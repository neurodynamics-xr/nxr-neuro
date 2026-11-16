#include "nxr/io/array_writer.h"
#include "nxr/io/store.h"

#include <blosc.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>

#include "chunking.h"
#include "crc32c.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/errors.h"


namespace nxr::io {

using Bytes = std::vector<std::uint8_t>;

namespace {
Error bad(const std::string& m) { return Error(code::invalid_argument, m); }
std::int64_t prod(const Shape& s) { return detail::product(s); }
}  // namespace

std::int64_t ArrayView::size() const { return prod(shape); }
std::size_t ArrayView::nbytes() const { return static_cast<std::size_t>(size()) * dtype_size(dtype); }

std::size_t ArrayMeta::itemsize() const { return dtype_size(dtype_from_string(data_type)); }

Row ArrayMeta::row() const {
  Shape c = chunks;
  return {{"data_type", data_type},
          {"chunk_shape_json", jdump(Json(c))},
          {"codecs_json", jdump(codecs)},
          {"fill_value_json", jdump(fill_value)}};
}

Json ArrayMeta::doc() const {
  Json d = Json::object();
  d["zarr_format"] = 3;
  d["node_type"] = "array";
  d["shape"] = shape;
  d["data_type"] = data_type;
  d["chunk_grid"] = Json{{"name", "regular"}, {"configuration", Json{{"chunk_shape", grid}}}};
  d["chunk_key_encoding"] = Json{{"name", "default"}, {"configuration", Json{{"separator", separator}}}};
  d["fill_value"] = fill_value;
  d["codecs"] = codecs;
  if (in_pieces) d["nxr_written_in_pieces"] = Json{{"must_understand", false}};
  return d;
}

Shape default_chunks(const Shape& shape, std::size_t itemsize) {
  const std::int64_t total = prod(shape) * static_cast<std::int64_t>(itemsize);
  Shape out;
  for (auto s : shape) out.push_back(std::max<std::int64_t>(1, s));
  if (total <= kBigBytes || shape.empty() || shape[0] <= 1) return out;
  Shape rest(shape.begin() + 1, shape.end());
  const std::int64_t row = prod(rest) * static_cast<std::int64_t>(itemsize);
  std::int64_t rows = std::max<std::int64_t>(1, kChunkBytes / std::max<std::int64_t>(row, 1));
  const std::int64_t n = shape[0];
  const std::int64_t nchunks = std::max<std::int64_t>(1, (n + rows - 1) / rows);
  rows = (n + nchunks - 1) / nchunks;
  out[0] = std::max<std::int64_t>(1, std::min(rows, n));
  return out;
}

namespace {
Json bytes_codec(std::size_t itemsize) {
  // zarr omits the endian for a one-byte type
  return itemsize == 1 ? Json{{"name", "bytes"}} : Json{{"name", "bytes"}, {"configuration", Json{{"endian", "little"}}}};
}
Json blosc_codec(std::size_t itemsize) {
  return Json{{"name", "blosc"},
              {"configuration", Json{{"typesize", static_cast<std::int64_t>(itemsize)}, {"cname", "zstd"}, {"clevel", 3}, {"shuffle", "shuffle"}, {"blocksize", 0}}}};
}
Json fill_for(DType t) {
  switch (t) {
    case DType::Bool: return false;
    case DType::Float16: case DType::Float32: case DType::Float64: return 0.0;
    case DType::Complex64: case DType::Complex128: return Json::array({0.0, 0.0});
    default: return 0;
  }
}
bool is_float(DType t) { return t == DType::Float16 || t == DType::Float32 || t == DType::Float64 || t == DType::Complex64 || t == DType::Complex128; }
}  // namespace

std::optional<Shape> default_shards(const Shape& shape, const Shape& chunks, std::size_t itemsize) {
  if (shape.empty() || chunks.size() != shape.size() || prod(shape) * static_cast<std::int64_t>(itemsize) <= kBigBytes || shape[0] <= chunks[0]) return std::nullopt;
  // ~64 MiB shards of the inner chunks along axis 0 (the last may be partial)
  const std::int64_t nchunks = (shape[0] + chunks[0] - 1) / chunks[0];
  const std::int64_t row = std::max<std::int64_t>(1, prod(chunks) * static_cast<std::int64_t>(itemsize));
  const std::int64_t per = std::max<std::int64_t>(1, std::min(nchunks, kShardBytes / row));
  Shape shards = chunks;
  shards[0] = chunks[0] * per;
  return shards;
}

ArrayMeta array_meta(DType dtype, const Shape& shape_in, const ArrayOptions& opt) {
  const std::size_t isz = dtype_size(dtype);
  Shape shape = shape_in;
  Shape chunks = opt.chunks ? *opt.chunks : default_chunks(shape, isz);
  for (auto& c : chunks) c = std::max<std::int64_t>(1, c);
  const bool compress = opt.compress ? *opt.compress : prod(shape) * static_cast<std::int64_t>(isz) > kBigBytes;
  Shape grid = chunks;
  std::optional<Shape> shards = opt.shards;
  if (!shards && !opt.chunks && !opt.unsharded && !opt.in_pieces) shards = default_shards(shape, chunks, isz);   // the default policy shards a large array
  if (shards) {
    grid = *shards;
    for (auto& g : grid) g = std::max<std::int64_t>(1, g);
    if (grid.size() != chunks.size()) throw bad("shards and chunks differ in rank");
    for (std::size_t d = 0; d < grid.size(); ++d)
      if (grid[d] % chunks[d]) throw bad("a shard (" + std::to_string(grid[d]) + ") must be a whole number of chunks (" + std::to_string(chunks[d]) + ")");
  }
  Json inner = Json::array({bytes_codec(isz)});
  if (compress) inner.push_back(blosc_codec(isz));
  Json codecs;
  if (shards) {
    codecs = Json::array({Json{{"name", "sharding_indexed"},
                               {"configuration", Json{{"chunk_shape", chunks},
                                                      {"codecs", inner},
                                                      {"index_codecs", Json::array({Json{{"name", "bytes"}, {"configuration", Json{{"endian", "little"}}}}, Json{{"name", "crc32c"}}})},
                                                      {"index_location", "end"}}}}});
  } else {
    codecs = inner;
  }
  ArrayMeta m;
  m.shape = shape;
  m.data_type = dtype_to_string(dtype);
  m.chunks = chunks;
  m.grid = grid;
  m.codecs = codecs;
  m.fill_value = fill_for(dtype);
  m.in_pieces = opt.in_pieces;
  return m;
}

std::string array_chunk_key(const std::string& at, const Shape& g, const std::string& separator) {
  std::string p = join_key(at, "c");
  for (auto gi : g) p += separator + std::to_string(gi);
  return p;
}

void erase_chunks(Store& store, const std::string& at) {
  store.erase_prefix(join_key(at, "c"));   // "/": the folder c/
  for (const auto& k : store.list(at).keys) if (k.rfind("c.", 0) == 0) store.erase(join_key(at, k));   // ".": the files c.0.1
}

void layout_array(Store& store, const std::string& at, const ArrayMeta& meta, const std::optional<std::string>& attributes) {
  erase_chunks(store, at);   // the chunks of an array laid out before
  store.set(join_key(at, "zarr.json"), zarr_json_text(meta.doc(), attributes && !attributes->empty() ? *attributes : "{}"));
}

namespace {

// Does the chunk hold only the fill value (zero)? A float compares by value (-0.0 == 0), as numpy does.
bool all_fill(const Bytes& b, DType t) {
  if (!is_float(t)) return std::all_of(b.begin(), b.end(), [](std::uint8_t x) { return x == 0; });
  switch (t) {
    case DType::Float32: case DType::Complex64: {
      const float* p = reinterpret_cast<const float*>(b.data());
      for (std::size_t i = 0; i < b.size() / 4; ++i) if (!(p[i] == 0.0f)) return false;
      return true;
    }
    case DType::Float64: case DType::Complex128: {
      const double* p = reinterpret_cast<const double*>(b.data());
      for (std::size_t i = 0; i < b.size() / 8; ++i) if (!(p[i] == 0.0)) return false;
      return true;
    }
    default: {  // float16: zero is 0x0000 or 0x8000
      const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(b.data());
      for (std::size_t i = 0; i < b.size() / 2; ++i) if ((p[i] & 0x7FFF) != 0) return false;
      return true;
    }
  }
}

Bytes blosc_encode(const Bytes& raw, std::size_t typesize) {
  static const bool init = [] { blosc_init(); return true; }();
  (void)init;
  Bytes out(raw.size() + BLOSC_MAX_OVERHEAD);
  const int n = blosc_compress_ctx(3, BLOSC_SHUFFLE, typesize, raw.size(), raw.data(), out.data(), out.size(), "zstd", 0, 1);
  if (n <= 0) throw Error(code::io, "blosc: cannot compress a chunk");
  out.resize(static_cast<std::size_t>(n));
  return out;
}

// the inner codec chain of an array (`bytes` then maybe `blosc`) read from its metadata
struct Plan {
  bool compress = false;
  bool sharded = false;
  Shape inner;  // the inner chunk (== grid when not sharded)
  Shape grid;   // the file's unit
  std::size_t isz = 0;
  DType dtype = DType::Float64;
  Shape shape;
  std::string sep = "/";
};

Plan plan_of(const ArrayMeta& m) {
  Plan p;
  p.dtype = dtype_from_string(m.data_type);
  p.isz = dtype_size(p.dtype);
  p.shape = m.shape;
  p.grid = m.grid;
  p.inner = m.chunks;
  p.sep = m.separator;
  const Json* chain = &m.codecs;
  if (m.codecs.size() == 1 && m.codecs[0]["name"] == "sharding_indexed") {
    p.sharded = true;
    chain = &m.codecs[0]["configuration"]["codecs"];
  }
  for (const auto& c : *chain) if (c["name"] == "blosc") p.compress = true;
  return p;
}

Bytes encode_chunk(const Plan& p, const Bytes& raw) { return p.compress ? blosc_encode(raw, p.isz) : raw; }

// the morton (Z) order of a grid, as zarr-python 3.4 writes a shard's inner chunks in: the coordinates of the ceiling hypercube decoded
// in code order, those outside the grid dropped
std::vector<Shape> morton_order(const Shape& shape) {
  const std::size_t n = shape.size();
  std::vector<int> bits(n);
  int total = 0, maxb = 0;
  for (std::size_t d = 0; d < n; ++d) {
    int b = 0;
    while ((std::int64_t{1} << b) < shape[d]) ++b;  // (c-1).bit_length()
    bits[d] = b;
    total += b;
    maxb = std::max(maxb, b);
  }
  std::vector<Shape> out;
  const std::int64_t nz = total > 0 ? (std::int64_t{1} << total) : 1;
  for (std::int64_t z = 0; z < nz; ++z) {
    Shape c(n, 0);
    int in = 0;
    for (int cb = 0; cb < maxb; ++cb)
      for (std::size_t d = 0; d < n; ++d)
        if (cb < bits[d]) { c[d] |= ((z >> in) & 1) << cb; ++in; }
    bool ok = true;
    for (std::size_t d = 0; d < n; ++d) ok = ok && c[d] < shape[d];
    if (ok) out.push_back(c);
  }
  return out;
}

void for_each_index(const Shape& dims, const std::function<void(const Shape&)>& f) {
  if (prod(dims) == 0) return;
  Shape idx(dims.size(), 0);
  while (true) {
    f(idx);
    std::size_t d = dims.size();
    bool more = false;
    while (d > 0) {
      --d;
      if (++idx[d] < dims[d]) { more = true; break; }
      idx[d] = 0;
    }
    if (!more) break;
  }
}

// The chunk of `data` at grid index `g` (chunk shape `ck`), padded with the fill (zero); false when the chunk lies wholly outside the array.
bool extract(const ArrayView& data, const Shape& ck, const Shape& g, Bytes& out) {
  out.assign(static_cast<std::size_t>(prod(ck)) * dtype_size(data.dtype), 0);
  for (std::size_t d = 0; d < ck.size(); ++d) if (g[d] * ck[d] >= data.shape[d]) return false;
  detail::copy_chunk_region(data.shape, ck, g, dtype_size(data.dtype), const_cast<std::uint8_t*>(static_cast<const std::uint8_t*>(data.data)), out.data(), true);
  return true;
}

}  // namespace

namespace {

// the array at `at`, as its own zarr.json states it (the row's image)
Plan load_plan(const Store& store, const std::string& at) {
  const auto raw = store.get(join_key(at, "zarr.json"));
  if (!raw) throw Error(code::not_found, at + ": no array laid out here");
  const Json doc = Json::parse(raw->begin(), raw->end());
  ArrayMeta m;
  m.shape = doc["shape"].get<Shape>();
  m.data_type = doc["data_type"].get<std::string>();
  m.grid = doc["chunk_grid"]["configuration"]["chunk_shape"].get<Shape>();
  m.codecs = doc["codecs"];
  m.separator = doc.contains("chunk_key_encoding") ? doc["chunk_key_encoding"].value("configuration", Json::object()).value("separator", "/") : "/";
  m.chunks = m.grid;
  if (m.codecs.size() == 1 && m.codecs[0]["name"] == "sharding_indexed") m.chunks = m.codecs[0]["configuration"]["chunk_shape"].get<Shape>();
  return plan_of(m);
}

// The cells of `data` (a whole array, or a SLAB of one starting at grid cell `offset`) written: a plain chunk a file, a shard a file with its inner
// chunks in morton order and then the index. A shard is STREAMED to its file, chunk after chunk, never held whole.
void write_cells(Store& store, const std::string& at, const Plan& p, const ArrayView& data, const Shape& offset, bool dense) {
  const Shape ngrid = detail::chunk_grid(data.shape, p.grid);
  Shape per(p.shape.size(), 1);  // inner chunks per shard along each axis
  if (p.sharded) for (std::size_t d = 0; d < per.size(); ++d) per[d] = p.grid[d] / p.inner[d];
  const std::vector<Shape> order = p.sharded ? morton_order(per) : std::vector<Shape>{};

  for_each_index(ngrid, [&](const Shape& g) {
    Shape ga(g);
    for (std::size_t d = 0; d < ga.size(); ++d) ga[d] += offset[d];
    if (!p.sharded) {
      Bytes raw;
      extract(data, p.grid, g, raw);
      if (dense || !all_fill(raw, p.dtype)) store.set(array_chunk_key(at, ga, p.sep), encode_chunk(p, raw));
      return;
    }
    // a SHARD: its inner chunks in morton order, then the index (offset, length) per chunk in C order, then the index's crc32c
    const std::int64_t nin = prod(per);
    std::vector<std::uint64_t> index(static_cast<std::size_t>(nin) * 2, ~std::uint64_t{0});
    const std::string file = array_chunk_key(at, ga, p.sep);
    std::unique_ptr<StoreWriter> out;
    std::uint64_t pos = 0;
    for (const Shape& k : order) {
      Shape gk(k.size());
      for (std::size_t d = 0; d < k.size(); ++d) gk[d] = g[d] * per[d] + k[d];
      Bytes raw;
      const bool inside = extract(data, p.inner, gk, raw);
      if (!inside || (!dense && all_fill(raw, p.dtype))) continue;
      const Bytes enc = encode_chunk(p, raw);
      std::int64_t flat = 0;
      for (std::size_t d = 0; d < k.size(); ++d) flat = flat * per[d] + k[d];
      index[static_cast<std::size_t>(flat) * 2] = pos;
      index[static_cast<std::size_t>(flat) * 2 + 1] = enc.size();
      if (!out) out = store.open_writer(file);   // the shard's value comes into being with its first chunk
      out->append(enc.data(), enc.size());
      pos += enc.size();
    }
    if (!out) return;  // an all-empty shard is not written
    Bytes ib(index.size() * 8);
    std::memcpy(ib.data(), index.data(), ib.size());
    const std::uint32_t crc = detail::crc32c(ib.data(), ib.size());
    out->append(ib.data(), ib.size());
    out->append(reinterpret_cast<const std::uint8_t*>(&crc), 4);
    out->close();
  });
}

}  // namespace

void populate_array(Store& store, const std::string& at, const ArrayView& data, bool dense) {
  // the layout is read back from the array's own zarr.json (the row's image), so the bytes follow what is stored
  const Plan p = load_plan(store, at);
  if (data.shape != p.shape) throw bad("populate_array: data is " + std::to_string(data.size()) + " values, the array's shape differs");
  if (dtype_to_string(data.dtype) != dtype_to_string(p.dtype)) throw bad("populate_array: data is " + std::string(dtype_to_string(data.dtype)) + ", the array " + dtype_to_string(p.dtype));
  if (data.size() == 0) return;
  if (p.shape.empty()) {  // a scalar: one chunk, key "c"
    Bytes raw(p.isz);
    std::memcpy(raw.data(), data.data, p.isz);
    if (dense || !all_fill(raw, p.dtype)) store.set(join_key(at, "c"), encode_chunk(p, raw));
    return;
  }
  write_cells(store, at, p, data, Shape(p.shape.size(), 0), dense);
}

void write_region(Store& store, const std::string& at, const Shape& start, const ArrayView& data) {
  const Plan p = load_plan(store, at);
  const std::size_t rank = p.shape.size();
  if (dtype_to_string(data.dtype) != dtype_to_string(p.dtype)) throw bad("write_region: data is " + std::string(dtype_to_string(data.dtype)) + ", the array " + dtype_to_string(p.dtype));
  if (start.size() != rank || data.shape.size() != rank) throw bad("write_region: the region's rank differs from the array's");
  bool whole = true;
  for (std::size_t d = 0; d < rank; ++d) {
    if (start[d] < 0 || data.shape[d] < 0 || start[d] + data.shape[d] > p.shape[d]) throw bad("write_region: the region lies outside the array on axis " + std::to_string(d));
    if (start[d] != 0 || data.shape[d] != p.shape[d]) whole = false;
  }
  if (whole) { populate_array(store, at, data); return; }
  if (data.size() == 0) return;
  if (p.sharded) throw Error(code::unsupported, at + ": a region of a SHARDED array is not written (its shard would be rewritten whole); write the array whole");
  const ArrayReader reader = ArrayReader::open(store, at);
  Shape c0(rank), cn(rank);
  for (std::size_t d = 0; d < rank; ++d) {
    c0[d] = start[d] / p.grid[d];
    cn[d] = (start[d] + data.shape[d] + p.grid[d] - 1) / p.grid[d] - c0[d];
  }
  const auto* src = static_cast<const std::uint8_t*>(data.data);
  for_each_index(cn, [&](const Shape& k) {
    Shape g(rank), lo(rank), hi(rank), cell(rank);
    for (std::size_t d = 0; d < rank; ++d) {
      g[d] = c0[d] + k[d];
      lo[d] = g[d] * p.grid[d];
      hi[d] = std::min(lo[d] + p.grid[d], p.shape[d]);
      cell[d] = hi[d] - lo[d];
    }
    Bytes buf = reader.read(lo, hi);   // the chunk as it is (its fill where absent)
    // the overlap of the region and this chunk, copied element by element (C order on both sides)
    Shape olo(rank), on(rank);
    for (std::size_t d = 0; d < rank; ++d) { olo[d] = std::max(lo[d], start[d]); on[d] = std::min(hi[d], start[d] + data.shape[d]) - olo[d]; }
    for_each_index(on, [&](const Shape& e) {
      std::int64_t fc = 0, fd = 0;
      for (std::size_t d = 0; d < rank; ++d) { fc = fc * cell[d] + (olo[d] + e[d] - lo[d]); fd = fd * data.shape[d] + (olo[d] + e[d] - start[d]); }
      std::memcpy(buf.data() + static_cast<std::size_t>(fc) * p.isz, src + static_cast<std::size_t>(fd) * p.isz, p.isz);
    });
    store.erase(array_chunk_key(at, g, p.sep));  // write_cells leaves an all-fill chunk out: the old value must not stay behind
    write_cells(store, at, p, ArrayView{buf.data(), p.dtype, cell}, g, false);
  });
}

struct SlabWriter::Impl {
  Store* store = nullptr;
  std::string at;
  Plan plan;
  std::size_t axis = 1;
};

SlabWriter::SlabWriter(Store& store, const std::string& at, std::size_t axis) : impl_(std::make_shared<Impl>()) {
  impl_->store = &store;
  impl_->at = at;
  impl_->plan = load_plan(store, at);
  impl_->axis = axis;
  if (impl_->plan.shape.empty() || axis >= impl_->plan.shape.size()) throw bad("SlabWriter: no such axis");
}

const Shape& SlabWriter::shape() const { return impl_->plan.shape; }

void SlabWriter::write(std::int64_t start, const ArrayView& slab, bool dense) {
  const Plan& p = impl_->plan;
  const std::size_t ax = impl_->axis;
  if (dtype_to_string(slab.dtype) != dtype_to_string(p.dtype)) throw bad("SlabWriter: the slab is " + std::string(dtype_to_string(slab.dtype)) + ", the array " + dtype_to_string(p.dtype));
  if (slab.shape.size() != p.shape.size()) throw bad("SlabWriter: the slab's rank differs from the array's");
  for (std::size_t d = 0; d < p.shape.size(); ++d)
    if (d != ax && slab.shape[d] != p.shape[d]) throw bad("SlabWriter: the slab must span the whole array on every axis but " + std::to_string(ax));
  const std::int64_t n = slab.shape[ax];
  if (start < 0 || start + n > p.shape[ax]) throw bad("SlabWriter: the slab lies outside the array");
  if (start % p.grid[ax] != 0 || (n % p.grid[ax] != 0 && start + n != p.shape[ax])) throw bad("SlabWriter: a slab is whole grid cells along its axis");
  if (slab.size() == 0) return;
  Shape offset(p.shape.size(), 0);
  offset[ax] = start / p.grid[ax];
  write_cells(*impl_->store, impl_->at, p, slab, offset, dense);
}

}  // namespace nxr::io
