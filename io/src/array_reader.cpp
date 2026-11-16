#include "nxr/io/array_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include <blosc.h>
#include <zlib.h>
#include <nlohmann/json.hpp>

#include "nxr/io/codec.h"
#include "nxr/io/errors.h"
#include "nxr/io/store.h"
#include "crc32c.h"

using nlohmann::json;

namespace nxr::io {

using Bytes = std::vector<std::uint8_t>;
using Dims = std::vector<std::int64_t>;

namespace {

[[noreturn]] void bad(const std::string& m) { throw ZarrFormatError("[nxr/io] " + m); }

std::int64_t product(const Dims& d) {
  std::int64_t p = 1;
  for (auto x : d) p *= x;
  return p;
}

using detail::crc32c;

// ── codec chain ───────────────────────────────────────────────────────────────────────────────────
struct Chain {
  bool big_endian = false;       // the `bytes` codec's endian
  std::vector<json> b2b;         // bytes->bytes codecs in written order (blosc, zstd, gzip, crc32c)
  bool sharded = false;
  json shard;                    // the sharding_indexed configuration
};

Chain parse_chain(const json& codecs, bool allow_shard) {
  Chain c;
  bool have_bytes = false;
  for (const auto& k : codecs) {
    const std::string name = k.at("name").get<std::string>();
    const json cfg = k.contains("configuration") ? k.at("configuration") : json::object();
    if (name == "bytes") {
      have_bytes = true;
      c.big_endian = cfg.value("endian", std::string("little")) == "big";
    } else if (name == "sharding_indexed") {
      if (!allow_shard) bad("nested sharding is not supported");
      have_bytes = true;
      c.sharded = true;
      c.shard = cfg;
    } else if (name == "blosc" || name == "zstd" || name == "gzip" || name == "crc32c") {
      if (!have_bytes) bad("codec \"" + name + "\" before the array->bytes codec");
      c.b2b.push_back(k);
    } else {
      bad("unsupported codec \"" + name + "\"");
    }
  }
  if (!have_bytes) bad("codec pipeline has no array->bytes codec");
  return c;
}

Bytes inflate_gzip(const std::uint8_t* p, std::size_t n, std::size_t expect) {
  z_stream z{};
  if (inflateInit2(&z, 15 + 32) != Z_OK) bad("gzip: cannot start");
  Bytes out(expect ? expect : n * 4 + 64);
  z.next_in = const_cast<Bytef*>(p);
  z.avail_in = static_cast<uInt>(n);
  std::size_t have = 0;
  int rc;
  do {
    if (have == out.size()) out.resize(out.size() * 2);
    z.next_out = out.data() + have;
    z.avail_out = static_cast<uInt>(out.size() - have);
    rc = inflate(&z, Z_NO_FLUSH);
    have = out.size() - z.avail_out;
    if (rc != Z_OK && rc != Z_STREAM_END) { inflateEnd(&z); bad("gzip: corrupt data"); }
  } while (rc != Z_STREAM_END);
  inflateEnd(&z);
  out.resize(have);
  return out;
}

Bytes decode_b2b(const Chain& c, Bytes buf, std::size_t raw) {
  for (std::size_t k = c.b2b.size(); k-- > 0;) {
    const std::string name = c.b2b[k].at("name").get<std::string>();
    const std::size_t expect = k == 0 ? raw : 0;
    if (name == "crc32c") {
      if (buf.size() < 4) bad("crc32c: chunk shorter than its checksum");
      std::uint32_t want;
      std::memcpy(&want, buf.data() + buf.size() - 4, 4);
      buf.resize(buf.size() - 4);
      if (crc32c(buf.data(), buf.size()) != want) bad("crc32c: checksum mismatch");
    } else if (name == "zstd") {
      buf = zstd_decompress(buf.data(), buf.size(), 0);
    } else if (name == "gzip") {
      buf = inflate_gzip(buf.data(), buf.size(), expect);
    } else if (name == "blosc") {
      std::size_t nbytes = 0, cbytes = 0, blocksize = 0;
      blosc_cbuffer_sizes(buf.data(), &nbytes, &cbytes, &blocksize);
      Bytes out(nbytes);
      if (nbytes) {
        const int r = blosc_decompress(buf.data(), out.data(), nbytes);
        if (r < 0) bad("blosc: corrupt data");
      }
      buf = std::move(out);
    }
  }
  if (buf.size() != raw) bad("decoded chunk is " + std::to_string(buf.size()) + " bytes, expected " + std::to_string(raw));
  return buf;
}

void swap_endian(Bytes& b, std::size_t itemsize, DType t) {
  const std::size_t w = (t == DType::Complex64 || t == DType::Complex128) ? itemsize / 2 : itemsize;
  if (w < 2) return;
  for (std::size_t i = 0; i + w <= b.size(); i += w) std::reverse(b.begin() + static_cast<std::ptrdiff_t>(i), b.begin() + static_cast<std::ptrdiff_t>(i + w));
}

// ── fill value ────────────────────────────────────────────────────────────────────────────────────
template <class T> void put(Bytes& b, std::size_t off, T v) { std::memcpy(b.data() + off, &v, sizeof(T)); }

double number(const json& j) {
  if (j.is_string()) {
    const std::string s = j.get<std::string>();
    if (s == "NaN") return std::numeric_limits<double>::quiet_NaN();
    if (s == "Infinity") return std::numeric_limits<double>::infinity();
    if (s == "-Infinity") return -std::numeric_limits<double>::infinity();
    bad("unsupported fill_value \"" + s + "\"");
  }
  if (j.is_boolean()) return j.get<bool>() ? 1.0 : 0.0;
  return j.get<double>();
}

Bytes fill_bytes(DType t, const json& fv) {
  Bytes b(dtype_size(t), 0);
  if (fv.is_null()) return b;
  switch (t) {
    case DType::Bool: case DType::UInt8: put<std::uint8_t>(b, 0, static_cast<std::uint8_t>(number(fv))); break;
    case DType::Int8: put<std::int8_t>(b, 0, static_cast<std::int8_t>(number(fv))); break;
    case DType::Int16: put<std::int16_t>(b, 0, static_cast<std::int16_t>(number(fv))); break;
    case DType::UInt16: put<std::uint16_t>(b, 0, static_cast<std::uint16_t>(number(fv))); break;
    case DType::Int32: put<std::int32_t>(b, 0, static_cast<std::int32_t>(number(fv))); break;
    case DType::UInt32: put<std::uint32_t>(b, 0, static_cast<std::uint32_t>(number(fv))); break;
    case DType::Int64: put<std::int64_t>(b, 0, fv.is_number_integer() ? fv.get<std::int64_t>() : static_cast<std::int64_t>(number(fv))); break;
    case DType::UInt64: put<std::uint64_t>(b, 0, fv.is_number_integer() ? fv.get<std::uint64_t>() : static_cast<std::uint64_t>(number(fv))); break;
    case DType::Float32: put<float>(b, 0, static_cast<float>(number(fv))); break;
    case DType::Float64: put<double>(b, 0, number(fv)); break;
    case DType::Complex64:
      if (fv.is_array()) { put<float>(b, 0, static_cast<float>(number(fv.at(0)))); put<float>(b, 4, static_cast<float>(number(fv.at(1)))); }
      else put<float>(b, 0, static_cast<float>(number(fv)));
      break;
    case DType::Complex128:
      if (fv.is_array()) { put<double>(b, 0, number(fv.at(0))); put<double>(b, 8, number(fv.at(1))); }
      else put<double>(b, 0, number(fv));
      break;
    case DType::Float16:
      if (number(fv) != 0.0) bad("a non-zero float16 fill_value is not supported");
      break;
  }
  return b;
}

// ── a chunk's value, read by ranges through the store ───────────────────────────────────────────────
struct StoredValue {
  const Store& store;
  std::string key;
  std::uint64_t size = 0;
  StoredValue(const Store& s, std::string k, std::uint64_t n) : store(s), key(std::move(k)), size(n) {}
  Bytes read(std::uint64_t off, std::uint64_t n) const {
    if (off + n > size) bad("a chunk reaches past the end of its file");
    auto b = store.get_range(key, off, n);
    if (!b) throw ZarrIOError("[nxr/io] cannot read " + key);
    return std::move(*b);
  }
};

}  // namespace

// ── the array ─────────────────────────────────────────────────────────────────────────────────────
struct ArrayReader::Impl {
  std::shared_ptr<const Store> owner;   // set when the reader keeps its store alive
  const Store* store = nullptr;
  std::string key;   // the array's prefix in the store
  Dims shape, grid;  // grid: the outer chunk shape (the shard, when sharded)
  DType dtype = DType::Float64;
  std::size_t itemsize = 0;
  Bytes fill;
  Chain outer;
  // sharded
  Dims inner;        // inner chunk shape
  Chain inner_chain, index_chain;
  bool index_at_end = true;
  // the chunk key encoding the array declares (`default`: "c" + sep + each index; `v2`: the indices alone), the separator "/" or "."
  bool key_v2 = false;
  std::string sep = "/";

  // Copy the part of `src` (a dense block of shape `sshape` whose first element sits at array coordinates `org`) that
  // falls inside [start, stop) into `out` (shape stop - start).
  void scatter(const Bytes& src, const Dims& sshape, const Dims& org, const Dims& start, const Dims& stop, Bytes& out) const {
    const std::size_t n = shape.size();
    if (n == 0) { std::memcpy(out.data(), src.data(), itemsize); return; }
    Dims lo(n), len(n), sst(n), ost(n);
    for (std::size_t d = 0; d < n; ++d) {
      lo[d] = std::max(org[d], start[d]);
      const std::int64_t hi = std::min({org[d] + sshape[d], stop[d], shape[d]});
      len[d] = hi - lo[d];
      if (len[d] <= 0) return;
    }
    sst[n - 1] = 1; ost[n - 1] = 1;
    for (std::size_t d = n - 1; d-- > 0;) { sst[d] = sst[d + 1] * sshape[d + 1]; ost[d] = ost[d + 1] * (stop[d + 1] - start[d + 1]); }
    Dims idx(n, 0);
    const std::size_t run = static_cast<std::size_t>(len[n - 1]) * itemsize;
    while (true) {
      std::int64_t so = 0, oo = 0;
      for (std::size_t d = 0; d < n; ++d) { so += (lo[d] - org[d] + idx[d]) * sst[d]; oo += (lo[d] - start[d] + idx[d]) * ost[d]; }
      std::memcpy(out.data() + static_cast<std::size_t>(oo) * itemsize, src.data() + static_cast<std::size_t>(so) * itemsize, run);
      std::size_t d = n - 1;  // advance the odometer over axes 0..n-2
      while (d > 0) {
        --d;
        if (++idx[d] < len[d]) goto next;
        idx[d] = 0;
      }
      break;
    next:;
    }
  }

  Bytes decode_chunk(const Chain& c, Bytes enc, const Dims& cshape) const {
    const std::size_t raw = static_cast<std::size_t>(product(cshape)) * itemsize;
    Bytes b = decode_b2b(c, std::move(enc), raw);
    if (c.big_endian) swap_endian(b, itemsize, dtype);
    return b;
  }

  void read_outer(const Dims& g, const Dims& start, const Dims& stop, Bytes& out) const {
    const std::size_t n = shape.size();
    std::string p = key_v2 ? std::string() : join_key(key, "c");
    for (std::size_t d = 0; d < g.size(); ++d) p += ((key_v2 && d == 0) ? "" : sep) + std::to_string(g[d]);
    if (key_v2) p = join_key(key, g.empty() ? "0" : p);
    const auto total = store->size(p);
    if (!total) return;  // missing chunk: the fill value
    Dims org(n);
    for (std::size_t d = 0; d < n; ++d) org[d] = g[d] * grid[d];
    const StoredValue file(*store, p, *total);
    if (!outer.sharded) {
      scatter(decode_chunk(outer, file.read(0, file.size), grid), grid, org, start, stop, out);
      return;
    }
    // sharded: the index, then only the inner chunks the region touches
    Dims ngrid(n);
    for (std::size_t d = 0; d < n; ++d) ngrid[d] = (grid[d] + inner[d] - 1) / inner[d];
    const std::int64_t nin = product(ngrid);
    const std::uint64_t entries = static_cast<std::uint64_t>(nin) * 16;
    // index codecs: bytes (+ crc32c): its stored size is entries (+4)
    std::uint64_t isize = entries;
    for (const auto& k : index_chain.b2b) if (k.at("name") == "crc32c") isize += 4;
    if (isize > file.size) bad("shard is shorter than its index");
    Bytes idx = file.read(index_at_end ? file.size - isize : 0, isize);
    idx = decode_b2b(index_chain, std::move(idx), static_cast<std::size_t>(entries));
    constexpr std::uint64_t none = std::numeric_limits<std::uint64_t>::max();
    Dims lo(n), hi(n);
    for (std::size_t d = 0; d < n; ++d) {
      lo[d] = std::max<std::int64_t>(0, (std::max(start[d], org[d]) - org[d]) / inner[d]);
      hi[d] = std::min<std::int64_t>(ngrid[d] - 1, (std::min({stop[d], org[d] + grid[d], shape[d]}) - 1 - org[d]) / inner[d]);
      if (hi[d] < lo[d]) return;
    }
    Dims k = lo;
    while (true) {
      std::int64_t flat = 0;
      for (std::size_t d = 0; d < n; ++d) flat = flat * ngrid[d] + k[d];
      std::uint64_t off, len;
      std::memcpy(&off, idx.data() + flat * 16, 8);
      std::memcpy(&len, idx.data() + flat * 16 + 8, 8);
      if (!(off == none && len == none)) {
        Dims iorg(n);
        for (std::size_t d = 0; d < n; ++d) iorg[d] = org[d] + k[d] * inner[d];
        scatter(decode_chunk(inner_chain, file.read(off, len), inner), inner, iorg, start, stop, out);
      }
      std::size_t d = n;
      bool more = false;
      while (d > 0) {
        --d;
        if (++k[d] <= hi[d]) { more = true; break; }
        k[d] = lo[d];
      }
      if (!more) break;
    }
  }
};

ArrayReader::ArrayReader(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ArrayReader::ArrayReader(ArrayReader&&) noexcept = default;
ArrayReader& ArrayReader::operator=(ArrayReader&&) noexcept = default;
ArrayReader::~ArrayReader() = default;

ArrayReader ArrayReader::open(std::shared_ptr<const Store> store, const std::string& key) {
  ArrayReader r = open(*store, key);
  r.impl_->owner = std::move(store);
  return r;
}

ArrayReader ArrayReader::open(const Store& store, const std::string& key) {
  static const bool blosc_ready = [] { blosc_init(); return true; }();
  (void)blosc_ready;
  const auto doc = store.get(join_key(key, "zarr.json"));
  if (!doc) throw ZarrIOError("[nxr/io] no array at " + key);
  const json j = json::parse(doc->begin(), doc->end());
  if (j.value("zarr_format", 0) != 3) bad("only Zarr v3 is read");
  if (j.value("node_type", std::string()) != "array") bad(key + " is not an array");
  auto im = std::make_unique<Impl>();
  im->store = &store;
  im->key = key;
  im->shape = j.at("shape").get<Dims>();
  im->dtype = dtype_from_string(j.at("data_type").get<std::string>());
  im->itemsize = dtype_size(im->dtype);
  const json& cg = j.at("chunk_grid");
  if (cg.at("name") != "regular") bad("only the regular chunk grid is read");
  im->grid = cg.at("configuration").at("chunk_shape").get<Dims>();
  const json& ke = j.at("chunk_key_encoding");
  // both separators are read, as the array's own zarr.json declares (nxr format 51 writes "."; "/" is every older dataset's)
  const std::string kname = ke.at("name").get<std::string>();
  if (kname != "default" && kname != "v2") bad("only the default and v2 chunk key encodings are read");
  im->key_v2 = kname == "v2";
  im->sep = ke.value("configuration", json::object()).value("separator", im->key_v2 ? "." : "/");
  if (im->sep != "/" && im->sep != ".") bad("the chunk key separator must be \"/\" or \".\"");
  if (im->grid.size() != im->shape.size()) bad("chunk_shape rank differs from the shape's");
  for (auto c : im->grid) if (c <= 0) bad("chunk_shape must be positive");
  im->fill = fill_bytes(im->dtype, j.contains("fill_value") ? j.at("fill_value") : json());
  im->outer = parse_chain(j.at("codecs"), true);
  if (im->outer.sharded) {
    const json& s = im->outer.shard;
    im->inner = s.at("chunk_shape").get<Dims>();
    if (im->inner.size() != im->shape.size()) bad("sharding chunk_shape rank differs from the shape's");
    for (auto c : im->inner) if (c <= 0) bad("sharding chunk_shape must be positive");
    im->inner_chain = parse_chain(s.at("codecs"), false);
    im->index_chain = parse_chain(s.at("index_codecs"), false);
    im->index_at_end = s.value("index_location", std::string("end")) != "start";
  }
  return ArrayReader(std::move(im));
}

const Dims& ArrayReader::shape() const { return impl_->shape; }
DType ArrayReader::dtype() const { return impl_->dtype; }
std::size_t ArrayReader::itemsize() const { return impl_->itemsize; }

Bytes ArrayReader::read() const { return read(Dims(impl_->shape.size(), 0), impl_->shape); }

Bytes ArrayReader::read(const Dims& start, const Dims& stop) const {
  const Impl& m = *impl_;
  const std::size_t n = m.shape.size();
  if (start.size() != n || stop.size() != n) bad("region rank differs from the array's");
  Dims ext(n);
  for (std::size_t d = 0; d < n; ++d) {
    if (start[d] < 0 || stop[d] < start[d] || stop[d] > m.shape[d]) bad("region outside the array");
    ext[d] = stop[d] - start[d];
  }
  const std::int64_t total = product(ext);
  Bytes out(static_cast<std::size_t>(total) * m.itemsize);
  if (total == 0) return out;
  const bool zero = std::all_of(m.fill.begin(), m.fill.end(), [](std::uint8_t x) { return x == 0; });
  if (!zero) for (std::size_t o = 0; o < out.size(); o += m.itemsize) std::memcpy(out.data() + o, m.fill.data(), m.itemsize);
  else std::fill(out.begin(), out.end(), 0);
  if (n == 0) { m.read_outer({}, start, stop, out); return out; }
  Dims lo(n), hi(n);
  for (std::size_t d = 0; d < n; ++d) { lo[d] = start[d] / m.grid[d]; hi[d] = (stop[d] - 1) / m.grid[d]; }
  Dims g = lo;
  while (true) {
    m.read_outer(g, start, stop, out);
    std::size_t d = n;
    bool more = false;
    while (d > 0) {
      --d;
      if (++g[d] <= hi[d]) { more = true; break; }
      g[d] = lo[d];
    }
    if (!more) break;
  }
  return out;
}

}  // namespace nxr::io
