#include "nxr/io/zarr_store.h"

#include <algorithm>
#include <cstring>
#include <filesystem>

#include "nxr/io/array_writer.h"
#include "nxr/io/codec_pipeline.h"
#include "chunking.h"
#include "metadata.h"

namespace fs = std::filesystem;

namespace nxr::io {

// ─── keys ─────────────────────────────────────────────────────────────────
static std::string chunk_key(const std::string& dir, const std::vector<std::int64_t>& g, const std::string& sep) {
  std::string p = join_key(dir, "c");
  for (std::int64_t gi : g) p += sep + std::to_string(gi);
  return p;
}

static nlohmann::json read_json(const Store& store, const std::string& key) {
  const auto raw = store.get(key);
  if (!raw) throw ZarrIOError("[nxr/io] cannot read " + key);
  return nlohmann::json::parse(raw->begin(), raw->end());
}

static std::string node_type_of(const Store& store, const std::string& dir) {
  const std::string j = join_key(dir, "zarr.json");
  if (!store.exists(j)) return "";
  try {
    return read_json(store, j).value("node_type", std::string());
  } catch (...) {
    return "";
  }
}

// Advance C-order grid odometer (dim 0 most-significant). Returns false on wrap.
static bool advance(std::vector<std::int64_t>& g, const std::vector<std::int64_t>& grid) {
  if (g.empty()) return false;
  int d = static_cast<int>(g.size()) - 1;
  while (d >= 0) {
    if (++g[d] < grid[d]) return true;
    g[d] = 0;
    --d;
  }
  return false;
}

// ─── store ────────────────────────────────────────────────────────────────
ZarrStore::ZarrStore(const fs::path& root) : store_(root) {
  fs::create_directories(root);
  if (!store_.exists("zarr.json")) store_.set("zarr.json", detail::make_group_json(nlohmann::json::object()).dump());
}

std::string ZarrStore::resolve(const std::string& path) const {
  std::string p = path;
  while (!p.empty() && p.front() == '/') p.erase(p.begin());
  if (p.find("..") != std::string::npos) {
    throw ZarrIOError("[nxr/io] path must not contain '..': " + path);
  }
  return p;
}

void ZarrStore::write_group(const std::string& path, const nlohmann::json& attributes) {
  store_.set(join_key(resolve(path), "zarr.json"), detail::make_group_json(attributes).dump());
}

bool ZarrStore::exists(const std::string& path) const {
  return store_.exists(join_key(resolve(path), "zarr.json"));
}

bool ZarrStore::is_group(const std::string& path) const {
  return node_type_of(store_, resolve(path)) == "group";
}

bool ZarrStore::is_array(const std::string& path) const {
  return node_type_of(store_, resolve(path)) == "array";
}

nlohmann::json ZarrStore::read_attributes(const std::string& path) const {
  const std::string j = join_key(resolve(path), "zarr.json");
  if (!store_.exists(j)) throw ZarrIOError("[nxr/io] no node at \"" + path + "\"");
  return read_json(store_, j).value("attributes", nlohmann::json::object());
}

std::vector<std::string> ZarrStore::list(const std::string& path) const {
  std::vector<std::string> out;
  const std::string dir = resolve(path);
  for (const auto& name : store_.list(dir).prefixes)
    if (store_.exists(join_key(join_key(dir, name), "zarr.json"))) out.push_back(name);
  return out;   // sorted by the store
}

void ZarrStore::update_attributes(const std::string& path, const nlohmann::json& patch) {
  const std::string j = join_key(resolve(path), "zarr.json");
  if (!store_.exists(j)) throw ZarrIOError("[nxr/io] no node at \"" + path + "\"");
  nlohmann::json meta = read_json(store_, j);
  nlohmann::json attrs = meta.value("attributes", nlohmann::json::object());
  attrs.update(patch);  // shallow merge: keys overwrite or add
  meta["attributes"] = attrs;
  store_.set(j, meta.dump());
}

void ZarrStore::delete_attribute(const std::string& path, const std::string& key) {
  const std::string j = join_key(resolve(path), "zarr.json");
  if (!store_.exists(j)) throw ZarrIOError("[nxr/io] no node at \"" + path + "\"");
  nlohmann::json meta = read_json(store_, j);
  if (meta.contains("attributes")) meta["attributes"].erase(key);
  store_.set(j, meta.dump());
}

void ZarrStore::delete_node(const std::string& path) {
  const std::string dir = resolve(path);
  if (dir.empty() || dir == ".") throw ZarrIOError("[nxr/io] refusing to delete the store root");
  store_.erase_prefix(dir);
}

ArrayMetadata ZarrStore::read_metadata(const std::string& path) const {
  const std::string mj = join_key(resolve(path), "zarr.json");
  if (!store_.exists(mj)) throw ZarrIOError("[nxr/io] no node at \"" + path + "\"");
  const nlohmann::json j = read_json(store_, mj);
  if (j.value("node_type", std::string()) != "array") {
    throw ZarrFormatError("[nxr/io] \"" + path + "\" is not an array");
  }
  return detail::parse_array_json(j);
}

void ZarrStore::write_raw(const std::string& path, const std::uint8_t* data, DType dtype,
                          const std::vector<std::int64_t>& shape, const WriteOptions& opts) {
  std::vector<std::int64_t> chunks = opts.chunks.empty() ? shape : opts.chunks;
  if (chunks.size() != shape.size()) {
    throw ZarrFormatError("[nxr/io] chunks rank != shape rank for \"" + path + "\"");
  }
  for (std::int64_t c : chunks) {
    if (c <= 0) throw ZarrFormatError("[nxr/io] chunk dimensions must be > 0 for \"" + path + "\"");
  }

  const std::string dir = resolve(path);
  erase_chunks(store_, dir);  // clear any stale chunks from a previous write
  store_.set(join_key(dir, "zarr.json"),
             detail::make_array_json(dtype, shape, chunks, opts.fill_value, opts.compress,
                                     opts.zstd_level, opts.attributes)
                 .dump());

  const std::size_t itemsize = dtype_size(dtype);
  const std::int64_t chunk_elems = detail::product(chunks);
  const std::vector<std::int64_t> grid = detail::chunk_grid(shape, chunks);
  const std::int64_t ngrid = detail::product(grid);
  const std::vector<std::uint8_t> fillp = detail::fill_pattern(dtype, opts.fill_value);
  const bool nonzero_fill = std::any_of(fillp.begin(), fillp.end(), [](std::uint8_t b) { return b != 0; });

  const CodecPipeline pipe = CodecPipeline::canonical(opts.compress, opts.zstd_level);

  std::vector<std::int64_t> g(shape.size(), 0);
  for (std::int64_t c = 0; c < ngrid; ++c) {
    std::vector<std::uint8_t> chunkbuf(static_cast<std::size_t>(chunk_elems) * itemsize, 0);
    if (nonzero_fill) {
      for (std::size_t off = 0; off < chunkbuf.size(); off += itemsize) {
        std::memcpy(chunkbuf.data() + off, fillp.data(), itemsize);
      }
    }
    detail::copy_chunk_region(shape, chunks, g, itemsize,
                              const_cast<std::uint8_t*>(data), chunkbuf.data(),
                              /*array_to_chunk=*/true);
    std::vector<std::uint8_t> enc = pipe.encode(chunkbuf.data(), chunkbuf.size());
    store_.set(chunk_key(dir, g, "."), enc);
    advance(g, grid);
  }
}

std::vector<std::uint8_t> ZarrStore::read_raw(const std::string& path,
                                              ArrayMetadata& out_meta) const {
  out_meta = read_metadata(path);
  const std::string dir = resolve(path);
  const std::size_t itemsize = dtype_size(out_meta.dtype);
  const std::int64_t total = detail::product(out_meta.shape);

  std::vector<std::uint8_t> out(static_cast<std::size_t>(total) * itemsize, 0);
  const std::vector<std::uint8_t> fillp = detail::fill_pattern(out_meta.dtype, out_meta.fill_value);
  if (std::any_of(fillp.begin(), fillp.end(), [](std::uint8_t b) { return b != 0; })) {
    for (std::size_t off = 0; off < out.size(); off += itemsize) {
      std::memcpy(out.data() + off, fillp.data(), itemsize);
    }
  }

  const std::int64_t chunk_elems = detail::product(out_meta.chunks);
  const std::vector<std::int64_t> grid = detail::chunk_grid(out_meta.shape, out_meta.chunks);
  const std::int64_t ngrid = detail::product(grid);

  const CodecPipeline pipe(out_meta.codecs);

  std::vector<std::int64_t> g(out_meta.shape.size(), 0);
  for (std::int64_t c = 0; c < ngrid; ++c) {
    if (auto stored = store_.get(chunk_key(dir, g, out_meta.separator))) {
      std::vector<std::uint8_t> enc = std::move(*stored);
      const std::size_t raw_size = static_cast<std::size_t>(chunk_elems) * itemsize;
      std::vector<std::uint8_t> dec = pipe.decode(enc.data(), enc.size(), raw_size);
      if (dec.size() != raw_size) {
        throw ZarrFormatError("[nxr/io] decoded chunk size mismatch at \"" + path + "\"");
      }
      detail::copy_chunk_region(out_meta.shape, out_meta.chunks, g, itemsize,
                                out.data(), dec.data(), /*array_to_chunk=*/false);
    }
    advance(g, grid);
  }
  return out;
}

}  // namespace nxr::io
