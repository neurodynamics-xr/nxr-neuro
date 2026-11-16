// nxr-io — an array's layout and its bytes, as nxr writes them (nxr.io.writer: ArrayMeta, array_meta, default_chunks, layout_array,
// populate_array). THE ROW FIRST: an array's layout is what its row states (`data_type` · `chunk_shape_json` · `codecs_json` · `fill_value_json`);
// `layout_array` writes its canonical zarr.json from that, `populate_array` the chunks.
//
//   ArrayMeta m = array_meta(DType::Float32, {1500, 1000});         // blosc (zstd, level 3, byte shuffle) above 4 MiB, axis 0 in ~2 MiB blocks
//   layout_array(store, "a/b", m);  populate_array(store, "a/b", view(data, {1500, 1000}));    // the array's keys: a/b/zarr.json, a/b/c/0/0 …
#pragma once
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/store.h"
#include "nxr/io/types.h"

namespace nxr::io {

using Shape = std::vector<std::int64_t>;

// A C-order, little-endian array in memory: a typed pointer and a shape. (Not owned.)
struct ArrayView {
  const void* data = nullptr;
  DType dtype = DType::Float64;
  Shape shape;
  std::int64_t size() const;
  std::size_t nbytes() const;
};

template <class T>
ArrayView view(const T* p, Shape shape) { return ArrayView{p, dtype_of<T>(), std::move(shape)}; }
template <class T>
ArrayView view(const std::vector<T>& v, Shape shape) { return ArrayView{v.data(), dtype_of<T>(), std::move(shape)}; }

// How an array is stored (all optional: the policy below decides what is not given).
struct ArrayOptions {
  std::optional<Shape> chunks;          // the chunk (the inner chunk, when sharded); default: the byte-budget policy
  std::optional<bool> compress;         // default: blosc for an array over 4 MiB
  std::optional<Shape> shards;          // the chunk grid becomes the shard, `chunks` the inner chunk
  bool unsharded = false;               // never shard by the default policy (an array written region by region: `write_region` refuses a sharded one)
  bool in_pieces = false;               // the array is written piece by piece (the Studio's WASM `put`): unsharded, and its zarr.json RECORDS it (`nxr_written_in_pieces`), so `compact` keeps it unsharded
  bool dense = false;                   // EVERY chunk on disk, even one equal to the fill value (write_empty_chunks)
};

// An array's stored layout, as the row states it, plus the chunk GRID (the shard, for a sharded array — the row's chunk is the inner one).
struct ArrayMeta {
  Shape shape;
  std::string data_type;
  Shape chunks, grid;
  Json codecs;
  Json fill_value;
  bool in_pieces = false;               // laid out unsharded on purpose, for region writes: zarr.json carries the extension field `nxr_written_in_pieces` ({"must_understand": false}, which zarr-python and zarrita ignore)
  std::string separator = ".";          // the chunk key separator: nxr format 51 writes "." (c.0.1 in the array's folder); an array laid out before has "/" (c/0/1)
  // the row's columns: data_type · chunk_shape_json · codecs_json · fill_value_json
  Row row() const;
  // the zarr.json document of the array, canonical order (SPEC §5)
  Json doc() const;
  std::size_t itemsize() const;
};

constexpr std::int64_t kBigBytes = 4 * 1024 * 1024;
constexpr std::int64_t kChunkBytes = 2 * 1024 * 1024;
constexpr std::int64_t kShardBytes = 64 * 1024 * 1024;   // the target shard of a large array (format 51)

// The key of the chunk (or shard) at grid index `g` of the array at `at`, for the declared separator: "a/c.0.1" or "a/c/0/1" (a scalar's: "a/c").
std::string array_chunk_key(const std::string& at, const Shape& g, const std::string& separator);
// Remove every chunk of the array at `at`, whichever separator it was written with (the folder `c/` and the files `c.*`).
void erase_chunks(Store& store, const std::string& at);

// The byte-budget policy: chunk axis 0 to ~2 MiB blocks once the array exceeds 4 MiB; otherwise one chunk (every extent floored at 1). `array_meta` shards
// an array over 4 MiB by default (format 51): the inner chunks above, a shard of up to ~64 MiB of them along axis 0 (a few files, not thousands).
Shape default_chunks(const Shape& shape, std::size_t itemsize);
// The shard the default policy gives an array of `shape` with inner chunk `chunks`: nullopt when it stays plain (the array is not over 4 MiB, or has no more rows than a chunk).
std::optional<Shape> default_shards(const Shape& shape, const Shape& chunks, std::size_t itemsize);
ArrayMeta array_meta(DType dtype, const Shape& shape, const ArrayOptions& opt = {});

// LAY OUT an array at key `at` from its row: the canonical zarr.json (its attributes the view's text when the node has a row).
void layout_array(Store& store, const std::string& at, const ArrayMeta& meta, const std::optional<std::string>& attributes = std::nullopt);
// POPULATE the laid-out array at key `at` with `data` (whole). `dense`: every chunk is written.
void populate_array(Store& store, const std::string& at, const ArrayView& data, bool dense = false);

// WRITE A REGION of a laid-out array: the values `data` at [start, start + data.shape) along every axis, the rest of the array kept as it is — each chunk the
// region touches read (an absent one is the fill value), patched, and written again (an all-fill chunk is removed: absent, as Zarr reads it). A plain
// chunked array only (a sharded one is refused: its shard would have to be rewritten whole). The region covering the whole array is populate_array.
void write_region(Store& store, const std::string& at, const Shape& start, const ArrayView& data);
// POPULATE a laid-out array one SLAB at a time, never holding it whole: `write(start, slab)` writes the grid cells of `slab` (the whole array on every axis but `axis`,
// along `axis` the elements [start, start + n), `start` and `n` whole grid cells but for the array's last) with the bytes `populate_array` writes for them.
// A sharded array's shard is streamed to its file chunk by chunk. The cells of a slab are written once (a slab is not merged with what is there).
class SlabWriter {
 public:
  SlabWriter(Store& store, const std::string& at, std::size_t axis = 1);   // the store must outlive the writer
  const Shape& shape() const;
  void write(std::int64_t start, const ArrayView& slab, bool dense = false);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace nxr::io
