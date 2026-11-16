// Format 51 (2026-11-14): standard Zarr with fewer files. Writers use the chunk key separator "." (c.0.1 in the array's folder); every reader reads
// both separators as the array's own zarr.json declares; an array over 4 MiB is sharded by default (~64 MiB shards of 2 MiB inner chunks), a small one
// stays one chunk. Synthetic arrays only.
#include "check.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <vector>

#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/store.h"
#include "nxr/io/zarr_store.h"

namespace fs = std::filesystem;
using namespace nxr::io;

namespace {
std::int64_t count_files(const fs::path& p) {
  std::int64_t n = 0;
  for (const auto& e : fs::recursive_directory_iterator(p)) if (e.is_regular_file()) ++n;
  return n;
}
std::int64_t count_dirs(const fs::path& p) {
  std::int64_t n = 0;
  for (const auto& e : fs::recursive_directory_iterator(p)) if (e.is_directory()) ++n;
  return n;
}
std::vector<float> ramp(std::int64_t n) {
  std::vector<float> v(static_cast<std::size_t>(n));
  std::iota(v.begin(), v.end(), 0.0f);
  return v;
}
}  // namespace

int main() {
  const fs::path tmp = fs::temp_directory_path() / "nxr_chunk_separator";
  fs::remove_all(tmp);
  auto fst = std::make_shared<FolderStore>(tmp);

  // both separators, written and read: the writer follows the array's own layout, the reader its zarr.json
  const auto v = ramp(10 * 7);
  for (const char* sep : {".", "/"}) {
    ArrayOptions o;
    o.chunks = Shape{10, 3};
    o.compress = false;
    ArrayMeta m = array_meta(DType::Float32, {10, 7}, o);
    m.separator = sep;
    const std::string at = std::string("plain") + (sep[0] == '.' ? "_dot" : "_slash");
    layout_array(*fst, at, m);
    populate_array(*fst, at, view(v, {10, 7}));
    const auto r = ArrayReader::open(fst, at);
    const Bytes b = r.read();
    CHECK(b.size() == v.size() * 4 && std::memcmp(b.data(), v.data(), b.size()) == 0);
    if (sep[0] == '.') CHECK(fs::exists(tmp / at / "c.0.2") && !fs::exists(tmp / at / "c"));
    else CHECK(fs::exists(tmp / at / "c" / "0" / "2"));
    // a region and a re-layout follow the declared separator, and the old chunks of the other encoding go
    const std::vector<float> patch(3, -1.0f);
    write_region(*fst, at, {2, 2}, view(patch, {1, 3}));
    const Bytes b2 = ArrayReader::open(fst, at).read({2, 2}, {3, 5});
    CHECK(std::memcmp(b2.data(), patch.data(), 12) == 0);
  }
  // a re-layout in the other separator removes the chunks of the first (no stale c.* beside c/, nor c/ beside c.*)
  {
    ArrayOptions o;
    o.chunks = Shape{10, 3};
    ArrayMeta m = array_meta(DType::Float32, {10, 7}, o);
    m.separator = "/";
    layout_array(*fst, "plain_dot", m);
    CHECK(!fs::exists(tmp / "plain_dot" / "c.0.2"));
    m.separator = ".";
    layout_array(*fst, "plain_slash", m);
    CHECK(!fs::exists(tmp / "plain_slash" / "c"));
  }
  // the older ZarrStore writes "." and reads both
  {
    ZarrStore zs(tmp / "zs.zarr");
    WriteOptions o;
    o.chunks = {10, 3};
    zs.write_array<float>("a", v, {10, 7}, o);
    CHECK(fs::exists(tmp / "zs.zarr" / "a" / "c.0.1"));
    CHECK(zs.read_array<float>("a") == v);
    fs::create_directories(tmp / "zs.zarr" / "a" / "c" / "0");
    // the same array in the "/" layout is read through its zarr.json
    fs::rename(tmp / "zs.zarr" / "a" / "c.0.0", tmp / "zs.zarr" / "a" / "c" / "0" / "0");
    fs::rename(tmp / "zs.zarr" / "a" / "c.0.1", tmp / "zs.zarr" / "a" / "c" / "0" / "1");
    fs::rename(tmp / "zs.zarr" / "a" / "c.0.2", tmp / "zs.zarr" / "a" / "c" / "0" / "2");
    std::string j = *FolderStore(tmp / "zs.zarr").get_text("a/zarr.json");
    j.replace(j.find("\"separator\":\".\""), 15, "\"separator\":\"/\"");
    FolderStore(tmp / "zs.zarr").set("a/zarr.json", j);
    CHECK(zs.read_array<float>("a") == v);
  }

  // a small array stays one chunk, one file; a large one is sharded by default: 100 MB = a few ~64 MiB shards, not ~50 chunk files
  {
    const ArrayMeta small = array_meta(DType::Float32, {512, 1000});   // 2 MB
    CHECK(small.codecs.size() == 1 && small.codecs[0]["name"] == "bytes" && small.grid == small.chunks);
    const ArrayMeta big = array_meta(DType::Float32, {25000, 1000});   // 100 MB
    CHECK(big.codecs[0]["name"] == "sharding_indexed");
    const std::int64_t shard_bytes = big.grid[0] * 1000 * 4;
    CHECK(shard_bytes <= kShardBytes && shard_bytes > kShardBytes / 2);
    CHECK(big.chunks[0] * 1000 * 4 <= kChunkBytes);
    const ArrayMeta ns = array_meta(DType::Float32, {25000, 1000}, ArrayOptions{.unsharded = true});
    CHECK(ns.codecs[0]["name"] == "bytes" || ns.codecs[0]["name"] == "blosc");
  }
  // a sharded array (default policy) round-trips whole and by region, in few files
  {
    const std::int64_t rows = 6000, cols = 1000;   // 24 MB: three 2 MB-chunks per... one shard
    const auto big = ramp(rows * cols);
    const ArrayMeta m = array_meta(DType::Float32, {rows, cols});
    CHECK(m.codecs[0]["name"] == "sharding_indexed");
    layout_array(*fst, "sh", m);
    populate_array(*fst, "sh", view(big, {rows, cols}));
    const auto r = ArrayReader::open(fst, "sh");
    const Bytes all = r.read();
    CHECK(all.size() == big.size() * 4 && std::memcmp(all.data(), big.data(), all.size()) == 0);
    const Bytes part = r.read({5000, 10}, {5003, 12});
    float x;
    std::memcpy(&x, part.data(), 4);
    CHECK(x == big[5000 * cols + 10]);
    CHECK(count_files(tmp / "sh") == 2);   // zarr.json and one shard
  }
  // files and folders of the same array under the old and the new policy (the numbers the report quotes)
  {
    const std::int64_t rows = 25000, cols = 1000;   // 100 MB
    const auto big = ramp(rows * cols);
    ArrayOptions o;
    o.unsharded = true;
    ArrayMeta old = array_meta(DType::Float32, {rows, cols}, o);
    old.separator = "/";
    layout_array(*fst, "old50", old);
    populate_array(*fst, "old50", view(big, {rows, cols}));
    const ArrayMeta now = array_meta(DType::Float32, {rows, cols});
    layout_array(*fst, "new51", now);
    populate_array(*fst, "new51", view(big, {rows, cols}));
    const auto f50 = count_files(tmp / "old50"), f51 = count_files(tmp / "new51");
    const auto d50 = count_dirs(tmp / "old50"), d51 = count_dirs(tmp / "new51");
    std::printf("100 MB array: format 50 %lld files, %lld folders (below the array's own); format 51 %lld files, %lld folders\n", (long long)f50, (long long)d50, (long long)f51, (long long)d51);
    CHECK(f51 < f50 && d51 < d50 && d51 == 0);
    const Bytes a = ArrayReader::open(fst, "old50").read({20000, 0}, {20002, 1000});
    const Bytes b = ArrayReader::open(fst, "new51").read({20000, 0}, {20002, 1000});
    CHECK(a == b);
  }
  fs::remove_all(tmp);
  return nxrtest::finish("chunk_separator");
}
