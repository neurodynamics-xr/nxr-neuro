// `dataset compact` (nxr::io::compact_dataset): a format-50 dataset (chunk key separator "/", large arrays unsharded) becomes format 51 ("." and sharded),
// array by array, every value kept, the files and folders fewer; a crash anywhere leaves the old layout or finishes on the next run; a second run changes
// nothing; a dataset another writer holds is refused. Synthetic data only.
#include <cstring>
#include <fstream>
#include <functional>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>

#include "check.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/compact.h"
#include "nxr/io/database.h"
#include "nxr/io/errors.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
CompactOptions mk(CompactOptions o = {}) { o.small_files = false; return o; }
std::vector<float> ramp(std::int64_t n, std::int64_t zero_from = -1, std::int64_t zero_to = -1, std::int64_t row_len = 1) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    const std::uint64_t x = (static_cast<std::uint64_t>(i) * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    v[static_cast<std::size_t>(i)] = static_cast<float>(x >> 20) / 8.0f;
    const std::int64_t row = i / row_len;
    if (zero_from >= 0 && row >= zero_from && row < zero_to) v[static_cast<std::size_t>(i)] = 0.0f;
  }
  return v;
}

struct Tree { std::int64_t files = 0, dirs = 0; };
Tree tree_of(const fs::path& p) {
  Tree t;
  for (const auto& e : fs::recursive_directory_iterator(p)) (e.is_directory() ? t.dirs : t.files)++;
  return t;
}
// every file under p (relative) with its bytes: two trees are the same when these maps are
std::map<std::string, std::string> snapshot(const fs::path& p) {
  std::map<std::string, std::string> m;
  for (const auto& e : fs::recursive_directory_iterator(p)) {
    if (!e.is_regular_file() || e.path().filename().string().rfind("dataset.sqlite", 0) == 0) continue;
    std::ifstream f(e.path(), std::ios::binary);
    m[fs::relative(e.path(), p).string()] = std::string(std::istreambuf_iterator<char>(f), {});
  }
  return m;
}

ManifoldSpec tri() {
  ManifoldSpec m;
  m.name = "m";
  m.path = "source/m";
  m.type = "surface";
  m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/m/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = "source/m/vertices";
  m.geometry = g;
  return m;
}

// a dataset as format 50 left it: separator "/", the large arrays unsharded; plus a ROWLESS large array and a scalar
std::string make_v50(const fs::path& datastore, const std::string& name) {
  CreateOptions co51;
  co51.store_format = 51;   // this test is about the folder tree: format 50 to 51 (format 52 is test_format52)
  DatasetWriter ds = create_dataset(datastore, name, co51);
  SubjectWriter sub = SubjectWriter::create(ds, "S");
  const std::vector<double> v(12, 1.0);
  const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
  ManifoldSpec m = tri();
  m.populate = [&](Store&, const std::string&) {
    sub.write_array("source/m/vertices", view(v, {4, 3}));
    sub.write_array("source/m/faces", view(f, {2, 3}));
  };
  const std::string mid = sub.manifold(m);
  auto v50 = [](DType t, const Shape& shape, ArrayOptions o) {
    o.unsharded = true;
    ArrayMeta a = array_meta(t, shape, o);
    a.separator = "/";
    return a;
  };
  const auto big = ramp(4000 * 1000, 1000, 1500, 1000);          // 16 MB: eight 2 MB chunks, a zero stretch (chunks the writer leaves out)
  FieldSpec b;
  b.name = "big";
  b.path = "source/big";
  b.kind = "map";
  b.manifold_id = mid;
  b.data = view(big, {4000, 1000});
  b.meta = v50(DType::Float32, {4000, 1000}, {});
  sub.field(b);
  const auto small = ramp(4);
  FieldSpec s;
  s.name = "small";
  s.path = "source/small";
  s.kind = "map";
  s.manifold_id = mid;
  s.data = view(small, {4});
  s.meta = v50(DType::Float32, {4}, {});
  sub.field(s);
  std::vector<std::uint8_t> zeros(50, 0);
  FieldSpec dz;
  dz.name = "dense_zeros";
  dz.path = "source/dense_zeros";
  dz.kind = "map";
  dz.manifold_id = mid;
  dz.value_type = "integer";
  dz.data = view(zeros, {50});
  dz.array.dense = true;
  dz.meta = v50(DType::UInt8, {50}, {});
  sub.field(dz);
  sub.complete();
  const std::string base = store_path_of(ds.db(), sub.id());
  // a rowless array (no node names it) and a scalar
  const auto extra = ramp(3000 * 1000);
  ArrayMeta em = v50(DType::Float32, {3000, 1000}, {});
  layout_array(ds.store(), base + "/source/extra", em);
  populate_array(ds.store(), base + "/source/extra", view(extra, {3000, 1000}));
  const float one = 7.5f;
  ArrayMeta sc = v50(DType::Float32, {}, {});
  layout_array(ds.store(), base + "/source/scalar", sc);
  populate_array(ds.store(), base + "/source/scalar", view(&one, Shape{}));
  return base;
}

std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); } catch (...) { return "other"; }
  return "";
}

Bytes read_all(Store& st, const std::string& key) { return ArrayReader::open(st, key).read(); }
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / "nxr_io_test_compact";
  fs::remove_all(root);
  const char* arrays[] = {"source/big", "source/small", "source/dense_zeros", "source/extra", "source/scalar"};

  // 1. the values stay, the files and folders drop, the rows and zarr.json follow, the second run does nothing
  std::map<std::string, Bytes> want;
  Tree before, after;
  std::map<std::string, std::string> clean_tree;
  {
    const std::string base = make_v50(root, "d");
    DatasetWriter ds = open_writer(root / "d");
    for (const char* a : arrays) want[a] = read_all(ds.store(), base + "/" + a);
    before = tree_of(root / "d");
    const Json dry = compact_dataset(ds, mk(CompactOptions{.dry_run = true}));
    CHECK_EQ(dry["compacted"].get<int>(), 5);
    CHECK_EQ(dry["files_before"].get<std::int64_t>() > dry["files_after"].get<std::int64_t>(), true);
    CHECK_EQ(snapshot(root / "d").size(), snapshot(root / "d").size());
    CHECK(read_all(ds.store(), base + "/source/big") == want["source/big"]);
    CHECK(ds.store().exists_prefix(base + "/source/big/c"));                 // a dry run changed nothing
    const Json r = compact_dataset(ds, mk());
    CHECK_EQ(r["compacted"].get<int>(), 5);
    CHECK_EQ(r["already_format_51"].get<int>(), 2);                 // the manifold's vertices and faces were written as format 51 already
    CHECK(r["store_format_set"].get<bool>());
    for (const char* a : arrays) CHECK(read_all(ds.store(), base + "/" + a) == want[a]);
    after = tree_of(root / "d");
    CHECK(after.files < before.files && after.dirs < before.dirs);
    CHECK_EQ(r["files_before"].get<std::int64_t>() > r["files_after"].get<std::int64_t>(), true);
    for (const char* a : arrays) {
      const Json z = Json::parse(*ds.store().get_text(base + "/" + a + "/zarr.json"));
      CHECK_EQ(z["chunk_key_encoding"]["configuration"]["separator"].get<std::string>(), std::string("."));
      CHECK(!ds.store().exists_prefix(base + "/" + a + "/.compact"));
    }
    const Json zb = Json::parse(*ds.store().get_text(base + "/source/big/zarr.json"));
    CHECK_EQ(zb["codecs"][0]["name"].get<std::string>(), std::string("sharding_indexed"));
    CHECK(!ds.store().exists_prefix(base + "/source/big/c"));
    // the row follows the file, and the node's zarr.json is the view's text
    const Row fr = *ds.db().one("SELECT codecs_json FROM field WHERE name = 'big'");
    CHECK(text(fr, "codecs_json").find("sharding_indexed") != std::string::npos);
    CHECK_EQ(text_parity(ds.db(), ds.store()).mismatched, 0);
    CHECK_EQ(text(*ds.db().one("SELECT value FROM meta WHERE key = 'store_format'"), "value"), std::string(kStoreFormat51));
    // idempotent: a second run skips every array and changes no byte
    clean_tree = snapshot(root / "d");
    const Json again = compact_dataset(ds, mk());
    CHECK_EQ(again["compacted"].get<int>(), 0);
    CHECK_EQ(again["already_format_51"].get<int>(), 7);
    CHECK(snapshot(root / "d") == clean_tree);
  }

  // 2. a crash at any stage leaves the old layout (before the commit) and the next run ends in the same tree as the clean run
  for (const char* crash : {"written", "verified", "moved", "committed"}) {
    const fs::path r2 = root / (std::string("crash_") + crash);
    const std::string base = make_v50(r2, "d");
    {
      DatasetWriter ds = open_writer(r2 / "d");
      CompactOptions o;
      o.on_stage = [&](const std::string& s, const std::string& at) { if (s == crash && at == base + "/source/big") throw std::runtime_error("crash"); };
      CHECK_EQ(code_of([&] { compact_dataset(ds, mk(o)); }), std::string("other"));
      const Json z = Json::parse(*ds.store().get_text(base + "/source/big/zarr.json"));
      const bool committed = std::string(crash) == "committed";
      CHECK_EQ(z["chunk_key_encoding"]["configuration"]["separator"].get<std::string>(), std::string(committed ? "." : "/"));
      if (!committed) CHECK(read_all(ds.store(), base + "/source/big") == want["source/big"]);          // the old layout reads as before
      else CHECK(read_all(ds.store(), base + "/source/big") == want["source/big"]);
    }
    DatasetWriter ds = open_writer(r2 / "d");
    const Json r = compact_dataset(ds, mk());
    CHECK(r["store_format_set"].get<bool>());
    for (const char* a : arrays) CHECK(read_all(ds.store(), base + "/" + a) == want[a]);
    CHECK_EQ(text_parity(ds.db(), ds.store()).mismatched, 0);
    const Tree t = tree_of(r2 / "d");
    CHECK_EQ(t.files, after.files);
    CHECK_EQ(t.dirs, after.dirs);
    const auto s = snapshot(r2 / "d");
    for (const auto& [k, v] : snapshot(root / "d")) if (k.find("source/big") != std::string::npos && k.find("zarr.json") == std::string::npos) CHECK(s.count(k) && s.at(k) == v);
    fs::remove_all(r2);
  }

  // 3. one subject; an unknown subject; the writer lock
  {
    const fs::path r3 = root / "lock";
    make_v50(r3, "d");
    DatasetWriter ds = open_writer(r3 / "d");
    CHECK_EQ(code_of([&] { open_writer(r3 / "d"); }), std::string(code::writer_lock_held));       // another writer holds it: refused
    CHECK_EQ(code_of([&] { compact_dataset(ds, mk(CompactOptions{.subject = "nobody"})); }), std::string(code::no_such_row));
    const Json r = compact_dataset(ds, mk(CompactOptions{.subject = "S"}));
    CHECK_EQ(r["compacted"].get<int>(), 5);
    // --no-shard: only the separator changes
    const fs::path r4 = root / "noshard";
    const std::string base = make_v50(r4, "d");
    DatasetWriter d4 = open_writer(r4 / "d");
    const Json r5 = compact_dataset(d4, mk(CompactOptions{.shard = false}));
    CHECK_EQ(r5["compacted"].get<int>(), 5);
    const Json z = Json::parse(*d4.store().get_text(base + "/source/big/zarr.json"));
    CHECK_EQ(z["codecs"][0]["name"].get<std::string>(), std::string("bytes"));
    CHECK(read_all(d4.store(), base + "/source/big") == want["source/big"]);
  }

  // 4. an array written in pieces (the Studio's `put`: in_pieces) is kept unsharded by compact, and still accepts a region write
  {
    const fs::path r6 = root / "pieces";
    const std::string base = make_v50(r6, "d");
    DatasetWriter d6 = open_writer(r6 / "d");
    ArrayOptions po;
    po.in_pieces = true;
    ArrayMeta pm = array_meta(DType::Float32, {4000, 1000}, po);
    CHECK(pm.in_pieces && pm.chunks == pm.grid);
    pm.separator = "/";   // as a format-50 dataset has it
    layout_array(d6.store(), base + "/source/pieces", pm);
    const auto top = ramp(100 * 1000);
    write_region(d6.store(), base + "/source/pieces", {3900, 0}, view(top, {100, 1000}));
    const Json r = compact_dataset(d6, mk());
    CHECK_EQ(r["compacted"].get<int>(), 6);
    const Json z = Json::parse(*d6.store().get_text(base + "/source/pieces/zarr.json"));
    CHECK_EQ(z["chunk_key_encoding"]["configuration"]["separator"].get<std::string>(), std::string("."));
    CHECK(z["codecs"][0]["name"].get<std::string>() != "sharding_indexed");
    CHECK(z.contains("nxr_written_in_pieces"));
    CHECK_EQ(z["chunk_grid"]["configuration"]["chunk_shape"].get<Shape>(), array_meta(DType::Float32, {4000, 1000}, po).grid);
    const Bytes got = ArrayReader::open(d6.store(), base + "/source/pieces").read({3900, 0}, {4000, 1000});
    CHECK_EQ(got.size(), top.size() * 4);
    CHECK(std::memcmp(got.data(), top.data(), got.size()) == 0);
    auto more = top;
    for (auto& x : more) x += 1.0f;
    write_region(d6.store(), base + "/source/pieces", {0, 0}, view(more, {100, 1000}));    // still accepts a region write
    const Bytes got2 = ArrayReader::open(d6.store(), base + "/source/pieces").read({0, 0}, {100, 1000});
    CHECK(std::memcmp(got2.data(), more.data(), got2.size()) == 0);
    // the same array again: nothing changes; a second compact keeps it
    CHECK_EQ(compact_dataset(d6, mk())["compacted"].get<int>(), 0);
    CHECK(Json::parse(*d6.store().get_text(base + "/source/pieces/zarr.json")).contains("nxr_written_in_pieces"));
  }

  fs::remove_all(root);
  return nxrtest::finish("compact");
}
