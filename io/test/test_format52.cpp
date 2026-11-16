// Format 52 (the small-file store at the dataset level): a synthetic dataset written at 52 reads back the same keys and bytes as the same data at 51, with a
// subject's files and folders cut to its large shards plus one database; `dataset compact` moves a 51 dataset to 52 with every key verified, and a crash anywhere
// in the move is finished by the next run; a node that fails or is removed leaves no key behind in either place. Synthetic data only.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <stdexcept>

#include "check.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/compact.h"
#include "nxr/io/database.h"
#include "nxr/io/dataset.h"
#include "nxr/io/errors.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::vector<float> ramp(std::int64_t n, std::uint64_t seed) {
  std::vector<float> v(static_cast<std::size_t>(n));
  std::uint64_t x = seed;
  for (auto& e : v) { x = (x * 6364136223846793005ULL + 1442695040888963407ULL); e = static_cast<float>(x >> 40) / 1024.0f; }
  return v;
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

// one subject: a surface, 24 small fields, a medium field (one chunk file near 2 MB) and a big one (a 16 MB shard), complete
std::string build(const fs::path& datastore, int store_format) {
  CreateOptions co;
  co.store_format = store_format;
  DatasetWriter ds = create_dataset(datastore, "d", co);
  SubjectWriter sub = SubjectWriter::create(ds, "S");
  const std::vector<double> v(12, 1.0);
  const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
  ManifoldSpec m = tri();
  m.populate = [&](Store&, const std::string&) {
    sub.write_array("source/m/vertices", view(v, {4, 3}));
    sub.write_array("source/m/faces", view(f, {2, 3}));
  };
  const std::string mid = sub.manifold(m);
  std::vector<std::vector<float>> keep;
  auto field = [&](const std::string& name, const std::vector<float>& data, Shape shape) {
    keep.push_back(data);
    FieldSpec s;
    s.name = name;
    s.path = "source/" + name;
    s.kind = "map";
    s.manifold_id = mid;
    s.data = view(keep.back(), shape);
    sub.field(s);
  };
  for (int i = 0; i < 24; ++i) field("small" + std::to_string(i), ramp(4, 100 + static_cast<std::uint64_t>(i)), {4});
  field("medium", ramp(600 * 1000, 7), {600, 1000});
  field("big", ramp(4000 * 1000, 9), {4000, 1000});
  sub.complete();
  return store_path_of(ds.db(), sub.id());
}

// every key under `prefix` with its bytes
void keys_of(const Store& st, const std::string& prefix, std::map<std::string, Bytes>& out) {
  const Listing l = st.list(prefix);
  for (const auto& k : l.keys) out[join_key(prefix, k)] = *st.get(join_key(prefix, k));
  for (const auto& p : l.prefixes) keys_of(st, join_key(prefix, p), out);
}

struct Disk { std::int64_t files = 0, dirs = 0, small_files = 0; };
Disk disk_of(const fs::path& p) {
  Disk d;
  for (const auto& e : fs::recursive_directory_iterator(p)) {
    if (e.is_directory()) { ++d.dirs; continue; }
    ++d.files;
    const std::string n = e.path().filename().string();
    if (n.rfind(kSmallFilesFile, 0) != 0 && e.file_size() < kSmallFileThreshold) ++d.small_files;   // a small value left as a file
  }
  return d;
}

std::map<std::string, Bytes> snapshot(const fs::path& datastore) {
  DatasetWriter ds = open_writer(datastore / "d", OpenOptions{.lock = true, .recover = false});
  std::map<std::string, Bytes> m;
  keys_of(ds.store(), "d/S.nxr.zarr", m);
  return m;
}

bool same_but_ids(const std::map<std::string, Bytes>& a, const std::map<std::string, Bytes>& b) {
  if (a.size() != b.size()) return false;
  for (const auto& [k, v] : a) {
    const auto it = b.find(k);
    if (it == b.end()) return false;
    if (k.size() >= 9 && k.compare(k.size() - 9, 9, "zarr.json") == 0) continue;   // carries the node's id: another dataset has another
    if (it->second != v) return false;
  }
  return true;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / "nxr_io_test_format52";
  fs::remove_all(root);

  // 1. the same data at 51 and at 52: the same keys and bytes; far fewer files and folders at 52
  const fs::path r51 = root / "f51", r52 = root / "f52";
  const std::string base = build(r51, 51);
  CHECK_EQ(build(r52, 52), base);
  Disk d51, d52;
  {
    DatasetWriter a = open_writer(r51 / "d"), b = open_writer(r52 / "d");
    CHECK_EQ(store_format_number(a.db()), 51);
    CHECK_EQ(store_format_number(b.db()), 52);
    std::map<std::string, Bytes> ka, kb;
    keys_of(a.store(), base, ka);
    keys_of(b.store(), base, kb);
    CHECK(!ka.empty());
    CHECK(same_but_ids(ka, kb));
    CHECK_EQ(text_parity(b.db(), b.store()).mismatched, 0);
    for (const char* arr : {"source/small3", "source/medium", "source/big"}) {
      const Bytes x = ArrayReader::open(a.store(), base + "/" + arr).read(), y = ArrayReader::open(b.store(), base + "/" + arr).read();
      CHECK(!x.empty() && x == y);
    }
    // a region read through a shard (a ranged get on a file) at 52
    const ArrayReader big = ArrayReader::open(b.store(), base + "/source/big");
    CHECK(big.read({100, 0}, {110, 1000}) == ArrayReader::open(a.store(), base + "/source/big").read({100, 0}, {110, 1000}));
  }
  d51 = disk_of(r51 / "d" / "S.nxr.zarr");
  d52 = disk_of(r52 / "d" / "S.nxr.zarr");
  std::printf("  synthetic subject, files / folders under the subject's folder: format 51 = %lld / %lld, format 52 = %lld / %lld (small values left as files at 52: %lld)\n",
              static_cast<long long>(d51.files), static_cast<long long>(d51.dirs), static_cast<long long>(d52.files), static_cast<long long>(d52.dirs), static_cast<long long>(d52.small_files));
  CHECK(d52.files < d51.files / 4 && d52.dirs < d51.dirs / 4);
  CHECK_EQ(d52.small_files, 0);
  CHECK(fs::exists(r52 / "d" / "S.nxr.zarr" / kSmallFilesFile) && !fs::exists(r51 / "d" / "S.nxr.zarr" / kSmallFilesFile));
  CHECK_EQ(d52.files, 3);   // the database, the medium chunk, the big shard: at rest one database file, no WAL or shared-memory file
  // a reader (no writer, no lock) sees the same, and a writer reopens it
  {
    const Dataset rd = Dataset::open(r52 / "d");
    DatastoreStore datastore(r52, SmallFileOptions{kSmallFileThreshold, true});
    PrefixStore keys(datastore, "d");
    const ArrayReader a = ArrayReader::open(keys, "S.nxr.zarr/source/small3");
    CHECK(!a.read().empty() && rd.query("SELECT name FROM subject").size() == 1);
    DatasetWriter ds = open_writer(r52 / "d");
    SubjectWriter sub = SubjectWriter::open(ds, "S");
    CHECK(!ArrayReader::open(sub.store(), "source/medium").read().empty());
  }

  // 2. a node that fails, and a node removed, leave no key in the table or the folder
  {
    DatasetWriter ds = open_writer(r52 / "d");
    SubjectWriter sub = SubjectWriter::open(ds, "S", OpenSubjectOptions{.immediate = true});
    const std::vector<double> v(12, 2.0);
    ManifoldSpec m = tri();
    m.name = "boom";
    m.path = "source/boom";
    m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/boom/faces"}}};
    m.geometry->positions_path = "source/boom/vertices";
    m.populate = [&](Store&, const std::string&) {
      sub.write_array("source/boom/vertices", view(v, {4, 3}));
      throw std::runtime_error("stop");
    };
    try { sub.manifold(m); } catch (const std::runtime_error&) {}
    CHECK(!sub.store().exists_prefix("source/boom"));
    const auto row = sub.by_name("field", "small5");
    CHECK(row.has_value() && sub.store().exists_prefix("source/small5"));
    sub.remove("field", text(*row, "id"));
    CHECK(!sub.store().exists_prefix("source/small5") && sub.store().exists_prefix("source/small6"));
  }

  // 3. compact 51 -> 52: every key the same before and after, the small files gone, the format set, a second run changes nothing
  {
    const fs::path r = root / "compact";
    build(r, 51);
    const auto before = snapshot(r);
    const Disk db = disk_of(r / "d" / "S.nxr.zarr");
    {   // a dry run changes nothing and reports what it would move
      DatasetWriter ds = open_writer(r / "d", OpenOptions{.lock = true, .recover = false});
      const Json dry = compact_dataset(ds, CompactOptions{.dry_run = true});
      CHECK(dry["small_keys_moved"].get<std::int64_t>() > 0);
      CHECK_EQ(store_format_number(ds.db()), 51);
    }
    CHECK(!fs::exists(r / "d" / "S.nxr.zarr" / kSmallFilesFile));
    Json rep;
    {
      DatasetWriter ds = open_writer(r / "d", OpenOptions{.lock = true, .recover = false});
      rep = compact_dataset(ds);
      CHECK_EQ(store_format_number(ds.db()), 52);
      CHECK(rep["store_format_set"].get<bool>());
    }
    CHECK(snapshot(r) == before);   // every key, every byte (zarr.json too: no row changed)
    const Disk da = disk_of(r / "d" / "S.nxr.zarr");
    std::printf("  compact 51 -> 52 of the same subject: files %lld -> %lld, folders %lld -> %lld (report: %lld -> %lld files, %lld small keys moved)\n", static_cast<long long>(db.files),
                static_cast<long long>(da.files), static_cast<long long>(db.dirs), static_cast<long long>(da.dirs), rep["files_on_disk_before"].get<long long>(),
                rep["files_on_disk_after"].get<long long>(), rep["small_keys_moved"].get<long long>());
    CHECK_EQ(da.files, d52.files);
    CHECK_EQ(da.dirs, d52.dirs);
    CHECK_EQ(da.small_files, 0);
    CHECK_EQ(rep["files_on_disk_after"].get<std::int64_t>(), da.files);
    CHECK_EQ(rep["folders_on_disk_after"].get<std::int64_t>(), da.dirs);
    CHECK_EQ(rep["files_on_disk_before"].get<std::int64_t>(), db.files);
    {
      DatasetWriter ds = open_writer(r / "d");
      CHECK_EQ(text_parity(ds.db(), ds.store()).mismatched, 0);
      const Json again = compact_dataset(ds);
      CHECK_EQ(again["small_keys_moved"].get<std::int64_t>(), 0);
      CHECK_EQ(again["compacted"].get<int>(), 0);
    }
    CHECK(snapshot(r) == before);
  }

  // 4. a crash in the move: after the first batch is in the table (the files still there), and after the files are removed; the next run finishes
  for (const char* crash : {"small_files_written", "small_files_removed"}) {
    const fs::path r = root / (std::string("crash_") + crash);
    build(r, 51);
    const auto before = snapshot(r);
    {
      DatasetWriter ds = open_writer(r / "d", OpenOptions{.lock = true, .recover = false});
      CompactOptions o;
      o.on_stage = [&](const std::string& s, const std::string&) { if (s == crash) throw std::runtime_error("crash"); };
      bool crashed = false;
      try { compact_dataset(ds, o); } catch (const std::runtime_error&) { crashed = true; }
      CHECK(crashed);
      CHECK_EQ(store_format_number(ds.db()), 51);   // not yet 52
    }
    CHECK(snapshot(r) == before);   // read through the table-first store: the same keys, the same bytes, whichever place holds them
    {
      DatasetWriter ds = open_writer(r / "d", OpenOptions{.lock = true, .recover = false});
      const Json fin = compact_dataset(ds);
      CHECK(fin["store_format_set"].get<bool>());
      CHECK_EQ(store_format_number(ds.db()), 52);
    }
    CHECK(snapshot(r) == before);
    const Disk da = disk_of(r / "d" / "S.nxr.zarr");
    CHECK_EQ(da.files, d52.files);
    CHECK_EQ(da.dirs, d52.dirs);
    CHECK_EQ(da.small_files, 0);
  }

  if (!std::getenv("NXR_KEEP_TEST_FILES")) fs::remove_all(root);
  return nxrtest::finish("format52");
}
