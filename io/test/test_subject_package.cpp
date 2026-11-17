// The subject package (store format 53), on synthetic data: one folder per subject, rows and small keys in one database, a digest of every key, the content hash, the
// per-package writer lock, recovery, the bundle equal to a dataset's, ids that clash.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <regex>

#include "check.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/bundle.h"
#include "nxr/io/hash.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}

ManifoldSpec triangle(const std::string& name) {
  ManifoldSpec m;
  m.name = name;
  m.path = "source/" + name;
  m.type = "surface";
  m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/" + name + "/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = "scs";
  g.positions_path = "source/" + name + "/vertices";
  m.geometry = g;
  return m;
}

// The same synthetic content into a subject of a dataset or of a package: a surface, a small field (a row of the small keys) and a large one (a file).
void add_content(SubjectWriter& sub, const std::string& tag) {
  const std::vector<double> v(12, 1.0);
  const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
  ManifoldSpec m = triangle("m" + tag);
  m.populate = [&, tag](Store&, const std::string&) {
    sub.write_array("source/m" + tag + "/vertices", view(v, {4, 3}));
    sub.write_array("source/m" + tag + "/faces", view(f, {2, 3}));
  };
  const std::string mid = sub.manifold(m);
  const std::vector<float> small(4, 2.0f);
  FieldSpec fs_;
  fs_.name = "small" + tag; fs_.path = "source/small" + tag; fs_.kind = "map"; fs_.manifold_id = mid; fs_.data = view(small, {4});
  sub.field(fs_);
  std::vector<double> big(400000);                    // 3.2 MB: its chunks are over the small-key threshold, so they are files
  for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<double>(i % 977) * 0.25;
  FieldSpec fb;
  fb.name = "big" + tag; fb.path = "source/big" + tag; fb.kind = "map"; fb.manifold_id = mid; fb.data = view(big, {400000});
  sub.field(fb);
}

Json normalised(const Json& j) {
  static const std::regex uuid("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$");
  if (j.is_string()) return std::regex_match(j.get<std::string>(), uuid) ? Json("<id>") : j;
  if (j.is_array()) { Json a = Json::array(); for (const auto& x : j) a.push_back(normalised(x)); return a; }
  if (j.is_object()) {
    Json o = Json::object();
    for (auto it = j.begin(); it != j.end(); ++it) {
      const std::string k = it.key();
      if (k.size() > 4 && k.compare(k.size() - 4, 4, "_utc") == 0) o[k] = "<time>";
      else if (k == "datastore") continue;     // the hint row: where each was made
      else o[k] = normalised(it.value());
    }
    return o;
  }
  return j;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_subject_package_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  CreatePackageOptions opt;
  opt.home_dataset = "demo";

  // a package is one folder: its database, its large shards, and nothing else once the writer is closed
  const fs::path pkg = root / "S1.nxr.zarr";
  std::string hash1;
  std::string subject_id;
  {
    CHECK_EQ(code_of([&] { CreatePackageOptions bad; create_package(root / "x.nxr.zarr", "x", bad); }), std::string(code::invalid_argument));   // the home dataset is named
    CHECK(!fs::exists(root / "x.nxr.zarr"));
    DatasetWriter w = create_package(pkg, "S1", opt);
    CHECK(w.is_package());
    CHECK(fs::exists(pkg / ".writer.lock"));
    CHECK_EQ(code_of([&] { open_package_writer(pkg); }), std::string(code::writer_lock_held));                 // one writer a package
    CHECK_EQ(code_of([&] { create_package(pkg, "S1", opt); }), std::string(code::already_exists));
    CHECK_EQ(code_of([&] { w.store(); }), std::string(code::invalid_argument));                              // no datastore here
    SubjectWriter sub = SubjectWriter::open(w, "S1");
    add_content(sub, "");
    subject_id = sub.id();
    sub.complete();
    CHECK_EQ(w.db().one("SELECT count(*) AS n FROM field")->at("n"), Value(std::int64_t(2)));
  }
  CHECK(!fs::exists(pkg / ".writer.lock"));
  CHECK(fs::exists(pkg / kPackageDatabaseFile));
  CHECK(!fs::exists(pkg / "small_files.sqlite"));
  CHECK(!fs::exists(pkg / "subject.sqlite-journal"));
  CHECK(!fs::exists(pkg / "subject.sqlite-wal"));
  {
    SubjectPackage p = SubjectPackage::open(pkg);
    const PackageRecord rec = p.record();
    CHECK_EQ(rec.subject_id, subject_id);
    CHECK_EQ(rec.store_format, 53);
    CHECK_EQ(rec.kind, std::string("subject"));
    CHECK_EQ(rec.content_hash.size(), std::size_t(64));
    hash1 = rec.content_hash;
    CHECK_EQ(p.content_hash(), hash1);
    // every key has its size and SHA-256; the keys are readable alone, a large chunk is a file and a zarr.json a row
    const VerifyReport v = p.verify();
    CHECK(v.ok);
    for (const auto& s : v.problems) std::printf("  problem: %s\n", s.c_str());
    CHECK(p.store().exists("source/small/zarr.json"));
    CHECK(p.store().exists("zarr.json"));
    const auto keys = list_all_keys(p.store());
    CHECK(keys.size() > 6);
    CHECK_EQ(static_cast<std::int64_t>(keys.size()), integer(*p.db().one("SELECT count(*) AS n FROM key_digest"), "n"));
    bool file_chunk = false;
    for (const auto& k : keys) if (fs::is_regular_file(pkg / k) && k.rfind("source/big", 0) == 0) file_chunk = true;
    CHECK(file_chunk);
    CHECK(p.bundle().has_value());
    CHECK(Json::parse(*p.store().get_text("source/small/zarr.json")).contains("attributes"));
  }

  // the bundle of a package equals the bundle of the same subject written into a dataset (ids and times aside)
  {
    DatasetWriter ds = create_dataset(root / "store", "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1", SubjectOptions{.default_template = false});
    add_content(sub, "");
    sub.complete();
    const Json from_dataset = *subject_bundle(ds.db(), sub.id());
    const Json from_package = *SubjectPackage::open(pkg).bundle();
    const Json a = normalised(from_dataset), b = normalised(from_package);
    for (auto it = a.begin(); it != a.end(); ++it)
      if (!b.contains(it.key()) || b[it.key()] != it.value()) std::printf("  bundle differs at %s\n    dataset: %s\n    package: %s\n", it.key().c_str(), it.value().dump().substr(0, 600).c_str(), b.contains(it.key()) ? b[it.key()].dump().substr(0, 600).c_str() : "(absent)");
    CHECK(a == b);
    for (const char* k : {"manifolds", "fields", "topologies"}) CHECK(from_package.contains(k));
    CHECK(!from_package["fields"].empty());
  }

  // a writer that changes nothing leaves the hash; one that writes changes it
  {
    { DatasetWriter w = open_package_writer(pkg); }
    CHECK_EQ(SubjectPackage::open(pkg).record().content_hash, hash1);
    {
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, "S1");
      const std::vector<float> more(3, 7.0f);
      ManifoldSpec ignored;
      const auto m = sub.by_name("manifold", "m");
      FieldSpec f;
      f.name = "more"; f.path = "source/more"; f.kind = "map"; f.manifold_id = text(*m, "id"); f.data = view(more, {3});
      sub.field(f);
      sub.complete();
    }
    const std::string hash2 = SubjectPackage::open(pkg).record().content_hash;
    CHECK(hash2 != hash1);
    CHECK(SubjectPackage::open(pkg).verify().ok);
    // removing it brings the keys back, and the rows, to what they were except the time of writing: the hash is a state, not a count
    {
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, "S1");
      sub.remove("field", text(*sub.by_name("field", "more"), "id"));
      CHECK(!sub.store().exists_prefix("source/more"));
    }
    CHECK(SubjectPackage::open(pkg).verify().ok);
    hash1 = SubjectPackage::open(pkg).record().content_hash;
  }

  // the hash is sensitive to the bytes: a changed shard is named by key
  {
    const fs::path copy = root / "S1-copy.nxr.zarr";
    fs::copy(pkg, copy, fs::copy_options::recursive);
    CHECK(SubjectPackage::open(copy).verify().ok);                               // a copy is the same package: same hash
    CHECK_EQ(SubjectPackage::open(copy).record().content_hash, hash1);
    std::string victim;
    for (const auto& k : list_all_keys(SubjectPackage::open(copy).store())) if (fs::is_regular_file(copy / k) && k.rfind("source/big", 0) == 0) { victim = k; break; }
    CHECK(!victim.empty());
    {
      std::fstream f(copy / victim, std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(40);
      char c = 0;
      f.read(&c, 1);
      f.seekp(40);
      c = static_cast<char>(c ^ 0x55);
      f.write(&c, 1);
    }
    const VerifyReport v = SubjectPackage::open(copy).verify();
    CHECK(!v.ok);
    bool named = false;
    for (const auto& s : v.problems) if (s.rfind(victim, 0) == 0) named = true;
    CHECK(named);
  }

  // recovery: a node left pending by a writer that stopped is rolled back, bytes and rows, at the next open
  {
    {
      DatasetWriter w = open_package_writer(pkg);
      SubjectWriter sub = SubjectWriter::open(w, "S1");
      const std::vector<float> lost(3, 9.0f);
      FieldSpec f;
      f.name = "lost"; f.path = "source/lost"; f.kind = "map"; f.manifold_id = text(*sub.by_name("manifold", "m"), "id"); f.data = view(lost, {3});
      sub.field(f);                       // pending: never completed
      CHECK(sub.store().exists("source/lost/zarr.json"));
    }
    DatasetWriter w = open_package_writer(pkg);
    CHECK(!w.db().one("SELECT 1 FROM field WHERE name = 'lost'").has_value());
    CHECK(!w.node_store().exists_prefix("demo/S1.nxr.zarr/source/lost"));
  }
  CHECK(SubjectPackage::open(pkg).verify().ok);
  CHECK_EQ(SubjectPackage::open(pkg).record().content_hash, hash1);

  // two packages, two writers at once: they share no file they write
  {
    DatasetWriter a = create_package(root / "A.nxr.zarr", "A", opt);
    DatasetWriter b = create_package(root / "B.nxr.zarr", "B", opt);
    CHECK(a.is_package() && b.is_package());
    CHECK(fs::exists(root / "A.nxr.zarr" / ".writer.lock") && fs::exists(root / "B.nxr.zarr" / ".writer.lock"));
  }

  // a folder that is no package, a package from the future
  CHECK_EQ(code_of([&] { SubjectPackage::open(root / "store"); }), std::string(code::not_found));
  {
    const fs::path future = root / "future.nxr.zarr";
    fs::copy(pkg, future, fs::copy_options::recursive);
    { Database d = Database::open(future / kPackageDatabaseFile); d.execute("UPDATE meta SET value = '54: later' WHERE key = 'store_format'"); }
    CHECK_EQ(code_of([&] { SubjectPackage::open(future); }), std::string(code::schema_version));
    CHECK_EQ(code_of([&] { open_package_writer(future); }), std::string(code::schema_version));
    CHECK(!fs::exists(future / ".writer.lock"));
    const fs::path old = root / "format52.nxr.zarr";
    fs::copy(pkg, old, fs::copy_options::recursive);
    { Database d = Database::open(old / kPackageDatabaseFile); d.execute("UPDATE meta SET value = '52: a dataset database' WHERE key = 'store_format'"); }
    CHECK_EQ(code_of([&] { SubjectPackage::open(old); }), std::string(code::not_a_database));
  }

  // ids: independent packages do not clash; two made from one counter's start do, and the home rows never do
  {
    CHECK(find_id_clashes({root / "A.nxr.zarr", root / "B.nxr.zarr", pkg}).empty());
    testing::deterministic(true, 0);
    { DatasetWriter w = create_package(root / "C.nxr.zarr", "C", opt); }
    testing::deterministic(true, 0);
    { DatasetWriter w = create_package(root / "D.nxr.zarr", "D", opt); }
    testing::deterministic(false);
    const auto clashes = find_id_clashes({root / "C.nxr.zarr", root / "D.nxr.zarr"});
    CHECK(!clashes.empty());
    for (const auto& c : clashes) CHECK(c.table != "dataset" && c.table != "datastore");
    // a row a package lists as an anchor (a template's, by design another package's id) is not a clash
    for (const char* name : {"C", "D"}) {
      Database d = Database::open(root / (std::string(name) + ".nxr.zarr") / kPackageDatabaseFile);
      d.execute("INSERT INTO anchor_row (table_name, row_id, from_subject_id) SELECT 'subject', id, id FROM subject");
    }
    CHECK(find_id_clashes({root / "C.nxr.zarr", root / "D.nxr.zarr"}).size() < clashes.size());
  }

  // SHA-256 in pieces equals SHA-256 whole
  {
    const std::string data(1000, 'q');
    Sha256 h;
    h.update(data.substr(0, 63));
    h.update(data.substr(63, 700));
    h.update(data.substr(763));
    CHECK_EQ(h.hex(), sha256_hex(data));
    CHECK_EQ(Sha256().hex(), sha256_hex(""));
  }

  fs::remove_all(root);
  return nxrtest::finish("subject_package");
}
