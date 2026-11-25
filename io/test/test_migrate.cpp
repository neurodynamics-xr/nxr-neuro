// `dataset split` and `dataset join` on synthetic datasets written by the current writer: a one-database dataset with a template, subjects registered to it, a group
// subject and large shards becomes subject packages with a catalogue, verified row for row and byte for byte, resumable at every stage, and back.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <map>

#include "check.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/bundle.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/hash.h"
#include "nxr/io/migrate.h"
#include "nxr/io/writer.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string what_of(const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { return e.what(); }
  return "";
}

std::string file_sha(const fs::path& p) {
  std::ifstream i(p, std::ios::binary);
  Sha256 h;
  std::vector<char> buf(1 << 20);
  while (i) { i.read(buf.data(), static_cast<std::streamsize>(buf.size())); h.update(reinterpret_cast<const std::uint8_t*>(buf.data()), static_cast<std::size_t>(i.gcount())); }
  return h.hex();
}

// every file of a folder tree, with its hash: "a second run changes no byte"
std::map<std::string, std::string> tree(const fs::path& root) {
  std::map<std::string, std::string> out;
  for (const auto& e : fs::recursive_directory_iterator(root))
    if (e.is_regular_file()) out[fs::relative(e.path(), root).generic_string()] = file_sha(e.path());
  return out;
}

ManifoldSpec triangle(const std::string& name) {
  ManifoldSpec m;
  m.name = name; m.path = "source/" + name; m.type = "surface"; m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/" + name + "/faces"}}};
  GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "source/" + name + "/vertices";
  m.geometry = g;
  return m;
}

std::string add_surface(SubjectWriter& sub, const std::string& name) {
  static const std::vector<double> v(12, 1.0);
  static const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
  ManifoldSpec m = triangle(name);
  m.populate = [&sub, name](Store&, const std::string&) { sub.write_array("source/" + name + "/vertices", view(v, {4, 3})); sub.write_array("source/" + name + "/faces", view(f, {2, 3})); };
  return sub.manifold(m);
}

std::string add_field(SubjectWriter& sub, const std::string& name, const std::string& manifold_id, std::size_t n) {
  std::vector<double> a(n);
  for (std::size_t i = 0; i < n; ++i) a[i] = static_cast<double>(i % 977) * 0.25;
  FieldSpec f;
  f.name = name; f.path = "source/" + name; f.kind = "map"; f.manifold_id = manifold_id; f.data = view(a, {static_cast<std::int64_t>(n)});
  return sub.field(f);
}

// A one-database dataset: a template (a surface, a subject-scope dictionary of tiles, a large field), two subjects registered to it (each with a field and an
// operator on the template's surface, a selection coded by the template's dictionary, a large field) (no group subject: format 56; the template holds the cohort).
void make_old(const fs::path& datastore, const std::string& name, int store_format) {
  CreateOptions co;
  co.store_format = store_format;
  DatasetWriter ds = create_dataset(datastore, name, co);
  SubjectOptions topt;
  topt.kind = "template";
  SubjectWriter t = SubjectWriter::create(ds, "@default_subject", topt);
  const std::string tm = add_surface(t, "cortex");
  add_field(t, "template_map", tm, 400000);
  DictionarySpec tiles;
  tiles.name = "tiles"; tiles.scope = "subject"; tiles.names = {"left", "right"};
  const DictionaryResult tile_dict = t.dictionary(tiles);
  DictionarySpec parcels;
  parcels.name = "Parcels"; parcels.scope = "app"; parcels.names = {"alpha", "beta"};
  t.dictionary(parcels);
  t.complete();
  std::map<std::string, std::string> small_of;
  for (const char* sname : {"S1", "S2"}) {
    SubjectWriter s = SubjectWriter::create(ds, sname);
    const std::string own = add_surface(s, "cortex");
    small_of[sname] = add_field(s, "small", own, 16);
    add_field(s, "on_template", tm, 4);
    add_field(s, "big", own, 400000);
    const std::vector<std::int32_t> codes = {0, 1, 0, 1};
    SelectionSpec sel;
    sel.name = "tiles"; sel.path = "source/tiles"; sel.type = "set"; sel.manifold_id = tm; sel.data = view(codes, {4});
    sel.cols.set("dictionary_id", tile_dict.id).set("cell", "0");
    s.selection(sel);
    OperatorSpec o;
    const std::vector<double> op(12, 0.5);
    o.name = "registration"; o.path = "registration"; o.kind = "transition"; o.from_manifold_id = own; o.n_rows = 3; o.n_cols = 4; o.data = view(op, {3, 4});
    o.cols.set("to_manifold_id", tm);
    s.operator_(o);
    s.complete();
  }
}

// every subject's keys with their SHA-256: from the format 52 table or from a package
std::map<std::string, std::map<std::string, std::string>> key_snapshot(const fs::path& folder) {
  std::map<std::string, std::map<std::string, std::string>> out;
  for (const auto& e : fs::directory_iterator(folder)) {
    const std::string n = e.path().filename().string();
    if (!e.is_directory() || n.find(".nxr.zarr") == std::string::npos || n.find(".replaced") != std::string::npos) continue;
    FolderStore files(e.path());
    std::unique_ptr<Database> db;
    std::unique_ptr<SmallFileStore> small;
    if (fs::is_regular_file(e.path() / kPackageDatabaseFile)) {
      db = std::make_unique<Database>(Database::open(e.path() / kPackageDatabaseFile, false, "", true));
      small = std::make_unique<SmallFileStore>(files, *db, SmallFileOptions{kSmallFileThreshold, true});
    } else {
      small = std::make_unique<SmallFileStore>(files, e.path() / kSmallFilesFile, SmallFileOptions{kSmallFileThreshold, true});
    }
    for (const auto& k : list_all_keys(*small)) { const auto b = small->get(k); out[n][k] = sha256_hex(std::string(b->begin(), b->end())); }
  }
  return out;
}

std::map<std::string, std::string> package_hashes(const fs::path& folder) {
  std::map<std::string, std::string> out;
  const Catalogue c = Catalogue::open(folder);
  for (const auto& e : c.entries()) out[e.name] = e.content_hash;
  return out;
}

bool same_database(const fs::path& a, const fs::path& b, std::string& why) {
  Database da = Database::open(a, false, "", true);
  da.execute("ATTACH DATABASE ? AS other", {b.string()});
  for (const auto& r : da.all("SELECT name FROM main.sqlite_master WHERE type = 'table' ORDER BY name")) {
    const std::string t = text(r, "name");
    if (t == "meta" || t == "sync_job") continue;   // what a database says about itself, and the outbox: not the dataset's rows
    const std::string l = "\"" + t + "\"";
    if (!da.one("SELECT 1 FROM other.sqlite_master WHERE type = 'table' AND name = ?", {t})) { why = "table " + t + " missing"; return false; }
    for (const char* d : {"main", "other"}) {
      const std::string o = std::string(d) == "main" ? "other" : "main";
      if (integer(*da.one("SELECT count(*) AS n FROM (SELECT * FROM " + std::string(d) + "." + l + " EXCEPT SELECT * FROM " + o + "." + l + ")"), "n") != 0) { why = "rows of " + t + " differ"; return false; }
    }
  }
  return true;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_migrate_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  SplitOptions so;
  so.registered_by = "test";

  make_old(root, "demo", 52);
  const fs::path ds = root / "demo";
  fs::copy_file(ds / kDatasetFile, root / "original.sqlite");
  const auto keys_before = key_snapshot(ds);
  CHECK_EQ(keys_before.size(), std::size_t(3));
  const auto shards_before = tree(ds);

  // a dry run changes nothing
  {
    SplitOptions dry = so;
    dry.dry_run = true;
    const Json plan = split_dataset(ds, dry);
    CHECK(plan["dry_run"] == true);
    CHECK_EQ(plan["subjects"].size(), std::size_t(3));
    CHECK(tree(ds) == shards_before);
  }
  CHECK(what_of([&] { split_dataset(ds, SplitOptions{}); }).find("registered_by") != std::string::npos);

  // a copy for each place a crash can stop, made now
  for (const char* stage : {"built", "verified", "committed", "registered"}) fs::copy(ds, root / (std::string("crash-") + stage), fs::copy_options::recursive);
  fs::copy(ds, root / "clean", fs::copy_options::recursive);

  // SPLIT
  const Json rep = split_dataset(ds, so);
  CHECK(!fs::exists(ds / kDatasetFile));
  CHECK(fs::exists(ds / "dataset.sqlite.format52"));
  CHECK(fs::exists(ds / kCatalogueFile) && fs::exists(ds / kIndexFile));
  CHECK_EQ(rep["subjects"].size(), std::size_t(3));
  for (const auto& s : rep["subjects"]) { CHECK(s["outcome"] == "migrated"); CHECK(s["keys"].get<int>() > 1); }
  const Catalogue cat = Catalogue::open(ds);
  CHECK_EQ(cat.entries().size(), std::size_t(3));
  CHECK_EQ(cat.find("@default_subject")->kind, std::string("template"));
  // each package: whole, its own bundle the original's, its template named, its anchors listed
  Database original = Database::open(ds / "dataset.sqlite.format52", false, "", true);
  for (const auto& e : cat.entries()) {
    const SubjectPackage p = SubjectPackage::open(cat.package_path(e));
    const VerifyReport v = p.verify();
    CHECK(v.ok);
    for (const auto& s : v.problems) std::printf("  %s: %s\n", e.name.c_str(), s.c_str());
    {   // the bundle is the original's but for the subject's template link (format 56: a subject names no template)
      Json a = *p.bundle(), b = *subject_bundle(original, e.subject_id);
      for (Json* x : {&a, &b}) if (x->contains("subject") && (*x)["subject"].is_object()) (*x)["subject"].erase("template_id");
      CHECK(a == b);
    }
    CHECK(!fs::exists(cat.package_path(e) / kSmallFilesFile));
    CHECK(fs::exists(cat.package_path(e) / "small_files.sqlite.format52"));
    CHECK(p.db().all("PRAGMA foreign_key_check").empty());
    const std::int64_t anchors = integer(*p.db().one("SELECT count(*) AS n FROM anchor_row"), "n");
    if (e.name == "S1" || e.name == "S2") {
      CHECK(p.record().template_reference.is_null() && is_null(p.subject_row(), "template_id"));   // format 56: a subject names no template
      CHECK(anchors > 0);
      CHECK(p.db().one("SELECT 1 FROM anchor_row WHERE table_name = 'manifold'").has_value());
      CHECK(p.db().one("SELECT 1 FROM anchor_row WHERE table_name = 'dictionary'").has_value());                 // the template's tiles
      CHECK(integer(*p.db().one("SELECT count(*) AS n FROM dictionary_entry e JOIN dictionary d ON d.id = e.dictionary_id WHERE d.name = 'tiles'"), "n") == 2);
    } else {
      CHECK_EQ(anchors, std::int64_t(0));
    }
  }
  // no key moved and no byte changed: the subjects' keys and their hashes are the same
  CHECK(key_snapshot(ds) == keys_before);
  // the whole: the index from the packages is the original database
  {
    std::string why;
    CHECK(same_database(ds / kIndexFile, ds / "dataset.sqlite.format52", why) || (std::printf("  %s\n", why.c_str()), false));
    CHECK(same_database(ds / "dataset.sqlite.format52", ds / kIndexFile, why));
  }
  // a second run changes no byte
  const auto after = tree(ds);
  const Json again = split_dataset(ds, so);
  CHECK(again["already_split"] == true);
  CHECK(tree(ds) == after);

  // CRASH at each stage of one subject, then the same command again: the same packages as the clean run
  const Json clean = split_dataset(root / "clean", so);
  const auto want = package_hashes(root / "clean");
  CHECK_EQ(want.size(), std::size_t(3));
  for (const char* stage : {"built", "verified", "committed", "registered"}) {
    const fs::path d = root / (std::string("crash-") + stage);
    SplitOptions stop = so;
    stop.on_stage = [&](const std::string& st, const std::string& subject) { if (st == stage && subject == "S1") throw Error(code::cancelled, "stopped at " + st); };
    CHECK_EQ(what_of([&] { split_dataset(d, stop); }), std::string("stopped at ") + stage);
    CHECK(fs::exists(d / kDatasetFile));                                       // the old database is untouched until the end
    CHECK(!fs::exists(d / kWriterLockFile));                                   // and the lock is let go
    const Json done = split_dataset(d, so);
    CHECK(done["subjects"].size() == 3);
    CHECK(package_hashes(d) == want);
    CHECK(key_snapshot(d) == keys_before);
    for (const auto& e : Catalogue::open(d).entries()) CHECK(SubjectPackage::open(Catalogue::open(d).package_path(e)).verify().ok);
    std::string why;
    CHECK(same_database(d / kIndexFile, d / "dataset.sqlite.format52", why));
    CHECK(!fs::exists(d / (std::string(kPackageDatabaseFile) + ".building")));
  }

  // JOIN: the one database back, table by table the original; the keys' bytes as they were
  {
    const auto shards = [&](const fs::path& d) { std::map<std::string, std::string> s; for (const auto& [k, v] : tree(d)) if (k.find("/source/") != std::string::npos && k.find("zarr.json") == std::string::npos) s[k] = v; return s; };
    const auto before_join = shards(ds);
    const Json j = join_dataset(ds);
    CHECK(fs::exists(ds / kDatasetFile));
    CHECK(!fs::exists(ds / kCatalogueFile) && fs::exists(ds / "catalogue.sqlite.format53"));
    CHECK(!fs::exists(ds / kIndexFile));
    std::string why;
    CHECK(same_database(ds / kDatasetFile, root / "original.sqlite", why) || (std::printf("  %s\n", why.c_str()), false));
    CHECK(same_database(root / "original.sqlite", ds / kDatasetFile, why));
    {
      Database a = Database::open(ds / kDatasetFile, false, "", true), b = Database::open(root / "original.sqlite", false, "", true);
      CHECK(a.all("SELECT key, value FROM meta ORDER BY key").size() == b.all("SELECT key, value FROM meta ORDER BY key").size());
      CHECK_EQ(text(*a.one("SELECT value FROM meta WHERE key = 'store_format'"), "value"), text(*b.one("SELECT value FROM meta WHERE key = 'store_format'"), "value"));
    }
    CHECK(key_snapshot(ds) == keys_before);
    CHECK(shards(ds) == before_join);
    CHECK(j["subjects"].size() == 3);
    for (const auto& e : fs::directory_iterator(ds)) if (e.is_directory() && e.path().filename().string().find(".nxr.zarr") != std::string::npos) {
      CHECK(fs::exists(e.path() / kSmallFilesFile));
      CHECK(!fs::exists(e.path() / kPackageDatabaseFile));
    }
    // the joined dataset is a dataset again: it opens as a writer, and its keys are the rows' text
    {
      DatasetWriter w = open_writer(ds);
      CHECK_EQ(text_parity(w.db(), w.store()).mismatched, 0);
    }
    CHECK(join_dataset(ds)["already_joined"] == true);
    // and splits again
    const Json second = split_dataset(ds, so);
    CHECK(second["subjects"].size() == 3);
    CHECK(key_snapshot(ds) == keys_before);
    CHECK(fs::exists(ds / "dataset.sqlite.format52-2"));                          // the earlier copy is kept, not overwritten
  }

  // a dataset below 52 is brought to 52 first
  {
    make_old(root / "older", "old", 51);
    const fs::path od = root / "older" / "old";
    CHECK_EQ(store_format_number(Database::open(od / kDatasetFile, false, "", true)), 51);
    const Json r = split_dataset(od, so);
    CHECK_EQ(r["subjects"].size(), std::size_t(3));
    const Catalogue c = Catalogue::open(od);
    for (const auto& e : c.entries()) CHECK(SubjectPackage::open(c.package_path(e)).verify().ok);
    CHECK_EQ(store_format_number(Database::open(od / "dataset.sqlite.format52", false, "", true)), 52);
  }

  fs::remove_all(root);
  return nxrtest::finish("migrate");
}
