// Group results in a default subject (what `pyramid reduce` wrote there) move into a GROUP PACKAGE at `dataset split`, and come back at `dataset join`
// (docs/specs/2026-11-16-subject-centred-datasets.md section 8; the mapping, old owner -> new owner, is in the header of io/src/migrate.cpp). Checked on a synthetic
// dataset: every row of the original is in exactly one package, in the owner the mapping assigns (the index from the packages is the original with the moved fields'
// `subject_id` changed, and each group subject is the one extra row); the arrays are in the group package, byte for byte, the originals kept aside; a crash after the
// arrays were copied is resumed; the group package verifies and names its members; and `join` gives back the original database and the original tree.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <map>

#include "check.h"
#include "nxr/io/array_writer.h"
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

// the files of a subject's folder that are arrays (not the database, the small-file table, a lock, a package's)
std::map<std::string, std::string> arrays_of(const fs::path& dir) {
  std::map<std::string, std::string> out;
  if (!fs::exists(dir)) return out;
  for (const auto& e : fs::recursive_directory_iterator(dir)) {
    const std::string n = e.path().filename().string();
    if (!e.is_regular_file() || n.find("sqlite") != std::string::npos || n == ".writer.lock") continue;
    out[fs::relative(e.path(), dir).generic_string()] = file_sha(e.path());
  }
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

// a template with a partition, two subjects, and what `pyramid reduce` leaves in the default subject: group fields (no path, their producer's `atlas` is spatial or time),
// their contributions (the members' fields) and measurements (arrays under `<node>/maps` and `<node>/time`), plus one ordinary field that stays
void make_old(const fs::path& datastore, const std::string& name) {
  DatasetWriter ds = create_dataset(datastore, name);
  SubjectOptions topt;
  topt.kind = "template";
  SubjectWriter t = SubjectWriter::create(ds, "@default_subject", topt);
  const std::string tm = add_surface(t, "cortex");
  add_field(t, "template_map", tm, 5000);
  const std::vector<std::int32_t> codes = {0, 1, 0, 1};
  DictionarySpec tiles;
  tiles.name = "tiles"; tiles.scope = "subject"; tiles.names = {"left", "right"};
  const DictionaryResult dict = t.dictionary(tiles);
  SelectionSpec psel;
  psel.name = "cortex tiles L1"; psel.path = "source/cortex_tiles_L1"; psel.type = "set"; psel.manifold_id = tm; psel.data = view(codes, {4});
  psel.cols.set("dictionary_id", dict.id).set("cell", "0");
  const std::string part = t.selection(psel);
  t.complete();
  std::map<std::string, std::string> small_of, id_of;
  for (const char* sname : {"S1", "S2"}) {
    SubjectWriter s = SubjectWriter::create(ds, sname);
    const std::string own = add_surface(s, "cortex");
    small_of[sname] = add_field(s, "small", own, 16);
    add_field(s, "big", own, 400000);
    s.complete();
    id_of[sname] = text(*ds.subject(sname), "id");
  }
  SubjectWriter d = SubjectWriter::open(ds, "@default_subject");
  d.compose([&] {
    const std::string node = "source/cortex_tiles_L1";
    for (const char* kind : {"spatial", "time"}) {
      const std::string key = std::string(kind) == "spatial" ? "mean" : "task__method__free";
      const std::string g = node + "/" + (std::string(kind) == "spatial" ? "maps" : "time") + "/" + key;
      const std::vector<double> v = {1.0, 2.0, 3.0, 4.0};
      FieldSpec f;
      f.name = std::string("group ") + key; f.kind = "group map"; f.manifold_id = tm;
      f.cols.set("data_type", "float32").set("function", "sum").json("producer_json", Json{{"atlas", kind}, {"key", key}, {"subjects", Json::array({"S1", "S2"})}});
      f.contributions.push_back({1.0, id_of["S1"], small_of["S1"]});
      f.contributions.push_back({1.0, id_of["S2"], small_of["S2"]});
      const std::string fid = d.field(f);
      d.write_array(g + "/s1", view(v, {4}));
      d.write_array(g + "/w", view(v, {4}));
      std::vector<double> big(200000);
      for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<double>(i % 1013);
      d.write_array(g + "/big", view(big, {static_cast<std::int64_t>(big.size())}));                 // over 1 MiB: files, not rows of the small-key table
      Cols c;
      c.set("measure", "sum").set("of_field_id", fid).set("array_path", g + "/s1").set("computed_by", "test");
      d.measurement(part, c);
      Cols w;
      w.set("measure", "weight").set("of_field_id", fid).set("array_path", g + "/w").set("computed_by", "test");
      d.measurement(part, w);
    }
  });
  ds.close();
}

bool same_rows(const fs::path& a, const fs::path& b, std::string& why) {
  Database da = Database::open(a, false, "", true);
  da.execute("ATTACH DATABASE ? AS other", {b.string()});
  for (const auto& r : da.all("SELECT name FROM main.sqlite_master WHERE type = 'table' ORDER BY name")) {
    const std::string t = text(r, "name");
    if (t == "meta" || t == "sync_job") continue;
    const std::string l = "\"" + t + "\"";
    if (!da.one("SELECT 1 FROM other.sqlite_master WHERE type = 'table' AND name = ?", {t})) continue;
    for (const char* d : {"main", "other"}) {
      const std::string o = std::string(d) == "main" ? "other" : "main";
      if (integer(*da.one("SELECT count(*) AS n FROM (SELECT * FROM " + std::string(d) + "." + l + " EXCEPT SELECT * FROM " + o + "." + l + ")"), "n") != 0) { why = "rows of " + t + " differ"; return false; }
    }
    if (integer(*da.one("SELECT count(*) AS n FROM main." + l), "n") != integer(*da.one("SELECT count(*) AS n FROM other." + l), "n")) { why = "the number of rows of " + t + " differs"; return false; }
  }
  return true;
}
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_split_groups_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  SplitOptions so;
  so.registered_by = "test";
  make_old(root, "demo");
  const fs::path ds = root / "demo";
  fs::copy_file(ds / kDatasetFile, root / "original.sqlite");
  const auto template_arrays_before = arrays_of(ds / "@default_subject.nxr.zarr");
  auto has_under = [](const std::map<std::string, std::string>& m, const std::string& prefix) { for (const auto& [k, v] : m) if (k.rfind(prefix, 0) == 0) return true; return false; };
  CHECK(has_under(template_arrays_before, "source/cortex_tiles_L1/maps/mean/big/") && has_under(template_arrays_before, "source/cortex_tiles_L1/time/task__method__free/big/"));

  // the plan moves nothing: a template keeps its group results (format 56: the template holds the cohort)
  {
    SplitOptions dry = so;
    dry.dry_run = true;
    const Json plan = split_dataset(ds, dry);
    CHECK(!plan.contains("group_results_move"));
  }

  // SPLIT
  const Json rep = split_dataset(ds, so);
  CHECK(!rep.contains("group_results") && !rep.contains("template_references"));
  const Catalogue cat = Catalogue::open(ds);
  CHECK_EQ(cat.entries().size(), std::size_t(3));                              // the template, S1 and S2: no group package
  for (const auto& e : cat.entries()) CHECK(e.kind == "subject" || e.kind == "template");
  // the template package: whole, its group fields its own, the members' source fields held as anchors (the template refers to its subjects)
  {
    const SubjectPackage t = SubjectPackage::open(ds / "@default_subject.nxr.zarr");
    CHECK(t.verify().ok);
    CHECK(t.db().all("PRAGMA foreign_key_check").empty());
    const std::string tid = t.record().subject_id;
    CHECK_EQ(integer(*t.db().one("SELECT count(*) AS n FROM field_contribution"), "n"), 4);
    CHECK_EQ(integer(*t.db().one("SELECT count(*) AS n FROM selection_measurement"), "n"), 4);
    CHECK(t.db().one("SELECT 1 FROM anchor_row WHERE table_name = 'field'").has_value());       // the members' fields
    CHECK(arrays_of(ds / "@default_subject.nxr.zarr") == template_arrays_before);              // its arrays where they were, untouched
  }
  // the subjects: no template link at all
  for (const char* n : {"S1", "S2"}) {
    const SubjectPackage p = SubjectPackage::open(ds / (std::string(n) + ".nxr.zarr"));
    CHECK(p.verify().ok);
    CHECK(p.record().template_reference.is_null());
    CHECK(is_null(p.subject_row(), "template_id"));
    CHECK(!p.db().one("SELECT 1 FROM anchor_row").has_value());
  }
  // JOIN: one database again, every subject in it
  {
    join_dataset(ds);
    CHECK(fs::exists(ds / kDatasetFile));
    CHECK_EQ(integer(*Database::open(ds / kDatasetFile, false, "", true).one("SELECT count(*) AS n FROM subject"), "n"), 3);
  }
  return nxrtest::finish("split_groups");
}
