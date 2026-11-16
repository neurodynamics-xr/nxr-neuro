// The dataset module on synthetic packages: the catalogue, registering (idempotent, conflict rules, refusals), the participants table (BIDS import and export),
// selecting subjects by list and by filter, the cross-subject index as a cache, group packages that record their members' hashes.
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/participants.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
std::string what_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// A package of one surface and one field, made for the dataset (so its home is the dataset).
fs::path make_package(const Catalogue& cat, const fs::path& at, const std::string& name) {
  DatasetWriter w = create_package(at, name, cat.package_options());
  SubjectWriter sub = SubjectWriter::open(w, name);
  const std::vector<double> v(12, 1.0);
  const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
  ManifoldSpec m;
  m.name = "cortex"; m.path = "source/cortex"; m.type = "surface"; m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/cortex/faces"}}};
  GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "source/cortex/vertices";
  m.geometry = g;
  m.populate = [&](Store&, const std::string&) { sub.write_array("source/cortex/vertices", view(v, {4, 3})); sub.write_array("source/cortex/faces", view(f, {2, 3})); };
  const std::string mid = sub.manifold(m);
  const std::vector<float> a(4, 2.0f);
  FieldSpec fs_; fs_.name = "map"; fs_.path = "source/map"; fs_.kind = "map"; fs_.manifold_id = mid; fs_.data = view(a, {4});
  sub.field(fs_);
  sub.complete();
  return at;
}

// change a package (a new field): a different content hash
void touch(const fs::path& pkg, const std::string& name) {
  DatasetWriter w = open_package_writer(pkg);
  SubjectWriter sub = SubjectWriter::open(w, name);
  const std::vector<float> a(3, 5.0f);
  static int n = 0;
  const std::string tag = n++ == 0 ? "extra" : "extra" + std::to_string(n);
  FieldSpec f; f.name = tag; f.path = "source/" + tag; f.kind = "map"; f.manifold_id = text(*sub.by_name("manifold", "cortex"), "id"); f.data = view(a, {3});
  sub.field(f);
  sub.complete();
}

void write_text(const fs::path& p, const std::string& s) { std::ofstream(p, std::ios::binary) << s; }
std::string read_text(const fs::path& p) { std::ifstream i(p, std::ios::binary); return std::string((std::istreambuf_iterator<char>(i)), std::istreambuf_iterator<char>()); }
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_catalogue_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path ds = root / "cohort";
  RegisterOptions reg;
  reg.registered_by = "test";

  CHECK_EQ(code_of([&] { Catalogue::open(ds); }), std::string(code::not_found));
  CHECK_EQ(code_of([&] { create_catalogue(ds, CreateCatalogueOptions{}); }), std::string(code::invalid_argument));   // the dataset is named
  Catalogue cat = create_catalogue(ds, CreateCatalogueOptions{.name = "demo"});
  CHECK_EQ(code_of([&] { create_catalogue(ds, CreateCatalogueOptions{.name = "demo"}); }), std::string(code::already_exists));
  CHECK(!fs::exists(ds / ".writer.lock"));
  CHECK(cat.entries().empty());
  CHECK(!cat.has_participants());

  // registering: new, then a rerun is a no-op
  for (const char* n : {"S1", "S2", "S3"}) make_package(cat, ds / (std::string(n) + ".nxr.zarr"), n);
  CHECK_EQ(scan_packages(ds).size(), std::size_t(3));
  auto res = register_packages(ds, scan_packages(ds), reg);
  CHECK_EQ(res.size(), std::size_t(3));
  for (const auto& r : res) { CHECK_EQ(r.outcome, std::string("registered")); if (r.outcome != "registered") std::printf("  %s\n", r.reason.c_str()); }
  const Catalogue c1 = Catalogue::open(ds);
  CHECK_EQ(c1.entries().size(), std::size_t(3));
  const auto e1 = *c1.find("S1");
  CHECK_EQ(e1.kind, std::string("subject"));
  CHECK_EQ(e1.path, std::string("S1.nxr.zarr"));
  CHECK_EQ(e1.store_format, 53);
  CHECK_EQ(e1.content_hash, SubjectPackage::open(ds / "S1.nxr.zarr").record().content_hash);
  CHECK_EQ(e1.n_surfaces, std::int64_t(1));
  CHECK_EQ(e1.n_fields, std::int64_t(1));
  CHECK_EQ(c1.find(e1.subject_id)->name, std::string("S1"));
  for (const auto& r : register_packages(ds, scan_packages(ds), reg)) CHECK_EQ(r.outcome, std::string("unchanged"));
  // the catalogue holds no manifold, field or participant row
  CHECK_EQ(integer(*c1.db().one("SELECT count(*) AS n FROM manifold"), "n"), std::int64_t(0));
  CHECK_EQ(integer(*c1.db().one("SELECT count(*) AS n FROM field"), "n"), std::int64_t(0));

  // a package a writer holds is refused; one that changed is `updated`
  {
    DatasetWriter w = open_package_writer(ds / "S1.nxr.zarr");
    const auto r = register_packages(ds, {ds / "S1.nxr.zarr"}, reg);
    CHECK_EQ(r[0].outcome, std::string("refused"));
    CHECK(has(r[0].reason, "writer"));
  }
  touch(ds / "S1.nxr.zarr", "S1");
  CHECK_EQ(Catalogue::open(ds).find("S1")->content_hash, e1.content_hash);   // the entry is the registered state, refreshed only by registering
  res = register_packages(ds, {ds / "S1.nxr.zarr"}, reg);
  CHECK_EQ(res[0].outcome, std::string("updated"));
  CHECK_EQ(Catalogue::open(ds).find("S1")->n_fields, std::int64_t(2));
  CHECK_EQ(Catalogue::open(ds).entries().size(), std::size_t(3));
  // a package changed outside a writer no longer hashes to its record
  {
    Database d = Database::open(ds / "S3.nxr.zarr" / kPackageDatabaseFile);
    d.execute("UPDATE field SET comment = 'edited outside a writer'");
  }
  res = register_packages(ds, {ds / "S3.nxr.zarr"}, reg);
  CHECK_EQ(res[0].outcome, std::string("refused"));
  CHECK(has(res[0].reason, "no longer hash"));
  { DatasetWriter w = open_package_writer(ds / "S3.nxr.zarr"); }          // open and close: the record follows the rows
  CHECK_EQ(register_packages(ds, {ds / "S3.nxr.zarr"}, reg)[0].outcome, std::string("updated"));

  // another id under a name the catalogue holds is refused, unless replaced on purpose; the old package is moved aside, never deleted
  {
    const fs::path other = root / "elsewhere";
    fs::create_directories(other);
    make_package(cat, ds / "S2-again.nxr.zarr", "S2");
    res = register_packages(ds, {ds / "S2-again.nxr.zarr"}, reg);
    CHECK_EQ(res[0].outcome, std::string("refused"));
    CHECK(has(res[0].reason, "--replace"));
    CHECK_EQ(Catalogue::open(ds).find("S2")->path, std::string("S2.nxr.zarr"));
    const std::string old_id = Catalogue::open(ds).find("S2")->subject_id;
    RegisterOptions rep = reg;
    rep.replace = true;
    res = register_packages(ds, {ds / "S2-again.nxr.zarr"}, rep);
    CHECK_EQ(res[0].outcome, std::string("replaced"));
    const auto c = Catalogue::open(ds);
    CHECK(c.find("S2")->subject_id != old_id);
    CHECK_EQ(c.find(old_id)->status, std::string("replaced"));
    CHECK(!fs::exists(ds / "S2.nxr.zarr"));
    bool aside = false;
    for (const auto& e : fs::directory_iterator(ds)) if (e.path().filename().string().rfind("S2.nxr.zarr.replaced-", 0) == 0 && fs::exists(e.path() / kPackageDatabaseFile)) aside = true;
    CHECK(aside);
    // a replaced entry is not a member by name
    CHECK(c.find("S2")->status == "registered");
  }

  // id clashes between packages are refused at registration, naming both
  {
    testing::deterministic(true, 0);
    make_package(cat, ds / "C.nxr.zarr", "C");
    testing::deterministic(true, 0);
    make_package(cat, ds / "D.nxr.zarr", "D");
    testing::deterministic(false);
    res = register_packages(ds, {ds / "C.nxr.zarr", ds / "D.nxr.zarr"}, reg);
    CHECK_EQ(res[0].outcome, std::string("registered"));
    CHECK_EQ(res[1].outcome, std::string("refused"));
    CHECK(has(res[1].reason, "is also in") && has(res[1].reason, "C.nxr.zarr"));
  }
  // a vocabulary copy that differs from the dataset's is refused by name
  {
    make_package(cat, ds / "V.nxr.zarr", "V");
    {
      DatasetWriter w = open_package_writer(ds / "V.nxr.zarr");
      w.db().execute("UPDATE dictionary_entry SET name = 'renamed' WHERE name = 'delta'");
    }
    res = register_packages(ds, {ds / "V.nxr.zarr"}, reg);
    CHECK_EQ(res[0].outcome, std::string("refused"));
    CHECK(has(res[0].reason, "frequency bands") && has(res[0].reason, "differs"));
  }
  // a folder that is no package, and one named twice
  res = register_packages(ds, {root, ds / "S1.nxr.zarr", ds / "S1.nxr.zarr"}, reg);
  CHECK_EQ(res[0].outcome, std::string("refused"));
  CHECK_EQ(res[2].outcome, std::string("refused"));
  CHECK_EQ(code_of([&] { register_packages(ds, {}, RegisterOptions{}); }), std::string(code::invalid_argument));   // who registers is named

  // the participants table: its own file, BIDS in and out
  const fs::path tsv = root / "participants.tsv", side = root / "participants.json";
  CHECK(!fs::exists(ds / kParticipantsFile));
  write_text(tsv, "participant_id\tage\tgroup\tsite\tsmoker\nS1\t71\tcontrol\ta\ttrue\nS2\t33\tpatient\tb\tfalse\nS3\tn/a\tcontrol\ta\tn/a\nC\t64\tpatient\tc\tfalse\n");
  write_text(side, R"({"age":{"Description":"age at scan","Units":"years"},"group":{"Levels":{"control":"controls","patient":"patients"}},"smoker":{"Type":"boolean"}})");
  {
    write_text(root / "bad.tsv", "participant_id\tage\nS1\t5\nnobody\t6\n");
    const std::string why = what_of([&] { participants_import(ds, root / "bad.tsv", std::nullopt); });
    CHECK(has(why, "nobody"));
    CHECK(!fs::exists(ds / kParticipantsFile));                                   // nothing written
    write_text(root / "bad2.tsv", "participant_id\tgroup\nS1\tother\n");
    CHECK(has(what_of([&] { participants_import(ds, root / "bad2.tsv", side); }), "not one of its levels"));
    CHECK(!fs::exists(ds / kParticipantsFile));
    const auto rep = participants_import(ds, tsv, side);
    CHECK_EQ(rep.variables.size(), std::size_t(4));
    CHECK_EQ(rep.n_subjects, std::int64_t(4));
    CHECK_EQ(rep.n_values, std::int64_t(14));
  }
  CHECK(Catalogue::open(ds).has_participants());
  CHECK(!fs::exists(ds / ".writer.lock"));
  {
    const auto vars = participant_variables(ds);
    CHECK_EQ(vars.size(), std::size_t(4));
    CHECK(vars[0].name == "age" && vars[0].type == VariableType::Number && vars[0].unit == std::string("years"));
    CHECK(vars[1].name == "group" && vars[1].type == VariableType::String && vars[1].levels && vars[1].levels->size() == 2);
    CHECK(vars[3].type == VariableType::Boolean);
    const auto vals = participant_values(ds);
    const std::string s3 = Catalogue::open(ds).find("S3")->subject_id;
    CHECK(vals.at(s3).count("group") && !vals.at(s3).count("age") && !vals.at(s3).count("smoker"));   // a missing value is a missing row
    // a rerun replaces the variables it defines
    participants_import(ds, tsv, side);
    CHECK_EQ(participant_variables(ds).size(), std::size_t(4));
    CHECK_EQ(participant_values(ds).size(), std::size_t(4));
  }
  participants_export(ds, root / "out.tsv", root / "out.json");
  {
    const std::string out = read_text(root / "out.tsv");
    CHECK(has(out, "participant_id\tage\tgroup\tsite\tsmoker\n"));
    CHECK(has(out, "S1\t71\tcontrol\ta\ttrue\n"));
    CHECK(has(out, "S3\tn/a\tcontrol\ta\tn/a\n"));
    const Json j = Json::parse(read_text(root / "out.json"));
    CHECK(j["age"]["Units"] == "years" && j["group"]["Levels"].contains("patient") && j["smoker"]["Type"] == "boolean");
    // the catalogue and the packages never carry participants
    CHECK_EQ(integer(*Catalogue::open(ds).db().one("SELECT count(*) AS n FROM subject_variable"), "n"), std::int64_t(0));
  }

  // selecting subjects: one of --subjects or --where
  {
    const auto names = [](const SelectedSubjects& s) { std::string o; for (const auto& m : s.members) o += (o.empty() ? "" : ",") + m.name; return o; };
    CHECK_EQ(code_of([&] { select_subjects(ds, SubjectSelection{}); }), std::string(code::invalid_argument));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{}); }), "no 'all subjects' default"));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{"S1"}, std::string("age > 1")}); }), "not both"));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{"S1", "ghost", "ghost2"}, std::nullopt}); }), "ghost, ghost2"));
    const auto by_list = select_subjects(ds, SubjectSelection{{"S2", "S1", "S2"}, std::nullopt});
    CHECK_EQ(names(by_list), std::string("S2,S1"));
    CHECK_EQ(by_list.how, std::string("subjects"));
    const auto old = select_subjects(ds, SubjectSelection{{}, std::string("group == 'control' and age >= 60")});
    CHECK_EQ(names(old), std::string("S1"));
    CHECK_EQ(old.filter_language_version, 1);
    CHECK(old.provenance()["filter"] == "group == 'control' and age >= 60");
    CHECK_EQ(old.members[0].content_hash, Catalogue::open(ds).find("S1")->content_hash);
    CHECK_EQ(names(select_subjects(ds, SubjectSelection{{}, std::string("age is missing")})), std::string("S3"));
    CHECK_EQ(names(select_subjects(ds, SubjectSelection{{}, std::string("not (age >= 60)")})), std::string("S2"));       // S3 (unknown) is not selected by accident
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{}, std::string("age > 200")}); }), "matches no subject"));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{}, std::string("agee > 1")}); }), "age, group, site, smoker"));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{}, std::string("age == 'x'")}); }), "is a number"));
    // a member changed since registering is stale and refused
    touch(ds / "S2-again.nxr.zarr", "S2");
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{"S2"}, std::nullopt}); }), "stale"));
    CHECK_EQ(register_packages(ds, {ds / "S2-again.nxr.zarr"}, reg)[0].outcome, std::string("updated"));
    CHECK_EQ(names(select_subjects(ds, SubjectSelection{{"S2"}, std::nullopt})), std::string("S2"));
  }
  // without a participants table a --where is refused, saying so
  {
    Catalogue bare = create_catalogue(root / "bare", CreateCatalogueOptions{.name = "bare"});
    make_package(bare, root / "bare" / "B1.nxr.zarr", "B1");
    CHECK_EQ(register_packages(root / "bare", {root / "bare" / "B1.nxr.zarr"}, reg)[0].outcome, std::string("registered"));
    CHECK(has(what_of([&] { select_subjects(root / "bare", SubjectSelection{{}, std::string("age > 1")}); }), "no participants table"));
    CHECK(has(what_of([&] { participants_export(root / "bare", root / "x.tsv", std::nullopt); }), "no participants table"));
  }

  // the index is a cache: built from the packages, current until a member changes, deletable
  {
    CHECK(!index_is_current(ds));
    const IndexReport r = build_index(ds);
    CHECK(r.rebuilt);
    CHECK(fs::exists(ds / kIndexFile));
    CHECK(!fs::exists(ds / (std::string(kIndexFile) + ".tmp")));
    CHECK(index_is_current(ds));
    CHECK(!build_index(ds).rebuilt);
    Database idx = Database::open(ds / kIndexFile, false, "", true);
    const Catalogue c = Catalogue::open(ds);
    std::int64_t members = 0, manifolds = 0, fields = 0;
    for (const auto& e : c.entries()) {
      if (e.status != "registered") continue;
      ++members;
      const SubjectPackage pkg = SubjectPackage::open(c.package_path(e));
      const Database& pk = pkg.db();
      manifolds += integer(*pk.one("SELECT count(*) AS n FROM manifold"), "n");
      fields += integer(*pk.one("SELECT count(*) AS n FROM field"), "n");
      // row for row: the subject's fields in the index are its package's
      CHECK_EQ(integer(*idx.one("SELECT count(*) AS n FROM field WHERE subject_id = ?", {e.subject_id}), "n"), integer(*pk.one("SELECT count(*) AS n FROM field"), "n"));
    }
    CHECK_EQ(integer(*idx.one("SELECT count(*) AS n FROM subject"), "n"), members);
    CHECK_EQ(integer(*idx.one("SELECT count(*) AS n FROM manifold"), "n"), manifolds);
    CHECK_EQ(integer(*idx.one("SELECT count(*) AS n FROM field"), "n"), fields);
    CHECK_EQ(integer(*idx.one("SELECT count(*) AS n FROM dictionary WHERE name = 'frequency bands'"), "n"), std::int64_t(1));   // one vocabulary, matched by key
    CHECK_EQ(integer(*idx.one("SELECT count(*) AS n FROM dataset"), "n"), std::int64_t(1));
    CHECK(idx.all("PRAGMA foreign_key_check").empty());
    // a member changes: the index is stale, and rebuilt
    touch(ds / "S3.nxr.zarr", "S3");
    CHECK(register_packages(ds, {ds / "S3.nxr.zarr"}, reg)[0].outcome == "updated");
    CHECK(!index_is_current(ds));
    CHECK(build_index(ds).rebuilt);
    CHECK(index_is_current(ds));
    // deleting it loses nothing
    idx.close();
    fs::remove(ds / kIndexFile);
    CHECK(build_index(ds).rebuilt);
    CHECK(Database::open(ds / kIndexFile, false, "", true).one("SELECT 1 FROM field WHERE name = 'extra'").has_value());
  }

  // a group package names its members and their hashes; a member changed later is seen
  {
    const auto members = select_subjects(ds, SubjectSelection{{"S1", "S3"}, std::nullopt});
    const fs::path g = create_group_package(ds, "controls", members, reg);
    CHECK(fs::exists(g / kPackageDatabaseFile));
    const auto e = *Catalogue::open(ds).find("controls");
    CHECK_EQ(e.kind, std::string("group"));
    CHECK_EQ(e.status, std::string("registered"));
    CHECK_EQ(SubjectPackage::open(g).record().members.size(), std::size_t(2));
    for (const auto& s : group_member_status(ds, "controls")) CHECK(s.current);
    CHECK(has(what_of([&] { create_group_package(ds, "controls", members, reg); }), "already holds"));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{"controls"}, std::nullopt}); }), "is a group"));
    CHECK(select_subjects(ds, SubjectSelection{{}, std::string("age >= 0")}).members.size() >= 1);    // a group is never a member of a cohort
    touch(ds / "S1.nxr.zarr", "S1");
    CHECK_EQ(register_packages(ds, {ds / "S1.nxr.zarr"}, reg)[0].outcome, std::string("updated"));
    int stale = 0;
    for (const auto& s : group_member_status(ds, "controls")) if (!s.current) { ++stale; CHECK_EQ(s.name, std::string("S1")); }
    CHECK_EQ(stale, 1);
    // the index folds the group in too
    CHECK(build_index(ds).rebuilt);
  }

  // status: a package that moved away is missing; one that changed behind the catalogue's back is stale
  {
    fs::rename(ds / "C.nxr.zarr", root / "C-away.nxr.zarr");
    touch(ds / "S3.nxr.zarr", "S3");
    std::map<std::string, std::string> st;
    for (const auto& e : refresh_member_status(ds)) st[e.name] = e.status;
    CHECK_EQ(st["C"], std::string("missing"));
    CHECK_EQ(st["S3"], std::string("stale"));
    CHECK_EQ(st["S1"], std::string("registered"));
    CHECK_EQ(Catalogue::open(ds).find("C")->status, std::string("missing"));
    CHECK(has(what_of([&] { select_subjects(ds, SubjectSelection{{"C"}, std::nullopt}); }), "missing"));
  }

  fs::remove_all(root);
  return nxrtest::finish("catalogue");
}
