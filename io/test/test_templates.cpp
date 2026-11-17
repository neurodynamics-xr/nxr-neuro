// Templates as packages: origin records, release and freezing, the two editions, the search path, a subject registered to a template, a registration between two
// templates, and the datastore's catalog.json no longer written. Synthetic data only.
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "check.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/templates.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) {
  try { f(); } catch (const Error& e) { return e.code(); }
  return "";
}
std::string what_of(const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { return e.what(); }
  return "";
}
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

std::string add_sphere(SubjectWriter& sub, const std::string& name) {
  static const std::vector<double> v = {1, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0};
  static const std::vector<std::int32_t> f = {0, 1, 2, 0, 2, 3};
  ManifoldSpec m;
  m.name = name; m.path = "source/" + name; m.type = "surface"; m.n_vertices = 4;
  m.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, "source/" + name + "/faces"}}};
  GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "source/" + name + "/vertices";
  m.geometry = g;
  m.populate = [&sub, name](Store&, const std::string&) { sub.write_array("source/" + name + "/vertices", view(v, {4, 3})); sub.write_array("source/" + name + "/faces", view(f, {2, 3})); };
  return sub.manifold(m);
}

// A template package: a sphere (redistributable), a large reference map on it (not), and a partition coded by a dictionary of its own.
fs::path make_template(const fs::path& at, const std::string& name, bool origins = true) {
  CreatePackageOptions po;
  po.home_dataset = "templates-work";
  po.subject.kind = "template";
  DatasetWriter w = create_package(at, name, po);
  {
    SubjectWriter sub = SubjectWriter::open(w, name);
    const std::string sphere = add_sphere(sub, "sphere");
    std::vector<double> map(400000);
    for (std::size_t i = 0; i < map.size(); ++i) map[i] = static_cast<double>(i % 313) * 0.5;
    FieldSpec f; f.name = "reference_map"; f.path = "source/reference_map"; f.kind = "map"; f.manifold_id = sphere; f.data = view(map, {400000});
    sub.field(f);
    DictionarySpec d; d.name = "tiles"; d.scope = "subject"; d.names = {"left", "right"};
    const DictionaryResult dict = sub.dictionary(d);
    const std::vector<std::int32_t> codes = {0, 1, 0, 1};
    SelectionSpec sel; sel.name = "tiles"; sel.path = "source/tiles"; sel.type = "set"; sel.manifold_id = sphere; sel.data = view(codes, {4});
    sel.cols.set("dictionary_id", dict.id).set("cell", "0");
    sub.selection(sel);
    sub.complete();
  }
  if (origins) {
    for (const auto& it : package_items(w.db())) {
      const bool free_to_share = it.name != "reference_map";
      set_origin(w.db(), it.table, it.id, OriginRecord{"a synthetic sphere", free_to_share ? "CC0" : "a restricted licence", "the test authors", free_to_share});
    }
  }
  return at;
}

std::string hash_of(const fs::path& p) { return SubjectPackage::open(p).record().content_hash; }
}  // namespace

int main() {
  const fs::path root = fs::temp_directory_path() / ("nxr_io_test_templates_" + std::to_string(::getpid()));
  fs::remove_all(root);
  fs::create_directories(root);
  ::unsetenv(kTemplatePathEnv);
  const fs::path shelf = root / "shelf";
  TemplateSearch here;
  here.command_paths = {shelf};
  here.use_environment = false;
  here.settings_file = root / "no-settings.json";

  // names
  {
    const TemplateRef r = parse_template_ref("fsaverage@1.0"), r2 = parse_template_ref("fsaverage@1.0:redistributable");
    CHECK(r.name == "fsaverage" && r.version == "1.0" && r.edition == "full" && r2.edition == "redistributable");
    CHECK_EQ(template_package_name("fsaverage", "1.0"), std::string("fsaverage@1.0.nxr.zarr"));
    CHECK_EQ(template_package_name("fsaverage", "1.0", "redistributable"), std::string("fsaverage@1.0+redistributable.nxr.zarr"));
    CHECK_EQ(code_of([&] { parse_template_ref("fsaverage"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { parse_template_ref("a/b@1"); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { parse_template_ref("a@1:other"); }), std::string(code::invalid_argument));
  }

  // an item with no origin record stops a release, naming it
  const fs::path bare = make_template(root / "work" / "bare.nxr.zarr", "bare", false);
  {
    const std::string why = what_of([&] { release_package(bare, ReleaseOptions{"bare", "1.0", "test", shelf}); });
    CHECK(has(why, "have no origin record") && has(why, "source/sphere") && has(why, "source/reference_map"));
    CHECK(!fs::exists(shelf / "bare@1.0.nxr.zarr"));
  }

  // RELEASE: a frozen copy in the shelf, the same content hash
  const fs::path work = make_template(root / "work" / "A.nxr.zarr", "A");
  const std::string work_hash = hash_of(work);
  CHECK(items_without_origin(SubjectPackage::open(work).db()).empty());
  ReleaseReport rel = release_package(work, ReleaseOptions{"A", "1.0", "test", shelf});
  CHECK_EQ(rel.folder.filename().string(), std::string("A@1.0.nxr.zarr"));
  CHECK_EQ(rel.content_hash, work_hash);
  CHECK_EQ(hash_of(rel.folder), work_hash);
  CHECK(SubjectPackage::open(rel.folder).verify().ok);
  CHECK(SubjectPackage::open(rel.folder).record().release.value("version", "") == "1.0");
  CHECK(!SubjectPackage::open(work).record().release.is_object());                         // the working package is not the release
  CHECK_EQ(code_of([&] { open_package_writer(rel.folder); }), std::string(code::not_writable));   // frozen
  CHECK_EQ(code_of([&] { release_package(rel.folder, ReleaseOptions{"A", "2.0", "test", shelf}); }), std::string(code::invalid_argument));   // already released
  CHECK_EQ(release_package(work, ReleaseOptions{"A", "1.0", "test", shelf}).content_hash, work_hash);   // the same release again is the same
  {
    { DatasetWriter w = open_package_writer(work); const auto& db = w.db(); (void)db; w.db().execute("UPDATE field SET comment = 'changed' WHERE name = 'reference_map'"); }
    CHECK(has(what_of([&] { release_package(work, ReleaseOptions{"A", "1.0", "test", shelf}); }), "frozen"));   // a changed package is another version
    CHECK_EQ(hash_of(rel.folder), work_hash);
  }

  // THE SEARCH PATH: command, environment, user setting, hint; the reference's hash decides
  {
    TemplateRef ref = parse_template_ref("A@1.0");
    ref.content_hash = work_hash;
    const ResolvedTemplate t = resolve_template(ref, here);
    CHECK_EQ(t.folder.filename().string(), std::string("A@1.0.nxr.zarr"));
    CHECK(t.record.release.value("name", "") == "A");
    // none: refused, listing every location searched
    TemplateSearch nowhere;
    nowhere.use_environment = false;
    nowhere.settings_file = root / "no-settings.json";
    CHECK(has(what_of([&] { resolve_template(ref, nowhere); }), "nowhere"));
    TemplateSearch wrong = here;
    wrong.command_paths = {root / "empty", root};
    const std::string why = what_of([&] { resolve_template(ref, wrong); });
    CHECK(has(why, (root / "empty").string()) && has(why, root.string()) && has(why, "not found"));
    // a hash that is not the package's is skipped and both hashes are named
    TemplateRef bad = ref;
    bad.content_hash = std::string(64, 'a');
    const std::string why2 = what_of([&] { resolve_template(bad, here); });
    CHECK(has(why2, work_hash) && has(why2, bad.content_hash));
    CHECK_EQ(code_of([&] { resolve_template(TemplateRef{"A", "1.0", "full", "", std::nullopt}, here); }), std::string(code::invalid_argument));   // a reference names its hash
    // environment, then settings, then the hint
    TemplateSearch env;
    env.use_environment = true;
    env.settings_file = root / "no-settings.json";
    ::setenv(kTemplatePathEnv, ((root / "elsewhere").string() + ":" + shelf.string()).c_str(), 1);
    CHECK_EQ(resolve_template(ref, env).folder, fs::weakly_canonical(shelf / "A@1.0.nxr.zarr"));
    const auto order = template_search_path(env);
    CHECK_EQ(order.size(), std::size_t(2));
    CHECK_EQ(order[0], root / "elsewhere");
    ::unsetenv(kTemplatePathEnv);
    TemplateSearch setting;
    setting.use_environment = false;
    setting.settings_file = root / "settings.json";
    { std::ofstream(*setting.settings_file) << Json{{"template_path", Json::array({shelf.string()})}}.dump(); }
    CHECK_EQ(resolve_template(ref, setting).record.content_hash, work_hash);
    TemplateSearch hint;
    hint.use_environment = false;
    hint.settings_file = root / "no-settings.json";
    TemplateRef hinted = ref;
    hinted.location_hint = shelf.string();
    CHECK_EQ(resolve_template(hinted, hint).record.content_hash, work_hash);
    // order: the command's location first (a copy of the right name and version with another hash is skipped, the later one used)
    {
      const fs::path decoy = root / "decoy";
      fs::create_directories(decoy);
      const fs::path other = make_template(root / "work" / "A2.nxr.zarr", "A2");
      { DatasetWriter w = open_package_writer(other); w.db().execute("UPDATE field SET comment = 'different' WHERE name = 'reference_map'"); }
      release_package(other, ReleaseOptions{"A", "1.0", "test", decoy});
      TemplateSearch both = here;
      both.command_paths = {decoy, shelf};
      CHECK_EQ(resolve_template(ref, both).folder, fs::weakly_canonical(shelf / "A@1.0.nxr.zarr"));
      CHECK_EQ(list_templates(both).size(), std::size_t(2));
    }
  }

  // TWO EDITIONS: the redistributable one keeps what its origin lets it, byte for byte, and lists what it left out
  {
    const ReleaseReport red = release_redistributable(rel.folder, shelf, "test");
    CHECK_EQ(red.folder.filename().string(), std::string("A@1.0+redistributable.nxr.zarr"));
    CHECK(red.content_hash != work_hash);
    CHECK(has(red.left_out.dump(), "reference_map"));
    CHECK(!has(red.left_out.dump(), "sphere\""));
    SubjectPackage full = SubjectPackage::open(rel.folder), part = SubjectPackage::open(red.folder);
    CHECK(part.verify().ok);
    CHECK(part.record().release.value("edition", "") == "redistributable");
    CHECK(part.record().release.value("derived_from", "") == work_hash);
    CHECK(!part.db().one("SELECT 1 FROM field WHERE name = 'reference_map'").has_value());
    CHECK(part.db().one("SELECT 1 FROM manifold WHERE name = 'sphere'").has_value());
    CHECK(!part.store().exists_prefix("source/reference_map"));
    // a kept item is the full edition's: the same key digests, and the same values read back
    for (const char* k : {"source/sphere/vertices/zarr.json", "source/sphere/faces/zarr.json", "source/tiles/zarr.json"})
      CHECK_EQ(integer(*part.db().one("SELECT size_bytes AS n FROM key_digest WHERE key = ?", {std::string(k)}), "n"), integer(*full.db().one("SELECT size_bytes AS n FROM key_digest WHERE key = ?", {std::string(k)}), "n"));
    CHECK(ArrayReader::open(part.store(), "source/sphere/vertices").read() == ArrayReader::open(full.store(), "source/sphere/vertices").read());
    CHECK(ArrayReader::open(part.store(), "source/tiles").read() == ArrayReader::open(full.store(), "source/tiles").read());
    TemplateRef pr = parse_template_ref("A@1.0:redistributable");
    pr.content_hash = red.content_hash;
    CHECK_EQ(resolve_template(pr, here).record.release.value("edition", ""), std::string("redistributable"));
    CHECK(has(what_of([&] { release_redistributable(red.folder, shelf, "test"); }), "released full edition"));
    CHECK(has(what_of([&] { release_redistributable(rel.folder, shelf, "test"); }), "already released"));
  }

  // A SUBJECT registered to a template: its anchor rows, its record, its bundle complete without the template
  TemplateRef aref = parse_template_ref("A@1.0");
  aref.content_hash = work_hash;
  const ResolvedTemplate A = resolve_template(aref, here);
  const fs::path ds = root / "cohort";
  Catalogue cat = create_catalogue(ds, CreateCatalogueOptions{.name = "cohort"});
  auto make_subject = [&](const std::string& name, const ResolvedTemplate& t) {
    CreatePackageOptions po = cat.package_options();
    DatasetWriter w = create_package(ds / (name + ".nxr.zarr"), name, po);
    const std::string sphere = template_manifold_id(t, "sphere");
    register_to_template(w, t, {{"manifold", sphere}});
    SubjectWriter sub = SubjectWriter::open(w, name);
    const std::vector<float> a(4, 3.0f);
    FieldSpec f; f.name = "on_template"; f.path = "source/on_template"; f.kind = "map"; f.manifold_id = sphere; f.data = view(a, {4});
    sub.field(f);
    sub.complete();
  };
  make_subject("S1", A);
  {
    SubjectPackage p = SubjectPackage::open(ds / "S1.nxr.zarr");
    const PackageRecord rec = p.record();
    CHECK(rec.template_reference["name"] == "A" && rec.template_reference["version"] == "1.0" && rec.template_reference["content_hash"] == work_hash);
    CHECK(text(p.subject_row(), "template_id") == A.record.subject_id);
    CHECK(p.db().all("PRAGMA foreign_key_check").empty());
    CHECK(anchor_differences(p.db(), A.folder).empty());
    const Json b = *p.bundle();
    bool template_surface = false;
    for (const auto& m : b["manifolds"]) if (m["name"] == "sphere" && m["subject_id"] == A.record.subject_id) template_surface = true;
    CHECK(template_surface);                                                       // the bundle follows the template from the package alone
    CHECK(p.verify().ok);
  }
  // registering checks the anchors against the template it names
  {
    RegisterOptions ro;
    ro.registered_by = "test";
    ro.templates = here;
    CHECK_EQ(register_packages(ds, {ds / "S1.nxr.zarr"}, ro)[0].outcome, std::string("registered"));
    CHECK_EQ(Catalogue::open(ds).template_references().size(), std::size_t(1));
    CHECK(Catalogue::open(ds).template_references()[0]["version"] == "1.0");
    make_subject("S2", A);
    {
      DatasetWriter w = open_package_writer(ds / "S2.nxr.zarr");
      w.db().execute("UPDATE manifold SET label = 'tampered' WHERE name = 'sphere' AND subject_id = ?", {A.record.subject_id});
    }
    const auto r = register_packages(ds, {ds / "S2.nxr.zarr"}, ro);
    CHECK_EQ(r[0].outcome, std::string("refused"));
    CHECK(has(r[0].reason, "anchor rows differ") && has(r[0].reason, "manifold"));
  }
  // a cohort is never pooled across template versions
  {
    const fs::path w2 = make_template(root / "work" / "A-v2.nxr.zarr", "A");
    { DatasetWriter w = open_package_writer(w2); w.db().execute("UPDATE field SET comment = 'second version' WHERE name = 'reference_map'"); }
    const ReleaseReport r2 = release_package(w2, ReleaseOptions{"A", "2.0", "test", shelf});
    TemplateRef a2 = parse_template_ref("A@2.0");
    a2.content_hash = r2.content_hash;
    const ResolvedTemplate A2 = resolve_template(a2, here);
    make_subject("S3", A2);
    RegisterOptions ro;
    ro.registered_by = "test";
    ro.templates = here;
    CHECK_EQ(register_packages(ds, {ds / "S3.nxr.zarr"}, ro)[0].outcome, std::string("registered"));
    CHECK_EQ(Catalogue::open(ds).template_references().size(), std::size_t(2));
    CHECK_EQ(select_subjects(ds, SubjectSelection{{"S1"}, std::nullopt}).members.size(), std::size_t(1));
    const std::string why = what_of([&] { select_subjects(ds, SubjectSelection{{"S1", "S3"}, std::nullopt}); });
    CHECK(has(why, "different template versions") && has(why, "A@1.0") && has(why, "A@2.0"));
  }

  // A REGISTRATION between two templates: its own versioned package
  {
    const fs::path wb = make_template(root / "work" / "B.nxr.zarr", "B");
    const ReleaseReport relb = release_package(wb, ReleaseOptions{"B", "3.1", "test", shelf});
    TemplateRef bref = parse_template_ref("B@3.1");
    bref.content_hash = relb.content_hash;
    RegistrationSpec spec;
    spec.name = "A-to-B";
    spec.from = aref; spec.to = bref;
    spec.from_manifold = "sphere"; spec.to_manifold = "sphere";
    spec.search = here;
    spec.method = "nearest vertex on the registered spheres";
    spec.warnings = {"the correspondence is the spheres', not anatomy's", "values are smoothed by resampling"};
    spec.origin = OriginRecord{"a synthetic registration", "CC0", "the test authors", true};
    spec.package.home_dataset = "templates-work";
    CHECK_EQ(code_of([&] { RegistrationSpec s = spec; s.method.clear(); create_registration_package(root / "work" / "bad.nxr.zarr", s, nullptr); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { RegistrationSpec s = spec; s.to = s.from; create_registration_package(root / "work" / "bad.nxr.zarr", s, nullptr); }), std::string(code::invalid_argument));
    // forward: A's vertex i goes to B's vertex perm[i]; backward is its inverse
    const std::vector<std::int32_t> perm = {2, 0, 3, 1}, inverse = {1, 3, 0, 2};
    auto write_one = [](SubjectWriter& sub, const std::string& name, const std::string& from, const std::string& to, const std::vector<std::int32_t>& p) {
      const std::vector<std::int32_t> indptr = {0, 1, 2, 3, 4}, indices = p;   // CSC: column j holds one entry, in row p[j]
      const std::vector<double> data(4, 1.0);
      OperatorSpec o;
      o.name = name; o.path = name; o.kind = "transition"; o.layout = "csc"; o.from_manifold_id = from; o.n_rows = 4; o.n_cols = 4;
      o.sparse = SparseArrays{{{"data", view(data, {4})}, {"indices", view(indices, {4})}, {"indptr", view(indptr, {5})}}};
      o.cols.set("to_manifold_id", to).set("nnz", 4);
      sub.operator_(o);
    };
    const fs::path regwork = create_registration_package(root / "work" / "A-to-B.nxr.zarr", spec, [&](SubjectWriter& sub, const RegistrationIds& ids) {
      write_one(sub, "forward", ids.from_manifold_id, ids.to_manifold_id, perm);
      write_one(sub, "backward", ids.to_manifold_id, ids.from_manifold_id, inverse);
    });
    {
      SubjectPackage p = SubjectPackage::open(regwork);
      CHECK_EQ(p.record().kind, std::string("registration"));
      CHECK(p.record().template_reference["from"]["name"] == "A" && p.record().template_reference["to"]["name"] == "B");
      CHECK(p.record().template_reference["to"]["content_hash"] == relb.content_hash);
      CHECK(p.record().description["method"] == "nearest vertex on the registered spheres" && p.record().description["warnings"].size() == 2);
      CHECK(p.db().all("PRAGMA foreign_key_check").empty());
      CHECK(items_without_origin(p.db()).empty());
      CHECK(integer(*p.db().one("SELECT count(*) AS n FROM anchor_row WHERE table_name = 'manifold'"), "n") >= 2);   // both templates' spheres
      CHECK(anchor_differences(p.db(), A.folder).size() > 0);                                                      // (B's anchors are not A's rows: each is checked against its own template)
    }
    const ReleaseReport rr = release_package(regwork, ReleaseOptions{"A-to-B", "1.0", "test", shelf});
    TemplateRef rref = parse_template_ref("A-to-B@1.0");
    rref.content_hash = rr.content_hash;
    const ResolvedTemplate reg = resolve_template(rref, here);
    CHECK_EQ(reg.record.kind, std::string("registration"));
    CHECK_EQ(code_of([&] { open_package_writer(reg.folder); }), std::string(code::not_writable));
    bool listed = false;
    for (const auto& l : list_templates(here)) if (l.name == "A-to-B" && l.kind == "registration") listed = true;
    CHECK(listed);
    // the round trip on the synthetic spheres: the operator applied by hand, forward then back, is the identity; a resampled field equals the permutation
    {
      SubjectPackage p = SubjectPackage::open(reg.folder);
      auto csc = [&](const std::string& name) {
        const auto ind = ArrayReader::open(p.store(), name + "/indices").read();
        std::vector<std::int32_t> out(4);
        std::memcpy(out.data(), ind.data(), 4 * sizeof(std::int32_t));
        return out;
      };
      const auto fwd = csc("forward"), bwd = csc("backward");
      const std::vector<double> x = {10, 20, 30, 40};
      std::vector<double> y(4), z(4);
      for (int j = 0; j < 4; ++j) y[static_cast<std::size_t>(fwd[static_cast<std::size_t>(j)])] = x[static_cast<std::size_t>(j)];
      for (int j = 0; j < 4; ++j) z[static_cast<std::size_t>(bwd[static_cast<std::size_t>(j)])] = y[static_cast<std::size_t>(j)];
      CHECK(z == x);
      CHECK(y == (std::vector<double>{20, 40, 10, 30}));
    }
    // a subject records the registration it uses
    {
      DatasetWriter w = open_package_writer(ds / "S1.nxr.zarr");
      record_registration(w, rref);
    }
    CHECK(SubjectPackage::open(ds / "S1.nxr.zarr").record().registration["name"] == "A-to-B");
  }

  // the datastore is retired: nothing of a package, a catalogue or a release writes a catalog.json
  {
    bool found = false;
    for (const auto& e : fs::recursive_directory_iterator(root)) if (e.path().filename() == "catalog.json") found = true;
    CHECK(!found);
  }

  fs::remove_all(root);
  return nxrtest::finish("templates");
}
