// The commands' library functions (nxr/io/task/commands.h): what each command of the command line does, moved out of cli/ so the command line, the Python binding
// and the WebAssembly binding call one function. Behaviour, refusals and results are the command line's, unchanged.
#include "nxr/io/task/commands.h"
#include "nxr/io/task/handle.h"
#include "commands_open.h"

#include <zlib.h>

#include <cstdio>
#include <fstream>
#include <set>

#include "nxr/convert/templates.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/compact.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/errors.h"
#include "nxr/io/migrate.h"
#include "nxr/io/npy.h"
#include "nxr/io/bundle.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/statistic.h"
#include "nxr/io/store.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/task.h"
#include "nxr/io/templates.h"
#include "nxr/io/writer.h"

namespace fs = std::filesystem;

namespace nxr::io::task {

using namespace commands_detail;

// ---- datasets ----

CommandResult create_dataset(const CreateDatasetRequest& rq, const Progress& progress) {
  // THE DEFAULT FORM: `dataset` makes a catalogue, the folder of a dataset that holds its subjects' packages (`create subject` adds one; `register package` lists
  // packages made elsewhere). `datastore` and `name` (the old spelling's) still make the one-database dataset DATASTORE/NAME; `split dataset` turns it into a catalogue.
  if (rq.dataset && rq.datastore) throw Refused("--dataset (a catalogue) and --datastore (a one-database dataset) are alternatives");
  if (!rq.dataset && !rq.datastore) throw Refused("the following arguments are required: --dataset");
  CommandResult r;
  if (rq.dataset) {
    const std::pair<const char*, bool> one_database[] = {{"--templates", rq.templates.has_value()}, {"--default", rq.default_.has_value()}, {"--modes", rq.modes.has_value()},
                                                         {"--atlas-finest", rq.atlas_finest.has_value()}, {"--lc-scales", rq.lc_scales.has_value()}, {"--store-format", rq.store_format.has_value()}};
    for (const auto& [x, given] : one_database)
      if (given) throw Refused(std::string(x) + " belongs to the one-database form (--datastore); a catalogue's template is a template package (`import template`, `list templates`)");
    CreateCatalogueOptions co;
    co.name = rq.name.value_or(fs::path(*rq.dataset).filename().string());
    if (co.name.empty()) co.name = fs::path(*rq.dataset).parent_path().filename().string();
    co.source_tool = rq.source_tool;
    co.source_protocol = rq.source_protocol;
    co.source_path = rq.source_path;
    co.created_by = rq.created_by;
    co.built_by = rq.created_by;
    const Catalogue c = create_catalogue(*rq.dataset, co);
    r.result = Json{{"dataset", text(c.dataset_row(), "name")}, {"id", text(c.dataset_row(), "id")}, {"folder", c.folder().string()}, {"form", "catalogue"}};
    return r;
  }
  const CreateOptions o = one_database_options(rq, progress);
  if (rq.templates) throw Refused("--templates writes the default subject with its frames and group tree, and needs the pipelines (-DNXR_NEURO_BUILD_PIPELINES=ON)");
  DatasetWriter ds = nxr::io::create_dataset(*rq.datastore, *rq.name, o);
  r.result = Json{{"dataset", text(ds.row(), "name")}, {"id", ds.id()}, {"database", (ds.folder() / kDatasetFile).string()}};
  ds.close();
  return r;
}

CreateOptions one_database_options(const CreateDatasetRequest& rq, const Progress& progress) {
  if (!rq.name) throw Refused("the following arguments are required: --name (with --datastore)");
  note(progress, "--datastore makes the one-database form; `create dataset --dataset PATH` makes a catalogue of subject packages (the default form).");
  if (!rq.templates && (rq.default_ || rq.modes || rq.atlas_finest || rq.lc_scales))
    throw Refused("--default, --modes, --atlas-finest and --lc-scales need --templates (the default subject)");
  CreateOptions o;
  o.source_tool = rq.source_tool;
  o.source_protocol = rq.source_protocol;
  o.source_path = rq.source_path;
  o.created_by = rq.created_by;
  if (rq.store_format) {
    if (*rq.store_format != "51" && *rq.store_format != "52") throw Refused("--store-format is 51 or 52");
    o.store_format = std::stoi(*rq.store_format);
  }
  return o;
}

CommandResult list_datasets(const ListDatasetsRequest& rq) {
  const Json cat = nxr::io::list_datasets(rq.datastore);
  CommandResult r;
  r.result = Json{{"datastore", cat.contains("id") ? cat["id"] : Json()}, {"datasets", cat.contains("datasets") ? cat["datasets"] : Json::array()}};
  return r;
}

CommandResult show_dataset(const ShowDatasetRequest& rq, const Progress& progress) {
  CommandResult r;
  if (is_catalogue_dataset(rq.dataset)) {   // a catalogue: its subjects by kind, from the entries (no package is opened)
    const Catalogue cat = Catalogue::open(rq.dataset);
    Json kinds = Json::object();
    std::int64_t total = 0;
    for (const auto& e : cat.entries()) if (e.status != "replaced") { kinds[e.kind] = (kinds.contains(e.kind) ? kinds[e.kind].get<std::int64_t>() : 0) + 1; ++total; }
    const Row d = cat.dataset_row();
    auto opt = [&](const char* c) { return is_null(d, c) ? Json() : Json(text(d, c)); };
    r.result = Json{{"dataset", text(d, "name")}, {"id", text(d, "id")}, {"path", cat.folder().string()}, {"form", "catalogue"}, {"subjects", total}, {"kinds", kinds},
                    {"source_tool", opt("source_tool")}, {"source_protocol", opt("source_protocol")}, {"created_utc", opt("created_utc")}};
    Json unavailable = Json::array();
    r.result["nodes"] = class_counts(rq.dataset, progress, &unavailable);
    if (!unavailable.empty()) r.result["nodes_unavailable"] = unavailable;
    return r;
  }
  Lease lease = open_reader(rq.dataset, std::nullopt, progress);
  DatasetWriter& ds = *lease;
  Json kinds = Json::object();
  std::int64_t total = 0;
  for (const auto& k : Dataset::of(ds).subject_kinds()) {
    kinds[k.kind] = k.n;
    total += k.n;
  }
  auto opt = [&](const char* c) { return is_null(ds.row(), c) ? Json() : Json(text(ds.row(), c)); };
  r.result = Json{{"dataset", text(ds.row(), "name")}, {"id", ds.id()}, {"path", text(ds.row(), "path")}, {"schema_version", schema_version()},
                  {"subjects", total}, {"kinds", kinds}, {"source_tool", opt("source_tool")}, {"source_protocol", opt("source_protocol")}, {"created_utc", opt("created_utc")}};
  r.result["nodes"] = class_counts(rq.dataset, progress, nullptr);
  return r;
}

CommandResult delete_dataset(const DeleteDatasetRequest& rq) {
  nxr::io::delete_dataset(rq.datastore, rq.name);
  CommandResult r;
  r.result = Json{{"deleted", rq.name}};
  return r;
}

CommandResult recover_dataset(const RecoverDatasetRequest& rq) {
  // the unfinished writes in pieces (add field | statistic --begin, never committed) are rolled back too: they live beside the database (nxr/io/task/commands.h `PieceOptions`)
  CommandResult r;
  if (is_catalogue_dataset(rq.dataset)) {   // a catalogue's packages recover when they are opened; what is left to roll back here is its unfinished writes
    r.result = Json{{"rolled_back", 0}, {"finished", 0}, {"writes_rolled_back", pieces::discard_all(rq.dataset)}};
    return r;
  }
  OpenOptions o;
  o.recover = false;
  DatasetWriter ds = open_one_database(rq.dataset, "recover dataset", o);
  const Dataset td = Dataset::of(ds);
  const Healed healed = td.recover();
  td.drain();
  write_catalog(ds.root());
  r.result = Json{{"rolled_back", healed.rolled_back}, {"finished", healed.finished}, {"writes_rolled_back", pieces::discard_all(rq.dataset)}};
  ds.close();
  return r;
}

CommandResult compact_dataset(const CompactDatasetRequest& rq) {
  // FORMAT 50 -> 51 -> 52, array by array (io/include/nxr/io/compact.h): the new layout beside the old, every value compared byte for byte, then swapped; resumable and
  // idempotent. The writer lock is taken (a dataset another writer holds is refused); nothing is recovered. A dry run reports the files and folders before and after.
  OpenOptions o;
  o.recover = false;
  DatasetWriter ds = open_one_database(rq.dataset, "compact dataset", o);
  CompactOptions co;
  co.subject = rq.subject;
  co.dry_run = rq.dry_run;
  co.shard = !rq.no_shard;
  co.small_files = !rq.no_small_files;
  CommandResult r;
  r.result = nxr::io::compact_dataset(ds, co);
  r.result.erase("stage");   // the report's own result marker: the command line adds it
  ds.close();
  return r;
}

CommandResult split_dataset(const SplitDatasetRequest& rq, const Progress& progress) {
  SplitOptions o;
  o.registered_by = rq.registered_by;
  o.dry_run = rq.dry_run;
  o.on_stage = [&](const std::string& stage, const std::string& subject) { if (progress) progress(Json{{"stage", stage}, {"subject", subject}}); };
  CommandResult r;
  r.result = Json{{"report", nxr::io::split_dataset(rq.dataset, o)}};
  return r;
}

CommandResult join_dataset(const JoinDatasetRequest& rq) {
  CommandResult r;
  r.result = Json{{"report", nxr::io::join_dataset(rq.dataset)}};
  return r;
}

CommandResult migrate_measurements(const MigrateMeasurementsRequest& rq, const Progress& progress) {
  // THE MEASUREMENTS' STORAGE (schema 53, io/migrate.h): one table for a value over a selection's members. Subject by subject, resumable, verified; the old statistics are
  // KEPT unless the explicit second step is asked.
  MeasurementStorageOptions o;
  o.dry_run = rq.dry_run;
  o.remove_statistics = rq.remove_statistics;
  o.on_stage = [&](const std::string& stage, const std::string& subject) { if (progress) progress(Json{{"stage", stage}, {"subject", subject}}); };
  CommandResult r;
  r.result = Json{{"report", migrate_measurement_storage(rq.dataset, o)}};
  return r;
}

CommandResult migrate_template_links(const MigrateTemplateLinksRequest& rq, const Progress& progress) {
  if (progress) progress(Json{{"stage", rq.dry_run ? "plan" : "migrate"}, {"dataset", rq.dataset}});
  CommandResult r;
  r.result = Json{{"report", nxr::io::migrate_template_links(rq.dataset, nxr::io::TemplateLinksOptions{rq.registered_by, rq.dry_run})}};
  return r;
}

// ---- subjects ----

CommandResult create_subject(const CreateSubjectRequest& rq, const Progress& progress) {
  if (rq.kind != "subject" && rq.kind != "template") throw Refused("argument --kind: invalid choice: '" + rq.kind + "' (choose from 'subject', 'template'; a cohort's results go into the dataset's template)");
  CommandResult r;
  if (is_catalogue_dataset(rq.dataset)) {   // a catalogue: a new package in the dataset's folder, registered
    const Catalogue cat = Catalogue::open(rq.dataset);
    SubjectOptions so;
    so.kind = rq.kind;
    so.created_by = rq.created_by;
    const fs::path pkg = member_package_path(cat, rq.name);
    DatasetWriter w = create_member_package(cat, rq.name, so, rq.created_by);
    const std::string sid = Dataset::of(w).package_subject_id();
    w.close();
    register_member(rq.dataset, pkg, rq.name, rq.created_by);
    r.result = Json{{"subject", rq.name}, {"kind", rq.kind}, {"id", sid}, {"path", pkg.filename().string()}};
    return r;
  }
  DatasetWriter ds = open_dataset(rq.dataset, std::nullopt, OpenOptions{}, progress);
  SubjectOptions o;
  o.kind = rq.kind;
  o.created_by = rq.created_by;
  SubjectWriter sub = SubjectWriter::create(ds, rq.name, o);
  sub.compose([] {});
  r.result = Json{{"subject", rq.name}, {"kind", rq.kind}, {"id", sub.id()}, {"path", text(sub.row(), "path")}};
  return r;
}

namespace {
// HOW MUCH A SUBJECT HOLDS, for the listings (the Subjects table, a subject's page): the nodes of each class, the volumes, surfaces and recordings among them, its acquisitions and the
// seconds its recordings hold
Json subject_counts(const Database& db, const std::string& subject_id) {
  auto count = [&](const std::string& sql) { const auto r = db.one(sql, {V(subject_id)}); return r ? integer(*r, "n") : std::int64_t(0); };
  Json out = Json{{"acquisitions", count("SELECT count(*) AS n FROM acquisition WHERE subject_id = ?")},
                  {"manifolds", count("SELECT count(*) AS n FROM manifold WHERE subject_id = ? AND status = 'complete'")},
                  {"surfaces", count("SELECT count(*) AS n FROM manifold WHERE subject_id = ? AND status = 'complete' AND type = 'surface' AND component_of_id IS NULL")},
                  {"volumes", count("SELECT count(*) AS n FROM manifold WHERE subject_id = ? AND status = 'complete' AND type = 'volume'")},
                  {"fields", count("SELECT count(*) AS n FROM field WHERE subject_id = ? AND status = 'complete'")},
                  {"recordings", count("SELECT count(*) AS n FROM field WHERE subject_id = ? AND status = 'complete' AND kind = 'recording'")},
                  {"selections", count("SELECT count(*) AS n FROM selection WHERE subject_id = ? AND status = 'complete'")},
                  {"operators", count("SELECT count(*) AS n FROM operator WHERE subject_id = ? AND status = 'complete'")},
                  {"statistics", has_table(db, "statistic") ? count("SELECT count(*) AS n FROM statistic WHERE subject_id = ? AND status = 'complete'") : std::int64_t(0)}};
  const auto sec = db.one("SELECT coalesce(sum(tm.n_vertices * td.spacing), 0) AS seconds FROM field f JOIN manifold pm ON pm.id = f.manifold_id AND pm.type = 'product' "
                          "JOIN geometry_dimension d1 ON d1.geometry_id = pm.geometry_id AND d1.ordinal = 1 JOIN manifold tm ON tm.id = d1.manifold_id "
                          "JOIN geometry_dimension td ON td.geometry_id = tm.geometry_id AND td.ordinal = 0 WHERE f.subject_id = ? AND f.kind = 'recording' AND f.status = 'complete'", {V(subject_id)});
  out["seconds"] = sec ? row_json(*sec)["seconds"] : Json(0.0);
  return out;
}
}  // namespace

CommandResult list_subjects(const ListSubjectsRequest& rq, const Progress& progress) {
  CommandResult r;
  if (fs::exists(fs::path(rq.dataset) / kCatalogueFile)) {   // a catalogue of subject packages: by name, or by a filter over the participants
    const Catalogue cat = Catalogue::open(rq.dataset);
    Json rows = Json::array();
    if (rq.subjects || rq.where) {
      SubjectSelection sel;
      if (rq.subjects) sel.subjects = *rq.subjects;
      sel.where = rq.where;
      const SelectedSubjects chosen = select_subjects(rq.dataset, sel);
      for (const auto& m : chosen.members) rows.push_back(m.to_json());
      r.result = Json{{"dataset", text(cat.dataset_row(), "name")}, {"subjects", rows}, {"selection", chosen.provenance()}};
    } else {
      for (const auto& e : cat.entries()) rows.push_back(e.to_json());
      r.result = Json{{"dataset", text(cat.dataset_row(), "name")}, {"subjects", rows}};
    }
    // each member's counts, read from its package (the catalogue records only some); a package that cannot be opened has none (null)
    for (auto& row : r.result["subjects"]) {
      row["counts"] = Json();
      if (!row.contains("path") || !row["path"].is_string() || !row.contains("subject_id") || !row["subject_id"].is_string()) continue;
      const fs::path pkg = fs::path(row["path"].get<std::string>()).is_absolute() ? fs::path(row["path"].get<std::string>()) : fs::path(rq.dataset) / row["path"].get<std::string>();
      try { const SubjectPackage p = SubjectPackage::open(pkg); row["counts"] = subject_counts(p.db(), row["subject_id"].get<std::string>()); } catch (const std::exception&) {}
    }
    return r;
  }
  if (rq.subjects || rq.where)
    throw Refused("--subjects and --where need a catalogue dataset (made by `create dataset --dataset PATH`); " + rq.dataset + " is a one-database dataset");
  Lease lease = open_reader(rq.dataset, std::nullopt, progress);
  DatasetWriter& ds = *lease;
  Json rows = Json::array();
  for (const auto& s : Dataset::of(ds).subjects()) rows.push_back(Json{{"name", s.name}, {"kind", s.kind}, {"status", s.status}, {"path", s.path}, {"counts", subject_counts(ds.db(), s.id)}});
  r.result = Json{{"dataset", text(ds.row(), "name")}, {"subjects", rows}};
  return r;
}

CommandResult show_subject(const ShowSubjectRequest& rq, const Progress& progress) {
  Lease lease = open_reader(rq.dataset, rq.subject, progress);
  DatasetWriter& ds = *lease;
  SubjectWriter s = SubjectWriter::open(ds, rq.subject);
  const NodeCounts n = Subject::of(s).node_counts();
  CommandResult r;
  r.result = Json{{"subject", s.name()}, {"kind", text(s.row(), "kind")}, {"path", text(s.row(), "path")}, {"manifolds", n.manifolds},
                  {"fields", n.fields}, {"operators", n.operators}, {"selections", n.selections}, {"counts", subject_counts(s.db(), s.id())}};
  return r;
}

CommandResult delete_subject(const DeleteSubjectRequest& rq) {
  if (rq.node) {   // one node, named CLASS:NAME (commands_delete.cpp); a catalogue's package opens through the catalogue
    DatasetWriter ds = is_catalogue_dataset(rq.dataset) ? open_dataset(rq.dataset, rq.subject, OpenOptions{}, {}) : open_one_database(rq.dataset, "delete subject");
    return delete_node(ds, rq.subject, *rq.node, rq.created_by, rq.by);
  }
  if (rq.by) throw Refused("--by goes with --node: it names the author of the node to delete");
  DatasetWriter ds = open_one_database(rq.dataset, "delete subject");
  OpenSubjectOptions oo;
  oo.created_by = rq.created_by;
  SubjectWriter sub = SubjectWriter::open(ds, rq.subject, oo);
  const Dataset td = Dataset::of(ds);
  td.remove_subject(sub.id());
  write_catalog(ds.root());
  CommandResult r;
  r.result = Json{{"removed", rq.subject}};
  return r;
}

CommandResult add_surface(const AddSurfaceRequest& rq, const Progress& progress) {
  // a mesh file, or (--vertices and --faces) the surface from arrays: vertices [n x 3], faces [m x 3] 0-based, as .npy files
  const bool arrays = rq.vertices || rq.faces;
  if (arrays && !rq.file.empty()) throw Refused("give a mesh FILE or --vertices and --faces (.npy), not both");
  if (!arrays && rq.file.empty()) throw Refused("the following arguments are required: FILE (or --vertices and --faces)");
  if (arrays && !(rq.vertices && rq.faces)) throw Refused("a surface from arrays takes both --vertices FILE.npy [n x 3] and --faces FILE.npy [m x 3]");
  if (arrays && !rq.name) throw Refused("a surface from arrays is named by --name");
  Mesh mesh;
  if (arrays) {
    const std::string vname = rq.vertices->label(), fname = rq.faces->label();
    const std::vector<std::int64_t> vshape = array_shape(*rq.vertices), fshape = array_shape(*rq.faces);
    if (vshape.size() != 2 || vshape[1] != 3) throw Refused("--vertices " + vname + ": the vertices are [n x 3]");
    if (fshape.size() != 2 || fshape[1] != 3) throw Refused("--faces " + fname + ": the faces are [m x 3], three vertex indices each (0-based)");
    const std::string fdtype = array_dtype(*rq.faces);
    if (fdtype.rfind("int", 0) != 0 && fdtype.rfind("uint", 0) != 0) throw Refused("--faces " + fname + ": vertex indices are integers (" + fdtype + " given)");
    mesh.vertices = array_all<double>(*rq.vertices);
    for (const auto i : array_all<std::int64_t>(*rq.faces)) {
      if (i < 0 || i >= vshape[0]) throw Refused("--faces " + fname + ": vertex " + std::to_string(i) + " is not one of the " + std::to_string(vshape[0]) + " vertices");
      mesh.faces.push_back(static_cast<std::int32_t>(i));
    }
  } else {
    mesh = read_mesh(rq.file);
  }
  const std::string name = sanitize_node_name(rq.name.value_or(mesh_name(rq.file)));
  DatasetWriter ds = open_dataset(rq.dataset, rq.subject, OpenOptions{}, progress);
  OpenSubjectOptions oo;
  oo.created_by = rq.by && !rq.by->empty() ? *rq.by : rq.created_by;   // the author (`--by`), else the program
  SubjectWriter sub = SubjectWriter::open(ds, rq.subject, oo);
  for (auto& x : mesh.vertices) x *= rq.scale;
  SurfaceOptions so;
  so.name = name;
  so.source = arrays ? rq.vertices->label() + ", " + rq.faces->label() : fs::path(rq.file).filename().string();
  so.primary = !rq.no_primary;
  SurfaceInfo info;
  sub.compose([&] { info = nxr::io::add_surface(sub, so, mesh.vertices, mesh.faces); });
  CommandResult r;
  r.result = Json{{"name", info.name}, {"folder", info.folder}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"atlases", info.atlases},
                  {"registration", info.registration}, {"hemispheres", info.hemispheres}, {"flipped", info.flipped}, {"mesh_health", info.mesh_health}};
  return r;
}

// ---- subject packages and the catalogue ----

CommandResult create_package(const CreatePackageRequest& rq) {
  CreatePackageOptions o;
  o.home_dataset = rq.home_dataset;
  o.subject.kind = rq.kind;
  o.subject.created_by = rq.created_by;
  o.built_by = rq.created_by;
  DatasetWriter w = nxr::io::create_package(rq.package, rq.name, o);
  CommandResult r;
  r.result = Json{{"package", rq.package}, {"subject_id", Dataset::of(w).package_subject_id()}};
  w.close();
  r.result["content_hash"] = SubjectPackage::open(rq.package).record().content_hash;
  return r;
}

CommandResult verify_package(const VerifyPackageRequest& rq) {
  const VerifyReport v = SubjectPackage::open(rq.package).verify();
  Json problems = Json::array();
  for (const auto& p : v.problems) problems.push_back(p);
  CommandResult r;
  r.result = Json{{"ok", v.ok}, {"content_hash", v.content_hash}, {"problems", problems}};
  return r;
}

CommandResult show_package(const ShowPackageRequest& rq) {
  CommandResult r;
  r.result = Json{{"record", SubjectPackage::open(rq.package).record().to_json()}};
  return r;
}

CommandResult register_package(const RegisterPackageRequest& rq) {
  std::vector<fs::path> pk(rq.package.begin(), rq.package.end());
  if (rq.scan) { if (!pk.empty()) throw Refused("--scan and --package are alternatives"); pk = scan_packages(rq.dataset); }
  if (pk.empty()) throw Refused("name the packages (--package P ...) or --scan the dataset folder");
  RegisterOptions ro;
  ro.registered_by = rq.registered_by;
  ro.replace = rq.replace;
  Json out = Json::array();
  for (const auto& p : register_packages(rq.dataset, pk, ro)) out.push_back(Json{{"package", p.package.string()}, {"name", p.name}, {"outcome", p.outcome}, {"reason", p.reason}});
  CommandResult r;
  r.result = Json{{"results", out}};
  return r;
}

CommandResult list_members(const ListMembersRequest& rq) {
  Json out = Json::array();
  for (const auto& e : refresh_member_status(rq.dataset)) out.push_back(e.to_json());
  CommandResult r;
  r.result = Json{{"subjects", out}};
  return r;
}

CommandResult create_index(const CreateIndexRequest& rq) {
  const IndexReport ir = build_index(rq.dataset, rq.force);
  CommandResult r;
  r.result = Json{{"rebuilt", ir.rebuilt}, {"subjects", ir.subjects}};
  return r;
}

CommandResult release_template(const ReleaseTemplateRequest& rq) {
  const ReleaseReport rr = release_package(rq.package, ReleaseOptions{rq.name, rq.template_version, rq.released_by, rq.into});
  CommandResult r;
  r.result = Json{{"folder", rr.folder.string()}, {"content_hash", rr.content_hash}};
  return r;
}

CommandResult release_redistributable(const ReleaseRedistributableRequest& rq) {
  const ReleaseReport rr = nxr::io::release_redistributable(rq.package, rq.into, rq.released_by);
  CommandResult r;
  r.result = Json{{"folder", rr.folder.string()}, {"content_hash", rr.content_hash}, {"left_out", rr.left_out}};
  return r;
}

CommandResult list_templates(const ListTemplatesRequest& rq) {
  Json out = Json::array();
  for (const auto& l : nxr::io::list_templates(search_of(rq.templates)))
    out.push_back(Json{{"folder", l.folder.string()}, {"name", l.name}, {"version", l.version}, {"edition", l.edition}, {"kind", l.kind}, {"content_hash", l.content_hash}});
  CommandResult r;
  r.result = Json{{"templates", out}};
  return r;
}

CommandResult create_template(const CreateTemplateRequest& rq) {
  nxr::convert::BuildRequest b;
  b.recipe = rq.recipe;
  b.sources = rq.sources;
  b.package = rq.package;
  if (rq.name) b.name = *rq.name;
  if (rq.home_dataset) b.home_dataset = *rq.home_dataset;
  b.created_by = rq.created_by;
  b.accept_unlisted_sources = rq.accept_unlisted_sources;
  b.cortex_mask = rq.cortex_mask;
  b.scalp = rq.scalp;
  if (rq.surface) b.surfaces = *rq.surface;
  if (rq.parcellation) b.annotations = *rq.parcellation;
  if (rq.hcp_trk) b.hcp_trk = fs::path(*rq.hcp_trk);
  if (rq.hcp_end_surface) b.hcp_end_surface = *rq.hcp_end_surface;
  if (rq.max_end_distance) b.hcp_max_end_distance_mm = *rq.max_end_distance;
  if (rq.hcp_points) b.hcp_points = *rq.hcp_points;
  CommandResult r;
  r.result = nxr::convert::build_template(b);
  return r;
}

// ---- reading ----

namespace {

std::string crc_text(const std::vector<std::uint8_t>& b) {
  uLong c = crc32(0L, Z_NULL, 0);
  for (std::size_t o = 0; o < b.size(); o += 1u << 30) c = crc32(c, b.data() + o, static_cast<uInt>(std::min<std::size_t>(1u << 30, b.size() - o)));
  char buf[16];
  std::snprintf(buf, sizeof buf, "%08lx", static_cast<unsigned long>(c));
  return buf;
}

// The dataset's keys (`<subject>/<array>/…`) as its format keeps them: files, or (format 52) a subject's small keys in its small_files.sqlite. Read only.
struct DatasetKeys {
  static fs::path normal(const fs::path& p) { fs::path q = fs::absolute(p).lexically_normal(); if (!q.has_filename()) q = q.parent_path(); return q; }
  explicit DatasetKeys(const fs::path& dataset_folder)
      : folder(normal(dataset_folder)), datastore(folder.parent_path(), SmallFileOptions{kSmallFileThreshold, true}), keys(datastore, folder.filename().string()) {}
  fs::path folder;
  DatastoreStore datastore;
  PrefixStore keys;
};

bool is_array(const Store& store, const std::string& key) {
  const std::string t = store.get_text(join_key(key, "zarr.json")).value_or("");
  return t.find("\"node_type\":\"array\"") != std::string::npos || t.find("\"node_type\": \"array\"") != std::string::npos;
}

// An array, or the arrays inside a group (a sparse operator or field is a group of data · indices · indptr).
void collect(const Store& store, const std::string& key, std::vector<std::string>& out) {
  if (!store.exists(join_key(key, "zarr.json"))) return;
  if (is_array(store, key)) { out.push_back(key); return; }
  for (const auto& k : store.list(key).prefixes) collect(store, join_key(key, k), out);
}

}  // namespace

CommandResult read_dataset(const ReadDatasetRequest& rq) {
  const fs::path folder = rq.folder;
  const Dataset ds = Dataset::open_readonly(folder);
  CommandResult r;
  if (rq.array) {
    const auto& [subject, path] = *rq.array;
    const auto s = ds.find_subject(subject);
    if (!s) throw Error(code::no_such_row, "no subject \"" + subject + "\"");
    const DatasetKeys dk(folder);
    const Store& store = dk.keys;
    const ArrayReader a = ArrayReader::open(store, join_key(s->path, path));
    std::vector<std::int64_t> lo(a.shape().size(), 0), hi = a.shape();
    for (std::size_t d = 0; d < rq.region.size(); ++d) {
      if (d >= lo.size()) throw Error(code::invalid_argument, "region has more axes than the array");
      lo[d] = rq.region[d][0];
      hi[d] = rq.region[d][1];
    }
    CommandArray out;
    out.name = path;
    out.dtype = dtype_to_string(a.dtype());
    out.bytes = a.read(lo, hi);
    out.shape.resize(lo.size());
    for (std::size_t d = 0; d < lo.size(); ++d) out.shape[d] = hi[d] - lo[d];
    r.result = Json{{"shape", out.shape}, {"dtype", out.dtype}, {"checksum", crc_text(out.bytes)}};
    if (rq.out) std::ofstream(*rq.out, std::ios::binary).write(reinterpret_cast<const char*>(out.bytes.data()), static_cast<std::streamsize>(out.bytes.size()));
    r.arrays.push_back(std::move(out));
    return r;
  }
  Json subjects = Json::array(), arrays = Json::array();
  for (const auto& s : ds.subjects()) subjects.push_back(Json{{"name", s.name}, {"path", s.path}, {"kind", s.kind}});
  std::set<std::string> seen;
  const DatasetKeys dk(folder);
  const Store& store = dk.keys;
  for (const auto& row : ds.stored_arrays()) {
    std::vector<std::string> keys;
    const std::string& sp = row.subject_path;
    collect(store, join_key(sp, row.path), keys);
    for (const auto& key : keys) {
      const std::string rel = key.substr(sp.size() + 1);   // the subject's own key, as the rows state it
      if (!seen.insert(sp + "\n" + rel).second) continue;
      const ArrayReader a = ArrayReader::open(store, key);
      arrays.push_back(Json{{"subject", row.subject_name}, {"record", row.record}, {"path", rel}, {"shape", a.shape()}, {"dtype", dtype_to_string(a.dtype())}, {"checksum", crc_text(a.read())}});
    }
  }
  r.result = Json{{"subjects", subjects}, {"arrays", arrays}};
  return r;
}

}  // namespace nxr::io::task
