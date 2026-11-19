// The commands' library functions (nxr/io/task/commands.h): what each command of the command line does, moved out of cli/ so the command line, the Python binding
// and the WebAssembly binding call one function. Behaviour, refusals and results are the command line's, unchanged.
#include "nxr/io/task/commands.h"

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
#include "nxr/io/small_file_store.h"
#include "nxr/io/store.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/surface.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/task.h"
#include "nxr/io/templates.h"
#include "nxr/io/writer.h"

namespace fs = std::filesystem;

namespace nxr::io::task {

namespace {

void note(const Progress& progress, const std::string& text) {
  if (progress) progress(Json{{"stage", "note"}, {"note", text}});
}

TemplateSearch search_of(const std::vector<std::string>& templates) {
  TemplateSearch s;
  for (const auto& p : templates) s.command_paths.push_back(p);
  return s;
}

// THE DATASET A COMMAND WORKS ON: a catalogue's subject package (found through the catalogue), or a one-database dataset (store formats 50-52), as every command opens it.
DatasetWriter open_dataset(const std::string& folder, const std::optional<std::string>& subject, const OpenOptions& o, const std::vector<std::string>& templates,
                           const Progress& progress) {
  if (is_catalogue_dataset(folder)) {
    if (!subject) throw Refused(folder + " is a catalogue of subject packages: name the subject to work on with --subject", true);
    return open_subject_package(folder, *subject, o, search_of(templates));
  }
  if (fs::is_regular_file(fs::path(folder) / "dataset.sqlite"))
    note(progress, folder + " is a one-database dataset; `nxr-neuro split dataset` makes it a catalogue of subject packages (the default form).");
  return open_writer(folder, o);
}

// The commands that make or change a one-database dataset itself refuse a catalogue, naming what to use.
DatasetWriter open_one_database(const std::string& folder, const std::string& what, const OpenOptions& o = {}) {
  if (is_catalogue_dataset(folder))
    throw Refused(what + " works on a one-database dataset; " + folder + " is a catalogue of subject packages (`list subjects`, `show subject` and the `compute` verbs open its packages)", true);
  return open_writer(folder, o);
}

OpenOptions reader() {
  OpenOptions o;
  o.lock = false;
  return o;
}

}  // namespace

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
    return r;
  }
  DatasetWriter ds = open_dataset(rq.dataset, std::nullopt, reader(), {}, progress);
  Json kinds = Json::object();
  std::int64_t total = 0;
  for (const auto& k : Dataset::of(ds).subject_kinds()) {
    kinds[k.kind] = k.n;
    total += k.n;
  }
  auto opt = [&](const char* c) { return is_null(ds.row(), c) ? Json() : Json(text(ds.row(), c)); };
  r.result = Json{{"dataset", text(ds.row(), "name")}, {"id", ds.id()}, {"path", text(ds.row(), "path")}, {"schema_version", schema_version()},
                  {"subjects", total}, {"kinds", kinds}, {"source_tool", opt("source_tool")}, {"source_protocol", opt("source_protocol")}, {"created_utc", opt("created_utc")}};
  return r;
}

CommandResult delete_dataset(const DeleteDatasetRequest& rq) {
  nxr::io::delete_dataset(rq.datastore, rq.name);
  CommandResult r;
  r.result = Json{{"deleted", rq.name}};
  return r;
}

CommandResult recover_dataset(const RecoverDatasetRequest& rq) {
  OpenOptions o;
  o.recover = false;
  DatasetWriter ds = open_one_database(rq.dataset, "recover dataset", o);
  const Dataset td = Dataset::of(ds);
  const Healed healed = td.recover();
  td.drain();
  write_catalog(ds.root());
  CommandResult r;
  r.result = Json{{"rolled_back", healed.rolled_back}, {"finished", healed.finished}};
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

// ---- subjects ----

CommandResult create_subject(const CreateSubjectRequest& rq, const Progress& progress) {
  if (rq.kind != "subject" && rq.kind != "template" && rq.kind != "group") throw Refused("argument --kind: invalid choice: '" + rq.kind + "' (choose from 'subject', 'template', 'group')");
  CommandResult r;
  if (is_catalogue_dataset(rq.dataset)) {   // a catalogue: a new package in the dataset's folder, registered
    const Catalogue cat = Catalogue::open(rq.dataset);
    CreatePackageOptions po = cat.package_options();
    po.subject.kind = rq.kind;
    po.subject.created_by = rq.created_by;
    po.built_by = rq.created_by;
    const fs::path pkg = cat.folder() / (rq.name + kPackageSuffix);
    DatasetWriter w = nxr::io::create_package(pkg, rq.name, po);
    const std::string sid = Dataset::of(w).package_subject_id();
    w.close();
    RegisterOptions ro;
    ro.registered_by = rq.created_by;
    const auto res = register_packages(rq.dataset, {pkg}, ro);
    if (res.empty() || res[0].outcome == "refused") throw Error(code::invalid_argument, rq.name + ": " + (res.empty() ? std::string("not registered") : res[0].reason));
    r.result = Json{{"subject", rq.name}, {"kind", rq.kind}, {"id", sid}, {"path", pkg.filename().string()}};
    return r;
  }
  DatasetWriter ds = open_dataset(rq.dataset, std::nullopt, OpenOptions{}, {}, progress);
  SubjectOptions o;
  o.kind = rq.kind;
  o.created_by = rq.created_by;
  SubjectWriter sub = SubjectWriter::create(ds, rq.name, o);
  sub.compose([] {});
  r.result = Json{{"subject", rq.name}, {"kind", rq.kind}, {"id", sub.id()}, {"path", text(sub.row(), "path")}};
  return r;
}

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
    return r;
  }
  if (rq.subjects || rq.where)
    throw Refused("--subjects and --where need a catalogue dataset (made by `create dataset --dataset PATH`); " + rq.dataset + " is a one-database dataset");
  DatasetWriter ds = open_dataset(rq.dataset, std::nullopt, reader(), {}, progress);
  Json rows = Json::array();
  for (const auto& s : Dataset::of(ds).subjects()) rows.push_back(Json{{"name", s.name}, {"kind", s.kind}, {"status", s.status}, {"path", s.path}});
  r.result = Json{{"dataset", text(ds.row(), "name")}, {"subjects", rows}};
  return r;
}

CommandResult show_subject(const ShowSubjectRequest& rq, const Progress& progress) {
  DatasetWriter ds = open_dataset(rq.dataset, rq.subject, reader(), rq.templates, progress);
  SubjectWriter s = SubjectWriter::open(ds, rq.subject);
  const NodeCounts n = Subject::of(s).node_counts();
  CommandResult r;
  r.result = Json{{"subject", s.name()}, {"kind", text(s.row(), "kind")}, {"path", text(s.row(), "path")}, {"manifolds", n.manifolds},
                  {"fields", n.fields}, {"operators", n.operators}, {"selections", n.selections}};
  return r;
}

CommandResult delete_subject(const DeleteSubjectRequest& rq) {
  DatasetWriter ds = open_one_database(rq.dataset, "delete subject");
  OpenSubjectOptions oo;
  oo.created_by = rq.created_by;
  SubjectWriter sub = SubjectWriter::open(ds, rq.subject, oo);
  CommandResult r;
  const Dataset td = Dataset::of(ds);
  if (!rq.node) {
    td.remove_subject(sub.id());
    write_catalog(ds.root());
    r.result = Json{{"removed", rq.subject}};
    return r;
  }
  const std::string& node = *rq.node;
  for (const Cls c : {Cls::manifold, Cls::field, Cls::operator_, Cls::selection}) {
    const char* t = to_string(c);
    if (const auto row = sub.find(t, node)) {
      td.remove(c, text(*row, "id"));
      r.result = Json{{"removed", node}, {"table", t}};
      return r;
    }
  }
  // no store path of that name: the node's NAME (a source estimate or a joint Selection has no path); unique within the subject, else refused with the matches
  std::vector<std::pair<Cls, std::string>> hits;   // class, id
  for (const Cls c : {Cls::manifold, Cls::field, Cls::operator_, Cls::selection, Cls::statistic})
    if (const auto row = sub.by_name(to_string(c), node)) hits.push_back({c, text(*row, "id")});
  if (hits.size() == 1) {
    td.remove(hits[0].first, hits[0].second);
    r.result = Json{{"removed", node}, {"table", to_string(hits[0].first)}, {"by", "name"}};
    return r;
  }
  if (hits.size() > 1) {
    std::string list;
    for (const auto& h : hits) list += (list.empty() ? "" : ", ") + std::string(to_string(h.first)) + " " + h.second;
    throw Error(code::invalid_argument, node + ": several nodes of " + rq.subject + " have this name (" + list + "); name a store path with --node instead");
  }
  throw Error(code::no_such_row, node + ": no node of " + rq.subject + " by store path or by name");
}

CommandResult add_surface(const AddSurfaceRequest& rq, const Progress& progress) {
  Mesh mesh = read_mesh(rq.file);
  const std::string name = sanitize_node_name(rq.name.value_or(mesh_name(rq.file)));
  DatasetWriter ds = open_dataset(rq.dataset, rq.subject, OpenOptions{}, rq.templates, progress);
  OpenSubjectOptions oo;
  oo.created_by = rq.created_by;
  SubjectWriter sub = SubjectWriter::open(ds, rq.subject, oo);
  for (auto& x : mesh.vertices) x *= rq.scale;
  SurfaceOptions so;
  so.name = name;
  so.source = fs::path(rq.file).filename().string();
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
  if (rq.templates) ro.templates = search_of(*rq.templates);
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
