// nxr-neuro — the command line, the verbs of nxr's: dataset · subject · surface (and read, step 1's reader).
//
//   nxr-neuro dataset create --datastore D --name N [--source-tool T --source-protocol P --source-path S] [--store-format 52|51]   (default 52: each subject's small keys in its small_files.sqlite)
//                            [--templates FS_SUBJECTS [--default fsaverage5] [--modes K] [--atlas-depth 8] [--lc-levels 4,6]]   with the default subject (io/default_subject.h)
//                            its frames (pipelines/frames.h frames_compute) and its group tree with the Levi-Civita connection (pipelines/tiles.h group_tree_compute)
//   nxr-neuro dataset list   --datastore D
//   nxr-neuro dataset show   --dataset D/N
//   nxr-neuro dataset delete --datastore D --name N
//   nxr-neuro dataset recover --dataset D/N
//   nxr-neuro subject create --dataset D/N --name S [--kind subject|template|group]
//   nxr-neuro subject list   --dataset D/N
//   nxr-neuro subject show   --dataset D/N --subject S
//   nxr-neuro subject delete --dataset D/N --subject S [--node PATH]
//   nxr-neuro surface add    --dataset D/N --subject S FILE [--name N] [--scale F] [--primary | --no-primary]
//   nxr-neuro convert list|scan|subject|surface ...            (Brainstorm: convert_command.cpp)
//   nxr-neuro eigen solve --dataset D/N --subject S [--modes K] [--surface N] [--replace]     (pipelines)
//   nxr-neuro headmodel compute | noisecov compute | inverse compute | psd compute ...                                           (cli/meeg_command.cpp)
//   nxr-neuro read <dataset folder> [--array <subject> <path> [--region a:b,c:d] [--out FILE]]
//
// Like nxr's, progress and results are JSON lines on stdout (`stage` keyed); a failure is a `stage: error` line and exit 1 (a usage error: exit 2).
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>

#include "nxr/io/default_subject.h"
#ifdef NXR_NEURO_WITH_PIPELINES
#include "nxr/pipelines/constant_q.h"
#include "nxr/pipelines/dataset.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/listing.h"
#include "nxr/pipelines/tiles.h"
#endif
#include "nxr/io/surface.h"
#include "cli.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/compact.h"
#include "nxr/io/task/dataset.h"
#include "nxr/io/task/task.h"
#include "nxr/io/writer.h"

#ifndef NXR_NEURO_VERSION
#define NXR_NEURO_VERSION "0.0.0"
#endif

using namespace nxr::io;
using namespace cli;
namespace fs = std::filesystem;

int read_command(int argc, char** argv);
int convert_main(const std::vector<std::string>& argv);
#ifdef NXR_NEURO_WITH_PIPELINES
int eigen_main(const std::vector<std::string>& argv);   // pipelines/tools/eigen_command.cpp
int phasegradient_main(const std::vector<std::string>& argv); // cli/meeg_command.cpp
int cyclepattern_main(const std::vector<std::string>& argv);  // cli/meeg_command.cpp
int measure_main(const std::vector<std::string>& argv);        // cli/measure_command.cpp
int surrogate_lagfree_spatial_events_compare_main(const std::vector<std::string>& argv);   // cli/measure_command.cpp
int surrogate_lagfree_vortex_compare_main(const std::vector<std::string>& argv);           // cli/measure_command.cpp
int pyramid_main(const std::vector<std::string>& argv);          // cli/measure_command.cpp     // cli/measure_command.cpp
int run_main(const std::vector<std::string>& argv);            // cli/run_command.cpp
int opticalflow_main(const std::vector<std::string>& argv);   // cli/meeg_command.cpp
int tiles_main(const std::vector<std::string>& argv);         // cli/meeg_command.cpp
#endif

namespace {

DatasetWriter dataset_of(const Args& a, bool lock = true) {
  OpenOptions o;
  o.lock = lock;
  return open_dataset(a, o);
}

// ── nxr-neuro dataset ──
// nxr-neuro constantq compute --dataset D/N --subject S (--recording R | --line NAME | --all) [--method hertz|cycles] [--voices N] [--cycles C] [--fmax HZ] [--fmin HZ]
//   THE CONSTANT-Q TILING of a time line (the Studio's time-tile policy; designed by compute, stored by io, joined in pipelines/constant_q.h): a ladder Selection whose members are the levels,
//   each a closed-form tessellation. The Studio's `backend/scripts/constant-q.mjs`: idempotent (a line that has one keeps it). --recording names a recording
//   (its time line), --line a time Line by name, --all every recording's time line.
#ifdef NXR_NEURO_WITH_PIPELINES
int constantq_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw Usage("nxr-neuro constantq: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro constantq compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--recording", "--line", "--method", "--voices", "--cycles", "--fmax", "--fmin"}, {"--all"}, {"--dataset", "--subject"});
  const int given = (a.maybe("--recording") ? 1 : 0) + (a.maybe("--line") ? 1 : 0) + (a.flag.count("--all") ? 1 : 0);
  if (given != 1) throw Usage(prog + ": error: name one of --recording R, --line NAME or --all");
  nxr::pipelines::ConstantQOptions p;
  if (const auto v = a.maybe("--method")) {
    if (*v != "hertz" && *v != "cycles") throw Usage(prog + ": error: --method must be hertz or cycles");
    p.method = *v;
  }
  if (const auto v = a.maybe("--voices")) p.voices = std::stoi(*v);
  if (const auto v = a.maybe("--cycles")) p.cycles = std::stod(*v);
  if (const auto v = a.maybe("--fmax")) p.f_max = std::stod(*v);
  if (const auto v = a.maybe("--fmin")) p.f_min = std::stod(*v);
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  // the time lines: a recording's (its product's time factor), a line by name, or every recording's
  std::vector<std::string> lines;
  const nxr::io::task::Subject ts = nxr::io::task::Subject::of(sub);
  if (const auto r = a.maybe("--recording")) {
    const auto line = ts.recording_time_line(*r);
    if (!line) throw Usage(prog + ": error: no recording named '" + *r + "' with a time line");
    lines.push_back(*line);
  } else if (const auto l = a.maybe("--line")) {
    const auto line = ts.line_named(*l);
    if (!line) throw Usage(prog + ": error: no time line named '" + *l + "'");
    lines.push_back(*line);
  } else {
    lines = ts.recording_time_lines();
  }
  Json made = Json::array();
  for (const std::string& line : lines) {
    nxr::pipelines::ConstantQWritten w;
    sub.compose([&] { w = nxr::pipelines::constant_q_write(sub, line, p, stamp()); });
    Json levels = Json::array();
    for (const auto& lv : w.levels) levels.push_back(Json{{"level", lv.level}, {"window_samples", lv.window_samples}, {"window_s", lv.window_s}, {"tiles", lv.tiles},
                                                               {"resolves_hz", Json::array({lv.resolves_lo, lv.resolves_hi})}});
    made.push_back(Json{{"ladder", w.name}, {"existed", w.existed}, {"levels", levels}});
  }
  emit(Json{{"stage", "result"}, {"ladders", made}});
  return 0;
}
#endif

int dataset_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro dataset: error: the following arguments are required: action");
  const std::string action = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro dataset " + action;
  if (action == "create") {
    const Args a = parse(prog, rest, {"--dataset", "--datastore", "--name", "--source-tool", "--source-protocol", "--source-path", "--templates", "--default", "--modes", "--atlas-depth", "--lc-levels", "--store-format"}, {}, {});
    // THE DEFAULT FORM: `--dataset PATH` makes a catalogue, the folder of a dataset that holds its subjects' packages (`create subject` adds one; `register package` lists
    // packages made elsewhere). `--datastore DIR --name N` (the old spelling's) still makes the one-database dataset DIR/N; `split dataset` turns it into a catalogue.
    if (a.maybe("--dataset") && a.maybe("--datastore")) throw Usage(prog + ": error: --dataset (a catalogue) and --datastore (a one-database dataset) are alternatives");
    if (!a.maybe("--dataset") && !a.maybe("--datastore")) throw Usage(prog + ": error: the following arguments are required: --dataset");
    if (a.maybe("--dataset")) {
      for (const char* x : {"--templates", "--default", "--modes", "--atlas-depth", "--lc-levels", "--store-format"})
        if (a.maybe(x)) throw Usage(prog + ": error: " + x + " belongs to the one-database form (--datastore); a catalogue's template is a template package (`import template`, `list templates`)");
      CreateCatalogueOptions co;
      co.name = a.maybe("--name").value_or(std::filesystem::path(a.get("--dataset")).filename().string());
      if (co.name.empty()) co.name = std::filesystem::path(a.get("--dataset")).parent_path().filename().string();
      co.source_tool = a.maybe("--source-tool");
      co.source_protocol = a.maybe("--source-protocol");
      co.source_path = a.maybe("--source-path");
      co.created_by = stamp();
      co.built_by = stamp();
      const Catalogue c = create_catalogue(a.get("--dataset"), co);
      emit(Json{{"stage", "result"}, {"dataset", text(c.dataset_row(), "name")}, {"id", text(c.dataset_row(), "id")}, {"folder", c.folder().string()}, {"form", "catalogue"}});
      return 0;
    }
    if (!a.maybe("--name")) throw Usage(prog + ": error: the following arguments are required: --name (with --datastore)");
    std::cerr << "note: --datastore makes the one-database form; `create dataset --dataset PATH` makes a catalogue of subject packages (the default form).\n";
    if (!a.maybe("--templates") && (a.maybe("--default") || a.maybe("--modes") || a.maybe("--atlas-depth") || a.maybe("--lc-levels")))
      throw Usage(prog + ": error: --default, --modes, --atlas-depth and --lc-levels need --templates (the default subject)");
    CreateOptions o;
    o.source_tool = a.maybe("--source-tool");
    o.source_protocol = a.maybe("--source-protocol");
    o.source_path = a.maybe("--source-path");
    o.created_by = stamp();
    if (const auto f = a.maybe("--store-format")) {
      if (*f != "51" && *f != "52") throw Usage(prog + ": error: --store-format is 51 or 52");
      o.store_format = std::stoi(*f);
    }
    if (const auto t = a.maybe("--templates")) {
      // THE DEFAULT SUBJECT, with the dataset: both or neither (the library function removes the dataset it made when a step fails)
#ifdef NXR_NEURO_WITH_PIPELINES
      nxr::pipelines::DatasetWithTemplateOptions w;
      w.dataset = o;
      w.subject.templates = *t;
      if (const auto v = a.maybe("--default")) w.subject.default_resolution = *v;
      if (const auto v = a.maybe("--atlas-depth")) w.atlas_depth = std::stoi(*v);
      if (const auto v = a.maybe("--lc-levels")) {             // the levels of the Levi-Civita connection, comma-separated (nxr's --lc-levels)
        w.lc_levels.clear();
        std::stringstream ss(*v);
        for (std::string x; std::getline(ss, x, ',');) {
          if (x.empty() || x.find_first_not_of("0123456789") != std::string::npos) throw Usage(prog + ": error: --lc-levels is a comma-separated list of levels (e.g. 4,6), not '" + *v + "'");
          w.lc_levels.push_back(std::stoi(x));
        }
      }
      if (const auto v = a.maybe("--modes")) {                 // the number of Hodge–Dirac modes to solve (as `eigen solve --modes`)
        try { w.modes = std::stoi(*v); } catch (...) { throw Usage(prog + ": error: --modes is a whole number (the number of modes to solve), not '" + *v + "'"); }
        if (*w.modes < 1) throw Usage(prog + ": error: --modes is a whole number of at least 1");
      }
      Json out = nxr::pipelines::create_dataset_with_template(a.get("--datastore"), a.get("--name"), w, stamp(), [](const Json& j) { emit(j); });
      out["stage"] = "result";
      emit(out);
#else
      throw Usage(prog + ": error: --templates writes the default subject with its frames and group tree, and needs the pipelines (-DNXR_NEURO_BUILD_PIPELINES=ON)");
#endif
    } else {
      DatasetWriter ds = create_dataset(a.get("--datastore"), a.get("--name"), o);
      emit(Json{{"stage", "result"}, {"dataset", text(ds.row(), "name")}, {"id", ds.id()}, {"database", (ds.folder() / kDatasetFile).string()}});
      ds.close();
    }
  } else if (action == "list") {
    const Args a = parse(prog, rest, {"--datastore"}, {}, {"--datastore"});
    const Json cat = list_datasets(a.get("--datastore"));
    emit(Json{{"stage", "result"}, {"datastore", cat.contains("id") ? cat["id"] : Json()}, {"datasets", cat.contains("datasets") ? cat["datasets"] : Json::array()}});
  } else if (action == "show") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    if (is_catalogue_dataset(a.get("--dataset"))) {   // a catalogue: its subjects by kind, from the entries (no package is opened)
      const Catalogue cat = Catalogue::open(a.get("--dataset"));
      Json kinds = Json::object();
      std::int64_t total = 0;
      for (const auto& e : cat.entries()) if (e.status != "replaced") { kinds[e.kind] = (kinds.contains(e.kind) ? kinds[e.kind].get<std::int64_t>() : 0) + 1; ++total; }
      const Row d = cat.dataset_row();
      auto opt = [&](const char* c) { return is_null(d, c) ? Json() : Json(text(d, c)); };
      emit(Json{{"stage", "result"}, {"dataset", text(d, "name")}, {"id", text(d, "id")}, {"path", cat.folder().string()}, {"form", "catalogue"}, {"subjects", total}, {"kinds", kinds},
                {"source_tool", opt("source_tool")}, {"source_protocol", opt("source_protocol")}, {"created_utc", opt("created_utc")}});
      return 0;
    }
    DatasetWriter ds = dataset_of(a, false);
    Json kinds = Json::object();
    std::int64_t total = 0;
    for (const auto& k : nxr::io::task::Dataset::of(ds).subject_kinds()) {
      kinds[k.kind] = k.n;
      total += k.n;
    }
    auto opt = [&](const char* c) { return is_null(ds.row(), c) ? Json() : Json(text(ds.row(), c)); };
    emit(Json{{"stage", "result"}, {"dataset", text(ds.row(), "name")}, {"id", ds.id()}, {"path", text(ds.row(), "path")}, {"schema_version", schema_version()},
              {"subjects", total}, {"kinds", kinds}, {"source_tool", opt("source_tool")}, {"source_protocol", opt("source_protocol")}, {"created_utc", opt("created_utc")}});
  } else if (action == "delete") {
    const Args a = parse(prog, rest, {"--datastore", "--name"}, {}, {"--datastore", "--name"});
    delete_dataset(a.get("--datastore"), a.get("--name"));
    emit(Json{{"stage", "result"}, {"deleted", a.get("--name")}});
  } else if (action == "recover") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    OpenOptions o;
    o.recover = false;
    DatasetWriter ds = open_one_database(a, "recover dataset", o);
    const nxr::io::task::Dataset td = nxr::io::task::Dataset::of(ds);
    const nxr::io::task::Healed healed = td.recover();
    td.drain();
    write_catalog(ds.root());
    emit(Json{{"stage", "result"}, {"rolled_back", healed.rolled_back}, {"finished", healed.finished}});
    ds.close();
  } else if (action == "compact") {
    // nxr-neuro dataset compact --dataset D/N [--subject S] [--dry-run] [--no-shard] [--no-small-files]
    //   FORMAT 50 -> 51 -> 52, array by array (io/include/nxr/io/compact.h): the new layout beside the old, every value compared byte for byte, then swapped; resumable and
    //   idempotent. The writer lock is taken (a dataset another writer holds is refused); nothing is recovered. --dry-run reports the files and folders before and after.
    const Args a = parse(prog, rest, {"--dataset", "--subject"}, {"--dry-run", "--no-shard", "--no-small-files"}, {"--dataset"});
    OpenOptions o;
    o.recover = false;
    DatasetWriter ds = open_one_database(a, "compact dataset", o);
    CompactOptions co;
    co.subject = a.maybe("--subject");
    co.dry_run = a.flag.count("--dry-run") > 0;
    co.shard = !a.flag.count("--no-shard");
    co.small_files = !a.flag.count("--no-small-files");
    emit(compact_dataset(ds, co));
    ds.close();
  } else {
    throw Usage("nxr-neuro dataset: error: argument action: invalid choice: '" + action + "' (choose from 'create', 'list', 'show', 'delete', 'recover', 'compact')");
  }
  return 0;
}

// ── nxr-neuro subject ──
int subject_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro subject: error: the following arguments are required: action");
  const std::string action = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro subject " + action;
  if (action == "list" || action == "show") {
    const bool show = action == "show";
    const Args a = show ? parse(prog, rest, {"--dataset", "--subject"}, {}, {"--dataset", "--subject"}) : parse(prog, rest, {"--dataset", "--subjects", "--where"}, {}, {"--dataset"});
    if (!show && std::filesystem::exists(std::filesystem::path(a.get("--dataset")) / kCatalogueFile)) {   // a catalogue of subject packages: by name, or by a filter over the participants
      const Catalogue cat = Catalogue::open(a.get("--dataset"));
      Json rows = Json::array();
      if (a.maybe("--subjects") || a.maybe("--where")) {
        SubjectSelection sel;
        if (const auto names = a.maybe("--subjects")) { std::stringstream ss(*names); for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) sel.subjects.push_back(x); }
        sel.where = a.maybe("--where");
        const SelectedSubjects chosen = select_subjects(a.get("--dataset"), sel);
        for (const auto& m : chosen.members) rows.push_back(m.to_json());
        emit(Json{{"stage", "result"}, {"dataset", text(cat.dataset_row(), "name")}, {"subjects", rows}, {"selection", chosen.provenance()}});
      } else {
        for (const auto& e : cat.entries()) rows.push_back(e.to_json());
        emit(Json{{"stage", "result"}, {"dataset", text(cat.dataset_row(), "name")}, {"subjects", rows}});
      }
      return 0;
    }
    if (!show && (a.maybe("--subjects") || a.maybe("--where"))) throw Usage(prog + ": error: --subjects and --where need a catalogue dataset (made by `create dataset --dataset PATH`); " + a.get("--dataset") + " is a one-database dataset");
    DatasetWriter ds = dataset_of(a, false);
    if (!show) {
      Json rows = Json::array();
      for (const auto& r : nxr::io::task::Dataset::of(ds).subjects())
        rows.push_back(Json{{"name", r.name}, {"kind", r.kind}, {"status", r.status}, {"path", r.path}});
      emit(Json{{"stage", "result"}, {"dataset", text(ds.row(), "name")}, {"subjects", rows}});
    } else {
      SubjectWriter s = SubjectWriter::open(ds, a.get("--subject"));
      const nxr::io::task::NodeCounts n = nxr::io::task::Subject::of(s).node_counts();
      emit(Json{{"stage", "result"}, {"subject", s.name()}, {"kind", text(s.row(), "kind")}, {"path", text(s.row(), "path")}, {"manifolds", n.manifolds},
                {"fields", n.fields}, {"operators", n.operators}, {"selections", n.selections}});
    }
    return 0;
  }
  if (action != "create" && action != "delete") throw Usage("nxr-neuro subject: error: argument action: invalid choice: '" + action + "' (choose from 'create', 'list', 'show', 'delete')");
  if (action == "create") {
    const Args a = parse(prog, rest, {"--dataset", "--name", "--kind"}, {}, {"--dataset", "--name"});
    const std::string kind = a.maybe("--kind").value_or("subject");
    if (kind != "subject" && kind != "template" && kind != "group") throw Usage(prog + ": error: argument --kind: invalid choice: '" + kind + "' (choose from 'subject', 'template', 'group')");
    if (is_catalogue_dataset(a.get("--dataset"))) {   // a catalogue: a new package in the dataset's folder, registered
      const Catalogue cat = Catalogue::open(a.get("--dataset"));
      CreatePackageOptions po = cat.package_options();
      po.subject.kind = kind;
      po.subject.created_by = stamp();
      po.built_by = stamp();
      const std::filesystem::path pkg = cat.folder() / (a.get("--name") + kPackageSuffix);
      DatasetWriter w = create_package(pkg, a.get("--name"), po);
      const std::string sid = nxr::io::task::Dataset::of(w).package_subject_id();
      w.close();
      RegisterOptions ro;
      ro.registered_by = stamp();
      const auto res = register_packages(a.get("--dataset"), {pkg}, ro);
      if (res.empty() || res[0].outcome == "refused") throw Error(code::invalid_argument, a.get("--name") + ": " + (res.empty() ? std::string("not registered") : res[0].reason));
      emit(Json{{"stage", "result"}, {"subject", a.get("--name")}, {"kind", kind}, {"id", sid}, {"path", pkg.filename().string()}});
      return 0;
    }
    DatasetWriter ds = dataset_of(a);
    SubjectOptions o;
    o.kind = kind;
    o.created_by = stamp();
    SubjectWriter sub = SubjectWriter::create(ds, a.get("--name"), o);
    sub.compose([] {});
    emit(Json{{"stage", "result"}, {"subject", a.get("--name")}, {"kind", kind}, {"id", sub.id()}, {"path", text(sub.row(), "path")}});
    return 0;
  }
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--node"}, {}, {"--dataset", "--subject"});
  DatasetWriter ds = open_one_database(a, "delete subject");
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  if (!a.maybe("--node")) {
    nxr::io::task::Dataset::of(ds).remove_subject(sub.id());
    write_catalog(ds.root());
    emit(Json{{"stage", "result"}, {"removed", a.get("--subject")}});
  } else {
    const std::string node = *a.maybe("--node");
    using nxr::io::task::Cls;
    const nxr::io::task::Dataset td = nxr::io::task::Dataset::of(ds);
    for (const Cls c : {Cls::manifold, Cls::field, Cls::operator_, Cls::selection}) {
      const char* t = nxr::io::task::to_string(c);
      if (const auto r = sub.find(t, node)) {
        td.remove(c, text(*r, "id"));
        emit(Json{{"stage", "result"}, {"removed", node}, {"table", t}});
        return 0;
      }
    }
    // no store path of that name: the node's NAME (a source estimate or a joint Selection has no path); unique within the subject, else refused with the matches
    std::vector<std::pair<Cls, std::string>> hits;   // class, id
    for (const Cls c : {Cls::manifold, Cls::field, Cls::operator_, Cls::selection, Cls::statistic})
      if (const auto r = sub.by_name(nxr::io::task::to_string(c), node)) hits.push_back({c, text(*r, "id")});
    if (hits.size() == 1) {
      td.remove(hits[0].first, hits[0].second);
      emit(Json{{"stage", "result"}, {"removed", node}, {"table", nxr::io::task::to_string(hits[0].first)}, {"by", "name"}});
      return 0;
    }
    if (hits.size() > 1) {
      std::string list;
      for (const auto& h : hits) list += (list.empty() ? "" : ", ") + std::string(nxr::io::task::to_string(h.first)) + " " + h.second;
      throw Error(code::invalid_argument, node + ": several nodes of " + a.get("--subject") + " have this name (" + list + "); name a store path with --node instead");
    }
    throw Error(code::no_such_row, node + ": no node of " + a.get("--subject") + " by store path or by name");
  }
  return 0;
}

// ── nxr-neuro surface ──
int surface_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "add") throw Usage("nxr-neuro surface: error: the following arguments are required: action (add)");
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const Args a = parse("nxr-neuro surface add", rest, {"--dataset", "--subject", "--name", "--scale"}, {"--primary", "--no-primary"}, {"--dataset", "--subject"}, 1);
  const double scale = a.maybe("--scale") ? std::stod(*a.maybe("--scale")) : 1.0;
  Mesh mesh = read_mesh(a.pos[0]);
  const std::string name = sanitize_node_name(a.maybe("--name").value_or(mesh_name(a.pos[0])));
  DatasetWriter ds = dataset_of(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  for (auto& x : mesh.vertices) x *= scale;
  SurfaceOptions so;
  so.name = name;
  so.source = fs::path(a.pos[0]).filename().string();
  so.primary = !a.flag.count("--no-primary");
  SurfaceInfo info;
  sub.compose([&] { info = add_surface(sub, so, mesh.vertices, mesh.faces); });
  emit(Json{{"stage", "result"}, {"name", info.name}, {"folder", info.folder}, {"n_vertices", info.n_vertices}, {"n_faces", info.n_faces}, {"atlases", info.atlases},
            {"registration", info.registration}, {"hemispheres", info.hemispheres}, {"flipped", info.flipped}, {"mesh_health", info.mesh_health}});
  return 0;
}



}  // namespace

// THE CHUNK LOOP's process defaults (io/include/nxr/io/chunk_loop.h), set once here so no pipeline passes them: a progress line after each chunk;
// Ctrl-C cancels the running job cleanly (nothing of it is stored; a second Ctrl-C kills the process); one job at a time per dataset folder's parent (a queue file in it).
namespace {
std::atomic<bool> cancel_requested{false};
extern "C" void on_interrupt(int) {
  cancel_requested = true;
  std::signal(SIGINT, SIG_DFL);
}
void chunk_loop_defaults(const std::vector<std::string>& args) {
  nxr::io::ChunkLoopOptions o;
  o.progress = [](const nxr::io::ChunkJob& j) {
    cli::emit(nxr::io::Json{{"stage", "chunk"}, {"done", j.done}, {"count", j.count}, {"chunk", j.visited.empty() ? nxr::io::Json() : nxr::io::Json(j.visited.back())},
                            {"seconds", std::round(j.seconds * 1000) / 1000}, {"state", j.state}});
  };
  o.cancelled = [] { return cancel_requested.load(); };
  for (std::size_t i = 0; i + 1 < args.size(); ++i)
    if (args[i] == "--dataset") {                                   // the queue: the folder that holds the dataset's folder
      const std::filesystem::path ds = std::filesystem::absolute(args[i + 1]).lexically_normal();
      const std::filesystem::path store = ds.filename().empty() ? ds.parent_path().parent_path() : ds.parent_path();
      if (std::filesystem::is_directory(store)) o.queue = store / ".nxr-chunk-queue.lock";
    }
  nxr::io::set_chunk_loop_defaults(std::move(o));
  std::signal(SIGINT, on_interrupt);
}
}  // namespace

// the old command words dispatched to their mains (the hidden aliases of the old spellings, and what every new command rewrites itself to)
// ── nxr-neuro list recordings|kernels|results (the old words: `__list`, which has no old spelling) ──
int list_nodes_main(const std::vector<std::string>& argv) {
#ifdef NXR_NEURO_WITH_PIPELINES
  if (argv.empty()) throw Usage("nxr-neuro list: error: the following arguments are required: what (recordings, kernels, results)");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const Args a = parse("nxr-neuro list " + what, rest, {"--dataset", "--subject"}, {}, {"--dataset", "--subject"});
  DatasetWriter ds = dataset_of(a, false);
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"));
  emit(Json{{"stage", "result"}, {"subject", sub.name()}, {what, nxr::pipelines::list_subject_nodes(ds, sub, what)}});
  return 0;
#else
  (void)argv;
  throw Error(code::unsupported, "nxr-neuro list needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
}

int package_main(const std::vector<std::string>& argv);   // cli/package_command.cpp
int import_main(const std::vector<std::string>& argv);    // cli/import_command.cpp
int export_main(const std::vector<std::string>& argv);    // cli/export_command.cpp
int run_legacy(const std::string& cmd, const std::vector<std::string>& rest) {
  if (cmd == "__list") return list_nodes_main(rest);
  if (cmd == "read") {
    std::vector<std::string> store = {"read"};
    store.insert(store.end(), rest.begin(), rest.end());
    std::vector<char*> av;
    for (auto& t : store) av.push_back(t.data());
    return read_command(static_cast<int>(av.size()), av.data());
  }
    if (cmd == "dataset") return dataset_main(rest);
    if (cmd == "package") return package_main(rest);
    if (cmd == "__import") return import_main(rest);
    if (cmd == "__export") return export_main(rest);
    if (cmd == "constantq") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return constantq_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro constantq needs the pipelines (nxr_compute + nxr_io): build with -DNXR_NEURO_BUILD_PIPELINES=ON (README: Pipelines)");
#endif
    }
    if (cmd == "subject") return subject_main(rest);
    if (cmd == "surface") return surface_main(rest);
    if (cmd == "convert") return convert_main(rest);
    if (cmd == "eigen") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return eigen_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro eigen needs the pipelines (nxr_compute + nxr_io): build with -DNXR_NEURO_BUILD_PIPELINES=ON (README: Pipelines)");
#endif
    }
    if (cmd == "run") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return run_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro run needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "pyramid") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return pyramid_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro pyramid needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "__surrogate_lagfree_spatial_events_compare") {               // `run surrogate_lagfree_spatial_events_compare` (the command table's internal words)
#ifdef NXR_NEURO_WITH_PIPELINES
      return surrogate_lagfree_spatial_events_compare_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro run surrogate_lagfree_spatial_events_compare needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "__surrogate_lagfree_vortex_compare") {                       // `run surrogate_lagfree_vortex_compare`
#ifdef NXR_NEURO_WITH_PIPELINES
      return surrogate_lagfree_vortex_compare_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro run surrogate_lagfree_vortex_compare needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "measure") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return measure_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro measure needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "phasegradient" || cmd == "cyclepattern" || cmd == "opticalflow" || cmd == "tiles") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return cmd == "phasegradient" ? phasegradient_main(rest) : cmd == "cyclepattern" ? cyclepattern_main(rest) : cmd == "opticalflow" ? opticalflow_main(rest) : tiles_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro " + cmd + " needs the pipelines (nxr_compute + nxr_io + nxr_meeg): build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
  throw Usage("nxr-neuro: error: unknown command '" + cmd + "'");
}


void install_chunk_loop_defaults(const std::vector<std::string>& args) { chunk_loop_defaults(args); }

int schema_version_for_version_flag() { return nxr::io::schema_version(); }
