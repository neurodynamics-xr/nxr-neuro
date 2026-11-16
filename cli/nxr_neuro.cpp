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

#include "nxr/io/constant_q.h"
#include "nxr/io/default_subject.h"
#ifdef NXR_NEURO_WITH_PIPELINES
#include "nxr/pipelines/dataset.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/listing.h"
#include "nxr/pipelines/tiles.h"
#endif
#include "nxr/io/surface.h"
#include "cli.h"
#include "nxr/io/compact.h"
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
int noisecov_main(const std::vector<std::string>& argv);   // cli/meeg_command.cpp
int psd_main(const std::vector<std::string>& argv);        // cli/meeg_command.cpp
int sourcepsd_main(const std::vector<std::string>& argv);  // cli/meeg_command.cpp
int sourcetensor_main(const std::vector<std::string>& argv);  // cli/meeg_command.cpp
int crossspectrum_main(const std::vector<std::string>& argv);           // cli/meeg_command.cpp
int phasegradient_main(const std::vector<std::string>& argv); // cli/meeg_command.cpp
int cyclepattern_main(const std::vector<std::string>& argv);  // cli/meeg_command.cpp
int quadrature_main(const std::vector<std::string>& argv);  // cli/meeg_command.cpp
int hodgepower_main(const std::vector<std::string>& argv);  // cli/meeg_command.cpp
int speedspectrum_main(const std::vector<std::string>& argv);  // cli/speedspectrum_command.cpp
int measure_main(const std::vector<std::string>& argv);        // cli/measure_command.cpp
int statistic_main(const std::vector<std::string>& argv);      // cli/measure_command.cpp
int patches_main(const std::vector<std::string>& argv);        // cli/measure_command.cpp
int cellquadrature_main(const std::vector<std::string>& argv); // cli/measure_command.cpp
int wishart_main(const std::vector<std::string>& argv);          // cli/measure_command.cpp
int pointspread_main(const std::vector<std::string>& argv);     // cli/measure_command.cpp
int spatialnull_main(const std::vector<std::string>& argv);     // cli/measure_command.cpp
int null_main(const std::vector<std::string>& argv);           // cli/measure_command.cpp
int repeat_main(const std::vector<std::string>& argv);         // cli/measure_command.cpp
int scaleselection_main(const std::vector<std::string>& argv); // cli/measure_command.cpp
int joint_main(const std::vector<std::string>& argv);          // cli/measure_command.cpp
int frametiling_main(const std::vector<std::string>& argv);
int pyramid_main(const std::vector<std::string>& argv);          // cli/measure_command.cpp     // cli/measure_command.cpp
int run_main(const std::vector<std::string>& argv);            // cli/run_command.cpp
int opticalflow_main(const std::vector<std::string>& argv);   // cli/meeg_command.cpp
int frames_main(const std::vector<std::string>& argv);        // cli/meeg_command.cpp
int tiles_main(const std::vector<std::string>& argv);         // cli/meeg_command.cpp
int filter_main(const std::string& verb, const std::vector<std::string>& argv);   // cli/meeg_command.cpp
int inverse_main(const std::vector<std::string>& argv);
int headmodel_main(const std::vector<std::string>& argv);
#endif

namespace {

DatasetWriter dataset_of(const Args& a, bool lock = true) {
  OpenOptions o;
  o.lock = lock;
  return open_writer(a.get("--dataset"), o);
}

// ── nxr-neuro dataset ──
// nxr-neuro constantq compute --dataset D/N --subject S (--recording R | --line NAME | --all) [--method hertz|cycles] [--voices N] [--cycles C] [--fmax HZ] [--fmin HZ]
//   THE CONSTANT-Q TILING of a time line (the Studio's time-tile policy; io/include/nxr/io/constant_q.h): a ladder Selection whose members are the levels,
//   each a closed-form tessellation. The Studio's `backend/scripts/constant-q.mjs`: idempotent (a line that has one keeps it). --recording names a recording
//   (its time line), --line a time Line by name, --all every recording's time line.
int constantq_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw Usage("nxr-neuro constantq: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro constantq compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--recording", "--line", "--method", "--voices", "--cycles", "--fmax", "--fmin"}, {"--all"}, {"--dataset", "--subject"});
  const int given = (a.maybe("--recording") ? 1 : 0) + (a.maybe("--line") ? 1 : 0) + (a.flag.count("--all") ? 1 : 0);
  if (given != 1) throw Usage(prog + ": error: name one of --recording R, --line NAME or --all");
  nxr::io::constantq::Params p;
  if (const auto v = a.maybe("--method")) {
    if (*v == "hertz") p.method = nxr::io::constantq::Method::Hertz;
    else if (*v == "cycles") p.method = nxr::io::constantq::Method::Cycles;
    else throw Usage(prog + ": error: --method must be hertz or cycles");
  }
  if (const auto v = a.maybe("--voices")) p.voices = std::stoi(*v);
  if (const auto v = a.maybe("--cycles")) p.cycles = std::stod(*v);
  if (const auto v = a.maybe("--fmax")) p.fMax = std::stod(*v);
  if (const auto v = a.maybe("--fmin")) p.fMin = std::stod(*v);
  DatasetWriter ds = open_writer(a.get("--dataset"));
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  // the time lines: a recording's (its product's time factor), a line by name, or every recording's
  std::vector<std::string> lines;
  const std::string q_time = "SELECT gd.manifold_id AS line FROM field f JOIN manifold m ON m.id = f.manifold_id JOIN geometry_dimension gd ON gd.geometry_id = m.geometry_id "
                             "JOIN manifold t ON t.id = gd.manifold_id WHERE f.subject_id = ? AND f.kind = 'recording' AND f.status = 'complete' AND t.type = 'line'";
  if (const auto r = a.maybe("--recording")) {
    const auto row = sub.db().one(q_time + " AND f.name = ?", {V(sub.id()), V(*r)});
    if (!row) throw Usage(prog + ": error: no recording named '" + *r + "' with a time line");
    lines.push_back(text(*row, "line"));
  } else if (const auto l = a.maybe("--line")) {
    const auto row = sub.db().one("SELECT id FROM manifold WHERE subject_id = ? AND name = ? AND type = 'line'", {V(sub.id()), V(*l)});
    if (!row) throw Usage(prog + ": error: no time line named '" + *l + "'");
    lines.push_back(text(*row, "id"));
  } else {
    for (const Row& row : sub.db().all(q_time, {V(sub.id())})) if (std::find(lines.begin(), lines.end(), text(row, "line")) == lines.end()) lines.push_back(text(row, "line"));
  }
  Json made = Json::array();
  for (const std::string& line : lines) {
    nxr::io::constantq::Written w;
    sub.compose([&] { w = nxr::io::constantq::write(sub, line, p, stamp()); });
    Json levels = Json::array();
    for (const auto& lv : w.tiling.levels) levels.push_back(Json{{"level", lv.level}, {"window_samples", lv.window_samples}, {"window_s", lv.window_s}, {"tiles", lv.count},
                                                               {"resolves_hz", Json::array({lv.resolves_lo, lv.resolves_hi})}});
    made.push_back(Json{{"ladder", w.name}, {"existed", w.existed}, {"levels", levels}});
  }
  emit(Json{{"stage", "result"}, {"ladders", made}});
  return 0;
}

int dataset_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro dataset: error: the following arguments are required: action");
  const std::string action = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro dataset " + action;
  if (action == "create") {
    const Args a = parse(prog, rest, {"--datastore", "--name", "--source-tool", "--source-protocol", "--source-path", "--templates", "--default", "--modes", "--atlas-depth", "--lc-levels", "--store-format"}, {},
                         {"--datastore", "--name"});
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
    DatasetWriter ds = dataset_of(a, false);
    Json kinds = Json::object();
    std::int64_t total = 0;
    for (const auto& r : ds.db().all("SELECT kind, count(*) AS n FROM subject WHERE dataset_id = ? AND status = 'complete' GROUP BY kind ORDER BY kind", {ds.id()})) {
      kinds[text(r, "kind")] = integer(r, "n");
      total += integer(r, "n");
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
    DatasetWriter ds = open_writer(a.get("--dataset"), o);
    const Recovered healed = recover(ds.db(), ds.store());
    drain(ds.db(), ds.store());
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
    DatasetWriter ds = open_writer(a.get("--dataset"), o);
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
    const Args a = show ? parse(prog, rest, {"--dataset", "--subject"}, {}, {"--dataset", "--subject"}) : parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    DatasetWriter ds = dataset_of(a, false);
    if (!show) {
      Json rows = Json::array();
      for (const auto& r : ds.db().all("SELECT name, kind, status, path FROM subject WHERE dataset_id = ? ORDER BY name", {ds.id()}))
        rows.push_back(Json{{"name", text(r, "name")}, {"kind", text(r, "kind")}, {"status", text(r, "status")}, {"path", text(r, "path")}});
      emit(Json{{"stage", "result"}, {"dataset", text(ds.row(), "name")}, {"subjects", rows}});
    } else {
      SubjectWriter s = SubjectWriter::open(ds, a.get("--subject"));
      auto count = [&](const char* t) { return integer(*ds.db().one(std::string("SELECT count(*) AS n FROM ") + t + " WHERE subject_id = ?", {s.id()}), "n"); };
      emit(Json{{"stage", "result"}, {"subject", s.name()}, {"kind", text(s.row(), "kind")}, {"path", text(s.row(), "path")}, {"manifolds", count("manifold")},
                {"fields", count("field")}, {"operators", count("operator")}, {"selections", count("selection")}});
    }
    return 0;
  }
  if (action != "create" && action != "delete") throw Usage("nxr-neuro subject: error: argument action: invalid choice: '" + action + "' (choose from 'create', 'list', 'show', 'delete')");
  if (action == "create") {
    const Args a = parse(prog, rest, {"--dataset", "--name", "--kind"}, {}, {"--dataset", "--name"});
    const std::string kind = a.maybe("--kind").value_or("subject");
    if (kind != "subject" && kind != "template" && kind != "group") throw Usage(prog + ": error: argument --kind: invalid choice: '" + kind + "' (choose from 'subject', 'template', 'group')");
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
  DatasetWriter ds = dataset_of(a);
  OpenSubjectOptions oo;
  oo.created_by = stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  if (!a.maybe("--node")) {
    remove_node(ds.db(), ds.store(), "subject", sub.id());
    write_catalog(ds.root());
    emit(Json{{"stage", "result"}, {"removed", a.get("--subject")}});
  } else {
    const std::string node = *a.maybe("--node");
    for (const char* t : {"manifold", "field", "operator", "selection"}) {
      if (const auto r = sub.find(t, node)) {
        remove_node(ds.db(), ds.store(), t, text(*r, "id"));
        emit(Json{{"stage", "result"}, {"removed", node}, {"table", t}});
        return 0;
      }
    }
    // no store path of that name: the node's NAME (a source estimate or a joint Selection has no path); unique within the subject, else refused with the matches
    std::vector<std::pair<std::string, std::string>> hits;   // table, id
    for (const char* t : {"manifold", "field", "operator", "selection", "statistic"})
      if (const auto r = sub.by_name(t, node)) hits.push_back({t, text(*r, "id")});
    if (hits.size() == 1) {
      remove_node(ds.db(), ds.store(), hits[0].first, hits[0].second);
      emit(Json{{"stage", "result"}, {"removed", node}, {"table", hits[0].first}, {"by", "name"}});
      return 0;
    }
    if (hits.size() > 1) {
      std::string list;
      for (const auto& h : hits) list += (list.empty() ? "" : ", ") + h.first + " " + h.second;
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
// Ctrl-C cancels the running job cleanly (nothing of it is stored; a second Ctrl-C kills the process); one job at a time per datastore (a queue file in it).
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
    if (args[i] == "--dataset") {                                   // the queue: the datastore that holds the dataset
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
    if (cmd == "constantq") return constantq_main(rest);
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
    if (cmd == "lowpass" || cmd == "highpass" || cmd == "bandpass" || cmd == "bandstop" || cmd == "notch") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return filter_main(cmd, rest);
#else
      throw Error(code::unsupported, "nxr-neuro " + cmd + " needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "run") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return run_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro run needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "statistic") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return statistic_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro statistic needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "pyramid") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return pyramid_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro pyramid needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "frametiling") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return frametiling_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro frametiling needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "joint") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return joint_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro joint needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "wishart") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return wishart_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro wishart needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "pointspread") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return pointspread_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro pointspread needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "spatialnull") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return spatialnull_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro spatialnull needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "null") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return null_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro null needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "repeat") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return repeat_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro repeat needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "cellquadrature") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return cellquadrature_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro cellquadrature needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "scaleselection") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return scaleselection_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro scaleselection needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "patches") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return patches_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro patches needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "measure") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return measure_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro measure needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "speedspectrum") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return speedspectrum_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro speedspectrum needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "noisecov" || cmd == "inverse" || cmd == "headmodel" || cmd == "psd" || cmd == "sourcepsd" || cmd == "sourcetensor" || cmd == "crossspectrum" || cmd == "phasegradient" || cmd == "cyclepattern" || cmd == "quadrature" || cmd == "hodgepower" || cmd == "opticalflow" || cmd == "frames" || cmd == "tiles") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return cmd == "noisecov" ? noisecov_main(rest) : cmd == "psd" ? psd_main(rest) : cmd == "sourcepsd" ? sourcepsd_main(rest) : cmd == "sourcetensor" ? sourcetensor_main(rest) : cmd == "crossspectrum" ? crossspectrum_main(rest) : cmd == "phasegradient" ? phasegradient_main(rest) : cmd == "cyclepattern" ? cyclepattern_main(rest) : cmd == "quadrature" ? quadrature_main(rest) : cmd == "hodgepower" ? hodgepower_main(rest) : cmd == "opticalflow" ? opticalflow_main(rest) : cmd == "frames" ? frames_main(rest) : cmd == "tiles" ? tiles_main(rest) : cmd == "inverse" ? inverse_main(rest) : headmodel_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro " + cmd + " needs the pipelines (nxr_compute + nxr_io + nxr_meeg): build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
  throw Usage("nxr-neuro: error: unknown command '" + cmd + "'");
}


void install_chunk_loop_defaults(const std::vector<std::string>& args) { chunk_loop_defaults(args); }

int schema_version_for_version_flag() { return nxr::io::schema_version(); }
