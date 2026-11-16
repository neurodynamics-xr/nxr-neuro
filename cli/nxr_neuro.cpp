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

const char* kUsage =
    "usage: nxr-neuro <command> ...\n"
    "  nxr-neuro constantq compute --dataset D/N --subject S (--recording R | --line NAME | --all) [--method hertz|cycles] [--voices N] [--cycles C] [--fmax HZ] [--fmin HZ]   the constant-Q time ladder of a time line (the Studio's time-tile policy)\n"
    "  nxr-neuro run DEFINITION --dataset D/N --subject S [--set NAME=VALUE ...] [--chunks K|A-B|all | --start S --stop S] [--block-samples N] [--block-frames N] --dry-run   a pipeline definition (nxr's YAML) planned on a subject: inputs found, arrows checked (run --list: the definitions and steps); --block-samples: the width the pointwise steps after the last filter run on (default 1024; 0 the whole chunk): memory only; --block-frames: the frames the steps after a frame tiling run on (default 256; 0 the whole chunk's frames): memory only\n"
    "  nxr-neuro dataset create|list|show|delete|recover|compact ...   the datasets of a datastore (--datastore D --name N; --dataset D/N); create --templates FS_SUBJECTS [--default fsaverage5] [--modes DIR] [--atlas-depth 8] [--lc-levels 4,6] writes the default subject (surfaces, parcellations, frames, group tree) with it\n"
    "  nxr-neuro subject create|list|show|delete ...           the subjects of a dataset (--dataset D/N --name S | --subject S [--node PATH])\n"
    "  nxr-neuro surface add --dataset D/N --subject S FILE    a mesh file (OBJ · PLY · STL) as a surface of a subject\n"
    "  nxr-neuro convert list|scan|subject|surface ...         a Brainstorm protocol: what it holds, and its anatomy into a dataset (nxr convert's verbs)\n"
    "  nxr-neuro eigen solve --dataset D/N --subject S [--modes K]   the Hodge-Dirac eigenmodes of a stored surface, cached \n"
    "  nxr-neuro headmodel compute --dataset D/N --subject S [--meg overlapping|single|none] [--no-eeg]   the sphere head model (gain), stored as an imported one\n"
    "  nxr-neuro sourcepsd compute --dataset D/N --subject S --recording R [--kernel NAME] [--bands alpha:8-12,...] [--normalisation relative]   the band power maps of sources (Brainstorm's PSD on sources, grouped in bands)\n"
    "  nxr-neuro crossspectrum compute --dataset D/N --subject S --recording R [--channels-of KERNEL | --types T,..] [--bands alpha:8-12,...] [the psd options | --estimator analytic [--level K | --window-seconds SEC]]   the sensor cross-spectral density per band (complex, channels x channels): every source block is K C K^T; analytic: one per window (a series); --resolve bins also stores each band bin by bin\n"
    "  nxr-neuro crossspectrum read --dataset D/N --subject S --name <stem>_<band>_bins (--pair A,B | --seed A) [--reading coherency|imag|phase|delay|psi] [--weights coherence|none]   readings derived from a band's cross-spectrum kept bin by bin: coherency, its imaginary part, the phase, the delay (slope of the phase over frequency, ms) or Nolte's phase-slope index; positive: the row channel leads\n"
    "  nxr-neuro phasegradient compute --dataset D/N --subject S --cross-spectrum NAME [--kernel NAME] [--signal normal|ambient] [--depth D] [--spatial-band modes|weyl:J|weyl:all]   from a stored band cross-spectrum: the phase gradient's sums per pyramid leaf (phase current in the canonical frames, power; gradient, speed, travelling index derived at read); at a spatial scale the per-vertex timing map and its inconsistency\n"
    "  nxr-neuro cyclepattern compute --dataset D/N --subject S --cross-spectrum NAME [--kernel NAME] [--package STEM] [--spatial-band modes|weyl:J|weyl:all] [--patterns Q] [--level K | --window-seconds SEC]   from a cross-spectrum per window (crossspectrum --estimator analytic): per spatial band the most consistent cycle patterns of the current (S, A, rho on the cortex) and how they recur across windows\n"
    "  nxr-neuro sourcetensor compute --dataset D/N --subject S --recording R [--kernel NAME] [--bands alpha:8-12,...] [the psd options]   the band cross-spectral tensor of the current (hermitian3), its tangential Stokes quaternion, normal power and share\n"
    "  nxr-neuro quadrature compute --dataset D/N --subject S --recording R [--kernel NAME] [--bands alpha:8-12,...] [--pairs K]   the quadrature spectrum of sources per band: quadratureFraction, the quadratureEnergy map, the top K quadraturePairs (p leads q) as source maps\n"
    "  nxr-neuro opticalflow compute --dataset D/N --subject S --recording R [--kernel NAME] [--package STEM] [--name STEM] [--start S] [--stop S] --family F --gain G --freq HZ [--centres HZ,..] [--cycles C] [--voices V] [--limits LO-HI] [--resolution HZ] [--oversample X] [--tolerance T] [--alpha A] [--depth D] [--frame-level L] [--start-s S] [--replace]   optical flow (Horn-Schunck) of the source amplitude of one filterbank member (wt; kernel applied to its complex coefficients), read in the Hodge-Dirac modes: exact (sources and sinks) and coexact (rotation) coefficients per frame, divergence, curl, speed and velocity as additive sums per leaf of the tile ladder's level D and per tower frame (measures flow_divergence, flow_curl, flow_speed, flow_velocity_*; `measure query` reads them, `_density` is the time mean); no per-vertex Fields\n"
    "  nxr-neuro frames compute --dataset D/N --subject S [--surface NAME] [--template SUBJECT] [--convention scs|ras] [--min-projection P] [--replace]   the canonical frames (north / west, from the registered sphere) and the anatomical frame (angle from north to anterior, second axis superior or medial, reliability mask) of a surface\n"
    "  nxr-neuro tiles compute --dataset D/N --subject S [--surface NAME] [--depth L] [--replace]   the dyadic tile ladder of a surface (the Studio's surfaceLadder, nxr geom.ladder): 2^l equal-area tiles per piece at level l, paired with scale by Weyl's law, stored as Selections; opticalflow / quadrature compute take --tiles [--tile-levels 1,2,3 | --tile-max-level N] [--no-levi-civita] to write their maps' tile summaries per level (vectors in frame components and Levi-Civita form)\n"
    "  nxr-neuro speedspectrum compute --dataset D/N --subject S --recording R --graph-family F --family F --gain G --sides analytic|two-sided [--fmin HZ --fmax HZ] [--speed-bins N]   the speed spectrum: mode energy per phase speed (2 pi f = c mu) from the joint eigenvalue x frequency bank, exact and coexact apart, with each tile's quadrature fraction\n"
    "  nxr-neuro hodgepower compute --dataset D/N --subject S --recording R [--kernel NAME] [--package STEM] [--modes N] [--bands alpha:8-12,...]   the band power of the Hodge-Dirac eigenmode coefficients of a free-orientation kernel\n"
    "  nxr-neuro measure compute --dataset D/N --subject S --labelling L --fields PATTERN --weight none|barycentric|voronoi [--measures mean,max,..] [--recipe NAME]   stored Fields measured over a partition's members (the Studio's measureOverPartition), with their recipe\n"
    "  nxr-neuro measure query --dataset D/N --subject S --selection JOINT --measure M [--field F] [--band I] [--spatial-band K] [--space-level L] [--time-level M] [--min-value X] [--max-value X] [--limit N] [--order asc|desc]   a measurement pyramid's array-backed measure read AT ANY SCALE: space rolled up by reshape to level L (-1 the whole cortex; default the stored depth), time on the tower to tiles of 2^M frames (default 0), one band; sum/max/min by the measure's merge, `<sum>_density` (power_sum_density, ...) from rolled-up sums over area x samples; rows {code, tile, band, value, weight, members}\n"
    "  nxr-neuro statistic compute --dataset D/N --subject S --over SELECTION --of FIELD|MAP --measure M --summaries median,mad,percentile:75,.. [--combine each|pool|mean] [--members-own]   a stored measurement summarised over its partition's members into a statistic (summaries across the members)\n"
    "  nxr-neuro joint compute --dataset D/N --subject S --recording R --space-level L --time-level M [--surface NAME] [--replace]   the JOINT partition of space x time: a closed-form Selection of the cells (a spatial tile of level L) x (a time tile of the constant-Q level M), code = (tile << B) | time tile, the Studio atlas's layout; `run cortical_cells` measures a source estimate over it (`run cortical_cell_pairs` over many pairs at once)\n"
    "  nxr-neuro frametiling compute --dataset D/N --subject S --recording R (--frame-level L | --centre HZ | --band LO,HI) [--frames-per-cycle 36] [--start-s SECONDS] [--replace]   the FRAME TILING on the TIME TOWER (D135): a closed-form spans Selection `<R> frames <s> s` on the recording's time line, the cycles of ONE tower level (86400 x 2^L s: -18 = 0.33 s, -25 = 2.57 ms; a level-m tile is code >> m); the level is named, or derived from the band as the coarsest whose frame is at or under 1/(centre x frames-per-cycle); --start-s anchors the day; the `frames` step samples a chunk's analytic field at each frame's start\n"
    "  nxr-neuro pyramid spatial-bands --dataset D/N --subject S [--surface NAME]   the surface's spatial bands (one per level of its tile ladder, the Weyl band, on the Eigenvalue line, 1/m^2): the within key of a measurement's spatial scale\n"
    "  nxr-neuro pyramid reduce --dataset D/N --subjects A,B,C [--kernels all|constrained|free|latest|METHOD|STAMP|NAME|GLOB,...] [--depth D]   the GROUP REDUCE: the NAMED subjects' group-tree pyramids (`pyramid build --tree group`) and group-tile map summaries summed cell for cell, time collapsed (frames and runs summed, maxima maxed), into the default subject's partition node, as nxr's atlas reduce writes them (kind `group map` / `source estimate`, pathless, function sum, field_contribution rows, per-subject means and densities); never all subjects by default; a member without the group tree, another leaf count, other bands or two kernels of one method for a recording are refused\n"
    "  nxr-neuro pyramid build --dataset D/N --subject S --recording R [--kernel K] [--tree subject|group] [--depth D] [--bands LO-HI,...] [--shape morlet|power_partition] [--halo-seconds 8] [--frame-level -18] [--start-s SECONDS] [--chunks k|a-b|all | --start S --stop S] [--keep-bad-segments] [--block-samples N] [--threads N] [--replace]   the MEASUREMENT PYRAMID of a recording's cortical currents: per band x leaf x frame, for a constrained-orientation kernel the band power (sum a|J|^2), envelope (sum a|J|) and envelope maximum, for a free-orientation kernel the orientation tensor in the canonical frames (t11 t22 tnn t12r t12i t1nr t1ni t2nr t2ni total; `frames compute` first); leaves = the surface ladder's level D (default: the finest every piece reaches; `--tree group`: the dataset's GROUP tree through the registered sphere, the same leaves in every subject, D default the dataset's atlas depth), bands default to octaves 1-64 Hz (`atlas bands`), frames = the tower tiling (-18 = 0.33 s); stored as nxr stores its atlas (source estimate Field without bytes, joint Selection (tile << B) | frame, array-backed measurement rows power_sum, envelope_sum, envelope_max); one chunk by default\n"
    "  nxr-neuro wishart compute --dataset D/N --subject S --recording R --windows NAME --bursts NAME --psd NAME --band LO,HI [--shape morlet|meyer] [--draws 1000] [--seed 1] [--replace]   the Wishart null of the second-moment measures: each burst's second moment (the stored analytic cross-spectrum windows summed over its tiles) against complex Wishart draws of its own zero-lag covariance (dof = duration x the band's effective bandwidth): the sensor-level lagged share, its p and BH q, and the run's, as statistics\n"
    "  nxr-neuro pointspread compute --dataset D/N --subject S --recording R [--forward NAME] [--kernel NAME] [--levels 2,3,..,8] [--reference-level 6] [--vertices 20 | --list V,V,..] [--fraction 0.5] [--spatial-shape mexhat] [--transform auto] [--name NAME] [--replace]   the instrument's blur: unit currents (x, y, z) at a sample of vertices through the stored head model and inverse kernel, read by the spatial events' chain (scale selection, the patch at a fraction of the peak): per source and orientation the peak's distance, the patch's diameter and area, the characteristic level and size, as a statistic of the recording\n"
    "  nxr-neuro spatialnull compute --dataset D/N --subject S --recording R [--events spatial_events] [--set NAME=VALUE ...] [--chunks all] [--batch 20] [--min-draws 20] [--max-draws 100] [--h 10] [--replace]   the lag-free null of the spatial events: draws of `spatial_events_null` in batches with early stopping (Besag-Clifford), per-vertex p and BH q of the home occupancy and the size distribution, and the null's mean occupancy, as statistics\n"
    "  nxr-neuro null compute --dataset D/N --subject S --recording-a A [--recording-b B] [--over TILES] [--stage NAME|INDEX] [--set NAME=VALUE ...] [--chunks all] [--batch 20] [--min-draws 20] [--max-draws 100] [--h 10] [--min-crossings 10] [--replace]   the lag-free null's tests of the vortex cores' crossings: draws of `vortex_null` in batches with early stopping (Besag-Clifford), per-tile p and BH q of the crossing consistency (and the run repeat with --recording-b), the data-to-null crossing ratio and the null's mean resultants, as statistics\n"
    "  nxr-neuro repeat compute --dataset D/N --subject S --recording-a A --recording-b B [--over TILES] [--stage NAME|INDEX] [--min-crossings 10] [--name NAME] [--replace]   the vortex cores' run-to-run repeat: two recordings' stored tile crossings (run vortex_cores on both): per tile the dot product of the two runs' mean resultant vectors, and the count-weighted mean cosine, as statistics (the p-value comes with the null)\n"
    "  nxr-neuro cellquadrature compute --dataset D/N --subject S --cross-spectrum NAME --space-level L [--kernel NAME] [--joint NAME] [--bursts NAME] [--weights area|none] [--tile-pairs on|off] [--transform auto|eigen|chebyshev|exact] [--modes N] [--order N] [--tolerance T] [--replace]   QUADRATURE per joint cell and per burst (travelling vs standing) from the STORED per-time-tile cross-spectrum of `crossspectrum compute --estimator analytic` (its tile level must equal the joint pair's time level): per spatial tile of level L, sum |T|^2 and sum |Im T|^2 of T = B C B^T (B the tile's rows of the kernel with the level's spatial band; a burst's C is its time tiles' sum), stored as ingredients, the quadrature fraction derived at read\n"
    "  nxr-neuro scaleselection compute --dataset D/N --subject S --recording R --space-levels L,L,.. --time-levels M,M,.. [--source cortical_event_pairs] [--response peak|mean] [--min-link 1] [--name NAME] [--replace]   each event's OWN size and rhythm: the best of the (spatial level x time level) pairs by standardized response, and its footprint across the pairs, from the stored cells and extents of `run cortical_event_pairs`\n"
    "  nxr-neuro patches compute --dataset D/N --subject S --map FIELD --level L [--sign positive|negative|magnitude] [--enter-sd 1.5] [--exit-sd 1.0] [--min-area M2] [--max-area M2] [--transform auto|eigen|chebyshev|exact] [--modes N] [--order N] [--tolerance T] [--summaries ..] [--replace]   PATCHES on a static cortical map (a PET map): where it stands out from its surroundings at the size of tile level L (the map filtered to the level's Weyl band, its tile means over the level's tiling, tiles above the baseline joined over mesh adjacency by hysteresis); stored as the band Field, tile and patch measurements, a baseline statistic, a patches Selection and statistics across the patches\n"
    "  nxr-neuro measure export --dataset D/N --labelling L --fields PATTERN --table NAME --out DIR [--statistics mean,sd,..] [--subjects A,B] [--recipe NAME]   the measurements as a table for statistics (a row a subject), its dictionary and provenance\n"
    "  nxr-neuro psd compute --dataset D/N --subject S --recording R [--set brainstorm|scipy|nxr-chain] [--segment-seconds SEC] [--through OPERATOR]   the Welch PSD of a recording, chunk by chunk, bad segments left out; --through a stored 2-row Operator (a rotation plane): its two-sided rotary spectrum\n"
    "  nxr-neuro lowpass|highpass|bandpass|bandstop|notch --dataset D/N --subject S --recording R --fpass HZ[,HZ] | --fstop LO,HI | --frequencies F,..   a zero-phase filter, chunk by chunk, into a new recording <R>_<verb>\n"
    "  nxr-neuro noisecov compute --dataset D/N --subject S --recording R [--start SEC --stop SEC]   the noise covariance of a recording's good channels, chunk by chunk\n"
    "  nxr-neuro inverse compute --dataset D/N --subject S (--noise-cov NAME | --recording R)   the unconstrained minimum-norm kernel (depth-weighted), stored as an imported one\n"
    "  nxr-neuro read <dataset folder> [--array <subject> <path> [--region a:b,c:d] [--out FILE]]   the arrays its rows name\n"
    "  nxr-neuro --version\n";

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

int main(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") { std::cout << kUsage; return argc < 2 ? 2 : 0; }
  const std::string cmd = argv[1];
  // for the parity tests only: ids a counter (from the number given) and a fixed clock, as nxr's side is run with
  if (const char* det = std::getenv("NXR_IO_DETERMINISTIC")) testing::deterministic(true, std::strtoull(det, nullptr, 10));
  if (cmd == "--version") { std::cout << "nxr-neuro " << NXR_NEURO_VERSION << " (database schema " << schema_version() << ")\n"; return 0; }
  if (cmd == "read") return read_command(argc - 1, argv + 1);
  const std::vector<std::string> rest(argv + 2, argv + argc);
  chunk_loop_defaults(rest);
  try {
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
    std::cerr << kUsage << "nxr-neuro: error: unknown command '" << cmd << "'\n";
    return 2;
  } catch (const Usage& e) {
    std::cerr << e.what() << '\n';
    return 2;
  } catch (const Error& e) {
    // Python prints a KeyError's message as its repr
    std::string m = e.what();
    if (e.python_name() == "KeyError") {
      const char q = m.find('\'') != std::string::npos && m.find('"') == std::string::npos ? '"' : '\'';
      std::string r(1, q);
      for (char c : m) { if (c == '\\') r += "\\\\"; else if (c == q) { r += '\\'; r += c; } else r += c; }
      m = r + q;
    }
    emit(Json{{"stage", "error"}, {"message", e.python_name() + ": " + m}});
    return 1;
  } catch (const std::exception& e) {
    emit(Json{{"stage", "error"}, {"message", std::string("RuntimeError: ") + e.what()}});
    return 1;
  }
}
