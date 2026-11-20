// nxr-neuro — the command line, the verbs of nxr's: dataset · subject · surface (and read, step 1's reader).
//
//   nxr-neuro dataset create --datastore D --name N [--source-tool T --source-protocol P --source-path S] [--store-format 52|51]   (default 52: each subject's small keys in its small_files.sqlite)
//                            [--templates FS_SUBJECTS [--default fsaverage5] [--modes K] [--atlas-finest \"parts 256\"] [--lc-scales \"parts 16,parts 64\"]]   with the default subject (io/default_subject.h)
//                            its frames (`run surface_frames`), its group tree (`run select_dyadic_surface` rule group) and the Levi-Civita connection (a `measure` over a level x level product)
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
//   nxr-neuro run eigenmodes --dataset D/N --subject S --surface N [--modes K] [--glued] [--replace]     (a definition)
//   nxr-neuro headmodel compute | noisecov compute | inverse compute | psd compute ...                                           (cli/meeg_command.cpp)
//   nxr-neuro read <dataset folder> [--array <subject> <path> [--region a:b,c:d] [--out FILE]]
//
// Like nxr's, progress and results are JSON lines on stdout (`stage` keyed); a failure is a `stage: error` line and exit 1 (a usage error: exit 2).
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>

#ifdef NXR_NEURO_WITH_PIPELINES
#include "nxr/pipelines/face.h"
#endif
#include "cli.h"
#include "nxr/io/chunk_loop.h"
#include "nxr/io/errors.h"

#ifndef NXR_NEURO_VERSION
#define NXR_NEURO_VERSION "0.0.0"
#endif

using namespace nxr::io;
using namespace cli;
namespace fs = std::filesystem;

int read_command(int argc, char** argv);
int convert_main(const std::vector<std::string>& argv);
#ifdef NXR_NEURO_WITH_PIPELINES
int measure_main(const std::vector<std::string>& argv);        // cli/measure_command.cpp
int surrogate_lagfree_spatial_events_compare_main(const std::vector<std::string>& argv);   // cli/measure_command.cpp
int surrogate_lagfree_vortex_compare_main(const std::vector<std::string>& argv);           // cli/measure_command.cpp
int run_main(const std::vector<std::string>& argv);            // cli/run_command.cpp
int compile_main(const std::vector<std::string>& argv);        // cli/run_command.cpp
#endif

namespace {

// ── nxr-neuro dataset ── (each action: parse, one library call (nxr/io/task/commands.h), print)
int dataset_main(const std::vector<std::string>& argv) {
  if (argv.empty()) throw Usage("nxr-neuro dataset: error: the following arguments are required: action");
  const std::string action = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro dataset " + action;
  if (action == "create") {
    const Args a = parse(prog, rest, {"--dataset", "--datastore", "--name", "--source-tool", "--source-protocol", "--source-path", "--templates", "--default", "--modes", "--atlas-finest", "--lc-scales", "--store-format"}, {}, {});
    task::CreateDatasetRequest rq;
    rq.dataset = a.maybe("--dataset");
    rq.datastore = a.maybe("--datastore");
    rq.name = a.maybe("--name");
    rq.source_tool = a.maybe("--source-tool");
    rq.source_protocol = a.maybe("--source-protocol");
    rq.source_path = a.maybe("--source-path");
    rq.templates = a.maybe("--templates");
    rq.default_ = a.maybe("--default");
    rq.atlas_finest = a.maybe("--atlas-finest");    // a quantity (parts per hemisphere, a tile diameter, a tile area): parsed where the grid is made
    rq.store_format = a.maybe("--store-format");
    rq.created_by = stamp();
    if (const auto v = a.maybe("--lc-scales")) {   // the tile scales of the Levi-Civita measurement, comma-separated quantities ("parts 16,parts 64")
      rq.lc_scales.emplace();
      std::stringstream ss(*v);
      for (std::string x; std::getline(ss, x, ',');) {
        if (x.empty()) throw Usage(prog + ": error: --lc-scales is a comma-separated list of quantities (e.g. \"parts 16,parts 64\" or \"20 mm,40 mm\"), not '" + *v + "'");
        rq.lc_scales->push_back(x);
      }
    }
    if (const auto v = a.maybe("--modes")) {        // the number of Hodge–Dirac modes to solve (as `run eigenmodes --modes`)
      try { rq.modes = std::stoi(*v); } catch (...) { throw Usage(prog + ": error: --modes is a whole number (the number of modes to solve), not '" + *v + "'"); }
    }
#ifdef NXR_NEURO_WITH_PIPELINES
    const task::CommandResult r = call(prog, [&] { return nxr::pipelines::face::create_dataset(rq, progress()); });
    if (rq.templates && !rq.dataset) {             // the library's report with the default subject: "stage" last, as it always printed
      Json out = r.result;
      out["stage"] = "result";
      emit(out);
      return 0;
    }
#else
    const task::CommandResult r = call(prog, [&] { return task::create_dataset(rq, progress()); });
#endif
    emit_result(r.result);
  } else if (action == "list") {
    const Args a = parse(prog, rest, {"--datastore"}, {}, {"--datastore"});
    emit_result(task::list_datasets({a.get("--datastore")}).result);
  } else if (action == "show") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    emit_result(call(prog, [&] { return task::show_dataset({a.get("--dataset")}, progress()); }).result);
  } else if (action == "delete") {
    const Args a = parse(prog, rest, {"--datastore", "--name"}, {}, {"--datastore", "--name"});
    emit_result(task::delete_dataset({a.get("--datastore"), a.get("--name")}).result);
  } else if (action == "recover") {
    const Args a = parse(prog, rest, {"--dataset"}, {}, {"--dataset"});
    emit_result(call(prog, [&] { return task::recover_dataset({a.get("--dataset")}); }).result);
  } else if (action == "compact") {
    // nxr-neuro dataset compact --dataset D/N [--subject S] [--dry-run] [--no-shard] [--no-small-files]   (FORMAT 50 -> 51 -> 52, array by array: io/include/nxr/io/compact.h)
    const Args a = parse(prog, rest, {"--dataset", "--subject"}, {"--dry-run", "--no-shard", "--no-small-files"}, {"--dataset"});
    task::CompactDatasetRequest rq;
    rq.dataset = a.get("--dataset");
    rq.subject = a.maybe("--subject");
    rq.dry_run = a.flag.count("--dry-run") > 0;
    rq.no_shard = a.flag.count("--no-shard") > 0;
    rq.no_small_files = a.flag.count("--no-small-files") > 0;
    emit_result(call(prog, [&] { return task::compact_dataset(rq); }).result);
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
  if (action == "list") {
    const Args a = parse(prog, rest, {"--dataset", "--subjects", "--where"}, {}, {"--dataset"});
    task::ListSubjectsRequest rq;
    rq.dataset = a.get("--dataset");
    if (const auto v = a.maybe("--subjects")) rq.subjects = split_commas(*v);
    rq.where = a.maybe("--where");
    emit_result(call(prog, [&] { return task::list_subjects(rq, progress()); }).result);
    return 0;
  }
  if (action == "show") {
    const Args a = parse(prog, rest, {"--dataset", "--subject"}, {}, {"--dataset", "--subject"});
    emit_result(call(prog, [&] { return task::show_subject({a.get("--dataset"), a.get("--subject"), values_of(a, "--templates")}, progress()); }).result);
    return 0;
  }
  if (action != "create" && action != "delete") throw Usage("nxr-neuro subject: error: argument action: invalid choice: '" + action + "' (choose from 'create', 'list', 'show', 'delete')");
  if (action == "create") {
    const Args a = parse(prog, rest, {"--dataset", "--name", "--kind"}, {}, {"--dataset", "--name"});
    task::CreateSubjectRequest rq;
    rq.dataset = a.get("--dataset");
    rq.name = a.get("--name");
    rq.kind = a.maybe("--kind").value_or("subject");
    rq.created_by = stamp();
    emit_result(call(prog, [&] { return task::create_subject(rq, progress()); }).result);
    return 0;
  }
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--node"}, {}, {"--dataset", "--subject"});
  task::DeleteSubjectRequest rq;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  rq.node = a.maybe("--node");
  rq.templates = values_of(a, "--templates");
  rq.created_by = stamp();
  emit_result(call(prog, [&] { return task::delete_subject(rq); }).result);
  return 0;
}

// ── nxr-neuro surface ──
int surface_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "add") throw Usage("nxr-neuro surface: error: the following arguments are required: action (add)");
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro surface add";
  // a mesh FILE, or the surface from arrays (--vertices and --faces, .npy): then no positional argument
  const bool arrays = std::find(rest.begin(), rest.end(), "--vertices") != rest.end() || std::find(rest.begin(), rest.end(), "--faces") != rest.end();
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--scale", "--vertices", "--faces", "--by"}, {"--primary", "--no-primary"}, {"--dataset", "--subject"}, arrays ? 0 : 1);
  task::AddSurfaceRequest rq;
  rq.scale = a.maybe("--scale") ? std::stod(*a.maybe("--scale")) : 1.0;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  if (!arrays) rq.file = a.pos[0];
  rq.vertices = a.maybe("--vertices");
  rq.faces = a.maybe("--faces");
  rq.name = a.maybe("--name");
  rq.primary = a.flag.count("--primary") > 0;
  rq.no_primary = a.flag.count("--no-primary") > 0;
  rq.templates = values_of(a, "--templates");
  rq.created_by = stamp();
  rq.by = a.maybe("--by");
  emit_result(call(prog, [&] { return task::add_surface(rq, progress()); }).result);
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
int list_class_main(const std::string& what, const std::vector<std::string>& rest);   // cli/node_command.cpp
int list_nodes_main(const std::vector<std::string>& argv) {
  for (const char* cls : {"manifolds", "fields", "selections", "operators", "statistics", "dictionaries", "spaces", "acquisitions", "jobs"})
    if (!argv.empty() && argv[0] == cls) return list_class_main(argv[0], std::vector<std::string>(argv.begin() + 1, argv.end()));
#ifdef NXR_NEURO_WITH_PIPELINES
  if (argv.empty()) throw Usage("nxr-neuro list: error: the following arguments are required: what (recordings, kernels, results)");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro list " + what;
  const Args a = parse(prog, rest, {"--dataset", "--subject"}, {}, {"--dataset", "--subject"});
  const nxr::pipelines::face::ListNodesRequest rq{a.get("--dataset"), a.get("--subject"), values_of(a, "--templates")};
  namespace face = nxr::pipelines::face;
  if (what == "recordings") emit_result(call(prog, [&] { return face::list_recordings(rq, progress()); }).result);
  else if (what == "kernels") emit_result(call(prog, [&] { return face::list_kernels(rq, progress()); }).result);
  else if (what == "results") emit_result(call(prog, [&] { return face::list_results(rq, progress()); }).result);
  else throw Error(code::invalid_argument, "list: '" + what + "' is not something a subject holds by kind (recordings, kernels or results)");
  return 0;
#else
  (void)argv;
  throw Error(code::unsupported, "nxr-neuro list needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
}

int package_main(const std::vector<std::string>& argv);   // cli/package_command.cpp
int import_main(const std::vector<std::string>& argv);    // cli/import_command.cpp
int export_main(const std::vector<std::string>& argv);    // cli/export_command.cpp
int crossspectrum_read_main(const std::vector<std::string>& rest);   // cli/meeg_command.cpp
int read_node_main(const std::vector<std::string>& argv);   // cli/node_command.cpp
int add_node_main(const std::vector<std::string>& argv);    // cli/node_command.cpp
int edit_node_main(const std::vector<std::string>& argv);   // cli/node_command.cpp
int run_legacy(const std::string& cmd, const std::vector<std::string>& rest) {
  if (cmd == "__list") return list_nodes_main(rest);
  if (cmd == "__read") return read_node_main(rest);           // `read field | statistic | operator | selection | manifold`
  if (cmd == "__edit") return edit_node_main(rest);           // `edit manifold | dictionary`
  if (cmd == "__add") return add_node_main(rest);             // `add field | selection | statistic | measurements | operator | points | line`
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
    if (cmd == "subject") return subject_main(rest);
    if (cmd == "surface") return surface_main(rest);
    if (cmd == "convert") return convert_main(rest);
    if (cmd == "run") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return run_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro run needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "__compile") {                                                 // `compile <definition>`
#ifdef NXR_NEURO_WITH_PIPELINES
      return compile_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro compile needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
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
    if (cmd == "crossspectrum" && !rest.empty() && rest[0] == "read") {       // `read crossspectrum`
#ifdef NXR_NEURO_WITH_PIPELINES
      return crossspectrum_read_main(std::vector<std::string>(rest.begin() + 1, rest.end()));
#else
      throw Error(code::unsupported, "nxr-neuro read crossspectrum needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
    if (cmd == "measure") {
#ifdef NXR_NEURO_WITH_PIPELINES
      return measure_main(rest);
#else
      throw Error(code::unsupported, "nxr-neuro measure needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
    }
  throw Usage("nxr-neuro: error: unknown command '" + cmd + "'");
}


void install_chunk_loop_defaults(const std::vector<std::string>& args) { chunk_loop_defaults(args); }

int schema_version_for_version_flag() { return nxr::io::schema_version(); }
