// nxr-neuro wishart compute --dataset D/N --subject S --recording R --windows NAME --bursts NAME --psd NAME --band LO,HI [--shape morlet|meyer] [--draws 1000] [--seed 1] [--replace]
//   THE WISHART NULL (nxr/pipelines/wishart_null.h): each burst's second moment (the stored analytic cross-spectrum windows summed over its tiles) against complex Wishart draws of its own zero-lag covariance (no lag; dof = duration x the band's effective bandwidth): the burst's sensor-level lagged share, its p and BH q, and the run's, as statistics.
// nxr-neuro pointspread compute --dataset D/N --subject S --recording R [--forward NAME] [--kernel NAME] [--levels 2,3,..,8] [--reference-level 6] [--vertices 20 | --list V,V,..] [--fraction 0.5] [--spatial-shape mexhat] [--transform auto] [--name NAME] [--replace]
//   THE INSTRUMENT'S BLUR (nxr/pipelines/pointspread.h): unit currents at a sample of vertices through the stored head model and kernel, read by the spatial events' chain; the patch size per source and orientation, as a statistic of the recording.
// nxr-neuro spatialnull compute --dataset D/N --subject S --recording R [--events spatial_events] [--set NAME=VALUE ...] [--chunks K|A-B|all | --start S --stop S] [--batch 20] [--min-draws 20] [--max-draws 100] [--h 10] [--replace]
//   THE LAG-FREE NULL OF THE SPATIAL EVENTS (nxr/pipelines/spatial_null.h): the draws of `spatial_events_null` in batches with early stopping, the per-vertex p and BH q of the home occupancy and the size distribution against them, as statistics.
// nxr-neuro null compute --dataset D/N --subject S --recording-a A [--recording-b B] [--over TILES] [--stage NAME|INDEX] [--set NAME=VALUE ...] [--chunks K|A-B|all | --start S --stop S] [--batch 20] [--min-draws 20] [--max-draws 100] [--h 10] [--min-crossings 10] [--replace]
//   THE LAG-FREE NULL'S TESTS (nxr/pipelines/null_compute.h; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10): the draws of `vortex_null` (the vortex chain on a lag-free surrogate; each a cached node, `--set` its parameters, `{recording}` the recording's name) in batches
//   with early stopping (Besag-Clifford), and the per-tile p and Benjamini-Hochberg q of the crossing consistency (and, with --recording-b, of the run repeat), the data-to-null crossing ratio and the null's mean resultants, as statistics.
// nxr-neuro joint compute --dataset D/N --subject S --recording R --space-level L --time-level M [--surface NAME] [--replace]
//   THE JOINT PARTITION of space x time (nxr/pipelines/joint.h; spec docs/specs/2026-11-13-cortical-events.md, phase 2a): a closed-form `set` Selection on (surface x the recording's time line), the cells
//   (a spatial tile of level L of the surface's tile ladder) x (a time tile of level M of the recording's constant-Q ladder), code = (tile << B) | time tile (the Studio atlas's layout). One per pair of
//   levels; `run cortical_cells` measures the source estimate over it. `tiles compute` and `constantq compute` first. A rerun keeps it; --replace rewrites it (and drops the measurements over it).
// nxr-neuro measure compute --dataset D/N --subject S --over TILING [--level K] --fields PATTERN [--measures mean,mean_square,max,min,abs_max] [--chunks K|A-B|all | --start S --stop S] [--replace]
//   OVER TIME: a time-resolved Field over a time partition (a constant-Q ladder's level, or <recording>_chunks): one array [rows x windows x components]
//   beside the Field, one row a measure (D96), `count` the last component; the span one chunk by default.
// nxr-neuro measure compute --dataset D/N --subject S --labelling L --fields PATTERN --weight none|barycentric|voronoi
//     [--measures mean,max,min,mean_square,abs_max,sum] [--recipe NAME] [--gradient] [--replace]
// nxr-neuro statistic compute --dataset D/N --subject S --over SELECTION --of FIELD|MAP --measure M --summaries median,mad,percentile:75,… [--combine each|pool|mean]
//     [--ddof N] [--name NAME] [--members-own] [--replace]      --members-own: the measure is the members' own (area, n_elements, n_tiles: rows of no Field); --of is then its parent
//   STATISTIC SUMMARIES (docs/specs/2026-11-13-statistic-summaries.md): a stored measurement summarised over its partition's members into a STATISTIC (method summary):
//   per row (each), pooled, or of the mean over rows; summaries mean · sd (n - ddof, ddof 1) · median · mad (1.4826 MAD) · percentile:p (numpy linear) · peak ·
//   kurtosis · skewness · min · max. Reads the measurement (array-backed or scalar rows), never the recording; a rerun is a hit.
// nxr-neuro measure export --dataset D/N --labelling L --fields PATTERN --table NAME --out DIR [--statistics mean,sd,…] [--subjects A,B] [--recipe NAME[@V]] [--exclude NAMES (default unassigned) | --keep-all] [--layout split|wide] [--scaling stored|density|power|amplitude|amplitude-density|db|db-density]
//     the stored measurements as a measurement table (one row a subject) with its data dictionary and provenance, a study release's layout
// Stored Fields measured over the members of a partition (nxr/pipelines/measure.h): per member and Field the Studio's measures (measureOverPartition),
// written as measurements of the labelling, with the recipe that made them. --weight has no default: none is Brainstorm's scout mean (the plain vertex
// mean), barycentric nxr's and Rheome's, voronoi the Studio's mass. JSON lines out, as the other pipelines.
#include <algorithm>
#include <cctype>
#include <sstream>

#include "cli.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/null_compute.h"
#include "nxr/pipelines/spatial_null.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/scaleselection.h"
#include "nxr/pipelines/measure.h"
#include "nxr/pipelines/joint.h"
#include "nxr/pipelines/group.h"
#include "nxr/pipelines/pyramid.h"

using namespace nxr::io;

namespace {
std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::istringstream in(s);
  for (std::string t; std::getline(in, t, sep);) if (!t.empty()) out.push_back(t);
  return out;
}
[[noreturn]] void usage(const std::string& prog, const std::string& m) { throw cli::Usage(prog + ": error: " + m); }
}  // namespace

int measure_export_main(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro measure export";
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--labelling", "--fields", "--table", "--out", "--statistics", "--subjects", "--recipe", "--exclude", "--layout", "--scaling"}, {"--keep-all"},
                                 {"--dataset", "--labelling", "--fields", "--table", "--out"});
  nxr::pipelines::MeasureExportOptions o;
  o.labelling = a.get("--labelling"); o.fields = a.get("--fields"); o.table = a.get("--table"); o.out = a.get("--out");
  if (const auto v = a.maybe("--statistics")) o.statistics = split(*v, ',');
  if (const auto v = a.maybe("--subjects")) o.subjects = split(*v, ',');
  o.recipe = a.maybe("--recipe");
  if (const auto v = a.maybe("--exclude")) o.exclude = split(*v, ',');
  if (a.flag.count("--keep-all")) o.exclude.clear();
  if (const auto v = a.maybe("--layout")) o.layout = *v;
  if (const auto v = a.maybe("--scaling")) o.scaling = *v;
  try {
    cli::emit(nxr::pipelines::measure_export(a.get("--dataset"), o));
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

// `measure query`: a pyramid's array-backed measurement at a chosen scale, rolled up AT READ (space by reshape to level L, time on the tower to level M, a band slice; derived values from rolled-up sums)
int measure_query_main(const std::vector<std::string>& argv) {
  const std::string prog = "nxr-neuro measure query";
  const cli::Args a = cli::parse(prog, argv, {"--dataset", "--subject", "--selection", "--field", "--measure", "--band", "--spatial-band", "--space-level", "--time-level", "--min-value", "--max-value", "--limit", "--order"}, {},
                                 {"--dataset", "--subject", "--selection", "--measure"});
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const std::string key = a.get("--selection");
    const nxr::io::task::Subject ts = nxr::io::task::Subject::of(sub);
    const auto sel = ts.by_id_or_name(nxr::io::task::Cls::selection, key);
    if (!sel) throw Error(code::no_such_row, sub.name() + ": no selection '" + key + "'");
    Json q = Json{{"selectionId", sel->id}, {"measure", a.get("--measure")}};
    try {
      if (const auto v = a.maybe("--field")) {
        const auto f = ts.by_id_or_name(nxr::io::task::Cls::field, *v);
        if (!f) throw Error(code::no_such_row, sub.name() + ": no field '" + *v + "'");
        q["ofFieldId"] = f->id;
      }
      if (const auto v = a.maybe("--band")) q["band"] = std::stoll(*v);
      if (const auto v = a.maybe("--spatial-band")) q["spatialBand"] = std::stoll(*v);
      if (const auto v = a.maybe("--space-level")) q["spaceLevel"] = std::stoi(*v);
      if (const auto v = a.maybe("--time-level")) q["timeLevel"] = std::stoi(*v);
      if (const auto v = a.maybe("--min-value")) q["minValue"] = std::stod(*v);
      if (const auto v = a.maybe("--max-value")) q["maxValue"] = std::stod(*v);
      if (const auto v = a.maybe("--limit")) q["limit"] = std::stoll(*v);
    } catch (const std::logic_error&) { usage(prog, "--band, --spatial-band, --space-level, --time-level and --limit are whole numbers; --min-value and --max-value numbers"); }
    if (const auto v = a.maybe("--order")) { if (*v != "asc" && *v != "desc") usage(prog, "--order is asc or desc"); q["order"] = *v; }
    if (!q.contains("spaceLevel") && !q.contains("timeLevel") && !q.contains("band") && !q.contains("spatialBand")) q["timeLevel"] = 0;     // this verb always reads the pyramid's cells
    const Json rows = ts.read_measurements(q);
    cli::emit(Json{{"stage", "result"}, {"selection", key}, {"measure", a.get("--measure")}, {"query", q}, {"rows", rows}, {"count", rows.size()}});
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int measure_main(const std::vector<std::string>& argv) {
  if (!argv.empty() && argv[0] == "query") return measure_query_main(std::vector<std::string>(argv.begin() + 1, argv.end()));
  if (!argv.empty() && argv[0] == "export") return measure_export_main(std::vector<std::string>(argv.begin() + 1, argv.end()));
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro measure: error: the following arguments are required: action (compute, query, export)");
  const std::string prog = "nxr-neuro measure compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--labelling", "--fields", "--weight", "--measures", "--recipe", "--over", "--level", "--chunks", "--start", "--stop"},
                                 {"--replace", "--gradient"}, {"--dataset", "--subject", "--fields"});
  if (a.maybe("--over")) {
    // OVER TIME: a time-resolved Field over a time partition (a constant-Q ladder's level, or a closed-form spans tiling such as <recording>_chunks)
    if (a.maybe("--labelling") || a.maybe("--weight") || a.flag.count("--gradient")) usage(prog, "--over measures over a time partition: --labelling, --weight and --gradient are for a surface partition");
    nxr::pipelines::MeasureOverTimeOptions t;
    t.over = *a.maybe("--over");
    if (const auto v = a.maybe("--level")) t.level = std::stoi(*v);
    t.fields = a.get("--fields");
    if (const auto v = a.maybe("--measures")) t.measures = split(*v, ',');
    t.chunks = a.maybe("--chunks");
    if (const auto v = a.maybe("--start")) t.start = std::stod(*v);
    if (const auto v = a.maybe("--stop")) t.stop = std::stod(*v);
    t.replace = a.flag.count("--replace") > 0;
    try {
      DatasetWriter ds = open_dataset(a);
      OpenSubjectOptions oo;
      oo.created_by = cli::stamp();
      SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
      const auto r = nxr::pipelines::measure_over_time(ds, sub, t, [](const Json& j) { cli::emit(j); });
      Json out = Json{{"stage", "result"}};
      for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
      cli::emit(out);
    } catch (const std::invalid_argument& e) {
      usage(prog, e.what());
    }
    return 0;
  }
  if (!a.maybe("--labelling") || !a.maybe("--weight")) usage(prog, "a surface partition needs --labelling and --weight (or --over for a time partition)");
  nxr::pipelines::MeasureComputeOptions o;
  o.labelling = a.get("--labelling");
  o.fields = a.get("--fields");
  o.weight = a.get("--weight");
  if (const auto v = a.maybe("--measures")) o.measures = split(*v, ',');
  o.recipe = a.maybe("--recipe");
  o.replace = a.flag.count("--replace") > 0;
  o.gradient = a.flag.count("--gradient") > 0;
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::measure_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    cli::emit(r.info);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int pyramid_main(const std::vector<std::string>& argv) {
  if (!argv.empty() && argv[0] == "spatial-bands") {
    const std::string prog = "nxr-neuro pyramid spatial-bands";
    const std::vector<std::string> rest(argv.begin() + 1, argv.end());
    const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--surface"}, {}, {"--dataset", "--subject"});
    nxr::pipelines::SpatialBandsOptions o;
    o.surface = a.maybe("--surface");
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::pyramid_spatial_bands(ds, sub, o);
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
    return 0;
  }
  if (!argv.empty() && argv[0] == "reduce") {
    // `pyramid reduce --dataset D/N --subjects A,B,C [--kernels SPEC] [--depth D]`: the group sums of the NAMED subjects' group-tree pyramids, cell for cell, in the default subject's store
    const std::string prog = "nxr-neuro pyramid reduce";
    const std::vector<std::string> rest(argv.begin() + 1, argv.end());
    const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subjects", "--kernels", "--depth", "--into"}, {}, {"--dataset", "--subjects"}, 0, {"--templates"});
    nxr::pipelines::PyramidReduceOptions o;
    { std::stringstream ss(a.get("--subjects")); std::string item; while (std::getline(ss, item, ',')) if (!item.empty()) o.subjects.push_back(item); }
    if (const auto v = a.maybe("--kernels")) o.kernels = *v;
    try { if (const auto v = a.maybe("--depth")) o.depth = std::stoi(*v); } catch (...) { usage(prog, "--depth is a whole number"); }
    if (is_catalogue_dataset(a.get("--dataset"))) {   // a catalogue: the members' packages are read, the result is a group package (one per analysis, named by --into)
      if (!a.maybe("--into")) throw cli::Usage(prog + ": error: " + a.get("--dataset") + " is a catalogue of subject packages: name the group package the result goes to with --into NAME");
      const auto g = nxr::pipelines::pyramid_reduce_group(a.get("--dataset"), o, *a.maybe("--into"), cli::template_search_of(a), [](const Json& j) { Json m = j; if (!m.contains("stage")) m["stage"] = "progress"; cli::emit(m); });
      Json out = Json{{"stage", "result"}};
      for (auto it = g.info.begin(); it != g.info.end(); ++it) out[it.key()] = it.value();
      cli::emit(out);
      return 0;
    }
    if (a.maybe("--into")) throw cli::Usage(prog + ": error: --into names a group package, which only a catalogue dataset has (`split dataset` makes one)");
    DatasetWriter ds = open_dataset(a);
    const auto r = nxr::pipelines::pyramid_reduce(ds, o, [](const Json& j) { Json m = j; if (!m.contains("stage")) m["stage"] = "progress"; cli::emit(m); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
    return 0;
  }
  if (argv.empty() || argv[0] != "build") throw cli::Usage("nxr-neuro pyramid: error: the following arguments are required: action (build, reduce, spatial-bands)");
  const std::string prog = "nxr-neuro pyramid build";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--recording", "--kernel", "--depth", "--tree", "--bands", "--shape", "--halo-seconds", "--frame-level", "--start-s", "--chunks", "--start", "--stop", "--block-samples", "--threads"},
                                 {"--replace", "--keep-bad-segments"}, {"--dataset", "--subject", "--recording"});
  nxr::pipelines::PyramidBuildOptions o;
  o.recording = a.get("--recording");
  o.kernel = a.maybe("--kernel");
  o.chunks = a.maybe("--chunks");
  if (const auto v = a.maybe("--shape")) o.shape = *v;
  if (const auto v = a.maybe("--tree")) o.tree = *v;
  try {
    if (const auto v = a.maybe("--depth")) o.depth = std::stoi(*v);
    if (const auto v = a.maybe("--halo-seconds")) o.halo_seconds = std::stod(*v);
    if (const auto v = a.maybe("--frame-level")) o.frame_level = std::stoi(*v);
    if (const auto v = a.maybe("--start-s")) o.start_s = std::stod(*v);
    if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
    if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
    if (const auto v = a.maybe("--block-samples")) o.block_samples = std::stoll(*v);
    if (const auto v = a.maybe("--threads")) o.threads = std::stoi(*v);
    if (const auto v = a.maybe("--bands")) {        // LO-HI,LO-HI,... Hz
      std::stringstream ss(*v);
      std::string item;
      while (std::getline(ss, item, ',')) {
        const auto dash = item.find('-');
        if (dash == std::string::npos) throw std::invalid_argument("x");
        o.bands.emplace_back(std::stod(item.substr(0, dash)), std::stod(item.substr(dash + 1)));
      }
    }
  } catch (...) { usage(prog, "--depth and --frame-level are whole numbers, --bands LO-HI,LO-HI (Hz), --halo-seconds, --start-s, --start and --stop numbers (s), --block-samples a whole number"); }
  o.exclude_bad_segments = a.flag.count("--keep-bad-segments") == 0;
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::pyramid_build(ds, sub, o, [](const Json& j) { Json m = j; if (!m.contains("stage")) m["stage"] = "progress"; cli::emit(m); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}



int spatialnull_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro spatialnull: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro spatialnull compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--recording", "--events", "--chunks", "--start", "--stop", "--batch", "--min-draws", "--max-draws", "--h"}, {"--replace"}, {"--dataset", "--subject", "--recording"}, 0, {"--set"});
  nxr::pipelines::SpatialNullOptions o;
  o.recording = a.get("--recording");
  if (const auto v = a.maybe("--events")) o.events = *v;
  if (a.multi.count("--set"))
    for (const auto& s : a.multi.at("--set")) {
      const auto eq = s.find('=');
      if (eq == std::string::npos || eq == 0) usage(prog, "--set " + s + ": NAME=VALUE");
      o.set[s.substr(0, eq)] = nxr::io::parse_yaml(s.substr(eq + 1), "--set " + s.substr(0, eq));
    }
  if (const auto v = a.maybe("--chunks")) o.chunks = *v;
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  auto num = [&](const char* k, int& dst) { if (const auto v = a.maybe(k)) { try { dst = std::stoi(*v); } catch (...) { usage(prog, std::string(k) + " is a whole number"); } } };
  num("--batch", o.batch); num("--min-draws", o.min_draws); num("--max-draws", o.max_draws); num("--h", o.h);
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::spatial_null_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int null_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro null: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro null compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--recording-a", "--recording-b", "--over", "--stage", "--chunks", "--start", "--stop", "--batch", "--min-draws", "--max-draws", "--h", "--min-crossings"}, {"--replace"}, {"--dataset", "--subject", "--recording-a"}, 0, {"--set"});
  nxr::pipelines::NullOptions o;
  o.recording_a = a.get("--recording-a");
  o.recording_b = a.maybe("--recording-b");
  o.over = a.maybe("--over");
  if (const auto v = a.maybe("--stage")) {
    const bool digits = !v->empty() && std::all_of(v->begin(), v->end(), [](unsigned char c) { return std::isdigit(c); });
    if (digits) o.stage = std::stoi(*v); else o.stage_name = *v;
  }
  if (a.multi.count("--set"))
    for (const auto& s : a.multi.at("--set")) {
      const auto eq = s.find('=');
      if (eq == std::string::npos || eq == 0) usage(prog, "--set " + s + ": NAME=VALUE");
      o.set[s.substr(0, eq)] = nxr::io::parse_yaml(s.substr(eq + 1), "--set " + s.substr(0, eq));
    }
  if (const auto v = a.maybe("--chunks")) o.chunks = *v;
  if (const auto v = a.maybe("--start")) o.start = std::stod(*v);
  if (const auto v = a.maybe("--stop")) o.stop = std::stod(*v);
  auto num = [&](const char* k, int& dst) { if (const auto v = a.maybe(k)) { try { dst = std::stoi(*v); } catch (...) { usage(prog, std::string(k) + " is a whole number"); } } };
  num("--batch", o.batch); num("--min-draws", o.min_draws); num("--max-draws", o.max_draws); num("--h", o.h); num("--min-crossings", o.min_crossings);
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::null_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}


int scaleselection_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro scaleselection: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro scaleselection compute";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--recording", "--space-levels", "--time-levels", "--source", "--response", "--min-link", "--name", "--surface"}, {"--replace"},
                                 {"--dataset", "--subject", "--recording", "--space-levels", "--time-levels"});
  nxr::pipelines::ScaleSelectionOptions o;
  o.recording = a.get("--recording");
  auto levels = [&](const char* flag) {
    std::vector<int> v;
    for (const std::string& t : split(a.get(flag), ',')) { try { v.push_back(std::stoi(t)); } catch (...) { usage(prog, std::string(flag) + " is a comma-separated list of whole numbers (levels)"); } }
    return v;
  };
  o.space_levels = levels("--space-levels");
  o.time_levels = levels("--time-levels");
  if (const auto v = a.maybe("--source")) o.source = *v;
  if (const auto v = a.maybe("--response")) o.response = *v;
  if (const auto v = a.maybe("--min-link")) { try { o.min_link = std::stoi(*v); } catch (...) { usage(prog, "--min-link is a whole number (at least 1)"); } }
  o.name = a.maybe("--name");
  o.surface = a.maybe("--surface");
  o.replace = a.flag.count("--replace") > 0;
  try {
    DatasetWriter ds = open_dataset(a);
    OpenSubjectOptions oo;
    oo.created_by = cli::stamp();
    SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
    const auto r = nxr::pipelines::scaleselection_compute(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

