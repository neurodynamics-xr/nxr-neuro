// nxr-neuro run surrogate_lagfree_pairs --dataset D/N --subject S --recording R --windows NAME --bursts NAME --psd NAME --band LO,HI [--shape morlet|meyer] [--draws 1000] [--seed 1] [--replace]   (a definition)
//   THE LAG-FREE PAIRS SURROGATE (definition surrogate_lagfree_pairs, node surrogate.lagfree_pairs; spec docs/specs/2026-11-17-surrogates.md): each burst's second moment against lag-free Wishart draws of its own zero-lag second moment.
// nxr-neuro pointspread compute --dataset D/N --subject S --recording R [--forward NAME] [--kernel NAME] [--levels 2,3,..,8] [--reference-level 6] [--vertices 20 | --list V,V,..] [--fraction 0.5] [--spatial-shape mexhat] [--transform auto] [--name NAME] [--replace]
//   THE INSTRUMENT'S BLUR (nxr/pipelines/pointspread.h): unit currents at a sample of vertices through the stored head model and kernel, read by the spatial events' chain; the patch size per source and orientation, as a statistic of the recording.
// nxr-neuro run surrogate_lagfree_spatial_events_compare --dataset D/N --subject S --recording R [--events spatial_events] [--set NAME=VALUE ...] [--chunks K|A-B|all | --start S --stop S] [--batch 20] [--min-draws 20] [--max-draws 100] [--h 10] [--replace]
//   THE LAG-FREE SURROGATE OF THE SPATIAL EVENTS (nxr/pipelines/surrogate_compare.h): the draws of `surrogate_lagfree_spatial_events` in batches with early stopping, the per-vertex p and BH q of the home occupancy and the size distribution against them, as statistics.
// nxr-neuro run surrogate_lagfree_vortex_compare --dataset D/N --subject S --recording-a A [--recording-b B] [--over TILES] [--stage NAME|INDEX] [--set NAME=VALUE ...] [--chunks K|A-B|all | --start S --stop S] [--batch 20] [--min-draws 20] [--max-draws 100] [--h 10] [--min-crossings 10] [--replace]
//   THE LAG-FREE SURROGATE OF THE VORTEX CHAIN (nxr/pipelines/surrogate_compare.h; spec docs/specs/2026-11-17-surrogates.md): the draws of `surrogate_lagfree_vortex` (the vortex chain on a lag-free surrogate; each a cached run, `--set` its parameters, `{recording}` the recording's name) in batches
//   with early stopping (Besag-Clifford), and the per-tile p and Benjamini-Hochberg q of the crossing consistency (and, with --recording-b, of the run repeat), the data-to-null crossing ratio and the null's mean resultants, as statistics.
// nxr-neuro measure query --dataset D/N --subject S --selection NAME --measure M [--field NAME] [--band "8-12 Hz"] [--spatial-scale "wavelength 40 mm"] [--tile "2 s"] [--min-value X] [--max-value X] [--limit N] [--order asc|desc]
//   Stored measurements over a selection's members, read at a scale STATED AS A QUANTITY (never a level index): the finest ingredients come from io, the roll-up to the coarser level and the derived
//   measures are compute's (the pipelines' query face), the level is printed with its snapped quantity. Measuring itself is `nxr-neuro run measure`; summarising `run summarize`.
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
#include "nxr/pipelines/surrogate_compare.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/measure.h"
#include "nxr/pipelines/face.h"

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
  nxr::pipelines::face::ExportMeasurementsRequest rq;
  rq.dataset = a.get("--dataset");
  rq.labelling = a.get("--labelling"); rq.fields = a.get("--fields"); rq.table = a.get("--table"); rq.out = a.get("--out");
  if (const auto v = a.maybe("--statistics")) rq.statistics = split(*v, ',');
  if (const auto v = a.maybe("--subjects")) rq.subjects = split(*v, ',');
  rq.recipe = a.maybe("--recipe");
  if (const auto v = a.maybe("--exclude")) rq.exclude = split(*v, ',');
  rq.keep_all = a.flag.count("--keep-all") > 0;
  rq.layout = a.maybe("--layout");
  rq.scaling = a.maybe("--scaling");
  cli::emit_result(cli::call(prog, [&] { return nxr::pipelines::face::export_measurements(rq); }).result);
  return 0;
}

// `measure query`: stored measurements over a selection's members at a scale stated as a quantity: the finest ingredients are io's, the roll-up through the grid's closed-form hierarchy and the derived measures
// are compute's (pipelines/face.h `measurement_query`); the level read is printed with its snapped quantity
int measure_query_main(const std::vector<std::string>& argv) {
  const std::string prog = "nxr-neuro measure query";
  const cli::Args a = cli::parse(prog, argv, {"--dataset", "--subject", "--selection", "--field", "--measure", "--band", "--spatial-scale", "--tile", "--min-value", "--max-value", "--limit", "--order", "--where", "--subjects", "--factor"}, {},
                                 {"--dataset", "--subject", "--selection", "--measure"});
  try {
    nxr::pipelines::face::QueryMeasurementsRequest rq;
    rq.dataset = a.get("--dataset");
    rq.subject = a.get("--subject");
    rq.selection = a.get("--selection");
    rq.measure = a.get("--measure");
    rq.field = a.maybe("--field");
    rq.band = a.maybe("--band");
    rq.spatial_scale = a.maybe("--spatial-scale");
    rq.tile = a.maybe("--tile");
    rq.factor = a.maybe("--factor");
    if (const auto v = a.maybe("--subjects")) rq.subjects = split(*v, ',');
    rq.where = a.maybe("--where");
    if (const auto v = a.maybe("--min-value")) rq.min_value = std::stod(*v);
    if (const auto v = a.maybe("--max-value")) rq.max_value = std::stod(*v);
    if (const auto v = a.maybe("--limit")) rq.limit = std::stoll(*v);
    rq.order = a.maybe("--order");
    cli::emit_result(cli::call(prog, [&] { return nxr::pipelines::face::query_measurements(rq); }).result);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int measure_main(const std::vector<std::string>& argv) {
  if (!argv.empty() && argv[0] == "query") return measure_query_main(std::vector<std::string>(argv.begin() + 1, argv.end()));
  if (!argv.empty() && argv[0] == "export") return measure_export_main(std::vector<std::string>(argv.begin() + 1, argv.end()));
  throw cli::Usage("nxr-neuro measure: error: the following arguments are required: action (query, export); measuring is `nxr-neuro run measure` (a field over a selection's members), summarising `nxr-neuro run summarize`");
}

int surrogate_lagfree_spatial_events_compare_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro run surrogate_lagfree_spatial_events_compare: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro run surrogate_lagfree_spatial_events_compare";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--recording", "--events", "--chunks", "--start", "--stop", "--batch", "--min-draws", "--max-draws", "--h"}, {"--replace"}, {"--dataset", "--subject", "--recording"}, 0, {"--set"});
  nxr::pipelines::SurrogateSpatialEventsCompareOptions o;
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
    const auto r = nxr::pipelines::surrogate_lagfree_spatial_events_compare(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}

int surrogate_lagfree_vortex_compare_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "compute") throw cli::Usage("nxr-neuro run surrogate_lagfree_vortex_compare: error: the following arguments are required: action (compute)");
  const std::string prog = "nxr-neuro run surrogate_lagfree_vortex_compare";
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse(prog, rest, {"--dataset", "--subject", "--recording-a", "--recording-b", "--over", "--stage", "--chunks", "--start", "--stop", "--batch", "--min-draws", "--max-draws", "--h", "--min-crossings"}, {"--replace"}, {"--dataset", "--subject", "--recording-a"}, 0, {"--set"});
  nxr::pipelines::SurrogateVortexCompareOptions o;
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
    const auto r = nxr::pipelines::surrogate_lagfree_vortex_compare(ds, sub, o, [](const Json& j) { cli::emit(j); });
    Json out = Json{{"stage", "result"}};
    for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
    cli::emit(out);
  } catch (const std::invalid_argument& e) {
    usage(prog, e.what());
  }
  return 0;
}


