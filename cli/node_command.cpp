// nxr-neuro read <class> · add <class> (docs/specs/2026-11-19-io-service.md, section 4): one stored node by name, read as arrays; what a user computed, stored as one unit.
//
//   nxr-neuro read field --dataset D --subject S --name N [--from S --to S | --all] [--rows 0,2] [--projectors active|none] [--shape-only] [--out FILE.npy]
//   nxr-neuro read statistic | operator --dataset D --subject S --name N [--out FILE.npy]
//   nxr-neuro read selection --dataset D --subject S --name N [--measure M [--field F]] [--out FILE.npy]
//   nxr-neuro read manifold --dataset D --subject S --name N [--family NAME] [--out FILE.npy]
//   nxr-neuro add field --dataset D --subject S --name N --manifold M --values FILE.npy [--time LINE] [--value-type T] [--kind K] [--unit U] [--cell 0|1|2] [--derived-from F] [--description TEXT] [--replace]
//   nxr-neuro add selection --dataset D --subject S --name N --manifold M --type partition|set|indices|spans (--members | --indices | --spans FILE.npy) [--entries A,B] [--description TEXT] [--replace]
//   nxr-neuro add statistic --dataset D --subject S --name N --measure M --method X --values FILE.npy --axes NAME:MANIFOLD ... [--value-type T] [--of-field F] [--partition P] [--unit U] [--description TEXT] [--replace]
//   nxr-neuro add measurements --dataset D --subject S --selection SEL --measure NAME=FILE.npy ... [--field F] [--unit U] [--replace]
//   nxr-neuro add operator --dataset D --subject S --name N --from-manifold A --to-manifold B (--values FILE.npy | --data F --indices F --indptr F) [--kind K] [--description TEXT] [--replace]
//   nxr-neuro add points --dataset D --subject S --name N --positions FILE.npy [--coordinate-system KEY] [--description TEXT] [--replace]
//   nxr-neuro add line --dataset D --subject S --name N --n N --spacing X --unit U [--origin X] [--dimension time|frequency|...] [--replace]
//
// Arrays go in and out as .npy files (C order, little-endian, the stored precision). The command parses, calls its library function once (nxr/pipelines/face.h for read,
// nxr/io/task/commands.h for add) and prints the result.
#include <string>
#include <vector>

#include "cli.h"
#include "registry.h"
#include "nxr/io/errors.h"
#ifdef NXR_NEURO_WITH_PIPELINES
#include "nxr/pipelines/face.h"
#endif

using namespace cli;
namespace task = nxr::io::task;

namespace {

double number_of(const std::string& prog, const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const double v = std::stod(text, &used);
    if (used == text.size()) return v;
  } catch (const std::exception&) {}
  throw Usage(prog + ": error: argument " + option + ": a number (got '" + text + "')");
}
std::int64_t integer_of(const std::string& prog, const std::string& option, const std::string& text) {
  try {
    std::size_t used = 0;
    const long long v = std::stoll(text, &used);
    if (used == text.size()) return v;
  } catch (const std::exception&) {}
  throw Usage(prog + ": error: argument " + option + ": a whole number (got '" + text + "')");
}

void need_noun(const std::vector<std::string>& argv, const std::string& verb, const char* nouns) {
  if (argv.empty()) throw Usage("nxr-neuro " + verb + ": error: the following arguments are required: what (" + nouns + ")");
}

}  // namespace

static task::CommandResult read_subject_call(const task::ReadSubjectRequest& rq) {
#ifdef NXR_NEURO_WITH_PIPELINES
  return nxr::pipelines::face::read_subject(rq, progress());   // with the nodes' types
#else
  return task::read_subject(rq, progress());
#endif
}
static task::CommandResult list_class_call(const task::ListClassRequest& rq) {
#ifdef NXR_NEURO_WITH_PIPELINES
  return nxr::pipelines::face::list_class(rq, progress());
#else
  return task::list_class(rq, progress());
#endif
}

// ── nxr-neuro read subject: what a subject holds, every node's description ──
static int read_subject_main(const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro read subject";
  const Args a = parse(prog, rest, {"--dataset", "--subject", "--classes"}, {"--related"}, {"--dataset", "--subject"});
  task::ReadSubjectRequest rq;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  if (const auto v = a.maybe("--classes")) rq.classes = split_commas(*v);
  rq.related = a.flag.count("--related") > 0;
  rq.templates = values_of(a, "--templates");
  emit_result(call(prog, [&] { return read_subject_call(rq); }).result);
  return 0;
}

// ── nxr-neuro list fields|manifolds|selections|operators|statistics|dictionaries|spaces|acquisitions|jobs ──
int list_class_main(const std::string& what, const std::vector<std::string>& rest) {
  const std::string prog = "nxr-neuro list " + what;
  std::set<std::string> valued = {"--dataset", "--subject", "--limit", "--after", "--columns"};
  if (what != "jobs") valued.insert("--name");
  if (what != "acquisitions") valued.insert("--kind");
  if (what == "manifolds" || what == "fields" || what == "selections" || what == "operators" || what == "statistics") valued.insert("--on");
  const Args a = parse(prog, rest, valued, {}, {"--dataset"}, 0, {"--templates"});
  task::ListClassRequest rq;
  rq.cls = what;
  rq.dataset = a.get("--dataset");
  rq.subject = a.maybe("--subject");
  rq.name = a.maybe("--name");
  rq.kind = a.maybe("--kind");
  rq.on = a.maybe("--on");
  rq.after = a.maybe("--after");
  if (const auto v = a.maybe("--limit")) rq.limit = integer_of(prog, "--limit", *v);
  if (const auto v = a.maybe("--columns")) rq.columns = split_commas(*v);
  rq.templates = values_of(a, "--templates");
  emit_result(call(prog, [&] { return list_class_call(rq); }).result);
  return 0;
}

// ── nxr-neuro read field|statistic|operator|selection|manifold ──
int read_node_main(const std::vector<std::string>& argv) {
  if (!argv.empty() && argv[0] == "subject") return read_subject_main(std::vector<std::string>(argv.begin() + 1, argv.end()));
#ifdef NXR_NEURO_WITH_PIPELINES
  need_noun(argv, "read", "subject, field, statistic, operator, selection, manifold");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro read " + what;
  std::set<std::string> valued = {"--dataset", "--subject", "--name", "--out"}, flags;
  if (what == "field") { valued.insert({"--from", "--to", "--rows", "--projectors"}); flags = {"--all", "--shape-only"}; }
  else if (what == "selection") valued.insert({"--measure", "--field"});
  else if (what == "manifold") valued.insert("--family");
  else if (what != "statistic" && what != "operator") throw Usage("nxr-neuro read: error: argument what: invalid choice: '" + what + "'");
  const Args a = parse(prog, rest, valued, flags, {"--dataset", "--subject", "--name"});
  namespace face = nxr::pipelines::face;
  face::ReadNodeRequest rq;
  rq.dataset = a.get("--dataset");
  rq.subject = a.get("--subject");
  rq.name = a.get("--name");
  if (const auto v = a.maybe("--from")) rq.from = parse_time("--from", *v);
  if (const auto v = a.maybe("--to")) rq.to = parse_time("--to", *v);
  rq.all = a.flag.count("--all") > 0;
  rq.shape_only = a.flag.count("--shape-only") > 0;
  if (const auto v = a.maybe("--rows")) for (const auto& r : split_commas(*v)) rq.rows.push_back(static_cast<std::int32_t>(integer_of(prog, "--rows", r)));
  if (const auto v = a.maybe("--projectors")) {
    if (*v != "active" && *v != "none") throw Usage(prog + ": error: argument --projectors: invalid choice: '" + *v + "' (choose from 'active', 'none')");
    rq.projectors = *v;
  }
  rq.measure = a.maybe("--measure");
  rq.field = a.maybe("--field");
  rq.family = a.maybe("--family");
  rq.out = a.maybe("--out");
  rq.templates = values_of(a, "--templates");
  const auto r = call(prog, [&] {
    if (what == "field") return face::read_field(rq);
    if (what == "statistic") return face::read_statistic(rq);
    if (what == "operator") return face::read_operator(rq);
    if (what == "selection") return face::read_selection(rq);
    return face::read_manifold(rq);
  });
  emit_result(r.result);
  return 0;
#else
  (void)argv;
  throw nxr::io::Error(nxr::io::code::unsupported, "nxr-neuro read <class> needs the pipelines: build with -DNXR_NEURO_BUILD_PIPELINES=ON");
#endif
}

// ── nxr-neuro add field|selection|statistic|measurements|operator|points|line ──
int add_node_main(const std::vector<std::string>& argv) {
  need_noun(argv, "add", "field, selection, statistic, measurements, operator, points, line");
  const std::string what = argv[0];
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const std::string prog = "nxr-neuro add " + what;
  const std::string by = stamp();
  auto common = [&](const Args& a, auto& rq) {
    rq.dataset = a.get("--dataset");
    rq.subject = a.get("--subject");
    rq.replace = a.flag.count("--replace") > 0;
    rq.templates = values_of(a, "--templates");
    rq.created_by = by;
  };
  task::CommandResult r;
  if (what == "field") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--manifold", "--values", "--time", "--value-type", "--kind", "--unit", "--cell", "--derived-from", "--description"},
                         {"--replace"}, {"--dataset", "--subject", "--name", "--manifold", "--values"});
    task::AddFieldRequest rq;
    common(a, rq);
    rq.name = a.get("--name");
    rq.manifold = a.get("--manifold");
    rq.values = a.get("--values");
    rq.time = a.maybe("--time");
    if (const auto v = a.maybe("--value-type")) rq.value_type = *v;
    if (const auto v = a.maybe("--kind")) rq.kind = *v;
    rq.unit = a.maybe("--unit");
    rq.cell = a.maybe("--cell");
    rq.derived_from = a.maybe("--derived-from");
    rq.description = a.maybe("--description");
    r = call(prog, [&] { return task::add_field(rq, progress()); });
  } else if (what == "selection") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--manifold", "--type", "--members", "--indices", "--spans", "--entries", "--description"}, {"--replace"},
                         {"--dataset", "--subject", "--name", "--manifold", "--type"});
    task::AddSelectionRequest rq;
    common(a, rq);
    rq.name = a.get("--name");
    rq.manifold = a.get("--manifold");
    rq.type = a.get("--type");
    rq.members = a.maybe("--members");
    rq.indices = a.maybe("--indices");
    rq.spans = a.maybe("--spans");
    if (const auto v = a.maybe("--entries")) rq.entries = split_commas(*v);
    rq.description = a.maybe("--description");
    r = call(prog, [&] { return task::add_selection(rq, progress()); });
  } else if (what == "statistic") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--measure", "--method", "--values", "--value-type", "--of-field", "--partition", "--unit", "--description"}, {"--replace"},
                         {"--dataset", "--subject", "--name", "--measure", "--method", "--values", "--axes"}, 0, {"--axes"});
    task::AddStatisticRequest rq;
    common(a, rq);
    rq.name = a.get("--name");
    rq.measure = a.get("--measure");
    rq.method = a.get("--method");
    rq.values = a.get("--values");
    rq.axes = values_of(a, "--axes");
    if (const auto v = a.maybe("--value-type")) rq.value_type = *v;
    rq.of_field = a.maybe("--of-field");
    rq.partition = a.maybe("--partition");
    rq.unit = a.maybe("--unit");
    rq.description = a.maybe("--description");
    r = call(prog, [&] { return task::add_statistic(rq, progress()); });
  } else if (what == "measurements") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--selection", "--field", "--unit"}, {"--replace"}, {"--dataset", "--subject", "--selection", "--measure"}, 0, {"--measure"});
    task::AddMeasurementsRequest rq;
    common(a, rq);
    rq.selection = a.get("--selection");
    rq.field = a.maybe("--field");
    rq.unit = a.maybe("--unit");
    rq.measure = values_of(a, "--measure");
    r = call(prog, [&] { return task::add_measurements(rq, progress()); });
  } else if (what == "operator") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--kind", "--from-manifold", "--to-manifold", "--values", "--data", "--indices", "--indptr", "--description"}, {"--replace"},
                         {"--dataset", "--subject", "--name", "--from-manifold", "--to-manifold"});
    task::AddOperatorRequest rq;
    common(a, rq);
    rq.name = a.get("--name");
    if (const auto v = a.maybe("--kind")) rq.kind = *v;
    rq.from_manifold = a.get("--from-manifold");
    rq.to_manifold = a.get("--to-manifold");
    rq.values = a.maybe("--values");
    rq.data = a.maybe("--data");
    rq.indices = a.maybe("--indices");
    rq.indptr = a.maybe("--indptr");
    rq.description = a.maybe("--description");
    r = call(prog, [&] { return task::add_operator(rq, progress()); });
  } else if (what == "points") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--positions", "--coordinate-system", "--description"}, {"--replace"}, {"--dataset", "--subject", "--name", "--positions"});
    task::AddPointsRequest rq;
    common(a, rq);
    rq.name = a.get("--name");
    rq.positions = a.get("--positions");
    rq.coordinate_system = a.maybe("--coordinate-system");
    rq.description = a.maybe("--description");
    r = call(prog, [&] { return task::add_points(rq, progress()); });
  } else if (what == "line") {
    const Args a = parse(prog, rest, {"--dataset", "--subject", "--name", "--n", "--spacing", "--origin", "--unit", "--dimension"}, {"--replace"},
                         {"--dataset", "--subject", "--name", "--n", "--spacing", "--unit"});
    task::AddLineRequest rq;
    common(a, rq);
    rq.name = a.get("--name");
    rq.n = integer_of(prog, "--n", a.get("--n"));
    rq.spacing = number_of(prog, "--spacing", a.get("--spacing"));
    if (const auto v = a.maybe("--origin")) rq.origin = number_of(prog, "--origin", *v);
    rq.unit = a.get("--unit");
    if (const auto v = a.maybe("--dimension")) rq.dimension = *v;
    r = call(prog, [&] { return task::add_line(rq, progress()); });
  } else {
    throw Usage("nxr-neuro add: error: argument what: invalid choice: '" + what + "' (choose from 'field', 'selection', 'statistic', 'measurements', 'operator', 'points', 'line', 'surface')");
  }
  emit_result(r.result);
  return 0;
}
