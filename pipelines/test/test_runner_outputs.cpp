// The runner stores every output a definition DECLARES through ONE task::Unit (docs/specs/2026-11-17-definition-outputs.md, sections 2, 3 and 6) on a small synthetic subject (no real data):
// toy nodes (pure, registered under grouped names) describe a field, a statistic (a channel axis on the recording's channels, a frequency line made from the node's description, the segment
// partition as a tiling), an operator, a partition and a manifold; the definition places them by link. Checked: every kind stored with its provenance and cache key; a manifold an output lies on made
// FIRST by the same run; all or nothing (an output that fails leaves nothing of the run); a rerun a cache hit; another key refused, then replaced with --replace (moved aside); a declaration the
// node does not meet refused; presets (defaults < preset < the caller's, an unknown one refused, the name in the provenance and the key); a shipped-style definition's field output stored as before;
// TYPED AXES: a node that declares them with an anonymous axis is refused before any data, and one whose axis is the recording's channel manifold is stored with that link.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>

#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/graph/node.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/task.h"

using namespace nxr::io;
namespace g = nxr::pipelines::graph;
namespace ng = nxr::graph;
namespace fs = std::filesystem;
namespace task = nxr::pipelines::task;

namespace {
bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }
std::string refusal(const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { return e.what(); }
  return "";
}
Json asj(const Json& j) { return j.is_string() ? Json::parse(j.get<std::string>()) : j; }   // a JSON column as the row gives it
g::Definition def(const std::string& yaml) { return g::parse(parse_yaml(yaml, "test.yaml"), "test.yaml", yaml); }

// a toy array node: no input, one output of the class `cls`, the array and its description built from the parameters
ng::NodeSpec toy(const std::string& name, const std::string& cls, std::function<ng::Array(const Json& params)> make) {
  ng::NodeSpec n;
  n.name = name;
  n.kind = ng::Kind::array;
  n.outputs.push_back({"out", "what it makes", false});
  n.parameters.push_back({"scale", Json(1.0), "a factor of every value", "-", {}, false});
  n.parameters.push_back({"broken", Json(false), "describe nothing (a fault)", "-", {}, false});
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a toy, on arrays");
  n.description = "a toy node of the runner's output test";
  n.plan = [cls](const ng::Types&, const Json&) {
    ng::Planned p;
    ng::PortType t;
    t.cls = cls == "field" ? ng::PortClass::field : cls == "operator" ? ng::PortClass::operator_ : cls == "selection" ? ng::PortClass::selection : ng::PortClass::constant;
    t.facts["cls"] = cls;
    p.out["out"] = t;
    return p;
  };
  n.run = [make](const std::any&, const ng::Arrays&) { return ng::Arrays{{"out", make(Json::object())}}; };
  n.prepare = [make](const ng::Types&, const ng::Arrays&, const Json& params, ng::Shared&) { return std::any(params); };
  n.run = [make](const std::any& prepared, const ng::Arrays&) { return ng::Arrays{{"out", make(std::any_cast<Json>(prepared))}}; };
  return n;
}
ng::Array arr(std::int64_t rows, std::int64_t cols, double scale, Json describe) {
  ng::Array a;
  a.rows = rows; a.cols = cols;
  for (std::int64_t i = 0; i < rows * cols; ++i) a.data.push_back(scale * static_cast<double>(i + 1));
  a.meta["describe"] = std::move(describe);
  return a;
}
double scale_of(const Json& p) { return p.value("scale", 1.0); }
void register_toys() {
  auto& r = ng::registry();
  r.add(toy("toy.map", "field", [](const Json& p) { return arr(3, 1, scale_of(p), Json{{"unit", "T"}, {"precision", "float32"}}); }));
  r.add(toy("toy.spectrum", "statistic", [](const Json& p) {
    if (p.value("broken", false)) { ng::Array a = arr(3, 5, 1.0, Json::object()); a.meta.erase("describe"); return a; }
    return arr(3, 5, scale_of(p), Json{{"measure", "power_density"}, {"method", "welch"}, {"unit", "T\xc2\xb2/Hz"}, {"precision", "float32"}, {"n_members", 4},
        {"axes", Json::array({Json{{"name", "channel"}, {"size", 3}}, Json{{"name", "frequency"}, {"size", 5}, {"line", Json{{"spacing", 0.5}, {"origin", 0.0}, {"unit", "Hz"}}}}})},
        {"tiling", Json{{"unit", "samples"}, {"origin", 0}, {"width", 100}, {"hop", 50}, {"count", 4}, {"skipped", 1}, {"end", 300}}},
        {"scaling", Json{{"kind", "density"}}}, {"options", Json{{"window", "hann"}}}});
  }));
  r.add(toy("toy.operator", "operator", [](const Json& p) { return arr(3, 3, scale_of(p), Json{{"params", Json{{"k", 2}}}, {"description", "a toy operator"}}); }));
  r.add(toy("toy.partition", "selection", [](const Json&) {
    ng::Array a = arr(3, 1, 1.0, Json{{"entries", Json::array({"left", "right"})}});
    a.data = {0, 0, 1};
    return a;
  }));
  r.add(toy("toy.line", "manifold", [](const Json&) { return arr(0, 0, 1.0, Json{{"start", 0.0}, {"step", 0.25}, {"count", 6}, {"unit", "Hz"}, {"dimension", "frequency"}}); }));
  r.add(toy("toy.online", "statistic", [](const Json& p) {
    return arr(6, 1, scale_of(p), Json{{"measure", "power"}, {"method", "mean"}, {"axes", Json::array({Json{{"name", "frequency"}, {"size", 6}}})}});
  }));
}
// a toy ACCUMULATE node that declares TYPED AXES (NodeTraits::typed_axes): a statistic per channel of the recording, its axis the recording's channel manifold; with `undeclared`, an anonymous axis
ng::NodeSpec typed_toy() {
  ng::NodeSpec n;
  n.name = "toy.typed_statistic";
  n.kind = ng::Kind::accumulate;
  n.inputs.push_back({"x", "the recording", false});
  n.outputs.push_back({"out", "one value a channel", false});
  n.parameters.push_back({"undeclared", Json(false), "give an anonymous axis (a fault)", "-", {}, false});
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "a toy, over the blocks");
  n.traits.typed_axes = true;
  n.description = "a toy node with typed axes";
  n.plan = [](const ng::Types& in, const Json& p) {
    ng::Planned out;
    ng::PortType t;
    t.cls = ng::PortClass::constant;
    t.facts["cls"] = "statistic";
    t.typed = true;
    const ng::PortType& x = in.at("x");
    t.domain.factors = {p.value("undeclared", false) || x.domain.empty() ? ng::anonymous(1) : x.domain.factors.front()};
    out.out["out"] = t;
    return out;
  };
  n.start = [](const std::any&) { return std::any(0); };
  n.step = [](const std::any&, std::any st, const ng::Blocks&, std::int64_t, const ng::Place&) { return std::make_pair(ng::Blocks{}, std::move(st)); };
  n.finish = [](const std::any&, std::any) { return ng::Arrays{{"out", arr(3, 1, 1.0, Json{{"measure", "mean"}, {"method", "mean"}})}}; };
  return n;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  register_toys();
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_runner_outputs_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const int M = 3, T = 600;
  const double sfreq = 100.0;
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = static_cast<float>(std::sin(0.01 * static_cast<double>(i)));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string cid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      cid = uuid7();
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = "rest"; rec.path = task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 256}; rec.array.compress = false; rec.cols.set("unit", "T");
      sub.field(std::move(rec));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  task::Subject s = task::Subject::of(sub);
  const std::string inputs = "inputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n";
  auto read_all = [&](const task::Node& n) {
    ArrayReader r = ArrayReader::open(sub.store(), *n.path);
    const auto raw = r.read();
    return std::make_pair(r.shape(), raw);
  };

  // ONE DEFINITION, FIVE KINDS: a Field on the recording's channels, the statistic (axes: the channels, a frequency line made from the node's description; the segment partition from its tiling), an
  // operator on the channels, a partition of them, and a line made by the run and an output that lies on it
  const std::string five = "pipeline: toys\n" + inputs +
      "parameters:\n  scale: {default: 1.0, meaning: a factor, unit: '-'}\n  label: {default: demo, meaning: part of a name, unit: '-'}\n  broken: false\n"
      "presets:\n  loud: {scale: 10.0}\n  quiet: {scale: 0.5, label: soft}\n"
      "steps:\n  before:\n"
      "    - {node: toy.map, with: {scale: $scale}, out: map}\n"
      "    - {node: toy.spectrum, with: {scale: $scale, broken: $broken}, out: spectrum}\n"
      "    - {node: toy.operator, with: {scale: $scale}, out: op}\n"
      "    - {node: toy.partition, out: part}\n"
      "    - {node: toy.line, out: freq}\n"
      "    - {node: toy.online, with: {scale: $scale}, out: onfreq}\n"
      "outputs:\n"
      "  map: {kind: field, kind_name: toy map, on: recording.manifold.0, name: '{recording}_$label_map'}\n"
      "  spectrum:\n    kind: statistic\n    of: recording\n    name: '{recording}_$label_psd'\n    axes: [{name: channel, on: recording.manifold.0}, {name: frequency, line: '{recording}_toy_freq'}]\n"
      "    partition: {tiling: '$label_segments', on: recording.time}\n"
      "  op: {kind: operator, operator_kind: toy operator, from: recording.manifold.0, to: recording.manifold.0, name: '{recording}_$label_op'}\n"
      "  part: {kind: selection, type: partition, on: recording.manifold.0, name: '{recording}_$label_parts'}\n"
      "  freq: {kind: manifold, type: line, name: '{recording}_$label_line'}\n"
      "  onfreq: {kind: statistic, of: recording, name: '{recording}_$label_onfreq', axes: [{name: frequency, on: freq}]}\n";
  g::RunOptions opt;
  opt.chunks = "all";
  const g::Definition d5 = def(five);

  // all or nothing: a node that describes nothing fails at the store; nothing of the run is left (not the line, not the field made before it)
  {
    const std::string what = refusal([&] { g::run(ds, sub, d5, Json{{"broken", true}}, opt); });
    CHECK(has(what, "describe"));
    CHECK(!s.find(task::Want{.cls = task::Cls::field, .name = std::string("rest_demo_map")}));
    CHECK(!s.line_named("rest_demo_line"));
    CHECK(!s.line_named("rest_toy_freq"));
  }
  const auto r1 = g::run(ds, sub, d5, Json::object(), opt);
  CHECK_EQ(r1.size(), static_cast<std::size_t>(1));
  CHECK(!r1[0]["hit"].get<bool>());
  CHECK_EQ(r1[0]["stored"].size(), static_cast<std::size_t>(6));
  const task::Node fld = s.require(task::Want{.cls = task::Cls::field, .name = std::string("rest_demo_map")}, "the toy map");
  CHECK_EQ(fld.kind.value_or(""), std::string("toy map"));
  CHECK_EQ(*fld.manifold_id, cid);
  CHECK_EQ(fld.producer["pipeline"]["name"].get<std::string>(), std::string("toys"));
  CHECK_EQ(fld.producer["output"].get<std::string>(), std::string("map"));
  CHECK(fld.producer.contains("cache"));
  {
    const auto [shape, raw] = read_all(fld);
    CHECK_EQ(shape, std::vector<std::int64_t>{3});
    float v[3];
    std::memcpy(v, raw.data(), sizeof v);
    CHECK(v[0] == 1.0f && v[2] == 3.0f);                                   // the node's float32 precision
  }
  const task::Node st = s.require(task::Want{.cls = task::Cls::statistic, .name = std::string("rest_demo_psd")}, "the toy statistic");
  CHECK_EQ(*st.path, std::string(*s.require(task::Want{.cls = task::Cls::field, .name = std::string("rest")}, "rec").path + "_statistics/rest_demo_psd"));
  CHECK_EQ(st.fact("measure").get<std::string>(), std::string("power_density"));
  CHECK_EQ(st.fact("method").get<std::string>(), std::string("welch"));
  {
    const Json axes = asj(st.fact("axes_json"));
    CHECK_EQ(axes.size(), static_cast<std::size_t>(2));
    CHECK_EQ(axes[0]["manifold_id"].get<std::string>(), cid);              // the channel axis lies on the recording's channels
    const auto line = s.line_named("rest_toy_freq");
    CHECK(line);
    CHECK_EQ(axes[1]["manifold_id"].get<std::string>(), *line);            // the frequency line made from the node's description, named by the definition
    const task::Geometry gm = s.geometry(*line);
    CHECK_EQ(gm.axes.at(0).n_vertices, static_cast<std::int64_t>(5));
    CHECK(std::abs(*gm.axes.at(0).spacing - 0.5) < 1e-12);
    const task::Node seg = s.require(task::Want{.cls = task::Cls::selection, .name = std::string("demo_segments")}, "the segment partition");
    CHECK_EQ(st.fact("partition_selection_id").get<std::string>(), seg.id);   // the partition the statistic names: the node's tiling, samples converted to seconds
    CHECK(seg.params["tiling"]["count"] == Json(5));
    CHECK(std::abs(seg.params["tiling"]["width"].get<double>() - 1.0) < 1e-12);
    CHECK(std::abs(seg.params["tiling"]["hop"].get<double>() - 0.5) < 1e-12);
    CHECK_EQ(*st.n_members, static_cast<std::int64_t>(4));
    CHECK_EQ(st.producer["output"].get<std::string>(), std::string("spectrum"));
  }
  const task::Node op = s.require(task::Want{.cls = task::Cls::operator_, .name = std::string("rest_demo_op")}, "the toy operator");
  CHECK_EQ(*op.from_manifold_id, cid);
  CHECK_EQ(*op.to_manifold_id, cid);
  CHECK_EQ(op.kind.value_or(""), std::string("toy operator"));
  CHECK_EQ(op.params["k"].get<int>(), 2);
  const task::Node part = s.require(task::Want{.cls = task::Cls::selection, .name = std::string("rest_demo_parts")}, "the toy partition");
  CHECK_EQ(*part.manifold_id, cid);
  CHECK_EQ(s.entry_count(*part.dictionary_id), static_cast<std::int64_t>(2));
  const auto fl = s.line_named("rest_demo_line");                          // the manifold made first, and the output that lies on it
  CHECK(fl);
  const task::Node onl = s.require(task::Want{.cls = task::Cls::statistic, .name = std::string("rest_demo_onfreq")}, "the statistic on the made line");
  CHECK_EQ(asj(onl.fact("axes_json"))[0]["manifold_id"].get<std::string>(), *fl);
  CHECK_EQ(s.geometry(*fl).axes.at(0).n_vertices, static_cast<std::int64_t>(6));

  // a rerun with the same key is a hit; another key is refused, and --replace moves the old aside
  const auto r2 = g::run(ds, sub, d5, Json::object(), opt);
  CHECK(r2[0]["hit"].get<bool>());
  const std::string other = refusal([&] { g::run(ds, sub, d5, Json{{"scale", 2.0}}, opt); });
  CHECK(has(other, "already exists") && has(other, "--replace"));
  g::RunOptions rep = opt;
  rep.replace = true;
  const auto r3 = g::run(ds, sub, d5, Json{{"scale", 2.0}}, rep);
  CHECK(!r3[0]["hit"].get<bool>() && r3[0]["replaced"].get<bool>());
  {
    const auto [shape, raw] = read_all(s.require(task::Want{.cls = task::Cls::field, .name = std::string("rest_demo_map")}, "the map again"));
    float v[3];
    std::memcpy(v, raw.data(), sizeof v);
    CHECK(v[1] == 4.0f);                                                   // replaced: 2 x the node's 2
  }

  // PRESETS: defaults < the preset < the caller's own; the preset's name in the provenance and the key; an unknown one refused
  const g::Definition dp = def(five);
  CHECK_EQ(g::parameters_of(dp, Json::object())["scale"].get<double>(), 1.0);
  CHECK_EQ(g::parameters_of(dp, Json::object(), "loud")["scale"].get<double>(), 10.0);
  CHECK_EQ(g::parameters_of(dp, Json::object(), "quiet")["label"].get<std::string>(), std::string("soft"));
  CHECK_EQ(g::parameters_of(dp, Json{{"scale", 3.0}}, "loud")["scale"].get<double>(), 3.0);
  CHECK(has(refusal([&] { g::parameters_of(dp, Json::object(), "nope"); }), "not a preset of toys"));
  rep.preset = "quiet";
  const auto r4 = g::run(ds, sub, dp, Json::object(), rep);
  CHECK(!r4[0]["hit"].get<bool>());
  const task::Node soft = s.require(task::Want{.cls = task::Cls::field, .name = std::string("rest_soft_map")}, "the preset's map");
  CHECK_EQ(soft.producer["preset"].get<std::string>(), std::string("quiet"));
  CHECK_EQ(soft.producer["parameters"]["scale"].get<double>(), 0.5);
  CHECK(g::run(ds, sub, dp, Json::object(), rep)[0]["hit"].get<bool>());   // the same preset: a hit
  rep.preset = "";
  CHECK(!g::run(ds, sub, dp, Json{{"label", "soft"}, {"scale", 0.5}}, rep)[0]["hit"].get<bool>());   // the same values without the preset's name: another key

  // what a declaration must meet is refused before any data is read
  CHECK(has(refusal([&] { g::run(ds, sub, def("pipeline: t\n" + inputs + "steps:\n  before: [{node: toy.spectrum, out: map}]\noutputs: {map: {kind: field, kind_name: x, on: recording.manifold.0}}\n"), Json::object(), opt, true); }), "declared field but its steps make a statistic"));
  CHECK(has(refusal([&] { g::run(ds, sub, def("pipeline: t\n" + inputs + "steps:\n  before: [{node: toy.map, out: map}]\noutputs: {map: {kind: field, kind_name: x, on: nothing.manifold.0}}\n"), Json::object(), opt, true); }), "nothing"));
  CHECK(has(refusal([&] { def("pipeline: t\n" + inputs + "steps:\n  before: [{node: toy.operator, out: op}]\noutputs: {op: {kind: operator, from: recording.manifold.0}}\n"); }), "operator_kind"));
  CHECK(has(refusal([&] { def("pipeline: t\n" + inputs + "presets: {p: {nope: 1}}\nsteps:\n  before: [{node: toy.map, out: map}]\noutputs: {map: {kind: field, kind_name: x, on: recording.manifold.0}}\n"); }), "not a parameter"));

  // TYPED AXES: a node that declares them has every axis of its outputs refused at preparation unless it is a stored thing, and the runner writes the links when it stores
  ng::registry().add(typed_toy());
  const std::string typed = "pipeline: typed\n" + inputs + "parameters:\n  undeclared: false\n"
      "steps:\n  chunks:\n    over: recording\n    do:\n      - {node: toy.typed_statistic, in: {x: recording}, with: {undeclared: $undeclared}, out: s}\n"
      "outputs:\n  s: {kind: statistic, of: recording, name: '{recording}_typed'}\n";
  const std::string undeclared = refusal([&] { g::run(ds, sub, def(typed), Json{{"undeclared", true}}, opt, true); });
  CHECK(has(undeclared, "typed axes") && has(undeclared, "undeclared") && has(undeclared, "anonymous"));
  g::run(ds, sub, def(typed), Json::object(), opt);
  {
    const task::Node ts = s.require(task::Want{.cls = task::Cls::statistic, .name = std::string("rest_typed")}, "the typed statistic");
    const Json axes = asj(ts.fact("axes_json"));
    CHECK_EQ(axes.size(), static_cast<std::size_t>(1));
    CHECK_EQ(axes[0]["manifold_id"].get<std::string>(), cid);              // the axis the node declared: the recording's channel manifold, by id
    CHECK_EQ(axes[0]["size"].get<std::int64_t>(), static_cast<std::int64_t>(3));
  }
  std::cout << "test_runner_outputs: ok\n";
  fs::remove_all(root);
  return 0;
}
