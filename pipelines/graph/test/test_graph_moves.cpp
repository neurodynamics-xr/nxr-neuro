// nxr_graph — the shared checker's moves (spec docs/specs/2026-11-17-type-system.md, sections 4 to 6, step T2), on synthetic types: one graph per move and rule that checks, and one that is
// refused with its message. No io, no file.
//   pointwise · apply (the subsumption, D106, the gauge) · compose · restrict · measure (the partition rules) · summarize · detect · make · the cardinal rule · the closed vocabulary ·
//   the checker inside `check` (a node with a signature and typed inputs) beside the old `plan`.
#include <cstdio>
#include <functional>
#include <string>

#include "nxr/graph/graph.h"
#include "nxr/graph/signature.h"

namespace g = nxr::graph;
using g::Json;
using VT = g::ValueType;

namespace {
int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)

bool refused(const std::function<void()>& f, const std::vector<std::string>& parts) {
  try { f(); } catch (const std::invalid_argument& e) {
    bool ok = true;
    for (const auto& part : parts) if (std::string(e.what()).find(part) == std::string::npos) { ok = false; std::printf("  refused, but with: %s\n  (wanted '%s')\n", e.what(), part.c_str()); }
    return ok;
  }
  std::printf("  not refused (wanted '%s')\n", parts.empty() ? "" : parts[0].c_str());
  return false;
}

const auto sensors = g::manifold_id("sensors", "rest", g::ManifoldKind::points);
const auto time = g::time_line("rest", 100.0);
const auto cortex = g::manifold_id("cortex", "kernel", g::ManifoldKind::surface);

g::Domain dom(std::vector<g::ManifoldRef> f) { g::Domain d; d.factors = std::move(f); return d; }

g::PortType field_t(std::vector<g::ManifoldRef> f, VT v = VT::real, const std::string& quantity = "recording", bool nonlinear = false) {
  g::PortType t;
  t.typed = true; t.cls = g::PortClass::field; t.domain = dom(std::move(f)); t.value = v; t.quantity = quantity; t.nonlinear = nonlinear;
  t.axis = g::Axis::samples; t.sfreq = 100.0;
  return t;
}
g::PortType kernel_t(const g::ManifoldRef& from, const g::ManifoldRef& to, VT v = VT::vector3) {
  g::PortType t;
  t.typed = true; t.cls = g::PortClass::operator_; t.domain = dom({from}); t.target = dom({to}); t.value = v; t.quantity = "inverse kernel";
  return t;
}
g::PortType selection_t(std::vector<g::ManifoldRef> on, const std::string& type, bool partition, const std::string& id) {
  g::PortType t;
  t.typed = true; t.cls = g::PortClass::selection; t.domain = dom(std::move(on)); t.quantity = type; t.sel = g::SelectionShape{type, partition, id, ""};
  return t;
}

// a use of the built-in node `node` on typed inputs (by port); the outputs get the names `outs` in the signature's order
g::Types use(const std::string& node, const g::Types& in, const Json& params = Json::object(), bool chunk = true, std::vector<std::string> outs = {"out"}) {
  const g::Signature* s = g::builtin_signature(node);
  if (!s) { std::printf("  no signature for %s\n", node.c_str()); ++failures; return {}; }
  g::TypeUse tu;
  tu.label = "chunks[0] " + node;
  tu.chunk_node = chunk;
  tu.in = in;
  for (const auto& [port, _] : in) tu.in_names[port] = port;
  for (std::size_t i = 0; i < s->outputs.size(); ++i) tu.out_names[s->outputs[i].first] = i < outs.size() ? outs[i] : outs[0] + "." + s->outputs[i].first;
  return g::type_use(node, *s, tu, params);
}

void test_pointwise() {
  const auto rec = field_t({sensors, time});
  const auto z = use("analytic", {{"x", rec}}, Json{{"band", Json::array({8.0, 16.0})}}).at("y");
  CHECK(z.value == VT::complex && z.res.band && (*z.res.band)[0] == 8 && g::same(z.domain, rec.domain) && z.typed);
  const auto env = use("envelope", {{"x", rec}}).at("y");
  CHECK(env.nonlinear && env.quantity == "envelope");
  CHECK(refused([&] { use("analytic", {{"x", z}}); }, {"port `x`", "is complex", "analytic signal is taken of a real field", "chunks[0] analytic -> out"}));
  // a toy node declared pointwise that drops a factor: the cardinal rule refuses it, whatever the node's own code does
  g::Signature bad = g::sig(g::Move::pointwise).in("x", g::field({g::var("M"), g::time_var("T")})).out("y", g::like("x").on({g::var("M")}));
  g::TypeUse tu; tu.label = "chunks[1] toy.drop"; tu.in = {{"x", rec}}; tu.in_names = {{"x", "rec"}}; tu.out_names = {{"y", "y"}};
  CHECK(refused([&] { g::type_use("toy.drop", bad, tu, Json::object()); }, {"is declared pointwise", "only `measure` removes a factor", "sensors(rest)"}));
}

void test_apply() {
  const auto rec = field_t({sensors, time}, VT::complex);
  const auto ks = g::restricted(sensors, "channels(kernel)"), cs = g::restricted(sensors, "channels(cross_spectrum)");
  const auto k = kernel_t(ks, cortex);
  // the kernel reads its channels of the whole recording (the subsumption): the output lies on the kernel's target, complex vector3 from a complex field
  const auto y = use("apply_kernel", {{"x", rec}, {"kernel", k}}).at("y");
  CHECK(g::print(y.domain) == "cortex(kernel) × time(rest)" && y.value == VT::vector3 && y.complex_parts && y.quantity == "current");
  // a field on the kernel's own channels
  CHECK(use("apply_kernel", {{"x", field_t({ks, time}, VT::complex)}, {"kernel", k}}).count("y") == 1);
  // another selection of the same manifold: refused naming both
  CHECK(refused([&] { use("apply_kernel", {{"x", field_t({cs, time}, VT::complex)}, {"kernel", k}}); },
                {"port `kernel`", "sensors(rest)|channels(kernel)", "sensors(rest)|channels(cross_spectrum)", "--channels-of"}));
  // a kernel from another manifold: no implicit move
  CHECK(refused([&] { use("apply_kernel", {{"x", field_t({cortex, time})}, {"kernel", k}}); }, {"port `kernel`"}));
  // D106
  const auto power = use("power", {{"x", rec}}).at("y");
  CHECK(power.nonlinear && power.quantity == "power");
  CHECK(refused([&] { use("apply_kernel", {{"x", power}, {"kernel", k}}); }, {"port `x`", "is non-linear", "D106", "apply it first"}));
  // the output's value: a free kernel and a fixed one; a form makes a real quantity; glued derives the surface
  CHECK(use("apply_kernel", {{"x", rec}, {"kernel", kernel_t(ks, cortex, VT::real)}}).at("y").value == VT::complex);
  const auto curl = use("apply_kernel", {{"x", rec}, {"kernel", k}}, Json{{"form", "curl_tan"}, {"glued", true}}).at("y");
  CHECK(curl.quantity == "curl_tan" && curl.value == VT::complex && g::print(curl.domain.factors[0]) == "glued(cortex(kernel))");
  // tiles: the ladder of the kernel's target, with its level; the ladder of the sensors is refused
  const auto tiles = selection_t({cortex}, "ladder", false, "tiles");
  const auto y5 = use("apply_kernel", {{"x", rec}, {"kernel", k}, {"tiles", tiles}}, Json{{"level", 5}}).at("y");
  CHECK(y5.res.ladder == "tiles" && y5.res.levels == std::vector<int>{5});
  CHECK(refused([&] { use("apply_kernel", {{"x", rec}, {"kernel", k}, {"tiles", selection_t({sensors}, "ladder", false, "tiles")}}, Json{{"level", 5}}); }, {"port `tiles`", "sensors(rest) is not cortex(kernel)"}));
  CHECK(refused([&] { use("apply_kernel", {{"x", rec}, {"kernel", k}, {"tiles", tiles}}); }, {"`tiles` comes with `level` or `levels`"}));
  // frames: the interpolation maps the time line onto the frame tiling's members; a non-linear x is refused
  const auto tiling = selection_t({time}, "tiling", true, "tiling");
  const auto fr = use("frames", {{"x", rec}, {"tiling", tiling}}).at("y");
  CHECK(g::print(fr.domain) == "sensors(rest) × members(tiling)" && fr.axis == g::Axis::frames && fr.domain.factors.back()->role == g::Role::frames);
  CHECK(refused([&] { use("frames", {{"x", power}, {"tiling", tiling}}); }, {"port `x`", "is non-linear"}));
  CHECK(refused([&] { use("frames", {{"x", rec}, {"tiling", selection_t({time}, "spans", true, "bad")}}); }, {"not a frame tiling"}));
  CHECK(refused([&] { use("frames", {{"x", rec}, {"tiling", selection_t({g::time_line("other")}, "tiling", true, "t2")}}); }, {"port `tiling`", "time(other) is not time(rest)"}));
}

void test_gauge() {
  const auto ks = g::restricted(sensors, "channels(kernel)");
  g::PortType k = kernel_t(ks, cortex);
  k.spin = 1; k.gauge = "gc/vertex:cortex";
  g::PortType x = field_t({ks, time}, VT::complex);
  CHECK(use("apply_kernel", {{"x", x}, {"kernel", k}}).at("y").spin == 1);           // a scalar field through a tangent operator: its output is spin 1 in the operator's gauge
  g::PortType t = field_t({cortex, time}, VT::complex);
  g::PortType kt = kernel_t(cortex, cortex);
  kt.spin = 1; kt.gauge = "gc/vertex:glued";
  t.spin = 1; t.gauge = "gc/vertex:cortex";
  CHECK(refused([&] { use("apply_kernel", {{"x", t}, {"kernel", kt}}); }, {"spin 1 in gauge gc/vertex:cortex", "gc/vertex:glued", "frame.rotate"}));
  kt.gauge = "gc/vertex:cortex";
  CHECK(use("apply_kernel", {{"x", t}, {"kernel", kt}}).count("y") == 1);
  t.gauge.clear();
  CHECK(refused([&] { use("apply_kernel", {{"x", t}, {"kernel", kt}}); }, {"records no gauge"}));
  // a vector3 value is gauge-free
  g::PortType v = field_t({cortex, time}, VT::vector3, "phase_current");
  CHECK(v.spin == 0 && v.gauge.empty());
}

void test_compose_restrict_make() {
  // compose: the ends meet
  g::Signature comp = g::sig(g::Move::compose).in("f", g::operator_(g::var("A"), g::var("B"))).in("g", g::operator_(g::var("B"), g::var("C"))).out("h", g::make_of(g::PortClass::operator_));
  comp.outputs[0].second.on({g::var("A")}).onto({g::var("C")});
  g::TypeUse tu; tu.label = "after[0] toy.compose"; tu.in_names = {{"f", "f"}, {"g", "g"}}; tu.out_names = {{"h", "h"}};
  tu.in = {{"f", kernel_t(sensors, cortex)}, {"g", kernel_t(cortex, g::manifold_id("sphere", "s"))}};
  const g::Types ok = g::type_use("toy.compose", comp, tu, Json::object());
  CHECK(g::print(ok.at("h").domain) == "sensors(rest)" && g::print(ok.at("h").target) == "sphere(s)");
  tu.in = {{"f", kernel_t(sensors, cortex)}, {"g", kernel_t(g::manifold_id("sensors", "other"), g::manifold_id("sphere", "s"))}};
  CHECK(refused([&] { g::type_use("toy.compose", comp, tu, Json::object()); }, {"port `g`", "sensors(other) is not cortex(kernel)"}));
  // restrict: at_level narrows the stack of levels; the domain is unchanged
  g::PortType stack = field_t({cortex, time});
  stack.res.ladder = "tiles"; stack.res.levels = {4, 5, 6};
  const auto one = use("at_level", {{"x", stack}}, Json{{"level", 5}}).at("y");
  CHECK(one.res.levels == std::vector<int>{5} && g::same(one.domain, stack.domain));
  CHECK(refused([&] { use("at_level", {{"x", one}}, Json{{"level", 5}}); }, {"is not a multi-level field"}));
  // make: no field input
  g::Signature mk = g::sig(g::Move::make).in("x", g::field({g::var("M")})).out("t", g::make_of(g::PortClass::selection));
  CHECK(refused([&] { mk.validate("toy.make"); }, {"a `make` node has no field input"}));
}

void test_measure() {
  const auto mag = [&](const std::string& quantity = "amplitude") {
    g::PortType t = field_t({cortex, time}, VT::real, quantity, true);
    t.res.band = std::array<double, 2>{8.0, 16.0};
    t.res.ladder = "tiles"; t.res.levels = {5};
    return t;
  };
  // a joint partition of the surface x the time line, at level 5 and 8-16 Hz
  g::PortType cells = selection_t({cortex, time}, "set", true, "cells");
  cells.res = mag().res;
  const auto m = use("measure", {{"x", mag()}, {"over", cells}}, Json{{"measure", "amplitude"}}, false).at("y");
  CHECK(g::print(m) == "measurement<members(cells); amplitude; band 8-16 Hz; level tiles@5>");
  CHECK(m.domain.factors[0]->label == "joint" && m.domain.factors[0]->of && g::same(m.domain.factors[0]->of, time));
  // the level IS the band: another level or another band is refused, naming both
  g::PortType x4 = mag(); x4.res.levels = {4};
  CHECK(refused([&] { use("measure", {{"x", x4}, {"over", cells}}, Json{{"measure", "amplitude"}}, false); },
                {"port `over`", "the joint partition `cells` is at level 5 of `tiles`", "x is limited to level 4 of `tiles`", "the level IS the band"}));
  g::PortType xb = mag(); xb.res.band = std::array<double, 2>{14.0, 30.0};
  CHECK(refused([&] { use("measure", {{"x", xb}, {"over", cells}}, Json{{"measure", "amplitude"}}, false); }, {"band 8-16 Hz", "band-limited to 14-30 Hz"}));
  // a partition of other manifolds
  CHECK(refused([&] { use("measure", {{"x", field_t({sensors, time}, VT::real, "amplitude")}, {"over", cells}}, Json{{"measure", "amplitude"}}, false); }, {"not factors of x"}));
  // the measure must fit the quantity; a complex x is refused
  CHECK(refused([&] { use("measure", {{"x", mag("power")}, {"over", cells}}, Json{{"measure", "amplitude"}}, false); }, {"the measure amplitude is a measure of amplitude or envelope"}));
  CHECK(refused([&] { use("measure", {{"x", field_t({cortex, time}, VT::complex)}, {"over", cells}}, Json{{"measure", "mean"}}, false); }, {"is complex", "this node reads real or vector3"}));
  // any selection on factors of x is measured (no property needed): a selection of one member is one region or time range
  CHECK(g::print(use("measure", {{"x", mag()}, {"over", selection_t({cortex, time}, "set", false, "loose")}}, Json{{"measure", "amplitude"}}, false).at("y").domain).find("members(loose)") != std::string::npos);
  // a time ladder: the level is chosen by the band; a field without a band is refused; the output's member set is named after the level
  g::PortType env = field_t({sensors, time}, VT::real, "envelope", true);
  env.res.band = std::array<double, 2>{8.0, 16.0};
  const auto ladder = selection_t({time}, "ladder", false, "ladder");
  const auto am = use("measure", {{"x", env}, {"over", ladder}}, Json{{"measure", "amplitude"}}, false).at("y");
  CHECK(g::print(am.domain) == "sensors(rest) × members(ladder@8-16)" && am.domain.factors.back()->role == g::Role::time_tiles);
  g::PortType nob = env; nob.res.band.reset();
  CHECK(refused([&] { use("measure", {{"x", nob}, {"over", ladder}}, Json{{"measure", "amplitude"}}, false); }, {"carries no band", "the level IS the band"}));
  // a chunk node measuring a partition of the stream needs an accumulate node; no partition: the run's span, also an accumulate node's
  CHECK(refused([&] { use("measure", {{"x", env}, {"over", selection_t({time}, "tiling", true, "tiles")}}, Json{{"measure", "mean"}}, true); }, {"needs an accumulate node"}));
  CHECK(refused([&] { use("measure", {{"x", env}}, Json{{"measure", "mean"}}, true); }, {"needs an accumulate node"}));
  CHECK(g::print(use("measure", {{"x", env}}, Json{{"measure", "mean"}}, false).at("y").domain) == "sensors(rest)");
  // the cardinal rule: only measure removes a factor, so a measurement has the member set in the place of the factors it partitions
  CHECK(am.cls == g::PortClass::measurement);
}

void test_summarize_detect() {
  g::PortType env = field_t({sensors, time}, VT::real, "envelope", true);
  env.res.band = std::array<double, 2>{8.0, 16.0};
  const auto ladder = selection_t({time}, "ladder", false, "ladder");
  const auto amp = use("measure", {{"x", env}, {"over", ladder}}, Json{{"measure", "amplitude"}}, false).at("y");
  // summarize: a statistic that remembers what it is of; the member set leaves the axes
  const auto st = use("summarize", {{"x", amp}}, Json{{"summaries", Json::array({"median", "mad"})}}, false).at("y");
  CHECK(g::print(st.domain) == "sensors(rest)" && st.of && st.of->quantity == "amplitude" && st.of->res.band && g::same(st.of->on, amp.domain));
  // a field is measured first
  CHECK(refused([&] { use("summarize", {{"x", env}}, Json::object(), false); }, {"port `x`", "is a field; a measurement is wanted", "measure the field over a partition first"}));
  // detect: events on a measurement over a time partition's members, a selection on the time line with an identity and what it was detected over
  const auto sp = use("detect_spans", {{"x", amp}, {"baseline", st}}, Json::object(), false, {"bursts"}).at("spans");
  CHECK(g::print(sp) == "selection<spans on time(rest); detected over ladder@8-16; as bursts; band 8-16 Hz>");
  CHECK(refused([&] { use("detect_spans", {{"x", env}}, Json::object(), false); }, {"is a field; a measurement is wanted", "it over the constant-Q level first"}));
  g::PortType other = st; other.of->quantity = "power";
  CHECK(refused([&] { use("detect_spans", {{"x", amp}, {"baseline", other}}, Json::object(), false); }, {"port `baseline`", "is not a statistic of x"}));
  // a measurement of events detected on another grid's members is refused (events themselves may overlap: no disjointness is asked)
  g::PortType amp2 = amp;
  amp2.domain.factors.back() = g::members("ladder@14-30", time, g::Role::time_tiles);
  amp2.quantity = "amplitude";
  CHECK(refused([&] { use("measure_events", {{"x", amp2}, {"events", sp}}, Json{{"measure", "amplitude"}}, false); }, {"port `events`", "were detected on the members of `ladder@8-16`", "`ladder@14-30`"}));
  const auto roll = use("measure_events", {{"x", amp}, {"events", sp}}, Json{{"measure", "amplitude"}}, false).at("y");
  CHECK(g::print(roll.domain) == "sensors(rest) × members(bursts)");
}

void test_vocabulary_and_outputs() {
  g::Signature s = g::sig(g::Move::pointwise).in("x", g::field({g::var("M")})).out("y", g::like("x").with_quantity("amplitudee"));
  CHECK(refused([&] { s.validate("toy.q"); }, {"toy.q", "amplitudee", "closed vocabulary"}));
  g::Signature dup = g::sig(g::Move::pointwise).in("x", g::field({g::var("M")})).out("y", g::like("z"));
  CHECK(refused([&] { dup.validate("toy.like"); }, {"is like 'z'"}));
  // the outputs' declared kinds against what the graph makes
  g::Types values;
  values["a"] = field_t({sensors, time});
  g::PortType st; st.typed = true; st.cls = g::PortClass::statistic; st.domain = dom({sensors});
  values["b"] = st;
  g::check_output_kinds(values, {{"a", "field"}, {"b", "statistic"}});
  CHECK(refused([&] { g::check_output_kinds(values, {{"b", "measurement"}}); }, {"outputs: b is declared a measurement", "statistic<"}));
}

// ── the checker inside `check`: a node with a signature and typed inputs, beside its old plan; untyped inputs are left to the plan
g::NodeSpec toy_analytic() {
  g::NodeSpec n;
  n.name = "toy.analytic";
  n.kind = g::Kind::chunk;
  n.inputs = {{"x", "", false}};
  n.outputs = {{"y", "", false}};
  n.traits = g::node_traits(g::TimeBehaviour::pointwise, g::Precision::float32, g::OutputPrecision::follows_input, "each column on its own");
  n.plan = [](const g::Types& in, const Json&) { g::Planned p; g::PortType t = in.at("x"); t.value = VT::complex; p.out["y"] = t; return p; };
  n.chunk = [](const std::any&, const g::Blocks& in, std::int64_t, const g::Place&) { return in; };
  n.signature = *g::builtin_signature("analytic");
  return n;
}

void test_in_check() {
  g::Registry reg;
  reg.add(toy_analytic());
  g::Graph gr = g::graph_from_json(Json{{"name", "t"}, {"inputs", {"rec"}}, {"outputs", {"z2"}}, {"chunks", Json::array({Json{{"node", "toy.analytic"}, {"in", {{"x", "rec"}}}, {"out", "z"}},
                                                                                                                              Json{{"node", "toy.analytic"}, {"in", {{"x", "z"}}}, {"out", "z2"}}})}});
  // typed: the second use is refused by the signature (z is complex), with the printed types
  g::PortType rec = field_t({sensors, time});
  CHECK(refused([&] { g::check(gr, {{"rec", rec}}, Json::object(), reg); }, {"chunks[1] toy.analytic -> z2", "port `x`", "is complex", "from chunks[0] toy.analytic", "field<sensors(rest) × time(rest); complex"}));
  // a single use checks, and the planned output carries the typed members
  g::Graph one = g::graph_from_json(Json{{"name", "t"}, {"inputs", {"rec"}}, {"outputs", {"z"}}, {"chunks", Json::array({Json{{"node", "toy.analytic"}, {"in", {{"x", "rec"}}}, {"out", "z"}}})}});
  const g::Checked c = g::check(one, {{"rec", rec}}, Json::object(), reg);
  CHECK(c.values.at("z").typed && c.values.at("z").value == VT::complex && g::print(c.values.at("z").domain) == "sensors(rest) × time(rest)");
  // an untyped input: the old plan alone (C1 unchanged), the second use is not refused by types
  g::PortType old;
  old.value = VT::real; old.rows = 3; old.axis = g::Axis::samples; old.sfreq = 100.0;
  CHECK(!g::check(gr, {{"rec", old}}, Json::object(), reg).values.at("z2").typed);
  // the registry keeps a built-in signature for a node of that name only when the ports fit
  g::NodeSpec n = toy_analytic();
  n.name = "analytic"; n.signature.reset(); n.inputs = {{"x", "", false}, {"extra", "", true}};
  g::Registry r2;
  r2.add(n);
  CHECK(r2.unmatched_signatures().size() == 1 && !r2.at("analytic").signature);
  n.inputs = {{"x", "", false}};
  g::Registry r3;
  r3.add(n);
  CHECK(r3.unmatched_signatures().empty() && r3.at("analytic").signature);
}

void test_builtin_table() {
  const auto names = g::builtin_signature_names();
  CHECK(names.size() == 31);
  for (const auto& n : names) {
    const g::Signature* s = g::builtin_signature(n);
    CHECK(s && !s->outputs.empty());
  }
}
// T5: the coordinate system is part of a manifold's type; no transition is inserted: the checker refuses and names `positions.transition`
void test_coordinates() {
  const auto device = g::manifold_id("sensors", "run1", g::ManifoldKind::points, g::Role::none, g::CoordinateSystem{"head", "device:run1", "m"});
  const auto head = g::manifold_id("sensors", "run1", g::ManifoldKind::points, g::Role::none, g::CoordinateSystem{"head", "scs", "m"});
  const auto surface = g::manifold_id("cortex", "kernel", g::ManifoldKind::surface, g::Role::none, g::CoordinateSystem{"head", "scs", "m"});
  CHECK(!g::same(device, head) && g::same(head, head));
  g::PortType tr; tr.typed = true; tr.cls = g::PortClass::operator_; tr.quantity = "transition";
  tr.transition = std::make_pair(g::CoordinateSystem{"head", "device:run1", "m"}, g::CoordinateSystem{"head", "scs", "m"});
  g::PortType pos; pos.typed = true; pos.cls = g::PortClass::manifold; pos.domain = dom({device});
  const auto moved = use("positions.transition", {{"positions", pos}, {"transition", tr}}, Json::object(), false, {"moved"}).at("moved");
  CHECK(moved.domain.factors[0]->cs && moved.domain.factors[0]->cs->name == "scs" && g::same(moved.domain.factors[0], head));
  g::PortType wrong = pos; wrong.domain = dom({head});
  CHECK(refused([&] { use("positions.transition", {{"positions", wrong}, {"transition", tr}}, Json::object(), false, {"moved"}); }, {"port `transition`", "starts in", "device:run1"}));
  // a kernel built on the head-frame sensors reads a recording whose sensors are still in the device frame: refused, the transition named, none inserted
  const auto k = kernel_t(g::restricted(head, "channels(kernel)"), surface);
  CHECK(refused([&] { use("apply_kernel", {{"x", field_t({device, time}, VT::complex)}, {"kernel", k}}); }, {"port `kernel`", "another coordinate system", "head/device:run1", "positions.transition"}));
  // a node that reads positions: its ports in one system, or in the one it names
  g::Signature s = g::sig(g::Move::make).in("channels", g::selection_any()).in("surface", g::selection_any()).out("g", g::make_of(g::PortClass::operator_)).positions_share({"channels", "surface"}).positions_in("surface", "scs");
  g::TypeUse tu; tu.label = "chunks[0] toy.gain"; tu.in_names = {{"channels", "c"}, {"surface", "s"}}; tu.out_names = {{"g", "g"}};
  tu.in = {{"channels", selection_t({device}, "indices", false, "c")}, {"surface", selection_t({surface}, "indices", false, "s")}};
  CHECK(refused([&] { g::type_use("toy.gain", s, tu, Json::object()); }, {"port `surface`", "head/scs", "head/device:run1", "positions.transition", "device:run1 -> scs"}));
  tu.in.at("channels") = selection_t({head}, "indices", false, "c");
  CHECK(g::type_use("toy.gain", s, tu, Json::object()).count("g") == 1);
}

}  // namespace

int main() {
  test_coordinates();
  test_pointwise();
  test_apply();
  test_gauge();
  test_compose_restrict_make();
  test_measure();
  test_summarize_detect();
  test_vocabulary_and_outputs();
  test_in_check();
  test_builtin_table();
  if (failures) { std::printf("test_graph_moves: %d failure(s)\n", failures); return 1; }
  std::printf("test_graph_moves: ok\n");
  return 0;
}
