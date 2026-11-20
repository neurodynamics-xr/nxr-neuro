// nxr-graph — `compute`: the types of arrays handed to a node or a graph, the same check as `run`, and `apply` (compute.h; spec docs/specs/2026-11-17-type-system.md, section 10).
#include "nxr/graph/compute.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace nxr::graph {

namespace {

const char* kinds_text = "field, operator, partition, constant";

ValueType parse_value(const std::string& s, bool& complex_parts) {
  complex_parts = false;
  if (s == "real") return ValueType::real;
  if (s == "complex") return ValueType::complex;
  if (s == "vector3") return ValueType::vector3;
  if (s == "complex vector3") { complex_parts = true; return ValueType::vector3; }
  if (s == "quaternion") return ValueType::quaternion;
  if (s == "hermitian3") return ValueType::hermitian3;
  if (s == "integer") return ValueType::integer;
  refuse("value '" + s + "' is not one of real, complex, vector3, complex vector3, quaternion, hermitian3, integer");
}
int components_of(ValueType v, bool complex_parts) {
  const int c = v == ValueType::vector3 ? 3 : v == ValueType::quaternion ? 4 : v == ValueType::hermitian3 ? 6 : v == ValueType::complex ? 2 : 1;
  return complex_parts ? c * 2 : c;
}

ManifoldRef renamed(const ManifoldRef& m, const std::string& prefix) {
  if (!m) return m;
  Manifold x = *m;
  if (x.form == Manifold::Form::anonymous) x.key = prefix + x.key;
  if (x.of) x.of = renamed(x.of, prefix);
  return std::make_shared<const Manifold>(std::move(x));
}
void rename_domain(Domain& d, const std::string& prefix) { for (auto& f : d.factors) f = renamed(f, prefix); }

void require_integer_members(const Array& a, const std::string& name) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double v = a.at(i);
    if (v != static_cast<double>(static_cast<std::int64_t>(v)) || v < -1)
      refuse("partition '" + name + "' must hold an integer member index per element (-1: none); it holds " + std::to_string(v));
  }
}

}  // namespace

ManifoldRef ArrayTyper::factor(const std::string& name, const std::string& label, std::int64_t n, ManifoldKind kind, bool time_axis, std::optional<double> spacing,
                               const ArrayMeta& m) {
  const auto cs = m.coordinate_systems.find(label);
  if (!label.empty()) {
    const auto it = labelled_.find(label);
    if (it != labelled_.end()) {
      const Manifold& have = *it->second;
      if (have.n != n)
        refuse("array '" + name + "': the label '" + label + "' names a manifold of " + std::to_string(have.n) + " elements (as an earlier array), and this array has " + std::to_string(n) +
               " along it; arrays that share a label lie on one manifold");
      if (cs != m.coordinate_systems.end() && have.cs && !(*have.cs == cs->second))
        refuse("array '" + name + "': the label '" + label + "' is in the coordinate system " + have.cs->space + "/" + have.cs->name + " (an earlier array), and " + cs->second.space + "/" + cs->second.name + " here");
      return it->second;
    }
  }
  Manifold x = *anonymous(next_++, kind);
  x.label = label.empty() ? (time_axis ? "time" : "array") : label;
  x.n = n;
  x.spacing = spacing;
  if (time_axis) { x.cs = CoordinateSystem{"time", "clock", "s"}; x.topology = Topology::path; }   // the samples of an array's time axis: a regular path in seconds
  if (cs != m.coordinate_systems.end()) x.cs = cs->second;
  ManifoldRef r = std::make_shared<const Manifold>(std::move(x));
  if (!label.empty()) labelled_[label] = r;
  return r;
}

PortType ArrayTyper::type_of(const std::string& name, const Array& a, const ArrayMeta& m) {
  PortType t;
  if (m.cls != "field" && m.cls != "operator" && m.cls != "partition" && m.cls != "constant")
    refuse("array '" + name + "': class '" + m.cls + "' is not one of " + kinds_text);
  t.unit = m.unit;
  t.rows = a.rows;
  bool complex_parts = false;
  ValueType v = ValueType::real;
  if (!m.value.empty()) v = parse_value(m.value, complex_parts);
  else if (a.comps == 2) v = ValueType::complex;
  else if (a.comps == 3) v = ValueType::vector3;
  else if (a.comps == 4) v = ValueType::quaternion;
  else if (a.comps == 6) { v = ValueType::vector3; complex_parts = true; }
  else if (a.comps != 1) refuse("array '" + name + "' has " + std::to_string(a.comps) + " components; say its value type (value = ...)");
  t.value = v;
  t.complex_parts = complex_parts;
  t.components = components_of(v, complex_parts);
  if (m.cls == "constant") return t;          // untyped: `typed` stays false, the nodes' plans check it
  t.typed = true;
  if (m.spin != 0 && v != ValueType::vector3 && m.gauge.empty())
    refuse("array '" + name + "' is a tangent array (spin " + std::to_string(m.spin) + ") without its gauge: its values are only meaningful in the gauge they were made in (say gauge = ...); "
           "a vector3 value needs none");
  t.spin = m.spin;
  t.gauge = m.gauge;

  if (m.cls == "operator") {
    t.cls = PortClass::operator_;
    const std::int64_t n_to = v == ValueType::vector3 ? a.rows / 3 : a.rows;
    if (v == ValueType::vector3 && a.rows % 3 != 0) refuse("operator '" + name + "' is vector3 and has " + std::to_string(a.rows) + " rows, not a multiple of 3");
    t.domain.factors = {factor(name, m.from, a.cols, ManifoldKind::points, false, std::nullopt, m)};
    t.target.factors = {factor(name, m.to, n_to, ManifoldKind::points, false, std::nullopt, m)};
    t.quantity = m.quantity;
    t.on = m.from; t.to = m.to;
    return t;
  }
  if (m.cls == "partition") {
    if (m.on.size() > 1) refuse("partition '" + name + "' partitions one factor: `on` names one label");
    require_integer_members(a, name);
    t.cls = PortClass::selection;
    t.value = ValueType::integer;
    t.domain.factors = {factor(name, m.on.empty() ? "" : m.on[0], static_cast<std::int64_t>(a.size()), ManifoldKind::points, false, std::nullopt, m)};
    t.sel = SelectionShape{"set", true, name, ""};
    t.quantity = "set";
    return t;
  }
  // a field
  t.cls = PortClass::field;
  const bool streamed = m.sfreq.has_value();
  if (streamed && *m.sfreq <= 0) refuse("array '" + name + "': the sample rate must be positive");
  if (m.on.size() > 2) refuse("array '" + name + "' has two axes (elements, samples): `on` names at most two labels");
  const std::string l0 = m.on.size() > 0 ? m.on[0] : "", l1 = m.on.size() > 1 ? m.on[1] : "";
  t.domain.factors = {factor(name, l0, a.rows, ManifoldKind::points, false, std::nullopt, m)};
  if (streamed) {
    t.domain.factors.push_back(factor(name, l1.empty() ? "time" : l1, a.cols, ManifoldKind::line, true, 1.0 / *m.sfreq, m));
    t.axis = Axis::samples;
    t.sfreq = *m.sfreq;
  } else if (a.cols > 1 || !l1.empty()) {
    t.domain.factors.push_back(factor(name, l1, a.cols, ManifoldKind::points, false, std::nullopt, m));
  }
  t.quantity = !m.quantity.empty() ? m.quantity : (streamed ? "recording" : "");
  if (!t.quantity.empty() && !is_quantity(t.quantity)) refuse("array '" + name + "': the quantity '" + t.quantity + "' is not in the closed vocabulary (docs/reference.md)");
  t.on = l0;
  return t;
}

Graph graph_of_node(const NodeSpec& node, const std::vector<std::string>& ports, const Json& with) {
  Graph g;
  g.name = node.name;
  for (const auto& p : ports)
    if (std::none_of(node.inputs.begin(), node.inputs.end(), [&](const PortSpec& q) { return q.name == p; })) refuse(node.name + " has no input port '" + p + "'");
  Use u;
  u.node = node.name;
  u.with = with;
  for (const auto& p : node.inputs) {
    if (!ports.empty() && std::find(ports.begin(), ports.end(), p.name) == ports.end()) {
      if (!p.optional) refuse(node.name + ": the required input port '" + p.name + "' is not given");
      continue;
    }
    g.inputs.push_back(p.name);
    u.in[p.name] = p.name;
  }
  for (const auto& p : node.outputs) {
    if (u.in.count(p.name)) refuse(node.name + ": the port '" + p.name + "' is both an input and an output; a graph of it needs distinct value names");
    u.out.push_back(p.name);
    g.outputs.push_back(p.name);
  }
  (node.kind == Kind::array ? g.before : g.chunks).push_back(std::move(u));
  return g;
}

Checked check_arrays(const Graph& g, const std::map<std::string, ComputeInput>& inputs, const Json& params, const Registry& registry, const DispatchContext& ctx) {
  ArrayTyper typer;
  Types types;
  for (const auto& name : g.inputs) {
    const auto it = inputs.find(name);
    if (it == inputs.end()) refuse("compute: the input '" + name + "' was not handed (the graph's inputs: " + [&] { std::string s; for (const auto& n : g.inputs) s += (s.empty() ? "" : ", ") + n; return s; }() + ")");
    types[name] = it->second.type ? *it->second.type : typer.type_of(name, it->second.array, it->second.meta);
  }
  for (const auto& [name, _] : inputs)
    if (std::find(g.inputs.begin(), g.inputs.end(), name) == g.inputs.end()) refuse("compute: '" + name + "' is not an input of the graph");
  return check(g, types, params, registry, ctx);
}

ComputeResult compute(const Graph& g, const std::map<std::string, ComputeInput>& inputs, const Json& params, const Registry& registry, const DispatchContext& ctx, const Blocking& blocking) {
  const Checked checked = check_arrays(g, inputs, params, registry, ctx);       // before any array is read
  Arrays in;
  for (const auto& [name, i] : inputs) in[name] = i.array;
  ComputeResult r;
  r.arrays = apply(checked, in, ctx, blocking);
  for (const auto& o : checked.outputs) r.types[o] = checked.values.at(o);
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  char id[32];
  std::snprintf(id, sizeof id, "%llx", static_cast<unsigned long long>(now));
  r.sidecar = sidecar_of(r.types, id);
  return r;
}

ComputeResult compute_node(const std::string& node, const std::map<std::string, ComputeInput>& inputs, const Json& params, const Registry& registry, const DispatchContext& ctx,
                           const Blocking& blocking) {
  const NodeSpec& spec = registry.at(node);
  std::vector<std::string> ports;
  for (const auto& [name, _] : inputs) ports.push_back(name);
  return compute(graph_of_node(spec, ports, params), inputs, Json::object(), registry, ctx, blocking);
}

Json sidecar_of(const Types& types, const std::string& run_id) {
  Json j = Json::object();
  j["nxr_types"] = 1;
  j["run"] = run_id;
  j["types"] = Json::object();
  j["axes"] = Json::object();
  for (const auto& [name, t] : types) {
    j["types"][name] = print(t);
    j["axes"][name] = Json{{"axis", static_cast<int>(t.axis)}, {"sfreq", t.sfreq ? Json(*t.sfreq) : Json()}, {"rows", t.rows}};   // what the printed form leaves out
  }
  return j;
}

PortType type_from_sidecar(const Json& sidecar, const std::string& name) {
  if (!sidecar.contains("nxr_types") || !sidecar.contains("types") || !sidecar["types"].contains(name))
    refuse("the sidecar holds no type for '" + name + "'");
  PortType t = parse_type(sidecar["types"][name].get<std::string>());
  const std::string prefix = sidecar.value("run", std::string()) + ":";
  rename_domain(t.domain, prefix);
  rename_domain(t.target, prefix);
  if (t.of) rename_domain(t.of->on, prefix);
  t.typed = true;
  if (sidecar.contains("axes") && sidecar["axes"].contains(name)) {
    const Json& a = sidecar["axes"][name];
    t.axis = static_cast<Axis>(a.value("axis", 0));
    if (a.contains("sfreq") && a["sfreq"].is_number()) t.sfreq = a["sfreq"].get<double>();
    t.rows = a.value("rows", static_cast<std::int64_t>(0));
    if (t.streamed() && !t.domain.factors.empty()) {      // the printed `#n` does not say that the last factor is the stream's line
      Manifold x = *t.domain.factors.back();
      x.kind = ManifoldKind::line;
      if (x.cs && x.cs->unit.empty()) x.cs->unit = "s";
      if (!x.cs) x.cs = CoordinateSystem{"time", "clock", "s"};
      if (x.topology == Topology::none) x.topology = Topology::path;
      if (t.axis == Axis::frames && !x.made_from) x.made_from = MadeFrom{"tiled", nullptr};
      t.domain.factors.back() = std::make_shared<const Manifold>(std::move(x));
    }
  }
  return t;
}

}  // namespace nxr::graph
