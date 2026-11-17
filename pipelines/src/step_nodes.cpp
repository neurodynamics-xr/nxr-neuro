// The registered steps as the executor's nodes (include/nxr/pipelines/step_nodes.h). A node here is a thin shell: its functions are the step's own, called with the
// description of the inputs a step plans on and the runner's `Bound`. The executor calls these shells through ITS dispatch point; no step is called from anywhere else.
#include "nxr/pipelines/step_nodes.h"
#include "nxr/pipelines/welch_steps.h"
#include "nxr/pipelines/meeg_steps.h"
#include "nxr/pipelines/partition_steps.h"

#include <variant>

#include "nxr/io/errors.h"

namespace nxr::pipelines::graph {

namespace {

namespace ng = nxr::graph;

Json row_json(const nxr::io::Row& r) {
  Json o = Json::object();
  for (const auto& [k, v] : r) {
    if (std::holds_alternative<std::int64_t>(v)) o[k] = std::get<std::int64_t>(v);
    else if (std::holds_alternative<double>(v)) o[k] = std::get<double>(v);
    else if (std::holds_alternative<std::string>(v)) o[k] = std::get<std::string>(v);
    else o[k] = nullptr;
  }
  return o;
}
nxr::io::Row json_row(const Json& o) {
  nxr::io::Row r;
  for (const auto& [k, v] : o.items()) {
    if (v.is_number_integer()) r[k] = v.get<std::int64_t>();
    else if (v.is_number()) r[k] = v.get<double>();
    else if (v.is_string()) r[k] = v.get<std::string>();
    else r[k] = std::monostate{};
  }
  return r;
}

ng::ValueType value_type_of(const Value& v) {
  if (v.complex) return ng::ValueType::complex;
  if (v.value_type == "vector3") return ng::ValueType::vector3;
  if (v.value_type == "quaternion") return ng::ValueType::quaternion;
  if (v.value_type == "hermitian3") return ng::ValueType::hermitian3;
  return ng::ValueType::real;
}
ng::PortClass class_of(const std::string& c) {
  if (c == "field") return ng::PortClass::field;
  if (c == "operator") return ng::PortClass::operator_;
  if (c == "selection") return ng::PortClass::selection;
  return ng::PortClass::constant;     // a statistic, a measurement, a span set, a labelling: stored constants and results
}

Values values_of(const ng::Types& in) {
  Values v;
  for (const auto& [port, t] : in) v[port] = value_of(t);
  return v;
}

// a step's functions as the node's: the same call, the step's own
ng::NodeSpec node_of(const StepSpec& st, const std::shared_ptr<Binding>& binding) {
  ng::NodeSpec n;
  n.name = st.name;
  n.kind = st.kind == "chunk" ? ng::Kind::chunk : st.kind == "accumulate" ? ng::Kind::accumulate : ng::Kind::array;   // setup and after run once, on arrays
  for (const auto& p : st.inputs) n.inputs.push_back({p, "", std::find(st.optional.begin(), st.optional.end(), p) != st.optional.end()});
  for (const auto& p : st.outputs) n.outputs.push_back({p, "", false});
  for (const auto& [k, v] : st.parameters.items()) n.parameters.push_back({k, v, "", "", {}, v.is_null()});   // a null default is "not given": the step reads it
  n.traits = st.traits;
  n.matches = st.matches;
  n.description = st.description;
  const StepSpec* s = &st;

  n.plan = [s, binding](const ng::Types& in, const Json& params) {
    const Values ins = values_of(in);
    Values made;
    try {
      made = s->plan(ins, params, PlanContext{*binding->bound});
    } catch (const nxr::io::Error& e) {
      ng::refuse(e.what());
    }
    ng::Planned out;
    for (const auto& [port, v] : made) out.out[port] = port_of(v);
    // the halo a chunk step needs is known from its prepared state (the taps designed, the frame tiling read); prepared here on the records alone, as the plan reads nothing stored
    if (s->kind == "chunk" && s->halo) {
      if (s->halo_at_plan) out.halo = s->halo_at_plan(ins, params);
      else {
        ExecContext ec{*binding->bound, nullptr, {}};
        out.halo = s->halo(s->prepare ? s->prepare(ins, params, ec) : std::any{});
      }
    }
    return out;
  };
  n.prepare = [s, binding](const ng::Types& in, const ng::Arrays&, const Json& params, ng::Shared& shared) {
    const Values ins = values_of(in);
    ExecContext ec{*binding->bound, binding->say, {}};
    ec.shared.swap(shared);                                         // the run's shared constants (the Studio's "uploaded once"): the first step to ask builds them
    std::any state;
    try {
      if (s->prepare) state = s->prepare(ins, params, ec);
    } catch (...) { ec.shared.swap(shared); throw; }
    ec.shared.swap(shared);
    return state;
  };
  if (s->kind == "chunk") {
    n.chunk = [s](const std::any& state, const Blocks& in, std::int64_t keep, const Place& at) { return s->chunk_at ? s->chunk_at(state, in, keep, at) : s->chunk(state, in, keep); };
    if (s->frame_place) n.frame_place = [s](const std::any& state, const Place& samples) { return s->frame_place(state, samples); };
  } else if (s->kind == "accumulate") {
    n.start = [s](const std::any& state) { return s->start(state); };
    n.step = [s](const std::any& state, std::any acc, const Blocks& in, std::int64_t, const Place& at) {
      s->fold(state, acc, in, at);
      return std::make_pair(Blocks{}, std::move(acc));
    };
    n.finish = [s](const std::any& state, std::any acc) { return s->finish(state, acc); };
  } else {
    n.run = [s, binding](const std::any& state, const ng::Arrays& in) {
      ng::Arrays ins;                                                // what an earlier step made: a found input is a description, never an array
      for (const auto& [port, a] : in) if (!a.meta.value("placeholder", false)) ins[port] = a;
      ExecContext ec{*binding->bound, binding->say, {}};
      return s->run(state, ins, ec);
    };
  }
  n.record = [s](const std::any& state) { return s->record && state.has_value() ? s->record(state) : Json::object(); };
  return n;
}

}  // namespace

// THE TYPE OF A STORED VALUE (type system T3): its domain's factors from the rows io describes (name, kind, line role, coordinate system, size), its quantity (a field's kind,
// an operator's kind, a selection's type), an operator's source and target. Only a value whose every factor io described is typed; the others are checked by their nodes' plans.
namespace {
ng::ManifoldKind kind_of_manifold(const std::string& t) {
  return t == "line" ? ng::ManifoldKind::line : t == "surface" ? ng::ManifoldKind::surface : t == "curve" ? ng::ManifoldKind::curve
         : (t == "volume" || t == "image" || t == "tetrahedral") ? ng::ManifoldKind::volume : ng::ManifoldKind::points;
}
ng::Role role_of_line(const std::string& r) {
  return r == "time" ? ng::Role::time : r == "frequency" ? ng::Role::frequency : r == "modes" ? ng::Role::modes : r == "levels" ? ng::Role::levels : ng::Role::none;
}
std::optional<ng::ManifoldRef> stored_manifold(const Value& v, const std::string& id) {
  if (id.empty() || !v.manifold_facts.contains(id)) return std::nullopt;
  const Json& f = v.manifold_facts[id];
  std::optional<ng::CoordinateSystem> cs;
  if (f.contains("cs")) cs = ng::CoordinateSystem{f["cs"].value("space", std::string()), f["cs"].value("name", std::string()), f["cs"].value("unit", std::string())};
  return ng::manifold_id(f.value("name", std::string()), id, kind_of_manifold(f.value("type", std::string())), role_of_line(f.value("role", std::string())), cs, f.value("n", std::int64_t(0)));
}
}  // namespace
void typed_from_rows(ng::PortType& t, const Value& v) {
  if (!v.row || v.cls == "statistic" || v.cls == "measurement") return;   // a statistic's type needs its OfType (the axes it was made over): typed by its node's plan
  auto domain = [&](const std::vector<std::string>& ids) -> std::optional<ng::Domain> {
    ng::Domain d;
    for (const auto& id : ids) { const auto m = stored_manifold(v, id); if (!m) return std::nullopt; d.factors.push_back(*m); }
    return d;
  };
  const nxr::io::Row& r = *v.row;
  if (v.cls == "operator") {
    const auto from = domain({v.manifolds.at(0)}), to = domain({v.manifolds.at(1)});
    if (!from || !to) return;
    t.domain = *from; t.target = *to;
    t.quantity = nxr::io::opt_text(r, "kind").value_or("");
    const std::string fv = nxr::io::opt_text(r, "from_value_type").value_or("scalar");
    t.from_value = fv == "vector3" ? ng::ValueType::vector3 : fv == "complex" ? ng::ValueType::complex : ng::ValueType::real;
  } else {
    const auto d = domain(v.manifolds);
    if (!d) return;
    t.domain = *d;
    if (v.cls == "field") t.quantity = nxr::io::opt_text(r, "kind").value_or("");
    if (v.cls == "selection" || v.cls == "spans" || v.cls == "labelling") {
      const std::string ty = nxr::io::opt_text(r, "type").value_or("");
      t.quantity = ty;
      ng::SelectionShape sh;
      sh.type = ty; sh.identity = nxr::io::text(r, "id");
      sh.partition = ty == "set" || ty == "ladder" || ty == "spans";
      t.sel = sh;
    }
  }
  if (!t.quantity.empty() && !ng::is_quantity(t.quantity)) { t.quantity.clear(); return; }   // a kind outside the closed vocabulary: left to the node's plan
  t.typed = true;
}

ng::PortType port_of(const Value& v) {
  ng::PortType t;
  t.cls = class_of(v.cls);
  t.value = value_type_of(v);
  if (v.chunked) {
    t.axis = v.facts.value("axis", std::string()) == "frames" ? ng::Axis::frames : ng::Axis::samples;
    if (v.rate > 0) t.sfreq = v.rate;
  }
  if (v.unit) t.unit = *v.unit;
  Json d = Json::object();
  d["cls"] = v.cls;
  d["manifolds"] = v.manifolds;
  d["value_type"] = v.value_type;
  d["complex"] = v.complex;
  d["time"] = v.time ? Json(*v.time) : Json();
  d["rate"] = v.rate;
  d["origin"] = v.origin;
  d["chunked"] = v.chunked;
  d["row"] = v.row ? row_json(*v.row) : Json();
  d["unit"] = v.unit ? Json(*v.unit) : Json();
  d["facts"] = v.facts;
  d["manifold_facts"] = v.manifold_facts;
  t.facts["value"] = std::move(d);
  typed_from_rows(t, v);
  return t;
}

Value value_of(const ng::PortType& t) {
  if (!t.facts.contains("value")) {                                       // what a native node (grouped names) made: described by its port type alone; `facts.cls` names a statistic, a manifold, ...
    Value v;
    v.cls = t.facts.contains("cls") && t.facts["cls"].is_string() ? t.facts["cls"].get<std::string>()
            : t.cls == ng::PortClass::field ? "field" : t.cls == ng::PortClass::operator_ ? "operator" : t.cls == ng::PortClass::constant ? "statistic" : "selection";
    v.complex = t.value == ng::ValueType::complex;
    v.value_type = t.value == ng::ValueType::vector3 ? "vector3" : t.value == ng::ValueType::quaternion ? "quaternion" : t.value == ng::ValueType::hermitian3 ? "hermitian3" : "scalar";
    v.chunked = t.streamed();
    v.rate = t.sfreq.value_or(0.0);
    if (!t.unit.empty()) v.unit = t.unit;
    v.facts = t.facts;
    return v;
  }
  const Json& d = t.facts.at("value");
  Value v;
  v.cls = d["cls"].get<std::string>();
  v.manifolds = d["manifolds"].get<std::vector<std::string>>();
  v.value_type = d["value_type"].get<std::string>();
  v.complex = d["complex"].get<bool>();
  if (!d["time"].is_null()) v.time = d["time"].get<std::string>();
  v.rate = d["rate"].get<double>();
  v.origin = d["origin"].get<double>();
  v.chunked = d["chunked"].get<bool>();
  if (!d["row"].is_null()) v.row = json_row(d["row"]);
  if (!d["unit"].is_null()) v.unit = d["unit"].get<std::string>();
  v.facts = d["facts"];
  return v;
}

ng::Registry node_registry(const std::shared_ptr<Binding>& binding) {
  ng::Registry r;
  register_welch_nodes();
  register_meeg_nodes();
  register_partition_nodes();
  for (const auto& st : registry()) if (!st.native) r.add(node_of(st, binding));
  for (const auto& name : nxr::graph::registry().names()) r.add(nxr::graph::registry().at(name));   // the native nodes (grouped names): pure, run as they are
  return r;
}

}  // namespace nxr::pipelines::graph
