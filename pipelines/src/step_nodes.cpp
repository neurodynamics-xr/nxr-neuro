// The registered steps as the executor's nodes (include/nxr/pipelines/step_nodes.h). A node here is a thin shell: its functions are the step's own, called with the
// description of the inputs a step plans on and the runner's `Bound`. The executor calls these shells through ITS dispatch point; no step is called from anywhere else.
#include "nxr/pipelines/step_nodes.h"
#include "nxr/graph/type_json.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/pipelines/welch_steps.h"
#include "nxr/pipelines/meeg_steps.h"
#include "nxr/pipelines/partition_steps.h"

#include <cmath>
#include <variant>

#include "nxr/quantity/quantity.h"

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
  if (c == "manifold") return ng::PortClass::manifold;   // a stored manifold is itself (type system section 8): `manifold<name(id)>`, never a constant
  return ng::PortClass::constant;     // a statistic, a measurement, a span set, a labelling: stored constants and results
}

Values values_of(const ng::Types& in) {
  Values v;
  for (const auto& [port, t] : in) v[port] = value_of(t);
  return v;
}

// a step's functions as the node's: the same call, the step's own
// a step's `band` given as a QUANTITY ("8-16 Hz", a definition's parameter) is the [lo, hi] pair in Hz the steps read
Json band_normalized(const Json& params) {
  if (!params.is_object() || !params.contains("band") || !params["band"].is_string()) return params;
  Json p = params;
  try {
    const nxr::quantity::Quantity q = nxr::quantity::parse(params["band"].get<std::string>());
    if (q.unit == "Hz" || q.chart == "log10") p["band"] = q.chart == "log10" ? Json::array({std::pow(10.0, q.lo), std::pow(10.0, q.hi)}) : Json::array({q.lo, q.hi});
  } catch (const std::invalid_argument&) {}                                 // left as given: the step refuses it with its own message
  return p;
}

ng::NodeSpec node_of(const StepSpec& st, const std::shared_ptr<Binding>& binding) {
  ng::NodeSpec n;
  n.name = st.name;
  n.kind = st.kind == "chunk" ? ng::Kind::chunk : st.kind == "accumulate" ? ng::Kind::accumulate : ng::Kind::array;   // setup and after run once, on arrays
  for (const auto& p : st.inputs) n.inputs.push_back({p, "", std::find(st.optional.begin(), st.optional.end(), p) != st.optional.end()});
  for (const auto& p : st.outputs) n.outputs.push_back({p, "", false});
  n.outputs_for = st.outputs_for;
  for (const auto& [k, v] : st.parameters.items()) n.parameters.push_back({k, v, "", "", {}, v.is_null()});   // a null default is "not given": the step reads it
  n.traits = st.traits;
  n.matches = st.matches;
  n.description = st.description;
  const StepSpec* s = &st;

  n.plan = [s, binding](const ng::Types& in, const Json& params_in) {
    const Json params = band_normalized(params_in);
    const Values ins = values_of(in);
    Values made;
    try {
      made = s->plan(ins, params, PlanContext{*binding->bound});
    } catch (const nxr::io::Error& e) {
      ng::refuse(e.what());
    }
    // TWO GENERIC CHECKS (spec docs/specs/2026-11-17-surrogates.md, sections 5 and 6), on the descriptions alone:
    //   a recording declared `read: shape` carries no sample: only a step that reads its shape only may take it;
    //   a value that AWAITS a linear operator (an unmixed surrogate: `awaits_operator`) may reach only linear steps that stay in the loop, until a step consumes the operator (its
    //   `pre_operator`, which clears the mark): a non-linear value or a reduction (a measurement, a statistic) made from it is refused.
    bool awaits = false;
    for (const auto& [port, v] : ins) {
      if (v.facts.value("read", std::string("samples")) == "shape" && !s->reads_shape_only)
        ng::refuse(s->name + ": its input `" + port + "` is the recording's SHAPE only (`read: shape` in the definition: no sample is read); this step reads samples. Only a step that reads no sample (a surrogate generator) takes it");
      if (v.facts.contains("awaits_operator")) awaits = true;
    }
    if (awaits)
      for (const auto& [port, v] : made)
        if (v.facts.contains("awaits_operator") && (v.facts.value("nonlinear", false) || v.cls != "field" || !v.chunked))
          ng::refuse(s->name + ": its output `" + port + "` is made from a value that awaits its linear operator (" + v.facts["awaits_operator"].dump() + ": an unmixed surrogate); a non-linear step or a reduction may read it only after the operator "
                     "is applied (give it to the first linear map as `pre_operator`, e.g. apply_kernel's)");
    ng::Planned out;
    for (const auto& [port, v] : made) out.out[port] = port_of(v);
    // the halo a chunk step needs is known from its prepared state (the taps designed, the frame tiling read); prepared here on the records alone, as the plan reads nothing stored.
    // An ACCUMULATING step may declare one too (the executor hands it the block with the loop's halo, as it does a Welch segment family's): its own columns are folded, the halo only read
    if ((s->kind == "chunk" || s->kind == "accumulate") && s->halo) {
      if (s->halo_at_plan) out.halo = s->halo_at_plan(ins, params);
      else {
        ExecContext ec{*binding->bound, nullptr, {}};
        out.halo = s->halo(s->prepare ? s->prepare(ins, params, ec) : std::any{});
      }
    }
    return out;
  };
  n.prepare = [s, binding](const ng::Types& in, const ng::Arrays& constants, const Json& params_in, ng::Shared& shared) {
    const Json params = band_normalized(params_in);
    const Values ins = values_of(in);
    ng::Arrays made;                                                // what an earlier node made (a found input is a placeholder: the step reads it through the binding)
    for (const auto& [port, a] : constants) if (!a.meta.value("placeholder", false)) made[port] = a;
    ExecContext ec{*binding->bound, binding->say, {}};
    ec.made = &made;
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
  // a graph manifold (`tile_network`'s fibre network) is typed a graph, never points (issue #298)
  return t == "line" ? ng::ManifoldKind::line : t == "surface" ? ng::ManifoldKind::surface : t == "curve" ? ng::ManifoldKind::curve : t == "graph" ? ng::ManifoldKind::graph
         : (t == "volume" || t == "image" || t == "tetrahedral") ? ng::ManifoldKind::volume : ng::ManifoldKind::points;
}
std::optional<ng::ManifoldRef> stored_manifold(const Value& v, const std::string& id) {
  if (id.empty() || !v.manifold_facts.contains(id)) return std::nullopt;
  const Json& f = v.manifold_facts[id];
  std::optional<ng::CoordinateSystem> cs;
  if (f.contains("cs")) cs = ng::CoordinateSystem{f["cs"].value("space", std::string()), f["cs"].value("name", std::string()), f["cs"].value("unit", std::string())};
  // a line's topology comes from the stored topology's kind (io's facts): a rank-1 lattice is a regular path, a path is a path, a circle is a cycle
  const std::string tk = f.value("topology", std::string());
  const ng::Topology topology = f.value("type", std::string()) != "line" ? ng::Topology::none : tk == "circle" ? ng::Topology::cycle : (tk == "lattice" || tk == "path") ? ng::Topology::path : ng::Topology::none;
  // io describes a time line by its role (the coordinate system `clock`, or a unit of seconds): its coordinate is in seconds
  if (f.value("role", std::string()) == "time") { if (!cs) cs = ng::CoordinateSystem{"time", "clock", "s"}; if (cs->unit.empty()) cs->unit = "s"; }
  return ng::manifold_id(f.value("name", std::string()), id, kind_of_manifold(f.value("type", std::string())), cs, f.value("n", std::int64_t(0)), topology);
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
  // THE GAUGE (G6, spec 5.1): a stored tangent quantity carries its gauge record; the port takes its gauge from it. One without a record (the port's signpost gauge) is refused
  // when read: it is recomputed, never converted.
  if (v.cls == "field") {
    const Json pj = Json::parse(nxr::io::opt_text(r, "producer_json").value_or("{}"), nullptr, false);
    const std::string name = nxr::io::text(r, "name"), kind = nxr::io::opt_text(r, "kind").value_or("");
    const int spin = pj.is_object() ? pj.value("spin", 0) : 0;
    const bool tangent = spin != 0 || (pj.is_object() && pj.value("gauge_dependent", false)) || (name.size() > 22 && name.compare(name.size() - 22, 22, "_canonical_gauge_angle") == 0) ||
                         (v.complex && kind.find("connection") != std::string::npos);
    const std::string gauge = pj.is_object() && pj.contains("gauge") && pj["gauge"].is_string() ? pj["gauge"].get<std::string>() : std::string();
    if (tangent && gauge.rfind("geometry-central@", 0) != 0)
      throw nxr::io::Error(nxr::io::code::invalid_argument,
                           name + " is stored in the port's signpost gauge (no gauge record); nxr-neuro now uses geometry-central@" + std::string(nxr::geometry::geometryCentralVersion()) +
                               "/vertex. Recompute it with the pipeline that made it (development data may be recomputed; the old arrays are kept until removed with the format's tools).");
    if (tangent) {
      t.spin = spin;
      t.gauge = gauge + "|" + pj.value("gauge_surface", std::string()) + "|" + (pj.contains("gauge_record") ? pj["gauge_record"].value("fingerprint", std::string()) : std::string());
    }
  }
  if (v.cls == "operator") {
    const auto from = domain({v.manifolds.at(0)}), to = domain({v.manifolds.at(1)});
    if (!from || !to) return;
    t.domain = *from; t.target = *to;
    t.domain.cells = nxr::io::opt_text(r, "from_cell").value_or("0");   // the cells at each end (D92): d0 maps '0' -> '1'; the Hodge-Dirac eigenvectors the line's '0' -> the triangulation's '0+1+2'
    t.target.cells = nxr::io::opt_text(r, "to_cell").value_or("0");
    t.quantity = nxr::io::opt_text(r, "kind").value_or("");
    if (v.facts.contains("transition") && v.facts["transition"].contains("from") && v.facts["transition"].contains("to")) {   // T5: a transition's two systems
      auto cs_of = [](const Json& c) { return ng::CoordinateSystem{c.value("space", std::string()), c.value("name", std::string()), c.value("unit", std::string())}; };
      t.transition = std::make_pair(cs_of(v.facts["transition"]["from"]), cs_of(v.facts["transition"]["to"]));
    }
    const std::string fv = nxr::io::opt_text(r, "from_value_type").value_or("scalar");
    t.from_value = fv == "vector3" ? ng::ValueType::vector3 : fv == "complex" ? ng::ValueType::complex : ng::ValueType::real;
  } else {
    const auto d = domain(v.manifolds);
    if (!d) return;
    t.domain = *d;
    if (v.cls == "field") {
      t.quantity = nxr::io::opt_text(r, "kind").value_or("");
      t.domain.cells = nxr::io::opt_text(r, "cell").value_or("0");           // which cells of its manifold the values lie on (D92)
      if (!nxr::io::is_null(r, "n_components")) t.components = static_cast<int>(nxr::io::integer(r, "n_components"));   // a vertex's components, as stored (a stack of levels included)
    }
    if (v.cls == "selection" || v.cls == "spans" || v.cls == "labelling") {
      std::string ty = nxr::io::opt_text(r, "type").value_or("");
      if (ty == "spans") {                                                  // a stored closed-form tiling of a line (a frame grid, a band's tiles) is read back as spans: its record (params.grid of kind time with a width) says it is one
        const Json pj = Json::parse(nxr::io::opt_text(r, "params_json").value_or("{}"), nullptr, false);
        if (pj.is_object() && pj.contains("grid") && pj["grid"].is_object() && pj["grid"].value("kind", std::string()) == "time" && pj["grid"].contains("width")) ty = "tiling";
      }
      t.quantity = ty;
      ng::SelectionShape sh;
      sh.type = ty; sh.identity = nxr::io::text(r, "id");
      sh.partition = ty == "set" || ty == "ladder" || ty == "spans";
      t.sel = sh;
    }
  }
  // a kind outside the closed vocabulary (an imported map's `PET <tracer>_suvr`): only the QUANTITY is left unset; the value keeps the type its record gives (its manifolds, cell,
  // value type, components, unit, time line), so a node that reads any quantity takes it and one that reads a named quantity refuses it as "of no declared quantity"
  if (!t.quantity.empty() && !ng::is_quantity(t.quantity)) t.quantity.clear();
  t.typed = true;
}

ng::PortType port_of(const Value& v) {
  ng::PortType t;
  t.cls = class_of(v.cls);
  t.value = value_type_of(v);
  t.components = t.value == ng::ValueType::vector3 ? 3 : t.value == ng::ValueType::quaternion ? 4 : t.value == ng::ValueType::hermitian3 ? 6 : 1;   // a vertex's components, as the signatures count them
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
            : t.cls == ng::PortClass::field ? "field" : t.cls == ng::PortClass::operator_ ? "operator" : t.cls == ng::PortClass::constant ? "statistic"
            : t.cls == ng::PortClass::manifold ? "manifold" : "selection";
    v.complex = t.value == ng::ValueType::complex || t.complex_parts;
    v.value_type = t.value == ng::ValueType::vector3 ? "vector3" : t.value == ng::ValueType::quaternion ? "quaternion" : t.value == ng::ValueType::hermitian3 ? "hermitian3" : "scalar";
    v.chunked = t.streamed();
    v.rate = t.sfreq.value_or(0.0);
    if (!t.unit.empty()) v.unit = t.unit;
    v.facts = t.facts;
    // THE AXES a typed node declared (NodeTraits::typed_axes): its domain (an operator's target too) rides with the value, so the runner checks it at preparation and writes its links at the store
    if (t.typed && !t.domain.empty()) {
      v.facts["domain"] = ng::to_json(t.domain);
      if (!t.target.empty()) v.facts["target"] = ng::to_json(t.target);
      v.facts["components"] = t.components;
    }
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
  if (d.contains("manifold_facts") && d["manifold_facts"].is_object()) v.manifold_facts = d["manifold_facts"];   // what io described of its manifolds (their sizes), as value_json wrote it
  return v;
}

ng::Registry node_registry(const std::shared_ptr<Binding>& binding) {
  ng::Registry r;
  register_welch_nodes();
  register_meeg_nodes();
  register_partition_nodes();
  for (const auto& st : registry()) if (!st.native) r.add(node_of(st, binding));
  for (const auto& name : nxr::graph::registry().names()) r.add(nxr::graph::registry().at(name));   // the native nodes (grouped names): pure, run as they are
  if (const ExternalNodes* x = ExternalNodes::current())                        // the nodes a caller declared by signature (compile only): checked as the others, never run
    for (const auto& n : x->nodes()) r.add(n);
  return r;
}

}  // namespace nxr::pipelines::graph
