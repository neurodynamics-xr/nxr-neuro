// nxr-graph — the types as data (type_json.h): the structured JSON form of a type, external node signatures, the compiled graph.
#include "nxr/graph/type_json.h"

#include <algorithm>
#include <memory>
#include <set>

namespace nxr::graph {

namespace {

// an enum's value from its name (the names `to_string` prints); refused naming what it reads and the valid names
template <class E, std::size_t N>
E enum_of(const Json& j, const E (&all)[N], const char* (*name)(E), const std::string& what) {
  if (!j.is_string()) refuse(what + ": a name is expected, not " + j.dump());
  const std::string s = j.get<std::string>();
  std::string valid;
  for (const E e : all) {
    if (s == name(e)) return e;
    valid += (valid.empty() ? "" : ", ") + std::string(name(e));
  }
  refuse(what + ": '" + s + "' is not one of " + valid);
}

constexpr ValueType kValues[] = {ValueType::real, ValueType::complex, ValueType::vector3, ValueType::quaternion, ValueType::hermitian3, ValueType::monogenic, ValueType::integer};
constexpr PortClass kClasses[] = {PortClass::field, PortClass::operator_, PortClass::selection, PortClass::partition, PortClass::constant, PortClass::measurement, PortClass::statistic, PortClass::manifold};
constexpr Axis kAxes[] = {Axis::none, Axis::samples, Axis::frames, Axis::members};
constexpr ManifoldKind kKinds[] = {ManifoldKind::points, ManifoldKind::line, ManifoldKind::surface, ManifoldKind::volume, ManifoldKind::curve, ManifoldKind::graph, ManifoldKind::members};
constexpr Role kRoles[] = {Role::none, Role::time, Role::frames, Role::time_tiles, Role::frequency, Role::modes, Role::levels};
constexpr Move kMoves[] = {Move::pointwise, Move::apply, Move::compose, Move::restrict, Move::measure, Move::summarize, Move::detect, Move::make};
constexpr Kind kNodeKinds[] = {Kind::array, Kind::chunk, Kind::accumulate};
constexpr TimeBehaviour kTimes[] = {TimeBehaviour::pointwise, TimeBehaviour::windowed, TimeBehaviour::stateful};
constexpr Precision kPrecisions[] = {Precision::float32, Precision::float64};
constexpr OutputPrecision kOutputs[] = {OutputPrecision::follows_input, OutputPrecision::float64};

const char* form_name(Manifold::Form f) {
  switch (f) {
    case Manifold::Form::id: return "id";
    case Manifold::Form::anonymous: return "anonymous";
    case Manifold::Form::variable: return "variable";
    case Manifold::Form::made: return "made";
    case Manifold::Form::restricted: return "restricted";
    case Manifold::Form::members: return "members";
    case Manifold::Form::derived: return "derived";
  }
  return "?";
}
constexpr Manifold::Form kForms[] = {Manifold::Form::id, Manifold::Form::anonymous, Manifold::Form::variable, Manifold::Form::made, Manifold::Form::restricted, Manifold::Form::members,
                                     Manifold::Form::derived};

Json cs_json(const CoordinateSystem& c) { return Json{{"space", c.space}, {"name", c.name}, {"unit", c.unit}}; }
CoordinateSystem cs_of(const Json& j, const std::string& what) {
  if (!j.is_object()) refuse(what + ": a coordinate system is {space, name, unit}");
  return CoordinateSystem{j.value("space", std::string()), j.value("name", std::string()), j.value("unit", std::string())};
}

std::string text_of(const Json& j, const char* key, const std::string& what) {
  if (!j.contains(key)) return {};
  if (!j[key].is_string()) refuse(what + "." + key + ": a string is expected, not " + j[key].dump());
  return j[key].get<std::string>();
}
bool flag_of(const Json& j, const char* key, const std::string& what) {
  if (!j.contains(key)) return false;
  if (!j[key].is_boolean()) refuse(what + "." + key + ": true or false, not " + j[key].dump());
  return j[key].get<bool>();
}

Json selection_json(const SelectionShape& s) {
  Json j{{"type", s.type}, {"partition", s.partition}};
  if (!s.identity.empty()) j["identity"] = s.identity;
  if (!s.detected_over.empty()) j["detected_over"] = s.detected_over;
  return j;
}

}  // namespace

// ───────────────────────────────────────────── manifolds, domains, resolutions ─────────────────────────────────────────────

Json to_json(const ManifoldRef& m) {
  if (!m) return nullptr;
  Json j = Json::object();
  j["form"] = form_name(m->form);
  if (!m->key.empty()) j["key"] = m->key;
  if (!m->label.empty()) j["label"] = m->label;
  j["kind"] = to_string(m->kind);
  j["role"] = to_string(m->role);
  if (m->cs) j["coordinate_system"] = cs_json(*m->cs);
  if (m->n > 0) j["n"] = m->n;
  if (m->spacing) j["spacing"] = *m->spacing;
  if (m->origin) j["origin"] = *m->origin;
  if (m->rate) j["rate"] = *m->rate;
  if (!m->unit.empty()) j["unit"] = m->unit;
  if (m->of) j["of"] = to_json(m->of);
  if (!m->by.empty()) j["by"] = m->by;
  if (m->irregular) j["irregular"] = true;
  if (m->rest) j["rest"] = true;
  if (m->subsume) j["subsume"] = true;
  j["text"] = print(m);
  return j;
}

ManifoldRef manifold_from_json(const Json& j) {
  if (j.is_string()) return parse_manifold(j.get<std::string>());
  if (!j.is_object()) refuse("manifold: an object or its printed form, not " + j.dump());
  const std::string w = "manifold " + j.value("text", j.dump());
  Manifold m;
  m.form = j.contains("form") ? enum_of(j["form"], kForms, form_name, w + ".form") : Manifold::Form::id;
  m.key = text_of(j, "key", w);
  m.label = text_of(j, "label", w);
  if (j.contains("kind")) m.kind = enum_of(j["kind"], kKinds, static_cast<const char* (*)(ManifoldKind)>(to_string), w + ".kind");
  else if (m.form == Manifold::Form::members) m.kind = ManifoldKind::members;
  if (j.contains("role")) m.role = enum_of(j["role"], kRoles, static_cast<const char* (*)(Role)>(to_string), w + ".role");
  if (j.contains("coordinate_system") && !j["coordinate_system"].is_null()) m.cs = cs_of(j["coordinate_system"], w + ".coordinate_system");
  if (j.contains("n")) m.n = j["n"].get<std::int64_t>();
  if (j.contains("spacing") && !j["spacing"].is_null()) m.spacing = j["spacing"].get<double>();
  if (j.contains("origin") && !j["origin"].is_null()) m.origin = j["origin"].get<double>();
  if (j.contains("rate") && !j["rate"].is_null()) m.rate = j["rate"].get<double>();
  m.unit = text_of(j, "unit", w);
  if (j.contains("of") && !j["of"].is_null()) m.of = manifold_from_json(j["of"]);
  m.by = text_of(j, "by", w);
  m.irregular = flag_of(j, "irregular", w);
  m.rest = flag_of(j, "rest", w);
  m.subsume = flag_of(j, "subsume", w);
  if ((m.form == Manifold::Form::restricted || m.form == Manifold::Form::derived) && !m.of) refuse(w + ": a " + form_name(m.form) + " manifold names what it is `of`");
  if ((m.form == Manifold::Form::restricted || m.form == Manifold::Form::members) && m.by.empty()) refuse(w + ": a " + form_name(m.form) + " manifold names the selection it is `by`");
  if (m.form != Manifold::Form::members && m.form != Manifold::Form::restricted && m.key.empty()) refuse(w + ": a " + form_name(m.form) + " manifold has a `key`");
  return std::make_shared<const Manifold>(std::move(m));
}

Json to_json(const Domain& d) {
  Json f = Json::array();
  for (const auto& m : d.factors) f.push_back(to_json(m));
  return Json{{"factors", f}, {"cells", d.cells}};
}
Domain domain_from_json(const Json& j) {
  Domain d;
  if (j.is_string()) {   // the printed form of a domain: its factors joined by " × "
    const std::string s = j.get<std::string>();
    std::size_t a = 0;
    for (;;) {
      const std::size_t b = s.find(" × ", a);
      d.factors.push_back(parse_manifold(s.substr(a, b == std::string::npos ? std::string::npos : b - a)));
      if (b == std::string::npos) break;
      a = b + std::string(" × ").size();
    }
    return d;
  }
  if (j.is_array()) { for (const auto& f : j) d.factors.push_back(manifold_from_json(f)); return d; }
  if (!j.is_object() || !j.contains("factors") || !j["factors"].is_array()) refuse("domain: {factors: [...], cells}, a list of factors, or its printed form; not " + j.dump());
  for (const auto& f : j["factors"]) d.factors.push_back(manifold_from_json(f));
  if (j.contains("cells")) d.cells = j["cells"].get<std::string>();
  return d;
}

Json to_json(const Resolution& r) {
  Json j = Json::object();
  if (r.band) j["band"] = Json::array({(*r.band)[0], (*r.band)[1]});
  if (!r.band_shape.empty()) j["band_shape"] = r.band_shape;
  if (!r.ladder.empty()) j["ladder"] = r.ladder;
  if (!r.levels.empty()) j["levels"] = r.levels;
  if (!r.spatial_shape.empty()) j["spatial_shape"] = r.spatial_shape;
  return j;
}
Resolution resolution_from_json(const Json& j) {
  Resolution r;
  if (j.is_null()) return r;
  if (!j.is_object()) refuse("resolution: {band, band_shape, ladder, levels, spatial_shape}, not " + j.dump());
  if (j.contains("band") && !j["band"].is_null()) {
    const Json& b = j["band"];
    if (!b.is_array() || b.size() != 2 || !b[0].is_number() || !b[1].is_number()) refuse("resolution.band: [low, high] in Hz, not " + b.dump());
    r.band = std::array<double, 2>{b[0].get<double>(), b[1].get<double>()};
  }
  r.band_shape = text_of(j, "band_shape", "resolution");
  r.ladder = text_of(j, "ladder", "resolution");
  if (j.contains("levels")) r.levels = j["levels"].get<std::vector<int>>();
  r.spatial_shape = text_of(j, "spatial_shape", "resolution");
  return r;
}

// ───────────────────────────────────────────── a type ─────────────────────────────────────────────

Json to_json(const PortType& t) {
  Json j = Json::object();
  j["class"] = to_string(t.cls);
  j["value"] = to_string(t.value);
  j["complex_parts"] = t.complex_parts;
  j["components"] = t.components;
  j["domain"] = to_json(t.domain);
  if (!t.target.empty()) j["target"] = to_json(t.target);
  if (t.cls == PortClass::operator_) j["from_value"] = to_string(t.from_value);
  j["axis"] = to_string(t.axis);
  j["streamed"] = t.streamed();
  if (t.sfreq) j["sfreq"] = *t.sfreq;
  if (t.rows > 0) j["rows"] = t.rows;
  j["quantity"] = t.quantity;
  j["unit"] = t.unit;
  if (!t.res.empty() || !t.res.band_shape.empty() || !t.res.spatial_shape.empty()) j["resolution"] = to_json(t.res);
  j["nonlinear"] = t.nonlinear;
  if (t.spin != 0) j["spin"] = t.spin;
  if (!t.gauge.empty()) j["gauge"] = t.gauge;
  if (t.sel) j["selection"] = selection_json(*t.sel);
  if (t.of) j["of"] = Json{{"quantity", t.of->quantity}, {"resolution", to_json(t.of->res)}, {"domain", to_json(t.of->on)}};
  if (t.transition) j["transition"] = Json{{"from", cs_json(t.transition->first)}, {"to", cs_json(t.transition->second)}};
  j["typed"] = t.typed;
  j["text"] = print(t);
  return j;
}

PortType port_type_from_json(const Json& j) {
  if (j.is_string()) return parse_type(j.get<std::string>());
  if (!j.is_object()) refuse("type: an object or its printed form, not " + j.dump());
  const std::string w = "type " + j.value("text", std::string("(structured)"));
  PortType t;
  t.typed = j.contains("typed") ? j["typed"].get<bool>() : true;
  if (j.contains("class")) t.cls = enum_of(j["class"], kClasses, static_cast<const char* (*)(PortClass)>(to_string), w + ".class");
  if (j.contains("value")) t.value = enum_of(j["value"], kValues, static_cast<const char* (*)(ValueType)>(to_string), w + ".value");
  t.complex_parts = flag_of(j, "complex_parts", w);
  if (j.contains("components")) t.components = j["components"].get<int>();
  if (j.contains("domain")) t.domain = domain_from_json(j["domain"]);
  if (j.contains("target")) t.target = domain_from_json(j["target"]);
  if (j.contains("from_value")) t.from_value = enum_of(j["from_value"], kValues, static_cast<const char* (*)(ValueType)>(to_string), w + ".from_value");
  if (j.contains("axis")) t.axis = enum_of(j["axis"], kAxes, static_cast<const char* (*)(Axis)>(to_string), w + ".axis");
  else if (flag_of(j, "streamed", w)) t.axis = Axis::samples;
  if (j.contains("sfreq") && !j["sfreq"].is_null()) t.sfreq = j["sfreq"].get<double>();
  if (j.contains("rows")) t.rows = j["rows"].get<std::int64_t>();
  t.quantity = text_of(j, "quantity", w);
  t.unit = text_of(j, "unit", w);
  if (j.contains("resolution")) t.res = resolution_from_json(j["resolution"]);
  t.nonlinear = flag_of(j, "nonlinear", w);
  if (j.contains("spin")) t.spin = j["spin"].get<int>();
  t.gauge = text_of(j, "gauge", w);
  if (j.contains("selection") && !j["selection"].is_null()) {
    const Json& s = j["selection"];
    t.sel = SelectionShape{text_of(s, "type", w + ".selection"), flag_of(s, "partition", w + ".selection"), text_of(s, "identity", w + ".selection"), text_of(s, "detected_over", w + ".selection")};
  }
  if (j.contains("of") && !j["of"].is_null()) {
    const Json& o = j["of"];
    t.of = OfType{text_of(o, "quantity", w + ".of"), o.contains("resolution") ? resolution_from_json(o["resolution"]) : Resolution{}, o.contains("domain") ? domain_from_json(o["domain"]) : Domain{}};
  }
  if (j.contains("transition") && !j["transition"].is_null())
    t.transition = std::make_pair(cs_of(j["transition"].value("from", Json()), w + ".transition.from"), cs_of(j["transition"].value("to", Json()), w + ".transition.to"));
  return t;
}

// ───────────────────────────────────────────── external nodes ─────────────────────────────────────────────

NodeSpec external_node(const Json& j) {
  if (!j.is_object() || !j.contains("name") || !j["name"].is_string() || j["name"].get<std::string>().empty()) refuse("an external node is an object with a `name`: " + j.dump());
  const std::string name = j["name"].get<std::string>();
  const std::string w = "external node " + name;
  static const std::set<std::string> keys = {"name", "kind", "move", "description", "inputs", "outputs", "parameters", "time", "window_rule", "computes_in", "output_precision"};
  for (const auto& [k, _] : j.items()) if (!keys.count(k)) refuse(w + ": unknown key '" + k + "'");

  NodeSpec n;
  n.name = name;
  n.description = j.value("description", std::string("declared by the caller (compile only)"));
  n.matches = "declared by the caller: no implementation in nxr-neuro";
  n.kind = j.contains("kind") ? enum_of(j["kind"], kNodeKinds, static_cast<const char* (*)(Kind)>(to_string), w + ".kind") : Kind::array;
  n.traits = node_traits(j.contains("time") ? enum_of(j["time"], kTimes, static_cast<const char* (*)(TimeBehaviour)>(to_string), w + ".time") : TimeBehaviour::pointwise,
                         j.contains("computes_in") ? enum_of(j["computes_in"], kPrecisions, static_cast<const char* (*)(Precision)>(to_string), w + ".computes_in") : Precision::float64,
                         j.contains("output_precision") ? enum_of(j["output_precision"], kOutputs, static_cast<const char* (*)(OutputPrecision)>(to_string), w + ".output_precision") : OutputPrecision::follows_input,
                         "declared by the caller (an external signature): checked by compile, never run here", j.value("window_rule", std::string()));

  Signature sg = sig(j.contains("move") ? enum_of(j["move"], kMoves, to_string, w + ".move") : Move::pointwise);
  if (n.kind == Kind::accumulate) sg.accumulating();
  // what the patterns cannot say, checked in refine: a variable bound only to a manifold of one kind; a port's components
  std::map<std::string, ManifoldKind> var_kind;
  std::map<std::string, int> components;
  auto factors = [&](const Json& list, const std::string& where) {
    if (!list.is_array()) refuse(where + ": a list of manifolds (patterns)");
    std::vector<ManifoldRef> f;
    for (const auto& x : list) {
      ManifoldRef m = manifold_from_json(x);
      if (m->form == Manifold::Form::variable && x.is_object() && x.contains("kind")) var_kind[m->key] = m->kind;
      f.push_back(std::move(m));
    }
    return f;
  };
  const Json ins = j.value("inputs", Json::array());
  if (!ins.is_array()) refuse(w + ".inputs: a list of ports");
  for (const auto& p : ins) {
    if (!p.is_object() || !p.contains("port") || !p["port"].is_string()) refuse(w + ".inputs: each port is {port: name, class, on, value, ...}");
    const std::string port = p["port"].get<std::string>();
    const std::string pw = w + ".inputs." + port;
    static const std::set<std::string> pkeys = {"port", "class", "on", "to", "value", "streamed", "linear", "optional", "over", "quantity", "hint", "components", "meaning"};
    for (const auto& [k, _] : p.items()) if (!pkeys.count(k)) refuse(pw + ": unknown key '" + k + "'");
    InPat pat;
    if (p.contains("class") && !(p["class"].is_string() && p["class"] == "any")) pat.cls = enum_of(p["class"], kClasses, static_cast<const char* (*)(PortClass)>(to_string), pw + ".class");
    if (p.contains("on")) { pat.dom.factors = factors(p["on"], pw + ".on"); pat.dom.cells = ""; pat.has_dom = true; }
    if (p.contains("to")) { pat.target.factors = factors(p["to"], pw + ".to"); pat.target.cells = ""; pat.has_target = true; }
    if (p.contains("value")) {
      const Json v = p["value"].is_array() ? p["value"] : Json::array({p["value"]});
      for (const auto& x : v) pat.values.push_back(enum_of(x, kValues, static_cast<const char* (*)(ValueType)>(to_string), pw + ".value"));
    }
    pat.is_streamed = flag_of(p, "streamed", pw);
    pat.is_linear = flag_of(p, "linear", pw);
    pat.is_optional = flag_of(p, "optional", pw);
    pat.is_over = flag_of(p, "over", pw);
    if (p.contains("quantity")) pat.quantity_in = p["quantity"].is_array() ? p["quantity"].get<std::vector<std::string>>() : std::vector<std::string>{p["quantity"].get<std::string>()};
    pat.hint = text_of(p, "hint", pw);
    if (p.contains("components")) components[port] = p["components"].get<int>();
    sg.in(port, pat);
    n.inputs.push_back(PortSpec{port, p.value("meaning", std::string()), pat.is_optional});
  }
  const Json outs = j.value("outputs", Json::array());
  if (!outs.is_array()) refuse(w + ".outputs: a list of ports");
  for (const auto& p : outs) {
    if (!p.is_object() || !p.contains("port") || !p["port"].is_string()) refuse(w + ".outputs: each port is {port: name, like, class, on, value, ...}");
    const std::string port = p["port"].get<std::string>();
    const std::string pw = w + ".outputs." + port;
    static const std::set<std::string> okeys = {"port", "like", "class", "on", "to", "value", "quantity", "nonlinear", "meaning"};
    for (const auto& [k, _] : p.items()) if (!okeys.count(k)) refuse(pw + ": unknown key '" + k + "'");
    OutPat o = p.contains("like") ? like(text_of(p, "like", pw)) : OutPat{};
    if (p.contains("class")) o.as(enum_of(p["class"], kClasses, static_cast<const char* (*)(PortClass)>(to_string), pw + ".class"));
    if (!p.contains("like") && !p.contains("class")) refuse(pw + ": an output is `like` an input port or of a `class`");
    if (p.contains("on")) o.on(factors(p["on"], pw + ".on"));
    if (p.contains("to")) o.onto(factors(p["to"], pw + ".to"));
    if (p.contains("value")) o.with_value(enum_of(p["value"], kValues, static_cast<const char* (*)(ValueType)>(to_string), pw + ".value"));
    if (p.contains("quantity")) o.with_quantity(text_of(p, "quantity", pw));
    if (flag_of(p, "nonlinear", pw)) o.non_linear();
    sg.out(port, o);
    n.outputs.push_back(PortSpec{port, p.value("meaning", std::string()), false});
  }
  const Json params = j.value("parameters", Json::array());
  if (!params.is_array()) refuse(w + ".parameters: a list of {name, default, meaning, unit, choices}");
  for (const auto& p : params) {
    if (!p.is_object() || !p.contains("name") || !p["name"].is_string()) refuse(w + ".parameters: each is {name, default, meaning, unit, choices}");
    ParamSpec ps;
    ps.name = p["name"].get<std::string>();
    ps.def = p.value("default", Json());
    ps.meaning = p.value("meaning", std::string());
    ps.unit = p.value("unit", std::string("-"));
    if (p.contains("choices")) ps.choices = p["choices"].get<std::vector<std::string>>();
    ps.optional = p.value("optional", false);
    n.parameters.push_back(std::move(ps));
  }
  if (!var_kind.empty() || !components.empty())
    sg.refine([var_kind, components](Unified& u, const Json&) {
      for (const auto& [v, kind] : var_kind) {
        const auto it = u.bind.vars.find(v);
        if (it == u.bind.vars.end() || !it->second || it->second->kind == kind) continue;
        for (const auto& [port, _] : u.in)                                       // the first port that bound it, for the message
          for (const auto& f : u.in.at(port).domain.factors)
            if (same(f, it->second)) u.fail(port, "its manifold " + print(f) + " is " + to_string(f->kind) + "; this node takes a " + to_string(kind) + " (" + v + ")");
        u.fail(u.in.begin()->first, "the manifold " + v + " = " + print(it->second) + " is " + to_string(it->second->kind) + "; this node takes a " + to_string(kind));
      }
      for (const auto& [port, c] : components)
        if (u.has(port) && u.type(port).components != c)
          u.fail(port, "has " + std::to_string(u.type(port).components) + " component(s) a vertex; this node reads " + std::to_string(c));
    });
  try { sg.validate(name); } catch (const std::invalid_argument& e) { refuse(w + ": " + e.what()); }
  n.signature = std::move(sg);

  // the node has no implementation: it is checked (its plan refuses an input the type system does not type), never run
  const std::string never = "the node " + name + " is declared by the caller (an external signature): nxr-neuro checks it (`compile`) and never runs it";
  n.plan = [name](const Types& in, const Json&) {
    for (const auto& [port, t] : in)
      if (!t.typed) refuse("port `" + port + "`: the value has no type in the type system's terms; " + name + " is declared by its signature alone, so its inputs must be typed");
    return Planned{};
  };
  if (n.kind == Kind::array) n.run = [never](const std::any&, const Arrays&) -> Arrays { refuse(never); };
  if (n.kind == Kind::chunk && n.traits.time != TimeBehaviour::stateful) n.chunk = [never](const std::any&, const Blocks&, std::int64_t, const Place&) -> Blocks { refuse(never); };
  if (n.kind == Kind::accumulate || n.traits.time == TimeBehaviour::stateful) {
    n.start = [never](const std::any&) -> std::any { refuse(never); };
    n.step = [never](const std::any&, std::any, const Blocks&, std::int64_t, const Place&) -> std::pair<Blocks, std::any> { refuse(never); };
  }
  if (n.kind == Kind::accumulate) n.finish = [never](const std::any&, std::any) -> Arrays { refuse(never); };
  return n;
}

std::vector<NodeSpec> external_nodes(const Json& doc) {
  const Json list = doc.is_object() && doc.contains("nodes") ? doc["nodes"] : doc;
  if (!list.is_array()) refuse("external signatures: {\"nodes\": [...]} or a list of nodes");
  std::vector<NodeSpec> out;
  std::set<std::string> seen;
  for (const auto& j : list) {
    NodeSpec n = external_node(j);
    if (!seen.insert(n.name).second) refuse("external signatures: the node " + n.name + " is declared twice");
    out.push_back(std::move(n));
  }
  return out;
}

// ───────────────────────────────────────────── the compiled graph ─────────────────────────────────────────────

namespace {

struct NodeRow { std::string label, section, node; std::map<std::string, std::string> in, out; Json row; };

Json assemble(const std::string& name, const Json& parameters, const std::vector<std::string>& inputs, const std::vector<std::string>& outputs, const Types& values,
              const std::vector<NodeRow>& rows) {
  Json in = Json::array(), out = Json::array(), nodes = Json::array(), edges = Json::array(), vals = Json::object();
  for (const auto& i : inputs) in.push_back(Json{{"name", i}, {"type", to_json(values.at(i))}});
  std::map<std::string, std::pair<std::size_t, std::string>> made;   // value -> (node index, its port)
  for (std::size_t k = 0; k < rows.size(); ++k) {
    const NodeRow& r = rows[k];
    for (const auto& [port, value] : r.in) {
      Json from = made.count(value) ? Json{{"node", made.at(value).first}, {"port", made.at(value).second}} : Json{{"input", value}};
      edges.push_back(Json{{"from", from}, {"value", value}, {"to", Json{{"node", k}, {"port", port}}}});
    }
    for (const auto& [port, value] : r.out) made[value] = {k, port};
    Json n = r.row;
    n["index"] = k;
    nodes.push_back(n);
  }
  for (const auto& [v, t] : values) vals[v] = to_json(t);
  for (const auto& o : outputs) out.push_back(Json{{"name", o}, {"type", to_json(values.at(o))}});
  return Json{{"name", name}, {"parameters", parameters}, {"inputs", in}, {"nodes", nodes}, {"edges", edges}, {"values", vals}, {"outputs", out}};
}

}  // namespace

Json compiled_json(const Checked& c) {
  std::vector<NodeRow> rows;
  for (const auto& n : c.nodes) {
    NodeRow r;
    r.in = n.in; r.out = n.out;
    Json row{{"label", n.label}, {"section", to_string(n.section)}, {"node", n.spec->name}, {"kind", to_string(n.spec->kind)}, {"time", to_string(n.spec->traits.time)},
             {"parameters", n.params}, {"in", n.in}, {"out", n.out}};
    if (n.section == Section::chunks) { row["halo"] = n.planned.halo; row["keep"] = n.keep; }
    r.row = std::move(row);
    rows.push_back(std::move(r));
  }
  Json j = assemble(c.name, c.parameters, c.inputs, c.outputs, c.values, rows);
  j["halo"] = c.halo;
  j["align"] = c.align;
  j["retime"] = c.retime ? Json(c.nodes[c.loop[*c.retime]].label) : Json();
  return j;
}

Json compiled_json(const Graph& g, const TypedGraph& typed, const Json& params, const std::function<const Signature*(const std::string&)>& signatures) {
  Json resolved = Json::object();
  for (const auto& [k, v] : g.parameters.items()) resolved[k] = v.is_object() && v.contains("default") ? v["default"] : v;
  if (params.is_object()) for (const auto& [k, v] : params.items()) resolved[k] = v;
  std::vector<NodeRow> rows;
  auto add = [&](const std::vector<Use>& uses, Section s) {
    for (std::size_t i = 0; i < uses.size(); ++i) {
      const Use& u = uses[i];
      NodeRow r;
      r.in = u.in;
      Json with = Json::object();
      for (const auto& [k, v] : u.with.items())
        with[k] = v.is_string() && !v.get<std::string>().empty() && v.get<std::string>()[0] == '$' && resolved.contains(v.get<std::string>().substr(1)) ? resolved[v.get<std::string>().substr(1)] : v;
      // the output ports in the order the use names them, as `type_check` pairs them (a node that makes a part of its outputs by a parameter: outputs_for)
      if (const Signature* sg = signatures(u.node)) {
        const NodeSpec* spec = registry().find(u.node);
        const std::vector<std::string> only = spec && spec->outputs_for ? spec->outputs_for(u.with) : std::vector<std::string>();
        std::size_t k = 0;
        for (const auto& [port, _] : sg->outputs) {
          if (spec && spec->outputs_for && std::find(only.begin(), only.end(), port) == only.end()) continue;
          if (k < u.out.size()) r.out[port] = u.out[k++];
        }
      }
      r.row = Json{{"label", std::string(to_string(s)) + "[" + std::to_string(i) + "] " + u.node}, {"section", to_string(s)}, {"node", u.node}, {"parameters", with}, {"in", u.in}, {"out", r.out}};
      rows.push_back(std::move(r));
    }
  };
  add(g.before, Section::before);
  add(g.chunks, Section::chunks);
  add(g.after, Section::after);
  Json j = assemble(g.name, resolved, g.inputs, g.outputs, typed.values, rows);
  j["halo"] = nullptr;
  j["align"] = nullptr;
  j["retime"] = nullptr;
  return j;
}

}  // namespace nxr::graph
