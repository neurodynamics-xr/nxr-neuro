// nxr-graph — THE SHARED CHECKER (signature.h; spec docs/specs/2026-11-17-type-system.md, sections 4 to 6): unification of a use's inputs with its node's signature, the
// outputs' types made from the signature, the move's rule (the cardinal rule among them), the node's refine.
#include <algorithm>
#include <set>
#include <sstream>

#include "nxr/graph/graph.h"
#include "nxr/graph/signature.h"

namespace nxr::graph {

const char* to_string(Move m) {
  switch (m) {
    case Move::pointwise: return "pointwise";
    case Move::apply: return "apply";
    case Move::compose: return "compose";
    case Move::restrict: return "restrict";
    case Move::measure: return "measure";
    case Move::summarize: return "summarize";
    case Move::detect: return "detect";
    case Move::make: return "make";
  }
  return "?";
}

// ───────────────────────────────────────────── the builder ─────────────────────────────────────────────

namespace {
InPat pattern(PortClass c, std::vector<ManifoldRef> f) {
  InPat p;
  p.cls = c;
  p.dom.factors = std::move(f);
  p.dom.cells = "";               // a pattern does not constrain the cells unless it says
  p.has_dom = true;
  return p;
}
}  // namespace

InPat field(std::vector<ManifoldRef> f) { return pattern(PortClass::field, std::move(f)); }
InPat measurement(std::vector<ManifoldRef> f) { return pattern(PortClass::measurement, std::move(f)); }
InPat measurement_any() { InPat p; p.cls = PortClass::measurement; return p; }
InPat selection(std::vector<ManifoldRef> f) { return pattern(PortClass::selection, std::move(f)); }
InPat selection_any() { InPat p; p.cls = PortClass::selection; return p; }
InPat statistic_any() { InPat p; p.cls = PortClass::statistic; return p; }
InPat any_port() { return InPat{}; }
InPat manifold_port(const ManifoldRef& m) { return pattern(PortClass::manifold, {m}); }
InPat transition_port() { InPat p; p.cls = PortClass::operator_; return p; }
InPat operator_(const ManifoldRef& from, const ManifoldRef& to) {
  InPat p;
  p.cls = PortClass::operator_;
  p.dom.factors = {from}; p.dom.cells = "";
  p.target.factors = {to}; p.target.cells = "";
  p.has_dom = p.has_target = true;
  return p;
}
OutPat like(const std::string& port) { OutPat o; o.like_port = port; return o; }
OutPat make_of(PortClass c) { OutPat o; o.cls = c; return o; }
Signature sig(Move m) { Signature s; s.move = m; return s; }

void Signature::validate(const std::string& node) const {
  std::set<std::string> in_ports, out_ports;
  for (const auto& [name, p] : inputs) {
    if (!in_ports.insert(name).second) refuse("signature of " + node + ": input port '" + name + "' is declared twice");
    for (const auto& q : p.quantity_in)
      if (!is_quantity(q)) refuse("signature of " + node + ": port '" + name + "' names the quantity '" + q + "', which is not in the closed vocabulary (docs/reference.md)");
  }
  for (const auto& [name, o] : outputs) {
    if (!out_ports.insert(name).second) refuse("signature of " + node + ": output port '" + name + "' is declared twice");
    if (o.quantity && !is_quantity(*o.quantity)) refuse("signature of " + node + ": output '" + name + "' names the quantity '" + *o.quantity + "', which is not in the closed vocabulary (docs/reference.md)");
    if (!o.like_port.empty() && !in_ports.count(o.like_port)) refuse("signature of " + node + ": output '" + name + "' is like '" + o.like_port + "', which is not an input port");
  }
  if (move == Move::make)
    for (const auto& [name, p] : inputs)
      if (p.cls && *p.cls == PortClass::field) refuse("signature of " + node + ": a `make` node has no field input ('" + name + "')");
}

// ───────────────────────────────────────────── refusals ─────────────────────────────────────────────

const PortType& Unified::type(const std::string& port) const {
  const auto it = in.find(port);
  if (it == in.end()) fail(port, "is not wired, and this node needs it here");
  return it->second;
}

void Unified::fail(const std::string& port, const std::string& message) const {
  std::string s = label + (first_out.empty() ? "" : " -> " + first_out) + ": port `" + port + "`: " + message;
  const auto it = in.find(port);
  if (it != in.end() && it->second.typed) {
    const auto nm = in_names.find(port);
    s += " (" + (nm == in_names.end() ? port : nm->second) + " = " + print(it->second);
    if (nm != in_names.end()) {
      const auto mb = made_by.find(nm->second);
      if (mb != made_by.end()) s += ", from " + mb->second;
    }
    s += ")";
  }
  throw TypeRefusal(s);
}

namespace {

std::string made_text(const Unified& u, const std::string& port) {
  const auto nm = u.in_names.find(port);
  if (nm == u.in_names.end()) return "";
  const auto mb = u.made_by.find(nm->second);
  return mb == u.made_by.end() ? "" : " (made by " + mb->second + ")";
}

bool value_matches(ValueType want, const PortType& t) {
  if (want == ValueType::complex) return t.value == ValueType::complex || t.complex_parts;
  return t.value == want && !t.complex_parts;
}

std::string band_text(const Resolution& r) {
  if (!r.band) return "no band";
  char a[32], b[32];
  std::snprintf(a, sizeof a, "%g", (*r.band)[0]);
  std::snprintf(b, sizeof b, "%g", (*r.band)[1]);
  return std::string(a) + "-" + b + " Hz";
}
std::string level_text(const Resolution& r) {
  if (r.ladder.empty() && r.levels.empty()) return "no spatial level";
  std::string s = "level ";
  for (std::size_t i = 0; i < r.levels.size(); ++i) s += (i ? "," : "") + std::to_string(r.levels[i]);
  return s + " of `" + r.ladder + "`";
}

// the identity of a selection read at `port`: its own, or the value's name
std::string identity_of(const Unified& u, const std::string& port) {
  const PortType& t = u.in.at(port);
  if (t.sel && !t.sel->identity.empty()) return t.sel->identity;
  const auto nm = u.in_names.find(port);
  return nm == u.in_names.end() ? port : nm->second;
}

std::string band_key(const Resolution& r) {   // "8-16", the level's name in a member set
  char a[32], b[32];
  std::snprintf(a, sizeof a, "%g", (*r.band)[0]);
  std::snprintf(b, sizeof b, "%g", (*r.band)[1]);
  return std::string(a) + "-" + b;
}

// ── the measure move's rule (spec section 5): the members of the selection replace the factors it lies on; the only rule that removes a factor
void measure_rule(Unified& u, const Signature& sg) {
  std::string over_port;
  for (const auto& [name, p] : sg.inputs) if (p.is_over) over_port = name;
  if (over_port.empty()) return;
  const PortType& x = u.type("x");
  std::vector<ManifoldRef> dom;
  const auto& xf = x.domain.factors;
  if (!u.has(over_port)) {
    // no partition given: the run's span is the one member; the streamed factor goes
    if (u.chunk_node) u.fail("x", "a chunk node measures over a partition (`over`); without one the whole run is the one member, which needs an accumulate node");
    if (xf.empty() || !x.domain.streamed()) u.fail("x", "x has no streamed factor to take the run's span over");
    dom.assign(xf.begin(), xf.end() - 1);
  } else {
    const PortType& P = u.in.at(over_port);
    const SelectionShape sel = P.sel.value_or(SelectionShape{});
    const std::string id = identity_of(u, over_port);
    // any selection on factors of x will do: one region or time range is a selection of one member. Two kinds are measured, and the factors a selection covers are the ones the
    // measurement loses:
    //   a GRID (sel.partition: a ladder, a tiling, tiles, a joint or product grid) covers the factors it is defined on, in order (a product grid of space x time covers two), at the levels the field
    //     is limited to (coupled levels: the level IS the band); only a grid has a closed-form hierarchy to roll up through;
    //   an IRREGULAR selection (detected events, scouts, an atlas: sel.partition false) is measured member by member: its members may overlap and have no levels, so nothing rolls up.
    const auto& pf = P.domain.factors;
    if (sel.type == "ladder") {
      // a time ladder: the level whose band is x's band (the level IS the band)
      if (pf.size() != 1 || xf.empty() || !same(pf[0], xf.back()))
        u.fail(over_port, "the ladder `" + id + "` is on " + print(P.domain) + "; x's time line is " + (xf.empty() ? "missing" : print(xf.back())));
      if (!x.res.band) u.fail("x", "x carries no band; a constant-Q level is chosen by the band (the level IS the band): band-limit it first");
      dom.assign(xf.begin(), xf.end() - 1);
      dom.push_back(members(id + "@" + band_key(x.res), xf.back(), Role::time_tiles));
    } else {
      // a partition of factors of x: consecutive, in order (a joint partition of space x time covers two)
      std::size_t at = xf.size();
      for (std::size_t i = 0; i + pf.size() <= xf.size() && at == xf.size(); ++i) {
        bool ok = !pf.empty();
        for (std::size_t j = 0; j < pf.size(); ++j) ok = ok && same(xf[i + j], pf[j]);
        if (ok) at = i;
      }
      if (at == xf.size())
        u.fail(over_port, "the partition `" + id + "` is on " + print(P.domain) + "; x lies on " + print(x.domain) + ": the partition's manifolds are not factors of x, in order");
      if (!P.res.empty()) {
        if (!P.res.same_level(x.res))
          u.fail(over_port, "the joint partition `" + id + "` is at " + level_text(P.res) + "; x is limited to " + level_text(x.res) + ". A measure over a joint partition takes a field at the partition's own levels (the level IS the band): set the spatial level of the node that made x to the partition's, or find the cells of x's level.");
        if (!P.res.same_band(x.res))
          u.fail(over_port, "the joint partition `" + id + "` is at " + band_text(P.res) + "; x is band-limited to " + band_text(x.res) + ". A measure over a joint partition takes a field at the partition's own band (the level IS the band): find the cells of x's band, or band-limit x to the partition's.");
      }
      const bool streamed_partitioned = x.domain.streamed() && at + pf.size() == xf.size();
      if (streamed_partitioned && u.chunk_node) u.fail(over_port, "the partition `" + id + "` is of the streamed factor; a measure over it needs an accumulate node (a chunk node keeps the stream)");
      dom.assign(xf.begin(), xf.begin() + static_cast<std::ptrdiff_t>(at));
      dom.push_back(members(id, pf.back(), streamed_partitioned ? Role::time_tiles : Role::none, pf.size() > 1 ? "joint" : "", P.sel && !P.sel->partition));   // an irregular selection's members are marked: nothing rolls up through them
      dom.insert(dom.end(), xf.begin() + static_cast<std::ptrdiff_t>(at + pf.size()), xf.end());
    }
  }
  for (auto& [port, o] : u.out) {
    if (o.cls != PortClass::measurement) continue;
    o.domain.factors = dom;
    o.domain.cells = "0";
    o.res = x.res;
  }
}

// ── the cardinal rule: only `measure` (and `summarize`) removes a factor
void cardinal_rule(const Unified& u, const Signature& sg, const std::string& node) {
  if (sg.move == Move::measure || sg.move == Move::summarize || sg.move == Move::detect || sg.move == Move::make || sg.move == Move::compose) return;
  const auto xin = u.in.find("x");
  if (xin == u.in.end() || xin->second.cls != PortClass::field) return;
  const PortType& x = xin->second;
  for (const auto& [port, o] : u.out) {
    if (o.cls != PortClass::field) continue;
    const auto& xf = x.domain.factors;
    const auto& of = o.domain.factors;
    std::size_t differ = 0;
    const bool counts = of.size() == xf.size();
    if (counts) for (std::size_t i = 0; i < xf.size(); ++i) if (!same(xf[i], of[i])) ++differ;
    const bool cells_same = o.domain.cells == x.domain.cells || sg.move == Move::apply;
    const bool ok = counts && cells_same && (sg.move == Move::pointwise ? differ == 0 : differ <= 1);
    if (!ok)
      throw TypeRefusal(u.label + (u.first_out.empty() ? "" : " -> " + u.first_out) + ": `" + node + "` is declared " + to_string(sg.move) + "; its output `" + port + "` lies on " + print(o.domain) +
                        " but its input x on " + print(x.domain) + ". " + (sg.move == Move::pointwise ? "A pointwise node leaves the domain as it is" : "This move changes at most one factor") +
                        ": only `measure` removes a factor (a reduction is a measurement over a partition).");
  }
}

// ── the gauge (G5): a tangent field is mapped only by an operator built in its own gauge
void gauge_rule(const Unified& u, Types& out, const Signature& sg) {
  if (sg.move != Move::apply) return;
  const PortType* x = u.has("x") ? &u.in.at("x") : nullptr;
  const std::pair<const std::string, PortType>* op = nullptr;
  std::string op_port;
  for (const auto& kv : u.in) if (kv.second.cls == PortClass::operator_) { op = &kv; op_port = kv.first; }
  if (!x || !op) return;
  if (x->spin != 0 && (op->second.spin != x->spin || op->second.gauge != x->gauge)) {
    u.fail(op_port, "x is spin " + std::to_string(x->spin) + " in gauge " + (x->gauge.empty() ? "(none)" : x->gauge) + "; the operator is spin " + std::to_string(op->second.spin) + " in gauge " +
                       (op->second.gauge.empty() ? "(none)" : op->second.gauge) + ". Insert frame.rotate, or recompute the stored field (it predates the gauge record).");
  }
  if (op->second.spin != 0)
    for (auto& [port, o] : out)
      if (o.cls == PortClass::field) { o.spin = op->second.spin; o.gauge = op->second.gauge; }
}

}  // namespace

// ───────────────────────────────────────────── type_use ─────────────────────────────────────────────

Types type_use(const std::string& node, const Signature& sg, const TypeUse& use, const Json& params) {
  Unified u;
  u.in = use.in;
  u.label = use.label;
  u.chunk_node = use.chunk_node;
  u.in_names = use.in_names;
  u.out_names = use.out_names;
  u.made_by = use.made_by;
  u.first_out = sg.outputs.empty() || !use.out_names.count(sg.outputs[0].first) ? std::string() : use.out_names.at(sg.outputs[0].first);

  // 1. THE INPUTS: class, value, linearity, quantity, gauge record, then the domain by unification
  for (const auto& [port, pat] : sg.inputs) {
    const auto it = use.in.find(port);
    if (it == use.in.end()) {
      if (!pat.is_optional) u.fail(port, "is not wired");
      continue;
    }
    const PortType& t = it->second;
    const std::string hint = pat.hint.empty() ? "" : " " + pat.hint;
    if (!t.typed) u.fail(port, "the value has no type in the type system's terms (no domain); the checker types only what is typed");
    if (pat.cls && t.cls != *pat.cls)
      u.fail(port, std::string("is a ") + to_string(t.cls) + "; a " + to_string(*pat.cls) + " is wanted." + hint);
    if (pat.is_streamed && !t.streamed()) u.fail(port, "is not a value inside the loop (samples or frames); this node reads a stream." + hint);
    if (!pat.values.empty()) {
      bool ok = false;
      for (const ValueType v : pat.values) ok = ok || value_matches(v, t);
      if (!ok) {
        std::string want;
        for (const ValueType v : pat.values) want += (want.empty() ? "" : " or ") + std::string(to_string(v));
        u.fail(port, std::string("is ") + (t.complex_parts && t.value != ValueType::complex ? "complex " : "") + to_string(t.value) + "; this node reads " + want + "." + hint);
      }
    }
    if (pat.is_linear && t.nonlinear)
      u.fail(port, "is non-linear" + made_text(u, port) + (t.quantity.empty() ? "" : ": " + t.quantity) + ". A linear map reads the chain before any non-linear node (D106): apply it first." + hint);
    if (!pat.quantity_in.empty() && std::find(pat.quantity_in.begin(), pat.quantity_in.end(), t.quantity) == pat.quantity_in.end()) {
      std::string want;
      for (const auto& q : pat.quantity_in) want += (want.empty() ? "" : " or ") + q;
      u.fail(port, "is " + (t.quantity.empty() ? std::string("of no declared quantity") : t.quantity) + "; this node reads " + want + "." + hint);
    }
    if ((t.cls == PortClass::field || t.cls == PortClass::operator_) && t.spin != 0 && t.gauge.empty())
      u.fail(port, "is a tangent value (spin " + std::to_string(t.spin) + ") that records no gauge; recompute it with a gauge record (G5).");
    std::string why;
    if (pat.has_dom && !unify(pat.dom, t.domain, u.bind, why))
      u.fail(port, "lies on " + print(t.domain) + "; " + why + "." + hint);
    if (pat.has_target && !unify(pat.target, t.target, u.bind, why))
      u.fail(port, "reaches " + print(t.target) + "; " + why + "." + hint);
  }

  // 1b. POSITIONS (T5): the manifolds a node reads positions of lie in ONE coordinate system, or in the one it names. Never a transition of the checker's own.
  auto position_cs = [&](const std::string& port) {
    std::vector<std::pair<ManifoldRef, std::string>> found;      // the factors that have positions, with where in the port they are
    const auto it = use.in.find(port);
    if (it == use.in.end()) return found;
    for (const Domain* d : {&it->second.domain, &it->second.target})
      for (const auto& f : d->factors)
        if (f && f->cs && f->kind != ManifoldKind::line && f->form != Manifold::Form::members) found.emplace_back(f, port);
    return found;
  };
  auto transition_hint = [&](const ManifoldRef& from, const ManifoldRef& to) {
    return "No transition is applied by the checker: add `positions.transition` with the transition operator " + from->cs->space + ": " + from->cs->name + " -> " + to->cs->name +
           " (a transition is stored when the recording's device-to-head transform was imported), then run again.";
  };
  for (const auto& group : sg.position_groups) {
    ManifoldRef first; std::string first_port;
    for (const auto& port : group)
      for (const auto& [m, _] : position_cs(port)) {
        if (!first) { first = m; first_port = port; continue; }
        if (first->cs->space != m->cs->space)
          u.fail(port, "its positions are in the space " + m->cs->space + "; those of `" + first_port + "` in " + first->cs->space + ". Positions of two spaces are not compared.");
        if (!same_cs(first->cs, m->cs))
          u.fail(port, "its positions (" + print(m) + ") are in " + m->cs->space + "/" + m->cs->name + " (" + m->cs->unit + "); those of `" + first_port + "` (" + print(first) + ") in " +
                       first->cs->space + "/" + first->cs->name + " (" + first->cs->unit + "). " + transition_hint(m, first));
      }
  }
  for (const auto& [port, want] : sg.positions_named)
    for (const auto& [m, _] : position_cs(port))
      if (m->cs->name != want)
        u.fail(port, "its positions (" + print(m) + ") are in " + m->cs->space + "/" + m->cs->name + "; this node needs " + want + ". " + transition_hint(m, [&] { Manifold x = *m; x.cs = CoordinateSystem{m->cs->space, want, m->cs->unit}; return std::make_shared<const Manifold>(std::move(x)); }()));
  // 1b. TWO GAUGES NEVER MEET (G5): fields of one spin in two gauges are not combined. The one conversion is `frame.rotate`; an operator is checked by gauge_rule.
  if (sg.move != Move::apply) {
    const std::pair<const std::string, PortType>* first = nullptr;
    for (const auto& kv : use.in) {
      const PortType& t = kv.second;
      if (t.cls != PortClass::field || t.spin == 0) continue;
      if (!first) { first = &kv; continue; }
      const PortType& a = first->second;
      if (a.spin != t.spin)
        u.fail(kv.first, "is spin " + std::to_string(t.spin) + " but `" + first->first + "` is spin " + std::to_string(a.spin) + ": values of different spin are not combined (a tangent vector, a line field and a conjugate transform differently under a change of gauge).");
      if (a.gauge != t.gauge)
        u.fail(kv.first, "is in gauge " + (t.gauge.empty() ? std::string("(none)") : t.gauge) + " but `" + first->first + "` is in gauge " + (a.gauge.empty() ? std::string("(none)") : a.gauge) +
                             ": two gauges never meet. Insert frame.rotate on one of them, or recompute the stored field (it predates the gauge record).");
    }
  }

  // 2. THE OUTPUTS, made from the signature
  std::map<std::string, std::string> sels;
  for (const auto& [port, t] : use.in)
    if (t.cls == PortClass::selection) sels[port] = identity_of(u, port);
  try {
    for (const auto& [port, op] : sg.outputs) {
      PortType o;
      if (!op.like_port.empty()) {
        const auto it = use.in.find(op.like_port);
        if (it == use.in.end()) u.fail(op.like_port, "is not wired; the output `" + port + "` is made like it");
        o = it->second;
        o.facts = Json::object();
      }
      o.typed = true;
      if (op.cls) o.cls = *op.cls;
      if (op.dom) {
        Domain d = instantiate(*op.dom, u.bind, sels);
        if (d.cells.empty()) d.cells = o.domain.cells.empty() ? "0" : o.domain.cells;
        o.domain = d;
      }
      if (op.target) {
        Domain d = instantiate(*op.target, u.bind, sels);
        if (d.cells.empty()) d.cells = "0";
        o.target = d;
      }
      if (op.value) o.value = *op.value;
      if (op.complex_parts) o.complex_parts = *op.complex_parts;
      if (op.quantity) o.quantity = *op.quantity;
      if (op.nonlinear) o.nonlinear = *op.nonlinear;
      if (o.cls == PortClass::selection && !o.sel) o.sel = SelectionShape{};
      u.out[port] = std::move(o);
    }
  } catch (const TypeRefusal&) {
    throw;
  } catch (const std::invalid_argument& e) {
    throw TypeRefusal(u.label + ": the signature of " + node + ": " + e.what());
  }

  // 3. THE MOVE'S RULES, then the node's refine (which may only narrow the outputs or refuse)
  switch (sg.move) {
    case Move::measure: measure_rule(u, sg); break;
    case Move::summarize: {
      // across a grid's members a statistic is a roll-up through the grid's closed-form hierarchy; across an irregular selection's members it is a `summary` statistic of their measurements
      const auto x = use.in.find("x");
      if (x != use.in.end() && x->second.cls == PortClass::field && sg.reduces.empty())
        u.fail("x", "a field is measured before it is summarized (the cardinal rule: a reduction is a measurement over a partition): measure it over a partition, then summarize the members.");
      break;
    }
    case Move::make:
      for (const auto& [port, t] : use.in)
        if (t.cls == PortClass::field) u.fail(port, "a `make` node reads no field");
      break;
    default: break;
  }
  try {
    if (sg.refine_fn) sg.refine_fn(u, params);
  } catch (const TypeRefusal&) {
    throw;
  } catch (const std::invalid_argument& e) {
    throw TypeRefusal(u.label + (u.first_out.empty() ? "" : " -> " + u.first_out) + ": " + e.what());
  }

  // 4. what every output must be: its selection identity, what a statistic is of, a quantity in the vocabulary, the cardinal rule, the gauge
  for (auto& [port, o] : u.out) {
    const std::string value_name = u.out_names.count(port) ? u.out_names.at(port) : u.first_out + "." + port;
    if (o.cls == PortClass::selection && o.sel && o.sel->identity.empty()) o.sel->identity = value_name;
    if (o.cls == PortClass::statistic && !o.of && use.in.count("x")) {
      const PortType& x = use.in.at("x");
      if (x.cls == PortClass::measurement || x.cls == PortClass::field) o.of = OfType{x.quantity, x.res, x.domain};
    }
    if (!is_quantity(o.quantity))
      throw TypeRefusal(u.label + ": the output `" + port + "` of " + node + " has the quantity '" + o.quantity + "', which is not in the closed vocabulary (docs/reference.md)");
  }
  cardinal_rule(u, sg, node);
  gauge_rule(u, u.out, sg);
  return u.out;
}

// ───────────────────────────────────────────── a graph typed by signatures ─────────────────────────────────────────────

TypedGraph type_check(const Graph& g, const Types& inputs, const Json& params, const std::function<const Signature*(const std::string&)>& signatures) {
  TypedGraph tg;
  for (const auto& n : g.inputs) {
    const auto it = inputs.find(n);
    if (it == inputs.end()) refuse("input '" + n + "' has no type");
    tg.values[n] = it->second;
  }
  Json resolved = Json::object();
  for (const auto& [k, v] : g.parameters.items()) resolved[k] = v.is_object() && v.contains("default") ? v["default"] : v;
  if (params.is_object()) for (const auto& [k, v] : params.items()) resolved[k] = v;

  struct Item { const Use* use; const char* section; std::size_t at; };
  std::vector<Item> items;
  for (std::size_t i = 0; i < g.before.size(); ++i) items.push_back({&g.before[i], "before", i});
  for (std::size_t i = 0; i < g.chunks.size(); ++i) items.push_back({&g.chunks[i], "chunks", i});
  for (std::size_t i = 0; i < g.after.size(); ++i) items.push_back({&g.after[i], "after", i});
  for (const Item& it : items) {
    const Use& u = *it.use;
    const std::string label = std::string(it.section) + "[" + std::to_string(it.at) + "] " + u.node;
    const Signature* s = signatures(u.node);
    if (!s) refuse(label + ": the node has no signature");
    TypeUse tu;
    tu.label = label;
    tu.chunk_node = std::string(it.section) == "chunks" && !s->accumulates;
    tu.made_by = tg.made_by;
    for (const auto& [port, value] : u.in) {
      const auto v = tg.values.find(value);
      if (v == tg.values.end()) refuse(label + ": port `" + port + "` reads '" + value + "', which is neither an input of the graph nor made by an earlier use");
      tu.in[port] = v->second;
      tu.in_names[port] = value;
    }
    for (std::size_t i = 0; i < s->outputs.size(); ++i)
      tu.out_names[s->outputs[i].first] = i < u.out.size() ? u.out[i] : (u.out.empty() ? label : u.out[0]) + "." + s->outputs[i].first;
    Json with = Json::object();
    for (const auto& [k, v] : u.with.items()) {
      if (v.is_string() && !v.get<std::string>().empty() && v.get<std::string>()[0] == '$') {
        const std::string ref = v.get<std::string>().substr(1);
        if (!resolved.contains(ref)) refuse(label + ": parameter '" + k + "' reads $" + ref + ", which is not a parameter of the graph");
        with[k] = resolved[ref];
      } else {
        with[k] = v;
      }
    }
    const Types made = type_use(u.node, *s, tu, with);
    for (const auto& [port, t] : made) {
      const std::string& name = tu.out_names.at(port);
      if (tg.values.count(name)) refuse(label + ": `out` names '" + name + "', which already exists");
      tg.values[name] = t;
      tg.made_by[name] = label;
    }
  }
  return tg;
}

void check_output_kinds(const Types& values, const std::map<std::string, std::string>& declared) {
  static const std::set<std::string> known = {"field", "operator", "selection", "measurement", "statistic", "manifold"};
  for (const auto& [name, kind0] : declared) {
    const auto v = values.find(name);
    if (v == values.end()) refuse("outputs: " + name + " is not a value of the graph");
    const std::string kind = kind0 == "spans" || kind0 == "labelling" ? "selection" : kind0;
    if (!known.count(kind)) continue;
    if (std::string(to_string(v->second.cls)) != kind)
      refuse("outputs: " + name + " is declared a " + kind0 + "; the graph makes " + print(v->second));
  }
}

}  // namespace nxr::graph
