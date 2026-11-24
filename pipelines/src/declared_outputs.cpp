// nxr-pipelines — declared outputs (declared_outputs.h): the definition places, the node describes, io writes.
#include "nxr/pipelines/declared_outputs.h"

#include <cmath>

#include "nxr/io/errors.h"
#include "nxr/selection/selection.h"
#include "nxr/io/fibers.h"
#include "nxr/io/finder.h"
#include "nxr/io/layout.h"
#include "nxr/io/selection_store.h"
#include "nxr/pipelines/input.h"
#include "nxr/io/hash.h"
#include "nxr/pipelines/bound.h"
#include "nxr/graph/node.h"
#include "nxr/graph/type_json.h"

namespace nxr::pipelines::graph {

namespace {
using nxr::io::text;
namespace ng = nxr::graph;

[[noreturn]] void bad(const std::string& output, const std::string& what) { refuse("outputs." + output + ": " + what); }

// THE GLUED CORTEX: the glued surface is the computational manifold, the imported two-hemisphere mesh the manifold of record. A SELECTION made on the glued surface, and a MEASUREMENT over one, are
// restricted back to the original like a Field: the member of each original vertex is its glued counterpart's, the bridge's elements are dropped (the member of an original vertex the glue left out stays -1).
std::vector<std::int32_t> restrict_members(const std::vector<std::int32_t>& on_glued, const GlueMap& g, std::int64_t n_original) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(n_original), -1);
  for (std::int64_t i = 0; i < g.n_glued() && i < static_cast<std::int64_t>(on_glued.size()); ++i) {
    const std::int32_t o = g.original_of[static_cast<std::size_t>(i)];
    if (o >= 0 && o < n_original) out[static_cast<std::size_t>(o)] = on_glued[static_cast<std::size_t>(i)];
  }
  return out;
}
Json restricted_record(const GlueMap& g) { return Json{{"glued", g.glued_id}, {"original", g.original_id}, {"dropped", "the bridge"}}; }
std::int64_t original_vertices(nxr::io::SubjectWriter& sub, const GlueMap& g, const std::string& o) {
  const auto orig = task::Subject::of(sub).by_id(task::Cls::manifold, g.original_id);
  if (!orig || !orig->n_vertices) bad(o, "the original surface of the glued one it lies on (" + g.original_id + ") has no vertex count");
  return *orig->n_vertices;
}
// the selection a measurement is over, ON THE ORIGINAL: itself when it does not lie on a glued surface (or is not per element); else its twin on the original (get-or-create by name,
// `<name> on <original>`), the member of each original vertex carried over, the dictionary shared
std::string selection_on_original(nxr::io::SubjectWriter& sub, const std::string& selection_id, const std::string& output) {
  const task::Subject ts = task::Subject::of(sub);
  const auto node = ts.by_id(task::Cls::selection, selection_id);
  if (!node || !node->manifold_id) return selection_id;
  const auto g = Bound(sub, nullptr).glue_of(*node->manifold_id);
  if (!g) return selection_id;
  const std::int64_t n_original = original_vertices(sub, *g, output);
  const auto orig = ts.by_id(task::Cls::manifold, g->original_id);
  const std::string name = node->name + " on " + orig->name;
  task::Want w{.cls = task::Cls::selection};
  w.name = name; w.manifold_id = g->original_id;
  if (const auto have = ts.find(w)) return have->id;
  if (!node->type || *node->type != "set" || !node->array_path)
    bad(output, "the selection " + node->name + " lies on the glued surface and is not a set per element: it cannot be restricted back to the original (make it on the original, or as a set per element)");
  const nxr::io::selection::ElementMembers em = ts.element_members(selection_id);
  nxr::io::selection::Description d;
  d.type = nxr::io::selection::Type::set;
  d.member_of = restrict_members(em.member_of, *g, n_original);
  d.member_codes = em.member_codes;
  nxr::io::selection::Placement at;
  at.name = name;
  at.path = ts.node_path(g->original_id, name);
  at.manifold_id = g->original_id;
  at.cell = "0";
  if (node->dictionary_id) at.dictionary_id = *node->dictionary_id;
  at.params = Json{{"restricted_from", restricted_record(*g)}, {"of_selection", selection_id}};
  return nxr::io::selection::save(sub, d, at).id;
}
std::string kind_of(const Json& spec) { return spec.value("kind", std::string()); }

// `{input}` -> the found input's name (or a parameter's value), `$parameter` -> its value (the longest parameter name that fits); twice, so that a parameter whose value holds `{input}` is filled
std::string fill_once(const Plan& P, const std::string& t) {
  std::string o;
  for (std::size_t i = 0; i < t.size();) {
    if (t[i] == '$') {
      std::size_t j = i + 1;
      while (j < t.size() && (std::isalnum(static_cast<unsigned char>(t[j])) || t[j] == '_')) ++j;
      std::string best;
      for (std::size_t e = j; e > i + 1; --e)
        if (P.parameters.contains(t.substr(i + 1, e - i - 1))) { best = t.substr(i + 1, e - i - 1); break; }
      if (!best.empty()) {
        const Json& v = P.parameters[best];
        o += v.is_string() ? v.get<std::string>() : v.dump();
        i += 1 + best.size();
        continue;
      }
    } else if (t[i] == '{') {
      const auto e = t.find('}', i);
      if (e != std::string::npos) {
        const std::string k = t.substr(i + 1, e - i - 1);
        if (P.found.count(k) && P.found.at(k).row) { o += text(*P.found.at(k).row, "name"); i = e + 1; continue; }
        if (P.parameters.contains(k)) { const Json& v = P.parameters[k]; o += v.is_string() ? v.get<std::string>() : v.dump(); i = e + 1; continue; }
      }
    }
    o += t[i++];
  }
  return o;
}
std::string fill(const Plan& P, const std::string& t) { return fill_once(P, fill_once(P, t)); }

// the id a link names: an output made already in this unit, a found input (its row's id), or one of its manifolds (`X.manifold.N`, `X.time`, `X.from`, `X.to`)
std::string link(const Plan& P, const std::string& ref, const MadeIds* made, const std::string& where);
// ALTERNATIVES `a|b`: the first one made in this unit or found (an output the node made only when the data asks for it: the glued surface of a cortex of two pieces, else the surface itself)
std::string link_first(const Plan& P, const std::string& ref, const MadeIds* made, const std::string& where) {
  std::size_t from = 0;
  while (from <= ref.size()) {
    std::size_t bar = ref.find('|', from);
    if (bar == std::string::npos) bar = ref.size();
    const std::string part = ref.substr(from, bar - from);
    from = bar + 1;
    if (part.empty()) bad(where, "the link '" + ref + "' has an empty alternative");
    if (!made) { link(P, part, nullptr, where); continue; }                    // at plan time every alternative must be a valid link
    if (made->count(part)) return made->at(part);
    if (P.definition.outputs.contains(part) && !P.found.count(part)) continue;   // an output not made (optional): the next alternative
    return link(P, part, made, where);
  }
  if (made) bad(where, "none of the alternatives '" + ref + "' is made or found");
  return "";
}
// a manifold of the dataset's DEFAULT subject by name (`default.Frequency`, `default.Eigenvalue`): the shared lines (as nxr). A band set lies on the dataset's line, in the subject that owns it.
bool is_default_link(const std::string& ref) { return ref.rfind("default.", 0) == 0; }
// the output's `on` as written, with the names its node's plan fixed (`default.{line}`: the line its rule lies on)
std::string on_named(const Plan& P, const std::string& o, std::string t) {
  if (P.values.count(o) && P.values.at(o).facts.contains("names"))
    for (const auto& [k, v] : P.values.at(o).facts["names"].items())
      for (std::size_t at; (at = t.find("{" + k + "}")) != std::string::npos;) t.replace(at, k.size() + 2, v.get<std::string>());
  return t;
}
std::string default_manifold_id(nxr::io::SubjectWriter& sub, const std::string& ref) {
  const std::string name = ref.substr(8);
  if (const auto tpl = sub.dataset().template_()) {
    task::Want w{.cls = task::Cls::manifold};
    w.name = name;
    if (const auto m = task::Subject::of(sub.dataset(), text(*tpl, "id")).find(w)) return m->id;
  }
  throw nxr::io::Error(nxr::io::code::no_such_row, sub.name() + ": the dataset's default subject has no manifold '" + name + "' (the shared line a band set lies on)");
}
std::string link(const Plan& P, const std::string& ref, const MadeIds* made, const std::string& where) {
  if (ref.find('|') != std::string::npos) return link_first(P, ref, made, where);
  if (made && made->count(ref)) return made->at(ref);
  if (is_default_link(ref)) {
    if (made) bad(where, "'" + ref + "' is a manifold of the default subject: resolved when the output is stored");
    return "";
  }
  if (P.definition.outputs.contains(ref) && !P.found.count(ref)) {
    if (made) bad(where, "'" + ref + "' is an output not made yet (manifolds are made first; list the others after what they lie on)");
    return "";                                                                // at plan time: an output of this definition, made in the unit
  }
  if (P.found.count(ref)) {
    if (!P.found.at(ref).row) bad(where, "'" + ref + "' has no stored row (a link names a found input)");
    return text(*P.found.at(ref).row, "id");
  }
  try {
    return nxr::io::finder::relation(ref, P.found, "outputs." + where);
  } catch (const nxr::io::Error& e) {
    bad(where, std::string("the link '") + ref + "': " + e.what());
  }
}
std::string link_of(const Plan& P, const Json& spec, const char* key, const MadeIds* made, const std::string& o) {
  if (!spec.contains(key) || !spec[key].is_string()) bad(o, std::string(key) + ": a link (INPUT.manifold.N, INPUT.time, an output made in the unit)");
  return link(P, spec[key].get<std::string>(), made, o);
}
// the found input a link starts from (its time line gives the samples' rate and origin)
const Value* base_input(const Plan& P, const std::string& ref) {
  const std::string b = ref.substr(0, ref.find('.'));
  return P.found.count(b) ? &P.found.at(b) : nullptr;
}

task::Values values_of(const ng::Array& a, const std::vector<std::int64_t>& shape, bool single, const std::string& o) {
  std::int64_t n = 1;
  for (const auto d : shape) n *= d;
  if (n != static_cast<std::int64_t>(a.size())) bad(o, std::to_string(a.size()) + " values for the shape the node describes (" + std::to_string(n) + ")");
  task::Values v;
  v.shape = shape;
  if (a.is_single()) {                                                      // a node that held its values in single precision (precision follows the input): no double copy
    if (single) v.f32 = a.single; else v.f64.assign(a.single.begin(), a.single.end());
  } else if (single) v.f32.assign(a.data.begin(), a.data.end());
  else v.f64 = a.data;
  return v;
}
// the node's values as integers when it says so (describe.dtype: i32 | u8): an index, a count, a form degree stored as such
task::Values stored_values_of(const ng::Array& a, const std::vector<std::int64_t>& shape, bool single, const Json& d, const std::string& o) {
  const std::string dtype = d.value("dtype", std::string());
  if (dtype.empty()) return values_of(a, shape, single, o);
  std::int64_t n = 1;
  for (const auto x : shape) n *= x;
  if (n != static_cast<std::int64_t>(a.data.size())) bad(o, std::to_string(a.data.size()) + " values for the shape the node describes (" + std::to_string(n) + ")");
  task::Values v;
  v.shape = shape;
  if (dtype == "i32") for (const double x : a.data) v.i32.push_back(static_cast<std::int32_t>(x));
  else if (dtype == "u8") for (const double x : a.data) v.u8.push_back(static_cast<std::uint8_t>(x));
  else bad(o, "describe.dtype is i32 or u8, not " + dtype);
  return v;
}
const Json& describe_of(const ng::Array& a, const std::string& o) {
  if (!a.meta.contains("describe") || !a.meta["describe"].is_object()) bad(o, "its node gives no description of what it made (meta.describe): a declared output is stored from it");
  return a.meta["describe"];
}
bool single_of(const Json& d) { return d.value("precision", std::string("float64")) == "float32"; }
std::vector<std::int64_t> ints(const Json& j) { std::vector<std::int64_t> v; for (const auto& x : j) v.push_back(x.get<std::int64_t>()); return v; }
std::vector<std::int64_t> shape_of(const ng::Array& a, const Json& d) {
  if (d.contains("shape")) return ints(d["shape"]);
  if (d.contains("axes")) { std::vector<std::int64_t> s; for (const auto& ax : d["axes"]) s.push_back(ax.value("size", std::int64_t(1))); return s; }
  return a.cols > 1 ? std::vector<std::int64_t>{a.rows, a.cols} : std::vector<std::int64_t>{a.rows * a.cols};
}

// a tiling the node described, in the unit of the line it lies on (samples of the found recording's line, or seconds)
task::Tiling tiling_of(const Plan& P, const Json& t, const std::string& on_ref, const std::string& o) {
  task::Tiling out;
  const std::string unit = t.value("unit", std::string("s"));
  double origin = t.value("origin", 0.0), width = t.value("width", 0.0), hop = t.value("hop", 0.0), end = t.value("end", 0.0);
  if (unit == "samples") {
    const Value* in = base_input(P, on_ref);
    if (!in || !(in->rate > 0)) bad(o, "a tiling in samples lies on the time line of a found recording (partition.on: INPUT.time)");
    origin = in->origin + origin / in->rate; width /= in->rate; hop /= in->rate; end = in->origin + end / in->rate;
  } else if (unit != "s") {
    bad(o, "the tiling's unit is samples or s");
  }
  out.origin = origin; out.width = width; out.hop = hop; out.end = end;
  out.count = t.value("count", std::int64_t(0));
  out.skipped = t.value("skipped", std::int64_t(0));
  out.unit = "s";
  return out;
}

// a dyadic GRID the node described (type grid): `describe` names the ladder and each level, the data is the tile code of every element per level [levels x elements]
task::GridOut grid_of(const ng::Array& a, const Json& d, const std::string& o) {
  task::GridOut g;
  g.label = d.value("label", std::string());
  g.description = d.value("description", std::string());
  g.unit = d.value("unit", std::string());
  g.params = d.value("params", Json::object());
  if (d.contains("dictionary")) g.dictionary = d["dictionary"];
  if (d.contains("dictionary_id") && d["dictionary_id"].is_string()) g.dictionary_id = d["dictionary_id"].get<std::string>();
  const Json levels = d.value("levels", Json::array());
  if (!levels.is_array() || levels.empty()) bad(o, "its node describes no levels (describe.levels)");
  if (a.rows != static_cast<std::int64_t>(levels.size())) bad(o, "its node gives " + std::to_string(a.rows) + " rows of tile codes for " + std::to_string(levels.size()) + " levels");
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const Json& x = levels[l];
    task::GridLevelOut L;
    L.name = x.value("name", std::string());
    if (L.name.empty()) bad(o, "level " + std::to_string(l) + " has no name");
    L.label = x.value("label", std::string());
    L.description = x.value("description", std::string());
    L.params = x.value("params", Json::object());
    L.tiles = x.value("tiles", std::int64_t(0));
    for (std::int64_t i = 0; i < a.cols; ++i) L.codes.push_back(static_cast<std::int32_t>(a.data[static_cast<std::size_t>(static_cast<std::int64_t>(l) * a.cols + i)]));
    if (x.contains("dictionary")) L.dictionary = x["dictionary"];
    if (x.contains("dictionary_id") && x["dictionary_id"].is_string()) L.dictionary_id = x["dictionary_id"].get<std::string>();
    for (const auto& ar : x.value("arrays", Json::array())) {
      task::GridArrayOut out;
      out.suffix = ar.value("suffix", std::string());
      out.shape = ints(ar.at("shape"));
      if (ar.value("dtype", std::string("f64")) == "i32") for (const auto& v : ar.at("values")) out.i32.push_back(v.get<std::int32_t>());
      else for (const auto& v : ar.at("values")) out.f64.push_back(v.get<double>());
      L.arrays.push_back(std::move(out));
    }
    if (x.contains("centres")) L.centres = x["centres"];
    if (x.contains("pairs")) L.pairs = x["pairs"];
    g.levels.push_back(std::move(L));
  }
  return g;
}

// ORIGIN CARRY-OVER (`carry_origin: INPUT`): a result derived from an input that came from somewhere (a template's fiber bundle) carries that input's `origin` (source, licence, terms, attribution) and
// `derived_from` (its name), as io's `derived_origin` says; null when the input has none. The input is a found manifold or node whose producer record holds the origin.
Json origin_of(const Plan& P, const std::string& input, const std::string& o) {
  if (!P.found.count(input) || !P.found.at(input).row) bad(o, "carry_origin: '" + input + "' is not an input found");
  const nxr::io::Row& row = *P.found.at(input).row;
  const Json producer = Json::parse(nxr::io::opt_text(row, "producer_json").value_or("{}"), nullptr, false);
  return nxr::io::fibers::derived_origin(producer, text(row, "name"));
}

// a field's manifold: a link, or a PRODUCT of manifolds written `A x B` (the Studio's product manifold, D37: surface x surface for a matrix between the vertices of one surface); the product is found or made
// by io from its factors, in order
std::vector<std::string> product_factors(const std::string& on) {
  std::vector<std::string> out;
  std::size_t from = 0;
  for (std::size_t at; (at = on.find(" x ", from)) != std::string::npos; from = at + 3) out.push_back(on.substr(from, at - from));
  if (!out.empty()) out.push_back(on.substr(from));
  return out;
}

std::string acquisition_of(const Plan& P, const std::string& input) {
  if (!P.found.count(input) || !P.found.at(input).row) return {};
  return nxr::io::opt_text(*P.found.at(input).row, "acquisition_id").value_or("");
}

// ── THE TYPED AXES (NodeTraits::typed_axes): a node that opts in declares every axis of its outputs as a stored thing; the runner refuses any other at preparation (before data flows) and
// writes the links when it stores. The forms that resolve: a stored manifold (id), the members of a selection (found, or one this run stores), a restriction of a resolvable manifold by such a
// selection or one of its members (`<selection>#<member>`), and a line made from a stored line by a transform (io's frequency line, keyed by the time line, the samples and the rate).
const Planned* maker_of(const Plan& P, const std::string& o) {
  for (const Planned& p : P.steps) for (const auto& [port, name] : p.outs) { (void)port; if (name == o) return &p; }
  return nullptr;
}
bool typed_axes_of(const Plan& P, const std::string& o) {
  const Planned* p = maker_of(P, o);
  if (!p || !p->step) return false;
  if (const ng::NodeSpec* n = ng::registry().find(p->step->name)) return n->traits.typed_axes;
  return p->step->traits.typed_axes;
}
// a selection a domain names: found (its id), or made by this run as the output `<node>.<port>` names (a declared selection); `#member` names one of its members
struct SelectionRef { std::string found_id, output, member; };
std::optional<SelectionRef> selection_ref(const Plan& P, const std::string& by) {
  SelectionRef r;
  std::string s = by;
  if (const auto h = s.find('#'); h != std::string::npos) { r.member = s.substr(h + 1); s = s.substr(0, h); }
  for (const auto& [k, v] : P.found) {
    (void)k;
    if (v.row && (v.cls == "selection" || v.cls == "spans" || v.cls == "labelling") && text(*v.row, "id") == s) { r.found_id = s; return r; }
  }
  const auto dot = s.rfind('.');
  if (dot == std::string::npos) return std::nullopt;
  const std::string node = s.substr(0, dot), port = s.substr(dot + 1);
  for (const Planned& p : P.steps)
    if (p.step && p.step->name == node && p.outs.count(port)) {
      const std::string out = p.outs.at(port);
      if (P.definition.outputs.contains(out) && kind_of(P.definition.outputs[out]) == "selection") { r.output = out; return r; }
    }
  return std::nullopt;
}
bool input_manifold(const Plan& P, const std::string& id) {
  for (const auto& [k, v] : P.found) { (void)k; for (const auto& m : v.manifolds) if (m == id) return true; }
  return false;
}
const char* form_name(ng::Manifold::Form f) {
  switch (f) {
    case ng::Manifold::Form::id: return "a stored manifold";
    case ng::Manifold::Form::anonymous: return "an anonymous manifold";
    case ng::Manifold::Form::variable: return "a variable";
    case ng::Manifold::Form::made: return "a made manifold";
    case ng::Manifold::Form::restricted: return "a restriction";
    case ng::Manifold::Form::members: return "a member set";
    case ng::Manifold::Form::derived: return "a derived manifold";
  }
  return "?";
}
// why the factor does not resolve to something stored (empty: it does)
std::string unresolved(const Plan& P, const ng::ManifoldRef& f) {
  if (!f) return "an empty factor";
  using F = ng::Manifold::Form;
  switch (f->form) {
    case F::id: return !f->key.empty() && input_manifold(P, f->key) ? std::string() : ng::print(f) + " is not a stored manifold the run's inputs lie on";
    case F::members: return selection_ref(P, f->by) ? std::string() : ng::print(f) + ": '" + f->by + "' is neither a selection found nor one this run stores (a declared selection output)";
    case F::restricted: {
      const std::string w = unresolved(P, f->of);
      if (!w.empty()) return w;
      return selection_ref(P, f->by) ? std::string() : ng::print(f) + ": '" + f->by + "' is neither a selection found nor one this run stores (a declared selection output)";
    }
    case F::made:
      if (!f->made_from || f->made_from->relation != "transform" || !f->made_from->of) return ng::print(f) + " is made here from no stored line (only a transform of a stored line is stored by key)";
      if (f->made_from->of->form != F::id || !input_manifold(P, f->made_from->of->key)) return ng::print(f) + " is made from " + ng::print(f->made_from->of) + ", which is not a stored line of the run's inputs";
      if (!f->rate || !(*f->rate > 0) || !f->spacing || !(*f->spacing > 0)) return ng::print(f) + " states no rate and spacing (the samples it is the transform of)";
      return std::string();
    default: return ng::print(f) + " is " + form_name(f->form) + ", not a stored thing";
  }
}
void check_typed_axes(const Plan& P, const std::string& o) {
  const Value& v = P.values.at(o);
  if (!v.facts.contains("domain")) bad(o, "typed axes: its node declares typed axes but its plan gives this output no typed domain (declare every axis as a stored manifold, a selection's members, a restriction or a transform of a stored line)");
  for (const char* key : {"domain", "target"}) {
    if (!v.facts.contains(key)) continue;
    const ng::Domain d = ng::domain_from_json(v.facts[key]);
    if (d.empty()) bad(o, std::string("typed axes: its ") + key + " has no axis");
    for (std::size_t i = 0; i < d.factors.size(); ++i)
      if (const std::string w = unresolved(P, d.factors[i]); !w.empty()) bad(o, std::string("typed axes: axis ") + std::to_string(i) + " of its " + key + " is undeclared: " + w);
  }
}
std::string selection_id_of(const Plan& P, const SelectionRef& r, const MadeIds& made, const std::string& o) {
  if (!r.found_id.empty()) return r.found_id;
  if (!made.count(r.output)) bad(o, "typed axes: the selection '" + r.output + "' is not stored before it");
  return made.at(r.output);
}
// one axis's record, in the statistic table's axes_json form: {name, manifold_id | selection_id (both for a restriction), member?, unit?}
Json axis_record(task::Unit& unit, const Plan& P, const MadeIds& made, const ng::ManifoldRef& f, const std::string& o) {
  using F = ng::Manifold::Form;
  Json r = Json::object();
  r["name"] = f->label.empty() ? std::string(f->form == F::members ? "members" : "axis") : f->label;
  if (f->form == F::id) r["manifold_id"] = f->key;
  else if (f->form == F::made) r["manifold_id"] = unit.frequency_line(f->made_from->of->key, std::llround(*f->rate / *f->spacing), *f->rate);
  else if (f->form == F::members || f->form == F::restricted) {
    const auto ref = selection_ref(P, f->by);
    if (!ref) bad(o, "typed axes: '" + f->by + "' does not resolve");
    if (f->form == F::restricted) {
      const Json of = axis_record(unit, P, made, f->of, o);
      if (of.contains("manifold_id")) r["manifold_id"] = of["manifold_id"];
      if (f->label.empty()) r["name"] = of["name"];
    }
    r["selection_id"] = selection_id_of(P, *ref, made, o);
    if (!ref->member.empty()) r["member"] = ref->member;
  } else {
    bad(o, "typed axes: " + ng::print(f) + " is not a stored thing");
  }
  if (const std::string u = ng::coordinate_unit(*f); !u.empty()) r["unit"] = u;
  return r;
}
// the records of a typed output's axes: each factor's, with its size (from the stored array's shape) and what the node adds about it (`describe.axes[i]`: the elements it covers)
Json axis_records(task::Unit& unit, const Plan& P, const MadeIds& made, const std::string& o, const char* key, const std::vector<std::int64_t>& sizes, const Json& hints) {
  const ng::Domain d = ng::domain_from_json(P.values.at(o).facts.at(key));
  Json out = Json::array();
  for (std::size_t i = 0; i < d.factors.size(); ++i) {
    Json r = axis_record(unit, P, made, d.factors[i], o);
    if (i < sizes.size()) r["size"] = sizes[i];
    if (hints.is_array() && i < hints.size() && hints[i].is_object()) for (const auto& [k, v] : hints[i].items()) if (!r.contains(k)) r[k] = v;
    out.push_back(std::move(r));
  }
  return out;
}
}  // namespace

bool is_declared(const Json& spec) {
  const std::string k = kind_of(spec);
  return k == "operator" || k == "manifold" || k == "selection" || k == "field" || (k == "statistic" && (spec.contains("axes") || spec.contains("of"))) ||
         (k == "measurement" && spec.contains("over"));
}

bool is_declared(const Plan& P, const std::string& o) {
  const Json& spec = P.definition.outputs[o];
  const std::string cls = P.values.count(o) ? P.values.at(o).cls : std::string();
  if (kind_of(spec) == "selection" && (cls == "spans" || cls == "labelling")) return false;
  return is_declared(spec);
}

std::string declared_name(const Plan& P, const std::string& o) {
  const Json& spec = P.definition.outputs[o];
  if (spec.contains("name") && spec["name"].is_string()) {
    std::string t = spec["name"].get<std::string>();
    if (P.values.count(o) && P.values.at(o).facts.contains("names"))          // what the node's plan fixed that names its output (a frame length): `{frame_s}`
      for (const auto& [k, v] : P.values.at(o).facts["names"].items())
        for (std::size_t at; (at = t.find("{" + k + "}")) != std::string::npos;) t.replace(at, k.size() + 2, v.get<std::string>());
    std::string filled = fill(P, t);
    if (P.values.count(o) && P.values.at(o).facts.contains("names"))          // a parameter's default may carry a name too (`{recording} time grid {grid_rule}`)
      for (const auto& [k, v] : P.values.at(o).facts["names"].items())
        for (std::size_t at; (at = filled.find("{" + k + "}")) != std::string::npos;) filled.replace(at, k.size() + 2, v.get<std::string>());
    return kind_of(spec) == "selection" ? filled : nxr::io::safe_name(filled);   // a selection keeps its name as written (`<recording> frames 0.0026 s`); its store path is made safe by io
  }
  const auto a = P.definition.anchor();
  return nxr::io::safe_name(P.definition.name + "." + o + (a && P.found.at(*a).row ? "." + text(*P.found.at(*a).row, "name") : std::string()));
}

void check_declared(const Plan& P, const std::string& o) {
  const Json& spec = P.definition.outputs[o];
  const std::string k = kind_of(spec);
  const Value& v = P.values.at(o);
  const std::string want = k == "operator" ? "operator" : k == "manifold" ? "manifold" : k == "selection" ? "selection" : k == "field" ? "field" : k == "measurement" ? "measurement" : "statistic";
  // an output the connectome's array nodes make (class `array`): a sparse matrix over a selection's members stored two codes a row, a sparse matrix stored as a field, an index list stored as a selection
  const bool array_output = v.cls == "array" && ((k == "measurement" && spec.value("codes", 1) == 2) || k == "field" || (k == "selection" && spec.value("type", std::string()) == "indices") ||
                            (k == "manifold" && spec.value("type", std::string()) == "points") || (k == "selection" && spec.value("type", std::string()) == "partition"));   // a montage: points [n x 3] and a label per point
  if (v.cls != want && !array_output) bad(o, "declared " + k + " but its steps make a " + v.cls);
  if (v.chunked) bad(o, "a value inside the chunk loop is not stored: only what leaves it");
  if (declared_name(P, o).empty()) bad(o, "its name is empty");
  const bool typed = typed_axes_of(P, o);
  if (typed) check_typed_axes(P, o);                                        // every axis a stored thing, refused here, before any data flows
  if (k == "field") {
    if (spec.contains("on") && spec["on"].is_string() && !product_factors(spec["on"].get<std::string>()).empty()) {
      for (const auto& f : product_factors(spec["on"].get<std::string>())) link(P, f, nullptr, o);
    } else if (spec.contains("on")) link_of(P, spec, "on", nullptr, o);
    else if (v.manifolds.empty()) bad(o, "a field output names the manifold it lies on (on: INPUT.manifold.N)");
    if (!spec.contains("kind_name") || !spec["kind_name"].is_string()) bad(o, "a field output names its kind (kind_name: …)");
  } else if (k == "statistic") {
    if (!spec.contains("of")) bad(o, "a statistic names what it is of (of: INPUT)");
    link_of(P, spec, "of", nullptr, o);
    for (const auto& ax : spec.value("axes", Json::array())) {
      if (!ax.is_object() || !ax.contains("name")) bad(o, "axes: each {name, on | line}");
      // ONE TABLE FOR VALUES OVER MEMBERS (consolidation spec 6.1): a value per member of a selection is a MEASUREMENT (`selection_measurement`), never a statistic. A statistic's axes are not a selection's members
      // (its `partition:` is provenance, the members it was computed from, never an axis); a statistic with such an axis is refused, naming what to declare instead.
      // a PICK (a selection of type indices: the channels a matrix covers, the elements an array is over) names elements of a manifold in an order, it does not partition anything: an axis over it is
      // the elements', so a statistic (a covariance on the picked channels) may have it. Only a partition's members make a measurement.
      auto is_pick = [&](const std::string& name) {
        if (P.definition.outputs.contains(name)) return kind_of(P.definition.outputs[name]) == "selection" && P.definition.outputs[name].value("type", std::string()) == "indices";
        return P.found.count(name) && P.found.at(name).cls == "selection" && P.found.at(name).row && nxr::io::opt_text(*P.found.at(name).row, "type") == std::optional<std::string>("indices");
      };
      const bool pick_axis = ax.contains("selection") && ax["selection"].is_string() && is_pick(ax["selection"].get<std::string>());
      const bool member_axis = (!pick_axis && (ax.contains("members") || ax.contains("selection"))) ||
                               (ax.contains("on") && ax["on"].is_string() && ((P.found.count(ax["on"].get<std::string>()) && P.found.at(ax["on"].get<std::string>()).cls == "selection") ||
                                                                              (P.definition.outputs.contains(ax["on"].get<std::string>()) && kind_of(P.definition.outputs[ax["on"].get<std::string>()]) == "selection")));
      if (member_axis) bad(o, "axis '" + ax["name"].get<std::string>() + "' runs over the members of a selection, so these values are a measurement: declare `kind: measurement` (over: the selection) instead of a statistic");
      if (ax.contains("on")) link(P, ax["on"].get<std::string>(), nullptr, o);
    }
    if (spec.contains("partition")) {
      const Json& p = spec["partition"];
      if (!p.is_object() || !p.contains("tiling") || !p.contains("on")) bad(o, "partition: {tiling: NAME, on: LINK}");
      link_of(P, p, "on", nullptr, o);
    }
  } else if (k == "measurement") {
    // over: a selection found, or an output of this run; of: a field found, or an output of this run (none: the members' own measures, area and n_elements)
    if (spec.contains("within") && !P.found.count(spec["within"].get<std::string>())) bad(o, "within: '" + spec["within"].get<std::string>() + "' is not a found input (the bank of bands)");
    for (const char* key : {"over", "of"}) {
      if (!spec.contains(key)) { if (std::string(key) == "over") bad(o, "a measurement output names the selection it is over (over: INPUT or OUTPUT)"); continue; }
      const std::string r = spec[key].get<std::string>();
      const bool optional_absent = !P.found.count(r) && P.definition.inputs.contains(r) && !P.definition.inputs[r].value("required", true);   // an optional input the subject lacks: its outputs are not made (the node hands none)
      if (!P.found.count(r) && !P.definition.outputs.contains(r) && !optional_absent) bad(o, std::string(key) + ": '" + r + "' is neither a found input nor an output of this definition");
    }
  } else if (k == "operator") {
    if (!typed || spec.contains("from")) link_of(P, spec, "from", nullptr, o);   // a typed operator's ends may come from its domain and target
    if (!typed || spec.contains("to")) link_of(P, spec, "to", nullptr, o);
  } else if (k == "selection") {
    const std::string t = spec.value("type", std::string());
    if (t != "partition" && t != "set" && t != "indices" && t != "spans" && t != "kernels" && t != "tiling" && t != "ladder" && t != "cells" && t != "grid" && t != "product") bad(o, "type: partition, set, indices, spans, kernels, tiling, ladder, cells, grid or product");
    if (t == "product") {                                                   // a product of selections (a grid when every factor is): its factors, each a link and, for a grid, the level it is at
      if (!spec.contains("factors") || !spec["factors"].is_array() || spec["factors"].size() < 2) bad(o, "a product selection's factors: a list of two or more {selection: LINK, level: N} (or LINK)");
      for (const auto& f : spec["factors"]) link(P, f.is_object() ? f.value("selection", std::string()) : f.get<std::string>(), nullptr, o);
    }
    if (typed && !spec.contains("on")) {                                    // a typed selection lies on its domain's one factor
      if (ng::domain_from_json(v.facts["domain"]).factors.size() != 1) bad(o, "a selection without `on` lies on its domain, which must have one axis");
    } else {
      link_of(P, spec, "on", nullptr, o);
    }
  } else {
    const std::string t = spec.value("type", std::string());
    if (t != "line" && t != "surface" && t != "product" && t != "points" && t != "complex") bad(o, "type: line, surface, product, points or complex");
    if (t == "product") {
      if (!spec.contains("factors") || !spec["factors"].is_array() || spec["factors"].size() < 2) bad(o, "a product's factors: a list of two or more links or outputs");
      for (const auto& f : spec["factors"]) link(P, f.get<std::string>(), nullptr, o);
    }
    if ((t == "surface" || t == "points" || t == "complex") && spec.contains("derived_from")) link_of(P, spec, "derived_from", nullptr, o);
  }
}

// a statistic OF a measurement the run stores (a summary of its members) is of the measurement's own field: found, or stored before it
std::string statistic_of(const Plan& P, const Json& spec, const std::string& o, const bool resolve_made_only_here) {
  (void)resolve_made_only_here;
  std::string of = spec["of"].get<std::string>();
  for (int hop = 0; hop < 8 && P.definition.outputs.contains(of) && kind_of(P.definition.outputs[of]) == "measurement"; ++hop) {
    const Json& ms = P.definition.outputs[of];
    if (!ms.contains("of") || !ms["of"].is_string()) bad(o, "of: '" + of + "' is a measurement made of no field: its statistic is of nothing stored");
    of = ms["of"].get<std::string>();
  }
  return of;
}

namespace {
// a typed output's axis as it is stored now (reading only): its manifold's id, or empty when it is not stored yet (a frequency line no run has made)
std::string stored_factor(const Plan& P, const task::Subject& ts, const ng::ManifoldRef& f) {
  using F = ng::Manifold::Form;
  if (!f) return {};
  if (f->form == F::id) return f->key;
  if (f->form == F::restricted) return stored_factor(P, ts, f->of);
  if (f->form == F::made && f->made_from && f->made_from->of && f->rate && f->spacing && *f->spacing > 0)
    return ts.frequency_line_of(f->made_from->of->key, std::llround(*f->rate / *f->spacing), *f->rate).value_or("");
  return {};
}
}  // namespace

std::vector<std::pair<task::Cls, std::string>> declared_places(const Plan& P, const std::string& o, nxr::io::SubjectWriter& sub) {
  const Json& spec = P.definition.outputs[o];
  const std::string k = kind_of(spec), name = declared_name(P, o);
  const task::Subject ts = task::Subject::of(sub);                          // io's naming rule for a node's store key
  std::vector<std::pair<task::Cls, std::string>> out;
  if (k == "field") {
    if (spec.contains("on") && spec["on"].is_string() && !product_factors(spec["on"].get<std::string>()).empty()) {
      // a product of manifolds: its id is known when io finds or makes it (the store); its node's place is checked there
      return out;
    }
    std::string mid = spec.contains("on") ? link_of(P, spec, "on", nullptr, o) : P.values.at(o).manifolds.at(0);
    if (mid.empty() && spec.contains("on")) {                                // on a manifold this run makes: its place is known once that manifold is stored (by its name); before, there is none
      const std::string on = spec["on"].get<std::string>();
      if (const auto id = ts.manifold_named(declared_name(P, on))) mid = *id;
    }
    if (!mid.empty()) out.emplace_back(task::Cls::field, ts.node_path(mid, name));
  } else if (k == "statistic") {
    const std::string of = statistic_of(P, spec, o, true);
    std::string path = "statistics/" + nxr::io::safe_name(name);
    if (P.found.count(of) && P.found.at(of).row && P.found.at(of).cls == "field") path = text(*P.found.at(of).row, "path") + "_statistics/" + nxr::io::safe_name(name);
    else if (!P.found.count(of) && P.definition.outputs.contains(of) && kind_of(P.definition.outputs[of]) == "field")        // of a field this run makes: beside it, where it is (or will be) stored
      for (const auto& [fcls, fpath] : declared_places(P, of, sub)) if (fcls == task::Cls::field) { path = fpath + "_statistics/" + nxr::io::safe_name(name); break; }
    out.emplace_back(task::Cls::statistic, path);
    if (spec.contains("partition")) out.emplace_back(task::Cls::selection, ts.node_path(link_of(P, spec["partition"], "on", nullptr, o), nxr::io::safe_name(fill(P, spec["partition"]["tiling"].get<std::string>()))));
  } else if (k == "operator") {
    out.emplace_back(task::Cls::operator_, nxr::io::safe_name(name));
  } else if (k == "selection" && !spec.contains("on") && typed_axes_of(P, o) && P.values.at(o).facts.contains("domain")) {
    // a typed selection lies on its domain's one axis: its place is known when that manifold is stored (a frequency line found by its key); before, there is none
    if (const std::string mid = stored_factor(P, ts, ng::domain_from_json(P.values.at(o).facts["domain"]).factors.at(0)); !mid.empty()) out.emplace_back(task::Cls::selection, ts.node_path(mid, name));
  } else if (k == "selection") {
    // on a manifold this run makes (a frequency line made with it): it has no place before that manifold is stored, and the store order puts the
    // manifold first, so it is checked there (a new manifold has no node to collide with)
    const bool on_made_here = spec.contains("on") && spec["on"].is_string() && P.definition.outputs.contains(spec["on"].get<std::string>());
    if (spec.contains("on") && spec["on"].is_string() && is_default_link(spec["on"].get<std::string>())) out.emplace_back(task::Cls::selection, ts.node_path(default_manifold_id(sub, on_named(P, o, spec["on"].get<std::string>())), name));
    else if (!on_made_here) out.emplace_back(task::Cls::selection, ts.node_path(link_of(P, spec, "on", nullptr, o), name));
    else if (const auto id = ts.manifold_named(declared_name(P, spec["on"].get<std::string>()))) out.emplace_back(task::Cls::selection, ts.node_path(*id, name));   // the manifold is stored already (a re-run): its place is known, so a re-run is a hit
  }
  return out;
}

// THE MAKER KEY of an output: the step that makes it (its name), its resolved parameters and its inputs' identities (the found input's id, or the name of an earlier output): equal keys are the same selection
Json maker_key(const Plan& P, const std::string& output) {
  for (const Planned& p : P.steps) {
    bool makes = false;
    for (const auto& [port, name] : p.outs) { (void)port; makes = makes || name == output; }
    if (!makes) continue;
    Json ins = Json::object();
    for (const auto& [port, t] : p.ins) ins[port] = P.found.count(t) && P.found.at(t).row ? Json(text(*P.found.at(t).row, "id")) : Json(t);
    return Json{{"maker", p.step->name}, {"parameters", p.params}, {"inputs", ins}};
  }
  return Json{{"maker", output}};
}

Json store_declared(task::Unit& unit, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& o, const ng::Array& a, const Json& span, MadeIds& made, bool replace, bool& replaced) {
  const Json& spec = P.definition.outputs[o];
  const std::string k = kind_of(spec), name = declared_name(P, o);
  static const Json none = Json::object();
  // a node whose array follows the connectome convention (a sparse matrix: meta.form coo; an index list) is stored from the definition alone; any other declared output is stored from the node's description
  const bool by_convention = !a.meta.contains("describe") && (a.meta.contains("form") || (k == "measurement" && spec.value("codes", 1) == 2) || (k == "selection" && spec.value("type", std::string()) == "indices"));
  const Json& d = by_convention ? none : describe_of(a, o);
  const std::string what = P.definition.name + " output " + o;
  auto clear = [&](task::Cls c, const std::string& path) {                    // another job's node there: refused, or moved aside with --replace
    const bool there = sub.find(task::to_string(c), path).has_value();
    unit.replace(c, path, replace, what);
    if (there) replaced = true;
  };
  Json report = Json{{"output", o}, {"kind", k}, {"name", name}};
  if (spec.contains("on") && spec["on"].is_string() && is_default_link(spec["on"].get<std::string>()) && !made.count(spec["on"].get<std::string>()))
    made[spec["on"].get<std::string>()] = default_manifold_id(sub, on_named(P, o, spec["on"].get<std::string>()));

  if (k == "manifold") {
    const std::string t = spec["type"].get<std::string>();
    std::string id;
    if (t == "line") {
      task::LineOut l;
      l.name = spec.contains("name") ? nxr::io::safe_name(fill(P, spec["name"].get<std::string>())) : name;
      l.alt = spec.contains("alt") ? nxr::io::safe_name(fill(P, spec["alt"].get<std::string>())) : l.name + "_alt";
      l.n = d.value("count", std::int64_t(0));
      l.spacing = d.value("step", 1.0);
      l.origin = d.value("start", 0.0);
      l.unit = d.value("unit", std::string("Hz"));
      l.dimension = d.value("dimension", std::string("frequency"));
      l.folder = d.value("folder", l.dimension == "time" ? std::string("time") : std::string("spectral"));
      l.cs_key = spec.value("cs", d.value("cs", l.unit == "s" ? std::string("clock") : std::string("hz")));
      if (l.n < 1) bad(o, "the line the node describes has no point (count)");
      id = unit.line(l);
    } else if (t == "surface") {
      task::SurfaceOut s;
      s.name = name; s.output = o;
      s.cs_key = spec.contains("cs") ? fill(P, spec["cs"].get<std::string>()) : d.value("cs", std::string("scs"));   // the definition's, else what the node made (a transition's target)
      if (s.cs_key.empty()) s.cs_key = "scs";
      if (spec.contains("derived_from")) s.derived_from_id = link_of(P, spec, "derived_from", &made, o);
      s.vertices = values_of(a, {a.rows, a.cols}, false, o);
      if (a.cols != 3) bad(o, "a surface's vertices are [n x 3]");
      for (const auto& f : d.value("faces", Json::array())) s.faces.push_back(f.get<std::int32_t>());
      if (d.contains("producer") && d["producer"].is_object()) s.producer = d["producer"];
      if (d.contains("description") && d["description"].is_string()) s.description = d["description"].get<std::string>();
      id = unit.surface(s);
    } else if (t == "points") {
      // POINTS with one coordinate each (an operator's mode axis: the eigenvalues): replaced with --replace, what lies on it moved aside with it
      task::PointsOut p;
      p.name = name; p.output = o;
      p.cs_key = spec.contains("cs") ? fill(P, spec["cs"].get<std::string>()) : d.value("cs", std::string("eigenvalue"));
      p.operator_name = d.value("operator", std::string());
      if (spec.contains("derived_from")) p.derived_from_id = link_of(P, spec, "derived_from", &made, o);
      if (d.contains("description") && d["description"].is_string()) p.description = d["description"].get<std::string>();
      if (d.contains("producer") && d["producer"].is_object()) p.producer = d["producer"];
      if (a.cols == 3) p.positions = values_of(a, {a.rows, a.cols}, false, o);   // points in space (a montage's electrodes), in the coordinate system the node names
      else p.positions = values_of(a, {a.rows * a.cols}, false, o);
      if (replace) clear(task::Cls::manifold, unit.path_of(p));
      id = unit.points(p);
    } else if (t == "complex") {
      // a CELL COMPLEX: vertices [n x 3] are the node's array, its edges [nE x 2] and faces [nF x 3] its description
      task::ComplexOut c;
      c.name = name; c.output = o;
      c.cs_key = spec.contains("cs") ? fill(P, spec["cs"].get<std::string>()) : d.value("cs", std::string("scs"));
      if (c.cs_key.empty()) c.cs_key = "scs";
      if (spec.contains("derived_from")) c.derived_from_id = link_of(P, spec, "derived_from", &made, o);
      if (d.contains("description") && d["description"].is_string()) c.description = d["description"].get<std::string>();
      if (d.contains("producer") && d["producer"].is_object()) c.producer = d["producer"];
      if (a.cols != 3) bad(o, "a complex's vertices are [n x 3]");
      c.vertices = values_of(a, {a.rows, a.cols}, false, o);
      for (const auto& e : d.value("edges", Json::array())) c.edges.push_back(e.get<std::int32_t>());
      for (const auto& f : d.value("faces", Json::array())) c.faces.push_back(f.get<std::int32_t>());
      if (replace) clear(task::Cls::manifold, unit.path_of(c));
      id = unit.cell_complex(c);
    } else {
      task::ProductOut pr;
      for (const auto& f : spec["factors"]) pr.factors.push_back(link(P, f.get<std::string>(), &made, o));
      id = unit.product(pr);
    }
    made[o] = id;
    report["id"] = id;
    return report;
  }

  if (k == "field") {
    task::FieldOut f;
    f.name = name; f.output = o; f.kind = spec["kind_name"].get<std::string>();
    if (spec.contains("on") && spec["on"].is_string() && !product_factors(spec["on"].get<std::string>()).empty()) {
      task::ProductOut pr;
      for (const auto& fac : product_factors(spec["on"].get<std::string>())) pr.factors.push_back(link(P, fac, &made, o));
      f.manifold_id = unit.product(pr);
    } else {
      f.manifold_id = spec.contains("on") ? link_of(P, spec, "on", &made, o) : P.values.at(o).manifolds.at(0);
    }
    f.value_type = d.value("value_type", P.values.at(o).value_type);
    if (d.contains("unit") && d["unit"].is_string()) f.unit = d["unit"].get<std::string>();
    else if (spec.contains("unit")) f.unit = spec["unit"].get<std::string>();
    if (d.contains("cell") && d["cell"].is_string()) f.cell = d["cell"].get<std::string>();            // which cells of its manifold the values sit on (edges '1', faces '2')
    f.description = spec.contains("description") ? spec["description"].get<std::string>() : d.value("description", f.kind + " \xe2\x80\x94 pipeline " + P.definition.name);
    // a SOURCE ESTIMATE: made by an operator found (by_operator: the inverse kernel) from a recording found (derived_from)
    if (spec.contains("derived_from") && P.found.count(spec["derived_from"].get<std::string>()) && P.found.at(spec["derived_from"].get<std::string>()).row) f.derived_from_id = text(*P.found.at(spec["derived_from"].get<std::string>()).row, "id");
    if (spec.contains("by_operator") && P.found.count(spec["by_operator"].get<std::string>()) && P.found.at(spec["by_operator"].get<std::string>()).row) f.by_operator_id = text(*P.found.at(spec["by_operator"].get<std::string>()).row, "id");
    if (spec.contains("of_operator")) {                                  // a field of an operator this run stored (stored after it: the operators' pass comes first)
      const std::string r = spec["of_operator"].get<std::string>();
      if (!made.count(r)) bad(o, "of_operator: '" + r + "' is not an operator output stored before it");
      f.of_operator_id = made.at(r);
    }
    if (d.contains("producer") && d["producer"].is_object()) f.producer = d["producer"];
    const bool sparse = a.meta.value("form", std::string()) == "coo";
    if (sparse) {
      // a SPARSE field (the node's COO [nnz x 3]: row, column, value; meta.shape the matrix's): stored as its entries, the matrix's layout in the producer record; the precision is the node's
      if (a.cols != 3) bad(o, "a sparse field is COO [nnz x 3]");
      f.data = values_of(a, {a.rows, a.cols}, a.meta.value("precision", std::string("float64")) == "float32", o);
      f.producer = f.producer.is_object() ? f.producer : Json::object();
      f.producer["layout"] = Json{{"form", "coo"}, {"shape", a.meta.value("shape", Json::array())}, {"columns", Json::array({"row", "column", "value"})}};
    } else if (!d.value("no_array", false)) {                                // no_array: a field whose values are its measurements
      ng::Array stored = a;
      std::vector<std::int64_t> shape = shape_of(a, d);
      // THE GLUED CORTEX: a result on the derived glued surface is RESTRICTED BACK to the original for storage (the glued -> original transfer, exact; the bridge dropped like the medial wall)
      if (const auto g = d.value("glue_map", false) ? std::optional<GlueMap>() : Bound(sub, nullptr).glue_of(f.manifold_id); g && !shape.empty() && shape[0] == g->n_glued()) {   // the glue map itself lies on the glued surface and stays there
        const auto orig = task::Subject::of(sub).by_id(task::Cls::manifold, g->original_id);
        if (!orig || !orig->n_vertices) bad(o, "the original surface of the glued one it lies on (" + g->original_id + ") has no vertex count");
        std::int64_t rest = 1;
        for (std::size_t i = 1; i < shape.size(); ++i) rest *= shape[i];
        stored.data = restrict_to_original(a.data, rest, *g, *orig->n_vertices);
        stored.rows = *orig->n_vertices; stored.cols = rest;
        shape[0] = *orig->n_vertices;
        f.producer = f.producer.is_object() ? f.producer : Json::object();
        f.producer["restricted_from"] = Json{{"glued", g->glued_id}, {"original", g->original_id}, {"dropped", "the bridge"}};
        f.manifold_id = g->original_id;
      }
      f.data = stored_values_of(stored, shape, single_of(d), d, o);
    }
    if (spec.contains("carry_origin")) {
      f.producer = f.producer.is_object() ? f.producer : Json::object();
      const Json og = origin_of(P, spec["carry_origin"].get<std::string>(), o);
      if (!og.is_null()) f.producer["origin"] = og;
    }
    const std::string path = unit.path_of(f);
    clear(task::Cls::field, path);
    report["id"] = unit.field(f); report["path"] = path;
    made[o] = report["id"].get<std::string>();
    made["path:" + o] = path;
    return report;
  }

  if (k == "measurement" && spec.value("codes", 1) == 2) {
    // TWO CODES A ROW (declared: over, codes: 2): the node's array is a sparse matrix over the selection's members, COO [nnz x 3] (row member, column member, value; meta.form coo): each entry a
    // row (code, within_code, value) of the selection, the member and the other. It is not of a field (`of`, when given, names the input the values are counted over: provenance only).
    if (a.cols != 3 || a.meta.value("form", std::string()) != "coo") bad(o, "codes: 2 takes a sparse matrix over the selection's members, COO [nnz x 3] (row, column, value)");
    const std::string over = spec["over"].get<std::string>();
    task::MeasurementOut m;
    m.output = o;
    std::string over_name;
    if (made.count(over)) { m.selection_id = made.at(over); over_name = declared_name(P, over); }
    else if (P.found.count(over) && P.found.at(over).row) { m.selection_id = text(*P.found.at(over).row, "id"); over_name = text(*P.found.at(over).row, "name"); }
    else bad(o, "over: '" + over + "' is neither a selection stored before it nor an input found");
    m.selection_id = selection_on_original(sub, m.selection_id, o);          // the glued cortex: a selection on it is read on the original
    const task::Subject ts = task::Subject::of(sub);
    m.pair_measure = name;
    m.pair_unit = spec.value("unit", std::string());
    if (ts.has_measure(m.selection_id, m.pair_measure, std::nullopt)) {
      if (!replace) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + what + ": the measure '" + m.pair_measure + "' over " + over_name + " already exists; pass --replace");
      unit.remove_measurements(m.selection_id, std::nullopt, m.pair_measure);
      replaced = true;
    }
    for (std::int64_t r = 0; r < a.rows; ++r) m.pairs.push_back({static_cast<std::int64_t>(a.data[static_cast<std::size_t>(3 * r)]), static_cast<std::int64_t>(a.data[static_cast<std::size_t>(3 * r + 1)]), a.data[static_cast<std::size_t>(3 * r + 2)]});
    report["rows"] = unit.measurements(m);
    report["over"] = m.selection_id;
    if (spec.contains("carry_origin")) report["origin"] = origin_of(P, spec["carry_origin"].get<std::string>(), o);
    return report;
  }

  if (k == "measurement") {
    // MEASUREMENTS OF A FIELD OVER A PARTITION (declared: over, of): the node's array is [members x measures], `describe.measures` names each column ({measure, unit}); a measure over a time partition
    // of a time-resolved field is [rows x members] a measure (`describe.values_shape`: the shape each measure's values take, their product the node's rows), in float32 when `describe.values_f32`.
    // `of` may be a field FOUND in the subject (stored beside it: <field path>_<partition name>/<measure>) or absent (the members' own measures); one this run stored is as before.
    task::MeasurementOut m;
    m.output = o;
    const std::string over = spec["over"].get<std::string>();
    std::string over_name;
    if (made.count(over)) { m.selection_id = made.at(over); over_name = declared_name(P, over); }
    else if (P.found.count(over) && P.found.at(over).row) { m.selection_id = text(*P.found.at(over).row, "id"); over_name = text(*P.found.at(over).row, "name"); }
    else bad(o, "over: '" + over + "' is neither a selection stored before it nor an input found");
    // A MEMBER OF THE FOUND SELECTION (`describe.over_member`: a level's tile set of the grid `over` names, which the node picked by the run's scale): the measurement is over that member
    if (d.contains("over_member") && d["over_member"].is_string()) {
      const auto member = Bound(sub, nullptr).selection(d["over_member"].get<std::string>());
      if (!member || nxr::io::opt_text(*member, "member_of_id") != std::optional<std::string>(m.selection_id)) bad(o, "its node's over_member is not a member of '" + over_name + "'");
      m.selection_id = text(*member, "id");
      over_name = text(*member, "name");
    }
    // THE GLUED CORTEX: a measurement over a selection that lies on the glued surface is over its twin on the original (the selection restricted back), its provenance saying so
    if (const std::string back = selection_on_original(sub, m.selection_id, o); back != m.selection_id) {
      if (!m.attributes.is_object()) m.attributes = Json::object();
      m.attributes["restricted_from"] = Json{{"selection", m.selection_id}};
      m.selection_id = back;
    }
    bool own = false;                                                       // the rows are not of a field stored in this run: a found field's, or the members' own
    if (spec.contains("of_statistic") && !spec.contains("of")) {            // OF A STORED STATISTIC (schema 54): its array is beside the statistic's, under the selection's name; no field
      const std::string sref = spec["of_statistic"].get<std::string>();
      if (!P.found.count(sref) || !P.found.at(sref).row || P.found.at(sref).cls != "statistic") bad(o, "of_statistic: '" + sref + "' is not a statistic found as an input");
      own = true;
      m.path = text(*P.found.at(sref).row, "path") + "_" + nxr::io::safe_name(over_name);
    } else if (spec.contains("of")) {
      const std::string of = spec["of"].get<std::string>();
      if (made.count(of)) {
        m.of_field_id = made.at(of);
        m.path = made.count("path:" + of) ? made.at("path:" + of) : name;
        {   // two measurements of one field of this run (over different selections) must not share an array: the second's path carries its selection's name
          int of_this = 0;
          for (const auto& [k2, sp2] : P.definition.outputs.items()) if (sp2.value("kind", std::string()) == "measurement" && sp2.contains("over") && sp2.value("of", std::string()) == of) ++of_this;
          if (of_this > 1) m.path += "_" + nxr::io::safe_name(over_name);
        }
      } else if (P.found.count(of) && P.found.at(of).row && P.found.at(of).cls == "field") {
        own = true;
        m.of_field_id = text(*P.found.at(of).row, "id");
        m.path = text(*P.found.at(of).row, "path") + "_" + nxr::io::safe_name(over_name);
      } else {
        bad(o, "of: '" + of + "' is neither a field output stored before it nor a field found");
      }
    } else {
      own = true;
      m.path = "measurements/" + nxr::io::safe_name(over_name) + "_members";
    }
    const Json measures = d.value("measures", Json::array());
    // a node may also give [rows x members x measures] (`describe.shape`): each measure's values are [rows x members]
    const bool cube = d.contains("shape") && d["shape"].is_array() && d["shape"].size() == 3;
    const std::int64_t crow = cube ? d["shape"][0].get<std::int64_t>() * d["shape"][1].get<std::int64_t>() : a.rows, ccol = cube ? d["shape"][2].get<std::int64_t>() : a.cols;
    // A COMPLEX measurement (the node says `complex`): its columns are re, im of each measure; each measure is stored as ONE array [rows x members x 2] (value_type complex, parts re, im)
    const bool complex_values = d.value("complex", false);
    if (complex_values && ccol % 2 != 0) bad(o, "its node says complex but gives " + std::to_string(ccol) + " columns (re, im of each measure)");
    if (!measures.is_array() || static_cast<std::int64_t>(measures.size()) != (complex_values ? ccol / 2 : ccol)) bad(o, "its node names " + std::to_string(measures.size()) + " measures for " + std::to_string(ccol) + " columns");
    if (static_cast<std::int64_t>(a.data.size()) != crow * ccol) bad(o, "its node gives " + std::to_string(a.data.size()) + " values for " + std::to_string(crow) + " rows x " + std::to_string(ccol) + " measures");
    std::vector<std::int64_t> vshape = cube ? std::vector<std::int64_t>{d["shape"][0].get<std::int64_t>(), d["shape"][1].get<std::int64_t>()} : std::vector<std::int64_t>{a.rows};
    if (d.contains("values_shape")) {
      vshape = ints(d["values_shape"]);
      std::int64_t n = 1;
      for (const auto x : vshape) n *= x;
      if (n != crow) bad(o, "its node's values_shape holds " + std::to_string(n) + " values a measure, its array " + std::to_string(crow) + " rows");
    }
    if (complex_values) vshape.push_back(2);
    const bool f32 = d.value("values_f32", false);
    // (the statistic a measurement is of, when it is: set below from `of_statistic`; named here so the check of rows already there is of the same key)
    std::optional<std::string> statid;
    if (spec.contains("of_statistic") && !spec.contains("of")) statid = text(*P.found.at(spec["of_statistic"].get<std::string>()).row, "id");
    if (own) {                                                              // another job's rows of the same measures: refused, or replaced with --replace (a member's own sizes depend on the weights alone: rewritten)
      const task::Subject ts = task::Subject::of(sub);
      const std::optional<std::string> ofid = m.of_field_id.empty() ? std::nullopt : std::optional<std::string>(m.of_field_id);
      for (const auto& mj : measures) {
        const std::string nm = mj["measure"].get<std::string>();
        if (!ts.has_measure(m.selection_id, nm, ofid, statid)) continue;
        if (!replace && (ofid || statid)) throw nxr::io::Error(nxr::io::code::already_exists, sub.name() + ": " + P.definition.name + " output " + o + ": the measure '" + nm + "' of this " + (statid ? "statistic" : "field") + " over " + over_name + " already exists; pass --replace");
        unit.remove_measurements(m.selection_id, ofid, nm, statid);
        if (ofid || statid) replaced = true;
      }
    }
    for (std::size_t c = 0; c < measures.size(); ++c) {
      task::MeasurementOut::Measure x;
      x.measure = measures[c]["measure"].get<std::string>();
      x.unit = measures[c].value("unit", std::string());
      x.values.shape = vshape;
      for (std::int64_t r = 0; r < crow; ++r)
        for (std::int64_t part = 0; part < (complex_values ? 2 : 1); ++part) {
          const double v = a.data[static_cast<std::size_t>(r * ccol) + (complex_values ? 2 * c + part : c)];
          if (f32) x.values.f32.push_back(static_cast<float>(v)); else x.values.f64.push_back(v);
        }
      m.measures.push_back(std::move(x));
    }
    if (d.contains("band")) m.band = d["band"];                             // ONE band: the band's own, or the member of a band set (`within`)
    if (d.contains("attributes") && d["attributes"].is_object()) m.attributes = d["attributes"];   // the array describes its own axes
    if (typed_axes_of(P, o)) {                                              // a typed node's axes: the links to what each one is (the statistic table's axes_json form)
      if (!m.attributes.is_object()) m.attributes = Json::object();
      m.attributes["axes"] = axis_records(unit, P, made, o, "domain", vshape, d.value("axes", Json()));
    }
    // A VALUE THAT IS NOT A SCALAR (a window's z zH, a tile's crossing counts): the value's type and its components' names, from the declaration or the node (consolidation spec 6.1); and the statistic it
    // is of, and the level it is at when it is one level of a grid (the product's factor levels are in the product itself)
    if (spec.contains("value_type") && spec["value_type"].is_string()) m.value_type = spec["value_type"].get<std::string>();
    else if (d.contains("value_type") && d["value_type"].is_string()) m.value_type = d["value_type"].get<std::string>();
    {
      const Json comps = spec.contains("components") ? spec["components"] : d.value("components", Json());
      if (comps.is_array()) for (const auto& c : comps) m.components.push_back(c.get<std::string>());
    }
    if (spec.contains("of_statistic")) { m.of_statistic_id = link_of(P, spec, "of_statistic", &made, o); if (!m.of_index) m.of_index = 0; }
    if (spec.contains("level") && spec["level"].is_number_integer()) m.level = spec["level"].get<std::int64_t>();
    else if (d.contains("level") && d["level"].is_number_integer()) m.level = d["level"].get<std::int64_t>();
    if (d.contains("snapped")) {                                            // what the stated quantity snapped to (the level's real size): in the array's attributes, as provenance
      if (!m.attributes.is_object()) m.attributes = Json::object();
      m.attributes["snapped"] = d["snapped"];
    }
    if (spec.contains("within")) {                                          // the bank of bands a measure is within (found: its code is the band's index, which the node names: describe.within_code)
      const std::string w = spec["within"].get<std::string>();
      if (!P.found.count(w) || !P.found.at(w).row) bad(o, "within: '" + w + "' is not a found selection");
      m.within_selection_id = text(*P.found.at(w).row, "id");
      if (d.contains("within_code")) m.within_code = d["within_code"].get<std::int64_t>();
    }
    if (d.contains("partition_params") && d["partition_params"].is_object()) unit.set_params(m.selection_id, d["partition_params"]);
    report["rows"] = unit.measurements(m);
    report["over"] = m.selection_id;
    return report;
  }

  if (k == "operator") {
    task::OperatorOut op;
    op.name = name; op.output = o; op.kind = spec["operator_kind"].get<std::string>();
    const bool typed_op = typed_axes_of(P, o);
    Json from_axes, to_axes;
    if (typed_op) {                                                         // a typed operator: its source and target as stored things (a target restricted to a selection names it)
      from_axes = axis_records(unit, P, made, o, "domain", {}, Json());
      if (P.values.at(o).facts.contains("target")) to_axes = axis_records(unit, P, made, o, "target", {}, Json());
    }
    op.from_manifold_id = typed_op && !spec.contains("from") ? from_axes.at(0).value("manifold_id", std::string()) : link_of(P, spec, "from", &made, o);
    op.to_manifold_id = typed_op && !spec.contains("to") && to_axes.is_array() && !to_axes.empty() ? to_axes.at(0).value("manifold_id", std::string()) : link_of(P, spec, "to", &made, o);
    op.from_value_type = d.value("from_value_type", std::string("scalar"));
    op.to_value_type = d.value("to_value_type", std::string("scalar"));
    if (d.contains("description")) op.description = d["description"].get<std::string>();
    if (d.contains("params")) op.params = d["params"];
    if (d.contains("producer") && d["producer"].is_object()) op.producer = d["producer"];
    // its links: the picks it maps from/to and the statistic it is derived from (outputs of this run, stored before it: their names sort first, or inputs found), the operator it
    // is derived from and the one it inverts (inputs found)
    auto id_of = [&](const char* key) -> std::optional<std::string> {
      if (!spec.contains(key)) return std::nullopt;
      const std::string r = spec[key].get<std::string>();
      if (made.count(r)) return made.at(r);
      if (P.found.count(r) && P.found.at(r).row) return text(*P.found.at(r).row, "id");
      bad(o, std::string(key) + ": '" + r + "' is neither an output stored before it nor an input found");
    };
    op.from_selection_id = id_of("from_selection"); op.to_selection_id = id_of("to_selection");
    if (typed_op) {                                                         // the axes' links, and a restriction's selection as the end's pick when the definition names none
      if (!op.params.is_object()) op.params = Json::object();
      op.params["axes"] = Json{{"from", from_axes}, {"to", to_axes}};
      if (!op.from_selection_id && from_axes.at(0).contains("selection_id")) op.from_selection_id = from_axes.at(0)["selection_id"].get<std::string>();
      if (!op.to_selection_id && to_axes.is_array() && !to_axes.empty() && to_axes.at(0).contains("selection_id")) op.to_selection_id = to_axes.at(0)["selection_id"].get<std::string>();
    }
    op.of_statistic_id = id_of("of_statistic"); op.derived_from_id = id_of("derived_from");
    if (const auto inv = id_of("inverts")) {
      if (d.value("inverts", true)) { op.inverts_id = inv; op.inverse_kind = spec.value("inverse_kind", std::string("regularized")); }
    }
    const std::string acq_from = spec.contains("acquisition_of") ? spec["acquisition_of"].get<std::string>() : spec.contains("derived_from") ? spec["derived_from"].get<std::string>() : std::string();
    if (const auto acq = acquisition_of(P, acq_from); !acq.empty()) op.acquisition = acq;
    if (d.contains("from_cell") && d["from_cell"].is_string()) op.from_cell = d["from_cell"].get<std::string>();
    if (d.contains("to_cell") && d["to_cell"].is_string()) op.to_cell = d["to_cell"].get<std::string>();
    if (d.contains("chunks")) op.chunk_shape = ints(d["chunks"]);
    if (d.contains("shards")) op.shard_shape = ints(d["shards"]);
    if (d.contains("sparse") && d["sparse"].is_object()) {
      // a SPARSE operator: the node's array holds the nnz values, its description the structure (csr: indptr [rows + 1], indices [nnz])
      const Json& sp = d["sparse"];
      task::OperatorOut::Sparse x;
      x.layout = sp.value("layout", std::string("csr"));
      x.n_rows = sp.value("rows", std::int64_t(0)); x.n_cols = sp.value("cols", std::int64_t(0));
      for (const auto& v : sp.value("indptr", Json::array())) x.indptr.push_back(v.get<std::int64_t>());
      for (const auto& v : sp.value("indices", Json::array())) x.indices.push_back(v.get<std::int32_t>());
      op.sparse = std::move(x);
      op.data = values_of(a, {static_cast<std::int64_t>(a.data.size())}, single_of(d), o);
    } else {
      op.data = values_of(a, {a.rows, a.cols}, single_of(d), o);
    }
    const std::string path = unit.path_of(op);
    clear(task::Cls::operator_, path);
    report["id"] = unit.operator_(op); report["path"] = path;
    made[o] = report["id"].get<std::string>();                              // a field of this operator (of_operator), and an operator derived from it, link to it
    return report;
  }

  if (k == "selection") {
    task::SelectionOut s;
    const std::string t = spec["type"].get<std::string>();
    s.name = name; s.output = o; s.type = t == "ladder" ? "tiling" : t;
    if (!spec.contains("on") && typed_axes_of(P, o)) {                      // a typed selection lies on its domain's one axis (a frequency line made by its key)
      const Json ax = axis_records(unit, P, made, o, "domain", {}, Json());
      s.manifold_id = ax.at(0).value("manifold_id", std::string());
      if (s.manifold_id.empty()) bad(o, "typed axes: the selection's axis is not a manifold");
    } else {
      s.manifold_id = link_of(P, spec, "on", &made, o);
    }
    if (d.contains("params")) s.params = d["params"];
    if (d.contains("description")) s.description = d["description"].get<std::string>();
    for (const auto& e : d.value("entries", Json::array())) s.entries.push_back(e.get<std::string>());
    if (t == "indices") for (const double c : a.data) s.codes.push_back(static_cast<std::int64_t>(c));
    else if (t == "partition" || t == "set") {      // a label per element: compute maps the labels to member numbers ONCE (one member per distinct label); io stores the numbers
      std::vector<std::int32_t> labels;
      for (const double c : a.data) labels.push_back(static_cast<std::int32_t>(c));
      nxr::selection::Lookup names;
      for (std::size_t k = 0; k < s.entries.size(); ++k) names.push_back(nxr::selection::Entry{static_cast<std::int64_t>(k), s.entries[k], std::nullopt});
      try {
        const nxr::selection::Selection set = nxr::selection::from_labels(labels, names);
        s.member_of = set.member_of;
        s.member_codes = set.member_codes;
      } catch (const std::invalid_argument& e) { bad(o, e.what()); }
    }
    else if (t == "kernels") {                     // a KERNEL SET: the node describes each member's anchor and profile
      for (const auto& k : d.value("kernels", Json::array())) {
        task::KernelOut x;
        x.start = k.at("start").get<double>(); x.stop = k.at("stop").get<double>(); x.code = k.at("code").get<std::int64_t>();
        x.family = k.at("family").get<std::string>(); x.chart = k.at("chart").get<std::string>(); x.gain = k.value("gain", 1.0);
        for (const auto& [pk, pv] : k.value("params", Json::object()).items()) x.params[pk] = pv.get<double>();
        s.kernels.push_back(std::move(x));
      }
      if (s.kernels.empty()) bad(o, "a kernel set needs its node's kernels (describe.kernels)");
    }
    else if (t == "grid") s.grid = grid_of(a, d, o);
    else if (t == "product") {
      // A PRODUCT of selections (the Studio's D100 productSelection; mixed radix, first factor fastest): a `set` with nothing stored per member; its factors in order, each a stored selection and, when it is a
      // level of a grid (the line's time grid, the surface's dyadic grid), the LEVEL it is at with the quantity it was snapped to (consolidation spec 6.2: coupled levels are never selections of their own).
      // The node's description gives the layout (`describe.layout`: product | joint) and the radices.
      std::size_t fk = 0;
      for (const auto& f : spec["factors"]) {
        int fac_level = -1;
        Json fac_snapped;
        s.factor_ids.push_back(link(P, f.is_object() ? f.value("selection", std::string()) : f.get<std::string>(), &made, o));
        // the level a node COUPLED (select.product `couple`: a band or a scale resolved in closed form at the run) is in the node's own record of the factor
        const Json* nodes_product = d.contains("product") ? &d["product"] : (d.contains("params") && d["params"].is_object() && d["params"].contains("product") ? &d["params"]["product"] : nullptr);
        if (nodes_product && nodes_product->contains("factors") && fk < (*nodes_product)["factors"].size()) {
          const Json& rf = (*nodes_product)["factors"][fk];
          bool coupled = false;
          for (const auto& cp : nodes_product->value("coupled", Json::array())) coupled = coupled || (cp.is_object() && cp.value("factor", std::size_t(~0)) == fk);
          if (coupled && rf.is_object() && rf.contains("level") && rf["level"].is_number_integer() && !(f.is_object() && f.contains("level"))) fac_level = rf["level"].get<int>();
          if (rf.is_object() && rf.contains("snapped") && !(f.is_object() && f.contains("snapped"))) fac_snapped = rf["snapped"];
        }
        ++fk;
        if (f.is_object() && f.contains("level")) fac_level = static_cast<int>(f["level"].is_string() && !f["level"].get<std::string>().empty() && f["level"].get<std::string>()[0] == '$'
                                                                                 ? P.parameters.at(f["level"].get<std::string>().substr(1)).get<std::int64_t>() : f["level"].get<std::int64_t>());
        if (f.is_object() && f.contains("snapped")) fac_snapped = f["snapped"];
        s.factor_levels.push_back(fac_level);
        s.params["factors_snapped"].push_back(fac_snapped);
      }
      if (d.contains("layout")) s.params["layout"] = d["layout"];
      if (d.contains("radices")) s.params["radices"] = d["radices"];
      if (d.contains("params") && d["params"].is_object() && d["params"].contains("product")) {      // the node's own record (select.product): the layout and each factor's radix, the member count of its level
        const Json& np = d["params"]["product"];
        if (!s.params.contains("layout") && np.contains("layout")) s.params["layout"] = np["layout"];
        if (!s.params.contains("radices") && np.contains("radices")) s.params["radices"] = np["radices"];
        if (s.radices.empty()) for (const auto& r : (np.contains("counts") ? np["counts"] : np.value("radices", Json::array()))) s.radices.push_back(r.get<std::int64_t>());   // the member count of each factor's level (io needs it for a level of a grid)
      }
      if (d.contains("product")) s.params["product"] = d["product"];       // the node's own record of each factor's level and snapped quantity, as it resolved them
    }
    else if (t == "cells") { s.n_members = d.at("n_members").get<std::int64_t>(); if (d.contains("label")) s.label = d["label"].get<std::string>(); }
    else if (t == "spans") for (const auto& x : d.value("spans", Json::array())) { task::Span sp; sp.start = x["start"].get<double>(); sp.stop = x["stop"].get<double>(); if (x.contains("code")) sp.code = x["code"].get<std::int64_t>(); sp.ordinal = static_cast<std::int64_t>(s.spans.size()); s.spans.push_back(sp); }
    else if (!d.contains("tiling")) bad(o, "its node gives no tiling (describe.tiling)");
    else s.tiling = tiling_of(P, d["tiling"], spec["on"].get<std::string>(), o);
    // THE GLUED CORTEX: a selection made on the derived glued surface is stored on the ORIGINAL (the manifold of record), its elements restricted back through the glue map (the bridge dropped)
    if (const auto g = d.value("glue_map", false) ? std::optional<GlueMap>() : Bound(sub, nullptr).glue_of(s.manifold_id)) {   // the bridge itself lies on the glued surface and stays there
      const std::int64_t n_original = original_vertices(sub, *g, o);
      bool restricted = false;
      if ((t == "partition" || t == "set") && static_cast<std::int64_t>(s.member_of.size()) == g->n_glued()) { s.member_of = restrict_members(s.member_of, *g, n_original); restricted = true; }
      else if (t == "indices") {
        std::vector<std::int64_t> kept;
        for (const std::int64_t c : s.codes) if (c >= 0 && c < g->n_glued() && g->original_of[static_cast<std::size_t>(c)] >= 0) kept.push_back(g->original_of[static_cast<std::size_t>(c)]);
        s.codes = std::move(kept); restricted = true;
      } else if (t == "grid" && s.grid) {
        for (task::GridLevelOut& L : s.grid->levels) if (static_cast<std::int64_t>(L.codes.size()) == g->n_glued()) { L.codes = restrict_members(L.codes, *g, n_original); restricted = true; }
      } else if (t == "product" || t == "spans" || t == "tiling" || t == "ladder" || t == "cells") {
        // not per vertex: nothing to restrict (a product of grids names its factors; spans lie on a line)
      } else {
        bad(o, "a " + t + " selection on the glued surface cannot be restricted back to the original (the bridge has no counterpart); make it on the original");
      }
      if (restricted) {
        s.manifold_id = g->original_id;
        if (!s.params.is_object()) s.params = Json::object();
        s.params["restricted_from"] = restricted_record(*g);
      }
    }
    const std::string path = unit.path_of(s);
    // A MAKER'S SELECTION, GET-OR-CREATE (`get_or_create: true`: a `make:` input stored): found by its maker key (the maker's name, its resolved parameters and its inputs' ids) it is reused; another one at the
    // same path is another job's, refused (or replaced with --replace). The key is in the selection's parameters.
    const std::string mkey = nxr::io::sha256_hex(maker_key(P, o).dump());
    if (!s.params.is_object()) s.params = Json::object();
    s.params["maker_key"] = mkey;
    if (spec.value("get_or_create", false))                                   // found by its maker key it is reused even under --replace: it is the same selection, and measurements lie over it
      if (const auto row = sub.find("selection", path)) {
        const Json pj = Json::parse(nxr::io::opt_text(*row, "params_json").value_or("{}"), nullptr, false);
        // a time grid IS its closed form (origin, width, hop, count, size and the band it is named for): the same grid made by another job (`select_time_grid`, a preset's maker) is the one found, not another
        const bool same_grid = pj.is_object() && pj.contains("grid") && s.params.contains("grid") && pj["grid"] == s.params["grid"] && pj.value("band", Json()) == s.params.value("band", Json());
        if (pj.is_object() && (pj.value("maker_key", std::string()) == mkey || same_grid)) {
          report["id"] = text(*row, "id"); report["path"] = path; report["reused"] = true;
          made[o] = report["id"].get<std::string>();
          return report;
        }
      }
    clear(task::Cls::selection, path);
    report["id"] = unit.selection(s); report["path"] = path;
    made[o] = report["id"].get<std::string>();
    return report;
  }

  // a statistic
  task::StatisticOut st;
  st.name = name; st.output = o;
  st.measure = d.value("measure", std::string()); st.method = d.value("method", std::string());
  if (st.measure.empty() || st.method.empty()) bad(o, "its node describes no measure or method");
  // a statistic OF a measurement the run stores (a summary of its members): it is of the measurement's own field (found, or stored before it)
  const std::string of = statistic_of(P, spec, o, false);
  // OF A GROUP RUN'S PER-SUBJECT INPUT (stacked over the members): what it is of lies in every member's own package, never in the group's, so the statistic is the group's own row (detached: no
  // link into a member's package; the members and each one's inputs are in its provenance, `group`)
  const bool of_members = P.found.count(of) && P.found.at(of).facts.is_object() && P.found.at(of).facts.contains("stack");
  const bool of_field = !P.found.count(of) || P.found.at(of).cls == "field";
  const std::string of_id = of_members ? std::string() : link(P, of, &made, o);
  if (of_members) st.detached = true;
  else if (of_field) st.of_field_id = of_id;
  else st.of_statistic_id = of_id;
  const std::string acq = of_members ? std::string() : acquisition_of(P, of);
  const bool typed_st = typed_axes_of(P, o) && !spec.contains("axes") && !d.contains("axes");   // a typed node's axes are its domain's
  const Json axes_desc = d.value("axes", Json::array());
  Json axes_decl = spec.value("axes", Json::array());
  if (!spec.contains("axes")) for (const auto& ax : axes_desc) axes_decl.push_back(Json{{"name", ax.value("name", std::string())}});   // none declared: the node's own axes, as it describes them
  if (axes_decl.size() != axes_desc.size()) bad(o, "declares " + std::to_string(axes_decl.size()) + " axes, its node describes " + std::to_string(axes_desc.size()));
  std::vector<std::int64_t> shape;
  Json axes = Json::array();
  for (std::size_t i = 0; i < axes_decl.size(); ++i) {
    const Json& decl = axes_decl[i];
    Json ax = axes_desc[i];
    if (decl["name"] != ax.value("name", Json())) bad(o, "axis " + std::to_string(i) + " is declared '" + decl["name"].get<std::string>() + "', its node names it " + ax.value("name", Json()).dump());
    const std::int64_t size = ax.value("size", std::int64_t(0));
    Json out = Json{{"name", decl["name"]}, {"size", size}};
    if (decl.contains("selection")) {                                     // an axis over a pick another output of the run made (a covariance over its channels)
      const std::string so = decl["selection"].get<std::string>();
      if (made.count(so)) out["selection_id"] = made.at(so);
      else if (P.found.count(so) && P.found.at(so).row) out["selection_id"] = text(*P.found.at(so).row, "id");     // or a selection found as an input (the tiles a repeat is per)
      else bad(o, "axis " + decl["name"].get<std::string>() + ": selection " + so + " is neither an output stored before it (list it first) nor an input found");
      for (const char* key : {"members", "values"}) if (ax.contains(key)) out[key] = ax[key];
    } else if (decl.contains("on")) {
      const std::string axis_manifold = link(P, decl["on"].get<std::string>(), &made, o);
      // an axis of PARTS of each element of the manifold it is linked to (the fused Hodge-Dirac weights: per mode, its normal-current and its flow part): the rows run part-major, size = parts x elements.
      // An axis names what it runs along by ONE of manifold_id | values (io's rule): its own entries (a part and a mode each) carry the stored mode axis they are of; an axis with none is that manifold's
      if (ax.contains("values") && ax["values"].is_array()) {
        Json vals = ax["values"];
        for (auto& v : vals) if (v.is_object()) v["manifold_id"] = axis_manifold;
        out["values"] = vals;
      } else {
        out["manifold_id"] = axis_manifold;
      }
      if (ax.contains("members")) out["members"] = ax["members"];        // a part of the manifold (the modes a selection keeps): its elements, by index
      if (decl.contains("parts")) out["parts"] = decl["parts"];
      else if (ax.contains("parts")) out["parts"] = ax["parts"];
    } else if (decl.contains("line")) {
      if (!ax.contains("line")) bad(o, "axis " + decl["name"].get<std::string>() + " declares a line; its node describes none");
      const Json& l = ax["line"];
      if (decl.contains("of_time") && l.contains("fft")) {              // the frequency line of a span of this time line (the node says it is the full one-sided line of n samples at fs): keyed by (time line, n, fs), made by the FFT operator (io's frequency_line); `line` is the old name, still found
        const std::string time_line = link(P, decl["of_time"].get<std::string>(), &made, o);
        out["manifold_id"] = unit.frequency_line(time_line, l["fft"].value("n", std::int64_t(0)), l["fft"].value("fs", 0.0), nxr::io::safe_name(fill(P, decl["line"].get<std::string>())));
        out["unit"] = l.value("unit", std::string("Hz"));
        if (ax.contains("members")) out["members"] = ax["members"];      // a part of the line (the bins a span keeps): its elements, by index
        axes.push_back(out);
        shape.push_back(size);
        continue;
      }
      task::LineOut lo;
      lo.name = nxr::io::safe_name(fill(P, decl["line"].get<std::string>())); lo.alt = lo.name + "_alt";
      lo.n = size; lo.spacing = l.value("spacing", 1.0); lo.origin = l.value("origin", 0.0);
      lo.unit = l.value("unit", std::string("Hz")); lo.dimension = l.value("dimension", std::string("frequency"));
      lo.folder = l.value("folder", lo.dimension == "time" ? std::string("time") : std::string("spectral"));
      lo.cs_key = l.value("cs", lo.unit == "s" ? std::string("clock") : std::string("hz"));
      if (!acq.empty()) lo.acquisition_id = acq;
      out["manifold_id"] = unit.line(lo);
      out["unit"] = lo.unit;
    } else {
      for (const char* key : {"values", "edges", "scale", "unit", "selection_id", "bands", "members"}) if (ax.contains(key)) out[key] = ax[key];
    }
    axes.push_back(out);
    shape.push_back(size);
  }
  if (typed_st) {                                                          // sizes from the node's shape (axes first, then the value's components), else the array's
    const std::vector<std::int64_t> sizes = d.contains("shape") ? ints(d["shape"]) : (a.cols > 1 ? std::vector<std::int64_t>{a.rows, a.cols} : std::vector<std::int64_t>{a.rows});
    axes = axis_records(unit, P, made, o, "domain", sizes, Json());
    shape = sizes;
    for (const auto& ax : axes)                                             // the partition it is computed from: its members' axis
      if (!st.partition_selection_id && ax.contains("selection_id") && !ax.contains("manifold_id")) { st.partition_selection_id = ax["selection_id"].get<std::string>(); if (ax.contains("size")) st.n_members = ax["size"].get<std::int64_t>(); }
  }
  st.axes = axes;
  if (spec.contains("over")) {                                              // a statistic over the members of a found selection (a summary of measurements over it)
    const std::string ov = spec["over"].get<std::string>();
    if (made.count(ov)) st.partition_selection_id = made.at(ov);
    else if (P.found.count(ov) && P.found.at(ov).row) st.partition_selection_id = text(*P.found.at(ov).row, "id");
    else bad(o, "over: '" + ov + "' is neither a selection stored before it nor an input found");
    if (d.contains("n_members")) st.n_members = d["n_members"].get<std::int64_t>();
    if (d.contains("members_skipped")) st.members_skipped = d["members_skipped"].get<std::int64_t>();
  }
  if (spec.contains("partition")) {
    const Json& p = spec["partition"];
    if (!d.contains("tiling")) bad(o, "declares a partition; its node describes no tiling");
    task::SelectionOut seg;
    seg.name = nxr::io::safe_name(fill(P, p["tiling"].get<std::string>())); seg.output = o; seg.type = "tiling";
    seg.manifold_id = link_of(P, p, "on", &made, o);
    seg.tiling = tiling_of(P, d["tiling"], p["on"].get<std::string>(), o);
    if (of_field && !of_members) seg.of_field_id = of_id;
    if (!acq.empty()) seg.acquisition = acq;
    const std::int64_t n = seg.tiling->count + seg.tiling->skipped;
    seg.description = "the " + std::to_string(n) + " segments of " + (P.found.count(of) && P.found.at(of).row ? text(*P.found.at(of).row, "name") : of) + " behind " + name + " (" + std::to_string(seg.tiling->skipped) + " left out)";
    const std::string spath = unit.path_of(seg);
    if (const auto again = made.find("partition:" + spath); again != made.end()) st.partition_selection_id = again->second;   // several statistics of one node share their partition
    else {
      clear(task::Cls::selection, spath);
      st.partition_selection_id = unit.selection(seg);
      made["partition:" + spath] = *st.partition_selection_id;
    }
    report["partition"] = Json{{"name", seg.name}, {"path", spath}, {"id", *st.partition_selection_id}};
    st.n_members = d.value("n_members", seg.tiling->count);
    st.members_skipped = d.value("members_skipped", seg.tiling->skipped);
  } else {
    if (d.contains("n_members")) st.n_members = d["n_members"].get<std::int64_t>();
    if (d.contains("members_skipped")) st.members_skipped = d["members_skipped"].get<std::int64_t>();
  }
  if (d.contains("components")) st.components = d["components"];
  if (d.contains("scaling")) st.scaling = d["scaling"];
  if (d.contains("dof") && d["dof"].is_number()) st.dof = d["dof"].get<double>();
  st.options = d.value("options", Json::object());
  if (d.contains("unit") && d["unit"].is_string()) st.unit = d["unit"].get<std::string>();
  if (d.contains("description")) st.description = d["description"].get<std::string>();
  st.label = d.value("label", st.method);
  if (d.contains("producer") && d["producer"].is_object()) st.producer = d["producer"];   // what the node records beside the provenance (a spectrum's resolution and rate: its readers ask for them)
  if (!acq.empty()) st.acquisition = acq;
  if (!span.is_null()) st.span = span;
  st.value_type = d.value("value_type", std::string("scalar"));
  if (d.contains("shape")) shape = ints(d["shape"]);                       // a value with components (complex, hermitian3, quaternion): the axes, then the components
  st.data = values_of(a, shape, single_of(d), o);
  const std::string path = unit.path_of(st);
  clear(task::Cls::statistic, path);
  report["id"] = unit.statistic(st); report["path"] = path;
  made[o] = report["id"].get<std::string>();
  return report;
}

}  // namespace nxr::pipelines::graph
