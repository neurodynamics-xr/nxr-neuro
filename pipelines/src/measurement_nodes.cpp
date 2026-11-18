// nxr-pipelines — THE MEASUREMENT NODES (measurement_nodes.h; spec docs/specs/2026-11-17-consolidation.md, sections 2 and 4.1).
//
// Each node is a thin wrapper of ONE compute implementation (compute/include/nxr/measurement/): it resolves the selection it is handed to the form compute takes (a Rule on a time
// line, a MemberOfElement on a surface, a ProductForm of them), folds the blocks the chunk service delivers, and describes what it made. It decides no data: the selection, the
// measures and the weights are the caller's. The ingredients (counts, weights, sums, maxima) are what is stored and what rolls up; the measures are derived from them at read
// (ratios never reduced themselves). Each measure's merge rule is DATA in the measures registry (measures.h), applied here in memory and by io in SQL.
//
// MEMORY (stated): `reduce` holds the ingredients of the run: rows x members doubles a few times over (members = the selection's members, a product's code space); never the signal.
#include "nxr/pipelines/measurement_nodes.h"

#include <algorithm>
#include <any>
#include <cmath>
#include <complex>
#include <memory>
#include <optional>
#include <limits>
#include <map>
#include <set>

#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/select/dyadic.h"
#include "nxr/measurement/estimate.h"
#include "nxr/measurement/measures.h"
#include "nxr/measurement/reduce.h"
#include "nxr/measurement/roll_up.h"
#include "nxr/measurement/summarize.h"
#include "nxr/quantity/quantity.h"
#include "nxr/select/events.h"
#include "nxr/select/grid.h"
#include "nxr/select/partition.h"
#include "nxr/selection/resolved.h"
#include "nxr/selection/selection.h"

namespace nxr::pipelines::graph {

namespace nm = nxr::measurement;
namespace ns = nxr::selection;

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
Value made(const Value& x) { Value v = x; v.row.reset(); return v; }

// ── the parts of a combined step ─────────────────────────────────────────────────────────────────────────────────
struct Wrapped { std::string which; std::any inner; };

const StepSpec& part_of(const std::vector<std::pair<std::string, StepSpec>>& parts, const std::string& which, const std::string& name, const std::string& param) {
  for (const auto& [k, s] : parts) if (k == which) return s;
  std::string list;
  for (const auto& [k, s] : parts) list += (list.empty() ? "" : ", ") + k;
  refuse(name + ": " + param + " " + (which.empty() ? std::string("is not given") : "'" + which + "' is not one of") + " (" + list + ")");
}
}  // namespace

StepSpec take_step(std::vector<StepSpec>& r, const std::string& name) {
  for (std::size_t i = 0; i < r.size(); ++i)
    if (r[i].name == name) { StepSpec s = std::move(r[i]); r.erase(r.begin() + static_cast<std::ptrdiff_t>(i)); return s; }
  refuse("add_measurement_nodes: no step '" + name + "' to take");
}

StepSpec combine_by_param(const std::string& name, const std::string& param, std::vector<std::pair<std::string, StepSpec>> parts, const std::string& description,
                          std::function<Json(const Json&, const std::string&)> adapt, std::function<std::string(const Values&, const Json&)> choose) {
  need(!parts.empty(), name + ": no parts");
  auto shared = std::make_shared<std::vector<std::pair<std::string, StepSpec>>>(std::move(parts));
  StepSpec s;
  s.name = name;
  s.kind = shared->front().second.kind;
  s.description = description;
  s.traits = shared->front().second.traits;
  std::string matches;
  Json params = Json::object();
  params[param] = nullptr;
  std::set<std::string> inputs_seen, outputs_seen;
  for (const auto& [k, p] : *shared) {
    need(p.kind == s.kind, name + ": the parts are of different kinds (" + p.kind + " and " + s.kind + ")");
    for (const auto& i : p.inputs) if (inputs_seen.insert(i).second) s.inputs.push_back(i);
    for (const auto& o : p.outputs) if (outputs_seen.insert(o).second) s.outputs.push_back(o);
    for (auto it = p.parameters.begin(); it != p.parameters.end(); ++it) if (!params.contains(it.key())) params[it.key()] = it.value();
    matches += (matches.empty() ? "" : "; ") + k + ": " + p.matches;
    s.reads_arrays = s.reads_arrays || p.reads_arrays;
  }
  // a port is optional unless every part names it and requires it
  for (const auto& i : s.inputs) {
    bool required_by_all = true;
    for (const auto& [k, p] : *shared) {
      const bool has = std::find(p.inputs.begin(), p.inputs.end(), i) != p.inputs.end();
      const bool opt = std::find(p.optional.begin(), p.optional.end(), i) != p.optional.end();
      if (!has || opt) required_by_all = false;
    }
    if (!required_by_all) s.optional.push_back(i);
  }
  s.parameters = params;
  s.matches = matches;
  const std::string param_name = param;
  if (!choose)                                                           // the part is named by the parameter itself: `out` names that part's outputs only
    s.outputs_for = [shared, param_name](const Json& with) {
      std::vector<std::string> all;
      std::set<std::string> seen;
      for (const auto& [k, p] : *shared) for (const auto& o : p.outputs) if (seen.insert(o).second) all.push_back(o);
      if (!with.is_object() || !with.contains(param_name) || !with[param_name].is_string()) return all;
      for (const auto& [k, p] : *shared) if (k == with[param_name].get<std::string>()) return p.outputs;
      return all;
    };
  auto sees = [adapt](const Json& p, const std::string& which) { return adapt ? adapt(p, which) : p; };
  auto pick = [param_name, choose](const Values& ins, const Json& p) -> std::string {
    if (choose) return choose(ins, p);
    return p.contains(param_name) && p[param_name].is_string() ? p[param_name].get<std::string>() : std::string();
  };
  s.plan = [shared, name, param_name, sees, pick](const Values& ins, const Json& p, const PlanContext& c) {
    const std::string which = pick(ins, p);
    const StepSpec& part = part_of(*shared, which, name, param_name);
    return part.plan(ins, sees(p, which), c);
  };
  s.prepare = [shared, name, param_name, sees, pick](const Values& ins, const Json& p, ExecContext& c) {
    const std::string which = pick(ins, p);
    const StepSpec& part = part_of(*shared, which, name, param_name);
    return std::any(Wrapped{which, part.prepare ? part.prepare(ins, sees(p, which), c) : std::any()});
  };
  s.resolve = [shared, name, param_name, sees, pick](const Values& ins, const Json& p, const Bound& b) {
    const std::string which = pick(ins, p);
    const StepSpec& part = part_of(*shared, which, name, param_name);
    Json j = part.resolve ? part.resolve(ins, sees(p, which), b) : Json::object();
    if (j.is_object()) j[param_name] = which;   // the chosen part is in the cache key
    return j;
  };
  auto inner = [shared, name, param_name](const std::any& st) -> std::pair<const StepSpec&, const std::any&> {
    const Wrapped& w = std::any_cast<const Wrapped&>(st);
    return {part_of(*shared, w.which, name, param_name), w.inner};
  };
  s.record = [inner](const std::any& st) { auto [p, in] = inner(st); return p.record ? p.record(in) : Json::object(); };
  s.halo = [inner](const std::any& st) { auto [p, in] = inner(st); return p.halo ? p.halo(in) : std::int64_t(0); };
  s.halo_at_plan = [shared, name, param_name, sees, pick](const Values& ins, const Json& p) {
    const std::string which = pick(ins, p);
    const StepSpec& part = part_of(*shared, which, name, param_name);
    return part.halo_at_plan ? part.halo_at_plan(ins, sees(p, which)) : std::int64_t(0);
  };
  s.chunk = [inner](const std::any& st, const Blocks& ins, std::int64_t keep) { auto [p, in] = inner(st); return p.chunk(in, ins, keep); };
  s.start = [inner](const std::any& st) { auto [p, in] = inner(st); return p.start(in); };
  s.fold = [inner](const std::any& st, std::any& acc, const Blocks& ins, const Place& at) { auto [p, in] = inner(st); p.fold(in, acc, ins, at); };
  s.finish = [inner](const std::any& st, std::any& acc) { auto [p, in] = inner(st); return p.finish(in, acc); };
  s.run = [inner](const std::any& st, const Arrays& ins, ExecContext& c) { auto [p, in] = inner(st); return p.run(in, ins, c); };
  // methods a part does not have stay null in the combined step only when no part has them
  auto any_has = [&](auto member) { for (const auto& [k, p] : *shared) if (p.*member) return true; return false; };
  if (!any_has(&StepSpec::chunk)) s.chunk = nullptr;
  if (!any_has(&StepSpec::start)) s.start = nullptr;
  if (!any_has(&StepSpec::fold)) s.fold = nullptr;
  if (!any_has(&StepSpec::finish)) s.finish = nullptr;
  if (!any_has(&StepSpec::run)) s.run = nullptr;
  if (!any_has(&StepSpec::prepare)) s.prepare = nullptr;
  return s;
}

namespace {

// ── field.form: `outer` (z zᴴ per sample, the general A M Aᴴ algebra's pointwise part) ───────────────────────────────
// A pointwise chunk step: x complex [M x n] -> y complex [M*M x n], row a*M + b = z_a conj(z_b). The reduction over a time grid is `reduce` (measure `sum`), never here.
StepSpec outer_form() {
  StepSpec s;
  s.name = "outer";
  s.kind = "chunk";
  s.inputs = {"x"};
  s.outputs = {"y"};
  s.parameters = Json::object();
  s.matches = "the Studio's fieldForms.ts `outer` (complex-extended); the channels' cross-spectral density of a window is its mean (nxr::signal::spectral::CrossWelch)";
  s.description = "z zᴴ per sample: the outer product of the channels' analytic signal (rows a * M + b = z_a conj z_b), pointwise";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float32, OutputPrecision::follows_input, "per-sample outer product in the block's own precision");
  s.plan = [](const Values& ins, const Json&, const PlanContext&) {
    const Value& x = ins.at("x");
    need(x.cls == "field" && x.time && x.chunked, "field.form outer: x must be a field on ... x time, inside the chunk loop");
    need(x.complex, "field.form outer: x must be complex (the analytic signal): z zᴴ of a real field is its square");
    Value y = made(x);
    y.facts["nonlinear"] = true;
    y.facts["made_by"] = "outer";
    y.facts["outer_of"] = x.manifolds.empty() ? std::string() : x.manifolds[0];
    return Values{{"y", y}};
  };
  s.chunk = [](const std::any&, const Blocks& ins, std::int64_t keep) {
    const Block& b0 = ins.at("x");
    const std::int64_t kl = std::min(keep, b0.left), kr = std::min(keep, b0.right), w = b0.own() + kl + kr, c0 = b0.left - kl, M = b0.rows;
    Block y{M * M, w, kl, kr, std::vector<float>(static_cast<std::size_t>(M * M * w)), std::vector<float>(static_cast<std::size_t>(M * M * w))};
    for (std::int64_t a = 0; a < M; ++a)
      for (std::int64_t c = 0; c < M; ++c)
        for (std::int64_t t = 0; t < w; ++t) {
          const std::size_t ia = static_cast<std::size_t>(a * b0.width + c0 + t), ic = static_cast<std::size_t>(c * b0.width + c0 + t), o = static_cast<std::size_t>((a * M + c) * w + t);
          const float ar = b0.re[ia], ai = b0.im[ia], cr = b0.re[ic], ci = b0.im[ic];
          y.re[o] = ar * cr + ai * ci;                       // Re(z_a conj z_c)
          y.im[o] = ai * cr - ar * ci;                       // Im(z_a conj z_c)
        }
    return Blocks{{"y", std::move(y)}};
  };
  return s;
}

// What a stored measurement is told by its node (declared_outputs.h, `describe`): [rows x members x components], one named column a component (a vector measure is stored as `<measure>_x/_y/_z`, as
// the registry's phase_current_x ...), so io's one writer (Unit::measurements) lays each out the way the Studio's array-backed measurements are, found by the measurement's key.
Json describe_measures(const Json& measures, std::int64_t R, std::int64_t T, std::int64_t C) {
  Json cols = Json::array();
  for (const auto& m : measures) {
    const std::string name = m.value("measure", std::string());
    const int k = m.value("components", 1);
    if (k == 3) for (const char* ax : {"_x", "_y", "_z"}) cols.push_back(Json{{"measure", name + ax}});
    else cols.push_back(Json{{"measure", name}});
  }
  return Json{{"measures", cols}, {"shape", Json::array({R, T, C})}, {"values_f32", false}};
}

// ── reduce ───────────────────────────────────────────────────────────────────────────────────────────────────────
// The selection a reduction is over, read from the record of the stored selection (its params_json) or from the description of the one made earlier in the run (`ExecContext::made`,
// the typed edge). One descriptor schema (written by select_nodes.cpp, read here):
//   time      {kind: time, origin, width, hop, count, size}   a time grid as a closed-form Rule on the line's samples (a level of the line's dyadic grid is width = hop = 2^k)
//   dyadic    {kind: dyadic, surface, level}                  a surface grid at a level: the member of each vertex
//   elements  {kind: elements, size}                          each element its own member (the channels)
//   span      {kind: span}                                    the run's span as ONE member
//   product   {kind: product, factors: [descriptor, ...], layout: "product" | "joint", bits}   mixed radix, the FIRST factor fastest (Studio D100; D129 joint: first factor's radix 2^bits)
// a STORED product's factors, as io keeps them (params.product: [{selection, members, level?}], the Studio's D100) read back as the descriptors the run's own product records: each factor's selection row says its grid
// (a time grid, a surface ladder at the level named, the run's span, a set of elements)
Json product_of_entries(const Json& entries, const ExecContext& c, const std::string& who) {
  Json out = Json::array();
  for (const auto& e : entries) {
    const auto row = c.bound.selection(e.value("selection", std::string()));
    need(row.has_value(), who + ": a factor of the stored product (" + e.dump() + ") is not among the subject's selections");
    const Json fp = Json::parse(nxr::io::opt_text(*row, "params_json").value_or("{}"), nullptr, false);
    const std::string nm = nxr::io::text(*row, "name");
    Json d;
    if (fp.is_object() && fp.contains("grid") && fp["grid"].is_object()) {
      d = fp["grid"];
      if (d.value("kind", std::string()) == "dyadic") {
        const auto at = nm.rfind(" tiles");
        need(at != std::string::npos && e.contains("level"), who + ": the surface factor " + nm + " of the stored product names no level");
        d["surface"] = nm.substr(0, at); d["level"] = e["level"]; d["tiles"] = e["members"];
      } else if (d.value("kind", std::string()) == "time" && !d.contains("size") && d.contains("samples")) d["size"] = d["samples"];
    } else if (fp.is_object() && fp.value("kind", std::string()) == "elements") d = Json{{"kind", "elements"}, {"size", e.value("members", std::int64_t(0))}};
    else if (fp.is_object() && fp.contains("level") && fp.contains("tiles")) d = Json{{"kind", "dyadic"}, {"surface", fp.value("surface", std::string())}, {"level", fp["level"]}, {"tiles", fp["tiles"]}};
    else refuse(who + ": the factor " + nm + " of the stored product records no grid");
    out.push_back(d);
  }
  return out;
}

Json descriptor_of(const Value& over, const ExecContext& c, const std::string& port, const std::string& who) {
  if (over.row) {
    const Json pj = Json::parse(nxr::io::opt_text(*over.row, "params_json").value_or("{}"), nullptr, false);
    if (pj.is_object() && pj.contains("product") && pj["product"].is_array())
      return Json{{"kind", "product"}, {"factors", product_of_entries(pj["product"], c, who)}, {"layout", pj.value("layout", std::string("product"))}, {"bits", pj.value("bits", 0)}};
    if (!(pj.is_object() && (pj.contains("grid") || pj.contains("product"))) && !nxr::io::is_null(*over.row, "array_path") && !nxr::io::is_null(*over.row, "n_members"))   // a set or partition: a member number per element
    {
      Json d = Json{{"kind", "labelled"}, {"array_path", nxr::io::text(*over.row, "array_path")}, {"members", nxr::io::integer(*over.row, "n_members")}};
      if (const auto m = nxr::io::is_null(*over.row, "manifold_id") ? std::nullopt : c.bound.manifold(nxr::io::text(*over.row, "manifold_id")))
        if (nxr::io::text(*m, "type") == "surface") d["surface"] = nxr::io::text(*m, "name");     // the surface its elements are the vertices of (a weight by vertex area)
      return d;
    }
    need(pj.is_object() && (pj.contains("grid") || pj.contains("product")), who + ": the selection " + nxr::io::text(*over.row, "name") + " records no grid or product (params.grid / params.product)");
    if (pj.contains("product")) return Json{{"kind", "product"}, {"factors", pj["product"]["factors"]}, {"layout", pj["product"].value("layout", std::string("product"))}, {"bits", pj["product"].value("bits", 0)}};
    Json grid = pj["grid"];
    if (pj.contains("band")) grid["band"] = pj["band"];                    // the band the grid is named for (rule band, frames): what its tiles resolve
    return grid;
  }
  need(c.made && c.made->count(port), who + ": `" + port + "` was not made earlier in the run and is not stored");
  const Json& d = c.made->at(port).meta.value("describe", Json::object());
  const Json& p = d.value("params", Json::object());
  need(p.contains("grid") || p.contains("product"), who + ": the selection made for `" + port + "` records no grid or product");
  if (p.contains("product")) return Json{{"kind", "product"}, {"factors", p["product"]["factors"]}, {"layout", p["product"].value("layout", std::string("product"))}, {"bits", p["product"].value("bits", 0)}};
  Json grid = p["grid"];
  if (p.contains("band")) grid["band"] = p["band"];
  return grid;
}

struct Form {
  std::string kind;                                  // time | dyadic | elements | span
  ns::Rule rule;                                     // time
  std::vector<std::int32_t> member_of;               // dyadic: the tile of every vertex
  std::int64_t members = 0, size = 0;
};

Form form_of(const Json& d, std::int64_t line_samples, ExecContext& c, const std::string& who) {
  Form f;
  f.kind = d.value("kind", std::string());
  if (f.kind == "time") {
    f.rule.origin = d.value("origin", std::int64_t(0));
    f.rule.width = d.value("width", std::int64_t(0));
    f.rule.hop = d.value("hop", f.rule.width);
    f.rule.size = d.value("size", line_samples);
    f.rule.count = d.value("count", std::int64_t(0));
    need(f.rule.width > 0 && f.rule.hop > 0 && f.rule.count > 0, who + ": the time grid has no width, hop or members");
    f.members = f.rule.count;
  } else if (f.kind == "dyadic") {
    const nxr::pipelines::SurfaceInput S = c.bound.surface(d["surface"].get<std::string>(), true);
    const nxr::pipelines::SurfaceLevel L = c.bound.surface_level(S, d["level"].get<int>());
    need(!d.contains("tiles") || !d["tiles"].is_number_integer() || d["tiles"].get<std::int64_t>() == L.nodes,
         who + ": the surface factor's record says " + d.value("tiles", Json()).dump() + " tiles at level " + std::to_string(d["level"].get<int>()) + ", the surface's own level has " + std::to_string(L.nodes));
    f.member_of = L.tile;
    f.members = L.nodes;
    f.size = S.n_vertices();
  } else if (f.kind == "labelled") {                                      // a stored set / partition: the member number of every element (negative: in no member)
    f.member_of = c.bound.array_int32(d["array_path"].get<std::string>());
    f.members = d["members"].get<std::int64_t>();
    f.size = static_cast<std::int64_t>(f.member_of.size());
  } else if (f.kind == "elements") {
    f.size = d.value("size", std::int64_t(0));
    f.members = f.size;
    f.member_of.resize(static_cast<std::size_t>(f.size));
    for (std::int64_t i = 0; i < f.size; ++i) f.member_of[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
  } else if (f.kind == "whole") {                                         // all the elements, one member
    f.size = d.value("size", std::int64_t(0));
    f.members = 1;
    f.kind = "elements";
    f.member_of.assign(static_cast<std::size_t>(f.size), 0);
  } else if (f.kind == "span") {
    f.rule = ns::Rule{0, line_samples, line_samples, 1, line_samples};
    f.kind = "time";
    f.members = 1;
  } else refuse(who + ": a selection of kind '" + f.kind + "' is not one `reduce` folds (time, dyadic, elements, span, or a product of them)");
  return f;
}

int components_of(const std::string& measure) { return measure == "phase_current" ? 3 : 1; }

// ── the ingredients as an Array (what crosses a node's edge and what the store writes): the vectors of nm::Ingredients (measures.h), one after the other, their names and lengths in meta ─────
// Layout of each vector: [row][member] (sum: [row][member][component]); the integers (count, nonfinite) are held as doubles (exact to 2^53). `held` (the values a median keeps) is never stored or rolled up.
struct IngredientColumns {
  std::vector<std::pair<std::string, std::vector<double>*>> d;
  std::vector<std::pair<std::string, std::vector<std::int64_t>*>> i;
};
IngredientColumns columns_of(nm::Ingredients& g) {
  IngredientColumns c;
  c.i = {{"count", &g.count}, {"nonfinite", &g.nonfinite}};
  c.d = {{"weight", &g.weight}, {"sum_sq", &g.sum_sq}, {"sum_cube", &g.sum_cube}, {"sum_fourth", &g.sum_fourth}, {"sum_abs", &g.sum_abs}, {"sum_sqrt_abs", &g.sum_sqrt_abs},
         {"max", &g.max}, {"min", &g.min}, {"abs_max", &g.abs_max}, {"sum", &g.sum}};
  return c;
}
Array pack_ingredients(const nm::Ingredients& g_in, std::int64_t rows, std::int64_t members) {
  nm::Ingredients g = g_in;
  IngredientColumns c = columns_of(g);
  Array a;
  a.rows = rows; a.cols = members;
  Json lens = Json::array(), order = Json::array();
  for (auto& [name, v] : c.i) { for (const std::int64_t x : *v) a.data.push_back(static_cast<double>(x)); lens.push_back(v->size()); order.push_back(name); }
  for (auto& [name, v] : c.d) { a.data.insert(a.data.end(), v->begin(), v->end()); lens.push_back(v->size()); order.push_back(name); }
  a.meta["ingredient_lengths"] = lens;
  a.meta["ingredient_order"] = order;
  a.meta["ingredient_components"] = g.components;
  a.meta["ingredient_unassigned"] = g.unassigned;
  return a;
}
nm::Ingredients unpack_ingredients(const Array& a) {
  need(!a.meta.contains("imag_offset"), "the ingredients are of a complex field (real and imaginary parts folded alike): rolling up or summarizing a complex measurement is not built; reduce a real quantity of it (a form) for that");
  nm::Ingredients g;
  g.rows = a.rows; g.members = a.cols;
  g.components = a.meta.value("ingredient_components", std::int64_t(1));
  g.unassigned = a.meta.value("ingredient_unassigned", std::int64_t(0));
  const Json& lens = a.meta.value("ingredient_lengths", Json::array());
  const Json& order = a.meta.value("ingredient_order", Json::array());
  IngredientColumns c = columns_of(g);
  need(lens.is_array() && order.is_array() && lens.size() == order.size() && lens.size() == c.i.size() + c.d.size(), "the ingredients carry no layout (an array `reduce` made)");
  std::size_t off = 0;
  for (std::size_t k = 0; k < order.size(); ++k) {
    const std::string name = order[k].get<std::string>();
    const std::size_t n = lens[k].get<std::size_t>();
    need(off + n <= a.data.size(), "the ingredients' layout is longer than their array");
    bool found = false;
    for (auto& [nm_, v] : c.i) if (nm_ == name) { v->assign(n, 0); for (std::size_t j = 0; j < n; ++j) (*v)[j] = static_cast<std::int64_t>(std::llround(a.data[off + j])); found = true; }
    for (auto& [nm_, v] : c.d) if (nm_ == name) { v->assign(a.data.begin() + static_cast<std::ptrdiff_t>(off), a.data.begin() + static_cast<std::ptrdiff_t>(off + n)); found = true; }
    need(found, "the ingredients name an unknown column '" + name + "'");
    off += n;
  }
  return g;
}
// the value of a measure for (row, member): a vector measure's component c is read with `component`, never by a spelled name
double read_value(const nm::Ingredients& g, std::int64_t r, std::int64_t m, const std::string& measure, int component) {
  return nm::value(g, r, m, measure, nm::Registry::standard(), component);
}
// the requested measures read from the ingredients: [rows x members x components]
Array values_of(const nm::Ingredients& g, const nm::Request& req, std::int64_t rows, std::int64_t members) {
  std::int64_t C = 0;
  for (const auto& m : req.measures) C += components_of(m);
  Array y{rows, members, std::vector<double>(static_cast<std::size_t>(rows * members * C), kNaN)};
  y.comps = C;
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t m = 0; m < members; ++m) {
      if (!(nm::value(g, r, m, "count") > 0)) continue;
      std::int64_t off = 0;
      for (const auto& name : req.measures) {
        const int k = components_of(name);
        for (int c = 0; c < k; ++c) y.data[static_cast<std::size_t>((r * members + m) * C + off + c)] = read_value(g, r, m, name, c);
        off += k;
      }
    }
  Json comps = Json::array();
  for (const auto& m : req.measures) comps.push_back(Json{{"measure", m}, {"components", components_of(m)}});
  y.meta["measures"] = comps;
  return y;
}

struct ReduceState {
  nm::Request req;
  std::vector<std::string> measures;
  std::vector<int> components;                       // per measure: 1, or 3 for a vector
  std::vector<Form> factors;                         // one (a grid) or several (a product, the FIRST fastest)
  std::vector<std::int64_t> radices;
  bool product = false;
  bool complex = false;                              // x is complex: the imaginary part is folded beside the real one (measures linear in the value only)
  bool static_map = false;                           // the field has no time axis: reduced once over the members of a space selection (Q2)
  ns::ProductForm product_form;
  std::vector<double> weights;
  double rate = 0, origin = 0;
  std::int64_t members = 0;
  Json over;
};
struct ReduceAcc { nm::Ingredients ing, ing_im; std::int64_t rows = 0; };


StepSpec reduce_node() {
  StepSpec s;
  s.name = "measurement.reduce";
  s.kind = "accumulate";
  s.inputs = {"x", "over", "exclude", "weights"};
  s.optional = {"over", "exclude", "weights"};
  s.outputs = {"values", "ingredients"};
  s.parameters = Json{{"measures", nullptr}, {"weight", "none"}, {"bad_segments", "exclude"}};
  s.matches = "the Studio's backend/src/db/measurements.ts reduceTiles and measureOverPartition (the measure registry, the lookup); compute nxr::measurement::reduce";
  s.description =
      "A field reduced over the members of ANY selection (a time grid, a surface grid, a product of them, a channel set, the run's span as one member), per row, in ONE pass: the additive "
      "INGREDIENTS (count, weight, sum, sum of squares, max, min, largest magnitude; per component for a vector) are folded block by block, and the requested measures are read from them "
      "(`values`); the ingredients (`ingredients`) are what is stored and what rolls up (`roll_up`: a merge rule per measure, data in the measures registry). Ratios (an amplitude, an sd) are derived "
      "at read, never reduced themselves. A member cut by an excluded span or by the run's edge is NaN, never a short mean. `measures` has no default; `weight` is none, or the surface's barycentric "
      "(or voronoi) vertex areas for a member of a surface grid.";
  s.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "ingredients accumulate in double (sums of float32 samples); the blocks arrive in time order");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value& x = ins.at("x");
    need(x.cls == "field" && x.chunked, "reduce: x must be a field (a streamed one, or a map with no time axis), inside the chunk loop (a measurement over a grid's members is `reduce.members`)");
    // Q2: a field with NO time axis (a map on a surface, a value per channel) is reduced over the members of a selection of its space, once: nothing streams, the map is the one block
    need(x.time || ins.count("over"), "reduce: a field with no time axis is reduced over a selection of its space (`over`); with none there is no member to reduce to");
    need(p["measures"].is_array() && !p["measures"].empty(), "reduce: measures is a list (no default: the caller's or the preset's), e.g. [mean, max]");
    for (const auto& m : p["measures"]) need(m.is_string(), "reduce: a measure is a name");
    // A COMPLEX field is reduced as it is by the measures that are LINEAR in the value (sum, mean): its real and imaginary parts are folded alike and the result stays complex (value_type complex, parts re, im).
    // A measure that needs an order or a magnitude (max, min, abs_max, mean_square, median, amplitude, ...) has no complex value: it refuses with its reason (give the field a form first).
    if (x.complex)
      for (const auto& m : p["measures"])
        need(m == "sum" || m == "mean", "reduce: x is complex and " + m.get<std::string>() + " needs a real quantity (an order or a magnitude); a complex field is reduced by the measures linear in its value (sum, mean), "
                                         "or give it a form first (field.form magnitude | power | phase_current)");
    need(p["bad_segments"] == "exclude" || p["bad_segments"] == "ignore", "reduce: bad_segments " + p["bad_segments"].dump() + ": exclude or ignore");
    need(p["weight"] == "none" || p["weight"] == "barycentric" || p["weight"] == "voronoi", "reduce: weight " + p["weight"].dump() + ": none, barycentric or voronoi");
    for (const auto& m : p["measures"]) need(p["weight"] == "none" || (m != "median" && m != "mad"), "reduce: " + m.get<std::string>() + " is the member's own value and has no weighted form (weight none only)");
    if (ins.count("exclude")) {
      const Value& ex = ins.at("exclude");
      need(x.time.has_value(), "reduce: exclude is a selection on x's time line, and x has none");
      need(ex.cls == "selection" && ex.manifolds[0] == *x.time, "reduce: exclude must be a selection on x's time line (" + *x.time + "), not on " + ex.manifolds[0]);
    }
    Value y = made(x);
    y.cls = "measurement";
    y.chunked = false;
    y.facts["measures"] = p["measures"];
    y.facts["measure"] = p["measures"][0];
    y.facts["weight"] = p["weight"];
    if (ins.count("over")) {
      const Value& ov = ins.at("over");
      need(ov.cls == "selection", "reduce: over must be a selection");
      Json over = Json{{"selection", ov.row ? Json(nxr::io::text(*ov.row, "name")) : Json(ov.facts.value("node", std::string("made in the run")))}, {"selection_id", ov.row ? Json(nxr::io::text(*ov.row, "id")) : Json(nullptr)}};
      Json gd = Json::object(), band = nullptr;
      if (ov.row) {                                                          // a stored grid: its record
        const Json pj = Json::parse(nxr::io::opt_text(*ov.row, "params_json").value_or("{}"), nullptr, false);
        if (pj.is_object() && pj.contains("grid")) gd = pj["grid"];
        if (pj.is_object() && pj.contains("band")) band = pj["band"];
      } else {                                                               // a grid made earlier in the run: what its maker's port carries (select_nodes.cpp)
        for (const char* k : {"window_samples", "tiles", "width_s", "origin", "resolves_hz", "level"}) if (ov.facts.contains(k)) over[k] = ov.facts[k];
      }
      if (gd.is_object() && gd.contains("width") && x.time) { over["window_samples"] = gd["width"]; over["tiles"] = gd["count"]; over["level"] = gd.value("level", -1); over["width_s"] = gd["width"].get<double>() / x.rate; over["origin"] = x.origin; }
      if (!band.is_null()) over["resolves_hz"] = Json{{"lo", band[0]}, {"hi", band[1]}};
      y.facts["over"] = over;
      if (ov.row) y.manifolds = {ov.manifolds.empty() ? (x.time ? *x.time : std::string()) : ov.manifolds[0]};
      if (!x.time) y.rate = 0;
    } else {
      y.manifolds.erase(std::remove(y.manifolds.begin(), y.manifolds.end(), *x.time), y.manifolds.end());   // the run's span is the one member: the streamed factor goes
      y.time.reset();
      y.rate = 0;
    }
    return Values{{"values", y}, {"ingredients", y}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    ReduceState st;
    const Value& x = ins.at("x");
    for (const auto& m : p["measures"]) { st.measures.push_back(m.get<std::string>()); st.components.push_back(components_of(m.get<std::string>())); }
    st.req.measures = st.measures;
    st.req.weighting = p["weight"] == "none" ? nm::Weighting::none : nm::Weighting::given;
    st.rate = x.rate; st.origin = x.origin;
    st.complex = x.complex;
    std::int64_t samples = 0;                                              // a map has no time line: nothing streams
    if (x.time) {
      const auto line = c.bound.manifold(*x.time);
      need(line.has_value(), "reduce: x's time line is not among the subject's rows");
      samples = nxr::io::integer(*line, "n_vertices");
    }
    if (!ins.count("over")) {
      st.factors.push_back(form_of(Json{{"kind", "span"}}, samples, c, "reduce"));
    } else {
      const Json d = descriptor_of(ins.at("over"), c, "over", "reduce");
      st.over = d;
      if (d.value("kind", std::string()) == "product") {
        st.product = true;
        for (const auto& f : d["factors"]) st.factors.push_back(form_of(f, samples, c, "reduce"));
        // the code space: mixed radix, the FIRST factor fastest; a joint's first factor's radix is the next power of two (Studio D129): code = (tile << bits) | frame with the frames first
        for (const auto& f : st.factors) st.radices.push_back(f.members);
        if (d.value("layout", std::string("product")) == "joint") {
          int bits = 0;
          while ((std::int64_t(1) << bits) < st.radices[0]) ++bits;
          st.radices[0] = std::int64_t(1) << bits;
        }
        st.members = 1;
        for (const auto r : st.radices) st.members *= r;
        for (const auto& f : st.factors) {
          ns::Form form;
          if (f.kind == "time") form = f.rule;
          else { ns::MemberOfElement m; m.member_of = f.member_of; m.members = f.members; form = m; }
          st.product_form.factors.push_back(form);
          st.product_form.counts.push_back(f.members);
        }
        st.product_form.radices = st.radices;
      } else {
        st.factors.push_back(form_of(d, samples, c, "reduce"));
        st.members = st.factors[0].members;
      }
    }
    if (!st.product) st.members = st.factors[0].members;
    st.static_map = !x.time;
    need(st.product || st.factors[0].kind == "time" || !x.time, "reduce: over a selection of space alone a streamed field is not reduced here (pointwise in time: a spatial grid x a time grid is a `select.product`); give the product with the time grid");
    need(x.time || !st.product, "reduce: a map has no time axis: its selection is of its space alone, not a product with a time grid");
    if (p["weight"] != "none") {
      for (const auto& f : st.factors) {
        const Json& fd = st.product ? st.over["factors"][static_cast<std::size_t>(&f - st.factors.data())] : st.over;
        if (f.kind != "dyadic" && !(f.kind == "labelled" && fd.contains("surface"))) continue;
        const std::string surface = fd["surface"].get<std::string>();
        const nxr::pipelines::SurfaceInput S = c.bound.surface(surface, true);
        if (p["weight"] == "voronoi") {                                      // the mixed Voronoi mass of the input triangulation (the Studio's, nxr's and Rheome's)
          nxr::geometry::IntrinsicSurface isurf(S.vertices.data(), static_cast<int>(S.n_vertices()), S.faces.data(), static_cast<int>(S.faces.size() / 3));
          st.weights = isurf.inputMixedVoronoiMass();
        } else {
          nxr::select::Mesh mesh; mesh.n = static_cast<int>(S.n_vertices()); mesh.positions = S.vertices; mesh.faces = S.faces;
          st.weights = nxr::select::vertexAreas(mesh);
        }
      }
      need(!st.weights.empty(), "reduce: weight " + p["weight"].dump() + " needs a member set of a surface grid to weight its vertices by");
    }
    return std::any(std::move(st));
  };
  s.start = [](const std::any&) { return std::any(ReduceAcc{}); };
  s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
    const ReduceState& st = std::any_cast<const ReduceState&>(st_);
    ReduceAcc& acc = std::any_cast<ReduceAcc&>(acc_);
    const Block& b = ins.at("x");
    const bool pooled_rows = st.product || st.factors[0].kind != "time";       // a product or a space grid pools the rows (the elements); a time grid keeps them (channels, frames)
    // a vector field's rows are its elements' components, consecutive (element x components + component): one component per call (Request::component), the rows of one component the elements
    int ncomp = 1;
    for (const int k : st.components) ncomp = std::max(ncomp, k);
    need(b.rows % ncomp == 0, "reduce: x has " + std::to_string(b.rows) + " rows, not a whole number of " + std::to_string(ncomp) + "-component elements");
    const std::int64_t elements = b.rows / ncomp;
    acc.rows = pooled_rows ? 1 : elements;
    if (acc.ing.members == 0) {
      bool hold = false;                                                       // a median keeps the values (exact, unweighted, never rolled up)
      for (const auto& m : st.measures) hold = hold || m == "median" || m == "mad";
      acc.ing = nm::Ingredients(acc.rows, st.members, ncomp, hold);
      if (st.complex) acc.ing_im = nm::Ingredients(acc.rows, st.members, ncomp, hold);
    }
    nm::Request req = st.req;
    req.per_row = !pooled_rows;
    const std::int64_t n = b.own();
    if (st.static_map) {                                                   // THE MAP, ONE BLOCK: its elements are the rows of a one-sample block; folded by the members of their selection, the one reduction of compute
      std::vector<float> run(static_cast<std::size_t>(elements));
      ns::MemberOfElement m;
      m.member_of = st.factors[0].member_of; m.members = st.factors[0].members;
      need(static_cast<std::int64_t>(m.member_of.size()) == elements, "reduce: the selection has " + std::to_string(m.member_of.size()) + " elements, the map " + std::to_string(elements));
      for (int part = 0; part < (st.complex ? 2 : 1); ++part) {
        const std::vector<float>& src = part == 0 ? b.re : b.im;
        nm::Ingredients& into = part == 0 ? acc.ing : acc.ing_im;
        for (int comp = 0; comp < ncomp; ++comp) {
          req.component = comp;
          for (std::int64_t r = 0; r < elements; ++r) run[static_cast<std::size_t>(r)] = src[static_cast<std::size_t>((r * ncomp + comp) * b.width + b.left)];
          nm::reduce(into, run.data(), 1, elements, m, st.weights.empty() ? nullptr : st.weights.data(), req);
        }
      }
      return;
    }
    auto good = [&](std::int64_t j) { return !(place.good && !place.good->empty() && !(*place.good)[static_cast<std::size_t>(j)]); };
    std::vector<float> run;                                               // [rows x samples] of one run of good samples, contiguous
    for (std::int64_t a = 0; a < n;) {
      if (!good(a)) { ++a; continue; }
      std::int64_t e = a;
      while (e < n && good(e)) ++e;
      const std::int64_t len = e - a;
      run.resize(static_cast<std::size_t>(elements * len));
      for (int part = 0; part < (st.complex ? 2 : 1); ++part) {
        const std::vector<float>& src = part == 0 ? b.re : b.im;
        nm::Ingredients& into = part == 0 ? acc.ing : acc.ing_im;
        for (int comp = 0; comp < ncomp; ++comp) {
          req.component = comp;
          for (std::int64_t r = 0; r < elements; ++r) std::copy_n(src.begin() + (r * ncomp + comp) * b.width + b.left + a, len, run.begin() + r * len);
          if (st.product) nm::reduce(into, run.data(), elements, len, place.first + a, st.product_form, st.weights.empty() ? nullptr : st.weights.data(), req);
          else nm::reduce(into, run.data(), elements, len, place.first + a, st.factors[0].rule, req);
        }
      }
      a = e;
    }
  };
  s.finish = [](const std::any& st_, std::any& acc_) {
    const ReduceState& st = std::any_cast<const ReduceState&>(st_);
    ReduceAcc& acc = std::any_cast<ReduceAcc&>(acc_);
    need(acc.rows > 0, "reduce: no good sample in the span");
    const std::int64_t R = acc.rows, T = st.members;
    std::int64_t C = 0;
    for (const int k : st.components) C += k;
    const std::int64_t C1 = C;                                              // the columns of the REAL part; a complex value has them twice (re, im per measure)
    if (st.complex) C = 2 * C1;
    Array y{R, T, std::vector<double>(static_cast<std::size_t>(R * T * C), kNaN)};
    y.comps = C;
    // a member WHOLE in the span and free of bad samples has its full count of samples; any other is NaN (a short mean is never a tile's mean)
    std::int64_t whole = 0;
    const bool timed = !st.product && st.factors[0].kind == "time";
    for (std::int64_t r = 0; r < R; ++r)
      for (std::int64_t m = 0; m < T; ++m) {
        if (timed) {
          const ns::Rule& u = st.factors[0].rule;
          const std::int64_t a0 = std::max<std::int64_t>(0, u.origin + m * u.hop), a1 = std::min(u.size, u.origin + m * u.hop + u.width);
          if (nm::value(acc.ing, r, m, "count") != static_cast<double>(a1 - a0)) continue;
        } else if (!(nm::value(acc.ing, r, m, "count") > 0)) continue;
        ++whole;
        std::int64_t off = 0;
        for (std::size_t k = 0; k < st.measures.size(); ++k) {
          if (st.complex) {                                                 // one measure, two columns: re then im
            y.data[static_cast<std::size_t>((r * T + m) * C + 2 * off)] = read_value(acc.ing, r, m, st.measures[k], 0);
            y.data[static_cast<std::size_t>((r * T + m) * C + 2 * off + 1)] = read_value(acc.ing_im, r, m, st.measures[k], 0);
            off += 1;
            continue;
          }
          for (int c = 0; c < st.components[k]; ++c) y.data[static_cast<std::size_t>((r * T + m) * C + off + c)] = read_value(acc.ing, r, m, st.measures[k], c);
          off += st.components[k];
        }
      }
    need(whole > 0, "reduce: no member lies whole and good in the span");
    Json comps = Json::array();
    for (std::size_t k = 0; k < st.measures.size(); ++k) comps.push_back(Json{{"measure", st.measures[k]}, {"components", st.components[k]}});
    y.meta = Json{{"measures", comps}, {"members", T}, {"whole_members", whole}, {"over", st.over}, {"radices", st.radices}, {"rate", st.rate}, {"origin", st.origin}};
    y.meta["describe"] = describe_measures(comps, R, T, C1);
    if (st.complex) {                                                       // the stored array is [rows x members x 2] a measure: value_type complex, parts re, im (declared_outputs.cpp)
      y.meta["describe"]["complex"] = true;
      y.meta["describe"]["shape"] = Json::array({R, T, C});
      y.meta["describe"]["value_type"] = "complex";
      y.meta["describe"]["components"] = Json::array({"re", "im"});
    }
    if (st.over.is_object() && st.over.value("kind", std::string()) == "time" && st.over.value("width", std::int64_t(0)) > 0 && st.rate > 0) {   // a time grid's tile, as the detection after it reads it
      y.meta["window_samples"] = st.over["width"];
      y.meta["width_s"] = st.over["width"].get<double>() / st.rate;
      y.meta["tiles"] = st.over.value("count", std::int64_t(0));
      if (st.over.contains("band") && st.over["band"].is_array() && st.over["band"].size() == 2) y.meta["resolves_hz"] = Json{{"lo", st.over["band"][0]}, {"hi", st.over["band"][1]}};
    }
    // the ingredients, packed [rows x members x K] by the measurement module (what is stored, and what rolls up)
    Array g = pack_ingredients(acc.ing, R, T);
    if (st.complex) {                                                       // the imaginary part's ingredients follow the real part's, same layout (`imag_offset`: where they begin)
      const Array gi = pack_ingredients(acc.ing_im, R, T);
      g.meta["imag_offset"] = static_cast<std::int64_t>(g.data.size());
      g.data.insert(g.data.end(), gi.data.begin(), gi.data.end());
    }
    for (const auto& [k, v] : y.meta.items()) g.meta[k] = v;               // the values' record beside the ingredients' own layout (ingredient_lengths, ...)
    return Arrays{{"values", std::move(y)}, {"ingredients", std::move(g)}};
  };
  return s;
}

// ── reduce.members ───────────────────────────────────────────────────────────────────────────────────────────────
// A measurement over a grid G's members, over the members of another selection given in the first one's member numbers (events detected over G): nxr::measurement::merge. The ingredients
// of each event are the merge of its cells', never a second pass over the signal (rule 13).
StepSpec reduce_members_node() {
  StepSpec s;
  s.name = "measurement.reduce_members";
  s.kind = "after";
  s.inputs = {"x", "over"};
  s.outputs = {"values", "ingredients"};
  s.parameters = Json{{"measures", nullptr}};
  s.matches = "the Studio's measureOverPartition over a stored measurement; compute nxr::measurement::merge";
  s.description = "the ingredients of a measurement over a grid's members merged over the members of an irregular selection given on them (each event's cells): the event's value from its cells' "
                  "ingredients by each measure's merge rule (sums add, maxima merge), never a second pass over the signal. The events' members may overlap; no level, no roll-up.";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the ingredients: double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value &x = ins.at("x"), &ev = ins.at("over");
    need(x.cls == "measurement", "reduce.members: x must be a measurement over a grid's members (`reduce` with a grid)");
    need(ev.cls == "selection" || ev.cls == "spans", "reduce.members: over must be a selection (events) given on the grid's members");
    need(p["measures"].is_array() && !p["measures"].empty(), "reduce.members: measures is a list (no default)");
    Value y = made(x);
    y.facts["measures"] = p["measures"];
    y.facts["measure"] = p["measures"][0];
    y.facts["over"] = Json{{"selection", ev.row ? Json(nxr::io::text(*ev.row, "name")) : Json("the events made in the run")}, {"irregular", true}, {"detected_over", x.facts.value("over", Json::object())}};
    return Values{{"values", y}, {"ingredients", y}};
  };
  s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array& g = ins.at("x");
    const Array& ev = ins.at("over");                                     // [events x 4]: first tile, stop tile, row (-1: every), peak -- the events' cells as ranges of the grid's members
    ns::Elements groups;
    groups.size = g.cols;
    for (std::int64_t k = 0; k < ev.rows; ++k) {
      const double* e = ev.data.data() + k * 4;
      for (std::int64_t m = static_cast<std::int64_t>(e[0]); m < static_cast<std::int64_t>(e[1]); ++m) groups.elements.push_back(m);
      groups.offsets.push_back(static_cast<std::int64_t>(groups.elements.size()));
    }
    // the measures of the events' CELLS merge by their rule; an event's OWN measures (its duration, the interval to the next event, its tile count) are read from the events' tiles
    const std::set<std::string> own = {"duration", "interval_next", "n_tiles"};
    nm::Request req;
    for (const auto& m : p["measures"]) if (!own.count(m.get<std::string>())) req.measures.push_back(m.get<std::string>());
    const nm::Ingredients fine = unpack_ingredients(g);
    const nm::Ingredients merged = nm::merge(fine, groups, req);
    const Array cells = values_of(merged, req, g.rows, groups.members());
    const std::int64_t E = groups.members(), R = g.rows;
    const double width_s = g.meta.value("width_s", 0.0);
    std::vector<nxr::select::TileEvent> evs;
    for (std::int64_t k = 0; k < ev.rows; ++k) { const double* e = ev.data.data() + k * 4; evs.push_back({static_cast<std::int64_t>(e[0]), static_cast<std::int64_t>(e[1]), static_cast<std::int64_t>(e[2])}); }
    std::int64_t C = 0, Cc = 0;
    Json comps = Json::array();
    for (const auto& m : p["measures"]) { const int k = components_of(m.get<std::string>()); C += k; comps.push_back(Json{{"measure", m}, {"components", k}}); }
    for (const auto& m : req.measures) Cc += components_of(m);
    Array y{R, E, std::vector<double>(static_cast<std::size_t>(R * E * C), kNaN)};
    y.comps = C;
    std::int64_t off = 0, coff = 0;
    for (const auto& mj : p["measures"]) {
      const std::string m = mj.get<std::string>();
      const int k = components_of(m);
      if (own.count(m)) {
        need(width_s > 0, "reduce_members: " + m + " is read from the events' tiles; x carries no tile width (reduce over a time grid)");
        const std::vector<double> v = m == "duration" ? nxr::select::eventDurations(evs, width_s) : m == "interval_next" ? nxr::select::eventIntervalNext(evs, width_s) : [&] {
          std::vector<double> n; for (const auto& e : evs) n.push_back(static_cast<double>(e.stop - e.first)); return n; }();
        for (std::int64_t e = 0; e < E; ++e)
          for (std::int64_t r = 0; r < R; ++r)
            if (evs[static_cast<std::size_t>(e)].row < 0 || evs[static_cast<std::size_t>(e)].row == r) y.data[static_cast<std::size_t>((r * E + e) * C + off)] = v[static_cast<std::size_t>(e)];
      } else {
        for (std::int64_t r = 0; r < R; ++r)
          for (std::int64_t e = 0; e < E; ++e)
            for (int c = 0; c < k; ++c) y.data[static_cast<std::size_t>((r * E + e) * C + off + c)] = cells.data[static_cast<std::size_t>((r * E + e) * Cc + coff + c)];
        coff += k;
      }
      off += k;
    }
    // an event found on one row (a channel) is measured on that row alone: the other rows hold no value for it (an event of row -1, found on the mean over rows, is measured on every row)
    for (std::int64_t e = 0; e < E; ++e)
      if (evs[static_cast<std::size_t>(e)].row >= 0)
        for (std::int64_t r = 0; r < R; ++r)
          if (r != evs[static_cast<std::size_t>(e)].row) for (std::int64_t c = 0; c < C; ++c) y.data[static_cast<std::size_t>((r * E + e) * C + c)] = kNaN;
    y.meta = Json{{"members", E}, {"of", g.meta}, {"measures", comps}, {"describe", describe_measures(comps, R, E, C)}};
    Array gi = pack_ingredients(merged, g.rows, E);
    for (const auto& [kk, vv] : y.meta.items()) gi.meta[kk] = vv;
    return Arrays{{"values", std::move(y)}, {"ingredients", std::move(gi)}};
  };
  return s;
}

// ── reduce.statistic ─────────────────────────────────────────────────────────────────────────────────────────────
// A STORED STATISTIC reduced over the members of a selection of its first axis (a PSD [channels x frequencies] over a channel set): `measure` of a statistic (schema 54, consolidation spec 6.1). The
// reduction is the same function as a field's (`nxr::measurement::reduce`, members of elements): the rows of the measurement are the statistic's OTHER axes (the frequencies, in order), the
// members the selection's. The statistic is read as it is stored; nothing is computed from the recording. Elements the selection leaves out (member -1) do not count. The values are the measures
// as the registry derives them from the ingredients (a mean is sum / count), stored as [rows x members x components] OF THE STATISTIC (`of_statistic_id`, no field).
StepSpec reduce_statistic_node() {
  StepSpec s;
  s.name = "measurement.reduce_statistic";
  s.kind = "after";
  s.inputs = {"x", "over"};
  s.outputs = {"values"};
  s.parameters = Json{{"measures", nullptr}};
  s.reads_arrays = true;
  s.matches = "the Studio's measureOverPartition over a stored statistic's first axis; compute nxr::measurement::reduce (members of elements)";
  s.description = "A stored statistic (a PSD [channels x frequencies]) reduced over the members of a selection of its first axis, by registry measures: one value per (row, member), the rows the statistic's other axes "
                  "in order. The same reduction as a field's over a set of elements, so a channel set's mean spectrum is a measurement read by name, never a loop inside a node. Elements the selection does not hold "
                  "(member -1) are left out. The measurement is OF the statistic.";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the stored statistic's values: double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value &x = ins.at("x"), &ov = ins.at("over");
    need(x.cls == "statistic", "reduce.statistic: x must be a stored statistic (a field is `reduce`)");
    need(ov.cls == "selection", "reduce.statistic: over must be a selection of the statistic's first axis (a set of its elements, e.g. a channel set as one member)");
    need(p["measures"].is_array() && !p["measures"].empty(), "reduce.statistic: measures is a list (no default), e.g. [mean]");
    for (const auto& m : p["measures"]) need(m.is_string(), "reduce.statistic: a measure is a name");
    Value y;
    y.cls = "measurement";
    y.manifolds = ov.manifolds;
    y.facts["measures"] = p["measures"];
    y.facts["measure"] = p["measures"][0];
    y.facts["of_statistic"] = ins.at("x").row ? Json(nxr::io::text(*ins.at("x").row, "name")) : Json("a statistic");
    y.facts["over"] = Json{{"selection", ov.row ? Json(nxr::io::text(*ov.row, "name")) : Json("the selection made in the run")}};
    return Values{{"values", y}};
  };
  s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  s.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array &x = ins.at("x"), &ov = ins.at("over");
    const std::int64_t E = x.rows, R = x.cols;                                  // [elements x rows]: the first axis is the members' elements, the rest are the rows
    need(ov.rows == E, "reduce.statistic: the selection has " + std::to_string(ov.rows) + " elements, the statistic's first axis " + std::to_string(E));
    need(!x.meta.contains("value_type") || x.meta["value_type"] == "scalar", "reduce.statistic: the statistic's values are " + x.meta.value("value_type", std::string()) + "; a scalar statistic is reduced");
    ns::MemberOfElement set;
    set.member_of.resize(static_cast<std::size_t>(E));
    std::int64_t members = ov.meta.value("n_members", std::int64_t(0));
    if (ov.meta.contains("describe") && ov.meta["describe"].contains("n_members")) members = std::max(members, ov.meta["describe"]["n_members"].get<std::int64_t>());
    for (std::int64_t e = 0; e < E; ++e) { set.member_of[static_cast<std::size_t>(e)] = static_cast<std::int32_t>(ov.data[static_cast<std::size_t>(e)]); members = std::max<std::int64_t>(members, set.member_of[static_cast<std::size_t>(e)] + 1); }
    set.members = members;
    nm::Request req;
    for (const auto& m : p["measures"]) req.measures.push_back(m.get<std::string>());
    std::vector<double> rows_by_elements(static_cast<std::size_t>(R * E));          // the reduction reads [rows x elements]
    for (std::int64_t e = 0; e < E; ++e) for (std::int64_t r = 0; r < R; ++r) rows_by_elements[static_cast<std::size_t>(r * E + e)] = x.data[static_cast<std::size_t>(e * R + r)];
    nm::Ingredients acc(R, members, 1);
    nm::reduce(acc, rows_by_elements.data(), R, E, set, nullptr, req);
    Array y = values_of(acc, req, R, members);
    const Json comps = y.meta["measures"];
    std::int64_t C = 0;
    for (const auto& m : comps) C += m.value("components", 1);
    y.meta = Json{{"members", members}, {"measures", comps}, {"statistic", x.meta.value("name", std::string())}, {"describe", describe_measures(comps, R, members, C)}};
    if (x.meta.contains("unit")) for (auto& c : y.meta["describe"]["measures"]) c["unit"] = x.meta["unit"];
    y.meta["describe"]["attributes"] = Json{{"rows", "the statistic's axes after the first, in order"}, {"of_statistic", x.meta.value("name", std::string())}};
    return Arrays{{"values", std::move(y)}};
  };
  return s;
}

// ── roll_up ──────────────────────────────────────────────────────────────────────────────────────────────────────
// The ingredients through a GRID's closed-form hierarchy: the coarser level as a QUANTITY (a duration, a tile size, `whole`), resolved in closed form (spec 3.3); or across subjects
// (the same members in several subjects, summed code for code: `across`). The checker refuses an irregular selection (rule 18: the family mark and the move, never a parameter name).
StepSpec roll_up_node() {
  StepSpec s;
  s.name = "measurement.roll_up";
  s.kind = "after";
  s.inputs = {"x", "grid"};
  s.optional = {"grid"};
  s.outputs = {"values", "ingredients"};
  s.parameters = Json{{"to", nullptr}, {"measures", nullptr}, {"factor", 0}};
  s.matches = "the Studio's measurement roll-up through a grid's hierarchy (the merge rule of each measure); compute nxr::measurement::roll_up";
  s.description =
      "a measurement's ingredients carried up a GRID's closed-form hierarchy to a coarser level (`to`: a duration such as `2 s`, a tile size such as `40 mm`, `pieces`, or `whole`: one member; "
      "for a product, `factor` names the factor rolled): sums add, maxima merge, means by sum and count, ratios derived at read (the merge rule of each measure is data in the measures "
      "registry). Refused over an irregular selection (no levels).";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the ingredients: double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value& x = ins.at("x");
    need(x.cls == "measurement", "roll_up: x must be a measurement (`reduce`)");
    need(p["to"].is_string() || p["to"].is_number(), "roll_up: `to` is the coarser level as a quantity (e.g. \"2 s\", \"40 mm\", \"pieces\", \"whole\"); a bare level index is only `index:N`");
    if (p["to"].is_string()) {
      const std::string t = p["to"].get<std::string>();
      if (t != "whole" && t != "pieces") { try { nxr::quantity::parse(t); } catch (const std::invalid_argument& e) { refuse(std::string("roll_up: to: ") + e.what()); } }
    } else refuse("roll_up: `to` is a quantity with a unit (e.g. \"2 s\"), not the bare number " + p["to"].dump());
    const Json over = x.facts.value("over", Json::object());
    need(!over.value("irregular", false), "roll_up: x is over an irregular selection (" + over.value("selection", std::string("?")) + "): it has no levels to roll up through; summarize its members instead");
    Value y = made(x);
    y.facts["rolled_to"] = p["to"];
    if (p["measures"].is_array()) y.facts["measures"] = p["measures"];
    return Values{{"values", y}, {"ingredients", y}};
  };
  s.prepare = [](const Values&, const Json& p, ExecContext&) { return std::any(p); };
  s.run = [](const std::any& st, const Arrays& ins, ExecContext& ctx) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array& g = ins.at("x");
    nm::Request req;
    for (const auto& m : g.meta.value("measures", Json::array())) req.measures.push_back(m.value("measure", std::string()));
    const std::string to = p["to"].get<std::string>();
    ns::Elements groups;
    groups.size = g.cols;
    if (to == "whole") {
      groups.elements.resize(static_cast<std::size_t>(g.cols));
      for (std::int64_t i = 0; i < g.cols; ++i) groups.elements[static_cast<std::size_t>(i)] = i;
      groups.offsets.push_back(g.cols);
    } else {
      // the groups of a coarser level of the grid the ingredients are on, in the finer level's member numbers (closed form: code >> shift)
      const Json over = g.meta.value("over", Json::object());
      need(over.value("kind", std::string()) == "time", "roll_up: to " + to + ": only a time grid (a duration) or `whole` is rolled in this run; a surface grid's levels roll up through its stored closed form");
      const nxr::quantity::Quantity q = nxr::quantity::parse(to);
      const double rate = g.meta.value("rate", 0.0);
      need(rate > 0, "roll_up: the ingredients carry no sampling rate");
      const auto level = nxr::select::level_of_duration(q.hi, rate);
      const int from = static_cast<int>(std::lround(std::log2(static_cast<double>(over.value("width", std::int64_t(1))))));
      need(level.level >= from, "roll_up: " + to + " is finer than the grid's tiles (" + std::to_string(over.value("width", std::int64_t(1))) + " samples): a roll-up goes coarser");
      const ns::Line line{"s", over.value("size", std::int64_t(0)), g.meta.value("origin", 0.0), 1.0 / rate};
      const ns::Selection grid = nxr::select::time_grid(line, from);
      groups = nm::coarser_groups(grid, from, level.level);
      if (ctx.say) ctx.say(Json{{"stage", "roll_up"}, {"to", to}, {"level", level.level}, {"snapped_s", level.snapped}});
    }
    const nm::Ingredients fine = unpack_ingredients(g);
    const nm::Ingredients coarse = nm::roll_up(fine, groups, req);
    Array y = values_of(coarse, req, g.rows, groups.members());
    y.meta = Json{{"members", groups.members()}, {"rolled_to", to}, {"of", g.meta}, {"measures", g.meta.value("measures", Json::array())}};
    { std::int64_t C = 0; for (const auto& m : y.meta["measures"]) C += m.value("components", 1); y.meta["describe"] = describe_measures(y.meta["measures"], y.rows, y.cols, C); }
    Array gi = pack_ingredients(coarse, g.rows, groups.members());
    gi.meta = y.meta;
    return Arrays{{"values", std::move(y)}, {"ingredients", std::move(gi)}};
  };
  return s;
}

// ── summarize ────────────────────────────────────────────────────────────────────────────────────────────────────
// the existing step's plan, ports and parameters; its run is nxr::measurement::summarize (stats.h stays as its arithmetic)
void redirect_summarize(StepSpec& sc) {
  sc.description = "a measurement's members summarised into a STATISTIC (members -> statistic; across an irregular set it is a `summary`; never of a field): per row (each), pooled, or the mean over rows; "
                   "NaN members are skipped and counted";
  sc.reads_arrays = true;                                                // a stored measurement found by name is read as an array [rows x members]
  sc.matches = "nxr::measurement::summarize (the summaries of compute nxr::stats: numpy median and linear percentile, Rheome's robust SD, MATLAB's sd, kurtosis, skewness)";
  sc.run = [](const std::any& st, const Arrays& ins, ExecContext&) {
    const Json& p = std::any_cast<const Json&>(st);
    const Array& x = ins.at("x");
    nm::Summaries req;
    for (const auto& n : p["summaries"]) req.names.push_back(n.get<std::string>());
    req.ddof = p["ddof"].get<int>();
    const std::string combine = p["combine"].get<std::string>();
    req.combine = combine == "each" ? nm::Combine::each : combine == "pool" ? nm::Combine::pool : nm::Combine::mean;
    // THE MEASURES OF x are its columns [rows x members x measures]: each is summarised, the statistic's columns measure-major (measure c, summary s at column c * summaries + s)
    const std::int64_t C = std::max<std::int64_t>(1, x.comps);
    std::vector<std::string> column_names;
    for (const auto& m : x.meta.value("measures", Json::array())) {
      const std::string name = m.value("measure", std::string());
      if (m.value("components", 1) == 3) for (const char* ax : {"_x", "_y", "_z"}) column_names.push_back(name + ax);
      else column_names.push_back(name);
    }
    std::int64_t first = 0, count = C;                                       // `measure`: one of the columns alone
    if (p["measure"].is_string()) {
      const std::string want = p["measure"].get<std::string>();
      bool found = false;
      std::int64_t c = 0;
      for (const auto& m : x.meta.value("measures", Json::array())) {
        const int k = m.value("components", 1);
        if (m.value("measure", std::string()) == want) { first = c; count = k; found = true; }
        c += k;
      }
      if (!found && C == 1) { first = 0; count = 1; found = true; }
      need(found, "summarize: measure " + p["measure"].dump() + " is not one of x's measures");
    }
    if (static_cast<std::int64_t>(column_names.size()) != C) { column_names.clear(); for (std::int64_t c = 0; c < C; ++c) column_names.push_back(C == 1 ? x.meta.value("measure", std::string()) : "column " + std::to_string(c)); }
    // no members at all (a run that found no event): one NaN member, so the counts and sums read 0 and the rest NaN
    const std::int64_t S = static_cast<std::int64_t>(req.names.size());
    Array y{x.rows, count * S, std::vector<double>(static_cast<std::size_t>(x.rows * count * S), kNaN)};
    for (std::int64_t j = 0; j < count; ++j) {
      const std::int64_t c = first + j;
      Eigen::MatrixXd values(x.rows, std::max<std::int64_t>(1, x.cols));
      for (std::int64_t r = 0; r < x.rows; ++r)
        for (std::int64_t m = 0; m < values.cols(); ++m) values(r, m) = x.cols ? x.data[static_cast<std::size_t>((r * x.cols + m) * C + c)] : kNaN;
      const Eigen::MatrixXd out = nm::summarize(values, req);               // rows' x summaries
      if (j == 0) { y.rows = out.rows(); y.data.assign(static_cast<std::size_t>(out.rows() * count * S), kNaN); }
      for (Eigen::Index r = 0; r < out.rows(); ++r) for (Eigen::Index k = 0; k < out.cols(); ++k) y.data[static_cast<std::size_t>(r * count * S + j * S + k)] = out(r, k);
    }
    Json sj = Json::array();
    for (const auto& n : req.names) sj.push_back(Json{{"name", n}});
    Json measure_values = Json::array();
    std::string measure_text;
    for (std::int64_t j = 0; j < count; ++j) { const std::string& n = column_names[static_cast<std::size_t>(first + j)]; measure_values.push_back(Json{{"name", n}}); measure_text += (j ? "+" : "") + n; }
    y.meta = Json{{"summaries", sj}, {"combine", p["combine"]}, {"ddof", p["ddof"]}, {"members", x.cols}, {"measures", measure_values}, {"of", x.meta}};
    // what a DECLARED statistic output is stored from: the measure(s) summarised, its rows, measures and summaries as axes, the options
    Json row_values = Json::array();                                       // the rows: the combination they were made into, or (each) the row's number
    for (std::int64_t r = 0; r < y.rows; ++r) row_values.push_back(Json{{"name", combine == "each" ? std::to_string(r) : combine}});
    Json axes = Json::array({Json{{"name", "row"}, {"size", y.rows}, {"values", row_values}}});
    if (count > 1) axes.push_back(Json{{"name", "measure"}, {"size", count}, {"values", measure_values}});
    axes.push_back(Json{{"name", "summary"}, {"size", S}, {"values", sj}});
    y.cols = count * S;
    y.meta["describe"] = Json{{"measure", count > 1 ? column_names[static_cast<std::size_t>(first)] : measure_text}, {"method", "summary"}, {"label", "summary"}, {"axes", axes},
                              {"n_members", x.cols}, {"options", Json{{"summaries", sj}, {"combine", p["combine"]}, {"ddof", p["ddof"]}}},
                              {"description", "the " + measure_text + " of " + x.meta.value("name", std::string()) + " summarised (" + combine + ")"}};
    return Arrays{{"y", std::move(y)}};
  };
}

// ── estimate ─────────────────────────────────────────────────────────────────────────────────────────────────────
// A statistic from a partition of the stream: the MEAN over a family of segments (they may overlap) of a per-segment quantity, folded block by block (`SegmentMean`), or the counts of
// fixed bins (`Histogram`). The per-segment quantity (taper and FFT; X Xᴴ; |X|²) is a pointwise node before it; `estimate` only means or counts. A segment cut by an excluded span is
// skipped and counted.
struct EstimateState { std::string method; std::vector<double> edges; std::string rows_manifold, quantity; };
struct EstimateAcc { std::optional<nm::Histogram> hist; nm::SegmentMean mean; std::int64_t rows = 0, cols = 0; };

StepSpec estimate_node() {
  StepSpec s;
  s.name = "measurement.estimate";
  s.kind = "accumulate";
  s.inputs = {"x", "exclude"};
  s.optional = {"exclude"};
  s.outputs = {"y"};
  s.parameters = Json{{"method", nullptr}, {"edges", nullptr}};
  s.matches = "nxr's Welch segment mean (scipy.signal.welch) and nxr's log-binned histograms; compute nxr::measurement::SegmentMean, Histogram";
  s.description = "a statistic from the stream: method `histogram` (fixed bins `edges`: counts of the good samples of a real field) or `sample` (the mean over time of a per-sample quantity, "
                  "samples cut by an excluded span left out and counted); Welch's segments are `welch.*` nodes on SegmentMean";
  s.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64, "counts and sums in double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value& x = ins.at("x");
    need(x.cls == "field" && x.time && x.chunked, "estimate: x must be a field on ... x time, inside the chunk loop");
    need(!x.complex, "estimate: x is complex; estimate a real quantity of it (give it a form first: field.form magnitude | power | phase_current)");
    need(p["method"] == "histogram" || p["method"] == "sample", "estimate: method " + p["method"].dump() + ": histogram or sample");
    if (p["method"] == "histogram") need(p["edges"].is_array() && p["edges"].size() >= 2, "estimate: histogram needs `edges` (no default: the caller's bins)");
    Value y = made(x);
    y.cls = "statistic";
    y.chunked = false;
    y.facts["method"] = p["method"];
    y.manifolds.erase(std::remove(y.manifolds.begin(), y.manifolds.end(), *x.time), y.manifolds.end());
    y.time.reset();
    return Values{{"y", y}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    EstimateState st;
    st.method = p["method"].get<std::string>();
    st.rows_manifold = ins.at("x").manifolds.empty() ? std::string() : ins.at("x").manifolds[0];
    st.quantity = ins.at("x").facts.value("made_by", std::string());
    if (st.method == "histogram") for (const auto& e : p["edges"]) st.edges.push_back(e.get<double>());
    return std::any(std::move(st));
  };
  s.start = [](const std::any& st_) {
    const EstimateState& st = std::any_cast<const EstimateState&>(st_);
    EstimateAcc a;
    if (st.method == "histogram") { nm::Histogram h; h.edges = st.edges; h.counts.assign(st.edges.size() - 1, 0); a.hist = std::move(h); }
    return std::any(std::move(a));
  };
  s.fold = [](const std::any& st_, std::any& acc_, const Blocks& ins, const Place& place) {
    const EstimateState& st = std::any_cast<const EstimateState&>(st_);
    EstimateAcc& acc = std::any_cast<EstimateAcc&>(acc_);
    const Block& b = ins.at("x");
    const std::int64_t n = b.own();
    auto good = [&](std::int64_t j) { return !(place.good && !place.good->empty() && !(*place.good)[static_cast<std::size_t>(j)]); };
    acc.rows = b.rows;
    if (st.method == "histogram") {
      for (std::int64_t r = 0; r < b.rows; ++r)
        for (std::int64_t a = 0; a < n;) {
          if (!good(a)) { ++a; continue; }
          std::int64_t e = a;
          while (e < n && good(e)) ++e;
          acc.hist->add(b.re.data() + r * b.width + b.left + a, e - a);
          a = e;
        }
      return;
    }
    Eigen::MatrixXcd v = Eigen::MatrixXcd::Zero(b.rows, 1);
    std::int64_t used = 0;
    for (std::int64_t j = 0; j < n; ++j) {
      if (!good(j)) continue;
      for (std::int64_t r = 0; r < b.rows; ++r) v(r, 0) = std::complex<double>(b.re[static_cast<std::size_t>(r * b.width + b.left + j)], 0.0);
      acc.mean.add(place.first + j, v);
      ++used;
    }
    acc.cols += used;
  };
  s.finish = [](const std::any& st_, std::any& acc_) {
    const EstimateState& st = std::any_cast<const EstimateState&>(st_);
    EstimateAcc& acc = std::any_cast<EstimateAcc&>(acc_);
    if (st.method == "histogram") {
      Array y{1, static_cast<std::int64_t>(acc.hist->counts.size()), std::vector<double>(acc.hist->counts.begin(), acc.hist->counts.end())};
      y.meta = Json{{"method", "histogram"}, {"edges", acc.hist->edges}};
      const std::int64_t nb = static_cast<std::int64_t>(acc.hist->counts.size());
      y.meta["generic_statistic"] = Json{{"measure", "count"}, {"method", "histogram"}, {"shape", Json::array({1, nb})},
                                         {"axes", Json::array({Json{{"name", "pool"}, {"values", Json::array({Json{{"name", "every row, every good sample"}}})}, {"size", 1}}, Json{{"name", "bin"}, {"size", nb}, {"edges", acc.hist->edges}}})},
                                         {"description", "the counts of the good samples of the field in the given bins"}, {"options", Json{{"edges", acc.hist->edges}}}};
      return Arrays{{"y", std::move(y)}};
    }
    need(acc.mean.used > 0, "estimate: no good sample in the span");
    const Eigen::MatrixXcd m = acc.mean.mean();
    Array y{m.rows(), 1, std::vector<double>(static_cast<std::size_t>(m.rows()))};
    for (Eigen::Index r = 0; r < m.rows(); ++r) y.data[static_cast<std::size_t>(r)] = m(r, 0).real();
    y.meta = Json{{"method", "sample"}, {"used", acc.mean.used}, {"skipped", acc.mean.skipped}};
    // the generic mean of a real per-sample field (the relative power, a magnitude): [row] on the field's first factor, stored as a statistic of method mean
    Json row_axis = Json{{"name", "row"}, {"size", m.rows()}};
    if (!st.rows_manifold.empty()) row_axis["manifold_id"] = st.rows_manifold;
    y.meta["generic_statistic"] = Json{{"measure", st.quantity == "relative" ? std::string("relative_power") : st.quantity == "magnitude" ? std::string("amplitude") : std::string("mean")}, {"method", "mean"}, {"shape", Json::array({m.rows(), 1})},
                                       {"axes", Json::array({row_axis, Json{{"name", "estimate"}, {"values", Json::array({Json{{"name", "mean"}}})}, {"size", 1}}})},
                                       {"n_members", acc.mean.used}, {"description", "the mean over the good samples of the field, per row"}, {"options", Json{{"used", acc.mean.used}, {"skipped", acc.mean.skipped}}}};
    return Arrays{{"y", std::move(y)}};
  };
  return s;
}

}  // namespace

void add_measurement_nodes(std::vector<StepSpec>& r) {
  // field.form: the three forms' steps, and `outer`, as one node chosen by `form`
  {
    std::vector<std::pair<std::string, StepSpec>> parts;
    parts.emplace_back("magnitude", take_step(r, "magnitude"));
    parts.emplace_back("power", take_step(r, "power"));
    parts.emplace_back("phase_current", take_step(r, "phase_current"));
    parts.emplace_back("outer", outer_form());
    r.push_back(combine_by_param("signal.field_form", "form", std::move(parts),
        "the pointwise FORM of a field (the Studio's fieldForms.ts, complex-extended): magnitude |z| (per element, over its components), power |z|^2, phase_current Im(z conj grad z) on the tangent "
        "plane, outer z z^H; a reduction over time or space is `reduce` after it, never here. (The Studio's own named forms of a typed field -- curl, div, abs -- stay the node `form`.)"));
  }
  // the retired reductions: reduce_cells and tile_max are roll_up along the space factor, measure and measure_events are reduce, summarize_measurements is summarize
  // (already gone from the registry when the old verbs' steps were dropped: take what is there)
  for (const char* n : {"measure", "tile_max", "reduce_cells", "measure_events"})
    r.erase(std::remove_if(r.begin(), r.end(), [&](const StepSpec& s) { return s.name == n; }), r.end());
  StepSpec sc = take_step(r, "measurement.summarize");
  redirect_summarize(sc);
  r.push_back(std::move(sc));
  r.push_back(reduce_node());
  r.push_back(reduce_members_node());
  r.push_back(reduce_statistic_node());
  r.push_back(roll_up_node());
  // estimate: ONE node, the statistic of a partition of the stream (spec 2.5; core_histogram, relative_histogram and baseline are folded in). `method`: `sample` (the mean over the good columns: a band's
  // power per vertex and level is the BASELINE, any other real field the generic mean) or `histogram` (counts: fixed bins when `edges` are given, else the log bins of nxr's threshold over a
  // curl (cores) or over the relative power outside the bursts). What x IS tells the part; the choice is in the cache key.
  {
    std::vector<std::pair<std::string, StepSpec>> parts;
    parts.emplace_back("baseline", take_step(r, "baseline"));
    parts.emplace_back("core_histogram", take_step(r, "core_histogram"));
    parts.emplace_back("relative_histogram", take_step(r, "relative_histogram"));
    parts.emplace_back("mean", estimate_node());
    parts.emplace_back("fixed", estimate_node());
    r.push_back(combine_by_param(
        "measurement.estimate", "method", std::move(parts),
        "a statistic from the stream, by `method`: `sample` = the mean over the good columns of x (the BASELINE of a band's power per vertex and level, left out: `exclude` the bad segments and `outside` the bursts; "
        "any other real field: its mean, samples cut by an excluded span counted); `histogram` = counts of the good frames' values: fixed bins when `edges` are given, else nxr's 4096 log bins (range from the first "
        "`range_frames` good frames) over |curl| of the current (the cores' threshold) or over the relative power outside the bursts (the patches' threshold). Welch's segments are `welch.*` nodes on SegmentMean.",
        [](const Json& p, const std::string& which) {
          Json q = p;
          if (which == "mean") q["method"] = "sample";
          if (which == "fixed") q["method"] = "histogram";
          return q;
        },
        [](const Values& ins, const Json& p) -> std::string {
          const std::string m = p.contains("method") && p["method"].is_string() ? p["method"].get<std::string>() : std::string();
          const std::string made = ins.count("x") ? ins.at("x").facts.value("made_by", std::string()) : std::string();
          if (m == "sample") return made == "power" ? "baseline" : "mean";
          if (m == "histogram") return p.contains("edges") && p["edges"].is_array() ? "fixed" : (made == "relative" ? "relative_histogram" : "core_histogram");
          return m;
        }));
  }
}

}  // namespace nxr::pipelines::graph
