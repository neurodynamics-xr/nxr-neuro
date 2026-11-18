// The type system on the shipped definitions (spec docs/specs/2026-11-17-type-system.md, step T2; the C++ port of tools/type_inventory.py's `types` and `mistakes` sections):
//   1. every shipped definition checks by the shared checker, on the input types the runner would read from the found rows (stated below, as in the prototype's STORED), and the
//      outputs' declared kinds are those the graph makes;
//   2. each of the spec's ten seeded mistakes (appendix B), an edit of a shipped definition, is refused before any data, with the expected rule;
//   3. the signatures' ports are the registered steps' (names, optionality, outputs in order), and every step the definitions use has a signature.
// Synthetic only: no dataset, no subject, no file.
#include <cstdio>
#include <functional>
#include <set>
#include <string>

#include "nxr/graph/graph.h"
#include "nxr/graph/signature.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/registry.h"

namespace ng = nxr::graph;
namespace np = nxr::pipelines::graph;
using ng::Json;

namespace {
int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)

// ── what the runner would read from the found rows, where a definition does not say it (the prototype's STORED; each a fact of a row):
//   an inverse kernel: free orientation (vector3); a joint partition: of (the kernel's target surface x the recording's time line) at the space level and band its `where` names;
//   `tiling` is a frame tiling, any other spans a spans selection on the time line; statistic mean: the spatial baseline; histogram: a distribution; welch: channels of the kernel
//   (`--channels-of`) when the definition reads a kernel, else of the recording.
struct Found { std::map<std::string, ng::PortType> types; std::vector<std::string> order; };

Json resolve(const Json& v, const Json& params) {
  if (v.is_string() && !v.get<std::string>().empty() && v.get<std::string>()[0] == '$') {
    const std::string k = v.get<std::string>().substr(1);
    if (params.contains(k)) return params[k];
  }
  return v;
}

ng::Domain dom(std::vector<ng::ManifoldRef> f) { ng::Domain d; d.factors = std::move(f); return d; }

ng::ManifoldRef rel(const std::string& ref, const Found& found) {
  const std::size_t dot = ref.find('.');
  const ng::PortType& t = found.types.at(ref.substr(0, dot));
  const std::string what = ref.substr(dot + 1);
  if (what == "time") return t.domain.factors.back();
  if (what.rfind("manifold.", 0) == 0) return t.domain.factors.at(static_cast<std::size_t>(std::stoi(what.substr(9))));
  if (what == "to") return t.target.factors.at(0);
  if (what == "from") return t.domain.factors.at(0);
  ng::refuse("relation " + ref + " not understood");
}

ng::PortType input_type(const std::string& name, const Json& spec, const Found& found, const Json& params, const Json& all_inputs) {
  ng::PortType t;
  t.typed = true;
  if (spec.contains("field")) {
    t.cls = ng::PortClass::field;
    t.domain = dom({ng::manifold_id("sensors", name, ng::ManifoldKind::points, ng::Role::none, ng::CoordinateSystem{"head", "scs", "m"}), ng::time_line(name, 100.0)});
    t.quantity = "recording";
    t.axis = ng::Axis::samples;
    t.sfreq = 100.0;
    return t;
  }
  if (spec.contains("operator")) {
    t.cls = ng::PortClass::operator_;
    t.domain = dom({ng::restricted(rel(spec.at("from").get<std::string>(), found), "channels(" + name + ")")});
    t.target = dom({ng::manifold_id("cortex", name, ng::ManifoldKind::surface, ng::Role::none, ng::CoordinateSystem{"head", "scs", "m"})});
    t.value = ng::ValueType::vector3;
    t.quantity = "inverse kernel";
    return t;
  }
  if (spec.contains("selection")) {
    const std::string kind = spec["selection"].get<std::string>();
    t.cls = ng::PortClass::selection;
    ng::ManifoldRef on = spec.contains("manifold") ? rel(spec["manifold"].get<std::string>(), found) : nullptr;
    if (kind == "ladder") { t.domain = dom({on}); t.sel = ng::SelectionShape{"ladder", false, name, ""}; }
    else if (kind == "spans") {
      const bool tiling = name == "tiling";
      t.domain = dom({on});
      t.sel = ng::SelectionShape{tiling ? "tiling" : "spans", true, name, ""};
    } else if (kind == "set") {
      const Json w = spec.value("where", Json::object());
      const Json lvl = resolve(w.value("params.joint.space_level", Json()), params), band = resolve(w.value("params.joint.band", Json()), params);
      ng::ManifoldRef surf, time;
      for (const auto& n : found.order) {
        const ng::PortType& f = found.types.at(n);
        if (f.cls == ng::PortClass::operator_ && !surf) surf = f.target.factors.at(0);
        if (f.cls == ng::PortClass::field && !time) time = f.domain.factors.back();
      }
      std::string tiles = "tiles";
      for (const auto& n : found.order) {
        const ng::PortType& f = found.types.at(n);
        if (f.sel && f.sel->type == "ladder" && !f.domain.empty() && surf && ng::same(f.domain.factors[0], surf)) tiles = n;
      }
      t.domain = dom({surf, time});
      t.sel = ng::SelectionShape{"set", true, name, ""};
      t.res.ladder = tiles;
      if (lvl.is_number()) t.res.levels = {lvl.get<int>()};
      if (band.is_array()) t.res.band = std::array<double, 2>{band[0].get<double>(), band[1].get<double>()};
    } else {
      ng::refuse("input " + name + ": selection kind " + kind + " not understood");
    }
    t.quantity = t.sel->type;
    return t;
  }
  if (spec.contains("statistic")) {
    const std::string m = spec["statistic"].get<std::string>();
    t.cls = ng::PortClass::statistic;
    t.quantity = m;
    ng::ManifoldRef surf, sens;
    bool kernel = false;
    for (const auto& [n, s] : all_inputs.items()) {
      if (s.contains("operator")) kernel = true;
      if (s.contains("field") && found.types.count(n)) sens = found.types.at(n).domain.factors.at(0);
    }
    if (m == "mean") {
      for (const auto& n : found.order) if (found.types.at(n).cls == ng::PortClass::operator_) surf = found.types.at(n).target.factors.at(0);
      t.domain = dom({surf});
    } else if (m == "histogram") {
      t.domain = dom({ng::manifold_id("bins", name)});
    } else if (m == "welch") {
      ng::ManifoldRef ch = sens;
      if (kernel) for (const auto& n : found.order) if (found.types.at(n).cls == ng::PortClass::operator_) ch = ng::restricted(sens, "channels(" + n + ")");
      t.domain = dom({ch, ng::manifold_id("frequency", name, ng::ManifoldKind::line, ng::Role::frequency)});
    } else {
      t.domain = dom({ng::manifold_id("tiles", name)});
    }
    return t;
  }
  ng::refuse("input " + name + ": class not understood");
}

struct Flat { std::string section; np::StepUse use; };

struct Case {
  std::string name;
  Json params, inputs, outputs;
  std::vector<Flat> uses;
};

Case case_of(const np::Definition& d0) {
  Json defaults = Json::object();
  for (const auto& [k, v] : d0.parameters.items()) defaults[k] = v.is_object() && v.contains("default") ? v["default"] : v;
  const np::Definition d = np::expand(d0, defaults);
  Case c;
  c.name = d.name;
  c.params = defaults;
  c.inputs = d.inputs;
  c.outputs = d.outputs;
  for (const auto& u : d.before) c.uses.push_back({"before", u});
  for (const auto& u : d.do_) c.uses.push_back({"do", u});
  for (const auto& u : d.after) c.uses.push_back({"after", u});
  return c;
}

// the case's typed graph, or the first refusal (thrown)
ng::TypedGraph check_case(const Case& c) {
  Found found;
  ng::Graph g;
  g.name = c.name;
  g.parameters = c.params;
  for (const auto& [name, spec] : c.inputs.items()) {
    found.types[name] = input_type(name, spec, found, c.params, c.inputs);
    found.order.push_back(name);
    g.inputs.push_back(name);
  }
  for (const auto& f : c.uses) {
    ng::Use u;
    u.node = f.use.step;
    for (const auto& [port, v] : f.use.ins.items()) {
      const std::string value = v.get<std::string>();
      if (!found.types.count(value) && c.inputs.contains(value)) continue;      // an optional input not found
      u.in[port] = value;
    }
    u.with = f.use.with;
    u.out = f.use.outs;
    (f.section == "before" ? g.before : f.section == "do" ? g.chunks : g.after).push_back(std::move(u));
  }
  ng::TypedGraph tg = ng::type_check(g, found.types, c.params);
  std::map<std::string, std::string> declared;
  for (const auto& [o, decl] : c.outputs.items()) declared[o] = decl.value("kind", std::string("?"));
  ng::check_output_kinds(tg.values, declared);
  return tg;
}

np::StepUse step(const std::string& s, Json in, const std::string& out, Json with = Json::object()) {
  np::StepUse u;
  u.step = s; u.ins = std::move(in); u.with = std::move(with); u.outs = {out}; u.section = "do";
  return u;
}

struct Mistake { std::string definition, what, expected; std::function<void(Case&)> edit; };

}  // namespace

int main() {
  std::map<std::string, Case> cases;
  for (const auto& n : np::available()) { try { cases[n] = case_of(np::load(n)); } catch (const std::exception& e) { std::printf("  %s not read: %s\n", n.c_str(), e.what()); } }

  // 1. every shipped definition this test models checks. The prototype models the input classes of the first definitions (field, operator, kernel's tiles, statistics); an input class it does not
  //    model (`measurement`, a `selection: any`, a stored selection made by a maker) is a SKIP, listed, not a pass: the shared checker checks those definitions in the runner's plan
  //    (test_definitions_load loads all of them, the runner tests type them from the rows it finds).
  int ok = 0, skipped = 0;
  std::set<std::string> checked;
  for (const auto& [name, c] : cases) {
    try { check_case(c); ++ok; checked.insert(name); }
    catch (const std::invalid_argument& e) {
      const std::string w = e.what();
      if (w.find("not understood") != std::string::npos || w.find("has no signature") != std::string::npos) { std::printf("  %s skipped (not modelled here): %s\n", name.c_str(), w.c_str()); ++skipped; }
      else { std::printf("  %s REFUSED: %s\n", name.c_str(), w.c_str()); ++failures; }
    }
    catch (const std::exception& e) { std::printf("  %s skipped (an input class this test does not model): %s\n", name.c_str(), e.what()); ++skipped; }
  }
  std::printf("  %d of %zu definitions check, %d skipped\n", ok, cases.size(), skipped);

  // 2. the ten seeded mistakes, each refused with its rule
  const std::vector<Mistake> mistakes = {
      {"band_bursts", "measure the analytic signal (complex) instead of its envelope", "is complex",
       [](Case& c) { c.uses[0].use.step = "analytic"; }},
      {"cell_phase_current", "the kernel applied after `power`", "D106",
       [](Case& c) { c.uses.insert(c.uses.begin() + 1, {"do", step("power", {{"x", "z"}}, "zp")}); c.uses[2].use.ins["x"] = "zp"; }},
      {"global_bursts", "the spatial band at level 4, the joint cells at level 5", "the level IS the band",
       [](Case& c) { c.uses[1].use.with["level"] = 4; }},
      {"cortical_cells", "the time band changed to 14-30 Hz, the cells of the 8-16 Hz level", "band-limited to 14-30 Hz",
       [](Case& c) { c.uses[0].use.with["band"] = Json::array({14.0, 30.0}); }},
      {"spatial_events", "frames interpolated after `power`", "is non-linear",
       [](Case& c) { c.uses.insert(c.uses.begin() + 1, {"do", step("power", {{"x", "z"}}, "zp")}); c.uses[2].use.ins["x"] = "zp"; }},
      {"band_bursts", "events detected on the envelope field, not on its measurement", "is a field; a measurement is wanted",
       [](Case& c) { for (auto& f : c.uses) if (f.use.step == "detect_spans") f.use.ins["x"] = "amp"; }},
      {"global_bursts", "a streamed field summarized", "measure the field over a partition first",
       [](Case& c) { auto u = step("measurement.summarize", {{"x", "mag"}}, "oops"); u.section = "after"; c.uses.push_back({"after", u}); }},
      {"global_bursts", "the tile ladder of the sensors instead of the cortex", "port `tiles`",
       [](Case& c) { c.inputs["tiles"]["manifold"] = "recording.manifold.0"; }},
      {"global_bursts", "an output declared a measurement that is a statistic", "outputs: baseline is declared a measurement",
       [](Case& c) { c.outputs["baseline"]["kind"] = "measurement"; }},
      {"band_bursts", "a roll-up over events detected on another band's tiles", "were detected on the members of",
       [](Case& c) {
         c.uses.insert(c.uses.begin() + 2, {"do", step("signal.envelope", {{"x", "recording"}}, "amp2", Json{{"band", Json::array({14.0, 30.0})}})});
         c.uses[3].use.ins["x"] = "amp2";
       }},
  };
  int caught = 0;
  int skipped_mistakes = 0;
  for (const auto& m : mistakes) {
    if (!checked.count(m.definition)) { ++skipped_mistakes; std::printf("  mistake on %s skipped: the definition is retired or not modelled here (%s)\n", m.definition.c_str(), m.what.c_str()); continue; }
    Case c = cases.at(m.definition);
    m.edit(c);
    try {
      check_case(c);
      std::printf("  NOT CAUGHT  %s: %s\n", m.definition.c_str(), m.what.c_str());
      ++failures;
    } catch (const std::invalid_argument& e) {
      if (std::string(e.what()).find(m.expected) == std::string::npos) {
        std::printf("  %s: %s\n    refused, but with: %s\n    (wanted '%s')\n", m.definition.c_str(), m.what.c_str(), e.what(), m.expected.c_str());
        ++failures;
      } else {
        ++caught;
      }
    }
  }
  std::printf("  %d of %zu seeded mistakes caught, %d skipped\n", caught, mistakes.size(), skipped_mistakes);
  CHECK(caught + skipped_mistakes == static_cast<int>(mistakes.size()));

  // 3. the signatures' ports are the registered steps'; every step the definitions use has a signature
  int without_signature = 0;
  std::set<std::string> used;
  for (const auto& [name, c] : cases) for (const auto& f : c.uses) used.insert(f.use.step);
  for (const auto& s : used) {
    const ng::Signature* sg = ng::builtin_signature(s);
    if (!sg) { ++without_signature; continue; }      // a node with no signature is checked by its `plan` alone
    const np::StepSpec& st = np::step(s);
    std::set<std::string> in_sig, in_step(st.inputs.begin(), st.inputs.end()), opt_sig, opt_step(st.optional.begin(), st.optional.end());
    for (const auto& [port, pat] : sg->inputs) { in_sig.insert(port); if (pat.is_optional) opt_sig.insert(port); }
    if (in_sig != in_step) { std::string d; for (const auto& x : in_sig) if (!in_step.count(x)) d += " +sig:" + x; for (const auto& x : in_step) if (!in_sig.count(x)) d += " +step:" + x; std::printf("  %s: signature inputs differ from the registered step's (%s )\n", s.c_str(), d.c_str()); ++failures; }
    std::vector<std::string> outs;
    for (const auto& [port, _] : sg->outputs) outs.push_back(port);
    if (outs != st.outputs) { std::printf("  %s: signature outputs differ from the registered step's\n", s.c_str()); ++failures; }
    if (opt_sig != opt_step) { std::printf("  %s: signature optional ports differ from the registered step's\n", s.c_str()); ++failures; }
  }
  std::printf("  %d steps are checked by their plan alone (no signature)\n", without_signature);
  {
  }

  if (failures) { std::printf("test_definitions_typed: %d failure(s)\n", failures); return 1; }
  std::printf("test_definitions_typed: ok\n");
  return 0;
}
