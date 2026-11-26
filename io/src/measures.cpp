// nxr-io — measurements (see measures.h): the Studio's measure registry, measurement table and repository.
#include "nxr/io/measures.h"
#include "nxr/io/selection_store.h"
#include <cstring>
#include "nxr/io/writer.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/statistic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

#include "nxr/io/bundle.h"
#include "nxr/io/errors.h"
#include "nxr/io/rollup.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/io/store.h"

namespace nxr::io {

namespace {
const char* const kRegistryText =
#include "gen/measures_json.inc"
    ;
// nxr-neuro's own measures (schema/measures_nxr.json;), codes in its reserved range
const char* const kOwnText =
#include "gen/measures_nxr_json.inc"
    ;
Value v(const Json& x) {
  if (x.is_null()) return Value(std::monostate{});
  if (x.is_number_integer()) return Value(x.get<std::int64_t>());
  if (x.is_number()) return Value(x.get<double>());
  if (x.is_boolean()) return Value(static_cast<std::int64_t>(x.get<bool>()));
  if (x.is_string()) return Value(x.get<std::string>());
  return Value(x.dump());
}
}  // namespace

// the derived formulas (measurements.ts), by name; then nxr-neuro's own (the phase family: ratios of sums, never sums of ratios)
double measure_formula(const std::string& name, const std::map<std::string, double>& m) {
  auto g = [&](const char* k) { return m.at(k); };
  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  auto ratio = [&](double a, double b) { return b > 0 ? a / b : kNaN; };   // no power on the member: no phase gradient (left out, as a non-finite value)
  if (name == "relative_power") return g("total_power") != 0 ? g("power_sum") / g("total_power") : 0.0;   // Brainstorm's relative power (a zero total gives 0)
  if (name == "phase_gradient_1") return ratio(g("phase_current_1"), g("phase_power"));
  if (name == "phase_gradient_2") return ratio(g("phase_current_2"), g("phase_power"));
  // the vector phase gradient of the cells and bursts (spec docs/specs/2026-11-13-burst-phase-current.md): each component of the mean phase current over the mean power (rad/m); its negative is the direction the phase travels
  if (name == "phase_gradient_x") return ratio(g("phase_current_x"), g("power"));
  if (name == "phase_gradient_y") return ratio(g("phase_current_y"), g("power"));
  if (name == "phase_gradient_z") return ratio(g("phase_current_z"), g("power"));
  if (name == "phase_gradient_magnitude") return ratio(std::hypot(g("phase_current_1"), g("phase_current_2")), g("phase_power"));
  if (name == "phase_gradient_direction_deg") return std::hypot(g("phase_current_1"), g("phase_current_2")) > 0 ? std::atan2(g("phase_current_2"), g("phase_current_1")) * 180.0 / M_PI : kNaN;
  if (name == "travelling_index") return ratio(std::hypot(g("phase_current_1"), g("phase_current_2")), std::sqrt(std::max(0.0, g("phase_power") * g("phase_gradient_power"))));
  if (name == "phase_coherence") return ratio(std::hypot(g("phase_current_1"), g("phase_current_2")), g("phase_current_magnitude"));
  if (name == "quadrature_fraction" || name == "tile_pair_quadrature_fraction") return ratio(std::sqrt(std::max(0.0, g(name == "quadrature_fraction" ? "cross_spectrum_quadrature_norm2" : "tile_pair_cross_spectrum_quadrature_norm2"))), std::sqrt(std::max(0.0, g(name == "quadrature_fraction" ? "cross_spectrum_norm2" : "tile_pair_cross_spectrum_norm2"))));
  // the crossings of the tiles (spec piece 7): R = |S|/n and the unbiased rho2 = (|S|^2 - n)/(n(n - 1)) of a tile's n crossings, from the stored vector sum and count (derived at read, never stored)
  if (name == "crossing_consistency") return g("crossings") > 0 ? std::sqrt(g("crossing_sum_x") * g("crossing_sum_x") + g("crossing_sum_y") * g("crossing_sum_y") + g("crossing_sum_z") * g("crossing_sum_z")) / g("crossings") : kNaN;
  if (name == "crossing_rho2") {
    const double n = g("crossings"), s2 = g("crossing_sum_x") * g("crossing_sum_x") + g("crossing_sum_y") * g("crossing_sum_y") + g("crossing_sum_z") * g("crossing_sum_z");
    return n > 1 ? (s2 - n) / (n * (n - 1)) : kNaN;
  }
  // the split of a phase 1-form (nxr::signal::splitPhase): shares of its ⋆1 energy, ratios of the members' sums of vertexEnergy (never a sum of ratios)
  if (name == "exact_share") return ratio(g("exact_energy"), g("one_form_energy"));
  if (name == "coexact_share") return ratio(g("coexact_energy"), g("one_form_energy"));
  if (name == "residual_share") return g("one_form_energy") > 0 ? std::max(0.0, 1.0 - (g("exact_energy") + g("coexact_energy")) / g("one_form_energy")) : kNaN;
  if (name == "exact_misfit_share") return ratio(g("exact_misfit_energy"), g("one_form_energy"));
  if (name == "variance") return g("mean_square") - g("mean") * g("mean");
  if (name == "sd") return std::sqrt(std::max(0.0, g("mean_square") - g("mean") * g("mean")));
  if (name == "rms") return std::sqrt(std::max(0.0, g("mean_square")));
  if (name == "shape_factor") return ratio(std::sqrt(std::max(0.0, g("mean_square"))), g("mean_abs"));                                  // MATLAB ShapeFactor
  if (name == "impulse_factor") return ratio(g("abs_max"), g("mean_abs"));                                                              // MATLAB ImpulseFactor
  if (name == "clearance_factor") return ratio(g("abs_max"), g("mean_sqrt_abs") * g("mean_sqrt_abs"));                                  // MATLAB ClearanceFactor
  // the standardized 3rd and 4th central moments WITHIN a member (1/n, MATLAB skewness and kurtosis with flag 1; kurtosis not excess), from the
  // rolled-up raw moments: m2 = E2 − μ², m3 = E3 − 3μE2 + 2μ³, m4 = E4 − 4μE3 + 6μ²E2 − 3μ⁴. NaN for a member with no spread.
  if (name == "skewness" || name == "kurtosis") {
    const double mu = g("mean"), e2 = g("mean_square"), e3 = g("mean_cube");
    const double m2 = e2 - mu * mu;
    if (!(m2 > 0)) return kNaN;
    if (name == "skewness") return (e3 - 3 * mu * e2 + 2 * mu * mu * mu) / std::pow(m2, 1.5);
    const double e4 = g("mean_fourth");
    return (e4 - 4 * mu * e3 + 6 * mu * mu * e2 - 3 * mu * mu * mu * mu) / (m2 * m2);
  }
  if (name == "peak_to_peak") return g("max") - g("min");
  if (name == "crest_factor") return g("mean_square") > 0 ? g("abs_max") / std::sqrt(g("mean_square")) : 0.0;
  if (name == "envelope_variance") return g("power") - g("amplitude") * g("amplitude");
  throw Error(code::unsupported, "measure " + name + ": a derived measure with no formula here");
}
// db.measurements.query (database.ts): one stored measure rolled up


// the Studio's registry followed by nxr-neuro's own; an own code outside the reserved range, or a code or name in both, is refused (a build error caught by test_measures)
const Json& measure_registry() {
  static const Json j = [] {
    Json r = Json::parse(kRegistryText);
    const Json own = Json::parse(kOwnText);
    const std::int64_t lo = own["range"][0], hi = own["range"][1];
    for (const auto& m : own["measures"]) {
      const std::int64_t c = m["code"];
      if (c < lo || c > hi) throw Error(code::invalid_argument, "measure " + m["name"].get<std::string>() + ": code " + std::to_string(c) + " is outside nxr-neuro's range");
      for (const auto& e : r["measures"])
        if (e["code"] == m["code"] || e["name"] == m["name"]) throw Error(code::invalid_argument, "measure " + m["name"].get<std::string>() + ": its code or name is already the Studio's");
      r["measures"].push_back(m);
    }
    return r;
  }();
  return j;
}
std::optional<Json> measure_spec(const std::string& name) {
  for (const auto& m : measure_registry()["measures"]) if (m["name"] == name) return m;
  return std::nullopt;
}
bool measure_is_derived(const std::string& name) {
  const auto s = measure_spec(name);
  return s && !(*s)["derived_from"].is_null();
}
const char* merge_rule_name(MergeRule r) {
  switch (r) {
    case MergeRule::sum: return "sum";
    case MergeRule::max: return "max";
    case MergeRule::min: return "min";
    case MergeRule::mean: return "mean";
    case MergeRule::none: return "none";
    case MergeRule::derived: return "derived";
  }
  return "sum";
}
namespace {
MergeRule rule_of_text(const std::string& name, const std::string& m) {
  for (const MergeRule r : {MergeRule::sum, MergeRule::max, MergeRule::min, MergeRule::mean, MergeRule::none, MergeRule::derived})
    if (m == merge_rule_name(r)) return r;
  throw Error(code::invalid_argument, "measure " + name + ": merge '" + m + "' is not a merge rule (sum, max, min, mean, none or derived)");
}
const char* kDeclBegin = "-- \xe2\x94\x80\xe2\x94\x80 measure_declaration (begin";
const char* kDeclEnd = "-- \xe2\x94\x80\xe2\x94\x80 measure_declaration (end)";
bool table_exists(const Database& db, const char* name) { return db.one("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", {Value(std::string(name))}).has_value(); }
}  // namespace

std::string undeclared_refusal(const std::string& name) {
  return "measure " + name + " is neither declared nor registered, so how its values merge is unknown and its members are never rolled up: declare it when writing it "
         "(its merge rule: sum, max, min, mean or none)";
}
MergeRule measure_merge(const std::string& name) {
  if (const auto s = measure_spec(name)) {
    const std::string m = (*s).value("merge", std::string());
    if (m.empty()) throw Error(code::invalid_argument, "measure " + name + ": the registry gives it no merge rule (sum, max, min, mean, none or derived)");
    return rule_of_text(name, m);
  }
  throw Error(code::unsupported, undeclared_refusal(name));
}

MeasureDeclaration measure_declaration_of(const Json& j) {
  if (!j.is_object()) throw Error(code::invalid_argument, "a measure declaration is an object {name, merge, form, unit, weight, description}");
  static const std::set<std::string> keys = {"name", "merge", "form", "unit", "weight", "description"};
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (!keys.count(it.key())) throw Error(code::invalid_argument, "a measure declaration has no key '" + it.key() + "' (name, merge, form, unit, weight, description)");
    if (!it.value().is_null() && !it.value().is_string()) throw Error(code::invalid_argument, "a measure declaration's '" + it.key() + "' is a text");
  }
  MeasureDeclaration d;
  if (!j.contains("name") || !j["name"].is_string() || j["name"].get<std::string>().empty()) throw Error(code::invalid_argument, "a measure declaration names its measure (name)");
  d.name = j["name"].get<std::string>();
  auto o = [&](const char* k) { return j.contains(k) && j[k].is_string() ? std::optional<std::string>(j[k].get<std::string>()) : std::nullopt; };
  d.merge = o("merge"); d.form = o("form"); d.unit = o("unit"); d.weight = o("weight"); d.description = o("description");
  return d;
}
Json measure_declaration_json(const MeasureDeclaration& d) {
  Json j = Json{{"name", d.name}};
  if (d.merge) j["merge"] = *d.merge;
  if (d.form) j["form"] = *d.form;
  if (d.unit) j["unit"] = *d.unit;
  if (d.weight) j["weight"] = *d.weight;
  if (d.description) j["description"] = *d.description;
  return j;
}

bool ensure_measure_declarations(Database& db) {
  if (table_exists(db, "measure_declaration")) return false;
  const std::string s = model_sql();
  const auto a = s.find(kDeclBegin), b = s.find(kDeclEnd);
  if (a == std::string::npos || b == std::string::npos || b < a) throw Error(code::not_a_database, "model.sql: no measure_declaration block between its markers");
  const auto line = s.find('\n', a);
  const std::string ddl = s.substr(line + 1, b - line - 1);
  db.tx([&] {
    db.exec_script(ddl);
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {Value(std::string("table:measure_declaration")), Value(now_utc() + " added in place (additive; user_version unchanged)")});
  });
  return true;
}

bool ensure_measurement_producer(Database& db) {
  if (has_column(db, "selection_measurement", "producer_json")) return false;
  db.tx([&] {
    db.execute("ALTER TABLE selection_measurement ADD COLUMN producer_json TEXT");
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {Value(std::string("columns:selection_measurement.producer_json")), Value(now_utc() + " added in place (additive; user_version unchanged)")});
  });
  return true;
}

bool ensure_measurement_rows_manifold(Database& db) {
  if (has_column(db, "selection_measurement", "rows_manifold_id")) return false;
  ensure_measurement_producer(db);   // the columns in the model's order
  db.tx([&] {
    db.execute("ALTER TABLE selection_measurement ADD COLUMN rows_manifold_id TEXT REFERENCES manifold (id) ON DELETE CASCADE");
    db.execute("DROP INDEX IF EXISTS selection_measurement_key");
    db.execute("CREATE UNIQUE INDEX selection_measurement_key ON selection_measurement (selection_id, coalesce(code, -1), coalesce(of_field_id, ''), coalesce(of_statistic_id, ''), coalesce(of_index, -1), "
               "coalesce(within_selection_id, ''), coalesce(within_code, -1), measure, coalesce(rows_manifold_id, ''))");
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {Value(std::string("columns:selection_measurement.rows_manifold_id")), Value(now_utc() + " added in place, the key made again with it (additive; user_version unchanged)")});
  });
  return true;
}

std::optional<Json> declared_measure(const Database& db, const std::string& name) {
  if (!table_exists(db, "measure_declaration")) return std::nullopt;
  const auto r = db.one("SELECT * FROM measure_declaration WHERE name = ?", {Value(name)});
  if (!r) return std::nullopt;
  Json j = Json{{"name", name}, {"merge", text(*r, "merge")}};
  for (const char* k : {"form", "unit", "weight", "description", "declared_utc", "declared_by"})
    if (const auto v = opt_text(*r, k)) j[k] = *v;
  return j;
}

Json declare_measure(Database& db, const MeasureDeclaration& d, const std::optional<std::string>& by) {
  const std::string who = "measure " + d.name;
  const auto reg = measure_spec(d.name);
  if (!reg && !d.merge) throw Error(code::invalid_argument, who + " is not in the registry: its declaration gives its merge rule (sum, max, min, mean or none)");
  if (d.merge) rule_of_text(d.name, *d.merge);
  // what the declaration says, the registry's values filled in for a known name; a value that differs from the registry's is a contradiction
  MeasureDeclaration full = d;
  if (reg) {
    auto agree = [&](const char* key, std::optional<std::string>& mine) {
      const Json& r = (*reg)[key];
      if (!r.is_string()) return;
      if (mine && *mine != r.get<std::string>())
        throw Error(code::invalid_argument, who + ": the declaration gives " + key + " '" + *mine + "', the registry '" + r.get<std::string>() + "' (one name keeps one meaning: use another name)");
      if (!mine) mine = r.get<std::string>();
    };
    agree("merge", full.merge); agree("form", full.form); agree("unit", full.unit); agree("weight", full.weight);
    if (!full.description && (*reg)["description"].is_string()) full.description = (*reg)["description"].get<std::string>();
  } else if (*full.merge == "derived") {
    throw Error(code::invalid_argument, who + ": a derived measure is computed at read by a formula io holds, and only registered measures have one; store its additive ingredients (each declared) and derive it at read");
  }
  ensure_measure_declarations(db);
  if (const auto had = declared_measure(db, d.name)) {
    for (const auto& [key, mine] : std::vector<std::pair<const char*, const std::optional<std::string>*>>{{"merge", &full.merge}, {"form", &full.form}, {"unit", &full.unit}, {"weight", &full.weight}}) {
      const std::optional<std::string> theirs = had->contains(key) ? std::optional<std::string>((*had)[key].get<std::string>()) : std::nullopt;
      if (*mine && theirs && **mine != *theirs)
        throw Error(code::invalid_argument, who + ": the subject already declares " + key + " '" + *theirs + "'; this declaration gives '" + **mine + "' (one name keeps one meaning: use another name)");
      if (*mine && !theirs) throw Error(code::invalid_argument, who + ": the subject declares it without " + key + "; this declaration gives '" + **mine + "' (a declaration is not changed: use another name)");
    }
    return *had;
  }
  db.execute("INSERT INTO measure_declaration (name, merge, form, unit, weight, description, declared_utc, declared_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
             {Value(d.name), Value(*full.merge), V(full.form), V(full.unit), V(full.weight), V(full.description), Value(now_utc()), V(by)});
  return *declared_measure(db, d.name);
}

MergeRule measure_merge(const Database& db, const std::string& name) {
  if (const auto d = declared_measure(db, name)) return rule_of_text(name, (*d)["merge"].get<std::string>());
  return measure_merge(name);
}
ReadRule measure_read_rule(const Database& db, const std::string& name) {
  if (const auto d = declared_measure(db, name)) return {rule_of_text(name, (*d)["merge"].get<std::string>()), true};
  if (measure_spec(name)) return {measure_merge(name), true};
  return {MergeRule::none, false};
}
Json measure_merge_table() {
  Json t = Json::object();
  for (const auto& m : measure_registry()["measures"]) t[m["name"].get<std::string>()] = m.value("merge", std::string());
  return t;
}
std::string measure_reduction(const std::string& name) {
  const MergeRule r = measure_merge(name);
  return r == MergeRule::derived ? "mean" : merge_rule_name(r);
}

std::string ensure_measure_dictionary(Database& db, const std::optional<std::string>& by) {
  const Json& reg = measure_registry();
  selection::AppLookup spec;
  spec.name = reg["dictionary"]["name"].get<std::string>();
  spec.scope = reg["dictionary"]["scope"].get<std::string>();
  spec.description = reg["dictionary"]["description"].get<std::string>();
  spec.by = by;
  spec.refresh_when_incomplete = true;   // the Studio rewrites the entries when the registry names a measure the dictionary lacks
  for (const auto& m : reg["measures"]) {
    selection::AppEntry e;
    e.code = m["code"].get<std::int64_t>();
    e.name = m["name"].get<std::string>();
    e.ordinal = e.code;
    if (m["unit"].is_string()) e.measurement_unit = m["unit"].get<std::string>();
    if (m.value("merge", std::string()) != "derived" && m["merge"].is_string()) e.function = m["merge"].get<std::string>();   // the dictionary's `function` is the merge rule (the one key; the Studio's `reduction`)
    if (!m["weight"].is_null()) e.params = Json{{"weight", m["weight"]}};
    e.attributes = Json{{"description", m["description"]}};
    spec.entries.push_back(std::move(e));
  }
  return selection::ensure_app_lookup(db, spec);
}

}  // namespace nxr::io
