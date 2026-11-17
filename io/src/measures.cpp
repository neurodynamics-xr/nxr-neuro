// nxr-io — measurements (see measures.h): the Studio's measure registry, measurement table and repository.
#include "nxr/io/measures.h"
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
std::string measure_reduction(const std::string& name) {
  const auto s = measure_spec(name);
  return s && (*s)["reduction"].is_string() ? (*s)["reduction"].get<std::string>() : "mean";
}

std::string ensure_measure_dictionary(Database& db, const std::optional<std::string>& by) {
  const Json& reg = measure_registry();
  const std::string name = reg["dictionary"]["name"];
  auto entries = [&](const std::string& id) {
    db.execute("DELETE FROM dictionary_entry WHERE dictionary_id = ?", {Value(id)});
    for (const auto& m : reg["measures"]) {
      const Json params = m["weight"].is_null() ? Json(nullptr) : Json(Json{{"weight", m["weight"]}}.dump());
      db.execute("INSERT INTO dictionary_entry (dictionary_id, code, name, color_json, measurement, measurement_unit, function, params_json, attributes_json, ordinal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                 {Value(id), v(m["code"]), v(m["name"]), Value(std::monostate{}), Value(std::monostate{}), v(m["unit"]), v(m["reduction"]), v(params),
                  Value(Json{{"description", m["description"]}}.dump()), v(m["code"])});
    }
  };
  if (const auto have = db.one("SELECT * FROM dictionary WHERE scope = 'app' AND name = ?", {Value(name)})) {
    const std::string id = std::get<std::string>(have->at("id"));
    std::set<std::string> known;
    for (const Row& e : db.all("SELECT name FROM dictionary_entry WHERE dictionary_id = ?", {Value(id)})) known.insert(std::get<std::string>(e.at("name")));
    bool missing = false;
    for (const auto& m : reg["measures"]) if (!known.count(m["name"].get<std::string>())) missing = true;
    if (missing) db.tx([&] {
      // the Studio rewrites the row as it is (the dictionary's own `by` passes through measureDictionaryRows, then `have` is upserted unchanged) and its entries
      entries(id);
    });
    return id;
  }
  const std::string id = uuid7();
  db.tx([&] {
    db.execute("INSERT INTO dictionary (id, name, scope, dataset_id, subject_id, description, created_utc, created_by, modified_utc, modified_by) VALUES (?, ?, ?, NULL, NULL, ?, NULL, ?, NULL, ?)",
               {Value(id), Value(name), v(reg["dictionary"]["scope"]), v(reg["dictionary"]["description"]), by ? Value(*by) : Value(std::monostate{}), by ? Value(*by) : Value(std::monostate{})});
    entries(id);
  });
  return id;
}

}  // namespace nxr::io
