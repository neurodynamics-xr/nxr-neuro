// nxr::measurement — the measures registry, the ingredients and how a value is read from them (measurement/measures.h). Compiled with -ffp-contract=off: the accumulation is a port
// of the Studio's measureOverPartition (no fused multiply-add in JavaScript), so its sums equal the Studio's to the last bit.
#include "nxr/measurement/measures.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace nxr::measurement {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

MeasureSpec S(const char* name, const char* form, const char* ingredient, Merge merge, const char* weight = "", std::vector<std::string> derived = {}) {
  return MeasureSpec{name, form, ingredient, merge, weight, std::move(derived)};
}

std::vector<MeasureSpec> standard_table() {
  using M = Merge;
  std::vector<MeasureSpec> t = {
      // the field as it is (the Studio's measures.json)
      S("mean", "none", "sum", M::mean, "area"), S("mean_square", "none", "sum_sq", M::mean, "area"), S("sum", "none", "sum", M::sum), S("sum_square", "none", "sum_sq", M::sum),
      S("weight", "none", "weight", M::sum), S("max", "none", "max", M::max), S("min", "none", "min", M::min), S("abs_max", "none", "abs_max", M::max), S("count", "none", "count", M::sum),
      S("median", "none", "held", M::none),
      // nxr-neuro's own raw moments (2026-11-13): the ingredients of the shape, impulse and clearance factors, skewness and kurtosis, derived at read
      S("mean_abs", "none", "sum_abs", M::mean, "area"), S("mean_sqrt_abs", "none", "sum_sqrt_abs", M::mean, "area"), S("mean_cube", "none", "sum_cube", M::mean, "area"),
      S("mean_fourth", "none", "sum_fourth", M::mean, "area"),
      // a complex field's forms
      S("amplitude", "magnitude", "sum", M::mean, "n_elements"), S("power", "power", "sum", M::mean, "n_elements"), S("power_sum", "power", "sum", M::sum),
      S("envelope_sum", "magnitude", "sum", M::sum), S("envelope_max", "magnitude", "max", M::max), S("phase_current", "phase_current", "sum", M::sum),
      S("phase_current_1", "phase_current", "sum", M::sum), S("phase_current_2", "phase_current", "sum", M::sum), S("phase_current_magnitude", "phase_current", "sum", M::sum),
      S("total_power", "power", "sum", M::sum), S("phase_power", "power", "sum", M::sum), S("phase_gradient_power", "power", "sum", M::sum),
      // forms on the complex's edges and faces (the tile integrals)
      S("outflow", "edge_1form", "sum", M::sum), S("circulation", "face_2form", "sum", M::sum),
      // the ⋆1 energies of a phase 1-form and of its split's parts (vertexEnergy: each edge's half at each end), summed over a member
      S("one_form_energy", "edge_1form", "sum", M::sum), S("exact_energy", "edge_1form", "sum", M::sum), S("coexact_energy", "edge_1form", "sum", M::sum),
      S("exact_misfit_energy", "edge_1form", "sum", M::sum),
      // the member's own (no field): area is Σw, n_elements the count
      S("area", "member", "weight", M::sum), S("n_elements", "member", "count", M::sum), S("duration", "member", "member", M::sum), S("n_tiles", "member", "member", M::sum),
      S("n_cells", "member", "member", M::sum), S("interval_next", "member", "member", M::none), S("centre", "member", "member", M::none), S("first_cell", "member", "member", M::none),
      S("last_cell", "member", "member", M::none), S("path_length", "member", "member", M::none), S("levi_civita", "frames", "member", M::none),
      // the orientation tensor and the gradient sums (nxr measure.vectors.band_tensor, vector_stats)
      S("tensor_11", "hermitian3", "sum", M::sum), S("tensor_22", "hermitian3", "sum", M::sum), S("tensor_nn", "hermitian3", "sum", M::sum), S("tensor_12_re", "hermitian3", "sum", M::sum),
      S("tensor_12_im", "hermitian3", "sum", M::sum), S("tensor_1n_re", "hermitian3", "sum", M::sum), S("tensor_1n_im", "hermitian3", "sum", M::sum), S("tensor_2n_re", "hermitian3", "sum", M::sum),
      S("tensor_2n_im", "hermitian3", "sum", M::sum), S("tensor_total", "hermitian3", "sum", M::sum),
      S("gradient_1", "gradient", "sum", M::sum), S("gradient_2", "gradient", "sum", M::sum), S("gradient_n", "gradient", "sum", M::sum), S("gradient_11", "gradient", "sum", M::sum),
      S("gradient_22", "gradient", "sum", M::sum), S("gradient_12", "gradient", "sum", M::sum), S("gradient_magnitude", "gradient", "sum", M::sum), S("gradient_weight", "gradient", "sum", M::sum),
      S("cross_spectrum_norm2", "outer", "sum", M::sum), S("cross_spectrum_quadrature_norm2", "outer", "sum", M::sum), S("tile_pair_cross_spectrum_norm2", "outer", "sum", M::sum),
      S("tile_pair_cross_spectrum_quadrature_norm2", "outer", "sum", M::sum),
      S("crossing_sum_x", "none", "sum", M::sum), S("crossing_sum_y", "none", "sum", M::sum), S("crossing_sum_z", "none", "sum", M::sum), S("crossings", "none", "count", M::sum),
      S("phase_current_x", "phase_current", "sum", M::mean, "n_elements"), S("phase_current_y", "phase_current", "sum", M::mean, "n_elements"),
      S("phase_current_z", "phase_current", "sum", M::mean, "n_elements"),
  };
  // derived at read, never stored or rolled up (merge none)
  const std::vector<std::pair<const char*, std::vector<std::string>>> derived = {
      {"variance", {"mean_square", "mean"}}, {"sd", {"mean_square", "mean"}}, {"rms", {"mean_square"}}, {"peak_to_peak", {"max", "min"}}, {"crest_factor", {"abs_max", "mean_square"}},
      {"envelope_variance", {"power", "amplitude"}}, {"shape_factor", {"mean_square", "mean_abs"}}, {"impulse_factor", {"abs_max", "mean_abs"}},
      {"clearance_factor", {"abs_max", "mean_sqrt_abs"}}, {"skewness", {"mean", "mean_square", "mean_cube"}}, {"kurtosis", {"mean", "mean_square", "mean_cube", "mean_fourth"}},
      {"phase_gradient_1", {"phase_current_1", "phase_power"}}, {"phase_gradient_2", {"phase_current_2", "phase_power"}},
      {"phase_gradient_magnitude", {"phase_current_1", "phase_current_2", "phase_power"}}, {"phase_gradient_direction_deg", {"phase_current_1", "phase_current_2"}},
      {"travelling_index", {"phase_current_1", "phase_current_2", "phase_power", "phase_gradient_power"}}, {"phase_coherence", {"phase_current_1", "phase_current_2", "phase_current_magnitude"}},
      {"quadrature_fraction", {"cross_spectrum_norm2", "cross_spectrum_quadrature_norm2"}},
      {"tile_pair_quadrature_fraction", {"tile_pair_cross_spectrum_norm2", "tile_pair_cross_spectrum_quadrature_norm2"}},
      {"phase_gradient_x", {"phase_current_x", "power"}}, {"phase_gradient_y", {"phase_current_y", "power"}}, {"phase_gradient_z", {"phase_current_z", "power"}},
      {"relative_power", {"power_sum", "total_power"}}, {"crossing_consistency", {"crossing_sum_x", "crossing_sum_y", "crossing_sum_z", "crossings"}}, {"crossing_rho2", {"crossing_sum_x", "crossing_sum_y", "crossing_sum_z", "crossings"}},
      {"exact_share", {"exact_energy", "one_form_energy"}}, {"coexact_share", {"coexact_energy", "one_form_energy"}},
      {"residual_share", {"one_form_energy", "exact_energy", "coexact_energy"}}, {"exact_misfit_share", {"exact_misfit_energy", "one_form_energy"}},
  };
  for (const auto& d : derived) t.push_back(S(d.first, "none", "", M::none, "", d.second));
  return t;
}

double finite_or_nan(double v) { return std::isfinite(v) ? v : kNaN; }
}  // namespace

const char* merge_name(Merge m) {
  switch (m) {
    case Merge::sum: return "sum";
    case Merge::max: return "max";
    case Merge::min: return "min";
    case Merge::mean: return "mean";
    case Merge::none: return "none";
  }
  return "none";
}

Merge merge_from(const std::string& name) {
  if (name == "sum") return Merge::sum;
  if (name == "max") return Merge::max;
  if (name == "min") return Merge::min;
  if (name == "mean") return Merge::mean;
  if (name.empty() || name == "none" || name == "null") return Merge::none;
  throw std::invalid_argument("measurement: the merge rule '" + name + "' is not one of sum, max, min, mean, none");
}

Registry::Registry(std::vector<MeasureSpec> specs) : specs_(std::move(specs)) {
  std::unordered_map<std::string, int> seen;
  for (const MeasureSpec& s : specs_) if (s.name.empty() || ++seen[s.name] > 1) throw std::invalid_argument("measurement: the registry holds an empty or repeated measure name ('" + s.name + "')");
}

const Registry& Registry::standard() {
  static const Registry r(standard_table());
  return r;
}

const MeasureSpec* Registry::find(const std::string& name) const {
  for (const MeasureSpec& s : specs_) if (s.name == name) return &s;
  return nullptr;
}

const MeasureSpec& Registry::at(const std::string& name) const {
  if (const MeasureSpec* s = find(name)) return *s;
  std::string valid;
  for (std::size_t i = 0; i < specs_.size() && i < 24; ++i) valid += (i ? ", " : "") + specs_[i].name;
  throw std::invalid_argument("measurement: '" + name + "' is not a measure of the registry (" + valid + ", ...)");
}

bool Registry::derived(const std::string& name) const {
  const MeasureSpec* s = find(name);
  return s && !s->derived_from.empty();
}

double Registry::derive(const std::string& name, const std::function<double(const std::string&)>& g) const {
  const MeasureSpec& s = at(name);
  if (s.derived_from.empty()) throw std::invalid_argument("measurement: '" + name + "' is not derived (it is read from the ingredients)");
  auto sq = [](double x) { return x * x; };
  double v;
  if (name == "variance") v = g("mean_square") - sq(g("mean"));
  else if (name == "sd") v = std::sqrt(std::max(g("mean_square") - sq(g("mean")), 0.0));
  else if (name == "rms") v = std::sqrt(g("mean_square"));
  else if (name == "peak_to_peak") v = g("max") - g("min");
  else if (name == "crest_factor") v = g("abs_max") / std::sqrt(g("mean_square"));
  else if (name == "envelope_variance") v = g("power") - sq(g("amplitude"));
  else if (name == "shape_factor") v = std::sqrt(g("mean_square")) / g("mean_abs");
  else if (name == "impulse_factor") v = g("abs_max") / g("mean_abs");
  else if (name == "clearance_factor") v = g("abs_max") / sq(g("mean_sqrt_abs"));
  else if (name == "skewness" || name == "kurtosis") {
    const double m = g("mean"), m2raw = g("mean_square"), m3raw = g("mean_cube");
    const double m2 = m2raw - m * m;
    if (name == "skewness") v = (m3raw - 3 * m * m2raw + 2 * m * m * m) / std::pow(m2, 1.5);
    else v = (g("mean_fourth") - 4 * m * m3raw + 6 * m * m * m2raw - 3 * m * m * m * m) / (m2 * m2);
  }
  else if (name == "relative_power") v = g("total_power") != 0 ? g("power_sum") / g("total_power") : 0.0;   // Brainstorm's relative power (a zero total gives 0)
  else if (name == "phase_gradient_1") v = g("phase_current_1") / g("phase_power");
  else if (name == "phase_gradient_2") v = g("phase_current_2") / g("phase_power");
  else if (name == "phase_gradient_magnitude") v = std::hypot(g("phase_current_1"), g("phase_current_2")) / g("phase_power");
  else if (name == "phase_gradient_direction_deg") v = std::atan2(g("phase_current_2"), g("phase_current_1")) * 180.0 / M_PI;
  else if (name == "travelling_index") v = std::hypot(g("phase_current_1"), g("phase_current_2")) / std::sqrt(g("phase_power") * g("phase_gradient_power"));
  else if (name == "phase_coherence") v = std::hypot(g("phase_current_1"), g("phase_current_2")) / g("phase_current_magnitude");
  else if (name == "quadrature_fraction") v = std::sqrt(g("cross_spectrum_quadrature_norm2") / g("cross_spectrum_norm2"));
  else if (name == "tile_pair_quadrature_fraction") v = std::sqrt(g("tile_pair_cross_spectrum_quadrature_norm2") / g("tile_pair_cross_spectrum_norm2"));
  else if (name == "phase_gradient_x") v = g("phase_current_x") / g("power");
  else if (name == "phase_gradient_y") v = g("phase_current_y") / g("power");
  else if (name == "phase_gradient_z") v = g("phase_current_z") / g("power");
  else if (name == "crossing_consistency" || name == "crossing_rho2") {
    const double n = g("crossings"), s2 = sq(g("crossing_sum_x")) + sq(g("crossing_sum_y")) + sq(g("crossing_sum_z"));
    v = name == "crossing_consistency" ? std::sqrt(s2) / n : (s2 - n) / (n * (n - 1));
  }
  // the split of a phase 1-form: shares of its ⋆1 energy (ratios of the sums over a member; NaN where it has no energy)
  else if (name == "exact_share") v = g("one_form_energy") > 0 ? g("exact_energy") / g("one_form_energy") : kNaN;
  else if (name == "coexact_share") v = g("one_form_energy") > 0 ? g("coexact_energy") / g("one_form_energy") : kNaN;
  else if (name == "residual_share") v = g("one_form_energy") > 0 ? std::max(0.0, 1.0 - (g("exact_energy") + g("coexact_energy")) / g("one_form_energy")) : kNaN;
  else if (name == "exact_misfit_share") v = g("one_form_energy") > 0 ? g("exact_misfit_energy") / g("one_form_energy") : kNaN;
  else throw std::invalid_argument("measurement: the registry derives '" + name + "' but compute has no formula for it");
  return finite_or_nan(v);
}

Ingredients::Ingredients(std::int64_t rows_, std::int64_t members_, std::int64_t components_, bool hold) : rows(rows_), members(members_), components(components_) {
  if (rows_ < 0 || members_ < 0 || components_ < 1) throw std::invalid_argument("measurement: ingredients of rows >= 0, members >= 0 and components >= 1");
  const std::size_t n = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(members_);
  count.assign(n, 0); nonfinite.assign(n, 0);
  weight.assign(n, 0.0); sum_sq.assign(n, 0.0); sum_cube.assign(n, 0.0); sum_fourth.assign(n, 0.0); sum_abs.assign(n, 0.0); sum_sqrt_abs.assign(n, 0.0);
  max.assign(n, -kInf); min.assign(n, kInf); abs_max.assign(n, 0.0);
  sum.assign(n * static_cast<std::size_t>(components_), 0.0);
  if (hold) held.assign(n, {});
}

const std::vector<std::string>& field_measures() { static const std::vector<std::string> m = {"mean", "max", "min", "mean_square", "abs_max"}; return m; }
const std::vector<std::string>& member_measures() { static const std::vector<std::string> m = {"area", "n_elements"}; return m; }
bool known_measure(const std::string& name, const Registry& registry) { return registry.find(name) != nullptr; }

double value(const Ingredients& g, std::int64_t row, std::int64_t member, const std::string& measure, const Registry& registry, std::int64_t component) {
  const MeasureSpec& s = registry.at(measure);
  if (!s.derived_from.empty()) throw std::invalid_argument("measurement: '" + measure + "' is derived: read it with Registry::derive from its ingredients' measures");
  if (row < 0 || row >= g.rows || member < 0 || member >= g.members) throw std::invalid_argument("measurement: value of (row " + std::to_string(row) + ", member " + std::to_string(member) + ") is outside the ingredients");
  if (component < 0 || component >= g.components) throw std::invalid_argument("measurement: component " + std::to_string(component) + " of " + std::to_string(g.components));
  const std::size_t k = g.at(row, member);
  const double w = g.weight[k];
  const std::string& in = s.ingredient;
  double v;
  if (in == "count") v = static_cast<double>(g.count[k]);
  else if (in == "weight") v = w;
  else if (in == "max") v = g.max[k];
  else if (in == "min") v = g.min[k];
  else if (in == "abs_max") v = g.abs_max[k];
  else if (in == "held") {
    if (g.held.empty()) throw std::invalid_argument("measurement: a median needs the values held (Ingredients(..., hold = true), Request::measures names it)");
    std::vector<double> h = g.held[k];                                     // numpy median of the finite values: the middle one, the mean of the two middle ones when even
    if (h.empty()) return kNaN;
    const std::size_t m = h.size() / 2;
    std::nth_element(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(m), h.end());
    const double hi = h[m];
    if (h.size() % 2 == 1) return hi;
    return 0.5 * (*std::max_element(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(m)) + hi);
  }
  else {
    double total;
    if (in == "sum") total = g.sum[g.at(row, member, component)];
    else if (in == "sum_sq") total = g.sum_sq[k];
    else if (in == "sum_cube") total = g.sum_cube[k];
    else if (in == "sum_fourth") total = g.sum_fourth[k];
    else if (in == "sum_abs") total = g.sum_abs[k];
    else if (in == "sum_sqrt_abs") total = g.sum_sqrt_abs[k];
    else throw std::invalid_argument("measurement: '" + measure + "' is a member's own measure (" + in + "): it is computed from the member, not accumulated");
    v = s.merge == Merge::mean ? (w > 0 ? total / w : 0.0) : total;       // a mean of a member of zero weight is the reference's 0
  }
  return finite_or_nan(v);
}

}  // namespace nxr::measurement
