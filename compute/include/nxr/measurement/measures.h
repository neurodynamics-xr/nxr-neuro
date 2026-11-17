// nxr::measurement (compute) — THE MEASURES AND THEIR INGREDIENTS (spec docs/specs/2026-11-17-consolidation.md, section 2.1): the ONE definition of what a measure over a selection's
// member is, how it is read from the additive INGREDIENTS the reduction accumulates, and how it MERGES when members are rolled up. The reference is the Cortical Flow Studio's
// `measureOverPartition` (backend/src/db/measurements.ts): one pass accumulates, per member, the weight Σw, the weighted sums Σw·x, Σw·x², the count, max, min and max |x|; the
// measures are read from those (partition.cpp's accumulation, line for line, moved here).
//
// THE MERGE RULE IS DATA (PI, 2026-11-17, rule 35: roll-ups live in io's QUERY and in compute, one definition): each measure's merge rule (sum, max, min, mean as the sum of
// weighted values over the sum of weights, none: never rolled up) is a row of the measures registry (io/schema/measures.json: `reduction`, `weight`, `derived_from`; plus `form`,
// the form of the field the measure is taken of). io applies the rule in SQL, compute in memory (`roll_up.h`), and a parity test compares the two. Ratios (amplitude = sum / weight,
// sd from mean and mean_square, speed, travelling index) are DERIVED at read from the stored ingredients, never reduced themselves.
//
// FORMS (the Studio's field forms, complex-extended; applied pointwise by the node `field.form` before a reduction):
//   none            the field as it is        magnitude  |z|        power  |z|^2        phase_current  the vector Im(conj(z) grad z) in the canonical frame (components)
//   outer           z z^H (a matrix per sample)           edge_1form / face_2form   a 1-form on edges, a 2-form on faces (outflow, circulation)
//   member          the member itself (area, element count, duration, centre ...: no field)
//
// Pure functions of arrays and tables: no file, no row. std::invalid_argument names the measure and the valid ones.
#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace nxr::measurement {

/** how a measure combines when members merge */
enum class Merge { sum, max, min, mean, none };
const char* merge_name(Merge m);
/** "sum" | "max" | "min" | "mean" | "none" (an empty name or "null": none); std::invalid_argument for another */
Merge merge_from(const std::string& name);

/** one row of the registry */
struct MeasureSpec {
  std::string name;
  std::string form = "none";                 // the form of the field it is taken of
  std::string ingredient;                    // the ingredient it reads: count, weight, sum, sum_sq, sum_cube, sum_fourth, sum_abs, sum_sqrt_abs, max, min, abs_max, held, member
  Merge merge = Merge::sum;
  std::string weight;                        // merge mean: the measure whose values weight the mean when it is rolled up (area, n_elements, weight, count)
  std::vector<std::string> derived_from;     // non-empty: derived at read from these measures (merge none: it is never stored or rolled up)
};

/** the registry: the table io reads from io/schema/measures.json and hands to compute as data, or the built-in standard table (the same measures) */
class Registry {
 public:
  Registry() = default;
  explicit Registry(std::vector<MeasureSpec> specs);
  /** the standard table: the Studio's measures (measures.json) and nxr-neuro's own (measures_nxr.json) */
  static const Registry& standard();
  const MeasureSpec* find(const std::string& name) const;
  /** std::invalid_argument naming the valid measures for an unknown name */
  const MeasureSpec& at(const std::string& name) const;
  const std::vector<MeasureSpec>& all() const { return specs_; }
  bool derived(const std::string& name) const;
  /** the value of a DERIVED measure from `stored(name)` of its ingredients' measures (NaN when one is not finite); std::invalid_argument for a measure that is not derived, or one
   *  whose formula compute does not have. Formulas: variance, sd, rms, peak_to_peak, crest_factor, envelope_variance, shape_factor, impulse_factor, clearance_factor, skewness,
   *  kurtosis, phase_gradient_{1,2,x,y,z,magnitude,direction_deg}, travelling_index, phase_coherence, quadrature_fraction, tile_pair_quadrature_fraction, crossing_consistency, crossing_rho2. */
  double derive(const std::string& name, const std::function<double(const std::string&)>& stored) const;

 private:
  std::vector<MeasureSpec> specs_;
};

/** THE ADDITIVE INGREDIENTS per row and member, as the Studio's measureOverPartition accumulates them. Rows are what stays unreduced (frames, channels, a band); members are the
 *  selection's. Everything adds (sums, counts) or merges by max / min, so a member read block by block is the blocks' ingredients combined. A vector measure keeps `sum` per
 *  component; the other ingredients are those of component 0. */
struct Ingredients {
  std::int64_t rows = 0, members = 0, components = 1;
  std::vector<std::int64_t> count, nonfinite;                  // [rows x members]: values added, values not finite
  std::vector<double> weight, sum_sq, sum_cube, sum_fourth, sum_abs, sum_sqrt_abs, max, min, abs_max;   // [rows x members]
  std::vector<double> sum;                                     // [rows x members x components]: Σw·x
  std::vector<std::vector<double>> held;                       // [rows x members] the finite values, only when a median is wanted (exact, unweighted, never rolled up)
  std::int64_t unassigned = 0;                                 // elements in no member (a negative member number), counted once
  Ingredients() = default;
  /** zeroed: weights and sums 0, max -inf, min +inf (the identities of the merges); `hold` keeps the values for a median */
  Ingredients(std::int64_t rows_, std::int64_t members_, std::int64_t components_ = 1, bool hold = false);
  std::size_t at(std::int64_t row, std::int64_t member) const { return static_cast<std::size_t>(row * members + member); }
  std::size_t at(std::int64_t row, std::int64_t member, std::int64_t component) const { return static_cast<std::size_t>((row * members + member) * components + component); }
};

/** the value of `measure` for (row, member), as the reference reads it; NaN when it is not finite (an empty member for max / min; a non-finite value). A member of no weight has the
 *  reference's 0 for a mean. `component` for a vector measure. Derived measures need `Registry::derive` (they read other measures). */
double value(const Ingredients& ing, std::int64_t row, std::int64_t member, const std::string& measure, const Registry& registry = Registry::standard(), std::int64_t component = 0);
/** the measures this producer writes by default, in the reference's order: with a field mean, max, min, mean_square, abs_max; without, area and n_elements */
const std::vector<std::string>& field_measures();
const std::vector<std::string>& member_measures();
bool known_measure(const std::string& name, const Registry& registry = Registry::standard());

}  // namespace nxr::measurement
