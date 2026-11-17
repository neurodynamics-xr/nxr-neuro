// nxr-io — MEASUREMENTS on a selection's members (D68): the measure registry and the reads and writes of `selection_measurement`, a port of the
// Cortical Flow Studio's (backend/src/db/measurements.ts: MEASURES, ensureMeasureDictionary; db/database.ts measurements; db/repository.ts
// measurementRepository), the reference. The registry's data is `schema/measures.json`, generated from the Studio's (bindings/wasm-io/tools/gen_measures.mts),
// followed by nxr-neuro's own, `schema/measures_nxr.json` (codes 1000-1999, reserved: never the Studio's); the DERIVED measures' formulas are code here.
//   A measure is STORED (mean, max, …: a reduction to roll it up by, `code >> shift`) or DERIVED (variance, sd, rms, peak_to_peak, crest_factor,
//   envelope_variance; nxr-neuro's phase_gradient_1/2, _magnitude, _direction_deg, travelling_index, phase_coherence): never written, computed at read
//   from its ingredients rolled up first (a ratio of sums, never a sum of ratios).
#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"

namespace nxr::io {

// The registry (schema/measures.json): {dictionary: {name, scope, description}, measures: [{code, name, reduction, unit, weight, span, derived_from, description}]}.
const Json& measure_registry();
// One measure's entry, or nullopt for a name the registry does not know (a subject may add its own).
std::optional<Json> measure_spec(const std::string& name);
bool measure_is_derived(const std::string& name);
// How a measure's values combine: its reduction, `mean` when it names none.
std::string measure_reduction(const std::string& name);

// A derived measure's formula over its ingredients' values by name (NaN where undefined); throws Error(unsupported) for a name with none. The row-level reads in selection_measurement_rows.cpp apply it.
double measure_formula(const std::string& name, const std::map<std::string, double>& ingredients);

// The app-wide `measure` dictionary, seeded if absent and completed if it lacks a registry entry; returns its id (ensureMeasureDictionary).
std::string ensure_measure_dictionary(Database& db, const std::optional<std::string>& by);

// WRITE rows: each {selection_id, code?, of_field_id?, measure, value?, weight?, within_selection_id?, within_code?, unit?, computed_utc?, computed_by?,
// array_path?, component?}. A derived measure is refused; the dictionary is ensured; a selection's OWN rows (code null) stamp its modified_utc. Returns the count.
std::int64_t write_measurements(Database& db, const Json& rows, const std::optional<std::string>& by);
// READ: the roll-up (`code >> shift`, reduced; a derived measure from its ingredients rolled up first), thresholded, ordered: [{code, value, weight, members}].
// q: {selectionId, measure, ofFieldId?, shift?, reduction?, minValue?, maxValue?, limit?, order?, withinSelectionId?, withinCode?}.
Json query_measurements(const Database& db, const Json& q);
// READ-TIME ROLL-UP of ARRAY-BACKED pyramid measurements (D132; nxr `rollup.space/time`, the Studio's `readAtlas`; spec docs/specs/2026-11-14-measurement-pyramid.md, step 3): the same query with the Store the arrays lie in.
// Without `spaceLevel`, `timeLevel` or `band` it is the query above. With any of them, `selectionId` is a pyramid's JOINT Selection (params_json.joint: layout [band, tile, frame]) or a spatial partition
// (params_json.atlas, arrays [leaf]), and the answer is a measurement ON ITS CELLS at a chosen scale, as rows [{code, tile, band, value, weight, members}]:
//   q: {selectionId, measure, ofFieldId? (default: the only field measured), band? (the row's within_code; needed when the joint holds several), spaceLevel? (ℓ ≤ the stored depth D, −1 the whole cortex; default D),
//       timeLevel? (m ≥ 0: tiles of 2^m frames aligned on the tower; default 0), minValue?, maxValue?, limit?, order?}
//   code   the space node at ℓ (leaf code >> (D − ℓ)); tile: the tower code of the time tile ((code0 >> m) + i; null for a spatial partition); band; weight: its area (m²) × its good samples; members: leaves × frames merged.
// A STORED measure (its array-backed row) merges by its reduction (rollup.h `merge_of`: sum, max, min); the space and time roll-ups commute. A DERIVED value is computed from rolled-up sums, never rolled up:
//   <sum measure>_density = Σ / (area × samples) per m² per sample (power_sum_density, tensor_total_density, ...; the area from the partition's leaf areas, the samples the frame tiling's `count` of the field);
//   mean = sum / weight, mean_square = sum_square / weight, variance = mean_square − mean² of a spatial partition's `weight`, `sum`, `sum_square` rows. NaN or non-finite cells (an empty cell, no samples) are left out.
// Throws Error (no_such_row, invalid_argument) naming what is missing.
class Store;
Json query_measurements(const Database& db, const Store& store, const Json& q);

// What a selection's MEMBERS carry: [{of_field_id, measure, members, min, max}], derived ones offered where all their ingredients are (min/max null).
Json measurement_kinds(const Database& db, const std::string& selection_id);
// DELETE rows: where {selectionId, ofFieldId?, measure?}; returns how many.
std::int64_t delete_measurements(Database& db, const Json& where);

// A STORED MEASUREMENT's values as [rows x members] (float64; NaN where a member has none): the input of a statistic summary (spec
// docs/specs/2026-11-13-statistic-summaries.md). Either form the model stores: ARRAY-BACKED (a row with array_path and component: [rows x members x components],
// the envelope's and `measure over time`'s) or SCALAR rows (one value per member: one row of values, `measure compute`'s). What was measured is a Field
// (`of_field_id`) or a band of a map statistic (`of_statistic_id`, `of_index`). A derived measure is not stored: refused (summarize its ingredients).
class SubjectWriter;
struct MeasurementValues {
  std::int64_t rows = 0, members = 0;
  std::vector<double> values;               // [rows x members]
  bool array_backed = false;
  std::optional<std::string> unit;
  std::vector<std::string> computed_by;     // the producers of the rows read (their identity, for a cache key)
};
MeasurementValues measurement_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id,
                                     const std::optional<std::string>& of_statistic_id, const std::optional<std::int64_t>& of_index, const std::string& measure);

// A DERIVED measure of ARRAY-BACKED or scalar-row ingredients, member by member as stored (no roll-up): each ingredient read by `measurement_values` (of the same Field or statistic), then the measure's formula applied
// elementwise (a ratio of the stored sums, derived at read, never stored); NaN where an ingredient is NaN or the formula is undefined. Refuses a measure that is not derived.
MeasurementValues derived_measurement_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id,
                                             const std::optional<std::string>& of_statistic_id, const std::optional<std::int64_t>& of_index, const std::string& measure);

}  // namespace nxr::io
