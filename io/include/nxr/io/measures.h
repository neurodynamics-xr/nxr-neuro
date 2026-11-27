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

// The registry (schema/measures.json): {dictionary: {name, scope, description}, measures: [{code, name, merge, form, unit, weight, span, derived_from, description}]}
// `merge` is the ONE rule key (sum, max, min, mean, none, derived: how the value rolls up, shared by io's SQL and compute's roll_up); `form` the form of the field it is taken of (none, magnitude, power, phase_current, outer, hermitian3, gradient, edge_1form, face_2form, frames, member).
const Json& measure_registry();
// One measure's entry, or nullopt for a name the registry does not know (a subject may add its own).
std::optional<Json> measure_spec(const std::string& name);
bool measure_is_derived(const std::string& name);
// How a measure's values combine: its merge rule (below), `mean` for a derived one (registered names; Error(unsupported) for an unknown one).
std::string measure_reduction(const std::string& name);

// THE MERGE RULE IS DATA (schema/measures.json and measures_nxr.json, the key `merge`): the ONE definition of how a measure's values
// combine when members are rolled up (through a grid's hierarchy, or across subjects). io applies it in SQL (measurement_query.h), compute in memory (nxr::measurement), with a parity test
// between the two; the table below is what io hands compute (pipelines carry it across, as io and compute never meet).
//   sum      the members' values add                             max, min   the extreme of the members'
//   mean     the weighted mean: sum(value * weight) / sum(weight), the weight the member's own (its area, its sample count): a mean through sums and a count, never a mean of means
//   none     never merged (a median, a member's own centre, a duration of one event): a roll-up that would group members is refused
//   derived  not stored: computed at read from its ingredients, themselves merged first (a ratio of sums, never a sum of ratios)
enum class MergeRule { sum, max, min, mean, none, derived };
const char* merge_rule_name(MergeRule r);
// A REGISTERED measure's merge rule. A name the registry does not know has no rule here: Error(unsupported), naming the declaration it needs (never a guessed `sum`).
MergeRule measure_merge(const std::string& name);

// DECLARED MEASURES (docs/reference.md, "Declared measures"). The registry is a vocabulary of known names and their defaults, not a gate: any measure may be stored, and
// the writer of its measurements DECLARES it (its merge rule, and its form, unit, weight measure and description where they apply). The declaration is kept in the subject's
// `measure_declaration` table, one row a name, so one name keeps one meaning in a subject: a second declaration that differs is refused, and so is one that contradicts the
// registry's entry for a known name (a declaration may leave out what the registry gives). `derived` is declared only for a registered name (its formula is io's).
struct MeasureDeclaration {
  std::string name;
  std::optional<std::string> merge;            // sum · max · min · mean · none · derived; required for a name the registry does not know
  std::optional<std::string> form, unit, weight, description;
};
// {name, merge, form?, unit?, weight?, description?} (the form a request and a producer record carry); Error(invalid_argument) for another shape or an unknown key
MeasureDeclaration measure_declaration_of(const Json& j);
Json measure_declaration_json(const MeasureDeclaration& d);
// the table, added in place to a database made before it (additive; user_version unchanged); true when it was added
bool ensure_measure_declarations(Database& db);
// Check a declaration against the registry and the subject's earlier one, and store it (in the caller's transaction, or its own); returns the declaration as stored (the
// registry's values filled in for a known name). Error(invalid_argument) for a contradiction, naming both values.
Json declare_measure(Database& db, const MeasureDeclaration& d, const std::optional<std::string>& by);
// the subject's declaration of a name, or nullopt
std::optional<Json> declared_measure(const Database& db, const std::string& name);
// THE RULE A ROLL-UP USES: the subject's declaration, else the registry's; a measure neither declared nor registered is Error(unsupported) (it is never summed by default)
MergeRule measure_merge(const Database& db, const std::string& name);
// the rule a READ at the stored level may use: as above, but a measure neither declared nor registered reads as `none` (each member its own value), `known` false;
// a read that would merge its members is then refused with `undeclared_refusal`
struct ReadRule { MergeRule rule; bool known; };
ReadRule measure_read_rule(const Database& db, const std::string& name);
std::string undeclared_refusal(const std::string& name);

// THE PRODUCER OF A MEASUREMENT (`selection_measurement.producer_json`): the column added in place to a database made before it (additive; user_version unchanged); true when added.
// write_measurements stores a row's `producer_json` (an object) when the row carries one.
bool ensure_measurement_producer(Database& db);
// THE MANIFOLD A MEASUREMENT'S ROWS LIE ON (`selection_measurement.rows_manifold_id`, issue #264): the column added in place and the store's key made again with it (additive; user_version
// unchanged); true when added. Two measurements of one field over one selection with the same measure on two manifolds (the two hemispheres) are then two rows of the key.
bool ensure_measurement_rows_manifold(Database& db);
// What a measurement of a selection IS and how it was made, for a read: {declaration (the subject's, else the registry's entry, else null), declared (true when the
// subject declares it), producers: [the distinct producer records of its rows]} (`read selection --measure` gives them beside the values)
Json measurement_provenance(const Database& db, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::string& measure,
                           const std::optional<std::string>& of_statistic_id = std::nullopt);   // of a statistic: only its rows' producers
// {"<measure>": "sum" | "max" | "min" | "mean" | "none" | "derived", ...} for every registered measure: the table compute reads.
Json measure_merge_table();

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
  Json producers = Json::array();           // the producer records (producer_json) of the rows read, and of no other measurement of the measure (#290)
};
// `rows_manifold_id`: only the array whose rows lie on that manifold (issue #264); without it, arrays of the measure on several manifolds are refused, naming them
// (the unnamed manifold of an array stored without --rows-on counts as one of them, #290).
MeasurementValues measurement_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id,
                                     const std::optional<std::string>& of_statistic_id, const std::optional<std::int64_t>& of_index, const std::string& measure,
                                     const std::optional<std::string>& rows_manifold_id = std::nullopt);

// A DERIVED measure of ARRAY-BACKED or scalar-row ingredients, member by member as stored (no roll-up): each ingredient read by `measurement_values` (of the same Field or statistic), then the measure's formula applied
// elementwise (a ratio of the stored sums, derived at read, never stored); NaN where an ingredient is NaN or the formula is undefined. Refuses a measure that is not derived.
// A MEASURE PER MEMBER of a selection, stored or derived (a ratio of stored sums, at read), whatever it was measured of: the field or statistic its rows (or its first ingredient's) are of,
// found among the selection's measurements (refused when they are of several). [rows x members], as `measurement_values` gives it.
MeasurementValues member_measure_values(SubjectWriter& sub, const std::string& selection_id, const std::string& measure);
MeasurementValues derived_measurement_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id,
                                             const std::optional<std::string>& of_statistic_id, const std::optional<std::int64_t>& of_index, const std::string& measure);

}  // namespace nxr::io
