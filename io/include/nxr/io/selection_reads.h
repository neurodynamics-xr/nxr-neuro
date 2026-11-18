// nxr-io — READS of the selection module as stored rows (docs/specs/2026-11-17-selection-module.md). Pipelines and the command line read selections, their members, their lookups
// and their measurements through these functions and carry no SQL of their own: only io touches the database. Each returns the stored row(s) with every column, as `Database::one`
// would, so a caller reads the columns it needs (`text(row, "id")`); the typed readers of selection_store.h are the other face of the same rows.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"

namespace nxr::io::selection {

// ── selections ───────────────────────────────────────────────────────────────────────────────────────────────────
/** the complete selection of that name in the subject, optionally of a stored type and on a manifold */
std::optional<Row> named_row(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& stored_type = std::nullopt,
                             const std::optional<std::string>& manifold_id = std::nullopt);
/** the complete top-level `ladder` of that name in the subject (a ladder is never a member), optionally on a manifold */
std::optional<Row> ladder_row(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& manifold_id = std::nullopt);
/** the member of the ladder or set `parent_id` of that name (complete) */
std::optional<Row> member_row(const Database& db, const std::string& parent_id, const std::string& name);
/** the members of `parent_id` by member ordinal (the complete ones, or all) */
std::vector<Row> member_rows(const Database& db, const std::string& parent_id, bool complete_only = true);
/** the complete selection of that name over that field */
std::optional<Row> row_over_field(const Database& db, const std::string& subject_id, const std::string& of_field_id, const std::string& name);
/** the complete selections over that field */
std::vector<Row> rows_over_field(const Database& db, const std::string& subject_id, const std::string& of_field_id);
/** the complete array-backed `set` of that name, optionally on a manifold (the lowest path when several): a labelling of channels or vertices */
std::optional<Row> labelling_row(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& manifold_id = std::nullopt);
/** the complete sets of a subject whose parameters hold the key (a joint holds `joint`, a cell labelling `cells_of`), by name */
std::vector<Row> set_rows_with_param(const Database& db, const std::string& subject_id, const std::string& param_key);
/** the complete selections of a subject that a recording's atlas tree made: `tree` ("group") in params `atlas.tree`, with a joint (params `joint` present) or without */
std::vector<Row> joint_rows_on_tree(const Database& db, const std::string& subject_id, const std::string& tree);
/** the complete level partition of a tree (params `atlas.tree` = tree, no `joint`) whose name ends ` group tiles L<level>` */
std::optional<Row> tree_level_row(const Database& db, const std::string& subject_id, const std::string& tree, int level);
/** the default atlas ladder's row of a subject (params `atlas.default` = 1), or nothing */
std::optional<Row> default_atlas_ladder_row(const Database& db, const std::string& subject_id);
/** the span bounds [start, stop) of a spans selection, in seconds, by ordinal */
std::vector<std::pair<double, double>> span_bounds(const Database& db, const std::string& selection_id);

// ── lookups (a selection's dictionary) ───────────────────────────────────────────────────────────────────────────
/** code -> name of a dictionary's entries */
std::map<std::int64_t, std::string> lookup_names(const Database& db, const std::string& dictionary_id);
/** the code of the first entry (by name) among `names`, or nothing */
std::optional<std::int64_t> lookup_code_of_any(const Database& db, const std::string& dictionary_id, const std::vector<std::string>& names);
/** the entry of that name in the dictionary of that name (any scope), as its stored row (params_json ...), or nothing */
std::optional<Row> lookup_entry_named(const Database& db, const std::string& dictionary_name, const std::string& entry_name);

// ── measurements ─────────────────────────────────────────────────────────────────────────────────────────────────
/** where an array-backed measurement keeps its values: the array and the component holding them */
struct MeasurementArray { std::string array_path; std::int64_t component = 0; };
/** the array-backed row of `measure` of the selection over the field */
std::optional<MeasurementArray> measurement_array(const Database& db, const std::string& selection_id, const std::string& of_field_id, const std::string& measure);
/** the array of `measure` over the field in any selection (the first row with an array) */
std::optional<MeasurementArray> measurement_array_over_field(const Database& db, const std::string& of_field_id, const std::string& measure);
/** the (measure, array) pairs of the selection's array-backed rows (a measure's last array when it has several) */
std::map<std::string, std::string> measurement_arrays(const Database& db, const std::string& selection_id);
/** the measures of the selection over the field, by name, leaving out `except` */
std::vector<std::string> measures_over_field(const Database& db, const std::string& selection_id, const std::string& of_field_id, const std::optional<std::string>& except = std::nullopt);
/** the distinct `computed_by` of the selection's rows over the field, optionally of one measure */
std::vector<std::string> computed_by_over_field(const Database& db, const std::string& selection_id, const std::string& of_field_id, const std::optional<std::string>& measure = std::nullopt);
/** the selection a selection's rows are measured within (their `within_selection_id`), or nothing */
std::optional<std::string> within_selection_of(const Database& db, const std::string& selection_id);
/** the selection's array-backed rows with the kind and unit of the field they are of (columns `fkind`, `funit` beside the row's own) */
std::vector<Row> array_rows_with_field(const Database& db, const std::string& selection_id);
/** what a measurement row is of: a field, a map slice of a statistic (`statistic_id`, `index` when the slice is named), or none (the members' own values) */
struct MeasuredOf {
  std::optional<std::string> field_id, statistic_id;
  std::optional<std::int64_t> index;
};
/** the distinct (computed_by, unit) of the rows of `measure` on the selection of that kind */
std::vector<Row> computed_by_and_unit(const Database& db, const std::string& selection_id, const std::string& measure, const MeasuredOf& of);

}  // namespace nxr::io::selection
