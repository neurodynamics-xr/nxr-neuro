#pragma once
// nxr-io — THE STATISTIC NODE (docs/specs/2026-11-13-statistics.md): what is computed FROM a partition's members over a span of a field (field ->
// selection/measurement -> statistic): a Welch PSD (the mean over the segments), a cross-spectrum, a distribution, a percentile. Its
// row (table `statistic`, model.sql) says what the values are (`measure`), how they were estimated (`method`), of what (`of_field_id` · `of_statistic_id`),
// from which partition (`partition_selection_id`) and how many of its members (`n_members`), over which span (`span_json`), along which axes
// (`axes_json`); the values are an array at its path. Brainstorm's timefreq file is the model. Written by SubjectWriter::statistic.
//
// WITHOUT MIGRATION: a dataset made before the table has none; the first statistic written adds the table, its triggers and view and widens sync_job's
// table list IN PLACE (ensure_statistics, the block of model.sql between its markers; user_version unchanged, recorded in meta). A reader treats a
// dataset without the table as one with no statistics.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"

namespace nxr::io {

class SubjectWriter;

/** whether the database has a table of that name */
bool has_table(const Database& db, const std::string& name);
bool has_column(const Database& db, const std::string& table, const std::string& column);
/** the statistic table, its triggers and view, and the measurements of statistics (below), added in place to a database made before them; true when it added any */
bool ensure_statistics(Database& db);
/** MEASUREMENTS OF A STATISTIC: selection_measurement's `of_statistic_id` and `of_index` (a slice along the statistic's other axis: a band), its unique key and
 *  the selection's zarr.json view widened, in place, once; true when it added them */
bool ensure_statistic_measurements(Database& db);
/** the statistics registry (schema/statistics.json): {measures: [{name, unit_rule, description}], methods: [{name, description}]} */
const Json& statistic_registry();
bool statistic_measure_known(const std::string& name);
bool statistic_method_known(const std::string& name);

/** a stored statistic: its row, its record parsed, and its values as float64 in C order */
struct StoredStatistic {
  Row row;
  Json axes, components, span, options, producer, scaling;
  std::vector<std::int64_t> shape;
  std::vector<double> values;
};
/** the subject's complete statistics (optionally of one field, of one measure), by name; none when the dataset has no table */
std::vector<Row> statistics_of(const Database& db, const std::string& subject_id, const std::optional<std::string>& of_field_id = std::nullopt,
                               const std::optional<std::string>& measure = std::nullopt);
/** a statistic of the subject by name (an Error no_such_row naming the subject's statistics when there is none) */
// `with_values` false: the record, the axes and the shape only (no value read: a caller that reads a region of the array itself)
StoredStatistic read_statistic(SubjectWriter& sub, const std::string& name, bool with_values = true);

/** A MAP OF A STATISTIC (2026-11-13, the maintainers: map statistics are read as maps): a statistic whose first axis lies on a manifold (the vertices of a surface,
 *  the modes) and whose second lists named entries (bands: {name, …}) is a stack of maps; slice k is entry k's map, named `<statistic>_<entry name>` (as
 *  the Field of that band was named), `of_index` k in a measurement of it */
struct MapSlice {
  Row statistic;
  std::int64_t index = 0, n = 0;               // the slice along the second axis; the first axis's size
  std::string name, manifold_id;
  std::optional<std::string> unit;
  std::string value_type = "scalar";           // each value, as a field's (scalar · complex · vector3 · quaternion · hermitian3 …)
  std::int64_t n_components = 1;
};
/** the map slices of the subject's statistics on a manifold, by name */
std::vector<MapSlice> map_slices(const Database& db, const std::string& subject_id, const std::string& manifold_id);
/** a slice's values [n x n_components] (float64), each value's components in the field's order (hermitian3: 3 x 3 x re, im) */
std::vector<double> read_map_slice(SubjectWriter& sub, const MapSlice& slice);

}  // namespace nxr::io
