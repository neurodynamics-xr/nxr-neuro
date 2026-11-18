// nxr-io — READS of a subject's other rows (fields, operators, manifolds with their geometry, subjects, the dataset) as stored rows. Pipelines and the command line ask
// these functions and carry no SQL of their own: only io touches the database. Each returns the stored row(s) with every column, as `Database::one` would. Selections and
// their measurements are in selection_reads.h; a row by its id is `Database::read(table, id)`.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"

namespace nxr::io {

// ── fields ───────────────────────────────────────────────────────────────────────────────────────────────────────
/** which field: the name, and what else is known of it (a kind, its manifold, whether it holds an array: `stored` true = has a path, false = rows only, none = either) */
struct FieldQuery {
  std::string name;
  std::optional<std::string> kind, manifold_id;
  std::optional<bool> stored;
  bool complete_only = true;
};
std::optional<Row> find_field(const Database& db, const std::string& subject_id, const FieldQuery& q);
/** the complete fields of a subject on a manifold, in the order they were stored */
std::vector<Row> fields_on_manifold(const Database& db, const std::string& subject_id, const std::string& manifold_id);
/** the rows-only (byte-less) fields of a subject that a group atlas made (`producer_json.atlas` is `spatial` or `time`) */
std::vector<Row> group_atlas_fields(const Database& db, const std::string& subject_id);

// ── operators ────────────────────────────────────────────────────────────────────────────────────────────────────
/** the complete operator of that kind and name */
std::optional<Row> find_operator(const Database& db, const std::string& subject_id, const std::string& kind, const std::string& name);
/** the complete operators of that kind that hold an array */
std::vector<Row> stored_operators(const Database& db, const std::string& subject_id, const std::string& kind);

// ── manifolds and their geometry ─────────────────────────────────────────────────────────────────────────────────
/** the complete manifold of that name, optionally of a type */
std::optional<Row> find_manifold(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& type = std::nullopt);
/** a manifold with its geometry's `positions_path` and `cs_id` (the geometry's coordinate system) beside its own columns */
std::optional<Row> manifold_with_geometry(const Database& db, const std::string& manifold_id);
/** the complete manifold of that name with the same two geometry columns */
std::optional<Row> manifold_with_geometry_named(const Database& db, const std::string& subject_id, const std::string& name);
/** the complete surface of that name, or (no name) the subject's primary manifold, with `positions_path` */
std::optional<Row> surface_with_geometry(const Database& db, const std::string& subject_id, const std::optional<std::string>& name);
/** the path of a topology's faces (its rank-2 cells) */
std::optional<std::string> topology_faces_path(const Database& db, const std::string& topology_id);
/** the length unit of a manifold's coordinate system ("" when it records none) */
std::string manifold_coordinate_unit(const Database& db, const std::string& manifold_id);
/** the coordinate system a manifold's geometry is in ("" when it records none) */
std::string manifold_coordinate_system(const Database& db, const std::string& manifold_id);
/** the NAME of the coordinate system a manifold's geometry is in (the key a stored montage or kernel records), "" when it records none */
std::string manifold_coordinate_system_name(const Database& db, const std::string& manifold_id);
/** the name of a geometry's coordinate system */
std::optional<std::string> coordinate_system_name_of_geometry(const Database& db, const std::string& geometry_id);
/** the manifolds a product manifold's geometry is made of, by factor */
std::vector<std::string> factor_manifolds(const Database& db, const std::string& manifold_id);
/** a closed-form line's first factor as stored: its spacing, origin (seconds) and sample count */
struct LineFactor { double spacing = 0; std::optional<double> origin; std::int64_t count = 0; };
std::optional<LineFactor> line_factor(const Database& db, const std::string& manifold_id);
/** an acquisition's label ("" when none) */
std::string acquisition_label_of(const Database& db, const std::string& acquisition_id);

// ── subjects and the dataset ─────────────────────────────────────────────────────────────────────────────────────
/** the complete subjects of kind `subject`, by name (columns id, name) */
std::vector<Row> complete_subjects(const Database& db);
/** the complete subject of that name, optionally in a dataset and of a kind (columns: all) */
std::optional<Row> find_subject(const Database& db, const std::string& name, const std::optional<std::string>& dataset_id = std::nullopt, const std::optional<std::string>& kind = std::nullopt);
/** the dataset's row (id, name) of a one-dataset database */
std::optional<Row> dataset_row(const Database& db);

// ── cached results ───────────────────────────────────────────────────────────────────────────────────────────────
/** the column that holds a node's producer record: a selection's is `params_json` (its producer is in `params_json.search`), the other nodes' is `producer_json` */
std::string producer_column(const std::string& table);
/** the complete rows of a node table (manifold, selection, field, operator, statistic) of the subject that hold a producer record, optionally at a path; nothing for a table the database lacks */
std::vector<Row> producer_rows(const Database& db, const std::string& subject_id, const std::string& table, const std::optional<std::string>& path);

}  // namespace nxr::io
