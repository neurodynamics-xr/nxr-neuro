// nxr-io — WHAT IS MEASURED ON A SELECTION (the selection module's measurement part; docs/specs/2026-11-17-selection-module.md): the `selection_measurement` rows (a
// measure's value per member of a partition, or a selection's own), and the arrays an array-backed measurement keeps. One place for the writes, reads and deletes; the
// writer's `measurement` / `set_own_measurements`, the task API's `MeasurementOut` and the importers' region values call these.
//
// The registry-level machinery (the measure registry, derived measures, the pyramid reads, `measurement_values`) stays in measures.h, rollup.h and statistic.h:
// this file is the row-level CRUD and forwards the batch calls to them, so a caller finds everything about a selection's measurements here.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/records/rows.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/writer.h"

namespace nxr::io::selection {

/** a measurement row of a selection of this subject (`row`: measure, code?, of_field_id?, value?, array_path?, unit?, computed_by?, within_selection_id?, within_code?, ...);
 *  inside the caller's composition */
void save_measurement(SubjectWriter& sub, const std::string& selection_id, const Cols& row);
/** a measurement row with its array: `values` are written at `array_path` (laid out and populated) and the row names it */
void save_measurement_array(SubjectWriter& sub, const std::string& selection_id, const std::string& array_path, const ArrayView& values, const Cols& row);
/** one value per member, as rows (no array): `code` the member's, `of_field_id` none when it is the members' own (a size) */
struct MemberValue { std::int64_t code = 0; double value = 0; std::optional<std::int64_t> within_code; };
std::int64_t save_member_values(SubjectWriter& sub, const std::string& selection_id, const std::string& measure, const std::vector<MemberValue>& values,
                                const std::optional<std::string>& unit, const std::string& computed_by, const std::optional<std::string>& within_selection_id = std::nullopt);
/** the batch write of the measure registry (rows as JSON: {selectionId, code?, ofFieldId?, measure, value?, ...}); refuses a derived measure; returns the count (measures.h) */
std::int64_t save_measurements(Database& db, const Json& rows, const std::optional<std::string>& by = std::nullopt);

/** what was measured on a selection: its own rows (code null), or every row (its members' too) */
std::vector<records::SelectionMeasurementRow> measurements_of(const Database& db, const std::string& selection_id, bool own = true);
/** what a selection's members carry: [{of_field_id, measure, members, min, max}] (measures.h) */
Json measurement_kinds_of(const Database& db, const std::string& selection_id);

/** which measurement rows of a selection a question is about; a member left unset does not restrict */
struct MeasurementFilter {
  std::optional<std::string> rows_manifold_id;      // only the rows on that manifold (issue #264)
  std::optional<std::string> measure;
  std::optional<std::string> of_field_id;
  bool of_field_none = false;                       // only the members' own values (no field: a size, a count of streamlines)
  std::optional<std::string> of_statistic_id;       // rows of a map slice of that statistic ...
  std::optional<std::int64_t> of_index;             // ... and of that slice, when named
  bool whole_only = false;                          // only rows with no `within` partition
};
/** does the selection hold a measurement row of that kind? */
bool has_measurement(const Database& db, const std::string& selection_id, const MeasurementFilter& filter);
/** the selections that hold `measure` of the field, by id */
std::vector<std::string> selections_measured(const Database& db, const std::string& of_field_id, const std::string& measure);

/** the arrays measurements name: of one field, of a subject (as (selection id, path)), of a dataset */
std::vector<std::string> measurement_array_paths_of_field(const Database& db, const std::string& of_field_id);
std::vector<std::pair<std::string, std::string>> measurement_array_paths_of_subject(const Database& db, const std::string& subject_id);
std::vector<SubjectArray> measurement_array_paths_in_dataset(const Query& all);
/** the own rows (code null) of a subject's selections, as stored rows (the subject bundle) */
std::vector<Row> subject_own_measurement_rows(const Database& db, const std::string& subject_id);

/** replace a selection's OWN rows (code null); its members' rows are the measurement path's */
void update_own_measurements(SubjectWriter& sub, const std::string& selection_id, const std::vector<Cols>& rows);
/** DELETE rows: of a selection, optionally of one field and one measure; returns how many (measures.h's delete) */
std::int64_t remove_measurements(Database& db, const std::string& selection_id, const std::optional<std::string>& of_field_id = std::nullopt,
                                 const std::optional<std::string>& measure = std::nullopt, const std::optional<std::string>& of_statistic_id = std::nullopt);

}  // namespace nxr::io::selection
