// nxr-io — THE MEASUREMENT QUERY API: what is measured, asked as a table, with filters and roll-ups, in SQL (roll-ups live in io's query as well as in compute, for group analysis and
// statistical modelling; docs/specs/2026-11-17-consolidation.md sections 2.3 and 12 D1; ARCHITECTURE.md, io's wording extended to say so).
//
//   WHAT IT ANSWERS   one row per member (subject, selection, member number) for a measure, or a coarser row once members are merged: the long table a mixed model or a group test reads.
//   FILTERS           by selection (id, or a NAME, which names the same selection in every subject: a group tree's grid), by member (number, or label through the selection's lookup), by LEVEL IN PHYSICAL UNITS
//                     (a level of a grid whose tile is 2.5 to 5 s, or 8 to 16 mm), by subject (ids, names, or a participants filter: a group), by field and band, by value.
//   ROLL-UPS          through a grid's closed-form hierarchy (member number `code >> shift`, a line's time grid and a dyadic surface grid alike; one factor of a product, mixed radix), to the whole,
//                     and ACROSS SUBJECTS (the same member of every selected subject, code for code). Each measure merges by its registry rule (measures.h `MergeRule`, DATA in schema/measures.json,
//                     the one definition compute shares): sum, max, min, a mean through its sums and weights, never a median; a derived measure is computed from its merged ingredients (ratios of
//                     sums, never sums of ratios).
//   WHERE             SQL over `selection_measurement` (and `selection`, `dictionary_entry`, `subject`, `field`): scalar member rows. A measurement stored as an ARRAY (rows x members x components) is read
//                     through the store and merged by the same rule in memory (rollup.h): `rolled_values`, `rolled_values_across`.
//   NOT HERE          the maths of a measure (compute's `reduce`, `roll_up`, `summarize`) and any file layout beyond the arrays a measurement row names.
// The roll-up arithmetic exists twice by decision (SQL here, memory in compute): both read the merge rule from the registry, and a parity test compares them on the same members.
#pragma once
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/measures.h"
#include "nxr/io/store.h"

namespace nxr::io {
class SubjectWriter;
}

namespace nxr::io::query {

/** a physical size, inclusive at both ends: the size of a grid's level as its level dictionary records it (`unit` "s" for a time grid's tile, the surface's unit, "mm", for a dyadic tile's diameter) */
struct LevelRange {
  std::string unit;
  std::optional<double> min, max;
  std::optional<std::size_t> factor;        // for a PRODUCT selection: which factor's level (its `selection_factor.level` against that grid's level sizes); a level row or a tiling needs none
};

struct Filter {
  std::vector<std::string> selections;      // selection ids or names (a name = that selection in every subject); empty: every selection that carries the measure
  std::optional<std::string> of_field;      // a field id or name: what was measured
  std::vector<std::int64_t> members;        // member NUMBERS (0 .. n - 1: the index of the member in its selection)
  std::vector<std::int64_t> labels;         // member LABELS (a set's dictionary codes), turned into member numbers by the selection's labels array (needs the Store of the subject)
  std::optional<LevelRange> level;          // only selections at a level of a grid whose size is in range
  std::vector<std::string> subjects;        // subject ids or names; empty: every subject in the database
  std::optional<std::string> within_selection;   // the band a row is within (a bank of bands; with `within_code`)
  std::optional<std::int64_t> within_code;
  std::optional<double> min_value, max_value;    // on the (rolled-up) value
};

struct RollUp {
  int shift = 0;                            // coarser by `shift` levels: member number `code >> shift` (a dyadic or time grid, closed form); 0: as stored
  std::optional<std::size_t> factor;        // a PRODUCT selection: roll this factor by `shift` (mixed radix, first factor fastest) and keep the others
  bool whole = false;                       // merge every member into one (level -1)
  bool across_subjects = false;             // merge the same member of every selected subject, code for code (the group tree's codes)
  bool merges() const { return shift > 0 || whole || across_subjects; }
};

struct Query {
  std::string measure;                      // a stored measure, or a derived one (computed from its ingredients merged first)
  Filter filter;
  RollUp roll;
  std::string order;                        // "asc" | "desc" by value; empty: by subject, selection, member number
  std::int64_t limit = 0;                   // 0: all
};

/** The rows: [{subject_id, selection_id, code, value, weight, members}] (subject_id and selection_id are null when the rows are merged across subjects; `members` = how many stored members merged into the row;
 *  `weight` the sum of the members' weights). Error(unsupported) naming the measure when it is `none` and members would be merged; Error(invalid_argument) for an unknown subject or selection. Scalar rows only:
 *  a selection whose measurement is an array is read by `rolled_values`. `labels` in the filter need `store` (the Store of the subject the selections are in); everything else is SQL. */
Json members(const Database& db, const Query& q, const Store* store = nullptr);

/** the stored SQL of the query, for reading and tests: the statement for the measure's rows with the parameters in order (a derived measure: its first ingredient) */
struct Statement { std::string sql; std::vector<Value> params; };
Statement statement(const Database& db, const Query& q, const std::vector<std::string>& selection_ids);

/** the selection ids a filter reaches for a measure: the selections that carry it, narrowed by name, subject, field, band and LEVEL (by physical size). Ordered by subject, then name. */
std::vector<std::string> selections(const Database& db, const std::string& measure, const Filter& filter);

/** the physical size of the level a selection lies at (the grid's level dictionary: `measurement`, `measurement_unit`), or nothing: a level row of a grid, a tiling (its width, its unit), a product with a
 *  `factor` (that factor's level). */
struct LevelSize { double size = 0; std::string unit; };
std::optional<LevelSize> level_size(const Database& db, const std::string& selection_id, std::optional<std::size_t> factor = std::nullopt);

/** the subject ids a participants filter selects (the filter language of subject_filter.h over the dataset's participants table): a GROUP, for the `subjects` of a filter. Error(invalid_argument) as the language says. */
std::vector<std::string> subjects_where(const std::filesystem::path& dataset_folder, const std::string& filter_text);

/** the dataset's cross-subject index opened read-only, built first when it is not current (dataset_index.h): the database the group queries above run on */
Database open_index(const std::filesystem::path& dataset_folder);

// ── measurements stored as ARRAYS (rows x members x components) ────────────────────────────────────────────────

/** a stored measurement's values [rows x members] merged as `roll` says by the measure's registry rule (rollup.h): shift, one factor of a product, the whole. A mean rolls through the stored `weight` of the same
 *  field (a mean of means is refused); a derived measure from its merged ingredients, elementwise. `across_subjects` is ignored here: see `rolled_values_across`. Throws Error. */
MeasurementValues rolled_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& of_statistic_id,
                                const std::optional<std::int64_t>& of_index, const std::string& measure, const RollUp& roll);

/** the same selection in several subjects (the group tree's grid: equal members), merged code for code by the measure's registry rule, after each is rolled as `roll` says */
struct SubjectArray {
  SubjectWriter* subject = nullptr;
  std::string selection_id;
  std::optional<std::string> of_field_id, of_statistic_id;
  std::optional<std::int64_t> of_index;
};
MeasurementValues rolled_values_across(const std::vector<SubjectArray>& subjects, const std::string& measure, const RollUp& roll);

}  // namespace nxr::io::query
