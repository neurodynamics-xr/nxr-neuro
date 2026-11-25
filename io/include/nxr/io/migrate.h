// nxr-io — MIGRATION between the one-database dataset (store formats 50 to 52: `<dataset>/dataset.sqlite` and the subjects' stores) and subject packages with a
// catalogue (store format 53) — docs/specs/2026-11-16-subject-centred-datasets.md section 8. Both run IN PLACE, under the dataset's writer lock, SUBJECT BY SUBJECT,
// verified before anything is marked done, RESUMABLE (a crash leaves the old dataset whole or a subject finished; a second run changes no byte) and REVERSIBLE; the
// old database is kept, renamed, never deleted. A migration is one explicit command, never silent on first use: a build that knows format 53 still reads 50-52.
//
// SPLIT (`split_dataset`):
//   1. FORMAT FIRST: a dataset below 52 is brought to 52 by `compact_dataset` (the subjects' arrays and small keys are verified byte for byte there).
//   2. THE CATALOGUE is made carrying the dataset's own `dataset` and `datastore` rows (the ids stay).
//   3. THE TEMPLATES FIRST (a template's hash is what the subjects name): each subject of kind `template` becomes a package; the dataset's default subject is
//      referenced as `<dataset>-default`, version `local`, edition `full`, its content hash. It is not released into a shared templates folder: replacing it by a
//      released template with the same anatomy is a separate, explicit step.
//   4. EACH SUBJECT: `subject.sqlite.building` is made in the subject's own folder: the rows it owns (every row carrying its subject_id, the children of its rows,
//      the home dataset and datastore rows, and a copy of every shared vocabulary dictionary, ids unchanged), the ANCHOR rows of the template (and of any other
//      subject) its rows point at, with their parents and the children a bundle follows (listed in `anchor_row`), the subject's small keys copied from its
//      `small_files.sqlite` into the package's own `small_files` table, `key_digest` (every key hashed, shards included), the record and the content hash.
//      VERIFIED: no foreign-key violation; every row is the original's, value for value; the bundle from the package is the bundle from the original database;
//      the keys are the same and every small key's bytes are unchanged (no shard moves). Only then the package is committed (`subject.sqlite.building` is renamed
//      to `subject.sqlite`, the old `small_files.sqlite` renamed `small_files.sqlite.format52`) and registered in the catalogue (the commit of that subject).
//   5. THE WHOLE: the index is built from the packages and compared with the original database table by table, row for row (a row of the original that no package
//      holds, or any value that differs, refuses the end: nothing is renamed and the run can be repeated once it is understood); `variable_dictionary` and
//      `subject_variable` rows, if any, move to `participants.sqlite`.
//   6. THE END: the original database is renamed `dataset.sqlite.format52`.
// Group fields `pyramid reduce` wrote into the default subject stay in its package for now (moving them changes the rows' owner, which a verified row-for-row
// migration cannot do): a later, explicit step moves them into a group package.
//
// JOIN (`join_dataset`) writes the one database back: the index build, written as `dataset.sqlite` (the original's `meta` kept when `dataset.sqlite.format52`
// exists), the participants table moved back into it, each subject's small keys back into `small_files.sqlite` (byte for byte); then the catalogue, the
// participants file and each `subject.sqlite` are renamed aside (`*.format53`), never deleted. `split` then `join` gives back the original database, compared table by
// table, and every key's bytes.
#pragma once
#include <filesystem>
#include <functional>
#include <string>

#include "nxr/io/json_text.h"

namespace nxr::io {

struct SplitOptions {
  std::string registered_by;                     // recorded with each catalogue entry: required
  bool dry_run = false;                          // report the plan; change nothing (not even bring the dataset to format 52)
  // Called at the stages of a subject's migration ("built", "verified", "committed", "registered") with its name: a test throws from it to stop there.
  std::function<void(const std::string& stage, const std::string& subject)> on_stage;
};
struct JoinOptions {
  std::function<void(const std::string& stage, const std::string& subject)> on_stage;   // "database", "keys", "aside"
};

// The report: per subject its outcome (`migrated`, `already`), the template, the counts of rows and anchor rows, the keys and bytes verified.
Json split_dataset(const std::filesystem::path& dataset_folder, const SplitOptions& opt);
Json join_dataset(const std::filesystem::path& dataset_folder, const JoinOptions& opt = {});

// FORMAT 56 FOR A CATALOGUE MADE BEFORE IT (`nxr-neuro migrate template-links`): THE TEMPLATE HOLDS THE COHORT, a subject names no template. In the catalogue and in
// every member package: `subject.template_id` set NULL; in each subject package the rows it carried as anchors from a template (its `subject` row, whatever else)
// removed with their `anchor_row` entries, refused (that package left as it was) when a row of the subject still points at one of them; the record's template
// reference dropped. Then every database is opened to write (55 -> 56 in place), each package's content hash recomputed, the packages registered again and the
// index (a cache) rebuilt. A catalogue holding a group package (kind `group`) is refused, naming it. No array is touched.
// A ONE-DATABASE dataset: every `template_id` cleared and each subject of kind `group` FOLDED INTO the dataset's one template (its fields re-owned by the template,
// their arrays moved from the group's folder to the template's, the group's row removed, its emptied folder kept aside as `<folder>.format55-group`); a group owning
// anything but fields, or an array path the template already holds, is refused. Then the database is brought to 56.
struct TemplateLinksOptions {
  std::string registered_by;                     // recorded with each catalogue entry: required
  bool dry_run = false;                          // report what would change; change nothing
};
Json migrate_template_links(const std::filesystem::path& dataset_folder, const TemplateLinksOptions& opt);

// ── THE SELECTIONS' STORAGE (schema 52): label arrays and the JSON form of 2026-11-17 to member numbers + labels beside them, a constant-Q ladder's level code to the power of two
// of its window (a ladder lacking the window is refused), a spatial dyadic ladder's rule noted; see migrate_selection.cpp. In place, resumable, `dry_run` changes nothing. Run it BEFORE
// `split_dataset` (a split refuses a dataset not converted). Sets `meta.selection_storage`. Report: counts per subject.
struct SelectionStorageOptions {
  bool dry_run = false;
  std::function<void(const std::string& stage, const std::string& subject)> on_stage;
};
Json migrate_selection_storage(const std::filesystem::path& dataset_folder, const SelectionStorageOptions& opt = {});

// ── THE MEASUREMENTS' STORAGE (schema 53; docs/specs/2026-11-17-consolidation.md section 6.3): ONE table holds a value over a selection's members, `selection_measurement`. The `statistic` rows
// that are a value per member of a partition (their axes_json has an axis on the partition's members) become array-backed measurements; a product made only to couple a level is rewritten as a
// product of (grid, level); the day-anchored tower tilings are marked read-only. In place, subject by subject, RESUMABLE and verified; WRITTEN, NOT RUN: the fixture alone is migrated, in the final round.
//   1. For each such statistic: its values are rewritten into [rows x members x components] at `<path>.members` (rows = the other axes in order, components = the trailing parts; the statistic's options,
//      producer, axes, value type, components and unit in the array's attributes), VERIFIED byte for byte against the source, and one `selection_measurement` row is made (selection_id = the partition,
//      of_field_id and of_statistic_id as the statistic's, measure, array_path, unit, computed_by). A measurement of that measure already on the partition is a CONFLICT (reported, nothing overwritten).
//      The statistic is marked in its options (`migrated_to_measurement`) and, by default, LEFT IN PLACE with its array: nothing is deleted unless the removal step is requested.
//   2. A product whose factor is a LEVEL ROW of a ladder (made only to couple a level) names the ladder and the level instead (`selection_factor.level` = the level's identity, the level row's ordinal, `params_json.product`
//      entries gain `level`); the codes are unchanged. The level rows stay.
//   3. A frame tiling on the tower (params_json.atlas.frame_level / frame_code0) gets `retired_grid: "tower"`: its measurements stay readable, new runs use the line's time grid.
//   4. meta `measurement_storage` = "members" (a dataset made now has it).
// `remove_statistics` is the EXPLICIT second step, run only when the option is set: each migrated statistic whose measurement is verified is removed (a statistic whose parent was it is re-pointed to the
// measurement: of_statistic_id null, of_field_id kept, options.of_measurement); the drain then removes the old array by its queued delete job.
struct MeasurementStorageOptions {
  bool dry_run = false;                           // report the plan; change nothing
  bool remove_statistics = false;                 // the second step above (never taken by default)
  std::function<void(const std::string& stage, const std::string& subject)> on_stage;   // "converted", "products", "tilings", "removed": a test throws from it to stop there
};
// The report: per subject the statistics converted, conflicts (by name), products rewritten, tilings marked, statistics removed; the dataset's totals and `dry_run`.
Json migrate_measurement_storage(const std::filesystem::path& dataset_folder, const MeasurementStorageOptions& opt = {});

}  // namespace nxr::io
