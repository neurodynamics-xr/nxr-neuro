// nxr-io — the PARTICIPANTS TABLE: what is recorded about a dataset's subjects (age, sex, group), in its own file `<dataset>/participants.sqlite`, apart from the
// catalogue and the packages because it can be the most sensitive part of a dataset (docs/specs/2026-11-16-subject-centred-datasets.md section 3.1). A dataset is
// shared, published or copied to a cluster WITHOUT it; no subject and no catalogue ever needs it. It is never copied, exported or published unless a command
// names it: `participants_export` writes only to the paths it is given.
//
// Its two tables have the model's `variable_dictionary` and `subject_variable` columns (BIDS participants.json and participants.tsv); the foreign keys to rows of
// another file (the dataset, the subject) are dropped, the subject being the catalogue's id. A missing value is a missing row.
// BIDS participants.tsv: tab-separated, a header, a `participant_id` column matched to the catalogue's SUBJECT NAMES, `n/a` for missing. participants.json (the
// sidecar): per variable `Description`, `Units`, `Levels` (an object whose keys are the levels), and, as an extension, `Type` ("number", "string" or "boolean").
// Without a `Type` a variable is a `number` when every value it has is a number and a `string` otherwise.
#pragma once
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/subject_filter.h"

namespace nxr::io {

// The variables recorded (insertion order); empty when the dataset has no participants table.
std::vector<VariableInfo> participant_variables(const std::filesystem::path& dataset_folder);
// Every subject's values by SUBJECT ID (a subject with no row has none).
std::map<std::string, ParticipantValues> participant_values(const std::filesystem::path& dataset_folder);

struct ParticipantsImportReport {
  std::vector<std::string> variables;      // those the file defined
  std::int64_t n_subjects = 0;             // subjects with at least one value
  std::int64_t n_values = 0;
};
// Import `tsv` (and `sidecar`) into the dataset's participants table, under the dataset's writer lock. All or nothing: a participant the catalogue does not hold is
// refused by name and nothing is written; a value that does not fit its variable's type or levels is refused with its row. A variable the file defines replaces
// that variable's earlier definition and values; variables it does not define are kept.
ParticipantsImportReport participants_import(const std::filesystem::path& dataset_folder, const std::filesystem::path& tsv, const std::optional<std::filesystem::path>& sidecar);
// Write the table as BIDS participants.tsv (every subject of the catalogue with a value, `n/a` for missing) and, when `sidecar` is given, participants.json.
void participants_export(const std::filesystem::path& dataset_folder, const std::filesystem::path& tsv, const std::optional<std::filesystem::path>& sidecar);

}  // namespace nxr::io
