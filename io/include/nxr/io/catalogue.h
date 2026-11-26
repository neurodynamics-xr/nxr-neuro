// nxr-io — the DATASET MODULE: a dataset is a light CATALOGUE of subject packages (docs/specs/2026-11-16-subject-centred-datasets.md sections 3, 5 and 6).
//
//   <dataset>/                    any path; addressed by its own path (there is no datastore and no catalog.json)
//     catalogue.sqlite            the dataset: its members (by id, path, versions, content hash): subjects and the dataset's ONE template
//     participants.sqlite         the participants table (participants.h): its own file, so it can be withheld; optional
//     index.sqlite                a CACHE of every member's rows, built from the packages (dataset_index.h); deletable
//     .writer.lock                held only while the dataset module's files are written, for seconds
//     <subject>.nxr.zarr/         subject packages (subject_package.h)
//
// THE DEPENDENCY GOES ONE WAY: this module reads subject packages; a subject package never reads it. The catalogue holds the model's own `dataset` and `subject`
// rows (a copy of each member's own row, refreshed at registration) and one table of its own, `subject_entry`; it holds no manifold, field, selection or
// operator row and no participant variable.
//
// THE TEMPLATE HOLDS THE COHORT (format 56): a subject names no template and carries nothing of it. The dataset's own template subject (`dataset_template`) holds each
// subject's registration onto it, each subject's results carried onto it, and the averages and statistics over them; it refers to its subjects (their rows as
// anchor rows, their arrays read from their own packages). A subject step that needs the template's definitions reads it read-only (`open_dataset_template`).
//
// REGISTERING replaces staging and merge: no array is read or moved; the package's record and rows are checked and one short transaction is written.
//   the same subject id and content hash   -> `unchanged` (a rerun is a no-op)
//   the same id, a new hash                -> the entry is refreshed (the package is the authority on its own content): `updated`
//   another id under a name or path the catalogue holds -> refused, unless `replace`, which marks the old entry `replaced` and moves its package aside
//                                              (`<folder>.replaced-<date>`), never deleting it
//   a failed check registers nothing for that package and does not stop the others.
// A catalogue writes only packages inside its own folder; one it lists from elsewhere (by absolute path) is read-only to it.
#pragma once
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/subject_filter.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/templates.h"

namespace nxr::io {

constexpr const char* kCatalogueFile = "catalogue.sqlite";
constexpr const char* kParticipantsFile = "participants.sqlite";
constexpr const char* kIndexFile = "index.sqlite";
constexpr int kCatalogueVersion = 1;

struct CreateCatalogueOptions {
  std::string name;                                   // the dataset's name: required
  std::optional<std::string> source_tool, source_protocol, source_path, created_by, built_by;
};

// A member as the catalogue records it.
struct CatalogueEntry {
  std::string subject_id, name, path, kind;           // path: relative to the dataset folder, or absolute for a package elsewhere
  int schema_version = 0, store_format = 0;
  std::string content_hash, registered_utc, registered_by;
  std::string status;                                 // registered · missing · stale · replaced
  std::int64_t n_acquisitions = 0, n_recordings = 0, n_surfaces = 0, n_fields = 0, n_operators = 0;
  Json to_json() const;
};

class Catalogue {
 public:
  // Read only; refuses a folder with no catalogue (`not_found`) and a catalogue at a version this build does not know (`schema_version`).
  static Catalogue open(const std::filesystem::path& folder);
  Catalogue(Catalogue&&) noexcept;
  Catalogue& operator=(Catalogue&&) noexcept;
  ~Catalogue();

  const std::filesystem::path& folder() const { return folder_; }
  const Database& db() const { return *db_; }
  Row dataset_row() const;
  // The members, by name (replaced entries last).
  std::vector<CatalogueEntry> entries() const;
  // By name or id; a replaced entry is found only by its id.
  std::optional<CatalogueEntry> find(const std::string& name_or_id) const;
  // Where a member's package is.
  std::filesystem::path package_path(const CatalogueEntry& entry) const;
  bool has_participants() const;
  // What to give `create_package` so a package made for this dataset carries this dataset as its home (its name, id and hint row).
  CreatePackageOptions package_options() const;

 private:
  Catalogue() = default;
  std::filesystem::path folder_;
  std::unique_ptr<Database> db_;
};

// CREATE a dataset folder's catalogue (the folder is made when absent; one that holds a catalogue is refused: `already_exists`).
Catalogue create_catalogue(const std::filesystem::path& folder, const CreateCatalogueOptions& opt);

// CREATE a catalogue that carries a dataset's own `dataset` and `datastore` rows as they are (the migration of a dataset database: the ids stay). `lock` false: the
// caller already holds the dataset folder's writer lock.
Catalogue create_catalogue_from(const std::filesystem::path& folder, const Row& dataset, const Row& datastore, const std::string& built_by, bool lock = true);

struct RegisterOptions {
  bool replace = false;                               // another subject id under a name the catalogue holds replaces it (the old package is moved aside)
  std::string registered_by;                          // the program or person recorded with each entry: required
};
struct RegisterResult {
  std::filesystem::path package;
  std::string name;                                   // the subject's, when the package could be read
  std::string outcome;                                // registered · unchanged · updated · replaced · refused
  std::string reason;                                 // for a refusal: what failed, naming the package and the other party
};
// Register the packages, each in its own short transaction under the dataset's writer lock. Every check of section 6 runs: the versions are known, the package is
// complete (no pending node, no writer lock, a record whose hash is the package's), its ids clash with no other member's (and no other candidate's in the same
// call), its shared vocabulary agrees by key with the dataset's. A refusal names the package and registers nothing for it.
std::vector<RegisterResult> register_packages(const std::filesystem::path& folder, const std::vector<std::filesystem::path>& packages, const RegisterOptions& opt);
// The packages (`*.nxr.zarr` holding a subject.sqlite) directly inside the folder, sorted: what `--scan` registers.
std::vector<std::filesystem::path> scan_packages(const std::filesystem::path& folder);
// Every member's package looked at: `missing` when it is gone, `stale` when its content hash differs from the entry's; written to the entries. Returns them.
std::vector<CatalogueEntry> refresh_member_status(const std::filesystem::path& folder);

// OPEN ONE SUBJECT of a catalogue dataset for computing: the subject's package, found through the catalogue (read only), opened as `open_package_writer` does (its own
// writer lock; `opt.lock` false for a read), so every pipeline written for a dataset runs on it unchanged. A subject is self-contained: nothing is routed to a template.
// The template itself is opened this way too (it is a member); a group analysis opens it as its writer. Refused, by name: a dataset that is not a catalogue
// (`not_found`), an unknown subject, a member that is not `registered` or whose package is missing.
DatasetWriter open_subject_package(const std::filesystem::path& dataset_folder, const std::string& subject, const OpenOptions& opt = {});
// THE DATASET'S TEMPLATE: the one registered member of kind `template`. None: `not_found` ("<dataset>: no template subject"); more than one: `invalid_argument`, naming
// them (a dataset has ONE template, its own copy).
CatalogueEntry dataset_template(const std::filesystem::path& dataset_folder);
// The dataset's template, read only (no lock): for a subject step that needs the template's definitions (its tiling, its sphere). What the step stores is the
// subject's own; its provenance names the template (`producer.template = {name, subject_id}`).
SubjectPackage open_dataset_template(const std::filesystem::path& dataset_folder);
// The template of the dataset a member's package lies in (the package's folder is inside its dataset's), opened as `open_subject_package` does (`opt.lock` false
// for a read). Refused when `member` is not a package or its folder is not inside a catalogue dataset.
DatasetWriter open_template_of(const DatasetWriter& member, const OpenOptions& opt = {});
// Whether the folder is a catalogue dataset (holds `catalogue.sqlite`); a one-database dataset holds `dataset.sqlite`.
bool is_catalogue_dataset(const std::filesystem::path& dataset_folder);

// SELECTING SUBJECTS: exactly one of `subjects` (names or ids of the catalogue) or `where` (a filter over the participant variables, subject_filter.h); neither or
// both is refused, and so is a selection that matches nothing.
struct SubjectSelection {
  std::vector<std::string> subjects;
  std::optional<std::string> where;
};
struct SelectedSubjects {
  std::vector<CatalogueEntry> members;                // each with its content hash
  std::string how;                                    // "subjects" or "where"
  std::string filter_text;                            // the --where text, or empty
  int filter_language_version = 0;                    // 0 when the selection was a list
  Json provenance() const;                            // what a result records: how, the filter and its version, each member's name, id and hash
};
// Refused too: an unknown name (by name), a member that is not `registered` or whose package is missing or differs from its entry (`stale`: register again), and
// a member that is not a subject (a template or a group is never a member of a cohort).
SelectedSubjects select_subjects(const std::filesystem::path& folder, const SubjectSelection& selection);

// A GROUP ANALYSIS (format 56: the template holds the cohort): the members of a cohort read each from its own package (read only, no lock), the result written into
// THE DATASET'S TEMPLATE (its writer lock: one cohort writer at a time). `open` selects the cohort through the catalogue (`select_subjects`: the same refusals) and
// opens the dataset's template (`dataset_template`: none or two refused). `anchor_contributions` carries the members' rows a result's contributions name into the
// template as anchor rows (the template refers to its subjects; a subject never refers to the template). `finish` closes the template (its record and content hash)
// and registers it again. A result's provenance records the members and their content hashes (`producer.group`); `cohort_member_status` reads them back.
class GroupAnalysis {
 public:
  static GroupAnalysis open(const std::filesystem::path& dataset_folder, const std::vector<std::string>& subjects, const std::string& registered_by);
  GroupAnalysis(GroupAnalysis&&) noexcept;
  GroupAnalysis& operator=(GroupAnalysis&&) noexcept;
  ~GroupAnalysis();
  // A member's package, read only; the first call opens it.
  DatasetWriter& member(const std::string& name);
  // The dataset's template, open to write: where the cohort's results go.
  DatasetWriter& template_writer();
  // The template's name.
  const std::string& template_name() const;
  const SelectedSubjects& selected() const;
  // (subject id, field id) pairs: each member's subject row and the source fields' rows, with what they name, carried into the template. Run outside a transaction.
  void anchor_contributions(const std::vector<std::pair<std::string, std::string>>& subject_and_field);
  // Close the template and register it again (its content hash changed); returns the registration outcome.
  std::string finish();

 private:
  GroupAnalysis() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Each member a cohort result was made from, against the catalogue now: `producer_group` is the result's `producer.group` ({members: [{subject_id, name,
// content_hash}], ...}). `current` false when the member is gone, replaced, or its package has changed since (re-run, re-imported): the result is FLAGGED, never deleted.
struct GroupMemberStatus {
  std::string subject_id, name, recorded_hash, current_hash;   // current_hash empty when the member is gone from the catalogue
  bool current = false;                                        // the member is as the result saw it
};
std::vector<GroupMemberStatus> cohort_member_status(const std::filesystem::path& dataset_folder, const Json& producer_group);

}  // namespace nxr::io
