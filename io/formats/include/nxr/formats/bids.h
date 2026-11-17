// BIDS (the Brain Imaging Data Structure): a folder parsed into an in-memory description. Pure: it lists and reads folders and small files (JSON,
// TSV) through a `Source`, and imports nothing; a recording's data files are named, never read (the format's own reader does that).
//
//   bids::DirectorySource src("/data/ds-rest");
//   bids::Dataset ds = bids::read_dataset(src);
//   for (const auto& s : ds.subjects) for (const auto& ses : s.sessions) for (const auto& rec : ses.recordings) { rec.data_files; rec.sidecar; rec.channels; }
//
// Follows the BIDS specification (v1.10): file names are `key-value` entities joined by `_`, then a suffix and an extension (`sub-01_ses-a_task-rest_run-02_eeg.edf`),
// in the specification's entity order; folders `sub-<label>[/ses-<label>]/<datatype>/`; the dataset's `dataset_description.json`, `participants.tsv`
// and `participants.json`; `*_scans.tsv`; and the INHERITANCE PRINCIPLE for metadata files: a `.json` (or `.tsv`) file in a folder applies to the
// files in that folder and below whose entities include all of its own and whose suffix it names; JSON values of a deeper file override those of a
// shallower one key by key; for tables (channels, events, electrodes) the most specific file replaces the others. MNE-BIDS (BSD-3-Clause) is the
// reference the goldens come from (test/golden/gen_bids.py); the rules are the specification's, not MNE-BIDS's code.
//
// What is parsed: every subject and session folder, every file in a datatype folder (those that are not BIDS-named go in `issues`, never dropped
// silently), recordings of the electrophysiology datatypes (meg, eeg, ieeg) with split files grouped and companion files (BrainVision `.vmrk`/`.eeg`,
// EEGLAB `.fdt`) attached, their merged sidecar, channels, events, electrodes and coordinate-system files, the scans table, and the derivative
// datasets under `derivatives/`. A BIDS-derivatives dataset (DatasetType `derivative`) reads the same way.
#pragma once
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"
#include "nxr/formats/byte_source.h"
#include "nxr/formats/tsv.h"

namespace nxr::formats::bids {

// ---- entities and file names --------------------------------------------------------------------------------------------------------------

// The BIDS entities in the order a file name carries them: short key (`sub`) and long name (`subject`).
struct EntityDef { const char* key; const char* name; };
const std::vector<EntityDef>& entity_definitions();
int entity_rank(const std::string& key);   // position in the specification's order, -1 for a key the specification does not define

struct Entities {
  std::vector<std::pair<std::string, std::string>> items;   // (key, value) in the order the file name has them
  std::string suffix;                                         // `eeg`, `events`, `T1w`, `participants`, ...
  std::string extension;                                      // with its dots (`.nii.gz`, `.tsv`, `.fif`); empty for none

  bool has(const std::string& key) const { return get(key).has_value(); }
  std::optional<std::string> get(const std::string& key) const;
  void set(const std::string& key, const std::string& value);   // replaces, or appends
  bool in_canonical_order() const;                              // keys the specification does not define are never canonical
  bool subset_of(const Entities& other, const std::set<std::string>& ignore = {}) const;   // every entity here is in `other` with the same value (but those in `ignore`)
};

// `sub-01_ses-a_task-rest_eeg.edf` -> entities, suffix, extension. Refuses (format_error) a name that is not of that form: an empty key or value, a
// character outside [a-zA-Z0-9], a repeated key, no suffix. A name with no entity (`participants.tsv`) is valid.
Entities parse_name(const std::string& file_name);
// The name back: entities (in the order given, or sorted into the specification's order), `_`, suffix, extension.
std::string build_name(const Entities& e, bool canonical_order = true);

// ---- the folder source --------------------------------------------------------------------------------------------------------------------

struct Entry { std::string name; bool is_dir = false; };

class Source {
 public:
  virtual ~Source() = default;
  virtual std::vector<Entry> list(const std::string& dir) const = 0;   // names in `dir` (relative to the dataset root, "" is the root), sorted
  virtual bool exists(const std::string& path) const = 0;
  virtual std::shared_ptr<const ByteSource> open(const std::string& path) const = 0;
};

class DirectorySource final : public Source {
 public:
  explicit DirectorySource(std::string root);   // io error when it is not a folder
  std::vector<Entry> list(const std::string& dir) const override;
  bool exists(const std::string& path) const override;
  std::shared_ptr<const ByteSource> open(const std::string& path) const override;
  const std::string& root() const { return root_; }
 private:
  std::string root_;
};

// A tree in memory (tests, a browser client): files by relative path; folders implied.
class MemoryTree final : public Source {
 public:
  void add(const std::string& path, const std::string& content);
  void add_dir(const std::string& path);   // an empty folder, or a folder that is itself a data file (a CTF `.ds`)
  std::vector<Entry> list(const std::string& dir) const override;
  bool exists(const std::string& path) const override;
  std::shared_ptr<const ByteSource> open(const std::string& path) const override;
 private:
  std::vector<std::pair<std::string, std::string>> files_;
  std::set<std::string> dirs_;
};

// ---- the description ----------------------------------------------------------------------------------------------------------------------

struct Issue { std::string path; std::string message; };   // something the reader parsed around, never silently

struct DatasetDescription {
  nlohmann::json raw;                                  // the whole file
  std::string name, bids_version;                      // the two required fields
  std::optional<std::string> dataset_type, license, how_to_acknowledge, dataset_doi, acknowledgements, funding_text;
  std::vector<std::string> authors, funding, ethics_approvals, references_and_links;
  nlohmann::json generated_by, source_datasets;        // arrays of objects as the file has them (null when absent)
};

struct File {
  std::string path;          // relative to the dataset root, with `/`
  std::string datatype;      // the folder it is in (`anat`, `eeg`, ...); empty for a file at the subject or session level
  Entities entities;
};

struct TableFile { std::string path; tsv::Table table; };
struct JsonFile { std::string path; nlohmann::json content; };

// An electrophysiology recording (a data file of suffix meg, eeg or ieeg).
struct Recording {
  std::string datatype;                        // meg, eeg, ieeg
  Entities entities;                           // of the first part: its `split` entity, if any, is in it
  std::vector<std::string> data_files;         // the parts in `split` order (one for a recording that is not split); a CTF `.ds` is a folder
  std::vector<std::string> companions;         // files sharing the data file's name with another extension (BrainVision `.vmrk` and `.eeg`, EEGLAB `.fdt`)
  std::vector<std::string> sidecar_files;      // the `<suffix>.json` files that apply, shallowest first (their merge is `sidecar`)
  nlohmann::json sidecar;                      // the merge, deeper overriding shallower key by key; an empty object when there is none
  std::optional<TableFile> channels, events;   // the most specific applicable table
  std::optional<JsonFile> channels_sidecar, events_sidecar;
  std::vector<TableFile> electrodes;           // one per `space` entity (all that apply)
  std::vector<JsonFile> coordsystems;          // one per `space` entity, or the one without
  std::vector<std::string> other_files;        // files of the same recording with other suffixes (`_headshape.*`, `_photo.*`, `_markers.*`)
};

struct Session {
  std::string label;                           // empty when the subject has no session folders
  std::vector<Recording> recordings;
  std::vector<File> files;                     // the other files of the session (anatomy, functional, ...)
  std::optional<TableFile> scans;              // `*_scans.tsv`
};

struct Subject {
  std::string label;
  std::vector<Session> sessions;               // one session with an empty label for a subject without `ses-` folders
  std::vector<File> files;                     // files directly in the subject folder (`_sessions.tsv`, a subject-level sidecar)
};

struct Dataset {
  DatasetDescription description;
  std::optional<TableFile> participants;
  std::optional<JsonFile> participants_sidecar;
  std::vector<Subject> subjects;
  std::vector<std::string> derivatives;        // `derivatives/<name>` folders (each a dataset of its own; read them with a source rooted there)
  std::vector<std::string> top_level_files;    // README, CHANGES, LICENSE, .bidsignore, `task-*_*.json` at the root, ...
  std::vector<Issue> issues;
};

struct ReadOptions {
  bool strict = false;                         // refuse (format_error) instead of listing an issue for a file whose name is not BIDS-formed
  std::vector<std::string> subjects;           // labels (without `sub-`); empty: all
};

Dataset read_dataset(const Source& src, const ReadOptions& options = {});
DatasetDescription parse_dataset_description(const nlohmann::json& j);   // refuses a file without `Name` or `BIDSVersion`

// ---- the inheritance principle ------------------------------------------------------------------------------------------------------------

// The files named `<entities>_<suffix><extension>` that apply to `target` (a path relative to the root), found from the root down to the target's
// folder, least specific first (shallower folder first; within a folder, fewer entities first). `ignore` lists entities of a candidate that need not
// be in the target (`space` for electrodes and coordinate systems).
std::vector<std::string> applicable_files(const Source& src, const std::string& target, const std::string& suffix, const std::string& extension,
                                          const std::set<std::string>& ignore = {});
// The JSON files that apply, merged (deeper overrides shallower, key by key; nested objects are replaced, not merged, as the specification says).
nlohmann::json merged_sidecar(const Source& src, const std::string& target, const std::string& suffix, std::vector<std::string>* used = nullptr);

}  // namespace nxr::formats::bids
