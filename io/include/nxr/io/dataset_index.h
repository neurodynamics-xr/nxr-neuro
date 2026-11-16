// nxr-io — the CROSS-SUBJECT INDEX: a CACHE of every member's rows, for the readers that want SQL across subjects (docs/specs/2026-11-16-subject-centred-datasets.md
// section 3.3). `<dataset>/index.sqlite` is a database in the model's schema holding every registered package's rows in the shape of a dataset database, so a
// reader of one reads it unchanged.
//   - BUILT FROM THE PACKAGES, never written otherwise; its `meta.index_json` records the content hash of every package it was built from. A reader finds the index
//     stale when a member's hash differs or a member is new or gone, and rebuilds it. Deleting it loses nothing.
//   - Its build is a merge without moving anything: each package's rows folded in, children with their parents; the home dataset and datastore rows become the
//     catalogue's one; a shared vocabulary dictionary (app or dataset scope) is matched across packages by its key (scope, name), the first copy kept and every
//     reference remapped onto it. Two copies of one key with different entries are a conflict, and an id two packages both own is a clash: the build reports each
//     by package and does not guess (nothing is written).
//   - A package whose version this build does not know refuses the build, naming it.
#pragma once
#include <filesystem>
#include <string>
#include <vector>

#include "nxr/io/catalogue.h"

namespace nxr::io {

struct IndexReport {
  bool rebuilt = false;
  std::int64_t n_subjects = 0;
  std::vector<std::string> subjects;        // names of the members folded in
};
struct IndexProblems {
  std::vector<std::string> clashes, conflicts;   // each names the packages involved
};

// Whether index.sqlite exists and was built from exactly the members' current content hashes.
bool index_is_current(const std::filesystem::path& dataset_folder);
// Build the index when it is not current (or `force`), under the dataset's writer lock; built beside the old one and renamed over it, so a reader sees one or the
// other. Throws (`id_clash` or `invalid_argument`, naming every clash and conflict) and leaves the old index as it was when a check fails.
IndexReport build_index(const std::filesystem::path& dataset_folder, bool force = false);

}  // namespace nxr::io
