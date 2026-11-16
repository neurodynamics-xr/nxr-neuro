// nxr-io — a subject's BUNDLE: every row its model needs, in one read, as JSON. A port of the Cortical Flow Studio's `bundleOf`
// (`backend/src/db/bundle.ts`, the reference; nxr's `Dataset.bundle` is the same), so an app builds its entities (`@nxr/io/entities`' `Model.fromBundle`)
// from nxr-neuro's read exactly as from the Studio's:
//   the subject, its dataset and datastore, acquisitions, manifolds, topologies and cells, geometries and dimensions, coordinate systems, spaces,
//   selections with their extents, spans and OWN measurements (code NULL; a member's are queried on demand), fields and contributions, operators and
//   their factors, statistics (complete rows; none before the statistic table), jobs, the dictionaries its selections name plus the vocabularies (`frequency bands`, `recipes`) at the scopes that apply, and their
//   entries; and FOLLOWING THE TEMPLATE (D127/D128): the manifolds of ANOTHER subject its rows name (the default subject's pial a registration maps from)
//   with their components, topologies, geometries, coordinate systems and spaces, the lineage its fields name in another subject (`lineage_fields`,
//   `lineage_operators`), and those subjects (`related_subjects`).
// Every row is a JSON object of its columns (integers, reals, text, null), as node:sqlite returns it.
#pragma once
#include <optional>
#include <string>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"

namespace nxr::io {

// The bundle of the subject `subject_id`; nullopt when there is no such subject.
std::optional<Json> subject_bundle(const Database& db, const std::string& subject_id);
// By the app's store path `<dataset.path>/<subject.path>`.
std::optional<Json> bundle_by_path(const Database& db, const std::string& store_path);
// A row as JSON (the bundle's form).
Json row_json(const Row& r);

}  // namespace nxr::io
