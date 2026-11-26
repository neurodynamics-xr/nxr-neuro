// nxr-io — COPY A SUBJECT'S STORED NODES FROM ANOTHER PACKAGE (`copy nodes`; register D171): a manifold, everything that lies on it or was made from it, and the arrays they
// name, into a package of the same subject, so what was solved once (an intrinsic triangulation with its Hodge-Dirac eigensolve, a sensor mesh, a glued cortex) is taken, never
// solved again. Nothing is computed: rows and bytes are copied, and every byte copied is checked against the digest the source recorded.
//
//   WHAT GOES: each named manifold (a seed) and, until nothing is added, every node that lies on or is made from one that goes: the manifolds derived from it or its components,
//   the fields and selections on it, the operators from or to it, the selections that are members of one that goes, the statistics of a field that goes, and every node the same
//   run made (the same cache key in its provenance: the outputs of one run go together). A product (a manifold times a time line) is never followed. A manifold's topology and
//   geometry rows go with it, a selection's members, extents, spans, factors and measurements, a field's contributions, an operator's factors, a dictionary's entries.
//   WHAT IT NAMES IN THE TARGET: every other row a copied row names is found in the target, and must be the same there: a node by its path (its row's counts and the digests of
//   its chunks equal), an acquisition by its label, a space and a coordinate system by their names, a dictionary by name and scope (its entries equal); the subject is the
//   target's. A space, a coordinate system or a dictionary the target lacks is copied; a node or an acquisition it lacks refuses the copy, naming it. A node that goes and that
//   the target already holds, the same, is KEPT (the target's row is used, nothing written); one the target holds and that differs refuses the copy.
//   THE ROWS ARE THE TARGET'S: every copied row gets a new id; every id in its columns and its JSON (the cache keys of its provenance included) is the target's, so a run asked
//   again in the target finds the copied result as its own. Each copied node's provenance gains `copied` {from_package, from_subject, from_id, from_content_hash, utc, by}.
//   CHECKED: the source is a package whose recorded content hash equals the hash of its rows and digests (nothing changed since it was closed), at this schema version, of a
//   subject of the same name; the target is a package; every copied chunk's size and SHA-256 in the target equal those the source recorded (a zarr.json document is rewritten
//   with the target's ids, and its attributes with the target's rows). BYTES FIRST, THE ROWS LAST, in one transaction; a failure removes what was written.
#pragma once
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"
#include "nxr/io/writer.h"

namespace nxr::io {

struct CopyNodesOptions {
  std::vector<std::string> manifolds;   // the seeds: manifolds of the source, by name or path
  bool dry_run = false;                 // report what would be copied, kept and found, writing nothing
  std::string by;                       // who copies (the program), in each copied node's `copied` record
};

// Copy into `target` (a subject of a package opened for writing) the seeds of the package at `source` and what goes with them. Returns
// {from {package, subject, content_hash}, copied [{class, name, path, id, from_id}], kept [{class, name, path, id}], found [{class, name, path}], keys, bytes, dry_run}.
Json copy_nodes(const std::filesystem::path& source, SubjectWriter& target, const CopyNodesOptions& opt, const std::function<void(const Json&)>& progress = {});

}  // namespace nxr::io
