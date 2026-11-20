// Writes in pieces, io's side (nxr/io/task/commands.h `PieceOptions`; docs/specs/2026-11-20-data-layer-queries.md, 3.7). io/src only.
//
// An unfinished write is a folder `<dataset>/.writes/<write id>/` holding `write.json` (what the one-shot command was asked, the shape, the dtype, the regions written, the owner) and
// `values.npy` (the whole array, zeros where nothing is written yet, a sparse file). It is outside the database, so it is invisible to every reader until the commit, and the commit is
// the one-shot command's own unit. Rolling a write back is removing its folder.
#pragma once
#include <string>
#include <vector>

#include "nxr/io/task/commands.h"

namespace nxr::io::task::pieces {

/** `add field | statistic` with `piece.begin | append | commit` */
CommandResult field(const AddFieldRequest& request, const Progress& progress);
CommandResult statistic(const AddStatisticRequest& request, const Progress& progress);

/** what `begin` checks against the subject, read only (the refusals of the one-shot command: the manifold, the name, the shape the values must have) */
void plan_field(const AddFieldRequest& request, const std::vector<std::int64_t>& shape, const Progress& progress);
void plan_statistic(const AddStatisticRequest& request, const std::vector<std::int64_t>& shape, const Progress& progress);

/** the unfinished writes of `dataset`: every one rolled back (`recover dataset`); those a handle began that has closed; those whose owner process is gone (a writable open's recovery) */
int discard_all(const std::string& dataset);
int discard_owned(const std::string& dataset, const std::string& handle_token);
int discard_orphans(const std::string& dataset);
/** the number of unfinished writes of `dataset` (`show dataset`'s counts do not include them; `recover dataset` reports them) */
int count(const std::string& dataset);

}  // namespace nxr::io::task::pieces
