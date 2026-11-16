// nxr-pipelines — the RUN-TO-RUN REPEAT of the vortex cores' crossings (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 8): does each tile's direction of core movement repeat from one recording to the next?
// A STATISTIC over two runs' `tile_crossings` (the measurements `run vortex_cores` stored over the same tile level; read through io, never a recording): per tile the dot product of the two runs' mean resultant vectors
// (the two senses pooled), for the tiles with at least `min_crossings` crossings in both runs; and, for the whole recording, the count-weighted mean cosine of the tiles' directions (nxr `repeat`, `tile_dot`; maths
// nxr::crossings::repeat). Stored as two statistics, both of the first run's Field (the second named in their options):
//   <name>.tiles   measure `repeat_dot`, method `run_repeat`, [tiles] (NaN where a run has fewer crossings); the axis is the level's Selection
//   <name>.whole   measure `repeat_cosine`, method `run_repeat`, [1]
// WHAT IT CANNOT SAY: the p-value (a tile's dot against the lag-free null's, Benjamini-Hochberg over the tiles) needs the null (piece 10), built next: here is the statistic only. A repeat is a description of two
// runs agreeing; two runs of the same subject share the cortex, the kernel and the instrument's leakage, so agreement is evidence only against the null.
#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io.h"

namespace nxr::pipelines {

struct RunRepeatOptions {
  std::string recording_a, recording_b;      // REQUIRED: the two recordings (Fields) whose `tile_crossings` are compared; the order does not change the statistic
  std::optional<std::string> over;           // the tile level's Selection (`<surface> tiles L<l>`) both were measured over; default: the one both have (refused when several)
  int stage = 0;                             // the stage by INDEX: rows 2s and 2s+1 of the stored crossings (0: the first stage of the run, `burst` by default); used when `stage_name` is not given
  std::optional<std::string> stage_name;     // the stage by NAME (burst, onset, within, offset): each run's own rows name their stages (` \xc2\xb7 stages ...` in `computed_by`), so two runs may list them in another order; refused when a run lacks it or does not name its stages
  int min_crossings = 10;                    // nxr NMIN_TILE: a tile counts when both runs have at least this many crossings (senses pooled)
  std::optional<std::string> name;           // default `vortex_cores.run_repeat.<a>.<b>` (and `.stage<s>` for a stage other than 0)
  bool replace = false;
};
struct RunRepeatResult { nxr::io::Json info; bool hit = false; };
/** the stage names a run's crossing rows carry (` · stages burst,onset` at the end of their `computed_by`); empty for crossings stored before they were named */
std::vector<std::string> crossing_stage_names(const std::vector<std::string>& computed_by);
/** the index of the stage `name` among `names` (the pair of rows 2i, 2i+1); refused with the names the run has (`which` names the run) */
int crossing_stage_index(const std::string& which, const std::vector<std::string>& names, const std::string& name);
RunRepeatResult run_repeat_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const RunRepeatOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
