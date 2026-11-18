// nxr-pipelines — the stage names of a run's stored vortex-core crossings (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 8). The run-to-run repeat itself is the `run_repeat` step
// (pipelines/src/steps_patches.cpp) of the `run_repeat` definition: a statistic over two runs' `tile_crossings` over the same tile level.
#pragma once
#include <string>
#include <vector>

namespace nxr::pipelines {

/** the stage names a run's crossing rows carry (` · stages burst,onset` at the end of their `computed_by`); empty for crossings stored before they were named */
std::vector<std::string> crossing_stage_names(const std::vector<std::string>& computed_by);
/** the index of the stage `name` among `names` (the pair of rows 2i, 2i+1); refused with the names the run has (`which` names the run) */
int crossing_stage_index(const std::string& which, const std::vector<std::string>& names, const std::string& name);

}  // namespace nxr::pipelines
