// nxr-pipelines — the mode analysis node (signal.mode_analysis) registered in nxr::graph::registry(); idempotent.
// See src/spectral_steps.cpp.
#pragma once

namespace nxr::pipelines::graph {

/** registers the spectral-coefficient nodes when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_spectral_nodes();

}  // namespace nxr::pipelines::graph
