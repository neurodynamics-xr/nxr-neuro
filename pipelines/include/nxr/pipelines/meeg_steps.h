#pragma once

namespace nxr::pipelines::graph {

/** registers the M/EEG nodes (grouped names `meeg.*`) in nxr::graph::registry() when they are not there yet; called before the registry of steps is first read */
void register_meeg_nodes();

}  // namespace nxr::pipelines::graph
