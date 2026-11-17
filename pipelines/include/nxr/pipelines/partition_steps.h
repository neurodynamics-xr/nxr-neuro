#pragma once

namespace nxr::pipelines::graph {

/** registers the partition nodes (grouped names: `joint.*`) in nxr::graph::registry() when they are not there yet; called before the registry of steps is first read */
void register_partition_nodes();

}  // namespace nxr::pipelines::graph
