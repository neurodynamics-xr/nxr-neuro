// nxr-pipelines — the flow through the fibre network and its connectome eigenmodes as graph nodes (connectome_flow_steps.cpp; spec docs/specs/2026-11-24-connectome-flow.md)
#pragma once

namespace nxr::pipelines::graph {

/** registers meeg.fiber_lengths, connectome.tile_pairs, connectome.laplacian_eigenmodes, connectome.tile_operator, connectome.tile_flow, connectome.tile_time_cells and
 *  connectome.network_flow when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_connectome_flow_nodes();

}  // namespace nxr::pipelines::graph
