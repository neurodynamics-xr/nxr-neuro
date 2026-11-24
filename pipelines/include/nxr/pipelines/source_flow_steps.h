// nxr-pipelines — the source mapping of the sensor-space flow as graph nodes (source_flow_steps.cpp; spec docs/specs/2026-11-24-source-flow.md)
#pragma once

namespace nxr::pipelines::graph {

/** registers signal.source_mode_operator, signal.cortical_flow and signal.cortical_time_cells when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_source_flow_nodes();

}  // namespace nxr::pipelines::graph
