// nxr-pipelines — the flow of a weighted cross-spectrum on a surface's field coordinates as graph nodes (sensor_flow_steps.cpp; spec docs/specs/2026-11-24-sensor-flow.md)
#pragma once

namespace nxr::pipelines::graph {

/** registers signal.overlapped_spectrum, signal.chunk_spectra, signal.spectral_flow, signal.atom_time_cells, stats.sign_flip_consistency, signal.sensor_power and stats.aperiodic_fit
 *  when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_sensor_flow_nodes();

}  // namespace nxr::pipelines::graph
