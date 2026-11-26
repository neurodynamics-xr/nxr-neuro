// nxr-pipelines — the bispectrum of the chunk spectra and the flow across frequencies as graph nodes (bispectrum_steps.cpp; spec docs/specs/2026-11-24-bispectrum.md)
#pragma once

namespace nxr::pipelines::graph {

/** registers signal.bispectrum and signal.flow_across_frequencies when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_bispectrum_nodes();

}  // namespace nxr::pipelines::graph
