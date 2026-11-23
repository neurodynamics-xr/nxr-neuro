// nxr-pipelines — the nodes of the mode-space Fourier coefficients (signal.mode_analysis, signal.spectral_coefficients) registered in nxr::graph::registry(); idempotent.
// See src/spectral_steps.cpp and the definition `spectral_coefficients`.
#pragma once

namespace nxr::pipelines::graph {

/** registers the spectral-coefficient nodes when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_spectral_nodes();

}  // namespace nxr::pipelines::graph
