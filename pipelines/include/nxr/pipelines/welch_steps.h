// nxr-pipelines — the Welch nodes (welch.psd, rotary.pair, welch.complex_psd) registered in nxr::graph::registry(); idempotent. See src/welch_steps.cpp.
#pragma once

namespace nxr::pipelines::graph {

/** registers the spectra's nodes (grouped names) in nxr::graph::registry() when they are not there yet; called before the registry of steps is first read */
void register_welch_nodes();

}  // namespace nxr::pipelines::graph
