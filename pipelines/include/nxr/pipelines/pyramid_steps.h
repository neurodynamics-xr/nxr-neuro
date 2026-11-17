// nxr-pipelines — the measurement pyramid's nodes (pyramid.band_analytic, pyramid.band_sums, pyramid.frame_samples, pyramid.estimate, pyramid.bands, pyramid.spatial_bands) registered in
// nxr::graph::registry(); idempotent. See src/pyramid_steps.cpp and docs/specs/2026-11-17-pyramid-grid.md.
#pragma once

namespace nxr::pipelines::graph {

/** registers the pyramid's nodes (grouped names `pyramid.*`) when they are not there yet; called by `register_partition_nodes` before the registry of steps is first read */
void register_pyramid_nodes();

}  // namespace nxr::pipelines::graph
