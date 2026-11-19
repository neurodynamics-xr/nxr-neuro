// nxr-pipelines — the nodes of the forms on a surface's intrinsic complex (signal.phase_split) registered in nxr::graph::registry(); idempotent. See src/form_steps.cpp and
// docs/specs/2026-11-19-intrinsic-form-nodes.md.
#pragma once

namespace nxr::pipelines::graph {

/** registers the intrinsic-form nodes when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_form_nodes();

}  // namespace nxr::pipelines::graph
