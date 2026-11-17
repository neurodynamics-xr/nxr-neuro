// nxr-pipelines — the optical flow's nodes (timebank.member_analytic, flow.activation, flow.max_amplitude, flow.velocity, flow.hodge_modes, flow.series) registered in
// nxr::graph::registry(); idempotent. See src/flow_steps.cpp and docs/specs/2026-11-17-flow-verbs.md.
#pragma once

namespace nxr::pipelines::graph {

/** registers the flow nodes when they are not there yet; called by `register_partition_nodes` with the other grouped nodes */
void register_flow_nodes();

}  // namespace nxr::pipelines::graph
