// nxr-pipelines — THE RUNNER'S SIDE OF THE EXECUTOR (docs/specs/2026-11-17-compute-and-run.md, step C5): the registered steps as nodes of the graph executor.
//
// The executor (`nxr_graph`, pipelines/graph/) knows arrays, blocks and types, never a row. The steps (registry.h) are still written on the model's description of a value
// (`finder::Value`: its manifolds, its rows) and ask the runner's `Bound` for what they need beside it. This file joins the two without a copy of a block or an array:
//
//   port_of(Value)   a found or planned value as a typed port, the executor's terms (axis, value type, unit, rate) set from it; the whole description rides in `facts`
//   value_of(port)   the description back, exactly, for a step's plan and the runner's store
//   node_registry    every registered step as a `NodeSpec` of a registry the executor's `check` reads, its functions calling the step's own through `Binding`
//
// The runner holds ONE `Binding` per run and points it at what the nodes may ask: the records only in a plan, the records and the stored constants in a run.
#pragma once
#include <functional>
#include <memory>

#include "nxr/graph/graph.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

/** what the nodes of a run ask: the runner's `Bound` (records in a plan; records and stored constants in a run) and where progress is said */
struct Binding {
  const Bound* bound = nullptr;
  std::function<void(const Json&)> say;
};

/** a value as a typed port; `chunked` is the axis `samples` (`frames` when the step says it made frames), the rest of the description is in `facts["value"]` */
nxr::graph::PortType port_of(const Value& v);
/** the description a port carries: the inverse of `port_of`, exactly */
Value value_of(const nxr::graph::PortType& t);
/** the registered steps as the executor's nodes, bound to `binding` (held by the nodes): a step's kind `setup`/`after` is an `array` node */
nxr::graph::Registry node_registry(const std::shared_ptr<Binding>& binding);

}  // namespace nxr::pipelines::graph
