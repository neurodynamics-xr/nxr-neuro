// nxr-pipelines — THE DISPATCH POINT: the one place a graph node's computation is called. The executor (runner.cpp) never calls a node's functions itself; it describes the
// call (`NodeCall`: which phase, the node's prepared state, its inputs) and hands it to `dispatch`, which checks the node's declared traits and runs it on the backend of the
// `DispatchContext`. Today the only backend is the CPU. A second backend (a GPU one) is added here, in `dispatch.cpp`, without touching a pipeline, a node or the runner.
// The spec: docs/specs/2026-11-12-chain-graph.md, "Node traits and the dispatch point"; the guard `cmake/check_dispatch.cmake` fails when a node's function is called anywhere else.
//
// What dispatch decides, and what it does not:
//   - it checks that the node declared its traits (registry.h `NodeTraits`) and refuses a node that did not;
//   - it refuses a `stateful` node in a block-parallel context (`DispatchContext::block_parallel`; no executor sets it yet);
//   - it picks the backend and runs the phase on it. It never changes a value: the CPU backend calls the node's own function.
#pragma once
#include <any>
#include <cstdint>

#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

/** where a node runs. Only the CPU exists. */
enum class Backend { cpu };

/** which of a node's functions a call runs */
enum class NodePhase {
  prepare,   // once, before the data: bind the node's constants (its state)
  start,     // an accumulate node: make its accumulator
  chunk,     // a chunk node: a block in, a block out (`chunk_at` when the node has it)
  fold,      // an accumulate node: fold a block into its accumulator
  finish,    // an accumulate node: finish the accumulator as arrays
  run        // a setup or after node: arrays in, arrays out
};

/** the context a call runs in */
struct DispatchContext {
  Backend backend = Backend::cpu;
  /** true when blocks may be run out of time order or in parallel: a `stateful` node is then refused. The runner folds blocks in time order, so it leaves this false. */
  bool block_parallel = false;
};

/** one call of a node: the phase, what it reads, and where its result goes. Build one with the `NodeCall::` functions; read the result from the member the phase fills. */
struct NodeCall {
  NodePhase phase = NodePhase::run;
  // reads
  const std::any* state = nullptr;            // the node's prepared state (every phase but `prepare`)
  const Values* planned_inputs = nullptr;     // prepare: the planned inputs
  const Json* parameters = nullptr;           // prepare: the node's parameters
  ExecContext* exec = nullptr;                // prepare, run
  const Blocks* blocks = nullptr;             // chunk, fold: the inputs inside the loop
  std::int64_t keep = 0;                      // chunk: the halo to leave on each side
  const Place* place = nullptr;               // chunk (a node with `chunk_at`), fold: where the own columns lie
  const Arrays* arrays = nullptr;             // run: the inputs outside the loop
  std::any* accumulator = nullptr;            // fold, finish
  // results
  std::any made_state;                        // prepare, start
  Blocks made_blocks;                         // chunk
  Arrays made_arrays;                         // finish, run

  static NodeCall prepare(const Values& inputs, const Json& parameters, ExecContext& exec);
  static NodeCall start(const std::any& state);
  static NodeCall chunk(const std::any& state, const Blocks& inputs, std::int64_t keep, const Place& place);
  static NodeCall fold(const std::any& state, std::any& accumulator, const Blocks& inputs, const Place& place);
  static NodeCall finish(const std::any& state, std::any& accumulator);
  static NodeCall run(const std::any& state, const Arrays& inputs, ExecContext& exec);
};

/** THE ONE CALL PATH: runs `call` on `node` on the context's backend, after the trait checks. Throws invalid_argument (a refusal with its reason) when the node declared no traits or is `stateful` in a block-parallel context. */
void dispatch(const StepSpec& node, NodeCall& call, const DispatchContext& context = {});

}  // namespace nxr::pipelines::graph
