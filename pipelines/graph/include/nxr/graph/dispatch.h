// nxr-graph — THE DISPATCH POINT: the one place a graph node's computation is called. The executor never calls a node's functions itself; it describes the
// call (`NodeCall`: the phase, the node's prepared state, its inputs) and hands it to `dispatch`, which checks the node's declared traits and runs the phase on the
// context's backend. Today the only backend is the CPU; a GPU backend is added in `dispatch.cpp` without touching a node or the executor.
// The guard `cmake/check_dispatch.cmake` fails when a node's function is called anywhere else under pipelines/graph/.
// (pipelines/dispatch.h is the runner's own, for its steps; the two are one when the runner is split, spec step C5.)
#pragma once
#include <any>
#include <cstdint>

#include "nxr/graph/node.h"

namespace nxr::graph {

enum class Backend { cpu };

enum class NodePhase {
  prepare,       // once, before the data: bind the node's constants
  start,         // a stateful or accumulate node: make its state
  chunk,         // a pointwise or windowed node: blocks in, blocks out
  step,          // a stateful or accumulate node: (block, state) -> (output, state)
  finish,        // an accumulate node: the state as arrays
  run,           // an array node: arrays in, arrays out
  frame_place,   // a retiming node: where a block's columns fall as frames
  record         // what preparation resolved
};

struct DispatchContext {
  Backend backend = Backend::cpu;
  /** true when blocks may be run out of time order or in parallel: a `stateful` node is then refused. The executor hands blocks in time order and leaves this false. */
  bool block_parallel = false;
};

/** one call of a node: the phase, what it reads, and where its result goes. Build one with the `NodeCall::` functions; read the result from the member the phase fills. */
struct NodeCall {
  NodePhase phase = NodePhase::run;
  const Types* types = nullptr;               // prepare
  const Arrays* arrays = nullptr;             // prepare: the constants; run: the inputs
  const Json* parameters = nullptr;           // prepare
  Shared* shared = nullptr;                   // prepare
  const std::any* prepared = nullptr;         // every phase but prepare
  const Blocks* blocks = nullptr;             // chunk, step
  std::int64_t keep = 0;                      // chunk, step
  const Place* place = nullptr;               // chunk, step, frame_place
  std::any state;                             // step, finish: the state moved in
  // results
  std::any made;                              // prepare, start: the prepared value or the state
  Blocks made_blocks;                         // chunk, step
  Arrays made_arrays;                         // finish, run
  std::any made_state;                        // step: the state moved out
  FramePlace made_frames;                     // frame_place
  Json made_json;                             // record

  static NodeCall prepare(const Types& types, const Arrays& constants, const Json& parameters, Shared& shared);
  static NodeCall start(const std::any& prepared);
  static NodeCall chunk(const std::any& prepared, const Blocks& in, std::int64_t keep, const Place& place);
  static NodeCall step(const std::any& prepared, std::any state, const Blocks& in, std::int64_t keep, const Place& place);
  static NodeCall finish(const std::any& prepared, std::any state);
  static NodeCall run(const std::any& prepared, const Arrays& in);
  static NodeCall frame_place(const std::any& prepared, const Place& samples);
  static NodeCall record(const std::any& prepared);
};

/** THE ONE CALL PATH: runs `call` on `node` on the context's backend, after the trait checks. Refuses (invalid_argument) a node that declared no traits, a `stateful`
 *  node in a block-parallel context, and a phase the node has no function for. */
void dispatch(const NodeSpec& node, NodeCall& call, const DispatchContext& context = {});

}  // namespace nxr::graph
