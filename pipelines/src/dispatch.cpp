// nxr-pipelines — THE DISPATCH POINT (dispatch.h). The only file that calls a node's computation functions (prepare, start, chunk, chunk_at, fold, finish, run);
// `cmake/check_dispatch.cmake` fails the build's tests when any other pipelines file does.
#include "nxr/pipelines/dispatch.h"

#include <string>

namespace nxr::pipelines::graph {

NodeCall NodeCall::prepare(const Values& inputs, const Json& parameters, ExecContext& exec) {
  NodeCall c; c.phase = NodePhase::prepare; c.planned_inputs = &inputs; c.parameters = &parameters; c.exec = &exec; return c;
}
NodeCall NodeCall::start(const std::any& state) {
  NodeCall c; c.phase = NodePhase::start; c.state = &state; return c;
}
NodeCall NodeCall::chunk(const std::any& state, const Blocks& inputs, std::int64_t keep, const Place& place) {
  NodeCall c; c.phase = NodePhase::chunk; c.state = &state; c.blocks = &inputs; c.keep = keep; c.place = &place; return c;
}
NodeCall NodeCall::fold(const std::any& state, std::any& accumulator, const Blocks& inputs, const Place& place) {
  NodeCall c; c.phase = NodePhase::fold; c.state = &state; c.accumulator = &accumulator; c.blocks = &inputs; c.place = &place; return c;
}
NodeCall NodeCall::finish(const std::any& state, std::any& accumulator) {
  NodeCall c; c.phase = NodePhase::finish; c.state = &state; c.accumulator = &accumulator; return c;
}
NodeCall NodeCall::run(const std::any& state, const Arrays& inputs, ExecContext& exec) {
  NodeCall c; c.phase = NodePhase::run; c.state = &state; c.arrays = &inputs; c.exec = &exec; return c;
}

namespace {

// the CPU backend: the node's own function, called as the runner called it before the dispatch point existed (no value changes)
void run_on_cpu(const StepSpec& node, NodeCall& call) {
  switch (call.phase) {
    case NodePhase::prepare: call.made_state = node.prepare(*call.planned_inputs, *call.parameters, *call.exec); return;
    case NodePhase::start: call.made_state = node.start(*call.state); return;
    case NodePhase::chunk:
      call.made_blocks = node.chunk_at ? node.chunk_at(*call.state, *call.blocks, call.keep, *call.place) : node.chunk(*call.state, *call.blocks, call.keep);
      return;
    case NodePhase::fold: node.fold(*call.state, *call.accumulator, *call.blocks, *call.place); return;
    case NodePhase::finish: call.made_arrays = node.finish(*call.state, *call.accumulator); return;
    case NodePhase::run: call.made_arrays = node.run(*call.state, *call.arrays, *call.exec); return;
  }
}

}  // namespace

void dispatch(const StepSpec& node, NodeCall& call, const DispatchContext& context) {
  check_traits(node);
  if (context.block_parallel && node.traits.time == TimeBehaviour::stateful)
    refuse(node.name + ": a stateful node (" + node.traits.note + ") cannot run on blocks in parallel or out of time order");
  switch (context.backend) {
    case Backend::cpu: run_on_cpu(node, call); return;
  }
  refuse(node.name + ": no such backend");
}

}  // namespace nxr::pipelines::graph
