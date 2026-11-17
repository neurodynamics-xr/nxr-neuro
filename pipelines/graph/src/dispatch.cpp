// nxr-graph — THE DISPATCH POINT (dispatch.h). The only file under pipelines/graph/ that calls a node's computation functions; `cmake/check_dispatch.cmake` fails
// the build's tests when any other file there does.
#include "nxr/graph/dispatch.h"

namespace nxr::graph {

NodeCall NodeCall::prepare(const Types& types, const Arrays& constants, const Json& parameters, Shared& shared) {
  NodeCall c; c.phase = NodePhase::prepare; c.types = &types; c.arrays = &constants; c.parameters = &parameters; c.shared = &shared; return c;
}
NodeCall NodeCall::start(const std::any& prepared) { NodeCall c; c.phase = NodePhase::start; c.prepared = &prepared; return c; }
NodeCall NodeCall::chunk(const std::any& prepared, const Blocks& in, std::int64_t keep, const Place& place) {
  NodeCall c; c.phase = NodePhase::chunk; c.prepared = &prepared; c.blocks = &in; c.keep = keep; c.place = &place; return c;
}
NodeCall NodeCall::step(const std::any& prepared, std::any state, const Blocks& in, std::int64_t keep, const Place& place) {
  NodeCall c; c.phase = NodePhase::step; c.prepared = &prepared; c.state = std::move(state); c.blocks = &in; c.keep = keep; c.place = &place; return c;
}
NodeCall NodeCall::finish(const std::any& prepared, std::any state) {
  NodeCall c; c.phase = NodePhase::finish; c.prepared = &prepared; c.state = std::move(state); return c;
}
NodeCall NodeCall::run(const std::any& prepared, const Arrays& in) { NodeCall c; c.phase = NodePhase::run; c.prepared = &prepared; c.arrays = &in; return c; }
NodeCall NodeCall::frame_place(const std::any& prepared, const Place& samples) {
  NodeCall c; c.phase = NodePhase::frame_place; c.prepared = &prepared; c.place = &samples; return c;
}
NodeCall NodeCall::record(const std::any& prepared) { NodeCall c; c.phase = NodePhase::record; c.prepared = &prepared; return c; }

namespace {

void missing(const NodeSpec& node, const char* fn) { refuse(node.name + ": the node has no `" + fn + "` function for this call"); }

// the CPU backend: the node's own function, called as it is registered (no value changes)
void run_on_cpu(const NodeSpec& node, NodeCall& call) {
  switch (call.phase) {
    case NodePhase::prepare:
      if (!node.prepare) { call.made = std::any{}; return; }
      call.made = node.prepare(*call.types, *call.arrays, *call.parameters, *call.shared);
      return;
    case NodePhase::start:
      if (!node.start) missing(node, "start");
      call.made = node.start(*call.prepared);
      return;
    case NodePhase::chunk:
      if (!node.chunk) missing(node, "chunk");
      call.made_blocks = node.chunk(*call.prepared, *call.blocks, call.keep, *call.place);
      return;
    case NodePhase::step: {
      if (!node.step) missing(node, "step");
      auto r = node.step(*call.prepared, std::move(call.state), *call.blocks, call.keep, *call.place);
      call.made_blocks = std::move(r.first);
      call.made_state = std::move(r.second);
      return;
    }
    case NodePhase::finish:
      if (!node.finish) missing(node, "finish");
      call.made_arrays = node.finish(*call.prepared, std::move(call.state));
      return;
    case NodePhase::run:
      if (!node.run) missing(node, "run");
      call.made_arrays = node.run(*call.prepared, *call.arrays);
      return;
    case NodePhase::frame_place:
      if (!node.frame_place) missing(node, "frame_place");
      call.made_frames = node.frame_place(*call.prepared, *call.place);
      return;
    case NodePhase::record:
      call.made_json = node.record ? node.record(*call.prepared) : Json::object();
      return;
  }
}

}  // namespace

void dispatch(const NodeSpec& node, NodeCall& call, const DispatchContext& context) {
  check_traits(node);
  if (context.block_parallel && node.traits.time == TimeBehaviour::stateful)
    refuse(node.name + ": a stateful node (" + node.traits.note + ") cannot run on blocks in parallel or out of time order");
  switch (context.backend) {
    case Backend::cpu: run_on_cpu(node, call); return;
  }
  refuse(node.name + ": no such backend");
}

}  // namespace nxr::graph
