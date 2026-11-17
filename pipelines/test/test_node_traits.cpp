// Node traits and the dispatch point (spec docs/specs/2026-11-12-chain-graph.md, "Node traits and the dispatch point").
//   - every registered node declares its time behaviour and numeric precision (a new node without a declaration fails here, naming it);
//   - a windowed node states its window rule;
//   - the dispatch point refuses a node with no declaration, and a stateful node in a block-parallel context, and otherwise runs the node's own function unchanged.
// The "dispatch is the only call path" half is the static guard cmake/check_dispatch.cmake (the test dispatch_is_the_only_call_path).
#include <cstdio>
#include <map>
#include <string>

#include "check.h"
#include "nxr/pipelines/dispatch.h"
#include "nxr/pipelines/registry.h"

namespace g = nxr::pipelines::graph;

namespace {
bool refused(const std::function<void()>& f, const std::string& part) {
  try { f(); } catch (const std::exception& e) { return std::string(e.what()).find(part) != std::string::npos; }
  return false;
}
}  // namespace

int main() {
  // 1. every registered node declares its traits
  const auto missing = g::undeclared_traits();
  for (const auto& n : missing) std::printf("  node without a declaration: %s\n", n.c_str());
  CHECK(missing.empty());
  CHECK(g::registry().size() >= 30);   // the registry was read, not empty

  std::map<std::string, int> by_time;
  int doubled = 0;
  for (const auto& node : g::registry()) {
    const g::NodeTraits& t = node.traits;
    CHECK(t.declared());
    CHECK(!t.uncertain);                                              // every classification has been settled (docs/reference.md, Node traits)
    CHECK(!t.note.empty());                                           // every classification says why
    if (t.time == g::TimeBehaviour::windowed) CHECK(!t.window_rule.empty());
    if (t.time != g::TimeBehaviour::windowed) CHECK(t.window_rule.empty());
    ++by_time[g::to_string(t.time)];
    doubled += node.computes_double_on_float_input();
    // precision follows the input: a node that computes in double on float32 input and gives a field-valued result declares `follows_input` and says it stores float32;
    // only statistics and tables of events (a sum, a count, a list) declare float64 output
    if (node.computes_double_on_float_input() && t.output == g::OutputPrecision::follows_input) {
      const bool says = t.note.find("float32") != std::string::npos;
      if (!says) std::printf("  node computing in double on float32 input whose note does not say it stores float32: %s\n", node.name.c_str());
      CHECK(says);
    }
  }
  std::printf("  nodes: %zu (pointwise %d, windowed %d, stateful %d); in double on float input: %d\n", g::registry().size(), by_time["pointwise"], by_time["windowed"], by_time["stateful"], doubled);

  // 2. the dispatch point checks the traits, then runs the node's own function
  g::ExecContext* no_context = nullptr;
  (void)no_context;
  int ran = 0;
  g::StepSpec node;
  node.name = "probe"; node.kind = "chunk";
  node.chunk = [&](const std::any&, const g::Blocks& ins, std::int64_t keep) { ++ran; g::Blocks out = ins; out["keep"] = g::Block{static_cast<std::int64_t>(keep), 0, 0, 0, {}, {}}; return out; };
  std::any state;
  const g::Blocks in{{"x", g::Block{2, 3, 0, 0, std::vector<float>(6, 1.0f), {}}}};
  const g::Place place{0, 3, nullptr};

  g::NodeCall undeclared_call = g::NodeCall::chunk(state, in, 4, place);
  CHECK(refused([&] { g::dispatch(node, undeclared_call); }, "declares no time behaviour"));   // no declaration: refused, not run
  CHECK_EQ(ran, 0);

  node.traits = g::node_traits(g::TimeBehaviour::pointwise, g::Precision::float32, g::OutputPrecision::follows_input, "a probe");
  g::NodeCall call = g::NodeCall::chunk(state, in, 4, place);
  g::dispatch(node, call);
  CHECK_EQ(ran, 1);
  CHECK_EQ(call.made_blocks.at("keep").rows, std::int64_t(4));                                // the node's own function ran, with the call's arguments
  CHECK_EQ(call.made_blocks.at("x").re.size(), std::size_t(6));

  node.traits = g::node_traits(g::TimeBehaviour::stateful, g::Precision::float32, g::OutputPrecision::follows_input, "a probe that carries state");
  g::DispatchContext parallel;
  parallel.block_parallel = true;
  g::NodeCall parallel_call = g::NodeCall::chunk(state, in, 4, place);
  CHECK(refused([&] { g::dispatch(node, parallel_call, parallel); }, "cannot run on blocks in parallel"));
  CHECK_EQ(ran, 1);
  g::NodeCall serial_call = g::NodeCall::chunk(state, in, 4, place);
  g::dispatch(node, serial_call);                                                             // the same node in time order is fine
  CHECK_EQ(ran, 2);

  node.traits = g::node_traits(g::TimeBehaviour::windowed, g::Precision::float32, g::OutputPrecision::follows_input, "a probe with a window");
  CHECK(!node.traits.declared());                                                             // windowed without a window rule is not a declaration
  return nxrtest::finish("test_node_traits");
}
