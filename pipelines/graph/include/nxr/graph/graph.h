// nxr-graph — A GRAPH OF NODES AND ITS CHECK. The spec: docs/specs/2026-11-17-compute-and-run.md, section 3.3.
//
// A `Graph` is the steps of a pipeline definition without the inputs' finding and the outputs' storing: named inputs, parameters, uses of registered nodes in
// three sections (`before` once, on arrays · `chunks` on every block · `after` once, on what the blocks reduced), named outputs. A graph is a pure function.
//
// `check` is the executor's whole plan, made on TYPES alone before any data flows (ARCHITECTURE.md, part 2): it refuses a bad arrow with the use, the port and the
// reason, and gives back what the executor needs, composed from the nodes' own declarations:
//   - the arrows: each node's `plan` on its inputs' types (value type, rows, manifolds, units); D106 (a linear map after a non-linear node is refused);
//   - the kinds by section (a chunk or accumulate node only in `chunks`; a `before` node takes no streamed value; an `after` node takes no block);
//   - the traits (every node declares them; a `stateful` node is refused where blocks may come out of order);
//   - the HALO, composed backwards over the chunk nodes: with h(n) the halo node n needs and keep(n) the halo it must leave for its consumers,
//       keep(n) = max over the chunk nodes c reading n of (h(c) + keep(c)),   0 for a node read only by accumulate nodes or outputs;
//       the loop's halo = max over the nodes reading a streamed input of (h(n) + keep(n));
//   - the retiming (the frame tiling): the nodes after it run on frames, and none of them may need a halo.
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"

namespace nxr::graph {

/** one use of a node: `in` maps the node's input ports to value names, `with` its parameters to a value or "$name" (a graph parameter), `out` names its outputs in the node's order */
struct Use {
  std::string node;
  std::map<std::string, std::string> in;
  Json with = Json::object();
  std::vector<std::string> out;
};

struct Graph {
  std::string name;
  int version = 1;
  std::string text;                                  // its canonical text; sha256(text) goes into the provenance (the runner's job)
  std::vector<std::string> inputs, outputs;          // value names
  Json parameters = Json::object();                  // name -> default, or {default, meaning, unit}
  std::vector<Use> before, chunks, after;
};
/** a graph from its JSON form {name, version, inputs: [..], outputs: [..], parameters: {..}, before|chunks|after: [{node, in: {..}, with: {..}, out: name | [names]}]}; faults refused naming where */
Graph graph_from_json(const Json& doc);

enum class Section { before, chunks, after };
const char* to_string(Section s);

/** one node as checked: where it sits, its resolved parameters, its inputs' types, what it makes, and the halo it needs and must keep */
struct CheckedNode {
  std::shared_ptr<const NodeSpec> spec;
  Section section = Section::chunks;
  std::string label;                                 // "chunks[2] welch.psd", for messages
  Json params;                                       // every parameter resolved (default, then `with`, $names replaced)
  std::map<std::string, std::string> in;             // input port -> value name
  std::map<std::string, std::string> out;            // output port -> value name
  Types in_types;                                    // the types of the inputs present, by port
  Planned planned;
  std::int64_t keep = 0;                             // the halo it leaves on each side for the chunk nodes after it
  bool uses_step() const { return spec->kind == Kind::accumulate || spec->traits.time == TimeBehaviour::stateful; }
};

/** a graph checked on types: immutable once returned, read by `Execution`. It holds copies of its nodes' specs, so it does not depend on the registry afterwards. */
struct Checked {
  std::string name;
  Json parameters;                                   // the graph's parameters, resolved
  std::vector<CheckedNode> nodes;                    // flat: before, then chunks, then after
  Types values;                                      // every value's type: the inputs and what the nodes make
  std::vector<std::string> inputs, outputs;
  std::int64_t halo = 0;                             // what each block handed to the executor must carry on each side
  std::int64_t align = 0;                            // the grid (samples) the run's span snaps to: the largest a node asks (Planned::align); 0: none
  std::vector<std::size_t> loop;                     // the indices (into `nodes`) of the chunks section, in order
  std::size_t tail = 0;                              // loop positions [0, tail) run on the whole block; the pointwise rest on sub-blocks (after the last node that needs a halo)
  std::optional<std::size_t> retime;                 // the loop position of the retiming node, if any: the nodes after it run on blocks of frames
  std::map<std::string, std::size_t> last_use;       // value -> the loop position of its last reader (a graph output is never released)
  bool has_state = false;                            // a stateful or accumulate node: blocks must meet in time order

  /** the plan as a person reads it (`--dry-run`): each use with its parameters and what it makes, the halo, the outputs */
  std::string text() const;
};

/** every arrow checked before any data: refuses (invalid_argument) with the use, the port and the reason.
 *  `inputs` types every graph input (streamed ones with axis samples); `params` sets the graph's parameters (unknown names refused). */
Checked check(const Graph& g, const Types& inputs, const Json& params = Json::object(), const Registry& registry = nxr::graph::registry(), const DispatchContext& ctx = {});

}  // namespace nxr::graph
