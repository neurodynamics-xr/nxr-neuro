// nxr-graph — A NODE: one pure function of matrices with typed ports, parameters with defaults, declared traits, and the reference it is held equal to.
// The spec: docs/specs/2026-11-17-compute-and-run.md, sections 2 and 3.2. A node's functions are called only through `dispatch` (dispatch.h).
//
//   kind   array        once, on whole arrays (a node in `before`, or in `after`)
//          chunk        on every block: pointwise or windowed (`chunk`), or stateful (`start` + `step`)
//          accumulate   `start`, `step` over the blocks in time order, `finish`: a reduction over a time partition (a measurement over a partition, never a
//                       bespoke accumulator: the cardinal rule of ARCHITECTURE.md)
//
// `NodeTraits` is the declaration every node makes where it is registered (ARCHITECTURE.md, "Backends and node traits"). pipelines/registry.h holds the same
// declaration for the runner's steps; the two are one when the runner is split (spec step C5).
#pragma once
#include <any>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/graph/signature.h"
#include "nxr/graph/types.h"

namespace nxr::graph {

/** How a node depends on time. A node must DECLARE one. */
enum class TimeBehaviour {
  undeclared,   // not yet declared: refused
  pointwise,    // each timestep is computed on its own: any block split or order gives the same values
  windowed,     // each timestep needs a halo or window around it: the rule is stated in `window_rule`
  stateful      // carries state forward in time: blocks must meet in time order
};
enum class Precision { undeclared, float32, float64 };
enum class OutputPrecision {
  undeclared,
  follows_input,   // the output has the precision of the node's input (stored precision follows the input)
  float64          // the output is double whatever the input is (an accumulated statistic, a table of events)
};
struct NodeTraits {
  TimeBehaviour time = TimeBehaviour::undeclared;
  std::string window_rule;                              // windowed: the halo or window, in words and units; empty otherwise
  Precision computes_in = Precision::undeclared;        // the precision its arithmetic runs in
  OutputPrecision output = OutputPrecision::undeclared;
  bool uncertain = false;                               // the classification is a judgement the maintainers should confirm
  std::string note;                                     // why, in a sentence
  bool declared() const { return time != TimeBehaviour::undeclared && computes_in != Precision::undeclared && output != OutputPrecision::undeclared && (time != TimeBehaviour::windowed || !window_rule.empty()); }
};
inline NodeTraits node_traits(TimeBehaviour time, Precision computes_in, OutputPrecision output, std::string note, std::string window_rule = {}, bool uncertain = false) {
  NodeTraits t;
  t.time = time; t.computes_in = computes_in; t.output = output; t.note = std::move(note); t.window_rule = std::move(window_rule); t.uncertain = uncertain;
  return t;
}
const char* to_string(TimeBehaviour t);
const char* to_string(Precision p);
const char* to_string(OutputPrecision p);

enum class Kind { array, chunk, accumulate };
const char* to_string(Kind k);

struct PortSpec { std::string name, meaning; bool optional = false; };
/** a parameter: its default (null: the caller must set it, unless `optional`), meaning and unit; the command line's help is made from these. `optional`: null is a value, "not given": the node decides what that means (a pipeline's optional bound) */
struct ParamSpec { std::string name; Json def; std::string meaning, unit; std::vector<std::string> choices; bool optional = false; };

/** what a node's plan gives back: the outputs' types (by output port name) and the halo it needs on each side (a chunk node; 0 otherwise) */
struct Planned {
  Types out;
  std::int64_t halo = 0;
  std::int64_t align = 0;   // samples: the grid the run's span must snap to (a power of two on the line's tiles; 0: none). `check` takes the largest over the graph (Checked::align)
};
/** node-held constants shared between nodes that ask for the same key (a bound operator built once: the Studio's "uploaded once") */
using Shared = std::map<std::string, std::any>;

struct NodeSpec {
  std::string name;                       // grouped: "signal.psd"
  Kind kind = Kind::chunk;
  std::vector<PortSpec> inputs, outputs;
  std::vector<ParamSpec> parameters;
  NodeTraits traits;
  /** the node's SIGNATURE (signature.h): its move and its ports' types with manifold variables. The checker runs it beside `plan` when every input is typed (types.h `typed`); a node
   *  without one is checked by its `plan` alone. `Registry::add` gives a node the built-in signature of its name (signatures.cpp) when the ports match. */
  std::optional<Signature> signature;
  bool linear_map = false;                // a linear map (a kernel, a form's operator): refused after a non-linear node (D106)
  std::string matches, description;       // the reference it is held equal to; one sentence

  /** PLAN: the arrows, on types only. `in` holds the types of the inputs that are present; `params` every parameter resolved. Refuses with refuse("...") naming the port and the reason. */
  /** the outputs a use with these parameters makes, a subset of `outputs` (a node whose parameter chooses a part, each part making its own: `select.events.spans` rule detect makes the spans, rule stages three
   *  more); null: every output. `out` of a use names exactly these, in the order of `outputs`. */
  std::function<std::vector<std::string>(const Json& params)> outputs_for;
  std::function<Planned(const Types& in, const Json& params)> plan;
  /** PREPARE (optional): once, before any block: the constants bound (taps designed, a matrix built), from the parameters and the inputs that are not streamed. Immutable for the run. */
  std::function<std::any(const Types& in, const Arrays& constants, const Json& params, Shared& shared)> prepare;
  /** chunk, pointwise or windowed: blocks in (the streamed inputs), blocks out, `keep` samples of halo left on each side for the nodes after it */
  std::function<Blocks(const std::any& prepared, const Blocks& in, std::int64_t keep, const Place& at)> chunk;
  /** stateful chunk and accumulate: (block, state) -> (output, state). State is a value, moved in and moved out; the executor owns it. An accumulate node returns no blocks. */
  std::function<std::any(const std::any& prepared)> start;
  std::function<std::pair<Blocks, std::any>(const std::any& prepared, std::any state, const Blocks& in, std::int64_t keep, const Place& at)> step;
  /** accumulate: the state's reduction as arrays */
  std::function<Arrays(const std::any& prepared, std::any state)> finish;
  /** array: arrays in, arrays out */
  std::function<Arrays(const std::any& prepared, const Arrays& in)> run;
  /** a RETIMING chunk node (the frame tiling): where a block's own columns fall as frames; its outputs carry axis `frames` */
  std::function<FramePlace(const std::any& prepared, const Place& samples)> frame_place;
  /** what preparation resolved that the parameters leave open: into the provenance */
  std::function<Json(const std::any& prepared)> record;
};

/** refuses (invalid_argument) a node whose traits are not fully declared, naming it */
void check_traits(const NodeSpec& node);

/** the nodes a graph may use, by name. A name is refused when empty or already taken. */
class Registry {
 public:
  void add(NodeSpec node);
  const NodeSpec* find(const std::string& name) const;
  /** a node by name; an unknown name is refused with the near names */
  const NodeSpec& at(const std::string& name) const;
  std::vector<std::string> names() const;
  /** the registered names nearest to `name` (the same group first, then the closest spelling) */
  std::vector<std::string> near(const std::string& name, std::size_t limit = 5) const;
  /** the nodes whose built-in signature was NOT attached because its ports are not the node's (a node's ports changed): empty when every signature fits */
  const std::vector<std::string>& unmatched_signatures() const { return unmatched_; }
 private:
  std::map<std::string, NodeSpec> nodes_;
  std::vector<std::string> unmatched_;
};
/** the built-in nodes. Empty until the runner's steps are registered as nodes (spec steps C5, C6). */
Registry& registry();

}  // namespace nxr::graph
