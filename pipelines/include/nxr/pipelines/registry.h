// nxr-pipelines — THE PIPELINE STEPS' REGISTRY: what a pipeline definition can name in `steps:`, and the types its arrows are checked with. A port of
// nxr's (nxr/pipelines/registry.py), the reference; the spec: docs/specs/2026-11-12-chain-graph.md §3 and §8b (what C++ forces: a compile-time table).
//
// A PIPELINE STEP is one node of the graph (the paramount architecture, CONTRIBUTING.md). It declares, once:
//   name        what a definition calls it (`step: band_filter`); nxr's names and ports exactly, new steps added to both tools
//   kind        setup       once, before the chunk loop, on no recording data
//               chunk       on every chunk of the recording (with the halo it asks for); its outputs stay inside the loop. POINTWISE: per element, per sample
//               accumulate  a REDUCTION over a time partition (the cardinal rule: reductions belong to the Selection system; the measurement functions
//                           folded chunk by chunk); its outputs leave the loop
//               after       once, after the loop, on what the loop reduced (never on recording data)
//   inputs      its ports (`in:` maps them to the definition's things); `optional` the ports that may be absent
//   outputs     what it makes (`out:` names them)
//   parameters  its parameters and their defaults (`with:` sets them)
//   traits      its TIME BEHAVIOUR and NUMERIC PRECISION (`NodeTraits` below): declared once, read and checked by the executor, so a backend that runs blocks
//               in parallel or in single precision knows what each node allows (docs/specs/2026-11-12-chain-graph.md, "Node traits and the dispatch point")
//   matches     the Studio's, nxr's or Rheome's function it is held equal to
//   plan        the arrows checked on the MODEL DESCRIPTION of each input (`finder::Value`: a field on its manifolds, of a type, ℝ or ℂ, inside the loop or
//               not; an operator from one manifold to another; a selection on one), before any data is read: what it makes, or a refusal with its reason
// Its computation is a pure function of `compute/` or `meeg/` (the same function a test calls directly); the runner binds and runs it.
#pragma once
#include <any>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/finder.h"
#include "nxr/pipelines/bound.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using Value = nxr::io::finder::Value;
using Values = std::map<std::string, Value>;

constexpr const char* kKinds[] = {"setup", "chunk", "accumulate", "after"};

/** what a step's plan is handed beside its inputs: the facts the runner bound (a manifold's type, a selection's tiling), read only. A node never reaches a database. */
struct PlanContext {
  const Bound& bound;
};

/** A VALUE INSIDE THE LOOP: a chunk's columns, the halo still round them (nxr's `Block`). Row-major [rows x width], float32 as the Studio's and nxr's;
 *  a ℂ value carries its imaginary part. The own columns are [left, width − right). Rows are the value's elements, a vertex's components consecutive. */
struct Block {
  std::int64_t rows = 0, width = 0, left = 0, right = 0;
  std::vector<float> re, im;
  std::int64_t own() const { return width - left - right; }
  bool complex() const { return !im.empty(); }
};
/** A VALUE OUTSIDE THE LOOP: an array [rows x cols] (cols 1 for a map), what an accumulate step finishes with and an after step takes */
struct Array {
  std::int64_t rows = 0, cols = 1;
  std::vector<double> data;
  nxr::io::Json meta = nxr::io::Json::object();   // what the step records about it for the store and the steps after it (a measurement's tiles, a span set's thresholds)
  std::int64_t comps = 1;                      // values an element holds: 1, or 3 for a vector measurement (the phase current's x y z); data is then [rows x cols x comps], the components consecutive
};
/** what a step is given while it runs: what the runner bound (a surface, a stored statistic, a kernel's matrix: constants the runner reads, never a recording). It holds no subject, dataset or file. */
struct ExecContext {
  const Bound& bound;
  std::function<void(const nxr::io::Json&)> say;                          // the run's progress: what preparation did or fell back on, said, never silent
  /** constants bound ONCE for the whole run and shared by the steps that ask for the same key (the Studio's "uploaded once"): a bound map several steps use is built by the first one's preparation */
  std::map<std::string, std::any> shared;
};
/** the chunk an accumulate step folds: its own samples and their good mask (the chunk service's, from the bad spans; empty: every sample good) */
struct Place {
  std::int64_t first = 0, n = 0;
  const std::vector<std::uint8_t>* good = nullptr;    // [n], the own columns'
};
/** where a RETIMING step (the frame tiling) puts a chunk's own columns: its first frame, its count, and which frames are good (outside the bad samples and inside the named spans; empty: all) */
struct FramePlace {
  std::int64_t first = 0, n = 0;
  std::vector<std::uint8_t> good;
};
using Blocks = std::map<std::string, Block>;
using Arrays = std::map<std::string, Array>;

/** How a node depends on time. A node must DECLARE one (a node that does not is refused by the executor, and the registry test fails). */
enum class TimeBehaviour {
  undeclared,   // not yet declared: refused
  pointwise,    // each timestep (frame, tile or event) is computed on its own, from its own columns: any block order, any block split, gives the same values
  windowed,     // each timestep needs a halo or a window around it (an FFT or FIR filter, the analytic signal, a Welch segment, an event's margin): the rule is stated in `window_rule`
  stateful      // carries state forward in time (a recursive filter, a state machine over frames, an accumulation that depends on the order of the blocks): blocks must meet in time order
};
/** The precision a node computes in, and what it hands on. */
enum class Precision { undeclared, float32, float64 };
enum class OutputPrecision {
  undeclared,
  follows_input,   // the output has the precision of the node's input (the stored precision follows the input)
  float64          // the output is double whatever the input is (an accumulated statistic, a table of events)
};
/** What a node declares about itself, where it is registered. The executor reads it (`check_traits`) and the dispatch point records it; nothing here changes a result. */
struct NodeTraits {
  TimeBehaviour time = TimeBehaviour::undeclared;
  std::string window_rule;                              // windowed: the halo or window, in words and units ("half the taps' length each side"); empty otherwise
  Precision computes_in = Precision::undeclared;        // the precision its arithmetic runs in
  OutputPrecision output = OutputPrecision::undeclared;
  bool uncertain = false;                               // the classification is a judgement the maintainers should confirm (listed in the PR that added it)
  std::string note;                                     // why, in a sentence: what makes it this time behaviour; what is accumulated in double
  bool declared() const { return time != TimeBehaviour::undeclared && computes_in != Precision::undeclared && output != OutputPrecision::undeclared && (time != TimeBehaviour::windowed || !window_rule.empty()); }
};
/** a declaration, written compactly where a node is registered */
inline NodeTraits node_traits(TimeBehaviour time, Precision computes_in, OutputPrecision output, std::string note, std::string window_rule = {}, bool uncertain = false) {
  NodeTraits t;
  t.time = time; t.computes_in = computes_in; t.output = output; t.note = std::move(note); t.window_rule = std::move(window_rule); t.uncertain = uncertain;
  return t;
}
const char* to_string(TimeBehaviour t);
const char* to_string(Precision p);
const char* to_string(OutputPrecision p);

struct StepSpec {
  std::string name, kind;
  std::vector<std::string> inputs, optional, outputs;
  Json parameters = Json::object();            // name → default (null: the definition must set it)
  std::string matches, description;
  std::function<Values(const Values& ins, const Json& params, const PlanContext& ctx)> plan;
  // ── execution (the kind's methods; each calls the pure functions of compute/ and meeg/) ──
  /** once, before the data: what every chunk uses (taps, a kernel's matrix, the mesh gradient), bound as a constant */
  std::function<std::any(const Values& ins, const Json& params, ExecContext& ctx)> prepare;
  /** what the REQUEST resolves to given what is stored (reading only, before anything is computed or written): part of the run's cache key, so a result is a hit only when the same choice is made
   *  (the spatial band: eigenmodes when stored or computable, else the series). Null: nothing resolved. */
  std::function<Json(const Values& ins, const Json& params, const Bound& bound)> resolve;
  /** what preparation RESOLVED that the parameters leave open (a series degree chosen from a tolerance, a transform's error): reported from the prepared state; the runner puts it in the run's
   *  provenance (producer_json of Fields, spans and statistics; the recipe entry the measurement rows name), so every option of a result is on record (the method rule). Generic: no step is special. */
  std::function<Json(const std::any& state)> record;
  /** a chunk step: the samples it needs on EACH side of a chunk's own columns */
  std::function<std::int64_t(const std::any& state)> halo;
  /** a chunk step: its outputs on the own columns and `keep` samples of halo on each side (what the steps after it still need) */
  std::function<Blocks(const std::any& state, const Blocks& ins, std::int64_t keep)> chunk;
  /** a chunk step that needs the chunk's PLACE (the first own sample): used instead of `chunk` when set */
  std::function<Blocks(const std::any& state, const Blocks& ins, std::int64_t keep, const Place& at)> chunk_at;
  /** a RETIMING chunk step (the frame tiling): its outputs' own columns are not the chunk's samples but the frames that fall in them. Says where they are, so the steps after it fold
   *  frames (their Place is the frame's: first frame, count, good), and the runner runs them on the whole chunk's frames, never on sample sub-blocks. Its outputs carry `facts.axis = "frames"`. */
  std::function<FramePlace(const std::any& state, const Place& samples)> frame_place;
  /** an accumulate step: its accumulator, a chunk folded into it (the own columns), and what it finishes as */
  std::function<std::any(const std::any& state)> start;
  std::function<void(const std::any& state, std::any& acc, const Blocks& ins, const Place& place)> fold;
  std::function<Arrays(const std::any& state, std::any& acc)> finish;
  /** a setup or after step: once, on arrays */
  std::function<Arrays(const std::any& state, const Arrays& ins, ExecContext& ctx)> run;
  /** the node's declared time behaviour and precision (see NodeTraits); set where the node is registered */
  NodeTraits traits;
  /** true when the node computes in double on float32 input: the maintainers' rule (stored precision follows the input) makes such a node flagged. Blocks inside the loop are float32; arrays outside are double. */
  bool computes_double_on_float_input() const { return traits.computes_in == Precision::float64 && (kind == "chunk" || kind == "accumulate"); }
};

/** the names of the registered nodes whose traits are not fully declared (empty when every node declares them) */
std::vector<std::string> undeclared_traits();
/** the executor's check before it runs a node: refuses (invalid_argument) a node with undeclared traits, naming it */
void check_traits(const StepSpec& node);

/** every registered step, by name */
const std::vector<StepSpec>& registry();
/** a step by name; an unknown name is refused (invalid_argument) with the registered names */
const StepSpec& step(const std::string& name);
/** every step as a pipeline's author reads it: name, kind, ports, parameters, what it matches */
Json catalogue();
/** a plan's refusal: an Error (invalid_argument) with the message, raised before any data is read */
[[noreturn]] void refuse(const std::string& message);

}  // namespace nxr::pipelines::graph
