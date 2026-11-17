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

#include "nxr/graph/node.h"
#include "nxr/graph/types.h"
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

/** The values and the node's declarations are the executor's (nxr_graph, pipelines/graph/): one type, no copies between the runner's steps and the executor.
 *  Block: a value INSIDE THE LOOP (a chunk's columns, the halo still round them). Array: a value OUTSIDE it (`meta`: what a step records about it for the steps after it and the store).
 *  Place: where a block's own columns lie. FramePlace: where a RETIMING step (the frame tiling) puts them. */
using nxr::graph::Array;
using nxr::graph::Arrays;
using nxr::graph::Block;
using nxr::graph::Blocks;
using nxr::graph::FramePlace;
using nxr::graph::Place;
using nxr::graph::NodeTraits;
using nxr::graph::OutputPrecision;
using nxr::graph::Precision;
using nxr::graph::TimeBehaviour;
using nxr::graph::node_traits;
using nxr::graph::to_string;
/** what a step is given while it runs: what the runner bound (a surface, a stored statistic, a kernel's matrix: constants the runner reads, never a recording). It holds no subject, dataset or file. */
struct ExecContext {
  const Bound& bound;
  std::function<void(const nxr::io::Json&)> say;                          // the run's progress: what preparation did or fell back on, said, never silent
  /** constants bound ONCE for the whole run and shared by the steps that ask for the same key (the Studio's "uploaded once"): a bound map several steps use is built by the first one's preparation */
  std::map<std::string, std::any> shared;
  /** the arrays an EARLIER NODE of the run made that this step reads (by its port: a `before` step's operator, e.g. a pre-operator composed into a kernel), never a found input (those are read through
   *  `bound`). Null when there are none. */
  const nxr::graph::Arrays* made = nullptr;
};
struct StepSpec {
  std::string name, kind;                      // kind: setup · chunk · accumulate · after, or `array` (a native node that runs once on arrays: in `before` or in `after`)
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
  /** a chunk step whose halo is known from its request alone: the plan asks this instead of preparing the step (a step whose preparation reads stored arrays). Null: the plan prepares it on the records and asks `halo`. */
  std::function<std::int64_t(const Values& ins, const Json& params)> halo_at_plan;
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
  bool reads_arrays = false;                   // a node that takes found inputs as arrays (a fiber bundle as its points, a surface as its vertices and faces, a partition as its member numbers): the runner reads them for it
  bool native = false;
  /** true for a chunk step that reads only the SHAPE of its streamed input (its rows, rate, span and good samples), never a sample: the one kind of step that may take a recording declared
   *  `read: shape` in a definition (spec docs/specs/2026-11-17-surrogates.md, section 6). Any other step handed such a value is refused at plan. */
  bool reads_shape_only = false;                         // a node of `nxr::graph::registry()` (grouped names, pure functions of compute/): this entry only describes it to the definition reader; the executor runs the node itself
  /** true when the node computes in double on float32 input: the maintainers' rule (stored precision follows the input) makes such a node flagged. Blocks inside the loop are float32; arrays outside are double. */
  bool computes_double_on_float_input() const { return traits.computes_in == Precision::float64 && (kind == "chunk" || kind == "accumulate"); }
};

/** the verbs as registered steps (verb_steps.cpp), appended to the registry */
void append_verb_steps(std::vector<StepSpec>& r);
/** the measurement verbs as registered steps (steps_measure.cpp): measures over the windows of a time tiling, over a partition's members, and scale selection */
void append_measure_steps(std::vector<StepSpec>& r);
void append_geometry_steps(std::vector<StepSpec>& r);   // steps_geometry.cpp
/** run_repeat, cellquadrature and patches as registered steps (steps_patches.cpp) */
void append_patches_steps(std::vector<StepSpec>& r);

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
