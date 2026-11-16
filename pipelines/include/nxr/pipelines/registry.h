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
//   matches     the Studio's, nxr's or Rheome's function it is held equal to
//   plan        the arrows checked on the MODEL DESCRIPTION of each input (`finder::Value`: a field on its manifolds, of a type, ℝ or ℂ, inside the loop or
//               not; an operator from one manifold to another; a selection on one), before any data is read: what it makes, or a refusal with its reason
// Its computation is a pure function of `compute/` or `meeg/` (the same function a test calls directly); the runner binds and runs it.
#pragma once
#include <any>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/writer.h"
#include "nxr/io/finder.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using Value = nxr::io::finder::Value;
using Values = std::map<std::string, Value>;

constexpr const char* kKinds[] = {"setup", "chunk", "accumulate", "after"};

/** what a step's plan may look up beside its inputs: the subject's rows (a manifold's type, a selection's tiling), read only */
struct PlanContext {
  const nxr::io::Database& db;
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
/** what a step is given while it runs: the subject (its stored nodes read BY NAME: a kernel, a surface; never a recording, which only the chunk service reads) */
struct ExecContext {
  nxr::io::SubjectWriter& sub;
  nxr::io::DatasetWriter* ds = nullptr;                                   // for a step whose preparation stores a node it needs (the spatial band's eigenmodes, when none are stored)
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
  std::function<Json(const Values& ins, const Json& params, nxr::io::SubjectWriter& sub)> resolve;
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
};

/** every registered step, by name */
const std::vector<StepSpec>& registry();
/** a step by name; an unknown name is refused (invalid_argument) with the registered names */
const StepSpec& step(const std::string& name);
/** every step as a pipeline's author reads it: name, kind, ports, parameters, what it matches */
Json catalogue();
/** a plan's refusal: an Error (invalid_argument) with the message, raised before any data is read */
[[noreturn]] void refuse(const std::string& message);

}  // namespace nxr::pipelines::graph
