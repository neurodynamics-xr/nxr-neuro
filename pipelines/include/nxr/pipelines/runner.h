// nxr-pipelines — THE RUNNER: one pipeline on one subject (`nxr-neuro run DEFINITION --dataset D --subject S`). A port of nxr's (nxr/pipelines/runner.py),
// the reference; the spec: docs/specs/2026-11-12-chain-graph.md §4. Generic: no analysis-specific code.
//
//   plan     the inputs found by the model's links (io/finder.h), the definition written as a graph of nodes and CHECKED ON TYPES by the executor (nxr_graph: arrows, kinds by section,
//            the halo composed, retiming); every output's manifold known: nothing read yet. `--dry-run` stops here and prints it.
//   run      the executor does every step (before · each chunk through the chunk and accumulate steps, state carried · after); the runner only streams: io's chunk service (`haloed_loop`)
//            hands each chunk of the span (always stated by the caller) with the halo the checked graph declares. A spatial band whose eigenmodes are not stored: the runner makes them
//            FIRST, as their own `eigenmodes` run (own provenance and cache key), says so, and starts again; nothing is stored hidden inside the run.
//   store    ONE task::Unit for the run (nodes, arrays, measurements, recipe, cache key together, or nothing): each output a Field of the subject, on the manifold its steps made it on, named `<pipeline>.<output>.<anchor>` in that manifold's folder
//            (D157), its producer_json the pipeline (name, version, the definition's sha256), the parameters, the inputs' ids, the span and the run's
//            facts; the CACHE KEY is the same (definition, parameters, inputs, span), so a rerun is a hit; another key's node is refused unless `replace`
//            (then moved aside, never deleted).
#pragma once
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/graph/graph.h"
#include "nxr/io/database.h"
#include "nxr/io/writer.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

struct Planned {
  StepUse use;
  const StepSpec* step = nullptr;
  Json params;                                 // the step's parameters: its defaults, then `with:` ($names resolved)
  std::map<std::string, std::string> ins;      // port → thing
  std::map<std::string, std::string> outs;     // the step's output → the definition's name
};

struct Plan {
  Definition definition;
  Json parameters;
  Values found;                                // the inputs found
  Values values;                               // every thing: the inputs and what the steps make
  std::vector<Planned> steps;
  nxr::graph::Graph graph;                     // the executor's form of the definition (its steps as uses of nodes, `with` written out); `execute` checks it again against the run's binding
  nxr::graph::Types types;                     // its inputs' types, from what was found

  std::vector<const Planned*> section(const std::string& sec) const;
  /** the plan as a person reads it (`--dry-run`) */
  std::string text(const nxr::io::Database& db) const;
};

/** every step's arrows checked on the inputs, in order; what each makes; the outputs' places: before any data is read */
Plan plan(const Definition& d, const Values& found, const Json& params, const nxr::io::Database& db, const std::string& subject_id);
/** the definition's parameters with `--set` overrides (an unknown name refused) */
Json parameters_of(const Definition& d, const Json& set);
/** what a run reads and how it stores: the span (the chunk service's request; neither is an error: the caller states it) and whether another job's outputs are replaced */
struct RunOptions {
  std::optional<std::string> chunks;
  std::optional<double> start, stop;
  bool replace = false;
  /** the width (samples) of the time SUB-BLOCKS a chunk's pointwise tail runs on: after the last chunk step that needs a halo (a band's taps), every step is pointwise in
   *  time, so it runs on the chunk's own columns a sub-block at a time and folds in time order. The result does not depend on it (as on the chunk size); only memory does.
   *  MEMORY held by the tail: about (the tail's rows, e.g. 3 x vertices of a free kernel) x block_samples x 8 bytes per complex value alive at once. 0: the whole chunk. */
  std::int64_t block_samples = 1024;
  /** the width (frames) of the BLOCKS OF FRAMES the steps after a frame tiling (`frames`) run on: they are pointwise in time, so each chunk's frames are cut into blocks, run through the tail and folded
   *  in frame order. The result does not depend on it (a frame's value is its own); only memory does. MEMORY held by the tail: (the tail's rows, e.g. the glued vertices x the levels x the components a
   *  vertex) x block_frames x 8 bytes per complex value alive at once, a few times over (the operator's product in double, the float copy). The steps BEFORE the tiling see the sensors only. 0: the whole chunk's frames. */
  std::int64_t block_frames = 256;
};
/** one planned run executed and its outputs stored (or found in the cache): {pipeline, anchor, stored: [{output, path, id, on}], run, hit} */
Json execute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const RunOptions& opt, const std::function<void(const Json&)>& progress = {});
/** `d` on the subject: every run planned first (a refusal stops all before any data is read), then each executed; `dry_run` plans only */
std::vector<Json> run(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Definition& d, const Json& set, const RunOptions& opt, bool dry_run = false,
                      const std::function<void(const Json&)>& progress = {});

/** every run of `d` on the subject planned (one per anchor match), none executed */
std::vector<Plan> plan_runs(const Definition& d, const nxr::io::Database& db, const std::string& subject_id, const Json& set);

}  // namespace nxr::pipelines::graph
