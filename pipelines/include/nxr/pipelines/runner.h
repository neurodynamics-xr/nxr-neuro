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
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/graph/graph.h"
#include "nxr/io/chunk_loop.h"
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
  std::string preset;                          // the named parameter set the run was asked for (`--set NAME`): in the provenance only, never the cache key (a preset is only a named parameter set); empty: none
  std::int64_t align = 0;                      // the grid (samples) the span snaps to: the largest a node of the graph asks (Planned::align); 0: none
  std::shared_ptr<const nxr::graph::Checked> checked;   // the graph as the plan checked it (every value's type, the halo): what `compile` hands back

  std::vector<const Planned*> section(const std::string& sec) const;
  /** the plan as a person reads it (`--dry-run`) */
  std::string text(const nxr::io::Database& db) const;
};

/** the executor's form of an (expanded) definition: its steps as uses of nodes with `with` written out for the run's parameters `params`, the inputs named in `present`
 *  (an optional input absent from it is left unwired; a required one refused), its outputs */
nxr::graph::Graph graph_of(const Definition& d, const Json& params, const std::vector<std::string>& present);
/** every step's arrows checked on the inputs, in order; what each makes; the outputs' places: before any data is read */
/** what a node may know of a recording's channels (types, quality, units: channel_facts), as a function of the recording's name; given by a caller that holds the subject's store (a plan reads the database only) */
using ChannelFacts = std::function<Json(const std::string& recording)>;
Plan plan(const Definition& d, const Values& found, const Json& params, const nxr::io::Database& db, const std::string& subject_id, const ChannelFacts& channels = nullptr);
/** the parameter sets of a definition's `plans:` header: one when no list is given, else one per combination of the lists' elements (each plan's element set, its list neutralised) */
std::vector<Json> plan_parameter_sets(const Definition& d, const Json& base);
/** the parameters with every origin (a parameter declared `chart: origin`) resolved to a point {kind: start | clock | event, seconds, ...}: an event from the stored selection's member */
Json resolve_origins(const Definition& d, const Json& params, const nxr::io::Database& db, const std::string& subject_id);
/** the definition's parameters: its defaults, then the preset's values (`preset`, a name of the definition's `presets:`; an unknown one refused), then the caller's `--set NAME=VALUE`
 *  overrides (an unknown name refused). The caller's own value wins over the preset's. */
Json parameters_of(const Definition& d, const Json& set, const std::string& preset = std::string());
/** what a run reads and how it stores: the span (the chunk service's request; neither is an error: the caller states it) and whether another job's outputs are replaced */
struct RunOptions {
  std::optional<std::string> chunks;
  std::optional<double> start, stop;
  bool replace = false;
  std::string requested_by;                    // the program (`nxr-neuro 0.1 / C++17`) the run's job row names (docs/specs/2026-11-20-data-layer-queries.md, 3.8); empty: no job row (a run made inside another command, e.g. a surrogate's draws)
  std::string preset;                          // the definition's named parameter set to start from (`--set NAME`); empty: the defaults
  /** the width (samples) of the time SUB-BLOCKS a chunk's pointwise tail runs on: after the last chunk step that needs a halo (a band's taps), every step is pointwise in
   *  time, so it runs on the chunk's own columns a sub-block at a time and folds in time order. The result does not depend on it (as on the chunk size); only memory does.
   *  MEMORY held by the tail: about (the tail's rows, e.g. 3 x vertices of a free kernel) x block_samples x 8 bytes per complex value alive at once. 0: the whole chunk. */
  std::int64_t block_samples = 1024;
  /** the width (frames) of the BLOCKS OF FRAMES the steps after a frame tiling (`frames`) run on: they are pointwise in time, so each chunk's frames are cut into blocks, run through the tail and folded
   *  in frame order. The result does not depend on it (a frame's value is its own); only memory does. MEMORY held by the tail: (the tail's rows, e.g. the glued vertices x the levels x the components a
   *  vertex) x block_frames x 8 bytes per complex value alive at once, a few times over (the operator's product in double, the float copy). The steps BEFORE the tiling see the sensors only. 0: the whole chunk's frames. */
  std::int64_t block_frames = 256;
  /** the recording's projectors on load (the chunk service): active (the default: its status-1 projectors applied to every block) or none (the samples as recorded). What was applied
   *  is in the run's provenance (`run.projectors`) and, when the recording has active projectors, in its cache key. */
  nxr::io::Projectors projectors = nxr::io::Projectors::active;
};
/** The per-subject inputs of a GROUP run (a definition of `scope: group`), stacked over the member subjects: for each `per_subject` selection input, the members' measurements (the measures the run's `measures`
 *  parameter names, of the field the member's `per_subject` field input found, when there is one) as ONE array [subjects x measures x rows x members]; the ids of what each member's input found, in the caller's order. */
struct GroupInputs {
  std::vector<std::string> subjects;                              // the members' names, in the caller's order (the stack's first axis)
  std::map<std::string, nxr::graph::Array> stacked;               // per_subject input → its values stacked over the subjects (meta: subjects, measures, rows, members, shape)
  std::map<std::string, std::vector<std::string>> ids;            // per_subject input → each member's found id
};
/** one planned run executed and its outputs stored (or found in the cache): {pipeline, anchor, stored: [{output, path, id, on}], run, hit}. `group`: the stacked inputs of a group run (run_group), whose
 *  outputs `sub` (the group's default subject) holds; null: a run of one subject. */
Json execute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Plan& P, const RunOptions& opt, const std::function<void(const Json&)>& progress = {}, const GroupInputs* group = nullptr);

/** A member subject of a group run: its name and its open writers (a catalogue dataset: its package). */
struct GroupMember {
  std::string name;
  nxr::io::DatasetWriter* ds = nullptr;
  nxr::io::SubjectWriter* sub = nullptr;
};
/** A GROUP RUN (consolidation spec 5; replaces `pyramid reduce` and group.cpp's subject merge): `d` (scope group) with its `per_subject` inputs found in EACH member (exactly one match each, else refused naming
 *  the subject), its other inputs and every output in the group's default subject (`group_ds`, `group_sub`). The members are stated by the caller (no default: none is refused). The members' inputs must lie on the
 *  group tree's codes: the runner compares the grid each member's selection records (`template` or `grid_key` in its parameters) and refuses a mismatch before any data is read. The cache key lists every member's
 *  input id; the provenance lists each subject's measurement id. `dry_run` plans only. One result per plan (`plans:` lists give several). */
std::vector<Json> run_group(nxr::io::DatasetWriter& group_ds, nxr::io::SubjectWriter& group_sub, const std::vector<GroupMember>& members, const Definition& d, const Json& set, const RunOptions& opt,
                            bool dry_run = false, const std::function<void(const Json&)>& progress = {});
/** `d` on the subject: every run planned first (a refusal stops all before any data is read), then each executed; `dry_run` plans only */
std::vector<Json> run(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const Definition& d, const Json& set, const RunOptions& opt, bool dry_run = false,
                      const std::function<void(const Json&)>& progress = {});

/** every run of `d` on the subject planned (one per anchor match), none executed */
std::vector<Plan> plan_runs(const Definition& d, const nxr::io::Database& db, const std::string& subject_id, const Json& set, const std::string& preset = std::string(), const ChannelFacts& channels = nullptr);
/** the same on an open subject, its channel facts read from it: what `compile` and a caller outside the runner use (the database handle stays in the runner) */
std::vector<Plan> plan_runs(const Definition& d, nxr::io::SubjectWriter& sub, const Json& set, const std::string& preset = std::string());

}  // namespace nxr::pipelines::graph
