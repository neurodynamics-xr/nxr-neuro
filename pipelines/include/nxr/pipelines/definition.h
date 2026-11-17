// nxr-pipelines — A PIPELINE DEFINITION: one YAML file, nxr's format (nxr/pipelines/definition.py, the reference; docs/specs/2026-11-12-chain-graph.md §2):
// what the pipeline reads (inputs, found by the model: io/finder.h), its parameters, its steps (before · chunks · after), what it stores (outputs).
//
//   pipeline: band_power_source_map          # the header: name, version, scope (subject)
//   inputs:   {recording: {field: recording, each: true}, kernel: {operator: inverse kernel, from: recording.manifold.0}}
//   parameters: {band: {default: [8.0, 12.0], meaning: "the band's edges", unit: Hz}}   # long form; the short form `band: [8.0, 12.0]` is the default alone (undocumented)
//   steps:
//     before: [ … ]                          # setup steps
//     chunks: {over: recording, do: [ … ]}   # THE CHUNK LOOP: chunk and accumulate steps, on every chunk of `over` (read by io's chunk service)
//     after:  [ … ]                          # after the loop
//   outputs:  {band_power: {kind: band power, unit: "A²·m²"}}
//
// A step: {step: <registered name>, in: {<port>: <thing>}, with: {<parameter>: <value or $name>}, out: <name> | [<names>]}.
// `parse` checks the STRUCTURE (sections, keys, every name defined before it is used, every `$name` a parameter, each step in the section its kind
// allows); the arrows are checked against the subject's rows by the plan (runner.h).
//
// DEFINITION LANGUAGE ADDITIONS (docs/specs/2026-11-17-consolidation.md, sections 4 and 5):
//   kind: chain            header: a FIELD CHAIN, a definition that makes one kind of pointwise field (no reduction, no detection); `run` alone it stores the field, composed into another it only
//                          contributes its steps
//   scope: group           header: the inputs marked `per_subject: true` are found in EACH member subject of a group; the others, and every output, are the group's default subject's
//   plans: {band: $bands, spatial_scale: $spatial_scales}   header: one PLAN per combination of the lists' elements (the first outermost): each plan has `band` and `spatial_scale` as the element and the
//                          lists neutralised (null), so each plan is its own cache key and its own stored result (pairs of a band and a spatial scale are several plans of one run)
//   inputs: x: {field: any, where: {name: $field}, chain: $chain}   a stored field by name, OR the field chain `$chain` composed in (its steps inlined before the consumer's, nothing stored, its name,
//                          parameters (`chain_parameters`) and sha256 in the provenance and the cache key); exactly one of the two is given
//   inputs: o: {selection: any, where: {name: $over}, make: $make}  a stored selection by name, OR made in the run by a maker `$make: {node: select.*, in: {port: thing}, with: {...}, store: {type, on, name}}`
//                          (get-or-create when `store:` is given: found by its maker key, never a refusal)
//   `when: $name` (or `when: "!$name"`, or `when: {name: value}`) on a step or an output: it is part of the run only when the parameter is given (not null, false, empty or 0)
//   parameter long form `{default, meaning, unit, chart, choices}`: a QUANTITY's unit and the chart it is in (consolidation spec, section 3); a bare level index is refused (only `chart: index` names one)
//
// REPETITION (phase 2c): an input or output entry with `for: {band: $bands, level: $space_levels}` and a block `- {for: {band: $bands}, do: [steps]}` are written out once per combination of the
// lists' elements (the first variable outermost), `{band}` in a name and `$band` as a value standing for the element: the flat definition is what the runner plans (`expand`).
#pragma once
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;

/** one step as the definition uses it */
struct StepUse {
  std::string step;
  Json ins = Json::object();                   // port → thing
  Json with = Json::object();                  // parameter → value (or "$name")
  std::vector<std::string> outs;
  std::string section;                         // before · do · after
  Json when;                                   // `when:` as written (null: always); resolved by `expand`, never seen by the plan
};

struct Definition {
  std::string name, scope, description;
  std::string kind = "pipeline";               // "pipeline" or "chain" (a field chain: `chain:` inputs of other definitions compose it)
  std::vector<std::string> per_subject;        // scope group: the inputs found in each member subject
  Json plans = Json::object();                 // header `plans:` (variable → `$list`): one plan per combination (see above); empty: one plan
  Json chains = Json::array();                 // the chains composed in by `expand`: [{input, chain, parameters, sha256}] (provenance; the cache key holds it)
  Json made = Json::array();                   // the makers composed in by `expand`: [{input, node, with, stored}] (provenance; the cache key holds it)
  std::string draw;                            // a header key `draw: <parameter>`: the run's stored nodes carry that parameter's value in their names (a null's draw: `seed`), so every draw is a node of its own
  int version = 1;
  Json inputs = Json::object(), parameters = Json::object(), outputs = Json::object();
  Json presets = Json::object();               // name -> {parameter: value}: the named parameter sets `run NAME --set PRESET` chooses (spec definition-outputs, section 6)
  Json parameter_docs = Json::object();        // name -> {meaning, unit, choices}: what a parameter declares about itself in the long form (empty: the short form, undocumented); `run <pipeline> --help` is made from it
  std::vector<StepUse> before, do_, after;
  std::optional<std::string> over;             // the input the chunk loop walks
  std::string source, text;                    // the file it was read from, and its text (its sha256 goes into the provenance)
  Json templ;                                  // a definition that repeats over list parameters (`for:`): its YAML as written (null: it is flat); `expand` writes it out with a run's parameters

  /** the input the pipeline runs once per match of (`each: true`) */
  std::optional<std::string> anchor() const;
  /** before, then the loop's, then after */
  std::vector<StepUse> uses() const;
};

/** a definition from its parsed YAML (`doc`); every structural fault is refused (invalid_argument) naming where */
Definition parse(const Json& doc, const std::string& source = "", const std::string& text = "");
/** the quantity replacing a retired level-index parameter name (docs/specs/2026-11-17-consolidation.md 3.5), or "" when the name is not one */
std::string quantity_for_index_parameter(const std::string& name);
/** whether a `when:` is true for the run's parameters */
bool when_holds(const Json& when, const Json& params);
/** the definition written out for a run's parameters: every `for:` (an input, an output, a block of steps) repeated over its lists' elements, the loop variables substituted ({band} in a name, $band as a value); every `when:` evaluated (a step or an
 *  output whose condition fails is left out); every `chain:` input composed (the chain's inputs prefixed `<input>_`, its steps inlined before the consumer's, its output named as the input) and every
 *  `make:` input made by a before-step (exactly one of the stored name and the chain/maker is given: both, or neither, are refused with the reason);
 *  a flat definition is returned as it is. Checked as any definition (structure, names defined before use). */
Definition expand(const Definition& d, const Json& params);
/** a definition by path (`*.yaml`) or by name among nxr-neuro's own (pipelines/definitions/, compiled in) */
Definition load(const std::string& name_or_path);
/** the names of nxr-neuro's own definitions */
std::vector<std::string> available();

}  // namespace nxr::pipelines::graph
