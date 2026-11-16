// nxr-pipelines — A PIPELINE DEFINITION: one YAML file, nxr's format (nxr/pipelines/definition.py, the reference; docs/specs/2026-11-12-chain-graph.md §2):
// what the pipeline reads (inputs, found by the model: io/finder.h), its parameters, its steps (before · chunks · after), what it stores (outputs).
//
//   pipeline: band_power_source_map          # the header: name, version, scope (subject)
//   inputs:   {recording: {field: recording, each: true}, kernel: {operator: inverse kernel, from: recording.manifold.0}}
//   parameters: {band: [8.0, 12.0]}
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
};

struct Definition {
  std::string name, scope, description;
  std::string draw;                            // a header key `draw: <parameter>`: the run's stored nodes carry that parameter's value in their names (a null's draw: `seed`), so every draw is a node of its own
  int version = 1;
  Json inputs = Json::object(), parameters = Json::object(), outputs = Json::object();
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
/** the definition written out for a run's parameters: every `for:` (an input, an output, a block of steps) repeated over its lists' elements, the loop variables substituted ({band} in a name, $band as a value);
 *  a flat definition is returned as it is. Checked as any definition (structure, names defined before use). */
Definition expand(const Definition& d, const Json& params);
/** a definition by path (`*.yaml`) or by name among nxr-neuro's own (pipelines/definitions/, compiled in) */
Definition load(const std::string& name_or_path);
/** the names of nxr-neuro's own definitions */
std::vector<std::string> available();

}  // namespace nxr::pipelines::graph
