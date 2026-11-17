// nxr-io — THE INPUT FINDER: a pipeline definition's `inputs:` found in a subject's rows by the model's classes and links, never by path. A port of nxr's
// (nxr/pipelines/inputs.py: resolve, value_of, relation), the reference; the spec: docs/specs/2026-11-12-chain-graph.md §1-2.
//
//   field: <kind>                  a Field with that `kind`                    (manifold: <relation>: it lies on that manifold, or has it as a factor)
//   operator: <kind>               an Operator with that `kind`                (from: / to: <relation>: its source / target manifold)
//   selection: <type>              a Selection of that `type`                  (manifold: <relation>: the manifold it indexes)
//   measurement: <measure>         a stored measurement, found by its KEY as the Studio keys `selection_measurement`: (over: the selection, of: the field or statistic,
//                                  the measure, within: the selection it is restricted within); `of` / `over` / `within` name inputs declared before it, `where` filters the selection's columns
//   statistic: <method>            a stored Statistic with that `method` (histogram, welch, …), found by `where: {name: …}`; it lies on no manifold (no from / to / manifold)
//   where: {<column>: v, params.<a.b>: v}   the row's columns, or a path into its `params_json`; v may be a `$parameter`, and `{X}` in it is input X's
//                                  name (the format names a recording's own nodes after it: `{recording}_bad_segments`, SPEC §6).
//                                  A `$parameter` whose value is `none` (or null) filters nothing: the entry is skipped (an optional name, e.g. `where: {name: $recording}`, default `none` = every one)
//   each: true                     the ANCHOR: the pipeline runs once per match; every other input is found for each
//   required: false                absent is allowed (the steps that take it go without)
//
// RELATIONS name a manifold through an input declared before: `X.manifold.i` (a field's i-th factor, in axis order), `X.time` (its time line: the factor
// whose role is time, SPEC §6), `X.from` / `X.to` (an operator's source / target).
//
// io's part: rows found and described. It reads no array and decides nothing about what a run computes (the architecture policy: data loads).
#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"

namespace nxr::io::finder {

/** A THING IN A PIPELINE, as the model describes it: an input found in the dataset, or what a step makes (nxr's `Value`). */
struct Value {
  std::string cls;                         // field · operator · selection · measurement (facts: measure, over {selection, selection_id}, of_field_id | of_statistic_id)
  std::vector<std::string> manifolds;      // a field: the manifold ids it lies on, in axis order (one, or a product's factors); an operator: {from, to}; a selection: {its manifold}
  std::string value_type = "scalar";       // a field's value type (an operator's TARGET's): scalar · vector3 · hermitian3 · … (the schema's)
  bool complex = false;                    // ℂ values: a field stored as `complex`, or one a step made complex (the analytic signal, a complex vector current)
  std::optional<std::string> time;         // a field's factor that is its time line (the axis the chunk loop walks)
  double rate = 0, origin = 0;             // that line's samples per second and first sample time (s)
  bool chunked = false;                    // inside the chunk loop: the value exists a chunk at a time
  std::optional<Row> row;                  // the stored row, for an input
  std::optional<std::string> unit;
  Json facts = Json::object();             // what a step records about what it made, for the steps after it (e.g. {"form": "absq", "phase": false})
  /** THE TYPES' FACTS of the stored manifolds this value lies on, from their rows (type system T3): per id, its name, type (points · line · surface · …), vertex count, line role
   *  (time · frequency · …) and coordinate system {space, name, unit} when its geometry records one. Filled by value_of for every manifold in `manifolds`. */
  Json manifold_facts = Json::object();
};

/** a manifold's factors in axis order (itself, when it is not a product) */
std::vector<std::string> factors(const Database& db, const std::string& manifold_id);
/** the model's description of a stored row of class `cls` */
Value value_of(const Database& db, const std::string& cls, const Row& row);
/** the manifold id a relation names (`X.manifold.i` · `X.time` · `X.from` · `X.to`); `where` prefixes an error */
std::string relation(const std::string& ref, const std::map<std::string, Value>& found, const std::string& where);
/** the inputs for each run: one mapping per anchor match (one, with no anchor). `inputs` is the definition's section (in its order); `params` the
 *  pipeline's parameters. An input matching nothing is refused when required; one matching several (other than the anchor) is refused as ambiguous.
 *  Every refusal is an Error (invalid_argument) naming the input. */
std::vector<std::map<std::string, Value>> resolve(const Database& db, const std::string& subject_id, const Json& inputs, const Json& params);
/** the inputs of a run, for its provenance: each one's class, name, id and kind */
Json describe(const std::map<std::string, Value>& found);

}  // namespace nxr::io::finder
