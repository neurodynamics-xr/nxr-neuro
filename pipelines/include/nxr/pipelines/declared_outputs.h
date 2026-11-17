// nxr-pipelines — DECLARED OUTPUTS (docs/specs/2026-11-17-definition-outputs.md, sections 2 and 3): what a definition's `outputs:` entry DECLARES (its kind, the found inputs it lies on or is of, the
// stored name and the names of the line and the partition it makes), bound to what the node DESCRIBED (`Array::meta["describe"]`: what the values are), and written by io in the run's one `task::Unit`.
//
//   field        {kind: field, kind_name, on: LINK}                                        a Field on the manifold the link names
//   statistic    {kind: statistic, of: INPUT, axes: [{name, on: LINK} | {name, line: NAME} | {name}], partition: {tiling: NAME, on: LINK}}
//   operator     {kind: operator, operator_kind, from: LINK, to: LINK}
//   selection    {kind: selection, type: partition | set | spans | tiling, on: LINK}
//   manifold     {kind: manifold, type: line | surface | product, name, alt, …}            made FIRST in the unit, so that another output can lie on it (`on: OUTPUT`)
//
// The existing short forms (`kind: spans`, `kind: labelling`, `kind: measurement`, a statistic without `axes:`, a Field's own kind name) are not declared outputs: the runner stores them as before.
// A LINK is `INPUT.manifold.N`, `INPUT.time`, `INPUT.from`, `INPUT.to` (io's finder) or the name of an output made in the same unit. A name is a template: `{input}` the found input's name, `$parameter` a
// parameter's value (the longest parameter name that fits).
#pragma once
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/writer.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/task.h"

namespace nxr::pipelines::graph {

/** whether an `outputs:` entry is one of the declared kinds above */
bool is_declared(const Json& spec);
/** a declared output checked against what its node's port type says and its links resolved, before any data is read: refuses (invalid_argument) naming the output */
void check_declared(const Plan& P, const std::string& output);
/** whether the plan's output is declared: the entry says so and its steps do not make a span set or a labelling (`{kind: selection, type: spans}` is the short form `{kind: spans}`, stored as before) */
bool is_declared(const Plan& P, const std::string& output);
/** the stored name an output takes (its template filled) */
std::string declared_name(const Plan& P, const std::string& output);
/** the places a declared output takes in the subject: {model class, store path}; a manifold takes none (found by name and reused, never replaced) */
std::vector<std::pair<task::Cls, std::string>> declared_places(const Plan& P, const std::string& output, nxr::io::SubjectWriter& sub);
/** a selection output made by a maker (`get_or_create: true`, a `make:` input stored): its maker key {maker, parameters, inputs' identities}; equal keys are the same selection, found and reused, never refused.
 *  A selection of `type: product` names its `factors` ({selection: LINK, level: N|$param, snapped}), with the level each grid factor is at (consolidation spec 6.2). A Field on the derived glued surface is stored
 *  RESTRICTED BACK to the original surface (PI rule 33), the bridge dropped, its producer saying so. A `measurement` output may state `value_type`, `components`, `of_statistic` and `level`; a `statistic` whose axes
 *  run over a selection's members is refused (it is a measurement). */
Json maker_key(const Plan& P, const std::string& output);
/** the ids of what a unit has made so far, by output name (manifolds first, so a later output can lie on them) */
using MadeIds = std::map<std::string, std::string>;
/** writes one declared output through the unit; `span` is the run's span record (null: none); returns the report entry {output, kind, name, path, id, ...} */
Json store_declared(task::Unit& unit, nxr::io::SubjectWriter& sub, const Plan& P, const std::string& output, const nxr::graph::Array& data, const Json& span, MadeIds& made, bool replace, bool& replaced);

}  // namespace nxr::pipelines::graph
