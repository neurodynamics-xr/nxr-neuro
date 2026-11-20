// nxr-graph — THE TYPES AS DATA: the structured JSON form of a type (both directions), node signatures given as JSON (external nodes, compile only), and a checked
// graph as JSON (the COMPILED GRAPH). The JSON form is for programs that build on nxr-neuro's types (a renderer, a notebook); the printed form (types.h `print`,
// `parse_type`) stays the one people read, and the two say the same thing: `print(port_type_from_json(to_json(t))) == print(t)` (test_graph_type_json).
//
//   a type      {"class": "field", "value": "complex", "complex_parts": false, "components": 1,
//                "domain": {"factors": [{"form": "id", "key": "rest", "label": "sensors", "kind": "points", "coordinate_system": {"space": "head", "name": "scs", "unit": "m"}},
//                                       {"form": "id", "key": "rest", "label": "time", "kind": "line", "topology": "path", "unit": "s", ...}], "cells": "0"},
//                "axis": "samples", "streamed": true, "sfreq": 100, "rows": 3, "quantity": "signal", "unit": "T", "resolution": {"band": [8, 12], "band_shape": "morlet"},
//                "nonlinear": false, "spin": 0, "typed": true, "text": "field<sensors(rest) × time(rest); complex; signal; band 8-12 Hz morlet; unit T>"}
//               A string is read as the printed form; a quantity is checked against the vocabulary where it is used (the checker), as for the printed form. `facts` (a node's own notes for its preparation) are never part of it.
//   a manifold  {"form": id | anonymous | variable | made | restricted | members | derived, "key", "label", "kind", "topology": path | cycle, "coordinate_system", "n", "spacing", "origin",
//                "rate", "unit", "samples", "made_from": {"relation": tiled | transform, "of": <line>}, "of": <manifold>, "by", "irregular", "rest", "subsume", "text"}; a string is read as the printed form
//                ("sensors(rest)", "M", "members(cells)"). A LINE has no role: it is a 1-D mesh with a topology, a coordinate (cs, origin, spacing, unit) and, when made from another line, `made_from`.
//                A VARIABLE's `unit` ("s": a line in seconds), `samples` (true: not tiled from another line) and `topology` constrain what it binds; a signature that sends `role` is refused.
//
// EXTERNAL NODES: a client (a renderer) declares nodes nxr-neuro does not implement, by their signatures, in the same terms as the registered nodes' (signature.h):
//   {"nodes": [{"name": "render.surface", "kind": "array", "move": "pointwise", "description": "...",
//               "inputs":  [{"port": "field", "class": "field", "on": [{"form": "variable", "key": "M", "kind": "surface"}], "value": ["real"], "components": 1},
//                           {"port": "surface", "class": "manifold", "on": ["M"]}],
//               "outputs": [],
//               "parameters": [{"name": "colormap", "default": "viridis", "meaning": "...", "unit": "-", "choices": ["viridis", "magma"]}]}]}
// An input pattern: port; class (field | measurement | selection | statistic | operator | manifold; absent: any); `on` (the domain's factors, manifold patterns; an operator's
// source) and `to` (an operator's target); value (one or a list of value types); streamed, linear, optional, over (booleans); quantity (a list); hint; components (a whole number
// the value's components must be). A variable's `kind` binds only a manifold of that kind. An output pattern: port; like (an input port); class; on; to; value; quantity; nonlinear.
// The move is one of signature.h's; kind is array (default) | chunk | accumulate (an accumulate node folds the stream); time (pointwise, default | windowed | stateful) with
// window_rule. Such a node is checked as any other, and refuses to run: it has no implementation here.
#pragma once
#include <functional>
#include <string>
#include <vector>

#include "nxr/graph/graph.h"
#include "nxr/graph/signature.h"
#include "nxr/graph/types.h"

namespace nxr::graph {

Json to_json(const ManifoldRef& m);
ManifoldRef manifold_from_json(const Json& j);
Json to_json(const Domain& d);
Domain domain_from_json(const Json& j);
Json to_json(const Resolution& r);
Resolution resolution_from_json(const Json& j);
/** the structured form of a type (the printed form under "text"); `facts` left out */
Json to_json(const PortType& t);
/** a type from its structured form, or from its printed form (a string); a member that is not given keeps PortType's default; a malformed one is refused naming it */
PortType port_type_from_json(const Json& j);

/** one node declared by its signature (the form above): a NodeSpec whose signature is the declared one, whose traits are declared, and whose functions refuse to run */
NodeSpec external_node(const Json& j);
/** the nodes of a document {"nodes": [...]} or a list of nodes; a name given twice is refused */
std::vector<NodeSpec> external_nodes(const Json& doc);

/** THE COMPILED GRAPH of a checked graph: {name, parameters, inputs: [{name, type}], nodes: [{index, label, section, node, kind, time, parameters, in, out, halo, keep}],
 *  edges: [{from: {input} | {node, port}, value, to: {node, port}}], values: {name: type}, outputs: [{name, type}], halo, align, retime}. Types in the structured form. */
Json compiled_json(const Checked& c);
/** the same for a graph typed by signatures alone (`type_check`): no plan ran, so `halo` and `align` are null and each node's `halo` is absent */
Json compiled_json(const Graph& g, const TypedGraph& typed, const Json& params,
                   const std::function<const Signature*(const std::string&)>& signatures = builtin_signature);

}  // namespace nxr::graph
