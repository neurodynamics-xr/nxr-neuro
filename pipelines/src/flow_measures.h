// The measures the flow nodes write (sensor_flow, source_flow, connectome_flow, the sign-flip consistency) DECLARED by them (measures.h, "declared measures"): how
// each one's values merge when members are rolled up, and the form of what it is taken of. The registry is a vocabulary, not a gate: a name it knows (power,
// exact_potential, coexact_stream, phase_current) keeps its entry and is not declared here. A p value, a consistency, a phase or a map never sums (merge none);
// sums over chunks, bins and members do (merge sum); counts add.
#pragma once
#include <string>

#include "nxr/graph/types.h"

namespace nxr::pipelines::graph {

/** {merge, form, description} of a flow measure this node family writes, or null for a name it does not declare (a registered one) */
inline nxr::graph::Json flow_measure_declaration(const std::string& measure) {
  using J = nxr::graph::Json;
  struct D { const char* name; const char* merge; const char* form; const char* description; };
  static const D table[] = {
      {"flow_coefficient", "sum", "mode", "the Hodge-Dirac coefficient of the phase current on one mode: linear in the additive 1-form, summed over chunks and bins"},
      {"vertex_phase_current_x", "sum", "phase_current", "the vertex total phase current, x: a sum over chunks, bins and tiles (not the mean phase_current_x of a member's vertices)"},
      {"vertex_phase_current_y", "sum", "phase_current", "the vertex total phase current, y: a sum over chunks, bins and tiles"},
      {"vertex_phase_current_z", "sum", "phase_current", "the vertex total phase current, z: a sum over chunks, bins and tiles"},
      {"remainder_potential", "none", "vertex", "the exact potential of the part beyond the kept modes: a map, never summed"},
      {"remainder_stream", "none", "vertex", "the coexact stream of the part beyond the kept modes: a map, never summed"},
      {"consistency", "none", "none", "the sign-flip consistency R1 of an element's time cells: a ratio, never summed"},
      {"p", "none", "none", "a p value: never summed"},
      {"familywise_p", "none", "none", "a family-wise p value: never summed"},
      {"threshold", "none", "member", "the member's family-wise threshold of the consistency"},
      {"median_consistency", "none", "member", "the median consistency over the member's elements"},
      {"median_null_95", "none", "member", "the median over the member's elements of the null's 95th percentile"},
      {"median_p", "none", "member", "the median p over the member's elements"},
      {"neighbour_correlation", "none", "member", "the correlation of neighbouring elements' consistencies"},
      {"cells", "sum", "member", "the time cells tested: a count"},
      {"groups_flipped", "sum", "member", "the sign-flip groups: a count"},
      {"cross_spectrum_real", "sum", "outer", "the real part of a pair's cross-spectrum: products add over chunks and bins"},
      {"cross_spectrum_imaginary", "sum", "outer", "the imaginary part of a pair's cross-spectrum: products add over chunks and bins"},
      {"mean_edge_flow", "none", "none", "the mean flow on an edge over the time cells: a mean, never summed"},
      {"control", "none", "none", "the distance-matched control's mean edge flow: a mean, never summed"},
      {"phase", "none", "none", "an angle: never summed"},
      {"divergence", "sum", "none", "the network divergence of the edge flow at a tile: linear in the flow"},
      {"potential", "none", "vertex", "the graph potential of the edge flow: a map, never summed"},
      {"potential_coefficient", "sum", "mode", "the coefficient of the graph potential on one connectome mode: linear (results made before the graph manifold; the network now writes flow_coefficient on its stored modes)"},
      {"weighted_gradient_squares", "sum", "none", "the conductance-weighted squares of the potential's gradient: an energy"},
      {"weighted_flow_squares", "sum", "none", "the conductance-weighted squares of the edge flow: an energy"},
      {"weighted_cycle_squares", "sum", "none", "the conductance-weighted squares of the edge flow's cycle (harmonic) part on the graph: an energy"},
  };
  for (const D& d : table)
    if (measure == d.name) return J{{"merge", d.merge}, {"form", d.form}, {"description", d.description}};
  return J();
}

/** each entry {measure, ...} of a node's `describe.measures` given its declaration (merge, form, description), where it has one */
inline void declare_flow_measures(nxr::graph::Json& measures) {
  if (!measures.is_array()) return;
  for (auto& m : measures) {
    if (!m.is_object() || !m.contains("measure") || m.contains("merge")) continue;
    const nxr::graph::Json d = flow_measure_declaration(m["measure"].get<std::string>());
    if (d.is_object()) for (const auto& [k, v] : d.items()) m[k] = v;
  }
}

}  // namespace nxr::pipelines::graph
