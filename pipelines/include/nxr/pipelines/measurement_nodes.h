// nxr-pipelines — THE MEASUREMENT NODES (docs/specs/2026-11-17-consolidation.md, section 2 and 4.1): the four operations on measurements, each ONE node over ONE compute
// implementation (compute/include/nxr/measurement/), and the one pointwise node that gives a field its form.
//
//   field.form     chunk       a field's FORM (the Studio's fieldForms.ts, complex-extended): `magnitude` |z|, `power` |z|², `phase_current` Im(z̄ ∇z), `outer` z zᴴ
//   reduce         accumulate  a field over the members of ANY selection (a time grid, a surface grid, a product of them, a span, a channel set): the additive ingredients, folded
//                              block by block (nxr::measurement::reduce); the measures are read from them (measures.h); the only node that removes a factor
//   reduce.members after       a measurement over a grid's members, over the members of another selection given on them (events over a grid): nxr::measurement::merge
//   roll_up        after       a measurement's ingredients through a GRID's closed-form hierarchy (a coarser level, `whole`), or across subjects: nxr::measurement::roll_up
//   summarize      after       a measurement's members to a statistic (mean, sd, median, mad, percentile, ...): nxr::measurement::summarize
//   estimate       accumulate  a statistic from a partition of the stream (a segment mean, a histogram): nxr::measurement::SegmentMean, Histogram
//
// Roll-ups exist in io's query AND here: each measure's merge rule is DATA in the measures registry, applied by io in SQL and by compute in memory.
// A retired step name (measure, tile_max, reduce_cells, measure_events, magnitude, power, phase_current, summarize_measurements) is refused with the name of what replaces it
// (`retired_step_message`, registry.cpp).
#pragma once
#include <string>
#include <utility>
#include <vector>

#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

/** appends the measurement nodes to the steps `build()` made, taking the steps they replace out of `r` (their bodies are the nodes' parts) */
void add_measurement_nodes(std::vector<StepSpec>& r);

/** ONE step made of several parts chosen by the value of a parameter (`param`): the plan, preparation, chunk, fold, finish and run of the chosen part are called, the part's state
 *  held beside its name. The parts are the same kind; the ports are the union (a port is optional unless every part requires it); the parameters the union, `param` first. `adapt`
 *  (optional) changes the parameters a part sees (a rule that sets another part's parameter). Used by the makers (select_nodes.cpp) and `field.form`. */
StepSpec combine_by_param(const std::string& name, const std::string& param, std::vector<std::pair<std::string, StepSpec>> parts, const std::string& description,
                          std::function<Json(const Json&, const std::string&)> adapt = {},
                          std::function<std::string(const Values&, const Json&)> choose = {});
/** `choose` (optional): the part is chosen from the inputs and the parameters together instead of by `param`'s value alone (`estimate`: a power baseline or a distribution of curl or of relative power is told by what x is) */

/** the step named `name` taken out of `r` (moved); refuses when it is not there */
StepSpec take_step(std::vector<StepSpec>& r, const std::string& name);

}  // namespace nxr::pipelines::graph
