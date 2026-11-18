#pragma once
// nxr::select — REGIONS ON A GRAPH: the building block that finds connected sets of nodes of interest, the spatial equivalent of the time spans
// (nxr::select::crossings, keepSpans) with only the neighbourhood changed (spec docs/specs/2026-11-13-cortical-events.md, phase 1). PURE MATHS: values
// per node (a tile of a level, a vertex, a cell: the function never knows which), the pairs of nodes that are neighbours, the nodes' weights (their areas),
// numbers in; a label per node out. In time a node's neighbours are the previous and the next (a line graph); on the cortex two tiles are neighbours
// when a mesh edge joins them (nxr::select::tileAdjacency); a space × time graph adds each node's own copy in the next time tile. The function is the same.
//
//   detectRegions   the RULE (hysteresis, as `crossings`): the nodes with a value ≥ `exit` form CONNECTED COMPONENTS over the pairs whose two ends both are;
//                   a component is KEPT when it holds a node ≥ `enter` and minWeight ≤ Σ weight ≤ maxWeight. enter = exit: plain connected regions above
//                   one level. NaN (no value) is in no region and breaks a connection. NO MERGING of regions that do not touch:
//                   merging across small gaps is a named option of a pipeline and is not built here.
//   applySign       the SIGN is the caller's: a band-passed map has both signs, so the caller passes values already signed (positive: x; negative: −x, dips
//                   become bumps; magnitude: |x|). The thresholds are then on the signed values; detectRegions never knows.
// The regions are numbered by their lowest node, so the labelling is the same on every run. Σ weight is in the weights' units (m² for areas).
#include <Eigen/Dense>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace nxr::select {

enum class Sign { Positive, Negative, Magnitude };
Sign parseSign(const std::string& name);                 // "positive" | "negative" | "magnitude"; std::invalid_argument otherwise, naming the valid ones
const char* signName(Sign s);
/** positive: x; negative: −x; magnitude: |x| (NaN stays NaN). */
Eigen::VectorXd applySign(const Eigen::Ref<const Eigen::VectorXd>& x, Sign sign);

struct Region {
  std::vector<std::int32_t> nodes;            // ascending
  double weight = 0;                          // Σ node weights
  std::int32_t peak = -1;                     // the node of the largest value inside (the lowest on a tie)
  double value = 0;                           // that value
};
struct Regions {
  std::vector<std::int32_t> label;            // [nodes] the region's ordinal, -1 for a node in none
  std::vector<Region> regions;                // by ordinal
};

/** `values` [n] (NaN: no value), `pairs` the neighbours (a, b), either order, repeated pairs harmless, `weights` [n] ≥ 0 (finite). enter ≥ exit; minWeight ≤ maxWeight.
 *  std::invalid_argument for sizes that disagree, a node outside 0 … n−1, a negative or non-finite weight, an exit level above the entry level. */
Regions detectRegions(const Eigen::Ref<const Eigen::VectorXd>& values, const std::vector<std::pair<std::int32_t, std::int32_t>>& pairs, const Eigen::Ref<const Eigen::VectorXd>& weights,
                      double enter, double exit, double minWeight = 0, double maxWeight = std::numeric_limits<double>::infinity());

}  // namespace nxr::select
