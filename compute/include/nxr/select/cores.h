#pragma once
// nxr::cores — VORTEX CORES of a hatted curl map (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 5; family B, a spatial feature per frame). PURE MATHS on matrices: a map of values per vertex and frame,
// the vertices' neighbours, areas and positions, a threshold: numbers in, the cores out. The function never knows where the map came from (a recording's hatted curl, a synthetic field), which frames are asked for,
// or what unit the positions are in. A port of nxr's `_core_patches` (nxr/detect/vortex.py), the reference:
//
//   a core         a CONNECTED PATCH of one sense: in each frame, the vertices with sense·value ≥ threshold (sense +1, then −1: counter-clockwise then clockwise about the normal, the curl's sign), joined by the
//                  mesh edges whose two ends are both above. NO merging of patches that do not touch.
//   order          by frame, then sense (+1 first), then the patch's lowest vertex (nxr: the order of scipy's connected_components over the vertices in (frame, vertex) order).
//   centre         the |value|·area-WEIGHTED centroid of the patch's vertices, in the positions' unit (nxr: mm), as `_core_patches`.
//   area           Σ area of the patch's vertices (the areas' unit).
//   centre_vertex  the patch's vertex nearest its centre (the lowest on a tie): where the core sits on the mesh.      [nxr-neuro's own attribute]
//   peak           the largest sense·value in the patch: its strength (> 0).                                          [nxr-neuro's own attribute]
// The threshold is a percentile of |value| over the burst frames (nxr::histogram: `percentile` of the counts folded over the frames, a statistic), the caller's: a core is cut from a threshold it is given, never
// one it works out. Frames are independent: a map cut into blocks of frames, each given its first frame's number, gives the cores of the whole map (the same list, block by block).
#include <Eigen/Dense>
#include <cstdint>
#include <utility>
#include <vector>

namespace nxr::select {

struct Cores {
  std::vector<std::int32_t> frame;           // the frame of each core (the block's first frame added)
  std::vector<std::int8_t> sense;            // +1 / −1
  std::vector<std::int32_t> centre_vertex;
  Eigen::MatrixXd centre;                    // [cores × 3] in the positions' unit
  std::vector<double> area, peak;
  std::int64_t size() const { return static_cast<std::int64_t>(frame.size()); }
};

/** `values` [vertices × frames] (the hatted curl, real), `pairs` the vertices joined by a mesh edge (either order, repeats harmless), `area` [vertices] ≥ 0, `positions` [vertices × 3], `threshold` > 0, `firstFrame` the
 *  number of the block's first column. std::invalid_argument for sizes that disagree, a vertex outside 0 … V−1, a threshold that is not above 0, or a value that is not finite. */
Cores corePatches(const Eigen::Ref<const Eigen::MatrixXd>& values, const std::vector<std::pair<std::int32_t, std::int32_t>>& pairs, const Eigen::Ref<const Eigen::VectorXd>& area,
                  const Eigen::Ref<const Eigen::MatrixXd>& positions, double threshold, std::int32_t firstFrame = 0);

/** the cores of two blocks of frames joined in order (a block's after the previous one's) */
void append(Cores& into, const Cores& more);

}  // namespace nxr::select
