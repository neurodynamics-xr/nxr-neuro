#pragma once
// nxr::crossings — the CROSSINGS of tracks through tiles (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 7). PURE MATHS on arrays: steps in (a core's step from one frame to the next: its sense, where it started, how it moved,
// which track it is of), the surface's per-vertex frames and tiles in, per tile and sense the sum of the crossings' directions and their number out. The function never knows where the steps came from (stored tracks, a synthetic walk) or
// which frames the caller left in (a stage is the caller's mask). A port of nxr's `Tiles.crossings` (nxr/detect/vortex_cores.py), the reference:
//
//   a step     sits at the surface vertex NEAREST its start; that vertex's tile and normal are the step's. Its displacement is projected on the tangent plane there (a unit vector; a step shorter than 1e-9 has no direction and is left out).
//   carried    the unit vector is carried to its tile's CENTRE along the surface (the vector heat method's transport: the Levi-Civita form of nxr/ladder.h, a complex number c1 + i c2 in the vertex's frame turned by e^{-i rho}, the
//              same arithmetic as `ladder::lcResultant`); vectors at different places of a fold are never added in 3-D. A vertex with no frame or no transport carries 0 (the crossing still counts, as nxr's).
//   a crossing is ONE TRACK'S STEPS IN ONE TILE, ONE SENSE: their carried vectors added and made a unit vector. The crossings of a tile and sense are then added, and counted.
//   bridge     a step whose nearest vertex is on the callosal bridge (the glued surface's, not cortex: no tile) has no centre to carry to: its crossing is its track's steps' tangent unit vectors added in 3-D and made a unit vector,
//              summed in 3-D, in its own member.
// Derived at read (never here): R = |S| / n and the unbiased rho2 = (|S|^2 - n) / (n (n - 1)) of a tile's sum S of n crossings.
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

#include "nxr/geometry/frames.h"
#include "nxr/geometry/intrinsic_surface.h"

namespace nxr::select::track_crossings {

using geometry::Mat3;
constexpr std::int32_t kBridge = -2;      // a vertex's `node` on the bridge; -1: on no tile (a vertex cut away)

struct Steps {
  std::vector<std::int8_t> sense;         // +1 / -1
  Eigen::MatrixXd from, disp;             // [n x 3]: where the step started, how far it moved (the surface's unit)
  std::vector<std::int32_t> track;        // the track each step belongs to (any number: steps of one number are one track's)
  std::int64_t size() const { return static_cast<std::int64_t>(sense.size()); }
};
struct Surface {
  Eigen::MatrixXd positions;              // [V x 3] where a step is placed: its nearest vertex
  geometry::Mat3 normals, e1, e2;                   // [V x 3] per vertex (e1, e2: the frame the carried number is written in; NaN where there is none)
  Eigen::VectorXd rho;                    // [V] the frame angle at each tile vertex of its tile centre's e1 carried there (ladder::leviCivita(...).rho; NaN off the centre's component)
  std::vector<std::int32_t> node;         // [V] the vertex's tile (>= 0), -1 (none) or kBridge
};
struct Result {
  Eigen::MatrixXcd sum;                   // [tiles x 2]: column 0 sense +1, column 1 sense -1: the crossings' unit vectors carried to the tile centre, added (complex, in the centre's frame)
  Eigen::MatrixXi n;                      // [tiles x 2]: the crossings
  Eigen::Matrix<double, 2, 3> bridge = Eigen::Matrix<double, 2, 3>::Zero();   // [sense +1 / -1 x xyz]: the bridge's crossings' unit vectors, added in 3-D
  Eigen::Vector2i bridge_n = Eigen::Vector2i::Zero();
  std::int64_t steps_in = 0, steps_used = 0, no_direction = 0, off_tiles = 0;   // the steps asked for, those that made a crossing, those too short, those on no tile
};

/** `use` [steps] (empty: all): the steps counted (a stage's frames). std::invalid_argument for arrays that disagree, a vertex-less surface, a sense that is not +-1, or a tile >= `tiles`. */
Result crossings(const Steps& steps, const std::vector<std::uint8_t>& use, const Surface& surface, int tiles);

/** tile sums (complex, in the centres' frames) as 3-D tangent vectors at the centres: c1 e1(centre) + c2 e2(centre) [tiles x 3] (0 where the tile has no centre). */
Eigen::MatrixXd embedAtCentres(const Eigen::VectorXcd& w, const std::vector<int>& centres, const geometry::Mat3& e1, const geometry::Mat3& e2);

/** the vertex of `positions` nearest each query point [q x 3] (the lowest index on a tie) */
std::vector<std::int32_t> nearestVertex(const Eigen::MatrixXd& positions, const Eigen::MatrixXd& query);

// ── the RUN-TO-RUN REPEAT (spec piece 8): does a tile's direction repeat from one recording to the next? ──
// nxr `repeat` and `tile_dot` (vortex_cores.py), on tile sums already made: a tile's two senses POOLED (a core turns sense each half cycle, so the direction of the crossings is the tile's, not the sense's: the two senses' 3-D
// sums are vectors at the same centre and add), the mean resultant vector m = S / n of each run, and only the tiles with at least `minCrossings` crossings in BOTH runs (10, nxr's NMIN_TILE) are compared:
//   dot     per tile, m_a . m_b: large only when both runs are consistent and agree; 0 where either is inconsistent; negative when they go opposite ways (NaN: fewer crossings in a run)
//   cosine  the whole recording's COUNT-WEIGHTED mean of the tiles' cosines S_a . S_b / (|S_a||S_b|), weight min(n_a, n_b): a tile whose sum vanishes in a run has no direction and is left out (NaN: no tile qualifies)
// Its p-value needs the null (piece 10); this holds the statistic only.
struct Pooled { Eigen::MatrixXd sum; Eigen::VectorXd n; };   // [tiles x 3], [tiles]
Pooled poolSenses(const Eigen::MatrixXd& plus, const Eigen::VectorXd& nPlus, const Eigen::MatrixXd& minus, const Eigen::VectorXd& nMinus);
struct Repeat {
  Eigen::VectorXd dot;                    // [tiles]
  double cosine = std::numeric_limits<double>::quiet_NaN();
  std::int64_t tilesUsed = 0;             // the tiles with at least minCrossings in both runs (and a direction in both, for the cosine)
};
Repeat repeat(const Pooled& a, const Pooled& b, int minCrossings = 10);

}  // namespace nxr::select::track_crossings
