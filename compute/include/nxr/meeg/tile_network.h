// nxr::meeg — THE FIBRE NETWORK ON THE TILES OF A SURFACE and the flow on it, as pure functions (docs/specs/2026-11-24-connectome-flow.md). Matrices in, matrices out;
// no file, no subject. The reference is nxr's flow through the fibre network (`flownet.py`: tiles of the surface ladder as nodes, tile pairs joined by streamlines as
// edges, the connectome Laplacian and its eigenmodes, the divergence and the graph potential of an edge flow).
//
//   fiberLengths            each fibre's length: the sum of its segments' lengths, in the points' unit
//   tileCodesAtLevel        a dyadic grid's finest code per vertex shifted to a coarser level (code >> (depth - level)); -1 stays -1
//   tilePairs               the tiles (the codes some vertex holds, ascending), each tile's centroid and piece, and every pair of tiles (i < j): the count of streamlines
//                           whose two ends lie in the two tiles, their median length, the centroids' distance, whether the tiles lie on different pieces
//   laplacianEigenmodes     the graph Laplacian L = D - W of a weighted network on the tiles (W symmetric, from the pairs' weights) and ALL its eigenmodes (one dense
//                           symmetric eigensolve: the tile graph is small), eigenvalues ascending; the null modes (eigenvalue below a stated fraction of the largest) counted
//   networkFlow             an edge flow F on the network's edges (pairs i < j, i → j): the divergence d = Bᵀ F (B the incidence, -1 at i, +1 at j), the graph potential
//                           phi solving L phi = Bᵀ W F as the 1/eigenvalue gain on the eigenmodes (the minimum-norm solution: the null modes left out), its
//                           coefficients on the eigenmodes, and the sums Σ w (B phi)² and Σ w F² (the gradient share is their ratio, derived at read)
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::meeg {

/** points [nF·nP x 3] fibre-major (row-major xyz); nP >= 2 → lengths [nF] in the points' unit. */
std::vector<double> fiberLengths(const std::vector<double>& points, std::int64_t pointsPerFiber);
std::vector<double> fiberLengths(const std::vector<float>& points, std::int64_t pointsPerFiber);

/** finest codes [V] (-1 off the grid) at `depth` → codes at `level` (0 <= level <= depth). */
std::vector<std::int64_t> tileCodesAtLevel(const std::vector<std::int64_t>& finestCodes, int depth, int level);

struct TileNetwork {
  std::vector<std::int64_t> codes;          // [T] the tiles' codes, ascending
  std::vector<std::int64_t> pieces;         // [T] each tile's piece (code >> level: the hemisphere slot)
  std::vector<std::int64_t> vertexCount;    // [T]
  Eigen::MatrixXd centroids;                // [T x 3] the mean position of each tile's vertices
  // the pairs i < j in the order (0,1), (0,2), ..., (1,2), ...: P = T (T - 1) / 2
  std::vector<std::int32_t> first, second;  // [P] tile indices (rows of `codes`)
  Eigen::VectorXd count;                    // [P] streamlines whose two ends lie in the two tiles
  Eigen::VectorXd medianLength;             // [P] their median length (NaN when none)
  Eigen::VectorXd distance;                 // [P] the centroids' Euclidean distance (the positions' unit)
  Eigen::VectorXd across;                   // [P] 1 when the two tiles lie on different pieces
  std::int64_t endsOutside = 0, sameTile = 0, gatedOut = 0;   // fibres left out: an end on no tile; both ends in one tile; removed by the gate
};
/** tileCodes [V] (codes at the level, -1 off), positions [V x 3], level (the piece is code >> level), endVertex [2nF] in block order (-1: no vertex), keep [nF]
 *  (empty: every fibre), lengths [nF] (empty: medians NaN). */
TileNetwork tilePairs(const std::vector<std::int64_t>& tileCodes, const Eigen::MatrixXd& positions, int level, const std::vector<std::int32_t>& endVertex,
                      const std::vector<std::uint8_t>& keep, const std::vector<double>& lengths);

/** the index of pair (i, j), i < j, of T tiles in tilePairs' order */
inline std::int64_t pairIndex(std::int64_t i, std::int64_t j, std::int64_t T) { return i * T - i * (i + 1) / 2 + (j - i - 1); }

struct LaplacianModes {
  Eigen::VectorXd eigenvalues;   // [T] ascending
  Eigen::MatrixXd eigenvectors;  // [T x T] columns orthonormal
  Eigen::MatrixXd laplacian;     // [T x T]
  std::int64_t nullModes = 0;    // eigenvalues below nullTolerance x the largest: one per connected component of the network
};
/** the weights [P] of the pairs (0 where no edge) of T tiles in tilePairs' order → L = D - W and its eigenmodes. */
LaplacianModes laplacianEigenmodes(const Eigen::VectorXd& pairWeights, std::int64_t tiles, double nullTolerance);

struct NetworkFlow {
  Eigen::VectorXd divergence;    // [T] Bᵀ F
  Eigen::VectorXd potential;     // [T] phi
  Eigen::VectorXd coefficients;  // [T] u_kᵀ phi per eigenmode (0 on the null modes)
  double weightedGradientSquares = 0;   // Σ_e w_e (B phi)_e²
  double weightedFlowSquares = 0;       // Σ_e w_e F_e²
};
/** edge flow [P] and weights [P] of the pairs (an edge: weight > 0; the flow of a non-edge is not read), the network's eigenmodes. */
NetworkFlow networkFlow(const Eigen::VectorXd& pairFlow, const Eigen::VectorXd& pairWeights, const LaplacianModes& modes);

}  // namespace nxr::meeg
