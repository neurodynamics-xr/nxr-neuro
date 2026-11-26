// nxr::meeg — THE FIBRE NETWORK ON THE TILES OF A SURFACE and the flow on it, as pure functions (docs/specs/2026-11-24-connectome-flow.md). Matrices in, matrices out;
// no file, no subject. The reference is nxr's flow through the fibre network (`flownet.py`: tiles of the surface ladder as nodes, tile pairs joined by streamlines as
// edges, the connectome Laplacian and its eigenmodes, the divergence and the graph potential of an edge flow).
//
//   fiberLengths            each fibre's length: the sum of its segments' lengths, in the points' unit
//   tileCodesAtLevel        a dyadic grid's finest code per vertex shifted to a coarser level (code >> (depth - level)); -1 stays -1
//   tilePairs               the tiles (the codes some vertex holds, ascending), each tile's centroid and piece, and every pair of tiles (i < j): the count of streamlines
//                           whose two ends lie in the two tiles, their median length, the centroids' distance, whether the tiles lie on different pieces
//   graphHodgeDirac         the Hodge–Dirac modes of a GRAPH (a 1-complex: d0 = B the incidence, -1 at the tail, +1 at the tip; ⋆0 = diag(node weights), ⋆1 = diag(edge
//                           weights); no 2-cells): the exact (gradient) modes of Bᵀ⋆1 B ψ = λ ⋆0 ψ, λ > 0 (one dense generalized symmetric eigensolve: the graph is small;
//                           the constants, one per connected component, out), each as its MIXED FORM [ψ/√2 ; Bψ/(μ√2)], and the harmonic (cycle) modes, the kernel of
//                           Bᵀ⋆1 (E - V + components of them), each [0 ; h], ⋆1-orthonormal; all orthonormal under diag(⋆0, ⋆1). The surfaces' convention (eigen_steps)
//   networkFlow             an edge flow F on the graph's edges: the divergence d = Bᵀ F, the flow's coefficient on every mode (its projection on the mode's 1-form part: 2 ⟨F,
//                           φ_k⟩⋆1 on an exact mode, ⟨F, h_k⟩⋆1 on a cycle mode), the graph potential phi = Σ_exact c_k ψ_k / μ_k (the weighted least-squares potential:
//                           Bᵀ⋆1 B phi = Bᵀ⋆1 F, the constants out), and the sums Σ ⋆1 (B phi)² and Σ ⋆1 F² (the gradient share is their ratio, derived at read)
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

/** a graph's Hodge–Dirac modes (graphHodgeDirac): K = cycle modes + exact modes, ascending μ (the cycle modes at μ = 0 first) */
struct GraphModes {
  std::int64_t nodes = 0, edges = 0, components = 0, cycles = 0;   // V, E, the connected components (the null 0-forms left out), the cycle modes (E - V + components)
  Eigen::VectorXd mu;            // [K] ascending; 0 on a cycle mode
  std::vector<int> kind;         // [K] 0 exact, 1 harmonic (a cycle)
  Eigen::MatrixXd forms;         // [(V + E) x K] each mode's mixed form: rows 0-forms [0, V) · 1-forms [V, V + E)
};
/** edges [E x 2] (tail, tip; tail != tip) of a graph of `nodes` vertices, node weights [V] > 0 (⋆0), edge weights [E] > 0 (⋆1); an eigenvalue at most `nullTolerance` x the
 *  largest is a constant (one per connected component). std::invalid_argument on a weight that is not positive, an edge outside the vertices or a loop. */
GraphModes graphHodgeDirac(const std::vector<std::int32_t>& edges, std::int64_t nodes, const Eigen::VectorXd& nodeWeights, const Eigen::VectorXd& edgeWeights, double nullTolerance);

struct NetworkFlow {
  Eigen::VectorXd divergence;    // [V] Bᵀ F
  Eigen::VectorXd potential;     // [V] phi
  Eigen::VectorXd coefficients;  // [K] the flow's coefficient on each mode (the modes' order)
  double weightedGradientSquares = 0;   // Σ_e ⋆1_e (B phi)_e²  (= Σ_exact c_k² / 2)
  double weightedCycleSquares = 0;      // Σ_cycles c_k²: the cycle part's ⋆1 energy
  double weightedFlowSquares = 0;       // Σ_e ⋆1_e F_e²
};
/** an edge flow [E] (tail → tip) on the graph the modes were solved on (its edges [E x 2] and edge weights [E], ⋆1) → its divergence, coefficients, potential and sums. */
NetworkFlow networkFlow(const Eigen::VectorXd& edgeFlow, const std::vector<std::int32_t>& edges, const Eigen::VectorXd& edgeWeights, const GraphModes& modes);

}  // namespace nxr::meeg
