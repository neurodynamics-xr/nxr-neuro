#pragma once
// nxr::magnetic — THE MAGNETIC LAPLACIAN AND ITS EIGENMODES: the phase summary over time, trials and subjects (docs/specs/2026-11-12-magnetic-laplacian.md,
// approved by the). A graph's edges carry complex weights W_uv = w_uv e^{iθ_uv} (a lag θ and a strength w); ONE complex map x with
// x_u ≈ e^{iθ_uv} x_v is sought: relaxed to Σ |x_u|² = 1, that is the lowest eigenvector of the magnetic Laplacian (Singer, 2011: angular synchronisation).
//
//   magneticLaplacian(n, edges, W, normalise)   L = D − W (W Hermitian: W_vu = conj W_uv), D_uu = Σ_v |W_uv|; Hermitian, positive semi-definite.
//                                               Its quadratic form is Σ_edges |W_uv| |x_u − e^{iθ_uv} x_v|²: a map fitting every lag exactly has L x = 0.
//       normalise  none · symmetric (D^{-1/2} L D^{-1/2}: Singer's form)
//   W is the caller's: the Laplacian does not know what the weights are made of (an averaged cross-spectrum, its coherency, anything else; maintainers,
//   2026-11-12: build the operator, not what is supplied to it).
//   hermitianEigs(L, k)        the lowest k eigenpairs of a sparse Hermitian matrix (Spectra 1.2.0 HermEigsSolver, shift-invert through Eigen's complex
//                              SimplicialLDLT); each eigenvector of unit norm, its largest component real and positive (a vector's global phase is
//                              arbitrary: this fixes it so a result is reproducible)
//   hermitianEigsDense(L, k)   the same by a dense Hermitian solve (Eigen's SelfAdjointEigenSolver): the test oracle, small matrices only
//   winding(x, faces)          the winding number of arg x round each face (the wrapped phase differences along its edges, summed, / 2π): ±1 at a
//                              vortex core's face, 0 elsewhere
//
// The method knows only its input, the edge weights. What it cannot resolve: the map is defined up to a global rotation (only phase
// differences mean anything); lags over half a cycle between neighbours alias; where |x| is small the phase is undetermined; λ₁ compares only on the
// same graph and options.
//
// PURE MATHS: arrays in, arrays out.
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace nxr::magnetic {

using Complex = std::complex<double>;
using Hermitian = Eigen::SparseMatrix<Complex>;
using Edges = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;   // each undirected edge once: (u, v)
using Faces = Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

enum class Normalise { None, Symmetric };

/** the normalisation by its name ("none", "symmetric"); std::invalid_argument naming the valid ones otherwise */
Normalise normaliseOf(const std::string& name);

/** L on n vertices from the edges' weights (W[e] = W_uv for edges(e, 0) = u, edges(e, 1) = v; W_vu = conj). Throws std::invalid_argument on a shape
 *  mismatch, an edge outside [0, n), a self-loop, an edge given twice, or a non-finite weight. With `symmetric`, a vertex of no weight (D_uu = 0) keeps a
 *  zero row (it is in no edge's fit). */
Hermitian magneticLaplacian(int n, const Edges& edges, const std::vector<Complex>& W, Normalise normalise);

/** THE TRANSPORTED MAGNETIC LAPLACIAN (a connection Laplacian whose transport is the surface's rotation times the measured lag): on n vertices with a
 *  b-vector a vertex (b = 3: a current), W_uv = w_uv e^{iθ_uv} R_uv (R_uv [b x b] orthogonal, carrying v's vector to u: nxr::transport), W_vu = W_uvᴴ,
 *  D_uu = Σ_v w_uv I_b; L = D − W [b·n x b·n], Hermitian, PSD, its quadratic form Σ_edges w_uv |x_u − e^{iθ_uv} R_uv x_v|². Its lowest eigenvector is ONE
 *  vector map x (b complex numbers a vertex) whose transported lags fit the measured ones: no per-vertex sign is chosen; only a global phase remains.
 *  `normalise` symmetric: D^{-1/2} L D^{-1/2}. Throws std::invalid_argument on mismatched sizes, a self-loop, an edge twice, a negative or non-finite
 *  weight, or a rotation of the wrong shape. */
Hermitian transportLaplacian(int n, const Edges& edges, const Eigen::VectorXd& w, const Eigen::VectorXd& theta, const std::vector<Eigen::MatrixXd>& R, Normalise normalise);

/** THE BLOCK MAGNETIC LAPLACIAN: b complex numbers a vertex, ARBITRARY complex b x b edge blocks W_uv (blocks[e] for edges(e) = (u, v); W_vu = W_uvᴴ),
 *  degree D_uu = Σ_v ‖W_uv‖₂ I_b (the blocks' spectral norms: PSD for any blocks, since |x_uᴴ W x_v| ≤ ‖W‖ |x_u| |x_v|); L = D − W [b·n x b·n]. b = 1 is
 *  magneticLaplacian; blocks w e^{iθ} R (R orthogonal) transportLaplacian. THE JOINT DATA-BLOCK ROUTE: with
 *  each edge's normalised cross-spectrum block (T_uv / √(tr T_uu tr T_vv)) its lowest eigenvector is one complex b-vector a vertex fitting every pair's
 *  relation (orientation and lag together) consistently over the graph; only a global phase is free. `normalise` symmetric: D^{-1/2} L D^{-1/2}.
 *  Throws std::invalid_argument on mismatched sizes, a self-loop, an edge twice, blocks of unequal or wrong shape, or a non-finite entry. */
Hermitian blockMagneticLaplacian(int n, const Edges& edges, const std::vector<Eigen::MatrixXcd>& blocks, Normalise normalise);

struct Eigenpairs {
  Eigen::VectorXd values;        // [k] ascending
  Eigen::MatrixXcd vectors;      // [n x k], column i the i-th eigenvector (unit norm, its largest component real and positive)
  int converged = 0;             // how many the iterative solver converged (k for the dense oracle)
};

/** the lowest k eigenpairs: shift-invert at `sigma` (below the spectrum: L − σI positive definite for a PSD L), `tol` the solver's relative tolerance.
 *  Throws nxr::core::Error (EigensolveInvalidK) for k outside [1, n − 1], (EigensolveNotConverged) when fewer than k converge. */
Eigenpairs hermitianEigs(const Hermitian& L, int k, double sigma = -1e-6, double tol = 1e-10);
/** the dense oracle (n ≤ 4096) */
Eigenpairs hermitianEigsDense(const Hermitian& L, int k);

/** the winding number of arg x round each face (faces CCW): an integer per face */
std::vector<int> winding(const Eigen::VectorXcd& x, const Faces& faces);

}  // namespace nxr::magnetic
