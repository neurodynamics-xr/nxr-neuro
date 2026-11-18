#pragma once
// nxr::core — CONGRUENCE ON A PATTERN: (A M Aᴴ) at the pairs asked for, never the whole n x n product. General algebra (the architecture policy:
// general algebra stays its own function): a source cross-spectrum on a mesh's edges is K C Kᴴ on the edge pattern (the magnetic Laplacian's input,
// docs/specs/2026-11-12-magnetic-laplacian.md), and any other A M Aᴴ restricted to a sparsity pattern is the same call.
//
// PURE MATHS: arrays in, arrays out. O(|pairs| · m²) with A M precomputed once: O(n m²) + O(|pairs| m).
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <vector>

namespace nxr::core {

using Pairs = Eigen::Matrix<int32_t, Eigen::Dynamic, 2, Eigen::RowMajor>;

/** out[p] = (A M Aᴴ)(pairs(p, 0), pairs(p, 1)) = Σ_ab A[u, a] M[a, b] conj(A[v, b]); A [n x m] real, M [m x m] complex. Throws std::invalid_argument on a
 *  shape mismatch or a pair outside [0, n). */
std::vector<std::complex<double>> congruence(const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M, const Pairs& pairs);

/** The WHOLE product A M Aᴴ [n x n] (A real [n x m], M complex [m x m]): for a small n (a spatial band's coefficients), where every pair is wanted.
 *  Throws std::invalid_argument on a shape mismatch. */
Eigen::MatrixXcd congruence(const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M);

/** BLOCKS on a pattern: A [b·n x m] holds b consecutive rows an element (a free-orientation kernel: x y z a source); out[p] is the b x b block
 *  (A M Aᴴ)(element u, element v) for pairs(p) = (u, v): the cross-spectrum of u's b components with v's (nxr::meeg::crossTensor's, batched).
 *  Throws std::invalid_argument on a shape mismatch or a pair outside [0, n). */
std::vector<Eigen::MatrixXcd> blockCongruence(const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M, const Pairs& pairs, int b);

/** SPARSE CONGRUENCE: projection · connections · projectionᵀ for a sparse projection [m x n] and sparse connections [n x n] (a connection matrix smoothed over a mesh, a coarse-graining of a graph): the result is [m x m] and
 *  sparse, formed in the precision of the inputs. General algebra: it knows nothing of what the projection and the connections are. Throws std::invalid_argument on a shape mismatch. */
template <class Scalar>
Eigen::SparseMatrix<Scalar> sparseCongruence(const Eigen::SparseMatrix<Scalar>& projection, const Eigen::SparseMatrix<Scalar>& connections);

/** The same product formed in blocks of `block` columns of the result, each block pruned before the next is formed: entries with |value| < `drop_below` are not kept
 *  (0 keeps every entry that is not exactly zero). The result equals sparseCongruence followed by the pruning of entries below `drop_below`; the memory is one block of
 *  the product, not the whole. Throws std::invalid_argument on a shape mismatch or block < 1. */
template <class Scalar>
Eigen::SparseMatrix<Scalar> sparseCongruenceBlocked(const Eigen::SparseMatrix<Scalar>& projection, const Eigen::SparseMatrix<Scalar>& connections, std::int64_t block, Scalar drop_below);

// ── the K C K^T blocks of a free-orientation kernel (from meeg/tensor.h): the cross-spectral TENSOR of a source current ──
//   sourceTensor(K, S, Sq)   per source v, the complex 3x3 Hermitian T_v = K_v (S + i Sq) K_v^T, K_v the source's three kernel rows (x y z), S + i Sq a band's sensor cross-spectral density
//                            (S its real part, Sq its imaginary part Im(X X^H)). Re T is symmetric, Im T antisymmetric; tr T = the source's band power summed over its orientations. Never a
//                            source time series: channels^2 per band.
//   crossTensor(K, S, Sq, i, j)   the 3x3 block BETWEEN two sources, T_ij = K_i (S + i Sq) K_j^T (T_ji = T_ij^H; T_ii is sourceTensor's), read on demand: the full 3N x 3N matrix is never formed.
/** [N] tensors, each 9 real and 9 imaginary parts, row-major 3x3 (x y z). */
struct SourceTensors {
  Eigen::Index sources = 0;
  std::vector<double> re, im;                 // [N x 9] each
  const double* reOf(Eigen::Index v) const { return re.data() + 9 * v; }
  const double* imOf(Eigen::Index v) const { return im.data() + 9 * v; }
};
/** K [3N x M] (three rows a source: x y z), S and Sq [M x M] (Sq antisymmetric). Throws std::invalid_argument on a shape mismatch. */
SourceTensors sourceTensor(const Eigen::MatrixXd& K, const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq);
/** The block T_ij (re, im: 9 each, row-major) between sources i and j of K [3N x M]. Throws std::invalid_argument on a shape mismatch or a source outside K. */
void crossTensor(const Eigen::MatrixXd& K, const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq, Eigen::Index i, Eigen::Index j, double* re, double* im);

}  // namespace nxr::core
