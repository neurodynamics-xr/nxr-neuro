#pragma once
// nxr::algebra — CONGRUENCE ON A PATTERN: (A M Aᴴ) at the pairs asked for, never the whole n x n product. General algebra (the architecture policy:
// general algebra stays its own function): a source cross-spectrum on a mesh's edges is K C Kᴴ on the edge pattern (the magnetic Laplacian's input,
// docs/specs/2026-11-12-magnetic-laplacian.md), and any other A M Aᴴ restricted to a sparsity pattern is the same call.
//
// PURE MATHS: arrays in, arrays out. O(|pairs| · m²) with A M precomputed once: O(n m²) + O(|pairs| m).
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <vector>

namespace nxr::algebra {

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

}  // namespace nxr::algebra
