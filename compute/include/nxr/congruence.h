#pragma once
// nxr::algebra — CONGRUENCE ON A PATTERN: (A M Aᴴ) at the pairs asked for, never the whole n x n product. General algebra (the architecture policy:
// general algebra stays its own function): a source cross-spectrum on a mesh's edges is K C Kᴴ on the edge pattern (the magnetic Laplacian's input,
// docs/specs/2026-11-12-magnetic-laplacian.md), and any other A M Aᴴ restricted to a sparsity pattern is the same call.
//
// PURE MATHS: arrays in, arrays out. O(|pairs| · m²) with A M precomputed once: O(n m²) + O(|pairs| m).
#include <Eigen/Dense>
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

}  // namespace nxr::algebra
