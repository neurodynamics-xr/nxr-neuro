#pragma once
// nxr::core: general algebra of matrices, with no domain: a Hermitian factor, the symmetric square root, the antisymmetric factor of Im(F F^H), and a diagonal row scaling.
// (hermitianFactor and skewFactor were declared in quadrature.h; divideRows in scaling.h.)
#include <Eigen/Dense>
#include <complex>
#include <vector>

namespace nxr::core {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

/** A factor L [N x r] of a Hermitian positive semi-definite S = L L^H from its eigendecomposition (L = V sqrt(max(e,0))); eigenvalues below `tolerance` x the largest are dropped, so r <= N is its rank. */
MatrixXcd hermitianFactor(const MatrixXcd& S, double tolerance = 1e-12);

/** The SYMMETRIC square root of a real symmetric positive semi-definite matrix (the symmetrised input; negative eigenvalues clipped to 0; Rheome pairsqrt): the one root R = R^T with R R = A.
 *  Not `hermitianFactor`: a factor L with S = L L^H is not unique (any L U, U unitary, is one), the symmetric root is; they are different objects and stay two functions. */
MatrixXd symmetricSquareRoot(const MatrixXd& A);

// ── the quadrature spectrum in factored form ────────────────────────────────────────────────────────────────────────

struct QuadratureFactor {
    MatrixXd Q;     // [N x m] orthonormal columns, m = min(N, 2r)
    MatrixXd S;     // [m x m] antisymmetric: Im(F F^H) = Q S Q^T (exactly: not normalised)
    double reference = 0;   // ||F||_F^2 = tr C, the scale of C: a pair weaker than 1e-10 of it is rounding, not quadrature, and planes leaves it out
};
/** Im(F F^H) = J R^T - R J^T (F = R + iJ) as Q S Q^T, through the QR of [R J]. nxr's `lagged` (which also scales S to unit norm: S / ||S||_F). */
QuadratureFactor skewFactor(const MatrixXcd& F);

// ── a diagonal scaling of a matrix's rows ───────────────────────────────────────────────────────────────────────────
// D⁻¹ X, each row i of X divided by d_i: pointwise, it never reduces across rows or columns, so a matrix cut into blocks of columns gives the same result block by block.
/** Y(i, j) = X(i, j) / d(i). std::invalid_argument when d does not have X's rows, or an entry of d is not finite and above 0 (the message names the first). */
Eigen::MatrixXd divideRows(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::VectorXd>& d);

// ── convolution along the columns ───────────────────────────────────────────────────────────────────────────────────
/** MATLAB's conv(x, taps, 'valid') for each row of X: [rows x n] from [rows x (n + L - 1)], the L (odd) SYMMETRIC taps applied along the columns, the valid part only (output column j is centred on input column
 *  j + (L-1)/2; the taps are not flipped, which for symmetric taps is the same). std::invalid_argument when L is even or X has fewer than L columns. */
Eigen::MatrixXd conv(const Eigen::MatrixXd& X, const Eigen::VectorXd& taps);

}  // namespace nxr::core
