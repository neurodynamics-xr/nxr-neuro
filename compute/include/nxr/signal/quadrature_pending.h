#pragma once
// nxr::quadrature: the members of the former quadrature.h that belong to the core (algebra, congruence) and geometry (mesh) toolboxes. They wait here for the batches that own those toolboxes (B4, B1).
// Moved from quadrature.h (toolbox refactor): declarations unchanged; the conventions are in quadrature.h.
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <vector>

#include <limits>

#include "nxr/hodge_weights.h"
#include "nxr/selection/resolved.h"

namespace nxr::quadrature {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

/** A factor L [N x r] of a Hermitian positive semi-definite S = L L^H from its eigendecomposition (L = V sqrt(max(e,0))); eigenvalues below `tolerance` x the largest are dropped, so r <= N is its rank. */
MatrixXcd hermitianFactor(const MatrixXcd& S, double tolerance = 1e-12);

// ── the quadrature spectrum in factored form ────────────────────────────────────────────────────────────────────────

struct QuadratureFactor {
    MatrixXd Q;     // [N x m] orthonormal columns, m = min(N, 2r)
    MatrixXd S;     // [m x m] antisymmetric: Im(F F^H) = Q S Q^T (exactly: not normalised)
    double reference = 0;   // ||F||_F^2 = tr C, the scale of C: a pair weaker than 1e-10 of it is rounding, not quadrature, and quadraturePairs leaves it out
};
/** Im(F F^H) = J R^T - R J^T (F = R + iJ) as Q S Q^T, through the QR of [R J]. nxr's `lagged` (which also scales S to unit norm: S / ||S||_F). */
QuadratureFactor quadratureFactor(const MatrixXcd& F);

/** The quadrature energy of each row of Fr [n x r] against all rows of F [N x r]: e_i = sum_j (Im C_ij)^2 with C_ij = Fr_i . conj(F_j), from Gram products
 *  e_i = (Fr_i G Fr_i^H - Re(Fr_i G2 Fr_i^T)) / 2, G = F^H F, G2 = F^H conj(F). Their sum over i is nxr's `im_rows(Fr, F)`. */
VectorXd quadratureEnergy(const MatrixXcd& Fr, const MatrixXcd& F);

// ── the energy of a block of a source cross-spectrum, without forming it ──────────────────────────────────────────────────

/** The two SUMS that make a quadrature fraction: ‖T‖_F² = Σ|T_ij|² and ‖Im T‖_F² = Σ|Im T_ij|², T = B C Bᵀ, B real [n x m], C Hermitian [m x m] (a stored Σ z zᴴ). T [n x n] is never formed: with the
 *  Gram G = BᵀB [m x m] and C = Cr + i Cq (Cr symmetric, Cq antisymmetric), ‖B X Bᵀ‖² = tr(X G Xᵀ G) for real X, so ‖Re T‖² = tr(Cr G Cr G), ‖Im T‖² = −tr(Cq G Cq G), ‖T‖² = ‖Re T‖² + ‖Im T‖².
 *  C is symmetrised ((C + Cᴴ)/2) first. These are SUMS OF PRODUCTS and the only things to store; the fraction ‖Im T‖/‖T‖ is their ratio, derived at read (the squares add across what is summed; the roots do not).
 *  They are NOT additive over a union of blocks (the cross blocks are missing) nor over time (Σ_t ‖T_t‖² ≠ ‖T of Σ_t C_t‖²: sum the C first). */
struct CongruenceNorms { double total = 0, quadrature = 0; };
CongruenceNorms congruenceNorms(const MatrixXd& B, const MatrixXcd& C);

// ── the direction a pattern moves in ───────────────────────────────────────────────────────────────────────────────

/** Area [nF] of each face of a mesh (positions [nV x 3], faces [nF x 3]). */
VectorXd faceAreas(const hodge::RowMat3& positions, const hodge::FaceMat& faces);

}  // namespace nxr::quadrature
