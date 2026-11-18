#pragma once
// nxr::signal — THE LEADING PATTERNS of a Hermitian second-moment matrix (a cross-spectrum over a window), and how alike two patterns are
// (spec docs/specs/2026-11-13-cycle-pattern.md). PURE MATHS: arrays in, arrays out; nothing here knows a window, a band or a cortex.
//
//   leading          the q leading eigenvectors of C (unit norm), their eigenvalues and C's trace: a pattern's share of C is λ_i / trace (derived at read)
//   rotateToLargest  a complex pattern's free phase fixed: m e^{iθ} with θ = −½ arg Σ m_k², so that Re(m e^{iθ}) has the largest norm over θ (phase 0 is the
//                    frame of largest variance); ‖Im‖² / ‖Re‖² of the result is 0 for a standing pattern and 1 for a pattern that travels evenly
//   similarity       |⟨u_i, v_i⟩| per column: phase-free (invariant to e^{iθ} on either), 1 for the same pattern
#include <Eigen/Dense>
#include <vector>

#include "nxr/core/algebra.h"

namespace nxr::signal {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

struct Leading {
  Eigen::MatrixXcd vectors;    // [n x q] unit-norm eigenvectors, largest eigenvalue first
  Eigen::VectorXd values;      // [q] their eigenvalues (≥ 0 for a second-moment matrix, up to rounding)
  double trace = 0;            // tr C
};
/** The q leading eigenpairs of a Hermitian C [n x n] (its Hermitian part ½(C + Cᴴ) is used). Throws std::invalid_argument when C is not square or q is
 *  outside [1, n]. Each vector is returned with rotateToLargest applied. */
Leading leading(const Eigen::MatrixXcd& C, int q);
/** m e^{iθ}, θ = −½ arg Σ m_k² (θ = 0 when Σ m_k² = 0: a pattern that travels evenly has no frame of largest variance). */
Eigen::VectorXcd rotateToLargest(const Eigen::VectorXcd& m);
/** |⟨U.col(i), V.col(i)⟩| for each column of two [n x q] matrices of unit vectors. Throws std::invalid_argument on a shape mismatch. */
Eigen::VectorXd similarity(const Eigen::MatrixXcd& U, const Eigen::MatrixXcd& V);

// Moved from quadrature.h (toolbox refactor): declarations unchanged.
// ── quadrature pairs ───────────────────────────────────────────────────────────────────────────────────────────────

struct QuadraturePair {
    VectorXd p, q;  // [N] unit, orthogonal; p leads q by a quarter cycle (see above)
    double sigma = 0;   // the pair's quadrature amplitude (units of the spectrum Im C; nxr's sigma is this over ||Im C||_F)
};
/** The k strongest pairs of Im C = Q S Q^T, strongest first (fewer if Im C has fewer nonzero pairs). From the Hermitian matrix -iS, whose eigenvalues are +-sigma. */
std::vector<QuadraturePair> planes(const core::QuadratureFactor& f, int k);
/** The strongest pair. */
QuadraturePair top_plane(const core::QuadratureFactor& f);
/** nxr's `planes(Q, S, k)` and `top_plane(F)` through the same code: the pairs straight from a complex factor. */
std::vector<QuadraturePair> planes(const MatrixXcd& F, int k);

// ── a read-out's top pair as sensor filters ────────────────────────────────────────────────────────────────────────

struct QuadraturePairFilters {
    VectorXd fx, fy;    // [C] sensor filters: the pair's patterns read back to the sensors, fx = W^T p, fy = W^T q (negated if the sense came out -1)
    int sense = 1;      // +1: the rotation fx -> fy is at +f (counter-clockwise); -1 when the pair's rotary power was larger at -f (then fy is negated)
    double sigma = 0;   // the pair's amplitude (of Im(W S W^H))
    double positive = 0, negative = 0;   // the rotary power of z = fx.x + i fy.x at +f and -f in the band: w^T S conj(w), w^H S w, w = fx + i fy
};
/** The top quadrature pair of the read-out W [R x C] (rows: the modes' weights, a kernel's rows ...) of a band's sensor cross-spectrum S [C x C] (Hermitian): F = W L, S = L L^H.
 *  nxr's `rotation_plane`, from the band's cross-spectrum instead of a time series. */
QuadraturePairFilters rotationPlaneFilters(const MatrixXd& W, const MatrixXcd& S);

}  // namespace nxr::signal
