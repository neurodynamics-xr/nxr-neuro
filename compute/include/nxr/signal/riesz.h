#pragma once
// nxr::monogenic — the SPATIAL ANALYTIC SIGNAL (monogenic signal) of a band-passed field on a surface; PURE MATHS: matrices in, matrices out. NOT in Rheome, nxr or the Studio: the maths is Felsberg and Sommer (2001), written from the paper.
//
// A spatial band leaves a SIGNED field; |f| dips to zero where lobes of opposite sign meet, as a band-passed x(t) does between its peaks. The analytic signal's spatial analogue is
// the Riesz transform R f = ∇ Δ^{-1/2} f, a tangent vector field, and with it
//   spatial amplitude   A = sqrt(f² + |R f|²)        unsigned: no zero-crossing dips
//   spatial phase       atan2(|R f|, f)  in [0, π]   0 on a positive lobe's crest, π/2 on its flank, π on a negative lobe's trough
//   orientation         R f / |R f|                  the local wave-vector direction (a unit tangent vector), NaN where R f = 0
//
// R f is LINEAR, so a caller folds it into a linear map once (a kernel, a band): R f = G (Φ diag(g/√λ) Φᵀ ⋆0) f, g the band's gain, Φ the Laplace–Beltrami modes, ⋆0 the vertex mass,
// G the mesh gradient. The constant mode (λ = 0) has no Riesz part. This file holds the pointwise pieces; every function is POINTWISE (per vertex, per sample): nothing is reduced.
//
// LAYOUT: a field holds `components` numbers a vertex, row v·components + a, one column a sample (as nxr::phase). A Riesz part holds 3 (x y z) a vertex and component.
//
//   rieszGain(g, lambdaMin)        g(λ)/√λ, and 0 for λ ≤ lambdaMin (the constant mode has no Riesz part)
//   rieszOfScaled(G, S)            the mesh gradient of Δ^{-1/2} F f: [3V·c × m] (x y z a vertex, per component) from S [V·c × m]
//   amplitude(Z, zc, R, rc)        sqrt(Σ|z_a|² + Σ|r|²) per vertex and sample                                                  [V × m]
//   phase(f, R)                    atan2(|R f|, f) of a REAL field f [V × m] and its Riesz part R [3V × m] (one component)        [V × m]
//   orientation(R)                 R f / |R f| per vertex and sample [3V × m]; NaN where |R f| = 0
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <functional>

namespace nxr::signal {

using Gradient = std::array<Eigen::SparseMatrix<double>, 3>;   // nxr::phase::Gradient: the three [V × V] operators x, y, z

/** g(λ)/√λ; 0 where λ ≤ lambdaMin (default 1e-3 1/m^2: no cortex has a non-constant mode there) or g(λ) = 0 (so a band's gain, zero away from the band, stays zero). */
std::function<double(double)> rieszGain(const std::function<double(double)>& g, double lambdaMin = 1e-3);

/** The gradient of a scalar field per component: S [V·components × m] (row v·components + a) → [3V·components × m], row (v·components + a)·3 + d. */
Eigen::MatrixXd rieszOfScaled(const Gradient& G, const Eigen::MatrixXd& S, int components = 1);
Eigen::MatrixXcd rieszOfScaled(const Gradient& G, const Eigen::MatrixXcd& S, int components = 1);

/** A = sqrt(Σ_a |z_a|² + Σ_b |r_b|²) per vertex: Z [V·zc × m], R [V·rc × m] (rc = 3·zc for a Riesz part), same V and m → [V × m]. */
Eigen::MatrixXd amplitude(const Eigen::MatrixXcd& Z, int zc, const Eigen::MatrixXcd& R, int rc);
/** atan2(|R|, f) in [0, π]: f [V × m] real, R [3V × m] its Riesz part. */
Eigen::MatrixXd phase(const Eigen::MatrixXd& f, const Eigen::MatrixXd& R);
/** R / |R| per vertex: [3V × m]; NaN where |R| = 0. */
Eigen::MatrixXd orientation(const Eigen::MatrixXd& R);

}  // namespace nxr::signal
