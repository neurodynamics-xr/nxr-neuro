#pragma once
// nxr::quadrature — the part of a cross-spectrum that no zero-delay mixing can make, and the measures built on it. PURE MATHS (arrays in, arrays out). A port of nxr's
// `nxr.spectral.lagged` (MIT) with exact names.
//
// CONVENTIONS (one place, used by every function below)
//   Spectrum   a signal x(t) has the Fourier coefficient X(f) = sum_t x(t) e^{-2 pi i f t} (numpy's rfft; nxr::fft::r2c). A component cos(2 pi f t + phi) has X ∝ e^{i phi}.
//   Cross-spectral matrix  C_ij = E[X_i conj(X_j)], C = F F^H for a complex factor F [N x r] (its columns are r independent complex patterns). C is Hermitian.
//   Lead       arg C_ij = phi_i - phi_j. Im C_ij > 0 means row i LEADS row j (it peaks earlier, by phaseLag / (2 pi f) seconds).
//
//   coSpectrum(C)          Re C   the in-phase (co-) spectrum: symmetric
//   quadratureSpectrum(C)  Im C   the quadrature spectrum: ANTISYMMETRIC. Zero-delay mixing (a head model, an inverse, any real matrix K applied to the sensors) turns C into K C K^T, so Im C becomes
//                                 K (Im C) K^T: it adds no quadrature of its own. Sensors in phase (Im C = 0) stay in phase at every source; a lag read at the sources was in the sensors.
//   quadratureFactor(F)    (Q, S) with Im C = Q S Q^T, Q [N x m] orthonormal columns, S [m x m] antisymmetric (m <= 2r): through the QR of [Re F, Im F]; never an N x N matrix.
//   quadratureFraction     ||Im C||_F / ||C||_F  in [0, 1]; 0 for any in-phase mixture.
//   quadratureEnergy       per row i, sum_j (Im C_ij)^2 = ||Im(F_i F^H)||^2: the quadrature spectrum's energy in row i (nxr's `im_rows` is the sum of this over its rows).
//   quadraturePairs        the canonical decomposition of the antisymmetric Im C into eigenvalue pairs +- i sigma_k: Im C = sum_k sigma_k (p_k q_k^T - q_k p_k^T), p_k, q_k orthonormal real
//                          patterns, sigma_k > 0 the pair's quadrature amplitude, strongest first. BY THE LEAD CONVENTION ABOVE p LEADS q BY A QUARTER CYCLE: the activity is
//                          p cos(wt) + q sin(wt): p -> q -> -p -> -q, counter-clockwise in the (p, q) plane (the rotary spectrum's +f). (nxr's x = p, y = q.)
//                          The term sigma (p q^T - q p^T) is invariant under a rotation of (p, q) in their plane, so the pair is a plane with an orientation; we fix the gauge:
//                          the row where p^2 + q^2 is largest has q = 0 and p > 0.
//   quadratureShare        the share of ||Im C||_F^2 the pairs explain: ||B^T (Im C) B||_F^2 / ||Im C||_F^2, B the pairs' orthonormal basis (= sum 2 sigma_k^2 / ||Im C||_F^2).
//   impropriety            rho(u) = |u^T u| / u^H u (Schreier & Scharf's degree of impropriety) of a complex pattern: 1 a standing pattern (one real pattern times one phase),
//                          0 a circular one (a pure travelling or rotating pattern). 1 - impropriety is what nxr called the travel index.
//   rotarySpectrum         the rotary spectrum (Gonella 1972) of z = p.x + i q.x: the power at +f (p -> q) and at -f, by complex Morlet transform in time (nxr's `rotary_power`).
//   quadraturePairFilters  the top quadrature pair of a read-out of the sensors in a band, as two sensor filters and its rotary sense.
//   weightedPhaseGradient  P = Im(conj(z) grad z) = |z|^2 grad arg z, area-weighted over a log-map disc, each face's vector transported to the anchor: the direction and phaseGradientAlignment.
//   standingSurrogate      the best standing approximation of a complex map.
//   meanResultantLength · rayleighTest   circular statistics of directions.
//   phaseLag · timeLag     arg C_ij (rad) and arg C_ij / (2 pi f) (s): positive when row i leads row j.
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <vector>

#include <limits>

#include "nxr/chart.h"
#include "nxr/frames.h"
#include "nxr/hodge_weights.h"
#include "nxr/selection/resolved.h"

namespace nxr::quadrature {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

// ── the two halves of a cross-spectral matrix ───────────────────────────────────────────────────────────────────────

/** Re C. */
MatrixXd coSpectrum(const MatrixXcd& C);
/** Im C = (C - C^H)/(2i) for a Hermitian C: antisymmetric. */
MatrixXd quadratureSpectrum(const MatrixXcd& C);

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

/** ||Im C||_F / ||C||_F of C = F F^H, from the small Gram matrices: ||C||^2 = ||F^H F||^2, ||Im C||^2 = (||C||^2 - Re tr(G conj(G)))/2 with G = F^T F. nxr's `im_fraction`. */
double quadratureFractionFromFactor(const MatrixXcd& F);
/** ||Im C||_F / ||C||_F of a stored Hermitian matrix C. nxr's `lagged_fraction`. */
double quadratureFraction(const MatrixXcd& C);

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
/** The same for every group of rows of A [n x m] (rowGroup[i] in [0, groups) or −1: no group) and every matrix of Cs: out.total and out.quadrature are [groups x Cs.size()], NaN for a group with no row or
 *  a matrix with no value (an empty 0 x 0 matrix in Cs). The Gram of each group is formed once. Throws std::invalid_argument on a shape mismatch.
 *  AREA WEIGHTS (rowWeights [n], empty: all 1): the sums become Σ_ij w_i w_j |T_ij|² and Σ_ij w_i w_j |Im T_ij|² (w the vertex areas: a tile's pairs of vertices count by the mass they stand for, so the value does not
 *  change when a region is sampled more finely). With W = diag(w) on the group's rows, ‖W^½ B X Bᵀ W^½‖² = tr(X G X G) for real X, G = Bᵀ W B: the same trace form with the weighted Gram, nothing else changes. */
struct GroupedCongruenceNorms { MatrixXd total, quadrature; };
GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights = VectorXd());

/** The same for the BLOCKS BETWEEN groups: for each pair (a, b) of groups, T_ab = B_a C B_bᵀ (the part of the source cross-spectrum that couples tile a with tile b; a wave running from one tile into the next is there),
 *  Σ_ij w_i w'_j |T_ab,ij|² and Σ_ij w_i w'_j |Im T_ab,ij|² without forming T_ab. ‖W_a^½ B_a X B_bᵀ W_b^½‖² = tr(X G_b Xᵀ G_a) for real X (G_a = B_aᵀ W_a B_a): with C = Cr + i Cq, Cr symmetric and Cq
 *  antisymmetric, ‖Re T_ab‖² = tr(Cr G_b Cr G_a), ‖Im T_ab‖² = −tr(Cq G_b Cq G_a), ‖T_ab‖² = their sum; each trace is Σ_ij (X G_b)_ij (X G_a)_ji, so with X G_g held for every group once per C a pair costs m² and
 *  nothing more (a = b is the tile's own block, the grouped form above). T_ba is T_abᴴ: the same norms, so a pair is listed once. Also returns the groups' own blocks (as groupedCongruenceNorms).
 *  Memory: the groups' Grams and two m x m products per group at a time. Throws std::invalid_argument on a shape mismatch or a pair outside [0, groups). */
struct PairCongruenceNorms { GroupedCongruenceNorms tiles; MatrixXd pairTotal, pairQuadrature; };   // pair* are [pairs x Cs.size()], NaN for a pair with an empty group or an empty C
PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const std::vector<int>& rowGroup, int groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs,
                                        const VectorXd& rowWeights = VectorXd());
/** The same two functions with the groups as the RESOLVED SELECTION FORM (selection/resolved.h, section 8 of the selection spec): the member number of each row (-1: in no member) and the member count.
 *  A grid (a level of a dyadic grid, an elements grid) or an irregular set whose members are disjoint, resolved by resolve_member_of; the groups are the members, an empty member included (its row is NaN). */
GroupedCongruenceNorms groupedCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<MatrixXcd>& Cs, const VectorXd& rowWeights = VectorXd());
PairCongruenceNorms pairCongruenceNorms(const MatrixXd& A, const selection::MemberOfElement& groups, const std::vector<std::pair<int, int>>& pairs, const std::vector<MatrixXcd>& Cs,
                                        const VectorXd& rowWeights = VectorXd());

// ── quadrature pairs ───────────────────────────────────────────────────────────────────────────────────────────────

struct QuadraturePair {
    VectorXd p, q;  // [N] unit, orthogonal; p leads q by a quarter cycle (see above)
    double sigma = 0;   // the pair's quadrature amplitude (units of the spectrum Im C; nxr's sigma is this over ||Im C||_F)
};
/** The k strongest pairs of Im C = Q S Q^T, strongest first (fewer if Im C has fewer nonzero pairs). From the Hermitian matrix -iS, whose eigenvalues are +-sigma. */
std::vector<QuadraturePair> quadraturePairs(const QuadratureFactor& f, int k);
/** The strongest pair. */
QuadraturePair quadraturePair(const QuadratureFactor& f);
/** nxr's `planes(Q, S, k)` and `top_plane(F)` through the same code: the pairs straight from a complex factor. */
std::vector<QuadraturePair> quadraturePairs(const MatrixXcd& F, int k);
/** The share of ||Im C||_F^2 the pairs explain (see the header). */
double quadratureShare(const QuadratureFactor& f, const std::vector<QuadraturePair>& pairs);

/** The rotary power of a pair in the factor's patterns: P+ = ||F^T w||^2 and P- = ||F^H w||^2 with w = p + iq (the power of z = p.x + i q.x at +f and -f, summed over the factor's columns),
 *  and the sense (+1 when P+ >= P-). For a pair of Im(F F^H) as quadraturePairs gives it, P+ - P- = 4 sigma > 0: the sense is always +1; the numbers are the check. */
struct RotaryPower { double positive = 0, negative = 0; int sense = 1; };
RotaryPower rotaryPower(const MatrixXcd& F, const VectorXd& p, const VectorXd& q);

// ── standing or circular ───────────────────────────────────────────────────────────────────────────────────────────

/** rho(u) = |u^T u| / (u^H u) in [0, 1]; NaN for the zero pattern. */
double impropriety(const VectorXcd& u);

/** The rotary spectrum by complex Morlet transform: power(f, t) = |(z * w_{+f})(t)|^2 and |(z * w_{-f})(t)|^2, w_{+-f}(s) = exp(-s^2/(2 sd^2)) e^{+-2 pi i f s} / sum exp(-s^2/(2 sd^2)),
 *  sd = ncycles / (2 pi |f|), s in [-4 sd, 4 sd) at the sampling step ('same' length and centering as scipy.signal.fftconvolve). A standing pattern is symmetric. nxr's `rotary_power`. */
struct RotarySpectrum {
    MatrixXd positive;  // [freqs x T] at +f: z turning p -> q
    MatrixXd negative;  // [freqs x T] at -f
};
RotarySpectrum rotarySpectrum(const VectorXcd& z, double sfreq, const std::vector<double>& freqs, double ncycles = 7.0);

// ── a read-out's top pair as sensor filters ────────────────────────────────────────────────────────────────────────

struct QuadraturePairFilters {
    VectorXd fx, fy;    // [C] sensor filters: the pair's patterns read back to the sensors, fx = W^T p, fy = W^T q (negated if the sense came out -1)
    int sense = 1;      // +1: the rotation fx -> fy is at +f (counter-clockwise); -1 when the pair's rotary power was larger at -f (then fy is negated)
    double sigma = 0;   // the pair's amplitude (of Im(W S W^H))
    double positive = 0, negative = 0;   // the rotary power of z = fx.x + i fy.x at +f and -f in the band: w^T S conj(w), w^H S w, w = fx + i fy
};
/** The top quadrature pair of the read-out W [R x C] (rows: the modes' weights, a kernel's rows ...) of a band's sensor cross-spectrum S [C x C] (Hermitian): F = W L, S = L L^H.
 *  nxr's `rotation_plane`, from the band's cross-spectrum instead of a time series. */
QuadraturePairFilters quadraturePairFilters(const MatrixXd& W, const MatrixXcd& S);

// ── the direction a pattern moves in ───────────────────────────────────────────────────────────────────────────────

/** Area [nF] of each face of a mesh (positions [nV x 3], faces [nF x 3]). */
VectorXd faceAreas(const hodge::RowMat3& positions, const hodge::FaceMat& faces);

struct PhaseGradient {
    double direction = 0;                   // radians: the angle of the pattern's travel direction, -sum A_f P_f, in the ANCHOR's gauge (from its reference axis e1 toward e2 = n x e1); the gauge angle is always reported
    double phaseGradientAlignment = 0;      // T = |sum A_f P_f| / sum A_f |z_f| |grad z_f|, the sum of the TRANSPORTED vectors: 1 one plane wave, 0 a standing pattern
    double directionFromAnterior = std::numeric_limits<double>::quiet_NaN();   // only when `anatomy` was given: the travel direction as atan2(second, anterior) at the anchor (frames::angleFromAnterior of anatomicalComponents); NaN if unreliable there
    Eigen::Vector3d weightedSum = Eigen::Vector3d::Zero();   // the transported sum as an ambient vector in the anchor's tangent plane
};
/** The anatomical frame of the surface (frames::anatomicalFrame / carryAnatomical) with the reference frame it was measured from (e1, e2 [V x 3]). */
struct AnatomicalFrame { const frames::Anatomical* anatomical; const frames::Mat3* e1; const frames::Mat3* e2; };
/** The area-weighted phase gradient of the complex map z [V] over the LOG-MAP DISC of radius `radius` round vertex `anchor` (nxr::chart: heat distance on the intrinsic mesh). On a face f: z_f = mean
 *  of z at its vertices, grad z_f the face gradient of the hat functions (FaceOperators::grad), P_f = Im(conj(z_f) grad z_f) = |z|^2 grad arg z. For a wave a e^{-i k.x} P = -|a|^2 k and the pattern,
 *  Re[z e^{i phi}] with phi rising in time, travels along -P. Zero for ANY standing pattern (z real up to one phase).
 *  P_f lives in its face's plane, so it is never summed as a raw 3-D vector (across a sulcal wall such sums cancel): each face's P_f is read as an angle at each of its vertices in the disc (a third of the
 *  face's area each; ChartKit::gaugeHeading) and carried to the anchor's tangent plane by the vector heat transport (ChartKit::transported: angle at the anchor = angle at w - arg X_w), then summed there.
 *  `direction` is atan2(-S.e2, -S.e1) of that sum in the anchor's gauge; with `anatomy` (the surface's anatomical frame and its reference e1, e2) `directionFromAnterior` is also reported, read through frames::anatomicalComponents / angleFromAnterior at the anchor.
 *  `maxRadius` the chart's validity radius (ChartKit::disc refuses a larger `radius`). A disc whose faces carry no |z| |grad z| has alignment 0. `faces` and `ops` are the mesh `kit` was made from. */
PhaseGradient weightedPhaseGradient(const VectorXcd& z, const hodge::FaceOperators& ops, const hodge::FaceMat& faces, const VectorXd& areas, chart::ChartKit& kit, int anchor, double radius,
                                    const AnatomicalFrame* anatomy = nullptr, double maxRadius = std::numeric_limits<double>::infinity());
/** The same over a NEIGHBOURHOOD given as a selection: `disc` is the Elements form of a `distance` selection (centre vertex `anchor`, a radius) resolved by selection::resolve_distance, which always holds
 *  the centre; the disc is its member 0 (the one member of the resolved distance). The radius is not a parameter here: the neighbourhood is the selection. */
PhaseGradient weightedPhaseGradient(const VectorXcd& z, const hodge::FaceOperators& ops, const hodge::FaceMat& faces, const VectorXd& areas, chart::ChartKit& kit, int anchor,
                                    const selection::Elements& disc, const AnatomicalFrame* anatomy = nullptr);
/** Z rotated so that sum Z^2 is real and positive, its real part kept, then rotated back: the standing pattern with the same real part up to its best phase (alignment 0 by construction). */
VectorXcd standingSurrogate(const VectorXcd& z);

// ── directions ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** R = |mean e^{i a}| in [0, 1]. */
double meanResultantLength(const VectorXd& angles);
/** The mean direction arg mean e^{i a} (rad). */
double meanDirection(const VectorXd& angles);
struct Rayleigh { double R = 0, p = 1, meanDirection = 0; std::int64_t n = 0; };
/** Rayleigh's test of uniformity: R, and p ~ exp(-n R^2) (the large-sample approximation nxr used), and the mean direction. */
Rayleigh rayleighTest(const VectorXd& angles);

// ── delays ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/** arg C_ij in (-pi, pi], rad, for every entry: positive where row i leads row j. */
MatrixXd phaseLag(const MatrixXcd& C);
double phaseLag(std::complex<double> cij);
/** arg C_ij / (2 pi f), seconds: row i leads row j by this time (negative: lags). Unambiguous within half a cycle (|delay| < 1 / (2 f)). */
MatrixXd timeLag(const MatrixXcd& C, double frequencyHz);
double timeLag(std::complex<double> cij, double frequencyHz);

}  // namespace nxr::quadrature
