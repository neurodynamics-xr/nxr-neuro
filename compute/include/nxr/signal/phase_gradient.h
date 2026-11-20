#pragma once
// nxr::signal — the PHASE of a complex field on a surface, POINTWISE (per vertex, per sample): how the timing of an oscillation changes across the cortex.
// The formulas are nxr's phase reader (nxr/detect/core_phase.py, PhaseReader.read_out: grad phi = Im(conj(z) grad z) / |z|^2, no unwrapping; velocity
// u = -omega grad phi / |grad phi|^2), the reference; the mesh gradient is its `gradient_and_kernel` GV (the P1 face gradients, area-averaged onto the
// vertices).
//
// THE CARDINAL RULE (CONTRIBUTING.md): every function here is POINTWISE: nothing is summed or averaged across samples or vertices. Over a span of time or a
// region, a phase gradient is a reduction, and reductions belong to the Selection system: measure the phase current and the power over a partition (time
// and/or space), then take the ratio at read (a derived measure).
//
// A COMPLEX FIELD Z holds `components` complex numbers a vertex (1: a scalar such as the normal current; 3: a vector such as a free-orientation current),
// row v·components + a, one column a sample (a frame, a segment's coefficient: the function does not know). With several components the formulas sum over
// them (Im(Jᴴ ∇J), |J|², |∇J|²): the phase of the vector as a whole, independent of the axes, and blind to how its orientation turns from vertex to vertex.
//
//   vertexGradient(P, F)              the mesh gradient: three [V x V] operators (x, y, z), each the area-weighted mean of the incident faces' P1 gradients
//                                     (nxr's GV). Positions in any length unit; the gradients are per that unit.
//   power(Z, c)                       |z|^2 per vertex and sample                                                 [V x m]
//   phaseCurrent(Z, c, G, n)          j = Im(z̄ ∇z), on the tangent plane (n the unit vertex normals): THE PHASE CURRENT, |z|^2 ∇φ   [3V x m] (x y z a vertex)
//                                     (the same quantity on the edges of an intrinsic complex: intrinsic_forms.h's phaseCurrent(c, z); the operator passed is the
//                                     caller's choice of discretisation, G here the reference)
//   gradientPower(Z, c, G, n)         |∇z|^2 of the tangent gradient                                              [V x m]
//   phaseGradient(j, p)               ∇φ = j / |z|^2 (rad per length; points AGAINST the travel); NaN where |z|^2 = 0                  [3V x m]
//   phaseVelocity(g, omega)           u = −ω ∇φ / |∇φ|^2 (length per second, along the travel) and its length, the speed; NaN where ∇φ = 0
//   travellingIndex(j, p, q)          |j| / sqrt(|z|^2 |∇z|^2), 0 … 1 (Cauchy–Schwarz): 1 one plane wave, 0 a standing pattern; NaN where p q = 0
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cstdint>
#include <vector>

#include <limits>

#include "nxr/geometry/charts.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/frames.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/core/algebra.h"
#include "nxr/geometry/mesh.h"
#include "nxr/selection/resolved.h"

namespace nxr::signal {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::VectorXd;

using Gradient = std::array<Eigen::SparseMatrix<double>, 3>;

/** the mesh's vertex gradient (nxr's GV): P [V x 3] positions, F [nF x 3] faces (CCW) -> G[d] [V x V], d = x, y, z */
Gradient vertexGradient(const Eigen::MatrixXd& P, const Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor>& F);

/** |z|^2 summed over the components: [V x m] */
Eigen::MatrixXd power(const Eigen::MatrixXcd& Z, int components);
/** the phase current Im(z̄ ∇z) summed over the components, projected on the tangent plane of the unit normals N [V x 3]: [3V x m] */
Eigen::MatrixXd phaseCurrent(const Eigen::MatrixXcd& Z, int components, const Gradient& G, const Eigen::MatrixXd& N);
/** |∇z|^2 of the tangent gradient, summed over the components: [V x m] */
Eigen::MatrixXd gradientPower(const Eigen::MatrixXcd& Z, int components, const Gradient& G, const Eigen::MatrixXd& N);

/** ∇φ = j / p: [3V x m] from j [3V x m] and p [V x m] */
Eigen::MatrixXd phaseGradient(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p);
struct Velocity {
    Eigen::MatrixXd velocity;   // [3V x m]
    Eigen::MatrixXd speed;      // [V x m]
};
/** u = −ω g / |g|^2 and |u| = ω / |g|, ω in rad/s */
Velocity phaseVelocity(const Eigen::MatrixXd& g, double omega);
/** |j| / sqrt(p q): [V x m] */
Eigen::MatrixXd travellingIndex(const Eigen::MatrixXd& j, const Eigen::MatrixXd& p, const Eigen::MatrixXd& q);

// Moved from quadrature.h (toolbox refactor): declarations unchanged.
struct PhaseGradient {
    double direction = 0;                   // radians: the angle of the pattern's travel direction, -sum A_f P_f, in the ANCHOR's gauge (from its reference axis e1 toward e2 = n x e1); the gauge angle is always reported
    double phaseGradientAlignment = 0;      // T = |sum A_f P_f| / sum A_f |z_f| |grad z_f|, the sum of the TRANSPORTED vectors: 1 one plane wave, 0 a standing pattern
    double directionFromAnterior = std::numeric_limits<double>::quiet_NaN();   // only when `anatomy` was given: the travel direction as atan2(second, anterior) at the anchor (frames::angleFromAnterior of anatomicalComponents); NaN if unreliable there
    Eigen::Vector3d weightedSum = Eigen::Vector3d::Zero();   // the transported sum as an ambient vector in the anchor's tangent plane
};
/** The anatomical frame of the surface (frames::anatomicalFrame / carryAnatomical) with the reference frame it was measured from (e1, e2 [V x 3]). */
struct AnatomicalFrame { const geometry::Anatomical* anatomical; const geometry::Mat3* e1; const geometry::Mat3* e2; };
/** The area-weighted phase gradient of the complex map z [V] over the LOG-MAP DISC of radius `radius` round vertex `anchor` (nxr::chart: heat distance on the intrinsic mesh). On a face f: z_f = mean
 *  of z at its vertices, grad z_f the face gradient of the hat functions (FaceOperators::grad), P_f = Im(conj(z_f) grad z_f) = |z|^2 grad arg z. For a wave a e^{-i k.x} P = -|a|^2 k and the pattern,
 *  Re[z e^{i phi}] with phi rising in time, travels along -P. Zero for ANY standing pattern (z real up to one phase).
 *  P_f lives in its face's plane, so it is never summed as a raw 3-D vector (across a sulcal wall such sums cancel): each face's P_f is read as an angle at each of its vertices in the disc (a third of the
 *  face's area each; ChartKit::gaugeHeading) and carried to the anchor's tangent plane by the vector heat transport (geometry::transportedEach: angle at the anchor = angle at w - arg X_w), then summed there.
 *  `direction` is atan2(-S.e2, -S.e1) of that sum in the anchor's gauge; with `anatomy` (the surface's anatomical frame and its reference e1, e2) `directionFromAnterior` is also reported, read through frames::anatomicalComponents / angleFromAnterior at the anchor.
 *  `maxRadius` the chart's validity radius (geometry::discs refuses a larger `radius`). A disc whose faces carry no |z| |grad z| has alignment 0. `faces` and `ops` are the mesh `kit` was made from. */
PhaseGradient weightedPhaseGradient(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit, int anchor, double radius,
                                    const AnatomicalFrame* anatomy = nullptr, double maxRadius = std::numeric_limits<double>::infinity());
/** The same over a NEIGHBOURHOOD given as a selection: `disc` is the Elements form of a `distance` selection (centre vertex `anchor`, a radius) resolved by selection::resolve_distance, which always holds
 *  the centre; the disc is its member 0 (the one member of the resolved distance). The radius is not a parameter here: the neighbourhood is the selection. */
PhaseGradient weightedPhaseGradient(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit, int anchor,
                                    const selection::Elements& disc, const AnatomicalFrame* anatomy = nullptr);

/** The same for many anchors in one call: one vector heat solver and one heat solver for all of them (the batch of `weightedPhaseGradient`; the same values, bit for bit). */
std::vector<PhaseGradient> weightedPhaseGradients(const VectorXcd& z, const operators::FaceOperators& ops, const operators::FaceMat& faces, const VectorXd& areas, const geometry::ChartKit& kit,
                                                  const std::vector<int>& anchors, double radius, const AnatomicalFrame* anatomy = nullptr, double maxRadius = std::numeric_limits<double>::infinity());

// ── THE PHASE GRADIENT OF A BAND from the sensor cross-spectrum (from meeg/wave.h): the spectral form of nxr's phase reader ──
// PhaseReader.read_out: grad phi = Im(conj(z) grad z) / |z|^2, no unwrapping; velocity u = -omega grad phi / |grad phi|^2), averaged over the band's
// segments and bins with power weights. Whether the activity moves across the cortex, which way and how fast, without forming a source time series.
//
// A scalar source signal z_v = a_v X (a_v [1 x M] its row: the normal current n_v^T K_v, or one ambient component of K_v; X a segment's sensor Fourier
// coefficients at a bin) and its tangent gradient g_v = B_v X (B_v [3 x M]: the mesh gradient of the rows a, projected on the tangent plane at v). With
// the band's cross-spectrum C = S + i Sq, C_ab = <X_a conj(X_b)>:
//   power       P_v = <|z_v|^2>             = a_v S a_v^T
//   phase sum   H_v = Im <conj(z_v) g_v>     = B_v Sq a_v^T            (3-vector: the power-weighted sum of Im(conj z grad z))
//   gradient    Q_v = <|g_v|^2>             = tr(B_v S B_v^T)
// SUMS: they add over segments, bins, signals (the three ambient components) and vertices, so a region's or a group's readout is a ratio of its sums.
// Read from them (phaseReadout):
//   grad phi    H / P                         rad per length: the power-weighted mean phase gradient (points AGAINST the travel: phi = omega t - k.x)
//   velocity    -omega (H/P) / |H/P|^2         the phase velocity, length per second, along the travel
//   travelling  |H| / sqrt(P Q)               0 … 1 (Cauchy-Schwarz): 1 a single plane wave (all of the gradient is a phase gradient), 0 a standing pattern
//                                             (a real spatial pattern times e^{i omega t}) or waves going opposite ways that cancel
struct PhaseSums {
  Eigen::VectorXd power;          // [N] P
  Eigen::MatrixXd phase;          // [N x 3] H (x y z)
  Eigen::VectorXd gradientPower;  // [N] Q
};
/** A [N x M] the signal's rows, B[3] [N x M] each the x y z of their tangent gradients, S and Sq [M x M] the band's cross-spectrum. */
PhaseSums phaseSums(const Eigen::MatrixXd& A, const Eigen::MatrixXd B[3], const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq);
/** a += b (the sums of another signal, segment or band). */
void addPhaseSums(PhaseSums& a, const PhaseSums& b);

struct PhaseReadout {
  Eigen::MatrixXd gradient;       // [N x 3] H / P, rad per length (NaN where P = 0)
  Eigen::MatrixXd velocity;       // [N x 3] -omega (H/P) / |H/P|^2 (NaN where H = 0)
  Eigen::VectorXd speed;          // [N] omega / |H/P| (NaN where H = 0)
  Eigen::VectorXd travelling;     // [N] |H| / sqrt(P Q) (NaN where P Q = 0)
};
/** omega in rad/s: the speed is in the positions' length per second. */
PhaseReadout phaseReadout(const PhaseSums& s, double omega);


/** The phase current H [N x 3] (x y z) read in each vertex's tangent frame: j1 = H . e1, j2 = H . e2 (e1 north, e2 west of the canonical frames) and its length |j| = hypot(j1, j2); `e1`, `e2` [N x 3]. A vertex whose frame is
 *  not finite (a singular vertex of the registered sphere) gets NaN in all three: the caller leaves it out of every sum it makes. Throws std::invalid_argument on a shape mismatch. */
struct PhaseCurrentFrames {
  Eigen::VectorXd j1, j2, magnitude;   // [N]
};
PhaseCurrentFrames phaseCurrentInFrames(const Eigen::MatrixXd& H, const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2);

}  // namespace nxr::signal
