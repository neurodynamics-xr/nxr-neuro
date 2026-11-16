// nxr::meeg — the PHASE GRADIENT of a band, read from the sensor cross-spectrum: the spectral form of nxr's phase reader (nxr.detect.core_phase,
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
#pragma once
#include <Eigen/Dense>

namespace nxr::meeg {

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

}  // namespace nxr::meeg
