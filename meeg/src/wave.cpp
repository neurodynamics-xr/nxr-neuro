// nxr::meeg — the phase gradient of a band from the sensor cross-spectrum (wave.h).
#include "nxr/meeg/wave.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::meeg {

PhaseSums phaseSums(const Eigen::MatrixXd& A, const Eigen::MatrixXd B[3], const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq) {
  const Eigen::Index N = A.rows(), M = A.cols();
  if (S.rows() != M || S.cols() != M || Sq.rows() != M || Sq.cols() != M) throw std::invalid_argument("phaseSums: the cross-spectrum is not channels x channels");
  for (int d = 0; d < 3; ++d) if (B[d].rows() != N || B[d].cols() != M) throw std::invalid_argument("phaseSums: a gradient block is not the signal's shape");
  PhaseSums s;
  const Eigen::MatrixXd AS = A * S, AQ = A * Sq.transpose();     // B_v Sq a_v^T = B_v . (a_v Sq^T)
  s.power = (AS.array() * A.array()).rowwise().sum();
  s.phase.resize(N, 3);
  s.gradientPower = Eigen::VectorXd::Zero(N);
  for (int d = 0; d < 3; ++d) {
    s.phase.col(d) = (B[d].array() * AQ.array()).rowwise().sum();
    s.gradientPower += ((B[d] * S).array() * B[d].array()).rowwise().sum().matrix();
  }
  return s;
}

void addPhaseSums(PhaseSums& a, const PhaseSums& b) {
  if (a.power.size() == 0) { a = b; return; }
  if (a.power.size() != b.power.size()) throw std::invalid_argument("addPhaseSums: different vertex counts");
  a.power += b.power;
  a.phase += b.phase;
  a.gradientPower += b.gradientPower;
}

PhaseReadout phaseReadout(const PhaseSums& s, double omega) {
  const Eigen::Index N = s.power.size();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PhaseReadout r;
  r.gradient.setConstant(N, 3, nan);
  r.velocity.setConstant(N, 3, nan);
  r.speed.setConstant(N, nan);
  r.travelling.setConstant(N, nan);
  for (Eigen::Index v = 0; v < N; ++v) {
    const double P = s.power(v), Q = s.gradientPower(v);
    if (!(P > 0)) continue;
    const Eigen::RowVector3d g = s.phase.row(v) / P;
    r.gradient.row(v) = g;
    if (P * Q > 0) r.travelling(v) = s.phase.row(v).norm() / std::sqrt(P * Q);
    const double g2 = g.squaredNorm();
    if (g2 > 0) { r.velocity.row(v) = -omega * g / g2; r.speed(v) = omega / std::sqrt(g2); }
  }
  return r;
}


PhaseCurrentFrames phaseCurrentInFrames(const Eigen::MatrixXd& H, const Eigen::MatrixXd& e1, const Eigen::MatrixXd& e2) {
  const Eigen::Index N = H.rows();
  if (H.cols() != 3 || e1.rows() != N || e1.cols() != 3 || e2.rows() != N || e2.cols() != 3) throw std::invalid_argument("phaseCurrentInFrames: H, e1 and e2 are [N x 3]");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  PhaseCurrentFrames f;
  f.j1.setConstant(N, nan);
  f.j2.setConstant(N, nan);
  f.magnitude.setConstant(N, nan);
  for (Eigen::Index v = 0; v < N; ++v) {
    if (!e1.row(v).allFinite() || !e2.row(v).allFinite()) continue;
    f.j1(v) = H.row(v).dot(e1.row(v));
    f.j2(v) = H.row(v).dot(e2.row(v));
    f.magnitude(v) = std::hypot(f.j1(v), f.j2(v));
  }
  return f;
}

}  // namespace nxr::meeg
