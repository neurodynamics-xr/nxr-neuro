// nxr::meeg — a region's signal as the first principal component of its sources (scouts.h), written from the documented behaviour of Brainstorm's scout function PCA.
#include "nxr/meeg/scouts.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::meeg {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument("nxr::meeg: " + m); }
double signOf(double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : 0.0; }
}  // namespace

Eigen::VectorXd scoutSignFlip(const Eigen::MatrixXd& orientations) {
  if (orientations.cols() != 3) bad("the orientations are not n x 3");
  if (!orientations.allFinite()) bad("an orientation holds a NaN or an infinity: the signs cannot be flipped");
  const Eigen::Index n = orientations.rows();
  if (n <= 1) return Eigen::VectorXd::Ones(n);
  // the dominant orientation: the leading eigenvector of O^T O (the first right singular vector of O); the projections O v1 carry the signs (the first left singular vector's)
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(orientations.transpose() * orientations);
  const Eigen::VectorXd proj = orientations * es.eigenvectors().col(2);
  Eigen::VectorXd flip = proj.unaryExpr([](double x) { return signOf(x); });
  Eigen::Index pos = 0, neg = 0;
  for (Eigen::Index i = 0; i < n; ++i) { pos += flip(i) > 0; neg += flip(i) < 0; }
  if (pos < neg) flip = -flip;                                   // flip the fewest series
  else if (pos == neg) {                                          // a tie: the reference keeps its SVD's sign, which no rule reproduces; the first non-zero projection is made positive
    for (Eigen::Index i = 0; i < n; ++i) if (flip(i) != 0) { if (flip(i) < 0) flip = -flip; break; }
  }
  return flip;
}

RegionComponent regionComponent(const Eigen::MatrixXd& kernelRows, const Eigen::MatrixXd& covariance, const Eigen::VectorXd& flip) {
  const Eigen::Index n = kernelRows.rows(), M = kernelRows.cols();
  if (n == 0) bad("a region with no source");
  if (covariance.rows() != M || covariance.cols() != M) bad("the covariance is not on the kernel's " + std::to_string(M) + " channels");
  if (flip.size() != n) bad("the flip has " + std::to_string(flip.size()) + " entries for " + std::to_string(n) + " sources");
  if (!kernelRows.allFinite() || !covariance.allFinite()) bad("the kernel or the covariance holds a NaN or an infinity");
  const Eigen::MatrixXd DK = flip.asDiagonal() * kernelRows;                 // the flipped rows
  Eigen::MatrixXd S = DK * (0.5 * (covariance + covariance.transpose())) * DK.transpose();
  S = 0.5 * (S + S.transpose());
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
  const Eigen::VectorXd ev = es.eigenvalues().cwiseMax(0.0);                  // ascending; round-off below zero is zero
  if (!(ev(n - 1) > 0)) bad("a region whose sources have no variance over the span");
  const Eigen::VectorXd u = es.eigenvectors().col(n - 1);
  RegionComponent out;
  out.component = u.transpose() * DK;
  Eigen::Index peak = 0;
  for (Eigen::Index i = 1; i < n; ++i) if (std::abs(u(i)) > std::abs(u(peak))) peak = i;
  out.peakSource = peak;
  out.peak = DK.row(peak);
  out.explained = std::sqrt(ev(n - 1)) / ev.cwiseSqrt().sum();
  return out;
}

double regionComponentSign(double peakMax, double peakMin, double peakMean, double componentMax, double componentMin, double componentMean) {
  // the sign of the value of largest magnitude of a series with its mean removed: up (max - mean) against down (mean - min)
  auto excursion = [](double mx, double mn, double mean) { const double up = mx - mean, down = mean - mn; return up >= down ? 1.0 : -1.0; };
  return excursion(peakMax, peakMin, peakMean) * excursion(componentMax, componentMin, componentMean);
}

}  // namespace nxr::meeg
