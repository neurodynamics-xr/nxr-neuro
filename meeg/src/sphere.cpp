#include "nxr/meeg/sphere.h"

#include <cmath>
#include <stdexcept>

namespace nxr::meeg {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {
// The algebraic fit's centre (the points already in the frame of their weighted mean, scaled).
Vector3d algebraicCentre(const MatrixXd& x, const VectorXd& w) {
  const Eigen::Index n = x.rows();
  MatrixXd A(n, 4);
  VectorXd b(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    const double s = std::sqrt(w(i));
    A.block<1, 3>(i, 0) = 2 * s * x.row(i);
    A(i, 3) = s;
    b(i) = s * x.row(i).squaredNorm();
  }
  const VectorXd p = A.completeOrthogonalDecomposition().solve(b);
  return p.head<3>();
}
}  // namespace

Sphere fitSphere(const MatrixXd& points, const VectorXd& weights, const SphereFitOptions& opt) {
  const Eigen::Index n = points.rows();
  if (points.cols() != 3 || n < 4) throw std::invalid_argument("fitSphere: at least 4 points [n x 3] are needed");
  if (!points.allFinite()) throw std::invalid_argument("fitSphere: a NaN in the points");
  VectorXd w = weights.size() ? weights : VectorXd::Ones(n);
  if (w.size() != n) throw std::invalid_argument("fitSphere: the weights do not match the points");
  if ((w.array() < 0).any() || !w.allFinite() || w.sum() <= 0) throw std::invalid_argument("fitSphere: the weights must be non-negative and not all zero");
  const Vector3d mean = (points.transpose() * w) / w.sum();
  MatrixXd x = points.rowwise() - mean.transpose();
  double scale = std::sqrt((w.array() * x.rowwise().squaredNorm().array()).sum() / w.sum());
  if (!(scale > 0)) throw std::invalid_argument("fitSphere: all points coincide");
  x /= scale;
  Vector3d c = algebraicCentre(x, w);
  if (opt.cost == SphereCost::Geometric) {
    auto cost = [&](const VectorXd& p) {
      const VectorXd d = (x.rowwise() - p.transpose()).rowwise().norm();
      const double R = (w.array() * d.array()).sum() / w.sum();
      return (w.array() * (d.array() - R).square()).sum() / w.sum();
    };
    c = fminsearch(cost, c, opt.minimiser).x;
  }
  const VectorXd d = (x.rowwise() - c.transpose()).rowwise().norm();
  double radiusScaled;
  if (opt.cost == SphereCost::Geometric) radiusScaled = (w.array() * d.array()).sum() / w.sum();
  else {
    // R^2 = k + |c|^2 with k the weighted least-squares offset for this centre
    const VectorXd r = x.rowwise().squaredNorm().array() - 2 * (x * c).array();
    radiusScaled = std::sqrt((w.array() * r.array()).sum() / w.sum() + c.squaredNorm());
  }
  return {mean + scale * c, scale * radiusScaled};
}

std::vector<Sphere> sensorSpheres(const MatrixXd& surface, const MatrixXd& sensorPoints, const VectorXd& sensorWeights, const std::vector<int>& sensorChannel, int channels,
                                  const SensorSphereOptions& opt) {
  const Eigen::Index P = sensorPoints.rows();
  if (sensorPoints.cols() != 3 || sensorWeights.size() != P || static_cast<Eigen::Index>(sensorChannel.size()) != P)
    throw std::invalid_argument("sensorSpheres: points, weights and channels do not match");
  std::vector<Sphere> out;
  for (int ch = 0; ch < channels; ++ch) {
    VectorXd w = VectorXd::Zero(surface.rows());
    bool any = false;
    for (Eigen::Index p = 0; p < P; ++p) {
      if (sensorChannel[static_cast<std::size_t>(p)] != ch) continue;
      any = true;
      const VectorXd d = (surface.rowwise() - sensorPoints.row(p)).rowwise().norm();
      if ((d.array() <= 0).any()) throw std::invalid_argument("sensorSpheres: a sensor point lies on a surface vertex");
      w += std::abs(sensorWeights(p)) * d.array().pow(-opt.weightExponent).matrix();
    }
    out.push_back(fitSphere(surface, any && w.sum() > 0 ? w : VectorXd(), opt.fit));
  }
  return out;
}

}  // namespace nxr::meeg
