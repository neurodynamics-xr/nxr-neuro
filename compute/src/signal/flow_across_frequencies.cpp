// nxr::signal — the flow across frequencies (flow_across_frequencies.h).
#include "nxr/signal/flow_across_frequencies.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::signal {

namespace {
// the cosine of every pair of columns in the inner product with diagonal weights (none: plain); 0 where a column is zero
Eigen::MatrixXd cosines(const Eigen::MatrixXd& columns, const Eigen::VectorXd* weights) {
  const Eigen::MatrixXd weighted = weights ? Eigen::MatrixXd(weights->asDiagonal() * columns) : columns;
  Eigen::MatrixXd gram = columns.transpose() * weighted;
  const Eigen::VectorXd norms = gram.diagonal().cwiseMax(0.0).cwiseSqrt();
  for (Eigen::Index a = 0; a < gram.rows(); ++a)
    for (Eigen::Index b = 0; b < gram.cols(); ++b) gram(a, b) = norms(a) > 0 && norms(b) > 0 ? gram(a, b) / (norms(a) * norms(b)) : 0.0;
  return gram;
}
}  // namespace

FlowAcrossFrequencies flowAcrossFrequencies(const Eigen::MatrixXd& edgeCurrent, const Eigen::VectorXd& edgeWeights, const Eigen::MatrixXd& modeCoefficients,
                                            const std::vector<bool>& exactModes, const Eigen::MatrixXd& power, const Eigen::MatrixXd& vertexCurrent,
                                            const Eigen::VectorXd& frequencies) {
  const Eigen::Index A = frequencies.size(), S = power.rows();
  if (edgeCurrent.cols() != A || modeCoefficients.cols() != A || power.cols() != A || vertexCurrent.cols() != A)
    throw std::invalid_argument("flowAcrossFrequencies: every input has one column per atom");
  if (edgeWeights.size() != edgeCurrent.rows()) throw std::invalid_argument("flowAcrossFrequencies: one edge weight per edge");
  if (static_cast<Eigen::Index>(exactModes.size()) != modeCoefficients.rows()) throw std::invalid_argument("flowAcrossFrequencies: one kind per mode");
  if (vertexCurrent.rows() != 3 * S) throw std::invalid_argument("flowAcrossFrequencies: the vertex current has 3 rows per vertex");
  if ((frequencies.array() <= 0).any()) throw std::invalid_argument("flowAcrossFrequencies: the atoms' frequencies are > 0");
  FlowAcrossFrequencies out;
  out.edgeCosine = cosines(edgeCurrent, &edgeWeights);
  Eigen::MatrixXd travel = modeCoefficients, rotation = modeCoefficients;
  for (Eigen::Index k = 0; k < modeCoefficients.rows(); ++k) (exactModes[static_cast<std::size_t>(k)] ? rotation : travel).row(k).setZero();
  out.travelCosine = cosines(travel, nullptr);
  out.rotationCosine = cosines(rotation, nullptr);
  out.directionConsistency = Eigen::VectorXd::Zero(S);
  out.speedSlope = Eigen::VectorXd::Constant(S, std::numeric_limits<double>::quiet_NaN());
  out.speedIntercept = out.speedSlope;
  out.atomsUsed = Eigen::VectorXd::Zero(S);
  const Eigen::VectorXd logFrequency = frequencies.array().log();
  for (Eigen::Index v = 0; v < S; ++v) {
    Eigen::Vector3d resultant = Eigen::Vector3d::Zero();
    double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (Eigen::Index a = 0; a < A; ++a) {
      const Eigen::Vector3d current = vertexCurrent.block(3 * v, a, 3, 1);
      const double magnitude = current.norm();
      if (!(magnitude > 0) || !(power(v, a) > 0)) continue;
      resultant += current / magnitude;
      const double logSpeed = std::log(2.0 * M_PI * frequencies(a) * power(v, a) / magnitude);
      n += 1; sx += logFrequency(a); sy += logSpeed; sxx += logFrequency(a) * logFrequency(a); sxy += logFrequency(a) * logSpeed;
    }
    out.atomsUsed(v) = n;
    if (n > 0) out.directionConsistency(v) = resultant.norm() / n;
    const double spread = n * sxx - sx * sx;
    if (n >= 2 && spread > 0) {
      out.speedSlope(v) = (n * sxy - sx * sy) / spread;
      out.speedIntercept(v) = (sy - out.speedSlope(v) * sx) / n;
    }
  }
  return out;
}

}  // namespace nxr::signal
