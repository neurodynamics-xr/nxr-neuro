#include "nxr/signal/riesz.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::signal {

std::function<double(double)> rieszGain(const std::function<double(double)>& g, double lambdaMin) {
  return [g, lambdaMin](double l) {
    if (!(l > lambdaMin)) return 0.0;
    const double v = g(l);
    return v == 0.0 ? 0.0 : v / std::sqrt(l);
  };
}

namespace {
template <class M> M gradientOf(const Gradient& G, const M& S, int c) {
  const Eigen::Index V = G[0].rows();
  if (c < 1 || S.rows() != V * c) throw std::invalid_argument("monogenic: the field has " + std::to_string(S.rows()) + " rows, " + std::to_string(V) + " vertices x " + std::to_string(c) + " components");
  M out(3 * S.rows(), S.cols());
  for (int a = 0; a < c; ++a) {
    // the component's own rows (every c-th row), gradient per direction
    M Sa(V, S.cols());
    for (Eigen::Index v = 0; v < V; ++v) Sa.row(v) = S.row(v * c + a);
    for (int d = 0; d < 3; ++d) {
      const M Gd = G[d] * Sa;
      for (Eigen::Index v = 0; v < V; ++v) out.row((v * c + a) * 3 + d) = Gd.row(v);
    }
  }
  return out;
}
}  // namespace

Eigen::MatrixXd rieszOfScaled(const Gradient& G, const Eigen::MatrixXd& S, int components) { return gradientOf(G, S, components); }
Eigen::MatrixXcd rieszOfScaled(const Gradient& G, const Eigen::MatrixXcd& S, int components) { return gradientOf(G, S, components); }

Eigen::MatrixXd amplitude(const Eigen::MatrixXcd& Z, int zc, const Eigen::MatrixXcd& R, int rc) {
  if (zc < 1 || rc < 0 || Z.rows() % zc != 0) throw std::invalid_argument("monogenic: amplitude: bad component counts");
  const Eigen::Index V = Z.rows() / zc;
  if (R.rows() != V * rc || R.cols() != Z.cols()) throw std::invalid_argument("monogenic: amplitude: the Riesz part has " + std::to_string(R.rows()) + " x " + std::to_string(R.cols()) + ", expected " + std::to_string(V * rc) + " x " + std::to_string(Z.cols()));
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(V, Z.cols());
  for (Eigen::Index v = 0; v < V; ++v) {
    for (int a = 0; a < zc; ++a) A.row(v) += Z.row(v * zc + a).cwiseAbs2();
    for (int b = 0; b < rc; ++b) A.row(v) += R.row(v * rc + b).cwiseAbs2();
  }
  return A.cwiseSqrt();
}

Eigen::MatrixXd phase(const Eigen::MatrixXd& f, const Eigen::MatrixXd& R) {
  if (R.rows() != 3 * f.rows() || R.cols() != f.cols()) throw std::invalid_argument("monogenic: phase: R must be [3V x m] for f [V x m]");
  Eigen::MatrixXd p(f.rows(), f.cols());
  for (Eigen::Index v = 0; v < f.rows(); ++v)
    for (Eigen::Index t = 0; t < f.cols(); ++t)
      p(v, t) = std::atan2(std::sqrt(R(3 * v, t) * R(3 * v, t) + R(3 * v + 1, t) * R(3 * v + 1, t) + R(3 * v + 2, t) * R(3 * v + 2, t)), f(v, t));
  return p;
}

Eigen::MatrixXd orientation(const Eigen::MatrixXd& R) {
  if (R.rows() % 3 != 0) throw std::invalid_argument("monogenic: orientation: R must be [3V x m]");
  Eigen::MatrixXd o(R.rows(), R.cols());
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (Eigen::Index v = 0; v < R.rows() / 3; ++v)
    for (Eigen::Index t = 0; t < R.cols(); ++t) {
      const double n = std::sqrt(R(3 * v, t) * R(3 * v, t) + R(3 * v + 1, t) * R(3 * v + 1, t) + R(3 * v + 2, t) * R(3 * v + 2, t));
      for (int d = 0; d < 3; ++d) o(3 * v + d, t) = n > 0 ? R(3 * v + d, t) / n : nan;
    }
  return o;
}

}  // namespace nxr::signal
