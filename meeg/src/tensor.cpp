// nxr::meeg — the cross-spectral tensor of a source current and its readings at a vertex (tensor.h).
#include "nxr/meeg/tensor.h"

#include <stdexcept>

namespace nxr::meeg {

SourceTensors sourceTensor(const Eigen::MatrixXd& K, const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq) {
  if (K.rows() % 3 != 0) throw std::invalid_argument("sourceTensor: a free-orientation kernel has three rows a source");
  if (S.rows() != K.cols() || S.cols() != K.cols() || Sq.rows() != K.cols() || Sq.cols() != K.cols())
    throw std::invalid_argument("sourceTensor: the cross-spectrum is not channels x channels");
  const Eigen::Index N = K.rows() / 3;
  const Eigen::MatrixXd KS = K * S, KQ = K * Sq;            // [3N x M]: one product a part, then three dot products a pair of rows
  SourceTensors t;
  t.sources = N;
  t.re.assign(static_cast<std::size_t>(9 * N), 0.0);
  t.im.assign(static_cast<std::size_t>(9 * N), 0.0);
  for (Eigen::Index v = 0; v < N; ++v)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        const std::size_t k = static_cast<std::size_t>(9 * v + 3 * a + b);
        t.re[k] = KS.row(3 * v + a).dot(K.row(3 * v + b));
        t.im[k] = KQ.row(3 * v + a).dot(K.row(3 * v + b));
      }
  return t;
}

namespace {
// e_a^T M e_b for a 3x3 row-major M
double form(const double* M, const std::array<double, 3>& a, const std::array<double, 3>& b) {
  double s = 0;
  for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) s += a[static_cast<std::size_t>(i)] * M[3 * i + j] * b[static_cast<std::size_t>(j)];
  return s;
}
}  // namespace

Stokes stokes(const double* re, const double* im, const std::array<double, 3>& e1, const std::array<double, 3>& e2) {
  const double p11 = form(re, e1, e1), p22 = form(re, e2, e2), p12re = form(re, e1, e2), p12im = form(im, e1, e2);
  return Stokes{p11 + p22, p11 - p22, 2 * p12re, 2 * p12im};
}

double normalPower(const double* re, const std::array<double, 3>& n) { return form(re, n, n); }

double tracePower(const double* re) { return re[0] + re[4] + re[8]; }

void crossTensor(const Eigen::MatrixXd& K, const Eigen::MatrixXd& S, const Eigen::MatrixXd& Sq, Eigen::Index i, Eigen::Index j, double* re, double* im) {
  if (K.rows() % 3 != 0) throw std::invalid_argument("crossTensor: a free-orientation kernel has three rows a source");
  if (S.rows() != K.cols() || S.cols() != K.cols() || Sq.rows() != K.cols() || Sq.cols() != K.cols())
    throw std::invalid_argument("crossTensor: the cross-spectrum is not channels x channels");
  const Eigen::Index N = K.rows() / 3;
  if (i < 0 || j < 0 || i >= N || j >= N) throw std::invalid_argument("crossTensor: a source outside the kernel");
  const Eigen::MatrixXd Ki = K.middleRows(3 * i, 3), Kj = K.middleRows(3 * j, 3);
  const Eigen::Matrix3d R = Ki * S * Kj.transpose(), Q = Ki * Sq * Kj.transpose();
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { re[3 * a + b] = R(a, b); im[3 * a + b] = Q(a, b); }
}

}  // namespace nxr::meeg
