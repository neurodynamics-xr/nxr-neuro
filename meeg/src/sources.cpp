// nxr::meeg sources — see include/nxr/meeg/sources.h.
#include "nxr/meeg/sources.h"
#include <cmath>
#include <stdexcept>

namespace nxr::meeg {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static void check(const MatrixXd& K, Eigen::Index channels, int o) {
  if (o != 1 && o != 3) throw std::invalid_argument("sources: a kernel has 1 or 3 orientations per source");
  if (K.cols() != channels) throw std::invalid_argument("sources: the kernel has " + std::to_string(K.cols()) + " columns, the data " + std::to_string(channels) + " channels");
  if (K.rows() % o != 0) throw std::invalid_argument("sources: the kernel's rows are not a multiple of the orientations per source");
}

MatrixXd applyKernel(const MatrixXd& K, const MatrixXd& X, int o, Orientation orientation) {
  check(K, X.rows(), o);
  MatrixXd J = K * X;
  if (orientation == Orientation::Components) return J;
  const Eigen::Index N = J.rows() / o;
  MatrixXd out = MatrixXd::Zero(N, J.cols());
  for (Eigen::Index s = 0; s < N; ++s)
    for (int c = 0; c < o; ++c) out.row(s) += J.row(s * o + c).array().square().matrix();
  return out.array().sqrt();
}

MatrixXcd applyKernel(const MatrixXd& K, const MatrixXcd& X) {
  if (K.cols() != X.rows()) throw std::invalid_argument("sources: the kernel's columns and the coefficients' channels differ");
  return K.cast<std::complex<double>>() * X;
}

MatrixXd amplitude(const MatrixXcd& J, int o) {
  if (o != 1 && o != 3) throw std::invalid_argument("sources: a kernel has 1 or 3 orientations per source");
  if (J.rows() % o != 0) throw std::invalid_argument("sources: the rows are not a multiple of the orientations per source");
  const Eigen::Index N = J.rows() / o;
  MatrixXd out = MatrixXd::Zero(N, J.cols());
  for (Eigen::Index s = 0; s < N; ++s)
    for (int c = 0; c < o; ++c) out.row(s) += J.row(s * o + c).cwiseAbs2();
  return out.array().sqrt();
}

MatrixXd applyKernelAmplitude(const MatrixXd& K, const MatrixXcd& X, int o, Orientation orientation) {
  check(K, X.rows(), o);
  const MatrixXcd J = applyKernel(K, X);
  return orientation == Orientation::Norm ? amplitude(J, o) : MatrixXd(J.cwiseAbs());
}

}  // namespace nxr::meeg
