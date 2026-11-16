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

std::vector<Band> brainstormBands() {
  return {{"delta", 2, 4}, {"theta", 5, 7}, {"alpha", 8, 12}, {"beta", 15, 29}, {"gamma1", 30, 59}, {"gamma2", 60, 90}};
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

MatrixXd sourcePower(const MatrixXd& K, const std::vector<MatrixXd>& S, int o, Orientation orientation) {
  if (o != 1 && o != 3) throw std::invalid_argument("sources: a kernel has 1 or 3 orientations per source");
  if (K.rows() % o != 0) throw std::invalid_argument("sources: the kernel's rows are not a multiple of the orientations per source");
  const Eigen::Index rows = K.rows(), N = rows / o;
  MatrixXd P(orientation == Orientation::Norm ? N : rows, static_cast<Eigen::Index>(S.size()));
  for (std::size_t g = 0; g < S.size(); ++g) {
    if (S[g].rows() != K.cols() || S[g].cols() != K.cols()) throw std::invalid_argument("sources: a cross-spectrum is not channels x channels");
    const VectorXd d = ((K * S[g]).cwiseProduct(K)).rowwise().sum();
    if (orientation == Orientation::Components) { P.col(static_cast<Eigen::Index>(g)) = d; continue; }
    for (Eigen::Index s = 0; s < N; ++s) {
      double v = 0;
      for (int c = 0; c < o; ++c) v += d[s * o + c];
      P(s, static_cast<Eigen::Index>(g)) = v;
    }
  }
  return P;
}

MatrixXd relativePower(const MatrixXd& P, const VectorXd& total) {
  if (total.size() != P.rows()) throw std::invalid_argument("sources: one total per row");
  MatrixXd out = P;
  for (Eigen::Index i = 0; i < P.rows(); ++i) { if (total[i] != 0) out.row(i) = P.row(i) / total[i]; else out.row(i).setZero(); }
  return out;
}

}  // namespace nxr::meeg
