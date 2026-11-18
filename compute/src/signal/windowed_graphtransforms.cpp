// wgft / iwgft: the windowed graph Fourier transform of Shuman, Ricaud & Vandergheynst (2016), with the atoms built from the eigenbasis.
#include "nxr/signal/windowed_graphtransforms.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::signal::graphfilterbank {
namespace {
void checkBasis(const EigenTransform& t) {
  if (t.mass.rows() != 0 || t.mass.cols() != 0) throw std::invalid_argument("wgft: the windowed graph Fourier transform is defined for orthonormal eigenvectors of a Laplacian (identity mass); a mass-weighted pencil is not supported");
  if (t.phi.cols() != t.lambda.size()) throw std::invalid_argument("wgft: the eigenvectors and the eigenvalues disagree in number");
}
// T[:, i] = (T_i g) as a column per vertex i: sqrt(N) Phi diag(g^) Phi^T   (symmetric: column i = row i)
Eigen::MatrixXd translations(const EigenTransform& t, const Gain& kernel) {
  const double sq = std::sqrt(static_cast<double>(t.phi.rows()));
  Eigen::VectorXd gh(t.lambda.size());
  for (Eigen::Index l = 0; l < gh.size(); ++l) gh(l) = kernel(t.lambda(l));
  return sq * t.phi * gh.asDiagonal() * t.phi.transpose();
}
}  // namespace

Eigen::MatrixXd wgft(const EigenTransform& t, const Gain& kernel, const Eigen::VectorXd& f) {
  checkBasis(t);
  const Eigen::Index N = t.phi.rows(), K = t.phi.cols();
  if (f.size() != N) throw std::invalid_argument("wgft: the signal has " + std::to_string(f.size()) + " values, the basis " + std::to_string(N) + " vertices");
  const Eigen::MatrixXd T = translations(t, kernel);
  const double sq = std::sqrt(static_cast<double>(N));
  Eigen::MatrixXd S(N, K);
  for (Eigen::Index i = 0; i < N; ++i) S.row(i) = (sq * t.phi.transpose() * f.cwiseProduct(T.col(i))).transpose();
  return S;
}

Eigen::VectorXd iwgft(const EigenTransform& t, const Gain& kernel, const Eigen::MatrixXd& S) {
  checkBasis(t);
  const Eigen::Index N = t.phi.rows(), K = t.phi.cols();
  if (S.rows() != N || S.cols() != K) throw std::invalid_argument("iwgft: the coefficients must be [vertices x frequencies] = [" + std::to_string(N) + " x " + std::to_string(K) + "]");
  const Eigen::MatrixXd T = translations(t, kernel);
  const double sq = std::sqrt(static_cast<double>(N));
  Eigen::VectorXd num = Eigen::VectorXd::Zero(N);
  for (Eigen::Index i = 0; i < N; ++i) num += T.col(i).cwiseProduct(sq * t.phi * S.row(i).transpose());   // sum_k S(i,k) g_{i,k}(m) = sqrt(N) T_i(m) sum_k phi_k(m) S(i,k)
  Eigen::VectorXd den = Eigen::VectorXd::Zero(N);
  for (Eigen::Index i = 0; i < N; ++i) den += T.col(i).cwiseAbs2();
  Eigen::VectorXd f(N);
  for (Eigen::Index m = 0; m < N; ++m) {
    if (!(den(m) > 0.0)) throw std::invalid_argument("iwgft: the kernel has no energy at vertex " + std::to_string(m) + ": the transform is not invertible there");
    f(m) = num(m) / (static_cast<double>(N) * den(m));
  }
  return f;
}

}  // namespace nxr::signal::graphfilterbank
