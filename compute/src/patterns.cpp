// nxr::patterns — the leading patterns of a Hermitian matrix (patterns.h).
#include "nxr/patterns.h"

#include <cmath>
#include <complex>
#include <stdexcept>

#include <Eigen/Eigenvalues>

namespace nxr::patterns {

Eigen::VectorXcd rotateToLargest(const Eigen::VectorXcd& m) {
  const std::complex<double> s = (m.array() * m.array()).sum();
  if (std::abs(s) == 0) return m;
  return m * std::polar(1.0, -0.5 * std::arg(s));
}

Leading leading(const Eigen::MatrixXcd& C, int q) {
  if (C.rows() != C.cols()) throw std::invalid_argument("leading: C is square");
  if (q < 1 || q > C.rows()) throw std::invalid_argument("leading: q in [1, n]");
  const Eigen::MatrixXcd H = 0.5 * (C + C.adjoint());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
  if (es.info() != Eigen::Success) throw std::runtime_error("leading: the eigensolver did not converge");
  const Eigen::Index n = H.rows();
  Leading out;
  out.vectors.resize(n, q);
  out.values.resize(q);
  for (int i = 0; i < q; ++i) {
    out.values(i) = es.eigenvalues()(n - 1 - i);
    out.vectors.col(i) = rotateToLargest(es.eigenvectors().col(n - 1 - i));
  }
  out.trace = H.trace().real();
  return out;
}

Eigen::VectorXd similarity(const Eigen::MatrixXcd& U, const Eigen::MatrixXcd& V) {
  if (U.rows() != V.rows() || U.cols() != V.cols()) throw std::invalid_argument("similarity: U and V are both [n x q]");
  Eigen::VectorXd out(U.cols());
  for (Eigen::Index i = 0; i < U.cols(); ++i) out(i) = std::abs(U.col(i).dot(V.col(i)));
  return out;
}

}  // namespace nxr::patterns
