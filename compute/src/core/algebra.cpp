// nxr::core: general algebra (a Hermitian factor, the antisymmetric factor of Im(F F^H), a diagonal row scaling). No domain, no I/O.
#include "nxr/core/algebra.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nxr::core {
namespace {
MatrixXd antisymmetricPart(const MatrixXd& S) { return (S - S.transpose()) * 0.5; }
}  // namespace


MatrixXcd hermitianFactor(const MatrixXcd& S, double tolerance) {
    if (S.rows() != S.cols()) throw std::invalid_argument("hermitianFactor: the matrix must be square");
    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(S);
    const VectorXd& e = es.eigenvalues();
    const double top = std::max(0.0, e.size() ? e.maxCoeff() : 0.0);
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = e.size() - 1; i >= 0; --i) if (e[i] > tolerance * top && e[i] > 0) keep.push_back(i);
    MatrixXcd L(S.rows(), static_cast<Eigen::Index>(keep.size()));
    for (std::size_t k = 0; k < keep.size(); ++k) L.col(static_cast<Eigen::Index>(k)) = es.eigenvectors().col(keep[k]) * std::sqrt(e[keep[k]]);
    return L;
}

QuadratureFactor quadratureFactor(const MatrixXcd& F) {
    const Eigen::Index N = F.rows(), n = F.cols();
    MatrixXd RJ(N, 2 * n);
    RJ.leftCols(n) = F.real();
    RJ.rightCols(n) = F.imag();
    const Eigen::Index m = std::min(N, 2 * n);
    Eigen::HouseholderQR<MatrixXd> qr(RJ);
    QuadratureFactor f;
    f.Q = qr.householderQ() * MatrixXd::Identity(N, m);
    const MatrixXd Rq = qr.matrixQR().topRows(m).triangularView<Eigen::Upper>();
    MatrixXd M = MatrixXd::Zero(2 * n, 2 * n);
    M.bottomLeftCorner(n, n).setIdentity();
    M.topRightCorner(n, n) = -MatrixXd::Identity(n, n);        // [R J] M [R J]^T = J R^T - R J^T
    f.S = antisymmetricPart(Rq * M * Rq.transpose());
    f.reference = F.squaredNorm();
    return f;
}



Eigen::MatrixXd divideRows(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::VectorXd>& d) {
  if (d.size() != X.rows()) throw std::invalid_argument("divideRows: " + std::to_string(d.size()) + " divisors for " + std::to_string(X.rows()) + " rows");
  for (Eigen::Index i = 0; i < d.size(); ++i)
    if (!(std::isfinite(d(i)) && d(i) > 0)) throw std::invalid_argument("divideRows: the divisor of row " + std::to_string(i) + " is " + std::to_string(d(i)) + " (it must be finite and above 0)");
  Eigen::MatrixXd Y = X;
  for (Eigen::Index i = 0; i < Y.rows(); ++i) Y.row(i) /= d(i);
  return Y;
}


MatrixXd symmetricSquareRoot(const MatrixXd& A) {
  if (A.rows() != A.cols()) throw std::invalid_argument("symmetricSquareRoot: a square matrix");
  Eigen::SelfAdjointEigenSolver<MatrixXd> es(0.5 * (A + A.transpose()));
  const VectorXd e = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  return es.eigenvectors() * e.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace nxr::core
