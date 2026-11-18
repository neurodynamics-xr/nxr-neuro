// nxr::algebra — congruence on a pattern (congruence.h).
#include "nxr/core/congruence.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::core {

std::vector<std::complex<double>> congruence(const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M, const Pairs& pairs) {
  if (M.rows() != M.cols() || A.cols() != M.rows())
    throw std::invalid_argument("congruence: A is " + std::to_string(A.rows()) + " x " + std::to_string(A.cols()) + ", M " + std::to_string(M.rows()) + " x " +
                                std::to_string(M.cols()) + " (A's columns must be M's rows, M square)");
  const Eigen::MatrixXcd AM = A.cast<std::complex<double>>() * M;   // [n x m], once
  std::vector<std::complex<double>> out(static_cast<std::size_t>(pairs.rows()));
  for (Eigen::Index p = 0; p < pairs.rows(); ++p) {
    const Eigen::Index u = pairs(p, 0), v = pairs(p, 1);
    if (u < 0 || v < 0 || u >= A.rows() || v >= A.rows()) throw std::invalid_argument("congruence: a pair outside [0, " + std::to_string(A.rows()) + ")");
    // Σ_b (A M)[u, b] conj(A[v, b]); A is real, so conj(A) = A: a plain row-by-column product (Eigen's dot() would conjugate its first argument)
    out[static_cast<std::size_t>(p)] = (AM.row(u) * A.row(v).transpose().cast<std::complex<double>>())(0, 0);
  }
  return out;
}

std::vector<Eigen::MatrixXcd> blockCongruence(const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M, const Pairs& pairs, int b) {
  if (b < 1 || A.rows() % b != 0 || M.rows() != M.cols() || A.cols() != M.rows())
    throw std::invalid_argument("blockCongruence: A is " + std::to_string(A.rows()) + " x " + std::to_string(A.cols()) + ", M " + std::to_string(M.rows()) + " x " +
                                std::to_string(M.cols()) + ", blocks of " + std::to_string(b));
  const Eigen::Index n = A.rows() / b;
  const Eigen::MatrixXcd AM = A.cast<std::complex<double>>() * M;
  std::vector<Eigen::MatrixXcd> out(static_cast<std::size_t>(pairs.rows()));
  for (Eigen::Index p = 0; p < pairs.rows(); ++p) {
    const Eigen::Index u = pairs(p, 0), v = pairs(p, 1);
    if (u < 0 || v < 0 || u >= n || v >= n) throw std::invalid_argument("blockCongruence: a pair outside [0, " + std::to_string(n) + ")");
    out[static_cast<std::size_t>(p)] = AM.middleRows(u * b, b) * A.middleRows(v * b, b).transpose().cast<std::complex<double>>();   // A real: conj(A) = A
  }
  return out;
}

Eigen::MatrixXcd congruence(const Eigen::MatrixXd& A, const Eigen::MatrixXcd& M) {
  if (M.rows() != M.cols() || A.cols() != M.rows()) throw std::invalid_argument("congruence: A [n x m] and M [m x m]");
  const Eigen::MatrixXcd Ac = A.cast<std::complex<double>>();
  return Ac * M * Ac.transpose();
}


template <class Scalar>
Eigen::SparseMatrix<Scalar> sparseCongruence(const Eigen::SparseMatrix<Scalar>& projection, const Eigen::SparseMatrix<Scalar>& connections) {
  if (connections.rows() != connections.cols() || projection.cols() != connections.rows()) throw std::invalid_argument("sparseCongruence: projection is [m x n] and connections is [n x n]");
  Eigen::SparseMatrix<Scalar> projected = projection * connections;
  Eigen::SparseMatrix<Scalar> out = projected * projection.transpose();
  out.prune(Scalar(0), Scalar(0));   // entries that are exactly zero are not entries
  return out;
}

template <class Scalar>
Eigen::SparseMatrix<Scalar> sparseCongruenceBlocked(const Eigen::SparseMatrix<Scalar>& projection, const Eigen::SparseMatrix<Scalar>& connections, std::int64_t block, Scalar drop_below) {
  if (connections.rows() != connections.cols() || projection.cols() != connections.rows()) throw std::invalid_argument("sparseCongruenceBlocked: projection is [m x n] and connections is [n x n]");
  if (block < 1) throw std::invalid_argument("sparseCongruenceBlocked: block >= 1");
  const Eigen::SparseMatrix<Scalar> projected = projection * connections;
  const Eigen::SparseMatrix<Scalar> projectionTransposed = projection.transpose();
  const std::int64_t m = projection.rows();
  std::vector<Eigen::Triplet<Scalar>> kept;
  for (std::int64_t c0 = 0; c0 < m; c0 += block) {
    const std::int64_t w = std::min(block, m - c0);
    const Eigen::SparseMatrix<Scalar> part = projected * projectionTransposed.middleCols(c0, w);
    for (int k = 0; k < part.outerSize(); ++k)
      for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(part, k); it; ++it)
        if (it.value() != Scalar(0) && !(std::abs(it.value()) < drop_below)) kept.emplace_back(static_cast<int>(it.row()), static_cast<int>(c0 + k), it.value());
  }
  Eigen::SparseMatrix<Scalar> out(static_cast<int>(m), static_cast<int>(m));
  out.setFromTriplets(kept.begin(), kept.end());
  return out;
}

template Eigen::SparseMatrix<float> sparseCongruence(const Eigen::SparseMatrix<float>&, const Eigen::SparseMatrix<float>&);
template Eigen::SparseMatrix<double> sparseCongruence(const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&);
template Eigen::SparseMatrix<float> sparseCongruenceBlocked(const Eigen::SparseMatrix<float>&, const Eigen::SparseMatrix<float>&, std::int64_t, float);
template Eigen::SparseMatrix<double> sparseCongruenceBlocked(const Eigen::SparseMatrix<double>&, const Eigen::SparseMatrix<double>&, std::int64_t, double);

}  // namespace nxr::core
