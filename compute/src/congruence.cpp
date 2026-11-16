// nxr::algebra — congruence on a pattern (congruence.h).
#include "nxr/congruence.h"

#include <stdexcept>
#include <string>

namespace nxr::algebra {

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

}  // namespace nxr::algebra
