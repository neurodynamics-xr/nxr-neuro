// nxr::scaling::divideRows: each row divided by its own divisor (checked against the plain loop), the result of a matrix cut into blocks of columns equal to the whole's (pointwise), and the refusals.
#include <iostream>
#include <random>

#include "nxr/scaling.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_scaling\n";
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  Eigen::MatrixXd X(7, 11);
  for (Eigen::Index i = 0; i < X.rows(); ++i) for (Eigen::Index j = 0; j < X.cols(); ++j) X(i, j) = nd(rng);
  Eigen::VectorXd d(7);
  for (Eigen::Index i = 0; i < 7; ++i) d(i) = 0.5 + i;
  const Eigen::MatrixXd Y = nxr::scaling::divideRows(X, d);
  double e = 0;
  for (Eigen::Index i = 0; i < 7; ++i) for (Eigen::Index j = 0; j < 11; ++j) e = std::max(e, std::abs(Y(i, j) - X(i, j) / d(i)));
  EXPECT(e == 0, "each row divided by its divisor");
  const Eigen::MatrixXd A = nxr::scaling::divideRows(X.leftCols(4), d), B = nxr::scaling::divideRows(X.rightCols(7), d);
  EXPECT((A - Y.leftCols(4)).cwiseAbs().maxCoeff() == 0 && (B - Y.rightCols(7)).cwiseAbs().maxCoeff() == 0, "blocks of columns give the whole's columns (pointwise)");
  EXPECT((nxr::scaling::divideRows(Y, d.cwiseInverse().cwiseInverse()) - Y.cwiseProduct(d.cwiseInverse().replicate(1, 11))).cwiseAbs().maxCoeff() < 1e-14, "dividing twice is dividing by the square");
  auto throws = [&](const Eigen::VectorXd& dd, const char* part) { try { nxr::scaling::divideRows(X, dd); } catch (const std::invalid_argument& ex) { return std::string(ex.what()).find(part) != std::string::npos; } return false; };
  EXPECT(throws(Eigen::VectorXd::Ones(6), "6 divisors"), "a wrong length is refused");
  Eigen::VectorXd z = d; z(3) = 0;
  EXPECT(throws(z, "row 3"), "a zero divisor is refused, naming its row");
  z = d; z(5) = std::nan("");
  EXPECT(throws(z, "row 5"), "a non-finite divisor is refused");
  return failures ? 1 : 0;
}
