#include "nxr/scaling.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::scaling {

Eigen::MatrixXd divideRows(const Eigen::Ref<const Eigen::MatrixXd>& X, const Eigen::Ref<const Eigen::VectorXd>& d) {
  if (d.size() != X.rows()) throw std::invalid_argument("divideRows: " + std::to_string(d.size()) + " divisors for " + std::to_string(X.rows()) + " rows");
  for (Eigen::Index i = 0; i < d.size(); ++i)
    if (!(std::isfinite(d(i)) && d(i) > 0)) throw std::invalid_argument("divideRows: the divisor of row " + std::to_string(i) + " is " + std::to_string(d(i)) + " (it must be finite and above 0)");
  Eigen::MatrixXd Y = X;
  for (Eigen::Index i = 0; i < Y.rows(); ++i) Y.row(i) /= d(i);
  return Y;
}

}  // namespace nxr::scaling
