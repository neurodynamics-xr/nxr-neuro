#include "nxr/select/scale_selection.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::select::scale_selection {

LevelChoice selectLevel(const Eigen::Ref<const Eigen::MatrixXd>& power, const Eigen::Ref<const Eigen::VectorXd>& norm2, int levels) {
  if (levels < 1) throw std::invalid_argument("selectLevel: at least one level");
  if (power.rows() % levels != 0) throw std::invalid_argument("selectLevel: " + std::to_string(power.rows()) + " rows are not a whole number of " + std::to_string(levels) + " levels");
  if (norm2.size() != power.rows()) throw std::invalid_argument("selectLevel: " + std::to_string(norm2.size()) + " norms for " + std::to_string(power.rows()) + " rows");
  for (Eigen::Index i = 0; i < norm2.size(); ++i)
    if (!(std::isfinite(norm2(i)) && norm2(i) > 0)) throw std::invalid_argument("selectLevel: the squared norm of row " + std::to_string(i) + " is " + std::to_string(norm2(i)) + " (it must be finite and above 0)");
  const Eigen::Index E = power.rows() / levels, F = power.cols();
  LevelChoice out;
  out.index.resize(E, F);
  out.response.resize(E, F);
  for (Eigen::Index j = 0; j < F; ++j)
    for (Eigen::Index e = 0; e < E; ++e) {
      double best = -1;
      int at = 0;
      for (int l = 0; l < levels; ++l) {
        const double p = power(e * levels + l, j);
        if (!(std::isfinite(p) && p >= 0)) throw std::invalid_argument("selectLevel: the power at row " + std::to_string(e * levels + l) + ", column " + std::to_string(j) + " is " + std::to_string(p) + " (it must be finite and not below 0)");
        const double r = p / norm2(e * levels + l);
        if (r > best) { best = r; at = l; }
      }
      out.index(e, j) = at;
      out.response(e, j) = best;
    }
  return out;
}

}  // namespace nxr::select::scale_selection
