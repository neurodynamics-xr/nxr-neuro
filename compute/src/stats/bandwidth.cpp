// The bodies of timefilterbank::noiseBandwidth / equivalentBandwidth (Trapezoid) and synthesis::effectiveBandwidth (Sum): kept as the branches of the rule.
#include "nxr/stats/bandwidth.h"
#include <stdexcept>

namespace nxr::stats {

double enbw(const Eigen::VectorXd& power, double df, double reference, Rule rule) {
  if (power.size() < 2) throw std::invalid_argument("enbw: at least two points");
  double s = 0;
  for (Eigen::Index i = 0; i < power.size(); ++i) { const double w = (rule == Rule::Trapezoid && (i == 0 || i == power.size() - 1)) ? 0.5 : 1.0; s += w * power(i); }
  const double ref = reference > 0 ? reference : power.maxCoeff();
  if (!(ref > 0)) throw std::invalid_argument("enbw: the response passes nothing");
  return s * df / ref;
}

double statisticalBandwidth(const Eigen::VectorXd& power, double df, Rule rule) {
  if (power.size() < 2) throw std::invalid_argument("statisticalBandwidth: at least two points");
  double s2 = 0, s4 = 0;
  for (Eigen::Index i = 0; i < power.size(); ++i) {
    const double w = (rule == Rule::Trapezoid && (i == 0 || i == power.size() - 1)) ? 0.5 : 1.0;
    const double v2 = power(i);
    s2 += w * v2;
    s4 += w * v2 * v2;
  }
  if (!(s4 > 0)) throw std::invalid_argument("statisticalBandwidth: the spectrum holds no power");
  return rule == Rule::Trapezoid ? (s2 * df) * (s2 * df) / (s4 * df) : s2 * s2 * df / s4;
}

}  // namespace nxr::stats
