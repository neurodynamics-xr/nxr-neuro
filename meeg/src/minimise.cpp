#include "nxr/meeg/minimise.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nxr::meeg {

using Eigen::VectorXd;

NelderMeadOptions fminsearchOptions(int n) {
  NelderMeadOptions o;
  o.relativeStep = 0.05;
  o.zeroStep = 0.00025;
  o.xTolerance = 1e-4;
  o.fTolerance = 1e-4;
  o.maxEvaluations = 200 * n;
  return o;
}

MinimiseResult nelderMead(const std::function<double(const VectorXd&)>& f, const VectorXd& x0, const NelderMeadOptions& o) {
  const int n = static_cast<int>(x0.size());
  if (n < 1) throw std::invalid_argument("nelderMead: an empty start");
  if (o.maxEvaluations < 1) throw std::invalid_argument("nelderMead: maxEvaluations must be positive");
  int evals = 0;
  auto eval = [&](const VectorXd& x) { ++evals; return f(x); };
  std::vector<VectorXd> v(static_cast<std::size_t>(n) + 1, x0);
  std::vector<double> fv(static_cast<std::size_t>(n) + 1);
  for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i) + 1](i) = x0(i) != 0 ? x0(i) * (1 + o.relativeStep) : o.zeroStep;
  for (std::size_t i = 0; i < v.size(); ++i) fv[i] = eval(v[i]);
  std::vector<std::size_t> idx(v.size());
  auto order = [&] {
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return fv[a] < fv[b]; });
    std::vector<VectorXd> nv;
    std::vector<double> nf;
    for (std::size_t i : idx) { nv.push_back(v[i]); nf.push_back(fv[i]); }
    v = nv;
    fv = nf;
  };
  order();
  bool converged = false;
  while (evals < o.maxEvaluations) {
    double dx = 0;
    for (std::size_t i = 1; i < v.size(); ++i) dx = std::max(dx, (v[i] - v[0]).cwiseAbs().maxCoeff());
    if (dx <= o.xTolerance && fv.back() - fv[0] <= o.fTolerance) { converged = true; break; }
    VectorXd xbar = VectorXd::Zero(n);
    for (int i = 0; i < n; ++i) xbar += v[static_cast<std::size_t>(i)];
    xbar /= n;
    const VectorXd& worst = v.back();
    const VectorXd xr = xbar + o.reflection * (xbar - worst);
    const double fr = eval(xr);
    bool shrinkNow = false;
    if (fr < fv[0]) {
      const VectorXd xe = xbar + o.expansion * (xr - xbar);
      const double fe = eval(xe);
      if (fe < fr) { v.back() = xe; fv.back() = fe; } else { v.back() = xr; fv.back() = fr; }
    } else if (fr < fv[static_cast<std::size_t>(n) - 1]) {
      v.back() = xr;
      fv.back() = fr;
    } else if (fr < fv.back()) {                        // outside contraction
      const VectorXd xc = xbar + o.contraction * (xr - xbar);
      const double fc = eval(xc);
      if (fc <= fr) { v.back() = xc; fv.back() = fc; } else shrinkNow = true;
    } else {                                            // inside contraction
      const VectorXd xc = xbar - o.contraction * (xbar - worst);
      const double fc = eval(xc);
      if (fc < fv.back()) { v.back() = xc; fv.back() = fc; } else shrinkNow = true;
    }
    if (shrinkNow)
      for (std::size_t i = 1; i < v.size(); ++i) { v[i] = v[0] + o.shrink * (v[i] - v[0]); fv[i] = eval(v[i]); }
    order();
  }
  return {v[0], fv[0], evals, converged};
}

}  // namespace nxr::meeg
