// nxr::signal — the aperiodic background of a spectrum (see the header).
#include "nxr/signal/aperiodic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }
}  // namespace

AperiodicFit fitAperiodic(const Eigen::VectorXd& power, const Eigen::VectorXd& f, bool withKnee, int iterations, double threshold, double low, double high) {
  if (power.size() != f.size()) bad("fitAperiodic: " + std::to_string(power.size()) + " powers, " + std::to_string(f.size()) + " frequencies");
  const Eigen::Index n = f.size();
  std::vector<char> inRange(static_cast<std::size_t>(n));
  Eigen::VectorXd y(n), logF(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    inRange[static_cast<std::size_t>(i)] = f(i) >= low && f(i) <= high;
    if (inRange[static_cast<std::size_t>(i)] && !(f(i) > 0)) bad("fitAperiodic: a frequency in the range is not positive");
    y(i) = std::log10(std::max(power(i), 1e-300));
    logF(i) = f(i) > 0 ? std::log10(f(i)) : 0.0;
  }
  std::vector<char> keep = inRange;
  double offset = 0, exponent = 0, knee = 0;
  auto model = [&](Eigen::Index i) { return withKnee ? offset - std::log10(knee + std::pow(f(i), exponent)) : offset - exponent * logF(i); };
  auto indices = [&]() { std::vector<Eigen::Index> idx; for (Eigen::Index i = 0; i < n; ++i) if (keep[static_cast<std::size_t>(i)]) idx.push_back(i); return idx; };
  for (int it = 1; it <= iterations; ++it) {
    const std::vector<Eigen::Index> idx = indices();
    if (idx.size() < 2) bad("fitAperiodic: fewer than 2 points to fit");
    if (withKnee) {
      double best = std::numeric_limits<double>::infinity();
      double bestOffset = 0, bestExponent = 1, bestKnee = 0;
      for (int c = 1; c <= 60; ++c) {
        const double e = c / 10.0;
        for (int k = -1; k <= 10; ++k) {
          const double kk = k < 0 ? 0.0 : std::pow(10.0, -2.0 + 0.5 * k);
          double mean = 0;
          for (const auto i : idx) mean += y(i) + std::log10(kk + std::pow(f(i), e));
          mean /= static_cast<double>(idx.size());
          double err = 0;
          for (const auto i : idx) { const double r = y(i) - (mean - std::log10(kk + std::pow(f(i), e))); err += r * r; }
          if (err < best) { best = err; bestOffset = mean; bestExponent = e; bestKnee = kk; }
        }
      }
      offset = bestOffset; exponent = bestExponent; knee = bestKnee;
    } else {
      double mx = 0, my = 0;
      for (const auto i : idx) { mx += logF(i); my += y(i); }
      mx /= static_cast<double>(idx.size()); my /= static_cast<double>(idx.size());
      double sxy = 0, sxx = 0;
      for (const auto i : idx) { sxy += (logF(i) - mx) * (y(i) - my); sxx += (logF(i) - mx) * (logF(i) - mx); }
      const double slope = sxy / sxx;
      offset = my - slope * mx; exponent = -slope;
    }
    if (it == iterations) break;
    std::vector<double> residual;
    for (const auto i : idx) residual.push_back(y(i) - model(i));
    double mean = 0; for (const double r : residual) mean += r; mean /= static_cast<double>(residual.size());
    double var = 0; for (const double r : residual) var += (r - mean) * (r - mean);
    const double sd = residual.size() > 1 ? std::sqrt(var / static_cast<double>(residual.size() - 1)) : 0.0;
    if (!(sd > 0)) break;
    std::vector<char> next = keep;
    for (std::size_t j = 0; j < idx.size(); ++j) if (residual[j] > threshold * sd) next[static_cast<std::size_t>(idx[j])] = 0;
    if (std::count(next.begin(), next.end(), 1) < 4) { keep = inRange; break; }
    keep = next;
  }
  const std::vector<Eigen::Index> idx = indices();
  double mean = 0; for (const auto i : idx) mean += y(i); mean /= static_cast<double>(idx.size());
  double rss = 0, tss = 0;
  for (const auto i : idx) { const double r = y(i) - model(i); rss += r * r; tss += (y(i) - mean) * (y(i) - mean); }
  AperiodicFit out;
  out.offset = offset; out.exponent = exponent; out.knee = knee;
  out.r2 = 1.0 - rss / std::max(tss, 1e-300);
  out.pointsUsed = static_cast<Eigen::Index>(idx.size());
  return out;
}

Eigen::VectorXd evaluateAperiodic(const AperiodicFit& a, const Eigen::VectorXd& f) {
  Eigen::VectorXd out(f.size());
  for (Eigen::Index i = 0; i < f.size(); ++i)
    out(i) = a.knee > 0 ? std::pow(10.0, a.offset - std::log10(a.knee + std::pow(f(i), a.exponent))) : std::pow(10.0, a.offset - a.exponent * std::log10(f(i)));
  return out;
}

}  // namespace nxr::signal
