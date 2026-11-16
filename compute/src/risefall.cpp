// nxr::risefall — the double-sigmoid fit of Rheome's detect.risefall (risefall.h).
#include "nxr/risefall.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace nxr::risefall {

Model parseModel(const std::string& name) {
  if (name == "linear") return Model::Linear;
  if (name == "quadrature") return Model::Quadrature;
  if (name == "posterior") throw std::invalid_argument("risefall model 'posterior' (Rheome's grid posterior of tOn and tOff) is not built: use linear or quadrature");
  throw std::invalid_argument("risefall model '" + name + "': linear or quadrature");
}

namespace {
using Vec6 = Eigen::Matrix<double, 6, 1>;
double S(double x) { return 1.0 / (1.0 + std::exp(-x)); }
// MATLAB's round: half away from zero
double mround(double x) { return x < 0 ? -std::floor(-x + 0.5) : std::floor(x + 0.5); }
// MATLAB's movmean(x, k): the window centred (k odd) or [k/2 before, k/2 − 1 after] (k even), SHRUNK at the ends
Eigen::VectorXd movmean(const Eigen::VectorXd& x, int k) {
  const int n = static_cast<int>(x.size());
  const int before = k % 2 ? (k - 1) / 2 : k / 2, after = k % 2 ? (k - 1) / 2 : k / 2 - 1;
  Eigen::VectorXd out(n);
  for (int i = 0; i < n; ++i) { const int lo = std::max(0, i - before), hi = std::min(n - 1, i + after); out(i) = x.segment(lo, hi - lo + 1).mean(); }
  return out;
}
// MATLAB's prctile of a vector: the sorted values at the midpoints (i − 0.5)/n, linearly interpolated, clamped to the extremes
double prctile(Eigen::VectorXd y, double p) {
  std::sort(y.data(), y.data() + y.size());
  const double n = static_cast<double>(y.size()), r = p / 100.0 * n + 0.5;
  if (r <= 1) return y(0);
  if (r >= n) return y(y.size() - 1);
  const int lo = static_cast<int>(std::floor(r));
  return y(lo - 1) + (r - lo) * (y(lo) - y(lo - 1));
}
double model(const Vec6& p, double t, Model m) {
  const double g = p(1) * S((t - p(2)) / p(3)) * S(-(t - p(4)) / p(5));
  return m == Model::Linear ? p(0) + g : std::sqrt(p(0) * p(0) + g * g);
}
double cost(const Vec6& p, const Eigen::VectorXd& t, const Eigen::VectorXd& y, Model m) {
  double c = 0;
  for (Eigen::Index i = 0; i < t.size(); ++i) { const double r = model(p, t(i), m) - y(i); c += r * r; }
  return 0.5 * c;
}
// bounded Levenberg–Marquardt: the free variables (not held at a bound by the gradient) take a damped Gauss–Newton step, clipped to the bounds; central finite differences
int lsq(Vec6& p, const Vec6& lb, const Vec6& ub, const Eigen::VectorXd& t, const Eigen::VectorXd& y, Model m) {
  const Eigen::Index n = t.size();
  double lambda = 1e-3, c = cost(p, t, y, m);
  int it = 0;
  for (; it < 1000; ++it) {
    Eigen::MatrixXd J(n, 6);
    Eigen::VectorXd r(n);
    for (Eigen::Index i = 0; i < n; ++i) r(i) = model(p, t(i), m) - y(i);
    for (int k = 0; k < 6; ++k) {
      const double h = 1e-7 * std::max(1.0, std::abs(p(k)));
      Vec6 a = p, b = p;
      a(k) = std::min(ub(k), p(k) + h); b(k) = std::max(lb(k), p(k) - h);
      const double d = a(k) - b(k);
      for (Eigen::Index i = 0; i < n; ++i) J(i, k) = d > 0 ? (model(a, t(i), m) - model(b, t(i), m)) / d : 0.0;
    }
    const Vec6 g = J.transpose() * r;
    std::vector<int> fr;
    for (int k = 0; k < 6; ++k) if (!((p(k) <= lb(k) && g(k) > 0) || (p(k) >= ub(k) && g(k) < 0))) fr.push_back(k);   // a variable at a bound the gradient pushes into is held
    if (fr.empty()) break;
    Eigen::MatrixXd A(fr.size(), fr.size());
    Eigen::VectorXd gf(fr.size());
    for (std::size_t a = 0; a < fr.size(); ++a) { gf(a) = g(fr[a]); for (std::size_t b = 0; b < fr.size(); ++b) A(a, b) = J.col(fr[a]).dot(J.col(fr[b])); }
    bool moved = false;
    for (int tries = 0; tries < 30 && !moved; ++tries) {
      Eigen::MatrixXd D = A;
      for (Eigen::Index a = 0; a < D.rows(); ++a) D(a, a) += lambda * std::max(A(a, a), 1e-12);
      const Eigen::VectorXd step = D.ldlt().solve(-gf);
      Vec6 q = p;
      for (std::size_t a = 0; a < fr.size(); ++a) q(fr[a]) = std::min(ub(fr[a]), std::max(lb(fr[a]), p(fr[a]) + step(a)));
      const double cq = cost(q, t, y, m);
      if (cq < c) {
        const bool done = (c - cq) <= 1e-14 * std::max(c, 1e-300) + 1e-30 && (q - p).cwiseAbs().maxCoeff() < 1e-10;
        p = q; c = cq; lambda = std::max(lambda / 3, 1e-12); moved = true;
        if (done) return it + 1;
      } else lambda *= 4;
    }
    if (!moved) break;
  }
  return it;
}
}  // namespace

Fit fit(const Eigen::Ref<const Eigen::VectorXd>& a, double fs, double t0, double t1, const Options& o) {
  Fit R;
  if (!(fs > 0)) throw std::invalid_argument("risefall: the rate is above 0");
  const std::int64_t N = a.size();
  const std::int64_t i0 = std::max<std::int64_t>(1, static_cast<std::int64_t>(mround((t0 - o.margin) * fs)) + 1);
  const std::int64_t i1 = std::min<std::int64_t>(N, static_cast<std::int64_t>(mround((t1 + o.margin) * fs)) + 1);
  if (i1 - i0 < 10) return R;
  Eigen::VectorXd y = movmean(a.segment(i0 - 1, i1 - i0 + 1), std::max(1, static_cast<int>(mround(o.smooth * fs))));
  Eigen::VectorXd t(y.size());
  for (Eigen::Index i = 0; i < y.size(); ++i) t(i) = static_cast<double>(i0 - 1 + i) / fs;
  const double sc = y.cwiseAbs().maxCoeff();
  if (!(sc > 0) || !std::isfinite(sc)) return R;
  y /= sc;
  const double b0 = prctile(y, 5), ymax = y.maxCoeff(), ymin = y.minCoeff();
  double h0 = ymax - b0;
  Vec6 p0, lb, ub;
  p0 << b0, h0, t0, 0.1, t1, 0.1;
  lb << ymin - h0, 0, t(0), 0.01, t0, 0.01;
  if (o.model == Model::Quadrature) { h0 = std::sqrt(std::max(ymax * ymax - b0 * b0, 0.0)); p0(1) = h0; lb(0) = 0; }
  ub << ymax, 3 * h0, t1, o.tauMax, t(t.size() - 1), o.tauMax;
  for (int k = 0; k < 6; ++k) p0(k) = std::min(std::max(p0(k), lb(k)), ub(k));
  Vec6 p = p0;
  R.iterations = lsq(p, lb, ub, t, y, o.model);
  if (!(p(4) > p(2))) return R;
  double sse = 0;
  for (Eigen::Index i = 0; i < t.size(); ++i) { const double r = y(i) - model(p, t(i), o.model); sse += r * r; }
  const double ybar = y.mean(), sst = (y.array() - ybar).square().sum();
  R.b = p(0) * sc; R.h = p(1) * sc; R.tOn = p(2); R.tauR = p(3); R.tOff = p(4); R.tauF = p(5);
  R.r2 = 1 - sse / std::max(sst, std::numeric_limits<double>::min());
  R.buildup[0] = R.tOn - 2 * R.tauR; R.buildup[1] = R.tOn + 2 * R.tauR;
  R.decay[0] = R.tOff - 2 * R.tauF; R.decay[1] = R.tOff + 2 * R.tauF;
  if (R.tOff - 2 * R.tauF > R.tOn + 2 * R.tauR) { R.hasPlateau = true; R.plateau[0] = R.tOn + 2 * R.tauR; R.plateau[1] = R.tOff - 2 * R.tauF; }
  R.atBound = std::max(R.tauR, R.tauF) >= 0.99 * o.tauMax;
  R.ok = true;
  return R;
}

Stages stagesOf(double tOn, double tOff, double tauR, double tauF) {
  Stages S;
  if (!(std::isfinite(tOn) && std::isfinite(tOff) && std::isfinite(tauR) && std::isfinite(tauF) && tauR > 0 && tauF > 0)) return S;
  S.valid = true;
  S.onset[0] = tOn - 2 * tauR; S.onset[1] = tOn + 2 * tauR;
  S.offset[0] = tOff - 2 * tauF; S.offset[1] = tOff + 2 * tauF;
  if (tOff - 2 * tauF > tOn + 2 * tauR) { S.hasWithin = true; S.within[0] = tOn + 2 * tauR; S.within[1] = tOff - 2 * tauF; }
  return S;
}

bool passesGate(const Fit& f, double minR2) { return f.ok && !f.atBound && std::isfinite(f.r2) && f.r2 >= minR2; }

}  // namespace nxr::risefall
