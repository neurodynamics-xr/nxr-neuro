// nxr::lag — lag diagnostics (lag.h).
#include "nxr/stats/lag.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace nxr::stats {

LagStatistics lagStatistics(const std::vector<Complex>& S, const Eigen::VectorXd& Pu, const Eigen::VectorXd& Pv, double n, bool debias) {
  const Eigen::Index m = static_cast<Eigen::Index>(S.size());
  if (Pu.size() != m || Pv.size() != m) throw std::invalid_argument("lagStatistics: " + std::to_string(m) + " pairs, " + std::to_string(Pu.size()) + " and " + std::to_string(Pv.size()) + " powers");
  if (!(n > 1)) throw std::invalid_argument("lagStatistics: n (independent estimates) must exceed 1");
  LagStatistics r;
  r.lag.resize(m); r.coherence.resize(m); r.stderror.resize(m); r.snr.resize(m);
  for (Eigen::Index i = 0; i < m; ++i) {
    const Complex s = S[static_cast<std::size_t>(i)];
    const double denom = std::sqrt(std::max(Pu(i), 0.0) * std::max(Pv(i), 0.0));
    double g2 = denom > 0 ? std::norm(s) / (denom * denom) : 0.0;
    if (debias) g2 = std::max(0.0, (n * g2 - 1.0) / (n - 1.0));
    g2 = std::min(g2, 1.0);
    r.lag(i) = std::arg(s);
    r.coherence(i) = std::sqrt(g2);
    r.stderror(i) = g2 > 0 ? std::sqrt((1.0 - g2) / (2.0 * n * g2)) : std::numeric_limits<double>::infinity();
    r.snr(i) = std::isfinite(r.stderror(i)) && r.stderror(i) > 0 ? std::abs(r.lag(i)) / r.stderror(i) : (g2 >= 1.0 ? std::numeric_limits<double>::infinity() : 0.0);
  }
  return r;
}

Eigen::VectorXd holonomy(const Edges& edges, const Eigen::VectorXd& lag, const Faces& faces) {
  if (lag.size() != edges.rows()) throw std::invalid_argument("holonomy: " + std::to_string(edges.rows()) + " edges, " + std::to_string(lag.size()) + " lags");
  std::map<std::pair<int, int>, double> th;
  for (Eigen::Index e = 0; e < edges.rows(); ++e) {
    const int u = edges(e, 0), v = edges(e, 1);
    th[{u, v}] = lag(e);
    th[{v, u}] = -lag(e);
  }
  Eigen::VectorXd h(faces.rows());
  for (Eigen::Index f = 0; f < faces.rows(); ++f) {
    double s = 0;
    for (int c = 0; c < 3; ++c) {
      const auto it = th.find({faces(f, c), faces(f, (c + 1) % 3)});
      if (it == th.end()) throw std::invalid_argument("holonomy: face " + std::to_string(f) + " has an edge with no lag");
      s += it->second;
    }
    double w = std::remainder(s, 2 * M_PI);
    if (w <= -M_PI) w += 2 * M_PI;
    h(f) = w;
  }
  return h;
}

Eigen::MatrixXcd circularShift(const Eigen::MatrixXcd& Z, const std::vector<std::int64_t>& shifts) {
  if (static_cast<Eigen::Index>(shifts.size()) != Z.rows()) throw std::invalid_argument("circularShift: " + std::to_string(shifts.size()) + " shifts for " + std::to_string(Z.rows()) + " rows");
  const Eigen::Index n = Z.cols();
  Eigen::MatrixXcd out(Z.rows(), n);
  for (Eigen::Index r = 0; r < Z.rows(); ++r) {
    const std::int64_t k = n > 0 ? ((shifts[static_cast<std::size_t>(r)] % n) + n) % n : 0;
    for (Eigen::Index t = 0; t < n; ++t) out(r, (t + k) % n) = Z(r, t);
  }
  return out;
}

}  // namespace nxr::stats
