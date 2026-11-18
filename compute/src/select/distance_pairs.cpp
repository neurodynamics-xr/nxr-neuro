// nxr::select — pairs at a distance (distance_pairs.h).
#include "nxr/select/distance_pairs.h"

#include <algorithm>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nxr::select {

PairsAt pairsAtDistance(const Eigen::MatrixXd& positions, const Edges& edges, double d, double tolerance, int perVertex, Metric metric) {
  const Eigen::Index n = positions.rows();
  if (positions.cols() != 3) throw std::invalid_argument("pairsAtDistance: positions are [n x 3]");
  if (!(d > 0) || !(tolerance >= 0) || perVertex < 1) throw std::invalid_argument("pairsAtDistance: d > 0, tolerance >= 0, perVertex >= 1");
  const double lo = d * (1 - tolerance), hi = d * (1 + tolerance);
  std::vector<std::vector<std::pair<int, double>>> adj;
  if (metric == Metric::AlongEdges) {
    adj.resize(static_cast<std::size_t>(n));
    for (Eigen::Index e = 0; e < edges.rows(); ++e) {
      const int a = edges(e, 0), b = edges(e, 1);
      if (a < 0 || b < 0 || a >= n || b >= n) throw std::invalid_argument("pairsAtDistance: edge " + std::to_string(e) + " outside the vertices");
      const double l = (positions.row(a) - positions.row(b)).norm();
      adj[static_cast<std::size_t>(a)].push_back({b, l});
      adj[static_cast<std::size_t>(b)].push_back({a, l});
    }
  }
  std::set<std::pair<int, int>> chosen;
  std::vector<double> dist;
  std::vector<std::pair<int, int>> order;
  for (Eigen::Index s = 0; s < n; ++s) {
    std::vector<std::pair<int, double>> ring;                  // candidates v > s within [lo, hi]
    if (metric == Metric::Straight) {
      for (Eigen::Index v = s + 1; v < n; ++v) {
        const double dd = (positions.row(s) - positions.row(v)).norm();
        if (dd >= lo && dd <= hi) ring.push_back({static_cast<int>(v), dd});
      }
    } else {
      std::map<int, double> best;
      std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> q;
      q.push({0.0, static_cast<int>(s)});
      best[static_cast<int>(s)] = 0;
      while (!q.empty()) {
        const auto [du, u] = q.top();
        q.pop();
        if (du > best[u]) continue;
        for (const auto& [v, l] : adj[static_cast<std::size_t>(u)]) {
          const double nd = du + l;
          if (nd > hi) continue;
          const auto it = best.find(v);
          if (it == best.end() || nd < it->second) { best[v] = nd; q.push({nd, v}); }
        }
      }
      for (const auto& [v, dd] : best) if (v > s && dd >= lo && dd <= hi) ring.push_back({v, dd});
    }
    const std::size_t step = std::max<std::size_t>(1, ring.size() / static_cast<std::size_t>(perVertex));
    int taken = 0;
    for (std::size_t i = 0; i < ring.size() && taken < perVertex; i += step, ++taken) {
      if (chosen.insert({static_cast<int>(s), ring[i].first}).second) { order.push_back({static_cast<int>(s), ring[i].first}); dist.push_back(ring[i].second); }
    }
  }
  PairsAt r;
  r.pairs.resize(static_cast<Eigen::Index>(order.size()), 2);
  r.distance.resize(static_cast<Eigen::Index>(order.size()));
  for (std::size_t i = 0; i < order.size(); ++i) { r.pairs(static_cast<Eigen::Index>(i), 0) = order[i].first; r.pairs(static_cast<Eigen::Index>(i), 1) = order[i].second; r.distance(static_cast<Eigen::Index>(i)) = dist[i]; }
  return r;
}

}  // namespace nxr::select
