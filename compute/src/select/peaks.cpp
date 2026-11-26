// nxr::select — point events on a grid (peaks.h).
#include "nxr/select/peaks.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nxr::select {

std::vector<std::pair<Eigen::Index, Eigen::Index>> localMaxima(const Eigen::Ref<const Eigen::MatrixXd>& v, const Eigen::Ref<const Eigen::VectorXd>& level) {
  if (level.size() != v.rows()) throw std::invalid_argument("localMaxima: one level a row (" + std::to_string(level.size()) + " levels for " + std::to_string(v.rows()) + " rows)");
  std::vector<std::pair<Eigen::Index, Eigen::Index>> out;
  const Eigen::Index R = v.rows(), C = v.cols();
  for (Eigen::Index i = 0; i < R; ++i)
    for (Eigen::Index j = 0; j < C; ++j) {
      const double a = v(i, j);
      if (!std::isfinite(a) || !(a >= level(i))) continue;
      bool peak = true;
      for (Eigen::Index di = -1; di <= 1 && peak; ++di)
        for (Eigen::Index dj = -1; dj <= 1; ++dj) {
          const Eigen::Index r = i + di, c = j + dj;
          if (r < 0 || r >= R || c < 0 || c >= C) continue;
          if (std::isfinite(v(r, c)) && v(r, c) > a) { peak = false; break; }
        }
      if (peak) out.push_back({i, j});
    }
  return out;
}

std::vector<std::size_t> suppress(const std::vector<PointEvent>& e) {
  std::vector<std::size_t> order(e.size());
  std::iota(order.begin(), order.end(), std::size_t(0));
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return e[a].strength > e[b].strength; });
  std::vector<std::size_t> kept;
  for (const std::size_t p : order) {
    bool near = false;
    for (const std::size_t q : kept) {
      if (e[p].group != e[q].group) continue;
      if (std::abs(e[p].t - e[q].t) < std::max(e[p].tResolution, e[q].tResolution) && std::abs(e[p].f - e[q].f) < std::max(e[p].fResolution, e[q].fResolution)) { near = true; break; }
    }
    if (!near) kept.push_back(p);
  }
  return kept;
}

}  // namespace nxr::select
