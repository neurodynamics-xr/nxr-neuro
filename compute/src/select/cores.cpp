// nxr::cores (cores.h): nxr `_core_patches`, per frame.
#include "nxr/select/cores.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace nxr::select {

Cores corePatches(const Eigen::Ref<const Eigen::MatrixXd>& C, const std::vector<std::pair<std::int32_t, std::int32_t>>& pairs, const Eigen::Ref<const Eigen::VectorXd>& area,
                  const Eigen::Ref<const Eigen::MatrixXd>& positions, double threshold, std::int32_t firstFrame) {
  const Eigen::Index V = C.rows(), F = C.cols();
  if (area.size() != V || positions.rows() != V || positions.cols() != 3) throw std::invalid_argument("corePatches: the areas and positions are for " + std::to_string(V) + " vertices of values");
  if (!(threshold > 0) || !std::isfinite(threshold)) throw std::invalid_argument("corePatches: the threshold is above 0");
  if (!C.allFinite()) throw std::invalid_argument("corePatches: the values hold a non-finite number");
  // the neighbours as lists (CSR), sorted so a patch is found from its lowest vertex
  std::vector<std::int32_t> deg(static_cast<std::size_t>(V) + 1, 0);
  for (const auto& [a, b] : pairs) {
    if (a < 0 || b < 0 || a >= V || b >= V) throw std::invalid_argument("corePatches: an edge names a vertex outside 0 … " + std::to_string(V - 1));
    if (a == b) continue;
    ++deg[static_cast<std::size_t>(a) + 1]; ++deg[static_cast<std::size_t>(b) + 1];
  }
  for (std::size_t i = 1; i < deg.size(); ++i) deg[i] += deg[i - 1];
  std::vector<std::int32_t> nbr(static_cast<std::size_t>(deg.back())), fill(deg.begin(), deg.end() - 1);
  for (const auto& [a, b] : pairs) { if (a == b) continue; nbr[static_cast<std::size_t>(fill[static_cast<std::size_t>(a)]++)] = b; nbr[static_cast<std::size_t>(fill[static_cast<std::size_t>(b)]++)] = a; }
  Cores out;
  std::vector<double> cen;
  std::vector<std::int32_t> stack, members;
  std::vector<std::uint8_t> seen(static_cast<std::size_t>(V));
  for (Eigen::Index f = 0; f < F; ++f) {
    for (const int sg : {1, -1}) {
      std::fill(seen.begin(), seen.end(), 0);
      for (Eigen::Index v0 = 0; v0 < V; ++v0) {
        if (seen[static_cast<std::size_t>(v0)] || sg * C(v0, f) < threshold) continue;
        members.clear(); stack.assign(1, static_cast<std::int32_t>(v0)); seen[static_cast<std::size_t>(v0)] = 1;
        while (!stack.empty()) {
          const std::int32_t v = stack.back(); stack.pop_back(); members.push_back(v);
          for (std::int32_t k = deg[static_cast<std::size_t>(v)]; k < deg[static_cast<std::size_t>(v) + 1]; ++k) {
            const std::int32_t u = nbr[static_cast<std::size_t>(k)];
            if (!seen[static_cast<std::size_t>(u)] && sg * C(u, f) >= threshold) { seen[static_cast<std::size_t>(u)] = 1; stack.push_back(u); }
          }
        }
        double W = 0, A = 0, peak = 0;
        Eigen::Vector3d c = Eigen::Vector3d::Zero();
        for (const std::int32_t v : members) { const double w = area(v) * std::abs(C(v, f)); W += w; A += area(v); c += w * positions.row(v).transpose(); peak = std::max(peak, sg * C(v, f)); }
        c /= W;
        std::int32_t cv = members.front();
        double best = std::numeric_limits<double>::infinity();
        for (const std::int32_t v : members) { const double d = (positions.row(v).transpose() - c).squaredNorm(); if (d < best || (d == best && v < cv)) { best = d; cv = v; } }
        out.frame.push_back(static_cast<std::int32_t>(firstFrame + f)); out.sense.push_back(static_cast<std::int8_t>(sg)); out.centre_vertex.push_back(cv);
        out.area.push_back(A); out.peak.push_back(peak);
        cen.insert(cen.end(), {c.x(), c.y(), c.z()});
      }
    }
  }
  out.centre = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(cen.data(), static_cast<Eigen::Index>(cen.size() / 3), 3);
  return out;
}

void append(Cores& into, const Cores& more) {
  const Eigen::Index n0 = into.centre.rows();
  into.frame.insert(into.frame.end(), more.frame.begin(), more.frame.end());
  into.sense.insert(into.sense.end(), more.sense.begin(), more.sense.end());
  into.centre_vertex.insert(into.centre_vertex.end(), more.centre_vertex.begin(), more.centre_vertex.end());
  into.area.insert(into.area.end(), more.area.begin(), more.area.end());
  into.peak.insert(into.peak.end(), more.peak.begin(), more.peak.end());
  into.centre.conservativeResize(n0 + more.centre.rows(), 3);
  if (more.centre.rows()) into.centre.bottomRows(more.centre.rows()) = more.centre;
}

}  // namespace nxr::select
