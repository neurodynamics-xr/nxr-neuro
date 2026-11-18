// nxr::select — regions on a graph (regions.h).
#include "nxr/select/regions.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nxr::select {

Sign parseSign(const std::string& name) {
  if (name == "positive") return Sign::Positive;
  if (name == "negative") return Sign::Negative;
  if (name == "magnitude") return Sign::Magnitude;
  throw std::invalid_argument("sign '" + name + "': one of positive, negative, magnitude");
}
const char* signName(Sign s) { return s == Sign::Positive ? "positive" : s == Sign::Negative ? "negative" : "magnitude"; }

Eigen::VectorXd applySign(const Eigen::Ref<const Eigen::VectorXd>& x, Sign sign) {
  if (sign == Sign::Positive) return x;
  if (sign == Sign::Negative) return -x;
  return x.cwiseAbs();
}

namespace {
// union-find with the lowest node as the root, so a component is named by its lowest node
struct Sets {
  std::vector<std::int32_t> p;
  explicit Sets(std::size_t n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  std::int32_t find(std::int32_t a) { while (p[static_cast<std::size_t>(a)] != a) { p[static_cast<std::size_t>(a)] = p[static_cast<std::size_t>(p[static_cast<std::size_t>(a)])]; a = p[static_cast<std::size_t>(a)]; } return a; }
  void join(std::int32_t a, std::int32_t b) { a = find(a); b = find(b); if (a == b) return; if (a < b) p[static_cast<std::size_t>(b)] = a; else p[static_cast<std::size_t>(a)] = b; }
};
}  // namespace

Regions detectRegions(const Eigen::Ref<const Eigen::VectorXd>& values, const std::vector<std::pair<std::int32_t, std::int32_t>>& pairs, const Eigen::Ref<const Eigen::VectorXd>& weights,
                      double enter, double exit, double minWeight, double maxWeight) {
  const std::int64_t n = values.size();
  if (weights.size() != n) throw std::invalid_argument("detectRegions: one weight per node (" + std::to_string(n) + " values, " + std::to_string(weights.size()) + " weights)");
  if (!(exit <= enter)) throw std::invalid_argument("detectRegions: the exit level is at most the entry level");
  if (!(minWeight <= maxWeight)) throw std::invalid_argument("detectRegions: the smallest weight is at most the largest");
  for (std::int64_t i = 0; i < n; ++i)
    if (!std::isfinite(weights(i)) || weights(i) < 0) throw std::invalid_argument("detectRegions: node " + std::to_string(i) + " has a weight that is negative or not finite");
  Sets s(static_cast<std::size_t>(n));
  auto in = [&](std::int32_t a) { return values(a) >= exit; };       // NaN fails the comparison: no value, in no region
  for (const auto& [a, b] : pairs) {
    if (a < 0 || b < 0 || a >= n || b >= n) throw std::invalid_argument("detectRegions: a pair names a node outside 0 … " + std::to_string(n - 1));
    if (a != b && in(a) && in(b)) s.join(a, b);
  }
  // a component by its root (its lowest node); kept when it reaches `enter` and its weight is within the limits
  std::vector<std::int32_t> root(static_cast<std::size_t>(n), -1);
  std::vector<Region> comp(static_cast<std::size_t>(n));
  std::vector<char> entered(static_cast<std::size_t>(n), 0);
  for (std::int32_t i = 0; i < n; ++i) {
    if (!in(i)) continue;
    const std::int32_t r = s.find(i);
    root[static_cast<std::size_t>(i)] = r;
    Region& c = comp[static_cast<std::size_t>(r)];
    c.nodes.push_back(i);
    c.weight += weights(i);
    if (c.peak < 0 || values(i) > c.value) { c.peak = i; c.value = values(i); }
    if (values(i) >= enter) entered[static_cast<std::size_t>(r)] = 1;
  }
  Regions out;
  out.label.assign(static_cast<std::size_t>(n), -1);
  std::vector<std::int32_t> ordinal(static_cast<std::size_t>(n), -1);
  for (std::int32_t r = 0; r < n; ++r) {                                  // ascending roots = ascending lowest nodes
    const Region& c = comp[static_cast<std::size_t>(r)];
    if (c.nodes.empty() || !entered[static_cast<std::size_t>(r)] || c.weight < minWeight || c.weight > maxWeight) continue;
    ordinal[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(out.regions.size());
    out.regions.push_back(c);
  }
  for (std::int32_t i = 0; i < n; ++i)
    if (root[static_cast<std::size_t>(i)] >= 0) out.label[static_cast<std::size_t>(i)] = ordinal[static_cast<std::size_t>(root[static_cast<std::size_t>(i)])];
  return out;
}

}  // namespace nxr::select
