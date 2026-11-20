// nxr::select — the tangent edges of a burst on its envelope (edges.h); the maths of the reference detector (tangent_edges), written here.
#include "nxr/select/edges.h"

#include <algorithm>
#include <stdexcept>

namespace nxr::select {

namespace {
// going from the peak in direction `step` (±1): the first local minimum below half the peak, else the lowest point within `w` samples
std::int64_t trough(const Eigen::Ref<const Eigen::VectorXd>& a, std::int64_t ipk, int step, std::int64_t w) {
  const std::int64_t n = a.size();
  std::int64_t i = ipk, best = ipk;
  const std::int64_t lim = step < 0 ? std::max<std::int64_t>(0, ipk - w) : std::min<std::int64_t>(n - 1, ipk + w);
  while (i != lim) {
    const std::int64_t j = i + step;
    if (a(j) < a(best)) best = j;
    if (0 < j && j < n - 1 && a(j) <= a(j - 1) && a(j) <= a(j + 1) && a(j) < 0.5 * a(ipk)) return j;
    i = j;
  }
  return best;
}
}  // namespace

TangentEdges tangentEdges(const Eigen::Ref<const Eigen::VectorXd>& a, std::int64_t centre, std::int64_t halfWidth, std::int64_t reach) {
  const std::int64_t n = a.size();
  if (n < 3) throw std::invalid_argument("tangentEdges: an envelope of at least 3 samples");
  if (centre < 0 || centre >= n) throw std::invalid_argument("tangentEdges: the centre lies outside the envelope");
  if (halfWidth < 1 || reach < 1) throw std::invalid_argument("tangentEdges: halfWidth and reach are at least 1 sample");
  TangentEdges e;
  const std::int64_t lo = std::max<std::int64_t>(0, centre - halfWidth), hi = std::min<std::int64_t>(n - 1, centre + halfWidth);
  std::int64_t ipk = lo;
  for (std::int64_t i = lo; i <= hi; ++i) if (a(i) > a(ipk)) ipk = i;
  const std::int64_t tb = trough(a, ipk, -1, reach), ta = trough(a, ipk, +1, reach);
  e.peak = ipk; e.troughBefore = tb; e.troughAfter = ta;
  if (ipk - tb < 2 || ta - ipk < 2) return e;
  auto slope = [&](std::int64_t i) { return i == 0 ? a(1) - a(0) : i == n - 1 ? a(n - 1) - a(n - 2) : 0.5 * (a(i + 1) - a(i - 1)); };   // numpy.gradient
  std::int64_t ir = tb, ifl = ipk;
  for (std::int64_t i = tb; i <= ipk; ++i) if (slope(i) > slope(ir)) ir = i;
  for (std::int64_t i = ipk; i <= ta; ++i) if (slope(i) < slope(ifl)) ifl = i;
  if (slope(ir) <= 0 || slope(ifl) >= 0) return e;
  e.onset = static_cast<double>(ir) - (a(ir) - a(tb)) / slope(ir);
  e.offset = static_cast<double>(ifl) + (a(ifl) - a(ta)) / -slope(ifl);
  e.ok = true;
  return e;
}

}  // namespace nxr::select
