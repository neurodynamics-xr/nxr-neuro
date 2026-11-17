// nxr::nearest — see nxr/nearest.h.
#include "nxr/nearest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::nearest {

namespace {

template <class S>
void check(const std::vector<S>& positions, const std::vector<S>& query) {
  if (positions.size() % 3 != 0 || query.size() % 3 != 0) throw std::invalid_argument("nearestVertices: points are [n x 3]");
  if (positions.empty()) throw std::invalid_argument("nearestVertices: no vertex");
}

template <class S>
S squared(const S* p, const S* q) {
  const S a = p[0] - q[0], b = p[1] - q[1], c = p[2] - q[2];
  return a * a + b * b + c * c;
}

template <class S>
class KdTree {
 public:
  explicit KdTree(const std::vector<S>& pts) : p_(pts), order_(pts.size() / 3) {
    for (std::size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<std::int32_t>(i);
    build(0, static_cast<std::int64_t>(order_.size()), 0);
  }
  void nearest(const S* q, std::int32_t& best, S& best_d) const {
    best = -1;
    best_d = std::numeric_limits<S>::infinity();
    search(0, static_cast<std::int64_t>(order_.size()), 0, q, best, best_d);
  }

 private:
  const std::vector<S>& p_;
  std::vector<std::int32_t> order_;
  S at(std::int32_t i, int axis) const { return p_[static_cast<std::size_t>(i) * 3 + static_cast<std::size_t>(axis)]; }
  void build(std::int64_t lo, std::int64_t hi, int axis) {
    if (hi - lo < 2) return;
    const std::int64_t mid = (lo + hi) / 2;
    std::nth_element(order_.begin() + lo, order_.begin() + mid, order_.begin() + hi,
                     [&](std::int32_t a, std::int32_t b) { return at(a, axis) < at(b, axis) || (at(a, axis) == at(b, axis) && a < b); });
    build(lo, mid, (axis + 1) % 3);
    build(mid + 1, hi, (axis + 1) % 3);
  }
  void search(std::int64_t lo, std::int64_t hi, int axis, const S* q, std::int32_t& best, S& best_d) const {
    if (hi <= lo) return;
    const std::int64_t mid = (lo + hi) / 2;
    const std::int32_t i = order_[static_cast<std::size_t>(mid)];
    const S d = squared(&p_[static_cast<std::size_t>(i) * 3], q);
    if (d < best_d || (d == best_d && i < best)) { best_d = d; best = i; }
    const S diff = q[axis] - at(i, axis);
    const int next = (axis + 1) % 3;
    // the far side can hold an equally near vertex of a lower index, or one that rounding puts a hair below the best: it is searched when its plane is not farther than the best
    // by more than a few units in the last place, so the answer is the brute force's even on ties
    const std::int64_t nlo = diff < 0 ? lo : mid + 1, nhi = diff < 0 ? mid : hi, flo = diff < 0 ? mid + 1 : lo, fhi = diff < 0 ? hi : mid;
    search(nlo, nhi, next, q, best, best_d);
    const S slack = best_d * (S(1) + S(8) * std::numeric_limits<S>::epsilon());   // after the near side: best_d has changed
    if (diff * diff <= slack) search(flo, fhi, next, q, best, best_d);
  }
};

}  // namespace

template <class S>
Result<S> nearestVertices(const std::vector<S>& positions, const std::vector<S>& query) {
  check(positions, query);
  const KdTree<S> tree(positions);
  const std::size_t q = query.size() / 3;
  Result<S> r;
  r.vertex.resize(q);
  r.distance.resize(q);
  for (std::size_t i = 0; i < q; ++i) {
    S d;
    tree.nearest(&query[3 * i], r.vertex[i], d);
    r.distance[i] = std::sqrt(d);
  }
  return r;
}

template <class S>
Result<S> nearestVerticesBrute(const std::vector<S>& positions, const std::vector<S>& query) {
  check(positions, query);
  const std::size_t n = positions.size() / 3, q = query.size() / 3;
  Result<S> r;
  r.vertex.resize(q);
  r.distance.resize(q);
  for (std::size_t i = 0; i < q; ++i) {
    std::int32_t best = 0;
    S bd = squared(&positions[0], &query[3 * i]);
    for (std::size_t v = 1; v < n; ++v) {
      const S d = squared(&positions[3 * v], &query[3 * i]);
      if (d < bd) { bd = d; best = static_cast<std::int32_t>(v); }
    }
    r.vertex[i] = best;
    r.distance[i] = std::sqrt(bd);
  }
  return r;
}

template Result<float> nearestVertices(const std::vector<float>&, const std::vector<float>&);
template Result<double> nearestVertices(const std::vector<double>&, const std::vector<double>&);
template Result<float> nearestVerticesBrute(const std::vector<float>&, const std::vector<float>&);
template Result<double> nearestVerticesBrute(const std::vector<double>&, const std::vector<double>&);

}  // namespace nxr::nearest
