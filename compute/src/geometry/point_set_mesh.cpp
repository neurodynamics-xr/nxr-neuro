// nxr::geometry::pointset — the triangulation of a point set on a cap (see the header for the steps and the reference).
#include "nxr/geometry/point_set_mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace nxr::geometry::pointset {

namespace {

using Tri = std::array<std::int32_t, 3>;

double orient2(const std::vector<double>& p, std::int32_t a, std::int32_t b, std::int32_t c) {
  const long double ax = p[2 * a], ay = p[2 * a + 1], bx = p[2 * b], by = p[2 * b + 1], cx = p[2 * c], cy = p[2 * c + 1];
  return static_cast<double>((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
}

// > 0 when d lies strictly inside the circle through the counter-clockwise a, b, c (beyond a rounding margin relative to the terms)
bool inside_circle(const std::vector<double>& p, std::int32_t a, std::int32_t b, std::int32_t c, std::int32_t d) {
  const long double dx = p[2 * d], dy = p[2 * d + 1];
  const long double ax = p[2 * a] - dx, ay = p[2 * a + 1] - dy, bx = p[2 * b] - dx, by = p[2 * b + 1] - dy, cx = p[2 * c] - dx, cy = p[2 * c + 1] - dy;
  const long double a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
  const long double t1 = a2 * (bx * cy - by * cx), t2 = b2 * (ax * cy - ay * cx), t3 = c2 * (ax * by - ay * bx);
  const long double det = t1 - t2 + t3;
  const long double scale = std::fabs(t1) + std::fabs(t2) + std::fabs(t3);
  return det > 1e-12L * scale;
}

std::uint64_t key(std::int32_t a, std::int32_t b) { return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) | static_cast<std::uint32_t>(b); }

double dist3(const std::vector<double>& V, std::int32_t a, std::int32_t b) {
  const double x = V[3 * a] - V[3 * b], y = V[3 * a + 1] - V[3 * b + 1], z = V[3 * a + 2] - V[3 * b + 2];
  return std::sqrt(x * x + y * y + z * z);
}

// the undirected edges of a triangle list with their counts, sorted (a < b) as numpy's unique gives them
std::vector<std::pair<std::array<std::int32_t, 2>, int>> edge_counts(const std::vector<Tri>& F) {
  std::map<std::array<std::int32_t, 2>, int> m;
  for (const Tri& t : F)
    for (int k = 0; k < 3; ++k) {
      std::int32_t a = t[static_cast<std::size_t>(k)], b = t[static_cast<std::size_t>((k + 1) % 3)];
      if (a > b) std::swap(a, b);
      ++m[{a, b}];
    }
  return {m.begin(), m.end()};
}

// the boundary loops (vertex lists), the longest first; ties keep the order they were found in (nxr's _loops)
std::vector<std::vector<std::int32_t>> boundary_loops(const std::vector<Tri>& F) {
  std::vector<std::int32_t> order;                        // the vertices in the order they first appear on a boundary edge
  std::map<std::int32_t, std::vector<std::int32_t>> nb;
  for (const auto& [e, c] : edge_counts(F)) {
    if (c != 1) continue;
    for (int s = 0; s < 2; ++s) {
      const std::int32_t u = e[static_cast<std::size_t>(s)], w = e[static_cast<std::size_t>(1 - s)];
      if (!nb.count(u)) order.push_back(u);
      nb[u].push_back(w);
    }
  }
  std::set<std::int32_t> seen;
  std::vector<std::vector<std::int32_t>> loops;
  for (const std::int32_t s0 : order) {
    if (seen.count(s0)) continue;
    std::vector<std::int32_t> loop{s0};
    std::int32_t prev = -1, cur = s0;
    bool first = true;
    seen.insert(s0);
    while (true) {
      std::int32_t next = -1;
      for (const std::int32_t x : nb[cur])
        if ((first || x != prev) && !seen.count(x)) { next = x; break; }
      if (next < 0) break;
      prev = cur; cur = next; first = false;
      loop.push_back(cur); seen.insert(cur);
    }
    loops.push_back(std::move(loop));
  }
  std::stable_sort(loops.begin(), loops.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
  return loops;
}

// a loop closed by triangles: three vertices by one, more by the shortest diagonal, recursively (nxr's _fill_holes.tri)
void close_loop(const std::vector<double>& V, const std::vector<std::int32_t>& L, std::vector<Tri>& out) {
  const std::size_t k = L.size();
  if (k < 3) return;
  if (k == 3) { out.push_back({L[0], L[1], L[2]}); return; }
  std::tuple<double, std::size_t, std::size_t> best{std::numeric_limits<double>::infinity(), 0, 0};
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = i + 2; j < k; ++j) {
      if (i == 0 && j == k - 1) continue;
      const std::tuple<double, std::size_t, std::size_t> c{dist3(V, L[i], L[j]), i, j};
      if (c < best) best = c;
    }
  const std::size_t i = std::get<1>(best), j = std::get<2>(best);
  std::vector<std::int32_t> left(L.begin() + static_cast<std::ptrdiff_t>(i), L.begin() + static_cast<std::ptrdiff_t>(j) + 1);
  std::vector<std::int32_t> right(L.begin() + static_cast<std::ptrdiff_t>(j), L.end());
  right.insert(right.end(), L.begin(), L.begin() + static_cast<std::ptrdiff_t>(i) + 1);
  close_loop(V, left, out);
  close_loop(V, right, out);
}

// the triangles round vertex v grouped into fans (joined through an edge at v), in nxr's order (_fans: the last triangle of the star first)
std::vector<std::vector<std::size_t>> fans_at(const std::vector<Tri>& F, std::int32_t v) {
  std::vector<std::size_t> star;
  for (std::size_t f = 0; f < F.size(); ++f)
    if (F[f][0] == v || F[f][1] == v || F[f][2] == v) star.push_back(f);
  auto shared = [&](std::size_t f, std::size_t h) {
    int n = 0;
    for (const auto a : F[f]) for (const auto b : F[h]) n += a == b;
    return n;
  };
  std::vector<std::vector<std::size_t>> groups;
  while (!star.empty()) {
    std::vector<std::size_t> g{star.back()};
    star.pop_back();
    bool grow = true;
    while (grow) {
      grow = false;
      const std::vector<std::size_t> snapshot = star;
      for (const std::size_t f : snapshot) {
        bool joins = false;
        for (const std::size_t h : g) if (shared(f, h) == 2) { joins = true; break; }
        if (joins) { g.push_back(f); star.erase(std::find(star.begin(), star.end(), f)); grow = true; }
      }
    }
    groups.push_back(std::move(g));
  }
  return groups;
}

}  // namespace

std::vector<std::int32_t> planarDelaunay(const std::vector<double>& xy) {
  if (xy.size() % 2 != 0) throw std::invalid_argument("planarDelaunay: the points are not [n x 2]");
  const std::int32_t n = static_cast<std::int32_t>(xy.size() / 2);
  if (n < 3) throw std::invalid_argument("planarDelaunay: fewer than 3 points");
  std::vector<std::int32_t> ord(static_cast<std::size_t>(n));
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(), [&](std::int32_t a, std::int32_t b) { return std::make_pair(xy[2 * a], xy[2 * a + 1]) < std::make_pair(xy[2 * b], xy[2 * b + 1]); });
  for (std::size_t i = 1; i < ord.size(); ++i)
    if (xy[2 * ord[i]] == xy[2 * ord[i - 1]] && xy[2 * ord[i] + 1] == xy[2 * ord[i - 1] + 1])
      throw std::invalid_argument("planarDelaunay: points " + std::to_string(ord[i - 1]) + " and " + std::to_string(ord[i]) + " coincide");
  // the seed: the first points in order up to the first one off their line, fanned from it
  std::size_t k = 2;
  while (k < ord.size() && orient2(xy, ord[0], ord[1], ord[k]) == 0.0) ++k;
  if (k == ord.size()) throw std::invalid_argument("planarDelaunay: the points are all on one line");
  std::vector<Tri> T;
  std::vector<std::int32_t> hull;                                   // counter-clockwise
  const std::int32_t apex = ord[k];
  const bool left = orient2(xy, ord[0], ord[1], apex) > 0;
  for (std::size_t i = 0; i + 1 < k; ++i) T.push_back(left ? Tri{ord[i], ord[i + 1], apex} : Tri{ord[i + 1], ord[i], apex});
  if (left) { for (std::size_t i = 0; i < k; ++i) hull.push_back(ord[i]); hull.push_back(apex); }
  else { hull.push_back(ord[0]); hull.push_back(apex); for (std::size_t i = k - 1; i >= 1; --i) hull.push_back(ord[i]); }
  // the sweep: each next point is outside the hull so far; joined to every hull edge it sees
  for (std::size_t q = k + 1; q < ord.size(); ++q) {
    const std::int32_t p = ord[q];
    const std::size_t h = hull.size();
    std::vector<char> sees(h, 0);
    bool any = false;
    for (std::size_t i = 0; i < h; ++i) { sees[i] = orient2(xy, hull[i], hull[(i + 1) % h], p) < 0; any = any || sees[i]; }
    const std::size_t visible = static_cast<std::size_t>(std::count(sees.begin(), sees.end(), 1));
    if (!any || visible == h) throw std::invalid_argument("planarDelaunay: a point sees no hull edge, or every one (degenerate input)");
    std::size_t s = 0;                                              // the first edge of the visible run (the one whose predecessor is not visible)
    while (!(sees[s] && !sees[(s + h - 1) % h])) ++s;
    std::size_t e = s;
    while (sees[(e + 1) % h]) e = (e + 1) % h;
    for (std::size_t i = s;; i = (i + 1) % h) {
      T.push_back({hull[i], p, hull[(i + 1) % h]});
      if (i == e) break;
    }
    std::vector<std::int32_t> nh;                                   // the hull: from the end of the run round to its start, then p
    for (std::size_t i = (e + 1) % h;; i = (i + 1) % h) { nh.push_back(hull[i]); if (i == s) break; }
    nh.push_back(p);
    hull = std::move(nh);
  }
  // Lawson flips: an interior edge whose opposite vertex lies inside the other triangle's circumcircle is flipped, until none does
  std::unordered_map<std::uint64_t, std::size_t> owner;             // directed edge a -> b to the triangle holding it
  auto enroll = [&](std::size_t t) { for (int j = 0; j < 3; ++j) owner[key(T[t][static_cast<std::size_t>(j)], T[t][static_cast<std::size_t>((j + 1) % 3)])] = t; };
  auto third = [&](std::size_t t, std::int32_t a, std::int32_t b) { for (const auto v : T[t]) if (v != a && v != b) return v; return std::int32_t(-1); };
  for (std::size_t t = 0; t < T.size(); ++t) enroll(t);
  std::vector<std::pair<std::int32_t, std::int32_t>> stack;
  for (const Tri& t : T) for (int j = 0; j < 3; ++j) stack.emplace_back(t[static_cast<std::size_t>(j)], t[static_cast<std::size_t>((j + 1) % 3)]);
  std::size_t guard = 0;
  const std::size_t limit = 100 * T.size() * T.size() + 1000;
  while (!stack.empty()) {
    if (++guard > limit) throw std::invalid_argument("planarDelaunay: the flips do not settle (degenerate input)");
    const auto [a, b] = stack.back();
    stack.pop_back();
    const auto i1 = owner.find(key(a, b)), i2 = owner.find(key(b, a));
    if (i1 == owner.end() || i2 == owner.end()) continue;           // a hull edge
    const std::size_t t1 = i1->second, t2 = i2->second;
    const std::int32_t c = third(t1, a, b), d = third(t2, a, b);
    if (!inside_circle(xy, a, b, c, d)) continue;
    for (const std::size_t t : {t1, t2}) for (int j = 0; j < 3; ++j) owner.erase(key(T[t][static_cast<std::size_t>(j)], T[t][static_cast<std::size_t>((j + 1) % 3)]));
    T[t1] = {a, d, c};                                              // the quad a, d, b, c (counter-clockwise) split along c-d
    T[t2] = {d, b, c};
    enroll(t1); enroll(t2);
    stack.emplace_back(a, d); stack.emplace_back(d, b); stack.emplace_back(b, c); stack.emplace_back(c, a);
  }
  std::vector<std::int32_t> out;
  out.reserve(T.size() * 3);
  for (const Tri& t : T) out.insert(out.end(), t.begin(), t.end());
  return out;
}

PointSetMesh pointSetMesh(const std::vector<double>& V, const PointSetMeshOptions& o) {
  if (V.size() % 3 != 0) throw std::invalid_argument("pointSetMesh: the points are not [n x 3]");
  const std::int32_t n = static_cast<std::int32_t>(V.size() / 3);
  if (n < 3) throw std::invalid_argument("pointSetMesh: fewer than 3 points");
  for (const double x : V) if (!std::isfinite(x)) throw std::invalid_argument("pointSetMesh: a point has a non-finite coordinate");
  if (!(o.cut > 0)) throw std::invalid_argument("pointSetMesh: cut is a positive multiple of the median edge");
  const double pn = std::sqrt(o.pole[0] * o.pole[0] + o.pole[1] * o.pole[1] + o.pole[2] * o.pole[2]);
  if (!(pn > 0)) throw std::invalid_argument("pointSetMesh: the pole is a nonzero direction");
  // 1. the azimuthal equidistant projection about the pole, from the centroid
  double c[3] = {0, 0, 0};
  for (std::int32_t i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) c[k] += V[3 * i + k];
  for (double& x : c) x /= n;
  double u[3] = {o.pole[0] / pn, o.pole[1] / pn, o.pole[2] / pn}, e1[3], e2[3];
  if (o.pole[0] == 0 && o.pole[1] == 0 && o.pole[2] > 0) { e1[0] = 1; e1[1] = 0; e1[2] = 0; e2[0] = 0; e2[1] = 1; e2[2] = 0; }   // +z: the frame's own axes (nxr's)
  else {
    const double ref[3] = {std::fabs(u[0]) < 0.9 ? 1.0 : 0.0, std::fabs(u[0]) < 0.9 ? 0.0 : 1.0, 0.0};
    e1[0] = u[1] * ref[2] - u[2] * ref[1]; e1[1] = u[2] * ref[0] - u[0] * ref[2]; e1[2] = u[0] * ref[1] - u[1] * ref[0];
    const double l = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    for (double& x : e1) x /= l;
    e2[0] = u[1] * e1[2] - u[2] * e1[1]; e2[1] = u[2] * e1[0] - u[0] * e1[2]; e2[2] = u[0] * e1[1] - u[1] * e1[0];
  }
  std::vector<double> xy(static_cast<std::size_t>(2 * n));
  for (std::int32_t i = 0; i < n; ++i) {
    const double p[3] = {V[3 * i] - c[0], V[3 * i + 1] - c[1], V[3 * i + 2] - c[2]};
    const double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    if (!(r > 0)) throw std::invalid_argument("pointSetMesh: point " + std::to_string(i) + " is at the centroid (no direction to project)");
    const double z = p[0] * u[0] + p[1] * u[1] + p[2] * u[2], x = p[0] * e1[0] + p[1] * e1[1] + p[2] * e1[2], y = p[0] * e2[0] + p[1] * e2[1] + p[2] * e2[2];
    const double th = std::acos(std::max(-1.0, std::min(1.0, z / r))), ph = std::atan2(y, x);
    xy[static_cast<std::size_t>(2 * i)] = th * std::cos(ph);
    xy[static_cast<std::size_t>(2 * i + 1)] = th * std::sin(ph);
  }
  // 2. the plane Delaunay
  const std::vector<std::int32_t> D = planarDelaunay(xy);
  PointSetMesh out;
  std::vector<Tri> F;
  for (std::size_t t = 0; t < D.size() / 3; ++t) F.push_back({D[3 * t], D[3 * t + 1], D[3 * t + 2]});
  out.delaunay_triangles = static_cast<std::int64_t>(F.size());
  // 3. the cut: the median over every triangle's three edges (numpy's median: the mean of the two middle values of an even count)
  std::vector<double> len;
  for (const Tri& t : F) { len.push_back(dist3(V, t[0], t[1])); len.push_back(dist3(V, t[1], t[2])); len.push_back(dist3(V, t[0], t[2])); }
  std::sort(len.begin(), len.end());
  const std::size_t m = len.size();
  out.median_edge = m % 2 ? len[m / 2] : 0.5 * (len[m / 2 - 1] + len[m / 2]);
  const double longest = o.cut * out.median_edge;
  {
    std::vector<Tri> kept;
    for (const Tri& t : F)
      if (dist3(V, t[0], t[2]) < longest && dist3(V, t[1], t[0]) < longest && dist3(V, t[2], t[1]) < longest) kept.push_back(t);
    out.cut_triangles = static_cast<std::int64_t>(F.size() - kept.size());
    F = std::move(kept);
  }
  // 4. the pinches
  bool settled = false;
  for (int pass = 0; pass < o.repair_passes; ++pass) {
    std::vector<std::array<std::int32_t, 2>> bd;
    std::vector<int> deg(static_cast<std::size_t>(n), 0);
    for (const auto& [e, cnt] : edge_counts(F)) if (cnt == 1) { bd.push_back(e); ++deg[static_cast<std::size_t>(e[0])]; ++deg[static_cast<std::size_t>(e[1])]; }
    std::vector<std::int32_t> pinched;
    for (std::int32_t v = 0; v < n; ++v) if (deg[static_cast<std::size_t>(v)] > 2) pinched.push_back(v);
    if (pinched.empty()) { settled = true; break; }
    std::vector<Tri> add;
    std::set<std::size_t> drop;
    for (const std::int32_t v : pinched) {
      const auto fans = fans_at(F, v);
      std::set<std::int32_t> rim;
      for (const auto& e : bd) if (e[0] == v || e[1] == v) { rim.insert(e[0]); rim.insert(e[1]); }
      rim.erase(v);
      std::vector<std::vector<std::int32_t>> ends;
      for (const auto& g : fans) {
        std::set<std::int32_t> in;
        for (const std::size_t f : g) for (const auto x : F[f]) if (rim.count(x)) in.insert(x);
        ends.emplace_back(in.begin(), in.end());
      }
      bool found = false;
      std::tuple<double, std::int32_t, std::int32_t> best{0, 0, 0};
      for (std::size_t i = 0; i < fans.size(); ++i)
        for (std::size_t j = i + 1; j < fans.size(); ++j)
          for (const auto a : ends[i]) for (const auto b : ends[j]) {
            const std::tuple<double, std::int32_t, std::int32_t> cnd{dist3(V, a, b), a, b};
            if (!found || cnd < best) { best = cnd; found = true; }
          }
      if (found && std::get<0>(best) < longest) { add.push_back({v, std::get<1>(best), std::get<2>(best)}); ++out.pinches_closed; continue; }
      std::size_t small = 0;                                         // the smaller fan goes (the first of the smallest), if no point is lost with it
      for (std::size_t i = 1; i < fans.size(); ++i) if (fans[i].size() < fans[small].size()) small = i;
      const std::set<std::size_t> gone(fans[small].begin(), fans[small].end());
      std::set<std::int32_t> elsewhere, lost;
      for (std::size_t f = 0; f < F.size(); ++f) if (!gone.count(f)) for (const auto x : F[f]) elsewhere.insert(x);
      for (const std::size_t f : gone) for (const auto x : F[f]) if (!elsewhere.count(x)) lost.insert(x);
      if (!lost.empty()) throw std::invalid_argument("pointSetMesh: the gap at vertex " + std::to_string(v) + " is wider than the cut and dropping its smaller fan would lose point " + std::to_string(*lost.begin()));
      drop.insert(gone.begin(), gone.end());
      ++out.fans_dropped;
    }
    std::vector<Tri> next;
    for (std::size_t f = 0; f < F.size(); ++f) if (!drop.count(f)) next.push_back(F[f]);
    next.insert(next.end(), add.begin(), add.end());
    F = std::move(next);
  }
  if (!settled) throw std::invalid_argument("pointSetMesh: pinched vertices remain after " + std::to_string(o.repair_passes) + " passes");
  // 5. the holes: every boundary loop but the longest
  {
    const auto loops = boundary_loops(F);
    std::vector<Tri> add;
    for (std::size_t l = 1; l < loops.size(); ++l) close_loop(V, loops[l], add);
    out.hole_triangles = static_cast<std::int64_t>(add.size());
    F.insert(F.end(), add.begin(), add.end());
  }
  // 6. the orientation
  if (o.orientation == Orientation::outward)
    for (Tri& t : F) {
      double a[3], b[3], nrm[3], mid[3];
      for (int k = 0; k < 3; ++k) { a[k] = V[3 * t[1] + k] - V[3 * t[0] + k]; b[k] = V[3 * t[2] + k] - V[3 * t[0] + k]; mid[k] = (V[3 * t[0] + k] + V[3 * t[1] + k] + V[3 * t[2] + k]) / 3.0 - c[k]; }
      nrm[0] = a[1] * b[2] - a[2] * b[1]; nrm[1] = a[2] * b[0] - a[0] * b[2]; nrm[2] = a[0] * b[1] - a[1] * b[0];
      if (!(nrm[0] * mid[0] + nrm[1] * mid[1] + nrm[2] * mid[2] > 0)) std::swap(t[1], t[2]);
    }
  // the report
  std::vector<char> used(static_cast<std::size_t>(n), 0);
  for (const Tri& t : F) for (const auto x : t) used[static_cast<std::size_t>(x)] = 1;
  std::int64_t nv = 0;
  for (std::int32_t v = 0; v < n; ++v) { if (used[static_cast<std::size_t>(v)]) ++nv; else out.points_without_face.push_back(v); }
  out.edges = static_cast<std::int64_t>(edge_counts(F).size());
  out.boundary_loops = static_cast<int>(boundary_loops(F).size());
  out.euler_characteristic = nv - out.edges + static_cast<std::int64_t>(F.size());
  for (const Tri& t : F) out.faces.insert(out.faces.end(), t.begin(), t.end());
  return out;
}

}  // namespace nxr::geometry::pointset
