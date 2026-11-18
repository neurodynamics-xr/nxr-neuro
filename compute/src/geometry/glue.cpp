// nxr::glue — the Studio's glue.ts (glueMesh, sourcesOf), line for line: Map/Set insertion orders, strict-versus-tie comparisons, Math.round and
// Math.hypot (ladder::jsHypot) as JavaScript evaluates them.
#include "nxr/geometry/surface_maps.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "nxr/ladder.h"

namespace nxr::geometry::glue {
namespace {
using Pt = std::array<double, 3>;
using I = std::int32_t;
using L = std::int64_t;

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error(m); }
double jsRound(double x) { const double r = std::floor(x); return (x - r >= 0.5) ? r + 1 : r; }   // Math.round
double dist(const Pt& a, const Pt& b) { const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2]; return std::sqrt(x * x + y * y + z * z); }
Pt sub3(const Pt& a, const Pt& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

// the vertices' components through the faces' edges, among `on` (all when null): a label a vertex (the union-find root), −1 off `on`
std::vector<I> labels(std::size_t n, const std::vector<I>& F, const std::vector<std::uint8_t>* on) {
  std::vector<I> up(n);
  for (std::size_t i = 0; i < n; ++i) up[i] = static_cast<I>(i);
  auto find = [&](I i) { while (up[static_cast<size_t>(i)] != i) { up[static_cast<size_t>(i)] = up[static_cast<size_t>(up[static_cast<size_t>(i)])]; i = up[static_cast<size_t>(i)]; } return i; };
  for (std::size_t f = 0; f < F.size(); f += 3) {
    for (int k = 0; k < 3; ++k) {
      const I a = F[f + static_cast<size_t>(k)], b = F[f + static_cast<size_t>((k + 1) % 3)];
      if (on && !((*on)[static_cast<size_t>(a)] && (*on)[static_cast<size_t>(b)])) continue;
      const I ra = find(a), rb = find(b);
      if (ra != rb) up[static_cast<size_t>(ra)] = rb;
    }
  }
  std::vector<I> out(n, -1);
  for (std::size_t i = 0; i < n; ++i) if (!on || (*on)[i]) out[i] = find(static_cast<I>(i));
  return out;
}

// the members of `on` in its largest component (the first in order of appearance among equals: a Map's insertion order)
std::vector<std::uint8_t> largest(std::size_t n, const std::vector<I>& F, const std::vector<std::uint8_t>& on) {
  const std::vector<I> lab = labels(n, F, &on);
  std::vector<I> keys; std::unordered_map<I, L> count;
  for (std::size_t i = 0; i < n; ++i) if (lab[i] >= 0) { auto it = count.find(lab[i]); if (it == count.end()) { keys.push_back(lab[i]); count[lab[i]] = 1; } else ++it->second; }
  I best = -1; L most = 0;
  for (I k : keys) if (count[k] > most) { most = count[k]; best = k; }
  std::vector<std::uint8_t> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) out[i] = (lab[i] >= 0 && lab[i] == best) ? 1 : 0;
  return out;
}

// the faces' half-edges used once (a hole's rim), chained into loops in the faces' winding; a vertex on two rims is a pinch (returned as `pinch`)
struct Rims { std::vector<std::vector<I>> loops; I pinch = -1; };
Rims rims(const std::vector<I>& F, std::size_t n) {
  auto key = [&](I a, I b) { return static_cast<L>(a) * static_cast<L>(n) + b; };
  std::unordered_set<L> have;
  for (std::size_t f = 0; f < F.size(); f += 3) for (int k = 0; k < 3; ++k) have.insert(key(F[f + static_cast<size_t>(k)], F[f + static_cast<size_t>((k + 1) % 3)]));
  std::unordered_map<I, I> next; std::vector<I> order;
  Rims r;
  for (std::size_t f = 0; f < F.size(); f += 3) for (int k = 0; k < 3; ++k) {
    const I a = F[f + static_cast<size_t>(k)], b = F[f + static_cast<size_t>((k + 1) % 3)];
    if (have.count(key(b, a))) continue;
    if (next.count(a)) { r.pinch = a; return r; }
    next[a] = b; order.push_back(a);
  }
  std::size_t head = 0;
  while (!next.empty()) {
    while (!next.count(order[head])) ++head;                     // the first key still present, in insertion order
    const I a0 = order[head];
    std::vector<I> loop = {a0};
    I a = next[a0]; next.erase(a0);
    while (a != a0) { loop.push_back(a); const I b = next.at(a); next.erase(a); a = b; }
    r.loops.push_back(std::move(loop));
  }
  return r;
}

struct Disk { std::vector<I> faces, loop; };
// the faces of one piece with the patch `start` (its medial wall) cut out, grown until the hole is ONE simple loop; and that loop
Disk cutDisk(const std::vector<I>& F, const std::vector<std::uint8_t>& piece, const std::vector<std::uint8_t>& start, int which) {
  const std::size_t n = piece.size();
  std::vector<std::uint8_t> cut = largest(n, F, start);
  if (std::none_of(cut.begin(), cut.end(), [](std::uint8_t c) { return c != 0; }))
    fail("glue: hemisphere " + std::to_string(which) + " has no medial wall in the atlas \xe2\x80\x94 the hole is cut by anatomy only");
  for (int round = 0; round < 200; ++round) {
    std::vector<std::uint8_t> rest(n);
    for (std::size_t i = 0; i < n; ++i) rest[i] = (piece[i] && !cut[i]) ? 1 : 0;
    const std::vector<std::uint8_t> keep = largest(n, F, rest);
    for (std::size_t i = 0; i < n; ++i) if (piece[i] && !keep[i]) cut[i] = 1;
    std::vector<I> faces;
    for (std::size_t f = 0; f < F.size(); f += 3) {
      const I a = F[f], b = F[f + 1], c = F[f + 2];
      if (piece[static_cast<size_t>(a)] && !cut[static_cast<size_t>(a)] && !cut[static_cast<size_t>(b)] && !cut[static_cast<size_t>(c)]) { faces.push_back(a); faces.push_back(b); faces.push_back(c); }
    }
    Rims r = rims(faces, n);
    if (r.pinch >= 0) { cut[static_cast<size_t>(r.pinch)] = 1; continue; }
    if (r.loops.size() == 1) return Disk{faces, r.loops[0]};
    std::vector<std::vector<I>> ls = r.loops;
    std::stable_sort(ls.begin(), ls.end(), [](const std::vector<I>& x, const std::vector<I>& y) { return x.size() < y.size(); });
    for (std::size_t i = 0; i + 1 < ls.size(); ++i) for (I v : ls[i]) cut[static_cast<size_t>(v)] = 1;
  }
  fail("glue: the medial wall did not become a disk");
}

struct Arc { std::vector<double> cum; double per = 0; };
Arc arc(const std::vector<Pt>& pts) {
  Arc a; a.cum.push_back(0);
  for (std::size_t i = 1; i <= pts.size(); ++i) a.cum.push_back(a.cum[i - 1] + dist(pts[i - 1], pts[i % pts.size()]));
  a.per = a.cum[pts.size()];
  return a;
}
Pt sample(const std::vector<Pt>& pts, const std::vector<double>& cum, double per, double s) {
  const double Ls = s * per; std::size_t k = 0;
  while (k < pts.size() - 1 && cum[k + 1] <= Ls) ++k;
  const Pt& a = pts[k]; const Pt& b = pts[(k + 1) % pts.size()];
  const double seg = cum[k + 1] - cum[k], t = seg > 0 ? (Ls - cum[k]) / seg : 0;
  return {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
}

// triangles joining two loops that run the same way round, from B's vertex nearest A's first: the band of least total edge length (Fuchs, Kedem &
// Uselton 1977); a tie advances on A
template <class At> std::vector<I> stitch(const std::vector<I>& A, const std::vector<I>& B0, const At& at) {
  std::size_t j0 = 0;
  for (std::size_t j = 1; j < B0.size(); ++j) if (dist(at(B0[j]), at(A[0])) < dist(at(B0[j0]), at(A[0]))) j0 = j;
  std::vector<I> B(B0.begin() + static_cast<long>(j0), B0.end()); B.insert(B.end(), B0.begin(), B0.begin() + static_cast<long>(j0));
  const std::size_t m = A.size(), nb = B.size(), W = nb + 1;
  auto a = [&](std::size_t i) { return A[i % m]; };
  auto b = [&](std::size_t j) { return B[j % nb]; };
  std::vector<double> cost((m + 1) * W, std::numeric_limits<double>::infinity());
  std::vector<std::uint8_t> from((m + 1) * W, 0);
  cost[0] = 0;
  for (std::size_t i = 0; i <= m; ++i) for (std::size_t j = 0; j <= nb; ++j) {
    const double c = cost[i * W + j];
    if (i < m) { const double x = c + dist(at(a(i + 1)), at(b(j))); const std::size_t k = (i + 1) * W + j; if (x < cost[k] || (x == cost[k] && from[k] != 1)) { cost[k] = x; from[k] = 1; } }
    if (j < nb) { const double x = c + dist(at(a(i)), at(b(j + 1))); const std::size_t k = i * W + j + 1; if (x < cost[k]) { cost[k] = x; from[k] = 2; } }
  }
  std::vector<I> T;
  for (std::size_t i = m, j = nb; i > 0 || j > 0;) {
    if (from[i * W + j] == 1) { T.push_back(a(i - 1)); T.push_back(a(i)); T.push_back(b(j)); --i; }
    else { T.push_back(a(i)); T.push_back(b(j)); T.push_back(b(j - 1)); --j; }
  }
  return T;
}

double meanEdge(const std::vector<double>& P, const std::vector<I>& F) {
  double s = 0;
  for (std::size_t f = 0; f < F.size(); f += 3) for (int k = 0; k < 3; ++k) {
    const auto a = static_cast<size_t>(F[f + static_cast<size_t>(k)]), b = static_cast<size_t>(F[f + static_cast<size_t>((k + 1) % 3)]);
    s += dist({P[3 * a], P[3 * a + 1], P[3 * a + 2]}, {P[3 * b], P[3 * b + 1], P[3 * b + 2]});
  }
  return s / static_cast<double>(F.size());
}

L euler(const std::vector<I>& F, std::size_t n) {
  std::unordered_set<L> e;
  for (std::size_t f = 0; f < F.size(); f += 3) for (int k = 0; k < 3; ++k) {
    const L a = F[f + static_cast<size_t>(k)], b = F[f + static_cast<size_t>((k + 1) % 3)];
    e.insert(std::min(a, b) * static_cast<L>(n) + std::max(a, b));
  }
  return static_cast<L>(n) - static_cast<L>(e.size()) + static_cast<L>(F.size() / 3);
}
}  // namespace

Glued glueMesh(const std::vector<double>& P, const std::vector<I>& F, const std::vector<std::uint8_t>& wall) {
  const std::size_t n = P.size() / 3;
  const std::vector<I> lab = labels(n, F, nullptr);
  std::vector<I> roots; { std::unordered_set<I> seen; for (I l : lab) if (seen.insert(l).second) roots.push_back(l); }
  if (roots.size() != 2) fail("glue: the surface has " + std::to_string(roots.size()) + " pieces, not two");
  if (wall.size() != n) fail("glue: the medial-wall mask has " + std::to_string(wall.size()) + " entries for " + std::to_string(n) + " vertices");
  std::vector<std::vector<std::uint8_t>> pieces;
  for (I r : roots) { std::vector<std::uint8_t> p(n); for (std::size_t i = 0; i < n; ++i) p[i] = lab[i] == r ? 1 : 0; pieces.push_back(std::move(p)); }
  auto facesOf = [&](const std::vector<std::uint8_t>& p) { std::vector<I> o; for (std::size_t f = 0; f < F.size(); f += 3) if (p[static_cast<size_t>(F[f])]) { o.push_back(F[f]); o.push_back(F[f + 1]); o.push_back(F[f + 2]); } return o; };
  L chi[2];
  for (int q = 0; q < 2; ++q) {
    const std::vector<I> fs = facesOf(pieces[static_cast<size_t>(q)]);
    std::unordered_map<I, I> remap; std::vector<I> m2;
    for (I v : fs) if (!remap.count(v)) { const I k = static_cast<I>(remap.size()); remap[v] = k; }
    for (I v : fs) m2.push_back(remap[v]);
    chi[q] = euler(m2, remap.size());
  }
  std::vector<Disk> disks;
  for (int q = 0; q < 2; ++q) {
    std::vector<std::uint8_t> st(n);
    for (std::size_t i = 0; i < n; ++i) st[i] = (pieces[static_cast<size_t>(q)][i] && wall[i]) ? 1 : 0;
    disks.push_back(cutDisk(F, pieces[static_cast<size_t>(q)], st, q));
  }
  auto pt = [&](I v) -> Pt { const auto u = static_cast<size_t>(v); return {P[3 * u], P[3 * u + 1], P[3 * u + 2]}; };
  // the rims, oriented the same way round the axis from rim A's centre to rim B's
  auto centre = [&](const std::vector<I>& Lp) { Pt c = {0, 0, 0}; const double len = static_cast<double>(Lp.size()); for (I v : Lp) { const Pt p = pt(v); c[0] += p[0] / len; c[1] += p[1] / len; c[2] += p[2] / len; } return c; };
  const Pt ca = centre(disks[0].loop), axis = sub3(centre(disks[1].loop), ca);
  auto orient = [&](const std::vector<I>& Lp) {
    const Pt c = centre(Lp); double s = 0;
    for (std::size_t i = 0; i < Lp.size(); ++i) {
      const Pt p = sub3(pt(Lp[i]), c), q = sub3(pt(Lp[(i + 1) % Lp.size()]), c);
      s += (p[1] * q[2] - p[2] * q[1]) * axis[0] + (p[2] * q[0] - p[0] * q[2]) * axis[1] + (p[0] * q[1] - p[1] * q[0]) * axis[2];
    }
    std::vector<I> o = Lp;
    if (s < 0) std::reverse(o.begin(), o.end());
    const auto k = static_cast<long>(std::min_element(o.begin(), o.end()) - o.begin());   // from its lowest-numbered vertex
    std::vector<I> r(o.begin() + k, o.end()); r.insert(r.end(), o.begin(), o.begin() + k);
    return r;
  };
  const std::vector<I> A = orient(disks[0].loop), B = orient(disks[1].loop);
  // resampled to one count N, B's start the shift that brings it nearest A's
  const double h = meanEdge(P, F);
  std::vector<Pt> pa, pb; for (I v : A) pa.push_back(pt(v)); for (I v : B) pb.push_back(pt(v));
  const Arc ra = arc(pa), rb = arc(pb);
  const auto N = static_cast<std::size_t>(std::max(3.0, jsRound((ra.per + rb.per) / 2 / h)));
  std::vector<Pt> SA(N), SB(N);
  for (std::size_t i = 0; i < N; ++i) { SA[i] = sample(pa, ra.cum, ra.per, static_cast<double>(i) / static_cast<double>(N)); SB[i] = sample(pb, rb.cum, rb.per, static_cast<double>(i) / static_cast<double>(N)); }
  std::size_t shift = 0; double best = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < N; ++j) { double c = 0; for (std::size_t i = 0; i < N; ++i) { const double d = dist(SA[i], SB[(i + j) % N]); c += d * d; } if (c < best) { best = c; shift = j; } }
  std::vector<double> gaps(N);
  for (std::size_t i = 0; i < N; ++i) gaps[i] = dist(SA[i], SB[(i + shift) % N]);
  std::sort(gaps.begin(), gaps.end());
  const auto R = static_cast<std::size_t>(std::max(1.0, jsRound(gaps[N / 2] / h) - 1));
  // the kept vertices first (ascending), then ring r's N points
  std::set<I> usedSet(disks[0].faces.begin(), disks[0].faces.end()); usedSet.insert(disks[1].faces.begin(), disks[1].faces.end());
  const std::vector<I> used(usedSet.begin(), usedSet.end());
  std::vector<I> newOf(n, -1);
  for (std::size_t i = 0; i < used.size(); ++i) newOf[static_cast<size_t>(used[i])] = static_cast<I>(i);
  const std::size_t m = used.size(), total = m + R * N;
  std::vector<double> X(3 * total);
  for (std::size_t i = 0; i < m; ++i) for (int k = 0; k < 3; ++k) X[3 * i + static_cast<size_t>(k)] = P[3 * static_cast<size_t>(used[i]) + static_cast<size_t>(k)];
  for (std::size_t r = 1; r <= R; ++r) {
    const double t = static_cast<double>(r) / static_cast<double>(R + 1);
    for (std::size_t i = 0; i < N; ++i) { const Pt& a = SA[i]; const Pt& b = SB[(i + shift) % N]; const std::size_t o = 3 * (m + (r - 1) * N + i); for (int k = 0; k < 3; ++k) X[o + static_cast<size_t>(k)] = a[static_cast<size_t>(k)] + t * (b[static_cast<size_t>(k)] - a[static_cast<size_t>(k)]); }
  }
  auto at = [&](I v) -> Pt { const auto u = static_cast<size_t>(v); return {X[3 * u], X[3 * u + 1], X[3 * u + 2]}; };
  auto ring = [&](std::size_t r) { std::vector<I> o(N); for (std::size_t i = 0; i < N; ++i) o[i] = static_cast<I>(m + (r - 1) * N + i); return o; };
  std::vector<I> Am, Bm; for (I v : A) Am.push_back(newOf[static_cast<size_t>(v)]); for (I v : B) Bm.push_back(newOf[static_cast<size_t>(v)]);
  std::vector<I> bridge = stitch(Am, ring(1), at);
  for (std::size_t r = 1; r < R; ++r) { const auto s = stitch(ring(r), ring(r + 1), at); bridge.insert(bridge.end(), s.begin(), s.end()); }
  { const auto s = stitch(ring(R), Bm, at); bridge.insert(bridge.end(), s.begin(), s.end()); }
  // wound as the disks: a rim half-edge of disk A (its loop's own order) must not appear in the bridge too
  const std::vector<I>& la = disks[0].loop;
  const I x0 = newOf[static_cast<size_t>(la[0])], x1 = newOf[static_cast<size_t>(la[1])];
  bool same = false;
  for (std::size_t f = 0; f < bridge.size() && !same; f += 3) for (int k = 0; k < 3; ++k) if (bridge[f + static_cast<size_t>(k)] == x0 && bridge[f + static_cast<size_t>((k + 1) % 3)] == x1) same = true;
  if (same) for (std::size_t f = 0; f < bridge.size(); f += 3) std::swap(bridge[f + 1], bridge[f + 2]);
  Glued g;
  for (int q = 0; q < 2; ++q) for (I v : disks[static_cast<size_t>(q)].faces) g.faces.push_back(newOf[static_cast<size_t>(v)]);
  g.faces.insert(g.faces.end(), bridge.begin(), bridge.end());
  g.keep.resize(total);
  for (std::size_t i = 0; i < total; ++i) g.keep[i] = i < m ? used[i] : -1;
  g.positions = X;
  // the checks
  std::unordered_set<L> half;
  for (std::size_t f = 0; f < g.faces.size(); f += 3) for (int k = 0; k < 3; ++k) {
    const L hk = static_cast<L>(g.faces[f + static_cast<size_t>(k)]) * static_cast<L>(total) + g.faces[f + static_cast<size_t>((k + 1) % 3)];
    if (!half.insert(hk).second) fail("glue: a half-edge is used twice");
  }
  for (L hk : half) if (!half.count((hk % static_cast<L>(total)) * static_cast<L>(total) + hk / static_cast<L>(total))) fail("glue: an edge has one face");
  { const std::vector<I> lb = labels(total, g.faces, nullptr); std::unordered_set<I> s(lb.begin(), lb.end()); if (s.size() != 1) fail("glue: the result is not one piece"); }
  const L want = chi[0] + chi[1] - 2, got = euler(g.faces, total);
  if (got != want) fail("glue: \xcf\x87 = " + std::to_string(got) + ", not " + std::to_string(want));
  double area = 0, longest = 0;
  for (std::size_t f = 0; f < bridge.size(); f += 3) {
    const Pt p = at(bridge[f]), q = at(bridge[f + 1]), r = at(bridge[f + 2]), u = sub3(q, p), w = sub3(r, p);
    area += 0.5 * nxr::ladder::jsHypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
    longest = std::max({longest, dist(p, q), dist(q, r), dist(r, p)});
  }
  g.bridge = Bridge{static_cast<L>(R * N), static_cast<L>(R), area, longest};
  return g;
}

Sources sourcesOf(const std::vector<double>& P, const Glued& g) {
  const int K = 3;
  const std::size_t total = g.keep.size(), n = P.size() / 3;
  Sources s; s.k = K; s.idx.assign(total * K, -1); s.w.assign(total * K, 0.0f);
  std::vector<std::uint8_t> kept(n, 0);
  for (I v : g.keep) if (v >= 0) kept[static_cast<size_t>(v)] = 1;
  std::vector<I> walls;
  for (std::size_t v = 0; v < n; ++v) if (!kept[v]) walls.push_back(static_cast<I>(v));
  for (std::size_t i = 0; i < total; ++i) {
    if (g.keep[i] >= 0) { s.idx[i * K] = g.keep[i]; s.w[i * K] = 1; continue; }
    const double x = g.positions[3 * i], y = g.positions[3 * i + 1], z = g.positions[3 * i + 2];
    std::vector<std::pair<double, I>> best;
    for (I v : walls) {
      const auto u = static_cast<size_t>(v);
      const double d = nxr::ladder::jsHypot(P[3 * u] - x, P[3 * u + 1] - y, P[3 * u + 2] - z);
      if (best.size() < static_cast<size_t>(K) || d < best.back().first) {
        best.emplace_back(d, v);
        std::stable_sort(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        if (best.size() > static_cast<size_t>(K)) best.pop_back();
      }
    }
    double sum = 0;
    for (const auto& [d, v] : best) sum += 1 / std::max(d, 1e-9);
    for (std::size_t k = 0; k < best.size(); ++k) { s.idx[i * K + k] = best[k].second; s.w[i * K + k] = static_cast<float>(1 / std::max(best[k].first, 1e-9) / sum); }
  }
  return s;
}

}  // namespace nxr::glue
