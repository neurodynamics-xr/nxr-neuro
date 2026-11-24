// nxr::geometry::glue — the Studio's glue.ts (glueMesh, sourcesOf), line for line: Map/Set insertion orders, strict-versus-tie comparisons, Math.round and
// Math.hypot (ladder::jsHypot) as JavaScript evaluates them; except the cut, which removes exactly the medial wall (the reference widens it to one simple loop).
#include "nxr/geometry/surface_maps.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "nxr/select/dyadic.h"

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

// The EXACT cut of one hemisphere's medial wall: every wall vertex and every face touching one removed, nothing else. The hole's rim is walked as the
// boundary of what is kept: each removed face with exactly one wall vertex gives its other edge as a rim half-edge (in the kept side's winding, as the
// kept faces' unmatched half-edges run), and from a rim half-edge u -> v the next one is found by turning around v through the removed faces to the first
// vertex off the wall. Every rim half-edge has one successor and one predecessor, so the rim is a set of closed walks. A vertex may appear more than once in
// a walk (a PINCH): a cortex vertex whose faces all touch the wall is reached along its one edge to the rest of the cortex and left along the same edge
// (a spike u -> v -> u), and two lobes of the hole may meet at one vertex. Both are valid closed curves for a band of triangles (stitch keeps the band
// manifold at them). Refused, never repaired (the glue neither removes cortex nor keeps wall): a wall of several pieces, a rim of several walks (a wall
// piece with cortex enclosed by it, or several holes), a cortex vertex enclosed by the wall with no edge to the rest.
struct Disk { std::vector<I> faces, loop; std::size_t repeated = 0; };
Disk cutExactly(const std::vector<I>& F, const std::vector<std::uint8_t>& piece, const std::vector<std::uint8_t>& wall, int which) {
  const std::size_t n = piece.size();
  std::vector<std::uint8_t> wallHere(n, 0);
  std::size_t wallCount = 0;
  for (std::size_t i = 0; i < n; ++i) if (piece[i] && wall[i]) { wallHere[i] = 1; ++wallCount; }
  const std::string hemisphere = "glue: hemisphere " + std::to_string(which);
  if (wallCount == 0) fail(hemisphere + " has no medial wall in the atlas \xe2\x80\x94 the hole is cut by anatomy only");
  std::size_t wallPieces = 0;
  { const std::vector<I> lab = labels(n, F, &wallHere); std::unordered_set<I> roots; for (std::size_t i = 0; i < n; ++i) if (lab[i] >= 0) roots.insert(lab[i]); wallPieces = roots.size(); }
  auto key = [&](I a, I b) { return static_cast<L>(a) * static_cast<L>(n) + b; };
  std::unordered_map<L, std::size_t> faceOfHalfEdge;                          // a half-edge of this hemisphere -> its face's first index in F
  Disk d;
  std::vector<std::pair<I, I>> rimHalfEdges;                                  // in the faces' order: the walks start from the first one still unwalked
  std::vector<std::uint8_t> onKeptFace(n, 0);
  for (std::size_t f = 0; f < F.size(); f += 3) {
    if (!piece[static_cast<size_t>(F[f])]) continue;
    int wallCorners = 0;
    for (int k = 0; k < 3; ++k) { faceOfHalfEdge[key(F[f + static_cast<size_t>(k)], F[f + static_cast<size_t>((k + 1) % 3)])] = f; wallCorners += wallHere[static_cast<size_t>(F[f + static_cast<size_t>(k)])]; }
    if (wallCorners == 0) { for (int k = 0; k < 3; ++k) { d.faces.push_back(F[f + static_cast<size_t>(k)]); onKeptFace[static_cast<size_t>(F[f + static_cast<size_t>(k)])] = 1; } continue; }
    if (wallCorners != 1) continue;
    for (int k = 0; k < 3; ++k) {
      const I a = F[f + static_cast<size_t>(k)], b = F[f + static_cast<size_t>((k + 1) % 3)];
      if (!wallHere[static_cast<size_t>(a)] && !wallHere[static_cast<size_t>(b)]) rimHalfEdges.emplace_back(b, a);
    }
  }
  // the next rim half-edge after u -> v: turn around v from the removed face holding v -> u, through faces whose far corner is on the wall
  auto thirdCorner = [&](I from, I to) -> I {
    const auto it = faceOfHalfEdge.find(key(from, to));
    if (it == faceOfHalfEdge.end()) fail(hemisphere + ": the surface is not closed at the medial wall's rim (half-edge " + std::to_string(from) + " -> " + std::to_string(to) + " has no face)");
    const std::size_t f = it->second;
    for (int k = 0; k < 3; ++k) if (F[f + static_cast<size_t>(k)] == from) return F[f + static_cast<size_t>((k + 2) % 3)];
    fail(hemisphere + ": a face lookup went wrong");
  };
  auto successor = [&](I u, I v) -> std::pair<I, I> {
    I c = thirdCorner(v, u);
    for (std::size_t turn = 0; turn <= n; ++turn) {
      const I x = thirdCorner(v, c);
      if (!wallHere[static_cast<size_t>(x)]) return {v, x};
      c = x;
    }
    fail(hemisphere + ": turning around vertex " + std::to_string(v) + " at the medial wall's rim never left the wall");
  };
  std::unordered_set<L> walked;
  std::vector<std::vector<I>> walks;
  for (const auto& start : rimHalfEdges) {
    if (walked.count(key(start.first, start.second))) continue;
    std::vector<I> walk;
    std::pair<I, I> e = start;
    while (walked.insert(key(e.first, e.second)).second) { walk.push_back(e.first); e = successor(e.first, e.second); }
    if (e != start) fail(hemisphere + ": the medial wall's rim does not close");
    walks.push_back(std::move(walk));
  }
  std::unordered_set<I> onRim;
  std::size_t visits = 0;
  for (const auto& w : walks) for (I v : w) { onRim.insert(v); ++visits; }
  d.repeated = visits - onRim.size();
  std::size_t enclosed = 0;                                                   // cortex vertices with no kept face and no edge to the rim: enclosed by the wall
  for (std::size_t i = 0; i < n; ++i) if (piece[i] && !wallHere[i] && !onKeptFace[i] && !onRim.count(static_cast<I>(i))) ++enclosed;
  if (wallPieces != 1 || walks.size() != 1 || enclosed != 0)
    fail(hemisphere + ": the exact cut of its medial wall (" + std::to_string(wallCount) + " vertices in " + std::to_string(wallPieces) + " piece" + (wallPieces == 1 ? "" : "s") + ") leaves " +
         std::to_string(walks.size()) + " rim loop" + (walks.size() == 1 ? "" : "s") + " with " + std::to_string(d.repeated) + " pinch vertex visit" + (d.repeated == 1 ? "" : "s") +
         (enclosed ? " and " + std::to_string(enclosed) + " cortex vertices enclosed by the wall" : std::string()) +
         "; the bridge needs ONE hole per hemisphere, and the glue never removes cortex nor keeps wall: refused (the medial wall of the atlas is to be checked)");
  d.loop = std::move(walks[0]);
  return d;
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
// Uselton 1977); a tie advances on A. A loop may pass through a vertex twice (a pinch: the spike u -> v -> u of a cortex vertex whose faces all touched the
// wall): at such an APEX v (its neighbours in the loop the same vertex) the band must advance on the other loop at least once before it leaves v, or the
// two visits of u would share an edge to the same vertex of the other loop and the surface would not be manifold there. The search keeps, per cell, the
// best band ending with an advance on A and the best ending with one on B; without an apex it is the plain search, cell for cell.
template <class At> std::vector<I> stitch(const std::vector<I>& A, const std::vector<I>& B0, const At& at) {
  std::size_t j0 = 0;
  for (std::size_t j = 1; j < B0.size(); ++j) if (dist(at(B0[j]), at(A[0])) < dist(at(B0[j0]), at(A[0]))) j0 = j;
  std::vector<I> B(B0.begin() + static_cast<long>(j0), B0.end()); B.insert(B.end(), B0.begin(), B0.begin() + static_cast<long>(j0));
  const std::size_t m = A.size(), nb = B.size(), W = nb + 1;
  auto a = [&](std::size_t i) { return A[i % m]; };
  auto b = [&](std::size_t j) { return B[j % nb]; };
  auto apexOfA = [&](std::size_t i) { return i < m && m > 2 && A[(i + m - 1) % m] == A[(i + 1) % m]; };
  auto apexOfB = [&](std::size_t j) { return j < nb && nb > 2 && B[(j + nb - 1) % nb] == B[(j + 1) % nb]; };
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> costAfterA((m + 1) * W, inf), costAfterB((m + 1) * W, inf);   // the best band reaching the cell whose last advance was on A / on B
  for (std::size_t i = 0; i <= m; ++i) for (std::size_t j = 0; j <= nb; ++j) {
    const std::size_t k = i * W + j;
    const bool start = i == 0 && j == 0;
    // leaving on A: not from an apex of A reached by an advance on A (or the start); leaving on B: not from an apex of B reached on B (or the start)
    const double leaveOnA = start ? (apexOfA(0) ? inf : 0.0) : std::min(costAfterB[k], apexOfA(i) ? inf : costAfterA[k]);
    const double leaveOnB = start ? (apexOfB(0) ? inf : 0.0) : std::min(costAfterA[k], apexOfB(j) ? inf : costAfterB[k]);
    if (i < m && leaveOnA < inf) { const double x = leaveOnA + dist(at(a(i + 1)), at(b(j))); double& c = costAfterA[(i + 1) * W + j]; if (x < c) c = x; }
    if (j < nb && leaveOnB < inf) { const double x = leaveOnB + dist(at(a(i)), at(b(j + 1))); double& c = costAfterB[i * W + j + 1]; if (x < c) c = x; }
  }
  if (std::min(costAfterA[m * W + nb], costAfterB[m * W + nb]) == inf) fail("glue: no band of triangles joins the two loops");
  std::vector<I> T;
  // back from the end: the last advance the cheaper (a tie: on A), then each step back the cheaper band its cell allows leaving that way
  bool lastOnA = costAfterA[m * W + nb] <= costAfterB[m * W + nb];
  for (std::size_t i = m, j = nb; i > 0 || j > 0;) {
    if (lastOnA) {
      T.push_back(a(i - 1)); T.push_back(a(i)); T.push_back(b(j)); --i;
      const std::size_t k = i * W + j;
      const double viaA = apexOfA(i) ? inf : costAfterA[k];
      lastOnA = viaA <= costAfterB[k];
    } else {
      T.push_back(a(i)); T.push_back(b(j)); T.push_back(b(j - 1)); --j;
      const std::size_t k = i * W + j;
      const double viaB = apexOfB(j) ? inf : costAfterB[k];
      lastOnA = costAfterA[k] <= viaB;
    }
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

// Why faces F on n vertices are NOT a sphere (empty: they are one): the first property that fails, in this order: CLOSED (every edge on exactly two faces),
// CONSISTENTLY ORIENTED (the two faces of an edge traverse it in opposite directions), CONNECTED (one piece; a vertex on no face is a piece of its own),
// and the Euler characteristic V - E + F = 2 (a closed, connected, orientable surface of genus 0).
std::string notSphere(const std::vector<I>& F, std::size_t n) {
  std::unordered_map<L, int> facesOn;                                   // an undirected edge (low * n + high): the faces it is on
  std::unordered_set<L> halfEdges;                                      // a directed edge (from * n + to), each face's three in its winding
  L reused = 0;
  for (std::size_t f = 0; f < F.size(); f += 3) for (int k = 0; k < 3; ++k) {
    const L a = F[f + static_cast<size_t>(k)], b = F[f + static_cast<size_t>((k + 1) % 3)];
    ++facesOn[std::min(a, b) * static_cast<L>(n) + std::max(a, b)];
    if (!halfEdges.insert(a * static_cast<L>(n) + b).second) ++reused;
  }
  L one = 0, more = 0;
  for (const auto& [edge, count] : facesOn) { (void)edge; if (count == 1) ++one; else if (count > 2) ++more; }
  if (one || more)
    return "not closed: every edge must lie on exactly two faces; " + std::to_string(one) + " edges lie on one face and " + std::to_string(more) + " on three or more";
  if (reused) return "not consistently oriented: " + std::to_string(reused) + " edges are traversed in the same direction by both their faces";
  { const std::vector<I> lb = labels(n, F, nullptr); const std::unordered_set<I> pieces(lb.begin(), lb.end());
    if (pieces.size() != 1) return "not connected: " + std::to_string(pieces.size()) + " pieces (a vertex on no face counts as one)"; }
  const L chi = static_cast<L>(n) - static_cast<L>(facesOn.size()) + static_cast<L>(F.size() / 3);
  if (chi != 2) return "Euler characteristic V - E + F = " + std::to_string(chi) + ", not 2 (a sphere's)";
  return {};
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
  std::vector<Disk> disks;
  for (int q = 0; q < 2; ++q) disks.push_back(cutExactly(F, pieces[static_cast<size_t>(q)], wall, q));
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
  // the kept vertices first: EVERY vertex off the wall, ascending (a cortex vertex whose faces all touch the wall is kept, on the rim), then ring r's N points
  std::vector<I> used;
  for (std::size_t i = 0; i < n; ++i) if (!wall[i]) used.push_back(static_cast<I>(i));
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
  // THE RESULT IS A SPHERE: closed, consistently oriented, connected, V - E + F = 2; otherwise refused, naming the property that failed (two hemispheres that are not each
  // a sphere, or a bridge that does not close them up, never pass as a cortex of one piece)
  if (const std::string why = notSphere(g.faces, total); !why.empty()) fail("glue: the glued surface is not a sphere: " + why);
  g.edges = static_cast<L>(total) + static_cast<L>(g.faces.size() / 3) - 2;   // a sphere's: V - E + F = 2
  double area = 0, longest = 0;
  for (std::size_t f = 0; f < bridge.size(); f += 3) {
    const Pt p = at(bridge[f]), q = at(bridge[f + 1]), r = at(bridge[f + 2]), u = sub3(q, p), w = sub3(r, p);
    area += 0.5 * nxr::select::jsHypot(u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]);
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
      const double d = nxr::select::jsHypot(P[3 * u] - x, P[3 * u + 1] - y, P[3 * u + 2] - z);
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

}  // namespace nxr::geometry::glue
