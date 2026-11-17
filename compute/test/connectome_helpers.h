// Synthetic data for the connectome tests: a sphere as a surface, fibers whose ends are drawn clear of ties, an independent DENSE implementation of the
// reference's behaviour (tess_connectome: nearest vertex, fiber counts as E1' E2 + E2' E1 with the diagonal removed, mesh smoothing P W P', pruning by a fraction of the
// strongest weight, connected components). Nothing here is shared with the code under test beyond the mesh helper.
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <vector>

#include "mesh_helpers.h"

namespace ctest {

struct Lcg {   // a portable generator: the same numbers on every platform
  std::uint64_t s;
  explicit Lcg(std::uint64_t seed) : s(seed) {}
  double next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>((s >> 11) & ((1ULL << 53) - 1)) / static_cast<double>(1ULL << 53); }
  double range(double a, double b) { return a + (b - a) * next(); }
};

struct Bundle {
  std::int64_t nF = 0, nP = 0;
  std::vector<double> points;   // [nF*nP x 3]
};

inline void unitRandom(Lcg& g, double* p) {
  double n;
  do { for (int k = 0; k < 3; ++k) p[k] = g.range(-1, 1); n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); } while (n < 0.1 || n > 1);
  for (int k = 0; k < 3; ++k) p[k] /= n;
}

// nF fibers of nP points from one random end to another, the middle bent by a random amount; `self` fibers have both ends within a hair of one point; `dup` fibers repeat an earlier one.
inline Bundle makeBundle(Lcg& g, std::int64_t nF, std::int64_t nP, double radius, int self, int dup) {
  Bundle b;
  b.nF = nF; b.nP = nP;
  b.points.assign(static_cast<std::size_t>(nF * nP * 3), 0.0);
  for (std::int64_t f = 0; f < nF; ++f) {
    double a[3], c[3], bend[3];
    unitRandom(g, a);
    unitRandom(g, c);
    for (int k = 0; k < 3; ++k) { a[k] *= radius * g.range(0.97, 1.03); c[k] *= radius * g.range(0.97, 1.03); bend[k] = radius * 0.2 * g.range(-1, 1); }
    if (f < self) for (int k = 0; k < 3; ++k) c[k] = a[k] * (1 + 1e-6 * g.range(-1, 1));
    for (std::int64_t p = 0; p < nP; ++p) {
      const double t = static_cast<double>(p) / static_cast<double>(nP - 1), w = 4 * t * (1 - t);
      for (int k = 0; k < 3; ++k) b.points[static_cast<std::size_t>(3 * (f * nP + p) + k)] = (1 - t) * a[k] + t * c[k] + w * bend[k];
    }
  }
  for (int d = 0; d < dup; ++d) {   // the last `dup` fibers repeat fibers 0 .. dup-1
    const std::int64_t from = d, to = nF - 1 - d;
    for (std::int64_t i = 0; i < nP * 3; ++i) b.points[static_cast<std::size_t>(to * nP * 3 + i)] = b.points[static_cast<std::size_t>(from * nP * 3 + i)];
  }
  return b;
}

// ── the reference's behaviour, dense and independent ────────────────────────────────────────────────────────────
inline int bruteNearest(const std::vector<double>& pos, const double* q) {
  int best = 0; double bd = 1e300;
  for (std::size_t v = 0; v < pos.size() / 3; ++v) {
    double d = 0;
    for (int k = 0; k < 3; ++k) { const double t = pos[3 * v + static_cast<std::size_t>(k)] - q[k]; d += t * t; }
    if (d < bd) { bd = d; best = static_cast<int>(v); }
  }
  return best;
}
// per fiber [first end vertex, last end vertex] with the reference's nearest vertex
inline std::vector<std::array<int, 2>> referenceEnds(const std::vector<double>& pos, const Bundle& b) {
  std::vector<std::array<int, 2>> e;
  for (std::int64_t f = 0; f < b.nF; ++f)
    e.push_back({bruteNearest(pos, &b.points[static_cast<std::size_t>(3 * f * b.nP)]), bruteNearest(pos, &b.points[static_cast<std::size_t>(3 * (f * b.nP + b.nP - 1))])});
  return e;
}
// W = E1' E2 + E2' E1 (the diagonal kept: a fiber with both ends on one vertex adds 2 there)
inline Eigen::MatrixXd referenceCounts(const std::vector<std::array<int, 2>>& e, int V, const std::vector<std::uint8_t>& keep = {}) {
  Eigen::MatrixXd w = Eigen::MatrixXd::Zero(V, V);
  for (std::size_t f = 0; f < e.size(); ++f) {
    if (!keep.empty() && !keep[f]) continue;
    w(e[f][0], e[f][1]) += 1; w(e[f][1], e[f][0]) += 1;
  }
  return w;
}
inline Eigen::MatrixXd referenceSmoother(const std::vector<std::int32_t>& F, int V, int hops) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Identity(V, V);
  for (std::size_t i = 0; i < F.size(); i += 3)
    for (int x = 0; x < 3; ++x) { const int u = F[i + static_cast<std::size_t>(x)], v = F[i + static_cast<std::size_t>((x + 1) % 3)]; a(u, v) = 1; a(v, u) = 1; }
  for (int r = 0; r < V; ++r) a.row(r) /= a.row(r).sum();
  Eigen::MatrixXd p = Eigen::MatrixXd::Identity(V, V);
  for (int h = 0; h < hops; ++h) p = p * a;
  return p;
}
inline Eigen::MatrixXd withoutDiag(Eigen::MatrixXd m) { for (int i = 0; i < m.rows(); ++i) m(i, i) = 0; return m; }
// the connected components by breadth-first search, numbered by lowest vertex
inline std::vector<int> referenceComponents(const Eigen::MatrixXd& w) {
  const int n = static_cast<int>(w.rows());
  std::vector<int> lab(static_cast<std::size_t>(n), -1);
  int next = 0;
  for (int s = 0; s < n; ++s) {
    if (lab[static_cast<std::size_t>(s)] >= 0) continue;
    std::vector<int> q{s};
    lab[static_cast<std::size_t>(s)] = next;
    for (std::size_t h = 0; h < q.size(); ++h)
      for (int v = 0; v < n; ++v) if (w(q[h], v) != 0 && lab[static_cast<std::size_t>(v)] < 0) { lab[static_cast<std::size_t>(v)] = next; q.push_back(v); }
    ++next;
  }
  return lab;
}

}  // namespace ctest
