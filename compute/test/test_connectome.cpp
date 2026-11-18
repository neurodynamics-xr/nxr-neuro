// nxr::meeg, nxr::geometry::nearest and the sparse congruence: one test a function, and the VERTEX-level fiber counts, connection weights, pruning and largest component against an
// independent DENSE implementation of the reference's behaviour (tess_connectome; test/connectome_helpers.h) on synthetic fibers and a synthetic surface, gate none.
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "connectome_helpers.h"
#include "nxr/core/congruence.h"
#include "nxr/meeg/connectome.h"
#include "nxr/geometry/surface_maps.h"

using namespace nxr::meeg;
using Dense = Eigen::MatrixXd;

static bool refused(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
template <class Scalar>
static Sparse<Scalar> fromDense(const Eigen::MatrixXd& d) {
  std::vector<Eigen::Triplet<Scalar>> t;
  for (int i = 0; i < d.rows(); ++i) for (int j = 0; j < d.cols(); ++j) if (d(i, j) != 0) t.emplace_back(i, j, static_cast<Scalar>(d(i, j)));
  Sparse<Scalar> m(static_cast<int>(d.rows()), static_cast<int>(d.cols()));
  m.setFromTriplets(t.begin(), t.end());
  return m;
}
static double maxAbs(const Dense& d) { return d.size() ? d.cwiseAbs().maxCoeff() : 0; }

int main() {
  std::vector<double> vertexPositions; std::vector<int32_t> triangles;
  icosphere(2, vertexPositions, triangles, 0.08);   // 162 vertices, a sphere of 8 cm (metres)
  const int nV = static_cast<int>(vertexPositions.size() / 3);

  std::cout << "=== fiberEnds: the first and last rows, order f then nF + f; interleaving ===\n";
  {
    ctest::Lcg g(1);
    const auto b = ctest::makeBundle(g, 7, 5, 0.08, 0, 0);
    const auto e = fiberEnds<double>(b.points, 5);
    bool ok = e.size() == 6 * 7;
    for (int f = 0; f < 7; ++f) for (int k = 0; k < 3; ++k)
      ok = ok && e[static_cast<std::size_t>(3 * f + k)] == b.points[static_cast<std::size_t>(3 * (f * 5) + k)] && e[static_cast<std::size_t>(3 * (7 + f) + k)] == b.points[static_cast<std::size_t>(3 * (f * 5 + 4) + k)];
    EXPECT(ok, "fiberEnds rows");
    EXPECT(refused([&] { fiberEnds<double>(b.points, 1); }) && refused([&] { fiberEnds<double>(b.points, 4); }), "fiberEnds refuses nP < 2 and a ragged bundle");
    const std::vector<int> blockOrder = {10, 11, 12, 20, 21, 22};
    const auto inter = interleaveEnds(blockOrder);
    EXPECT((inter == std::vector<int>{10, 20, 11, 21, 12, 22}) && blockEnds(inter) == blockOrder, "interleaveEnds and blockEnds are inverse");
  }

  std::cout << "=== nearestVertices (k-d tree): equal to the brute force, the lowest index on ties, the distances ===\n";
  {
    ctest::Lcg g(2);
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<double> cloud, q;
      for (int i = 0; i < 2001; ++i) cloud.push_back(g.range(-1, 1));
      for (int i = 0; i < 900; ++i) q.push_back(g.range(-1.2, 1.2));
      if (pass == 0) {
        const auto t = nxr::geometry::nearest::nearestVertices<double>(cloud, q), r = nxr::geometry::nearest::nearestVerticesBrute<double>(cloud, q);
        EXPECT(t.vertex == r.vertex && t.distance == r.distance, "random cloud, float64: the tree equals the brute force");
      } else {
        const std::vector<float> cf(cloud.begin(), cloud.end()), qf(q.begin(), q.end());
        const auto t = nxr::geometry::nearest::nearestVertices<float>(cf, qf), r = nxr::geometry::nearest::nearestVerticesBrute<float>(cf, qf);
        EXPECT(t.vertex == r.vertex && t.distance == r.distance, "random cloud, float32: the tree equals the brute force");
        static_assert(std::is_same<decltype(t.distance)::value_type, float>::value, "float32 in, float32 out");
      }
    }
    const std::vector<double> lattice = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0}, q = {0.5, 0.5, 0, 0.5, 0, 0, 1, 1, 0.3};
    const auto r = nxr::geometry::nearest::nearestVertices<double>(lattice, q);
    EXPECT(r.vertex[0] == 0 && r.vertex[1] == 0 && r.vertex[2] == 3, "exact ties go to the lowest index");
    EXPECT(std::abs(r.distance[0] - std::sqrt(0.5)) < 1e-15 && std::abs(r.distance[1] - 0.5) < 1e-15 && std::abs(r.distance[2] - 0.3) < 1e-15, "the distances");
    std::vector<double> dup = {1, 1, 1, 0, 0, 0, 0, 0, 0};
    EXPECT(nxr::geometry::nearest::nearestVertices<double>(dup, {0, 0, 0}).vertex[0] == 1, "two vertices at one place: the lower index");
    EXPECT(refused([&] { nxr::geometry::nearest::nearestVertices<double>({}, {0, 0, 0}); }) && refused([&] { nxr::geometry::nearest::nearestVertices<double>({0, 0}, {0, 0, 0}); }), "refusals");
  }

  std::cout << "=== endConnectome: symmetric; the total is 2 x the fibers with distinct end vertices; a self-pair leaves no entry; two fibers on one pair give 2 ===\n";
  {
    const std::vector<std::int32_t> ends = {0, 0, 2, 5, 5, 1 /*first ends f=0..2: 0,0,2 ; wait: block order*/};
    // three fibers: (0,5), (0,5), (2,1) in block order [first ends | last ends]
    const std::vector<std::int32_t> e = {0, 0, 2, 5, 5, 1};
    const auto w = endConnectome<double>(e, {}, 6);
    EXPECT(w.coeff(0, 5) == 2 && w.coeff(5, 0) == 2 && w.coeff(2, 1) == 1 && w.coeff(1, 2) == 1 && w.nonZeros() == 4, "two fibers on one pair give 2, symmetric");
    const std::vector<std::int32_t> self = {3, 3, 4, 3, 3, 4};   // (3,3) twice and (4,4)? block: firsts 3,3,4 ; lasts 3,3,4 -> all self-pairs
    EXPECT(endConnectome<double>(self, {}, 6).nonZeros() == 0, "a fiber with both ends on one vertex leaves no entry");
    const std::vector<std::uint8_t> keep = {1, 0, 1};
    EXPECT(endConnectome<double>(e, keep, 6).coeff(0, 5) == 1, "a fiber the gate left out adds nothing");
    EXPECT(refused([&] { endConnectome<double>(e, {}, 5); }), "a vertex outside the surface is refused");
    (void)ends;
  }

  std::cout << "=== smoothingOperator: rows sum to 1; 0 hops is the identity; on a regular lattice the support after p hops is the p-ring ===\n";
  {
    const auto p3 = smoothingOperator<double>(triangles, nV, 3);
    bool sums = true;
    for (int i = 0; i < nV; ++i) { double s = 0; for (Sparse<double>::InnerIterator it(Sparse<double>(p3.transpose()), i); it; ++it) s += it.value(); (void)s; }
    const Dense dp = Dense(p3);
    for (int i = 0; i < nV; ++i) sums = sums && std::abs(dp.row(i).sum() - 1) < 1e-14;
    EXPECT(sums, "rows sum to 1 (3 hops)");
    EXPECT(Dense(smoothingOperator<double>(triangles, nV, 0)) == Dense::Identity(nV, nV), "0 hops is the identity");
    EXPECT((Dense(smoothingOperator<double>(triangles, nV, 3)) - ctest::referenceSmoother(triangles, nV, 3)).cwiseAbs().maxCoeff() < 1e-14, "equals the dense reference smoother");
    // a triangulated square lattice, six neighbours inside: the support of the centre after p hops is the hexagon of 1 + 3p(p+1) vertices
    const int n = 11;
    std::vector<std::int32_t> lat;
    for (int i = 0; i + 1 < n; ++i) for (int j = 0; j + 1 < n; ++j) {
      const int a = i * n + j, b = a + 1, c = a + n, d = c + 1;
      for (int v : {a, b, d, a, d, c}) lat.push_back(v);
    }
    bool ring = true;
    for (int p = 0; p <= 3; ++p) {
      const Dense s = Dense(smoothingOperator<double>(lat, n * n, p));
      int support = 0;
      for (int j = 0; j < n * n; ++j) support += s(5 * n + 5, j) > 0;
      ring = ring && support == 1 + 3 * p * (p + 1);
    }
    EXPECT(ring, "support after p hops is the p-ring (1, 7, 19, 37)");
    EXPECT(refused([&] { smoothingOperator<double>(triangles, nV, -1); }), "hops >= 0");
  }

  std::cout << "=== sparseCongruence: equals the dense projection · connections · projectionᵀ (and the blocked form, pruned) ===\n";
  {
    ctest::Lcg g(3);
    auto randomSparse = [&](int r, int c, double density, bool sym) {
      Dense d = Dense::Zero(r, c);
      for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) if (g.next() < density) d(i, j) = g.range(0.1, 2);
      if (sym) d = (d + d.transpose()).eval();
      return d;
    };
    for (int t = 0; t < 4; ++t) {
      const Dense projection = randomSparse(9 + t, 13, 0.3, false), connections = randomSparse(13, 13, 0.25, true);
      const Dense want = projection * connections * projection.transpose();
      EXPECT((Dense(nxr::core::sparseCongruence<double>(fromDense<double>(projection), fromDense<double>(connections))) - want).cwiseAbs().maxCoeff() < 1e-12, "random sparse " << t << ": equals the dense product");
      const double drop = 1.5;
      Dense pr = want;
      for (int i = 0; i < pr.rows(); ++i) for (int j = 0; j < pr.cols(); ++j) if (std::abs(pr(i, j)) < drop) pr(i, j) = 0;
      EXPECT((Dense(nxr::core::sparseCongruenceBlocked<double>(fromDense<double>(projection), fromDense<double>(connections), 4, drop)) - pr).cwiseAbs().maxCoeff() < 1e-12, "random sparse " << t << ": blocked and pruned equals the pruned dense product");
    }
    EXPECT(refused([&] { nxr::core::sparseCongruence<double>(fromDense<double>(Dense::Ones(2, 3)), fromDense<double>(Dense::Ones(2, 2))); }), "a shape mismatch is refused");
  }

  std::cout << "=== pruneBelow, componentsOf, largestComponent: by hand on a 4 x 4, two blocks, a tie, isolated vertices ===\n";
  {
    Dense w = Dense::Zero(4, 4);
    w(0, 1) = w(1, 0) = 10; w(2, 3) = w(3, 2) = 0.5;
    EXPECT(strongest<double>(fromDense<double>(w)) == 10, "strongest");
    const auto pruned = pruneBelow<double>(fromDense<double>(w), 0.1);   // t = 1: removes 0.5
    EXPECT(pruned.nonZeros() == 2 && pruned.coeff(0, 1) == 10, "entries below the fraction of the strongest are removed");
    EXPECT(pruneBelow<double>(fromDense<double>(w), 0.0).nonZeros() == 4 && pruneBelow<double>(Sparse<double>(4, 4), 0.5).nonZeros() == 0, "fraction 0 removes nothing; no entry, nothing removed");
    const auto lab = componentsOf<double>(pruned);
    EXPECT((lab == std::vector<std::int32_t>{0, 0, 1, 2}), "isolated vertices are components, numbered by their lowest vertex");
    std::vector<std::int64_t> sizes;
    EXPECT((largestComponent(lab, &sizes) == std::vector<std::int32_t>{0, 1}) && sizes == std::vector<std::int64_t>({2, 1, 1}), "the largest component and the sizes");
    Dense blocks = Dense::Zero(7, 7);   // {0,1,2} chain, {3,4}, {5}, {6}
    blocks(0, 1) = blocks(1, 0) = blocks(1, 2) = blocks(2, 1) = 1; blocks(3, 4) = blocks(4, 3) = 1;
    EXPECT((largestComponent(componentsOf<double>(fromDense<double>(blocks))) == std::vector<std::int32_t>{0, 1, 2}), "two blocks of different sizes");
    Dense tie = Dense::Zero(5, 5);
    tie(0, 1) = tie(1, 0) = 1; tie(2, 3) = tie(3, 2) = 1;
    EXPECT((largestComponent(componentsOf<double>(fromDense<double>(tie))) == std::vector<std::int32_t>{0, 1}), "a tie goes to the lowest-numbered component");
  }

  std::cout << "=== VERTEX-level connectome against the reference's behaviour (dense, independent), gate none ===\n";
  for (int hops : {0, 1, 3}) {
    for (double prune : {0.0, 1e-4, 0.05}) {
      ctest::Lcg g(100 + static_cast<std::uint64_t>(hops));
      const auto b = ctest::makeBundle(g, 240, 6, 0.08, 6, 8);   // 6 fibers with both ends on one vertex, 8 repeated fibers
      const auto ref = ctest::referenceEnds(vertexPositions, b);
      // the code under test, through the same chain
      const auto ends = fiberEnds<double>(b.points, b.nP);
      const auto near = nxr::geometry::nearest::nearestVertices<double>(vertexPositions, ends);
      const auto counts = endConnectome<double>(near.vertex, keepAll(b.nF), nV);
      const Dense refCounts = ctest::withoutDiag(ctest::referenceCounts(ref, nV));
      EXPECT(Dense(counts) == refCounts, "hops " << hops << " prune " << prune << ": fiber counts identical to the reference's (pattern and value)");
      if (prune == 0.0 || hops == 0) {
        // (the weights and the rest are checked once per hops/prune pair)
      }
      const auto weights = pruneBelow<double>(connectionWeights<double>(counts, triangles, hops), prune);
      const Dense smoothing = ctest::referenceSmoother(triangles, nV, hops);
      Dense refW = ctest::withoutDiag(smoothing * ctest::referenceCounts(ref, nV) * smoothing.transpose());
      // note: the reference smooths the matrix with its diagonal already removed
      refW = ctest::withoutDiag(smoothing * refCounts * smoothing.transpose());
      const double t = prune * refW.maxCoeff();
      for (int i = 0; i < nV; ++i) for (int j = 0; j < nV; ++j) if (refW(i, j) < t) refW(i, j) = 0;
      const Dense got(weights);
      bool samePattern = true;
      for (int i = 0; i < nV && samePattern; ++i) for (int j = 0; j < nV; ++j) if ((got(i, j) != 0) != (refW(i, j) != 0) && std::abs(refW(i, j) - t) > 1e-9 * (1 + t)) { samePattern = false; break; }
      const double rel = maxAbs(got - refW) / (refW.size() ? std::max(1e-300, maxAbs(refW)) : 1);
      EXPECT(samePattern && rel < 1e-12, "hops " << hops << " prune " << prune << ": connection weights: pattern identical, values to 1e-12 relative (" << rel << ")");
      const auto lab = componentsOf<double>(weights);
      const auto refLab = ctest::referenceComponents(refW);
      EXPECT(std::vector<int>(lab.begin(), lab.end()) == refLab, "hops " << hops << " prune " << prune << ": components identical to a breadth-first search");
      const auto big = largestComponent(lab);
      std::vector<int> cnt(refLab.size() + 1, 0);
      int bi = 0;
      for (int l : refLab) ++cnt[static_cast<std::size_t>(l)];
      for (std::size_t c = 0; c < cnt.size(); ++c) if (cnt[c] > cnt[static_cast<std::size_t>(bi)]) bi = static_cast<int>(c);
      std::vector<std::int32_t> refBig;
      for (int v = 0; v < nV; ++v) if (refLab[static_cast<std::size_t>(v)] == bi) refBig.push_back(v);
      EXPECT(big == refBig, "hops " << hops << " prune " << prune << ": the largest component's vertices");
    }
  }

  std::cout << "=== precision follows the input: float32 in, float32 out ===\n";
  {
    ctest::Lcg g(7);
    const auto b = ctest::makeBundle(g, 50, 4, 0.08, 0, 0);
    const std::vector<float> posf(vertexPositions.begin(), vertexPositions.end()), ptsf(b.points.begin(), b.points.end());
    const auto ends = fiberEnds<float>(ptsf, 4);
    const auto near = nxr::geometry::nearest::nearestVertices<float>(posf, ends);
    const auto w = connectionWeights<float>(endConnectome<float>(near.vertex, {}, nV), triangles, 2);
    static_assert(std::is_same<decltype(w)::Scalar, float>::value, "float32 weights");
    EXPECT(w.nonZeros() > 0, "float32 chain ran");
  }

  std::cout << (g_failures ? "FAILED\n" : "all connectome tests passed\n");
  return g_failures ? 1 : 0;
}
