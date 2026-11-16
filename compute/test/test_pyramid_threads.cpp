// The threaded kernel products of nxr::pyramid (bandSums, tensorBandSums, frameKernel) are BIT-IDENTICAL to the serial path for every thread count (the sums are folded in block order);
// synthetic data only. The medium case prints the speedup (a number, not a pass condition).
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "nxr/parallel.h"
#include "nxr/partition.h"
#include "nxr/pyramid.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)
using namespace nxr;

namespace {
struct Case {
  int V, M, n, members, tiles;
  Eigen::MatrixXd K3, e1, e2, en, Kc;
  Eigen::MatrixXcd A;
  std::vector<char> sing;
  std::vector<std::uint8_t> good;
  std::vector<std::int32_t> codes, tile;
  std::vector<double> area, wt;
};
Case make(int V, int M, int n, int members, int tiles, unsigned seed) {
  Case c{V, M, n, members, tiles, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}};
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd;
  c.K3 = Eigen::MatrixXd::NullaryExpr(3 * V, M, [&] { return nd(rng); });
  c.Kc = Eigen::MatrixXd::NullaryExpr(V, M, [&] { return nd(rng); });
  c.e1 = Eigen::MatrixXd::NullaryExpr(V, 3, [&] { return nd(rng); }); c.e2 = Eigen::MatrixXd::NullaryExpr(V, 3, [&] { return nd(rng); }); c.en = Eigen::MatrixXd::NullaryExpr(V, 3, [&] { return nd(rng); });
  c.A = Eigen::MatrixXcd::NullaryExpr(n, M, [&] { return std::complex<double>(nd(rng), nd(rng)); });
  c.sing.assign(V, 0);
  for (int v = 0; v < V; v += 17) c.sing[v] = 1;
  c.good.assign(n, 1);
  for (int s = 0; s < n; ++s) if (s % 23 == 5) c.good[s] = 0;
  c.codes.resize(V); c.area.resize(V); c.wt.resize(V);
  for (int v = 0; v < V; ++v) { c.codes[v] = v % 11 == 3 ? -1 : static_cast<int>(rng() % members); c.area[v] = 0.5 + (rng() % 100) / 100.0; c.wt[v] = c.sing[v] ? 0.0 : c.area[v]; }
  c.tile.resize(n);
  for (int s = 0; s < n; ++s) c.tile[s] = s % 37 == 0 ? -1 : static_cast<int>((static_cast<std::int64_t>(s) * tiles) / n);   // tiles cut across blocks
  return c;
}
bool same(const std::vector<double>& a, const std::vector<double>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0; }
double secs(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
}  // namespace

int main() {
  // bit-identity on a small case, an awkward block size (tiles cut by blocks, a partial last block)
  {
    Case c = make(300, 20, 700, 9, 25, 3);
    const pyramid::FrameKernel f1 = pyramid::frameKernel(c.K3, c.e1, c.e2, c.en, c.sing, 1);
    partition::AnalyticSums a1(c.members, c.tiles);
    partition::TensorSums t1(c.members, c.tiles);
    pyramid::bandSums(a1, c.Kc, c.A, c.good.data(), c.codes.data(), c.area.data(), c.tile.data(), 61, 1);
    pyramid::tensorBandSums(t1, f1, c.A, c.good.data(), c.codes.data(), c.wt.data(), c.area.data(), c.tile.data(), 61, 1);
    for (int th : {2, 3, 5, 16}) {
      const pyramid::FrameKernel f = pyramid::frameKernel(c.K3, c.e1, c.e2, c.en, c.sing, th);
      EXPECT(f.K1 == f1.K1 && f.K2 == f1.K2 && f.Kn == f1.Kn, "frameKernel is identical with " << th << " threads");
      partition::AnalyticSums a(c.members, c.tiles);
      partition::TensorSums t(c.members, c.tiles);
      pyramid::bandSums(a, c.Kc, c.A, c.good.data(), c.codes.data(), c.area.data(), c.tile.data(), 61, th);
      pyramid::tensorBandSums(t, f, c.A, c.good.data(), c.codes.data(), c.wt.data(), c.area.data(), c.tile.data(), 61, th);
      EXPECT(same(a.power, a1.power) && same(a.env, a1.env) && same(a.envmax, a1.envmax), "bandSums is bit-identical with " << th << " threads");
      for (int k = 0; k < 10; ++k) EXPECT(same(*t.arrays()[k], *t1.arrays()[k]), "tensor " << partition::TensorSums::keys()[k] << " is bit-identical with " << th << " threads");
    }
    double s = 0; for (double x : t1.t11) s += x;
    EXPECT(s > 0, "the case sums something");
  }
  // an error in a worker is rethrown, not lost
  {
    bool threw = false;
    try { parallel::orderedPieces(10, 3, [](std::int64_t i, int) { if (i == 4) throw std::runtime_error("x"); }, [](std::int64_t, int) {}); } catch (const std::runtime_error&) { threw = true; }
    EXPECT(threw, "an exception in a worker reaches the caller");
  }
  // the speedup on a medium case (printed)
  {
    const int th = parallel::resolveThreads(0);
    Case c = make(4000, 100, 8192, 64, 16, 7);
    const pyramid::FrameKernel f = pyramid::frameKernel(c.K3, c.e1, c.e2, c.en, c.sing, 1);
    partition::TensorSums t1(c.members, c.tiles), tn(c.members, c.tiles);
    auto t0 = std::chrono::steady_clock::now();
    pyramid::tensorBandSums(t1, f, c.A, c.good.data(), c.codes.data(), c.wt.data(), c.area.data(), c.tile.data(), 128, 1);
    const double s1 = secs(t0);
    t0 = std::chrono::steady_clock::now();
    pyramid::tensorBandSums(tn, f, c.A, c.good.data(), c.codes.data(), c.wt.data(), c.area.data(), c.tile.data(), 128, th);
    const double sn = secs(t0);
    bool ok = true; for (int k = 0; k < 10; ++k) ok = ok && same(*t1.arrays()[k], *tn.arrays()[k]);
    EXPECT(ok, "the medium case is bit-identical with " << th << " threads");
    std::cout << "  tensorBandSums, " << c.V * 3 << " x " << c.M << " kernel, " << c.n << " samples: 1 thread " << s1 << " s, " << th << " threads " << sn << " s, speedup " << s1 / sn << "x\n";
  }
  if (fails) { std::cout << fails << " failed\n"; return 1; }
  std::cout << "test_pyramid_threads: ok\n";
  return 0;
}
