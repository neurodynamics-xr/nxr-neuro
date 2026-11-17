// The vertex-level connectome against the maintainers' MATLAB reference, tess_connectome (golden/connectome_golden.txt, made once by golden/gen_connectome_golden.m on the
// synthetic sphere and fibers of test_connectome.cpp; the generator only calls the reference, and the golden is its output): the fiber counts identical in pattern and value, the
// smoothed connection weights, their pruning and the largest component's vertices at 0, 1 and 3 hops and EdgeThr 0, 1e-4 and 0.05 (pattern identical, values to 1e-12 relative:
// the order of the sparse sums differs from MATLAB's), gate none. First the golden's inputs are checked against the ones this test builds (connectome_helpers.h: the same
// icosphere, the same portable generator), so the reference and the code under test were given the same surface and fibers.
#include <cmath>
#include <iostream>

#include "connectome_helpers.h"
#include "golden_io.h"
#include "nxr/connectome.h"
#include "nxr/nearest.h"

#ifndef NXR_TEST_GOLDEN_DIR
#error "NXR_TEST_GOLDEN_DIR is not set"
#endif

using namespace nxr::connectome;
using Dense = Eigen::MatrixXd;

namespace {
// the full symmetric matrix of a golden's triplets (i j value, 1-based, the upper triangle)
Dense symmetricFrom(const Dense& t, int n) {
  Dense w = Dense::Zero(n, n);
  for (int r = 0; r < t.rows(); ++r) { const int i = static_cast<int>(t(r, 0)) - 1, j = static_cast<int>(t(r, 1)) - 1; w(i, j) = w(j, i) = t(r, 2); }
  return w;
}
double maxAbsDiff(const Dense& a, const Dense& b) { return a.size() ? (a - b).cwiseAbs().maxCoeff() : 0.0; }
}  // namespace

int main() {
  Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/connectome_golden.txt");
  EXPECT(g.good(), "connectome golden opens");

  std::vector<double> pos;
  std::vector<int32_t> tri;
  icosphere(2, pos, tri, 0.08);
  const int nV = static_cast<int>(pos.size() / 3);
  const Dense gv = g.mat("vertices"), gf = g.mat("faces");
  bool sameMesh = gv.rows() == nV && gf.rows() == static_cast<int>(tri.size() / 3);
  for (int v = 0; v < nV && sameMesh; ++v) for (int k = 0; k < 3; ++k) sameMesh = sameMesh && std::abs(gv(v, k) - pos[static_cast<std::size_t>(3 * v + k)]) < 1e-13;
  for (std::size_t i = 0; i < tri.size() && sameMesh; ++i) sameMesh = static_cast<int>(gf(static_cast<Eigen::Index>(i / 3), static_cast<Eigen::Index>(i % 3))) == tri[i] + 1;
  EXPECT(sameMesh, "the golden's sphere is this test's (vertices to 1e-13, triangles identical)");
  std::vector<double> gpos(static_cast<std::size_t>(3 * nV));   // the reference was given these
  for (int v = 0; v < nV; ++v) for (int k = 0; k < 3; ++k) gpos[static_cast<std::size_t>(3 * v + k)] = gv(v, k);

  for (int hops : {0, 1, 3}) {
    const std::string h = std::to_string(hops);
    std::cout << "=== " << hops << " hops ===\n";
    const Dense ends = g.mat("ends_" + h);   // [fibers x 6]: the first end, the last end
    const int nF = static_cast<int>(ends.rows());
    ctest::Lcg gen(100 + static_cast<std::uint64_t>(hops));
    const auto b = ctest::makeBundle(gen, 240, 6, 0.08, 6, 8);
    bool sameFibers = nF == b.nF;
    for (int f = 0; f < nF && sameFibers; ++f)
      for (int k = 0; k < 3; ++k)
        sameFibers = sameFibers && std::abs(ends(f, k) - b.points[static_cast<std::size_t>(3 * (f * b.nP) + k)]) < 1e-13 && std::abs(ends(f, 3 + k) - b.points[static_cast<std::size_t>(3 * (f * b.nP + b.nP - 1) + k)]) < 1e-13;
    EXPECT(sameFibers, "the golden's fiber ends are this test's (to 1e-13)");

    // the chain under test on what the reference read: nearest vertex, counts (gate none)
    std::vector<double> e(static_cast<std::size_t>(6 * nF));   // block order: the first ends, then the last ends
    for (int f = 0; f < nF; ++f) for (int k = 0; k < 3; ++k) { e[static_cast<std::size_t>(3 * f + k)] = ends(f, k); e[static_cast<std::size_t>(3 * (nF + f) + k)] = ends(f, 3 + k); }
    const auto near = nxr::nearest::nearestVertices<double>(gpos, e);
    const auto counts = endConnectome<double>(near.vertex, keepAll(nF), nV);
    EXPECT(Dense(counts) == symmetricFrom(g.mat("counts_" + h), nV), "fiber counts identical to the reference's (pattern and value)");

    int k = 0;
    for (double prune : {0.0, 1e-4, 0.05}) {
      const std::string key = h + "_" + std::to_string(k++);
      const Dense gw = symmetricFrom(g.mat("weights_" + key), nV);
      const Dense keepTable = g.mat("keep_" + key);
      const auto weights = pruneBelow<double>(connectionWeights<double>(counts, tri, hops), prune);
      const Dense got(weights);
      bool samePattern = true;
      for (int i = 0; i < nV && samePattern; ++i) for (int j = 0; j < nV; ++j) if ((got(i, j) != 0) != (gw(i, j) != 0)) { samePattern = false; break; }
      const double rel = maxAbsDiff(got, gw) / std::max(1e-300, gw.size() ? gw.cwiseAbs().maxCoeff() : 1.0);
      EXPECT(samePattern, "hops " << hops << " prune " << prune << ": connection weights, pattern identical to the reference's");
      EXPECT(rel < 1e-12, "hops " << hops << " prune " << prune << ": connection weights to 1e-12 relative (" << rel << ")");
      const auto big = largestComponent(componentsOf<double>(weights));
      bool sameKeep = static_cast<int>(big.size()) == keepTable.rows();
      for (std::size_t i = 0; i < big.size() && sameKeep; ++i) sameKeep = big[i] + 1 == static_cast<int>(keepTable(static_cast<Eigen::Index>(i), 0));
      EXPECT(sameKeep, "hops " << hops << " prune " << prune << ": the largest component's vertices identical to the reference's (" << big.size() << ")");
    }
  }
  std::cout << (g_failures ? "FAILED\n" : "all connectome golden tests passed\n");
  return g_failures ? 1 : 0;
}
