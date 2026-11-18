// Counting over a partition: equal to a DIRECT SUM over the partition of the vertex fiber counts (and, smoothed, of the vertex connection weights, formed here in the test only),
// for random partitions, a partition by hemisphere pair and a coarse one; at 0, 1 and 3 hops to 1e-12; independent of the order of the fibers; the within-region count.
#include <cmath>
#include <iostream>
#include <numeric>

#include "connectome_helpers.h"
#include "nxr/meeg/connectome.h"
#include "nxr/geometry/surface_maps.h"

using namespace nxr::meeg;
using Dense = Eigen::MatrixXd;

static Dense directSum(const Dense& vertexMatrix, const std::vector<std::int32_t>& label, int nRegions) {
  Dense out = Dense::Zero(nRegions, nRegions);
  const int n = static_cast<int>(vertexMatrix.rows());
  for (int u = 0; u < n; ++u) for (int v = 0; v < n; ++v)
    if (label[static_cast<std::size_t>(u)] >= 0 && label[static_cast<std::size_t>(v)] >= 0) out(label[static_cast<std::size_t>(u)], label[static_cast<std::size_t>(v)]) += vertexMatrix(u, v);
  return out;
}

int main() {
  std::vector<double> vertexPositions; std::vector<int32_t> triangles;
  icosphere(2, vertexPositions, triangles, 0.08);
  const int nV = static_cast<int>(vertexPositions.size() / 3);
  ctest::Lcg g(11);
  const auto b = ctest::makeBundle(g, 300, 5, 0.08, 10, 10);
  const auto ends = fiberEnds<double>(b.points, b.nP);
  const auto near = nxr::geometry::nearest::nearestVertices<double>(vertexPositions, ends);
  const auto ref = ctest::referenceEnds(vertexPositions, b);
  const Dense full = ctest::referenceCounts(ref, nV);   // with the diagonal: a fiber on one vertex adds 2 there

  struct Part { std::string name; int nRegions; std::vector<std::int32_t> label; };
  std::vector<Part> parts;
  {
    Part p{"random, 9 regions", 9, {}};
    for (int v = 0; v < nV; ++v) p.label.push_back(static_cast<std::int32_t>(g.next() * 9));
    parts.push_back(p);
    Part q{"random, 12 regions, some vertices in none", 12, {}};
    for (int v = 0; v < nV; ++v) q.label.push_back(g.next() < 0.15 ? -1 : static_cast<std::int32_t>(g.next() * 12));
    parts.push_back(q);
    Part h{"a hemisphere pair (z < 0, z >= 0)", 2, {}};
    for (int v = 0; v < nV; ++v) h.label.push_back(vertexPositions[static_cast<std::size_t>(3 * v + 2)] < 0 ? 0 : 1);
    parts.push_back(h);
    Part c{"one region", 1, std::vector<std::int32_t>(static_cast<std::size_t>(nV), 0)};
    parts.push_back(c);
    Part vtx{"vertices (the finest partition)", nV, {}};
    for (int v = 0; v < nV; ++v) vtx.label.push_back(v);
    parts.push_back(vtx);
  }
  for (const auto& part : parts)
    for (int hops : {0, 1, 3}) {
      const auto step = smoothingStep<double>(triangles, nV);
      const auto smoothedMemberships = smoothMemberships<double>(step, memberships<double>(part.label, part.nRegions), hops);
      const auto counts = partitionCounts<double>(near.vertex, keepAll(b.nF), smoothedMemberships);
      const Dense smoothing = ctest::referenceSmoother(triangles, nV, hops);
      const Dense want = directSum(smoothing * full * smoothing.transpose(), part.label, part.nRegions);
      const double err = (Dense(counts) - want).cwiseAbs().maxCoeff();
      EXPECT(err < 1e-12 * std::max(1.0, want.cwiseAbs().maxCoeff()), part.name << ", " << hops << " hops: equals the direct sum over the partition (" << err << ")");
      EXPECT((Dense(counts) - Dense(counts).transpose()).cwiseAbs().maxCoeff() < 1e-12, part.name << ", " << hops << " hops: symmetric");
      if (hops == 0) {   // plain fiber counts: total = 2 x fibers with both ends in a region
        double inRegions = 0;
        for (std::size_t f = 0; f < ref.size(); ++f) inRegions += (part.label[static_cast<std::size_t>(ref[f][0])] >= 0 && part.label[static_cast<std::size_t>(ref[f][1])] >= 0) ? 2 : 0;
        EXPECT(std::abs(Dense(counts).sum() - inRegions) < 1e-9, part.name << ": the total is 2 x the fibers with both ends in a region");
      }
    }

  std::cout << "=== a vertex partition at 0 hops is the vertex matrix; within-region counts; fiber order does not matter ===\n";
  {
    const auto oneHot = memberships<double>(parts[4].label, nV);
    EXPECT(Dense(partitionCounts<double>(near.vertex, {}, oneHot)) == full, "the vertex partition's counts are the vertex fiber counts (diagonal: self-pairs add 2)");
    const auto& two = parts[2];
    const auto c2 = Dense(partitionCounts<double>(near.vertex, {}, memberships<double>(two.label, 2)));
    double within0 = 0;
    for (const auto& e : ref) within0 += (two.label[static_cast<std::size_t>(e[0])] == 0 && two.label[static_cast<std::size_t>(e[1])] == 0) ? 2 : 0;
    EXPECT(c2(0, 0) == within0, "the within-region count is stored on the diagonal (a within-region fiber adds 2, both orders)");
    // order of the fibers: reverse them (block order: reverse each half)
    const std::size_t nF = near.vertex.size() / 2;
    std::vector<std::int32_t> rev(near.vertex.size());
    for (std::size_t f = 0; f < nF; ++f) { rev[f] = near.vertex[nF - 1 - f]; rev[nF + f] = near.vertex[2 * nF - 1 - f]; }
    const auto smoothedThreeHops = smoothMemberships<double>(smoothingStep<double>(triangles, nV), memberships<double>(parts[0].label, parts[0].nRegions), 3);
    EXPECT((Dense(partitionCounts<double>(rev, {}, smoothedThreeHops)) - Dense(partitionCounts<double>(near.vertex, {}, smoothedThreeHops))).cwiseAbs().maxCoeff() < 1e-12, "the result does not depend on the order of the fibers");
    // rolling up: counting over the union of regions equals the sum of the regions' counts (the pyramid's roll-up)
    const auto fine = Dense(partitionCounts<double>(near.vertex, {}, memberships<double>(parts[0].label, parts[0].nRegions)));
    std::vector<std::int32_t> coarse(parts[0].label.size());
    for (std::size_t v = 0; v < coarse.size(); ++v) coarse[v] = parts[0].label[v] / 3;
    const auto direct = Dense(partitionCounts<double>(near.vertex, {}, memberships<double>(coarse, 3)));
    Dense rolled = Dense::Zero(3, 3);
    for (int a = 0; a < 9; ++a) for (int c = 0; c < 9; ++c) rolled(a / 3, c / 3) += fine(a, c);
    EXPECT((rolled - direct).cwiseAbs().maxCoeff() < 1e-12, "rolling regions up equals counting over the coarser partition directly (0 hops)");
  }
  std::cout << (g_failures ? "FAILED\n" : "all partition tests passed\n");
  return g_failures ? 1 : 0;
}
