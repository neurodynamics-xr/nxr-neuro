// nxr::meeg tile network (nxr/meeg/tile_network.h) on small hand-made cases (no dataset):
//   fiberLengths: a straight fibre and a right angle;
//   tileCodesAtLevel: the finest codes of a two-piece grid shifted to coarser levels (piece = code >> level);
//   tilePairs: counts, numpy medians (odd and even counts), the gate, ends outside and in one tile, centroids, distances, across the pieces;
//   graphHodgeDirac: a path of four tiles (eigenvalues 2 - 2 cos(k pi / 4)), the constants of a network in two components, the cycle space's size (E - V + components),
//     the mixed forms orthonormal under diag(⋆0, ⋆1) on a weighted random network with node weights;
//   networkFlow: the potential against a direct (pseudo-inverse) solve of Bᵀ⋆1 B phi = Bᵀ⋆1 F; the coefficients synthesise the flow back (exact + cycle parts); a pure gradient
//     flow has gradient share 1 and its potential back up to a constant; a pure circulation is all cycle: no divergence, no gradient part.
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "check.h"
#include "nxr/meeg/tile_network.h"

using namespace nxr::meeg;

int main() {
  // ── lengths ──
  {
    const std::vector<double> points = {0, 0, 0, 1, 0, 0, 3, 0, 0,     0, 0, 0, 0, 2, 0, 2, 2, 0};
    const auto L = fiberLengths(points, 3);
    CHECK(L.size() == 2 && std::abs(L[0] - 3) < 1e-12 && std::abs(L[1] - 4) < 1e-12);
    const std::vector<float> single(points.begin(), points.end());
    CHECK(std::abs(fiberLengths(single, 3)[1] - 4) < 1e-6);
  }
  // ── codes at a level: two pieces, depth 2 (codes 4 piece + tile) → level 1 (2 piece + tile >> 1) ──
  {
    const std::vector<std::int64_t> finest = {0, 1, 2, 3, 4, 5, 6, 7, -1};
    const auto c1 = tileCodesAtLevel(finest, 2, 1);
    CHECK((c1 == std::vector<std::int64_t>{0, 0, 1, 1, 2, 2, 3, 3, -1}));
    CHECK((tileCodesAtLevel(finest, 2, 0) == std::vector<std::int64_t>{0, 0, 0, 0, 1, 1, 1, 1, -1}));
  }
  // ── pairs: four tiles (codes 0 1 on piece 0, 2 3 on piece 1 at level 1), one vertex per tile and one off ──
  {
    const std::vector<std::int64_t> codes = {0, 1, 2, 3, -1, 0};
    Eigen::MatrixXd X(6, 3);
    X << 0, 0, 0,  1, 0, 0,  0, 1, 0,  1, 1, 0,  9, 9, 9,  0, 0, 2;          // tile 0 holds vertices 0 and 5: centroid (0, 0, 1)
    // fibres (block order: first ends, then last ends): 0-1, 0-1, 0-1, 1-0, 2-3 (gated), 0-4 (off), 1-1 (one tile)
    const std::vector<std::int32_t> ends = {0, 5, 0, 1, 2, 0, 1,    1, 1, 1, 0, 3, 4, 1};
    const std::vector<std::uint8_t> keep = {1, 1, 1, 1, 0, 1, 1};
    const std::vector<double> lengths = {10, 20, 30, 40, 50, 60, 70};
    const TileNetwork n = tilePairs(codes, X, 1, ends, keep, lengths);
    CHECK(n.codes.size() == 4 && n.count.size() == 6);
    CHECK((n.pieces == std::vector<std::int64_t>{0, 0, 1, 1}));
    CHECK(n.count(pairIndex(0, 1, 4)) == 4 && n.count(pairIndex(2, 3, 4)) == 0);
    CHECK(std::abs(n.medianLength(pairIndex(0, 1, 4)) - 25) < 1e-12);     // numpy median of 10 20 30 40
    CHECK(std::isnan(n.medianLength(pairIndex(2, 3, 4))));
    CHECK(n.gatedOut == 1 && n.endsOutside == 1 && n.sameTile == 1);
    CHECK(std::abs(n.centroids(0, 2) - 1) < 1e-12 && std::abs(n.distance(pairIndex(0, 1, 4)) - std::sqrt(2.0)) < 1e-12);
    CHECK(n.across(pairIndex(0, 1, 4)) == 0 && n.across(pairIndex(1, 2, 4)) == 1);
    const TileNetwork odd = tilePairs(codes, X, 1, std::vector<std::int32_t>{0, 0, 0, 1, 1, 1}, {}, std::vector<double>{5, 1, 3});
    CHECK(odd.count(0) == 3 && std::abs(odd.medianLength(0) - 3) < 1e-12);
  }
  // ── the graph Hodge–Dirac modes of a path 0-1-2-3 (unit weights) and of two components ──
  {
    const std::vector<std::int32_t> path = {0, 1, 1, 2, 2, 3};
    const GraphModes m = graphHodgeDirac(path, 4, Eigen::VectorXd::Ones(4), Eigen::VectorXd::Ones(3), 1e-10);
    CHECK(m.components == 1 && m.cycles == 0 && m.mu.size() == 3);
    for (int k = 1; k < 4; ++k) CHECK(std::abs(m.mu(k - 1) * m.mu(k - 1) - (2 - 2 * std::cos(k * M_PI / 4))) < 1e-12);
    CHECK((m.forms.transpose() * m.forms - Eigen::MatrixXd::Identity(3, 3)).norm() < 1e-12);
    const GraphModes two = graphHodgeDirac(std::vector<std::int32_t>{0, 1, 2, 3}, 4, Eigen::VectorXd::Ones(4), Eigen::Vector2d(3, 2), 1e-10);
    CHECK(two.components == 2 && two.cycles == 0 && two.mu.size() == 2);
    bool refusedWeight = false;
    try { graphHodgeDirac(path, 4, Eigen::VectorXd::Ones(4), Eigen::Vector3d(1, 0, 1), 1e-10); } catch (const std::invalid_argument&) { refusedWeight = true; }
    CHECK(refusedWeight);
  }
  // ── a weighted random network: the modes orthonormal under the stars; the potential against a direct solve; the coefficients synthesise the flow; a gradient; a circulation ──
  {
    const int T = 12;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    std::vector<std::int32_t> edges;
    std::vector<double> wv;
    for (int i = 0; i < T; ++i)
      for (int j = i + 1; j < T; ++j)
        if (j == i + 1 || u(rng) < 0.4) { edges.push_back(i); edges.push_back(j); wv.push_back(0.5 + 20 * u(rng)); }   // connected (the path), then random chords
    const int E = static_cast<int>(wv.size());
    const Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(wv.data(), E);
    Eigen::VectorXd area(T);
    for (int i = 0; i < T; ++i) area(i) = 1e-4 * (1 + u(rng));
    const GraphModes m = graphHodgeDirac(edges, T, area, w, 1e-10);
    CHECK(m.components == 1 && m.cycles == E - T + 1 && m.mu.size() == E);                // (T - 1) exact + (E - T + 1) cycles
    Eigen::VectorXd stars(T + E);
    stars << area, w;
    CHECK((m.forms.transpose() * stars.asDiagonal() * m.forms - Eigen::MatrixXd::Identity(E, E)).norm() < 1e-9);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(E, T);
    for (int e = 0; e < E; ++e) { B(e, edges[static_cast<std::size_t>(2 * e)]) = -1; B(e, edges[static_cast<std::size_t>(2 * e + 1)]) = 1; }
    Eigen::VectorXd F(E);
    for (int e = 0; e < E; ++e) F(e) = u(rng) - 0.5;
    const NetworkFlow nf = networkFlow(F, edges, w, m);
    // the direct solve: Bᵀ⋆1 B phi = Bᵀ⋆1 F, the constant fixed by ⋆0 (Σ ⋆0 phi = 0, as the modes are ⋆0-orthogonal to the constants)
    const Eigen::MatrixXd L = B.transpose() * w.asDiagonal() * B;
    Eigen::VectorXd direct = L.completeOrthogonalDecomposition().solve(B.transpose() * w.cwiseProduct(F));
    direct.array() -= area.dot(direct) / area.sum();
    CHECK((nf.potential - direct).norm() < 1e-9 * std::max(1.0, direct.norm()));
    CHECK(std::abs(area.dot(nf.potential)) < 1e-12);
    CHECK((m.forms.bottomRows(E) * nf.coefficients - F).norm() < 1e-9);                  // complete: the exact and cycle parts synthesise the flow back
    CHECK(std::abs(nf.weightedGradientSquares + nf.weightedCycleSquares - nf.weightedFlowSquares) < 1e-9 * nf.weightedFlowSquares);
    CHECK((nf.divergence - B.transpose() * F).norm() < 1e-12);
    // a pure gradient: F = B phi0
    Eigen::VectorXd phi0(T);
    for (int i = 0; i < T; ++i) phi0(i) = std::sin(1.3 * i) + 0.2 * i;
    const NetworkFlow g = networkFlow(B * phi0, edges, w, m);
    CHECK(std::abs(g.weightedGradientSquares / g.weightedFlowSquares - 1) < 1e-10 && g.weightedCycleSquares < 1e-18 * g.weightedFlowSquares);
    CHECK((g.potential - (phi0.array() - area.dot(phi0) / area.sum()).matrix()).norm() < 1e-9);
    // a circulation around the triangle 0 → 1 → 2 → 0 with equal weights: no divergence, no gradient part, all cycle
    const std::vector<std::int32_t> tri = {0, 1, 0, 2, 1, 2};
    const GraphModes tm = graphHodgeDirac(tri, 3, Eigen::VectorXd::Ones(3), Eigen::VectorXd::Ones(3), 1e-10);
    CHECK(tm.cycles == 1 && tm.kind[0] == 1 && tm.mu(0) == 0);
    const NetworkFlow c = networkFlow(Eigen::Vector3d(1, -1, 1), tri, Eigen::VectorXd::Ones(3), tm);
    CHECK(c.divergence.norm() < 1e-12 && c.weightedGradientSquares < 1e-20 && std::abs(c.weightedCycleSquares - 3) < 1e-12 && std::abs(c.weightedFlowSquares - 3) < 1e-12);
  }
  return nxrtest::finish("test_tile_network");
}
