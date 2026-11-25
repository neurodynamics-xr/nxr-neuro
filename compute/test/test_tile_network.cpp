// nxr::meeg tile network (nxr/meeg/tile_network.h) on small hand-made cases (no dataset):
//   fiberLengths: a straight fibre and a right angle;
//   tileCodesAtLevel: the finest codes of a two-piece grid shifted to coarser levels (piece = code >> level);
//   tilePairs: counts, numpy medians (odd and even counts), the gate, ends outside and in one tile, centroids, distances, across the pieces;
//   laplacianEigenmodes: a path of four tiles (eigenvalues 2 - 2 cos(k pi / 4)), the null modes of a network in two components;
//   networkFlow: the potential against a direct (pseudo-inverse) solve of L phi = Bᵀ W F on a weighted random network; a pure gradient flow has gradient share 1 and its
//     potential back up to a constant; a pure circulation has share 0 and divergence 0.
#include <cmath>
#include <cstdio>
#include <random>
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
  // ── the Laplacian of a path 0-1-2-3 (unit weights) and of two components ──
  {
    Eigen::VectorXd w = Eigen::VectorXd::Zero(6);
    w(pairIndex(0, 1, 4)) = 1; w(pairIndex(1, 2, 4)) = 1; w(pairIndex(2, 3, 4)) = 1;
    const LaplacianModes m = laplacianEigenmodes(w, 4, 1e-10);
    for (int k = 0; k < 4; ++k) CHECK(std::abs(m.eigenvalues(k) - (2 - 2 * std::cos(k * M_PI / 4))) < 1e-12);
    CHECK(m.nullModes == 1);
    CHECK((m.eigenvectors.transpose() * m.eigenvectors - Eigen::MatrixXd::Identity(4, 4)).norm() < 1e-12);
    Eigen::VectorXd two = Eigen::VectorXd::Zero(6);
    two(pairIndex(0, 1, 4)) = 3; two(pairIndex(2, 3, 4)) = 2;
    CHECK(laplacianEigenmodes(two, 4, 1e-10).nullModes == 2);
  }
  // ── the potential: the eigenmode filter against a direct solve; a gradient flow; a circulation ──
  {
    const int T = 12, P = T * (T - 1) / 2;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0, 1);
    Eigen::VectorXd w = Eigen::VectorXd::Zero(P), F(P);
    for (int q = 0; q < P; ++q) { if (u(rng) < 0.4) w(q) = std::floor(1 + 20 * u(rng)); F(q) = u(rng) - 0.5; }
    for (int i = 0; i + 1 < T; ++i) if (w(pairIndex(i, i + 1, T)) == 0) w(pairIndex(i, i + 1, T)) = 1;   // connected
    const LaplacianModes m = laplacianEigenmodes(w, T, 1e-10);
    CHECK(m.nullModes == 1);
    const NetworkFlow nf = networkFlow(F, w, m);
    // the direct solve: L phi = Bᵀ W F, the minimum-norm solution (complete orthogonal decomposition)
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(T);
    for (int i = 0; i < T; ++i) for (int j = i + 1; j < T; ++j) { const int q = static_cast<int>(pairIndex(i, j, T)); if (w(q) > 0) { rhs(i) -= w(q) * F(q); rhs(j) += w(q) * F(q); } }
    const Eigen::VectorXd direct = m.laplacian.completeOrthogonalDecomposition().solve(rhs);
    CHECK((nf.potential - direct).norm() < 1e-10 * std::max(1.0, direct.norm()));
    CHECK(std::abs(nf.potential.sum()) < 1e-10);
    CHECK((m.eigenvectors.transpose() * nf.potential - nf.coefficients).norm() < 1e-10);
    // a pure gradient: F = phi0_j - phi0_i on the edges
    Eigen::VectorXd phi0(T);
    for (int i = 0; i < T; ++i) phi0(i) = std::sin(1.3 * i) + 0.2 * i;
    Eigen::VectorXd G(P);
    for (int i = 0; i < T; ++i) for (int j = i + 1; j < T; ++j) G(pairIndex(i, j, T)) = phi0(j) - phi0(i);
    const NetworkFlow g = networkFlow(G, w, m);
    CHECK(std::abs(g.weightedGradientSquares / g.weightedFlowSquares - 1) < 1e-10);
    CHECK((g.potential - (phi0.array() - phi0.mean()).matrix()).norm() < 1e-9);
    // a circulation around the triangle 0 → 1 → 2 → 0 with equal weights: no divergence, no gradient part
    Eigen::VectorXd tw = Eigen::VectorXd::Zero(3), C(3);
    tw.setOnes();
    C(pairIndex(0, 1, 3)) = 1; C(pairIndex(1, 2, 3)) = 1; C(pairIndex(0, 2, 3)) = -1;
    const NetworkFlow c = networkFlow(C, tw, laplacianEigenmodes(tw, 3, 1e-10));
    CHECK(c.divergence.norm() < 1e-12 && c.weightedGradientSquares < 1e-20 && std::abs(c.weightedFlowSquares - 3) < 1e-12);
  }
  return nxrtest::finish("test_tile_network");
}
