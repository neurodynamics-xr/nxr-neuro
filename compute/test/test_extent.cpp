// nxr::select on synthetic cells only: the cell code round-trips for every (tile, time tile) of several layouts (a power of two and not), refuses a frame beyond the layout and a gap code; the joint graph's
// edge count; standardize against a per-tile baseline (a tile with no spread left out); windowRegions on a path of tiles: an episode that moves one tile per time tile is ONE event (connected through the
// time links), a static one elsewhere another, min cells and min area drop them, a NaN cell breaks one; the event measures (nxr::measurement::reduce over an event's cells: cells, the union area, the weighted mean, the peak),
// the area per time tile, the centres per time tile and the path length against direct.
#include <cmath>
#include <iostream>

#include "nxr/select/extent.h"
#include "nxr/measurement/reduce.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::select;

template <class F> bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
  std::cout << "test_extent\n";
  {   // THE CELL CODE: the atlas's layout, round trip over every cell of several layouts
    bool ok = true;
    for (const std::int64_t F : {1, 2, 5, 8, 600, 1024}) {
      const int B = bitsFor(F);
      for (std::int64_t t = 0; t < 40 && ok; ++t)
        for (std::int64_t f = 0; f < F; f += (F > 64 ? 7 : 1)) {
          const auto c = cellCode(t, f, B, F);
          const auto back = cellOf(c, B, F);
          ok = ok && back.tile == t && back.frame == f && c == ((t << B) | f);
        }
    }
    EXPECT(ok, "cellOf(cellCode(tile, frame)) = (tile, frame) for frames 1, 2, 5, 8, 600, 1024");
    EXPECT(bitsFor(1) == 1 && bitsFor(2) == 1 && bitsFor(3) == 2 && bitsFor(5) == 3 && bitsFor(8) == 3 && bitsFor(9) == 4 && bitsFor(600) == 10 && bitsFor(1024) == 10 && bitsFor(1025) == 11, "bitsFor is max(1, ceil(log2(max(2, F))))");
    EXPECT(throws([] { cellCode(0, 5, 3, 5); }) && throws([] { cellCode(-1, 0, 3, 5); }) && throws([] { cellCode(0, -1, 3, 5); }), "a time tile beyond the layout, a negative tile or time tile are refused");
    EXPECT(throws([] { cellOf((3 << 3) | 6, 3, 5); }) && throws([] { cellOf(-1, 3, 5); }), "a gap code (time tile 6 of 5) and a negative code are refused");
    EXPECT(throws([] { cellCode(std::int64_t(1) << 30, 1, 2, 4); }), "a code beyond 31 bits is refused");
  }
  // A PATH OF 6 TILES (0-1-2-3-4-5), 7 time tiles
  const int S = 6, F = 7;
  Pairs path;
  for (int s = 0; s + 1 < S; ++s) path.emplace_back(s, s + 1);
  {
    const auto pr = spaceTimePairs(path, S, F);
    EXPECT(static_cast<int>(pr.size()) == F * 5 + (F - 1) * S, "the joint graph has pairs × frames + tiles × (frames − 1) edges");
    EXPECT(throws([&] { spaceTimePairs(Pairs{{0, 9}}, S, F); }), "a tile pair outside the level is refused");
  }
  Eigen::VectorXd area = Eigen::VectorXd::Constant(S, 2.0);
  area(2) = 3.0;
  // baseline 1 everywhere (median 1, SD 0.2). EPISODE A moves one tile per time tile, held together by the cells beside it: (1,1) (1,2) (2,2) (2,3) (3,3) and (4,3) (z = 5, 2.5, 5, 2.5, 5, 2:
  // the last above the exit level only). EPISODE B static at tile 5 in time tiles 5, 6 (z 2.5). A lone cell (0,6) of z 1.25 reaches the exit level, never the entry level: no event.
  Eigen::MatrixXd amp = Eigen::MatrixXd::Constant(S, F, 1.0);
  amp(1, 1) = 2.0; amp(1, 2) = 1.5; amp(2, 2) = 2.0; amp(2, 3) = 1.5; amp(3, 3) = 2.0; amp(4, 3) = 1.4;
  amp(5, 5) = 1.5; amp(5, 6) = 1.5; amp(0, 6) = 1.25;
  const Eigen::VectorXd med = Eigen::VectorXd::Ones(S);
  const Eigen::VectorXd sd = Eigen::VectorXd::Constant(S, 0.2);
  int skipped = -1;
  const Eigen::MatrixXd z = standardize(amp, med, sd, &skipped);
  EXPECT(std::abs(z(1, 1) - 5.0) < 1e-12 && std::abs(z(4, 3) - 2.0) < 1e-12 && z(0, 0) == 0.0 && skipped == 0, "standardize: (x - median) / SD per spatial tile");
  {
    Eigen::VectorXd sd0 = sd; sd0(2) = 0.0;
    int sk = 0;
    const Eigen::MatrixXd z0 = standardize(amp, med, sd0, &sk);
    EXPECT(sk == 1 && std::isnan(z0(2, 2)) && std::isnan(z0(2, 0)) && !std::isnan(z0(1, 1)), "a tile with no spread is NaN throughout and counted");
  }
  const auto W = windowRegions(z, path, area, 1.5, 1.0);
  auto node = [&](int s, int f) { return static_cast<std::size_t>(f * S + s); };
  EXPECT(W.regions.size() == 2, "two events: the moving episode (connected through space and the time links) and the static one; the lone cell is none");
  if (W.regions.size() == 2) {
    EXPECT(W.regions[0].nodes.size() == 6, "the moving episode holds its six cells");
    bool ok = true;
    for (const auto& c : std::vector<std::pair<int, int>>{{1, 1}, {1, 2}, {2, 2}, {2, 3}, {3, 3}, {4, 3}}) ok = ok && W.label[node(c.first, c.second)] == 0;
    EXPECT(ok, "the labels of the moving episode's cells are event 0");
    EXPECT(W.label[node(5, 5)] == 1 && W.label[node(5, 6)] == 1 && W.label[node(0, 6)] == -1, "the static episode is event 1 (numbered by its lowest cell); the lone cell is in none");
    EXPECT(std::abs(W.regions[0].weight - (2.0 + 2.0 + 3.0 + 3.0 + 2.0 + 2.0)) < 1e-12, "an event's weight is the sum of its cells' tile areas");
  }
  {   // two cells joined only DIAGONALLY in space-time ((1,1) and (2,2)) are two events
    Eigen::MatrixXd a2 = Eigen::MatrixXd::Constant(S, F, 1.0);
    a2(1, 1) = 2.0; a2(2, 2) = 2.0;
    EXPECT(windowRegions(standardize(a2, med, sd), path, area, 1.5, 1.0).regions.size() == 2, "cells diagonal in space-time are not neighbours");
  }
  {   // min cells, min area, a NaN cell
    EXPECT(windowRegions(z, path, area, 1.5, 1.0, 3).regions.size() == 1, "min_cells 3 drops the static event of 2 cells");
    const auto Wa = windowRegions(z, path, area, 1.5, 1.0, 1, 5.0);
    EXPECT(Wa.regions.size() == 1 && Wa.regions[0].nodes.size() == 6, "min_area 5 (the sum of the tile areas) drops the static event of area 4");
    Eigen::MatrixXd a3 = amp;
    a3(5, 5) = std::numeric_limits<double>::quiet_NaN();
    const auto W3 = windowRegions(standardize(a3, med, sd), path, area, 1.5, 1.0);
    EXPECT(W3.regions.size() == 2 && W3.regions[1].nodes.size() == 1, "a NaN cell (a bad time tile) is in no event and breaks the static event to its other cell");
    EXPECT(throws([&] { windowRegions(z, path, area, 1.0, 1.5); }) && throws([&] { windowRegions(z, path, area, 1.5, 1.0, 0); }), "an exit level above the entry level and a minimum of no cell are refused");
  }
  // THE ROLL-UPS and the path, on a labelling of the bridged episode and the static one
  const Eigen::MatrixXd& a2 = amp;
  const auto W2 = windowRegions(standardize(a2, med, sd), path, area, 1.5, 1.0);
  Labels lab = Labels::Constant(S, F, -1);
  for (std::size_t n = 0; n < W2.label.size(); ++n) lab(static_cast<Eigen::Index>(n % S), static_cast<Eigen::Index>(n / S)) = W2.label[n];
  const Eigen::VectorXd samples = Eigen::VectorXd::Constant(F, 100.0);
  Eigen::MatrixXd pk = a2.array() + 0.5;
  {   // THE EVENT MEASURES: reductions over the event's cells (the labelling is the partition): the cells' node numbers are the elements, the events the members
    namespace ms = nxr::measurement;
    const int cellsN = S * F;
    nxr::selection::Elements byEvent;
    byEvent.size = cellsN;
    for (int e = 0; e < 2; ++e) {
      for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) if (lab(s, f) == e) byEvent.elements.push_back(static_cast<std::int64_t>(f) * S + s);
      byEvent.offsets.push_back(static_cast<std::int64_t>(byEvent.elements.size()));
    }
    std::vector<double> amp(static_cast<std::size_t>(cellsN)), peak(static_cast<std::size_t>(cellsN)), wgt(static_cast<std::size_t>(cellsN));
    for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) { const std::size_t n = static_cast<std::size_t>(f) * S + s; amp[n] = a2(s, f); peak[n] = pk(s, f); wgt[n] = area(s) * samples(f); }
    ms::Ingredients means(1, 2), peaks(1, 2);
    ms::Request weighted; weighted.measures = {"mean"}; weighted.weighting = ms::Weighting::given;
    ms::Request plain; plain.measures = {"max"};
    ms::reduce(means, amp.data(), 1, cellsN, byEvent, wgt.data(), weighted);
    ms::reduce(peaks, peak.data(), 1, cellsN, byEvent, nullptr, plain);
    double wsum = 0, asum = 0, amax = -1;
    for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) if (lab(s, f) == 0) { wsum += area(s) * 100.0; asum += area(s) * 100.0 * a2(s, f); amax = std::max(amax, pk(s, f)); }
    EXPECT(means.count[0] == 6 && means.count[1] == 2, "event 0: 6 cells; event 1 (static): 2 cells");
    EXPECT(std::abs(ms::value(means, 0, 0, "mean") - asum / wsum) < 1e-12 && std::abs(means.weight[0] - wsum) < 1e-9 && std::abs(ms::value(peaks, 0, 0, "max") - amax) < 1e-12, "the amplitude is the area- and sample-weighted mean of the cells; the peak the largest");
    // the union area: the distinct spatial tiles of an event, each counted once (the member's own measure `area` over the event's tiles)
    nxr::selection::Elements tilesOf;
    tilesOf.size = S;
    for (int e = 0; e < 2; ++e) {
      for (int s = 0; s < S; ++s) { bool any = false; for (int f = 0; f < F; ++f) any = any || lab(s, f) == e; if (any) tilesOf.elements.push_back(s); }
      tilesOf.offsets.push_back(static_cast<std::int64_t>(tilesOf.elements.size()));
    }
    ms::Ingredients uni(1, 2);
    ms::Request geo; geo.measures = {"area"}; geo.weighting = ms::Weighting::given;
    ms::reduce(uni, static_cast<const double*>(nullptr), 1, S, tilesOf, area.data(), geo);
    EXPECT(uni.count[0] == 4 && std::abs(ms::value(uni, 0, 0, "area") - (2.0 + 3.0 + 2.0 + 2.0)) < 1e-12, "event 0: the union area is the area of its 4 distinct tiles (tile 1 counted once)");
    EXPECT(uni.count[1] == 1 && std::abs(ms::value(uni, 0, 1, "area") - 2.0) < 1e-12, "event 1: one tile, the area of one tile");
    // the area per time tile: the cells of (event, time tile) as members, the tile areas as weights
    nxr::selection::Elements byFrame;
    byFrame.size = cellsN;
    for (int e = 0; e < 2; ++e) for (int f = 0; f < F; ++f) {
      for (int s = 0; s < S; ++s) if (lab(s, f) == e) byFrame.elements.push_back(static_cast<std::int64_t>(f) * S + s);
      byFrame.offsets.push_back(static_cast<std::int64_t>(byFrame.elements.size()));
    }
    std::vector<double> ta(static_cast<std::size_t>(cellsN));
    for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) ta[static_cast<std::size_t>(f) * S + s] = area(s);
    ms::Ingredients ab(1, 2 * F);
    ms::reduce(ab, static_cast<const double*>(nullptr), 1, cellsN, byFrame, ta.data(), geo);
    EXPECT(std::abs(ms::value(ab, 0, 0 * F + 2, "area") - 5.0) < 1e-12 && std::abs(ms::value(ab, 0, 0 * F + 1, "area") - 2.0) < 1e-12 && ab.count[0 * F + 0] == 0 && ab.count[0 * F + 4] == 0 && std::abs(ms::value(ab, 0, 1 * F + 5, "area") - 2.0) < 1e-12,
           "the area per time tile sums its cells' tile areas; a time tile with no cell of the event holds none");
  }
  // a line mesh: vertex v of tile v at x = 10 v (the unit is whatever the positions carry); one vertex per tile of area 1
  std::vector<std::int32_t> tileOfVertex(S);
  std::vector<double> pos(static_cast<std::size_t>(S) * 3, 0.0), varea(S, 1.0);
  for (int v = 0; v < S; ++v) { tileOfVertex[static_cast<std::size_t>(v)] = v; pos[static_cast<std::size_t>(v) * 3] = 10.0 * v; }
  const auto C = centres(lab, 2, tileOfVertex, pos, varea);
  // frame 1: tile 1 -> vertex 1; frame 2: tiles {1,2} -> centroid 15 -> vertex 1 or 2 (tie: the lowest wins in tileCentres' strict <: vertex 1); frame 3: tiles {2,3,4} -> centroid 30 -> vertex 3
  EXPECT(C(0, 1) == 1 && C(0, 3) == 3 && C(0, 0) == -1 && C(1, 5) == 5 && C(1, 6) == 5, "the centre per time tile is the vertex nearest the centroid of its tiles' vertices");
  const Eigen::VectorXd P = pathLength(C, pos);
  const double direct = std::abs(10.0 * C(0, 2) - 10.0 * C(0, 1)) + std::abs(10.0 * C(0, 3) - 10.0 * C(0, 2));
  EXPECT(std::abs(P(0) - direct) < 1e-12 && std::abs(P(0) - 20.0) < 1e-12 && P(1) == 0.0, "the path length is the chord sum of the consecutive centres (20), and 0 for a static event");
  Eigen::VectorXd none = pathLength(Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic>::Constant(1, 3, -1), pos);
  EXPECT(std::isnan(none(0)), "an event with no centre has no path");
  std::cout << (failures ? "FAILED\n" : "ALL PASSED\n");
  return failures ? 1 : 0;
}
