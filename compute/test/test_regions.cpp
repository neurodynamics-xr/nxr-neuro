// nxr::regions on synthetic graphs only: the line graph reproduces nxr::intervals::crossings + keepSpans (hysteresis, the time rule), a small planar graph (a
// grid) gives connected regions with their weights and peaks, a NaN node breaks a region, the size limits and the entry level drop regions, regions that do not
// touch are never merged, the sign is the caller's (applySign), refusals.
#include <cmath>
#include <iostream>
#include <random>

#include "nxr/intervals.h"
#include "nxr/regions.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::regions;
using Pairs = std::vector<std::pair<std::int32_t, std::int32_t>>;

int main() {
  std::cout << "test_regions\n";
  {   // THE LINE GRAPH is the time rule: the same spans as crossings (enter high, extend while above the low level), kept by length (the weights are 1 per sample)
    std::mt19937 rng(7);
    std::normal_distribution<double> nd(0, 1);
    int agree = 0, runs = 40;
    for (int r = 0; r < runs; ++r) {
      const int n = 300;
      Eigen::VectorXd x(n);
      for (int i = 0; i < n; ++i) x(i) = nd(rng);
      Pairs line;
      for (int i = 0; i + 1 < n; ++i) line.emplace_back(i, i + 1);
      const auto spans = nxr::intervals::keepSpans(nxr::intervals::crossings(x, 1.5, 0.5), 3, 40);
      const auto reg = detectRegions(x, line, Eigen::VectorXd::Ones(n), 1.5, 0.5, 3, 40);
      bool same = spans.size() == reg.regions.size();
      for (std::size_t k = 0; same && k < spans.size(); ++k) {
        const auto& g = reg.regions[k];
        same = g.nodes.front() == spans[k].start && g.nodes.back() + 1 == spans[k].stop && static_cast<std::int64_t>(g.nodes.size()) == spans[k].length() && g.peak == spans[k].peak && g.value == spans[k].value;
      }
      agree += same;
    }
    std::printf("  line graph vs crossings + keepSpans: %d of %d random series identical\n", agree, runs);
    EXPECT(agree == runs, "on a line graph detectRegions gives the spans of crossings + keepSpans (start, stop, peak, value, length)");
  }
  // A 4 x 4 GRID, node = 4 r + c, weight 1 (a cell) or 2 (the last row), four-neighbour pairs
  const int R = 4, C = 4, n = R * C;
  Pairs grid;
  for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) { if (c + 1 < C) grid.emplace_back(r * C + c, r * C + c + 1); if (r + 1 < R) grid.emplace_back(r * C + c, (r + 1) * C + c); }
  Eigen::VectorXd w = Eigen::VectorXd::Ones(n);
  for (int c = 0; c < C; ++c) w(3 * C + c) = 2;
  Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
  // a region of 4 cells (rows 0-1, columns 0-1) with a peak of 5 at node 5; a lone cell 3 at 2.5; nodes 10 and 15 a diagonal apart (not neighbours)
  v(0) = 1.2; v(1) = 2.0; v(4) = 1.5; v(5) = 5.0;
  v(3) = 2.5;
  v(10) = 3.0; v(15) = 3.0;
  {
    const Regions g = detectRegions(v, grid, w, 2.0, 1.0);
    // the regions by their lowest node: {0,1,4,5}, {3}, {10}, {15}: four regions
    EXPECT(g.regions.size() == 4 && g.regions[0].nodes == std::vector<std::int32_t>({0, 1, 4, 5}) && g.regions[1].nodes == std::vector<std::int32_t>({3}) && g.regions[2].nodes == std::vector<std::int32_t>({10}) && g.regions[3].nodes == std::vector<std::int32_t>({15}),
           "regions numbered by their lowest node; nodes 10 and 15 (a diagonal apart) are two regions, never merged");
    EXPECT(g.regions[0].peak == 5 && g.regions[0].value == 5.0 && g.regions[0].weight == 4.0 && g.regions[3].weight == 2.0, "a region's peak node and value, and its weight (the last row's cells weigh 2)");
    EXPECT(g.label[5] == 0 && g.label[3] == 1 && g.label[10] == 2 && g.label[15] == 3 && g.label[2] == -1, "the label per node, -1 for none");
  }
  {   // hysteresis: the 2 x 2 block is kept at enter 5 (its peak) and the others dropped; at enter 6 nothing is kept; the low level alone does not make a region
    const Regions a = detectRegions(v, grid, w, 5.0, 1.0);
    EXPECT(a.regions.size() == 1 && a.regions[0].nodes.size() == 4, "enter 5: only the region reaching 5 is kept, with its whole extent above the exit level");
    EXPECT(detectRegions(v, grid, w, 6.0, 1.0).regions.empty(), "enter above every value: no region");
    const Regions b = detectRegions(v, grid, w, 1.0, 1.0);          // entry = exit: plain connected regions above one level
    EXPECT(b.regions.size() == 4 && b.regions[0].nodes.size() == 4, "enter = exit: plain connected regions");
    const Regions c = detectRegions(v, grid, w, 2.0, 1.6);          // nodes 0 (1.2) and 4 (1.5) fall below the exit level: the block shrinks to {1, 5}
    EXPECT(c.regions[0].nodes == std::vector<std::int32_t>({1, 5}), "a higher exit level shrinks the region to the nodes above it");
  }
  {   // size limits in the weights' units
    const Regions big = detectRegions(v, grid, w, 2.0, 1.0, 3.0);
    EXPECT(big.regions.size() == 1 && big.regions[0].weight == 4.0, "minWeight 3 keeps only the region of weight 4 (the others weigh 1, 1, 2)");
    const Regions small = detectRegions(v, grid, w, 2.0, 1.0, 0.0, 2.0);
    EXPECT(small.regions.size() == 3 && small.label[5] == -1, "maxWeight 2 drops the region of weight 4; the regions are renumbered");
  }
  {   // a NaN node is no node: it breaks a region in two and is never labelled
    Eigen::VectorXd u = Eigen::VectorXd::Constant(5, 3.0);
    u(2) = std::nan("");
    Pairs line = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    const Regions g = detectRegions(u, line, Eigen::VectorXd::Ones(5), 2.0, 1.0);
    EXPECT(g.regions.size() == 2 && g.label[2] == -1 && g.regions[0].nodes == std::vector<std::int32_t>({0, 1}) && g.regions[1].nodes == std::vector<std::int32_t>({3, 4}), "a NaN node breaks a region in two and belongs to none");
  }
  {   // the sign is the caller's: a dip is found by passing -x
    Eigen::VectorXd x(6);
    x << 0, -3, -4, 0, 2, 0;
    Pairs line = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}};
    const Eigen::VectorXd ones = Eigen::VectorXd::Ones(6);
    EXPECT(detectRegions(applySign(x, Sign::Positive), line, ones, 1.5, 1.0).regions.size() == 1, "positive: the bump");
    const Regions neg = detectRegions(applySign(x, Sign::Negative), line, ones, 1.5, 1.0);
    EXPECT(neg.regions.size() == 1 && neg.regions[0].nodes == std::vector<std::int32_t>({1, 2}) && neg.regions[0].value == 4.0, "negative: the dip, as a bump of -x");
    EXPECT(detectRegions(applySign(x, Sign::Magnitude), line, ones, 1.5, 1.0).regions.size() == 2, "magnitude: both");
    EXPECT(parseSign("negative") == Sign::Negative && std::string(signName(Sign::Magnitude)) == "magnitude", "sign names");
  }
  {   // refusals
    auto refused = [&](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
    const Eigen::VectorXd one = Eigen::VectorXd::Ones(3);
    EXPECT(refused([&] { detectRegions(one, {}, Eigen::VectorXd::Ones(2), 1, 1); }), "weights of another size are refused");
    EXPECT(refused([&] { detectRegions(one, {}, one, 1.0, 2.0); }), "an exit level above the entry level is refused");
    EXPECT(refused([&] { detectRegions(one, {{0, 3}}, one, 1, 1); }), "a pair outside the nodes is refused");
    EXPECT(refused([&] { detectRegions(one, {}, Eigen::VectorXd::Constant(3, -1.0), 1, 1); }), "a negative weight is refused");
    EXPECT(refused([&] { detectRegions(one, {}, one, 1, 1, 5, 2); }), "minWeight above maxWeight is refused");
    EXPECT(refused([&] { parseSign("up"); }), "an unknown sign is refused");
  }
  std::cout << (failures ? "FAILED\n" : "ok\n");
  return failures ? 1 : 0;
}
