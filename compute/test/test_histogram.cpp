// nxr::stats::histogram and nxr::measurement::Histogram against their reference, nxr's `_core_threshold` (golden: test/golden/histogram_golden.txt, gen_histogram_golden.py; synthetic data): the
// counts bin for bin, the percentiles to rounding; the fold is additive (any split into tiles sums to the same counts); median and MAD within one bin of
// the exact values; negative and non-finite values counted and left out; refusals.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "measurement_helpers.h"
#include "nxr/stats/histogram.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::stats::histogram;

int main() {
  std::cout << "test_histogram\n";
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/histogram_golden.txt");
  EXPECT(in.good(), "the golden is there");
  std::string tag;
  std::size_t R = 0, T = 0, CH = 0;
  in >> tag >> R >> T >> CH;
  std::vector<float> X(R * T);
  for (auto& v : X) in >> v;
  int nb = 0;
  double lo = 0, hi = 0;
  in >> tag >> nb >> lo >> hi;
  std::vector<std::pair<double, double>> pct;
  for (int i = 0; i < 3; ++i) { double p, v; in >> tag >> p >> v; pct.push_back({p, v}); }
  in >> tag;
  std::vector<std::int64_t> want(static_cast<std::size_t>(nb));
  for (auto& c : want) in >> c;
  // THE BINS from the first chunk's maximum, over every row (the reference ravels the chunk)
  float mx = 0;
  for (std::size_t r = 0; r < R; ++r) for (std::size_t t = 0; t < CH; ++t) mx = std::max(mx, X[r * T + t]);
  const LogBins b = logBinsFor(mx);
  EXPECT(b.bins == nb && std::abs(b.lo / lo - 1) < 1e-15 && std::abs(b.hi / hi - 1) < 1e-15, "the bins: hi = 4 x the first chunk's maximum, lo = hi x 1e-6");
  // THE COUNTS per chunk of 3000, summed: the reference's, bin for bin
  const std::size_t tiles = (T + CH - 1) / CH;
  const Counts c = histogramTiles(X.data(), R, T, CH, tiles, b);
  const auto all = sumTiles(c, 0, c.tiles - 1);
  int diff = 0;
  for (int k = 0; k < nb; ++k) diff += all[static_cast<std::size_t>(k)] != want[static_cast<std::size_t>(k)];
  EXPECT(diff == 0, "the counts equal nxr _core_threshold's, bin for bin (" + std::to_string(diff) + " bins differ)");
  EXPECT(all[0] >= 10 && all[static_cast<std::size_t>(nb - 1)] >= 2, "values below lo in bin 0; values above hi clipped into the top bin");
  double worst = 0;
  for (const auto& [p, v] : pct) worst = std::max(worst, std::abs(percentile(all, b, p) / v - 1));
  std::printf("  percentiles 50, 75, 98 vs nxr: %.2e\n", worst);
  EXPECT(worst < 1e-12, "the percentiles equal nxr's (the upper edge of the bin where the share reaches p)");
  // ADDITIVE: other tilings of the same samples (each folded with the same bins) sum to the same counts
  for (std::size_t w : {std::size_t(1), std::size_t(777), std::size_t(4096), T}) {
    const Counts cw = histogramTiles(X.data(), R, T, w, (T + w - 1) / w, b);
    EXPECT(sumTiles(cw, 0, cw.tiles - 1) == all, "tiles of " + std::to_string(w) + " samples sum to the same counts");
  }
  // MEDIAN AND MAD within one bin of the exact values (a relative bin width of ln(1e6) / 4095)
  {
    std::vector<double> v(X.begin(), X.end());
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    const double med = 0.5 * (v[n / 2 - 1] + v[n / 2]);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = std::abs(v[i] - med);
    std::sort(d.begin(), d.end());
    const double madx = 0.5 * (d[n / 2 - 1] + d[n / 2]);
    const double step = std::log(1e6) / (nb - 1);
    const double em = std::abs(std::log(median(all, b) / med)), ea = std::abs(std::log(mad(all, b) / madx));
    std::printf("  median %.5f (exact %.5f); mad %.5f (exact %.5f); one bin = %.4f in log\n", median(all, b), med, mad(all, b), madx, step);
    EXPECT(em <= step && ea <= 2 * step, "median within one bin, MAD within two (the median's bin and its own)");
  }
  // NEGATIVE AND NON-FINITE values: counted, left out
  {
    std::vector<float> y = {1.f, -2.f, NAN, 3.f, INFINITY, 0.5f};
    const Counts cy = histogramTiles(y.data(), 1, y.size(), 3, 2, b);
    const auto s = sumTiles(cy, 0, 1);
    std::int64_t kept = 0;
    for (auto x : s) kept += x;
    EXPECT(kept == 3 && cy.negative[0] == 1 && cy.nonfinite[0] == 1 && cy.nonfinite[1] == 1, "a negative and two non-finite values counted and left out");
  }
  // REFUSALS
  auto refused = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  EXPECT(refused([] { logBinsFor(0); }) && refused([&] { percentile(std::vector<std::int64_t>(static_cast<std::size_t>(nb), 0), b, 50); }) &&
         refused([&] { percentile(all, b, 0); }) && refused([&] { sumTiles(c, 0, c.tiles); }), "refusals: an all-zero first chunk, an empty distribution, p = 0, tiles out of range");
  std::cout << (failures ? "FAIL" : "PASS") << " test_histogram (" << failures << " failure(s))\n";
  return failures ? 1 : 0;
}
