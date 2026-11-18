#pragma once
// nxr::stats::histogram — THE DISTRIBUTION of a non-negative series as a measurement over a time partition (spec docs/specs/2026-11-13-single-events.md, the
// runner decisions of 2026-11-13: a global statistic is a stored node, its distribution, and percentiles are read off it). A port of nxr's
// `_core_threshold` (nxr/detect/vortex.py; the Studio's port: `curlCoreThreshold`, studio/src/engine/toolbox/measure/curlCores.ts), the reference:
//
//   bins      `bins` (4096) bins: bin 0 is [0, lo); bin b ≥ 1 is [lo·r^(b−1), lo·r^b), r = (hi/lo)^(1/(bins−1)); hi = 4 × the FIRST chunk's maximum,
//             lo = hi · 1e-6 (`logBinsFor`). Values above hi are clipped to hi (they land in the top bin), as the reference does.
//   the index b = min(floor(ln(a / lo) · sc) + 1, bins − 1), sc = (bins − 1) / ln(hi / lo), with the reference's precision under NumPy 2 (a / lo and
//             the log in float32, × sc and the floor in float64), so a value lands in the reference's bin.
//   fold      counts per TILE (`measurement::Histogram`, estimate.h: ADDITIVE, so a tile read chunk by chunk is the sum of its parts, and any run of tiles is the sum of theirs); the reference
//             pools |curl| over every vertex.
//   read      percentile(p): the reference's `np.geomspace(lo, hi, bins)[searchsorted(cdf, p / 100)]`, the UPPER edge of the first bin whose cumulative
//             share reaches p (exact to one bin: a relative width of ln(10⁶)/4095 ≈ 0.34 %); median = percentile(50); mad = the median of |x − median|
//             with each bin's mass at its upper edge (the same edges), exact to one bin.
// Pure: counts in, numbers out; the edges are the caller's (fixed once, from the first chunk, so every chunk shares them). A negative or non-finite value
// is not a value of a non-negative series: it is COUNTED (`negative`, `nonfinite`) and left out, never silently binned.
#include <cstdint>
#include <vector>

namespace nxr::stats::histogram {

constexpr int kBins = 4096;
constexpr double kHiFactor = 4.0, kLoRatio = 1e-6;

struct LogBins {
  double lo = 0, hi = 0;
  int bins = kBins;
};
/** the reference's bins from the first chunk's maximum: hi = factor × max, lo = hi × ratio (std::invalid_argument when max is not > 0 or bins < 2) */
LogBins logBinsFor(double firstChunkMax, int bins = kBins, double factor = kHiFactor, double ratio = kLoRatio);
/** the bin of a value (≥ 0, finite) */
int binOf(const LogBins& b, float value);
/** the upper edge of bin k: lo · r^k (bin 0's is lo) — `np.geomspace(lo, hi, bins)[k]` */
double upperEdge(const LogBins& b, int k);

/** the counts per tile (one `measurement::Histogram` a tile, estimate.h, stacked): the distribution is counted by `nxr::measurement::Histogram::add`, not here */
struct Counts {
  std::vector<std::int64_t> counts;     // [tiles x bins], row-major
  std::vector<std::int64_t> negative, nonfinite;   // [tiles] values left out
  int tiles = 0, bins = 0;
};
/** the sum of tiles' counts (a run of tiles: their distribution) [bins] */
std::vector<std::int64_t> sumTiles(const Counts& c, int first, int last);
/** the p-th percentile (0 < p ≤ 100) of a distribution [bins] (std::invalid_argument when it is empty) */
double percentile(const std::vector<std::int64_t>& counts, const LogBins& b, double p);
double median(const std::vector<std::int64_t>& counts, const LogBins& b);
/** the median absolute deviation from the median (not scaled: the robust SD is 1.4826 × it) */
double mad(const std::vector<std::int64_t>& counts, const LogBins& b);

}  // namespace nxr::stats::histogram
