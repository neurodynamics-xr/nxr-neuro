// nxr::stats::histogram — see include/nxr/histogram.h.
#include "nxr/stats/histogram.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace nxr::stats::histogram {

LogBins logBinsFor(double firstChunkMax, int bins, double factor, double ratio) {
  if (!(firstChunkMax > 0) || !std::isfinite(firstChunkMax)) throw std::invalid_argument("histogram: the first chunk's maximum must be finite and > 0 (an all-zero first chunk sets no bins)");
  if (bins < 2) throw std::invalid_argument("histogram: at least 2 bins");
  if (!(factor > 0) || !(ratio > 0 && ratio < 1)) throw std::invalid_argument("histogram: factor > 0 and 0 < ratio < 1");
  LogBins b;
  b.hi = firstChunkMax * factor;
  b.lo = b.hi * ratio;
  b.bins = bins;
  return b;
}

// The reference's arithmetic, as NumPy 2 promotes it (its values are float32): a = min(a, hi) and a / lo in FLOAT32 (a Python float is weak), log in
// float32, then × sc in FLOAT64 (sc is a NumPy float64 scalar, which promotes) and floor; b = 0 below lo, else min(floor(·) + 1, bins − 1).
int binOf(const LogBins& b, float value) {
  const float hi = static_cast<float>(b.hi), lo = static_cast<float>(b.lo);
  const double sc = static_cast<double>(b.bins - 1) / std::log(b.hi / b.lo);
  const float a = std::min(value, hi);
  if (!(a >= lo)) return 0;
  const float l = std::log(a / lo);
  const auto k = static_cast<std::int64_t>(std::floor(static_cast<double>(l) * sc)) + 1;
  return static_cast<int>(std::min<std::int64_t>(k, b.bins - 1));
}

double upperEdge(const LogBins& b, int k) {
  return b.lo * std::pow(b.hi / b.lo, static_cast<double>(k) / static_cast<double>(b.bins - 1));
}

std::vector<std::int64_t> sumTiles(const Counts& c, int first, int last) {
  if (first < 0 || last >= c.tiles || last < first) throw std::invalid_argument("histogram: tiles " + std::to_string(first) + " to " + std::to_string(last) + " of " + std::to_string(c.tiles));
  std::vector<std::int64_t> s(static_cast<std::size_t>(c.bins), 0);
  for (int t = first; t <= last; ++t)
    for (int k = 0; k < c.bins; ++k) s[static_cast<std::size_t>(k)] += c.counts[static_cast<std::size_t>(t) * static_cast<std::size_t>(c.bins) + static_cast<std::size_t>(k)];
  return s;
}

namespace {
// the reference's searchsorted(cumsum(h) / sum(h), q): the first bin whose cumulative share reaches q
int firstReaching(const std::vector<std::int64_t>& counts, double q) {
  const double total = static_cast<double>(std::accumulate(counts.begin(), counts.end(), std::int64_t(0)));
  if (!(total > 0)) throw std::invalid_argument("histogram: an empty distribution has no percentile");
  std::int64_t run = 0;
  for (std::size_t k = 0; k < counts.size(); ++k) {
    run += counts[k];
    if (static_cast<double>(run) / total >= q) return static_cast<int>(k);
  }
  return static_cast<int>(counts.size()) - 1;
}
}  // namespace

double binnedPercentile(const std::vector<std::int64_t>& counts, const LogBins& b, double p) {
  if (!(p > 0 && p <= 100)) throw std::invalid_argument("histogram: a percentile is in (0, 100]");
  if (static_cast<int>(counts.size()) != b.bins) throw std::invalid_argument("histogram: the counts do not have the bins' length");
  return upperEdge(b, firstReaching(counts, p / 100));
}

double binnedMedian(const std::vector<std::int64_t>& counts, const LogBins& b) { return binnedPercentile(counts, b, 50); }

double binnedMad(const std::vector<std::int64_t>& counts, const LogBins& b) {
  const double m = binnedMedian(counts, b);
  std::vector<std::pair<double, std::int64_t>> dev;
  for (int k = 0; k < b.bins; ++k)
    if (counts[static_cast<std::size_t>(k)] > 0) dev.emplace_back(std::abs(upperEdge(b, k) - m), counts[static_cast<std::size_t>(k)]);
  std::sort(dev.begin(), dev.end());
  std::int64_t total = 0;
  for (const auto& d : dev) total += d.second;
  std::int64_t run = 0;
  for (const auto& d : dev) {
    run += d.second;
    if (2 * run >= total) return d.first;
  }
  return dev.back().first;
}

}  // namespace nxr::stats::histogram
