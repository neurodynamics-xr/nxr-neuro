// Shared by the tests that need the cross-spectrum of a block over time tiles, or the counts of a distribution per time tile: the outer form of `nxr::measurement::reduce` over a rule (the tiles start at the block's first
// sample), with a tile past the data a zero matrix (what the old per-tile function returned).
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "nxr/histogram.h"
#include "nxr/measurement/estimate.h"
#include "nxr/measurement/reduce.h"

struct CrossSpectrumTiles {
  std::vector<Eigen::MatrixXcd> sum;      // [tiles], each [channels x channels]
  std::vector<std::int64_t> count;        // [tiles] samples summed (0 for a tile past the data)
};

inline CrossSpectrumTiles crossSpectrumTiles(const Eigen::MatrixXcd& Z, std::size_t width, std::size_t tiles) {
  nxr::selection::Rule rule;
  rule.origin = 0;
  rule.width = rule.hop = static_cast<std::int64_t>(width);
  rule.count = static_cast<std::int64_t>(tiles);
  rule.size = Z.cols();
  nxr::measurement::MatrixIngredients acc(rule.count);
  nxr::measurement::reduce(acc, Z, 0, rule);
  CrossSpectrumTiles out;
  out.count = acc.count;
  for (std::size_t t = 0; t < tiles; ++t) out.sum.push_back(acc.count[t] > 0 ? acc.sum[t] : Eigen::MatrixXcd::Zero(Z.rows(), Z.rows()));
  return out;
}

// the counts per tile of `width` samples of a [rows x samples] block (row-major float32), pooled over the rows: one `measurement::Histogram` a tile, folded with the same bins
inline nxr::histogram::Counts histogramTiles(const float* values, std::size_t rows, std::size_t samples, std::size_t width, std::size_t tiles, const nxr::histogram::LogBins& b) {
  nxr::histogram::Counts c;
  c.tiles = static_cast<int>(tiles);
  c.bins = b.bins;
  c.counts.assign(tiles * static_cast<std::size_t>(b.bins), 0);
  c.negative.assign(tiles, 0);
  c.nonfinite.assign(tiles, 0);
  for (std::size_t t = 0; t < tiles; ++t) {
    nxr::measurement::Histogram h = nxr::measurement::Histogram::log_bins(b);
    const std::size_t from = t * width, to = std::min(from + width, samples);
    if (to > from) for (std::size_t r = 0; r < rows; ++r) h.add(values + r * samples + from, static_cast<std::int64_t>(to - from));
    for (int k = 0; k < b.bins; ++k) c.counts[t * static_cast<std::size_t>(b.bins) + static_cast<std::size_t>(k)] = h.counts[static_cast<std::size_t>(k)];
    c.negative[t] = h.negative;
    c.nonfinite[t] = h.nonfinite;
  }
  return c;
}
