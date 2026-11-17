#include "nxr/band_amplitude.h"
#include "nxr/fft.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr {

Eigen::MatrixXd bandAmplitude(const Eigen::MatrixXcd& spectrum, const std::vector<BinRange>& ranges) {
  const std::int64_t bins = spectrum.cols();
  for (const auto& r : ranges)
    if (r.first < 0 || r.second <= r.first || r.second > bins)
      throw std::invalid_argument("bandAmplitude: the bin range [" + std::to_string(r.first) + ", " + std::to_string(r.second) + ") is empty or outside the " + std::to_string(bins) + " bins");
  const Eigen::MatrixXd magnitude = spectrum.cwiseAbs();
  Eigen::MatrixXd out(spectrum.rows(), static_cast<Eigen::Index>(ranges.size()));
  for (std::size_t b = 0; b < ranges.size(); ++b)
    out.col(static_cast<Eigen::Index>(b)) = magnitude.middleCols(ranges[b].first, ranges[b].second - ranges[b].first).rowwise().mean();
  return out;
}

BinRange bandBins(double lo_hz, double hi_hz, std::int64_t tile_samples, double sfreq) {
  if (tile_samples < 1 || !(sfreq > 0) || !(hi_hz > lo_hz) || lo_hz < 0) throw std::invalid_argument("bandBins: a tile of >= 1 samples, sfreq > 0 and a band 0 <= lo < hi");
  const double df = sfreq / static_cast<double>(tile_samples);
  const std::int64_t bins = tile_samples / 2 + 1;
  const std::int64_t first = static_cast<std::int64_t>(std::ceil(lo_hz / df - 1e-9));
  const std::int64_t last = std::min<std::int64_t>(static_cast<std::int64_t>(std::ceil(hi_hz / df - 1e-9)), bins);
  if (last <= first || first >= bins)
    throw std::invalid_argument("bandBins: the band [" + std::to_string(lo_hz) + ", " + std::to_string(hi_hz) + ") Hz holds no bin at " + std::to_string(df) + " Hz spacing");
  return {first, last};
}

Eigen::MatrixXd bandAmplitudeOverTiles(const Eigen::MatrixXd& x, std::int64_t tile_samples, double sfreq, double lo_hz, double hi_hz) {
  if (tile_samples < 1 || x.cols() < tile_samples) throw std::invalid_argument("bandAmplitudeOverTiles: the block holds fewer samples than one tile");
  const BinRange range = bandBins(lo_hz, hi_hz, tile_samples, sfreq);
  const std::int64_t tiles = x.cols() / tile_samples;
  Eigen::MatrixXd out(x.rows(), tiles);
  for (std::int64_t t = 0; t < tiles; ++t) {
    const Eigen::MatrixXd tile = x.middleCols(t * tile_samples, tile_samples);
    const Eigen::MatrixXcd X = fft::rfftColumns(tile.transpose()).transpose();   // rows x bins
    out.col(t) = bandAmplitude(X, {range});
  }
  return out;
}

}  // namespace nxr
