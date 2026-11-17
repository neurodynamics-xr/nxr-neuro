#include "nxr/band_amplitude.h"

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

}  // namespace nxr
