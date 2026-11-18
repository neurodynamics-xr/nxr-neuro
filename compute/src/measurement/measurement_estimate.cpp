// nxr::measurement — `estimate` (measurement/estimate.h).
#include "nxr/measurement/estimate.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nxr::measurement {

void SegmentMean::note(std::int64_t segment) {
  if (!seen_.insert(segment).second) throw std::invalid_argument("estimate: segment " + std::to_string(segment) + " was already counted (a segment's value is added once, complete)");
}

void SegmentMean::add(std::int64_t segment, const Eigen::Ref<const Eigen::MatrixXcd>& value) {
  if (used > 0 && (sum_.rows() != value.rows() || sum_.cols() != value.cols()))
    throw std::invalid_argument("estimate: a segment's value is " + std::to_string(value.rows()) + " x " + std::to_string(value.cols()) + ", the first was " + std::to_string(sum_.rows()) + " x " + std::to_string(sum_.cols()));
  note(segment);
  if (used == 0) sum_ = Eigen::MatrixXcd::Zero(value.rows(), value.cols());
  sum_ += value;
  ++used;
}

void SegmentMean::add_real(std::int64_t segment, const Eigen::Ref<const Eigen::MatrixXd>& value) {
  if (used > 0 && (sum_.rows() != value.rows() || sum_.cols() != value.cols()))
    throw std::invalid_argument("estimate: a segment's value is " + std::to_string(value.rows()) + " x " + std::to_string(value.cols()) + ", the first was " + std::to_string(sum_.rows()) + " x " + std::to_string(sum_.cols()));
  note(segment);
  if (used == 0) sum_ = Eigen::MatrixXcd::Zero(value.rows(), value.cols());
  for (Eigen::Index c = 0; c < value.cols(); ++c)
    for (Eigen::Index r = 0; r < value.rows(); ++r) sum_(r, c) += value(r, c);          // the real part, an addition of doubles as a real accumulator's
  ++used;
}

void SegmentMean::skip(std::int64_t segment) {
  note(segment);
  ++skipped;
}

Eigen::MatrixXcd SegmentMean::mean() const {
  if (used == 0) throw std::invalid_argument("estimate: no segment was added (every segment was skipped, or the record held none)");
  return sum_ / std::complex<double>(static_cast<double>(used), 0.0);
}

Eigen::MatrixXd SegmentMean::mean_real() const { return mean().real(); }

Histogram Histogram::log_bins(const histogram::LogBins& bins) {
  Histogram h;
  h.log = bins;
  h.counts.assign(static_cast<std::size_t>(bins.bins), 0);
  h.edges.resize(static_cast<std::size_t>(bins.bins));
  for (int k = 0; k < bins.bins; ++k) h.edges[static_cast<std::size_t>(k)] = histogram::upperEdge(bins, k);
  return h;
}

void Histogram::add(const float* v, std::int64_t n) {
  if (counts.size() != edges.size() || edges.empty()) { if (edges.empty()) throw std::invalid_argument("estimate: a histogram needs its bins' edges"); counts.assign(edges.size(), 0); }
  for (std::int64_t i = 0; i < n; ++i) {
    const float x = v[i];
    if (!std::isfinite(x)) { ++nonfinite; continue; }
    if (log) {
      if (x < 0) { ++negative; continue; }
      ++counts[static_cast<std::size_t>(histogram::binOf(*log, x))];
    } else {
      const auto it = std::upper_bound(edges.begin(), edges.end(), static_cast<double>(x));
      ++counts[static_cast<std::size_t>(it == edges.end() ? edges.size() - 1 : it - edges.begin())];
    }
  }
}

}  // namespace nxr::measurement
