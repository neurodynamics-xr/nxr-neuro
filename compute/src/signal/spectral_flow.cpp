// nxr::signal — the flow of a weighted cross-spectrum in a surface's field coordinates (see the header).
#include "nxr/signal/spectral_flow.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }
}  // namespace

Eigen::MatrixXcd overlappedSpectrum(const Eigen::MatrixXd& servingOperator, const Eigen::MatrixXd& samples, std::int64_t left, std::int64_t right, std::int64_t overlap,
                                    std::int64_t transformSamples, std::int64_t firstBin, std::int64_t binCount) {
  const std::int64_t width = samples.cols(), own = width - left - right;
  if (servingOperator.cols() != samples.rows()) bad("overlappedSpectrum: the serving operator reads " + std::to_string(servingOperator.cols()) + " channels, the samples have " + std::to_string(samples.rows()));
  if (left < 0 || right < 0 || own < 1) bad("overlappedSpectrum: the chunk has no samples of its own");
  if (overlap < 0 || transformSamples < overlap + own) bad("overlappedSpectrum: the transform (" + std::to_string(transformSamples) + " samples) does not hold the overlap and the chunk");
  const std::int64_t bins = transformSamples / 2 + 1;
  if (firstBin < 0 || binCount < 1 || firstBin + binCount > bins) bad("overlappedSpectrum: bins [" + std::to_string(firstBin) + ", " + std::to_string(firstBin + binCount) + ") outside the transform's " + std::to_string(bins));
  // compose first: the serving operator on the samples present (K rows, at most the channels), then the reflection and ONE transform of K rows
  const Eigen::MatrixXd coordinates = servingOperator * samples;                          // [K x width]
  const std::int64_t takeLeft = std::min(left, overlap), takeRight = std::min(right, transformSamples - overlap - own);
  const std::int64_t reflectLeft = overlap - takeLeft, reflectRight = transformSamples - overlap - own - takeRight;
  const std::int64_t start = left - takeLeft, stop = left + own + takeRight;             // the present samples taken: [start, stop) of the block
  const std::int64_t present = stop - start;
  if (present < 2 && (reflectLeft > 0 || reflectRight > 0)) bad("overlappedSpectrum: a reflection needs at least two samples, the chunk and its overlap hold " + std::to_string(present));
  if (reflectLeft > 0 && start != 0) bad("overlappedSpectrum: the left side is reflected only where the samples end (the recording's start)");
  if (reflectRight > 0 && stop != width) bad("overlappedSpectrum: the right side is reflected only where the samples end (the recording's end)");
  // numpy `reflect` (the edge sample not repeated); a side longer than the samples present is reflected again, as numpy pads repeatedly: the even extension of period
  // 2 (present - 1) (a short last chunk, e.g. a recording that ends a few seconds into its last chunk)
  const std::int64_t period = 2 * (present - 1);
  auto mirrored = [&](std::int64_t position) {                                            // position relative to `start`, any integer -> a present sample
    if (period == 0) return std::int64_t(0);
    std::int64_t m = position % period;
    if (m < 0) m += period;
    return m < present ? m : period - m;
  };
  const Eigen::Index K = coordinates.rows();
  std::vector<double> buffer(static_cast<std::size_t>(K * transformSamples));
  for (Eigen::Index r = 0; r < K; ++r) {
    double* row = buffer.data() + r * transformSamples;
    for (std::int64_t j = 0; j < transformSamples; ++j) row[j] = coordinates(r, start + mirrored(j - reflectLeft));
  }
  std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(K * bins));
  r2c(buffer.data(), spectrum.data(), {static_cast<std::size_t>(K), static_cast<std::size_t>(transformSamples)}, 1);
  Eigen::MatrixXcd out(K, binCount);
  for (Eigen::Index r = 0; r < K; ++r)
    for (std::int64_t k = 0; k < binCount; ++k) out(r, k) = spectrum[static_cast<std::size_t>(r * bins + firstBin + k)];
  return out;
}

Eigen::MatrixXd normaliseBySumOfSquaredGains(const Eigen::MatrixXd& gains, double floor) {
  const Eigen::RowVectorXd sum = gains.cwiseAbs2().colwise().sum();
  const double top = sum.size() ? sum.maxCoeff() : 0.0;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(gains.rows(), gains.cols());
  for (Eigen::Index k = 0; k < gains.cols(); ++k)
    if (top > 0 && sum(k) >= floor * top) out.col(k) = gains.col(k) / std::sqrt(sum(k));
  return out;
}

Eigen::MatrixXd cosineSquaredPartition(const Eigen::VectorXd& x, const std::vector<double>& centres) {
  const std::size_t n = centres.size();
  if (n < 2) bad("cosineSquaredPartition: at least two centres");
  for (std::size_t b = 1; b < n; ++b) if (!(centres[b] > centres[b - 1])) bad("cosineSquaredPartition: the centres are not ascending");
  const double halfPi = std::acos(0.0);
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const double u = x(i);
    if (u <= centres.front()) { g(0, i) = 1.0; continue; }
    if (u >= centres.back()) { g(static_cast<Eigen::Index>(n - 1), i) = 1.0; continue; }
    std::size_t b = 0;
    while (!(u >= centres[b] && u < centres[b + 1])) ++b;                                  // u between centres b and b + 1: member b falls, member b + 1 rises
    const double t = (u - centres[b]) / (centres[b + 1] - centres[b]);
    const double fall = std::cos(halfPi * t), rise = std::sin(halfPi * t);
    g(static_cast<Eigen::Index>(b), i) = fall * fall;
    g(static_cast<Eigen::Index>(b + 1), i) = rise * rise;
  }
  return g;
}

}  // namespace nxr::signal
