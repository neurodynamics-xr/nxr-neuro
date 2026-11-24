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
using RowMajorComplex = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
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

Eigen::MatrixXd gaussianAtomGains(const Eigen::VectorXd& frequencies, const std::vector<double>& centres, const std::vector<double>& standardDeviations, double gain) {
  if (centres.size() != standardDeviations.size()) bad("gaussianAtomGains: " + std::to_string(centres.size()) + " centres, " + std::to_string(standardDeviations.size()) + " standard deviations");
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(centres.size()), frequencies.size());
  for (std::size_t a = 0; a < centres.size(); ++a) {
    const double sd = standardDeviations[a];
    if (!(sd > 0)) bad("gaussianAtomGains: a standard deviation is not positive");
    for (Eigen::Index k = 0; k < frequencies.size(); ++k) {
      const double f = frequencies(k);
      if (f > 0) g(static_cast<Eigen::Index>(a), k) = gain * std::exp(-(f - centres[a]) * (f - centres[a]) / (2.0 * sd * sd));
    }
  }
  return g;
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

Eigen::MatrixXd mexicanHatPartition(const Eigen::VectorXd& mu, const std::vector<double>& centres) {
  const std::size_t n = centres.size();
  if (n < 2) bad("mexicanHatPartition: at least two centres");
  for (std::size_t b = 0; b < n; ++b) if (!(centres[b] > 0) || (b > 0 && !(centres[b] > centres[b - 1]))) bad("mexicanHatPartition: the centres are positive and ascending");
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), mu.size());
  std::vector<double> logGain(n);
  for (Eigen::Index i = 0; i < mu.size(); ++i) {
    if (mu(i) < 0) bad("mexicanHatPartition: a wavenumber is below 0");
    if (!(mu(i) > 0)) { g(0, i) = 1.0; continue; }                                         // the constant: every hat is 0 there; it belongs to the first member
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t b = 0; b < n; ++b) {                                                   // log of (lambda / c) e^(1 - lambda / c), lambda = mu², c = centre²
      const double x = (mu(i) * mu(i)) / (centres[b] * centres[b]);
      logGain[b] = std::log(x) + 1.0 - x;
      top = std::max(top, logGain[b]);
    }
    double sum = 0;
    for (std::size_t b = 0; b < n; ++b) sum += std::exp(logGain[b] - top);
    for (std::size_t b = 0; b < n; ++b) g(static_cast<Eigen::Index>(b), i) = std::exp(logGain[b] - top) / sum;
  }
  return g;
}

BandPairFlow bandPairFlow(const Eigen::MatrixXcd& C, const Eigen::MatrixXd& Psi, const Eigen::MatrixXd& G, const std::vector<std::int32_t>& edges, const Eigen::MatrixXd& H) {
  const Eigen::Index K = C.rows(), S = Psi.rows(), Q = H.rows();
  const Eigen::Index E = static_cast<Eigen::Index>(edges.size() / 2);
  if (C.cols() != K || Psi.cols() != K || G.cols() != K || G.rows() != 3 * S || H.cols() != K || edges.size() % 2 != 0)
    bad("bandPairFlow: a cross-spectrum [K x K], a synthesis [S x K], gradients [3S x K] and band gains [bands x K] of one K");
  for (const auto v : edges) if (v < 0 || v >= S) bad("bandPairFlow: an edge's vertex is outside the synthesis' vertices");
  BandPairFlow out;
  out.power = Eigen::MatrixXd::Zero(S, Q * Q);
  out.edgeCurrent = Eigen::MatrixXd::Zero(E, Q * Q);
  out.vertexCurrent = Eigen::MatrixXd::Zero(3 * S, Q * Q);
  const Eigen::MatrixXcd Cbar = C.conjugate();
  Eigen::MatrixXd synthesisAndGradients(S + 3 * S, K);
  synthesisAndGradients << Psi, G;
  for (Eigen::Index q = 0; q < Q; ++q) {
    // R_q = conj(C) diag(h_q) [Psiᵀ | Gᵀ]: [K x 4S], once per band
    const Eigen::MatrixXcd right = Cbar * (H.row(q).transpose().asDiagonal() * synthesisAndGradients.transpose()).cast<std::complex<double>>();
    for (Eigen::Index p = 0; p < Q; ++p) {
      const Eigen::Index pair = p * Q + q;
      const Eigen::MatrixXd left = Psi * H.row(p).transpose().asDiagonal();               // [S x K] Psi diag(h_p)
      for (Eigen::Index v = 0; v < S; ++v) {
        out.power(v, pair) = (left.row(v).cast<std::complex<double>>() * right.col(v)).value().real();
        for (int comp = 0; comp < 3; ++comp) out.vertexCurrent(3 * v + comp, pair) = (left.row(v).cast<std::complex<double>>() * right.col(S + 3 * v + comp)).value().imag();
      }
      for (Eigen::Index e = 0; e < E; ++e) {
        const std::int32_t tail = edges[static_cast<std::size_t>(2 * e)], tip = edges[static_cast<std::size_t>(2 * e + 1)];
        out.edgeCurrent(e, pair) = (left.row(tail).cast<std::complex<double>>() * right.col(tip)).value().imag();
      }
    }
  }
  return out;
}

CellFlow timeCellFlow(const Eigen::MatrixXcd& X, std::int64_t firstBin, std::int64_t N, std::int64_t ownFirst, std::int64_t ownSamples, std::int64_t cellSamples,
                      const Eigen::MatrixXd& Psi, const Eigen::MatrixXd& G, const std::vector<std::int32_t>& edges, const Eigen::VectorXd& h) {
  const Eigen::Index K = X.rows(), S = Psi.rows(), B = X.cols();
  const Eigen::Index E = static_cast<Eigen::Index>(edges.size() / 2);
  if (Psi.cols() != K || G.cols() != K || G.rows() != 3 * S || h.size() != K) bad("timeCellFlow: a spectrum [K x bins], a synthesis [S x K], gradients [3S x K] and a band gain [K] of one K");
  if (firstBin < 0 || firstBin + B > N) bad("timeCellFlow: the bins fall outside the transform");
  if (ownFirst < 0 || ownSamples < 1 || ownFirst + ownSamples > N) bad("timeCellFlow: the chunk's own samples fall outside the transform");
  if (cellSamples < 1) bad("timeCellFlow: a cell holds at least one sample");
  const std::int64_t cells = ownSamples / cellSamples;
  CellFlow out;
  out.edgeCurrent = Eigen::MatrixXd::Zero(E, cells);
  out.vertexCurrent = Eigen::MatrixXd::Zero(3 * S, cells);
  out.power = Eigen::MatrixXd::Zero(S, cells);
  if (cells == 0) return out;
  const std::int64_t used = cells * cellSamples;
  // the band signal back in time, rows in blocks: compose first (the synthesis and the band gain on the atom's bins only), then the inverse transform of N, the own samples kept
  auto inTime = [&](const Eigen::MatrixXd& rows) {                                         // rows [R x K] → [R x used] complex
    const Eigen::MatrixXcd spectral = (rows * h.asDiagonal()).cast<std::complex<double>>() * X;   // [R x B]
    const Eigen::Index R = spectral.rows();
    std::vector<std::complex<double>> buffer(static_cast<std::size_t>(R * N), std::complex<double>(0, 0));
    for (Eigen::Index r = 0; r < R; ++r) for (Eigen::Index k = 0; k < B; ++k) buffer[static_cast<std::size_t>(r * N + firstBin + k)] = spectral(r, k);
    c2c(buffer.data(), buffer.data(), {static_cast<std::size_t>(R), static_cast<std::size_t>(N)}, 1, false);
    RowMajorComplex z(R, used);
    for (Eigen::Index r = 0; r < R; ++r) for (std::int64_t t = 0; t < used; ++t) z(r, t) = buffer[static_cast<std::size_t>(r * N + ownFirst + t)];
    return z;
  };
  const RowMajorComplex z = inTime(Psi);                                                    // [S x used]
  for (Eigen::Index v = 0; v < S; ++v)
    for (std::int64_t c = 0; c < cells; ++c) {
      double sum = 0;
      for (std::int64_t t = c * cellSamples; t < (c + 1) * cellSamples; ++t) sum += std::norm(z(v, t));
      out.power(v, c) = sum;
    }
  for (Eigen::Index e = 0; e < E; ++e) {
    const std::int32_t tail = edges[static_cast<std::size_t>(2 * e)], tip = edges[static_cast<std::size_t>(2 * e + 1)];
    for (std::int64_t c = 0; c < cells; ++c) {
      double sum = 0;
      for (std::int64_t t = c * cellSamples; t < (c + 1) * cellSamples; ++t) sum += (std::conj(z(tail, t)) * z(tip, t)).imag();
      out.edgeCurrent(e, c) = sum;
    }
  }
  const Eigen::Index block = 64;                                                            // the gradients' rows in blocks: at most 64 x used complex at a time
  for (Eigen::Index r0 = 0; r0 < 3 * S; r0 += block) {
    const Eigen::Index rows = std::min(block, 3 * S - r0);
    const RowMajorComplex gz = inTime(G.middleRows(r0, rows));
    for (Eigen::Index r = 0; r < rows; ++r) {
      const Eigen::Index v = (r0 + r) / 3;
      for (std::int64_t c = 0; c < cells; ++c) {
        double sum = 0;
        for (std::int64_t t = c * cellSamples; t < (c + 1) * cellSamples; ++t) sum += (std::conj(z(v, t)) * gz(r, t)).imag();
        out.vertexCurrent(r0 + r, c) = sum;
      }
    }
  }
  return out;
}

}  // namespace nxr::signal
