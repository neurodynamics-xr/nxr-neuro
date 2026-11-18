// nxr::synthesis — the lag-free surrogates (synthesis/lagfree.h).
#include "nxr/synthesis/lagfree.h"

#include "nxr/core/algebra.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nxr::synthesis {

namespace {
std::uint64_t splitmix(std::uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
double uniform01(std::uint64_t bits) { return (static_cast<double>(bits >> 11) + 0.5) * (1.0 / 9007199254740992.0); }   // (0, 1)
}  // namespace

double standardNormalAt(std::uint64_t seed, std::uint64_t stream, std::uint64_t index) {
  const std::uint64_t h = splitmix(splitmix(splitmix(seed) ^ (stream + 0x632be59bd9b4e019ULL)) ^ index);
  const double u1 = uniform01(h), u2 = uniform01(splitmix(h ^ 0xa0761d6478bd642fULL));
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

Eigen::MatrixXd whiteNoiseBlock(std::uint64_t seed, int rows, std::int64_t first, std::int64_t n) {
  if (rows < 1 || n < 0) throw std::invalid_argument("whiteNoiseBlock: rows >= 1 and n >= 0");
  Eigen::MatrixXd w(rows, n);
  for (int r = 0; r < rows; ++r)
    for (std::int64_t j = 0; j < n; ++j) w(r, j) = standardNormalAt(seed, static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(first + j));   // a negative sample wraps: still one fixed value
  return w;
}

Eigen::VectorXd spectrumShapingTaps(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, double fs, int half) {
  if (freq.size() < 2 || freq.size() != psd.size()) throw std::invalid_argument("spectrumShapingTaps: freq and psd are of one length >= 2");
  if (!(fs > 0) || half < 1) throw std::invalid_argument("spectrumShapingTaps: fs > 0 and half >= 1");
  for (Eigen::Index i = 1; i < freq.size(); ++i) if (!(freq[i] > freq[i - 1])) throw std::invalid_argument("spectrumShapingTaps: freq increases");
  if (!(psd.maxCoeff() > 0)) throw std::invalid_argument("spectrumShapingTaps: the spectrum holds no power");
  // the amplitude on a fine grid up to Nyquist (linear in the power between the stored bins, zero outside them), and the taps as its cosine transform
  const double nyq = fs / 2, df = std::min(freq[1] - freq[0], nyq / 512.0) / 2;
  const std::int64_t nf = static_cast<std::int64_t>(std::floor(nyq / df)) + 1;
  Eigen::VectorXd a(nf);
  Eigen::Index j = 0;
  for (std::int64_t i = 0; i < nf; ++i) {
    const double f = static_cast<double>(i) * df;
    if (f < freq[0] || f > freq[freq.size() - 1]) { a[i] = 0; continue; }
    while (j + 2 < freq.size() && freq[j + 1] < f) ++j;
    const double t = (f - freq[j]) / (freq[j + 1] - freq[j]);
    a[i] = std::sqrt(std::max((1 - t) * psd[j] + t * psd[j + 1], 0.0));
  }
  Eigen::VectorXd h(2 * half + 1);
  for (int k = 0; k <= half; ++k) {
    double s = 0;
    for (std::int64_t i = 0; i < nf; ++i) s += a[i] * std::cos(6.283185307179586 * static_cast<double>(i) * df * k / fs) * (i == 0 || i == nf - 1 ? 0.5 : 1.0);
    const double w = 0.5 + 0.5 * std::cos(3.141592653589793 * k / (half + 1));
    h[half + k] = h[half - k] = s * w;
  }
  const double e = std::sqrt(h.squaredNorm());
  if (!(e > 0)) throw std::invalid_argument("spectrumShapingTaps: the taps carry no energy");
  return h / e;
}

Eigen::MatrixXd convolveValid(const Eigen::MatrixXd& white, const Eigen::VectorXd& taps) {
  const Eigen::Index L = taps.size();
  if (L % 2 == 0 || white.cols() < L) throw std::invalid_argument("convolveValid: odd taps and at least as many columns");
  const Eigen::Index n = white.cols() - L + 1;
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(white.rows(), n);
  for (Eigen::Index k = 0; k < L; ++k) y += taps[k] * white.middleCols(k, n);
  return y;
}

Eigen::MatrixXd zeroLagMixing(const Eigen::MatrixXd& realCrossSpectrum) {
  if (realCrossSpectrum.rows() != realCrossSpectrum.cols() || realCrossSpectrum.rows() == 0) throw std::invalid_argument("zeroLagMixing: a square cross-spectrum of at least one channel");
  const double meanVariance = realCrossSpectrum.trace() / static_cast<double>(realCrossSpectrum.rows());
  if (!(meanVariance > 0)) throw std::invalid_argument("zeroLagMixing: the cross-spectrum holds no power");
  return nxr::core::symmetricSquareRoot(realCrossSpectrum / meanVariance);
}

Eigen::MatrixXd shapedNoiseBlock(const Eigen::VectorXd& taps, std::uint64_t seed, int rows, std::int64_t first, std::int64_t n, const Eigen::VectorXd& scale) {
  if (scale.size() != 0 && scale.size() != n) throw std::invalid_argument("shapedNoiseBlock: scale has one value a sample");
  if (taps.size() % 2 == 0) throw std::invalid_argument("shapedNoiseBlock: odd taps");
  const std::int64_t half = (taps.size() - 1) / 2;
  const Eigen::MatrixXd w = whiteNoiseBlock(seed, rows, first - half, n + 2 * half);
  Eigen::MatrixXd x = convolveValid(w, taps);
  if (scale.size() != 0) x = x * scale.asDiagonal();
  return x;
}

Eigen::MatrixXd lagfreeBlock(const Eigen::MatrixXd& mixing, const Eigen::VectorXd& taps, std::uint64_t seed, std::int64_t first, std::int64_t n, const Eigen::VectorXd& scale) {
  if (mixing.rows() != mixing.cols()) throw std::invalid_argument("lagfreeBlock: the mixing is square");
  if (scale.size() != 0 && scale.size() != n) throw std::invalid_argument("lagfreeBlock: scale has one value a sample");
  if (taps.size() % 2 == 0) throw std::invalid_argument("lagfreeBlock: odd taps");
  return mixing * shapedNoiseBlock(taps, seed, static_cast<int>(mixing.rows()), first, n, scale);
}

double analyticGainPower(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, const Eigen::VectorXd& gain) {
  if (freq.size() != psd.size() || gain.size() != psd.size() || psd.size() < 2) throw std::invalid_argument("analyticGainPower: freq, psd and gain of one length >= 2");
  const Eigen::Index nf = psd.size();
  double num = 0, den = 0;
  for (Eigen::Index k = 0; k < nf; ++k) { const double w = (k == 0 || k == nf - 1) ? 0.5 : 1.0; num += w * psd[k] * gain[k] * gain[k]; den += w * psd[k]; }
  if (!(num > 0 && den > 0)) throw std::invalid_argument("analyticGainPower: the spectrum holds no power in the band");
  return 2.0 * num / den;
}

Eigen::VectorXd envelopeScale(const Eigen::VectorXd& power, double expected) {
  if (!(expected > 0)) throw std::invalid_argument("envelopeScale: the expected power is above 0");
  return (power.cwiseMax(0.0) / expected).cwiseSqrt();
}

Eigen::VectorXd interpolateTileSeries(const std::vector<double>& v, std::int64_t stride, std::int64_t first, std::int64_t n) {
  if (v.empty() || stride < 1) throw std::invalid_argument("interpolateTileSeries: values and a stride >= 1");
  Eigen::VectorXd y(n);
  const double last = static_cast<double>(v.size() - 1);
  for (std::int64_t j = 0; j < n; ++j) {
    const double p = std::min(std::max(static_cast<double>(first + j) / static_cast<double>(stride), 0.0), last);
    const std::size_t i = static_cast<std::size_t>(std::floor(p)), i1 = std::min(i + 1, v.size() - 1);
    const double t = p - static_cast<double>(i);
    y[j] = (1 - t) * v[i] + t * v[i1];
  }
  return y;
}

double effectiveBandwidth(const Eigen::VectorXd& freq, const Eigen::VectorXd& psd, const Eigen::VectorXd& gain) {
  if (freq.size() < 2 || freq.size() != psd.size() || (gain.size() != 0 && gain.size() != psd.size())) throw std::invalid_argument("effectiveBandwidth: freq, psd (and gain) of one length >= 2");
  double s1 = 0, s2 = 0;
  for (Eigen::Index i = 0; i < psd.size(); ++i) { const double p = psd[i] * (gain.size() ? gain[i] * gain[i] : 1.0); s1 += p; s2 += p * p; }
  if (!(s2 > 0)) throw std::invalid_argument("effectiveBandwidth: the spectrum holds no power");
  const double df = (freq[freq.size() - 1] - freq[0]) / static_cast<double>(freq.size() - 1);
  return s1 * s1 * df / s2;
}

Eigen::MatrixXcd lagfreeSecondMomentDraw(const Eigen::MatrixXd& ReQ0, double dof, std::uint64_t seed, std::uint64_t draw) {
  if (ReQ0.rows() != ReQ0.cols()) throw std::invalid_argument("lagfreeSecondMomentDraw: a square matrix");
  const int k = std::max(static_cast<int>(std::lround(dof)), 2), C = static_cast<int>(ReQ0.rows());
  const Eigen::MatrixXd L = nxr::core::symmetricSquareRoot(ReQ0);
  Eigen::MatrixXcd Z(C, k);
  const std::uint64_t s = seed ^ (0x517cc1b727220a95ULL * (draw + 1));
  for (int r = 0; r < C; ++r)
    for (int j = 0; j < k; ++j)
      Z(r, j) = std::complex<double>(standardNormalAt(s, 2 * static_cast<std::uint64_t>(r), static_cast<std::uint64_t>(j)), standardNormalAt(s, 2 * static_cast<std::uint64_t>(r) + 1, static_cast<std::uint64_t>(j))) / std::sqrt(2.0);
  const Eigen::MatrixXcd X = L.cast<std::complex<double>>() * Z;
  return X * X.adjoint() / static_cast<double>(k);
}

double laggedShare(const Eigen::MatrixXcd& Q) {
  const double t = Q.squaredNorm();
  return t > 0 ? Q.imag().squaredNorm() / t : std::numeric_limits<double>::quiet_NaN();
}

}  // namespace nxr::synthesis
