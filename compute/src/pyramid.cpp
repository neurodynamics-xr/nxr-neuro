// nxr::pyramid (pyramid.h).
#include "nxr/pyramid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "nxr/signal/fourier.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/core/parallel.h"

namespace nxr::pyramid {

Shape shapeOf(const std::string& name) {
  if (name == "morlet") return Shape::Morlet;
  if (name == "power_partition") return Shape::PowerPartition;
  throw std::invalid_argument("the band shape is morlet or power_partition, not '" + name + "'");
}
const char* shapeName(Shape s) { return s == Shape::Morlet ? "morlet" : "power_partition"; }

Eigen::MatrixXd bandGains(const Eigen::VectorXd& freqs, const Bands& bands, Shape shape) {
  const Eigen::Index nb = static_cast<Eigen::Index>(bands.size()), nf = freqs.size();
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(nb, nf);
  if (bands.empty()) return g;
  for (const auto& b : bands) if (!(b.second > b.first) || !(b.first > 0)) throw std::invalid_argument("a band needs 0 < lo < hi");
  if (shape == Shape::Morlet) {
    for (Eigen::Index b = 0; b < nb; ++b)
      for (Eigen::Index k = 0; k < nf; ++k) g(b, k) = freqs(k) == 0 ? 0.0 : nxr::signal::graphfilterbank::morletBand(std::abs(freqs(k)), bands[b].first, bands[b].second);
    return g;
  }
  // nxr band_gains: edges in log2 f, a cos/sin crossover half an octave wide centred on each shared edge, the outer edges taper to zero the same way
  std::vector<double> edges{bands[0].first};
  for (const auto& b : bands) edges.push_back(b.second);
  for (Eigen::Index b = 0; b + 1 < nb; ++b)
    if (std::abs(bands[b].second - bands[b + 1].first) > 1e-12 * bands[b].second) throw std::invalid_argument("power_partition: the bands must be adjacent (each starts where the last ends)");
  const double half = 0.25;
  for (Eigen::Index k = 0; k < nf; ++k) {
    const double f = std::abs(freqs(k));
    if (!(f > 0)) continue;
    const double lf = std::log2(std::max(f, 1e-12));
    auto rise = [&](double e) { const double t = std::clamp((lf - (e - half)) / (2 * half), 0.0, 1.0); return std::sin(0.5 * M_PI * t); };
    for (Eigen::Index b = 0; b < nb; ++b) {
      const double up = rise(std::log2(edges[b])), r = rise(std::log2(edges[b + 1]));
      g(b, k) = up * std::sqrt(std::clamp(1 - r * r, 0.0, 1.0));
    }
  }
  return g;
}

Spectrum spectrumOf(const Eigen::MatrixXd& block, double sfreq) {
  if (!(sfreq > 0)) throw std::invalid_argument("spectrumOf: the sampling rate is positive");
  if (block.rows() < 2) throw std::invalid_argument("spectrumOf: a block holds at least 2 samples");
  Spectrum s;
  s.sfreq = sfreq;
  s.X = nxr::signal::fftColumns(block.cast<std::complex<double>>(), true);
  s.freq = nxr::signal::fftfreq(static_cast<std::size_t>(block.rows()), 1.0 / sfreq);
  return s;
}

Eigen::MatrixXcd analyticBand(const Spectrum& spec, const Eigen::VectorXd& gain) {
  if (gain.size() != spec.freq.size()) throw std::invalid_argument("analyticBand: one gain a frequency bin");
  Eigen::MatrixXcd Y = spec.X;
  for (Eigen::Index k = 0; k < Y.rows(); ++k) {
    const double h = spec.freq(k) > 0 ? 2.0 * gain(k) : 0.0;     // positive frequencies doubled, negatives (and the Nyquist bin of an even length, which fftfreq calls negative) removed
    Y.row(k) *= h;
  }
  return nxr::signal::fftColumns(Y, false);
}

}  // namespace nxr::pyramid
