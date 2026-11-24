// crossSpectrumTiles and nxr::signal::wtBlock, on synthetic signals only:
//   the outer form      per tile Σ z zᴴ against a direct sum; Hermitian, its diagonal Σ|z|²; a partial last tile and a tile past the data; ADDITIVE (a
//                      window cut in two parts, summed, is the window); a known lag read back from the phase of an off-diagonal entry;
//   wtBlock    chunks with the band's halo (haloOf: its taps' length) transformed one by one give, on their own columns, what the whole record transformed at once
//                      gives (a Meyer band [8, 12] Hz); the FFT route equals the taps route (the same Meyer gain: even and odd taps) away from the
//                      record's ends; a Morlet member returns a tone's amplitude (gain 2: |c| = A); the own columns' shape.
#include "nxr/signal/wavelet.h"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "nxr/signal/graphfilterbank.h"
#include "measurement_helpers.h"
#include "nxr/signal/filters_fir.h"
#include "nxr/signal/timefilterbank.h"

using C = std::complex<double>;
static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_cross_spectrum\n";
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;

  // ── crossSpectrum ──
  {
    const Eigen::Index M = 5, n = 1000;
    Eigen::MatrixXcd Z(M, n);
    for (Eigen::Index i = 0; i < Z.size(); ++i) Z.data()[i] = C(nd(rng), nd(rng));
    const std::size_t width = 300, tiles = 5;                      // 300 · 300 · 300 · 100 (partial) · 0 (past the data)
    const auto r = crossSpectrumTiles(Z, width, tiles);
    double worst = 0, herm = 0, diag = 0;
    for (std::size_t t = 0; t < 4; ++t) {
      Eigen::MatrixXcd d = Eigen::MatrixXcd::Zero(M, M);
      const Eigen::Index from = static_cast<Eigen::Index>(t * width), to = std::min<Eigen::Index>(from + static_cast<Eigen::Index>(width), n);
      for (Eigen::Index s = from; s < to; ++s) for (Eigen::Index a = 0; a < M; ++a) for (Eigen::Index b = 0; b < M; ++b) d(a, b) += Z(a, s) * std::conj(Z(b, s));
      worst = std::max(worst, (r.sum[t] - d).cwiseAbs().maxCoeff() / d.cwiseAbs().maxCoeff());
      herm = std::max(herm, (r.sum[t] - r.sum[t].adjoint()).cwiseAbs().maxCoeff());
      for (Eigen::Index a = 0; a < M; ++a) diag = std::max(diag, std::abs(r.sum[t](a, a) - C(Z.row(a).segment(from, to - from).squaredNorm(), 0)));
    }
    EXPECT(worst < 1e-14 && herm < 1e-10 && diag < 1e-10, "Σ z zᴴ per tile == a direct sum (worst " << worst << "); Hermitian; its diagonal Σ|z|²");
    EXPECT(r.count == (std::vector<std::int64_t>{300, 300, 300, 100, 0}) && r.sum[4].isZero(), "a partial last tile (100 samples) and a tile past the data (0)");
    // additivity: tile 1 = [300, 600) cut at 450: the two parts' sums add up to it
    const auto a = crossSpectrumTiles(Z.middleCols(300, 150), 150, 1), b = crossSpectrumTiles(Z.middleCols(450, 150), 150, 1);
    EXPECT((a.sum[0] + b.sum[0] - r.sum[1]).cwiseAbs().maxCoeff() < 1e-10 && a.count[0] + b.count[0] == r.count[1], "a window cut in two: the parts add up to it");
    // a lag: channel 1 lags channel 0 by 0.7 rad: arg C_01 = 0.7
    Eigen::MatrixXcd L(2, 2000);
    for (Eigen::Index s = 0; s < 2000; ++s) { const double w = 2 * M_PI * 10 * s / 1000.0; L(0, s) = std::polar(1.0, w); L(1, s) = std::polar(0.5, w - 0.7); }
    const auto l = crossSpectrumTiles(L, 2000, 1);
    EXPECT(std::abs(std::arg(l.sum[0](0, 1)) - 0.7) < 1e-12 && std::abs(std::abs(l.sum[0](0, 1)) / 2000 - 0.5) < 1e-12, "a 0.7 rad lag read back from arg C_01");
  }

  // ── wtBlock: a record of 3 channels, 20000 samples at 1000 Hz (10 Hz in noise), in chunks of 4096 with a halo ──
  const double fs = 1000.0;
  const Eigen::Index channels = 3, n = 20000, chunk = 4096;
  Eigen::MatrixXd x(channels, n);
  for (Eigen::Index c = 0; c < channels; ++c)
    for (Eigen::Index s = 0; s < n; ++s) x(c, s) = std::sin(2 * M_PI * 10.0 * s / fs + c) + 0.5 * nd(rng);
  nxr::signal::timefilterbank::Design meyer(nxr::signal::timefilterbank::Family::Meyer, 2.0, nxr::signal::timefilterbank::Sides::Analytic);
  meyer.bands = {{8.0, 12.0}};                                   // the Signal chain's alpha band (the Studio's meyerBand), one member
  auto bandOf = [](const nxr::signal::timefilterbank::TimeFilterbank& b) {
    for (int m = 0; m < b.size(); ++m) if (std::abs(b.members[static_cast<std::size_t>(m)].lo - 8.0) < 1e-9 && std::abs(b.members[static_cast<std::size_t>(m)].hi - 12.0) < 1e-9) return m;
    return -1;
  };
  const auto whole = nxr::signal::wtBlock(meyer, x, fs, 0, static_cast<std::size_t>(n));
  const int band = bandOf(whole.bank);
  EXPECT(band >= 0 && whole.own[static_cast<std::size_t>(band)].rows() == channels && whole.own[static_cast<std::size_t>(band)].cols() == n, "the Meyer band [8, 12] Hz: one member");
  if (band < 0) return 1;
  const std::size_t H = nxr::signal::timefilterbank::haloOfTaps(whole.bank, band);
  double worst = 0, scale = whole.own[static_cast<std::size_t>(band)].cwiseAbs().maxCoeff();
  for (Eigen::Index first = 0; first < n; first += chunk) {
    const Eigen::Index own = std::min(chunk, n - first), left = std::min<Eigen::Index>(static_cast<Eigen::Index>(H), first), right = std::min<Eigen::Index>(static_cast<Eigen::Index>(H), n - first - own);
    if (left < static_cast<Eigen::Index>(H) || right < static_cast<Eigen::Index>(H)) continue;   // the record's ends: no context there (zero-padded), compared below only inside
    const auto t = nxr::signal::wtBlock(meyer, x.middleCols(first - left, left + own + right), fs, static_cast<std::size_t>(left), static_cast<std::size_t>(own));
    worst = std::max(worst, (t.own[static_cast<std::size_t>(band)] - whole.own[static_cast<std::size_t>(band)].middleCols(first, own)).cwiseAbs().maxCoeff() / scale);
  }
  EXPECT(worst < 1e-3, "chunks of 4096 with the band's halo, haloOf = " << H << " samples == the whole record transformed at once (worst relative " << worst << ")");
  // the FFT route and the taps route: the same Meyer gain
  {
    const auto gain = [](double f) { return nxr::signal::timefilterbank::meyerBand(f, 8.0, 12.0, 1.0 / 3.0); };
    const auto h = nxr::signal::gainTaps(gain, fs, false), g = nxr::signal::gainTaps(gain, fs, true);
    std::vector<float> xf(static_cast<std::size_t>(channels * n));
    for (Eigen::Index c = 0; c < channels; ++c) for (Eigen::Index s = 0; s < n; ++s) xf[static_cast<std::size_t>(c * n + s)] = static_cast<float>(x(c, s));
    const auto re = nxr::signal::convolveRows(xf.data(), channels, n, h, 0, n, false), im = nxr::signal::convolveRows(xf.data(), channels, n, g, 0, n, true);
    double d = 0;
    const Eigen::Index margin = 2000;                        // away from the record's ends (zero-padded there, reflected by the taps)
    for (Eigen::Index c = 0; c < channels; ++c)
      for (Eigen::Index s = margin; s < n - margin; ++s)
        d = std::max(d, std::abs(whole.own[static_cast<std::size_t>(band)](c, s) - C(re[static_cast<std::size_t>(c * n + s)], im[static_cast<std::size_t>(c * n + s)])) / scale);
    EXPECT(d < 1e-3, "the FFT route == the taps route (the band's even and odd taps) inside the record (worst relative " << d << ")");
  }
  // a Morlet member at 10 Hz, gain 2: a pure 10 Hz tone of amplitude A comes back with |c| = A
  {
    Eigen::MatrixXd tone(1, 8000);
    for (Eigen::Index s = 0; s < 8000; ++s) tone(0, s) = 1.5 * std::cos(2 * M_PI * 10.0 * s / fs);
    nxr::signal::timefilterbank::Design morlet(nxr::signal::timefilterbank::Family::Morlet, 2.0, nxr::signal::timefilterbank::Sides::Analytic);
    morlet.centres = {10.0};
    const auto t = nxr::signal::wtBlock(morlet, tone, fs, 2000, 4000);
    double lo = 1e9, hi = 0;
    for (Eigen::Index s = 0; s < 4000; ++s) { lo = std::min(lo, std::abs(t.own[0](0, s))); hi = std::max(hi, std::abs(t.own[0](0, s))); }
    EXPECT(t.own.size() == 1 && t.own[0].cols() == 4000 && std::abs(lo - 1.5) < 1e-3 && std::abs(hi - 1.5) < 1e-3, "Morlet at 10 Hz: a tone of amplitude 1.5 comes back with |c| in [" << lo << ", " << hi << "]");
  }
  EXPECT([&] { try { nxr::signal::wtBlock(meyer, x.leftCols(100), fs, 50, 60); } catch (const std::invalid_argument&) { return true; } return false; }(), "own columns outside the block refused");

  std::cout << (failures ? "FAIL" : "PASS") << " test_cross_spectrum (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
