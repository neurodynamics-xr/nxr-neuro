// nxr::spectral (the scaling chosen at read) and timebank::noiseBandwidth, on synthetic numbers and signals only:
//   the conversions: density ↔ power ↔ spectrum by the normalisers, amplitude and dB, the degree, refusals (a non-linear target on a product, a missing
//   normaliser, an unknown name); units;
//   WHITE NOISE three ways: Welch density (scipy scaling), Welch spectrum (÷ the window's ENBW per bin) and the analytic signal's band power (÷ its noise
//   bandwidth) all read as the same density, σ² / (fs / 2).
#include "nxr/signal/wavelet.h"
#include <cmath>
#include <iostream>
#include <random>

#include "nxr/signal/spectral_scaling.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/spectral.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::signal::spectral_scaling;

int main() {
  std::cout << "test_spectral_scaling\n";
  Scaling d; d.stored = Stored::Density; d.bandWidthHz = 4.0;
  EXPECT(std::abs(convert(2.0, d, Target::Power) - 8.0) < 1e-12 && std::abs(convert(2.0, d, Target::Amplitude) - std::sqrt(8.0)) < 1e-12 &&
         std::abs(convert(2.0, d, Target::Decibel) - 10 * std::log10(8.0)) < 1e-12 && std::abs(convert(2.0, d, Target::AmplitudeDensity) - std::sqrt(2.0)) < 1e-12,
         "a density: power = d B, amplitude = sqrt(power), dB = 10 log10(power), amplitude density = sqrt(d)");
  Scaling sp; sp.stored = Stored::Spectrum; sp.binEnbwHz = 0.375; sp.bandWidthHz = 4.0;
  EXPECT(std::abs(convert(3.0, sp, Target::Density) - 8.0) < 1e-12, "a spectrum value / the ENBW per bin is the density");
  Scaling bp; bp.stored = Stored::BandPower; bp.bandWidthHz = 4.0;
  EXPECT(std::abs(convert(8.0, bp, Target::Density) - 2.0) < 1e-12 && convert(8.0, bp, Target::Power) == 8.0, "a band power / its bandwidth is the density");
  Scaling q = d; q.degree = 2;
  EXPECT(std::abs(linearFactor(q, Target::Power) - 16.0) < 1e-12, "a product of two spectra converts by the factor squared");
  bool r1 = false, r2 = false, r3 = false;
  try { convert(1.0, q, Target::Decibel); } catch (const std::invalid_argument&) { r1 = true; }
  try { Scaling m; m.stored = Stored::Density; convert(1.0, m, Target::Power); } catch (const std::invalid_argument&) { r2 = true; }
  try { parseTarget("rms"); } catch (const std::invalid_argument&) { r3 = true; }
  EXPECT(r1 && r2 && r3, "refused: dB of a product of spectra, a missing bandwidth, an unknown target");
  EXPECT(std::isnan(convert(-1.0, d, Target::Amplitude)), "a negative power has no amplitude (NaN)");
  EXPECT(unitOf("A\xc2\xb2", d, Target::Density) == "A\xc2\xb2/Hz" && unitOf("A\xc2\xb2", d, Target::Decibel) == "dB re 1 A\xc2\xb2", "units follow the target");
  // WHITE NOISE, σ = 1.5 at fs = 200 Hz: the one-sided density is σ² / (fs / 2)
  const double fs = 200, sigma = 1.5, truth = sigma * sigma / (fs / 2);
  const int n = 1 << 17;
  std::mt19937 rng(4);
  std::normal_distribution<double> nd(0, sigma);
  Eigen::MatrixXd x(1, n);
  for (int i = 0; i < n; ++i) x(0, i) = nd(rng);
  double dens = 0, spec = 0, enbw = 0;
  {
    nxr::signal::spectral::WelchOptions wo = nxr::signal::spectral::brainstormPsdOptions(1.0); wo.sfreq = fs; wo.segmentSamples = 400;
    nxr::signal::spectral::WelchOptions ws = wo; ws.scaling = nxr::signal::spectral::Scaling::Spectrum;
    nxr::signal::spectral::Welch a(1, wo), b(1, ws);
    a.add(x); b.add(x);
    const auto ra = a.result(), rb = b.result();
    int k0 = 40, k1 = 60;                                   // 20 … 30 Hz at 0.5 Hz bins
    for (int k = k0; k <= k1; ++k) { dens += ra.psd(0, k); spec += rb.psd(0, k); }
    dens /= (k1 - k0 + 1); spec /= (k1 - k0 + 1);
    enbw = nxr::signal::spectral::enbwPerBin(ws);
  }
  Scaling ws; ws.stored = Stored::Spectrum; ws.binEnbwHz = enbw;
  nxr::signal::timefilterbank::Design dsg(nxr::signal::timefilterbank::Family::Meyer, 2.0, nxr::signal::timefilterbank::Sides::Analytic); dsg.bands = {{20.0, 30.0}};
  const nxr::signal::timefilterbank::TimeFilterbank bank = nxr::signal::timefilterbank::design(dsg, static_cast<std::size_t>(n), fs);
  const Eigen::MatrixXcd z = nxr::signal::wt(bank, Eigen::MatrixXd(x.transpose()))[0];
  Scaling an; an.stored = Stored::BandPower; an.bandWidthHz = nxr::signal::timefilterbank::noiseBandwidth(bank, 0);
  const double analytic = convert(z.squaredNorm() / n / 2, an, Target::Density);
  std::printf("  white noise density %.5f: Welch density %.5f, Welch spectrum / ENBW %.5f, analytic band power / B_n %.5f (B_n %.2f Hz)\n", truth, dens,
              convert(spec, ws, Target::Density), analytic, an.bandWidthHz);
  EXPECT(std::abs(dens / truth - 1) < 0.03 && std::abs(convert(spec, ws, Target::Density) / truth - 1) < 0.03 && std::abs(analytic / truth - 1) < 0.03,
         "white noise: the three estimators read as the same density (within 3 %)");
  std::cout << (failures ? "FAIL" : "PASS") << " test_spectral_scaling (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
