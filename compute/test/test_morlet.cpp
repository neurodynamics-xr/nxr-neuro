// nxr::signal::graphfilterbank::morletBand against nxr (golden: test/golden/gen_morlet_golden.py, from nxr/detect/vortex.py morlet_gain, the
// Studio's morlet_hat, nxr/filters/taps.py gain_taps): the gain, the even and odd taps of two bands, and the band's analytic signal of a synthetic series through the taps
// route (nxr nxr::taps) against nxr's FFT route, away from the series' ends. Then: unit peak at the arithmetic centre, FWHM = the band's width, a refusal.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/filters_fir.h"

static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

static std::map<std::string, std::vector<double>> load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("no golden at " + path);
  std::map<std::string, std::vector<double>> r;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::vector<std::string> w;
    for (std::string t; s >> t;) w.push_back(t);
    if (w.empty() || w[0] == "case") continue;
    std::size_t i = 1;
    std::string tag = w[0];
    for (; i < w.size(); ++i) {
      char* end = nullptr;
      const long n = std::strtol(w[i].c_str(), &end, 10);
      if (*end == 0 && n >= 0 && static_cast<std::size_t>(n) == w.size() - i - 1) break;
      tag += " " + w[i];
    }
    std::vector<double> v;
    for (std::size_t j = i + 1; j < w.size(); ++j) v.push_back(std::strtod(w[j].c_str(), nullptr));
    r[tag] = std::move(v);
  }
  return r;
}

int main() {
  using namespace nxr::signal;
  const auto g = load(kDir + "/morlet_golden.txt");
  const auto& xd = g.at("x 1");
  std::vector<float> x(xd.begin(), xd.end());
  struct Case { const char* name; double fs, lo, hi; };
  for (const Case& c : {Case{"alpha100", 100.0, 8.0, 12.0}, Case{"beta100", 100.0, 15.0, 29.0}}) {
    auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, c.lo, c.hi); };
    // the gain on the grid f = 0.5 … 60
    const auto& wg = g.at(std::string("gain ") + c.name);
    double worst = 0;
    for (std::size_t i = 0; i < wg.size(); ++i) worst = std::max(worst, std::fabs(gain(0.5 * static_cast<double>(i + 1)) - wg[i]));
    std::printf("  %s gain: max difference from nxr's morlet_gain/2 %.1e\n", c.name, worst);
    EXPECT(worst < 1e-12, c.name << ": the gain differs from nxr's");
    // the taps
    std::vector<std::vector<float>> th;
    for (bool odd : {false, true}) {
      const auto h = gainTaps(gain, c.fs, odd);
      const auto& want = g.at(std::string("taps ") + c.name + (odd ? " odd" : " even"));
      EXPECT(h.size() == want.size(), c.name << (odd ? " odd" : " even") << ": the truncation K differs from nxr's (" << h.size() << " vs " << want.size() << ")");
      double sc = 0, dw = 0;
      for (double v : want) sc = std::max(sc, std::fabs(v));
      for (std::size_t k = 0; k < std::min(h.size(), want.size()); ++k) dw = std::max(dw, std::fabs(h[k] - want[k]) / sc);
      std::printf("  %s %s taps: K %zu, max relative difference %.1e\n", c.name, odd ? "odd" : "even", h.size() - 1, dw);
      EXPECT(dw < 1e-6, c.name << ": taps differ from nxr's beyond float rounding");
      th.push_back(h);
    }
    // the analytic signal by taps against nxr's FFT route, away from the ends
    const std::size_t n = x.size(), K = th[0].size() - 1;
    const auto re = convolveRows(x.data(), 1, n, th[0], 0, n, false), im = convolveRows(x.data(), 1, n, th[1], 0, n, true);
    const auto& wa = g.at(std::string("analytic ") + c.name);
    double err = 0, scale = 0;
    for (std::size_t t = 0; t < n; ++t) scale = std::max(scale, std::hypot(wa[2 * t], wa[2 * t + 1]));
    for (std::size_t t = std::max<std::size_t>(K, 300); t + std::max<std::size_t>(K, 300) < n; ++t)
      err = std::max(err, std::hypot(re[t] - wa[2 * t], im[t] - wa[2 * t + 1]) / scale);
    std::printf("  %s analytic signal (taps vs nxr's FFT route, interior): max difference %.1e of the peak amplitude\n", c.name, err);
    EXPECT(err < 2e-3, c.name << ": the analytic signal differs from nxr's");
  }
  // unit peak at the arithmetic centre; half maximum at the band's edges' distance: FWHM = hi - lo
  const double lo = 15.0, hi = 29.0, f0 = (lo + hi) / 2;
  EXPECT(std::fabs(nxr::signal::graphfilterbank::morletBand(f0, lo, hi) - 1.0) < 1e-15, "the peak is not 1 at (lo + hi)/2");
  EXPECT(std::fabs(nxr::signal::graphfilterbank::morletBand(f0 - (hi - lo) / 2, lo, hi) - 0.5) < 1e-12, "the half maximum is not at half the width from the centre");
  EXPECT(std::fabs(nxr::signal::graphfilterbank::morletBand(f0 + (hi - lo) / 2, lo, hi) - 0.5) < 1e-12, "the half maximum is not at half the width from the centre (above)");
  bool refused = false;
  try { nxr::signal::graphfilterbank::morletBand(10.0, 12.0, 8.0); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "lo >= hi not refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
