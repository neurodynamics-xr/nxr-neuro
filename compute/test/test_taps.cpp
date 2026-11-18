// nxr::taps against nxr's own functions (golden: test/golden/gen_taps_golden.py, from nxr/filters/taps.py gain_taps and convolve_rows on
// nxr/filters/frames.py meyer_band; nxr's are the Studio's gainTaps and convolveRows): the even and odd taps of two Meyer bands (alpha at 1200 Hz, beta at
// 250 Hz) and the convolutions of random rows over a block's middle and over the whole block (its two ends reflected). Then what a pipeline relies on:
// THE HALO (a chunk convolved with K samples of context on each side equals the whole recording's result on its columns, to the bit); THE ANALYTIC SIGNAL
// (a 10 Hz cosine through the alpha band's even and odd taps is e^{iωt}: a constant amplitude and a phase advancing 2π·10/fs a sample); Σ h = G(0) = 0 for
// a band-pass; refusals.
#include "nxr/signal/filters_fir.h"

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
    std::vector<std::string> words;
    std::string w;
    while (s >> w) words.push_back(w);
    if (words.empty() || words[0] == "case") continue;
    // the tag is every word before the count (the first word that is the number of values left on the line)
    std::size_t i = 1;
    std::string tag = words[0];
    for (; i < words.size(); ++i) {
      char* end = nullptr;
      const long n = std::strtol(words[i].c_str(), &end, 10);
      if (*end == 0 && n >= 0 && static_cast<std::size_t>(n) == words.size() - i - 1) break;
      tag += " " + words[i];
    }
    std::vector<double> v;
    for (std::size_t j = i + 1; j < words.size(); ++j) v.push_back(std::strtod(words[j].c_str(), nullptr));
    r[tag] = std::move(v);
  }
  return r;
}

// values that differ from the golden as float32, and the largest relative difference (to the largest magnitude)
static std::pair<std::size_t, double> diff(const std::vector<float>& got, const std::vector<double>& want) {
  if (got.size() != want.size()) return {std::max(got.size(), want.size()), INFINITY};
  double scale = 0, worst = 0;
  std::size_t n = 0;
  for (double w : want) scale = std::max(scale, std::fabs(w));
  for (std::size_t i = 0; i < got.size(); ++i)
    if (got[i] != static_cast<float>(want[i])) { ++n; worst = std::max(worst, std::fabs(got[i] - want[i]) / scale); }
  return {n, worst};
}

int main() {
  using namespace nxr::signal;
  const auto g = load(kDir + "/taps_golden.txt");
  const std::vector<double>& xd = g.at("x 3 900");
  std::vector<float> x(xd.begin(), xd.end());
  struct Case { const char* name; double fs, lo, hi; };
  for (const Case& c : {Case{"alpha1200", 1200.0, 8.0, 12.0}, Case{"beta250", 250.0, 15.0, 29.0}}) {
    for (bool odd : {false, true}) {
      const std::string kind = odd ? "odd" : "even";
      const auto h = gainTaps([&](double f) { return nxr::signal::graphfilterbank::meyerBand(f, c.lo, c.hi, 1.0 / 3.0); }, c.fs, odd);
      const auto& want = g.at(std::string("taps ") + c.name + " " + kind);
      const auto dt = diff(h, want);
      std::printf("  %s %s taps: %zu of %zu (K %zu, nxr's %zu) differ from nxr's, max %.1e relative\n", c.name, kind.c_str(), dt.first, want.size(), h.size() - 1,
                  want.size() - 1, dt.second);
      EXPECT(h.size() == want.size(), c.name << " " << kind << ": the truncation K differs from nxr's");
      EXPECT(dt.second < 1e-6, c.name << " " << kind << " taps differ from nxr's beyond float rounding");
      // the convolutions, with nxr's own taps (so this checks the convolution alone)
      std::vector<float> hw(want.begin(), want.end());
      const std::size_t K = hw.size() - 1, left = std::min<std::size_t>(K, 300), width = 900 - 2 * left;
      for (const auto& [where, l, w] : {std::tuple<std::string, std::size_t, std::size_t>{"mid", left, width}, {"all", 0, 900}}) {
        const auto y = convolveRows(x.data(), 3, 900, hw, l, w, odd);
        const auto& wy = g.at(std::string("conv ") + c.name + " " + kind + " " + where + " " + std::to_string(l) + " " + std::to_string(w));
        const auto dy = diff(y, wy);
        if (dy.first) std::printf("  %s %s conv %s: %zu values differ, max %.1e\n", c.name, kind.c_str(), where.c_str(), dy.first, dy.second);
        EXPECT(dy.first == 0, c.name << " " << kind << " convolution (" << where << ") differs from nxr's");
      }
    }
  }
  // Σ h = G(0): a band-pass has no DC
  const double fs = 1200.0;
  auto alpha = [](double f) { return nxr::signal::graphfilterbank::meyerBand(f, 8.0, 12.0, 1.0 / 3.0); };
  const auto he = gainTaps(alpha, fs, false), ho = gainTaps(alpha, fs, true);
  double dc = he[0];
  for (std::size_t k = 1; k < he.size(); ++k) dc += 2.0 * he[k];
  EXPECT(std::fabs(dc) < 1e-6, "the even taps' sum is not G(0) = 0: " << dc);
  EXPECT(ho[0] == 0.0f, "the odd taps' centre is not 0");

  // THE HALO: a 4000-sample "recording", chunk [1500, 2500) convolved with K samples of context on each side, against the whole recording's result
  const std::size_t T = 4000, K = he.size() - 1, a = 1500, b = 2500;
  std::vector<float> rec(T);
  unsigned s = 7;
  for (auto& v : rec) { s = s * 1103515245u + 12345u; v = static_cast<float>(static_cast<double>(s) / 4294967296.0 - 0.5); }
  const auto whole = convolveRows(rec.data(), 1, T, he, 0, T);
  const std::vector<float> block(rec.begin() + static_cast<long>(a - K), rec.begin() + static_cast<long>(b + K));
  const auto part = convolveRows(block.data(), 1, block.size(), he, K, b - a);
  std::size_t halo_diff = 0;
  for (std::size_t t = 0; t < b - a; ++t) halo_diff += part[t] != whole[a + t];
  std::printf("  halo: chunk [%zu, %zu) with %zu samples of context each side against the whole recording: %zu values differ\n", a, b, K, halo_diff);
  EXPECT(halo_diff == 0, "a chunk with its halo does not equal the whole recording on its columns");

  // THE ANALYTIC SIGNAL of a 10 Hz cosine through the alpha band: e^{iωt} (amplitude 1, phase advancing ω/fs a sample)
  const double f0 = 10.0, w0 = 2 * M_PI * f0 / fs;
  std::vector<float> cosine(T);
  for (std::size_t t = 0; t < T; ++t) cosine[t] = static_cast<float>(std::cos(w0 * static_cast<double>(t)));
  const auto re = convolveRows(cosine.data(), 1, T, he, 0, T), im = convolveRows(cosine.data(), 1, T, ho, 0, T, true);
  double amp = 0, step = 0;
  for (std::size_t t = K; t + K < T; ++t) {
    const std::complex<double> z(re[t], im[t]), z1(re[t + 1 < T ? t + 1 : t], im[t + 1 < T ? t + 1 : t]);
    amp = std::max(amp, std::fabs(std::abs(z) - 1.0));
    double d = std::arg(z1) - std::arg(z);
    if (d < -M_PI) d += 2 * M_PI;
    if (t + 1 + K < T) step = std::max(step, std::fabs(d - w0));
  }
  std::printf("  analytic of a 10 Hz cosine (alpha band, away from the ends): amplitude within %.1e of 1, phase step within %.1e rad of ω/fs\n", amp, step);
  EXPECT(amp < 1e-3 && step < 1e-3, "the band's analytic signal of a cosine is not e^{iωt}");

  // REFUSALS
  bool refused = false;
  try { convolveRows(x.data(), 3, 900, he, 800, 200); } catch (const std::invalid_argument&) { refused = true; }
  EXPECT(refused, "columns outside the block: not refused");

  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
