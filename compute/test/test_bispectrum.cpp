// nxr::signal bispectrum and flow_across_frequencies on synthetic data (generated signals and random arrays; never a real recording):
//   atomBispectrum against the direct sums over every pair of bins (the bispectrum, the threenorm's cube sums, Kim and Powers' product and sum powers), for both groupings
//     (pairs of (second, sum) correlated, pairs of (first, second) convolved) and a first bin above 0; both bicoherences at most 1;
//   phase-amplitude coupling: a 40 Hz rhythm whose amplitude follows a 6 Hz rhythm's phase gives a threenorm bicoherence on the (6, 40, 40) atoms over 3 times that of the
//     same amplitude modulation with a phase unrelated to the 6 Hz rhythm and 5 times that of Gaussian noise;
//   harmonics and waveform shape on the (8, 8, 16) atoms: a sawtooth's bicoherence is over 5 times noise's; its biphase is +90 degrees for a slow rise and fast decay, -90 for the reverse;
//     a wave with sharp peaks (cos + 0.5 cos 2) has biphase 0 (positive real part), with sharp troughs 180;
//   flowAcrossFrequencies: the same field at two frequencies gives cosines 1, direction consistency 1 and a speed slope 1 (same wavelength) or 0 (same speed).
#include <cmath>
#include <complex>
#include <functional>
#include <algorithm>
#include <iostream>
#include <random>

#include "nxr/signal/bispectrum.h"
#include "nxr/signal/flow_across_frequencies.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/spectral_flow.h"

// Gaussian gains on a frequency line, standard deviations in Hz, 0 at f <= 0: the test's own synthetic atoms (a kernel set's gains are nxr::signal::kernelGains)
static Eigen::MatrixXd gaussianGains(const Eigen::VectorXd& frequencies, const std::vector<double>& centres, const std::vector<double>& standardDeviations, double gain) {
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(centres.size()), frequencies.size());
  for (std::size_t a = 0; a < centres.size(); ++a)
    for (Eigen::Index k = 0; k < frequencies.size(); ++k) {
      const double f = frequencies(k), sd = standardDeviations[a];
      if (f > 0) g(static_cast<Eigen::Index>(a), k) = gain * std::exp(-(f - centres[a]) * (f - centres[a]) / (2.0 * sd * sd));
    }
  return g;
}

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

namespace ns = nxr::signal;
using cd = std::complex<double>;

namespace {
// the direct sums over every pair of bins (i, j) whose sum i + j + firstBin is a kept bin
ns::TripleSums direct(const Eigen::MatrixXd& g, std::int64_t f0, const std::vector<ns::AtomTriple>& triples, const Eigen::MatrixXcd& X) {
  const Eigen::Index R = X.rows(), B = X.cols(), T = static_cast<Eigen::Index>(triples.size());
  ns::TripleSums s;
  s.bispectrum = Eigen::MatrixXcd::Zero(R, T);
  for (Eigen::MatrixXd* m : {&s.firstCube, &s.secondCube, &s.sumCube, &s.productPower, &s.sumPower}) *m = Eigen::MatrixXd::Zero(R, T);
  for (Eigen::Index t = 0; t < T; ++t)
    for (Eigen::Index r = 0; r < R; ++r)
      for (Eigen::Index i = 0; i < B; ++i)
        for (Eigen::Index j = 0; j < B; ++j) {
          const Eigen::Index k = i + j + f0;
          if (k >= B) continue;
          const auto& tr = triples[static_cast<std::size_t>(t)];
          const double w = g(tr.first, i) * g(tr.second, j) * g(tr.sum, k);
          s.bispectrum(r, t) += w * X(r, i) * X(r, j) * std::conj(X(r, k));
          s.firstCube(r, t) += w * std::pow(std::abs(X(r, i)), 3); s.secondCube(r, t) += w * std::pow(std::abs(X(r, j)), 3); s.sumCube(r, t) += w * std::pow(std::abs(X(r, k)), 3);
          s.productPower(r, t) += w * std::norm(X(r, i) * X(r, j)); s.sumPower(r, t) += w * std::norm(X(r, k));
        }
  return s;
}
// the direct cross sums: the first axis on X1, the second on X2, the sum on X3
Eigen::MatrixXcd directCross(const Eigen::MatrixXd& g, std::int64_t f0, const std::vector<ns::AtomTriple>& triples, const Eigen::MatrixXcd& X1, const Eigen::MatrixXcd& X2, const Eigen::MatrixXcd& X3) {
  const Eigen::Index R = X1.rows(), B = X1.cols(), T = static_cast<Eigen::Index>(triples.size());
  Eigen::MatrixXcd b = Eigen::MatrixXcd::Zero(R, T);
  for (Eigen::Index t = 0; t < T; ++t)
    for (Eigen::Index r = 0; r < R; ++r)
      for (Eigen::Index i = 0; i < B; ++i)
        for (Eigen::Index j = 0; i + j + f0 < B; ++j) {
          const auto& tr = triples[static_cast<std::size_t>(t)];
          b(r, t) += g(tr.first, i) * g(tr.second, j) * g(tr.sum, i + j + f0) * X1(r, i) * X2(r, j) * std::conj(X3(r, i + j + f0));
        }
  return b;
}
double relative(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) { return (a - b).norm() / std::max(1e-300, b.norm()); }

// a recording of `segments` segments of n samples at fs, each made by `make(t, segment)`; their one-sided spectra below `top` Hz, rows = 1
Eigen::MatrixXcd segmentSpectrum(const std::function<double(double, int)>& make, int segment, int n, double fs, Eigen::Index bins) {
  Eigen::MatrixXd x(n, 1);
  for (int t = 0; t < n; ++t) x(t, 0) = make(t / fs, segment);
  const Eigen::MatrixXcd X = ns::rfft(x, 0);
  return X.topRows(bins).transpose();
}
double threenormBicoherence(const ns::TripleSums& s, Eigen::Index t) {
  return std::abs(s.bispectrum(0, t)) / std::cbrt(s.firstCube(0, t) * s.secondCube(0, t) * s.sumCube(0, t));
}
}  // namespace

int main() {
  std::cout << "test_bispectrum\n";
  std::mt19937 rng(5);
  std::normal_distribution<double> nd(0, 1);
  std::uniform_real_distribution<double> ud(0, 1);

  // ── against the direct sums ──
  {
    const Eigen::Index R = 3, B = 30, G = 6;
    Eigen::MatrixXcd X(R, B);
    for (Eigen::Index r = 0; r < R; ++r) for (Eigen::Index k = 0; k < B; ++k) X(r, k) = cd(nd(rng), nd(rng));
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(G, B);
    for (Eigen::Index m = 0; m < G; ++m) for (Eigen::Index k = 2 + m; k < std::min<Eigen::Index>(B, 9 + 3 * m); ++k) g(m, k) = ud(rng);
    const std::vector<ns::AtomTriple> coupling{{0, 3, 3}, {1, 3, 3}, {2, 3, 3}, {0, 4, 4}};        // shared (second, sum): correlated once per pair
    const std::vector<ns::AtomTriple> harmonic{{1, 1, 4}, {1, 1, 5}, {1, 1, 3}, {2, 2, 5}};        // shared (first, second): convolved once per pair
    for (const std::int64_t f0 : {0, 2})
      for (const auto* triples : {&coupling, &harmonic}) {
        const ns::AtomTriples at = ns::atomTriples(g, f0, *triples);
        const ns::TripleSums ref = direct(g, f0, *triples, X);
        const ns::TripleSums three = ns::atomBispectrum(at, X, ns::BicoherenceNormalisation::threenorm);
        const ns::TripleSums kim = ns::atomBispectrum(at, X, ns::BicoherenceNormalisation::kim_powers);
        const std::string label = std::string(triples == &coupling ? "(second, sum) grouping" : "(first, second) grouping") + ", first bin " + std::to_string(f0);
        EXPECT(at.groupedBySecondAndSum == (triples == &coupling), label + ": the grouping with fewer distinct pairs is chosen");
        const double e = std::max({relative(three.bispectrum, ref.bispectrum), relative(kim.bispectrum, ref.bispectrum),
                                   relative(three.firstCube.cast<cd>(), ref.firstCube.cast<cd>()), relative(three.secondCube.cast<cd>(), ref.secondCube.cast<cd>()),
                                   relative(three.sumCube.cast<cd>(), ref.sumCube.cast<cd>()), relative(kim.productPower.cast<cd>(), ref.productPower.cast<cd>()),
                                   relative(kim.sumPower.cast<cd>(), ref.sumPower.cast<cd>())});
        EXPECT(e < 1e-12, label + ": every sum equals the direct sum over the pairs of bins (relative error " + std::to_string(e) + ")");
        double weightError = 0;
        for (std::size_t t = 0; t < triples->size(); ++t) {
          double w = 0;
          for (Eigen::Index i = 0; i < B; ++i) for (Eigen::Index j = 0; i + j + f0 < B; ++j) w += g((*triples)[t].first, i) * g((*triples)[t].second, j) * g((*triples)[t].sum, i + j + f0);
          weightError = std::max(weightError, std::abs(w - at.weight(static_cast<Eigen::Index>(t))) / std::max(1e-300, w));
        }
        EXPECT(weightError < 1e-12, label + ": the weights are the gains' sums over the pairs");
        double top = 0;
        for (Eigen::Index r = 0; r < R; ++r)
          for (Eigen::Index t = 0; t < three.bispectrum.cols(); ++t) {
            if (at.weight(t) == 0) continue;
            top = std::max(top, std::abs(three.bispectrum(r, t)) / std::cbrt(three.firstCube(r, t) * three.secondCube(r, t) * three.sumCube(r, t)));
            top = std::max(top, std::norm(kim.bispectrum(r, t)) / (kim.productPower(r, t) * kim.sumPower(r, t)));
          }
        EXPECT(top <= 1 + 1e-12, label + ": both bicoherences are at most 1 (largest " + std::to_string(top) + ")");
      }
    // every triple of bands: the first axis on band p's spectrum, the second on q's, the sum on r's
    Eigen::MatrixXcd Y(R, B);
    for (Eigen::Index r = 0; r < R; ++r) for (Eigen::Index k = 0; k < B; ++k) Y(r, k) = cd(nd(rng), nd(rng));
    const std::vector<Eigen::MatrixXcd> bandSpectra{X, Y};
    for (const auto* triples : {&coupling, &harmonic}) {
      const ns::AtomTriples at = ns::atomTriples(g, 1, *triples);
      const std::vector<ns::TripleSums> each = ns::bandTripleBispectrum(at, bandSpectra, ns::BicoherenceNormalisation::kim_powers);
      double e = 0;
      for (int p = 0; p < 2; ++p) for (int q = 0; q < 2; ++q) for (int r = 0; r < 2; ++r)
        e = std::max(e, relative(each[static_cast<std::size_t>((p * 2 + q) * 2 + r)].bispectrum, directCross(g, 1, *triples, bandSpectra[p], bandSpectra[q], bandSpectra[r])));
      EXPECT(e < 1e-12, std::string(triples == &coupling ? "(second, sum)" : "(first, second)") + " grouping: every band triple equals the direct cross sum (relative error " + std::to_string(e) + ")");
    }
    ns::TripleSums twice = ns::atomBispectrum(ns::atomTriples(g, 0, coupling), X, ns::BicoherenceNormalisation::threenorm);
    twice += twice;
    EXPECT(relative(twice.bispectrum, 2.0 * direct(g, 0, coupling, X).bispectrum) < 1e-12, "the sums of two chunks are the sum of each");
    bool refusedNegative = false;
    try { Eigen::MatrixXd bad = g; bad(0, 3) = -1; (void)ns::atomTriples(bad, 0, coupling); } catch (const std::invalid_argument&) { refusedNegative = true; }
    EXPECT(refusedNegative, "a negative gain is refused (the bounds need non-negative gains)");
  }

  // ── synthetic signals: 60 segments of 512 samples at 256 Hz, bins below 60 Hz ──
  const int n = 512, segments = 60;
  const double fs = 256.0;
  const Eigen::Index bins = 120;
  Eigen::VectorXd frequency(bins);
  for (Eigen::Index k = 0; k < bins; ++k) frequency(k) = k * fs / n;
  auto sums = [&](const std::function<double(double, int)>& make, const ns::AtomTriples& at) {
    ns::TripleSums total;
    for (int s = 0; s < segments; ++s) total += ns::atomBispectrum(at, segmentSpectrum(make, s, n, fs, bins), ns::BicoherenceNormalisation::threenorm);
    return total;
  };
  std::mt19937 signals(17);                                                      // the generated signals' own generator: independent of the blocks above
  std::vector<double> phase(segments), other(segments), carrier(segments);
  for (int s = 0; s < segments; ++s) { phase[s] = 2 * M_PI * ud(signals); other[s] = 2 * M_PI * ud(signals); carrier[s] = 2 * M_PI * ud(signals); }
  std::vector<std::vector<double>> noise(segments, std::vector<double>(n));
  for (auto& seg : noise) for (auto& v : seg) v = nd(signals);
  {
    // phase-amplitude coupling: the atoms (6 Hz, sd 0.6) and (40 Hz, sd 5): the high atom is wide enough to hold 40 +- 6 Hz (Zandvoort and Nolte 2021)
    Eigen::MatrixXd g(2, bins);
    g.row(0) = gaussianGains(frequency, {6.0}, {0.6}, 1.0);
    g.row(1) = gaussianGains(frequency, {40.0}, {5.0}, 1.0);
    const ns::AtomTriples at = ns::atomTriples(g, 0, {{0, 1, 1}});
    const double wl = 2 * M_PI * 6, wh = 2 * M_PI * 40;
    const auto coupled = [&](double t, int s) { return std::cos(wl * t + phase[s]) + (1 + 0.8 * std::cos(wl * t + phase[s])) * std::cos(wh * t + carrier[s]) + 0.3 * noise[s][static_cast<std::size_t>(std::lround(t * fs))]; };
    const auto uncoupled = [&](double t, int s) { return std::cos(wl * t + phase[s]) + (1 + 0.8 * std::cos(wl * t + other[s])) * std::cos(wh * t + carrier[s]) + 0.3 * noise[s][static_cast<std::size_t>(std::lround(t * fs))]; };
    const auto gaussian = [&](double t, int s) { return noise[s][static_cast<std::size_t>(std::lround(t * fs))]; };
    const double bc = threenormBicoherence(sums(coupled, at), 0), bu = threenormBicoherence(sums(uncoupled, at), 0), bn = threenormBicoherence(sums(gaussian, at), 0);
    std::cout << "    phase-amplitude coupling, threenorm bicoherence: coupled " << bc << ", unrelated phase " << bu << ", Gaussian noise " << bn << "\n";
    // the atoms' bins without the rhythms count in the threenorm too, so the coupled value is far from 1: the test is against the controls
    // an unrelated modulation leaves a residue of about the coupled value / sqrt(segments) (60 segments: about an eighth), so the bound is 3 times
    EXPECT(bc > 0.08 && bc > 3 * bu && bc > 5 * bn, "a 40 Hz amplitude following the 6 Hz phase gives a bicoherence on (6, 40, 40) over 3 times an unrelated modulation's and 5 times noise's");
  }
  {
    // harmonics and waveform shape: the atom (8 Hz, sd 1) on the first two axes and the same atom at f / 2 on the sum (its centre 16 Hz)
    Eigen::MatrixXd g(2, bins);
    g.row(0) = gaussianGains(frequency, {8.0}, {1.0}, 1.0);
    g.row(1) = gaussianGains(frequency / 2.0, {8.0}, {1.0}, 1.0);
    const ns::AtomTriples at = ns::atomTriples(g, 0, {{0, 0, 1}});
    const double w = 2 * M_PI * 8;
    auto frac = [](double v) { return v - std::floor(v); };
    const auto rising = [&](double t, int s) { return 2 * frac(8 * t + phase[s] / (2 * M_PI)) - 1; };     // slow rise, fast decay
    const auto falling = [&](double t, int s) { return -rising(t, s); };
    const auto peaks = [&](double t, int s) { return std::cos(w * t + phase[s]) + 0.5 * std::cos(2 * (w * t + phase[s])); };
    const auto troughs = [&](double t, int s) { return -peaks(t, s); };
    const auto gaussian = [&](double t, int s) { return noise[s][static_cast<std::size_t>(std::lround(t * fs))]; };
    auto biphase = [&](const ns::TripleSums& s) { return std::arg(s.bispectrum(0, 0)) * 180 / M_PI; };
    const ns::TripleSums r = sums(rising, at), f = sums(falling, at), p = sums(peaks, at), q = sums(troughs, at), z = sums(gaussian, at);
    std::cout << "    harmonics: threenorm bicoherence sawtooth " << threenormBicoherence(r, 0) << ", noise " << threenormBicoherence(z, 0) << "; biphase (degrees) slow rise "
              << biphase(r) << ", fast rise " << biphase(f) << ", sharp peaks " << biphase(p) << ", sharp troughs " << biphase(q) << "\n";
    EXPECT(threenormBicoherence(r, 0) > 0.15 && threenormBicoherence(r, 0) > 5 * threenormBicoherence(z, 0), "a sawtooth's harmonic gives a bicoherence on (8, 8, 16) over 5 times noise's");
    EXPECT(std::abs(biphase(r) - 90) < 5 && std::abs(biphase(f) + 90) < 5, "rise-decay asymmetry is the imaginary part: +90 degrees for a slow rise, -90 for a fast rise");
    EXPECT(std::abs(biphase(p)) < 5 && std::abs(std::abs(biphase(q)) - 180) < 5, "peak-trough asymmetry is the real part: 0 for sharp peaks, 180 for sharp troughs");
  }

  // ── the flow across frequencies ──
  {
    const Eigen::Index E = 5, M = 4, S = 3;
    Eigen::MatrixXd edges(E, 2), modes(M, 2), power(S, 2), current(3 * S, 2);
    for (Eigen::Index e = 0; e < E; ++e) edges(e, 0) = nd(rng);
    edges.col(1) = 3.0 * edges.col(0);
    for (Eigen::Index k = 0; k < M; ++k) modes(k, 0) = nd(rng);
    modes.col(1) = 0.5 * modes.col(0);
    for (Eigen::Index v = 0; v < S; ++v) { power(v, 0) = 1 + ud(rng); for (int c = 0; c < 3; ++c) current(3 * v + c, 0) = nd(rng); }
    const Eigen::VectorXd frequencies = Eigen::Vector2d(5.0, 10.0);
    // the same wavelength at both: the power-weighted phase gradient (current / power) the same, so speed doubles with frequency
    power.col(1) = 2.0 * power.col(0); current.col(1) = 2.0 * current.col(0);
    const ns::FlowAcrossFrequencies same = ns::flowAcrossFrequencies(edges, Eigen::VectorXd::Ones(E), modes, {true, true, false, false}, power, current, frequencies);
    EXPECT(std::abs(same.edgeCosine(0, 1) - 1) < 1e-12 && std::abs(same.travelCosine(0, 1) - 1) < 1e-12 && std::abs(same.rotationCosine(0, 1) - 1) < 1e-12,
           "the same field at two frequencies: every cosine is 1");
    EXPECT((same.directionConsistency.array() - 1).abs().maxCoeff() < 1e-12 && (same.speedSlope.array() - 1).abs().maxCoeff() < 1e-12, "same wavelength: direction consistency 1, speed slope 1");
    current.col(1) = 4.0 * current.col(0);                                                          // the gradient (current / power) doubles with the frequency: the same speed
    const ns::FlowAcrossFrequencies speed = ns::flowAcrossFrequencies(edges, Eigen::VectorXd::Ones(E), modes, {true, true, false, false}, power, current, frequencies);
    EXPECT(speed.speedSlope.array().abs().maxCoeff() < 1e-12, "same speed: speed slope 0");
    edges.col(1) = -edges.col(0);
    const ns::FlowAcrossFrequencies opposite = ns::flowAcrossFrequencies(edges, Eigen::VectorXd::Ones(E), modes, {true, true, false, false}, power, current, frequencies);
    EXPECT(std::abs(opposite.edgeCosine(0, 1) + 1) < 1e-12, "opposite edge currents: cosine -1");
  }
  std::cout << (failures ? "FAIL" : "PASS") << " test_bispectrum\n";
  return failures ? 1 : 0;
}
