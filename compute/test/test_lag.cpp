// nxr::stats (lagStatistics, holonomy, pairsAtDistance, circularShift) and timebank::equivalentBandwidth / independentEstimates, synthetic only:
//   the equivalent bandwidth of a rectangular band is its width; THE LAG'S STANDARD ERROR against a simulation: two channels sharing a band-limited
//   signal one sample apart, plus independent noise, 500 trials of 4 s through the Meyer alpha band and crossSpectrum: the spread of the estimated lag
//   equals se(θ) with n = independentEstimates (B·T), not with 2BT; the coherence's debiasing; holonomy 0 for lags that come from one phase map
//   (vortices included), ±δ on the two faces of a perturbed edge; pairs at a distance (straight and along the edges); a circular shift keeps each
//   channel's spectrum and destroys the lag between channels; refusals.
#include "nxr/signal/wavelet.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <random>
#include <set>

#include "mesh_helpers.h"
#include "nxr/signal/fourier.h"
#include "nxr/stats/lag.h"
#include "nxr/select/distance_pairs.h"
#include "measurement_helpers.h"
#include "nxr/signal/timefilterbank.h"

using C = std::complex<double>;
namespace tb = nxr::signal::timefilterbank;

int main() {
  std::cout << "test_lag\n";
  std::mt19937 rng(17);
  std::normal_distribution<double> nd;

  // THE EQUIVALENT BANDWIDTH: a rectangular band [8, 12] Hz is 4 Hz wide
  {
    tb::TimeFilterbank b{tb::Family::Meyer, 1.0, tb::Sides::Analytic, tb::EdgeBins::Single};
    b.fs = 200;
    tb::Member m;
    m.gain = [](double f) { return std::abs(f) >= 8 && std::abs(f) <= 12 ? 1.0 : 0.0; };
    b.members.push_back(m);
    const double B = tb::equivalentBandwidth(b, 0, 1 << 18);
    EXPECT(std::abs(B - 4.0) < 1e-2 && std::abs(tb::independentEstimates(b, 0, 800) - 16.0) < 0.05, "a rectangular [8, 12] Hz band: B = " << B << " Hz; n over 4 s = B·T = 16");
  }

  // THE LAG'S STANDARD ERROR against a simulation
  {
    const double fs = 200;
    const Eigen::Index n = 800, pad = 600;                        // 4 s, and margins the band's response fits in
    tb::Design meyer(tb::Family::Meyer, 2.0, tb::Sides::Analytic);
    meyer.bands = {{8.0, 12.0}};
    const auto bank = tb::timeBank(meyer, tb::gridLength(n + 2 * pad, tb::Padding::FastLen), fs);
    const double nEst = tb::independentEstimates(bank, 0, n), B = tb::equivalentBandwidth(bank, 0);
    std::vector<double> lags;
    double seSum = 0, gSum = 0;
    const int trials = 500;
    for (int k = 0; k < trials; ++k) {
      Eigen::MatrixXd x(2, n + 2 * pad + 1);
      for (Eigen::Index t = 0; t < x.cols(); ++t) x(0, t) = nd(rng);
      Eigen::MatrixXd y(2, n + 2 * pad);
      for (Eigen::Index t = 0; t < y.cols(); ++t) { y(0, t) = x(0, t + 1) + 0.8 * nd(rng); y(1, t) = x(0, t) + 0.8 * nd(rng); }   // channel 1 one sample behind
      const auto tr = nxr::signal::wtBlock(meyer, y, fs, static_cast<std::size_t>(pad), static_cast<std::size_t>(n));
      const auto cs = crossSpectrumTiles(tr.own[0], static_cast<std::size_t>(n), 1);
      const auto& S = cs.sum[0];
      const auto st = nxr::stats::lagStatistics({S(0, 1)}, Eigen::VectorXd::Constant(1, S(0, 0).real()), Eigen::VectorXd::Constant(1, S(1, 1).real()), nEst);
      lags.push_back(st.lag(0));
      seSum += st.stderror(0);
      gSum += st.coherence(0);
    }
    double mean = 0;
    for (double l : lags) mean += l;
    mean /= trials;
    double var = 0;
    for (double l : lags) var += (l - mean) * (l - mean);
    const double spread = std::sqrt(var / (trials - 1)), se = seSum / trials, ratio = spread / se, expected = 2 * M_PI * 10.0 / fs;
    std::cout << "    B = " << B << " Hz, n = B·T = " << nEst << " | mean |γ| " << gSum / trials << " | mean lag " << mean << " rad (one sample at ~10 Hz: " << expected
              << ") | spread of the lag " << spread << " rad, se with n = BT " << se << ", with n = 2BT " << se / std::sqrt(2.0) << "\n";
    EXPECT(std::abs(ratio - 1) < 0.2, "the spread of the estimated lag == se(θ) with n = B·T (ratio " << ratio << "; with 2BT it would be " << ratio * std::sqrt(2.0) << ")");
    EXPECT(std::abs(mean - expected) < 0.1, "the lag's mean is the delay's phase at the band's centre");
  }

  // lagStatistics: debiasing, no coherence, refusals
  {
    const auto a = nxr::stats::lagStatistics({C(0.3, 0.1), C(0, 0)}, Eigen::Vector2d(1, 1), Eigen::Vector2d(1, 1), 10, false);
    const auto b = nxr::stats::lagStatistics({C(0.3, 0.1)}, Eigen::VectorXd::Ones(1), Eigen::VectorXd::Ones(1), 10, true);
    const double g2 = 0.1, d2 = (10 * g2 - 1) / 9;
    EXPECT(std::abs(a.coherence(0) * a.coherence(0) - g2) < 1e-12 && std::abs(b.coherence(0) - std::sqrt(std::max(d2, 0.0))) < 1e-12 && std::isinf(a.stderror(1)) && a.snr(1) == 0,
           "|γ|² = 0.1 at n = 10: debiased " << d2 << " (floored at 0); no coherence: se = ∞, |θ|/se = 0");
    bool threw = false;
    try { nxr::stats::lagStatistics({C(1, 0)}, Eigen::VectorXd::Ones(2), Eigen::VectorXd::Ones(1), 10); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "mismatched sizes refused");
  }

  // HOLONOMY on an icosphere: lags from one phase map close (0), vortices included; a perturbed edge shows on its two faces
  std::vector<double> V;
  std::vector<int32_t> F;
  icosphere(3, V, F, 0.05);
  const int nV = static_cast<int>(V.size() / 3);
  std::set<std::pair<int, int>> es;
  for (std::size_t f = 0; f < F.size(); f += 3) for (int c = 0; c < 3; ++c) { const int a = F[f + c], b = F[f + (c + 1) % 3]; es.insert({std::min(a, b), std::max(a, b)}); }
  nxr::stats::Edges E(static_cast<Eigen::Index>(es.size()), 2);
  { Eigen::Index i = 0; for (auto [a, b] : es) { E(i, 0) = a; E(i, 1) = b; ++i; } }
  nxr::stats::Faces Fm(static_cast<Eigen::Index>(F.size() / 3), 3);
  for (std::size_t i = 0; i < F.size(); ++i) Fm(static_cast<Eigen::Index>(i / 3), static_cast<Eigen::Index>(i % 3)) = F[i];
  {
    Eigen::VectorXd th(E.rows());
    for (Eigen::Index e = 0; e < E.rows(); ++e) {
      const double pa = 3 * std::atan2(V[3 * E(e, 0) + 1], V[3 * E(e, 0)]), pb = 3 * std::atan2(V[3 * E(e, 1) + 1], V[3 * E(e, 1)]);
      th(e) = std::remainder(pa - pb, 2 * M_PI);
    }
    const Eigen::VectorXd h0 = nxr::stats::holonomy(E, th, Fm);
    th(5) += 0.5;
    const Eigen::VectorXd h1 = nxr::stats::holonomy(E, th, Fm);
    int moved = 0;
    double off = 0;
    for (Eigen::Index f = 0; f < h1.size(); ++f) {
      if (std::abs(h1(f)) > 1e-9) { ++moved; off = std::max(off, std::abs(std::abs(h1(f)) - 0.5)); }
    }
    EXPECT(h0.cwiseAbs().maxCoeff() < 1e-9 && moved == 2 && off < 1e-9, "lags of one phase map (with vortices of index ±3) close on every face; a perturbed edge shows ±0.5 on its two faces");
  }

  // PAIRS AT A DISTANCE: 2 cm ± 10 % on a 5 cm sphere
  {
    Eigen::MatrixXd P(nV, 3);
    for (int v = 0; v < nV; ++v) for (int d = 0; d < 3; ++d) P(v, d) = V[3 * v + d];
    const auto s = nxr::select::pairsAtDistance(P, E, 0.02, 0.1, 4, nxr::select::Metric::Straight);
    const auto g = nxr::select::pairsAtDistance(P, E, 0.02, 0.1, 4, nxr::select::Metric::AlongEdges);
    bool ok = s.pairs.rows() > nV && g.pairs.rows() > nV;
    std::map<int, int> per;
    std::set<std::pair<int, int>> uniq;
    for (Eigen::Index i = 0; i < s.pairs.rows(); ++i) {
      const double dd = (P.row(s.pairs(i, 0)) - P.row(s.pairs(i, 1))).norm();
      ok = ok && std::abs(dd - s.distance(i)) < 1e-12 && dd >= 0.018 && dd <= 0.022 && s.pairs(i, 0) < s.pairs(i, 1) && uniq.insert({s.pairs(i, 0), s.pairs(i, 1)}).second;
      ++per[s.pairs(i, 0)];
    }
    for (auto [v, c] : per) ok = ok && c <= 4;
    for (Eigen::Index i = 0; i < g.pairs.rows(); ++i) ok = ok && g.distance(i) >= 0.018 - 1e-12 && g.distance(i) <= 0.022 + 1e-12 && g.distance(i) >= (P.row(g.pairs(i, 0)) - P.row(g.pairs(i, 1))).norm() - 1e-12;
    EXPECT(ok, "pairs at 2 cm ± 10 %: " << s.pairs.rows() << " straight, " << g.pairs.rows() << " along the edges (≥ the straight distance), at most 4 a vertex, each once");
  }

  // THE CIRCULAR SHIFT: each channel's spectrum kept, the lag destroyed
  {
    const Eigen::Index n = 4096;
    Eigen::MatrixXcd Z(2, n);
    for (Eigen::Index t = 0; t < n; ++t) { const C s(nd(rng), nd(rng)); Z(0, t) = s; Z(1, t) = s * std::polar(1.0, 0.6); }   // identical up to a fixed lag
    const Eigen::MatrixXcd Zs = nxr::stats::circularShift(Z, {0, 1234});
    auto spectrum = [&](const Eigen::MatrixXcd& X, int r) {
      std::vector<C> in(X.cols()), out(X.cols());
      for (Eigen::Index t = 0; t < X.cols(); ++t) in[static_cast<std::size_t>(t)] = X(r, t);
      nxr::signal::c2c(in.data(), out.data(), {static_cast<std::size_t>(X.cols())}, 0, true);
      Eigen::VectorXd a(X.cols());
      for (Eigen::Index k = 0; k < X.cols(); ++k) a(k) = std::abs(out[static_cast<std::size_t>(k)]);
      return a;
    };
    const auto before = crossSpectrumTiles(Z, static_cast<std::size_t>(n), 1).sum[0], after = crossSpectrumTiles(Zs, static_cast<std::size_t>(n), 1).sum[0];
    const double cohBefore = std::abs(before(0, 1)) / std::sqrt(before(0, 0).real() * before(1, 1).real()), cohAfter = std::abs(after(0, 1)) / std::sqrt(after(0, 0).real() * after(1, 1).real());
    EXPECT((spectrum(Z, 1) - spectrum(Zs, 1)).cwiseAbs().maxCoeff() < 1e-9 * spectrum(Z, 1).maxCoeff() && std::abs(cohBefore - 1) < 1e-12 && cohAfter < 0.1,
           "a circular shift keeps the channel's spectrum (|FFT| equal) and destroys the lag: coherence " << cohBefore << " → " << cohAfter);
  }

  std::cout << (g_failures ? "FAIL" : "PASS") << " test_lag (" << g_failures << " failures)\n";
  return g_failures ? 1 : 0;
}
