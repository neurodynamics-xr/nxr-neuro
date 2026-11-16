// nxr::surrogate (surrogate.h; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10). SYNTHETIC ONLY. What is shown:
//   - the counter-based normals are standard (mean 0, variance 1) and a block does not depend on how the samples are cut: lagfreeBlock over [0, n) equals the blocks of any cut, to the last bit
//   - the surrogate has the data's SPECTRUM (a peaked spectrum: the periodogram's peak and shape, never flat) and its zero-lag COVARIANCE (the real covariance the mixing was made from)
//   - the surrogate is LAG-FREE: its cross-spectrum's imaginary part is zero within its sampling error, where the same noise with one channel delayed has a large one (the test has teeth)
//   - the scale is pointwise: a doubled scale in a stretch doubles the samples there and nowhere else
//   - the series interpolation, the effective bandwidth (a flat spectrum of width B gives B), the Wishart draw (mean Re Q0, no lag, its degrees of freedom), the sequential p and Benjamini-Hochberg; refusals
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>

#include "nxr/surrogate.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
template <class F> bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
// the Welch cross-spectrum of two rows: segments of `nseg` samples (Hann), mean of conj(Xa) Xb over segments, at bin k
struct Cross { std::vector<std::complex<double>> s01; std::vector<double> p0, p1; };
Cross welch(const Eigen::MatrixXd& x, int nseg) {
  const int nb = nseg / 2 + 1;
  Cross c; c.s01.assign(nb, 0); c.p0.assign(nb, 0); c.p1.assign(nb, 0);
  int count = 0;
  for (Eigen::Index s = 0; s + nseg <= x.cols(); s += nseg / 2, ++count)
    for (int k = 0; k < nb; ++k) {
      std::complex<double> a = 0, b = 0;
      for (int t = 0; t < nseg; ++t) {
        const double w = 0.5 - 0.5 * std::cos(6.283185307179586 * t / nseg), ph = -6.283185307179586 * k * t / nseg;
        const std::complex<double> e(std::cos(ph), std::sin(ph));
        a += w * x(0, s + t) * e; b += w * x(1, s + t) * e;
      }
      c.s01[k] += std::conj(a) * b; c.p0[k] += std::norm(a); c.p1[k] += std::norm(b);
    }
  for (int k = 0; k < nb; ++k) { c.s01[k] /= count; c.p0[k] /= count; c.p1[k] /= count; }
  return c;
}
}  // namespace

int main() {
  // THE NORMALS
  { const Eigen::MatrixXd w = nxr::surrogate::whiteBlock(7, 3, 0, 60000);
    double m = w.mean(), v = (w.array() - m).square().mean();
    std::cout << "  normals: mean " << m << ", variance " << v << "\n";
    EXPECT(std::abs(m) < 0.01 && std::abs(v - 1) < 0.02, "the counter-based normals are standard");
    EXPECT((nxr::surrogate::whiteBlock(7, 2, -5, 8).col(0) - nxr::surrogate::whiteBlock(7, 2, -7, 8).col(2)).norm() == 0, "a sample's value does not depend on the block it is asked in");
    EXPECT((nxr::surrogate::whiteBlock(7, 1, 0, 100) - nxr::surrogate::whiteBlock(8, 1, 0, 100)).norm() > 1, "another seed, another draw"); }

  // A PEAKED SPECTRUM and a covariance
  const double fs = 100.0;
  Eigen::VectorXd freq(201), psd(201);
  for (int i = 0; i < 201; ++i) { freq[i] = 0.25 * i; psd[i] = 1.0 / (1.0 + std::pow((freq[i] - 10.0) / 1.5, 2)) + 1e-4; }   // a peak at 10 Hz, 3 Hz wide, up to 50 Hz (Nyquist)
  const int half = 200;
  const Eigen::VectorXd taps = nxr::surrogate::shapingTaps(freq, psd, fs, half);
  EXPECT(taps.size() == 2 * half + 1 && std::abs(taps.squaredNorm() - 1) < 1e-12 && (taps - taps.reverse()).norm() < 1e-12, "the taps are symmetric with unit energy");
  Eigen::MatrixXd R(2, 2); R << 1.0, 0.6, 0.6, 1.0;
  const Eigen::MatrixXd Rh = nxr::surrogate::sqrtPsd(R);
  EXPECT((Rh * Rh - R).norm() < 1e-12, "sqrtPsd squares back");

  // CHUNK INDEPENDENCE
  { const std::int64_t n = 3000;
    const Eigen::MatrixXd whole = nxr::surrogate::lagfreeBlock(Rh, taps, 11, 0, n);
    Eigen::MatrixXd cut(2, n);
    std::int64_t at = 0;
    for (const std::int64_t len : {1000, 1, 777, 1222}) { cut.middleCols(at, len) = nxr::surrogate::lagfreeBlock(Rh, taps, 11, at, len); at += len; }
    std::cout << "  the surrogate cut in four unequal chunks differs from the whole by " << (whole - cut).cwiseAbs().maxCoeff() << "\n";
    EXPECT(at == n && (whole - cut).cwiseAbs().maxCoeff() == 0, "the surrogate does not depend on the chunk size"); }

  // THE SPECTRUM, THE COVARIANCE, NO LAG
  const std::int64_t N = 1 << 17;
  const Eigen::MatrixXd x = nxr::surrogate::lagfreeBlock(Rh, taps, 3, 0, N);
  { const Eigen::MatrixXd cov = x * x.transpose() / static_cast<double>(N);
    std::cout << "  covariance of the draw: " << cov(0, 0) << " " << cov(0, 1) << " " << cov(1, 1) << " (made from 1 0.6 1)\n";
    EXPECT((cov - R).cwiseAbs().maxCoeff() < 0.05, "the draw has the covariance the mixing was made from");
    const Cross c = welch(x, 512);
    // the periodogram's peak and shape against the target, between 2 and 30 Hz
    int pk = 0;
    for (int k = 1; k < 257; ++k) if (c.p0[k] > c.p0[pk]) pk = k;
    const double fpk = pk * fs / 512;
    double num = 0, da = 0, db = 0;
    for (int k = 0; k < 257; ++k) {
      const double f = k * fs / 512; if (f < 2 || f > 30) continue;
      // the target at f, linear in the stored bins
      const int i = std::min(static_cast<int>(f / 0.25), 199); const double t = f / 0.25 - i, tg = (1 - t) * psd[i] + t * psd[i + 1];
      num += std::sqrt(tg * c.p0[k]); da += tg; db += c.p0[k];
    }
    const double corr = num / std::sqrt(da * db);
    std::cout << "  periodogram peak " << fpk << " Hz (10), shape agreement (Bhattacharyya) " << corr << "\n";
    EXPECT(std::abs(fpk - 10.0) < 0.8 && corr > 0.97, "the draw has the data's spectrum, peaked, not flat");
    // LAG-FREE: the band's (6-14 Hz) cross-spectrum's imaginary part against its real part
    std::complex<double> s = 0; double p = 0;
    for (int k = 0; k < 257; ++k) { const double f = k * fs / 512; if (f < 6 || f > 14) continue; s += c.s01[k]; p += std::sqrt(c.p0[k] * c.p1[k]); }
    std::cout << "  cross-spectrum in 6-14 Hz: Re " << s.real() / p << ", Im " << s.imag() / p << " (of sqrt(P0 P1))\n";
    EXPECT(s.real() / p > 0.5 && std::abs(s.imag()) / p < 0.03, "no lag: the imaginary cross-spectrum is zero within its sampling error");
    // the same noise with channel 1 DELAYED by 2 samples (20 ms: 72 degrees at 10 Hz): a large imaginary part
    Eigen::MatrixXd y = x; y.row(1).tail(N - 2) = x.row(1).head(N - 2);
    const Cross d = welch(y, 512);
    std::complex<double> sd = 0; double pd = 0;
    for (int k = 0; k < 257; ++k) { const double f = k * fs / 512; if (f < 6 || f > 14) continue; sd += d.s01[k]; pd += std::sqrt(d.p0[k] * d.p1[k]); }
    std::cout << "  with a 20 ms lag: Im " << sd.imag() / pd << "\n";
    EXPECT(std::abs(sd.imag()) / pd > 0.15, "the test sees a lag"); }

  // PARITY WITH nxr's lagfree_real AND envelope_matched_null on the golden recording (test/golden/gen_surrogate_golden.py; 3 channels, 4096 samples at 100 Hz, a 10 Hz source with lags of 0, 2 and 4 samples and a 0.4 Hz modulation):
  // the same statistics of 40 draws of each, from the data's own spectrum and covariance
  { std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/surrogate_golden.txt");
    EXPECT(static_cast<bool>(in), "the golden is there");
    if (in) {
      int C = 0, T = 0; in >> C >> T;
      Eigen::MatrixXd X(C, T);
      for (int c = 0; c < C; ++c) for (int t = 0; t < T; ++t) in >> X(c, t);
      std::string key; double gs = 0, gi = 0, gr = 0, gec = 0, ger = 0; std::vector<double> gcov(6);
      while (in >> key) {
        if (key == "spec") in >> gs; else if (key == "imre") in >> gi >> gr; else if (key == "cov") for (auto& v : gcov) in >> v;
        else if (key == "env") in >> gec >> ger; else { std::string rest; std::getline(in, rest); }
      }
      // the data's spectrum (mean over channels, Welch 256), covariance and one-cycle RMS envelope, and the draws' own statistics
      const int seg = 256, nb = seg / 2 + 1;
      auto pspec = [&](const Eigen::MatrixXd& x) -> Eigen::VectorXd { Eigen::VectorXd P = Eigen::VectorXd::Zero(nb); int cnt = 0;
        for (Eigen::Index s0 = 0; s0 + seg <= x.cols(); s0 += seg / 2, ++cnt) for (int k = 0; k < nb; ++k) for (Eigen::Index r = 0; r < x.rows(); ++r) {
          std::complex<double> a = 0; for (int t = 0; t < seg; ++t) { const double w = 0.5 - 0.5 * std::cos(6.283185307179586 * t / seg), ph = -6.283185307179586 * k * t / seg; a += w * x(r, s0 + t) * std::complex<double>(std::cos(ph), std::sin(ph)); }
          P[k] += std::norm(a); }
        return P / (cnt * static_cast<double>(x.rows())); };
      const Eigen::MatrixXd Xc = X.colwise() - X.rowwise().mean();
      const Eigen::MatrixXd cov = Xc * Xc.transpose() / (T - 1);
      const Eigen::VectorXd Pd = pspec(X);
      Eigen::VectorXd f(nb); for (int k = 0; k < nb; ++k) f[k] = k * 100.0 / seg;
      const Eigen::VectorXd tp = nxr::surrogate::shapingTaps(f, Pd, 100.0, 200);
      const Eigen::MatrixXd Rh = nxr::surrogate::sqrtPsd(cov);
      auto oneCycle = [&](const Eigen::MatrixXd& x) -> Eigen::VectorXd { Eigen::VectorXd v(x.cols()), o(x.cols()); for (Eigen::Index t = 0; t < x.cols(); ++t) v[t] = x.col(t).squaredNorm() / x.rows();
        for (Eigen::Index t = 0; t < x.cols(); ++t) { double a = 0; for (Eigen::Index j = t - 5; j <= t + 4; ++j) if (j >= 0 && j < x.cols()) a += v[j]; o[t] = std::sqrt(a / 10); } return o; };
      const Eigen::VectorXd ed = oneCycle(X);
      const Eigen::VectorXd scale = ed / std::sqrt(cov.trace() / C);
      double spec = 0, im = 0, re = 0, ec = 0, er = 0; Eigen::MatrixXd cv = Eigen::MatrixXd::Zero(C, C);
      const int nd = 40;
      for (int d = 0; d < nd; ++d) {
        const Eigen::MatrixXd Y = nxr::surrogate::lagfreeBlock(Rh, tp, 100 + d, 0, T), E = nxr::surrogate::lagfreeBlock(Rh, tp, 200 + d, 0, T, scale);
        const Eigen::VectorXd Py = pspec(Y);
        { double a = 0, b = 0, c = 0; for (int k = 0; k < nb; ++k) { const double fk = k * 100.0 / seg; if (fk < 2 || fk > 30) continue; a += std::sqrt(Pd[k] * Py[k]); b += Pd[k]; c += Py[k]; } spec += a / std::sqrt(b * c) / nd; }
        { const Cross cr = welch(Y.topRows(2), seg); std::complex<double> sx = 0; double pp = 0; for (int k = 0; k < nb; ++k) { const double fk = k * 100.0 / seg; if (fk < 6 || fk > 14) continue; sx += cr.s01[k]; pp += std::sqrt(cr.p0[k] * cr.p1[k]); } im += sx.imag() / pp / nd; re += sx.real() / pp / nd; }
        { const Eigen::MatrixXd Yc = Y.colwise() - Y.rowwise().mean(); cv += Yc * Yc.transpose() / (T - 1) / nd; }
        { const Eigen::VectorXd en = oneCycle(E); const double me = en.mean(), md = ed.mean(); double sxy = 0, sxx = 0, syy = 0; for (Eigen::Index t = 0; t < T; ++t) { sxy += (en[t] - me) * (ed[t] - md); sxx += (en[t] - me) * (en[t] - me); syy += (ed[t] - md) * (ed[t] - md); } ec += sxy / std::sqrt(sxx * syy) / nd; er += me / md / nd; }
      }
      const double pairs[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}};
      double dcov = 0; for (int k = 0; k < 6; ++k) dcov = std::max(dcov, std::abs(cv(static_cast<int>(pairs[k][0]), static_cast<int>(pairs[k][1])) - gcov[static_cast<std::size_t>(k)]) / gcov[0]);
      std::cout << "  against nxr (mean of 40 draws; mine / nxr): spectrum agreement " << spec << " / " << gs << ", Im " << im << " / " << gi << ", Re " << re << " / " << gr << ", covariance differs by " << dcov << " of the variance, envelope correlation " << ec << " / " << gec << ", level ratio " << er << " / " << ger << "\n";
      EXPECT(std::abs(spec - gs) < 0.01 && spec > 0.97, "the draw's spectrum agrees with the data's as nxr's does");
      EXPECT(std::abs(im - gi) < 0.03 && std::abs(im) < 0.03 && std::abs(re - gr) < 0.06, "no lag, the zero-lag cross-spectrum kept: as nxr's");
      EXPECT(dcov < 0.12, "the covariance is the data's real covariance, as nxr's");
      // nxr rescales each draw to the data's RECALISED envelope (a reduction over channels and a cycle: correlation 0.98 by construction); here the scale is pointwise (the cardinal rule), so a draw follows the data's envelope up to its own sampling variation
      // (3 channels, one cycle = 10 samples: the largest it can be); the LEVEL is nxr's
      EXPECT(ec > 0.75 && std::abs(er - ger) < 0.06, "the draw follows the data's one-cycle envelope (its level as nxr's envelope-matched null; the correlation, a pointwise scale: lower by design)");
    } }

  // THE SCALE IS POINTWISE
  { Eigen::VectorXd sc = Eigen::VectorXd::Ones(1000); sc.segment(300, 200).setConstant(2.0);
    const Eigen::MatrixXd a = nxr::surrogate::lagfreeBlock(Rh, taps, 5, 0, 1000), b = nxr::surrogate::lagfreeBlock(Rh, taps, 5, 0, 1000, sc);
    Eigen::MatrixXd diff = b - a; diff.middleCols(300, 200) -= a.middleCols(300, 200);
    EXPECT(diff.cwiseAbs().maxCoeff() < 1e-12, "a scale multiplies its own samples only"); }

  // INTERPOLATION, BANDWIDTH
  { const std::vector<double> v = {1, 3, 2};
    const Eigen::VectorXd y = nxr::surrogate::interpolateSeries(v, 10, -5, 30);
    EXPECT(y[0] == 1 && std::abs(y[5] - 1) < 1e-12 && std::abs(y[10] - 2) < 1e-12 && std::abs(y[15] - 3) < 1e-12 && std::abs(y[25] - 2) < 1e-12 && y[29] == 2, "the series is linear between its values, held beyond them");
    Eigen::VectorXd f(1001), flat = Eigen::VectorXd::Zero(1001);
    for (int i = 0; i < 1001; ++i) { f[i] = 0.1 * i; if (f[i] >= 20 && f[i] < 28) flat[i] = 3.0; }
    const double B = nxr::surrogate::effectiveBandwidth(f, flat);
    std::cout << "  effective bandwidth of a flat 8 Hz band: " << B << "\n";
    EXPECT(std::abs(B - 8.0) < 0.2, "a flat spectrum of width B has effective bandwidth B");
    EXPECT(nxr::surrogate::effectiveBandwidth(f, psd.size() == f.size() ? psd : flat) > 0, "effective bandwidth is positive"); }

  // THE WISHART DRAW
  { Eigen::MatrixXd Q(3, 3); Q << 2, 0.5, 0.2, 0.5, 1, -0.3, 0.2, -0.3, 1.5;
    const int nd = 400; const double dof = 20;
    Eigen::MatrixXcd mean = Eigen::MatrixXcd::Zero(3, 3);
    for (int d = 0; d < nd; ++d) mean += nxr::surrogate::wishartDraw(Q, dof, 9, d) / nd;
    std::cout << "  Wishart mean: Re error " << (mean.real() - Q).cwiseAbs().maxCoeff() << ", Im " << mean.imag().cwiseAbs().maxCoeff() << "\n";
    EXPECT((mean.real() - Q).cwiseAbs().maxCoeff() < 0.08 && mean.imag().cwiseAbs().maxCoeff() < 0.08, "the Wishart draws have mean Re Q and no imaginary part");
    // fewer degrees of freedom: more spread
    double v20 = 0, v200 = 0;
    for (int d = 0; d < 200; ++d) { v20 += std::norm(nxr::surrogate::wishartDraw(Q, 20, 4, d)(0, 1) - Q(0, 1)); v200 += std::norm(nxr::surrogate::wishartDraw(Q, 200, 4, d)(0, 1) - Q(0, 1)); }
    EXPECT(v20 > 4 * v200, "the spread falls with the degrees of freedom (about 1/dof)");
    EXPECT((nxr::surrogate::wishartDraw(Q, 20, 4, 1) - nxr::surrogate::wishartDraw(Q, 20, 4, 1)).norm() == 0 && (nxr::surrogate::wishartDraw(Q, 20, 4, 1) - nxr::surrogate::wishartDraw(Q, 20, 4, 2)).norm() > 0, "seeded: one draw is one draw"); }

  // THE SEQUENTIAL P, BH
  { EXPECT(std::abs(nxr::surrogate::besagCliffordP(10, 40, 10) - 0.25) < 1e-12 && std::abs(nxr::surrogate::besagCliffordP(0, 99, 10) - 0.01) < 1e-12 && std::abs(nxr::surrogate::besagCliffordP(3, 19, 10) - 0.2) < 1e-12, "Besag-Clifford: h/n when stopped, (e+1)/(n+1) otherwise");
    Eigen::VectorXd p(5); p << 0.01, 0.04, 0.03, std::nan(""), 0.5;
    const Eigen::VectorXd q = nxr::surrogate::benjaminiHochberg(p);
    // sorted 0.01 0.03 0.04 0.5 (m = 4): raw 0.04 0.06 0.0533 0.5; the running minimum from the top: 0.04 0.0533 0.0533 0.5
    EXPECT(std::abs(q[0] - 0.04) < 1e-12 && std::abs(q[2] - 0.16 / 3) < 1e-12 && std::abs(q[1] - 0.16 / 3) < 1e-12 && std::abs(q[4] - 0.5) < 1e-12 && std::isnan(q[3]), "BH q-values (NaN left out)"); }

  // REFUSALS
  EXPECT(throws([&] { nxr::surrogate::shapingTaps(freq, Eigen::VectorXd::Zero(201), fs, 10); }), "a spectrum with no power is refused");
  EXPECT(throws([&] { nxr::surrogate::lagfreeBlock(Rh, taps, 1, 0, 10, Eigen::VectorXd::Ones(9)); }), "a scale of the wrong length is refused");
  EXPECT(throws([&] { nxr::surrogate::besagCliffordP(5, 3, 10); }), "more exceedances than draws is refused");
  EXPECT(throws([&] { nxr::surrogate::filterValid(Eigen::MatrixXd::Zero(1, 5), Eigen::VectorXd::Ones(7)); }), "a block shorter than the taps is refused");

  if (fails) { std::cout << fails << " failures\n"; return 1; }
  std::cout << "OK test_surrogate\n";
  return 0;
}
