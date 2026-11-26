// THE PER-SUBJECT FEATURES OF MEG FINGERPRINTING against Brainstorm's release of 2018-10-12 (golden: test/golden/fingerprint_features_golden.txt, made by gen_fingerprint_features_golden.m, which runs
// that release's process_bandpass 'Compute', MATLAB's hilbert, bst_corrn and bst_psd on synthetic region signals; synthetic data only):
//   the band-pass of the 2018 release (nxr::signal::filters, TransitionRule::Brainstorm2018, the ends padded with zeros, the record's mean removed): its order and its output, to rounding;
//   the Welch PSD of the release (Scaling::Brainstorm2018: |2 X / n|^2 at every bin; 2 s, 50 %, symmetric Hamming, each segment's mean removed): to rounding;
//   the amplitude envelope correlation: the streamed form (the FIR Hilbert transformer of signal.hilbert, zeros outside the half) against the release's (an FFT Hilbert of the whole half): the envelopes
//   equal away from the half's ends, the correlation to the tolerance the ends leave (printed);
//   the spectral feature's cut (the reference's six bands at 0.5 Hz: 8 + 8 + 10 + 34 + 40 + 201 = 301 bins, each from its band's spectrum), the connectome's band stack, Pearson from a covariance.
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "golden_io.h"
#include "nxr/signal/filters.h"
#include "nxr/signal/opticalflow.h"
#include "nxr/signal/spectral.h"
#include "nxr/stats/covariance.h"
#include "nxr/stats/identification.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using Eigen::MatrixXd;
namespace nf = nxr::signal::filters;
namespace ns = nxr::signal::spectral;

static bool refused(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

// the golden's closed-form region signals (gen_fingerprint_features_golden.m)
static MatrixXd signals(double fs, int N, int R) {
  MatrixXd x = MatrixXd::Zero(R, N);
  for (int r = 0; r < R; ++r) {
    for (int k = 0; k < 16; ++k) {
      const double f = 1.7 + 3.1 * k;
      for (int i = 0; i < N; ++i) {
        const double t = i / fs;
        const double m = 1 + 0.6 * std::cos(2 * M_PI * 0.13 * (k + 1) * t + 0.7 * (r % 2) + k);
        x(r, i) += m * std::cos(2 * M_PI * f * t + 1.3 * k + 0.5 * r) / (1 + 0.2 * k);
      }
    }
    for (int i = 0; i < N; ++i) x(r, i) += 0.3 * std::cos(2 * M_PI * 0.05 * (i / fs) + r) + 2 + r;
  }
  return x;
}
// the hilbert node's transformer half length (pipelines/src/fingerprint_steps.cpp, hilbert_half_length): the transition at DC spans lowest / 2
static std::size_t halfLength(double fs, double lowest, double beta) {
  const double A = beta / 0.1102 + 8.7, dw = 2 * M_PI * (lowest / 2) / fs;
  return static_cast<std::size_t>(std::ceil((A - 7.95) / (2.285 * dw) / 2));
}
static MatrixXd every(const MatrixXd& a, int step) {
  MatrixXd o(a.rows(), (a.cols() + step - 1) / step);
  for (Eigen::Index j = 0; j < o.cols(); ++j) o.col(j) = a.col(j * step);
  return o;
}

int main() {
  std::cout << "test_fingerprint_features\n";
  Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/fingerprint_features_golden.txt");
  EXPECT(g.good(), "the golden is there");
  { std::string header; std::getline(g.f, header); std::cout << "  golden: " << header.substr(0, 120) << "\n"; }
  const double fs = g.mat("fs")(0, 0);
  const MatrixXd half = g.mat("half");
  const int first = static_cast<int>(half(0, 0)), n = static_cast<int>(half(0, 1));
  const MatrixXd bands = g.mat("bands");
  const MatrixXd x = signals(fs, 8000, 4);
  const MatrixXd xw = x.middleCols(first, n);

  std::cout << "=== the band-pass, the Hilbert envelope and the AEC (Brainstorm 2018: process_bandpass, hilbert, bst_corrn) ===\n";
  for (int b = 0; b < bands.rows(); ++b) {
    const double lo = bands(b, 0), hi = bands(b, 1);
    const std::string s = std::to_string(b + 1), what = std::to_string(static_cast<int>(lo)) + "-" + std::to_string(static_cast<int>(hi)) + " Hz";
    const double order = g.mat("order_" + s)(0, 0);
    const MatrixXd yg = g.mat("bandpassed_" + s), eg = g.mat("envelope_" + s), ag = g.mat("aec_" + s);
    nf::FilterOptions o = nf::brainstormFilterOptions(fs);
    o.kind = nf::Kind::Bandpass; o.lowerEdge = lo; o.upperEdge = hi;
    o.transitionRule = nf::TransitionRule::Brainstorm2018; o.edge = nf::Edge::Zeros;
    const nf::Design d = nf::design(o);
    EXPECT(static_cast<double>(d.b.size() - 1) == order, what + ": the order " + std::to_string(d.b.size() - 1) + " is the release's (" + std::to_string(static_cast<int>(order)) + ")");
    const MatrixXd y = nf::bandpass(xw, lo, hi, o);   // the half is the record: its mean removed, its ends padded with zeros
    const double ey = relErr(every(y, 21), yg);
    EXPECT(ey < 1e-9, what + ": the band-passed half equals the release's (" + sci(ey) + ")");
    // the streamed envelope: the FIR Hilbert transformer, zeros outside the half (what signal.filter with ends span hands signal.hilbert)
    const std::size_t L = halfLength(fs, lo, 8.0);
    MatrixXd padded = MatrixXd::Zero(y.rows(), y.cols() + 2 * static_cast<Eigen::Index>(L));
    padded.middleCols(static_cast<Eigen::Index>(L), y.cols()) = y;
    const MatrixXd env = nxr::signal::hilbert(padded, nxr::signal::HilbertMethod::Fir, L, 8.0).middleCols(static_cast<Eigen::Index>(L), y.cols()).cwiseAbs();
    const MatrixXd envd = every(env, 21);
    // the envelopes away from the ends (the release's FFT is circular over the half: within about L samples of its ends the two differ)
    const Eigen::Index m0 = static_cast<Eigen::Index>(L / 21) + 1, m1 = envd.cols() - m0;
    double inner = 0, edge = 0;
    for (Eigen::Index j = 0; j < envd.cols(); ++j)
      for (Eigen::Index r = 0; r < envd.rows(); ++r) {
        const double e = std::abs(envd(r, j) - eg(r, j)) / eg.row(r).cwiseAbs().maxCoeff();
        (j >= m0 && j < m1 ? inner : edge) = std::max(j >= m0 && j < m1 ? inner : edge, e);
      }
    const MatrixXd aec = nxr::stats::correlation(nxr::stats::cov(env));
    const double ea = (aec - ag).cwiseAbs().maxCoeff();
    std::cout << "  " << what << ": Hilbert half length " << L << " samples (" << L / fs << " s); envelope error inside " << sci(inner) << ", within L of the ends " << sci(edge)
              << "; AEC max |difference| " << sci(ea) << "\n";
    EXPECT(m1 <= m0 || inner < 2e-3, what + ": the envelope equals the release's away from the half's ends (" + sci(inner) + ")");
    EXPECT(ea < 0.05, what + ": the amplitude envelope correlation is the release's to the tolerance the half's ends leave (" + sci(ea) + ")");
  }

  std::cout << "=== the Welch PSD of the release (bst_psd of 2018-10-12: 2 s, 50 %, power) ===\n";
  {
    const MatrixXd pg = g.mat("psd"), fg = g.mat("psd_freq");
    const double nw = g.mat("psd_windows")(0, 0);
    ns::WelchOptions wo;
    wo.sfreq = fs; wo.segmentSamples = ns::segmentSamplesFor(2.0, fs); wo.overlap = 0.5; wo.window = ns::Window::Hamming; wo.periodic = false; wo.detrend = ns::Detrend::Constant;
    wo.scaling = ns::Scaling::Brainstorm2018;
    const ns::WelchResult r = ns::psd(xw, wo);
    EXPECT(static_cast<double>(r.segments) == nw, "the windows: " + std::to_string(r.segments) + " (the release's " + std::to_string(static_cast<int>(nw)) + ")");
    const double ef = relErr(Eigen::Map<const Eigen::RowVectorXd>(r.frequencies.data(), static_cast<Eigen::Index>(r.frequencies.size())), fg);
    const double ep = relErr(r.psd, pg);
    std::cout << "  frequencies " << sci(ef) << ", power " << sci(ep) << " (relative to the largest)\n";
    EXPECT(ef < 1e-12 && ep < 1e-10, "the region spectrum equals the release's");
    ns::WelchOptions bad = wo;
    EXPECT(refused([&] { ns::ComplexWelch c(1, bad); }), "the 2018 scaling is the real Welch's only (a complex series refuses it)");
  }

  std::cout << "=== the spectral feature's cut: the reference's six bands at 0.5 Hz ===\n";
  {
    const int R = 3, K = 1201;     // 0 .. 600 Hz at 0.5 Hz
    std::vector<MatrixXd> spectra;
    for (int b = 0; b < 6; ++b) {
      MatrixXd s(R, K);
      for (int r = 0; r < R; ++r) for (int k = 0; k < K; ++k) s(r, k) = 1000.0 * b + 10.0 * r + 0.5 * k;   // band b, region r, its frequency
      spectra.push_back(s);
    }
    const std::vector<std::array<double, 2>> cut = {{0, 3.5}, {4, 7.5}, {8, 12.5}, {13, 29.5}, {30, 49.5}, {50, 150}};
    const nxr::stats::SpectralFeature f = nxr::stats::spectralFeature(spectra, 0.5, 0.0, cut);
    EXPECT((f.counts == std::vector<std::int64_t>{8, 8, 10, 34, 40, 201}) && f.values.cols() == 301, "the bins: 8 + 8 + 10 + 34 + 40 + 201 = " + std::to_string(f.values.cols()));
    bool from_its_band = f.frequencies.front() == 0 && f.frequencies.back() == 150;
    std::size_t j = 0;
    for (int b = 0; b < 6; ++b)
      for (std::int64_t c = 0; c < f.counts[static_cast<std::size_t>(b)]; ++c, ++j)
        for (int r = 0; r < R; ++r) from_its_band = from_its_band && f.values(r, static_cast<Eigen::Index>(j)) == 1000.0 * b + 10.0 * r + f.frequencies[j];
    EXPECT(from_its_band, "each bin is its band's spectrum at that frequency, 0 to 150 Hz inclusive, the bands side by side");
    EXPECT(refused([&] { nxr::stats::spectralFeature({spectra[0]}, 0.5, 0.0, cut); }), "refused: a spectrum count that is not the bands'");
    EXPECT(refused([&] { nxr::stats::spectralFeature({spectra[0]}, 0.5, 0.0, {{700, 800}}); }), "refused: a band that holds no bin");
  }

  std::cout << "=== the connectome's band stack; Pearson from a covariance ===\n";
  {
    std::vector<MatrixXd> M;
    for (int b = 0; b < 2; ++b) { MatrixXd m(3, 3); for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) m(i, k) = 100 * b + 10 * i + k; M.push_back(m); }
    const std::vector<double> v = nxr::stats::connectomeFeature(M);
    bool ok = v.size() == 18;
    for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) for (int b = 0; b < 2; ++b) ok = ok && v[static_cast<std::size_t>((i * 3 + k) * 2 + b)] == 100 * b + 10 * i + k;
    EXPECT(ok, "[r x r x bands] in C order: value (i r + j) bands + b");
    const nxr::stats::FeatureLayout up = nxr::stats::FeatureLayout::UpperTriangle;
    EXPECT(nxr::stats::featureTable(Eigen::Map<const MatrixXd>(v.data(), 1, 18), {3, 3, 2}, up).cols() == 12, "read by featureTable as per band the upper triangle with its diagonal (6 + 6)");
    const MatrixXd X = every(x.middleCols(0, 2000), 1);
    const MatrixXd C = nxr::stats::cov(X), Rm = nxr::stats::correlation(C);
    MatrixXd Xc = X.colwise() - X.rowwise().mean();
    for (Eigen::Index r = 0; r < Xc.rows(); ++r) Xc.row(r) /= Xc.row(r).norm();
    EXPECT((Rm - Xc * Xc.transpose()).cwiseAbs().maxCoeff() < 1e-12, "Pearson's r from the covariance is the normalised rows' product (bst_corrn, corrcoef)");
    EXPECT(refused([&] { nxr::stats::correlation(MatrixXd::Zero(2, 2)); }), "refused: a row with no variance");
  }

  std::cout << (failures ? "FAILED: " : "all passed: ") << failures << " failure(s)\n";
  return failures ? 1 : 0;
}
