// Gabor bursts (spec docs/specs/2026-11-20-gabor-bursts.md) on synthetic signals only: the time filter bank's Gabor family (its grid, the unit-energy normalisation, the amplitude of a matched
// Gaussian burst A = 2|W| / (π^(1/4) √σ)), one atom fitted to a planted burst (t0, f, σ, A, φ recovered within their resolution), two bursts at the same frequency told apart by their widths, the
// width test, the fit's R²; and the selection makers' maths: local maxima on a grid, suppression within each event's own resolution, the tangent edges of an envelope, groups by overlap.
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "nxr/select/edges.h"
#include "nxr/select/events.h"
#include "nxr/select/peaks.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/gabor.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
namespace tb = nxr::signal::timefilterbank;
namespace sg = nxr::signal;
namespace sl = nxr::select;
static constexpr double kPi = 3.14159265358979323846;

namespace {
template <class F> bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
// A e^(−(t−t0)²/2s²) cos(2πf(t−t0) + φ) added to x (sample i at i/fs)
void plant(Eigen::VectorXd& x, double fs, double t0, double f, double s, double A, double phi) {
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const double t = static_cast<double>(i) / fs - t0;
    x(i) += A * std::exp(-t * t / (2 * s * s)) * std::cos(2 * kPi * f * t + phi);
  }
}
Eigen::VectorXd noise(Eigen::Index n, double sd, unsigned seed) {
  std::mt19937 r(seed);
  std::normal_distribution<double> N(0, sd);
  Eigen::VectorXd x(n);
  for (auto& v : x) v = N(r);
  return x;
}
// the grid atom nearest (s, f): the width nearest in log, then the frequency nearest at it
std::pair<double, double> nearest(const tb::GaborGrid& g, double s, double f) {
  std::size_t best = 0;
  double score = 1e300;
  for (std::size_t a = 0; a < g.atoms.size(); ++a) {
    const double d = std::abs(std::log(g.atoms[a].first / s)) * 1e3 + std::abs(g.atoms[a].second - f);
    if (d < score) { score = d; best = a; }
  }
  return g.atoms[best];
}
}  // namespace

int main() {
  std::cout << "test_gabor\n";
  // ── the grid (the reference detector's widths and channel_atoms) ──
  {
    const tb::GaborGrid g = tb::gaborGrid(0.025, 0.64, 4, 8, 35, 0.5);
    EXPECT(g.widths.size() == 19 && std::abs(g.widths[0] - 0.025) < 1e-15 && std::abs(g.widths[18] - 0.025 * std::pow(2.0, 4.5)) < 1e-12, "19 widths, 25 ms · 2^(k/4), the last 2^4.5 times the first (at most 640 ms)");
    bool steps = true, edges = true, order = true;
    for (std::size_t a = 0; a < g.atoms.size(); ++a) {
      const double s = g.atoms[a].first, sf = 1 / (2 * kPi * s);
      const bool first = a == 0 || g.widthOf[a - 1] != g.widthOf[a];
      if (first) edges = edges && std::abs(g.atoms[a].second - std::max(8 - sf, 0.5 / s)) < 1e-12;
      else steps = steps && std::abs(g.atoms[a].second - g.atoms[a - 1].second - sf / 2) < 1e-9;
      edges = edges && g.atoms[a].second <= 35 + sf + 1e-6;
      order = order && s == g.widths[static_cast<std::size_t>(g.widthOf[a])];
    }
    EXPECT(steps && edges && order, "each width's frequencies run from max(fMin − σ_f, minFσ/σ) in steps of σ_f/2 up to fMax + σ_f, width-major");
    const tb::GaborGrid hi = tb::gaborGrid(0.01, 0.02, 4, 8, 12, 0.5);
    EXPECT(hi.widths.size() == 5 && hi.atoms.empty(), "a width whose first frequency (minFσ/σ) is above fMax has no atom");
    EXPECT(throws([] { tb::gaborGrid(0.1, 0.05, 4, 8, 35, 0.5); }) && throws([] { tb::gaborGrid(0.1, 0.2, 0, 8, 35, 0.5); }), "a width range upside down, or no width an octave, refused");
  }
  // ── the Gabor family: unit energy, one member an atom, analytic only ──
  const double fs = 250;
  {
    tb::Design d(tb::Family::Gabor, 1.0, tb::Sides::Analytic);
    d.atoms = {{0.1, 20.0}, {0.3, 12.0}};
    const std::size_t N = 1 << 14;
    const tb::TimeFilterbank b = tb::timeBank(d, N, fs);
    EXPECT(b.size() == 2 && b.normalisation == tb::Normalisation::UnitEnergy && b.members[0].width == 0.1 && b.members[0].fc == 20.0, "one member an atom, its width and frequency recorded; the bank says unit energy");
    const Eigen::MatrixXd G = tb::gains(b);
    const double df = fs / static_cast<double>(N);
    bool unit = true;
    for (int m = 0; m < 2; ++m) unit = unit && std::abs(G.col(m).squaredNorm() * df - 1) < 1e-6;
    EXPECT(unit, "unit energy: Σ_ν Ψ(ν)² Δν = 1 (Parseval: ∫|ψ|² dt = 1)");
    EXPECT(tb::familyOf("gabor") == tb::Family::Gabor, "the family by name");
    tb::Design two(tb::Family::Gabor, 1.0, tb::Sides::TwoSided);
    two.atoms = {{0.1, 20.0}};
    tb::Design none(tb::Family::Gabor, 1.0, tb::Sides::Analytic);
    EXPECT(throws([&] { tb::timeBank(two, N, fs); }) && throws([&] { tb::timeBank(none, N, fs); }), "a two-sided Gabor bank, or one with no atom, refused");
  }
  // ── the matched burst: |W| = (A/2) π^(1/4) √σ at σ = s; the bank's transform and the inner product agree ──
  {
    const double s = 0.13, f = 21.0, A = 1.7, phi = 0.7, t0 = 20.0;
    Eigen::VectorXd x = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(40 * fs));
    plant(x, fs, t0, f, s, A, phi);
    tb::Design d(tb::Family::Gabor, 1.0, tb::Sides::Analytic);
    d.atoms = {{s, f}};
    d.edgeBins = tb::EdgeBins::Drop;
    const tb::TimeFilterbank b = tb::timeBank(d, tb::gridLength(static_cast<std::size_t>(x.size()), tb::Padding::FastLen), fs);
    const tb::Coefficients W = sg::wt(b, Eigen::MatrixXd(x));
    const std::complex<double> w = W[0](static_cast<Eigen::Index>(t0 * fs), 0);
    const double expect = A / 2 * std::pow(kPi, 0.25) * std::sqrt(s);
    EXPECT(std::abs(std::abs(w) / expect - 1) < 1e-3, "the bank's coefficient at the burst's centre: |W| = (A/2) π^(1/4) √σ (A recovered as 2|W| / (π^(1/4) √σ) = " + std::to_string(2 * std::abs(w) / (std::pow(kPi, 0.25) * std::sqrt(s))) + ")");
    EXPECT(std::abs(std::arg(w) - phi) < 1e-3, "and arg W is the burst's phase");
    const std::complex<double> direct = sg::gaborCoefficient(x, fs, t0, f, s);
    EXPECT(std::abs(direct - w) < 1e-4 * std::abs(w), "the inner product over ±5σ equals the bank's transform there");
  }
  // ── each width at its own rate: wtDecimated is the full-rate transform at every D-th sample, exactly ──
  {
    Eigen::MatrixXd x(5000, 2);
    for (Eigen::Index c = 0; c < 2; ++c) x.col(c) = noise(5000, 1.0, 21 + static_cast<unsigned>(c));
    tb::Design d(tb::Family::Gabor, 1.0, tb::Sides::Analytic);
    d.atoms = {{0.2, 12.0}, {0.2, 12.8}};
    d.edgeBins = tb::EdgeBins::Drop;
    const int D = 16;
    const std::size_t N = static_cast<std::size_t>(D) * sg::next_fast_len(10000 / D + 1, true);
    const tb::TimeFilterbank b = tb::timeBank(d, N, fs);
    const tb::Coefficients full = sg::wt(b, x);
    const Eigen::MatrixXcd X = tb::spectrum(b, x);
    double worst = 0, scale = 0;
    for (int m = 0; m < 2; ++m) {
      const Eigen::MatrixXcd W = sg::wtDecimated(b, m, X, D);
      for (Eigen::Index c = 0; c < 2; ++c)
        for (Eigen::Index i = 0; i < W.cols(); ++i) { worst = std::max(worst, std::abs(W(c, i) - full[static_cast<std::size_t>(m)](i * D, c))); scale = std::max(scale, std::abs(full[static_cast<std::size_t>(m)](i * D, c))); }
    }
    EXPECT(worst < 1e-12 * scale, "the decimated transform equals the full rate at every D-th sample (" + std::to_string(worst / scale) + ")");
    EXPECT(throws([&] { sg::wtDecimated(b, 0, X, 11); }) && throws([&] { sg::wtDecimated(b, 0, X, static_cast<int>(N / 4)); }), "a decimation that does not divide the grid, or too coarse for the member's band, refused");
    const auto g = tb::morletLadderGain(8, 35, 1, 64, 8);
    EXPECT(std::abs(g(15) - 1) < 0.01 && std::abs(g(20) - 1) < 0.01 && g(4) < 1e-3 && g(60) < 1e-3, "the Morlet ladder's band gain: 1 inside the band (" + std::to_string(g(15)) + "), 0 far outside");
  }
  // ── the fit: a planted burst in noise recovered from the nearest grid atom ──
  const tb::GaborGrid grid = tb::gaborGrid(0.025, 0.64, 4, 8, 35, 0.5);
  const double step = std::pow(2.0, 0.25);
  {
    const double s = 0.13, f = 20.3, A = 1.0, phi = 0.7, t0 = 20.0;
    Eigen::VectorXd x = noise(static_cast<Eigen::Index>(40 * fs), 0.02, 3);
    plant(x, fs, t0, f, s, A, phi);
    const auto [s0, f0] = nearest(grid, s, f);
    const sg::GaborAtom a = sg::gaborFit(x, fs, t0 + 0.008, f0, s0);
    EXPECT(std::abs(a.f - f) < 1 / (2 * kPi * s) / 4 && std::abs(a.sigma / s - 1) < 0.03 && std::abs(a.amplitude - A) < 0.03 && std::abs(std::arg(std::polar(1.0, a.phase - (phi + 2 * kPi * f * 0.008)))) < 0.15,
           "the fit recovers f (" + std::to_string(a.f) + "), σ (" + std::to_string(a.sigma) + "), A (" + std::to_string(a.amplitude) + ") and φ (" + std::to_string(a.phase) + ") from the grid atom");
    EXPECT(std::abs(a.fwhm - 2.355 * a.sigma) < 1e-12 && std::abs(a.cycles - 2.355 * a.sigma * a.f) < 1e-12, "FWHM 2.355 σ and the cycles at half maximum 2.355 σ f");
    EXPECT(sg::gaborWidthPeak(x, fs, a, grid.widths.front(), grid.widths.back(), step), "the width test: |W| peaks along width at the matched σ");
    Eigen::VectorXd y = Eigen::VectorXd::Zero(x.size());
    plant(y, fs, t0, f, s, A, phi);
    const sg::GaborAtom exact = sg::gaborFit(y, fs, t0, f0, s0);
    EXPECT(sg::gaborR2(y, fs, exact) > 0.99, "R² of the atom against the clean burst's signal in its half-maximum window is near 1");
    EXPECT(sg::gaborR2(noise(x.size(), 1.0, 9), fs, exact) < 0.2, "and low against noise alone");
  }
  // ── two bursts at the same frequency, told apart by their widths ──
  {
    const double f = 18.0;
    Eigen::VectorXd x = noise(static_cast<Eigen::Index>(40 * fs), 0.02, 5);
    plant(x, fs, 10.0, f, 0.05, 1.0, 0.0);
    plant(x, fs, 25.0, f, 0.30, 1.0, 1.0);
    const auto [sa, fa] = nearest(grid, 0.05, f);
    const auto [sb, fb] = nearest(grid, 0.30, f);
    const sg::GaborAtom a = sg::gaborFit(x, fs, 10.0, fa, sa), b = sg::gaborFit(x, fs, 25.0, fb, sb);
    EXPECT(std::abs(a.sigma / 0.05 - 1) < 0.03 && std::abs(b.sigma / 0.30 - 1) < 0.03 && std::abs(a.f - f) < 0.5 && std::abs(b.f - f) < 0.1,
           "a short (σ " + std::to_string(a.sigma) + ") and a long (σ " + std::to_string(b.sigma) + ") burst at " + std::to_string(f) + " Hz each matched by its own width");
    EXPECT(std::abs(a.amplitude - 1) < 0.03 && std::abs(b.amplitude - 1) < 0.03, "both amplitudes 1");
    const sg::GaborAtom wrong = sg::gaborFit(x, fs, 25.0, fb, sa, 0);
    EXPECT(!sg::gaborWidthPeak(x, fs, wrong, grid.widths.front(), grid.widths.back(), step), "the long burst read at the short width is no peak along width");
    Eigen::VectorXd z = noise(x.size(), 0.02, 7);
    plant(z, fs, 20.0, f, 0.012, 1.0, 0.0);
    const sg::GaborAtom tiny = sg::gaborFit(z, fs, 20.0, f, grid.widths.front());
    EXPECT(!sg::gaborWidthPeak(z, fs, tiny, grid.widths.front(), grid.widths.back(), step), "a burst narrower than the ladder fails the width test (its σ is not inside the ladder)");
  }
  // ── the selection makers' maths ──
  {
    Eigen::MatrixXd v(3, 6);
    v << 1, 2, 1, 0, 5, 5,
         1, 9, 1, 0, 1, 0,
         0, 1, 0, NAN, 7, 0;
    const auto m = sl::localMaxima(v, Eigen::Vector3d(0, 0, 0));
    EXPECT(m.size() == 4 && m[0] == std::make_pair(Eigen::Index(0), Eigen::Index(4)) && m[1] == std::make_pair(Eigen::Index(0), Eigen::Index(5)) && m[2] == std::make_pair(Eigen::Index(1), Eigen::Index(1)) &&
               m[3] == std::make_pair(Eigen::Index(2), Eigen::Index(4)),
           "local maxima in the 3 x 3 box (ties count, the box clipped at the edges, NaN ignored), row-major");
    const auto lv = sl::localMaxima(v, Eigen::Vector3d(0, 10, 0));
    EXPECT(lv.size() == 3, "and at least the row's level: the 9 under its row's level 10 is dropped");
    EXPECT(throws([&] { sl::localMaxima(v, Eigen::Vector2d(0, 0)); }), "one level a row, or refused");
    std::vector<sl::PointEvent> p = {{10.0, 20, 0.1, 1.6, 1.0, 0}, {10.05, 20.5, 0.1, 1.6, 2.0, 0}, {10.05, 20.5, 0.1, 1.6, 0.5, 1}, {10.5, 20, 0.1, 1.6, 0.8, 0}, {10.02, 25, 0.1, 1.6, 0.9, 0}};
    const auto k = sl::suppress(p);
    EXPECT(k.size() == 4 && k[0] == 1 && k[1] == 4 && k[2] == 3 && k[3] == 2, "suppression: strongest first; the weaker inside its time and frequency resolution dropped; other groups and far events kept");
    // tangent edges of a Gaussian bump on a floor: the tangents meet the floor near t0 ∓ σ (√(2)·... for a Gaussian, the steepest slope is at ±σ and its tangent meets the floor at ±2σ)
    const double sig = 20;
    Eigen::VectorXd a(401);
    for (Eigen::Index i = 0; i < a.size(); ++i) a(i) = 0.1 + std::exp(-std::pow(static_cast<double>(i) - 200, 2) / (2 * sig * sig)) + 0.05 * std::exp(-std::pow(static_cast<double>(i) - 330, 2) / 200.0);
    const sl::TangentEdges e = sl::tangentEdges(a, 205, 20, 250);
    EXPECT(e.ok && e.peak == 200 && std::abs(e.onset - (200 - 2 * sig)) < 2 && std::abs(e.offset - (200 + 2 * sig)) < 2, "tangent edges of a Gaussian envelope: t0 ∓ 2σ (" + std::to_string(e.onset) + ", " + std::to_string(e.offset) + ")");
    EXPECT(e.troughAfter < 330, "the trough after the peak is the first local minimum below half the peak, not the next bump");
    EXPECT(!sl::tangentEdges(Eigen::VectorXd::LinSpaced(50, 0, 1), 49, 3, 10).ok, "no edges where the envelope has no rise and fall");
    const auto gr = sl::overlapGroups({0, 1, 5, 0.5, 6}, {2, 3, 7, 1.5, 8}, {8, 20, 8, 30, 9}, {12, 24, 12, 34, 13});
    EXPECT(gr == std::vector<std::int64_t>({0, 1, 2, 3, 2}), "groups by overlap in time AND frequency: the 0-2 s and 1-3 s events are apart in frequency");
    const auto gt = sl::overlapGroups({0, 1, 5, 0.5, 6}, {2, 3, 7, 1.5, 8});
    EXPECT(gt == std::vector<std::int64_t>({0, 0, 1, 0, 1}), "and by time alone when no frequency is given (chained)");
  }
  std::cout << (failures ? "FAIL" : "PASS") << " test_gabor (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
