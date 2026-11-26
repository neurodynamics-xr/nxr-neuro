// The Gabor bursts' compute functions against the reference detector the spec ports (golden/gabor_golden.txt, written by golden/gen_gabor_golden.py on a SYNTHETIC signal): the grid
// (gaborGrid = widths + channel_atoms' frequencies), each of the reference's candidates fitted (gaborFit = refine: f, σ, amplitude, phase, |W|, cycles), the burst test (gaborWidthPeak and the cycles
// = is_burst), the merge (suppress), and the tangent onset and offset of each merged burst (the envelope as the `edges` rule reads it, then tangentEdges = tangent_edges).
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "golden_io.h"
#include "nxr/select/edges.h"
#include "nxr/select/peaks.h"
#include "nxr/signal/gabor.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
namespace tb = nxr::signal::timefilterbank;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static constexpr double kPi = 3.14159265358979323846;

int main() {
  std::cout << "test_gabor_golden\n";
  Golden g(kDir + "/gabor_golden.txt");
  EXPECT(g.good(), "the reference's golden opens");
  g.tok(); const double fs = g.num();
  g.tok(); const double fmin = g.num(), fmax = g.num();
  g.tok(); const double smin = g.num(), smax = g.num(); const int perOct = static_cast<int>(g.num());
  g.tok(); const double minFSigma = g.num();
  g.tok(); const double mincyc = g.num();
  const Eigen::VectorXd x = g.mat("x").row(0).transpose();
  const Eigen::VectorXd sig = g.mat("sig").row(0).transpose();
  // ── the grid ──
  const tb::GaborGrid grid = tb::gaborGrid(smin, smax, perOct, fmin, fmax, minFSigma);
  double worst = 0;
  bool same = static_cast<Eigen::Index>(grid.widths.size()) == sig.size();
  for (Eigen::Index k = 0; same && k < sig.size(); ++k) worst = std::max(worst, std::abs(grid.widths[static_cast<std::size_t>(k)] - sig(k)));
  g.tok(); const int W = static_cast<int>(g.num());
  std::size_t a = 0;
  for (int k = 0; k < W; ++k) {
    g.tok(); const double s = g.num();
    const Eigen::MatrixXd fr = g.mat("freqs");
    for (Eigen::Index i = 0; i < fr.cols(); ++i, ++a) {
      same = same && a < grid.atoms.size();
      if (!same) break;
      worst = std::max({worst, std::abs(grid.atoms[a].first - s), std::abs(grid.atoms[a].second - fr(0, i))});
    }
  }
  same = same && a == grid.atoms.size();
  EXPECT(same && worst < 1e-12, "the grid: " + std::to_string(grid.widths.size()) + " widths, " + std::to_string(grid.atoms.size()) + " atoms, as the reference's (worst " + sci(worst) + ")");
  // ── each candidate fitted, and its burst test ──
  g.tok(); const int P = static_cast<int>(g.num());
  struct Ref { double t, f, s; nxr::signal::GaborAtom fit; bool burst; };
  std::vector<Ref> refs;
  double wf = 0, ws = 0, wa = 0, wp = 0, ww = 0, wc = 0;
  int burstsAgree = 0;
  const double step = std::pow(2.0, 1.0 / perOct);
  for (int p = 0; p < P; ++p) {
    g.tok();
    Ref r;
    r.t = g.num(); r.f = g.num(); r.s = g.num();
    g.tok();
    const double f = g.num(), s = g.num(), amp = g.num(), phase = g.num(), Wabs = g.num(), cycles = g.num();
    g.tok(); r.burst = g.num() != 0;
    r.fit = nxr::signal::gaborFit(x, fs, r.t, r.f, r.s);
    wf = std::max(wf, std::abs(r.fit.f - f)); ws = std::max(ws, std::abs(r.fit.sigma / s - 1)); wa = std::max(wa, std::abs(r.fit.amplitude / amp - 1));
    wp = std::max(wp, std::abs(std::arg(std::polar(1.0, r.fit.phase - phase)))); ww = std::max(ww, std::abs(r.fit.coefficient / Wabs - 1)); wc = std::max(wc, std::abs(r.fit.cycles - cycles));
    const bool burst = r.fit.f >= fmin && r.fit.f <= fmax && r.fit.cycles >= mincyc && nxr::signal::gaborWidthPeak(x, fs, r.fit, sig(0), sig(sig.size() - 1), step);
    burstsAgree += burst == r.burst;
    refs.push_back(r);
  }
  EXPECT(P > 0 && wf < 1e-9 && ws < 1e-9 && wa < 1e-9 && wp < 1e-9 && ww < 1e-9 && wc < 1e-9,
         std::to_string(P) + " candidates fitted as the reference's refine: f " + sci(wf) + " Hz, σ " + sci(ws) + ", amplitude " + sci(wa) + ", phase " + sci(wp) + " rad, |W| " + sci(ww) + ", cycles " + sci(wc));
  EXPECT(burstsAgree == P, "the burst test (in the band, the cycles, a peak along width) as the reference's is_burst for every candidate (" + std::to_string(burstsAgree) + " of " + std::to_string(P) + ")");
  // ── the merge ──
  std::vector<nxr::select::PointEvent> pe;
  std::vector<const Ref*> passing;
  for (const auto& r : refs) if (r.burst) { passing.push_back(&r); pe.push_back({r.t, r.fit.f, r.fit.sigma, 1 / (2 * kPi * r.fit.sigma), r.fit.coefficient, 0}); }
  const std::vector<std::size_t> kept = nxr::select::suppress(pe);
  const Eigen::MatrixXd merged = g.mat("merged");
  bool mergeSame = static_cast<Eigen::Index>(kept.size()) == merged.cols();
  for (std::size_t k = 0; mergeSame && k < kept.size(); ++k) mergeSame = static_cast<double>(kept[k]) == merged(0, static_cast<Eigen::Index>(k));
  EXPECT(mergeSame, "the merge: " + std::to_string(kept.size()) + " kept of " + std::to_string(pe.size()) + ", the same atoms in the same order as the reference's");
  // ── the tangent edges of each merged burst, the envelope read as the `edges` rule reads it ──
  double we = 0;
  int edgesSame = 0;
  for (std::size_t k = 0; k < kept.size(); ++k) {
    g.tok(); const double on = g.num(), off = g.num();
    const nxr::signal::GaborAtom& b = passing[kept[k]]->fit;
    const std::int64_t T = x.size(), ic = static_cast<std::int64_t>(std::llround(b.t0 * fs)), h = static_cast<std::int64_t>(std::max(3.0, 6 * b.sigma) * fs);
    const std::int64_t i0 = std::max<std::int64_t>(0, ic - h), i1 = std::min<std::int64_t>(T, ic + h);
    const double sr = std::max(b.sigma / 2, 3 / (2 * kPi * b.f));
    tb::Design d(tb::Family::Morlet, 2.0, tb::Sides::Analytic);
    d.centres = {b.f}; d.cycles = 2 * kPi * b.f * sr; d.edgeBins = tb::EdgeBins::Drop;
    const Eigen::MatrixXd win = x.segment(i0, i1 - i0);
    const tb::TimeFilterbank bank = tb::timeBank(d, tb::gridLength(static_cast<std::size_t>(i1 - i0), tb::Padding::FastLen), fs);
    const Eigen::VectorXd env = nxr::signal::wt(bank, win)[0].col(0).head(i1 - i0).cwiseAbs();
    const nxr::select::TangentEdges e = nxr::select::tangentEdges(env, ic - i0, std::max<std::int64_t>(1, static_cast<std::int64_t>(b.fwhm / 2 * fs)), static_cast<std::int64_t>(fs));
    if (std::isnan(on)) { edgesSame += !e.ok; continue; }
    if (!e.ok) continue;
    ++edgesSame;
    we = std::max({we, std::abs((e.onset + static_cast<double>(i0)) / fs - on), std::abs((e.offset + static_cast<double>(i0)) / fs - off)});
  }
  EXPECT(edgesSame == static_cast<int>(kept.size()) && we < 1e-6, "the tangent onset and offset of every merged burst as the reference's (worst " + sci(we) + " s)");
  std::cout << (failures ? "FAIL" : "PASS") << " test_gabor_golden (" << failures << " failures)\n";
  return failures ? 1 : 0;
}
