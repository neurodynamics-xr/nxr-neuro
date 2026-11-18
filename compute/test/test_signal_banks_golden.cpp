// test_signal_banks_golden: the NEW transforms (wt / iwt / gwt / igwt / jwt / ijwt, the energies as measurements) give, BITWISE, what the old filterbank functions gave on the synthetic inputs of signal_banks_inputs.h
// (golden/signal_banks_golden.txt, made by golden/gen_signal_banks_golden.cpp from the code of commit dc5f877, before the refactor). Synthetic only.
#include "golden_hash.h"
#include "signal_banks_inputs.h"
#include "nxr/signal/wavelet.h"
#include "nxr/signal/graphtransforms.h"
#include "nxr/signal/graphwavelet.h"
#include "nxr/signal/jointwavelet.h"
#include "nxr/measurement/band_energy.h"
#include "nxr/signal/spectral.h"
#include "nxr/synthesis/lagfree.h"
#include <iostream>
using namespace sbi;
namespace jb = nxr::signal::jointfilterbank;
namespace nxr_sig = nxr::signal; namespace ms = nxr::measurement;
struct Checker {
    goldenhash::Reader r; int failures = 0, checked = 0;
    explicit Checker(const std::string& p) : r(p) {}
    template <class D> void put(const std::string& n, const Eigen::MatrixBase<D>& A) { ++checked; std::string why; if (!r.same(n, A, &why)) { ++failures; std::cout << "  [FAIL] " << why << "\n"; } }
    void put(const std::string& n, double v) { ++checked; std::string why; if (!r.same(n, v, &why)) { ++failures; std::cout << "  [FAIL] " << why << "\n"; } }
    template <class V> void putAll(const std::string& n, const V& l) { for (std::size_t i = 0; i < l.size(); ++i) put(n + "." + std::to_string(i), l[i]); }
};
int main() {
    Checker w(std::string(NXR_TEST_GOLDEN_DIR) + "/signal_banks_golden.txt");
    if (!w.r.good()) { std::cout << "  [FAIL] the golden does not open\n"; return 1; }
    struct B { const char* n; tb::TimeFilterbank b; };
    std::vector<B> banks = {{"itersine", itersine()}, {"meyer", meyer()}, {"morlet", morlet()}};
    const MatrixXd x = tones(200, 2, 3, 50);
    for (auto& [name, bk] : banks) {
        const std::string p = std::string("tb.") + name + ".";
        const tb::Coefficients W = nxr_sig::wt(bk, x);
        w.putAll(p + "analysis", W);
        w.put(p + "synthDual", nxr_sig::iwt(bk, W, 200, fb::Synthesis::Dual));
        if (std::string(name) != "morlet") w.put(p + "synthTight", nxr_sig::iwt(bk, W, 200, fb::Synthesis::Tight));
        w.put(p + "scalogram", ms::scalogram(W).back());   // the golden holds the last member (the generator named every member alike)
        w.put(p + "energies", ms::energies(W));
        w.put(p + "centres", tb::centerFrequencies(bk));
        { auto r = tb::rates(bk, 2.0); w.put(p + "rates", r.first); }
        w.put(p + "frame", tb::frameOperator(bk)); w.put(p + "dual", tb::dualGains(bk));
        w.put(p + "gains", tb::gains(bk)); w.put(p + "gains", tb::freqz(bk).H);
        const MatrixXcd S = tb::spectrum(bk, x); w.put(p + "spectrum", S);
        w.put(p + "filterSpectrum", tb::filterSpectrum(bk, tb::gains(bk).col(1), S));
        w.put(p + "coefSpectrum", tb::coefficientSpectrum(bk, W[1]));
        w.put(p + "inverseReal", tb::inverseReal(bk, S, 200));
        w.put(p + "eqBw", tb::equivalentBandwidth(bk, 1)); w.put(p + "noiseBw", tb::noiseBandwidth(bk, 1));
        const auto own = nxr_sig::wtOwnRate(bk, x.col(0));
        for (std::size_t m = 0; m < own.size(); ++m) { w.put(p + "own" + std::to_string(m), own[m].coef); w.put(p + "ownT" + std::to_string(m), own[m].t); }
        tb::Design d(name == std::string("itersine") ? tb::Family::Itersine : tb::Family::Meyer, 2.0, tb::Sides::Analytic); d.voices = 2; d.limits = std::make_pair(2.0, 60.0);
        if (std::string(name) != "morlet") { const auto bt = nxr_sig::wtBlock(d, x.transpose(), kFs, 20, 150); w.putAll(p + "block", bt.own); }
    }
    { const auto bk = twoSided(); MatrixXcd z = noise(100, 2, 5).cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(100, 2, 6).cast<std::complex<double>>();
      const auto W = nxr_sig::wt(bk, z); w.putAll("tb.two.analysis", W); w.put("tb.two.synth", nxr_sig::iwtComplex(bk, W, 100)); w.put("tb.two.synthTight", nxr_sig::iwtComplex(bk, W, 100, fb::Synthesis::Tight));
      w.put("tb.two.inverseComplex", tb::inverseComplex(bk, tb::spectrum(bk, z), 100)); w.put("tb.two.centres", tb::centerFrequencies(bk)); }
    w.put("tb.logCentres", Eigen::Map<const VectorXd>(tb::centerFrequencies(tb::CentreGrid::Studio, 2, 60, 3, kFs).data(), (long)tb::centerFrequencies(tb::CentreGrid::Studio, 2, 60, 3, kFs).size()));
    w.put("tb.cwtCentres", Eigen::Map<const VectorXd>(tb::centerFrequencies(tb::CentreGrid::Cwt, 2, 60, 3).data(), (long)tb::centerFrequencies(tb::CentreGrid::Cwt, 2, 60, 3).size()));
    // Wt / Iwt streaming, chunks of 37 and the whole record (the node's use: add + finish on one block)
    for (auto& [name, bk] : banks) {
        nxr_sig::WtOptions o; o.tolerance = 1e-9; o.maxMargin = 4096;
        const MatrixXd xt = tones(900, 2, 9, 50).transpose();
        const auto layout = nxr_sig::wtLayout(bk, o); const auto bs = nxr_sig::wt(layout, xt);
        const std::string p = std::string("wt.") + name + ".";
        w.put(p + "margin", (double)layout.margin);
        for (std::size_t m = 0; m < bs.size(); ++m) { w.put(p + "coef" + std::to_string(m), bs[m].coef); w.put(p + "first" + std::to_string(m), (double)bs[m].first); }
        if (std::string(name) != "morlet") w.put(p + "iwt", nxr_sig::iwt(layout, bs, 900));
    }
    // the graph bank
    const Graph g = pathGraph(); const MatrixXd F = noise(g.n, 3, 11);
    for (auto fam : {fb::Family::Itersine, fb::Family::MexHat, fb::Family::Heat, fb::Family::LogItersine}) {
        const auto bank = graphBank(g, fam); const std::string p = "gb." + std::to_string((int)fam) + ".";
        const auto W = nxr_sig::gwt(bank, g.T, F); w.putAll(p + "eigen", W);
        w.put(p + "member1", nxr_sig::gwt(bank, 1, g.T, F));
        w.put(p + "synth", nxr_sig::igwt(bank, g.T, W)); w.put(p + "synthTight", nxr_sig::igwt(bank, g.T, W, fb::Synthesis::Tight));
        w.put(p + "scaleEnergy", ms::scaleEnergy(W)); w.put(p + "memberEnergy", ms::memberEnergy(W));
        w.put(p + "memberEnergyW", ms::memberEnergy(W, g.Md.diagonal()));
        gt::ChebyshevOperator cheb(g.L, g.M, g.lam.maxCoeff() * 1.01, 40);
        const auto Wc = nxr_sig::gwt(bank, cheb, F); w.putAll(p + "cheb", Wc);
        w.put(p + "chebMember1", nxr_sig::gwt(bank, 1, cheb, F)); w.put(p + "chebSynth", nxr_sig::igwt(bank, cheb, Wc));
        w.put(p + "framebounds", Eigen::MatrixXd((Eigen::VectorXd(4) << fb::framebounds(bank, g.lam).A, fb::framebounds(bank, g.lam).B, fb::framebounds(bank, g.lam).tightness, fb::framebounds(bank, g.lam).uncovered).finished()));
        w.put(p + "gains", fb::gains(bank, g.lam)); w.put(p + "dual", fb::dualGains(bank, g.lam));
        const MatrixXd Ck = noise(g.n, g.n, 13); const MatrixXcd C = (Ck * Ck.transpose()).cast<std::complex<double>>();
        const auto bm = ms::bandMoments(fb::gains(bank, g.lam), C, true); w.put(p + "bandPower", bm.power); w.putAll(p + "bandMatrices", bm.matrices);
        w.put(p + "bandPowerDiag", ms::bandPower(fb::gains(bank, g.lam), C.diagonal().real()));
        const MatrixXcd Fc = F.cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(g.n, 3, 12).cast<std::complex<double>>();
        const auto Wcx = nxr_sig::gwt(bank, g.T, Fc); w.putAll(p + "eigenC", Wcx); w.put(p + "synthC", nxr_sig::igwt(bank, g.T, Wcx)); w.put(p + "memberEnergyC", ms::memberEnergy(Wcx));
    }
    { const auto c = gt::chebyshevCoefficients([](double l) { return std::exp(-0.3 * l); }, 5.0, 20); w.put("gb.chebCoef", c); w.put("gb.chebEval", gt::chebyshevEval(c, 5.0, 1.7)); }
    // the joint bank
    {
        const Graph g2 = pathGraph(12); const auto gbank = graphBank(g2);
        tb::Design td(tb::Family::Itersine, 2.0, tb::Sides::Analytic); td.voices = 2; td.limits = std::make_pair(2.0, 60.0); td.edgeBins = tb::EdgeBins::Drop;
        const auto tbk = tb::design(td, 256, kFs);
        const auto jbank = jb::make(gbank, tbk); const MatrixXd Cm = noise(12, 50, 21);
        const auto Wj = nxr_sig::jwt(jbank, g2.lam, Cm); w.putAll("jb.modes", Wj);
        w.put("jb.synth", nxr_sig::ijwt(jbank, g2.lam, Wj, 50)); w.put("jb.scalogram", ms::memberTotals(Wj));
        const MatrixXd Fj = noise(12, 50, 22); const auto We = nxr_sig::jwt(jbank, g2.T, Fj); w.putAll("jb.eigen", We); w.put("jb.eigenSynth", nxr_sig::ijwt(jbank, g2.T, We, 50));
        gt::ChebyshevOperator cheb(g2.L, g2.M, g2.lam.maxCoeff() * 1.01, 40);
        const auto Wch = nxr_sig::jwt(jbank, cheb, Fj); w.putAll("jb.cheb", Wch); w.put("jb.chebSynth", nxr_sig::ijwt(jbank, cheb, Wch, 50));
        const MatrixXcd Cspec = MatrixXcd::Random(12, (long)tbk.nfft / 2 + 1);
        const auto Y = nxr_sig::jwtSpectrum(jbank, g2.lam, Cspec); w.putAll("jb.Y", Y); w.put("jb.synthSpectrum", nxr_sig::ijwtSpectrum(jbank, g2.lam, Y)); w.put("jb.synthSpectrumTight", nxr_sig::ijwtSpectrum(jbank, g2.lam, Y, fb::Synthesis::Tight));
        w.put("jb.frame", jb::frameOperator(jbank, g2.lam));
        const auto pb = jb::framebounds(jbank, g2.lam, jb::BoundsBy::Product); const auto fbd = jb::framebounds(jbank, g2.lam);
        w.put("jb.productBounds", Eigen::MatrixXd((Eigen::VectorXd(2) << pb.A, pb.B).finished())); w.put("jb.framebounds", Eigen::MatrixXd((Eigen::VectorXd(2) << fbd.A, fbd.B).finished()));
        // kernels
        const auto sk = jb::speedkernel(3.0, 2.0); const auto rk = jb::ridgeKernel(3.0, 1.0, +1);
        Eigen::MatrixXd K(8, 4); for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) { const double l = 0.5 + i, f = 1.0 + 3 * j; K(i, j) = std::abs(sk(l, f)); }
        w.put("jb.speedkernel", K);
        Eigen::MatrixXd R(8, 4); for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) R(i, j) = std::abs(rk(0.5 + i, 1.0 + 3 * j)); w.put("jb.ridgeKernel", R);
        tb::Design t2(tb::Family::Itersine, 1.0, tb::Sides::TwoSided); t2.voices = 2; t2.limits = std::make_pair(2.0, 60.0);
        const auto jb2 = jb::make(gbank, tb::design(t2, 256, kFs)); const MatrixXcd Cz = noise(12, 50, 31).cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(12, 50, 32).cast<std::complex<double>>();
        const auto Wz = nxr_sig::jwt(jb2, g2.lam, Cz); w.putAll("jb.two", Wz); w.put("jb.twoSynth", nxr_sig::ijwtComplex(jb2, g2.lam, Wz, 50));
        Eigen::MatrixXd R2(8, 4); const auto rk2 = jb::ridgeKernel(3.0, 1.0, -1); for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) R2(i, j) = std::abs(rk2(0.5 + i, -1.0 - 3 * j)); w.put("jb.ridgeKernelNeg", R2);
    }
    // bandwidths
    { nxr::signal::spectral::WelchOptions o; o.sfreq = 250; w.put("enbw.welch", nxr::signal::spectral::enbwPerBin(o));
      auto o2 = nxr::signal::spectral::brainstormPsdOptions(250, 2); w.put("enbw.welchBst", nxr::signal::spectral::enbwPerBin(o2));
      VectorXd f = VectorXd::LinSpaced(200, 0, 100), P = (1 + (f.array() / 20).square()).inverse().matrix(), G = (-(f.array() - 30).square() / 50).exp().matrix();
      w.put("enbw.lagfree", nxr::synthesis::effectiveBandwidth(f, P)); w.put("enbw.lagfreeGain", nxr::synthesis::effectiveBandwidth(f, P, G)); }
    // speedkernel (Rheome: omega = 2 pi f, angular width d) and ridgeKernel (nxr: f in Hz, width in Hz, sense) are the same ridge, exp(-1/2 ((f - v mu / 2 pi) / width)^2), for f > 0 and sense +1 with d = 2 pi width:
    // they agree to rounding there (the golden's two hashes differ by it), and the two are written differently (rounding), so they are NOT merged: the plan merges only what is identical
    { const auto sk = jb::speedkernel(3.0, 2.0 * M_PI * 2.0), rk = jb::ridgeKernel(3.0, 2.0, +1); double e = 0, onNeg = 0;
      for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) { const double l = 0.5 + i, f = 1.0 + 3 * j; e = std::max(e, std::abs(std::abs(sk(l, f)) - std::abs(rk(l, f)))); }
      onNeg = std::abs(sk(2.0, -3.0)); (void)onNeg;
      std::cout << "speedkernel vs ridgeKernel (d = 2 pi width, f > 0): largest difference " << e << "\n"; if (e > 1e-12) ++w.failures; }
    std::cout << w.checked << " arrays compared bitwise, " << w.failures << " differ\n";
    return w.failures ? 1 : 0;
}
