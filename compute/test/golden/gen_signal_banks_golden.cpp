// Generator of signal_banks_golden.txt: the OLD transforms (compiled against the code of 2026-11-18 before the wt / iwt / gwt / igwt / jwt / ijwt refactor, commit dc5f877) on the synthetic
// inputs of compute/test/signal_banks_inputs.h. The test compares the new code to the recorded hashes. Not built by CMake (it needs the old API); to remake it, check out dc5f877.
#include "../golden_hash.h"
#include "../signal_banks_inputs.h"
#include "nxr/signal/wavelet.h"
#include "nxr/signal/graphtransforms.h"
#include "nxr/signal/spectral.h"
#include "nxr/synthesis/lagfree.h"
#include <iostream>
using namespace sbi;
namespace jb = nxr::signal::jointfilterbank;
int main(int argc, char** argv) {
    goldenhash::Writer w(argc > 1 ? argv[1] : "signal_banks_golden.txt");
    struct B { const char* n; tb::TimeFilterbank b; };
    std::vector<B> banks = {{"itersine", itersine()}, {"meyer", meyer()}, {"morlet", morlet()}};
    const MatrixXd x = tones(200, 2, 3, 50);
    for (auto& [name, bk] : banks) {
        const std::string p = std::string("tb.") + name + ".";
        const tb::Coefficients W = tb::analysis(bk, x);
        w.putAll(p + "analysis", W);
        w.put(p + "synthDual", tb::synthesis(bk, W, 200, fb::Synthesis::Dual));
        if (std::string(name) != "morlet") w.put(p + "synthTight", tb::synthesis(bk, W, 200, fb::Synthesis::Tight));
        for (auto& m : tb::scalogram(W)) w.put(p + "scalogram", m);
        w.put(p + "energies", tb::energies(W));
        w.put(p + "centres", tb::centerFrequencies(bk));
        { auto r = tb::rates(bk, 2.0); w.put(p + "rates", r.first); }
        w.put(p + "frame", tb::frameOperator(bk)); w.put(p + "dual", tb::dualGains(bk));
        w.put(p + "gains", tb::gains(bk));
        const MatrixXcd S = tb::spectrum(bk, x); w.put(p + "spectrum", S);
        w.put(p + "filterSpectrum", tb::filterSpectrum(bk, tb::gains(bk).col(1), S));
        w.put(p + "coefSpectrum", tb::coefficientSpectrum(bk, W[1]));
        w.put(p + "inverseReal", tb::inverseReal(bk, S, 200));
        w.put(p + "eqBw", tb::equivalentBandwidth(bk, 1)); w.put(p + "noiseBw", tb::noiseBandwidth(bk, 1));
        const auto own = tb::analysisOwnRate(bk, x.col(0));
        for (std::size_t m = 0; m < own.size(); ++m) { w.put(p + "own" + std::to_string(m), own[m].coef); w.put(p + "ownT" + std::to_string(m), own[m].t); }
        tb::Design d(name == std::string("itersine") ? tb::Family::Itersine : tb::Family::Meyer, 2.0, tb::Sides::Analytic); d.voices = 2; d.limits = std::make_pair(2.0, 60.0);
        if (std::string(name) != "morlet") { const auto bt = tb::analysisOfBlock(d, x.transpose(), kFs, 20, 150); w.putAll(p + "block", bt.own); }
    }
    { const auto bk = twoSided(); MatrixXcd z = noise(100, 2, 5).cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(100, 2, 6).cast<std::complex<double>>();
      const auto W = tb::analysis(bk, z); w.putAll("tb.two.analysis", W); w.put("tb.two.synth", tb::synthesisComplex(bk, W, 100)); w.put("tb.two.synthTight", tb::synthesisComplex(bk, W, 100, fb::Synthesis::Tight));
      w.put("tb.two.inverseComplex", tb::inverseComplex(bk, tb::spectrum(bk, z), 100)); w.put("tb.two.centres", tb::centerFrequencies(bk)); }
    w.put("tb.logCentres", Eigen::Map<const VectorXd>(tb::logCentres(kFs, 2, 60, 3).data(), (long)tb::logCentres(kFs, 2, 60, 3).size()));
    w.put("tb.cwtCentres", Eigen::Map<const VectorXd>(tb::cwtCentres(2, 60, 3).data(), (long)tb::cwtCentres(2, 60, 3).size()));
    // Wt / Iwt streaming, chunks of 37 and the whole record (the node's use: add + finish on one block)
    for (auto& [name, bk] : banks) {
        nxr::signal::Options o; o.tolerance = 1e-9; o.maxMargin = 4096;
        const MatrixXd xt = tones(900, 2, 9, 50).transpose();
        nxr::signal::Wt wt(bk, 2, o);
        auto b1 = wt.add(xt); auto b2 = wt.finish();
        const std::string p = std::string("wt.") + name + ".";
        w.put(p + "margin", (double)wt.margin());
        for (std::size_t m = 0; m < b1.size(); ++m) {
            Eigen::MatrixXcd all(b1[m].coef.rows() + b2[m].coef.rows(), 2); all << b1[m].coef, b2[m].coef;
            w.put(p + "coef" + std::to_string(m), all); w.put(p + "first" + std::to_string(m), (double)b1[m].first);
        }
        if (std::string(name) != "morlet") { nxr::signal::Iwt iw(bk, 2, o); auto r1 = iw.add(b1); auto r2 = iw.add(b2); (void)r1; (void)r2; w.put(p + "iwt", iw.finish(900)); }
    }
    // the graph bank
    const Graph g = pathGraph(); const MatrixXd F = noise(g.n, 3, 11);
    for (auto fam : {fb::Family::Itersine, fb::Family::MexHat, fb::Family::Heat, fb::Family::LogItersine}) {
        const auto bank = graphBank(g, fam); const std::string p = "gb." + std::to_string((int)fam) + ".";
        const auto W = fb::analysis(bank, g.T, F); w.putAll(p + "eigen", W);
        w.put(p + "member1", fb::analysisMember(bank, 1, g.T, F));
        w.put(p + "synth", fb::igwt(bank, g.T, W)); w.put(p + "synthTight", fb::igwt(bank, g.T, W, fb::Synthesis::Tight));
        w.put(p + "scaleEnergy", fb::scaleEnergy(W)); w.put(p + "memberEnergy", fb::memberEnergy(W));
        w.put(p + "memberEnergyW", fb::memberEnergy(W, g.Md.diagonal()));
        gt::ChebyshevOperator cheb(g.L, g.M, g.lam.maxCoeff() * 1.01, 40);
        const auto Wc = fb::analysis(bank, cheb, F); w.putAll(p + "cheb", Wc);
        w.put(p + "chebMember1", fb::analysisMember(bank, 1, cheb, F)); w.put(p + "chebSynth", fb::igwt(bank, cheb, Wc));
        w.put(p + "framebounds", Eigen::MatrixXd((Eigen::VectorXd(4) << fb::framebounds(bank, g.lam).A, fb::framebounds(bank, g.lam).B, fb::framebounds(bank, g.lam).tightness, fb::framebounds(bank, g.lam).uncovered).finished()));
        w.put(p + "gains", fb::gains(bank, g.lam)); w.put(p + "dual", fb::dualGains(bank, g.lam));
        const MatrixXd Ck = noise(g.n, g.n, 13); const MatrixXcd C = (Ck * Ck.transpose()).cast<std::complex<double>>();
        const auto bm = fb::bandMoments(bank, g.lam, C, true); w.put(p + "bandPower", bm.power); w.putAll(p + "bandMatrices", bm.matrices);
        w.put(p + "bandPowerDiag", fb::bandPower(bank, g.lam, C.diagonal().real()));
        const MatrixXcd Fc = F.cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(g.n, 3, 12).cast<std::complex<double>>();
        const auto Wcx = fb::analysis(bank, g.T, Fc); w.putAll(p + "eigenC", Wcx); w.put(p + "synthC", fb::igwt(bank, g.T, Wcx)); w.put(p + "memberEnergyC", fb::memberEnergy(Wcx));
    }
    { const auto c = gt::chebyshevCoefficients([](double l) { return std::exp(-0.3 * l); }, 5.0, 20); w.put("gb.chebCoef", c); w.put("gb.chebEval", gt::chebyshevEval(c, 5.0, 1.7)); }
    // the joint bank
    {
        const Graph g2 = pathGraph(12); const auto gbank = graphBank(g2);
        tb::Design td(tb::Family::Itersine, 2.0, tb::Sides::Analytic); td.voices = 2; td.limits = std::make_pair(2.0, 60.0); td.edgeBins = tb::EdgeBins::Drop;
        const auto tbk = tb::design(td, 256, kFs);
        const auto jbank = jb::make(gbank, tbk); const MatrixXd Cm = noise(12, 50, 21);
        const auto Wj = jb::analysis(jbank, g2.lam, Cm); w.putAll("jb.modes", Wj);
        w.put("jb.synth", jb::synthesis(jbank, g2.lam, Wj, 50)); w.put("jb.scalogram", jb::scalogram(Wj));
        const MatrixXd Fj = noise(12, 50, 22); const auto We = jb::analysis(jbank, g2.T, Fj); w.putAll("jb.eigen", We); w.put("jb.eigenSynth", jb::synthesis(jbank, g2.T, We, 50));
        gt::ChebyshevOperator cheb(g2.L, g2.M, g2.lam.maxCoeff() * 1.01, 40);
        const auto Wch = jb::analysis(jbank, cheb, Fj); w.putAll("jb.cheb", Wch); w.put("jb.chebSynth", jb::synthesis(jbank, cheb, Wch, 50));
        const MatrixXcd Cspec = MatrixXcd::Random(12, (long)tbk.nfft / 2 + 1);
        const auto Y = jb::analysisSpectrum(jbank, g2.lam, Cspec); w.putAll("jb.Y", Y); w.put("jb.synthSpectrum", jb::synthesisSpectrum(jbank, g2.lam, Y)); w.put("jb.synthSpectrumTight", jb::synthesisSpectrum(jbank, g2.lam, Y, fb::Synthesis::Tight));
        w.put("jb.frame", jb::frameOperator(jbank, g2.lam));
        const auto pb = jb::productBounds(jbank, g2.lam); const auto fbd = jb::framebounds(jbank, g2.lam);
        w.put("jb.productBounds", Eigen::MatrixXd((Eigen::VectorXd(2) << pb.A, pb.B).finished())); w.put("jb.framebounds", Eigen::MatrixXd((Eigen::VectorXd(2) << fbd.A, fbd.B).finished()));
        // kernels
        const auto sk = jb::speedkernel(3.0, 2.0); const auto rk = jb::ridgeKernel(3.0, 1.0, +1);
        Eigen::MatrixXd K(8, 4); for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) { const double l = 0.5 + i, f = 1.0 + 3 * j; K(i, j) = std::abs(sk(l, f)); }
        w.put("jb.speedkernel", K);
        Eigen::MatrixXd R(8, 4); for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) R(i, j) = std::abs(rk(0.5 + i, 1.0 + 3 * j)); w.put("jb.ridgeKernel", R);
        tb::Design t2(tb::Family::Itersine, 1.0, tb::Sides::TwoSided); t2.voices = 2; t2.limits = std::make_pair(2.0, 60.0);
        const auto jb2 = jb::make(gbank, tb::design(t2, 256, kFs)); const MatrixXcd Cz = noise(12, 50, 31).cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(12, 50, 32).cast<std::complex<double>>();
        const auto Wz = jb::analysis(jb2, g2.lam, Cz); w.putAll("jb.two", Wz); w.put("jb.twoSynth", jb::synthesisComplex(jb2, g2.lam, Wz, 50));
        Eigen::MatrixXd R2(8, 4); const auto rk2 = jb::ridgeKernel(3.0, 1.0, -1); for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) R2(i, j) = std::abs(rk2(0.5 + i, -1.0 - 3 * j)); w.put("jb.ridgeKernelNeg", R2);
    }
    // bandwidths
    { nxr::signal::spectral::WelchOptions o; o.sfreq = 250; w.put("enbw.welch", nxr::signal::spectral::enbwPerBin(o));
      auto o2 = nxr::signal::spectral::brainstormPsdOptions(250, 2); w.put("enbw.welchBst", nxr::signal::spectral::enbwPerBin(o2));
      VectorXd f = VectorXd::LinSpaced(200, 0, 100), P = (1 + (f.array() / 20).square()).inverse().matrix(), G = (-(f.array() - 30).square() / 50).exp().matrix();
      w.put("enbw.lagfree", nxr::synthesis::effectiveBandwidth(f, P)); w.put("enbw.lagfreeGain", nxr::synthesis::effectiveBandwidth(f, P, G)); }
    std::cout << "written\n";
}
