// nxr::jointbank: parity with Rheome (@jointfilterbank + +jtv, goldens/gen_jointbank_golden.m) on a synthetic sphere spectrum, with nxr's ridge_filter
// (gen_timebank_nxr.py), and the joint frame's own laws (separable bounds are products, exact and polynomial transforms, rotation sense on a ridge,
// refusals). Synthetic data only.
#include "nxr/jointbank.h"
#include "golden_io.h"
#include "mesh_helpers.h"
#include <random>
using namespace nxr::jointbank;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace fb = nxr::filterbank;
namespace tb = nxr::timebank;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static constexpr double kPi = 3.14159265358979323846;
using cd = std::complex<double>;

static MatrixXd noise(int n, int c, unsigned seed) { std::mt19937 r(seed); std::normal_distribution<double> N; MatrixXd x(n, c); for (auto& v : x.reshaped()) v = N(r); return x; }
static MatrixXcd cnoise(int n, int c, unsigned seed) { return noise(n, c, seed).cast<cd>() + cd(0, 1) * noise(n, c, seed + 1).cast<cd>(); }
template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
static tb::Design TD(tb::Family f, double g, tb::Sides s) { return tb::Design(f, g, s); }

static void rheome() {
    std::cout << "=== parity with Rheome @jointfilterbank + +jtv (synthetic sphere spectrum x itersine time bank, gain 2: peak 2, sum g^2 = 4, DC and Nyquist dropped) ===\n";
    Golden g(kDir + "/jointbank_golden.txt");
    EXPECT(g.good(), "Rheome joint golden opens");
    const VectorXd lam = g.mat("lambda").col(0);
    const MatrixXcd C = g.cmat("Cre", "Cim");
    EXPECT(g.tok() == "params", "params");
    const int N = (int)g.num(); const double fs = g.num(); const int V = (int)g.num();
    tb::Design td = TD(tb::Family::Itersine, 2.0, tb::Sides::Analytic); td.edgeBins = tb::EdgeBins::Drop; td.voices = V;
    const tb::TimeBank time = tb::design(td, (size_t)N, fs);
    double worst = 0;
    std::string word;
    while (g.f >> word && word == "case") {
        const std::string name = g.tok(), fam = g.tok(); const int Ng = (int)g.num(), Nt = (int)g.num(), Nk = (int)g.num();
        const fb::Bank graph = fb::design(fam == "itersine" ? fb::Family::Itersine : fb::Family::MexHat, fb::Range::fromSpectrum(lam));
        std::vector<Kernel> kern; if (name == "mexhat-speed") kern.push_back(speedKernel(40, 6));
        const JointBank jb = make(graph, time, kern);
        EXPECT(graph.size() == Ng && time.size() == Nt && jb.size() == Ng * Nt * Nk, name + ": " + std::to_string(jb.size()) + " members = " + std::to_string(Ng) + " x " + std::to_string(Nt) + " x " + std::to_string(Nk));
        const MatrixXd S = g.mat("S"), bg = g.mat("bounds");
        const double eS = relErr(frameOperator(jb, lam), S);
        // Rheome's A, B run over the whole plane (DC and Nyquist columns are uncovered when dropped); ours exclude those two bins
        MatrixXd inner = S.middleCols(1, S.cols() - 2); if (N % 2 == 1) inner = S.middleCols(1, S.cols() - 1);
        const auto b = bounds(jb, lam);
        const double eB = std::max(std::abs(b.A - inner.minCoeff()), std::abs(b.B - inner.maxCoeff())) / std::max(1.0, inner.maxCoeff());
        EXPECT(std::abs(bg(0, 0)) < 1e-12, name + ": Rheome's plane A = 0 only at the dropped DC / Nyquist columns (" + sci(bg(0, 0)) + ")");
        std::vector<int> pick; { const MatrixXd p = g.mat("pick"); for (int i = 0; i < p.cols(); ++i) pick.push_back((int)p(0, i)); }
        const std::vector<MatrixXcd> Y = analysisSpectrum(jb, lam, C);
        double eY = 0; for (int m : pick) eY = std::max(eY, relErr(Y[(size_t)m - 1], g.cmat("Yre" + std::to_string(m), "Yim" + std::to_string(m))));
        const MatrixXcd Cd = g.cmat("Cdre", "Cdim"), Ct = g.cmat("Ctre", "Ctim");
        const double eD = relErr(synthesisSpectrum(jb, lam, Y, fb::Synthesis::Dual), Cd), eT = relErr(synthesisSpectrum(jb, lam, Y, fb::Synthesis::Tight), Ct);
        const MatrixXd jS = g.mat("jtvS"); const MatrixXcd jC = g.cmat("jtvCre", "jtvCim");
        const double eJ = std::max(relErr(frameOperator(jb, lam), jS), relErr(synthesisSpectrum(jb, lam, Y, fb::Synthesis::Dual), jC));
        const double e = std::max({eS, eB, eY, eD, eT, eJ});
        worst = std::max(worst, e);
        EXPECT(e < 1e-11, name + ": frame operator " + sci(eS) + ", bounds " + sci(eB) + ", wt " + sci(eY) + ", iwt dual " + sci(eD) + " tight " + sci(eT) + ", +jtv bounds/dual " + sci(eJ));
        if (name == "itersine-sep") {
            const auto pb = productBounds(jb, lam);
            EXPECT(std::abs(pb.A - b.A) < 1e-10 && std::abs(pb.B - b.B) < 1e-10, "separable: A, B = products of the graph and time bounds (" + sci(pb.A - b.A) + ", " + sci(pb.B - b.B) + "), A = B = 4 = gain^2 (tight): " + sci(b.A - 4));
        } else {
            EXPECT(throws([&] { productBounds(jb, lam); }), name + ": productBounds refuses a bank with a joint kernel");
        }
        if (g.tok() != "end") { std::cout << "  [FAIL] golden structure\n"; ++g_failures; return; }
    }
    EXPECT(worst < 1e-11, "worst " + sci(worst));
}

static void nxrRidge() {
    std::cout << "=== parity with nxr spectral.joint.ridge_filter (a signed ridge on a two-sided axis) ===\n";
    Golden g(kDir + "/timebank_nxr_golden.txt");
    for (const char* skip : {"window_edges", "window_gains", "window_gains_full", "itersine_gains", "morlet_fc", "morlet_hat", "z", "morlet_freqs", "morlet_re", "morlet_im", "rot_xy", "rot_Aplus", "rot_Aminus"}) g.mat(skip);
    const VectorXd mu = g.mat("ridge_mu").row(0).transpose();
    const MatrixXcd Z = g.cmat("ridge_Zre", "ridge_Zim");
    for (int s : {+1, -1}) {
        const MatrixXcd R = g.cmat(s > 0 ? "ridge+1_re" : "ridge-1_re", s > 0 ? "ridge+1_im" : "ridge-1_im");
        if (s > 0) { /* params come last */ }
        const double v = 3.0, width = 1.5;
        const Kernel K = ridgeKernel(v, width, s);
        const int n = (int)Z.cols(); const double fs = 100;
        const VectorXd f = nxr::fft::fftfreq((size_t)n, 1 / fs);
        MatrixXcd out(Z.rows(), n);
        for (int p = 0; p < Z.rows(); ++p) {
            Eigen::VectorXcd X = nxr::fft::fft(Eigen::VectorXcd(Z.row(p).transpose()));
            for (int k = 0; k < n; ++k) X(k) *= K(mu(p) * mu(p), f(k));
            out.row(p) = nxr::fft::ifft(X).transpose();
        }
        const double e = relErr(out, R);
        EXPECT(e < 1e-13, std::string("ridgeKernel sense ") + (s > 0 ? "+1" : "-1") + " == nxr ridge_filter " + sci(e));
    }
}

static void laws() {
    std::cout << "=== the joint frame's own laws ===\n";
    // a synthetic sphere spectrum and a modes x time series
    VectorXd lam(36); { int i = 0; for (int l = 0; l <= 5; ++l) for (int k = 0; k < 2 * l + 1; ++k) lam(i++) = l * (l + 1.0); }
    const fb::Bank graph = fb::design(fb::Family::Itersine, fb::Range::fromSpectrum(lam));
    for (double gn : {1.0, 2.0, 3.0}) {
        tb::Design td = TD(tb::Family::Itersine, gn, tb::Sides::Analytic); td.voices = 2;
        const size_t T = 120; const double fs = 64;
        const JointBank jb = make(graph, tb::design(td, T, fs));
        MatrixXd C = noise(36, (int)T, 5); C.array() += 0.7;       // DC offset
        const auto b = bounds(jb, lam);
        const double G = gn * gn;           // the time bank's sum of g^2 (the graph itersine bank is tight at 1)
        Coefficients W = analysis(jb, lam, C);
        const double eD = relErr(synthesis(jb, lam, W, T, fb::Synthesis::Dual), C), eT = relErr(synthesis(jb, lam, W, T, fb::Synthesis::Tight), C);
        double en = 0; for (const auto& w : W) en += w.cwiseAbs2().sum();
        EXPECT(std::abs(b.A - G) < 1e-10 && std::abs(b.B - G) < 1e-10 && isTight(jb, lam) && eD < 1e-12 && eT < 1e-12 && std::abs(en - G / 2 * C.squaredNorm()) < 1e-9 * C.squaredNorm(),
               "analytic joint bank (gain " + std::to_string((int)gn) + "): tight A = B = gain^2 = " + std::to_string((int)G) + ", modes x time reconstruct exactly (dual " + sci(eD) + ", tight " + sci(eT) + "), Parseval (gain^2/2)Σ‖c‖²");
        // the time-domain path equals the spectrum-domain one
        const MatrixXcd spec = tb::spectrum(jb.time, MatrixXd(C.transpose())).transpose();
        const auto Y = analysisSpectrum(jb, lam, spec);
        double e = 0;
        for (int m = 0; m < jb.size(); ++m) e = std::max(e, relErr(MatrixXcd(W[(size_t)m].transpose()), tb::filterSpectrum(jb.time, VectorXd::Ones(spec.cols()), MatrixXcd(Y[(size_t)m].transpose()))));
        EXPECT(e < 1e-13, "modes x time analysis == FFT, multiply on (lambda, f), inverse FFT " + sci(e));
    }
    // two-sided: complex mode series, tight, reconstruct; Meyer time side
    {
        const size_t T = 100; const double fs = 50;
        tb::Design td = TD(tb::Family::Meyer, 1.0, tb::Sides::TwoSided);
        const JointBank jb = make(graph, tb::design(td, T, fs));
        const MatrixXcd C = cnoise(36, (int)T, 8);
        const Coefficients W = analysis(jb, lam, C);
        const double eD = relErr(synthesisComplex(jb, lam, W, T, fb::Synthesis::Dual), C), eT = relErr(synthesisComplex(jb, lam, W, T, fb::Synthesis::Tight), C);
        const auto b = bounds(jb, lam);
        EXPECT(std::abs(b.A - 1) < 1e-10 && std::abs(b.B - 1) < 1e-10 && eD < 1e-12 && eT < 1e-12, "two-sided joint bank: tight A = B = 1 incl. DC and Nyquist, complex series reconstruct exactly (dual " + sci(eD) + ", tight " + sci(eT) + ")");
        EXPECT(throws([&] { analysis(make(graph, tb::design(TD(tb::Family::Meyer, 1.0, tb::Sides::Analytic), T, fs)), lam, C); }), "an analytic joint bank REFUSES a complex mode series");
        // a (lambda, t) lag kernel is refused with the reason; a (lambda, f) kernel is handed f in HERTZ, signed
        std::string why; try { make(graph, tb::design(TD(tb::Family::Meyer, 1.0, tb::Sides::Analytic), T, fs), std::vector<LagKernel>{LagKernel{[](double, double) { return cd(1); }}}); } catch (const std::invalid_argument& e) { why = e.what(); }
        EXPECT(why.find("time LAG") != std::string::npos && why.find("cannot multiply") != std::string::npos, "a (lambda, t) kernel is refused by the joint bank: \"" + why.substr(0, 70) + "...\"");
        { Kernel probe = [](double, double f) { return cd(f); };
          const tb::TimeBank two = tb::design(TD(tb::Family::Meyer, 1.0, tb::Sides::TwoSided), T, fs);
          const JointBank jbp = make(graph, two, {probe});
          const MatrixXcd Wm = memberGains(jbp, 0, lam); const VectorXd fr = tb::frequencies(two); const MatrixXd gg = fb::gains(graph, lam); const MatrixXd tg = tb::gains(two);
          double e = 0; for (int k = 0; k < (int)lam.size(); ++k) for (int j = 0; j < fr.size(); ++j) e = std::max(e, std::abs(Wm(k, j) - fr(j) * gg(k, 0) * tg(j, 0)));
          EXPECT(e < 1e-13, "joint-bank kernels receive f in Hz, signed (a kernel returning f reproduces the signed fftfreq grid) " + sci(e)); }
    }
    // rotation sense on a ridge: a +/- signed ridge kernel pair separates counter-clockwise from clockwise travelling patterns
    {
        const size_t T = 256; const double fs = 128, v = 8;
        tb::Design td = TD(tb::Family::Itersine, 1.0, tb::Sides::TwoSided); td.voices = 2;
        const JointBank jb = make(graph, tb::design(td, T, fs), {ridgeKernel(v, 3, +1), ridgeKernel(v, 3, -1)});
        MatrixXcd ccw(36, (int)T), cw(36, (int)T);
        for (int k = 0; k < 36; ++k) { const double f = v * std::sqrt(lam(k)) / (2 * kPi); for (int t = 0; t < (int)T; ++t) { const cd z = std::exp(cd(0, 2 * kPi * f * t / fs)); ccw(k, t) = z; cw(k, t) = std::conj(z); } }
        const int per = jb.size() / 2;
        auto split = [&](const MatrixXcd& Cm) { const VectorXd e = energies(analysis(jb, lam, Cm)); return std::make_pair(e.head(per).sum(), e.tail(per).sum()); };
        const auto [a1, a2] = split(ccw); const auto [b1, b2] = split(cw);
        EXPECT(a1 > 10 * a2 && b2 > 10 * b1, "ridge sense: ccw pattern energy +ridge " + sci(a1) + " vs -ridge " + sci(a2) + "; cw pattern +ridge " + sci(b1) + " vs -ridge " + sci(b2));
    }
    // axes: the joint bank carries its axes; compatible() names every mismatch (sizes agreeing proves nothing)
    {
        const tb::TimeBank a1 = tb::design(TD(tb::Family::Meyer, 1.0, tb::Sides::Analytic), 100, 50), a2 = tb::design(TD(tb::Family::Meyer, 1.0, tb::Sides::Analytic), 100, 60);
        const Axes x = axes(make(graph, a1), lam), y = axes(make(graph, a1), lam), z = axes(make(graph, a2), lam, true);
        VectorXd lam2 = lam; lam2(5) += 1;
        const Axes w2 = axes(make(graph, a1), lam2);
        const Axes tw = axes(make(graph, tb::design(TD(tb::Family::Meyer, 1.0, tb::Sides::TwoSided), 100, 50)), lam);
        EXPECT(compatible(x, y).empty() && compatible(x, z).size() == 2 && compatible(x, w2).size() == 1 && compatible(x, tw).size() >= 1,
               "compatible(): same axes agree; a different rate and a padded boundary are named (" + std::to_string(compatible(x, z).size()) + "), a different spectrum of the same size is caught, analytic vs two-sided is caught");
    }
    // vertices: an icosphere pencil from the part-1 golden; exact and Chebyshev
    {
        Golden g(kDir + "/filterbank_golden.txt");
        const MatrixXd Ld = g.mat("L"), Md = g.mat("M"); g.mat("lambda");
        const int n = (int)Ld.rows();
        Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(Ld, Md);
        fb::EigenTransform Tr; Tr.phi = es.eigenvectors(); Tr.lambda = es.eigenvalues(); Tr.mass = Md.sparseView();
        for (int i = 0; i < Tr.lambda.size(); ++i) if (std::abs(Tr.lambda(i)) < 1e-10) Tr.lambda(i) = 0;
        const fb::Bank heat = fb::design(fb::Family::Heat, fb::Range::fromSpectrum(Tr.lambda));
        const size_t T = 64;
        tb::Design td = TD(tb::Family::Itersine, 1.0, tb::Sides::Analytic); td.voices = 1; td.minBins = 4;
        const JointBank jb = make(heat, tb::design(td, T, 32));
        const MatrixXd F = noise(n, (int)T, 3);
        const Coefficients We = analysis(jb, Tr, F);
        const double eR = relErr(synthesis(jb, Tr, We, T, fb::Synthesis::Dual), F);
        // consistent with the mode-domain path lifted by Phi
        const MatrixXd Cm = Tr.phi.transpose() * (Tr.mass * F);
        const Coefficients Wm = analysis(jb, Tr.lambda, Cm);
        double eL = 0; for (int m = 0; m < jb.size(); ++m) eL = std::max(eL, relErr(We[(size_t)m], MatrixXcd(Tr.phi.cast<cd>() * Wm[(size_t)m])));
        EXPECT(eR < 1e-11 && eL < 1e-12, "vertices x time (exact): analysis == Phi . modes analysis " + sci(eL) + ", dual synthesis reconstructs the field " + sci(eR));
        const fb::ChebyshevOperator cheb(Ld.sparseView(), Md.sparseView(), 1.01 * Tr.lambda.maxCoeff(), 80);
        const Coefficients Wc = analysis(jb, cheb, F);
        double eC = 0; for (int m = 0; m < jb.size(); ++m) eC = std::max(eC, relErr(Wc[(size_t)m], We[(size_t)m]));
        const double eCs = relErr(synthesis(jb, cheb, Wc, T, fb::Synthesis::Dual), F);
        EXPECT(eC < 1e-8, "vertices x time (Chebyshev order 80, heat bank): analysis == exact " + sci(eC));
        EXPECT(eCs < 1e-5, "vertices x time (Chebyshev): dual synthesis o analysis reconstructs the field " + sci(eCs));
        const JointBank jk = make(heat, tb::design(td, T, 32), {ridgeKernel(1, 1, +1)});
        EXPECT(throws([&] { analysis(jk, cheb, F); }), "the polynomial route REFUSES a bank with a joint kernel (it couples lambda and f)");
        const MatrixXcd Fc = cnoise(n, (int)T, 4);
        tb::Design tt = TD(tb::Family::Meyer, 1.0, tb::Sides::TwoSided);
        const JointBank j2 = make(heat, tb::design(tt, T, 32));
        const Coefficients W2 = analysis(j2, Tr, Fc);
        EXPECT(relErr(synthesisComplex(j2, Tr, W2, T), Fc) < 1e-11, "vertices x time, complex field on a two-sided bank (exact) reconstructs " + sci(relErr(synthesisComplex(j2, Tr, W2, T), Fc)));
        EXPECT(relErr(analysis(j2, cheb, Fc)[0], W2[0]) < 1e-7, "vertices x time, complex field (Chebyshev) == exact " + sci(relErr(analysis(j2, cheb, Fc)[0], W2[0])));
    }
}

int main() {
    rheome();
    nxrRidge();
    laws();
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}
