// nxr::kernels: parity with Rheome's +filters (space-time kernels, sphere_kernel, steer, spin, scatter), jointfilterbank.speedkernel / band and filters.psitime
// (goldens/gen_kernels_golden.m, on a synthetic sphere spectrum and icosphere), plus the kernels' own identities. Synthetic data only.
#include "nxr/signal/kernels.h"
#include "nxr/signal/jointfilterbank.h"
#include "golden_io.h"
#include "mesh_helpers.h"
using namespace nxr::signal;
namespace fb = nxr::signal::graphfilterbank;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
using cd = std::complex<double>;

static void eq(const std::string& name, const MatrixXd& a, const MatrixXd& b) { const double e = relErr(a, b); EXPECT(e < 1e-11, name + " == Rheome, rel " + sci(e)); }
static void eqc(const std::string& name, const MatrixXcd& a, const MatrixXcd& b) { const double e = relErr(a, b); EXPECT(e < 1e-11, name + " == Rheome, rel " + sci(e)); }
template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
    Golden g(kDir + "/kernels_golden.txt");
    EXPECT(g.good(), "Rheome kernels golden opens");
    std::cout << "=== parity with Rheome +filters ===\n";
    const VectorXd lam = g.mat("lambda").col(0);
    const VectorXd grid = g.mat("grid").row(0).transpose(), half = g.mat("half").row(0).transpose(), t = g.mat("t").row(0).transpose();
    const MatrixXd sh = g.mat("signedHalf");
    // signed frequencies: right on a full FFT grid AND on a one-sided one (the half is stated; the old rule wrapped a one-sided grid by Fs = N df and made its top half negative)
    const VectorXd ws = signedFrequencies(grid, Half::Full);
    const MatrixXd sf = g.mat("signedFull");
    EXPECT(relErr(MatrixXd(signedFrequencies(half, Half::Positive).transpose()), sh) < 1e-14 && relErr(MatrixXd(signedFrequencies(half, Half::Positive).transpose()), MatrixXd(half.transpose())) < 1e-14,
           "signedFrequencies(one-sided grid, Positive) == Rheome's signedfrequencies(.., 'positive'): unchanged, every bin non-negative (a 20 Hz bin reads +20 Hz)");
    EXPECT(relErr(MatrixXd(ws.transpose()), sf) < 1e-11 && relErr(MatrixXd(ws), MatrixXd(nxr::signal::fftfreq(grid.size(), 1.0 / 40.0))) < 1e-11,
           "signedFrequencies(full grid, Full) == Rheome's, and is fftfreq (Rheome maps +fs/2 to -fs/2 as numpy does)");
    EXPECT(half.minCoeff() >= 0 && signedFrequencies(half, Half::Full)(half.size() - 1) < 0, "  ... what the old unconditional wrap did to the one-sided grid's top bin: " + std::to_string(signedFrequencies(half, Half::Full)(half.size() - 1)) + " Hz for " + std::to_string(half(half.size() - 1)) + " Hz (hence the stated half)");
    eq("gabor", gabor(lam, ws, 20, 10, 2), g.mat("gabor"));
    eq("travwave (both senses, |f|)", travwave(lam, ws, 3, 2, 0), g.mat("travwave"));
    eq("travwave sense +1", travwave(lam, ws, 3, 2, 1), g.mat("travwave_p"));
    eq("travwave sense -1", travwave(lam, ws, 3, 2, -1), g.mat("travwave_m"));
    eq("travwave sense +1 (one-sided grid)", travwave(lam, signedFrequencies(half, Half::Positive), 3, 2, 1), g.mat("travwave_half"));
    eq("gabor (one-sided grid)", gabor(lam, signedFrequencies(half, Half::Positive), 20, 10, 2), g.mat("gabor_half"));
    eqc("resonator", resonator(lam, ws, 10, 6), g.cmat("resRe", "resIm"));
    eq("stmatern", stmatern(lam, ws, 5, 1.5, 2), g.mat("stmatern"));
    eq("temporalBandpass", MatrixXd(temporalBandpass(ws, 10, 3).transpose()), g.mat("bandpass"));
    eq("diffusion (continuum, D = 1.3)", diffusion(lam, t, 1.3), g.mat("diffusion"));
    eq("wave (continuum)", wave(lam, t, 1.7), g.mat("wave"));
    eq("dampedwave (continuum)", dampedwave(lam, t, 1.7, 0.2), g.mat("dampedwave"));
    eq("kleingordon (continuum, omega^2 = c^2 lambda + m^2)", kleingordon(lam, t, 1.2, 0.3), g.mat("kleingordon"));
    const MatrixXcd sch = schrodinger(lam, t, 1.5);
    eqc("schrodinger", sch, g.cmat("schRe", "schIm"));
    EXPECT((sch.cwiseAbs().array() - 1).abs().maxCoeff() < 1e-14, "schrodinger is unitary: |G| = 1 for every mode and lag");
    // the Chebyshev-angle variants reproduce the earlier definitions' numbers exactly
    eq("wave, Variant::Chebyshev (= the old wave)", wave(lam, t, 1.7, Variant::Chebyshev, 0), g.mat("wave_cheb"));
    eq("wave, Variant::Chebyshev, lmax 50", wave(lam, t, 1.7, Variant::Chebyshev, 50), g.mat("wave_cheb_lmax"));
    eq("dampedwave, Variant::Chebyshev", dampedwave(lam, t, 1.7, 0.2, Variant::Chebyshev, 0), g.mat("dampedwave_cheb"));
    eq("kleingordon, Variant::Chebyshev", kleingordon(lam, t, 1.2, 0.3, Variant::Chebyshev, 0), g.mat("kleingordon_cheb"));
    Eigen::VectorXd t3(3); t3 << 0.05, 0.2, 0.5;
    eq("heat", heat(lam, t3, 0), g.mat("heat"));
    eq("heat (lmax 30)", heat(lam, t3, 30), g.mat("heat_lmax"));
    eq("mexhat", mexhat(lam, t3), g.mat("mexhat"));
    Eigen::VectorXd a(2), b(2); a << 0.05, 0.1; b << 0.2, 0.5;
    eq("diffgauss", diffgauss(lam, a, b), g.mat("diffgauss"));
    const VectorXd gam = g.mat("gamma").col(0), gl = g.mat("gl").col(0);
    eq("sphere_kernel", MatrixXd(sphere_kernel(gam, gl)), g.mat("sphere_kernel"));
    const VectorXd f1 = g.mat("f1").row(0).transpose();
    eq("raisedCosineBand == filters.psitime", MatrixXd(raisedCosineBand(f1, 8, 14, 1.5).transpose()), g.mat("psitime"));
    eq("raisedCosineBand == jointfilterbank.band(f), f in Hz", MatrixXd(raisedCosineBand(f1, 8, 14, 1.5).transpose()), g.mat("band_hz"));
    { nxr::signal::jointfilterbank::Kernel K = nxr::signal::jointfilterbank::speedkernel(40, 6); const MatrixXd ref = g.mat("speed"); MatrixXd out(lam.size(), f1.size());
      for (int i = 0; i < lam.size(); ++i) for (int j = 0; j < f1.size(); ++j) out(i, j) = K(lam(i), f1(j)).real();
      eq("speedkernel (f in Hz, omega = 2 pi f inside)", out, ref); }
    const MatrixXd Wq = g.mat("Wq"), q0 = g.mat("q0");
    eq("steer", steer(Wq, Eigen::Vector4d(q0.col(0))), g.mat("steer"));
    const MatrixXd V = g.mat("V"), ax = g.mat("axis"), th = g.mat("theta");
    eq("spin", spin(V, ax, VectorXd(th.col(0))), g.mat("spin"));
    { const MatrixXd s = spin(V, ax, VectorXd(th.col(0)));
      double e = 0; for (int i = 0; i < V.rows(); ++i) e = std::max(e, std::abs(s.row(i).norm() - V.row(i).norm()));
      EXPECT(e < 1e-13, "spin preserves every vector's norm " + sci(e));
      Eigen::MatrixXd axn = ax.row(0); const MatrixXd s2 = spin(V, axn, VectorXd::Constant(1, 0.7));
      double ea = 0; for (int i = 0; i < V.rows(); ++i) { const Eigen::Vector3d n = ax.row(0).transpose().normalized(); ea = std::max(ea, std::abs(s2.row(i).dot(n.transpose()) - V.row(i).dot(n.transpose()))); }
      EXPECT(ea < 1e-13, "spin about an axis leaves the component along it unchanged " + sci(ea)); }

    std::cout << "=== graph wavelet scattering (filters.scatter) on a synthetic icosphere ===\n";
    const MatrixXd Ld = g.mat("icoL"), Md = g.mat("icoM"), lm = g.mat("icoLambda");
    const MatrixXcd Fc = g.cmat("FcRe", "FcIm");
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(Ld, Md);
    fb::EigenTransform T; T.phi = es.eigenvectors(); T.lambda = es.eigenvalues(); T.mass = Md.sparseView();
    for (int i = 0; i < T.lambda.size(); ++i) if (std::abs(T.lambda(i)) < 1e-10) T.lambda(i) = 0;
    std::string word;
    int cases = 0;
    while (g.f >> word && word == "scatter") {
        const std::string fam = g.tok(); const int P = (int)g.num();
        const MatrixXd sig = g.mat("Sigma"), S0re = g.mat("S0re"), S0im = g.mat("S0im"), S1 = g.mat("S1"), pairs = g.mat("Pairs"), S2 = g.mat("S2");
        const fb::GraphFilterbank bank = fb::design(fam == "mexhat" ? fb::Family::MexHat : fam == "heat" ? fb::Family::Heat : fb::Family::Itersine, fb::Range::fromSpectrum(T.lambda));
        const Scatter S = scatter(bank, T, Fc, 2);
        MatrixXd s1(S.S1.size(), 3); for (size_t m = 0; m < S.S1.size(); ++m) s1.row((Eigen::Index)m) = S.S1[m].row(0);
        MatrixXd pr(S.pairs.size(), 2), s2(S.pairs.size(), 3);
        for (size_t q = 0; q < S.pairs.size(); ++q) { pr(q, 0) = S.pairs[q].first + 1; pr(q, 1) = S.pairs[q].second + 1; s2.row((Eigen::Index)q) = S.S2[q].row(0); }
        MatrixXd sg = MatrixXd(S.sigma.transpose()); for (auto& v : sg.reshaped()) if (std::isnan(v)) v = -1;      // NaN = the low-pass has no scale
        MatrixXd sgr = sig; for (auto& v : sgr.reshaped()) if (std::isnan(v)) v = -1;
        const double e = std::max({relErr(MatrixXcd(S.S0), MatrixXcd(S0re.cast<cd>() + cd(0, 1) * S0im.cast<cd>())), relErr(s1, S1), relErr(pr, pairs), relErr(s2, S2), relErr(sg, sgr)});
        EXPECT(e < 1e-10 && (int)S.pairs.size() == P, "scatter " + fam + " (" + std::to_string(bank.size()) + " members, " + std::to_string(P) + " second-order pairs): S0, S1, S2, Sigma, Pairs == Rheome, rel " + sci(e));
        ++cases;
    }
    EXPECT(cases == 3, "three scatter cases (mexhat, itersine, heat)");
    EXPECT(throws([&] { scatter(fb::design(fb::Family::Heat, fb::Range::fromSpectrum(T.lambda)), T, Fc, 3); }), "scatter order 3 is refused");

    std::cout << "=== the kernels' own identities ===\n";
    {   // wave and dampedwave at beta = 0 agree; kleingordon at mu = 0 is the wave; resonator is unity at f = 0 and the peak of |H| sits at f0 (Q = 6: |H(f0)| = Q)
        VectorXd tt = VectorXd::LinSpaced(9, 0, 3);
        EXPECT((dampedwave(lam, tt, 1.1, 0) - wave(lam, tt, 1.1)).cwiseAbs().maxCoeff() < 1e-12 && (kleingordon(lam, tt, 1.1, 0) - wave(lam, tt, 1.1)).cwiseAbs().maxCoeff() < 1e-12, "dampedwave(beta = 0) = kleingordon(m = 0) = wave");
        VectorXd f(3); f << 0, 10, 20;
        const MatrixXcd H = resonator(lam, f, 10, 6);
        EXPECT(std::abs(H(0, 0) - 1.0) < 1e-14 && std::abs(std::abs(H(0, 1)) - 6.0) < 1e-12, "resonator: H(0) = 1 and |H(f0)| = Q");
        VectorXd l0(3); l0 << 0, 1, 4;
        EXPECT(relErr(diffgauss(l0, VectorXd::Constant(1, 0.1), VectorXd::Constant(1, 0.3)), (heat(l0, VectorXd::Constant(1, 0.1), 0) - heat(l0, VectorXd::Constant(1, 0.3), 0)).eval()) < 1e-15, "diffgauss = heat(t1) - heat(t2)");
        EXPECT(throws([&] { diffgauss(l0, VectorXd::Constant(1, 0.3), VectorXd::Constant(1, 0.1)); }) && throws([&] { heat(l0, VectorXd::Constant(1, -1.0), 0); }), "diffgauss t1 >= t2 and heat t < 0 are refused");
        // sphere_kernel: the heat series integrates to the degree-0 gain over the sphere (the mean): ∫K dΩ = g_0
        const int Ng = 4001; VectorXd gm = VectorXd::LinSpaced(Ng, 0, 3.14159265358979323846), glh(30); for (int l = 0; l < 30; ++l) glh(l) = std::exp(-0.3 * l * (l + 1.0));
        const VectorXd K = sphere_kernel(gm, glh); double I = 0; for (int i = 1; i < Ng; ++i) I += 0.5 * (K(i) * std::sin(gm(i)) + K(i - 1) * std::sin(gm(i - 1))) * (gm(i) - gm(i - 1));
        EXPECT(std::abs(2 * 3.14159265358979323846 * I - glh(0)) < 1e-6, "sphere_kernel: ∫ K dΩ = g_0 (" + sci(std::abs(2 * 3.14159265358979323846 * I - glh(0))) + ")");
    }
    std::cout << "=== the ruling's definitions ===\n";
    {   // wave on the sphere: the continuum form IS the exact spectrum omega_l = c sqrt(l(l+1)) / R
        const double R = 0.1, c = 12; VectorXd l = VectorXd::LinSpaced(13, 0, 12), lm2(13), tt = VectorXd::LinSpaced(200, 0, 0.05);
        for (int i = 0; i < 13; ++i) lm2(i) = l(i) * (l(i) + 1) / (R * R);
        MatrixXd ex(13, 200); for (int i = 0; i < 13; ++i) for (int j = 0; j < 200; ++j) ex(i, j) = std::cos(c * std::sqrt(l(i) * (l(i) + 1)) / R * tt(j));
        const double e = (wave(lm2, tt, c) - ex).cwiseAbs().maxCoeff();
        EXPECT(e < 1e-12, "continuum wave == the exact sphere solution cos(t c sqrt(l(l+1))/R): " + sci(e));
        const double eChe = (wave(lm2, tt, c, Variant::Chebyshev, 0) - ex).cwiseAbs().maxCoeff();
        EXPECT(eChe > 0.5, "  ... and the Chebyshev variant is not that dispersion (max deviation " + sci(eChe) + ")");
        // Klein-Gordon: m only through omega^2; the DC mode rings at m; m = 0 is the wave; diffusion / schrodinger are not normalised by lambda_max
        VectorXd lk(3); lk << 0, 4, 100; VectorXd tk = VectorXd::LinSpaced(21, 0, 2);
        const MatrixXd KG = kleingordon(lk, tk, 2.0, 7.0);
        double ek = 0; for (int i = 0; i < 3; ++i) for (int j = 0; j < 21; ++j) ek = std::max(ek, std::abs(KG(i, j) - std::cos(std::sqrt(4.0 * lk(i) + 49.0) * tk(j))));
        EXPECT(ek < 1e-14 && (KG.row(0).transpose() - tk.unaryExpr([](double x) { return std::cos(7.0 * x); })).cwiseAbs().maxCoeff() < 1e-14, "kleingordon: omega^2 = c^2 lambda + m^2, the lambda = 0 mode rings at m");
        VectorXd lk2(4); lk2 << 0, 4, 100, 1e4;
        EXPECT((diffusion(lk2, tk, 0.7).topRows(3) - diffusion(lk, tk, 0.7)).cwiseAbs().maxCoeff() < 1e-15 && (schrodinger(lk2, tk, 2).topRows(3) - schrodinger(lk, tk, 2)).cwiseAbs().maxCoeff() < 1e-15,
               "diffusion and schrodinger do not depend on the largest eigenvalue present (physical units)");
        // travwave: a wave one way and the other separate on a signed axis
        const double cs = 4, k = 5; const double f0 = cs * k / (2 * 3.14159265358979323846);
        VectorXd fa = VectorXd::LinSpaced(401, -2 * f0, 2 * f0), lk3(1); lk3 << k * k;
        const MatrixXd Gp = travwave(lk3, fa, cs, 0.5, 1), Gm = travwave(lk3, fa, cs, 0.5, -1), G0 = travwave(lk3, fa, cs, 0.5, 0);
        Eigen::Index ip, im; Gp.row(0).maxCoeff(&ip); Gm.row(0).maxCoeff(&im);
        EXPECT(std::abs(fa(ip) - f0) < 0.011 * f0 && std::abs(fa(im) + f0) < 0.011 * f0 && [&] { double e = 0; for (int j = 0; j < fa.size(); ++j) if (std::abs(fa(j)) > f0 / 2) e = std::max(e, std::abs(Gp(0, j) + Gm(0, j) - G0(0, j))); return e < 1e-3; }() && throws([&] { travwave(lk3, fa, cs, 0.5, 2); }),
               "travwave: sense +1 peaks at +c sqrt(lambda)/2pi, -1 at -, 0 is |f| (the sum of the two); sense 2 is refused");
        // a synthetic travelling pattern each way: z(t) = exp(+-i 2 pi f0 t) on the mode of lambda = k^2, through the signed grid of an FFT
        const int N = 200; const double fs = 100, f0b = 3.0, cb = 2 * 3.14159265358979323846 * f0b / k;
        VectorXd fb_ = nxr::signal::fftfreq(N, 1.0 / fs); VectorXd lk4(1); lk4 << k * k;
        MatrixXcd Z(1, N), Zl(1, N); for (int j = 0; j < N; ++j) { const double ph = 2 * 3.14159265358979323846 * f0b * j / fs; Z(0, j) = cd(std::cos(ph), std::sin(ph)); Zl(0, j) = std::conj(Z(0, j)); }
        MatrixXcd SR = nxr::signal::fftColumns(MatrixXcd(Z.transpose()), true).transpose(), SL = nxr::signal::fftColumns(MatrixXcd(Zl.transpose()), true).transpose();
        auto share = [&](const MatrixXcd& S, int sense) { const MatrixXd G = travwave(lk4, fb_, cb, 1.0, sense); return (S.array().abs2() * G.array().square()).sum() / S.array().abs2().sum(); };
        EXPECT(share(SR, 1) > 0.99 && share(SR, -1) < 1e-12 && share(SL, -1) > 0.99 && share(SL, 1) < 1e-12 && share(SR, 0) > 0.99 && share(SL, 0) > 0.99,
               "a synthetic wave travelling each way: sense +1 keeps the +f one only, -1 the other, 0 both");
    }
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}
