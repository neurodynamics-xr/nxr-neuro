// nxr::filterbank: parity with Rheome (@graphfilterbank + +graphtransform, goldens from golden/gen_filterbank_golden.m on a synthetic
// icosphere), with the Studio's dyadic frame (golden/gen_meyer_golden.mts), and the frame's own laws (tightness, reconstruction,
// eigenbasis vs Chebyshev). Synthetic data only.
#include "nxr/signal/graphwavelet.h"
#include "nxr/measurement/band_energy.h"
#include "nxr/signal/graphfilterbank.h"
#include "mesh_helpers.h"
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include "nxr/signal/timefilterbank.h"
using namespace nxr::signal::graphfilterbank;
using nxr::signal::timefilterbank::meyerBand;
using namespace nxr::signal::graphtransforms;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static std::string tok(std::istream& is) { std::string t; if (!(is >> t)) { std::cout << "  [FAIL] golden ends early\n"; ++g_failures; std::exit(1); } return t; }
static double num(std::istream& is) { return std::strtod(tok(is).c_str(), nullptr); }
static MatrixXd readMat(std::istream& is, const std::string& name) {
    std::string t = tok(is);
    if (t != name) { std::cout << "  [FAIL] golden: expected '" << name << "', got '" << t << "'\n"; ++g_failures; std::exit(1); }
    int r = (int)num(is), c = (int)num(is);
    MatrixXd A(r, c);
    for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) A(i, j) = num(is);
    return A;
}
static double rel(const MatrixXd& a, const MatrixXd& b) {   // max |a-b| / max(1, max|b|); NaNs must coincide
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e9;
    double d = 0, m = 1;
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        const double x = a.data()[i], y = b.data()[i];
        if (std::isnan(x) != std::isnan(y)) return 1e9;
        if (std::isnan(x)) continue;
        d = std::max(d, std::abs(x - y)); m = std::max(m, std::abs(y));
    }
    return d / m;
}
static std::string sci(double v) { std::ostringstream s; s.precision(2); s << std::scientific << v; return s.str(); }
static MatrixXd stack(const Coefficients& W, int m) { return W[(size_t)m]; }

struct Case { std::string name, wavelet, range; DesignOptions o; };
static Case parseCase(std::istream& is) {
    Case c; c.name = tok(is); c.wavelet = tok(is); c.range = tok(is);
    if (tok(is) != "opts") std::exit(1);
    std::string line; std::getline(is, line);
    std::istringstream ls(line); std::string k;
    while (ls >> k) {
        if (k == "Wavelet") { ls >> k; }
        else if (k == "FineLimit") { ls >> k; c.o.fineLimit = FineLimit::Usable; }
        else if (k == "VoicesPerOctave") { int v; ls >> v; c.o.voices = v; }
        else if (k == "NumFilters") { int v; ls >> v; c.o.numFilters = v; }
        else if (k == "LPFactor") { double v; ls >> v; c.o.lpFactor = v; }
        else if (k == "ScaleLimits") { std::string a, b; ls >> a >> b; c.o.scaleLimits = std::make_pair(std::atof(a.c_str() + 1), std::atof(b.c_str())); }
    }
    return c;
}
// the bank family by name: the library's graphfilterbank::familyOf

int main() {
    std::ifstream f(std::string(NXR_TEST_GOLDEN_DIR) + "/filterbank_golden.txt");
    EXPECT(f.good(), "Rheome golden opens");
    MatrixXd Ld = readMat(f, "L"), Md = readMat(f, "M"), lamG = readMat(f, "lambda"), X = readMat(f, "X");
    Eigen::SparseMatrix<double> L = Ld.sparseView(), M = Md.sparseView();
    const VectorXd lam = lamG.col(0);
    const int n = (int)Ld.rows();

    // the modes, solved here (dense, mass-orthonormal) — must agree with Rheome's spectrum
    Eigen::GeneralizedSelfAdjointEigenSolver<MatrixXd> es(Ld, Md);
    EigenTransform T; T.phi = es.eigenvectors(); T.lambda = es.eigenvalues(); T.mass = M;
    for (int i = 0; i < T.lambda.size(); ++i) if (std::abs(T.lambda(i)) < 1e-10) T.lambda(i) = 0;
    std::cout << "\n=== the synthetic pencil (icosphere, " << n << " vertices) ===\n";
    EXPECT(rel(T.lambda, lam) < 1e-10, "C++ dense modes: spectrum == Rheome's, rel " + sci(rel(T.lambda, lam)));
    EXPECT((T.phi.transpose() * Md * T.phi - MatrixXd::Identity(n, n)).cwiseAbs().maxCoeff() < 1e-10, "Phi is mass-orthonormal");
    EXPECT(std::abs(estimateLmax(L, M) - lam.maxCoeff()) < 1e-3 * lam.maxCoeff(), "estimateLmax == the largest eigenvalue");

    std::string t;
    double worstGain = 0, worstW = 0, worstCheb = 0;
    int banks = 0, chebBanks = 0;
    std::map<std::string, GraphFilterbank> built;
    std::cout << "=== parity with Rheome @graphfilterbank + +graphtransform ===\n";
    while (f >> t && t == "bank") {
        const Case c = parseCase(f);
        const Range range = c.range == "lmax" ? Range::fromLmax(lam.maxCoeff()) : Range::fromSpectrum(lam);
        const GraphFilterbank bank = design(familyOf(c.wavelet), range, c.o);
        built[c.name] = bank;
        const MatrixXd Hg = readMat(f, "H"), tg = readMat(f, "t"), bg = readMat(f, "bounds"), cg = readMat(f, "cwn"), mg = readMat(f, "massLost");
        const VectorXd grid = c.range == "lmax" ? VectorXd::LinSpaced(512, 0.0, lam.maxCoeff()) : lam;
        const double eH = rel(gains(bank, grid), Hg);
        const Bounds b = c.range == "lmax" ? framebounds(bank) : framebounds(bank, lam);
        MatrixXd bm(1, 4); bm << b.A, b.B, b.tightness, b.uncovered;
        const double eB = rel(bm, bg), eT = rel(scales(bank).transpose(), tg), eC = rel(centerWavenumbers(bank).transpose(), cg),
                     eM = rel(massLost(bank).transpose(), mg);
        worstGain = std::max({worstGain, eH, eB, eT, eC, eM});
        if (std::max({eH, eB, eT, eC, eM}) > 1e-11 || bank.size() != Hg.cols())
            { std::cout << "  [FAIL] " << c.name << ": members " << bank.size() << " vs " << Hg.cols() << ", gains " << sci(eH) << ", bounds " << sci(eB)
                        << ", scales " << sci(eT) << ", centres " << sci(eC) << ", massLost " << sci(eM) << "\n"; ++g_failures; }
        ++banks;
        if (c.name == "mexhat" || c.name == "heat" || c.name == "itersine" || c.name == "logitersine") {
            Coefficients W = nxr::signal::gwt(bank, T, X);
            double e = 0;
            for (int m = 0; m < bank.size(); ++m) e = std::max(e, rel(stack(W, m), readMat(f, "W" + std::to_string(m + 1))));
            const MatrixXd Xd = readMat(f, "Xdual"), Xt = readMat(f, "Xtight"), Se = readMat(f, "scaleEnergy"), Me = readMat(f, "memberEnergy");
            const double eD = rel(nxr::signal::igwt(bank, T, W, Synthesis::Dual), Xd), eTi = rel(nxr::signal::igwt(bank, T, W, Synthesis::Tight), Xt),
                         eS = rel(nxr::measurement::scaleEnergy(W), Se), eMe = rel(nxr::measurement::memberEnergy(W), Me);
            worstW = std::max({worstW, e, eD, eTi, eS, eMe});
            EXPECT(std::max({e, eD, eTi, eS, eMe}) < 1e-9, c.name + ": eigenbasis analysis " + sci(e) + ", dual " + sci(eD) + ", tight " + sci(eTi) + ", energies " + sci(std::max(eS, eMe)) + " (rel to Rheome)");
            // Chebyshev on the pencil, same lmax and order as Rheome's T
            std::string ct; f >> ct; // "Wc1" is next: put back by re-reading through readMat with the token already taken
            MatrixXd first(n, X.cols());
            { int r = (int)num(f), cc = (int)num(f); first.resize(r, cc); for (int i = 0; i < r; ++i) for (int j = 0; j < cc; ++j) first(i, j) = num(f); }
            static std::unique_ptr<ChebyshevOperator> cheb;
            if (!cheb) { std::ifstream g(std::string(NXR_TEST_GOLDEN_DIR) + "/filterbank_golden.txt"); std::string s; double lm = 0;
                         while (std::getline(g, s)) if (s.rfind("cheb ", 0) == 0) { lm = std::atof(s.c_str() + s.rfind(' ')); }
                         cheb = std::make_unique<ChebyshevOperator>(L, M, lm, 60); }
            Coefficients Wc = nxr::signal::gwt(bank, *cheb, X);
            double ec = rel(Wc[0], first);
            for (int m = 1; m < bank.size(); ++m) ec = std::max(ec, rel(Wc[(size_t)m], readMat(f, "Wc" + std::to_string(m + 1))));
            worstCheb = std::max(worstCheb, ec); ++chebBanks;
            EXPECT(ec < 1e-9, c.name + ": Chebyshev (order 60, M-pencil) analysis == Rheome's, rel " + sci(ec));
            // and against the exact eigenbasis, member by member (the low-pass weak spot is reported, not hidden)
            double ee = 0; for (int m = 0; m < bank.size(); ++m) ee = std::max(ee, rel(Wc[(size_t)m], W[(size_t)m]));
            // heat (analytic) is exact; mexhat's quartic low-pass and itersine's finite smoothness make the series converge slowly
            // (Rheome's own warning): order 60 on this 42-vertex pencil is within 1e-1, and the error falls with the order.
            EXPECT(ee < 1e-1, c.name + ": Chebyshev (order 60) vs exact eigenbasis, max rel " + sci(ee));
        }
        if (tok(f) != "end") { std::cout << "  [FAIL] golden structure\n"; return 1; }
    }
    EXPECT(banks == 11 && chebBanks == 4 && worstGain < 1e-11, std::to_string(banks) + " banks: gains, bounds, scales, centres, massLost match Rheome, worst rel " + sci(worstGain));
    EXPECT(worstW < 1e-9 && worstCheb < 1e-9, "analysis/synthesis (eigen) worst " + sci(worstW) + ", Chebyshev worst " + sci(worstCheb));

    std::cout << "=== the frame's own laws ===\n";
    for (const char* k : {"itersine", "logitersine"}) {
        const GraphFilterbank& b = built[k];
        const Bounds bd = framebounds(b, lam);
        EXPECT(std::abs(bd.A - 1) < 1e-12 && std::abs(bd.B - 1) < 1e-12 && isframetight(b), std::string(k) + ": tight, A = B = 1 (A " + sci(bd.A - 1) + ", B " + sci(bd.B - 1) + ")");
        Coefficients W = nxr::signal::gwt(b, T, X);
        EXPECT(rel(nxr::signal::igwt(b, T, W, Synthesis::Tight), X) < 1e-12, std::string(k) + ": tight synthesis o analysis = identity");
        EXPECT(rel(nxr::signal::igwt(b, T, W, Synthesis::Dual), X) < 1e-12, std::string(k) + ": dual synthesis o analysis = identity");
        double en = 0; for (int m = 0; m < b.size(); ++m) en += (T.phi.transpose() * (M * W[(size_t)m])).squaredNorm();
        EXPECT(std::abs(en - (T.phi.transpose() * (M * X)).squaredNorm()) < 1e-9 * en, std::string(k) + ": Parseval (member energies add to the field's)");
    }
    for (const char* k : {"mexhat", "heat"}) {
        const GraphFilterbank& b = built[k];
        EXPECT(!isframetight(b) && framebounds(b, lam).A > 0, std::string(k) + ": not tight, but covers the spectrum (A > 0)");
        EXPECT(rel(nxr::signal::igwt(b, T, nxr::signal::gwt(b, T, X), Synthesis::Dual), X) < 1e-11, std::string(k) + ": the canonical dual reconstructs exactly");
        EXPECT(rel(nxr::signal::igwt(b, T, nxr::signal::gwt(b, T, X), Synthesis::Tight), X) > 1e-2, std::string(k) + ": tight synthesis does NOT reconstruct (A != B)");
    }
    {   // eigenbasis vs Chebyshev: heat is smooth, so the series converges fast; error falls with the order
        const GraphFilterbank& b = built["heat"];
        const auto W = nxr::signal::gwt(b, T, X);
        double prev = 1e9; bool mono = true; double last = 0;
        for (int order : {10, 20, 40, 80}) {
            ChebyshevOperator C(L, M, 1.01 * lam.maxCoeff(), order);
            auto Wc = nxr::signal::gwt(b, C, X);
            double e = 0; for (int m = 0; m < b.size(); ++m) e = std::max(e, rel(Wc[(size_t)m], W[(size_t)m]));
            mono = mono && e <= prev * 1.0001; prev = e; last = e;
        }
        EXPECT(mono && last < 1e-10, "heat: Chebyshev error falls with the order, order 80 rel " + sci(last));
        ChebyshevOperator C(L, M, 0, 80);   // lmax estimated
        EXPECT(C.lmax() > lam.maxCoeff() && C.lmax() < 1.05 * lam.maxCoeff(), "ChebyshevOperator: lmax estimated (Lanczos x 1.01) bounds the spectrum");
        const auto Wc = nxr::signal::gwt(b, C, X);
        EXPECT(rel(nxr::signal::igwt(b, C, Wc, Synthesis::Dual), X) < 1e-6, "heat: Chebyshev dual synthesis o analysis ~ identity, rel " + sci(rel(nxr::signal::igwt(b, C, Wc, Synthesis::Dual), X)));
        const GraphFilterbank& li = built["logitersine"];
        ChebyshevOperator C2(L, M, 1.01 * lam.maxCoeff(), 200);
        EXPECT(rel(nxr::signal::igwt(li, C2, nxr::signal::gwt(li, C2, X), Synthesis::Tight), X) < 1e-2, "logitersine: Chebyshev tight reconstruction (order 200), rel " + sci(rel(nxr::signal::igwt(li, C2, nxr::signal::gwt(li, C2, X), Synthesis::Tight), X)));
        // the identity mass: a graph pencil
        ChebyshevOperator C3(L, {}, 0, 80);
        EXPECT(C3.rows() == n && C3.lmax() > 0, "ChebyshevOperator without a mass (identity)");
    }
    {   // complex fields: the operator is real, so real and imaginary parts are independent
        const GraphFilterbank& b = built["logitersine"];
        Eigen::MatrixXcd Z(n, 2); Z.real() = X; Z.imag() = X.rowwise().reverse() * 0.5;
        auto Wz = nxr::signal::gwt(b, T, Z);
        auto Wr = nxr::signal::gwt(b, T, X);
        EXPECT((Wz[1].real() - Wr[1]).cwiseAbs().maxCoeff() < 1e-12 && Wz[1].imag().cwiseAbs().maxCoeff() > 1e-3, "complex field: real part == real analysis, imaginary part carried");
        EXPECT((nxr::signal::igwt(b, T, Wz, Synthesis::Dual) - Z).cwiseAbs().maxCoeff() < 1e-11, "complex field: reconstruction");
        ChebyshevOperator C(L, M, 1.01 * lam.maxCoeff(), 40);
        auto Wzc = nxr::signal::gwt(b, C, Z);
        EXPECT((Wzc[2].real() - nxr::signal::gwt(b, C, X)[2]).cwiseAbs().maxCoeff() < 1e-12, "complex field through Chebyshev: real part independent");
    }
    {   // guards
        auto throws = [](auto fn) { try { fn(); } catch (const std::exception&) { return true; } return false; };
        EXPECT(throws([] { design(Family::MexHat, Range::fromLimits(-1, 5)); }), "a SIGNED range throws");
        EXPECT(throws([] { design(Family::Heat, Range::fromLmax(0)); }), "lmax <= 0 throws");
        EXPECT(throws([&] { DesignOptions o; o.scaleLimits = {2.0, 1.0}; design(Family::MexHat, Range::fromLmax(10), o); }), "scaleLimits not increasing throws");
        EXPECT(throws([&] { ChebyshevOperator bad(L, M, 0.1 * lam.maxCoeff(), 20); }), "Chebyshev lmax below the spectrum throws (it would diverge)");
        EXPECT(throws([] { dyadic({1.0, 1.2}); }), "dyadic: overlapping transitions throw");
        EXPECT(throws([] { dyadic({2.0, 1.0}); }), "dyadic: edges must ascend");
    }

    std::cout << "=== Meyer's dyadic frame vs the Studio (wavelets/dyadicFrame.ts) ===\n";
    {
        std::ifstream g(std::string(NXR_TEST_GOLDEN_DIR) + "/meyer_golden.txt");
        EXPECT(g.good(), "Studio golden opens");
        std::string w; double worst = 0; int frames = 0;
        while (g >> w) {
            if (w == "frame") {
                std::string name = tok(g); int nm = (int)num(g), nx = (int)num(g); double hw = num(g); int ne = (int)num(g);
                tok(g); std::vector<double> edges(ne); for (auto& e : edges) e = num(g);
                tok(g); VectorXd xs(nx); for (int i = 0; i < nx; ++i) xs(i) = num(g);
                MatrixXd want(nx, nm); for (int m = 0; m < nm; ++m) { tok(g); for (int i = 0; i < nx; ++i) want(i, m) = num(g); }
                GraphFilterbank b = name == "weyl" ? weyl(1.8, 6) : name == "rank" ? rank(4) : dyadic(edges, hw);
                const double e = rel(gains(b, xs), want); worst = std::max(worst, e); ++frames;
                const Bounds bd = framebounds(b, xs.cwiseMax(0.0));
                EXPECT(e < 1e-13 && b.size() == nm && std::abs(bd.A - 1) < 1e-12 && std::abs(bd.B - 1) < 1e-12,
                       "meyer '" + name + "': " + std::to_string(nm) + " members, gains == Studio rel " + sci(e) + ", tight (A = B = 1)");
                if (name == "weyl") EXPECT(b.members.front().level == 0 && b.members[3].level == 3 && b.members.back().level == 6, "weyl levels: scaling 0, wavelet j, top = levels");
            } else if (w == "rankcoords") {
                int k = (int)num(g); VectorXd lm(k); for (int i = 0; i < k; ++i) lm(i) = num(g);
                tok(g); MatrixXd m1(1, k), m4(1, k); for (int i = 0; i < k; ++i) m1(0, i) = num(g);
                tok(g); for (int i = 0; i < k; ++i) m4(0, i) = num(g);
                EXPECT(rel(rankCoordinates(lm, 1).transpose(), m1) == 0 && rel(rankCoordinates(lm, 4).transpose(), m4) == 0, "rankCoordinates (mult 1 and 4) == Studio");
            } else if (w == "band") {
                const double xs[] = {0, 0.3, 0.7, 1, 1.4, 2, 3, 100}; double e = 0;
                for (double x : xs) e = std::max(e, std::abs(meyerBand(x, 0.5, 2, 1.0 / 3) - num(g)));
                EXPECT(e < 1e-15, "meyerBand == Studio");
            }
        }
        EXPECT(frames == 3 && worst < 1e-13, "3 Meyer frames, worst rel " + sci(worst));
    }
    {   // ONE member's band (the spatial band of a level): the same matrix as that member of the full analysis, by modes and by the series; the series' error is measurable
        const GraphFilterbank wb = weyl(4 * M_PI * 3, 4);      // an area of 12 pi: edges 1, 2, 4, 8
        const VectorXd lmx = VectorXd::LinSpaced(5, 0.0, 12.0);
        Eigen::SparseMatrix<double> Is(n, n); Is.setIdentity();
        EigenTransform Tw = T;
        const Coefficients all = nxr::signal::gwt(wb, Tw, X);
        const MatrixXd one = nxr::signal::gwt(wb, 2, Tw, X);
        EXPECT((one - all[2]).cwiseAbs().maxCoeff() < 1e-12, "analysisMember (modes) == that member of analysis");
        ChebyshevOperator cop(L, M, 0, 60);
        const MatrixXd oc = nxr::signal::gwt(wb, 2, cop, X);
        EXPECT((oc - nxr::signal::gwt(wb, cop, X)[2]).cwiseAbs().maxCoeff() < 1e-12, "analysisMember (series) == that member of analysis");
        const double e5 = chebyshevError(wb.members[2].gain, cop.lmax(), 5), e60 = chebyshevError(wb.members[2].gain, cop.lmax(), 60), e200 = chebyshevError(wb.members[2].gain, cop.lmax(), 200);
        std::printf("  series error of a Weyl band on [0, %.1f]: order 5 %.3g, 60 %.3g, 200 %.3g\n", cop.lmax(), e5, e60, e200);
        EXPECT(e200 < e60 && e60 < e5, "the series' error falls with its order");
        bool refused = false;
        try { nxr::signal::gwt(wb, 9, Tw, X); } catch (const std::invalid_argument&) { refused = true; }
        EXPECT(refused, "a member outside the bank is refused");
    }
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
