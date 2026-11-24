// The windowed graph Fourier transform, jft / ijft, a bank on an operator's Chebyshev bound, TimeFilterbank kinds, xwt and wcoherence. Goldens from golden/gen_signal_grid_golden.py (NumPy applied to the
// published formulas of Shuman et al. 2016, GSPBox's jft, Grinsted et al. 2004; synthetic data only).
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/graphtransforms.h"
#include "nxr/signal/jointtransforms.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet_cross.h"
#include "nxr/signal/windowed_graphtransforms.h"
#include "golden_keyvalues.h"
#include "mesh_helpers.h"
#include <Eigen/Sparse>
using namespace nxr::signal;
namespace gfb = nxr::signal::graphfilterbank;
namespace gt = nxr::signal::graphtransforms;

static std::string sci(double v) { std::ostringstream s; s.precision(2); s << std::scientific << v; return s.str(); }
template <class F> static bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
    const KeyValues g(std::string(NXR_TEST_GOLDEN_DIR) + "/signal_grid_golden.txt");
    const int N = 12;
    gt::EigenTransform t;
    t.phi = g.mat("graph.phi", N, N);
    t.lambda = g.vec("graph.lambda");
    const Eigen::MatrixXd Ld = g.mat("graph.L", N, N);

    std::cout << "\n=== windowed graph Fourier transform (Shuman, Ricaud & Vandergheynst 2016) ===\n";
    {
        const gt::Gain kernel = [](double l) { return std::exp(-0.8 * l); };
        const Eigen::VectorXd f = g.vec("wgft.f");
        const Eigen::MatrixXd S = gt::wgft(t, kernel, f);
        const double e = (S - g.mat("wgft.S", N, N)).cwiseAbs().maxCoeff();
        EXPECT(S.rows() == N && S.cols() == N && e < 1e-12, "wgft: S f(i, k) == the paper's atoms M_k T_i g (numpy eigh): max error " + sci(e));
        const Eigen::VectorXd back = gt::iwgft(t, kernel, S);
        const double eb = (back - g.vec("wgft.back")).cwiseAbs().maxCoeff(), er = (back - f).cwiseAbs().maxCoeff();
        EXPECT(eb < 1e-12 && er < 1e-10, "iwgft: the synthesis equals the reference's (" + sci(eb) + ") and recovers the signal (" + sci(er) + ")");
        gt::EigenTransform withMass = t; withMass.mass = Eigen::SparseMatrix<double>(N, N); withMass.mass.setIdentity();
        EXPECT(throws([&] { gt::wgft(withMass, kernel, f); }), "wgft: a non-identity mass is refused");
        EXPECT(throws([&] { gt::wgft(t, kernel, Eigen::VectorXd::Zero(N - 1)); }), "wgft: a signal of the wrong length is refused");
    }
    std::cout << "=== cone of influence vs MATLAB wcoherence ===\n";
    {
        const KeyValues m(std::string(NXR_TEST_GOLDEN_DIR) + "/wcoherence_coi_golden.txt");
        const double fs = m.vec("fs")(0), coiScalar = m.vec("fourier_factor")(0) / m.vec("sigma_t")(0);   // the wavelet's constants, as MATLAB's own function gives them
        for (const char* tag : {"N64", "N63"}) {
            const std::string key = tag;
            const Eigen::Index n = static_cast<Eigen::Index>(m.vec(key + ".n")(0));
            const Eigen::VectorXd freq = m.vec(key + ".freq_hz"), coiHz = m.vec(key + ".coi_hz");
            const Eigen::VectorXd mine = coneOfInfluenceFrequency(n, 1.0 / fs, coiScalar, freq.maxCoeff());
            const double e = mine.size() == coiHz.size() ? ((mine - coiHz).cwiseAbs().array() / coiHz.array()).maxCoeff() : 1e300;
            EXPECT(e < 1e-12, std::string("coneOfInfluenceFrequency (") + std::to_string(n) + " samples): relative error vs the cone wcoherence returns " + sci(e));
            // the mask: a coefficient is inside the cone where its frequency is below the cone's (MATLAB's plot: below the dashed line), i.e. its period is above the cone's
            const Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> inside = insideConeOfInfluence(freq.cwiseInverse(), coneOfInfluenceFrequency(n, 1.0 / fs, coiScalar, freq.maxCoeff()).cwiseInverse());   // the cone as periods, capped at the highest frequency as MATLAB returns it
            long wrong = 0, inCount = 0;
            for (Eigen::Index s = 0; s < freq.size(); ++s)
                for (Eigen::Index t = 0; t < n; ++t) { const bool want = freq(s) < coiHz(t) * (1.0 - 1e-12); wrong += inside(s, t) != want; inCount += inside(s, t); }
            EXPECT(wrong == 0 && inCount > 0 && inCount < inside.size(), std::string("insideConeOfInfluence (") + std::to_string(n) + " samples): " + std::to_string(inCount) + " of " + std::to_string(inside.size()) + " inside, " + std::to_string(wrong) + " differ from MATLAB's cone");
        }
        EXPECT(throws([] { coneOfInfluence(0, 1.0, 1.0); }) && throws([] { coneOfInfluence(8, 0.0, 1.0); }) && throws([] { coneOfInfluenceFrequency(8, 1.0, 1.0, 0.0); }), "coneOfInfluence: bad length, step, constant and cap refused");
    }
    std::cout << "=== jft / ijft (GSPBox: graph Fourier transform, unitary DFT over time) ===\n";
    {
        const int T = 16;
        const Eigen::MatrixXd F = g.mat("jft.F", N, T);
        const Eigen::MatrixXcd Fh = jointtransforms::jft(t, F);
        const double e = (Fh - g.cmat("jft.Fhat", N, T)).cwiseAbs().maxCoeff();
        EXPECT(e < 1e-12, "jft (real signal): max error vs numpy " + sci(e));
        const Eigen::MatrixXcd Fc = g.cmat("jft.Fc", N, T);
        const double ec = (jointtransforms::jft(t, Fc) - g.cmat("jft.Fchat", N, T)).cwiseAbs().maxCoeff();
        EXPECT(ec < 1e-12, "jft (complex signal): max error vs numpy " + sci(ec));
        EXPECT((jointtransforms::ijftReal(t, Fh) - F).cwiseAbs().maxCoeff() < 1e-12 && (jointtransforms::ijft(t, jointtransforms::jft(t, Fc)) - Fc).cwiseAbs().maxCoeff() < 1e-12, "ijft(jft(F)) == F, real and complex");
        EXPECT(std::abs(Fh.cwiseAbs2().sum() - F.cwiseAbs2().sum()) < 1e-10 * F.cwiseAbs2().sum(), "jft is unitary (Parseval)");
    }
    std::cout << "=== a bank on an operator's Chebyshev bound ===\n";
    {
        Eigen::SparseMatrix<double> L = Ld.sparseView(1e-300, 1);
        // a bank on the operator's Chebyshev bound: the transform estimates lambda_max (Lanczos), the bank is designed on [0, that bound]
        struct { gfb::GraphFilterbank bank; gt::ChebyshevOperator transform; } ob{{}, gt::ChebyshevOperator(L, {}, 0.0, 40)};
        ob.bank = gfb::design(gfb::Family::Heat, gfb::Range::fromLmax(ob.transform.lmax()));
        const double lmaxTrue = g.vec("graph.lmax")(0);
        EXPECT(ob.transform.lmax() >= lmaxTrue && ob.transform.lmax() < 1.2 * lmaxTrue, "a bank on the Chebyshev bound: the Chebyshev bound " + std::to_string(ob.transform.lmax()) + " covers the true lambda_max " + std::to_string(lmaxTrue) + " (Lanczos)");
        EXPECT(ob.bank.size() > 0 && std::abs(ob.bank.lmax - ob.transform.lmax()) < 1e-12 && ob.transform.order() == 40, "a bank on the Chebyshev bound: the bank is designed on [0, bound]; the transform has Rheome's default order 40");
        // the polynomial route stands for the exact one: g(L) f by the Chebyshev series vs through the eigenbasis, for the bank's members
        const Eigen::VectorXd f = g.vec("wgft.f");
        double worst = 0;
        const Eigen::VectorXd lam = t.lambda;
        for (int m = 0; m < ob.bank.size(); ++m) {
            const auto& gain = ob.bank.members[m].gain;
            const Eigen::VectorXd poly = gt::chebyshev(ob.transform, gain, Eigen::MatrixXd(f)).col(0);
            Eigen::VectorXd c = t.phi.transpose() * f;
            for (int k = 0; k < c.size(); ++k) c(k) *= gain(lam(k));
            worst = std::max(worst, (poly - t.phi * c).cwiseAbs().maxCoeff());
        }
        EXPECT(worst < 1e-3, "a bank on the Chebyshev bound: the polynomial transform equals the exact eigenbasis filter for every member (max " + sci(worst) + ")");
        struct { gfb::GraphFilterbank bank; gt::ChebyshevOperator transform; } itersine{{}, gt::ChebyshevOperator(L, {}, 0.0, 60)};
        itersine.bank = gfb::design(gfb::Family::LogItersine, gfb::Range::fromLmax(itersine.transform.lmax()));
        EXPECT(itersine.transform.order() == 60 && itersine.bank.kind == "logitersine", "a bank on the Chebyshev bound: the family and the order are the caller's");
    }
    std::cout << "=== TimeFilterbank kinds ===\n";
    {
        using namespace timefilterbank;
        Design d(Family::Itersine, 1.0, Sides::Analytic); d.voices = 2;
        const TimeFilterbank bank = timeBank(d, 1024, 128.0);
        const auto k = kinds(bank);
        bool ok = k.size() == static_cast<std::size_t>(bank.size()) && k.size() >= 3 && k.front() == "lowpass" && k.back() == "highpass";
        for (std::size_t i = 1; ok && i + 1 < k.size(); ++i) ok = k[i] == "band";
        EXPECT(ok, "kinds (itersine, analytic): lowpass, then bands, then highpass: " + std::to_string(k.size()) + " members");
        Design m(Family::Meyer, 1.0, Sides::Analytic);
        const auto km = kinds(timeBank(m, 1024, 128.0));
        EXPECT(km.front() == "lowpass" && km.back() == "highpass", "kinds (Meyer): the scaling member is a low-pass, the top member a high-pass");
        Design two(Family::Itersine, 1.0, Sides::TwoSided); two.voices = 2;
        const auto kt = kinds(timeBank(two, 1024, 128.0));
        int dc = 0; for (auto& s : kt) dc += s == "dc";
        EXPECT(dc == 1, "kinds (two-sided): the DC bin is a member of its own ('dc')");
    }
    std::cout << "=== xwt and wcoherence (Grinsted, Moore & Jevrejeva 2004) ===\n";
    {
        for (const char* name : {"a", "b", "c", "d"}) {
            const std::string p = std::string("wcoh.") + name;
            const Eigen::VectorXd dims = g.vec(p + ".dims");
            const int S = static_cast<int>(dims(0)), n = static_cast<int>(dims(1));
            const Eigen::MatrixXcd Wx = g.cmat(p + ".Wx", S, n), Wy = g.cmat(p + ".Wy", S, n);
            const double ex = (xwt(Wx, Wy) - g.cmat(p + ".xwt", S, n)).cwiseAbs().maxCoeff();
            WaveletCoherenceOptions o; o.dt = dims(2); o.scale = g.vec(p + ".scale"); o.scaleSmoothingSteps = dims(3);
            const Eigen::MatrixXd R = wcoherence(Wx, Wy, o);
            const double er = (R - g.mat(p + ".R2", S, n)).cwiseAbs().maxCoeff();
            EXPECT(ex < 1e-13 && er < 1e-11 && R.minCoeff() >= 0 && R.maxCoeff() <= 1.0 + 1e-12, std::string("case ") + "(max R2 " + std::to_string(R.maxCoeff()) + ") " + name + " (" + std::to_string(S) + " scales x " + std::to_string(n) + " samples, scale smoothing " + std::to_string(dims(3)) + "): xwt " + sci(ex) + ", wcoherence " + sci(er) + ", 0 <= R^2 <= 1");
        }
        const std::vector<Eigen::MatrixXcd> A = {Eigen::MatrixXcd::Constant(3, 2, {1, 1}), Eigen::MatrixXcd::Constant(2, 2, {0, 2})}, B = {Eigen::MatrixXcd::Constant(3, 2, {1, -1}), Eigen::MatrixXcd::Constant(2, 2, {2, 0})};
        const auto X = xwt(A, B);
        EXPECT(X.size() == 2 && std::abs(X[0](0, 0) - std::complex<double>(0, 2)) < 1e-15 && std::abs(X[1](1, 1) - std::complex<double>(0, 4)) < 1e-15, "xwt of two members' blocks: W_x conj(W_y) per member");
        WaveletCoherenceOptions bad; bad.scale = Eigen::VectorXd::Ones(2);
        EXPECT(throws([&] { wcoherence(Eigen::MatrixXcd::Ones(3, 8), Eigen::MatrixXcd::Ones(3, 8), bad); }) && throws([&] { xwt(Eigen::MatrixXcd::Ones(3, 8), Eigen::MatrixXcd::Ones(3, 7)); }), "a scale per row and equal shapes are required");
    }
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
