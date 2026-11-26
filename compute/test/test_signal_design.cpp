// fir1, butter, mscohere, xcorr and istft against SciPy (goldens from golden/gen_signal_design_golden.py: scipy.signal.firwin / butter / coherence / correlate / istft on synthetic data).
#include "nxr/signal/correlation.h"
#include "nxr/signal/filters.h"
#include "nxr/signal/filters_fir_design.h"
#include "nxr/signal/filters_iir.h"
#include "nxr/signal/istft.h"
#include "nxr/signal/spectral.h"
#include "golden_keyvalues.h"
#include "mesh_helpers.h"
using namespace nxr::signal;
using nxr::signal::spectral::Window;
using nxr::signal::spectral::Detrend;
using nxr::signal::spectral::windowOf;

static double maxErr(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return a.size() == b.size() ? (a - b).cwiseAbs().maxCoeff() : 1e9; }
static double maxErr(const Eigen::VectorXcd& a, const Eigen::VectorXcd& b) { return a.size() == b.size() ? (a - b).cwiseAbs().maxCoeff() : 1e9; }
static std::string sci(double v) { std::ostringstream s; s.precision(2); s << std::scientific << v; return s.str(); }
template <class F> static bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
    const KeyValues g(std::string(NXR_TEST_GOLDEN_DIR) + "/signal_design_golden.txt");
    std::cout << "\n=== fir1 vs scipy.signal.firwin ===\n";
    {
        struct C { const char* name; int order; std::vector<double> wn; Band type; };
        const std::vector<C> cases = {{"low", 30, {0.3}, Band::Lowpass}, {"high", 30, {0.4}, Band::Highpass}, {"bandpass", 40, {0.2, 0.5}, Band::Bandpass},
                                      {"stop", 40, {0.2, 0.5}, Band::Bandstop}, {"low_odd_order", 21, {0.15}, Band::Lowpass}};
        double worst = 0;
        for (const auto& c : cases) {
            const std::size_t taps = c.order + 1;
            const std::pair<const char*, Window> wins[] = {{"hamming", Window::Hamming}, {"hann", Window::Hann}, {"blackman", Window::Blackman}};
            for (const auto& w : wins) {
                const Eigen::VectorXd win = windowOf(taps, w.second, false);
                const double e1 = maxErr(fir1(c.order, c.wn, c.type, win), g.vec(std::string("fir1.") + c.name + "." + w.first));
                worst = std::max(worst, e1);
                if (w.second == Window::Hamming) worst = std::max(worst, maxErr(fir1(c.order, c.wn, c.type), g.vec(std::string("fir1.") + c.name + ".hamming")));   // the default window is Hamming
            }
        }
        EXPECT(worst < 1e-14, "fir1: low, high, band-pass, band-stop; Hamming (default), Hann, Blackman: max tap error " + sci(worst) + " < 1e-14");
        const double e = maxErr(fir1(40, {0.2, 0.5}, Band::Bandpass, nxr::signal::filters::kaiser(41, 5.0), false), g.vec("fir1.bandpass.noscale.kaiser5"));
        EXPECT(e < 1e-14, "fir1: a Kaiser window, unscaled ('noscale'): error " + sci(e));
        const Eigen::VectorXd h = fir1(30, {0.3}, Band::Lowpass);
        EXPECT(std::abs(h.sum() - 1.0) < 1e-13 && (h - h.reverse()).cwiseAbs().maxCoeff() < 1e-15, "fir1: unit DC gain, symmetric (linear phase)");
        EXPECT(throws([] { fir1(31, {0.4}, Band::Highpass); }), "fir1: a high-pass of odd order is refused");
        EXPECT(throws([] { fir1(20, {1.2}, Band::Lowpass); }) && throws([] { fir1(20, {0.5, 0.2}, Band::Bandpass); }) && throws([] { fir1(20, {0.3, 0.5}, Band::Lowpass); }), "fir1: bad edges refused");
    }
    std::cout << "=== butter vs scipy.signal.butter ===\n";
    {
        struct C { const char* name; int n; std::vector<double> wn; Band type; };
        const std::vector<C> cases = {{"low", 4, {0.25}, Band::Lowpass}, {"high", 5, {0.4}, Band::Highpass}, {"bandpass", 3, {0.2, 0.45}, Band::Bandpass},
                                      {"stop", 4, {0.15, 0.5}, Band::Bandstop}, {"low1", 1, {0.1}, Band::Lowpass}, {"low8", 8, {0.3}, Band::Lowpass},
                                      {"high1", 1, {0.35}, Band::Highpass}, {"high6", 6, {0.15}, Band::Highpass}, {"bandpass1", 1, {0.2, 0.6}, Band::Bandpass}, {"bandpass6", 6, {0.1, 0.3}, Band::Bandpass},
                                      {"stop1", 1, {0.3, 0.6}, Band::Bandstop}, {"stop5", 5, {0.25, 0.4}, Band::Bandstop}, {"low7", 7, {0.05}, Band::Lowpass}, {"high9", 9, {0.8}, Band::Highpass}};
        for (const auto& c : cases) {
            const TransferFunction tf = butter(c.n, c.wn, c.type);
            const Eigen::VectorXd b = g.vec(std::string("butter.") + c.name + ".b"), a = g.vec(std::string("butter.") + c.name + ".a");
            const double eb = maxErr(tf.b, b) / b.cwiseAbs().maxCoeff(), ea = maxErr(tf.a, a) / a.cwiseAbs().maxCoeff();
            EXPECT(eb < 1e-11 && ea < 1e-11 && tf.a(0) == 1.0, std::string("butter ") + c.name + " (order " + std::to_string(c.n) + "): relative error b " + sci(eb) + ", a " + sci(ea) + " < 1e-11");

            // the zeros, poles and gain, in scipy's order (output = 'zpk')
            const ZeroPoleGain zpk = butterZpk(c.n, c.wn, c.type);
            const Eigen::VectorXcd zg = g.cvec(std::string("butter.") + c.name + ".z"), pg = g.cvec(std::string("butter.") + c.name + ".p");
            double ez = 0, ep = 0;
            bool sizes = static_cast<Eigen::Index>(zpk.z.size()) == zg.size() && static_cast<Eigen::Index>(zpk.p.size()) == pg.size();
            for (Eigen::Index i = 0; sizes && i < zg.size(); ++i) ez = std::max(ez, std::abs(zpk.z[static_cast<std::size_t>(i)] - zg(i)));
            for (Eigen::Index i = 0; sizes && i < pg.size(); ++i) ep = std::max(ep, std::abs(zpk.p[static_cast<std::size_t>(i)] - pg(i)));
            const double kg = g.vec(std::string("butter.") + c.name + ".k")(0);
            EXPECT(sizes && ez < 1e-12 && ep < 1e-12 && std::abs(zpk.k - kg) < 1e-12 * std::abs(kg), std::string("butterZpk ") + c.name + ": zeros " + sci(ez) + ", poles " + sci(ep) + ", gain " + sci(std::abs(zpk.k - kg) / std::abs(kg)) + " vs scipy output='zpk'");
            // the second-order sections, in scipy's order (output = 'sos', pairing 'nearest')
            const Eigen::MatrixXd sos = butterSos(c.n, c.wn, c.type);
            const Eigen::Index ns = static_cast<Eigen::Index>(g.at(std::string("butter.") + c.name + ".sos").size() / 6);
            const Eigen::MatrixXd sg = g.mat(std::string("butter.") + c.name + ".sos", ns, 6);
            const double es = sos.rows() == ns ? (sos - sg).cwiseAbs().maxCoeff() : 1e300;
            EXPECT(sos.rows() == ns && es < 1e-12, std::string("butterSos ") + c.name + ": " + std::to_string(ns) + " sections, max error " + sci(es) + " vs scipy output='sos'");
        }
        EXPECT(throws([] { butter(0, {0.3}, Band::Lowpass); }) && throws([] { butter(2, {0.3}, Band::Bandpass); }) && throws([] { butter(2, {1.0}, Band::Lowpass); }), "butter: bad order and edges refused");
    }
    std::cout << "=== mscohere vs scipy.signal.coherence ===\n";
    {
        const Eigen::VectorXd x = g.vec("coh.x"), y = g.vec("coh.y");
        struct C { const char* name; int nper, nov, nfft; Detrend det; };
        const std::vector<C> cases = {{"a", 256, 128, 0, Detrend::Constant}, {"b", 200, 150, 256, Detrend::Linear}, {"c", 128, 0, 0, Detrend::None}, {"d", 255, 100, 0, Detrend::Constant}};
        for (const auto& c : cases) {
            const Coherence r = mscohere(x, y, 100.0, g.vec(std::string("coh.") + c.name + ".window"), c.nov, c.nfft, c.det);
            const double ef = maxErr(r.f, g.vec(std::string("coh.") + c.name + ".f")), ec = maxErr(r.Cxy, g.vec(std::string("coh.") + c.name + ".C"));
            EXPECT(ef < 1e-12 && ec < 1e-10, std::string("mscohere case ") + c.name + ": frequencies " + sci(ef) + ", coherence " + sci(ec) + " (nperseg " + std::to_string(c.nper) + ", overlap " + std::to_string(c.nov) + ")");
        }
        EXPECT(throws([&] { mscohere(x, y, 100.0, Eigen::VectorXd::Ones(5000)); }), "mscohere: a segment longer than the signal is refused");
    }
    std::cout << "=== xcorr vs scipy.signal.correlate ===\n";
    {
        const Eigen::VectorXd xr = g.vec("xc.xr"), yr = g.vec("xc.yr"), ys = g.vec("xc.ys");
        const Xcorr a = xcorr(xr, yr);
        const Eigen::VectorXd want = g.vec("xc.real");
        EXPECT(maxErr(a.r, want) < 1e-12 && a.lags(0) == -36 && a.lags(a.lags.size() - 1) == 36, "xcorr (real, equal lengths): raw sums == correlate 'full', lags -36 .. 36");
        const Xcorr s = xcorr(xr, ys);
        const Eigen::VectorXd lw = g.vec("xc.short.lags");
        bool lagsOk = s.lags.size() == lw.size(); for (Eigen::Index i = 0; lagsOk && i < lw.size(); ++i) lagsOk = s.lags(i) == static_cast<int>(lw(i));
        EXPECT(lagsOk && maxErr(s.r, g.vec("xc.short")) < 1e-12, "xcorr (unequal lengths 37, 23): values and lags == correlate / correlation_lags");
        const XcorrComplex c = xcorr(g.cvec("xc.xc"), g.cvec("xc.yc"));
        EXPECT(maxErr(c.r, g.cvec("xc.complex")) < 1e-12, "xcorr (complex): sum x[n+m] conj(y[n]) == correlate");
        const Xcorr au = xcorr(xr, xr);
        EXPECT(maxErr(au.r, g.vec("xc.auto")) < 1e-12, "xcorr (autocorrelation) == correlate(x, x)");
        const Xcorr lim = xcorr(xr, yr, 5);
        EXPECT(lim.lags.size() == 11 && maxErr(lim.r, want.segment(36 - 5, 11)) < 1e-12, "xcorr: maxlag 5 keeps lags -5 .. 5");
        const Xcorr big = xcorr(xr, yr, 40);
        EXPECT(big.lags.size() == 81 && big.r(0) == 0.0 && std::abs(big.r(40) - want(36)) < 1e-12, "xcorr: maxlag beyond the overlap pads zeros (MATLAB)");
        const Xcorr co = xcorr(xr, xr, -1, XcorrScale::Coeff);
        EXPECT(std::abs(co.r(36) - 1.0) < 1e-13 && co.r.cwiseAbs().maxCoeff() <= 1.0 + 1e-13, "xcorr Coeff: the zero-lag autocorrelation is 1");
        const Xcorr bi = xcorr(xr, yr, -1, XcorrScale::Biased), un = xcorr(xr, yr, -1, XcorrScale::Unbiased);
        EXPECT(std::abs(bi.r(36 + 3) - want(36 + 3) / 37.0) < 1e-13 && std::abs(un.r(36 + 3) - want(36 + 3) / 34.0) < 1e-13, "xcorr Biased (/N) and Unbiased (/(N-|m|))");
        EXPECT(throws([&] { xcorr(xr, ys, -1, XcorrScale::Biased); }), "xcorr: a scaled correlation of unequal lengths is refused");
    }
    std::cout << "=== istft vs scipy.signal.istft ===\n";
    {
        for (const char* name : {"a", "b", "c", "d", "e"}) {
            const std::string p = std::string("istft.") + name;
            const Eigen::VectorXd sh = g.vec(p + ".shape");
            const int bins = static_cast<int>(sh(0)), nseg = static_cast<int>(sh(1)), nper = static_cast<int>(sh(2)), nov = static_cast<int>(sh(3)), nfft = static_cast<int>(sh(4)), boundary = static_cast<int>(sh(5));
            Eigen::MatrixXcd Z(bins, nseg);   // the golden is time-major: nseg rows of `bins`
            const Eigen::VectorXcd flat = g.cvec(p + ".Z");
            for (int j = 0; j < nseg; ++j) for (int i = 0; i < bins; ++i) Z(i, j) = flat(j * bins + i);
            const bool psd = std::string(name) == "d";
            const Eigen::VectorXd x = istft(Z, g.vec(p + ".window"), nov, nfft, boundary != 0, psd ? StftScaling::Psd : StftScaling::Spectrum, 2.0);
            const Eigen::VectorXd want = g.vec(p + ".x");
            const double e = maxErr(x, want);
            EXPECT(e < 1e-12, std::string("istft case ") + name + " (nperseg " + std::to_string(nper) + ", overlap " + std::to_string(nov) + ", nfft " + std::to_string(nfft) + (boundary ? ", boundary" : ", no boundary") + (psd ? ", psd scaling" : "") + "): max error " + sci(e));
        }
        const Eigen::VectorXd sh = g.vec("istft.two.shape");
        const int rows = static_cast<int>(sh(0)), nseg = static_cast<int>(sh(1));
        Eigen::MatrixXcd Z(rows, nseg);
        const Eigen::VectorXcd flat = g.cvec("istft.two.Z");
        for (int j = 0; j < nseg; ++j) for (int i = 0; i < rows; ++i) Z(i, j) = flat(j * rows + i);
        const double e = maxErr(istftTwoSided(Z, g.vec("istft.two.window"), 32, true), g.cvec("istft.two.x"));
        EXPECT(e < 1e-12, "istft (two-sided, complex signal): max error " + sci(e));
        EXPECT(throws([&] { istft(Eigen::MatrixXcd::Zero(10, 3), Eigen::VectorXd::Ones(8), 4, 16); }), "istft: bins that do not fit nfft are refused");
    }
    std::cout << (g_failures ? "\nFAILURES\n" : "\nALL PASSED\n");
    return g_failures ? 1 : 0;
}
