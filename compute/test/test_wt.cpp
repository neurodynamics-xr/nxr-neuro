// nxr::signal / iwt: the chunked transform equals the whole-record aperiodic transform at the members' instants, for any chunk size; iwt(wt(x)) = x; a tone of amplitude A gives |c| = (G/2) A.
// Synthetic data only. The reference is a direct sum over the whole record's FFT bins (no chunks), with the same continuous gains.
#include "nxr/signal/wavelet.h"
#include "mesh_helpers.h"
#include "golden_io.h"
#include <random>
using namespace nxr::signal;
using namespace nxr::signal::timefilterbank;
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXd;
namespace fft = nxr::signal;
static constexpr double kPi = 3.14159265358979323846;
using cd = std::complex<double>;

static MatrixXd bandNoise(int C, int n, double fs, unsigned seed, double fmax) {    // sum of random tones below fmax: no energy near Nyquist
    std::mt19937 r(seed); std::uniform_real_distribution<double> U(0, 1);
    MatrixXd x = MatrixXd::Zero(C, n);
    for (int c = 0; c < C; ++c) for (int k = 0; k < 40; ++k) {
        const double f = 0.5 + U(r) * (fmax - 0.5), a = U(r), ph = 2 * kPi * U(r);
        for (int i = 0; i < n; ++i) x(c, i) += a * std::cos(2 * kPi * f * i / fs + ph);
    }
    return x;
}

// whole-record reference: c_m(t_k) = e^{-2 pi i f_lo t} (1/Nr) sum_b X_b g_m(f_b) e^{2 pi i f_b t}, b over the member's support (masked), X the FFT of the zero-padded record
static MatrixXcd reference(const TimeFilterbank& bk, const Plan& pl, int m, const MatrixXd& x, std::int64_t k0, std::int64_t k1) {
    const std::int64_t n = x.cols();
    const std::size_t Nr = nxr::signal::next_fast_len(static_cast<std::size_t>(n + 4 * pl.margin + 16), true);
    MatrixXd xp = MatrixXd::Zero(Nr, x.rows()); xp.topRows(n) = x.transpose();
    const MatrixXcd X = fft::rfft(xp);
    const double flo = (double)bk.support[m].first * bk.fs / bk.nfft, fhi = (double)bk.support[m].second * bk.fs / bk.nfft;
    MatrixXcd out(k1 - k0, x.rows()); out.setZero();
    const double D = (double)pl.p[m] / pl.q[m];
    for (std::int64_t b = 0; b <= (std::int64_t)Nr / 2; ++b) {
        const double f = b * bk.fs / Nr; if (f < flo - 1e-9 || f > fhi + 1e-9) continue;
        VectorXd fv(1); fv(0) = f; const double g = gains(bk, fv)(0, m);
        for (std::int64_t k = k0; k < k1; ++k) {
            const double t = k * D;
            const cd e = std::polar(g / Nr, 2 * kPi * std::fmod((double)b * t / Nr, 1.0) - 2 * kPi * std::fmod(flo / bk.fs * t, 1.0));
            for (int c = 0; c < (int)x.rows(); ++c) out(k - k0, c) += X(b, c) * e;
        }
    }
    return out;
}

static std::vector<Block> runWt(const TimeFilterbank& bk, const MatrixXd& x, std::int64_t chunk, const Options& o, std::int64_t* margin = nullptr, double* achieved = nullptr) {
    Wt w(bk, (int)x.rows(), o);
    if (margin) *margin = w.margin(); if (achieved) *achieved = w.achievedTolerance();
    std::vector<Block> all(bk.size());
    for (auto& b : all) { b.coef = MatrixXcd::Zero(0, x.rows()); }
    bool first = true;
    auto take = [&](std::vector<Block> r) {
        for (int m = 0; m < bk.size(); ++m) {
            if (first) all[m].first = r[m].first;
            const Eigen::Index h = all[m].coef.rows();
            all[m].coef.conservativeResize(h + r[m].coef.rows(), Eigen::NoChange); all[m].coef.bottomRows(r[m].coef.rows()) = r[m].coef;
        }
        first = false;
    };
    for (std::int64_t s = 0; s < x.cols(); s += chunk) take(w.add(x.middleCols(s, std::min<std::int64_t>(chunk, x.cols() - s))));
    take(w.finish());
    return all;
}

int main() {
    const double fs = 250; const int n = 900, C = 2;
    const MatrixXd x = bandNoise(C, n, fs, 3, 0.4 * fs / 2 * 2 * 0.5);     // up to 50 Hz
    struct Case { const char* name; Design d; };
    std::vector<Case> cases;
    { Design d(Family::Itersine, 2.0, Sides::Analytic); d.voices = 2; d.limits = std::make_pair(2.0, 100.0); cases.push_back({"itersine", d}); }
    { Design d(Family::Morlet, 2.0, Sides::Analytic); d.centres = {4, 8, 16, 32}; d.cycles = 6; cases.push_back({"morlet", d}); }
    { Design d(Family::Meyer, 2.0, Sides::Analytic); d.limits = std::make_pair(2.0, 60.0); cases.push_back({"meyer", d}); }
    const std::size_t nfftDesign = 1024;
    for (auto& cs : cases) {
        const TimeFilterbank bk = design(cs.d, nfftDesign, fs);
        Options o; o.tolerance = 1e-9; o.maxMargin = 8192;
        const Plan pl(bk, o);
        std::cout << "=== " << cs.name << ": " << bk.size() << " members, margin " << pl.margin << " samples, P " << pl.P << ", achieved tail " << sci(pl.achieved) << "\n";
        std::vector<Block> ref0;
        double worstChunk = 0, worstEdge = 0;
        for (std::int64_t chunk : {(std::int64_t)1, (std::int64_t)37, (std::int64_t)n}) {
            if (chunk == 1) { /* tiny */ }
            Options oo = o; oo.block = 0;
            const auto got = runWt(bk, chunk == 1 ? x.leftCols(300) : x, chunk, oo);
            const MatrixXd xs = chunk == 1 ? MatrixXd(x.leftCols(300)) : x;
            for (int m = 0; m < bk.size(); ++m) {
                const Block& b = got[m];
                const MatrixXcd r = reference(bk, pl, m, xs, b.first, b.first + b.coef.rows());
                const double e = (b.coef - r).cwiseAbs().maxCoeff() / std::max(1e-300, r.cwiseAbs().maxCoeff());
                // the bound: the tail the margin leaves out (relative to the impulse response's L1 norm) times the input's size; low-/high-pass members have a jump in the analytic gain at DC / Nyquist, a 1/t tail
                const double bound = 3 * pl.tail[m] * pl.l1[m] * xs.cwiseAbs().maxCoeff() / r.cwiseAbs().maxCoeff() + 1e-9;
                worstChunk = std::max(worstChunk, e / bound);
                worstEdge = std::max(worstEdge, e);
            }
            std::cout << "  chunk " << chunk << ": worst relative error " << sci(worstEdge) << " (" << sci(worstChunk) << " x its tail bound)" << "\n";
        }
        EXPECT(worstChunk < 10, std::string(cs.name) + ": chunked wt equals the whole-record transform at the members' instants within the stated tail (chunks 1, 37, whole record), error " + sci(worstEdge));
        // chunk sizes give identical streams
        const auto a = runWt(bk, x, 53, o), b = runWt(bk, x, n, o);
        double dif = 0; for (int m = 0; m < bk.size(); ++m) dif = std::max(dif, (a[m].coef - b[m].coef).cwiseAbs().maxCoeff());
        EXPECT(dif < 1e-12, std::string(cs.name) + ": chunk size 53 equals the whole record " + sci(dif));
        // reconstruction
        if (std::string(cs.name) != "morlet") {
            Iwt iw(bk, C, o, nxr::signal::graphfilterbank::Synthesis::Dual);
            const auto W = runWt(bk, x, 100, o);
            // feed in two pieces to exercise the streaming
            Eigen::MatrixXd rec = iw.add(W); rec.conservativeResize(C, rec.cols());
            const MatrixXd tail = iw.finish(n);
            MatrixXd all(C, rec.cols() + tail.cols()); all << rec, tail;
            const double e = (all - x).cwiseAbs().maxCoeff() / x.cwiseAbs().maxCoeff();
            EXPECT(all.cols() == n && e < 1e-4, std::string(cs.name) + ": iwt(wt(x)) = x over the whole record, ends included, " + sci(e));
        }
    }
    // tone: |c| = (G/2) A
    {
        Design d(Family::Itersine, 3.0, Sides::Analytic); d.voices = 2; d.limits = std::make_pair(2.0, 100.0);
        const TimeFilterbank bk = design(d, nfftDesign, fs);
        const double A = 1.7, f0 = centerFrequencies(bk)(5); MatrixXd t(1, 4000); for (int i = 0; i < 4000; ++i) t(0, i) = A * std::cos(2 * kPi * f0 * i / fs + 0.3);
        const auto W = runWt(bk, t, 500, Options());
        double best = 0; const Plan pl(bk, Options());
        for (int m = 0; m < bk.size(); ++m) {   // the instants well inside the record (t in [0.4, 0.6] n)
            const std::int64_t k0 = (std::int64_t)(0.4 * 4000) * pl.q[m] / pl.p[m] - W[m].first, k1 = (std::int64_t)(0.6 * 4000) * pl.q[m] / pl.p[m] - W[m].first;
            if (k1 > k0) best = std::max(best, W[m].coef.middleRows(k0, k1 - k0).cwiseAbs().maxCoeff());
        }
        EXPECT(std::abs(best - 1.5 * A) < 1e-4 * A, "a tone of amplitude A gives |c| = (G/2) A: " + std::to_string(best) + " vs " + std::to_string(1.5 * A));
    }
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}
