// nxr::welch: parity with scipy.signal.welch, nxr (qc.sensors.psd and spectral.chain._welch) and Brainstorm's bst_psd output (goldens from golden/gen_welch_golden.py
// and .m, synthetic signals), chunk-size invariance (identical to the last bit), the known levels (a sine of amplitude A, white noise of variance s^2),
// a bad span left out, and the refusals.
#include "nxr/welch.h"
#include "nxr/rotary.h"
#include "golden_io.h"
#include "mesh_helpers.h"
#include <random>
#include <optional>
using namespace nxr::welch;
using Eigen::MatrixXd;
using Eigen::VectorXd;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static constexpr double kPi = 3.14159265358979323846;

// max |a - b| / max |b| per record (a row of the PSD is one channel)
static double rowRel(const MatrixXd& a, const MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e9;
    double worst = 0;
    for (Eigen::Index i = 0; i < a.rows(); ++i) worst = std::max(worst, (a.row(i) - b.row(i)).cwiseAbs().maxCoeff() / b.row(i).cwiseAbs().maxCoeff());
    return worst;
}
template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

static void parity() {
    std::cout << "=== parity with scipy.signal.welch, nxr and Brainstorm (golden/welch_golden.txt) ===\n";
    Golden g(kDir + "/welch_golden.txt");
    EXPECT(g.good(), "welch golden opens");
    const MatrixXd x = g.mat("x");
    const MatrixXd gm = g.mat("good");
    const double fs = g.mat("fs")(0, 0);
    std::vector<std::uint8_t> good(static_cast<std::size_t>(gm.cols()));
    for (std::size_t i = 0; i < good.size(); ++i) good[i] = gm(0, static_cast<Eigen::Index>(i)) > 0.5;
    auto check = [&](const char* name, const MatrixXd& got, double tol) {
        const MatrixXd want = g.mat(name);
        const double e = rowRel(got, want);
        EXPECT(e < tol, std::string(name) + " rel err " + sci(e));
        std::cout << "  " << name << ": " << sci(e) << "\n";
    };
    // scipy.signal.welch defaults (Hann periodic, constant detrend, density) = nxr's qc.sensors.psd with n = 4 s
    WelchOptions o = scipyWelchOptions(fs, 400);
    check("scipy_hann_const_n400", welch(x, o).psd, 1e-12);
    {
        const WelchResult r = welch(x, o, &good);
        check("scipy_hann_const_n400_bad", r.psd, 1e-12);
        EXPECT(r.segments == static_cast<std::int64_t>(g.mat("scipy_hann_const_n400_bad_segments")(0, 0)), "segments left in");
    }
    o = scipyWelchOptions(fs, 301); o.overlap = 100.0 / 301; o.window = Window::Hamming; o.detrend = Detrend::Linear;
    check("scipy_hamming_linear_n301_ov100", welch(x, o).psd, 1e-12);
    o = scipyWelchOptions(fs, 256); o.overlap = 64.0 / 256; o.window = Window::Blackman; o.detrend = Detrend::None; o.scaling = Scaling::Spectrum;
    check("scipy_blackman_none_spectrum_n256_ov64", welch(x, o).psd, 1e-12);
    o = scipyWelchOptions(fs, nxr::welch::segmentSamplesFor(4.0, fs));
    check("nxr_qc_psd", welch(x, o).psd, 1e-12);
    // nxr.spectral.chain._welch (2 s segments, no detrend, per-bin chain scaling)
    o = nxrChainWelchOptions(fs, 2.0);
    {
        const WelchResult r = welch(x, o);
        check("nxr_chain", r.psd, 1e-12);
        EXPECT(r.segments == static_cast<std::int64_t>(g.mat("nxr_chain_segments")(0, 0)), "nxr segments");
        const WelchResult b = welch(x, o, &good);
        check("nxr_chain_bad", b.psd, 1e-12);
        EXPECT(b.segments == static_cast<std::int64_t>(g.mat("nxr_chain_bad_segments")(0, 0)), "nxr segments with the bad span");
    }
    // Brainstorm's defaults (bst_psd: Hamming, mean removed, power per Hz, 50 %): its output
    auto defaults = [&](double sec, const char* name) {
        const WelchResult r = welch(x, brainstormPsdOptions(fs, sec));
        check(name, r.psd, 1e-12);
        EXPECT(r.segments == static_cast<std::int64_t>(g.mat(std::string(name) + "_windows")(0, 0)), std::string(name) + " window count");
    };
    defaults(2.0, "bst_default_2s");
    {
        const WelchResult r = welch(x, brainstormPsdOptions(fs, 2.0), &good);
        check("bst_bad_1", r.psd, 1e-12);
        EXPECT(r.segments == static_cast<std::int64_t>(g.mat("bst_bad_1_windows")(0, 0)), "Brainstorm skips the same windows");
    }
    defaults(1.0, "bst_default_1s");
}

static void chunkInvariance() {
    std::cout << "=== the result does not depend on the chunk size ===\n";
    std::mt19937 rng(5);
    std::normal_distribution<double> nd;
    const int C = 4, T = 5000;
    MatrixXd x(C, T);
    for (auto& v : x.reshaped()) v = nd(rng) + 3;
    std::vector<std::uint8_t> good(T, 1);
    for (int i = 1700; i < 2100; ++i) good[i] = 0;
    for (int i = 4000; i < 4003; ++i) good[i] = 0;
    for (const bool withMask : {false, true}) {
        for (const auto& o : {brainstormPsdOptions(100.0, 2.0), nxrChainWelchOptions(100.0, 1.5), scipyWelchOptions(100.0, 301)}) {
            const WelchResult whole = welch(x, o, withMask ? &good : nullptr);
            for (int chunk : {1, 7, 100, 199, 300, 301, 1024, 4999, 5000}) {
                Welch w(C, o);
                for (int s = 0; s < T; s += chunk) {
                    const int m = std::min(chunk, T - s);
                    if (withMask) w.add(x.middleCols(s, m), std::vector<std::uint8_t>(good.begin() + s, good.begin() + s + m)); else w.add(x.middleCols(s, m));
                }
                const WelchResult r = w.result();
                EXPECT(r.segments == whole.segments && r.psd == whole.psd, "chunk " + std::to_string(chunk) + (withMask ? " (masked)" : "") + ": identical, " + sci((r.psd - whole.psd).cwiseAbs().maxCoeff()));
            }
        }
    }
}

static void levels() {
    std::cout << "=== known levels ===\n";
    const double fs = 250.0;
    const int T = 25000;
    const std::size_t n = 500;       // 2 s: bins every 0.5 Hz
    MatrixXd x(2, T);
    std::mt19937 rng(9);
    std::normal_distribution<double> nd;
    for (int t = 0; t < T; ++t) { x(0, t) = 3.0 * std::sin(2 * kPi * 12.5 * t / fs); x(1, t) = 2.0 * nd(rng); }
    WelchOptions o = scipyWelchOptions(fs, n);
    const WelchResult r = welch(x, o);
    EXPECT(r.frequencies.size() == 251 && std::abs(r.frequencies[1] - 0.5) < 1e-12 && std::abs(r.frequencies[250] - 125.0) < 1e-12, "frequencies k fs/n up to Nyquist");
    // the power of a sine (A^2/2) is the PSD's integral; a bin-centred sine puts it in 3 bins of a Hann window (periodic)
    const double power = r.psd.row(0).sum() * (fs / static_cast<double>(n));
    EXPECT(std::abs(power - 4.5) < 1e-9, "sine of amplitude 3: integral of the PSD = A^2/2, got " + std::to_string(power));
    Eigen::Index kmax; r.psd.row(0).maxCoeff(&kmax);
    EXPECT(kmax == 25, "peak at 12.5 Hz");
    // white noise of variance s^2 has a one-sided density 2 s^2 / fs (interior bins): within the scatter of 100 segments
    const double level = r.psd.row(1).segment(5, 240).mean();
    EXPECT(std::abs(level / (2 * 4.0 / fs) - 1) < 0.03, "white noise level 2 s^2 / fs, got ratio " + std::to_string(level / (2 * 4.0 / fs)));
    // 'Spectrum' scaling: the peak of a bin-centred sine is A^2/2 with a rectangular window
    WelchOptions rect = scipyWelchOptions(fs, n); rect.window = Window::Rectangular; rect.scaling = Scaling::Spectrum;
    EXPECT(std::abs(welch(x, rect).psd(0, 25) - 4.5 * 1.0) < 1e-9 * 4.5 * 2, "spectrum scaling: the sine's peak bin is A^2/2 (one-sided, rectangular window)");
    // a bad span: the result equals the average over the segments that miss it
    std::vector<std::uint8_t> good(T, 1);
    for (int i = 6000; i < 6100; ++i) good[i] = 0;
    MatrixXd y = x;
    y.middleCols(6000, 100).setConstant(1e6);   // garbage in the bad span must not matter
    const WelchResult clean = welch(x, o, &good), junk = welch(y, o, &good);
    EXPECT(clean.psd == junk.psd && clean.segments == junk.segments && clean.segmentsSkipped >= 1, "samples under the mask are never read into the estimate");
    const WelchResult none = welch(x, o);
    EXPECT(clean.segments + clean.segmentsSkipped == none.segments, "segments left in + left out = the segments of the grid");
}

// THE ROTARY SPECTRUM: the two-sided Welch PSD of z = a_x . X + i a_y . X (nxr.timefreq.psd of the complex series; golden/gen_rotary_welch_golden.py, synthetic signals):
// parity with scipy, +f forward and -f backward, chunk-size independence to the last bit, a bad span left out.
static void rotary() {
    std::cout << "=== the rotary spectrum: the two-sided PSD of a complex pair (golden/rotary_welch_golden.txt) ===\n";
    Golden g(kDir + "/rotary_welch_golden.txt");
    EXPECT(g.good(), "rotary golden opens");
    const MatrixXd X = g.mat("X"), A = g.mat("A");
    const double fs = g.mat("fs")(0, 0);
    const Eigen::MatrixXcd z = nxr::rotary::pairSeries(A, X);
    auto check = [&](const char* fname, const char* name, const ComplexWelchResult& r) {
        const MatrixXd f = g.mat(fname), want = g.mat(name);
        EXPECT((r.frequencies.transpose() - f.row(0)).cwiseAbs().maxCoeff() < 1e-9, std::string(fname) + " the signed frequencies");
        const double e = rowRel(r.psd, want);
        EXPECT(e < 1e-12, std::string(name) + " rel err " + sci(e));
        std::cout << "  " << name << ": " << sci(e) << "\n";
    };
    WelchOptions o = scipyWelchOptions(fs, 400);
    check("rotary_f_n400", "rotary_hann_const_n400", complexWelch(z, o));
    o = scipyWelchOptions(fs, 301); o.overlap = 100.0 / 301; o.window = Window::Hamming; o.detrend = Detrend::Linear;   // odd: no -fs/2 bin
    check("rotary_f_n301", "rotary_hamming_linear_n301_ov100", complexWelch(z, o));
    // the sense: the peak of a forward pair (x = cos, y = sin) at +f, a backward pair at -f
    {
        const int T = 4000;
        Eigen::MatrixXd P(2, T);
        for (int t = 0; t < T; ++t) { const double ph = 2 * kPi * 10.0 * t / fs; P(0, t) = std::cos(ph); P(1, t) = std::sin(ph); }
        MatrixXd I = MatrixXd::Identity(2, 2);
        WelchOptions w = scipyWelchOptions(fs, 400);
        const ComplexWelchResult fwd = complexWelch(nxr::rotary::pairSeries(I, P), w);
        Eigen::Index k; fwd.psd.row(0).maxCoeff(&k);
        EXPECT(std::abs(fwd.frequencies[k] - 10.0) < 1e-9, "a forward-rotating pair peaks at +f");
        MatrixXd back = P; back.row(1) *= -1;
        const ComplexWelchResult bwd = complexWelch(nxr::rotary::pairSeries(I, back), w);
        bwd.psd.row(0).maxCoeff(&k);
        EXPECT(std::abs(bwd.frequencies[k] + 10.0) < 1e-9, "a backward-rotating pair peaks at -f");
        EXPECT(fwd.frequencies[0] == -fs / 2, "the signed line starts at -fs/2");
    }
    // chunk size does not matter: identical to the last bit; a bad span is never read
    {
        const Eigen::Index T = z.cols();
        const WelchOptions w = scipyWelchOptions(fs, 256);
        const ComplexWelchResult whole = complexWelch(z, w);
        for (Eigen::Index chunk : {Eigen::Index(1), Eigen::Index(97), Eigen::Index(700), T}) {
            ComplexWelch cw(1, w);
            for (Eigen::Index s = 0; s < T; s += chunk) cw.add(z.middleCols(s, std::min(chunk, T - s)));
            const ComplexWelchResult r = cw.result();
            EXPECT(r.psd == whole.psd && r.segments == whole.segments, "chunk " + std::to_string(chunk) + ": the same to the last bit");
        }
        std::vector<std::uint8_t> good(static_cast<std::size_t>(T), 1);
        for (int i = 900; i < 1250; ++i) good[static_cast<std::size_t>(i)] = 0;
        Eigen::MatrixXcd junk = z;
        for (int i = 900; i < 1250; ++i) junk(0, i) = std::complex<double>(1e9, -1e9);
        const ComplexWelchResult clean = complexWelch(z, w, &good), dirty = complexWelch(junk, w, &good);
        EXPECT(clean.psd == dirty.psd && clean.segmentsSkipped >= 1 && clean.segments + clean.segmentsSkipped == whole.segments, "samples under the mask are never read");
    }
    EXPECT(throws([&] { nxr::rotary::pairSeries(MatrixXd::Zero(3, 4), MatrixXd::Zero(4, 9)); }), "a pair is two rows");
    EXPECT(throws([&] { nxr::rotary::pairSeries(MatrixXd::Zero(2, 3), MatrixXd::Zero(4, 9)); }), "the rows act on the data's channels");
    EXPECT(throws([&] { ComplexWelch(1, [] { WelchOptions b = scipyWelchOptions(100.0, 100); b.overlap = 1.0; return b; }()); }), "overlap 1");
}

// THE SEGMENT GRID as a pure rule: tiles (2^k samples, the nearest level; a power-of-two width; an explicit level), Brainstorm's and scipy's round, nxr's even; and its refusals
static void segments() {
    std::cout << "=== segmentFor: the segment rules ===\n";
    using O = std::optional<double>; using S = std::optional<std::int64_t>; using L = std::optional<int>;
    auto at = [](const char* rule, O sec, S n, L k, double fs = 1200.0, double ov = 0.5) { return segmentFor(rule, sec, n, k, fs, ov); };
    EXPECT(at("tile", O{}, S{}, L{}).samples == 4096 && at("tile", O{}, S{}, L{}).level == 12, "4 s at 1200 Hz is level 12 (4096 samples, 3.41 s)");
    EXPECT(at("tile", O{2.0}, S{}, L{}, 100.0).samples == 256 && at("tile", O{2.0}, S{}, L{}, 100.0).level == 8, "2 s at 100 Hz is level 8");
    EXPECT(at("tile", O{}, S{128}, L{}).level == 7 && at("tile", O{}, S{}, L{9}).samples == 512, "a power-of-two width, an explicit level");
    EXPECT(at("round", O{2.0}, S{}, L{}, 100.0).samples == 200 && at("round", O{2.0}, S{}, L{}, 100.0).level == 0, "round: round(seconds fs) samples, no grid");
    EXPECT(at("round", O{}, S{301}, L{}).samples == 301 && at("even", O{2.0}, S{}, L{}, 100.5).samples == 202 && at("even", O{2.0}, S{}, L{}, 100.0).samples == 200, "samples win; even makes the count even");
    EXPECT(throws([&] { at("tile", O{}, S{200}, L{}); }), "a width that is no tile");
    EXPECT(throws([&] { at("tile", O{}, S{128}, L{7}); }), "level and samples are alternatives");
    EXPECT(throws([&] { at("tile", O{}, S{}, L{0}); }) && throws([&] { at("tile", O{}, S{}, L{25}); }), "a level is 1 to 24");
    EXPECT(throws([&] { at("tile", O{2.0}, S{}, L{}, 100.0, 0.3); }), "a hop that is no tile");
    EXPECT(at("tile", O{2.0}, S{}, L{}, 100.0, 0.75).samples == 256, "overlap 3/4 keeps the hop a tile");
    EXPECT(throws([&] { at("round", O{2.0}, S{}, L{8}); }), "a level with another rule");
    EXPECT(throws([&] { at("nope", O{2.0}, S{}, L{}); }) && throws([&] { at("tile", O{-1.0}, S{}, L{}); }), "an unknown rule, a negative duration");
}

static void refusals() {
    std::cout << "=== refusals ===\n";
    WelchOptions o = scipyWelchOptions(100.0, 100);
    EXPECT(throws([&] { Welch(0, o); }), "no channels");
    WelchOptions bad = o; bad.overlap = 1.0;
    EXPECT(throws([&] { Welch(1, bad); }), "overlap 1");
    bad = o; bad.segmentSamples = 1;
    EXPECT(throws([&] { Welch(1, bad); }), "a one-sample segment");
    bad = o; bad.sfreq = 0;
    EXPECT(throws([&] { Welch(1, bad); }), "no sampling frequency");
    Welch w(2, o);
    EXPECT(throws([&] { w.add(MatrixXd::Zero(3, 10)); }), "a chunk with the wrong channels");
    EXPECT(throws([&] { w.add(MatrixXd::Zero(2, 10), std::vector<std::uint8_t>(9, 1)); }), "a mask of the wrong length");
    const WelchResult empty = welch(MatrixXd::Zero(2, 50), o);
    EXPECT(empty.segments == 0 && empty.psd.cols() == 51 && empty.psd.isZero(), "a record shorter than a segment: no segment, a zero PSD");
}

static void linearMap() {
    std::cout << "=== the Welch of a linear map (LinearWelch) ===\n";
    std::mt19937 rng(11);
    std::normal_distribution<double> nd;
    const int C = 6, T = 1700, rows = 12;
    MatrixXd x(C, T), A(rows, C);
    for (auto& v : x.reshaped()) v = nd(rng);
    for (auto& v : A.reshaped()) v = nd(rng);
    std::vector<std::uint8_t> good(T, 1);
    for (int i = 700; i < 760; ++i) good[i] = 0;                // a bad stretch: the segments touching it are left out
    for (const char* which : {"brainstorm", "scipy-odd"}) {
        WelchOptions o = std::string(which) == "brainstorm" ? brainstormPsdOptions(100.0, 2.0) : scipyWelchOptions(100.0, 201);   // an even and an odd segment length
        const WelchResult direct = welch(A * x, o, &good);
        const std::size_t first = 3, last = 40;
        for (std::size_t group : {std::size_t(1), std::size_t(3)}) {
            LinearWelch lw(A, group, o, first, last);
            for (int t = 0; t < T; t += 333) {
                const int n = std::min(333, T - t);
                lw.add(x.middleCols(t, n), std::vector<std::uint8_t>(good.begin() + t, good.begin() + t + n));
            }
            const LinearWelchResult r = lw.result();
            MatrixXd want(rows / static_cast<int>(group), last - first + 1);
            for (int o2 = 0; o2 < want.rows(); ++o2) {
                want.row(o2).setZero();
                for (std::size_t g = 0; g < group; ++g) want.row(o2) += direct.psd.block(o2 * static_cast<int>(group) + static_cast<int>(g), first, 1, last - first + 1);
            }
            const double e = (r.psd - want).cwiseAbs().maxCoeff() / want.cwiseAbs().maxCoeff();
            EXPECT(r.segments == direct.segments && r.segmentsSkipped == direct.segmentsSkipped && direct.segmentsSkipped > 0, std::string(which) + ", " + std::to_string(group) + " row(s) an output: the same segments as Welch on A x");
            EXPECT(e < 1e-12 && r.psd.cols() == static_cast<Eigen::Index>(last - first + 1) && std::abs(r.frequencies[0] - direct.frequencies[first]) < 1e-12, std::string(which) + ", " + std::to_string(group) + " row(s) an output: equal to Welch on A x over the bins kept, " + sci(e));
            // independent of the chunk size: another cutting gives the same sums to rounding
            LinearWelch lw2(A, group, o, first, last);
            lw2.add(x, good);
            EXPECT((lw2.result().psd - r.psd).cwiseAbs().maxCoeff() / r.psd.cwiseAbs().maxCoeff() < 1e-13, std::string(which) + ", " + std::to_string(group) + ": independent of the chunk size");
        }
    }
    // the whole spectrum with the Nyquist bin and DC (the one-sided doubling): bins 0 .. n/2
    {
        WelchOptions o = scipyWelchOptions(100.0, 100);
        const WelchResult direct = welch(A * x, o);
        LinearWelch lw(A, 1, o, 0, 50);
        lw.add(x);
        EXPECT((lw.result().psd - direct.psd).cwiseAbs().maxCoeff() / direct.psd.cwiseAbs().maxCoeff() < 1e-12, "bins 0 .. Nyquist: DC and Nyquist not doubled, the rest doubled, as Welch");
    }
    // refusals
    WelchOptions o = scipyWelchOptions(100.0, 100);
    EXPECT(throws([&] { LinearWelch(A, 5, o, 0, 10); }), "rows not a multiple of the group");
    EXPECT(throws([&] { LinearWelch(A, 1, o, 10, 4); }) && throws([&] { LinearWelch(A, 1, o, 0, 51); }), "bins out of order or outside the spectrum");
    LinearWelch lw(A, 1, o, 0, 10);
    EXPECT(throws([&] { lw.add(MatrixXd::Zero(3, 10)); }), "a chunk with the wrong channels");
}

int main() {
    parity();
    chunkInvariance();
    levels();
    rotary();
    linearMap();
    segments();
    refusals();
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}
