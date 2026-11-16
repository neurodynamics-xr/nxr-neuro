// nxr::filter: parity with Brainstorm's filters (golden/filter_bst_golden.txt from gen_filter_golden.m) on synthetic signals; the frequency response at known points (passband, stopband, the -6 dB / -3 dB points the references define, DC, Nyquist); zero phase / no delay;
// chunk-size invariance of the streaming filter (FIR to rounding, IIR to its stated tolerance); the refusals.
#include "nxr/filter.h"
#include "golden_io.h"
#include "mesh_helpers.h"
#include <random>
using namespace nxr::filter;
using Eigen::MatrixXd;
using Eigen::VectorXd;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static constexpr double kPi = 3.14159265358979323846;

template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
static double maxAbs(const MatrixXd& a, const MatrixXd& b) { return a.rows() == b.rows() && a.cols() == b.cols() ? (a - b).cwiseAbs().maxCoeff() : 1e9; }

static void brainstormParity() {
    std::cout << "=== parity with Brainstorm (golden/filter_bst_golden.txt) ===\n";
    Golden g(kDir + "/filter_bst_golden.txt");
    EXPECT(g.good(), "Brainstorm golden opens");
    const double fs = g.mat("fs")(0, 0);
    const MatrixXd x = g.mat("x");
    const double scale = x.cwiseAbs().maxCoeff();
    struct Case { const char* name; Kind kind; double lo, hi; bool b; bool relax; bool mirror; };
    const Case cases[] = {{"lp30", Kind::Lowpass, 0, 30, true, false, false}, {"hp5", Kind::Highpass, 5, 0, true, false, false}, {"bp1_20", Kind::Bandpass, 1, 20, true, false, false},
                          {"bp8_40", Kind::Bandpass, 8, 40, true, false, false}, {"lp5", Kind::Lowpass, 0, 5, false, false, false}, {"hp0p5", Kind::Highpass, 0.5, 0, false, false, false},
                          {"bp1_20_relax", Kind::Bandpass, 1, 20, true, true, false}};
    for (const Case& c : cases) {
        const MatrixXd want = g.mat(c.name);
        const std::size_t K = static_cast<std::size_t>(g.mat(std::string(c.name) + "_transient")(0, 0));
        const std::size_t order = static_cast<std::size_t>(g.mat(std::string(c.name) + "_order")(0, 0));
        FilterOptions o = brainstormFilterOptions(fs);
        o.kind = c.kind; o.lowerEdge = c.lo; o.upperEdge = c.hi;
        if (c.relax) o.ripple = 1e-2;
        Design d = design(o);
        const std::size_t ours = d.transient;
        EXPECT(ours + 1 >= K && ours <= K + 1, std::string(c.name) + ": our transient " + std::to_string(ours) + " within 1 sample of Brainstorm's " + std::to_string(K));
        o.transientSamples = K;               // Brainstorm's K stated: the record's ends are then compared exactly
        d = design(o);
        EXPECT(static_cast<std::size_t>(d.order()) == order, std::string(c.name) + ": order " + std::to_string(order));
        if (c.b) {
            const MatrixXd b = g.mat(std::string(c.name) + "_b");
            const double e = maxAbs(d.b.transpose(), b);
            EXPECT(e < 1e-13, std::string(c.name) + ": coefficients (fir1 + kaiser) max err " + sci(e));
        }
        const MatrixXd got = c.kind == Kind::Lowpass ? lowpass(x, c.hi, o) : c.kind == Kind::Highpass ? highpass(x, c.lo, o) : bandpass(x, c.lo, c.hi, o);
        const double e = maxAbs(got, want);
        EXPECT(e < 1e-11 * scale, std::string(c.name) + ": output (incl. both ends) max err " + sci(e));
    }
}

static void brainstormIirParity() {
    std::cout << "=== parity with Brainstorm's notch and band-stop (IIR, forward-backward) ===\n";
    Golden g(kDir + "/filter_bst_golden.txt");
    // skip to the 250 Hz part: the reader is sequential, so read past the FIR records
    for (const char* n : {"fs", "x"}) g.mat(n);
    for (const char* c : {"lp30", "hp5", "bp1_20", "bp8_40"}) { g.mat(c); g.mat(std::string(c) + "_transient"); g.mat(std::string(c) + "_order"); g.mat(std::string(c) + "_b"); }
    g.mat("lp5"); g.mat("lp5_transient"); g.mat("lp5_order");
    g.mat("hp0p5"); g.mat("hp0p5_transient"); g.mat("hp0p5_order");
    g.mat("bp1_20_relax"); g.mat("bp1_20_relax_transient"); g.mat("bp1_20_relax_order"); g.mat("bp1_20_relax_b");
    g.mat("bp1_20_mirror"); g.mat("bp1_20_mirror_transient"); g.mat("bp1_20_mirror_order");
    const double fs = g.mat("fs2")(0, 0);
    const MatrixXd x = g.mat("x2");
    const double scale = x.cwiseAbs().maxCoeff();
    FilterOptions o = brainstormFilterOptions(fs);
    {
        const MatrixXd want = g.mat("notch50_100_bw1");
        const double e = maxAbs(notch(x, {50, 100}, o), want);
        EXPECT(e < 1e-11 * scale, "notch 50, 100 Hz (3-dB bandwidth 1 Hz), both ends included: max err " + sci(e));
    }
    {
        FilterOptions p = o; p.notchBandwidth = 2.5;
        const MatrixXd want = g.mat("notch50_bw2p5");
        const double e = maxAbs(notch(x, {50}, p), want);
        EXPECT(e < 1e-11 * scale, "notch 50 Hz, bandwidth 2.5 Hz: max err " + sci(e));
    }
    {
        const MatrixXd want = g.mat("bandstop50_w1p5");
        const double e = maxAbs(bandstop(x, 50 - 0.75, 50 + 0.75, o), want);
        EXPECT(e < 1e-7 * scale, "band-stop 50 Hz, width 1.5 Hz (Butterworth, order 4 prototype): max err " + sci(e) + " (the reference's own polynomial form is ill-conditioned)");
    }
    {
        // Brainstorm's process_bandstop given a LIST of centre frequencies filters the first one only (observed: the others pass untouched), so the golden of [50 100] is the 50 Hz band-stop
        const MatrixXd want = g.mat("bandstop50_100_w3");
        const double e = maxAbs(bandstop(x, 50 - 1.5, 50 + 1.5, o), want);
        EXPECT(e < 1e-7 * scale, "band-stop 50 Hz, width 3 Hz (Brainstorm applies the first of a list): max err " + sci(e));
    }
}

static void responses() {
    std::cout << "=== frequency response at known points ===\n";
    const double fs = 1000;
    auto gain = [&](const Design& d, double f) { return std::abs(d.response(f, fs)); };
    {
        FilterOptions o = brainstormFilterOptions(fs); o.kind = Kind::Lowpass; o.upperEdge = 40;
        const Design d = design(o);
        EXPECT(std::abs(gain(d, 0) - 1) < 1e-4 && std::abs(gain(d, 20) - 1) < 1e-3, "Brainstorm low-pass 40 Hz: unit gain at DC and in the passband (ripple 1e-3)");
        EXPECT(gain(d, 41) < 1e-3 && gain(d, 100) < 1e-3 && gain(d, 500) < 1e-3, "stopband from 41 Hz (transition 1 Hz) at or below -60 dB, Nyquist included");
        EXPECT(std::abs(gain(d, 40.5) - 0.5) < 2e-3, "-6 dB at the middle of the transition band (40.5 Hz): " + sci(gain(d, 40.5)));
        EXPECT((d.b - d.b.reverse()).cwiseAbs().maxCoeff() < 1e-15 && d.b.size() % 2 == 1, "linear phase: symmetric coefficients, odd length");
    }
    {
        FilterOptions o = brainstormFilterOptions(fs); o.kind = Kind::Highpass; o.lowerEdge = 5;
        const Design d = design(o);
        EXPECT(gain(d, 0) < 1e-3 && gain(d, 4.4) < 1e-3, "Brainstorm high-pass 5 Hz: stopband (DC to 4.5 Hz) below -60 dB");
        EXPECT(std::abs(gain(d, 5) - 1) < 1e-3 && std::abs(gain(d, 500) - 1) < 1e-3, "passband from 5 Hz, unit gain at Nyquist");
        EXPECT(std::abs(gain(d, 4.75) - 0.5) < 2e-3, "-6 dB at 4.75 Hz");
        EXPECT(static_cast<std::size_t>(d.order()) == 7252, "order 7252 (kaiserord, 0.5 Hz transition at 1 kHz)");
    }
    {
        FilterOptions o = brainstormFilterOptions(fs); o.kind = Kind::Bandpass; o.lowerEdge = 8; o.upperEdge = 12;
        const Design d = design(o);
        EXPECT(std::abs(gain(d, 10) - 1) < 1e-3 && gain(d, 6.5) < 1e-3 && gain(d, 13.5) < 1e-3 && gain(d, 0) < 1e-3, "Brainstorm band-pass 8-12 Hz: passband, both stopbands (1 Hz transition bands above 5 Hz)");
        EXPECT(std::abs(gain(d, 7.5) - 0.5) < 2e-3 && std::abs(gain(d, 12.5) - 0.5) < 2e-3, "-6 dB at 7.5 and 12.5 Hz");
    }
    {
        FilterOptions o = brainstormFilterOptions(fs); o.kind = Kind::Notch; o.frequencies = {50, 100, 150}; o.notchBandwidth = 2.0;
        const Design d = design(o);
        EXPECT(gain(d, 50) < 1e-12 && gain(d, 100) < 1e-12 && gain(d, 150) < 1e-12, "notch: zero at the line frequency and its harmonics");
        {
            // the 3-dB points of the single pass lie where tan(pi f / fs) = tan(pi f0 / fs) -/+ ... (the digital notch is symmetric on the warped axis): find them and check |H|^2 = 1/2 and their distance = the bandwidth
            const double w0 = std::tan(kPi * 50 / fs), bwT = std::tan(kPi * 2.0 / fs);      // Mitra-Kuo: the 3-dB width in the warped (tan) variable
            double lo = 40, hi = 50;                    // bisect |H|^2 = 1/2 below the notch and above it
            for (int k = 0; k < 80; ++k) { const double m = (lo + hi) / 2; (gain(d, m) > 0.5 ? lo : hi) = m; }
            const double f1 = lo; lo = 50; hi = 60;
            for (int k = 0; k < 80; ++k) { const double m = (lo + hi) / 2; (gain(d, m) < 0.5 ? lo : hi) = m; }
            const double f2 = hi;
            (void)w0; (void)bwT;
            EXPECT(std::abs((f2 - f1) - 2.0) < 0.02, "forward-backward notch: |H|^2 = 1/2 at two frequencies 2 Hz apart (the 3-dB bandwidth): " + std::to_string(f1) + " and " + std::to_string(f2));
        }
        EXPECT(std::abs(gain(d, 0) - 1) < 1e-12 && std::abs(gain(d, 500) - 1) < 1e-3 && std::abs(gain(d, 75) - 1) < 5e-3, "unit gain at DC, Nyquist and between the notches");
    }
    {
        FilterOptions o = brainstormFilterOptions(fs); o.kind = Kind::Bandstop; o.lowerEdge = 45; o.upperEdge = 55;
        const Design d = design(o);
        {
            // the zero is at the geometric centre of the edges on the pre-warped axis (butter's band-stop)
            const double fc = std::atan(std::sqrt(std::tan(kPi * 45 / fs) * std::tan(kPi * 55 / fs))) * fs / kPi;
            EXPECT(gain(d, fc) < 1e-12 && gain(d, 50) < 1e-6, "band-stop: zero at the centre (" + std::to_string(fc) + " Hz, the geometric centre of the edges on the bilinear axis)");
        }
        EXPECT(std::abs(gain(d, 45) - 0.5) < 1e-9 && std::abs(gain(d, 55) - 0.5) < 1e-6, "|H|^2 = 1/2 at the band edges (Butterworth -3 dB, twice)");
        EXPECT(std::abs(gain(d, 0) - 1) < 1e-12 && std::abs(gain(d, 500) - 1) < 1e-9 && gain(d, 30) > 0.999, "unit gain at DC and Nyquist, flat away from the band");
    }
    std::cout << "=== zero phase and no delay ===\n";
    {
        const double sf = 200;
        const int T = 4000;
        MatrixXd x(1, T);
        for (int i = 0; i < T; ++i) x(0, i) = std::sin(2 * kPi * 9.0 * i / sf + 0.7);
        FilterOptions o = brainstormFilterOptions(sf); o.removeMean = false;
        for (const Kind k : {Kind::Lowpass, Kind::Bandpass}) {
            const MatrixXd y = k == Kind::Lowpass ? lowpass(x, 30, o) : bandpass(x, 6, 12, o);
            const double G = std::abs(design([&] { FilterOptions p = o; p.kind = k; p.lowerEdge = 6; p.upperEdge = k == Kind::Lowpass ? 30 : 12; return p; }()).response(9.0, sf));
            const double e = (y.middleCols(1000, 2000) - G * x.middleCols(1000, 2000)).cwiseAbs().maxCoeff();
            EXPECT(e < 1e-9, std::string(k == Kind::Lowpass ? "low-pass" : "band-pass") + ": a sine comes out in phase, scaled by the gain only: max err " + sci(e));
        }
        FilterOptions n = brainstormFilterOptions(sf); n.removeMean = false;
        const MatrixXd z = notch(x, {50}, n);
        EXPECT((z.middleCols(1000, 2000) - x.middleCols(1000, 2000)).cwiseAbs().maxCoeff() < 1e-3, "notch: a sine away from the notch passes in phase");
    }
}

static void chunkInvariance() {
    std::cout << "=== the result does not depend on the chunk size ===\n";
    std::mt19937 rng(5);
    std::normal_distribution<double> nd;
    const int C = 3, T = 9000;
    MatrixXd x(C, T);
    for (int c = 0; c < C; ++c) { double w = 0; for (int i = 0; i < T; ++i) { w += 0.1 * nd(rng); x(c, i) = w + nd(rng) + 5 * (c + 1) + 2 * std::sin(2 * kPi * 50.0 * i / 250.0); } }
    struct Spec { const char* name; FilterOptions o; double tol; };
    std::vector<Spec> specs;
    { FilterOptions o = brainstormFilterOptions(250); o.kind = Kind::Bandpass; o.lowerEdge = 2; o.upperEdge = 40; specs.push_back({"Brainstorm band-pass (FIR)", o, 1e-12}); }
    { FilterOptions o = brainstormFilterOptions(250); o.kind = Kind::Lowpass; o.upperEdge = 30; specs.push_back({"Brainstorm low-pass (FIR, mean restored)", o, 1e-12}); }
    { FilterOptions o = brainstormFilterOptions(250); o.kind = Kind::Notch; o.frequencies = {50, 100}; o.notchBandwidth = 3.0; specs.push_back({"Brainstorm notch (IIR forward-backward)", o, 1e-12}); }
    { FilterOptions o = brainstormFilterOptions(250); o.kind = Kind::Bandstop; o.lowerEdge = 45; o.upperEdge = 55; specs.push_back({"Brainstorm band-stop (IIR forward-backward)", o, 1e-10}); }
    for (const Spec& s : specs) {
        VectorXd mean = x.rowwise().mean();
        Filter whole(C, s.o, mean);
        MatrixXd a = whole.add(x), b = whole.finish();
        MatrixXd ref(C, T);
        ref << a, b;
        EXPECT(ref.cols() == T && whole.samplesOut() == T, std::string(s.name) + ": every sample comes out once, in order");
        EXPECT(whole.lookAhead() > 0 && static_cast<int>(whole.lookAhead()) < T / 2, std::string(s.name) + ": look-ahead " + std::to_string(whole.lookAhead()) + " samples");
        double worst = 0;
        for (int chunk : {1, 7, 100, 333, 1000, 1023, 1024, 2500, 8999, 9000}) {
            Filter f(C, s.o, mean);
            MatrixXd y(C, 0);
            auto append = [&](const MatrixXd& m) { MatrixXd t(C, y.cols() + m.cols()); t << y, m; y = t; };
            for (int i = 0; i < T; i += chunk) append(f.add(x.middleCols(i, std::min(chunk, T - i))));
            append(f.finish());
            const double e = y.cols() == T ? (y - ref).cwiseAbs().maxCoeff() : 1e9;
            worst = std::max(worst, e);
            EXPECT(e < s.tol * 20, std::string(s.name) + ", chunk " + std::to_string(chunk) + ": max difference from the whole record " + sci(e));
        }
        std::cout << "   " << s.name << ": worst over chunk sizes " << sci(worst) << "\n";
    }
}

static void refusals() {
    std::cout << "=== refusals ===\n";
    MatrixXd x = MatrixXd::Random(2, 5000);
    FilterOptions o = brainstormFilterOptions(100);
    EXPECT(throws([&] { lowpass(x, 50, o); }), "a low-pass edge at Nyquist");
    EXPECT(throws([&] { highpass(x, 0, o); }), "a high-pass edge at 0");
    EXPECT(throws([&] { bandpass(x, 20, 10, o); }), "lower edge above the upper");
    EXPECT(throws([&] { bandpass(x, 10, 60, o); }), "an upper edge beyond Nyquist");
    EXPECT(throws([&] { notch(x, {}, o); }), "a notch without frequencies");
    EXPECT(throws([&] { notch(x, {50}, o); }), "a notch at Nyquist");
    EXPECT(throws([&] { bandstop(x, 40, 30, o); }), "a band-stop with the edges swapped");
    EXPECT(throws([&] { notch(MatrixXd::Random(1, 4), {10}, o); }), "an IIR filter on a record shorter than filtfilt's extension");
    FilterOptions b = brainstormFilterOptions(100); b.kind = Kind::Highpass; b.lowerEdge = 5;
    EXPECT(throws([&] { Filter(2, b); }), "a mean-removing FIR filter without the record's channel means");
    b.removeMean = false;
    Filter f(2, b);
    EXPECT(throws([&] { f.add(MatrixXd::Zero(3, 10)); }), "a chunk with the wrong number of channels");
    f.add(x.leftCols(100));
    f.finish();
    EXPECT(throws([&] { f.add(x.leftCols(10)); }), "add after finish");
    FilterOptions w = brainstormFilterOptions(1000); w.kind = Kind::Bandpass; w.lowerEdge = 20; w.upperEdge = 400; w.transitionBandwidth = 150;
    EXPECT(throws([&] { design(w); }), "a low-pass edge whose transition band passes Nyquist");
}

int main() {
    responses();
    chunkInvariance();
    refusals();
    brainstormParity();
    brainstormIirParity();
    std::cout << (g_failures ? "FAILED\n" : "all passed\n");
    return g_failures ? 1 : 0;
}
