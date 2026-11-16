// nxr::timebank: parity with Rheome's @timefilterbank (goldens/gen_timebank_golden.m) and MATLAB's cwtfilterbank (Morse · Amor · Bump), with nxr's frames.py /
// kernels.py / joint.py / rotary.py (gen_timebank_nxr.py) and the Studio's windowFrame / chain.ts (gen_timebank_studio.mts), and the frame's own laws
// (tight reconstruction, DC and Nyquist identical across gain, two-sided rotation sense, refusals). Synthetic data only.
#include "nxr/timebank.h"
#include "golden_io.h"
#include "mesh_helpers.h"
#include <random>
using namespace nxr::timebank;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace fft = nxr::fft;
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static constexpr double kPi = 3.14159265358979323846;

static Design D(Family f, double g, Sides s) { return Design(f, g, s); }
static MatrixXd noise(int n, int c, unsigned seed) { std::mt19937 r(seed); std::normal_distribution<double> N; MatrixXd x(n, c); for (auto& v : x.reshaped()) v = N(r); return x; }
static MatrixXcd cnoise(int n, int c, unsigned seed) { return noise(n, c, seed).cast<std::complex<double>>() + std::complex<double>(0, 1) * noise(n, c, seed + 1).cast<std::complex<double>>(); }
template <class F> static bool throws(F&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

static void rheomeItersine() {
    std::cout << "=== parity with Rheome @timefilterbank (itersine, gain 2, analytic, DC and Nyquist dropped as Rheome does) ===\n";
    Golden g(kDir + "/timebank_golden.txt");
    EXPECT(g.good(), "Rheome timebank golden opens");
    double worst = 0;
    int cases = 0;
    while (g.tok() == "itersine") {
        const int N = (int)g.num(); const double fs = g.num(); const int V = (int)g.num(); const double a = g.num(); const bool hasLim = g.num() != 0; const double lo = g.num(), hi = g.num();
        const MatrixXd H = g.mat("H"), fcG = g.mat("fc"), binsG = g.mat("bins"), bG = g.mat("bounds"), rG = g.mat("rates"), NmG = g.mat("Nm"), x = g.mat("x");
        const MatrixXcd C = g.cmat("Cre", "Cim");
        Design d = D(Family::Itersine, 2.0, Sides::Analytic);
        d.edgeBins = EdgeBins::Drop; d.voices = V; d.anchor = a;
        if (hasLim) d.limits = std::make_pair(lo, hi);
        const TimeBank b = design(d, (size_t)N, fs);
        const double eH = relErr(gains(b), H);
        const double eFc = relErr(centerFrequencies(b).transpose(), fcG);
        MatrixXd bins(b.size(), 2);
        for (int m = 0; m < b.size(); ++m) { bins(m, 0) = (double)b.support[(size_t)m].first + 1; bins(m, 1) = (double)b.support[(size_t)m].second + 1; }
        const double eBins = relErr(bins, binsG);
        const auto bd = bounds(b);
        MatrixXd bm(1, 4); bm << bd.A / 2, bd.B / 2, bd.tightness, bd.uncovered;     // Rheome reports S/2
        const double eB = relErr(bm, bG);
        const auto [r, Nm] = rates(b);
        MatrixXd Nmd(1, (int)Nm.size()); for (size_t i = 0; i < Nm.size(); ++i) Nmd(0, (int)i) = (double)Nm[i];
        const double eR = std::max(relErr(r.transpose(), rG), relErr(Nmd, NmG));
        const Coefficients W = analysis(b, x);
        double eC = 0; for (int m = 0; m < b.size(); ++m) eC = std::max(eC, relErr(MatrixXcd(W[(size_t)m].topRows(N)), MatrixXcd(C.col(m))));
        const auto own = analysisOwnRate(b, x.col(0));
        double eO = 0;
        for (int m = 0; m < b.size(); ++m) {
            const MatrixXd o = g.mat("own" + std::to_string(m + 1));
            MatrixXcd ref(o.rows(), 1); ref.col(0) = o.col(0).cast<std::complex<double>>() + std::complex<double>(0, 1) * o.col(1).cast<std::complex<double>>();
            eO = std::max({eO, relErr(MatrixXcd(own[(size_t)m].coef), ref), relErr(MatrixXd(own[(size_t)m].t), MatrixXd(o.col(2)))});
        }
        const double e = std::max({eH, eFc, eBins, eB, eR, eC, eO});
        worst = std::max(worst, e); ++cases;
        EXPECT(e < 1e-11, "N=" + std::to_string(N) + " V=" + std::to_string(V) + " (" + std::to_string(b.size()) + " members): gains " + sci(eH) + ", centres " + sci(eFc) + ", bins " + sci(eBins) +
                          ", bounds " + sci(eB) + ", rates " + sci(eR) + ", coefficients " + sci(eC) + ", own-rate " + sci(eO));
        if (g.tok() != "end") { std::cout << "  [FAIL] golden structure\n"; ++g_failures; return; }
    }
    EXPECT(cases == 4 && worst < 1e-11, "4 Rheome banks: worst " + sci(worst));
}

static void cwt() {
    std::cout << "=== parity with MATLAB cwtfilterbank (Rheome's reference bank, gain 2: peak 2) ===\n";
    Golden g(kDir + "/timebank_golden.txt");
    std::string t;
    while (g.f >> t && t != "cwt") { /* skip the itersine sections */ }
    int cases = 0; double worst = 0;
    do {
        const std::string wv = g.tok(); const int N = (int)g.num(); const double fs = g.num(); const int V = (int)g.num(); const double lo = g.num(), hi = g.num();
        std::string rest; std::getline(g.f, rest);
        const MatrixXd fcG = g.mat("fc"), H = g.mat("H"), x = g.mat("x");
        const MatrixXcd C = g.cmat("Cre", "Cim"), W = g.cmat("Wre", "Wim");
        Design d = D(wv == "morse" ? Family::Morse : wv == "amor" ? Family::Amor : Family::Bump, 2.0, Sides::Analytic);
        if (rest.find("TimeBandwidth 30") != std::string::npos) d.timeBandwidth = 30;
        d.edgeBins = EdgeBins::Single;
        (void)lo;
        // MATLAB raises the bottom to where the wavelet's support fits the record (cwtfreqbounds, not ported): the lowest centre is passed, the geometric grid from fmax is what is tested
        const std::vector<double> c = cwtCentres(fcG.minCoeff(), hi, V);
        MatrixXd cm(1, (int)c.size()); for (size_t i = 0; i < c.size(); ++i) cm(0, (int)i) = c[c.size() - 1 - i];     // MATLAB lists the centres descending
        const double eFc = relErr(cm, fcG);
        d.centres = c;
        const TimeBank b = design(d, (size_t)(2 * N), fs);
        MatrixXd Hd(H.rows(), H.cols()); for (int m = 0; m < b.size(); ++m) Hd.col(m) = H.col(b.size() - 1 - m);       // ascending
        const MatrixXd Gc = gains(b, frequencies(b));                                  // the continuous design, no DC / Nyquist rule
        const double eH = relErr(Gc, Hd);
        // coefficients exactly as cwtfilterbank forms them: symmetric extension to 2N, first N rows
        const MatrixXcd X = spectrum(b, extendSymmetric(x));
        double eC = 0, eW = 0;
        for (int m = 0; m < b.size(); ++m) {
            const MatrixXcd cmx = filterSpectrum(b, Gc.col(m), X);
            eC = std::max(eC, relErr(MatrixXcd(cmx.topRows(N)), MatrixXcd(C.col(b.size() - 1 - m))));
            eW = std::max(eW, relErr(MatrixXcd(cmx.topRows(N).col(0)), MatrixXcd(W.col(b.size() - 1 - m))));
        }
        // and through analysis() with the DC / Nyquist rule: only the Nyquist bin differs, by the tail of the top scale
        const Coefficients Wa = analysis(b, extendSymmetric(x));
        double eA = 0; for (int m = 0; m < b.size(); ++m) eA = std::max(eA, relErr(MatrixXcd(Wa[(size_t)m].topRows(N)), MatrixXcd(W.col(b.size() - 1 - m).cast<std::complex<double>>())));
        const double e = std::max({eFc, eH, eC, eW});
        worst = std::max(worst, e); ++cases;
        EXPECT(e < 1e-10 && eA < 1e-4, wv + (rest.empty() ? "" : " " + rest) + " (" + std::to_string(b.size()) + " scales): centres " + sci(eFc) + ", gains " + sci(eH) + ", coefficients " + sci(eC) +
                                          ", == cwtfilterbank.wt " + sci(eW) + "; via analysis() (Nyquist rule) " + sci(eA));
        if (g.tok() != "end") { std::cout << "  [FAIL] golden structure\n"; ++g_failures; return; }
    } while (g.f >> t && t == "cwt");
    EXPECT(cases == 4 && worst < 1e-10, "4 cwtfilterbank banks: worst " + sci(worst));
}

static void nxrAndStudio() {
    std::cout << "=== parity with nxr (frames.py, kernels.py, joint.py, rotary.py) and the Studio (dyadicFrame.ts, chain.ts) ===\n";
    const size_t n = 192; const double fs = 100;
    Golden g(kDir + "/timebank_nxr_golden.txt");
    const MatrixXd edges = g.mat("window_edges"), wg = g.mat("window_gains"), wgf = g.mat("window_gains_full"), itg = g.mat("itersine_gains"), mfc = g.mat("morlet_fc"), mhat = g.mat("morlet_hat");
    Golden e(kDir + "/timebank_studio_golden.txt");
    const MatrixXd eedges = e.mat("window_edges"), ewg = e.mat("window_gains"), efc = e.mat("fc"), ebm = e.mat("bank_morlet"), ebmey = e.mat("bank_meyer"), esum = e.mat("meyer_mother_sum");
    const std::vector<double> we = windowEdges(n, fs);
    MatrixXd wem(1, (int)we.size()); for (size_t i = 0; i < we.size(); ++i) wem(0, (int)i) = we[i];
    EXPECT(relErr(wem, edges) < 1e-14 && relErr(wem, eedges) < 1e-14, "window edges == nxr window_frame == Studio windowFrame");
    // Meyer, gain One: Σg² = 1 as nxr and the Studio; two-sided shares |f|
    const TimeBank mey1 = design(D(Family::Meyer, 1.0, Sides::Analytic), n, fs);
    const MatrixXd f1 = frequencies(mey1);
    EXPECT(relErr(gains(mey1, f1), wg) < 1e-13 && relErr(gains(mey1, f1), ewg) < 1e-13, "Meyer bands (gain 1) == nxr window_frame gains " + sci(relErr(gains(mey1, f1), wg)) + " == Studio windowFrame " + sci(relErr(gains(mey1, f1), ewg)));
    const TimeBank meyTwo = design(D(Family::Meyer, 1.0, Sides::TwoSided), n, fs);
    MatrixXd sym = MatrixXd::Zero(wgf.rows(), mey1.size());            // the two-sided bank's sum of the + and − member at each |f| equals the one-sided gain
    const MatrixXd g2 = gains(meyTwo); const int M1 = mey1.size();
    for (int m = 0; m < M1; ++m) for (int k = 0; k < g2.rows(); ++k) sym(k, m) = (m == 0 ? std::sqrt(g2(k, 0) * g2(k, 0) + g2(k, 1) * g2(k, 1) + g2(k, 2) * g2(k, 2)) : std::hypot(g2(k, 2 * m + 1), g2(k, 2 * m + 2)));   // [DC, scaling+-, band+-, ..., top+-]
    EXPECT(relErr(sym, wgf) < 1e-13, "Meyer two-sided members (+f and −f) reassemble nxr's gains on |f| " + sci(relErr(sym, wgf)));
    // nxr's itersine_time_frame keeps sum g^2 = 2 (peak sqrt 2); Rheome's bank is now peak = gain, so the match is at gain sqrt 2 (nxr's convention differs from the ruling: flagged)
    Design di = D(Family::Itersine, std::sqrt(2.0), Sides::Analytic); di.voices = 4; di.edgeBins = EdgeBins::Drop;     // nxr zeroes DC and Nyquist, and keeps sum g^2 = 2 (peak sqrt 2): the gain that matches it
    const TimeBank it2 = design(di, n, fs);
    EXPECT(relErr(gains(it2), itg) < 1e-13, "itersine (gain sqrt 2 = nxr's sum g^2 = 2, edge bins dropped) == nxr itersine_time_frame " + sci(relErr(gains(it2), itg)));
    // Morlet hats (nxr, gain One) and the Studio's 2 × Morlet (gain Two)
    std::vector<double> fc; for (int i = 0; i < mfc.cols(); ++i) fc.push_back(mfc(0, i));
    Design dm = D(Family::Morlet, 1.0, Sides::Analytic); dm.centres = fc; dm.cycles = 6;
    const TimeBank mo1 = design(dm, n, fs);
    MatrixXd hat = gains(mo1, f1); for (int k = 0; k < hat.rows(); ++k) if (f1(k) == 0) hat.row(k).setZero();     // the hat itself has a Gaussian tail at DC; the bank is on f > 0
    MatrixXd hatRef = mhat; hatRef.row(0).setZero();
    EXPECT(relErr(hat, hatRef) < 1e-13, "Morlet (gain 1) == nxr morlet_hat on f > 0 " + sci(relErr(hat, hatRef)));
    dm.gain = 2.0;
    const TimeBank mo2 = design(dm, n, fs);
    MatrixXd g2m = gains(mo2, f1), eref = ebm; g2m.row(0).setZero(); eref.row(0).setZero();
    EXPECT(relErr(g2m, eref) < 1e-13 && relErr(logCentres(fs, 2, 40, 4).size() == (size_t)efc.cols() ? MatrixXd::Zero(1, 1) : MatrixXd::Ones(1, 1), MatrixXd::Zero(1, 1)) < 1e-14,
           "Morlet (gain 2: peak 2) == the Studio's bankGain = 2 x Morlet " + sci(relErr(g2m, eref)) + "; logCentres == waveletFreqs");
    { const std::vector<double> lc = logCentres(fs, 2, 40, 4); MatrixXd m(1, (int)lc.size()); for (size_t i = 0; i < lc.size(); ++i) m(0, (int)i) = lc[i]; EXPECT(relErr(m, efc) < 1e-14 && relErr(m, mfc) < 1e-14, "logCentres == Studio waveletFreqs == nxr wavelet_freqs"); }
    std::cout << "  [info] the Studio's Meyer MOTHER bank (chain.ts 'meyer', 4 voices, scaled 1/sqrt(voices)) has Σ(g/2)² in [" << esum(0, 0) << ", " << esum(0, 1) << "]: tight only at 1 voice; not ported\n";
    (void)ebmey;
    // nxr's morlet(): complex series at signed frequencies = the two-sided Morlet bank's members (periodic, gain One)
    const MatrixXd zz = g.mat("z"), frq = g.mat("morlet_freqs");
    const MatrixXcd Mref = g.cmat("morlet_re", "morlet_im");
    MatrixXcd z(zz.cols(), 1); z.col(0) = zz.row(0).transpose().cast<std::complex<double>>() + std::complex<double>(0, 1) * zz.row(1).transpose().cast<std::complex<double>>();
    Design dt = D(Family::Morlet, 1.0, Sides::TwoSided); dt.centres = {5.0, 12.5, 9.0}; dt.cycles = 6;
    const TimeBank mt = design(dt, n, fs);
    const Coefficients Wt = analysis(mt, z);
    double eM = 0;
    for (int i = 0; i < frq.cols(); ++i) {        // member of centre |f0| and sign of f0
        const double f0 = frq(0, i); int c = f0 == 5 || f0 == -5 ? 0 : (f0 == 12.5 || f0 == -12.5) ? 1 : 2;
        const int m = 2 * c + (f0 > 0 ? 0 : 1);
        eM = std::max(eM, relErr(MatrixXcd(Wt[(size_t)m].col(0).transpose()), MatrixXcd(Mref.row(i))));
    }
    EXPECT(eM < 1e-12, "two-sided Morlet members == nxr spectral.joint.morlet at +-f0 (periodic grid) " + sci(eM));
    // nxr's rotary amplitudes A+- = |z_+|, |z_-| : the + (resp. -) members' synthesis of z = x + iy
    const MatrixXd xy = g.mat("rot_xy"), Ap = g.mat("rot_Aplus"), Am = g.mat("rot_Aminus");
    MatrixXcd zr(xy.cols(), 1); zr.col(0) = xy.row(0).transpose().cast<std::complex<double>>() + std::complex<double>(0, 1) * xy.row(1).transpose().cast<std::complex<double>>();
    const TimeBank tm = design(D(Family::Itersine, 1.0, Sides::TwoSided), n, fs);
    const Coefficients Wr = analysis(tm, zr);
    const MatrixXd Gt = gains(tm);
    MatrixXcd Zp = MatrixXcd::Zero((long)n, 1), Zm = MatrixXcd::Zero((long)n, 1);
    for (int m = 0; m < tm.size(); ++m) {
        const MatrixXcd S = coefficientSpectrum(tm, Wr[(size_t)m]);
        if (tm.members[(size_t)m].sign == 0) continue;
        (tm.members[(size_t)m].sign > 0 ? Zp : Zm) += (S.array().colwise() * Gt.col(m).cast<std::complex<double>>().array()).matrix();
    }
    const MatrixXd Ap2 = inverseComplex(tm, Zp, n).cwiseAbs(), Am2 = inverseComplex(tm, Zm, n).cwiseAbs();
    const double eRot = std::max(relErr(MatrixXd(Ap2.transpose()), Ap), relErr(MatrixXd(Am2.transpose()), Am));
    EXPECT(eRot < 1e-10, "rotary amplitudes: A+ = |z_+| and A- = |z_-| from the two-sided +f / -f members == nxr rotary_amplitudes (hilbert) " + sci(eRot) + "  [A+ " + sci(Ap.mean()) + " vs " + sci(Ap2.mean()) + ", A- " + sci(Am.mean()) + " vs " + sci(Am2.mean()) + "]");
}

static void laws() {
    std::cout << "=== the frame's own laws ===\n";
    const double fs = 128;
    for (size_t n : {192u, 151u}) {      // an even and an odd record (odd has no Nyquist bin)
        MatrixXd x = noise((int)n, 3, 7);
        x.array() += 1.5;                                                           // a DC offset
        if (n % 2 == 0) for (int i = 0; i < (int)n; ++i) x(i, 0) += 0.8 * ((i % 2) ? -1 : 1);     // and a Nyquist component
        for (Family fam : {Family::Itersine, Family::Meyer}) {
            for (double gn : {1.0, 2.0, 3.0}) {
                for (Padding pad : {Padding::None, Padding::FastLen}) {
                    Design d = D(fam, gn, Sides::Analytic); d.voices = 3;
                    const TimeBank b = design(d, gridLength(n, pad), fs);
                    const auto bd = bounds(b);
                    const double G = gn * gn;        // Σ g² = gain² (the gain is the peak)
                    const Coefficients W = analysis(b, x);
                    const double eT = relErr(synthesis(b, W, n, nxr::filterbank::Synthesis::Tight), x), eD = relErr(synthesis(b, W, n, nxr::filterbank::Synthesis::Dual), x);
                    const double S = frameOperator(b)(1);
                    double en = 0; for (const auto& w : W) en += w.cwiseAbs2().sum();
                    const double eP = std::abs(en - G / 2 * x.squaredNorm()) / x.squaredNorm();
                    const std::string name = std::string(fam == Family::Itersine ? "itersine" : "meyer") + " gain " + std::to_string((int)gn) + " n=" + std::to_string(n) + (pad == Padding::None ? " periodic" : " padded");
                    // Parseval holds exactly for the periodic grid; padded, the sum runs over the padded record (the same identity on nfft)
                    EXPECT(std::abs(bd.A - G) < 1e-12 && std::abs(bd.B - G) < 1e-12 && isTight(b) && eT < 1e-12 && eD < 1e-12 && eP < 1e-12 && std::abs(S - G) < 1e-12,
                           name + ": S = gain^2 = " + std::to_string((int)G) + " (A " + sci(bd.A - G) + ", B " + sci(bd.B - G) + "), tight " + sci(eT) + ", dual " + sci(eD) + " reconstruct DC+Nyquist exactly; Parseval Σ‖w‖² = (gain²/2)Σx² " + sci(eP));
                    // DC and Nyquist: S is G/2 there for both gains — identical relative handling, never doubled
                    const VectorXd Sf = frameOperator(b);
                    const bool edgeOk = std::abs(Sf(0) - G / 2) < 1e-12 && (b.nfft % 2 == 1 || std::abs(Sf(Sf.size() - 1) - G / 2) < 1e-12);
                    EXPECT(edgeOk, name + ": S(DC) = S(Nyquist) = gain²/2 = " + std::to_string(G / 2));
                    // reconstruction: x = (2/gain²) Re ifft(Σ g W) on the interior bins (DC and Nyquist counted once)
                    const MatrixXd Gm = gains(b);
                    MatrixXcd Z = MatrixXcd::Zero(Gm.rows(), x.cols());
                    for (int m = 0; m < b.size(); ++m) Z += (coefficientSpectrum(b, W[(size_t)m]).array().colwise() * Gm.col(m).cast<std::complex<double>>().array()).matrix();
                    MatrixXcd Zf = MatrixXcd::Zero((long)b.nfft, x.cols()); Zf.topRows(Z.rows()) = Z;
                    const MatrixXd rec = ((2 / G) * fft::fftColumns(Zf, false).real()).topRows((Eigen::Index)n);
                    EXPECT(relErr(rec, x) < 1e-12, name + ": x = (2/gain²) Re ifft(Sum g W), DC and Nyquist not doubled " + sci(relErr(rec, x)));
                }
            }
        }
    }
    // Drop: Rheome's convention loses exactly the DC and Nyquist content
    {
        const size_t n = 192; MatrixXd x = noise((int)n, 1, 3); x.array() += 2.0;
        Design d = D(Family::Itersine, 2.0, Sides::Analytic); d.edgeBins = EdgeBins::Drop;
        const TimeBank b = design(d, n, 100);
        const MatrixXd r = synthesis(b, analysis(b, x), n, nxr::filterbank::Synthesis::Tight);
        MatrixXd lost = x; { MatrixXcd X = fft::rfftColumns(x); const std::complex<double> dc = X(0, 0), ny = X(X.rows() - 1, 0); X.setZero(); X(0, 0) = dc; X(X.rows() - 1, 0) = ny; lost = fft::irfftColumns(X, n); }
        EXPECT(relErr(MatrixXd(x - r), lost) < 1e-12, "EdgeBins::Drop (Rheome): reconstruction misses exactly the DC and Nyquist content " + sci(relErr(MatrixXd(x - r), lost)));
    }
    // two-sided: tight over the whole axis, no factor 2; reconstructs a complex record exactly, at any gain
    for (Family fam : {Family::Itersine, Family::Meyer}) for (double gn : {1.0, 2.0}) {
        Design d = D(fam, gn, Sides::TwoSided); d.voices = 3;
        const size_t n = 200;
        const TimeBank b = design(d, n, 128);
        const auto bd = bounds(b);
        const MatrixXcd z = cnoise((int)n, 2, 11);
        const Coefficients W = analysis(b, z);
        const double eT = relErr(synthesisComplex(b, W, n, nxr::filterbank::Synthesis::Tight), z), eD = relErr(synthesisComplex(b, W, n, nxr::filterbank::Synthesis::Dual), z);
        double en = 0; for (const auto& w : W) en += w.cwiseAbs2().sum();
        EXPECT(std::abs(bd.A - gn * gn) < 1e-12 && std::abs(bd.B - gn * gn) < 1e-12 && eT < 1e-12 && eD < 1e-12 && std::abs(en - gn * gn * z.squaredNorm()) < 1e-9 * gn * gn * z.squaredNorm(),
               std::string(fam == Family::Itersine ? "itersine" : "meyer") + " two-sided gain " + std::to_string((int)gn) + ": S = gain^2 on every bin incl. DC and Nyquist, tight " + sci(eT) + ", dual " + sci(eD) + ", Parseval Σ‖w‖² = gain² Σ|z|²");
    }
    // two-sided separates a counter-clockwise from a clockwise rotating pattern
    {
        const size_t n = 256; const double fs = 128, f0 = 16;
        Design d = D(Family::Itersine, 1.0, Sides::TwoSided); d.voices = 3;
        const TimeBank b = design(d, n, fs);
        MatrixXcd ccw(n, 1), cw(n, 1);
        for (int i = 0; i < (int)n; ++i) { const double ph = 2 * kPi * f0 * i / fs; ccw(i, 0) = std::complex<double>(std::cos(ph), std::sin(ph)); cw(i, 0) = std::conj(ccw(i, 0)); }
        double eP = 0, eM = 0, eTot = 0;
        const Coefficients Wc = analysis(b, ccw), Ww = analysis(b, cw);
        for (int m = 0; m < b.size(); ++m) { (b.members[(size_t)m].sign < 0 ? eM : eP) += Wc[(size_t)m].cwiseAbs2().sum(); eTot += Wc[(size_t)m].cwiseAbs2().sum(); }
        double wP = 0, wM = 0;
        for (int m = 0; m < b.size(); ++m) (b.members[(size_t)m].sign < 0 ? wM : wP) += Ww[(size_t)m].cwiseAbs2().sum();
        EXPECT(eM < 1e-24 * eTot && wP < 1e-24 * eTot && std::abs(eP - (double)n) < 1e-9 && std::abs(wM - (double)n) < 1e-9,
               "two-sided: a counter-clockwise tone has all its energy in the +f members (" + sci(eP) + " vs " + sci(eM) + " at -f), a clockwise one in the -f members (" + sci(wM) + " vs " + sci(wP) + ")");
        // the analytic bank on the real part cannot tell them apart
        MatrixXd re = ccw.real(), re2 = cw.real();
        const TimeBank ba = design(D(Family::Itersine, 2.0, Sides::Analytic), n, fs);
        EXPECT(relErr(MatrixXd(analysis(ba, re)[3].cwiseAbs()), MatrixXd(analysis(ba, re2)[3].cwiseAbs())) < 1e-12, "analytic bank: the real parts of a ccw and a cw rotation are identical (no sense)");
    }
    // refusals
    {
        const TimeBank ba = design(D(Family::Itersine, 2.0, Sides::Analytic), 128, 100);
        EXPECT(throws([&] { analysis(ba, cnoise(128, 1, 1)); }), "an ANALYTIC bank REFUSES a complex input (std::invalid_argument)");
        EXPECT(throws([&] { design(D(Family::Itersine, 0.0, Sides::Analytic), 128, 100); }) && throws([&] { design(D(Family::Meyer, -1.0, Sides::TwoSided), 128, 100); }), "a gain of zero or below is refused (the gain is the peak in-band response)");
        EXPECT(throws([&] { design(D(Family::Morlet, 1.0, Sides::Analytic), 128, 100); }), "Morlet without centres is refused (nothing is assumed)");
        EXPECT(throws([&] { synthesisComplex(ba, analysis(ba, noise(128, 1, 2)), 128); }), "synthesisComplex on an analytic bank is refused");
    }
    // ONE MEANING OF GAIN: a tone of amplitude A inside a member's passband gives |c| = (gain/2) A on the analytic bank, in every family and for every gain;
    // a complex tone on the two-sided bank gives gain * A
    {
        const size_t n = 256; const double fs = 128, A = 1.7;
        auto tone = [&](double f0) { MatrixXd x(n, 1); for (int i = 0; i < (int)n; ++i) x(i, 0) = A * std::cos(2 * kPi * f0 * i / fs); return x; };
        auto ctone = [&](double f0) { MatrixXcd z(n, 1); for (int i = 0; i < (int)n; ++i) z(i, 0) = A * std::exp(std::complex<double>(0, 2 * kPi * f0 * i / fs)); return z; };
        double worstA = 0, worstT = 0; int count = 0;
        for (double gn : {1.0, 2.0, 3.0}) {
            for (Family fam : {Family::Itersine, Family::Meyer, Family::Morlet, Family::Morse, Family::Amor, Family::Bump}) {
                Design d = D(fam, gn, Sides::Analytic); d.voices = 2;
                double f0 = 8.0;                                          // a bin of the 256-point grid, and a member centre (Itersine: 2^(6/2); Morlet / cwt: given)
                if (fam == Family::Meyer) f0 = 5.5;                       // inside the plateau of the 4 … 8 Hz octave band (edges 0.5·2^k, transitions ±1/3 octave)
                if (fam == Family::Morlet || fam == Family::Morse || fam == Family::Amor || fam == Family::Bump) d.centres = {8.0};
                const TimeBank b = design(d, n, fs);
                const Coefficients W = analysis(b, tone(f0));
                double best = 0; for (const auto& w : W) best = std::max(best, w.cwiseAbs().maxCoeff());
                worstA = std::max(worstA, std::abs(best - gn / 2 * A) / (gn / 2 * A));
                ++count;
            }
            for (Family fam : {Family::Itersine, Family::Meyer}) {
                Design d = D(fam, gn, Sides::TwoSided); d.voices = 2;
                const TimeBank b = design(d, n, fs);
                const Coefficients W = analysis(b, ctone(fam == Family::Meyer ? 5.5 : 8.0));
                double best = 0; for (const auto& w : W) best = std::max(best, w.cwiseAbs().maxCoeff());
                worstT = std::max(worstT, std::abs(best - gn * A) / (gn * A));
            }
        }
        EXPECT(worstA < 1e-9 && worstT < 1e-9 && count == 18, "a tone of amplitude A in a member's passband gives |c| = (gain/2) A, in all six families at gains 1, 2, 3 (worst rel " + sci(worstA) + "); a complex tone, two-sided, gives gain A (" + sci(worstT) + ")");
    }
    // DECIMATION of a band's analytic signal: the kept columns are every d-th from the offset; Σ z zᴴ decimated, × d, is the full-rate sum (alpha
    // [8, 12] Hz at 1200 Hz, d = 4: 300 Hz, far above the band); the factor and offset are checked
    {
        const size_t n = 12000; const double fs = 1200;
        Design d = D(Family::Meyer, 2.0, Sides::Analytic); d.bands = {{8.0, 12.0}};
        const TimeBank b = design(d, n, fs);
        const MatrixXcd Z = analysis(b, noise(n, 3, 9))[0].transpose();          // [3 x n]
        const MatrixXcd Zd = decimate(Z, 4, 1);
        bool cols = Zd.cols() == static_cast<Eigen::Index>((n - 1 + 3) / 4) && Zd.col(5) == Z.col(1 + 20);
        const MatrixXcd full = Z * Z.adjoint(), dec = 4.0 * (Zd * Zd.adjoint());
        const double rel = (dec - full).norm() / full.norm();
        EXPECT(cols && rel < 1e-3, "decimate keeps every d-th column from the offset; Σ z zᴴ × d equals the full-rate sum (alpha at 1200 → 300 Hz: rel " + sci(rel) + ")");
        EXPECT(throws([&] { decimate(Z, 0, 0); }) && throws([&] { decimate(Z, 4, 4); }), "a factor below 1 or an offset outside [0, d) is refused");
    }
    // Chebyshev-free sanity: scalogram / energies shapes and cwt dual
    {
        const TimeBank b = design(D(Family::Meyer, 2.0, Sides::Analytic), 128, 100);
        const Coefficients W = analysis(b, noise(128, 2, 4));
        EXPECT(scalogram(W).size() == W.size() && energies(W).rows() == b.size() && energies(W).cols() == 2, "scalogram and energies have the members x channels shape");
    }
}

int main() {
    rheomeItersine();
    cwt();
    nxrAndStudio();
    laws();
    std::cout << "\n" << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}
