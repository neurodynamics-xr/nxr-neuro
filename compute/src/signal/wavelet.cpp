#include "nxr/signal/wavelet.h"
#include "nxr/signal/fourier.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nxr::signal::timefilterbank {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXd;

// ── the Fourier primitives of a bank's grid (declared in wavelet.h) ────────────────────────────────────────────────────────────────────────────────────────────────────────

static void requireSides(const TimeFilterbank& b, Sides s, const char* what) {
    if (b.sides != s) throw std::invalid_argument(std::string("timebank::") + what + (s == Sides::Analytic ? ": needs an analytic bank" : ": needs a two-sided bank"));
}

MatrixXcd spectrum(const TimeFilterbank& b, const MatrixXd& x) {
    if (static_cast<std::size_t>(x.rows()) > b.nfft) throw std::invalid_argument("timebank: the record is longer than the bank's grid");
    MatrixXd xp = MatrixXd::Zero(static_cast<Eigen::Index>(b.nfft), x.cols());
    xp.topRows(x.rows()) = x;
    if (b.sides == Sides::Analytic) return rfft(xp);
    return fft(xp.cast<cd>());
}

MatrixXcd spectrum(const TimeFilterbank& b, const MatrixXcd& x) {
    if (b.sides == Sides::Analytic)
        throw std::invalid_argument("timebank: an ANALYTIC bank is one-sided and cannot take a complex signal (it would drop its negative frequencies silently); "
                                    "use a two-sided bank, or pass the real signal");
    if (static_cast<std::size_t>(x.rows()) > b.nfft) throw std::invalid_argument("timebank: the record is longer than the bank's grid");
    MatrixXcd xp = MatrixXcd::Zero(static_cast<Eigen::Index>(b.nfft), x.cols());
    xp.topRows(x.rows()) = x;
    return fft(xp);
}

MatrixXcd filterSpectrum(const TimeFilterbank& b, const VectorXd& g, const MatrixXcd& X) {
    if (g.size() != X.rows()) throw std::invalid_argument("timebank::filterSpectrum: one gain per bin");
    MatrixXcd Y = MatrixXcd::Zero(static_cast<Eigen::Index>(b.nfft), X.cols());
    for (Eigen::Index j = 0; j < X.cols(); ++j)
        for (Eigen::Index k = 0; k < X.rows(); ++k) Y(k, j) = X(k, j) * g(k);
    return ifft(Y);
}

MatrixXcd coefficientSpectrum(const TimeFilterbank& b, const MatrixXcd& W) {
    MatrixXcd S = fft(W);
    return b.sides == Sides::Analytic ? MatrixXcd(S.topRows(static_cast<Eigen::Index>(rfftBins(b.nfft)))) : S;
}

MatrixXd inverseReal(const TimeFilterbank& b, const MatrixXcd& S, std::size_t n) {
    requireSides(b, Sides::Analytic, "inverseReal");
    return irfft(S, b.nfft).topRows(static_cast<Eigen::Index>(n));
}

MatrixXcd inverseComplex(const TimeFilterbank& b, const MatrixXcd& S, std::size_t n) {
    requireSides(b, Sides::TwoSided, "inverseComplex");
    return ifft(S).topRows(static_cast<Eigen::Index>(n));
}

}  // namespace nxr::signal::timefilterbank

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;
using cd = std::complex<double>;
using i64 = std::int64_t;
static constexpr double kPi = 3.14159265358979323846;

// ── full rate (the bodies of timefilterbank::analysis, synthesis, ...: moved, not edited) ───────────────────────────────

namespace tb = signal::timefilterbank;

static void requireSides(const tb::TimeFilterbank& b, tb::Sides s, const char* what) {
    if (b.sides != s) throw std::invalid_argument(std::string("wt: ") + what + (s == tb::Sides::Analytic ? ": needs an analytic bank" : ": needs a two-sided bank"));
}
// the bins of an analytic grid that are real and unmirrored: DC, and Nyquist for an even length
static bool isEdgeBin(const tb::TimeFilterbank& b, Eigen::Index k) {
    return b.sides == tb::Sides::Analytic && (k == 0 || (b.nfft % 2 == 0 && static_cast<std::size_t>(k) == b.nfft / 2));
}

static tb::Coefficients analyse(const tb::TimeFilterbank& b, const MatrixXcd& X) {
    const MatrixXd G = tb::gains(b);
    tb::Coefficients W;
    for (int m = 0; m < b.size(); ++m) W.push_back(tb::filterSpectrum(b, G.col(m), X));
    return W;
}

tb::Coefficients wt(const tb::TimeFilterbank& b, const MatrixXd& x) { return analyse(b, tb::spectrum(b, x)); }

tb::Coefficients bandAnalytic(const MatrixXd& x, double fs, const std::vector<std::pair<double, double>>& bands, tb::Family shape, double halfWidth) {
    if (x.rows() < 2) throw std::invalid_argument("bandAnalytic: x holds at least 2 samples");
    if (bands.empty()) throw std::invalid_argument("bandAnalytic: at least one band");
    tb::Design design(shape, 2.0, tb::Sides::Analytic);
    design.bands = bands;
    design.halfWidth = halfWidth;
    design.edgeBins = tb::EdgeBins::Drop;
    return wt(tb::timeBank(design, static_cast<std::size_t>(x.rows()), fs), x);
}

std::vector<MatrixXd> bandEnvelope(const MatrixXd& x, double fs, const std::vector<std::pair<double, double>>& bands, tb::Family shape, double halfWidth) {
    const tb::Coefficients z = bandAnalytic(x, fs, bands, shape, halfWidth);
    std::vector<MatrixXd> out;
    out.reserve(z.size());
    for (const auto& m : z) out.push_back(m.cwiseAbs());
    return out;
}
tb::Coefficients wt(const tb::TimeFilterbank& b, const MatrixXcd& x) { return analyse(b, tb::spectrum(b, x)); }

// sum_m w_m * spec(W_m); w = the dual g/S, or g / S_nominal (G^2, and G^2/2 at an analytic DC / Nyquist under Single) for the tight synthesis
static MatrixXcd combine(const tb::TimeFilterbank& b, const tb::Coefficients& W, signal::graphfilterbank::Synthesis mode) {
    if (static_cast<int>(W.size()) != b.size()) throw std::invalid_argument("iwt: one coefficient block per member");
    MatrixXd w;
    if (mode == signal::graphfilterbank::Synthesis::Dual) w = tb::dualGains(b);
    else {
        w = tb::gains(b);
        const double S0 = b.gain * b.gain;
        for (Eigen::Index k = 0; k < w.rows(); ++k) w.row(k) /= (isEdgeBin(b, k) && b.edgeBins == tb::EdgeBins::Single) ? S0 / 2 : S0;
    }
    MatrixXcd Z;
    for (int m = 0; m < b.size(); ++m) {
        const MatrixXcd Sm = tb::coefficientSpectrum(b, W[static_cast<std::size_t>(m)]);
        if (m == 0) Z = MatrixXcd::Zero(Sm.rows(), Sm.cols());
        for (Eigen::Index j = 0; j < Sm.cols(); ++j)
            for (Eigen::Index k = 0; k < Sm.rows(); ++k) Z(k, j) += w(k, m) * Sm(k, j);
    }
    return Z;
}

MatrixXd iwt(const tb::TimeFilterbank& b, const tb::Coefficients& W, std::size_t n, signal::graphfilterbank::Synthesis mode) {
    requireSides(b, tb::Sides::Analytic, "iwt");
    return tb::inverseReal(b, combine(b, W, mode), n);
}

MatrixXcd iwtComplex(const tb::TimeFilterbank& b, const tb::Coefficients& W, std::size_t n, signal::graphfilterbank::Synthesis mode) {
    requireSides(b, tb::Sides::TwoSided, "iwtComplex");
    return tb::inverseComplex(b, combine(b, W, mode), n);
}

MatrixXcd wt(const tb::TimeFilterbank& b, int member, const MatrixXd& x, const VectorXd& at) {
    requireSides(b, tb::Sides::Analytic, "wt");
    if (member < 0 || member >= b.size()) throw std::invalid_argument("wt: no member " + std::to_string(member));
    const i64 C = x.rows(), width = x.cols();
    const std::size_t N = next_fast_len(static_cast<std::size_t>(width), true);
    MatrixXd xp = MatrixXd::Zero(static_cast<Eigen::Index>(N), C);
    xp.topRows(width) = x.transpose();
    const MatrixXcd X = rfft(xp);                                                 // [N/2 + 1 x C]
    // the member's bins on the window's grid, and the gains there (zero outside the member's support on the bank's grid)
    const double df = b.fs / static_cast<double>(N), nf = static_cast<double>(N) / static_cast<double>(b.nfft);
    i64 lo = static_cast<i64>(std::ceil(static_cast<double>(b.support[static_cast<std::size_t>(member)].first) * nf - 1e-9));
    i64 hi = static_cast<i64>(std::floor(static_cast<double>(b.support[static_cast<std::size_t>(member)].second) * nf + 1e-9));
    hi = std::min<i64>(hi, static_cast<i64>(N / 2)); if (hi < lo) lo = hi = std::min<i64>(lo, static_cast<i64>(N / 2));
    VectorXd f = VectorXd::LinSpaced(static_cast<Eigen::Index>(hi - lo + 1), static_cast<double>(lo), static_cast<double>(hi)) * df;
    VectorXd g = tb::gains(b, f).col(member);
    // the bins where the gain is zero to rounding (below 1e-16 of its peak) add nothing: skip them (a Meyer band's nominal support runs to Nyquist)
    const double gcut = 1e-16 * g.cwiseAbs().maxCoeff();
    i64 k0 = 0, k1 = hi - lo; while (k0 <= k1 && std::abs(g(k0)) <= gcut) ++k0; while (k1 >= k0 && std::abs(g(k1)) <= gcut) --k1;
    if (k1 < k0) return MatrixXcd::Zero(C, at.size());
    g = VectorXd(g.segment(k0, k1 - k0 + 1)); lo += k0; hi = lo + (k1 - k0);
    MatrixXcd out(C, at.size());
    const i64 nb = hi - lo + 1;
    std::vector<cd> gp(static_cast<std::size_t>(nb));                                  // g_b e^{2 pi i b t / N} for the member's bins, per instant
    for (Eigen::Index j = 0; j < at.size(); ++j) {
        const double w = 2 * kPi * at(j) / static_cast<double>(N);
        cd ph = std::polar(1.0, w * static_cast<double>(lo)); const cd step = std::polar(1.0, w);
        for (i64 k = 0; k < nb; ++k) {
            if (k % 64 == 0) ph = std::polar(1.0, w * static_cast<double>(lo + k));        // re-seeded: the recurrence's error stays at rounding
            gp[static_cast<std::size_t>(k)] = g(k) * ph;
            ph *= step;
        }
        for (i64 c = 0; c < C; ++c) {
            const cd* col = X.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(X.rows()) + static_cast<std::size_t>(lo);
            double re = 0, im = 0;
            for (i64 k = 0; k < nb; ++k) { const double a = col[k].real(), b2 = col[k].imag(), pr = gp[static_cast<std::size_t>(k)].real(), pi = gp[static_cast<std::size_t>(k)].imag(); re += a * pr - b2 * pi; im += a * pi + b2 * pr; }
            out(c, j) = cd(re, im) / static_cast<double>(N);
        }
    }
    return out;
}

BlockTransform wtBlock(const tb::TimeFilterbank& bank, const Eigen::MatrixXd& block, std::size_t left, std::size_t own) {
  const std::size_t width = static_cast<std::size_t>(block.cols());
  if (left + own > width) throw std::invalid_argument("wtBlock: the own columns [left, left + own) are not inside the block");
  const std::size_t grid = tb::gridLength(width, tb::Padding::FastLen);
  if (bank.nfft != grid)
    throw std::invalid_argument("wtBlock: the bank is designed on a grid of " + std::to_string(bank.nfft) + " samples; a block of " + std::to_string(width) +
                                " samples needs one of " + std::to_string(grid) + " (timeBank(design, gridLength(width, FastLen), fs))");
  BlockTransform r;
  const tb::Coefficients all = wt(bank, Eigen::MatrixXd(block.transpose()));   // wt takes [samples x signals]
  r.own.reserve(all.size());
  for (const auto& w : all) r.own.push_back(w.middleRows(static_cast<Eigen::Index>(left), static_cast<Eigen::Index>(own)).transpose());
  return r;
}

MatrixXcd wtDecimated(const tb::TimeFilterbank& b, int member, const MatrixXcd& X, int D) {
    requireSides(b, tb::Sides::Analytic, "wtDecimated");
    if (member < 0 || member >= b.size()) throw std::invalid_argument("wtDecimated: no member " + std::to_string(member));
    if (D < 1 || b.nfft % static_cast<std::size_t>(D) != 0) throw std::invalid_argument("wtDecimated: the decimation " + std::to_string(D) + " does not divide the bank's grid (" + std::to_string(b.nfft) + ")");
    const i64 nb = static_cast<i64>(b.nfft / 2 + 1);
    if (X.rows() != nb) throw std::invalid_argument("wtDecimated: the spectrum has " + std::to_string(X.rows()) + " bins, the bank's grid " + std::to_string(nb));
    const i64 M = static_cast<i64>(b.nfft) / D, C = X.cols();
    const tb::Gain1& gain = b.members[static_cast<std::size_t>(member)].gain;      // this member's gain alone (the bank's other members are not evaluated)
    VectorXd g(nb);
    for (i64 k = 0; k < nb; ++k) g(k) = gain(static_cast<double>(k) * b.fs / static_cast<double>(b.nfft));
    double peak = 0;
    for (i64 k = 1; k < nb; ++k) peak = std::max(peak, std::abs(g(k)));
    const double cut = 1e-16 * peak;
    i64 k0 = 1, k1 = nb - 1;                                                       // DC and Nyquist dropped (the analytic member's real, unmirrored bins)
    if (b.nfft % 2 == 0) k1 = nb - 2;
    while (k0 <= k1 && std::abs(g(k0)) <= cut) ++k0;
    while (k1 >= k0 && std::abs(g(k1)) <= cut) --k1;
    if (k1 < k0) return MatrixXcd::Zero(C, M);
    const i64 J = k1 - k0 + 1;
    if (J > M) throw std::invalid_argument("wtDecimated: the member spans " + std::to_string(J) + " bins, more than the " + std::to_string(M) + " of the decimated grid (decimate less)");
    MatrixXcd Y = MatrixXcd::Zero(M, C);
    for (i64 c = 0; c < C; ++c) for (i64 j = 0; j < J; ++j) Y(j, c) = X(k0 + j, c) * g(k0 + j);
    const MatrixXcd y = ifft(Y);                                                   // (1/M) Σ_j Y_j e^{2πi j m / M}
    MatrixXcd out(C, M);
    const double scale = static_cast<double>(M) / static_cast<double>(b.nfft);
    for (i64 m = 0; m < M; ++m) {
        const cd ph = std::polar(scale, 2 * kPi * static_cast<double>((k0 * m) % M) / static_cast<double>(M));
        for (i64 c = 0; c < C; ++c) out(c, m) = ph * y(m, c);
    }
    return out;
}

}  // namespace nxr::signal
