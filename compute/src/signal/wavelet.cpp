#include "nxr/signal/wavelet.h"
#include "nxr/signal/fourier.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nxr::signal {
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXcd; using Eigen::VectorXd;
using cd = std::complex<double>;
using i64 = std::int64_t;
static constexpr double kPi = 3.14159265358979323846;

static i64 roundUp(i64 n, i64 m) { return (n + m - 1) / m * m; }

// the member gains at frequencies f, zero outside the member's support on the bank's grid
static MatrixXd maskedGains(const signal::timefilterbank::TimeFilterbank& b, const VectorXd& f) {
    MatrixXd g = signal::timefilterbank::gains(b, f);
    for (int m = 0; m < b.size(); ++m) {
        const double lo = static_cast<double>(b.support[m].first) * b.fs / static_cast<double>(b.nfft) - 1e-9 * b.fs / b.nfft;
        const double hi = static_cast<double>(b.support[m].second) * b.fs / static_cast<double>(b.nfft) + 1e-9 * b.fs / b.nfft;
        for (Eigen::Index i = 0; i < f.size(); ++i) if (f(i) < lo || f(i) > hi) g(i, m) = 0;
    }
    return g;
}

WtLayout wtLayout(const signal::timefilterbank::TimeFilterbank& bk, const WtOptions& opt) {
    WtLayout pl; pl.bank = bk; pl.options = opt;
    const auto& bank = pl.bank; auto& rate = pl.rate; auto& Nm = pl.Nm; auto& p = pl.p; auto& q = pl.q; auto& P = pl.P; auto& tail = pl.tail; auto& l1 = pl.l1; auto& achieved = pl.achieved;
    auto& margin = pl.margin; auto& block = pl.block; auto& window = pl.window; auto& frame = pl.frame; auto& b1 = pl.b1; auto& b2 = pl.b2; auto& gain = pl.gain;
    if (bank.sides != signal::timefilterbank::Sides::Analytic) throw std::invalid_argument("wt: an analytic bank (a real recording); a two-sided bank has no own-rate sub-bands");
    const int M = bank.size();
    auto [r, nm] = signal::timefilterbank::rates(bank, opt.oversample);
    rate = r; Nm = nm;
    for (int m = 0; m < M; ++m) {
        const i64 g = std::gcd<i64, i64>(static_cast<i64>(bank.nfft), static_cast<i64>(Nm[m]));
        p.push_back(static_cast<i64>(bank.nfft) / g); q.push_back(static_cast<i64>(Nm[m]) / g);
        P = std::lcm(P, p.back());
    }
    // the impulse responses on a fine grid, the grid doubled while a member's tail has not decayed to the tolerance (up to maxMargin)
    std::int64_t L = 0; achieved = 0; tail.assign(M, 0.0); l1.assign(M, 0.0);
    std::size_t N0 = next_fast_len(std::max<std::size_t>(4 * bank.nfft, 4096), true);
    for (int m = 0; m < M; ++m) {
        std::size_t g = N0; double tl = 0, total = 0; i64 Lm = 0;
        for (;;) {
            VectorXd f0 = VectorXd::LinSpaced(static_cast<Eigen::Index>(g / 2 + 1), 0, static_cast<double>(g / 2)) * (bank.fs / static_cast<double>(g));
            const MatrixXd G0 = maskedGains(bank, f0);
            VectorXcd Y = VectorXcd::Zero(static_cast<Eigen::Index>(g));
            for (Eigen::Index i = 0; i < G0.rows(); ++i) Y(i) = G0(i, m);
            const VectorXcd h = ifft(Y) * static_cast<double>(g);       // a_m for X = 1 (up to the 1/g of a window)
            VectorXd a = VectorXd::Zero(static_cast<Eigen::Index>(g / 2 + 1));   // sum of |h| at |t| = i, both sides
            total = 0;
            for (std::size_t i = 0; i < g; ++i) { const std::size_t t = std::min(i, g - i); a(static_cast<Eigen::Index>(t)) += std::abs(h(static_cast<Eigen::Index>(i))); total += std::abs(h(static_cast<Eigen::Index>(i))); }
            tl = 0; Lm = static_cast<i64>(g / 2);
            for (i64 t = static_cast<i64>(g / 2); t >= 0; --t) {      // grow the tail from the far end while it stays under the tolerance
                if (tl + a(t) > opt.tolerance * total) { Lm = t; break; }
                tl += a(t); Lm = t;
            }
            if (Lm >= static_cast<i64>(g / 2) - 1) tl += a(static_cast<Eigen::Index>(Lm));     // did not decay within the grid: report what is seen (a lower bound)
            if (Lm < static_cast<i64>(g / 2) - 1 || static_cast<i64>(g) >= 2 * static_cast<i64>(opt.maxMargin)) break;
            g = next_fast_len(2 * g, true);
        }
        Lm = std::min<i64>(Lm, static_cast<i64>(opt.maxMargin));
        tail[m] = tl / total; l1[m] = total / static_cast<double>(g); L = std::max(L, Lm); achieved = std::max(achieved, tail[m]);
    }
    margin = roundUp(std::max<i64>(L, 1), P);
    block = roundUp(opt.block ? static_cast<i64>(opt.block) : std::max<i64>(2 * margin, std::min<i64>(4 * 2 * margin, static_cast<i64>(opt.maxBlock))), P);
    window = block + 2 * margin;
    // window grid
    const double df = bank.fs / static_cast<double>(window);
    VectorXd fw = VectorXd::LinSpaced(static_cast<Eigen::Index>(window / 2 + 1), 0, static_cast<double>(window / 2)) * df;
    const MatrixXd Gw = maskedGains(bank, fw);
    frame = Gw.array().square().rowwise().sum();
    for (int m = 0; m < M; ++m) {
        const double nf = static_cast<double>(window) / static_cast<double>(bank.nfft);
        i64 lo = static_cast<i64>(std::ceil(static_cast<double>(bank.support[m].first) * nf - 1e-9));
        i64 hi = static_cast<i64>(std::floor(static_cast<double>(bank.support[m].second) * nf + 1e-9));
        hi = std::min<i64>(hi, static_cast<i64>(window / 2));
        if (hi < lo) { lo = hi = std::min<i64>(lo, static_cast<i64>(window / 2)); }
        const i64 Mm = window * q[m] / p[m];
        if (hi - lo + 1 > Mm) throw std::invalid_argument("wt: the member's own rate is too low for its window bins (oversample < 1?)");
        b1.push_back(lo); b2.push_back(hi);
        gain.push_back(Gw.col(m).segment(lo, hi - lo + 1));
    }
    return pl;
}

// cycles (mod 1) of the phase  b1 t_j / N - i1 (s_w + t_j) / nfft,  t_j = j p / q,  exactly, in integers
static double phaseCycles(const WtLayout& pl, int m, i64 sw, i64 j) {
    const __int128 N = pl.window, nfft = static_cast<__int128>(pl.bank.nfft), i1 = static_cast<__int128>(pl.bank.support[m].first);
    const __int128 p = pl.p[m], q = pl.q[m], b1 = pl.b1[m];
    const __int128 num = -i1 * (static_cast<__int128>(sw) * q + static_cast<__int128>(j) * p) * N + b1 * static_cast<__int128>(j) * p * nfft;
    const __int128 den = q * N * nfft;
    __int128 r = num % den; if (r < 0) r += den;
    return static_cast<double>(static_cast<long double>(r) / static_cast<long double>(den));
}

// ── wt: the window's coefficients ─────────────────────────────────────────────────────────────────────────────────────

std::vector<Block> wt(const WtLayout& pl, const MatrixXd& x, const std::vector<int>& members, std::int64_t start) {
    if (start % pl.P != 0) throw std::invalid_argument("wt: the window starts on the lattice period (a multiple of " + std::to_string(pl.P) + " samples)");
    const int M = pl.bank.size(); const int channels = static_cast<int>(x.rows());
    std::vector<char> want(static_cast<std::size_t>(M), members.empty() ? 1 : 0);
    for (int m : members) { if (m < 0 || m >= M) throw std::invalid_argument("wt: no member " + std::to_string(m)); want[static_cast<std::size_t>(m)] = 1; }
    std::vector<Block> out(M);
    i64 next = -pl.margin;                                      // the first sample of the next core
    for (int m = 0; m < M; ++m) { out[m].first = (start + next) * pl.q[m] / pl.p[m]; out[m].coef = MatrixXcd::Zero(0, channels); }
    const i64 n = x.cols();
    // the windows the record holds whole (as a stream fed the whole record would take), then the rest to the tail
    for (bool final : {false, true}) {
    while (final ? next < n + pl.margin : next + pl.block + pl.margin <= n) {
        const i64 sw = next - pl.margin;
        MatrixXd w = MatrixXd::Zero(pl.window, channels);
        for (i64 i = 0; i < pl.window; ++i) {
            const i64 g = sw + i;
            if (g >= 0 && g < n) w.row(i) = x.col(g).transpose();
        }
        const MatrixXcd X = rfft(w);
        for (int m = 0; m < M; ++m) {
            if (!want[static_cast<std::size_t>(m)]) continue;
            const i64 Mm = pl.window * pl.q[m] / pl.p[m];
            const i64 nb = pl.b2[m] - pl.b1[m] + 1;
            MatrixXcd Y = MatrixXcd::Zero(Mm, channels);
            for (i64 r = 0; r < nb; ++r) Y.row(r) = X.row(pl.b1[m] + r) * pl.gain[m](r);
            const MatrixXcd y = ifft(Y);
            const i64 j0 = pl.margin * pl.q[m] / pl.p[m];
            i64 j1 = (pl.margin + pl.block) * pl.q[m] / pl.p[m];
            if (final) { // instants k with k D < n + R  <=>  k p < (n + R) q
                i64 cnt = 0; while (j0 + cnt < Mm && (sw / pl.p[m] * pl.q[m] + j0 + cnt) * pl.p[m] < (n + pl.margin) * pl.q[m] && j0 + cnt < (pl.margin + pl.block) * pl.q[m] / pl.p[m]) ++cnt;
                j1 = j0 + cnt;
            }
            if (j1 <= j0) continue;
            MatrixXcd c(j1 - j0, channels);
            const double scale = static_cast<double>(Mm) / static_cast<double>(pl.window);
            for (i64 j = j0; j < j1; ++j) {
                const double ph = 2 * kPi * phaseCycles(pl, m, start + sw, j);
                c.row(j - j0) = y.row(j) * (scale * cd(std::cos(ph), std::sin(ph)));
            }
            Block& o = out[m];
            const Eigen::Index have = o.coef.rows();
            o.coef.conservativeResize(have + c.rows(), Eigen::NoChange);
            o.coef.bottomRows(c.rows()) = c;
        }
        next += pl.block;
    }
    }
    return out;
}

// ── iwt ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd iwt(const WtLayout& pl, const std::vector<Block>& blocks, std::int64_t n, signal::graphfilterbank::Synthesis mode, std::int64_t start) {
    if (start % pl.P != 0) throw std::invalid_argument("iwt: the window starts on the lattice period (a multiple of " + std::to_string(pl.P) + " samples)");
    const int M = pl.bank.size();
    if (static_cast<int>(blocks.size()) != M) throw std::invalid_argument("iwt: one block per member");
    const int channels = blocks[0].coef.cols();
    std::vector<i64> first(M), end(M);
    for (int m = 0; m < M; ++m) { first[m] = blocks[m].first; end[m] = first[m] + blocks[m].coef.rows(); if (blocks[m].coef.cols() != channels) throw std::invalid_argument("iwt: every member has the same channels"); }
    std::vector<MatrixXd> parts;
    i64 next = 0;
    while (next < n) {
        const i64 sw = start + next - pl.margin;                    // global: the lattice and the baseband phase are the record's
        const Eigen::Index nbins = pl.window / 2 + 1;
        MatrixXcd Xr = MatrixXcd::Zero(nbins, channels);
        for (int m = 0; m < M; ++m) {
            const i64 Mm = pl.window * pl.q[m] / pl.p[m];
            const i64 k0 = sw * pl.q[m] / pl.p[m];                 // sw is a multiple of p
            MatrixXcd a = MatrixXcd::Zero(Mm, channels);
            for (i64 j = 0; j < Mm; ++j) {
                const i64 k = k0 + j;
                if (k < first[m] || k >= end[m]) continue;
                const double ph = -2 * kPi * phaseCycles(pl, m, sw, j);
                const cd e(std::cos(ph), std::sin(ph));
                for (int c = 0; c < channels; ++c) a(j, c) = blocks[m].coef(k - first[m], c) * e;
            }
            const MatrixXcd Y = fft(a) * (static_cast<double>(pl.window) / static_cast<double>(Mm));   // = X g
            const i64 nb = pl.b2[m] - pl.b1[m] + 1;
            for (i64 r = 0; r < nb; ++r) {
                const i64 b = pl.b1[m] + r;
                const double g = pl.gain[m](r);
                const double S = mode == signal::graphfilterbank::Synthesis::Dual ? pl.frame(b) : pl.bank.gain * pl.bank.gain;
                if (mode == signal::graphfilterbank::Synthesis::Dual && !(S > 1e-12 * pl.frame.maxCoeff())) continue;
                Xr.row(b) += Y.row(r) * (g / S);
            }
        }
        const MatrixXd xw = irfft(Xr, static_cast<std::size_t>(pl.window));
        const i64 cnt = std::min<i64>(pl.block, n - next);
        parts.push_back(xw.middleRows(pl.margin, cnt).transpose());
        next += pl.block;
    }
    i64 cols = 0; for (auto& p : parts) cols += p.cols();
    MatrixXd out(channels, cols); i64 c0 = 0;
    for (auto& p : parts) { out.middleCols(c0, p.cols()) = p; c0 += p.cols(); }
    return out;
}

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

std::vector<OwnRate> wtOwnRate(const tb::TimeFilterbank& b, const VectorXd& x, double oversample) {
    requireSides(b, tb::Sides::Analytic, "wtOwnRate");
    MatrixXd col = x;
    const MatrixXcd X = tb::spectrum(b, col);
    const MatrixXd G = tb::gains(b);
    std::vector<OwnRate> out;
    for (int m = 0; m < b.size(); ++m) {
        const auto [i1, i2] = b.support[static_cast<std::size_t>(m)];
        const std::size_t nb = i2 - i1 + 1, Nm = next_fast_len(static_cast<std::size_t>(std::ceil(oversample * static_cast<double>(nb))), true);
        Eigen::VectorXcd Y = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(Nm));
        for (std::size_t i = 0; i < nb; ++i) Y(static_cast<Eigen::Index>(i)) = X(static_cast<Eigen::Index>(i1 + i), 0) * G(static_cast<Eigen::Index>(i1 + i), m);
        OwnRate r;
        r.coef = ifft(Y) * (static_cast<double>(Nm) / static_cast<double>(b.nfft));
        r.t = VectorXd::LinSpaced(static_cast<Eigen::Index>(Nm), 0, static_cast<double>(Nm - 1)) * (static_cast<double>(b.nfft) / static_cast<double>(Nm) / b.fs);
        r.rate = static_cast<double>(Nm) * b.fs / static_cast<double>(b.nfft);
        r.weight = static_cast<double>(b.nfft) / static_cast<double>(Nm);
        out.push_back(std::move(r));
    }
    return out;
}

MatrixXcd wt(const tb::TimeFilterbank& b, int member, const MatrixXd& x, const VectorXd& at) {
    requireSides(b, tb::Sides::Analytic, "wt");
    if (member < 0 || member >= b.size()) throw std::invalid_argument("wt: no member " + std::to_string(member));
    const i64 C = x.rows(), width = x.cols();
    const std::size_t N = next_fast_len(static_cast<std::size_t>(width), true);
    MatrixXd xp = MatrixXd::Zero(static_cast<Eigen::Index>(N), C);
    xp.topRows(width) = x.transpose();
    const MatrixXcd X = rfft(xp);                                                 // [N/2 + 1 x C]
    // the member's bins on the window's grid, and the gains there (zero outside the member's support on the bank's grid, as the own-rate transform)
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

BlockTransform wtBlock(const tb::Design& design, const Eigen::MatrixXd& block, double fs, std::size_t left, std::size_t own) {
  const std::size_t width = static_cast<std::size_t>(block.cols());
  if (left + own > width) throw std::invalid_argument("wtBlock: the own columns [left, left + own) are not inside the block");
  BlockTransform r{tb::design(design, tb::gridLength(width, tb::Padding::FastLen), fs), {}};
  const tb::Coefficients all = wt(r.bank, Eigen::MatrixXd(block.transpose()));   // wt takes [samples x signals]
  r.own.reserve(all.size());
  for (const auto& w : all) r.own.push_back(w.middleRows(static_cast<Eigen::Index>(left), static_cast<Eigen::Index>(own)).transpose());
  return r;
}

}  // namespace nxr::signal
