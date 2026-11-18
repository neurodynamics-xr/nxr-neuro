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
static MatrixXd maskedGains(const signal::timefilterbank::TimeBank& b, const VectorXd& f) {
    MatrixXd g = signal::timefilterbank::gains(b, f);
    for (int m = 0; m < b.size(); ++m) {
        const double lo = static_cast<double>(b.support[m].first) * b.fs / static_cast<double>(b.nfft) - 1e-9 * b.fs / b.nfft;
        const double hi = static_cast<double>(b.support[m].second) * b.fs / static_cast<double>(b.nfft) + 1e-9 * b.fs / b.nfft;
        for (Eigen::Index i = 0; i < f.size(); ++i) if (f(i) < lo || f(i) > hi) g(i, m) = 0;
    }
    return g;
}

Plan::Plan(const signal::timefilterbank::TimeBank& bk, const Options& opt) : bank(bk), options(opt) {
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
    std::size_t N0 = nextFastLen(std::max<std::size_t>(4 * bank.nfft, 4096), true);
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
            g = nextFastLen(2 * g, true);
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
}

// cycles (mod 1) of the phase  b1 t_j / N - i1 (s_w + t_j) / nfft,  t_j = j p / q,  exactly, in integers
static double phaseCycles(const Plan& pl, int m, i64 sw, i64 j) {
    const __int128 N = pl.window, nfft = static_cast<__int128>(pl.bank.nfft), i1 = static_cast<__int128>(pl.bank.support[m].first);
    const __int128 p = pl.p[m], q = pl.q[m], b1 = pl.b1[m];
    const __int128 num = -i1 * (static_cast<__int128>(sw) * q + static_cast<__int128>(j) * p) * N + b1 * static_cast<__int128>(j) * p * nfft;
    const __int128 den = q * N * nfft;
    __int128 r = num % den; if (r < 0) r += den;
    return static_cast<double>(static_cast<long double>(r) / static_cast<long double>(den));
}

// ── Wt ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Wt::Wt(const signal::timefilterbank::TimeBank& bank, int channels, const Options& options) : plan_(bank, options), channels_(channels), buf_(MatrixXd::Zero(channels, 0)) { next_ = -plan_.margin; }

std::vector<Block> Wt::add(const MatrixXd& chunk) {
    if (finished_) throw std::logic_error("wt: add after finish");
    if (chunk.rows() != channels_) throw std::invalid_argument("wt: the chunk has the wrong number of channels");
    const Eigen::Index held = buf_.cols();
    buf_.conservativeResize(Eigen::NoChange, held + chunk.cols());
    buf_.rightCols(chunk.cols()) = chunk;
    total_ += chunk.cols();
    return drain(false);
}

std::vector<Block> Wt::finish() { finished_ = true; return drain(true); }

std::vector<Block> Wt::drain(bool final) {
    const Plan& pl = plan_; const int M = pl.bank.size();
    std::vector<Block> out(M);
    for (int m = 0; m < M; ++m) { out[m].first = next_ * pl.q[m] / pl.p[m]; out[m].coef = MatrixXcd::Zero(0, channels_); }
    const i64 n = total_;
    while (final ? next_ < n + pl.margin : next_ + pl.block + pl.margin <= n) {
        const i64 sw = next_ - pl.margin;
        MatrixXd w = MatrixXd::Zero(pl.window, channels_);
        for (i64 i = 0; i < pl.window; ++i) {
            const i64 g = sw + i;
            if (g >= 0 && g < n) w.row(i) = buf_.col(g - bufStart_).transpose();
        }
        const MatrixXcd X = rfftColumns(w);
        for (int m = 0; m < M; ++m) {
            const i64 Mm = pl.window * pl.q[m] / pl.p[m];
            const i64 nb = pl.b2[m] - pl.b1[m] + 1;
            MatrixXcd Y = MatrixXcd::Zero(Mm, channels_);
            for (i64 r = 0; r < nb; ++r) Y.row(r) = X.row(pl.b1[m] + r) * pl.gain[m](r);
            const MatrixXcd y = fftColumns(Y, false);
            const i64 j0 = pl.margin * pl.q[m] / pl.p[m];
            i64 j1 = (pl.margin + pl.block) * pl.q[m] / pl.p[m];
            if (final) j1 = std::min<i64>(j1, j0 + (n + pl.margin - next_) * pl.q[m] / pl.p[m] + 1);   // instants with t < n + R
            if (final) { // instants k with k D < n + R  <=>  k p < (n + R) q
                i64 cnt = 0; while (j0 + cnt < Mm && (sw / pl.p[m] * pl.q[m] + j0 + cnt) * pl.p[m] < (n + pl.margin) * pl.q[m] && j0 + cnt < (pl.margin + pl.block) * pl.q[m] / pl.p[m]) ++cnt;
                j1 = j0 + cnt;
            }
            if (j1 <= j0) continue;
            MatrixXcd c(j1 - j0, channels_);
            const double scale = static_cast<double>(Mm) / static_cast<double>(pl.window);
            for (i64 j = j0; j < j1; ++j) {
                const double ph = 2 * kPi * phaseCycles(pl, m, sw, j);
                c.row(j - j0) = y.row(j) * (scale * cd(std::cos(ph), std::sin(ph)));
            }
            Block& o = out[m];
            const Eigen::Index have = o.coef.rows();
            o.coef.conservativeResize(have + c.rows(), Eigen::NoChange);
            o.coef.bottomRows(c.rows()) = c;
        }
        next_ += pl.block;
        const i64 keep = next_ - pl.margin;                     // the next window starts here
        if (keep > bufStart_ && buf_.cols() > 0) {
            const i64 drop = std::min<i64>(keep - bufStart_, buf_.cols());
            buf_ = MatrixXd(buf_.rightCols(buf_.cols() - drop));
            bufStart_ += drop;
        }
    }
    return out;
}

// ── Iwt ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Iwt::Iwt(const signal::timefilterbank::TimeBank& bank, int channels, const Options& options, signal::graphfilterbank::Synthesis mode) : plan_(bank, options), channels_(channels), mode_(mode) {
    const int M = bank.size();
    have_.assign(M, std::vector<VectorXcd>(channels));
    start_.assign(M, 0); end_.assign(M, 0);
    for (int m = 0; m < M; ++m) { start_[m] = -plan_.margin * plan_.q[m] / plan_.p[m]; end_[m] = start_[m]; }
}

Eigen::MatrixXd Iwt::add(const std::vector<Block>& blocks) {
    const int M = plan_.bank.size();
    if (static_cast<int>(blocks.size()) != M) throw std::invalid_argument("iwt: one block per member");
    for (int m = 0; m < M; ++m) {
        const Block& b = blocks[m];
        if (b.coef.rows() == 0) continue;
        if (b.first != end_[m]) throw std::invalid_argument("iwt: the coefficients must arrive in order, without gaps");
        for (int c = 0; c < channels_; ++c) {
            VectorXcd& v = have_[m][c]; const Eigen::Index h = v.size();
            v.conservativeResize(h + b.coef.rows()); v.tail(b.coef.rows()) = b.coef.col(c);
        }
        end_[m] += b.coef.rows();
    }
    return drain(false, 0);
}

Eigen::MatrixXd Iwt::finish(std::int64_t samples) { return drain(true, samples); }

Eigen::MatrixXd Iwt::drain(bool final, std::int64_t n) {
    const Plan& pl = plan_; const int M = pl.bank.size();
    std::vector<MatrixXd> parts;
    auto ready = [&]() {          // all members hold the instants up to the window's end
        for (int m = 0; m < M; ++m) if (end_[m] * pl.p[m] < (next_ + pl.block + pl.margin) * pl.q[m]) return false;
        return true;
    };
    while (final ? next_ < n : ready()) {
        const i64 sw = next_ - pl.margin;
        const Eigen::Index nbins = pl.window / 2 + 1;
        MatrixXcd Xr = MatrixXcd::Zero(nbins, channels_);
        for (int m = 0; m < M; ++m) {
            const i64 Mm = pl.window * pl.q[m] / pl.p[m];
            const i64 k0 = sw * pl.q[m] / pl.p[m];                 // sw is a multiple of p
            MatrixXcd a = MatrixXcd::Zero(Mm, channels_);
            for (i64 j = 0; j < Mm; ++j) {
                const i64 k = k0 + j;
                if (k < start_[m] || k >= end_[m]) continue;
                const double ph = -2 * kPi * phaseCycles(pl, m, sw, j);
                const cd e(std::cos(ph), std::sin(ph));
                for (int c = 0; c < channels_; ++c) a(j, c) = have_[m][c](k - start_[m]) * e;
            }
            const MatrixXcd Y = fftColumns(a, true) * (static_cast<double>(pl.window) / static_cast<double>(Mm));   // = X g
            const i64 nb = pl.b2[m] - pl.b1[m] + 1;
            for (i64 r = 0; r < nb; ++r) {
                const i64 b = pl.b1[m] + r;
                const double g = pl.gain[m](r);
                const double S = mode_ == signal::graphfilterbank::Synthesis::Dual ? pl.frame(b) : pl.bank.gain * pl.bank.gain;
                if (mode_ == signal::graphfilterbank::Synthesis::Dual && !(S > 1e-12 * pl.frame.maxCoeff())) continue;
                Xr.row(b) += Y.row(r) * (g / S);
            }
        }
        const MatrixXd x = irfftColumns(Xr, static_cast<std::size_t>(pl.window));
        i64 cnt = pl.block; if (final) cnt = std::min<i64>(cnt, n - next_);
        parts.push_back(x.middleRows(pl.margin, cnt).transpose());
        next_ += pl.block;
        // drop coefficients older than the next window
        for (int m = 0; m < M; ++m) {
            const i64 keep = (next_ - pl.margin) * pl.q[m] / pl.p[m];
            if (keep > start_[m]) {
                const i64 d = std::min<i64>(keep - start_[m], end_[m] - start_[m]);
                for (int c = 0; c < channels_; ++c) have_[m][c] = VectorXcd(have_[m][c].tail(have_[m][c].size() - d));
                start_[m] += d;
                if (start_[m] < keep) { start_[m] = end_[m] = keep; }
            }
        }
    }
    i64 cols = 0; for (auto& p : parts) cols += p.cols();
    MatrixXd out(channels_, cols); i64 c0 = 0;
    for (auto& p : parts) { out.middleCols(c0, p.cols()) = p; c0 += p.cols(); }
    return out;
}

}  // namespace nxr::signal
