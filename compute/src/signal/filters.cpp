// nxr::signal::filters — see include/nxr/filter.h.
#include "nxr/signal/filters.h"
#include "nxr/signal/fourier.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace nxr::signal::filters {

using Eigen::MatrixXd;
using Eigen::VectorXd;
static constexpr double kPi = 3.14159265358979323846;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

[[noreturn]] static void bad(const std::string& m) { throw std::invalid_argument("filter: " + m); }

FilterOptions brainstormFilterOptions(double sfreq) {
    FilterOptions o;
    o.sfreq = sfreq;
    o.ripple = 1e-3;
    o.edge = Edge::MeanOfTransient;
    o.transientEnergy = 0.99;
    o.removeMean = true;
    return o;
}

// ── windows ─────────────────────────────────────────────────────────────────────────────────────────────────────

static double besselI0(double x) {   // the power series sum ((x/2)^k / k!)^2
    double sum = 1, term = 1;
    const double q = x * x / 4;
    for (int k = 1; k < 500; ++k) {
        term *= q / (static_cast<double>(k) * k);
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    return sum;
}

VectorXd kaiser(std::size_t n, double beta) {
    VectorXd w(static_cast<Eigen::Index>(n));
    if (n == 1) { w[0] = 1; return w; }
    const double den = besselI0(beta), N = static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        const double r = 2.0 * static_cast<double>(i) / N - 1.0;
        w[static_cast<Eigen::Index>(i)] = besselI0(beta * std::sqrt(std::max(0.0, 1 - r * r))) / den;
    }
    return w;
}

static double sinc(double x) { return x == 0 ? 1.0 : std::sin(kPi * x) / (kPi * x); }

std::pair<std::size_t, double> kaiserord(double transitionHz, double sfreq, double ripple) {
    const double A = -20 * std::log10(ripple);
    const double dw = 2 * kPi * transitionHz / sfreq;
    double beta = 0;
    if (A > 50) beta = 0.1102 * (A - 8.7);
    else if (A > 21) beta = 0.5842 * std::pow(A - 21, 0.4) + 0.07886 * (A - 21);
    const double n = A > 21 ? (A - 7.95) / (2.285 * dw) : 5.79 / (2.285 * dw);
    return {static_cast<std::size_t>(std::ceil(n)), beta};
}

// ── FIR designs ─────────────────────────────────────────────────────────────────────────────────────────────────

// MATLAB fir1(N, Wn, type, window, 'noscale'): ideal band-pass kernels (units of Nyquist), windowed.
static VectorXd fir1(std::size_t order, const std::vector<std::pair<double, double>>& passbands, const VectorXd& win) {
    const Eigen::Index n = static_cast<Eigen::Index>(order) + 1;
    VectorXd b(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        const double m = static_cast<double>(i) - static_cast<double>(order) / 2;
        double h = 0;
        for (const auto& [lo, hi] : passbands) h += (hi >= 1 ? sinc(m) : hi * sinc(hi * m)) - (lo <= 0 ? 0.0 : lo * sinc(lo * m));
        b[i] = h * win[i];
    }
    return b;      // Brainstorm calls fir1 with 'noscale': the gain at the centre of the first passband is not renormalised to 1
}

static Design designKaiser(const FilterOptions& o) {
    const double ny = o.sfreq / 2;
    Design d;
    d.method = Method::Fir;
    double lower = o.lowerEdge, upper = o.upperEdge;
    if (o.kind == Kind::Lowpass) lower = 0;
    if (o.kind == Kind::Highpass) upper = 0;
    if ((o.kind == Kind::Highpass || o.kind == Kind::Bandpass) && !(lower > 0 && lower < ny)) bad("the high-pass edge must be between 0 and Nyquist (" + std::to_string(ny) + " Hz)");
    if ((o.kind == Kind::Lowpass || o.kind == Kind::Bandpass) && !(upper > 0 && upper < ny)) bad("the low-pass edge must be between 0 and Nyquist (" + std::to_string(ny) + " Hz)");
    if (o.kind == Kind::Bandpass && !(lower < upper)) bad("the lower edge must be below the upper edge");
    // Brainstorm's rule: 0.5 Hz transition bands, 1 Hz when the high-pass edge is above 5 Hz and for a low-pass alone
    const double tb = o.transitionBandwidth ? *o.transitionBandwidth : (o.kind == Kind::Lowpass ? 1.0 : lower <= 5 ? 0.5 : 1.0);
    if (!(tb > 0)) bad("the transition bandwidth must be positive");
    double width = tb;
    std::vector<std::pair<double, double>> pass;      // in units of Nyquist
    if (lower > 0) {
        const double stop = std::max(lower - tb, 0.0);
        width = std::min(width, lower - stop);
    }
    if (upper > 0 && upper + tb > ny) bad("the low-pass edge plus its transition band is beyond Nyquist");
    if (o.kind == Kind::Lowpass) pass = {{0, (2 * upper + tb) / o.sfreq}};
    else if (o.kind == Kind::Highpass) pass = {{(2 * lower - std::min(tb, lower)) / o.sfreq, 1}};
    else if (o.kind == Kind::Bandpass) pass = {{(2 * lower - std::min(tb, lower)) / o.sfreq, (2 * upper + tb) / o.sfreq}};
    else bad("the Kaiser FIR design makes low-pass, high-pass and band-pass filters");
    auto [n, beta] = kaiserord(width, o.sfreq, o.ripple);
    std::size_t order = n + (n % 2);
    d.b = fir1(order, pass, kaiser(order + 1, beta));
    return d;
}

// K: the shortest one-sided impulse response (from the centre) that holds `energy` of the energy; at least 1
static std::size_t transientOf(const VectorXd& b, double energy) {
    const Eigen::Index h = b.size() / 2;
    const VectorXd e = b.tail(b.size() - h).array().square();
    const double total = e.sum();
    double cum = 0;
    for (Eigen::Index i = 0; i < e.size(); ++i) {
        cum += e[i];
        if (cum >= energy * total) return std::max<std::size_t>(static_cast<std::size_t>(i), 1);
    }
    return static_cast<std::size_t>(h);
}

// ── IIR designs ─────────────────────────────────────────────────────────────────────────────────────────────────

// Mitra and Kuo's second-order notch with 3-dB bandwidth bw at f0
static Section notchSection(double f0, double bw, double fs) {
    const double w0 = 2 * kPi * f0 / fs, Bw = 2 * kPi * bw / fs;
    const double alpha = (1 - std::tan(Bw / 2)) / (1 + std::tan(Bw / 2)), beta = std::cos(w0);
    const double g = (1 + alpha) / 2;
    return {g, -2 * beta * g, g, -beta * (1 + alpha), alpha};
}

// butter(N, [f1 f2]/(fs/2), 'stop'): the analog low-pass prototype's poles, the band-stop transformation, the bilinear transform; N biquads, unit gain at DC
static std::vector<Section> butterBandstop(int N, double f1, double f2, double fs) {
    using cd = std::complex<double>;
    const double u1 = 2 * fs * std::tan(kPi * f1 / fs), u2 = 2 * fs * std::tan(kPi * f2 / fs);     // the pre-warped edges, rad/s
    const double bw = u2 - u1, w0sq = u1 * u2;
    const double fs2 = 2 * fs;
    std::vector<cd> poles;
    for (int k = 0; k < N; ++k) {
        const double th = kPi * (2.0 * k + N + 1.0) / (2.0 * N);
        const cd p(std::cos(th), std::sin(th));                  // a left-half-plane pole of the prototype
        const cd half = bw / (2.0 * p);
        const cd root = std::sqrt(half * half - w0sq);
        for (const cd& s : {half + root, half - root}) poles.push_back((fs2 + s) / (fs2 - s));
    }
    const double theta0 = 2 * std::atan(std::sqrt(w0sq) / fs2);
    std::vector<Section> out;
    std::vector<bool> used(poles.size(), false);
    for (std::size_t i = 0; i < poles.size(); ++i) {
        if (used[i]) continue;
        used[i] = true;
        std::size_t best = poles.size();
        double bd = 1e300;
        for (std::size_t j = i + 1; j < poles.size(); ++j) {
            if (used[j]) continue;
            const double dd = std::abs(poles[j] - std::conj(poles[i]));
            if (dd < bd) { bd = dd; best = j; }
        }
        used[best] = true;
        const double a1 = -2 * poles[i].real(), a2 = std::norm(poles[i]);
        const double g = (1 + a1 + a2) / (2 - 2 * std::cos(theta0));
        out.push_back({g, -2 * std::cos(theta0) * g, g, a1, a2});
    }
    return out;
}

static void sectionState(const Section& s, double& z1, double& z2) {      // the steady state of a unit constant input
    const double G = (s.b0 + s.b1 + s.b2) / (1 + s.a1 + s.a2);
    z2 = s.b2 - s.a2 * G;
    z1 = s.b1 - s.a1 * G + z2;
}

static void runSection(const Section& s, const double* in, double* out, std::size_t n, double z1, double z2) {
    for (std::size_t i = 0; i < n; ++i) {
        const double y = s.b0 * in[i] + z1;
        z1 = s.b1 * in[i] - s.a1 * y + z2;
        z2 = s.b2 * in[i] - s.a2 * y;
        out[i] = y;
    }
}

static void runStage(const std::vector<Section>& st, std::vector<double>& x, bool steady) {
    for (const Section& s : st) {
        double z1 = 0, z2 = 0;
        if (steady) { sectionState(s, z1, z2); z1 *= x[0]; z2 *= x[0]; }
        runSection(s, x.data(), x.data(), x.size(), z1, z2);
    }
}

// MATLAB filtfilt on one signal: odd extension of 3 x (length - 1) samples, started from the steady state, forward then backward
static void filtfiltStage(const std::vector<Section>& st, const double* x, std::size_t n, double* y) {
    const std::size_t nfact = 3 * 2 * st.size();
    if (n <= nfact) bad("a forward-backward filter needs more than " + std::to_string(nfact) + " samples, got " + std::to_string(n));
    std::vector<double> e(n + 2 * nfact);
    for (std::size_t i = 0; i < nfact; ++i) e[i] = 2 * x[0] - x[nfact - i];
    std::copy(x, x + n, e.begin() + static_cast<std::ptrdiff_t>(nfact));
    for (std::size_t i = 0; i < nfact; ++i) e[nfact + n + i] = 2 * x[n - 1] - x[n - 2 - i];
    runStage(st, e, true);
    std::reverse(e.begin(), e.end());
    runStage(st, e, true);
    std::reverse(e.begin(), e.end());
    std::copy(e.begin() + static_cast<std::ptrdiff_t>(nfact), e.begin() + static_cast<std::ptrdiff_t>(nfact + n), y);
}

// the samples after which a stage's one-sided impulse response has left less than `tol` of its L1 norm
static std::size_t stageMargin(const std::vector<Section>& st, double tol, std::size_t cap) {
    std::vector<double> h;
    const std::size_t block = 4096;
    std::size_t quiet = 0;
    double l1 = 0;
    std::vector<double> state;
    std::vector<std::pair<double, double>> z(st.size(), {0.0, 0.0});
    while (h.size() < cap && quiet < 4) {
        double peak = 0;
        for (std::size_t i = 0; i < block; ++i) {
            double v = h.empty() && i == 0 ? 1.0 : 0.0;
            for (std::size_t k = 0; k < st.size(); ++k) {
                const Section& s = st[k];
                const double y = s.b0 * v + z[k].first;
                z[k].first = s.b1 * v - s.a1 * y + z[k].second;
                z[k].second = s.b2 * v - s.a2 * y;
                v = y;
            }
            h.push_back(v);
            l1 += std::abs(v);
            peak = std::max(peak, std::abs(v));
        }
        quiet = peak < 1e-4 * tol * l1 ? quiet + 1 : 0;
    }
    double tail = 0;
    std::size_t m = h.size();
    for (std::size_t i = h.size(); i-- > 0;) {
        tail += std::abs(h[i]);
        if (tail > tol * l1) break;
        m = i;
    }
    return m;
}

static Design designIir(const FilterOptions& o) {
    const double ny = o.sfreq / 2;
    Design d;
    d.method = Method::Iir;
    if (o.kind == Kind::Notch) {
        if (o.frequencies.empty()) bad("a notch needs frequencies");
        const double bw = o.notchBandwidth ? *o.notchBandwidth : 1.0;
        if (!(bw > 0)) bad("the notch bandwidth must be positive");
        for (double f : o.frequencies) {
            if (!(f > 0 && f < ny)) bad("a notch frequency must be between 0 and Nyquist (" + std::to_string(ny) + " Hz), got " + std::to_string(f));
            d.stages.push_back({notchSection(f, bw, o.sfreq)});
        }
    } else if (o.kind == Kind::Bandstop) {
        if (!(o.lowerEdge > 0 && o.upperEdge < ny && o.lowerEdge < o.upperEdge)) bad("a band-stop needs 0 < lower < upper < Nyquist (" + std::to_string(ny) + " Hz)");
        if (o.bandstopOrder < 1) bad("the band-stop order must be at least 1");
        d.stages.push_back(butterBandstop(o.bandstopOrder, o.lowerEdge, o.upperEdge, o.sfreq));
    } else {
        bad("the IIR designs make notch and band-stop filters");
    }
    std::size_t margin = 0;
    for (const auto& st : d.stages) margin += std::max(stageMargin(st, o.tolerance, o.maxMargin), 3 * 2 * st.size() + 1);
    d.lookBehind = d.lookAhead = std::min(margin, o.maxMargin);
    return d;
}

Design design(const FilterOptions& o) {
    if (!(o.sfreq > 0)) bad("the sampling frequency must be positive");
    Method m = o.method ? *o.method : (o.kind == Kind::Bandstop || o.kind == Kind::Notch) ? Method::Iir : Method::Fir;
    Design d;
    if (m == Method::Iir) {
        d = designIir(o);
    } else {
        d = designKaiser(o);
        d.transient = o.transientSamples ? std::max<std::size_t>(*o.transientSamples, 1) : transientOf(d.b, o.transientEnergy);
        d.lookBehind = d.lookAhead = static_cast<std::size_t>(d.b.size() / 2);
    }
    return d;
}

std::complex<double> Design::response(double freqHz, double sfreq) const {
    const double w = 2 * kPi * freqHz / sfreq;
    if (method == Method::Fir) {
        std::complex<double> s = 0;
        const double h = static_cast<double>(b.size() - 1) / 2;
        for (Eigen::Index i = 0; i < b.size(); ++i) s += b[i] * std::exp(std::complex<double>(0, -w * (static_cast<double>(i) - h)));
        return s;
    }
    double g = 1;
    const std::complex<double> z1 = std::exp(std::complex<double>(0, -w)), z2 = z1 * z1;
    for (const auto& st : stages)
        for (const Section& s : st) g *= std::norm((s.b0 + s.b1 * z1 + s.b2 * z2) / (1.0 + s.a1 * z1 + s.a2 * z2));
    return g;
}

// ── the pure filter ─────────────────────────────────────────────────────────────────────────────────────────────

static MatrixXd iirWindow(const MatrixXd& w, const Design& design_) {
    MatrixXd out(w.rows(), w.cols());
    std::vector<double> x(static_cast<std::size_t>(w.cols())), y(x.size());
    for (Eigen::Index c = 0; c < w.rows(); ++c) {
        for (Eigen::Index i = 0; i < w.cols(); ++i) x[static_cast<std::size_t>(i)] = w(c, i);
        for (const auto& st : design_.stages) {
            filtfiltStage(st, x.data(), x.size(), y.data());
            x.swap(y);
        }
        for (Eigen::Index i = 0; i < w.cols(); ++i) out(c, i) = x[static_cast<std::size_t>(i)];
    }
    return out;
}

// The window's FIR output at the positions that have a whole filter around them; the ends of the record (atStart / atEnd) are padded as the option set says first.
// Positions of the window with the filter's support outside the window and not an end of the record are not valid: the caller never asks for them.
static MatrixXd firWindow(const MatrixXd& w, const FilterOptions& opt_, const Design& design_, const VectorXd& mean_, bool restoreMean_, bool atStart, bool atEnd) {
    const Eigen::Index W = w.cols(), N = design_.b.size(), h = N / 2, C = w.rows();
    const Eigen::Index lp = atStart ? h : 0, rp = atEnd ? h : 0;
    const Eigen::Index P = lp + W + rp;
    MatrixXd out(C, W);
    out.setZero();
    if (P < N) return out;
    const std::size_t nfft = next_fast_len(static_cast<std::size_t>(P), true);
    const std::size_t nb = rfftBins(nfft);
    // the filter's spectrum
    std::vector<cd> H(nb);
    {
        std::vector<double> hb(nfft, 0.0);
        for (Eigen::Index i = 0; i < N; ++i) hb[static_cast<std::size_t>(i)] = design_.b[i];
        r2c(hb.data(), H.data(), {nfft}, 0);
    }
    const Eigen::Index K = std::min<Eigen::Index>(static_cast<Eigen::Index>(design_.transient), W);
    const Eigen::Index group = 16;
    RowMat buf(group, static_cast<Eigen::Index>(nfft));
    std::vector<cd> spec(static_cast<std::size_t>(group) * nb);
    for (Eigen::Index c0 = 0; c0 < C; c0 += group) {
        const Eigen::Index g = std::min(group, C - c0);
        buf.setZero();
        for (Eigen::Index r = 0; r < g; ++r) {
            const Eigen::Index c = c0 + r;
            const double m = mean_.size() ? mean_[c] : 0.0;
            double* p = buf.row(r).data();
            for (Eigen::Index i = 0; i < W; ++i) p[lp + i] = w(c, i) - m;
            if (atStart && lp > 0) {
                if (opt_.edge == Edge::MeanOfTransient) {
                    double s = 0;
                    for (Eigen::Index i = 0; i < K; ++i) s += p[lp + i];
                    std::fill(p, p + lp, s / static_cast<double>(K));
                }
            }
            if (atEnd && rp > 0) {
                const Eigen::Index last = lp + W - 1;
                if (opt_.edge == Edge::MeanOfTransient) {
                    double s = 0;
                    for (Eigen::Index i = 0; i < K; ++i) s += p[last - i];
                    std::fill(p + last + 1, p + last + 1 + rp, s / static_cast<double>(K));
                }
            }
        }
        r2c(buf.data(), spec.data(), {static_cast<std::size_t>(g), nfft}, 1);
        for (Eigen::Index r = 0; r < g; ++r)
            for (std::size_t k = 0; k < nb; ++k) spec[static_cast<std::size_t>(r) * nb + k] *= H[k];
        c2r(spec.data(), buf.data(), {static_cast<std::size_t>(g), nfft}, 1);
        for (Eigen::Index r = 0; r < g; ++r) {
            const double m = restoreMean_ && mean_.size() ? mean_[c0 + r] : 0.0;
            const double* p = buf.row(r).data();
            // full-convolution index of the output at window position i: lp + i + h  (the filter's centre aligned with the sample)
            for (Eigen::Index i = 0; i < W; ++i) {
                const Eigen::Index q = lp + i + h;
                if (q - (N - 1) >= 0 && q < P) out(c0 + r, i) = p[q] + m;        // a whole filter lies inside the padded window
                else out(c0 + r, i) = 0;
            }
        }
    }
    return out;
}

MatrixXd filter(const MatrixXd& w, const FilterOptions& options, const Design& design_, const VectorXd& meanIn, std::size_t left, std::size_t own, bool atStart, bool atEnd) {
    if (static_cast<Eigen::Index>(left + own) > w.cols()) bad("a window's own columns lie outside it");
    const bool highpassPart = options.kind == Kind::Highpass || options.kind == Kind::Bandpass;
    VectorXd mean;
    bool restoreMean = false;
    if (design_.method == Method::Fir && options.removeMean) {
        if (meanIn.size() != w.rows()) bad("the record's channel means are needed (removeMean): " + std::to_string(meanIn.size()) + " given for " + std::to_string(w.rows()) + " channels");
        mean = meanIn;
        restoreMean = !highpassPart;
    }
    const MatrixXd y = design_.method == Method::Fir ? firWindow(w, options, design_, mean, restoreMean, atStart, atEnd) : iirWindow(w, design_);
    return y.middleCols(static_cast<Eigen::Index>(left), static_cast<Eigen::Index>(own));
}
MatrixXd filter(const MatrixXd& w, const FilterOptions& options, const VectorXd& mean, std::size_t left, std::size_t own, bool atStart, bool atEnd) {
    return filter(w, options, design(options), mean, left, own, atStart, atEnd);
}

// ── whole-record verbs ──────────────────────────────────────────────────────────────────────────────────────────

// A whole array is one window: no halo, both ends the record's.
static MatrixXd runWhole(const MatrixXd& x, const FilterOptions& o) {
    if (x.rows() < 1) bad("at least one channel");
    VectorXd mean;
    if (o.removeMean) mean = x.rowwise().mean();
    if (x.cols() == 0) return MatrixXd(x.rows(), 0);
    return filter(x, o, design(o), mean, 0, static_cast<std::size_t>(x.cols()), true, true);
}

MatrixXd lowpass(const MatrixXd& x, double upperEdge, FilterOptions o) { o.kind = Kind::Lowpass; o.upperEdge = upperEdge; return runWhole(x, o); }
MatrixXd highpass(const MatrixXd& x, double lowerEdge, FilterOptions o) { o.kind = Kind::Highpass; o.lowerEdge = lowerEdge; return runWhole(x, o); }
MatrixXd bandpass(const MatrixXd& x, double lowerEdge, double upperEdge, FilterOptions o) { o.kind = Kind::Bandpass; o.lowerEdge = lowerEdge; o.upperEdge = upperEdge; return runWhole(x, o); }
MatrixXd bandstop(const MatrixXd& x, double lowerEdge, double upperEdge, FilterOptions o) { o.kind = Kind::Bandstop; o.lowerEdge = lowerEdge; o.upperEdge = upperEdge; return runWhole(x, o); }
MatrixXd notch(const MatrixXd& x, const std::vector<double>& frequencies, FilterOptions o) { o.kind = Kind::Notch; o.frequencies = frequencies; return runWhole(x, o); }

}  // namespace nxr::signal::filters
