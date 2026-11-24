// nxr::signal::graphfilterbank — see include/nxr/filterbank.h and docs/specs/2026-11-10-filterbanks.md.
// Ported line for line from Rheome (@graphfilterbank/private/designbank.m, +graphtransform/{eigen,chebyshev}.m,
// +filters/frame_{gains,bounds,analysis,synthesis}.m) and checked against nxr's filters.frames and the Studio's dyadicFrame.ts.
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/timefilterbank.h"

#include <Eigen/Cholesky>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsSolver.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nxr::signal::graphfilterbank {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kUsable = 4.744;   // (1 + u)e^(−u) = 0.05: a member's mass beyond λmax under 5 %

// MATLAB's logspace(log10(a), log10(b), n): 10.^linspace — the last point exactly b.
std::vector<double> logspace(double a, double b, int n) {
    std::vector<double> v(static_cast<std::size_t>(n));
    const double la = std::log10(a), lb = std::log10(b);
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = std::pow(10.0, i == n - 1 ? lb : la + i * (lb - la) / (n - 1));
    return v;
}

Member plain(Role role, double t, graphtransforms::Gain g) { Member m; m.role = role; m.scale = t; m.gain = std::move(g); return m; }

double itersineWindow(double u) {   // sin(π/2 cos²(πu/2)) on |u| ≤ 1
    return std::abs(u) <= 1 ? std::sin(0.5 * kPi * std::pow(std::cos(0.5 * kPi * u), 2)) : 0.0;
}

}  // namespace

// ── Range ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Range Range::fromLmax(double lmax) { Range r; r.lmax = lmax; return r; }
Range Range::fromLimits(double lmin, double lmax) { Range r; r.lmin = lmin; r.lmax = lmax; return r; }
Range Range::fromSpectrum(const Eigen::VectorXd& lambda) {
    Range r;
    r.lambda = lambda;
    std::sort(r.lambda.data(), r.lambda.data() + r.lambda.size());
    if (r.lambda.size() == 0) throw std::invalid_argument("filterbank: empty spectrum");
    r.lmax = r.lambda(r.lambda.size() - 1);
    for (Eigen::Index i = 0; i < r.lambda.size(); ++i)
        if (r.lambda(i) > r.lmax * 1e-12) { r.lmin = r.lambda(i); break; }
    return r;
}

// ── design (Rheome designbank) ───────────────────────────────────────────────────────────────────────────────────────────

GraphFilterbank design(Family family, const Range& range, const DesignOptions& o) {
    const double lmax = range.lmax, lmin = range.lmin;
    if (!std::isnan(lmin) && lmin < 0)
        throw std::invalid_argument("filterbank::design: the spectral range is SIGNED; mexhat and heat diverge for lambda < 0 — square the operator (D^2 is PSD)");
    if (!std::isfinite(lmax) || lmax <= 0) throw std::invalid_argument("filterbank::design: lmax must be finite and positive");
    if (!std::isnan(lmin) && lmax <= lmin) throw std::invalid_argument("filterbank::design: invalid range");
    if (o.numFilters != 0 && o.numFilters < 2) throw std::invalid_argument("filterbank::design: numFilters must be 0 or >= 2");
    if (o.voices < 1 || o.voices > 48) throw std::invalid_argument("filterbank::design: voices must be in [1, 48]");

    double tmin, tmax;
    if (o.scaleLimits) {
        tmin = o.scaleLimits->first; tmax = o.scaleLimits->second;
        if (!(tmin > 0 && tmax > tmin)) throw std::invalid_argument("filterbank::design: scaleLimits must be [lo hi], positive and increasing");
    } else {
        // λmin of 0 carries no scale (the DC mode): fall back to the lpFactor rule rather than t_max = Inf.
        const double lminEff0 = (std::isnan(lmin) || lmin <= 0) ? lmax / o.lpFactor : lmin;
        tmin = o.fineLimit == FineLimit::Basis ? 1.0 / lmax : kUsable / lmax;
        tmax = 2.0 / lminEff0;
    }
    const double lminEff = 2.0 / tmax;
    const int V = o.voices;

    GraphFilterbank b;
    b.lmax = lmax; b.lmin = lmin; b.lpFactor = o.lpFactor; b.lambda = range.lambda;

    switch (family) {
    case Family::Itersine: {
        b.kind = "itersine";
        const int Nf = o.numFilters ? o.numFilters : 8;
        const double overlap = 2, scale = lmax / (Nf - overlap + 1) * overlap;
        auto kf = [](double x) { return (x >= -0.5 && x <= 0.5) ? std::sin(0.5 * kPi * std::pow(std::cos(kPi * x), 2)) : 0.0; };
        for (int ii = 1; ii <= Nf; ++ii)
            b.members.push_back(plain(Role::Member, kNaN, [=](double l) { return kf(l / scale - (ii - overlap / 2) / overlap) / std::sqrt(overlap) * std::sqrt(2.0); }));
        b.total = 1.0;   // exact on [0, λmax]: neighbours half a window apart, sin² + cos² = 1
        break;
    }
    case Family::LogItersine: {
        b.kind = "logitersine";
        const double kTop = std::sqrt(lmax), kMin = std::sqrt(std::max(lminEff, kEps));
        const int J = std::max(1, static_cast<int>(std::floor(V * std::log2(kTop / kMin))));
        auto xi = [=](double l) { return V * std::log2(std::max(std::sqrt(l), std::numeric_limits<double>::min()) / kTop); };
        auto w = [](double u) { return itersineWindow(u); };
        b.members.resize(static_cast<std::size_t>(J) + 2);
        b.members[0] = plain(Role::Scaling, kNaN, [=](double l) { const double x = xi(l); const double ww = w(x + J);
            return x < -J ? std::sqrt(std::max(1 - ww * ww, 0.0)) : 0.0; });   // 1 at λ = 0 (ξ → −∞)
        for (int jj = 1; jj <= J; ++jj)   // ascending in k: coarse first
            b.members[static_cast<std::size_t>(J + 1 - jj)] = plain(Role::Member, kNaN, [=](double l) { return w(xi(l) + jj); });
        b.members.back() = plain(Role::Top, kNaN, [=](double l) { const double x = xi(l); const double ww = w(x + 1);
            return x > -1 ? std::sqrt(std::max(1 - ww * ww, 0.0)) : 0.0; });
        b.total = 1.0;
        break;
    }
    case Family::MexHat: {
        b.kind = "mexhat";
        if (tmax <= tmin) throw std::invalid_argument("filterbank::design: t_min exceeds t_max (the coarse end is finer than the fine end)");
        const int Nf = o.numFilters ? o.numFilters : std::max(2, static_cast<int>(std::round(V * std::log2(std::sqrt(tmax / tmin)))) + 1);
        const auto tw = logspace(tmin, tmax, Nf);
        // GSPBox's scaling function: quartic roll-off at 1.2x the wavelet peak
        b.members.push_back(plain(Role::Scaling, kNaN, [=](double l) { return (1.2 * std::exp(-1.0)) * std::exp(-std::pow(l / (0.4 * lminEff), 4)); }));
        for (double t : tw) b.members.push_back(plain(Role::Wavelet, t, [=](double l) { const double x = l * t; return x * std::exp(-x); }));
        break;
    }
    case Family::Heat: {
        b.kind = "heat";
        // the fine end pinned at 1/λmax regardless of the policy, one octave coarser at the top
        const double tTop = 2 * tmax;
        const int n = o.numFilters ? o.numFilters : std::max(2, static_cast<int>(std::round(V * std::log2(std::sqrt(tTop / tmin)))) + 1);
        const auto tw = logspace(tTop, tmin, n);
        for (double t : tw) b.members.push_back(plain(Role::Member, t, [=](double l) { return std::exp(-(l * t)); }));
        break;
    }
    }
    return b;
}

// ── Meyer's dyadic frame (the Studio, nxr) ───────────────────────────────────────────────────────────────────────────────

double meyerNu(double x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return std::pow(x, 4) * (35 - 84 * x + 70 * x * x - 20 * std::pow(x, 3));
}

double mexicanHat(double lambda, double centre) {
    if (!(centre > 0)) throw std::invalid_argument("filterbank::mexicanHat: the centre is above 0");
    const double x = lambda / centre;
    return x <= 0 ? 0.0 : x * std::exp(1.0 - x);
}

double mexicanHatReach(double centre, double floor) {
    if (!(centre > 0) || !(floor > 0 && floor < 1)) throw std::invalid_argument("filterbank::mexicanHatReach: a centre above 0 and a floor in (0, 1)");
    double lo = 1.0, hi = 2.0;                                  // x e^(1-x) falls from 1 at x = 1: bracket then bisect
    while (hi * std::exp(1.0 - hi) > floor) { lo = hi; hi *= 2; }
    for (int i = 0; i < 100; ++i) { const double m = 0.5 * (lo + hi); (m * std::exp(1.0 - m) > floor ? lo : hi) = m; }
    return centre * hi;
}

GraphFilterbank dyadic(const std::vector<double>& edges, double h) {
    if (edges.empty()) throw std::invalid_argument("filterbank::dyadic: no edges");
    for (std::size_t k = 1; k < edges.size(); ++k) {
        if (!(edges[k] > edges[k - 1])) throw std::invalid_argument("filterbank::dyadic: edges must ascend");
        if (std::log2(edges[k] / edges[k - 1]) < 2 * h - 1e-9)
            throw std::invalid_argument("filterbank::dyadic: edges closer than two half-widths — the transitions would overlap");
    }
    if (!(h > 0)) throw std::invalid_argument("filterbank::dyadic: halfWidth must be positive");
    GraphFilterbank b; b.kind = "meyer"; b.total = 1.0; b.halfWidth = h; b.edges = edges;
    const double inf = std::numeric_limits<double>::infinity();
    auto add = [&](Role r, double lo, double hi, double level) {
        Member m; m.role = r; m.lo = lo; m.hi = hi; m.level = level;
        m.gain = [=](double x) { return timefilterbank::meyerBand(x, lo, hi, h); };
        b.members.push_back(std::move(m));
    };
    add(Role::Scaling, 0.0, edges[0], 0);
    for (std::size_t k = 1; k < edges.size(); ++k) add(Role::Wavelet, edges[k - 1], edges[k], static_cast<double>(k));
    add(Role::Top, edges.back(), inf, static_cast<double>(edges.size()));
    return b;
}

GraphFilterbank weyl(double area, int levels, double h) {
    if (!(area > 0) || levels < 1) throw std::invalid_argument("filterbank::weyl: an area and at least one level");
    std::vector<double> e;
    for (int j = 0; j < levels; ++j) e.push_back(4 * kPi * std::pow(2.0, j) / area);
    return dyadic(e, h);
}

GraphFilterbank rank(int levels, double h) {
    if (levels < 1) throw std::invalid_argument("filterbank::rank: at least one level");
    std::vector<double> e;
    for (int j = 0; j < levels; ++j) e.push_back(std::pow(2.0, j) + 0.5);
    return dyadic(e, h);
}

Eigen::VectorXd rankCoordinates(const Eigen::VectorXd& lam, int mult) {
    std::vector<Eigen::Index> order(static_cast<std::size_t>(lam.size()));
    for (Eigen::Index i = 0; i < lam.size(); ++i) order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](Eigen::Index a, Eigen::Index b) { return std::abs(lam(a)) < std::abs(lam(b)); });
    Eigen::VectorXd out(lam.size());
    for (std::size_t r = 0; r < order.size(); ++r) out(order[r]) = static_cast<double>(r / static_cast<std::size_t>(mult) + 1);
    return out;
}

// ── the frame ────────────────────────────────────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd gains(const GraphFilterbank& bank, const Eigen::VectorXd& lambda) {
    Eigen::MatrixXd H(lambda.size(), bank.size());
    for (int m = 0; m < bank.size(); ++m)
        for (Eigen::Index i = 0; i < lambda.size(); ++i) H(i, m) = bank.members[static_cast<std::size_t>(m)].gain(lambda(i));
    return H;
}

Eigen::MatrixXd dualGains(const GraphFilterbank& bank, const Eigen::VectorXd& lambda) {
    Eigen::MatrixXd H = gains(bank, lambda);
    Eigen::VectorXd S = H.array().square().rowwise().sum();
    const double smax = S.size() ? std::max(S.maxCoeff(), std::numeric_limits<double>::min()) : 0.0;
    for (Eigen::Index i = 0; i < S.size(); ++i) {
        const bool bad = S(i) <= 1e-12 * smax;
        H.row(i) *= bad ? 0.0 : 1.0 / S(i);
    }
    return H;
}

Bounds framebounds(const GraphFilterbank& bank, const Eigen::VectorXd& lambda) {
    const Eigen::MatrixXd H = gains(bank, lambda);
    const Eigen::VectorXd S = H.array().square().rowwise().sum();
    Bounds b;
    b.A = S.minCoeff(); b.B = S.maxCoeff();
    b.tightness = b.B / std::max(b.A, kEps);
    const double thr = kEps * std::max(b.B, std::numeric_limits<double>::min());
    b.uncovered = (S.array() <= thr).cast<double>().mean();
    return b;
}

Bounds framebounds(const GraphFilterbank& bank) {
    if (bank.lambda.size()) return framebounds(bank, bank.lambda);
    if (!(bank.lmax > 0)) throw std::invalid_argument("filterbank::bounds: the bank has no spectral range; pass the spectrum");
    return framebounds(bank, Eigen::VectorXd::LinSpaced(512, 0.0, bank.lmax));
}

bool isframetight(const GraphFilterbank& bank, double tol) { return std::abs(framebounds(bank).tightness - 1) <= tol; }

Eigen::VectorXd scales(const GraphFilterbank& bank) {
    Eigen::VectorXd t(bank.size());
    for (int m = 0; m < bank.size(); ++m) t(m) = bank.members[static_cast<std::size_t>(m)].scale;
    return t;
}
Eigen::VectorXd widths(const GraphFilterbank& bank) { return (2 * scales(bank).array()).sqrt(); }

Eigen::VectorXd centerWavenumbers(const GraphFilterbank& bank) {
    if (!(bank.lmax > 0)) throw std::invalid_argument("filterbank::centerWavenumbers: the bank has no spectral range");
    const double lo = std::max(std::isnan(bank.lmin) ? bank.lmax / bank.lpFactor : bank.lmin, kEps);
    const Eigen::VectorXd lg = Eigen::VectorXd::LinSpaced(512, lo, bank.lmax);
    const Eigen::MatrixXd H = gains(bank, lg).cwiseAbs();
    Eigen::VectorXd k = Eigen::VectorXd::Zero(bank.size());
    for (int m = 0; m < bank.size(); ++m) {
        const double s = H.col(m).sum();
        if (s > 0) k(m) = std::sqrt(lg.cwiseProduct(H.col(m)).sum() / s);
    }
    return k;
}

Eigen::VectorXd massLost(const GraphFilterbank& bank) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(bank.size());
    if (bank.kind == "mexhat") {
        for (int m = 1; m < bank.size(); ++m) { const double u = bank.members[static_cast<std::size_t>(m)].scale * bank.lmax; v(m) = (1 + u) * std::exp(-u); }
        // the scaling function is a quartic roll-off, not an exponential: integrate it (trapezoid on 4096 points of [0, 6 λmax])
        const int N = 4096; double beyond = 0, all = 0;
        for (int i = 0; i + 1 < N; ++i) {
            const double l0 = 6 * bank.lmax * i / (N - 1), l1 = 6 * bank.lmax * (i + 1) / (N - 1);
            const double g0 = std::abs(bank.members[0].gain(l0)), g1 = std::abs(bank.members[0].gain(l1));
            all += 0.5 * (g0 + g1) * (l1 - l0);
            if (l0 > bank.lmax && l1 > bank.lmax) beyond += 0.5 * (g0 + g1) * (l1 - l0);   // Rheome: trapz over the samples with λ > λmax
        }
        v(0) = beyond / std::max(all, kEps);
    } else if (bank.kind == "heat") {
        for (int m = 0; m < bank.size(); ++m) v(m) = std::exp(-bank.members[static_cast<std::size_t>(m)].scale * bank.lmax);
    } else if (bank.kind == "meyer") {
        v.setConstant(kNaN);
    }
    return v;
}

Family familyOf(const std::string& name) {
    if (name == "mexhat") return Family::MexHat;
    if (name == "heat") return Family::Heat;
    if (name == "itersine") return Family::Itersine;
    if (name == "logitersine") return Family::LogItersine;
    throw std::invalid_argument("graph bank family '" + name + "' is not one of mexhat, heat, itersine, logitersine, meyer, weyl, rank");
}

GraphFilterbank graphBank(const NamedBank& b) {
    if (b.family == "meyer") return dyadic(b.edges, b.halfWidth);
    if (b.family == "weyl") return weyl(b.area, b.levels, b.halfWidth);
    if (b.family == "rank") return rank(b.levels, b.halfWidth);
    return design(familyOf(b.family), b.range, b.options);
}

}  // namespace nxr::signal::graphfilterbank
