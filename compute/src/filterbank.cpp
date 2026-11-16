// nxr::filterbank — see include/nxr/filterbank.h and docs/specs/2026-11-10-filterbanks.md.
// Ported line for line from Rheome (@graphfilterbank/private/designbank.m, +graphtransform/{eigen,chebyshev}.m,
// +filters/frame_{gains,bounds,analysis,synthesis}.m) and checked against nxr's filters.frames and the Studio's dyadicFrame.ts.
#include "nxr/filterbank.h"

#include <Eigen/Cholesky>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsSolver.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nxr::filterbank {
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

Member plain(Role role, double t, Gain g) { Member m; m.role = role; m.scale = t; m.gain = std::move(g); return m; }

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

Bank design(Family family, const Range& range, const DesignOptions& o) {
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

    Bank b;
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

namespace {
double angle(double x, double e, double h) { return (kPi / 2) * meyerNu((std::log2(x / e) + h) / (2 * h)); }
}

double meyerBand(double x, double lo, double hi, double h) {
    if (!(x > 0)) return lo > 0 ? 0.0 : 1.0;
    const double rise = lo > 0 ? std::sin(angle(x, lo, h)) : 1.0;
    const double fall = std::isfinite(hi) ? std::cos(angle(x, hi, h)) : 1.0;
    return rise * fall;
}

double morletBand(double x, double lo, double hi) {
    if (!(hi > lo)) throw std::invalid_argument("filterbank::morletBand: needs lo < hi");
    const double f0 = (lo + hi) / 2, sigma = (hi - lo) / (2 * std::sqrt(2 * std::log(2.0)));
    const double z = (std::fabs(x) - f0) / sigma;
    return std::exp(-0.5 * z * z);
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

Bank dyadic(const std::vector<double>& edges, double h) {
    if (edges.empty()) throw std::invalid_argument("filterbank::dyadic: no edges");
    for (std::size_t k = 1; k < edges.size(); ++k) {
        if (!(edges[k] > edges[k - 1])) throw std::invalid_argument("filterbank::dyadic: edges must ascend");
        if (std::log2(edges[k] / edges[k - 1]) < 2 * h - 1e-9)
            throw std::invalid_argument("filterbank::dyadic: edges closer than two half-widths — the transitions would overlap");
    }
    if (!(h > 0)) throw std::invalid_argument("filterbank::dyadic: halfWidth must be positive");
    Bank b; b.kind = "meyer"; b.total = 1.0; b.halfWidth = h; b.edges = edges;
    const double inf = std::numeric_limits<double>::infinity();
    auto add = [&](Role r, double lo, double hi, double level) {
        Member m; m.role = r; m.lo = lo; m.hi = hi; m.level = level;
        m.gain = [=](double x) { return meyerBand(x, lo, hi, h); };
        b.members.push_back(std::move(m));
    };
    add(Role::Scaling, 0.0, edges[0], 0);
    for (std::size_t k = 1; k < edges.size(); ++k) add(Role::Wavelet, edges[k - 1], edges[k], static_cast<double>(k));
    add(Role::Top, edges.back(), inf, static_cast<double>(edges.size()));
    return b;
}

Bank weyl(double area, int levels, double h) {
    if (!(area > 0) || levels < 1) throw std::invalid_argument("filterbank::weyl: an area and at least one level");
    std::vector<double> e;
    for (int j = 0; j < levels; ++j) e.push_back(4 * kPi * std::pow(2.0, j) / area);
    return dyadic(e, h);
}

Bank rank(int levels, double h) {
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

Eigen::MatrixXd gains(const Bank& bank, const Eigen::VectorXd& lambda) {
    Eigen::MatrixXd H(lambda.size(), bank.size());
    for (int m = 0; m < bank.size(); ++m)
        for (Eigen::Index i = 0; i < lambda.size(); ++i) H(i, m) = bank.members[static_cast<std::size_t>(m)].gain(lambda(i));
    return H;
}

Eigen::MatrixXd dualGains(const Bank& bank, const Eigen::VectorXd& lambda) {
    Eigen::MatrixXd H = gains(bank, lambda);
    Eigen::VectorXd S = H.array().square().rowwise().sum();
    const double smax = S.size() ? std::max(S.maxCoeff(), std::numeric_limits<double>::min()) : 0.0;
    for (Eigen::Index i = 0; i < S.size(); ++i) {
        const bool bad = S(i) <= 1e-12 * smax;
        H.row(i) *= bad ? 0.0 : 1.0 / S(i);
    }
    return H;
}

Bounds bounds(const Bank& bank, const Eigen::VectorXd& lambda) {
    const Eigen::MatrixXd H = gains(bank, lambda);
    const Eigen::VectorXd S = H.array().square().rowwise().sum();
    Bounds b;
    b.A = S.minCoeff(); b.B = S.maxCoeff();
    b.tightness = b.B / std::max(b.A, kEps);
    const double thr = kEps * std::max(b.B, std::numeric_limits<double>::min());
    b.uncovered = (S.array() <= thr).cast<double>().mean();
    return b;
}

Bounds bounds(const Bank& bank) {
    if (bank.lambda.size()) return bounds(bank, bank.lambda);
    if (!(bank.lmax > 0)) throw std::invalid_argument("filterbank::bounds: the bank has no spectral range; pass the spectrum");
    return bounds(bank, Eigen::VectorXd::LinSpaced(512, 0.0, bank.lmax));
}

bool isTight(const Bank& bank, double tol) { return std::abs(bounds(bank).tightness - 1) <= tol; }

Eigen::VectorXd scales(const Bank& bank) {
    Eigen::VectorXd t(bank.size());
    for (int m = 0; m < bank.size(); ++m) t(m) = bank.members[static_cast<std::size_t>(m)].scale;
    return t;
}
Eigen::VectorXd widths(const Bank& bank) { return (2 * scales(bank).array()).sqrt(); }

Eigen::VectorXd centerWavenumbers(const Bank& bank) {
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

Eigen::VectorXd massLost(const Bank& bank) {
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

// ── Chebyshev ────────────────────────────────────────────────────────────────────────────────────────────────────────────

Eigen::VectorXd chebyshevCoefficients(const Gain& g, double lmax, int K) {
    const double a = lmax / 2;
    Eigen::VectorXd gj(K + 1), c(K + 1);
    for (int j = 0; j <= K; ++j) gj(j) = g(a * (std::cos(kPi * (j + 0.5) / (K + 1)) + 1));
    for (int k = 0; k <= K; ++k) {
        double s = 0;
        for (int j = 0; j <= K; ++j) s += gj(j) * std::cos(kPi * k * (j + 0.5) / (K + 1));
        c(k) = (2.0 / (K + 1)) * s;
    }
    c(0) /= 2;
    return c;
}

double chebyshevEval(const Eigen::VectorXd& c, double lmax, double lambda) {
    const double t = 2 * lambda / lmax - 1;
    double b1 = 0, b2 = 0;
    for (Eigen::Index k = c.size() - 1; k >= 1; --k) { const double b0 = 2 * t * b1 - b2 + c(k); b2 = b1; b1 = b0; }
    return c(0) + t * b1 - b2;
}

double estimateLmax(const Eigen::SparseMatrix<double>& L, const Eigen::SparseMatrix<double>& mass) {
    const Eigen::Index n = L.rows();
    Eigen::SparseMatrix<double> B = mass;
    if (B.rows() == 0) { B.resize(n, n); B.setIdentity(); }
    try {
        if (n <= 40) {   // small: exact
            Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Eigen::MatrixXd(L), Eigen::MatrixXd(B), Eigen::EigenvaluesOnly);
            return std::abs(es.eigenvalues()(n - 1)) > std::abs(es.eigenvalues()(0)) ? std::abs(es.eigenvalues()(n - 1)) : std::abs(es.eigenvalues()(0));
        }
        const int ncv = static_cast<int>(std::min<Eigen::Index>(20, n));
        Spectra::SparseSymMatProd<double> opA(L);
        Spectra::SparseCholesky<double> opB(B);
        Spectra::SymGEigsSolver<Spectra::SparseSymMatProd<double>, Spectra::SparseCholesky<double>, Spectra::GEigsMode::Cholesky> s(opA, opB, 1, ncv);
        s.init();
        s.compute(Spectra::SortRule::LargestMagn, 500, 1e-4);
        if (s.info() == Spectra::CompInfo::Successful) return std::abs(s.eigenvalues()(0));
    } catch (...) { /* fall through */ }
    return 0.0;
}

struct ChebyshevOperator::Impl {
    Eigen::SparseMatrix<double> L, M;
    bool hasMass = false;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt;
    double lmax = 0;
    int order = 40;
    Eigen::MatrixXd Aop(const Eigen::MatrixXd& X) const {
        Eigen::MatrixXd Y = L * X;
        return hasMass ? Eigen::MatrixXd(llt.solve(Y)) : Y;
    }
};

ChebyshevOperator::ChebyshevOperator(const Eigen::SparseMatrix<double>& L, const Eigen::SparseMatrix<double>& mass, double lmax, int order)
    : p_(std::make_unique<Impl>()) {
    if (L.rows() != L.cols()) throw std::invalid_argument("ChebyshevOperator: L must be square");
    if (order < 1) throw std::invalid_argument("ChebyshevOperator: order must be >= 1");
    p_->L = L; p_->order = order;
    if (mass.rows() != 0) {
        if (mass.rows() != L.rows() || mass.cols() != L.cols()) throw std::invalid_argument("ChebyshevOperator: mass must match L");
        p_->M = mass; p_->hasMass = true;
        p_->llt.compute(p_->M);   // factor once, reuse every recursion step
        if (p_->llt.info() != Eigen::Success) throw std::invalid_argument("ChebyshevOperator: the mass is not positive definite");
    }
    // λmax MUST bound the operator's whole spectrum: a smaller value makes T_k grow like (2x)^k and the recursion diverge.
    double lest = estimateLmax(L, mass);
    if (lest <= 0) {   // power iteration, deterministic: a lower bound
        std::mt19937 rng(0); std::normal_distribution<double> nd;
        Eigen::MatrixXd x(L.rows(), 1); for (Eigen::Index i = 0; i < x.rows(); ++i) x(i) = nd(rng);
        x.normalize();
        for (int i = 0; i < 100; ++i) { Eigen::MatrixXd y = p_->Aop(x); const double nv = y.norm(); if (nv == 0) break; x = y / nv; lest = std::abs((x.transpose() * p_->Aop(x))(0)); }
    }
    if (lmax <= 0) lmax = 1.01 * lest;
    else if (lmax < 0.99 * lest)
        throw std::invalid_argument("ChebyshevOperator: lmax is below the operator's spectrum (at least " + std::to_string(lest) +
                                    "); the recursion would diverge. Omit lmax to have it estimated.");
    p_->lmax = lmax;
}
ChebyshevOperator::~ChebyshevOperator() = default;
ChebyshevOperator::ChebyshevOperator(ChebyshevOperator&&) noexcept = default;
ChebyshevOperator& ChebyshevOperator::operator=(ChebyshevOperator&&) noexcept = default;
int ChebyshevOperator::rows() const { return static_cast<int>(p_->L.rows()); }
double ChebyshevOperator::lmax() const { return p_->lmax; }
int ChebyshevOperator::order() const { return p_->order; }

Eigen::MatrixXd ChebyshevOperator::apply(const Gain& g, const Eigen::MatrixXd& X) const {
    const int K = p_->order;
    const Eigen::VectorXd c = chebyshevCoefficients(g, p_->lmax, K);
    const double a = p_->lmax / 2;
    // Clenshaw on Ltil = L/a − I, which maps [0, λmax] onto [−1, 1]
    auto Ltil = [&](const Eigen::MatrixXd& Y) -> Eigen::MatrixXd { return p_->Aop(Y) / a - Y; };
    Eigen::MatrixXd b2 = Eigen::MatrixXd::Zero(X.rows(), X.cols()), b1 = b2;
    for (int k = K; k >= 1; --k) {
        Eigen::MatrixXd bk = 2 * Ltil(b1) - b2 + c(k) * X;
        b2 = std::move(b1); b1 = std::move(bk);
    }
    return Ltil(b1) - b2 + c(0) * X;
}

Eigen::MatrixXcd ChebyshevOperator::apply(const Gain& g, const Eigen::MatrixXcd& X) const {
    Eigen::MatrixXcd Y(X.rows(), X.cols());
    Y.real() = apply(g, Eigen::MatrixXd(X.real()));
    Y.imag() = apply(g, Eigen::MatrixXd(X.imag()));
    return Y;
}

// ── analysis and synthesis ───────────────────────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd forward(const EigenTransform& t, const Eigen::MatrixXd& F) {
    if (t.phi.rows() != F.rows()) throw std::invalid_argument("filterbank: field rows differ from the eigenbasis rows");
    return t.mass.rows() ? Eigen::MatrixXd(t.phi.transpose() * (t.mass * F)) : Eigen::MatrixXd(t.phi.transpose() * F);
}
Eigen::MatrixXcd forward(const EigenTransform& t, const Eigen::MatrixXcd& F) {
    if (t.phi.rows() != F.rows()) throw std::invalid_argument("filterbank: field rows differ from the eigenbasis rows");
    const Eigen::MatrixXcd phiT = t.phi.transpose().cast<std::complex<double>>();
    if (!t.mass.rows()) return phiT * F;
    Eigen::MatrixXcd BF(F.rows(), F.cols());
    BF.real() = t.mass * Eigen::MatrixXd(F.real());
    BF.imag() = t.mass * Eigen::MatrixXd(F.imag());
    return phiT * BF;
}
Eigen::MatrixXd inverse(const EigenTransform& t, const Eigen::MatrixXd& C) {
    if (t.phi.cols() != C.rows()) throw std::invalid_argument("filterbank: coefficient rows differ from the number of modes");
    return t.phi * C;
}
Eigen::MatrixXcd inverse(const EigenTransform& t, const Eigen::MatrixXcd& C) {
    if (t.phi.cols() != C.rows()) throw std::invalid_argument("filterbank: coefficient rows differ from the number of modes");
    return t.phi.cast<std::complex<double>>() * C;
}

namespace {

Eigen::MatrixXcd join(const Eigen::MatrixXd& re, const Eigen::MatrixXd& im) {
    Eigen::MatrixXcd z(re.rows(), re.cols()); z.real() = re; z.imag() = im; return z;
}

template <class Op> CoefficientsC complexOf(const Eigen::MatrixXcd& F, Op op) {   // op(real matrix) → Coefficients
    Coefficients re = op(Eigen::MatrixXd(F.real())), im = op(Eigen::MatrixXd(F.imag()));
    CoefficientsC out;
    for (std::size_t m = 0; m < re.size(); ++m) out.push_back(join(re[m], im[m]));
    return out;
}

void split(const CoefficientsC& W, Coefficients& re, Coefficients& im) {
    for (const auto& w : W) { re.emplace_back(w.real()); im.emplace_back(w.imag()); }
}

void checkMembers(const Bank& bank, std::size_t nW) {
    if (nW != static_cast<std::size_t>(bank.size())) throw std::invalid_argument("filterbank: one coefficient block per member is required");
}

}  // namespace

Coefficients analysis(const Bank& bank, const EigenTransform& t, const Eigen::MatrixXd& F) {
    if (t.phi.cols() != t.lambda.size()) throw std::invalid_argument("filterbank: phi and lambda disagree");
    const Eigen::MatrixXd C = forward(t, F);
    const Eigen::MatrixXd H = gains(bank, t.lambda);
    Coefficients W;
    for (int m = 0; m < bank.size(); ++m) W.emplace_back(t.phi * (H.col(m).asDiagonal() * C));
    return W;
}
CoefficientsC analysis(const Bank& bank, const EigenTransform& t, const Eigen::MatrixXcd& F) {
    return complexOf(F, [&](const Eigen::MatrixXd& X) { return analysis(bank, t, X); });
}

Coefficients analysis(const Bank& bank, const ChebyshevOperator& t, const Eigen::MatrixXd& F) {
    Coefficients W;
    for (const auto& m : bank.members) W.emplace_back(t.apply(m.gain, F));
    return W;
}
CoefficientsC analysis(const Bank& bank, const ChebyshevOperator& t, const Eigen::MatrixXcd& F) {
    return complexOf(F, [&](const Eigen::MatrixXd& X) { return analysis(bank, t, X); });
}

static const Member& memberOf(const Bank& bank, int member) {
    if (member < 0 || member >= bank.size()) throw std::invalid_argument("filterbank: member " + std::to_string(member) + " is outside the bank (0 … " + std::to_string(bank.size() - 1) + ")");
    return bank.members[static_cast<std::size_t>(member)];
}
Eigen::MatrixXd analysisMember(const Bank& bank, int member, const EigenTransform& t, const Eigen::MatrixXd& F) {
    const Member& m = memberOf(bank, member);
    if (t.phi.cols() != t.lambda.size()) throw std::invalid_argument("filterbank: phi and lambda disagree");
    Eigen::VectorXd g(t.lambda.size());
    for (Eigen::Index k = 0; k < g.size(); ++k) g(k) = m.gain(t.lambda(k));
    return t.phi * (g.asDiagonal() * forward(t, F));
}
Eigen::MatrixXd analysisMember(const Bank& bank, int member, const ChebyshevOperator& t, const Eigen::MatrixXd& F) { return t.apply(memberOf(bank, member).gain, F); }

double chebyshevError(const Gain& g, double lmax, int order, int samples) {
    const Eigen::VectorXd c = chebyshevCoefficients(g, lmax, order);
    double worst = 0;
    for (int i = 0; i <= samples; ++i) {
        const double l = lmax * i / samples;
        worst = std::max(worst, std::abs(g(l) - chebyshevEval(c, lmax, l)));
    }
    return worst;
}

Eigen::MatrixXd synthesis(const Bank& bank, const EigenTransform& t, const Coefficients& W, Synthesis mode) {
    checkMembers(bank, W.size());
    const Eigen::MatrixXd H = mode == Synthesis::Dual ? dualGains(bank, t.lambda) : gains(bank, t.lambda);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(t.phi.cols(), W[0].cols());
    for (int m = 0; m < bank.size(); ++m) C += H.col(m).asDiagonal() * forward(t, W[static_cast<std::size_t>(m)]);
    return t.phi * C;
}
Eigen::MatrixXcd synthesis(const Bank& bank, const EigenTransform& t, const CoefficientsC& W, Synthesis mode) {
    Coefficients re, im; split(W, re, im);
    return join(synthesis(bank, t, re, mode), synthesis(bank, t, im, mode));
}

Eigen::MatrixXd synthesis(const Bank& bank, const ChebyshevOperator& t, const Coefficients& W, Synthesis mode) {
    checkMembers(bank, W.size());
    // the dual weights g/S as functions of λ; S is covered wherever it exceeds 1e-12 of its largest value on [0, λmax]
    double smax = 0;
    for (int i = 0; i < 512; ++i) {
        const double l = t.lmax() * i / 511.0; double s = 0;
        for (const auto& m : bank.members) s += m.gain(l) * m.gain(l);
        smax = std::max(smax, s);
    }
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(W[0].rows(), W[0].cols());
    for (int m = 0; m < bank.size(); ++m) {
        const Gain& g = bank.members[static_cast<std::size_t>(m)].gain;
        Gain w = g;
        if (mode == Synthesis::Dual)
            w = [&bank, &g, smax](double l) { double s = 0; for (const auto& q : bank.members) s += q.gain(l) * q.gain(l);
                                              return s <= 1e-12 * smax ? 0.0 : g(l) / s; };
        F += t.apply(w, W[static_cast<std::size_t>(m)]);
    }
    return F;
}
Eigen::MatrixXcd synthesis(const Bank& bank, const ChebyshevOperator& t, const CoefficientsC& W, Synthesis mode) {
    Coefficients re, im; split(W, re, im);
    return join(synthesis(bank, t, re, mode), synthesis(bank, t, im, mode));
}

Eigen::MatrixXd scaleEnergy(const Coefficients& W) {
    Eigen::MatrixXd E = Eigen::MatrixXd::Zero(W.at(0).rows(), W[0].cols());
    for (const auto& w : W) E += w.array().square().matrix();
    return E;
}
Eigen::MatrixXd memberEnergy(const Coefficients& W) {
    Eigen::MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) E.row(static_cast<Eigen::Index>(m)) = W[m].array().square().colwise().sum();
    return E;
}

Eigen::MatrixXd memberEnergy(const Coefficients& W, const Eigen::VectorXd& weights) {
    Eigen::MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) {
        if (W[m].rows() != weights.size()) throw std::invalid_argument("memberEnergy: " + std::to_string(weights.size()) + " weights for " + std::to_string(W[m].rows()) + " vertices");
        E.row(static_cast<Eigen::Index>(m)) = (W[m].array().square().colwise() * weights.array()).colwise().sum();
    }
    return E;
}

Eigen::MatrixXd memberEnergy(const CoefficientsC& W) {
    Eigen::MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) E.row(static_cast<Eigen::Index>(m)) = W[m].array().abs2().colwise().sum();
    return E;
}

Eigen::MatrixXd memberEnergy(const CoefficientsC& W, const Eigen::VectorXd& weights) {
    Eigen::MatrixXd E(static_cast<Eigen::Index>(W.size()), W.at(0).cols());
    for (std::size_t m = 0; m < W.size(); ++m) {
        if (W[m].rows() != weights.size()) throw std::invalid_argument("memberEnergy: " + std::to_string(weights.size()) + " weights for " + std::to_string(W[m].rows()) + " vertices");
        E.row(static_cast<Eigen::Index>(m)) = (W[m].array().abs2().colwise() * weights.array()).colwise().sum();
    }
    return E;
}

BandMoments bandMoments(const Bank& bank, const Eigen::VectorXd& lambda, const Eigen::MatrixXcd& C, bool withMatrices) {
    if (C.rows() != C.cols() || C.rows() != lambda.size())
        throw std::invalid_argument("bandMoments: C is " + std::to_string(C.rows()) + " x " + std::to_string(C.cols()) + " for " + std::to_string(lambda.size()) + " modes");
    const Eigen::MatrixXd G = gains(bank, lambda);                         // [K x M]
    BandMoments r;
    r.power = G.array().square().matrix().transpose() * C.diagonal().real();
    if (withMatrices) {
        r.matrices.reserve(static_cast<std::size_t>(G.cols()));
        for (Eigen::Index m = 0; m < G.cols(); ++m) {
            const Eigen::VectorXcd g = G.col(m).cast<std::complex<double>>();
            r.matrices.push_back(g.asDiagonal() * C * g.asDiagonal());
        }
    }
    return r;
}

Eigen::VectorXd bandPower(const Bank& bank, const Eigen::VectorXd& lambda, const Eigen::VectorXd& modePower) {
    if (modePower.size() != lambda.size()) throw std::invalid_argument("bandPower: " + std::to_string(modePower.size()) + " mode powers for " + std::to_string(lambda.size()) + " modes");
    return gains(bank, lambda).array().square().matrix().transpose() * modePower;
}

}  // namespace nxr::filterbank
