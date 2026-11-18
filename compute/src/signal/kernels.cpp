// nxr::signal — see include/nxr/spacetime_kernels.h. Ported from rheome/+rheome/+filters.
#include "nxr/signal/kernels.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nxr::signal {

using cd = std::complex<double>;
static constexpr double kPi = 3.14159265358979323846;
static constexpr double kEps = 2.220446049250313e-16;

VectorXd signedFrequencies(const VectorXd& w, Half half) {
    if (half == Half::Positive) return w;      // a one-sided grid 0 … fs/2: every bin is non-negative, nothing wraps
    if (w.size() < 2) throw std::invalid_argument("signedFrequencies: needs at least two bins");
    const double Fs = static_cast<double>(w.size()) * (w(1) - w(0));
    VectorXd s = w;
    for (Eigen::Index i = 0; i < s.size(); ++i) s(i) = w(i) - Fs * (w(i) >= Fs / 2 ? 1.0 : 0.0);
    return s;
}

MatrixXd gabor(const VectorXd& lambda, const VectorXd& f, double k0, double f0, double sf) {
    const double sk = std::max(k0 / 3, kEps);
    MatrixXd G(lambda.size(), f.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) {
        const double k = std::sqrt(std::max(lambda(i), 0.0));
        const double sl = std::exp(-std::pow(k - k0, 2) / (2 * sk * sk));
        for (Eigen::Index j = 0; j < f.size(); ++j)
            G(i, j) = sl * (std::exp(-std::pow(f(j) - f0, 2) / (2 * sf * sf)) + std::exp(-std::pow(f(j) + f0, 2) / (2 * sf * sf)));
    }
    return G;
}

MatrixXd travwave(const VectorXd& lambda, const VectorXd& f, double c, double width, int sense) {
    if (sense < -1 || sense > 1) throw std::invalid_argument("travwave: sense is +1, -1 or 0");
    const double w = std::max(width, kEps);
    MatrixXd G(lambda.size(), f.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) {
        const double fr = c * std::sqrt(std::max(lambda(i), 0.0)) / (2 * kPi);
        for (Eigen::Index j = 0; j < f.size(); ++j) G(i, j) = std::exp(-std::pow((sense == 0 ? std::abs(f(j)) : f(j)) - (sense == 0 ? fr : sense * fr), 2) / (2 * w * w));
    }
    return G;
}

MatrixXcd resonator(const VectorXd& lambda, const VectorXd& f, double f0, double Q) {
    f0 = std::max(f0, kEps); Q = std::max(Q, kEps);
    MatrixXcd G(lambda.size(), f.size());
    for (Eigen::Index j = 0; j < f.size(); ++j) {
        const cd H = (f0 * f0) / (cd(f0 * f0 - f(j) * f(j), f(j) * (f0 / Q)));
        for (Eigen::Index i = 0; i < lambda.size(); ++i) G(i, j) = H;
    }
    return G;
}

MatrixXd stmatern(const VectorXd& lambda, const VectorXd& f, double kappa, double nu, double v) {
    MatrixXd G(lambda.size(), f.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i)
        for (Eigen::Index j = 0; j < f.size(); ++j) G(i, j) = std::pow(kappa * kappa + lambda(i) + std::pow(f(j) / v, 2), -nu);
    return G;
}

VectorXd temporalBandpass(const VectorXd& f, double f0, double bw) {
    const double b = std::max(bw, kEps);
    VectorXd g(f.size());
    for (Eigen::Index j = 0; j < f.size(); ++j) g(j) = std::exp(-std::pow(std::abs(f(j)) - f0, 2) / (2 * b * b));
    return g;
}

VectorXd raisedCosineBand(const VectorXd& f, double f1, double f2, double width) {
    const double bw = f2 - f1;
    if (!(bw > 0)) throw std::invalid_argument("raisedCosineBand: the band must increase");
    width = std::max(0.0, std::min(width, bw / 2));
    VectorXd g(f.size());
    for (Eigen::Index j = 0; j < f.size(); ++j) {
        const double x = f(j);
        double v = (x > f1 && x < f2) ? 1.0 : 0.0;
        if (width > 0) {
            if (x > f1 && x < f1 + width) v = 0.5 * (1 - std::cos(kPi * (x - f1) / width));
            if (x > f2 - width && x < f2) v = 0.5 * (1 + std::cos(kPi * (x - (f2 - width)) / width));
        }
        if (x == f1 || x == f2) v = 0.0;
        g(j) = std::max(v, 0.0);
    }
    return g;
}

static double lmaxOf(const VectorXd& lambda, double lmax) { return lmax > 0 ? lmax : lambda.maxCoeff(); }

MatrixXd diffusion(const VectorXd& lambda, const VectorXd& t, double D) {
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::exp(-D * lambda(i) * std::abs(t(j)));
    return G;
}

// θ_k = acos(clamp(1 − 2(λ + shift)/λmax)): the Chebyshev change of variable
static VectorXd theta(const VectorXd& lambda, double lmax, double shift) {
    const double sc = 2 / std::max(lmax, kEps);
    VectorXd th(lambda.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) th(i) = std::acos(std::max(std::min(1 - sc * (lambda(i) + shift), 1.0), -1.0));
    return th;
}

MatrixXd wave(const VectorXd& lambda, const VectorXd& t, double c) {
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) { const double om = c * std::sqrt(std::max(lambda(i), 0.0)); for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::cos(om * t(j)); }
    return G;
}

MatrixXd wave(const VectorXd& lambda, const VectorXd& t, double c, Variant variant, double lmax) {
    if (variant == Variant::Continuum) return wave(lambda, t, c);
    const VectorXd th = theta(lambda, lmaxOf(lambda, lmax), 0.0);
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::cos(c * (th(i) * t(j)));
    return G;
}

MatrixXd dampedwave(const VectorXd& lambda, const VectorXd& t, double c, double beta, Variant variant, double lmax) {
    MatrixXd G = wave(lambda, t, c, variant, lmax);
    for (Eigen::Index j = 0; j < t.size(); ++j) G.col(j) *= std::exp(-beta * std::abs(t(j)));
    return G;
}
MatrixXd dampedwave(const VectorXd& lambda, const VectorXd& t, double c, double beta) { return dampedwave(lambda, t, c, beta, Variant::Continuum, 0.0); }

MatrixXd kleingordon(const VectorXd& lambda, const VectorXd& t, double c, double m) {
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) { const double om = std::sqrt(c * c * std::max(lambda(i), 0.0) + m * m); for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::cos(om * t(j)); }
    return G;
}

MatrixXd kleingordon(const VectorXd& lambda, const VectorXd& t, double c, double mu, Variant variant, double lmax) {
    if (variant == Variant::Continuum) return kleingordon(lambda, t, c, mu);
    const VectorXd th = theta(lambda, lmaxOf(lambda, lmax), mu);
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::cos(c * (th(i) * t(j)));
    return G;
}

MatrixXcd schrodinger(const VectorXd& lambda, const VectorXd& t, double hbar) {
    if (hbar == 0) throw std::invalid_argument("schrodinger: hbar must be nonzero");
    MatrixXcd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::exp(cd(0, -lambda(i) * t(j) / hbar));
    return G;
}

MatrixXd heat(const VectorXd& lambda, const VectorXd& t, double lmax) {
    if ((t.array() < 0).any()) throw std::invalid_argument("heat: t >= 0");
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t.size(); ++j) G(i, j) = std::exp(-(lmax > 0 ? lambda(i) / lmax : lambda(i)) * t(j));
    return G;
}

MatrixXd mexhat(const VectorXd& lambda, const VectorXd& t) {
    if ((t.array() < 0).any()) throw std::invalid_argument("mexhat: t >= 0");
    MatrixXd G(lambda.size(), t.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t.size(); ++j) { const double tl = lambda(i) * t(j); G(i, j) = tl * std::exp(-tl); }
    return G;
}

MatrixXd diffgauss(const VectorXd& lambda, const VectorXd& t1, const VectorXd& t2) {
    if (t1.size() != t2.size()) throw std::invalid_argument("diffgauss: t1 and t2 must have the same number of elements");
    if ((t1.array() < 0).any()) throw std::invalid_argument("diffgauss: t1 >= 0");
    if ((t1.array() >= t2.array()).any()) throw std::invalid_argument("diffgauss: t1 < t2 (a band-pass with non-negative gain)");
    MatrixXd G(lambda.size(), t1.size());
    for (Eigen::Index i = 0; i < lambda.size(); ++i) for (Eigen::Index j = 0; j < t1.size(); ++j) G(i, j) = std::exp(-lambda(i) * t1(j)) - std::exp(-lambda(i) * t2(j));
    return G;
}

VectorXd sphere_kernel(const VectorXd& gamma, const VectorXd& gl) {
    if (gl.size() < 1) throw std::invalid_argument("sphere_kernel: needs the gains of degree 0 at least");
    const Eigen::Index L = gl.size() - 1;
    VectorXd K(gamma.size());
    for (Eigen::Index i = 0; i < gamma.size(); ++i) {
        const double x = std::cos(gamma(i));
        double Plm1 = 1.0, k = gl(0) / (4 * kPi);
        if (L >= 1) {
            double Pl = x; k += gl(1) * 3 / (4 * kPi) * Pl;
            for (Eigen::Index l = 2; l <= L; ++l) {
                const double Pn = ((2.0 * l - 1) * x * Pl - (l - 1.0) * Plm1) / static_cast<double>(l);
                k += gl(l) * (2.0 * l + 1) / (4 * kPi) * Pn;
                Plm1 = Pl; Pl = Pn;
            }
        }
        K(i) = k;
    }
    return K;
}

// Hamilton product, [w x y z]
static Eigen::Vector4d qmul(const Eigen::Vector4d& a, const Eigen::Vector4d& b) {
    return Eigen::Vector4d(a(0) * b(0) - a(1) * b(1) - a(2) * b(2) - a(3) * b(3),
                           a(0) * b(1) + a(1) * b(0) + a(2) * b(3) - a(3) * b(2),
                           a(0) * b(2) - a(1) * b(3) + a(2) * b(0) + a(3) * b(1),
                           a(0) * b(3) + a(1) * b(2) - a(2) * b(1) + a(3) * b(0));
}

MatrixXd steer(const MatrixXd& W, const Eigen::Vector4d& q0in) {
    if (W.rows() % 4 != 0) throw std::invalid_argument("steer: the row count must be a multiple of 4 (a quaternion field)");
    const Eigen::Vector4d p = q0in / std::max(q0in.norm(), kEps);
    MatrixXd Wq(W.rows(), W.cols());
    for (Eigen::Index c = 0; c < W.cols(); ++c)
        for (Eigen::Index r = 0; r < W.rows(); r += 4) Wq.block<4, 1>(r, c) = qmul(W.block<4, 1>(r, c), p);   // q ⊗ p  (right multiplication by p)
    return Wq;
}

MatrixXd spin(const MatrixXd& V, const MatrixXd& axis, const VectorXd& theta) {
    const Eigen::Index n = V.rows();
    if (V.cols() != 3) throw std::invalid_argument("spin: V must be [n x 3]");
    if (!(axis.rows() == n || axis.rows() == 1) || axis.cols() != 3) throw std::invalid_argument("spin: axis must be [n x 3] or [1 x 3]");
    if (!(theta.size() == n || theta.size() == 1)) throw std::invalid_argument("spin: theta must be [n] or a scalar");
    MatrixXd out(n, 3);
    for (Eigen::Index i = 0; i < n; ++i) {
        Eigen::Vector3d ax = axis.row(axis.rows() == 1 ? 0 : i);
        ax /= std::max(ax.norm(), kEps);
        const double h = 0.5 * theta(theta.size() == 1 ? 0 : i);
        const Eigen::Vector4d q(std::cos(h), std::sin(h) * ax(0), std::sin(h) * ax(1), std::sin(h) * ax(2));
        const Eigen::Vector4d qc(q(0), -q(1), -q(2), -q(3));
        const Eigen::Vector4d v(0, V(i, 0), V(i, 1), V(i, 2));
        const Eigen::Vector4d r = qmul(qmul(q, v), qc);
        out.row(i) = r.tail<3>().transpose();
    }
    return out;
}

std::vector<bool> lowPass(const signal::graphfilterbank::GraphFilterbank& bank) {
    std::vector<bool> low(bank.members.size(), false);
    if (bank.kind == "heat") return low;
    for (std::size_t m = 0; m < bank.members.size(); ++m) {
        if (bank.members[m].role == signal::graphfilterbank::Role::Scaling) { low[m] = true; continue; }
        if (!(bank.lmax > 0)) continue;
        double peak = 0, g0 = std::abs(bank.members[m].gain(0.0));
        for (int i = 0; i < 512; ++i) peak = std::max(peak, std::abs(bank.members[m].gain(bank.lmax * i / 511.0)));
        low[m] = g0 > 0.5 * peak;
    }
    return low;
}

Scatter scatter(const signal::graphfilterbank::GraphFilterbank& bank, const signal::graphfilterbank::EigenTransform& t, const MatrixXcd& F, int order, const MatrixXd& poolIn) {
    if (order != 1 && order != 2) throw std::invalid_argument("scatter: order is 1 or 2");
    const Eigen::Index nV = t.phi.rows(), nT = F.cols();
    if (F.rows() != nV) throw std::invalid_argument("scatter: F must have one row per vertex of the basis");
    const MatrixXd H = signal::graphfilterbank::gains(bank, t.lambda);                    // [K × M]
    const int M = static_cast<int>(H.cols());
    MatrixXd w;
    if (poolIn.size() == 0) {
        VectorXd a = t.mass.size() > 0 ? VectorXd(t.mass * VectorXd::Ones(nV)) : VectorXd::Ones(nV);
        w = a / std::max(a.sum(), std::numeric_limits<double>::min());
    } else {
        w = poolIn;
        if (w.rows() != nV) throw std::invalid_argument("scatter: pool must have one row per vertex");
    }
    const MatrixXd wt = w.transpose();                                       // [nR × nV]
    Scatter S;
    // a member's scale is 1/(its centre wavenumber), defined for every family; the low-pass has none (NaN) and takes no part in the pairing
    VectorXd kc;
    if (bank.lmax > 0) kc = signal::graphfilterbank::centerWavenumbers(bank);
    else {   // a bank with no range of its own (Meyer): the gain-weighted centroid of √λ over the transform's spectrum
        kc = VectorXd::Zero(M);
        for (int m = 0; m < M; ++m) { const double s = H.col(m).cwiseAbs().sum(); if (s > 0) kc(m) = (t.lambda.array().max(0.0).sqrt() * H.col(m).cwiseAbs().array()).sum() / s; }
    }
    const std::vector<bool> low = lowPass(bank);
    S.sigma = VectorXd::Constant(M, std::numeric_limits<double>::quiet_NaN());
    for (int m = 0; m < M; ++m) if (!low[static_cast<std::size_t>(m)] && kc(m) > 0 && std::isfinite(kc(m))) S.sigma(m) = 1.0 / kc(m);
    const MatrixXd PhiT = t.phi.transpose();
    auto B = [&](const auto& X) { return t.mass.size() > 0 ? MatrixXcd((t.mass.template cast<cd>()) * X.template cast<cd>()) : MatrixXcd(X.template cast<cd>()); };
    const MatrixXcd Cb = PhiT.cast<cd>() * B(F);
    S.S0 = wt.cast<cd>() * F;
    for (int m1 = 0; m1 < M; ++m1) {
        MatrixXcd G1 = Cb; for (Eigen::Index j = 0; j < nT; ++j) G1.col(j).array() *= H.col(m1).array();
        const MatrixXd U1 = (t.phi.cast<cd>() * G1).cwiseAbs();
        S.S1.push_back(wt * U1);
        if (order < 2) continue;
        const MatrixXd C1 = PhiT * (t.mass.size() > 0 ? MatrixXd(t.mass * U1) : U1);
        for (int m2 = 0; m2 < M; ++m2) {
            if (!(S.sigma(m2) > S.sigma(m1))) continue;       // NaN (a low-pass) compares false: no pair
            MatrixXd G2 = C1; for (Eigen::Index j = 0; j < nT; ++j) G2.col(j).array() *= H.col(m2).array();
            S.pairs.push_back({m1, m2});
            S.S2.push_back(wt * (t.phi * G2).cwiseAbs());
        }
    }
    return S;
}

}  // namespace nxr::signal
