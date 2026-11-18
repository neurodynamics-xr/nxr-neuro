// nxr::signal jwt / ijwt — the bodies of jointfilterbank::analysis, analysisSpectrum, synthesis, synthesisComplex, synthesisSpectrum (moved, not edited): a joint bank only designs.
#include "nxr/signal/jointwavelet.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::signal {
using namespace jointfilterbank;
using Eigen::MatrixXcd; using Eigen::MatrixXd; using Eigen::VectorXd;
namespace fb = nxr::signal::graphfilterbank;
namespace tb = nxr::signal::timefilterbank;
using cd = std::complex<double>;

static bool isEdgeColumn(const JointFilterbank& b, Eigen::Index j) {
    return b.time.sides == tb::Sides::Analytic && (j == 0 || (b.time.nfft % 2 == 0 && static_cast<std::size_t>(j) == b.time.nfft / 2));
}

// ── spectrum domain ────────────────────────────────────────────────────────────────────────────────────────────────────────

std::vector<MatrixXcd> jwtSpectrum(const JointFilterbank& b, const VectorXd& lambda, const MatrixXcd& C) {
    const Plane F = plane(b, lambda);
    if (C.rows() != lambda.size() || C.cols() != F.f.size()) throw std::invalid_argument("jwtSpectrum: C must be [K x F] on the bank's grid");
    std::vector<MatrixXcd> Y;
    for (int m = 0; m < b.size(); ++m) Y.push_back(C.array() * jointfilters(b, F, lambda, m).array());
    return Y;
}

MatrixXcd ijwtSpectrum(const JointFilterbank& b, const VectorXd& lambda, const std::vector<MatrixXcd>& Y, Synthesis mode) {
    if (static_cast<int>(Y.size()) != b.size()) throw std::invalid_argument("ijwtSpectrum: one block per member");
    const Plane F = plane(b, lambda);
    MatrixXd S; double tol = 0;
    if (mode == Synthesis::Dual) { S = frameOperator(b, F, lambda); tol = 1e-12 * S.maxCoeff(); }
    MatrixXcd C = MatrixXcd::Zero(lambda.size(), F.f.size());
    for (int m = 0; m < b.size(); ++m) {
        MatrixXcd W = jointfilters(b, F, lambda, m);
        if (mode == Synthesis::Dual)
            for (Eigen::Index j = 0; j < W.cols(); ++j) for (Eigen::Index i = 0; i < W.rows(); ++i) W(i, j) = S(i, j) > tol ? W(i, j) / S(i, j) : cd(0);
        C += W.conjugate().cwiseProduct(Y[static_cast<std::size_t>(m)]);
    }
    return C;
}

// ── modes × time ───────────────────────────────────────────────────────────────────────────────────────────────────────────

static Coefficients analyseSpectrum(const JointFilterbank& b, const VectorXd& lambda, const MatrixXcd& spec /* [F x K] */) {
    const Plane F = plane(b, lambda);
    const Eigen::VectorXd ones = VectorXd::Ones(F.f.size());
    Coefficients W;
    for (int m = 0; m < b.size(); ++m) {
        const MatrixXcd Ym = jointfilters(b, F, lambda, m).transpose().array() * spec.array();      // [F x K]
        W.push_back(tb::filterSpectrum(b.time, ones, Ym).transpose());                          // [K x nfft]
    }
    return W;
}

Coefficients jwt(const JointFilterbank& b, const VectorXd& lambda, const MatrixXd& C) {
    if (C.rows() != lambda.size()) throw std::invalid_argument("jwt: C must be [K x T]");
    return analyseSpectrum(b, lambda, tb::spectrum(b.time, MatrixXd(C.transpose())));
}
Coefficients jwt(const JointFilterbank& b, const VectorXd& lambda, const MatrixXcd& C) {
    if (C.rows() != lambda.size()) throw std::invalid_argument("jwt: C must be [K x T]");
    return analyseSpectrum(b, lambda, tb::spectrum(b.time, MatrixXcd(C.transpose())));       // an analytic time bank refuses
}

// Σ_i w_i ⊙ spec(W_i); the dual w = conj(W_i)/S, tight w = conj(W_i) / S_nominal (both factors tight)
static MatrixXcd combineModes(const JointFilterbank& b, const VectorXd& lambda, const Coefficients& W, Synthesis mode) {
    if (static_cast<int>(W.size()) != b.size()) throw std::invalid_argument("ijwt: one block per member");
    const Plane F = plane(b, lambda);
    MatrixXd S;
    double tol = 0, nominal = 1;
    if (mode == Synthesis::Dual) { S = frameOperator(b, F, lambda); tol = 1e-12 * S.maxCoeff(); }
    else {
        if (!b.separable() || std::isnan(b.graph.total) || std::isnan(b.time.total))
            throw std::invalid_argument("jointbank: tight synthesis needs a separable bank of two tight banks; use the dual");
        nominal = b.graph.total * b.time.total;
    }
    MatrixXcd Z = MatrixXcd::Zero(F.f.size(), lambda.size());
    for (int m = 0; m < b.size(); ++m) {
        MatrixXcd w = jointfilters(b, F, lambda, m).conjugate();                                      // [K x F]
        if (mode == Synthesis::Dual) { for (Eigen::Index j = 0; j < w.cols(); ++j) for (Eigen::Index i = 0; i < w.rows(); ++i) w(i, j) = S(i, j) > tol ? w(i, j) / S(i, j) : cd(0); }
        else {   // tight: divide by the nominal S: G·G_t for the interior, half of it on an analytic DC / Nyquist bin (Single)
            for (Eigen::Index j = 0; j < w.cols(); ++j) {
                const double edge = (isEdgeColumn(b, j) && b.time.edgeBins == tb::EdgeBins::Single) ? 0.5 : 1.0;
                w.col(j) /= nominal * edge;
            }
        }
        Z += (w.transpose().array() * tb::coefficientSpectrum(b.time, W[static_cast<std::size_t>(m)].transpose()).array()).matrix();
    }
    return Z;   // [F x K]
}

MatrixXd ijwt(const JointFilterbank& b, const VectorXd& lambda, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseReal(b.time, combineModes(b, lambda, W, mode), n).transpose();
}
MatrixXcd ijwtComplex(const JointFilterbank& b, const VectorXd& lambda, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseComplex(b.time, combineModes(b, lambda, W, mode), n).transpose();
}

// ── vertices ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

static MatrixXcd toModes(const fb::EigenTransform& t, const MatrixXcd& F) {
    const MatrixXd PhiT = t.phi.transpose();
    MatrixXcd BF = t.mass.size() > 0 ? MatrixXcd(t.mass.cast<cd>() * F) : F;
    return PhiT.cast<cd>() * BF;
}

Coefficients jwt(const JointFilterbank& b, const fb::EigenTransform& t, const MatrixXd& F) {
    const MatrixXd PhiT = t.phi.transpose();
    const MatrixXd Cm = t.mass.size() > 0 ? MatrixXd(PhiT * (t.mass * F)) : MatrixXd(PhiT * F);
    Coefficients W = jwt(b, t.lambda, Cm);
    for (auto& w : W) w = t.phi.cast<cd>() * w;
    return W;
}
Coefficients jwt(const JointFilterbank& b, const fb::EigenTransform& t, const MatrixXcd& F) {
    Coefficients W = jwt(b, t.lambda, toModes(t, F));
    for (auto& w : W) w = t.phi.cast<cd>() * w;
    return W;
}

static Coefficients toCoefficientModes(const fb::EigenTransform& t, const Coefficients& W) {
    Coefficients M;
    for (const auto& w : W) M.push_back(toModes(t, w));
    return M;
}
MatrixXd ijwt(const JointFilterbank& b, const fb::EigenTransform& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return t.phi * ijwt(b, t.lambda, toCoefficientModes(t, W), n, mode);
}
MatrixXcd ijwtComplex(const JointFilterbank& b, const fb::EigenTransform& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return t.phi.cast<cd>() * ijwtComplex(b, t.lambda, toCoefficientModes(t, W), n, mode);
}

// polynomial route: separable only
static void requireSeparable(const JointFilterbank& b) {
    if (!b.separable()) throw std::invalid_argument("jointbank: the polynomial (Chebyshev) route needs a SEPARABLE bank; a joint kernel couples lambda and f, which a series in the operator cannot see: use the EigenTransform");
}

static Coefficients chebAnalysis(const JointFilterbank& b, const fb::ChebyshevOperator& t, const MatrixXcd& spec /* [F x n] */) {
    requireSeparable(b);
    const MatrixXd Gt = tb::gains(b.time);
    const VectorXd ones = VectorXd::Ones(Gt.rows());
    Coefficients W(static_cast<std::size_t>(b.size()));
    for (std::size_t it = 0; it < static_cast<std::size_t>(b.time.size()); ++it) {
        const MatrixXcd Ft = tb::filterSpectrum(b.time, Gt.col(static_cast<Eigen::Index>(it)), spec).transpose();      // [n x nfft]
        for (std::size_t ig = 0; ig < static_cast<std::size_t>(b.graph.size()); ++ig)
            W[ig + static_cast<std::size_t>(b.graph.size()) * it] = chebyshev(t, b.graph.members[ig].gain, Ft);
    }
    (void)ones;
    return W;
}

Coefficients jwt(const JointFilterbank& b, const fb::ChebyshevOperator& t, const MatrixXd& F) {
    return chebAnalysis(b, t, tb::spectrum(b.time, MatrixXd(F.transpose())));
}
Coefficients jwt(const JointFilterbank& b, const fb::ChebyshevOperator& t, const MatrixXcd& F) {
    return chebAnalysis(b, t, tb::spectrum(b.time, MatrixXcd(F.transpose())));
}

static MatrixXcd chebCombine(const JointFilterbank& b, const fb::ChebyshevOperator& t, const Coefficients& W, Synthesis mode) {
    requireSeparable(b);
    if (static_cast<int>(W.size()) != b.size()) throw std::invalid_argument("ijwt: one block per member");
    // graph weights g/S_g (dual) or g/total (tight), as functions of lambda; time weights from the time bank
    const fb::GraphFilterbank& g = b.graph;
    const double Smax = std::max(fb::framebounds(g).B, 1e-300);
    std::vector<fb::Gain> wg;
    for (std::size_t ig = 0; ig < g.members.size(); ++ig) {
        if (mode == Synthesis::Tight) {
            if (std::isnan(g.total) || std::isnan(b.time.total)) throw std::invalid_argument("jointbank: tight synthesis needs two tight banks; use the dual");
            const fb::Gain gi = g.members[ig].gain; const double tot = g.total;
            wg.push_back([gi, tot](double l) { return gi(l) / tot; });
        } else {
            const fb::GraphFilterbank* gp = &g; const std::size_t i = ig;
            wg.push_back([gp, i, Smax](double l) { double S = 0; for (const auto& m : gp->members) { const double x = m.gain(l); S += x * x; } return S > 1e-12 * Smax ? gp->members[i].gain(l) / S : 0.0; });
        }
    }
    MatrixXd wt;
    if (mode == Synthesis::Dual) wt = tb::dualGains(b.time);
    else {
        wt = tb::gains(b.time);
        const double G = b.time.gain * b.time.gain;      // Σ g² = gain² for a tight time bank
        for (Eigen::Index k = 0; k < wt.rows(); ++k) {
            const bool edge = b.time.sides == tb::Sides::Analytic && (k == 0 || (b.time.nfft % 2 == 0 && static_cast<std::size_t>(k) == b.time.nfft / 2));
            wt.row(k) /= (edge && b.time.edgeBins == tb::EdgeBins::Single) ? G / 2 : G;
        }
    }
    MatrixXcd Z;
    for (std::size_t it = 0; it < static_cast<std::size_t>(b.time.size()); ++it)
        for (std::size_t ig = 0; ig < static_cast<std::size_t>(g.size()); ++ig) {
            const MatrixXcd U = chebyshev(t, wg[ig], W[ig + static_cast<std::size_t>(g.size()) * it]);                       // [n x nfft]
            const MatrixXcd Us = tb::coefficientSpectrum(b.time, U.transpose());                                        // [F x n]
            if (Z.size() == 0) Z = MatrixXcd::Zero(Us.rows(), Us.cols());
            for (Eigen::Index j = 0; j < Us.cols(); ++j) for (Eigen::Index k = 0; k < Us.rows(); ++k) Z(k, j) += wt(k, static_cast<Eigen::Index>(it)) * Us(k, j);
        }
    return Z;
}

MatrixXd ijwt(const JointFilterbank& b, const fb::ChebyshevOperator& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseReal(b.time, chebCombine(b, t, W, mode), n).transpose();
}
MatrixXcd ijwtComplex(const JointFilterbank& b, const fb::ChebyshevOperator& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseComplex(b.time, chebCombine(b, t, W, mode), n).transpose();
}


}  // namespace nxr::signal
