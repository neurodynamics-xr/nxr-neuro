// nxr::signal::jointfilterbank — see include/nxr/jointbank.h and docs/specs/2026-11-10-filterbanks.md.
#include "nxr/signal/jointfilterbank.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace nxr::signal::jointfilterbank {

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace fb = nxr::signal::graphfilterbank;
namespace tb = nxr::signal::timefilterbank;

JointFilterbank make(const fb::GraphFilterbank& graph, const tb::TimeFilterbank& time, std::vector<Kernel> kernels) {
    JointFilterbank b; b.graph = graph; b.time = time; b.kernels = std::move(kernels);
    const std::size_t Ng = static_cast<std::size_t>(graph.size()), Nt = static_cast<std::size_t>(time.size()), Nk = std::max<std::size_t>(1, b.kernels.size());
    for (std::size_t k = 0; k < Nk; ++k) for (std::size_t t = 0; t < Nt; ++t) for (std::size_t g = 0; g < Ng; ++g) b.index.push_back({g, t, k});
    return b;
}

JointFilterbank make(const fb::GraphFilterbank&, const tb::TimeFilterbank&, const std::vector<LagKernel>&) {
    throw std::invalid_argument("jointbank::make: a (lambda, t) kernel acts on a time LAG, as a propagator (diffusion, wave, ...), and cannot multiply a joint "
                                "(lambda, f) spectrum, which the joint bank is evaluated on; transform it along the time axis first and pass it as a (lambda, f) kernel");
}

// the factors on the plane, computed once: graph gains [K × Ng], time gains [F × Nt]
struct Factors { MatrixXd Gg, Gt; VectorXd f; };
static Factors factors(const JointFilterbank& b, const VectorXd& lambda) { return {fb::gains(b.graph, lambda), tb::gains(b.time), tb::bins(b.time)}; }

static MatrixXcd member(const JointFilterbank& b, const Factors& F, const VectorXd& lambda, int m) {
    const Index ix = b.index.at(static_cast<std::size_t>(m));
    MatrixXcd W = (F.Gg.col(static_cast<Eigen::Index>(ix.g)) * F.Gt.col(static_cast<Eigen::Index>(ix.t)).transpose()).cast<cd>();
    if (!b.separable()) {
        const Kernel& K = b.kernels[ix.k];
        for (Eigen::Index j = 0; j < W.cols(); ++j)
            for (Eigen::Index i = 0; i < W.rows(); ++i) W(i, j) *= K(lambda(i), F.f(j));
    }
    return W;
}

MatrixXcd jointfilters(const JointFilterbank& b, int m, const VectorXd& lambda) { return member(b, factors(b, lambda), lambda, m); }

static MatrixXd frameOp(const JointFilterbank& b, const Factors& F, const VectorXd& lambda) {
    MatrixXd S = MatrixXd::Zero(lambda.size(), F.f.size());
    for (int m = 0; m < b.size(); ++m) S += member(b, F, lambda, m).cwiseAbs2();
    return S;
}

MatrixXd frameOperator(const JointFilterbank& b, const VectorXd& lambda) { return frameOp(b, factors(b, lambda), lambda); }

static bool isEdgeColumn(const JointFilterbank& b, Eigen::Index j) {
    return b.time.sides == tb::Sides::Analytic && (j == 0 || (b.time.nfft % 2 == 0 && static_cast<std::size_t>(j) == b.time.nfft / 2));
}

fb::Bounds framebounds(const JointFilterbank& b, const VectorXd& lambda) {
    const MatrixXd S = frameOperator(b, lambda);
    double A = std::numeric_limits<double>::infinity(), B = 0, unc = 0, count = 0;
    for (Eigen::Index j = 0; j < S.cols(); ++j) {
        if (isEdgeColumn(b, j)) continue;
        for (Eigen::Index i = 0; i < S.rows(); ++i) { A = std::min(A, S(i, j)); B = std::max(B, S(i, j)); if (S(i, j) <= 1e-12) ++unc; ++count; }
    }
    fb::Bounds r; r.A = A; r.B = B; r.tightness = B / std::max(A, 2.220446049250313e-16); r.uncovered = count > 0 ? unc / count : 0.0;
    return r;
}

fb::Bounds productBounds(const JointFilterbank& b, const VectorXd& lambda) {
    if (!b.separable()) throw std::invalid_argument("jointbank::productBounds: the bank has joint kernels; its frame operator is not a product");
    const fb::Bounds g = fb::framebounds(b.graph, lambda), t = tb::framebounds(b.time);
    fb::Bounds r; r.A = g.A * t.A; r.B = g.B * t.B; r.tightness = r.B / std::max(r.A, 2.220446049250313e-16);
    r.uncovered = 1.0 - (1.0 - g.uncovered) * (1.0 - t.uncovered);
    return r;
}

bool isframetight(const JointFilterbank& b, const VectorXd& lambda, double tol) { const fb::Bounds r = framebounds(b, lambda); return std::abs(r.B / std::max(r.A, 2.220446049250313e-16) - 1) <= tol; }

// ── spectrum domain ────────────────────────────────────────────────────────────────────────────────────────────────────────

std::vector<MatrixXcd> analysisSpectrum(const JointFilterbank& b, const VectorXd& lambda, const MatrixXcd& C) {
    const Factors F = factors(b, lambda);
    if (C.rows() != lambda.size() || C.cols() != F.f.size()) throw std::invalid_argument("jointbank::analysisSpectrum: C must be [K x F] on the bank's grid");
    std::vector<MatrixXcd> Y;
    for (int m = 0; m < b.size(); ++m) Y.push_back(C.array() * member(b, F, lambda, m).array());
    return Y;
}

MatrixXcd synthesisSpectrum(const JointFilterbank& b, const VectorXd& lambda, const std::vector<MatrixXcd>& Y, Synthesis mode) {
    if (static_cast<int>(Y.size()) != b.size()) throw std::invalid_argument("jointbank::synthesisSpectrum: one block per member");
    const Factors F = factors(b, lambda);
    MatrixXd S; double tol = 0;
    if (mode == Synthesis::Dual) { S = frameOp(b, F, lambda); tol = 1e-12 * S.maxCoeff(); }
    MatrixXcd C = MatrixXcd::Zero(lambda.size(), F.f.size());
    for (int m = 0; m < b.size(); ++m) {
        MatrixXcd W = member(b, F, lambda, m);
        if (mode == Synthesis::Dual)
            for (Eigen::Index j = 0; j < W.cols(); ++j) for (Eigen::Index i = 0; i < W.rows(); ++i) W(i, j) = S(i, j) > tol ? W(i, j) / S(i, j) : cd(0);
        C += W.conjugate().cwiseProduct(Y[static_cast<std::size_t>(m)]);
    }
    return C;
}

// ── modes × time ───────────────────────────────────────────────────────────────────────────────────────────────────────────

static Coefficients analyseSpectrum(const JointFilterbank& b, const VectorXd& lambda, const MatrixXcd& spec /* [F x K] */) {
    const Factors F = factors(b, lambda);
    const Eigen::VectorXd ones = VectorXd::Ones(F.f.size());
    Coefficients W;
    for (int m = 0; m < b.size(); ++m) {
        const MatrixXcd Ym = member(b, F, lambda, m).transpose().array() * spec.array();      // [F x K]
        W.push_back(tb::filterSpectrum(b.time, ones, Ym).transpose());                          // [K x nfft]
    }
    return W;
}

Coefficients analysis(const JointFilterbank& b, const VectorXd& lambda, const MatrixXd& C) {
    if (C.rows() != lambda.size()) throw std::invalid_argument("jointbank::analysis: C must be [K x T]");
    return analyseSpectrum(b, lambda, tb::spectrum(b.time, MatrixXd(C.transpose())));
}
Coefficients analysis(const JointFilterbank& b, const VectorXd& lambda, const MatrixXcd& C) {
    if (C.rows() != lambda.size()) throw std::invalid_argument("jointbank::analysis: C must be [K x T]");
    return analyseSpectrum(b, lambda, tb::spectrum(b.time, MatrixXcd(C.transpose())));       // an analytic time bank refuses
}

// Σ_i w_i ⊙ spec(W_i); the dual w = conj(W_i)/S, tight w = conj(W_i) / S_nominal (both factors tight)
static MatrixXcd combineModes(const JointFilterbank& b, const VectorXd& lambda, const Coefficients& W, Synthesis mode) {
    if (static_cast<int>(W.size()) != b.size()) throw std::invalid_argument("jointbank::synthesis: one block per member");
    const Factors F = factors(b, lambda);
    MatrixXd S;
    double tol = 0, nominal = 1;
    if (mode == Synthesis::Dual) { S = frameOp(b, F, lambda); tol = 1e-12 * S.maxCoeff(); }
    else {
        if (!b.separable() || std::isnan(b.graph.total) || std::isnan(b.time.total))
            throw std::invalid_argument("jointbank: tight synthesis needs a separable bank of two tight banks; use the dual");
        nominal = b.graph.total * b.time.total;
    }
    MatrixXcd Z = MatrixXcd::Zero(F.f.size(), lambda.size());
    for (int m = 0; m < b.size(); ++m) {
        MatrixXcd w = member(b, F, lambda, m).conjugate();                                      // [K x F]
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

MatrixXd synthesis(const JointFilterbank& b, const VectorXd& lambda, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseReal(b.time, combineModes(b, lambda, W, mode), n).transpose();
}
MatrixXcd synthesisComplex(const JointFilterbank& b, const VectorXd& lambda, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseComplex(b.time, combineModes(b, lambda, W, mode), n).transpose();
}

// ── vertices ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

static MatrixXcd toModes(const fb::EigenTransform& t, const MatrixXcd& F) {
    const MatrixXd PhiT = t.phi.transpose();
    MatrixXcd BF = t.mass.size() > 0 ? MatrixXcd(t.mass.cast<cd>() * F) : F;
    return PhiT.cast<cd>() * BF;
}

Coefficients analysis(const JointFilterbank& b, const fb::EigenTransform& t, const MatrixXd& F) {
    const MatrixXd PhiT = t.phi.transpose();
    const MatrixXd Cm = t.mass.size() > 0 ? MatrixXd(PhiT * (t.mass * F)) : MatrixXd(PhiT * F);
    Coefficients W = analysis(b, t.lambda, Cm);
    for (auto& w : W) w = t.phi.cast<cd>() * w;
    return W;
}
Coefficients analysis(const JointFilterbank& b, const fb::EigenTransform& t, const MatrixXcd& F) {
    Coefficients W = analysis(b, t.lambda, toModes(t, F));
    for (auto& w : W) w = t.phi.cast<cd>() * w;
    return W;
}

static Coefficients toCoefficientModes(const fb::EigenTransform& t, const Coefficients& W) {
    Coefficients M;
    for (const auto& w : W) M.push_back(toModes(t, w));
    return M;
}
MatrixXd synthesis(const JointFilterbank& b, const fb::EigenTransform& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return t.phi * synthesis(b, t.lambda, toCoefficientModes(t, W), n, mode);
}
MatrixXcd synthesisComplex(const JointFilterbank& b, const fb::EigenTransform& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return t.phi.cast<cd>() * synthesisComplex(b, t.lambda, toCoefficientModes(t, W), n, mode);
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
            W[ig + static_cast<std::size_t>(b.graph.size()) * it] = t.apply(b.graph.members[ig].gain, Ft);
    }
    (void)ones;
    return W;
}

Coefficients analysis(const JointFilterbank& b, const fb::ChebyshevOperator& t, const MatrixXd& F) {
    return chebAnalysis(b, t, tb::spectrum(b.time, MatrixXd(F.transpose())));
}
Coefficients analysis(const JointFilterbank& b, const fb::ChebyshevOperator& t, const MatrixXcd& F) {
    return chebAnalysis(b, t, tb::spectrum(b.time, MatrixXcd(F.transpose())));
}

static MatrixXcd chebCombine(const JointFilterbank& b, const fb::ChebyshevOperator& t, const Coefficients& W, Synthesis mode) {
    requireSeparable(b);
    if (static_cast<int>(W.size()) != b.size()) throw std::invalid_argument("jointbank::synthesis: one block per member");
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
            const MatrixXcd U = t.apply(wg[ig], W[ig + static_cast<std::size_t>(g.size()) * it]);                       // [n x nfft]
            const MatrixXcd Us = tb::coefficientSpectrum(b.time, U.transpose());                                        // [F x n]
            if (Z.size() == 0) Z = MatrixXcd::Zero(Us.rows(), Us.cols());
            for (Eigen::Index j = 0; j < Us.cols(); ++j) for (Eigen::Index k = 0; k < Us.rows(); ++k) Z(k, j) += wt(k, static_cast<Eigen::Index>(it)) * Us(k, j);
        }
    return Z;
}

MatrixXd synthesis(const JointFilterbank& b, const fb::ChebyshevOperator& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseReal(b.time, chebCombine(b, t, W, mode), n).transpose();
}
MatrixXcd synthesisComplex(const JointFilterbank& b, const fb::ChebyshevOperator& t, const Coefficients& W, std::size_t n, Synthesis mode) {
    return tb::inverseComplex(b.time, chebCombine(b, t, W, mode), n).transpose();
}

VectorXd scalogram(const Coefficients& W) {
    VectorXd e(static_cast<Eigen::Index>(W.size()));
    for (std::size_t m = 0; m < W.size(); ++m) e(static_cast<Eigen::Index>(m)) = W[m].cwiseAbs2().sum();
    return e;
}

Axes axes(const JointFilterbank& b, const VectorXd& lambda, bool padded) {
    Axes a; a.lambda = lambda; a.f = tb::bins(b.time); a.fs = b.time.fs; a.nfft = b.time.nfft; a.half = b.time.sides;
    a.boundary = padded ? Boundary::Path : Boundary::Ring;
    return a;
}

std::vector<std::string> iscompatible(const Axes& a, const Axes& b, double tol) {
    std::vector<std::string> why;
    if (a.lambda.size() != b.lambda.size()) why.push_back("lambda: " + std::to_string(a.lambda.size()) + " vs " + std::to_string(b.lambda.size()) + " modes");
    else if (a.lambda.size() > 0 && (a.lambda - b.lambda).cwiseAbs().maxCoeff() > tol * std::max(a.lambda.maxCoeff(), 2.220446049250313e-16))
        why.push_back("lambda: same count, different values (different anatomy or basis)");
    if (a.f.size() != b.f.size()) why.push_back("frequency: " + std::to_string(a.f.size()) + " vs " + std::to_string(b.f.size()) + " bins");
    else if (a.f.size() > 0 && (a.f - b.f).cwiseAbs().maxCoeff() > tol * std::max(a.f.cwiseAbs().maxCoeff(), 2.220446049250313e-16)) why.push_back("frequency: same count, different values");
    if (a.half != b.half) why.push_back("half: one is analytic (one-sided, complex), the other two-sided (signed)");
    if (a.boundary != b.boundary) why.push_back("boundary: ring (periodic) vs path (padded): different implied time extension");
    return why;
}

Kernel speedkernel(double c, double d) {
    if (!(d > 0)) throw std::invalid_argument("jointbank::speedkernel: the width d must be positive");
    return [c, d](double lambda, double f) { const double w = 2 * 3.14159265358979323846 * f - c * std::sqrt(lambda); return cd(std::exp(-w * w / (2 * d * d))); };
}

Kernel ridgeKernel(double v, double width, int sense) {
    if (!(width > 0) || (sense != 1 && sense != -1)) throw std::invalid_argument("jointbank::ridgeKernel: width > 0 and sense = +1 or -1");
    return [v, width, sense](double lambda, double f) {
        const double z = (f - sense * v * std::sqrt(std::max(lambda, 0.0)) / (2 * 3.14159265358979323846)) / width;
        return cd(std::exp(-0.5 * z * z));
    };
}

}  // namespace nxr::signal::jointfilterbank
