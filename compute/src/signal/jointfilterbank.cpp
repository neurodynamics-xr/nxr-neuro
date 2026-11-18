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
Plane plane(const JointFilterbank& b, const VectorXd& lambda) { return {fb::gains(b.graph, lambda), tb::gains(b.time), tb::bins(b.time)}; }

MatrixXcd jointfilters(const JointFilterbank& b, const Plane& F, const VectorXd& lambda, int m) {
    const Index ix = b.index.at(static_cast<std::size_t>(m));
    MatrixXcd W = (F.Gg.col(static_cast<Eigen::Index>(ix.g)) * F.Gt.col(static_cast<Eigen::Index>(ix.t)).transpose()).cast<cd>();
    if (!b.separable()) {
        const Kernel& K = b.kernels[ix.k];
        for (Eigen::Index j = 0; j < W.cols(); ++j)
            for (Eigen::Index i = 0; i < W.rows(); ++i) W(i, j) *= K(lambda(i), F.f(j));
    }
    return W;
}

MatrixXcd jointfilters(const JointFilterbank& b, int m, const VectorXd& lambda) { return jointfilters(b, plane(b, lambda), lambda, m); }

MatrixXd frameOperator(const JointFilterbank& b, const Plane& F, const VectorXd& lambda) {
    MatrixXd S = MatrixXd::Zero(lambda.size(), F.f.size());
    for (int m = 0; m < b.size(); ++m) S += jointfilters(b, F, lambda, m).cwiseAbs2();
    return S;
}

MatrixXd frameOperator(const JointFilterbank& b, const VectorXd& lambda) { return frameOperator(b, plane(b, lambda), lambda); }

static bool isEdgeColumn(const JointFilterbank& b, Eigen::Index j) {
    return b.time.sides == tb::Sides::Analytic && (j == 0 || (b.time.nfft % 2 == 0 && static_cast<std::size_t>(j) == b.time.nfft / 2));
}

static fb::Bounds boundsOfPlane(const JointFilterbank& b, const VectorXd& lambda) {
    const MatrixXd S = frameOperator(b, lambda);
    double A = std::numeric_limits<double>::infinity(), B = 0, unc = 0, count = 0;
    for (Eigen::Index j = 0; j < S.cols(); ++j) {
        if (isEdgeColumn(b, j)) continue;
        for (Eigen::Index i = 0; i < S.rows(); ++i) { A = std::min(A, S(i, j)); B = std::max(B, S(i, j)); if (S(i, j) <= 1e-12) ++unc; ++count; }
    }
    fb::Bounds r; r.A = A; r.B = B; r.tightness = B / std::max(A, 2.220446049250313e-16); r.uncovered = count > 0 ? unc / count : 0.0;
    return r;
}

static fb::Bounds boundsOfProduct(const JointFilterbank& b, const VectorXd& lambda) {
    if (!b.separable()) throw std::invalid_argument("framebounds: the bank has joint kernels; its frame operator is not a product");
    const fb::Bounds g = fb::framebounds(b.graph, lambda), t = tb::framebounds(b.time);
    fb::Bounds r; r.A = g.A * t.A; r.B = g.B * t.B; r.tightness = r.B / std::max(r.A, 2.220446049250313e-16);
    r.uncovered = 1.0 - (1.0 - g.uncovered) * (1.0 - t.uncovered);
    return r;
}

fb::Bounds framebounds(const JointFilterbank& b, const VectorXd& lambda, BoundsBy by) { return by == BoundsBy::Plane ? boundsOfPlane(b, lambda) : boundsOfProduct(b, lambda); }

bool isframetight(const JointFilterbank& b, const VectorXd& lambda, double tol) { const fb::Bounds r = framebounds(b, lambda); return std::abs(r.B / std::max(r.A, 2.220446049250313e-16) - 1) <= tol; }

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
