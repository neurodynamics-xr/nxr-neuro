#include "nxr/signal/graphtransforms.h"

#include <Eigen/Cholesky>
#include <Spectra/MatOp/SparseCholesky.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymGEigsSolver.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace nxr::signal::graphtransforms {
namespace {
constexpr double kPi = 3.14159265358979323846;
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

Eigen::MatrixXd chebyshev(const ChebyshevOperator& op, const Gain& g, const Eigen::MatrixXd& X) {
    const int K = op.p_->order;
    const Eigen::VectorXd c = chebyshevCoefficients(g, op.p_->lmax, K);
    const double a = op.p_->lmax / 2;
    // Clenshaw on Ltil = L/a − I, which maps [0, λmax] onto [−1, 1]
    auto Ltil = [&](const Eigen::MatrixXd& Y) -> Eigen::MatrixXd { return op.p_->Aop(Y) / a - Y; };
    Eigen::MatrixXd b2 = Eigen::MatrixXd::Zero(X.rows(), X.cols()), b1 = b2;
    for (int k = K; k >= 1; --k) {
        Eigen::MatrixXd bk = 2 * Ltil(b1) - b2 + c(k) * X;
        b2 = std::move(b1); b1 = std::move(bk);
    }
    return Ltil(b1) - b2 + c(0) * X;
}

Eigen::MatrixXcd chebyshev(const ChebyshevOperator& op, const Gain& g, const Eigen::MatrixXcd& X) {
    Eigen::MatrixXcd Y(X.rows(), X.cols());
    Y.real() = chebyshev(op, g, Eigen::MatrixXd(X.real()));
    Y.imag() = chebyshev(op, g, Eigen::MatrixXd(X.imag()));
    return Y;
}

// ── analysis and synthesis ───────────────────────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd gft(const EigenTransform& t, const Eigen::MatrixXd& F) {
    if (t.phi.rows() != F.rows()) throw std::invalid_argument("filterbank: field rows differ from the eigenbasis rows");
    return t.mass.rows() ? Eigen::MatrixXd(t.phi.transpose() * (t.mass * F)) : Eigen::MatrixXd(t.phi.transpose() * F);
}
Eigen::MatrixXcd gft(const EigenTransform& t, const Eigen::MatrixXcd& F) {
    if (t.phi.rows() != F.rows()) throw std::invalid_argument("filterbank: field rows differ from the eigenbasis rows");
    const Eigen::MatrixXcd phiT = t.phi.transpose().cast<std::complex<double>>();
    if (!t.mass.rows()) return phiT * F;
    Eigen::MatrixXcd BF(F.rows(), F.cols());
    BF.real() = t.mass * Eigen::MatrixXd(F.real());
    BF.imag() = t.mass * Eigen::MatrixXd(F.imag());
    return phiT * BF;
}
Eigen::MatrixXd igft(const EigenTransform& t, const Eigen::MatrixXd& C) {
    if (t.phi.cols() != C.rows()) throw std::invalid_argument("filterbank: coefficient rows differ from the number of modes");
    return t.phi * C;
}
Eigen::MatrixXcd igft(const EigenTransform& t, const Eigen::MatrixXcd& C) {
    if (t.phi.cols() != C.rows()) throw std::invalid_argument("filterbank: coefficient rows differ from the number of modes");
    return t.phi.cast<std::complex<double>>() * C;
}

double chebyshevError(const Gain& g, double lmax, int order, int samples) {
    const Eigen::VectorXd c = chebyshevCoefficients(g, lmax, order);
    double worst = 0;
    for (int i = 0; i <= samples; ++i) {
        const double l = lmax * i / samples;
        worst = std::max(worst, std::abs(g(l) - chebyshevEval(c, lmax, l)));
    }
    return worst;
}

}  // namespace nxr::signal::graphtransforms
