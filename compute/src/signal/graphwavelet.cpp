// nxr::signal gwt / igwt — the bodies of graphfilterbank::analysis, analysisMember and igwt (moved, not edited): a graph bank only designs.
#include "nxr/signal/graphwavelet.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace nxr::signal {
using namespace graphfilterbank;
using namespace graphtransforms;

namespace {

Eigen::MatrixXcd join(const Eigen::MatrixXd& re, const Eigen::MatrixXd& im) {
    Eigen::MatrixXcd z(re.rows(), re.cols()); z.real() = re; z.imag() = im; return z;
}

template <class Op> CoefficientsC intrinsicComplex(const Eigen::MatrixXcd& F, Op op) {   // op(real matrix) → Coefficients
    Coefficients re = op(Eigen::MatrixXd(F.real())), im = op(Eigen::MatrixXd(F.imag()));
    CoefficientsC out;
    for (std::size_t m = 0; m < re.size(); ++m) out.push_back(join(re[m], im[m]));
    return out;
}

void split(const CoefficientsC& W, Coefficients& re, Coefficients& im) {
    for (const auto& w : W) { re.emplace_back(w.real()); im.emplace_back(w.imag()); }
}

void checkMembers(const GraphFilterbank& bank, std::size_t nW) {
    if (nW != static_cast<std::size_t>(bank.size())) throw std::invalid_argument("filterbank: one coefficient block per member is required");
}

}  // namespace

Coefficients gwt(const GraphFilterbank& bank, const EigenTransform& t, const Eigen::MatrixXd& F) {
    if (t.phi.cols() != t.lambda.size()) throw std::invalid_argument("filterbank: phi and lambda disagree");
    const Eigen::MatrixXd C = gft(t, F);
    const Eigen::MatrixXd H = gains(bank, t.lambda);
    Coefficients W;
    for (int m = 0; m < bank.size(); ++m) W.emplace_back(t.phi * (H.col(m).asDiagonal() * C));
    return W;
}
CoefficientsC gwt(const GraphFilterbank& bank, const EigenTransform& t, const Eigen::MatrixXcd& F) {
    return intrinsicComplex(F, [&](const Eigen::MatrixXd& X) { return gwt(bank, t, X); });
}

Coefficients gwt(const GraphFilterbank& bank, const ChebyshevOperator& t, const Eigen::MatrixXd& F) {
    Coefficients W;
    for (const auto& m : bank.members) W.emplace_back(chebyshev(t, m.gain, F));
    return W;
}
CoefficientsC gwt(const GraphFilterbank& bank, const ChebyshevOperator& t, const Eigen::MatrixXcd& F) {
    return intrinsicComplex(F, [&](const Eigen::MatrixXd& X) { return gwt(bank, t, X); });
}

static const graphfilterbank::Member& memberOf(const GraphFilterbank& bank, int member) {
    if (member < 0 || member >= bank.size()) throw std::invalid_argument("filterbank: member " + std::to_string(member) + " is outside the bank (0 … " + std::to_string(bank.size() - 1) + ")");
    return bank.members[static_cast<std::size_t>(member)];
}
Eigen::MatrixXd gwt(const GraphFilterbank& bank, int member, const EigenTransform& t, const Eigen::MatrixXd& F) {
    const Member& m = memberOf(bank, member);
    if (t.phi.cols() != t.lambda.size()) throw std::invalid_argument("filterbank: phi and lambda disagree");
    Eigen::VectorXd g(t.lambda.size());
    for (Eigen::Index k = 0; k < g.size(); ++k) g(k) = m.gain(t.lambda(k));
    return t.phi * (g.asDiagonal() * gft(t, F));
}
Eigen::MatrixXd gwt(const GraphFilterbank& bank, int member, const ChebyshevOperator& t, const Eigen::MatrixXd& F) { return chebyshev(t, memberOf(bank, member).gain, F); }

Eigen::MatrixXd igwt(const GraphFilterbank& bank, const EigenTransform& t, const Coefficients& W, Synthesis mode) {
    checkMembers(bank, W.size());
    const Eigen::MatrixXd H = mode == Synthesis::Dual ? dualGains(bank, t.lambda) : gains(bank, t.lambda);
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(t.phi.cols(), W[0].cols());
    for (int m = 0; m < bank.size(); ++m) C += H.col(m).asDiagonal() * gft(t, W[static_cast<std::size_t>(m)]);
    return t.phi * C;
}
Eigen::MatrixXcd igwt(const GraphFilterbank& bank, const EigenTransform& t, const CoefficientsC& W, Synthesis mode) {
    Coefficients re, im; split(W, re, im);
    return join(igwt(bank, t, re, mode), igwt(bank, t, im, mode));
}

Eigen::MatrixXd igwt(const GraphFilterbank& bank, const ChebyshevOperator& t, const Coefficients& W, Synthesis mode) {
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
        F += chebyshev(t, w, W[static_cast<std::size_t>(m)]);
    }
    return F;
}
Eigen::MatrixXcd igwt(const GraphFilterbank& bank, const ChebyshevOperator& t, const CoefficientsC& W, Synthesis mode) {
    Coefficients re, im; split(W, re, im);
    return join(igwt(bank, t, re, mode), igwt(bank, t, im, mode));
}


}  // namespace nxr::signal
