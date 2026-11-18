// nxr::analytic — amplitude, phase, unwrapping, instantaneous frequency, and wt's carrier (see the header).
#include "nxr/signal/complex_signal.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace nxr::signal {
namespace {
constexpr double kPi = 3.14159265358979323846;
using i128 = __int128;

void checkMember(const wt::Plan& plan, int m) {
    if (m < 0 || m >= static_cast<int>(plan.p.size())) throw std::invalid_argument("analytic: no such member");
}
}  // namespace

Eigen::MatrixXd abs(const Eigen::MatrixXcd& c) { return c.cwiseAbs(); }

Eigen::MatrixXd angle(const Eigen::MatrixXcd& c) { return c.array().arg().matrix(); }

Eigen::MatrixXd unwrap(const Eigen::MatrixXd& phase, double tolerance) {
    Eigen::MatrixXd out = phase;
    for (Eigen::Index j = 0; j < out.cols(); ++j) {
        double add = 0;
        for (Eigen::Index k = 1; k < out.rows(); ++k) {
            const double d = phase(k, j) - phase(k - 1, j);
            if (std::abs(d) > tolerance) {
                double r = std::remainder(d, 2 * kPi);           // in [-pi, pi]
                if (r == -kPi && d > 0) r = kPi;                 // MATLAB keeps a jump of +pi as +pi
                add += r - d;
            }
            out(k, j) = phase(k, j) + add;
        }
    }
    return out;
}

Eigen::MatrixXd instfreq(const Eigen::MatrixXcd& z, double rate) {
    if (!(rate > 0)) throw std::invalid_argument("analytic: the rate must be positive");
    const Eigen::Index n = z.rows();
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(n, z.cols());
    if (n < 2) return f;
    const double s = rate / (2 * kPi);
    for (Eigen::Index j = 0; j < z.cols(); ++j) {
        Eigen::VectorXd d(n - 1);
        for (Eigen::Index k = 0; k + 1 < n; ++k) d(k) = std::arg(z(k + 1, j) * std::conj(z(k, j)));
        f(0, j) = s * d(0);
        f(n - 1, j) = s * d(n - 2);
        for (Eigen::Index k = 1; k + 1 < n; ++k) f(k, j) = s * 0.5 * (d(k - 1) + d(k));
    }
    return f;
}

double carrierFrequency(const wt::Plan& plan, int m) {
    checkMember(plan, m);
    const auto& b = plan.bank;
    return static_cast<double>(b.support[static_cast<std::size_t>(m)].first) * b.fs / static_cast<double>(b.nfft);
}

Eigen::VectorXd instants(const wt::Plan& plan, int m, const wt::Block& block) {
    checkMember(plan, m);
    const double p = static_cast<double>(plan.p[static_cast<std::size_t>(m)]), q = static_cast<double>(plan.q[static_cast<std::size_t>(m)]);
    Eigen::VectorXd t(block.coef.rows());
    for (Eigen::Index r = 0; r < t.size(); ++r) t(r) = static_cast<double>(block.first + r) * p / (q * plan.bank.fs);
    return t;
}

Eigen::MatrixXcd carrier(const wt::Plan& plan, int m, const wt::Block& block) {
    checkMember(plan, m);
    // phase in cycles: f_lo t_k = (i1 fs / nfft) (k p / (q fs)) = i1 k p / (q nfft): reduced exactly mod (q nfft) in integers
    const i128 i1 = plan.bank.support[static_cast<std::size_t>(m)].first;
    const i128 p = plan.p[static_cast<std::size_t>(m)], q = plan.q[static_cast<std::size_t>(m)];
    const i128 den = q * static_cast<i128>(plan.bank.nfft);
    Eigen::MatrixXcd out = block.coef;
    for (Eigen::Index r = 0; r < out.rows(); ++r) {
        i128 num = (i1 * p % den) * ((static_cast<i128>(block.first) + r) % den) % den;
        if (num < 0) num += den;
        const double cycles = static_cast<double>(num) / static_cast<double>(den);
        out.row(r) *= std::polar(1.0, 2 * kPi * cycles);
    }
    return out;
}

}  // namespace nxr::signal
