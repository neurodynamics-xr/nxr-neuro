// nxr::signal — amplitude, phase, unwrapping, instantaneous frequency (see the header).
#include "nxr/signal/complex_signal.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace nxr::signal {
namespace {
constexpr double kPi = 3.14159265358979323846;
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

}  // namespace nxr::signal
