// nxr::analytic: abs / angle / unwrap / instfreq on known series.
#include "nxr/signal/complex_signal.h"

#include <cmath>
#include <cstdio>

using namespace nxr;

static int failures = 0;
static void check(bool ok, const char* what, double value) {
    if (!ok) { std::printf("FAIL %s (%.3e)\n", what, value); ++failures; }
}
static constexpr double kPi = 3.14159265358979323846;
static double wrapd(double a) { return std::remainder(a, 2 * kPi); }

int main() {
    // abs, angle
    Eigen::MatrixXcd c(2, 1);
    c << std::complex<double>(3, 4), std::complex<double>(-1, 0);
    check(std::abs(signal::abs(c)(0, 0) - 5) < 1e-15, "abs", 0);
    check(std::abs(signal::angle(c)(1, 0) - kPi) < 1e-15, "angle of -1 is +pi", 0);

    // unwrap: a phase ramp of 0.3 cycles a sample, wrapped, comes back
    const int n = 200;
    Eigen::MatrixXd ramp(n, 2), wrapped(n, 2);
    for (int k = 0; k < n; ++k) {
        ramp(k, 0) = 0.1 + 2 * kPi * 0.3 * k;
        ramp(k, 1) = -0.4 - 2 * kPi * 0.45 * k;
        wrapped(k, 0) = wrapd(ramp(k, 0));
        wrapped(k, 1) = wrapd(ramp(k, 1));
    }
    const double uw = (signal::unwrap(wrapped) - ramp).cwiseAbs().maxCoeff();
    check(uw < 1e-9, "unwrap recovers a ramp", uw);

    // instfreq: a linear chirp f(t) = f0 + k t is exact at the interior (the centred step of a quadratic phase)
    {
        const double rate = 500, f0 = 10, kk = 4;
        Eigen::MatrixXcd z(n, 1);
        for (int i = 0; i < n; ++i) { const double t = i / rate; z(i, 0) = std::polar(2.0, 2 * kPi * (f0 * t + 0.5 * kk * t * t) + 0.3); }
        const Eigen::MatrixXd f = signal::instfreq(z, rate);
        double e = 0;
        for (int i = 1; i + 1 < n; ++i) e = std::max(e, std::abs(f(i, 0) - (f0 + kk * i / rate)));
        check(e < 1e-9, "instfreq of a chirp", e);
    }

    if (failures) return 1;
    std::printf("test_analytic: ok\n");
    return 0;
}
