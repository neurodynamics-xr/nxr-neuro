// nxr::analytic: abs / angle / unwrap / instfreq on known series, and wt's carrier: a tone's phase and frequency recovered from wt coefficients.
#include "nxr/signal/complex_signal.h"

#include <cmath>
#include <cstdio>

using namespace nxr;
using namespace nxr::signal::timefilterbank;

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

    // wt's carrier: a tone A cos(2 pi f t + phi) at a Morlet member's centre -> |a| = A, angle(a) = 2 pi f t + phi, instfreq = f
    {
        const double fs = 250, f = 8, A = 1.7, phi = 0.6;
        Design d(Family::Morlet, 2.0, Sides::Analytic); d.centres = {4, 8, 16}; d.cycles = 6;
        const TimeFilterbank bk = timeBank(d, 1024, fs);
        const int N = 20 * static_cast<int>(fs);
        Eigen::MatrixXd x(1, N);
        for (int i = 0; i < N; ++i) x(0, i) = A * std::cos(2 * kPi * f * i / fs + phi);
        const signal::WtLayout pl = signal::wtLayout(bk);
        std::vector<signal::Block> all = signal::wt(pl, x);
        int m = -1;
        for (int j = 0; j < static_cast<int>(bk.members.size()); ++j) if (std::abs(bk.members[static_cast<std::size_t>(j)].fc - f) < 1e-9) m = j;
        check(m >= 0, "a member at 8 Hz", 0);
        // join the member's blocks
        std::vector<std::complex<double>> a; std::vector<double> t;
        for (const auto* part : {&all}) {
            const signal::Block& b = (*part)[static_cast<std::size_t>(m)];
            if (b.coef.rows() == 0) continue;
            const Eigen::MatrixXcd cb = signal::carrier(pl, m, b);
            const Eigen::VectorXd tb = signal::instants(pl, m, b);
            for (Eigen::Index r = 0; r < cb.rows(); ++r) { a.push_back(cb(r, 0)); t.push_back(tb(r)); }
        }
        double eA = 0, eP = 0, eF = 0; int used = 0;
        Eigen::MatrixXcd z(static_cast<Eigen::Index>(a.size()), 1);
        for (std::size_t i = 0; i < a.size(); ++i) z(static_cast<Eigen::Index>(i), 0) = a[i];
        const Eigen::MatrixXd fi = signal::instfreq(z, pl.rate(m));
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (t[i] < 3 || t[i] > 17) continue;                     // away from the record's ends
            ++used;
            eA = std::max(eA, std::abs(std::abs(a[i]) - A) / A);
            eP = std::max(eP, std::abs(wrapd(std::arg(a[i]) - (2 * kPi * f * t[i] + phi))));
            eF = std::max(eF, std::abs(fi(static_cast<Eigen::Index>(i), 0) - f));
        }
        check(used > 100, "carrier: samples checked", used);
        std::printf("carrier: %d samples, |a| %.1e, phase %.1e, instfreq %.1e\n", used, eA, eP, eF);
        check(eA < 1e-6, "carrier: |a| = A", eA);
        check(eP < 1e-6, "carrier: angle(a) = 2 pi f t + phi", eP);
        check(eF < 1e-6, "carrier: instfreq = f", eF);
        check(signal::carrierFrequency(pl, m) > 0, "a band member's shift is above DC", 0);

        // the carrier factor is exact far into a recording: instants K and K + q nfft get the same factor
        signal::Block b0{0, Eigen::MatrixXcd::Ones(1, 1)}, b1{static_cast<std::int64_t>(pl.q[static_cast<std::size_t>(m)]) * static_cast<std::int64_t>(bk.nfft) * 1000003LL, Eigen::MatrixXcd::Ones(1, 1)};
        const double ex = std::abs(signal::carrier(pl, m, b0)(0, 0) - signal::carrier(pl, m, b1)(0, 0));
        check(ex < 1e-12, "carrier phase exact at a large instant", ex);
    }

    if (failures) return 1;
    std::printf("test_analytic: ok\n");
    return 0;
}
