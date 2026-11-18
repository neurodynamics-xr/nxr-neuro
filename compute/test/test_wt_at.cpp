// nxr::signal wt(bank, member, window, instants): the member's analytic signal read at given instants (a frame Selection's, not whole samples), exactly. Synthetic data only.
//   a tone on a bin of the window's grid: |a(t)| = (G/2) A and the phase 2 pi f t + phi at fractional t, to rounding;
//   whole-sample instants equal the full-rate wt of the same window (the bank designed on the window's grid);
//   (against the old taps + sinc path: test_frames_wt.)
#include "nxr/signal/wavelet.h"
#include "mesh_helpers.h"
#include <random>
using namespace nxr::signal; namespace tb = nxr::signal::timefilterbank;
using Eigen::MatrixXd; using Eigen::VectorXd; using Eigen::MatrixXcd;
static constexpr double kPi = 3.14159265358979323846;

int main() {
    const double fs = 1000; const int n = 4096;
    tb::Design d(tb::Family::Meyer, 2.0, tb::Sides::Analytic); d.bands = {{4, 30}}; d.edgeBins = tb::EdgeBins::Drop;      // a wide band: its plateau holds the tone at its full gain
    const tb::TimeFilterbank bank = tb::design(d, n, fs);
    const double f0 = 10 * 1.0, binf = std::round(f0 * n / fs) * fs / n, A = 1.3, phi = 0.4;   // a frequency on a bin of the grid
    MatrixXd x(2, n); for (int i = 0; i < n; ++i) { x(0, i) = A * std::cos(2 * kPi * binf * i / fs + phi); x(1, i) = 2 * A * std::sin(2 * kPi * binf * i / fs); }
    VectorXd at(5); at << 100.25, 777.5, 1500.125, 2222.9, 3001.333;
    const MatrixXcd a = wt(bank, 0, x, at);
    double eAmp = 0, ePh = 0;
    for (int j = 0; j < at.size(); ++j) {
        eAmp = std::max({eAmp, std::abs(std::abs(a(0, j)) - A), std::abs(std::abs(a(1, j)) - 2.0 * A)});
        ePh = std::max(ePh, std::abs(std::remainder(std::arg(a(0, j)) - (2 * kPi * binf * at(j) / fs + phi), 2 * kPi)));
    }
    EXPECT(eAmp < 1e-9 * A && ePh < 1e-9, "a tone on a bin: |a| = (G/2) A with G = 2 and the phase of the tone at fractional instants, " + std::to_string(eAmp) + " / " + std::to_string(ePh));
    // whole samples: the full-rate wt of the window
    {
        std::mt19937 g(5); std::normal_distribution<double> N; MatrixXd z(1, n); for (int i = 0; i < n; ++i) z(0, i) = N(g);
        const auto W = wt(bank, MatrixXd(z.transpose()));
        VectorXd ints(4); ints << 10, 500, 2047, 4000;
        const MatrixXcd b = wt(bank, 0, z, ints);
        double e = 0; for (int j = 0; j < ints.size(); ++j) e = std::max(e, std::abs(b(0, j) - W[0]((int)ints(j), 0)));
        EXPECT(e < 1e-11, "whole-sample instants equal the full-rate wt of the window " + std::to_string(e));
    }
    EXPECT([&] { try { wt(bank, 7, x, at); } catch (const std::invalid_argument&) { return true; } return false; }(), "a member outside the bank is refused");
    std::cout << (g_failures ? "FAILED" : "all passed") << " (" << g_failures << " failures)\n";
    return g_failures ? 1 : 0;
}
