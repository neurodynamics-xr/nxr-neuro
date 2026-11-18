// nxr::meeg crossspectrum readings: coherency (unit diagonal, |.| <= 1 for a rank-one cross-spectrum, = 1), phaseSlope recovers a known delay from C_ab(f) = exp(i 2 pi f tau)
// (row leads: tau > 0), through a phase that wraps many times, with the weights option; phase_slope_index has the sign of the delay, is antisymmetric, and equals a
// hand-written sum; PSI against a numpy implementation from Nolte et al. 2008 is the golden test (test_crossspectrum_bins_golden).
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <complex>
#include <vector>

#include "check.h"
#include "nxr/signal/crossspectral.h"

using namespace nxr::signal;
using namespace nxr::core;
using cd = std::complex<double>;

int main() {
  const double tau = 0.013, tau2 = -0.004;                       // s: channel 0 leads channel 1 by 13 ms; channel 2 leads 0 by -4 ms (lags)
  const int F = 17;
  Eigen::VectorXd f(F);
  std::vector<Eigen::MatrixXcd> C;
  for (int k = 0; k < F; ++k) {
    f[k] = 8 + 0.25 * k;
    const double amp = 1.0 + 0.1 * k;                            // power varies over the band
    // X_0 = x, X_1 = x e^{-i 2 pi f tau} (delayed), X_2 = x e^{+i 2 pi f (-tau2)}: C_ab = <X_a conj X_b>
    Eigen::Vector3cd X(1, std::polar(1.0, -2 * M_PI * f[k] * tau), std::polar(1.0, 2 * M_PI * f[k] * tau2));
    C.push_back(amp * (X * X.adjoint()));
  }
  const auto H = coherency(C);
  double worst = 0;
  for (const auto& h : H) for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) worst = std::max(worst, std::abs(std::abs(h(a, b)) - 1.0));
  std::printf("  coherency modulus of a rank-one C: %.2e\n", worst);
  CHECK(worst < 1e-13);
  for (auto w : {PhaseSlopeOptions::Weights::Coherence, PhaseSlopeOptions::Weights::None}) {
    PhaseSlopeOptions o; o.weights = w;
    const PhaseSlope ps = phaseSlope(C, f, o);
    std::printf("  delay 0->1 %.6f ms (13), 2->0 %.6f ms (-4), residual %.2e\n", ps.delayMs(0, 1), ps.delayMs(2, 0), ps.residual(0, 1));
    CHECK(std::abs(ps.delayMs(0, 1) - 13.0) < 1e-9 && std::abs(ps.delayMs(1, 0) + 13.0) < 1e-9 && std::abs(ps.delayMs(0, 0)) < 1e-9);
    CHECK(std::abs(ps.delayMs(2, 0) - tau2 * 1000) < 1e-9 && ps.residual(0, 1) < 1e-9);
  }
  const Eigen::MatrixXd psi = phase_slope_index(C);
  double hand = 0;
  for (int k = 0; k + 1 < F; ++k) hand += (std::conj(H[k](0, 1)) * H[k + 1](0, 1)).imag();
  std::printf("  PSI 0->1 %.6f (hand %.6f)\n", psi(0, 1), hand);
  CHECK(psi(0, 1) > 0 && std::abs(psi(0, 1) - hand) < 1e-14 && std::abs(psi(0, 1) + psi(1, 0)) < 1e-14 && psi(2, 0) < 0 && std::abs(psi(1, 1)) < 1e-14);
  // a single bin: no slope
  CHECK(std::isnan(phaseSlope({C[0]}, f.head(1)).delayMs(0, 1)));
  // GOLDEN: the phase-slope index against MNE-Connectivity's `_compute_psi` (0.9.0) on synthetic cross-spectra (gen_psi_golden.py)
  {
    std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/psi_golden.txt");
    std::string line;
    std::getline(in, line);
    int nf = 0, m = 0;
    in >> nf >> m;
    Eigen::VectorXd fg(nf);
    for (auto& v : fg) in >> v;
    std::vector<Eigen::MatrixXcd> Cg(nf, Eigen::MatrixXcd(m, m));
    for (int k = 0; k < nf; ++k) for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) { double x; in >> x; Cg[k](a, b) = x; }
    for (int k = 0; k < nf; ++k) for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) { double x; in >> x; Cg[k](a, b) += cd(0, x); }
    Eigen::MatrixXd ref(m, m);
    for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) in >> ref(a, b);
    CHECK(static_cast<bool>(in));
    const double err = (phase_slope_index(Cg) - ref).cwiseAbs().maxCoeff();
    std::printf("  PSI vs mne-connectivity _compute_psi: %.2e (of %.3f)\n", err, ref.cwiseAbs().maxCoeff());
    CHECK(err < 1e-12);
    // and the delays the golden's data were made with (12 ms, -6 ms, 20 ms; row leads when its delay is smaller): 0->1 about +12 ms
    const PhaseSlope ps = phaseSlope(Cg, fg);
    std::printf("  golden data's phase slope 0->1 %.2f ms (made with 12), 0->2 %.2f (-6), 0->3 %.2f (20)\n", ps.delayMs(0, 1), ps.delayMs(0, 2), ps.delayMs(0, 3));
    CHECK(std::abs(ps.delayMs(0, 1) - 12) < 3 && std::abs(ps.delayMs(0, 2) + 6) < 3 && std::abs(ps.delayMs(0, 3) - 20) < 3);
  }
  return nxrtest::finish("test_crossspectrum_bins");
}
