// nxr::meeg wave: the phase sums from the cross-spectrum equal the same averages taken over an ensemble of complex amplitudes X (C = mean X X^H), and the
// readouts of known patterns are known: a plane wave z = e^{-i k.x} on a line of points gives grad phi = -k, the velocity along +k at omega/|k|, travelling 1;
// a standing wave cos(k.x) gives travelling 0; two waves going opposite ways with equal power cancel (travelling 0).
#include <cmath>
#include <complex>
#include <random>

#include "check.h"
#include "nxr/meeg/wave.h"

using namespace nxr::meeg;
using Eigen::MatrixXd;
using cd = std::complex<double>;

int main() {
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  // the SUMS against an ensemble: z = A X, g = B X, X complex
  {
    const int N = 5, M = 7, E = 400;
    MatrixXd A(N, M), B[3];
    for (auto& v : A.reshaped()) v = nd(rng);
    for (auto& b : B) { b.resize(N, M); for (auto& v : b.reshaped()) v = nd(rng); }
    Eigen::MatrixXcd X(M, E);
    for (int i = 0; i < M; ++i) for (int e = 0; e < E; ++e) X(i, e) = cd(nd(rng), nd(rng));
    const Eigen::MatrixXcd C = X * X.adjoint() / E;
    const PhaseSums s = phaseSums(A, B, C.real(), C.imag());
    const Eigen::MatrixXcd z = A * X;
    double worst = 0;
    for (int v = 0; v < N; ++v) {
      const double P = z.row(v).squaredNorm() / E;
      double Q = 0;
      Eigen::Vector3d H;
      for (int d = 0; d < 3; ++d) {
        const Eigen::RowVectorXcd g = B[d].row(v) * X;
        H[d] = (z.row(v).conjugate().array() * g.array()).sum().imag() / E;
        Q += g.squaredNorm() / E;
      }
      worst = std::max({worst, std::abs(P - s.power(v)) / P, (H - s.phase.row(v).transpose()).norm() / H.norm(), std::abs(Q - s.gradientPower(v)) / Q});
    }
    std::printf("  sums vs ensemble: %.2e\n", worst);
    CHECK(worst < 1e-12);
  }
  // KNOWN PATTERNS on a line of points x_j = j h (one channel a point, so a = identity and B = the central difference along x)
  {
    const int N = 64; const double h = 0.01, k = 2 * M_PI / (16 * h), omega = 2 * M_PI * 10;
    MatrixXd A = MatrixXd::Identity(N, N), B[3] = {MatrixXd::Zero(N, N), MatrixXd::Zero(N, N), MatrixXd::Zero(N, N)};
    for (int j = 1; j + 1 < N; ++j) { B[0](j, j + 1) = 0.5 / h; B[0](j, j - 1) = -0.5 / h; }
    auto sums_of = [&](const std::vector<Eigen::VectorXcd>& amps) {     // C = sum over independent patterns of their X X^H
      Eigen::MatrixXcd C = Eigen::MatrixXcd::Zero(N, N);
      for (const auto& x : amps) C += x * x.adjoint();
      return phaseSums(A, B, C.real(), C.imag());
    };
    Eigen::VectorXcd plane(N), back(N), standing(N);
    for (int j = 0; j < N; ++j) { plane(j) = std::exp(cd(0, -k * j * h)); back(j) = std::exp(cd(0, k * j * h)); standing(j) = std::cos(k * j * h); }
    const PhaseReadout p = phaseReadout(sums_of({plane}), omega), st = phaseReadout(sums_of({standing}), omega), two = phaseReadout(sums_of({plane, back}), omega);
    const int v = N / 2;
    const double kd = std::sin(k * h) / h;                                    // what the central difference reads of e^{-ikx}
    std::printf("  plane: grad %.3f (expect %.3f), speed %.3f (expect %.3f), travelling %.3f; standing %.3f; two opposite %.3f\n", p.gradient(v, 0), -kd, p.speed(v), omega / kd,
                p.travelling(v), st.travelling(v), two.travelling(v));
    CHECK(std::abs(p.gradient(v, 0) + kd) < 1e-9 && p.velocity(v, 0) > 0 && std::abs(p.speed(v) - omega / kd) < 1e-6 * omega / kd);
    CHECK(std::abs(p.travelling(v) - std::sin(k * h) / h / kd) < 1e-9);       // |H| / sqrt(P Q) = 1 for a plane wave read by the same difference
    CHECK(std::isnan(st.travelling(v)));                                     // at an antinode the gradient is 0: no phase to read
    CHECK(st.travelling(v + 2) < 1e-12 && two.travelling(v + 2) < 1e-12);
  }
  // the phase current read in the tangent frames: j1 = H.e1, j2 = H.e2, |j|; a vertex with no frame is NaN in all three
  {
    MatrixXd H(3, 3), e1(3, 3), e2(3, 3);
    H << 1, 2, 3, 0, 0, 5, 1, 1, 1;
    e1 << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    e2 << 0, 1, 0, 1, 0, 0, std::nan(""), 0, 0;
    const PhaseCurrentFrames f = phaseCurrentInFrames(H, e1, e2);
    CHECK(f.j1(0) == 1 && f.j2(0) == 2 && std::abs(f.magnitude(0) - std::sqrt(5.0)) < 1e-15);
    CHECK(f.j1(1) == 0 && f.j2(1) == 0 && f.magnitude(1) == 0);
    CHECK(std::isnan(f.j1(2)) && std::isnan(f.j2(2)) && std::isnan(f.magnitude(2)));
    bool threw = false;
    try { phaseCurrentInFrames(MatrixXd(2, 3), e1, e2); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
  return nxrtest::finish("test_wave");
}
