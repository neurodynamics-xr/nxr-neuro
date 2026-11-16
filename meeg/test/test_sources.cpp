// nxr::meeg sources: J = K X (real, complex), the orientation handling (components, norm, amplitude), fixed kernels, and the source power from a cross-spectrum equal to the power of J.
#include <random>

#include "check.h"
#include "nxr/meeg/sources.h"

using namespace nxr::meeg;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;

int main() {
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  auto rnd = [&](int r, int c) { MatrixXd m(r, c); for (auto& v : m.reshaped()) v = nd(rng); return m; };
  const int M = 5, N = 4, T = 64;
  const MatrixXd K = rnd(3 * N, M), X = rnd(M, T);
  const MatrixXd J = applyKernel(K, X);
  CHECK((J - K * X).cwiseAbs().maxCoeff() < 1e-14);
  const MatrixXd Jn = applyKernel(K, X, 3, Orientation::Norm);
  CHECK_EQ(Jn.rows(), N);
  for (int s = 0; s < N; ++s) for (int t = 0; t < T; ++t)
    CHECK(std::abs(Jn(s, t) - std::sqrt(J(3 * s, t) * J(3 * s, t) + J(3 * s + 1, t) * J(3 * s + 1, t) + J(3 * s + 2, t) * J(3 * s + 2, t))) < 1e-12);
  // complex coefficients: linear in the real and imaginary parts; the amplitude is the norm over the three orientations
  const MatrixXcd Z = rnd(M, T).cast<std::complex<double>>() + std::complex<double>(0, 1) * rnd(M, T).cast<std::complex<double>>();
  const MatrixXcd Jz = applyKernel(K, Z);
  CHECK((Jz.real() - K * Z.real()).cwiseAbs().maxCoeff() < 1e-12 && (Jz.imag() - K * Z.imag()).cwiseAbs().maxCoeff() < 1e-12);
  const MatrixXd A = amplitude(Jz, 3);
  CHECK_EQ(A.rows(), N);
  CHECK(std::abs(A(1, 7) - std::sqrt(std::norm(Jz(3, 7)) + std::norm(Jz(4, 7)) + std::norm(Jz(5, 7)))) < 1e-12);
  CHECK((applyKernelAmplitude(K, Z, 3, Orientation::Norm) - A).cwiseAbs().maxCoeff() < 1e-14);
  CHECK_EQ(applyKernelAmplitude(K, Z, 3, Orientation::Components).rows(), 3 * N);
  // a fixed kernel: one row per source; the norm is |J|
  const MatrixXd Kf = K.topRows(N);
  CHECK((applyKernel(Kf, X, 1, Orientation::Norm) - (Kf * X).cwiseAbs()).cwiseAbs().maxCoeff() < 1e-14);
  // source power: diag(K S K^T) with S = X X^T is the sum of J^2 over time; the three orientations summed, or apart
  const MatrixXd S = X * X.transpose();
  const MatrixXd Pc = sourcePower(K, {S}, 3, Orientation::Components), Ps = sourcePower(K, {S}, 3, Orientation::Norm);
  CHECK((Pc.col(0) - J.array().square().rowwise().sum().matrix()).cwiseAbs().maxCoeff() < 1e-9);
  CHECK((Ps.col(0) - Jn.array().square().rowwise().sum().matrix()).cwiseAbs().maxCoeff() < 1e-9);
  const MatrixXd Pr = relativePower(Ps, Ps.col(0) * 2);
  CHECK(std::abs(Pr(2, 0) - 0.5) < 1e-14);
  CHECK(brainstormBands().size() == 6 && brainstormBands()[2].name == "alpha" && brainstormBands()[2].lo == 8 && brainstormBands()[2].hi == 12);
  bool threw = false;
  try { applyKernel(K, rnd(M + 1, T)); } catch (const std::invalid_argument&) { threw = true; }
  CHECK(threw);
  return nxrtest::finish("test_sources");
}
