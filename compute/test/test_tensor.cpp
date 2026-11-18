// nxr::meeg tensor: the cross-spectral tensor of a free-orientation source and its readings. Its trace is the band power (sourcePower, Norm) exactly;
// it is Hermitian; known currents give known Stokes parameters (the cross-spectrum S + i Sq = X X^H of a current's complex amplitude X at the band):
// turning from e1 to e2 (S3 = +S0), the other way (S3 = -S0), swinging along e1 (S1 = S0), along 45 degrees (S2 = S0), along the normal (normal power =
// the trace, S0 = 0); and a turn of the frame by psi turns S1 + i S2 by 2 psi and leaves S0, S3 and |S1 + i S2| alone.
#include <cmath>
#include <complex>
#include <random>

#include "check.h"
#include "nxr/meeg/sources.h"
#include "nxr/core/congruence.h"
#include "nxr/signal/polarisation.h"

using namespace nxr::meeg;
using namespace nxr::signal;
using namespace nxr::core;
using Eigen::MatrixXd;
using cd = std::complex<double>;

// a single source whose kernel is the identity (its three channels ARE its x y z), current of complex amplitude X: S + i Sq = X X^H
static SourceTensors one(const Eigen::Vector3cd& X) {
  const Eigen::Matrix3cd C = X * X.adjoint();
  return sourceTensor(MatrixXd::Identity(3, 3), C.real(), C.imag());
}

int main() {
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  auto rnd = [&](int r, int c) { MatrixXd m(r, c); for (auto& v : m.reshaped()) v = nd(rng); return m; };
  // the TRACE is the band power, exactly; Hermitian
  {
    const int M = 7, N = 5;
    const MatrixXd K = rnd(3 * N, M), A = rnd(M, M), B = rnd(M, M);
    const MatrixXd S = A * A.transpose(), Sq = B - B.transpose();
    const SourceTensors t = sourceTensor(K, S, Sq);
    const MatrixXd P = sourcePower(K, {S}, 3, Orientation::Norm);
    double worst = 0, herm = 0;
    for (int v = 0; v < N; ++v) {
      worst = std::max(worst, std::abs(tracePower(t.reOf(v)) - P(v, 0)) / std::abs(P(v, 0)));
      for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) herm = std::max({herm, std::abs(t.reOf(v)[3 * a + b] - t.reOf(v)[3 * b + a]), std::abs(t.imOf(v)[3 * a + b] + t.imOf(v)[3 * b + a])});
    }
    std::printf("  trace vs band power: %.2e; Hermitian to %.2e\n", worst, herm);
    CHECK(worst < 1e-13 && herm < 1e-12);
  }
  const std::array<double, 3> e1 = {1, 0, 0}, e2 = {0, 1, 0}, n = {0, 0, 1};
  const cd I(0, 1);
  // TURNING from e1 to e2: x = cos, y = sin -> amplitudes (1, -i, 0) (sin at +f is -i cos)
  {
    const SourceTensors t = one(Eigen::Vector3cd(1, -I, 0));
    const Stokes s = stokes(t.reOf(0), t.imOf(0), e1, e2);
    std::printf("  turning e1 -> e2: S = (%.3f, %.3f, %.3f, %.3f)\n", s.s0, s.s1, s.s2, s.s3);
    CHECK(std::abs(s.s0 - 2) < 1e-12 && std::abs(s.s1) < 1e-12 && std::abs(s.s2) < 1e-12 && std::abs(s.s3 - 2) < 1e-12);
    const Stokes r = stokes(one(Eigen::Vector3cd(1, I, 0)).reOf(0), one(Eigen::Vector3cd(1, I, 0)).imOf(0), e1, e2);
    CHECK(std::abs(r.s3 + 2) < 1e-12);                                   // the other way round
  }
  // SWINGING along e1, along 45 degrees, along the normal
  {
    const SourceTensors a = one(Eigen::Vector3cd(1, 0, 0)), d = one(Eigen::Vector3cd(1, 1, 0)), z = one(Eigen::Vector3cd(0, 0, 1));
    const Stokes sa = stokes(a.reOf(0), a.imOf(0), e1, e2), sd = stokes(d.reOf(0), d.imOf(0), e1, e2), sz = stokes(z.reOf(0), z.imOf(0), e1, e2);
    CHECK(std::abs(sa.s1 - sa.s0) < 1e-12 && std::abs(sa.s2) < 1e-12 && std::abs(sa.s3) < 1e-12);
    CHECK(std::abs(sd.s2 - sd.s0) < 1e-12 && std::abs(sd.s1) < 1e-12 && std::abs(sd.s3) < 1e-12);
    CHECK(std::abs(normalPower(z.reOf(0), n) - tracePower(z.reOf(0))) < 1e-12 && std::abs(sz.s0) < 1e-12);
  }
  // THE GAUGE: the frame turned by psi
  {
    const SourceTensors t = one(Eigen::Vector3cd(cd(0.7, 0.2), cd(-0.3, 0.9), cd(0.4, -0.1)));
    const Stokes s = stokes(t.reOf(0), t.imOf(0), e1, e2);
    const double psi = 0.6;
    const std::array<double, 3> f1 = {std::cos(psi), std::sin(psi), 0}, f2 = {-std::sin(psi), std::cos(psi), 0};
    const Stokes r = stokes(t.reOf(0), t.imOf(0), f1, f2);
    const cd L(s.s1, s.s2), Lr(r.s1, r.s2);
    CHECK(std::abs(r.s0 - s.s0) < 1e-12 && std::abs(r.s3 - s.s3) < 1e-12 && std::abs(std::abs(Lr) - std::abs(L)) < 1e-12);
    CHECK(std::abs(Lr - L * std::exp(cd(0, -2 * psi))) < 1e-12);         // the axis is measured from e1: turning the frame by psi turns it back by 2 psi
  }
  // a BLOCK between two sources: T_ii is sourceTensor's (to the bit: the same products), T_ji = T_ij^H
  {
    const int M = 6, N = 4;
    const MatrixXd K = rnd(3 * N, M), A = rnd(M, M), B = rnd(M, M);
    const MatrixXd S = A * A.transpose(), Sq = B - B.transpose();
    const SourceTensors t = sourceTensor(K, S, Sq);
    double diag = 0, adj = 0, rij[9], iij[9], rji[9], iji[9];
    for (int v = 0; v < N; ++v) {
      crossTensor(K, S, Sq, v, v, rij, iij);
      for (int k = 0; k < 9; ++k) diag = std::max({diag, std::abs(rij[k] - t.reOf(v)[k]), std::abs(iij[k] - t.imOf(v)[k])});
    }
    crossTensor(K, S, Sq, 1, 3, rij, iij);
    crossTensor(K, S, Sq, 3, 1, rji, iji);
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) adj = std::max({adj, std::abs(rji[3 * a + b] - rij[3 * b + a]), std::abs(iji[3 * a + b] + iij[3 * b + a])});
    std::printf("  block T_ii vs sourceTensor: %.2e; T_ji vs T_ij^H: %.2e\n", diag, adj);
    CHECK(diag < 1e-12 && adj < 1e-12);
    bool outside = false;
    try { crossTensor(K, S, Sq, 0, N, rij, iij); } catch (const std::invalid_argument&) { outside = true; }
    CHECK(outside);
  }
  return nxrtest::finish("test_tensor");
}
