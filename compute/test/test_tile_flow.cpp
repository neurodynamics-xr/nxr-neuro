// nxr::signal tile flow (nxr/signal/tile_flow.h) on generated signals (no dataset):
//   servedSynthesis: the inverse of a complete serving operator, the rank of an incomplete one;
//   tileOperator: the kernel's rows averaged over each tile, composed with a synthesis;
//   a planted directed lag (read through pairFlow and timeCellPairFlow): tile 1 follows tile 0 by tau on one component; the whole-run cross-spectrum's phase gives the lag back (-arg S_01 / 2 pi f = -tau, the reference's
//   sign) and every time cell's Im sum is positive (tile 1 lags tile 0: the phase flows 0 -> 1); the per-cell sums add up to the whole own span's time-domain products.
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "check.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/source_flow.h"
#include "nxr/signal/tile_flow.h"

using namespace nxr::signal;
using cd = std::complex<double>;

int main() {
  std::mt19937 rng(3);
  std::normal_distribution<double> normal(0, 1);
  // ── the synthesis of a serving operator ──
  {
    Eigen::MatrixXd S(5, 5);
    for (int i = 0; i < 25; ++i) S(i / 5, i % 5) = normal(rng);
    const ServedSynthesis s = servedSynthesis(S);
    CHECK(s.rank == 5 && (S * s.synthesis - Eigen::MatrixXd::Identity(5, 5)).norm() < 1e-9);
    Eigen::MatrixXd low = S.topRows(3);
    const ServedSynthesis l = servedSynthesis(low);
    CHECK(l.rank == 3 && (low * l.synthesis - Eigen::MatrixXd::Identity(3, 3)).norm() < 1e-9);
  }
  // ── the tile operator ──
  {
    const int V = 4, C = 3;
    Eigen::MatrixXd K(3 * V, C);
    for (int i = 0; i < 3 * V * C; ++i) K(i / C, i % C) = normal(rng);
    Eigen::MatrixXd Q(C, 2);
    for (int i = 0; i < 2 * C; ++i) Q(i / 2, i % 2) = normal(rng);
    const Eigen::MatrixXd M = tileOperator(K, {0, 0, 1, -1}, 2, Q);
    CHECK(M.rows() == 6 && M.cols() == 2);
    CHECK((M.middleRows(0, 3) - 0.5 * (K.middleRows(0, 3) + K.middleRows(3, 3)) * Q).norm() < 1e-12);
    CHECK((M.middleRows(3, 3) - K.middleRows(6, 3) * Q).norm() < 1e-12);
  }
  // ── a planted lag between two tiles (the operator the identity on 6 coordinates: tile t component c = coordinate 3 t + c) ──
  {
    const double fs = 200;
    const std::int64_t N = 4096, own0 = 1024, own = 2048, cell = 256;
    const double f = fs * 205.0 / static_cast<double>(N), tau = 0.012;
    Eigen::MatrixXd x(6, N);
    for (std::int64_t t = 0; t < N; ++t) {
      const double s = static_cast<double>(t) / fs;
      for (int r = 0; r < 6; ++r) x(r, t) = 0.05 * normal(rng);
      x(0, t) += std::cos(2 * M_PI * f * s);
      x(3, t) += std::cos(2 * M_PI * f * (s - tau));
    }
    const Eigen::MatrixXcd Y = rfft(x, 1);                                                    // [6 x N/2 + 1]
    Eigen::MatrixXd gains(1, Y.cols());
    for (Eigen::Index k = 0; k < Y.cols(); ++k) { const double fk = static_cast<double>(k) * fs / static_cast<double>(N); gains(0, k) = k == 0 ? 0 : 2 * std::exp(-(fk - f) * (fk - f) / 2.0); }
    const Eigen::MatrixXd M = Eigen::MatrixXd::Identity(6, 6);
    const Eigen::MatrixXd tileReadout = componentMajor(M, 3);                                 // two tiles of three components
    const std::vector<std::int32_t> tilePair{0, 1};
    const PairFlow flow = pairFlow(Y * gains.row(0).transpose().asDiagonal(), {tileReadout}, 3, tilePair, nullptr);
    const cd crossSpectrum01(flow.pairReal(0, 0), -flow.pairImaginary(0, 0));                  // S_01 = Σ z_0 conj(z_1), the conjugate of the pair sum
    const double lag = -std::arg(crossSpectrum01) / (2 * M_PI * f);
    std::printf("  planted lag %.4f s, read %.4f s (the reference's sign: -tau)\n", tau, lag);
    CHECK(std::abs(lag + tau) < 1e-3);
    CHECK(crossSpectrum01.imag() > 0);
    const Eigen::MatrixXcd X = Y * gains.row(0).asDiagonal();
    const TimeCellPairFlow c = timeCellPairFlow(X, 0, N, own0, own, 0, cell, tileReadout, 3, tilePair, nullptr);
    CHECK(c.firstCell == 0 && c.samples.size() == static_cast<std::size_t>(own / cell));
    const TimeCellPairFlow shifted = timeCellPairFlow(X, 0, N, own0, own, 100, cell, tileReadout, 3, tilePair, nullptr);   // the own span starting at line sample 100: a part cell at each end
    CHECK(shifted.firstCell == 0 && shifted.samples.front() == cell - 100 && shifted.samples.back() == 100 && std::abs(shifted.pairImaginary.sum() - c.pairImaginary.sum()) < 1e-9 * std::abs(c.pairImaginary.sum()));
    CHECK(c.pairImaginary.rows() == 1 && c.pairImaginary.cols() == own / cell && c.power.rows() == 2);
    bool positive = true;
    for (Eigen::Index j = 0; j < c.pairImaginary.cols(); ++j) positive = positive && -c.pairImaginary(0, j) > 0;   // Σ Im(z_0 conj z_1) > 0: tile 1 lags tile 0
    CHECK(positive);
    // the cells' sums are the time-domain sums of the band signal over the own span (an inverse transform of the weighted bins, unscaled)
    double whole = 0;                                                                         // Σ over the three components of tile 0 (rows 0 1 2) with tile 1 (rows 3 4 5)
    for (int comp = 0; comp < 3; ++comp) {
      std::vector<cd> z0(static_cast<std::size_t>(N), cd(0, 0)), z1(static_cast<std::size_t>(N), cd(0, 0));
      for (Eigen::Index k = 0; k < X.cols(); ++k) { z0[static_cast<std::size_t>(k)] = X(comp, k); z1[static_cast<std::size_t>(k)] = X(3 + comp, k); }
      c2c(z0.data(), z0.data(), {static_cast<std::size_t>(N)}, 0, false);
      c2c(z1.data(), z1.data(), {static_cast<std::size_t>(N)}, 0, false);
      for (std::int64_t t = own0; t < own0 + own; ++t) whole += (z0[static_cast<std::size_t>(t)] * std::conj(z1[static_cast<std::size_t>(t)])).imag();
    }
    CHECK(std::abs(-c.pairImaginary.sum() - whole) < 1e-9 * std::abs(whole));
  }
  return nxrtest::finish("test_tile_flow");
}
