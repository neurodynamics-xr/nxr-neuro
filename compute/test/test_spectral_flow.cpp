// nxr::signal spectral_flow on synthetic data (generated signals, random coordinates, an icosphere's edges):
//   overlappedSpectrum: the operator applied before the transform equals after it; the left side reflected at a recording's start equals the transform of the
//     explicitly reflected samples (numpy `reflect`); a chunk with its whole overlap present is the plain transform of those samples;
//   gaussianAtomGains, normaliseBySumOfSquaredGains: the squared gains sum to one on the support; cosineSquaredPartition and mexicanHatPartition: the gains sum to one at every point;
//   bandPairFlow: the pairs of an amplitude partition sum to the whole field's values; the whole field's values equal the direct sums over the draws of the
//     cross-spectrum (power |z|², phaseCurrent(z), Im(conj(z) G y));
//   timeCellFlow: the cells' sums equal the direct sums of the inverse transform's samples.
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/spectral_flow.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_spectral_flow\n";
  namespace ns = nxr::signal;
  using cd = std::complex<double>;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0, 1);
  auto randn = [&](Eigen::Index r, Eigen::Index c) { Eigen::MatrixXd m(r, c); for (Eigen::Index i = 0; i < r; ++i) for (Eigen::Index j = 0; j < c; ++j) m(i, j) = nd(rng); return m; };
  auto crandn = [&](Eigen::Index r, Eigen::Index c) { return Eigen::MatrixXcd(randn(r, c).cast<cd>() + cd(0, 1) * randn(r, c).cast<cd>()); };

  // ── overlappedSpectrum ──
  {
    const int channels = 4, K = 3, own = 16, overlap = 8, N = 32;
    const Eigen::MatrixXd op = randn(K, channels);
    // an interior chunk: the whole overlap present on both sides
    const Eigen::MatrixXd x = randn(channels, overlap + own + overlap);
    const Eigen::MatrixXcd A = ns::overlappedSpectrum(op, x, overlap, overlap, overlap, N, 0, N / 2 + 1);
    const Eigen::MatrixXcd direct = op.cast<cd>() * ns::rfft(x, 1);
    EXPECT((A - direct).cwiseAbs().maxCoeff() < 1e-10, "interior chunk: the plain transform of the chunk and its overlap; the operator before the transform = after it");
    // the recording's start: no sample on the left, reflected
    const Eigen::MatrixXd y = randn(channels, own + overlap);
    Eigen::MatrixXd reflected(channels, N);
    for (int j = 0; j < overlap; ++j) reflected.col(j) = y.col(overlap - j);
    reflected.rightCols(own + overlap) = y;
    const Eigen::MatrixXcd B = ns::overlappedSpectrum(op, y, 0, overlap, overlap, N, 2, 5);
    EXPECT((B - (op.cast<cd>() * ns::rfft(reflected, 1)).middleCols(2, 5)).cwiseAbs().maxCoeff() < 1e-10, "the recording's start reflected (numpy reflect), bins kept");
    // the recording's end, a part chunk: the right side reflected to the same transform length
    const int part = 10;
    const Eigen::MatrixXd w = randn(channels, overlap + part);
    Eigen::MatrixXd endReflected(channels, N);
    endReflected.leftCols(overlap + part) = w;
    for (int j = 0; j < N - overlap - part; ++j) endReflected.col(overlap + part + j) = w.col(overlap + part - 2 - j);
    const Eigen::MatrixXcd D = ns::overlappedSpectrum(op, w, overlap, 0, overlap, N, 0, N / 2 + 1);
    EXPECT((D - op.cast<cd>() * ns::rfft(endReflected, 1)).cwiseAbs().maxCoeff() < 1e-10, "a part chunk at the recording's end: reflected on the right to one transform length");
    bool refused = false;
    try { ns::overlappedSpectrum(op, randn(channels, 4), 0, 0, overlap, N, 0, 3); } catch (const std::invalid_argument&) { refused = true; }
    EXPECT(refused, "refused: more reflection than the samples hold");
  }

  // ── the atoms and the partitions ──
  {
    Eigen::VectorXd f = Eigen::VectorXd::LinSpaced(2000, 0.0, 60.0);
    std::vector<double> centres, sds;
    for (int k = 0; k < 17; ++k) { centres.push_back(4.0 * std::pow(2.0, k / 5.0)); sds.push_back(centres.back() / 12.0); }
    const Eigen::MatrixXd g = ns::gaussianAtomGains(f, centres, sds, 2.0);
    EXPECT(g(0, 0) == 0.0 && std::abs(g.row(5).maxCoeff() - 2.0) < 1e-2, "Gaussian atoms: 0 at 0 Hz, gain 2 at the centre");
    const Eigen::MatrixXd n = ns::normaliseBySumOfSquaredGains(g, 1e-3);
    const Eigen::RowVectorXd s2 = n.cwiseAbs2().colwise().sum();
    double worst = 0; int support = 0;
    for (Eigen::Index k = 0; k < f.size(); ++k) if (s2(k) > 0) { ++support; worst = std::max(worst, std::abs(s2(k) - 1.0)); }
    EXPECT(support > 1000 && worst < 1e-12, "normalised atoms: the squared gains sum to one on the support");
    Eigen::VectorXd x(6);
    x << -std::numeric_limits<double>::infinity(), 0.0, 1.3, 2.0, 2.7, 9.0;
    const Eigen::MatrixXd p = ns::cosineSquaredPartition(x, {1.0, 2.0, 3.0});
    EXPECT((p.colwise().sum().array() - 1.0).abs().maxCoeff() < 1e-12 && p(0, 0) == 1.0 && p(2, 5) == 1.0 && p(1, 3) == 1.0, "cos² partition: the gains sum to one; the constant in the first band; the last the complement");
    Eigen::VectorXd mu(6);
    mu << 0.0, 0.5, 2.0, 4.0, 8.0, 400.0;
    const Eigen::MatrixXd hat = ns::mexicanHatPartition(mu, {1.0, 2.0, 4.0});
    double peakShare = hat(1, 2);
    EXPECT((hat.colwise().sum().array() - 1.0).abs().maxCoeff() < 1e-12 && hat(0, 0) == 1.0 && hat(2, 5) > 0.999 && (hat.array() >= 0).all(),
           "Mexican hat partition: the gains sum to one; the constant in the first band; far above the last centre, the last member");
    EXPECT(peakShare > hat(0, 2) && peakShare > hat(2, 2), "Mexican hat partition: a member holds most at its own centre");
    // its member is the graph bank's hat in lambda = mu², up to the division by the sum
    const double raw1 = ns::graphfilterbank::mexicanHat(4.0, 4.0), raw0 = ns::graphfilterbank::mexicanHat(4.0, 1.0), raw2 = ns::graphfilterbank::mexicanHat(4.0, 16.0);
    EXPECT(std::abs(peakShare - raw1 / (raw0 + raw1 + raw2)) < 1e-12, "Mexican hat partition: the graph bank's hat divided by the sum of the hats");
  }

  // ── bandPairFlow ──
  std::vector<double> Vx; std::vector<int32_t> F;
  icosphere(1, Vx, F, 1.0);
  const ns::IntrinsicComplex c = ns::intrinsicComplex(nxr::geometry::surfaceArrays(Vx.data(), static_cast<int>(Vx.size() / 3), F.data(), static_cast<int>(F.size() / 3)));
  const int S = c.nV, K = 9, draws = 40;
  const Eigen::MatrixXd Psi = randn(S, K), G = randn(3 * S, K);
  {
    const Eigen::MatrixXcd Y = crandn(K, draws);
    const Eigen::MatrixXcd C = Y * Y.adjoint();
    Eigen::VectorXd mu = Eigen::VectorXd::LinSpaced(K, 0.5, 4.0);
    const Eigen::MatrixXd H = ns::cosineSquaredPartition(mu.array().log() / std::log(2.0), {0.0, 1.0, 2.0});
    const ns::BandPairFlow pairs = ns::bandPairFlow(C, Psi, G, c.edges, H);
    const ns::BandPairFlow whole = ns::bandPairFlow(C, Psi, G, c.edges, Eigen::MatrixXd::Ones(1, K));
    const Eigen::MatrixXcd z = Psi.cast<cd>() * Y, gz = G.cast<cd>() * Y;
    const Eigen::VectorXd power = z.cwiseAbs2().rowwise().sum();
    const Eigen::VectorXd current = ns::phaseCurrent(c, z).rowwise().sum();
    Eigen::VectorXd vertex(3 * S);
    for (int r = 0; r < 3 * S; ++r) { double s = 0; for (int n = 0; n < draws; ++n) s += (std::conj(z(r / 3, n)) * gz(r, n)).imag(); vertex(r) = s; }
    EXPECT((whole.power.col(0) - power).cwiseAbs().maxCoeff() < 1e-8 * power.maxCoeff(), "whole field: power = Σ |z|² over the draws");
    EXPECT((whole.edgeCurrent.col(0) - current).cwiseAbs().maxCoeff() < 1e-8 * current.cwiseAbs().maxCoeff(), "whole field: edge phase current = Σ phaseCurrent(z)");
    EXPECT((whole.vertexCurrent.col(0) - vertex).cwiseAbs().maxCoeff() < 1e-8 * vertex.cwiseAbs().maxCoeff(), "whole field: vertex phase current = Σ Im(conj(z) G y)");
    EXPECT((pairs.power.rowwise().sum() - whole.power.col(0)).cwiseAbs().maxCoeff() < 1e-8 * power.maxCoeff() &&
           (pairs.edgeCurrent.rowwise().sum() - whole.edgeCurrent.col(0)).cwiseAbs().maxCoeff() < 1e-8 * current.cwiseAbs().maxCoeff() &&
           (pairs.vertexCurrent.rowwise().sum() - whole.vertexCurrent.col(0)).cwiseAbs().maxCoeff() < 1e-8 * vertex.cwiseAbs().maxCoeff(), "the band pairs of an amplitude partition sum to the whole");
  }

  // ── timeCellFlow ──
  {
    const int N = 64, first = 3, bins = 9, ownFirst = 16, own = 32, cell = 8;
    const Eigen::MatrixXcd X = crandn(K, bins);
    const Eigen::VectorXd h = Eigen::VectorXd::LinSpaced(K, 0.2, 1.0);
    const ns::CellFlow cells = ns::timeCellFlow(X, first, N, ownFirst, own, cell, Psi, G, c.edges, h);
    Eigen::MatrixXcd full = Eigen::MatrixXcd::Zero(N, 4 * S);                                // time along rows for ns::ifft (axis 0)
    const Eigen::MatrixXcd spectral = (Eigen::MatrixXd(Psi.rows() + G.rows(), K) << Psi, G).finished().cast<cd>() * h.cast<cd>().asDiagonal() * X;
    for (int k = 0; k < bins; ++k) full.row(first + k) = spectral.col(k).transpose();
    const Eigen::MatrixXcd t = ns::ifft(full, 0);                                             // [N x 4S]
    double err = 0, scale = 0;
    for (int q = 0; q < own / cell; ++q) {
      for (int e = 0; e < c.nE; ++e) {
        const int tail = c.edges[2 * e], tip = c.edges[2 * e + 1];
        double s = 0; for (int n = ownFirst + q * cell; n < ownFirst + (q + 1) * cell; ++n) s += (std::conj(t(n, tail)) * t(n, tip)).imag();
        err = std::max(err, std::abs(s - cells.edgeCurrent(e, q))); scale = std::max(scale, std::abs(s));
      }
      for (int r = 0; r < 3 * S; ++r) {
        double s = 0; for (int n = ownFirst + q * cell; n < ownFirst + (q + 1) * cell; ++n) s += (std::conj(t(n, r / 3)) * t(n, S + r)).imag();
        err = std::max(err, std::abs(s - cells.vertexCurrent(r, q)));
      }
      for (int v = 0; v < S; ++v) {
        double s = 0; for (int n = ownFirst + q * cell; n < ownFirst + (q + 1) * cell; ++n) s += std::norm(t(n, v));
        err = std::max(err, std::abs(s - cells.power(v, q)));
      }
    }
    EXPECT(cells.edgeCurrent.cols() == own / cell && err < 1e-10 * std::max(scale, 1e-30), "time cells: the sums over each cell equal the direct sums of the inverse transform");
  }

  std::cout << (failures ? "FAILED\n" : "all passed\n");
  return failures ? 1 : 0;
}
