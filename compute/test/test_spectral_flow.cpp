// nxr::signal spectral_flow on synthetic data (generated signals, random coordinates, an icosphere's edges):
//   overlappedSpectrum: the operator applied before the transform equals after it; the left side reflected at a recording's start equals the transform of the
//     explicitly reflected samples (numpy `reflect`); a chunk with its whole overlap present is the plain transform of those samples;
//   normaliseBySumOfSquaredGains (on Gaussian atoms): the squared gains sum to one on the support; cosineSquaredPartition: the gains sum to one at every point;
//   pairFlow on the field coordinates: the pairs of an amplitude partition sum to the whole field's values; the whole field's values equal the direct sums over the draws
//     of the cross-spectrum (power |z|², phaseCurrent(z), Im(conj(z) sharp(d0 z))). (The time cells: test_flow_readout.)
#include <cmath>
#include <complex>
#include <iostream>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/spectral_flow.h"

// Gaussian gains on a frequency line, standard deviations in Hz, 0 at f <= 0: the test's own synthetic atoms (a kernel set's gains are nxr::signal::kernelGains)
static Eigen::MatrixXd gaussianGains(const Eigen::VectorXd& frequencies, const std::vector<double>& centres, const std::vector<double>& standardDeviations, double gain) {
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(centres.size()), frequencies.size());
  for (std::size_t a = 0; a < centres.size(); ++a)
    for (Eigen::Index k = 0; k < frequencies.size(); ++k) {
      const double f = frequencies(k), sd = standardDeviations[a];
      if (f > 0) g(static_cast<Eigen::Index>(a), k) = gain * std::exp(-(f - centres[a]) * (f - centres[a]) / (2.0 * sd * sd));
    }
  return g;
}

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
    // a SHORT last chunk: the right side needs more reflection than the samples hold; reflected again (numpy pads `reflect` repeatedly): grown pass by pass here
    const int shortPart = 3;
    const Eigen::MatrixXd u = randn(channels, overlap + shortPart);
    std::vector<Eigen::VectorXd> grown;
    for (int j = 0; j < overlap + shortPart; ++j) grown.push_back(u.col(j));
    while (static_cast<int>(grown.size()) < N) {
      const int n = static_cast<int>(grown.size());
      for (int j = 0; j < n - 1 && static_cast<int>(grown.size()) < N; ++j) grown.push_back(grown[static_cast<std::size_t>(n - 2 - j)]);
    }
    Eigen::MatrixXd shortReflected(channels, N);
    for (int j = 0; j < N; ++j) shortReflected.col(j) = grown[static_cast<std::size_t>(j)];
    const Eigen::MatrixXcd S = ns::overlappedSpectrum(op, u, overlap, 0, overlap, N, 0, N / 2 + 1);
    EXPECT((S - op.cast<cd>() * ns::rfft(shortReflected, 1)).cwiseAbs().maxCoeff() < 1e-10, "a short last chunk: reflected again where the samples are too few (numpy reflect)");
    bool refused = false;
    try { ns::overlappedSpectrum(op, randn(channels, 1), 0, 0, overlap, N, 0, 3); } catch (const std::invalid_argument&) { refused = true; }
    EXPECT(refused, "refused: one sample cannot be reflected");
  }

  // ── the atoms and the partitions ──
  {
    Eigen::VectorXd f = Eigen::VectorXd::LinSpaced(2000, 0.0, 60.0);
    std::vector<double> centres, sds;
    for (int k = 0; k < 17; ++k) { centres.push_back(4.0 * std::pow(2.0, k / 5.0)); sds.push_back(centres.back() / 12.0); }
    const Eigen::MatrixXd g = gaussianGains(f, centres, sds, 2.0);
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
  }

  // ── pairFlow on the field coordinates: the whole field against the direct sums over the draws; the band pairs of an amplitude partition sum to the whole ──
  std::vector<double> Vx; std::vector<int32_t> F;
  icosphere(1, Vx, F, 1.0);
  const ns::IntrinsicComplex c = ns::intrinsicComplex(nxr::geometry::surfaceArrays(Vx.data(), static_cast<int>(Vx.size() / 3), F.data(), static_cast<int>(F.size() / 3)));
  const int S = c.nV, K = 9, draws = 40;
  const Eigen::MatrixXd Psi = randn(S, K);
  const Eigen::SparseMatrix<double> sharpWeights = ns::sharpOperator(c);
  {
    const Eigen::MatrixXcd Y = crandn(K, draws);
    Eigen::VectorXd mu = Eigen::VectorXd::LinSpaced(K, 0.5, 4.0);
    const Eigen::MatrixXd H = ns::cosineSquaredPartition(mu.array().log() / std::log(2.0), {0.0, 1.0, 2.0});
    std::vector<Eigen::MatrixXd> bandReadouts;
    for (Eigen::Index q = 0; q < H.rows(); ++q) bandReadouts.push_back(Psi * H.row(q).transpose().asDiagonal());
    const ns::PairFlow pairs = ns::pairFlow(Y, bandReadouts, 1, c.edges, &sharpWeights);
    const ns::PairFlow whole = ns::pairFlow(Y, {Psi}, 1, c.edges, &sharpWeights);
    const Eigen::MatrixXcd z = Psi.cast<cd>() * Y;
    const Eigen::VectorXd power = z.cwiseAbs2().rowwise().sum();
    const Eigen::VectorXd current = ns::phaseCurrent(c, z).rowwise().sum();
    Eigen::VectorXd vertex = Eigen::VectorXd::Zero(3 * S);                                    // Σ_draws Im(conj(z_v) (grad z)_v), the gradient sharp(d0 z) of each draw
    const Eigen::MatrixXd gradientReal = ns::sharp(c, Eigen::MatrixXd(c.d0 * z.real())), gradientImaginary = ns::sharp(c, Eigen::MatrixXd(c.d0 * z.imag()));
    for (int r = 0; r < 3 * S; ++r) for (int n = 0; n < draws; ++n) vertex(r) += (std::conj(z(r / 3, n)) * cd(gradientReal(r, n), gradientImaginary(r, n))).imag();
    EXPECT((whole.power.col(0) - power).cwiseAbs().maxCoeff() < 1e-8 * power.maxCoeff(), "whole field: power = Σ |z|² over the draws");
    EXPECT((whole.pairImaginary.col(0) - current).cwiseAbs().maxCoeff() < 1e-8 * current.cwiseAbs().maxCoeff(), "whole field: edge phase current = Σ phaseCurrent(z)");
    EXPECT((whole.vertexCurrent.col(0) - vertex).cwiseAbs().maxCoeff() < 1e-8 * vertex.cwiseAbs().maxCoeff(), "whole field: vertex phase current = Σ Im(conj(z) sharp(d0 z))");
    EXPECT((pairs.power.rowwise().sum() - whole.power.col(0)).cwiseAbs().maxCoeff() < 1e-8 * power.maxCoeff() &&
           (pairs.pairImaginary.rowwise().sum() - whole.pairImaginary.col(0)).cwiseAbs().maxCoeff() < 1e-8 * current.cwiseAbs().maxCoeff() &&
           (pairs.vertexCurrent.rowwise().sum() - whole.vertexCurrent.col(0)).cwiseAbs().maxCoeff() < 1e-8 * vertex.cwiseAbs().maxCoeff(), "the band pairs of an amplitude partition sum to the whole");
  }

  std::cout << (failures ? "FAILED\n" : "all passed\n");
  return failures ? 1 : 0;
}
