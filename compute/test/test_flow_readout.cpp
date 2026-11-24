// nxr::signal flow_readout on synthetic data (an icosphere's complex and its Hodge-Dirac modes, random 1-forms):
//   flowSplit: the coefficients on the stored edge rows, the potential and the stream equal projectOneForm / exactPotential / coexactStream column by column;
//     with the remainder split, the remainder's parts equal OneFormRemainderSplit on w minus its synthesis, and the energies are the ⋆1 norms;
//   splitPhases: a batch of 1-forms equals splitPhase on each;
//   pairFlow: with one component and the readouts Psi diag(h_q) it equals bandPairFlow (dense and factored orders); with three components readoutPairFlow; on tiles
//     (all pairs i < j, no sharp) the conjugate of the tile cross-spectrum sums S_ij = Σ z_i conj(z_j) of accumulateTileCrossSpectra;
//   OneFormRemainderSplit: an exact 1-form d0 phi gives back phi and no stream; a coexact 1-form gives back its stream and no potential.
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>

#include "mesh_helpers.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/source_flow.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/signal/tile_flow.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)

int main() {
  std::cout << "test_flow_readout\n";
  namespace ns = nxr::signal;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0, 1);
  auto randn = [&](Eigen::Index r, Eigen::Index c) { Eigen::MatrixXd m(r, c); for (Eigen::Index i = 0; i < r; ++i) for (Eigen::Index j = 0; j < c; ++j) m(i, j) = nd(rng); return m; };
  using ComplexValue = std::complex<double>;
  auto crandn = [&](Eigen::Index r, Eigen::Index c) { return Eigen::MatrixXcd(randn(r, c).cast<ComplexValue>() + ComplexValue(0, 1) * randn(r, c).cast<ComplexValue>()); };
  auto relative = [](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); };

  // ── an icosphere, its complex and its modes ──
  std::vector<double> vertexPositions; std::vector<int32_t> faces;
  icosphere(2, vertexPositions, faces, 1.0);
  const int vertexCount = static_cast<int>(vertexPositions.size() / 3), faceCount = static_cast<int>(faces.size() / 3);
  const auto surface = nxr::geometry::surfaceArrays(vertexPositions.data(), vertexCount, faces.data(), faceCount);
  const ns::IntrinsicComplex complex = ns::intrinsicComplex(surface);
  const auto hodgeDirac = nxr::operators::hodgedirac::assemble(surface);
  nxr::operators::HodgeSolveOptions solveOptions; solveOptions.limit = 60;
  const ns::Modes modes = ns::modesOf(nxr::operators::hodgeDiracModes(hodgeDirac, 120, false, solveOptions), hodgeDirac.nV, hodgeDirac.nE, hodgeDirac.nF);
  const Eigen::SparseMatrix<double> sharpWeights = ns::sharpOperator(complex);

  // ── flowSplit ──
  {
    const Eigen::MatrixXd edgeCurrent = randn(complex.nE, 3);
    const ns::FlowSplit plain = ns::flowSplit(complex, modes, edgeCurrent, nullptr);
    const Eigen::MatrixXd coefficients = ns::projectOneForm(complex, modes, edgeCurrent);
    EXPECT(relative(plain.coefficients, coefficients) < 1e-12 && relative(plain.potential, ns::exactPotential(modes, coefficients)) < 1e-12
               && relative(plain.stream, ns::coexactStream(complex, modes, coefficients)) < 1e-12,
           "flowSplit: coefficients on the stored edge rows, potential sum (c / mu) exact, stream likewise");
    EXPECT(plain.remainderPotential.size() == 0 && plain.remainderStream.size() == 0 && plain.remainderEnergy.size() == 0, "flowSplit: no remainder parts unless a remainder split is given");
    const Eigen::VectorXd energy = edgeCurrent.cwiseAbs2().transpose() * complex.star1;
    EXPECT(relative(plain.flowEnergy, energy) < 1e-12, "flowSplit: the flow's ⋆1 energy per column");
    const ns::OneFormRemainderSplit remainderSplit(complex);
    const ns::FlowSplit withRemainder = ns::flowSplit(complex, modes, edgeCurrent, &remainderSplit);
    const Eigen::MatrixXd remainder = edgeCurrent - ns::synthesizeOneForm(complex, modes, coefficients);
    double worst = 0;
    for (Eigen::Index column = 0; column < edgeCurrent.cols(); ++column) {
      const ns::OneFormRemainderSplit::Parts parts = remainderSplit.split(remainder.col(column));
      worst = std::max({worst, relative(withRemainder.remainderPotential.col(column), parts.potential), relative(withRemainder.remainderStream.col(column), parts.stream)});
    }
    const Eigen::VectorXd remainderEnergy = remainder.cwiseAbs2().transpose() * complex.star1;
    EXPECT(worst < 1e-12 && relative(withRemainder.remainderEnergy, remainderEnergy) < 1e-12, "flowSplit: the remainder beyond the kept modes split by the sparse solves, its energy");
  }

  // ── splitPhases: a batch equals splitPhase on each ──
  {
    const Eigen::MatrixXd oneForms = randn(complex.nE, 4);
    const ns::PhaseSplits batch = ns::splitPhases(complex, modes, oneForms, sharpWeights);
    double worst = 0;
    for (Eigen::Index column = 0; column < oneForms.cols(); ++column) {
      const ns::PhaseSplit one = ns::splitPhase(complex, modes, oneForms.col(column));
      worst = std::max({worst, relative(batch.coefficients.col(column), one.coef), relative(batch.exact.col(column), one.exact), relative(batch.coexact.col(column), one.coexact),
                        relative(batch.potential.col(column), one.potential), relative(batch.stream.col(column), one.stream), relative(batch.propagation.col(column), one.propagation.col(0))});
    }
    std::printf("  splitPhases vs splitPhase: %.2e\n", worst);
    EXPECT(worst < 1e-12, "splitPhases: a batch of 1-forms equals splitPhase on each");
  }


  // ── the field coordinates: the constant and the exact modes' vertex parts, each at unit mass norm; a cos² partition of their eigenvalues ──
  std::vector<int> exactModes;
  for (Eigen::Index k = 0; k < modes.mu.size(); ++k) if (modes.kind[static_cast<std::size_t>(k)] == 0) exactModes.push_back(static_cast<int>(k));
  const Eigen::Index coordinateCount = static_cast<Eigen::Index>(exactModes.size()) + 1;
  Eigen::MatrixXd synthesis(vertexCount, coordinateCount);
  Eigen::VectorXd log2Eigenvalue(coordinateCount);
  synthesis.col(0).setConstant(1.0 / std::sqrt(complex.star0.sum())); log2Eigenvalue(0) = -std::numeric_limits<double>::infinity();
  for (Eigen::Index j = 1; j < coordinateCount; ++j) {
    synthesis.col(j) = modes.P.col(exactModes[static_cast<std::size_t>(j - 1)]);
    synthesis.col(j) /= std::sqrt(synthesis.col(j).cwiseAbs2().dot(complex.star0));
    log2Eigenvalue(j) = std::log2(modes.mu(exactModes[static_cast<std::size_t>(j - 1)]));
  }
  const Eigen::MatrixXd bandGains = ns::cosineSquaredPartition(log2Eigenvalue, {log2Eigenvalue(3), log2Eigenvalue(coordinateCount - 1)});
  std::vector<Eigen::MatrixXd> bandReadouts;
  for (Eigen::Index q = 0; q < bandGains.rows(); ++q) bandReadouts.push_back(synthesis * bandGains.row(q).transpose().asDiagonal());

  // ── pairFlow: the sensor readout (one component, mesh edges, sharp) = bandPairFlow ──
  {
    const Eigen::MatrixXd gradients = ns::sharp(complex, Eigen::MatrixXd(complex.d0 * synthesis));
    for (const Eigen::Index draws : {Eigen::Index(4), Eigen::Index(3 * coordinateCount)}) {             // factored (few draws) and dense
      const Eigen::MatrixXcd weightedColumns = crandn(coordinateCount, draws);
      const ns::BandPairFlow want = ns::bandPairFlow(weightedColumns * weightedColumns.adjoint(), synthesis, gradients, complex.edges, bandGains);
      const ns::PairFlow got = ns::pairFlow(weightedColumns, bandReadouts, 1, complex.edges, &sharpWeights);
      const double gap = std::max({relative(got.power, want.power), relative(got.pairImaginary, want.edgeCurrent), relative(got.vertexCurrent, want.vertexCurrent)});
      std::printf("  pairFlow vs bandPairFlow (%s): %.2e\n", draws < coordinateCount ? "factored" : "dense", gap);
      EXPECT(gap < 1e-10, std::string("pairFlow = bandPairFlow (one component), ") + (draws < coordinateCount ? "factored order" : "dense order"));
    }
  }

  // ── pairFlow: the source readout (three components) = readoutPairFlow ──
  {
    std::vector<Eigen::MatrixXd> readouts;
    for (int band = 0; band < 2; ++band) readouts.push_back(randn(3 * vertexCount, coordinateCount));
    for (const Eigen::Index draws : {Eigen::Index(5), Eigen::Index(2 * coordinateCount)}) {
      const Eigen::MatrixXcd weightedColumns = crandn(coordinateCount, draws);
      const ns::BandPairFlow want = ns::readoutPairFlow(weightedColumns, readouts, 3, complex.edges, sharpWeights);
      const ns::PairFlow got = ns::pairFlow(weightedColumns, readouts, 3, complex.edges, &sharpWeights);
      const double gap = std::max({relative(got.power, want.power), relative(got.pairImaginary, want.edgeCurrent), relative(got.vertexCurrent, want.vertexCurrent)});
      EXPECT(gap < 1e-10, "pairFlow = readoutPairFlow (three components, " << (2 * draws < coordinateCount ? "factored" : "dense") << " order; gap " << gap << ")");
    }
  }

  // ── pairFlow: tiles (every pair i < j, no sharp) = the conjugate of the tile cross-spectrum sums ──
  {
    const Eigen::Index tiles = 6, bins = 7;
    const Eigen::MatrixXd tileOperator = randn(3 * tiles, coordinateCount);                  // tile-major: rows 3 t + component
    const Eigen::MatrixXcd coordinates = crandn(coordinateCount, bins);
    Eigen::MatrixXd gains(1, bins);
    for (Eigen::Index k = 0; k < bins; ++k) gains(0, k) = 0.2 + 0.1 * static_cast<double>(k);
    std::vector<Eigen::MatrixXcd> sums;
    ns::accumulateTileCrossSpectra(tileOperator * coordinates, gains, sums);
    std::vector<std::int32_t> tilePairs;
    for (std::int32_t i = 0; i < tiles; ++i) for (std::int32_t j = i + 1; j < tiles; ++j) { tilePairs.push_back(i); tilePairs.push_back(j); }
    const Eigen::MatrixXcd weightedColumns = coordinates * gains.row(0).transpose().asDiagonal();
    const ns::PairFlow got = ns::pairFlow(weightedColumns, {ns::componentMajor(tileOperator, 3)}, 3, tilePairs, nullptr);
    const Eigen::MatrixXcd& crossSpectrum = sums[0];
    Eigen::VectorXd power(tiles), pairReal(tilePairs.size() / 2), pairImaginary(tilePairs.size() / 2);
    for (Eigen::Index i = 0; i < tiles; ++i) power(i) = crossSpectrum(i, i).real();
    for (std::size_t pair = 0; pair < tilePairs.size() / 2; ++pair) {
      const ComplexValue s = crossSpectrum(tilePairs[2 * pair], tilePairs[2 * pair + 1]);
      pairReal(static_cast<Eigen::Index>(pair)) = s.real(); pairImaginary(static_cast<Eigen::Index>(pair)) = -s.imag();
    }
    const double gap = std::max({relative(got.power.col(0), power), relative(got.pairReal.col(0), pairReal), relative(got.pairImaginary.col(0), pairImaginary)});
    EXPECT(gap < 1e-10 && got.vertexCurrent.size() == 0, "pairFlow on tiles = conj(S_ij), S_ii of accumulateTileCrossSpectra; no vertex current without sharp (gap " << gap << ")");
  }

  // ── OneFormRemainderSplit ──
  {
    const ns::OneFormRemainderSplit splitter(complex);
    Eigen::VectorXd potential = randn(vertexCount, 1);
    potential = potential.array() - potential.dot(complex.star0) / complex.star0.sum();
    const ns::OneFormRemainderSplit::Parts exactParts = splitter.split(complex.d0 * potential);
    EXPECT(relative(exactParts.potential, potential) < 1e-6 && exactParts.stream.cwiseAbs().maxCoeff() < 1e-6 * potential.cwiseAbs().maxCoeff(), "an exact 1-form: its potential back, no stream");
    Eigen::VectorXd starredStream = randn(complex.nF, 1);
    starredStream = starredStream.array() - starredStream.mean();                // orthogonal to the face Laplacian's constant
    const Eigen::VectorXd twoFormPotential = starredStream.cwiseQuotient(complex.star2);
    const Eigen::VectorXd coexact = complex.star1.cwiseInverse().cwiseProduct(complex.d1.transpose() * starredStream);
    const ns::OneFormRemainderSplit::Parts coexactParts = splitter.split(coexact);
    Eigen::VectorXd streamWant = ns::vertexDensity(complex, twoFormPotential);
    streamWant = streamWant.array() - streamWant.dot(complex.star0) / complex.star0.sum();
    EXPECT(relative(coexactParts.stream, streamWant) < 1e-6 && coexactParts.potential.cwiseAbs().maxCoeff() < 1e-6 * coexact.cwiseAbs().maxCoeff(), "a coexact 1-form: its stream back, no potential");
  }

  std::cout << (failures ? "FAILED " : "OK ") << failures << "\n";
  return failures ? 1 : 0;
}
