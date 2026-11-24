// nxr::signal flow_readout on synthetic data (an icosphere's complex and its Hodge-Dirac modes, random 1-forms):
//   flowSplit: the coefficients on the stored edge rows, the potential and the stream equal projectOneForm / exactPotential / coexactStream column by column;
//     with the remainder split, the remainder's parts equal OneFormRemainderSplit on w minus its synthesis, and the energies are the ⋆1 norms;
//   splitPhases: a batch of 1-forms equals splitPhase on each;
//   pairFlow: with one component and the readouts Psi diag(h_q) it equals bandPairFlow (dense and factored orders); with three components readoutPairFlow; on tiles
//     (all pairs i < j, no sharp) the conjugate of the tile cross-spectrum sums S_ij = Σ z_i conj(z_j) of accumulateTileCrossSpectra;
//   timeCellPairFlow: equals readoutTimeCellFlow (three components), timeCellFlow (one component, Psi diag(h)) and tileTimeCells (conjugated); both orders of the products
//     (the readout applied before or after the inverse transform) equal a direct inverse transform written out here; a chunk cut in two, its cells anchored at the line's
//     sample 0 and a cell straddling the cut, adds back to the uncut chunk;
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

  // ── timeCellPairFlow ──
  {
    const std::int64_t transformSamples = 64, firstBin = 3, bins = 9, ownFirst = 16, ownSamples = 32, cellSamples = 8;
    const Eigen::MatrixXcd spectrum = crandn(coordinateCount, bins);
    Eigen::VectorXd bandGain(coordinateCount);
    for (Eigen::Index j = 0; j < coordinateCount; ++j) bandGain(j) = 1.0 / (1.0 + static_cast<double>(j));
    auto gap = [&](const ns::TimeCellPairFlow& got, const Eigen::MatrixXd& power, const Eigen::MatrixXd& pairImaginary, const Eigen::MatrixXd& vertexCurrent) {
      double worst = std::max(relative(got.power, power), relative(got.pairImaginary, pairImaginary));
      if (vertexCurrent.size()) worst = std::max(worst, relative(got.vertexCurrent, vertexCurrent));
      return worst;
    };
    // the sensor version: one component, the synthesis with its band gain
    {
      const Eigen::MatrixXd gradients = ns::sharp(complex, Eigen::MatrixXd(complex.d0 * synthesis));
      const ns::CellFlow want = ns::timeCellFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, cellSamples, synthesis, gradients, complex.edges, bandGain);
      const ns::TimeCellPairFlow got = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, synthesis * bandGain.asDiagonal(), 1, complex.edges, &sharpWeights);
      const double worst = gap(got, want.power, want.edgeCurrent, want.vertexCurrent);
      EXPECT(worst < 1e-10 && got.firstCell == 0 && got.samples == std::vector<std::int64_t>(4, cellSamples), "timeCellPairFlow = timeCellFlow (one component; gap " << worst << ")");
    }
    // the source version: three components
    const Eigen::MatrixXd sourceReadout = randn(3 * vertexCount, coordinateCount);
    {
      const ns::CellFlow want = ns::readoutTimeCellFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, cellSamples, sourceReadout, 3, complex.edges, sharpWeights);
      const ns::TimeCellPairFlow got = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights);
      const double worst = gap(got, want.power, want.edgeCurrent, want.vertexCurrent);
      EXPECT(worst < 1e-10, "timeCellPairFlow = readoutTimeCellFlow (three components; gap " << worst << ")");
    }
    // the tile version: every pair i < j, no sharp, cells anchored at the line with a straddling cell (lineFirst 5)
    {
      const Eigen::Index tiles = 5;
      const Eigen::MatrixXd tileOperator = randn(3 * tiles, coordinateCount);              // tile-major
      std::vector<std::int32_t> tilePairs;
      for (std::int32_t i = 0; i < tiles; ++i) for (std::int32_t j = i + 1; j < tiles; ++j) { tilePairs.push_back(i); tilePairs.push_back(j); }
      const ns::TileCells want = ns::tileTimeCells(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 5, cellSamples, tileOperator);
      const ns::TimeCellPairFlow got = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 5, cellSamples, ns::componentMajor(tileOperator, 3), 3, tilePairs, nullptr);
      const double worst = std::max({relative(got.power, want.power), relative(got.pairReal, want.pairReal), relative(got.pairImaginary, Eigen::MatrixXd(-want.pairImaginary))});
      EXPECT(worst < 1e-10 && got.firstCell == want.firstCell && got.samples == want.samples && got.vertexCurrent.size() == 0,
             "timeCellPairFlow = tileTimeCells conjugated, the same cells and sample counts (gap " << worst << ")");
    }
    // both orders of the products against a direct inverse transform: z(t) = (1/N) Σ_k X(:, k) e^{2 pi i (firstBin + k) t / N}, then z = U y per sample
    auto direct = [&](const Eigen::MatrixXd& readout, int components, const std::vector<std::int32_t>& pairs) {
      const Eigen::Index vertices = readout.rows() / components;
      ns::TimeCellPairFlow want;
      const std::int64_t cells = ownSamples / cellSamples;
      want.power = Eigen::MatrixXd::Zero(vertices, cells);
      want.pairReal = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(pairs.size() / 2), cells);
      want.pairImaginary = want.pairReal;
      for (std::int64_t sample = 0; sample < ownSamples; ++sample) {
        Eigen::VectorXcd coordinatesInTime = Eigen::VectorXcd::Zero(spectrum.rows());
        for (Eigen::Index k = 0; k < bins; ++k)
          coordinatesInTime += spectrum.col(k) * std::polar(1.0 / static_cast<double>(transformSamples), 2 * M_PI * static_cast<double>((firstBin + k) * (ownFirst + sample)) / static_cast<double>(transformSamples));
        const Eigen::VectorXcd z = readout.cast<ComplexValue>() * coordinatesInTime;
        const std::int64_t cell = sample / cellSamples;
        for (int c = 0; c < components; ++c) {
          for (Eigen::Index v = 0; v < vertices; ++v) want.power(v, cell) += std::norm(z(c * vertices + v));
          for (std::size_t pair = 0; pair < pairs.size() / 2; ++pair) {
            const ComplexValue product = std::conj(z(c * vertices + pairs[2 * pair])) * z(c * vertices + pairs[2 * pair + 1]);
            want.pairReal(static_cast<Eigen::Index>(pair), cell) += product.real();
            want.pairImaginary(static_cast<Eigen::Index>(pair), cell) += product.imag();
          }
        }
      }
      return want;
    };
    {
      const std::vector<std::int32_t> fewPairs = {0, 1, 1, 2, 0, 3};
      const Eigen::MatrixXd smallReadout = randn(3 * 4, coordinateCount);                   // fewer rows than columns: the readout first
      const ns::TimeCellPairFlow small = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, smallReadout, 3, fewPairs, nullptr);
      const ns::TimeCellPairFlow smallWant = direct(smallReadout, 3, fewPairs);
      const ns::TimeCellPairFlow large = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, sourceReadout, 3, fewPairs, nullptr);
      const ns::TimeCellPairFlow largeWant = direct(sourceReadout, 3, fewPairs);
      const double smallGap = std::max({relative(small.power, smallWant.power), relative(small.pairReal, smallWant.pairReal), relative(small.pairImaginary, smallWant.pairImaginary)});
      const double largeGap = std::max({relative(large.power, largeWant.power), relative(large.pairReal, largeWant.pairReal), relative(large.pairImaginary, largeWant.pairImaginary)});
      EXPECT(smallGap < 1e-10 && largeGap < 1e-10, "timeCellPairFlow: the readout applied first (" << smallGap << ") or after the inverse transform (" << largeGap << ") = the direct sums");
    }
    // a chunk cut in two at a sample that is not a cell boundary: the parts, added by the line's cell index, are the uncut chunk's cells
    {
      const std::int64_t lineFirst = 3, cut = 13;
      const ns::TimeCellPairFlow whole = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, lineFirst, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights);
      const ns::TimeCellPairFlow first = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, cut, lineFirst, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights);
      const ns::TimeCellPairFlow second = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst + cut, ownSamples - cut, lineFirst + cut, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights);
      Eigen::MatrixXd power = Eigen::MatrixXd::Zero(whole.power.rows(), whole.power.cols()), pairImaginary = Eigen::MatrixXd::Zero(whole.pairImaginary.rows(), whole.pairImaginary.cols());
      Eigen::MatrixXd vertexCurrent = Eigen::MatrixXd::Zero(whole.vertexCurrent.rows(), whole.vertexCurrent.cols());
      std::vector<std::int64_t> samples(whole.samples.size(), 0);
      for (const ns::TimeCellPairFlow* part : {&first, &second})
        for (Eigen::Index cell = 0; cell < part->power.cols(); ++cell) {
          const Eigen::Index at = part->firstCell + cell - whole.firstCell;
          power.col(at) += part->power.col(cell); pairImaginary.col(at) += part->pairImaginary.col(cell); vertexCurrent.col(at) += part->vertexCurrent.col(cell);
          samples[static_cast<std::size_t>(at)] += part->samples[static_cast<std::size_t>(cell)];
        }
      const double worst = std::max({relative(power, whole.power), relative(pairImaginary, whole.pairImaginary), relative(vertexCurrent, whole.vertexCurrent)});
      EXPECT(worst < 1e-10 && samples == whole.samples && whole.samples.front() == cellSamples - lineFirst, "a chunk cut in two: its parts add back to the uncut chunk's cells (gap " << worst << ")");
    }
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
