// nxr::signal flow_readout on synthetic data (an icosphere's complex and its Hodge-Dirac modes, random 1-forms):
//   flowSplit: the coefficients on the stored edge rows, the potential and the stream equal projectOneForm / exactPotential / coexactStream column by column;
//     with the remainder split, the remainder's parts equal OneFormRemainderSplit on w minus its synthesis, and the energies are the ⋆1 norms;
//   splitPhases: a batch of 1-forms equals splitPhase on each;
//   pairFlow: equals its definition written out (U_p conj(C) U_qᵀ with C = Y Yᴴ; the vertex phase current by the gradient sharp(d0 U_q), the golden's) for one component
//     (the sensor readouts), three components, dense and factored orders; on tiles it is the conjugate of S_ij = Σ_c Z_c diag(g²) Z_cᴴ;
//   timeCellPairFlow: equals its definition written out (a direct inverse transform, the vertex current by each sample's gradient) for one and three components and on tiles,
//     the cells anchored at the line's sample 0 with their sample counts; both orders of the products agree; a chunk cut in two adds back to the uncut chunk;
//   takeWholeCells: a chunk streamed in three parts gives the uncut chunk's whole cells, in order; the part cells at the line's ends are never returned;
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

  // ── pairFlow against its definition written out: per band pair and component Z = U_p conj(C) U_qᵀ with C = Y Yᴴ; power Re Z(v, v); pair sums Z(tail, tip);
  //    the vertex phase current Im(U_p,v conj(C) (grad U_q)_vᵀ) with the gradient sharp(d0 U_q) (the golden's definition, the gradient formed here) ──
  auto directPairFlow = [&](const Eigen::MatrixXcd& weightedColumns, const std::vector<Eigen::MatrixXd>& readouts, int components, const std::vector<std::int32_t>& pairs, bool withVertexCurrent) {
    const Eigen::Index bands = static_cast<Eigen::Index>(readouts.size()), vertices = readouts.front().rows() / components, pairCount = static_cast<Eigen::Index>(pairs.size() / 2);
    const Eigen::MatrixXcd conjugateCrossSpectrum = (weightedColumns * weightedColumns.adjoint()).conjugate();
    ns::PairFlow want;
    want.power = Eigen::MatrixXd::Zero(vertices, bands * bands);
    want.pairReal = Eigen::MatrixXd::Zero(pairCount, bands * bands);
    want.pairImaginary = want.pairReal;
    if (withVertexCurrent) want.vertexCurrent = Eigen::MatrixXd::Zero(3 * vertices, bands * bands);
    for (Eigen::Index p = 0; p < bands; ++p)
      for (Eigen::Index q = 0; q < bands; ++q)
        for (int c = 0; c < components; ++c) {
          const Eigen::MatrixXd left = readouts[static_cast<std::size_t>(p)].middleRows(c * vertices, vertices), right = readouts[static_cast<std::size_t>(q)].middleRows(c * vertices, vertices);
          const Eigen::MatrixXcd reading = left.cast<ComplexValue>() * conjugateCrossSpectrum * right.transpose().cast<ComplexValue>();
          const Eigen::Index pair = p * bands + q;
          for (Eigen::Index v = 0; v < vertices; ++v) want.power(v, pair) += reading(v, v).real();
          for (Eigen::Index e = 0; e < pairCount; ++e) {
            const ComplexValue z = reading(pairs[static_cast<std::size_t>(2 * e)], pairs[static_cast<std::size_t>(2 * e + 1)]);
            want.pairReal(e, pair) += z.real(); want.pairImaginary(e, pair) += z.imag();
          }
          if (!withVertexCurrent) continue;
          const Eigen::MatrixXd gradient = ns::sharp(complex, Eigen::MatrixXd(complex.d0 * right));   // [3V x columns]
          const Eigen::MatrixXcd withGradient = left.cast<ComplexValue>() * conjugateCrossSpectrum * gradient.transpose().cast<ComplexValue>();   // [V x 3V]
          for (Eigen::Index v = 0; v < vertices; ++v) for (int k = 0; k < 3; ++k) want.vertexCurrent(3 * v + k, pair) += withGradient(v, 3 * v + k).imag();
        }
    return want;
  };
  auto pairGap = [&](const ns::PairFlow& got, const ns::PairFlow& want) {
    double worst = std::max({relative(got.power, want.power), relative(got.pairReal, want.pairReal), relative(got.pairImaginary, want.pairImaginary)});
    if (want.vertexCurrent.size()) worst = std::max(worst, relative(got.vertexCurrent, want.vertexCurrent));
    return worst;
  };
  {
    // the sensor readout: one component, the synthesis per band Psi diag(h_q), the mesh's edges with sharp
    for (const Eigen::Index draws : {Eigen::Index(4), Eigen::Index(3 * coordinateCount)}) {             // factored (few draws) and dense
      const Eigen::MatrixXcd weightedColumns = crandn(coordinateCount, draws);
      const double worst = pairGap(ns::pairFlow(weightedColumns, bandReadouts, 1, complex.edges, &sharpWeights), directPairFlow(weightedColumns, bandReadouts, 1, complex.edges, true));
      EXPECT(worst < 1e-10, "pairFlow, one component (the sensor readouts), " << (draws < coordinateCount ? "factored" : "dense") << " order = its definition, the vertex current by the gradient (gap " << worst << ")");
    }
    // a source readout: three components
    std::vector<Eigen::MatrixXd> readouts;
    for (int band = 0; band < 2; ++band) readouts.push_back(randn(3 * vertexCount, coordinateCount));
    for (const Eigen::Index draws : {Eigen::Index(5), Eigen::Index(2 * coordinateCount)}) {
      const Eigen::MatrixXcd weightedColumns = crandn(coordinateCount, draws);
      const double worst = pairGap(ns::pairFlow(weightedColumns, readouts, 3, complex.edges, &sharpWeights), directPairFlow(weightedColumns, readouts, 3, complex.edges, true));
      EXPECT(worst < 1e-10, "pairFlow, three components, " << (2 * draws < coordinateCount ? "factored" : "dense") << " order = its definition (gap " << worst << ")");
    }
    // tiles: every pair i < j, no sharp; the stored S_ij = Σ_c Z_c diag(g²) Z_cᴴ (written out) is the conjugate of the pair sums
    const Eigen::Index tiles = 6, bins = 7;
    const Eigen::MatrixXd tileOperator = randn(3 * tiles, coordinateCount);                  // tile-major: rows 3 t + component
    const Eigen::MatrixXcd coordinates = crandn(coordinateCount, bins);
    Eigen::VectorXd gains(bins);
    for (Eigen::Index k = 0; k < bins; ++k) gains(k) = 0.2 + 0.1 * static_cast<double>(k);
    Eigen::MatrixXcd tileCrossSpectrum = Eigen::MatrixXcd::Zero(tiles, tiles);
    for (int c = 0; c < 3; ++c) {
      Eigen::MatrixXcd componentCoefficients(tiles, bins);
      for (Eigen::Index t = 0; t < tiles; ++t) componentCoefficients.row(t) = (tileOperator.row(3 * t + c).cast<ComplexValue>() * coordinates).cwiseProduct(gains.transpose().cast<ComplexValue>());
      tileCrossSpectrum += componentCoefficients * componentCoefficients.adjoint();
    }
    std::vector<std::int32_t> tilePairs;
    for (std::int32_t i = 0; i < tiles; ++i) for (std::int32_t j = i + 1; j < tiles; ++j) { tilePairs.push_back(i); tilePairs.push_back(j); }
    const ns::PairFlow got = ns::pairFlow(coordinates * gains.asDiagonal(), {ns::componentMajor(tileOperator, 3)}, 3, tilePairs, nullptr);
    double worst = 0;
    for (Eigen::Index i = 0; i < tiles; ++i) worst = std::max(worst, std::abs(got.power(i, 0) - tileCrossSpectrum(i, i).real()));
    for (std::size_t pair = 0; pair < tilePairs.size() / 2; ++pair) {
      const ComplexValue stored = tileCrossSpectrum(tilePairs[2 * pair], tilePairs[2 * pair + 1]);
      worst = std::max({worst, std::abs(got.pairReal(static_cast<Eigen::Index>(pair), 0) - stored.real()), std::abs(got.pairImaginary(static_cast<Eigen::Index>(pair), 0) + stored.imag())});
    }
    worst /= tileCrossSpectrum.cwiseAbs().maxCoeff();
    EXPECT(worst < 1e-10 && got.vertexCurrent.size() == 0, "pairFlow on tiles = conj(S_ij), S_ii of the tiles' cross-spectrum; no vertex current without sharp (gap " << worst << ")");
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
    // the definition written out: z(t) = (1/N) Σ_k X(:, k) e^{2 pi i (firstBin + k) t / N}, z = U y per sample, sums over the cells of the line (anchored at its sample 0);
    // the vertex phase current Σ_t Im(conj(z_v) (grad z)_v) with the gradient sharp(d0 z) of each sample (the golden's definition)
    auto direct = [&](std::int64_t lineFirst, const Eigen::MatrixXd& readout, int components, const std::vector<std::int32_t>& pairs, bool withVertexCurrent) {
      const Eigen::Index vertices = readout.rows() / components;
      ns::TimeCellPairFlow want;
      want.firstCell = lineFirst / cellSamples;
      const std::int64_t cells = (lineFirst + ownSamples - 1) / cellSamples - want.firstCell + 1;
      want.power = Eigen::MatrixXd::Zero(vertices, cells);
      want.pairReal = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(pairs.size() / 2), cells);
      want.pairImaginary = want.pairReal;
      if (withVertexCurrent) want.vertexCurrent = Eigen::MatrixXd::Zero(3 * vertices, cells);
      want.samples.assign(static_cast<std::size_t>(cells), 0);
      for (std::int64_t sample = 0; sample < ownSamples; ++sample) {
        Eigen::VectorXcd coordinatesInTime = Eigen::VectorXcd::Zero(spectrum.rows());
        for (Eigen::Index k = 0; k < bins; ++k)
          coordinatesInTime += spectrum.col(k) * std::polar(1.0 / static_cast<double>(transformSamples), 2 * M_PI * static_cast<double>((firstBin + k) * (ownFirst + sample)) / static_cast<double>(transformSamples));
        const Eigen::VectorXcd z = readout.cast<ComplexValue>() * coordinatesInTime;
        const std::int64_t cell = (lineFirst + sample) / cellSamples - want.firstCell;
        ++want.samples[static_cast<std::size_t>(cell)];
        for (int c = 0; c < components; ++c) {
          const Eigen::VectorXcd zc = z.segment(c * vertices, vertices);
          for (Eigen::Index v = 0; v < vertices; ++v) want.power(v, cell) += std::norm(zc(v));
          for (std::size_t pair = 0; pair < pairs.size() / 2; ++pair) {
            const ComplexValue product = std::conj(zc(pairs[2 * pair])) * zc(pairs[2 * pair + 1]);
            want.pairReal(static_cast<Eigen::Index>(pair), cell) += product.real();
            want.pairImaginary(static_cast<Eigen::Index>(pair), cell) += product.imag();
          }
          if (!withVertexCurrent) continue;
          Eigen::MatrixXd parts(vertices, 2);
          parts << zc.real(), zc.imag();
          const Eigen::MatrixXd gradient = ns::sharp(complex, Eigen::MatrixXd(complex.d0 * parts));   // [3V x 2]: the gradient of the real and imaginary parts
          for (Eigen::Index v = 0; v < vertices; ++v)
            for (int k = 0; k < 3; ++k) want.vertexCurrent(3 * v + k, cell) += (std::conj(zc(v)) * ComplexValue(gradient(3 * v + k, 0), gradient(3 * v + k, 1))).imag();
        }
      }
      return want;
    };
    auto cellGap = [&](const ns::TimeCellPairFlow& got, const ns::TimeCellPairFlow& want) {
      double worst = std::max({relative(got.power, want.power), relative(got.pairReal, want.pairReal), relative(got.pairImaginary, want.pairImaginary)});
      if (want.vertexCurrent.size()) worst = std::max(worst, relative(got.vertexCurrent, want.vertexCurrent));
      return (got.firstCell == want.firstCell && got.samples == want.samples) ? worst : 1.0;
    };
    // the sensor readout: one component, the synthesis with its band gain, the mesh's edges with sharp
    {
      const Eigen::MatrixXd readout = synthesis * bandGain.asDiagonal();
      const double worst = cellGap(ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, readout, 1, complex.edges, &sharpWeights),
                                   direct(0, readout, 1, complex.edges, true));
      EXPECT(worst < 1e-10, "timeCellPairFlow, one component = its definition, the vertex current by the gradient (gap " << worst << ")");
    }
    // a source readout: three components
    const Eigen::MatrixXd sourceReadout = randn(3 * vertexCount, coordinateCount);
    {
      const double worst = cellGap(ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights),
                                   direct(0, sourceReadout, 3, complex.edges, true));
      EXPECT(worst < 1e-10, "timeCellPairFlow, three components = its definition (gap " << worst << ")");
    }
    // tiles: every pair i < j, no sharp, the cells anchored at the line with cut cells at both ends (lineFirst 5)
    {
      const Eigen::Index tiles = 5;
      const Eigen::MatrixXd tileReadout = ns::componentMajor(randn(3 * tiles, coordinateCount), 3);
      std::vector<std::int32_t> tilePairs;
      for (std::int32_t i = 0; i < tiles; ++i) for (std::int32_t j = i + 1; j < tiles; ++j) { tilePairs.push_back(i); tilePairs.push_back(j); }
      const ns::TimeCellPairFlow got = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 5, cellSamples, tileReadout, 3, tilePairs, nullptr);
      const double worst = cellGap(got, direct(5, tileReadout, 3, tilePairs, false));
      EXPECT(worst < 1e-10 && got.samples.front() == 3 && got.samples.back() == 5 && got.vertexCurrent.size() == 0,
             "timeCellPairFlow on tiles = its definition, the cut cells at the line's anchoring with their sample counts (gap " << worst << ")");
    }
    // both orders of the products (the readout applied before or after the inverse transform) against the definition
    {
      const std::vector<std::int32_t> fewPairs = {0, 1, 1, 2, 0, 3};
      const Eigen::MatrixXd smallReadout = randn(3 * 4, coordinateCount);                   // fewer rows than columns: the readout first
      const ns::TimeCellPairFlow small = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, smallReadout, 3, fewPairs, nullptr);
      const ns::TimeCellPairFlow smallWant = direct(0, smallReadout, 3, fewPairs, false);
      const ns::TimeCellPairFlow large = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownSamples, 0, cellSamples, sourceReadout, 3, fewPairs, nullptr);
      const ns::TimeCellPairFlow largeWant = direct(0, sourceReadout, 3, fewPairs, false);
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
    // streamed in three parts through takeWholeCells: the uncut chunk's whole cells (the first, cut by lineFirst 3, and the last part cell never returned)
    {
      const std::int64_t lineFirst = 3;
      const std::int64_t ownLength = 30;                                                     // line samples [3, 33): cells 0 (5 samples), 1..3 whole, 4 (1 sample)
      const ns::TimeCellPairFlow whole = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst, ownLength, lineFirst, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights);
      ns::TimeCellPairFlow carried;
      std::vector<ns::TimeCellPairFlow> taken;
      std::int64_t start = 0;
      for (const std::int64_t length : {std::int64_t(9), std::int64_t(10), std::int64_t(11)}) {
        const ns::TimeCellPairFlow part = ns::timeCellPairFlow(spectrum, firstBin, transformSamples, ownFirst + start, length, lineFirst + start, cellSamples, sourceReadout, 3, complex.edges, &sharpWeights);
        taken.push_back(ns::takeWholeCells(carried, part, cellSamples));
        start += length;
      }
      Eigen::MatrixXd power(whole.power.rows(), 0), vertexCurrent(whole.vertexCurrent.rows(), 0);
      std::int64_t firstTaken = -1;
      for (const auto& t : taken) {
        if (t.power.cols() == 0) continue;
        if (firstTaken < 0) firstTaken = t.firstCell;
        power.conservativeResize(Eigen::NoChange, power.cols() + t.power.cols()); power.rightCols(t.power.cols()) = t.power;
        vertexCurrent.conservativeResize(Eigen::NoChange, vertexCurrent.cols() + t.vertexCurrent.cols()); vertexCurrent.rightCols(t.vertexCurrent.cols()) = t.vertexCurrent;
      }
      const double worst = power.cols() == 3 ? std::max(relative(power, whole.power.middleCols(1, 3)), relative(vertexCurrent, whole.vertexCurrent.middleCols(1, 3))) : 1.0;
      EXPECT(firstTaken == 1 && power.cols() == 3 && worst < 1e-10 && carried.samples == std::vector<std::int64_t>{1},
             "takeWholeCells: three parts give the uncut chunk's whole cells 1..3; the end cells never returned (gap " << worst << ")");
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
