// nxr::signal — the flow readout (see the header).
#include "nxr/signal/flow_readout.h"

#include <algorithm>
#include <complex>
#include <stdexcept>
#include <string>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
using ComplexValue = std::complex<double>;
using RowMajorComplex = Eigen::Matrix<ComplexValue, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// rows [rows x the atom's bins] (only the bins listed) inverted to time; the own samples kept [rows x ownSamples]
RowMajorComplex inverseOwnSamples(const Eigen::MatrixXcd& rows, const std::vector<Eigen::Index>& bins, std::int64_t firstBin, std::int64_t transformSamples, std::int64_t ownFirst,
                                  std::int64_t ownSamples) {
  std::vector<ComplexValue> buffer(static_cast<std::size_t>(rows.rows() * transformSamples), ComplexValue(0, 0));
  for (Eigen::Index r = 0; r < rows.rows(); ++r)
    for (std::size_t q = 0; q < bins.size(); ++q) buffer[static_cast<std::size_t>(r * transformSamples + firstBin + bins[q])] = rows(r, static_cast<Eigen::Index>(q));
  c2c(buffer.data(), buffer.data(), {static_cast<std::size_t>(rows.rows()), static_cast<std::size_t>(transformSamples)}, 1, false);
  const Eigen::Map<const RowMajorComplex> inTime(buffer.data(), rows.rows(), transformSamples);
  return inTime.middleCols(ownFirst, ownSamples);
}
}  // namespace

TimeCellPairFlow timeCellPairFlow(const Eigen::MatrixXcd& X, std::int64_t firstBin, std::int64_t N, std::int64_t ownFirst, std::int64_t ownSamples, std::int64_t lineFirst,
                                  std::int64_t cellSamples, const Eigen::MatrixXd& readout, int components, const std::vector<std::int32_t>& pairs,
                                  const Eigen::SparseMatrix<double>* sharpWeights) {
  const Eigen::Index columns = X.rows(), binCount = X.cols();
  if (components < 1 || readout.rows() % components != 0 || readout.cols() != columns) bad("timeCellPairFlow: a readout [components V x columns] on the spectrum's rows");
  const Eigen::Index vertices = readout.rows() / components, pairCount = static_cast<Eigen::Index>(pairs.size() / 2);
  if (pairs.size() % 2 != 0) bad("timeCellPairFlow: the pairs are (tail, tip)");
  if (sharpWeights && (sharpWeights->rows() != 3 * vertices || sharpWeights->cols() != pairCount)) bad("timeCellPairFlow: the sharp weights are [3V x P] (the pairs are the complex's edges)");
  if (firstBin < 0 || firstBin + binCount > N) bad("timeCellPairFlow: the bins fall outside the transform");
  if (ownFirst < 0 || ownSamples < 1 || ownFirst + ownSamples > N) bad("timeCellPairFlow: the chunk's own samples fall outside the transform");
  if (cellSamples < 1 || lineFirst < 0) bad("timeCellPairFlow: a cell holds at least one sample; the line starts at sample 0");
  for (const auto v : pairs) if (v < 0 || v >= vertices) bad("timeCellPairFlow: a pair's vertex is outside the readout's vertices");
  TimeCellPairFlow out;
  out.firstCell = lineFirst / cellSamples;
  const std::int64_t cells = (lineFirst + ownSamples - 1) / cellSamples - out.firstCell + 1;
  out.power = Eigen::MatrixXd::Zero(vertices, cells);
  out.pairReal = Eigen::MatrixXd::Zero(pairCount, cells);
  out.pairImaginary = Eigen::MatrixXd::Zero(pairCount, cells);
  out.samples.assign(static_cast<std::size_t>(cells), 0);
  std::vector<std::int64_t> cellStart(static_cast<std::size_t>(cells));                     // each cell's first own sample (relative to ownFirst)
  for (std::int64_t c = 0; c < cells; ++c) {
    const std::int64_t lo = std::max(lineFirst, (out.firstCell + c) * cellSamples), hi = std::min(lineFirst + ownSamples, (out.firstCell + c + 1) * cellSamples);
    out.samples[static_cast<std::size_t>(c)] = hi - lo;
    cellStart[static_cast<std::size_t>(c)] = lo - lineFirst;
  }
  // the atom's bins only (its gains are in X)
  std::vector<Eigen::Index> bins;
  for (Eigen::Index k = 0; k < binCount; ++k) if (X.col(k).squaredNorm() > 0) bins.push_back(k);
  Eigen::MatrixXcd reached(columns, static_cast<Eigen::Index>(bins.size()));
  for (std::size_t q = 0; q < bins.size(); ++q) reached.col(static_cast<Eigen::Index>(q)) = X.col(bins[q]);
  // one cell's vertex signals of one component [V x the cell's samples] → its sums
  auto accumulate = [&](std::int64_t c, const RowMajorComplex& block) {
    for (Eigen::Index v = 0; v < vertices; ++v) out.power(v, c) += block.row(v).squaredNorm();
    for (Eigen::Index e = 0; e < pairCount; ++e) {
      const ComplexValue s = block.row(pairs[static_cast<std::size_t>(2 * e)]).dot(block.row(pairs[static_cast<std::size_t>(2 * e + 1)]));   // Σ_t conj(z_tail) z_tip
      out.pairReal(e, c) += s.real();
      out.pairImaginary(e, c) += s.imag();
    }
  };
  if (!bins.empty() && vertices < columns) {
    // compose first: per component the readout on the atom's bins [V x bins], then V inverse transforms (at most columns x ownSamples held)
    for (int component = 0; component < components; ++component) {
      const RowMajorComplex z = inverseOwnSamples(readout.middleRows(component * vertices, vertices).cast<ComplexValue>() * reached, bins, firstBin, N, ownFirst, ownSamples);
      for (std::int64_t c = 0; c < cells; ++c) accumulate(c, z.middleCols(cellStart[static_cast<std::size_t>(c)], out.samples[static_cast<std::size_t>(c)]));
    }
  } else if (!bins.empty()) {
    // the inverse transform on the columns' side, then per cell z = U y (never more than one cell's vertex signals held)
    const RowMajorComplex inTime = inverseOwnSamples(reached, bins, firstBin, N, ownFirst, ownSamples);
    for (int component = 0; component < components; ++component) {
      const Eigen::MatrixXcd componentReadout = readout.middleRows(component * vertices, vertices).cast<ComplexValue>();
      for (std::int64_t c = 0; c < cells; ++c)
        accumulate(c, componentReadout * inTime.middleCols(cellStart[static_cast<std::size_t>(c)], out.samples[static_cast<std::size_t>(c)]));
    }
  }
  if (sharpWeights) out.vertexCurrent = *sharpWeights * out.pairImaginary;                   // Σ_e sharp(v, e) Σ_t Im(conj(z_v) (z_tip - z_tail)): the edge sum at both ends
  return out;
}

PairFlow pairFlow(const Eigen::MatrixXcd& Y, const std::vector<Eigen::MatrixXd>& readouts, int components, const std::vector<std::int32_t>& pairs,
                  const Eigen::SparseMatrix<double>* sharpWeights) {
  const Eigen::Index bands = static_cast<Eigen::Index>(readouts.size()), columns = Y.rows(), draws = Y.cols();
  if (bands < 1 || components < 1 || readouts.front().rows() % components != 0) bad("pairFlow: at least one readout of whole vertices");
  const Eigen::Index vertices = readouts.front().rows() / components, pairCount = static_cast<Eigen::Index>(pairs.size() / 2);
  if (pairs.size() % 2 != 0) bad("pairFlow: the pairs are (tail, tip)");
  for (const auto& readout : readouts) if (readout.rows() != components * vertices || readout.cols() != columns) bad("pairFlow: every readout [components V x columns], columns those of the weighted columns");
  if (sharpWeights && (sharpWeights->rows() != 3 * vertices || sharpWeights->cols() != pairCount)) bad("pairFlow: the sharp weights are [3V x P] (the pairs are the complex's edges)");
  for (const auto v : pairs) if (v < 0 || v >= vertices) bad("pairFlow: a pair's vertex is outside the readout's vertices");
  PairFlow out;
  out.power = Eigen::MatrixXd::Zero(vertices, bands * bands);
  out.pairReal = Eigen::MatrixXd::Zero(pairCount, bands * bands);
  out.pairImaginary = Eigen::MatrixXd::Zero(pairCount, bands * bands);
  if (sharpWeights) out.vertexCurrent = Eigen::MatrixXd::Zero(3 * vertices, bands * bands);
  if (draws == 0) return out;
  // the cheaper order (compose first): dense conj(C) [columns x columns] once, or the factored conj(Y) (Yᵀ ·) when Y has fewer draws than half the columns
  const bool factored = 2 * draws < columns;
  Eigen::MatrixXd crossSpectrumReal, crossSpectrumImaginaryConjugate;                       // conj(C) = Re C - i Im C, applied to a REAL readout as two real products
  if (!factored) { const Eigen::MatrixXcd C = Y * Y.adjoint(); crossSpectrumReal = C.real(); crossSpectrumImaginaryConjugate = -C.imag(); }
  std::vector<Eigen::MatrixXd> transposed;                                                 // U_bᵀ [columns x components V]: each vertex's row a contiguous column
  for (const auto& readout : readouts) transposed.push_back(readout.transpose());
  Eigen::SparseMatrix<double, Eigen::RowMajor> sharpByEdge;                                // [P x 3V]: each edge's (at most 6) weights
  if (sharpWeights) sharpByEdge = Eigen::SparseMatrix<double, Eigen::RowMajor>(sharpWeights->transpose());
  for (Eigen::Index q = 0; q < bands; ++q) {
    Eigen::MatrixXcd right;                                                                 // conj(C) U_qᵀ [columns x components V], once per band
    if (factored) right = Y.conjugate() * (Y.transpose() * transposed[static_cast<std::size_t>(q)].cast<ComplexValue>());
    else {
      right.resize(columns, transposed[static_cast<std::size_t>(q)].cols());
      right.real() = crossSpectrumReal * transposed[static_cast<std::size_t>(q)];
      right.imag() = crossSpectrumImaginaryConjugate * transposed[static_cast<std::size_t>(q)];
    }
    for (Eigen::Index p = 0; p < bands; ++p) {
      const Eigen::MatrixXd& left = transposed[static_cast<std::size_t>(p)];
      const Eigen::Index pair = p * bands + q;
      auto entry = [&](int c, Eigen::Index a, Eigen::Index b) { return left.col(c * vertices + a).cast<ComplexValue>().dot(right.col(c * vertices + b)); };   // U_p,a conj(C) U_q,bᵀ (U real)
      Eigen::VectorXd diagonalImaginary = Eigen::VectorXd::Zero(vertices);
      for (int c = 0; c < components; ++c)
        for (Eigen::Index v = 0; v < vertices; ++v) {
          const ComplexValue z = entry(c, v, v);
          out.power(v, pair) += z.real();
          diagonalImaginary(v) += z.imag();
        }
      for (Eigen::Index e = 0; e < pairCount; ++e) {
        const Eigen::Index tail = pairs[static_cast<std::size_t>(2 * e)], tip = pairs[static_cast<std::size_t>(2 * e + 1)];
        ComplexValue forward(0, 0);
        double backward = 0;
        for (int c = 0; c < components; ++c) { forward += entry(c, tail, tip); if (sharpWeights) backward += entry(c, tip, tail).imag(); }
        out.pairReal(e, pair) = forward.real();
        out.pairImaginary(e, pair) = forward.imag();
        if (!sharpWeights) continue;
        // the vertex phase current Σ_e sharp(v, e) (Im Z(v, tip_e) - Im Z(v, tail_e)); sharp at v reads only v's own edges: at the tail Im Z(tail, tip) - Im Z(tail, tail),
        // at the tip Im Z(tip, tip) - Im Z(tip, tail)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sharpByEdge, e); it; ++it) {
          const Eigen::Index vertex = it.col() / 3;
          if (vertex == tail) out.vertexCurrent(it.col(), pair) += it.value() * (forward.imag() - diagonalImaginary(tail));
          else out.vertexCurrent(it.col(), pair) += it.value() * (diagonalImaginary(tip) - backward);
        }
      }
    }
  }
  return out;
}

TimeCellPairFlow takeWholeCells(TimeCellPairFlow& carried, const TimeCellPairFlow& part, std::int64_t cellSamples) {
  const Eigen::Index cells = part.power.cols();
  if (static_cast<Eigen::Index>(part.samples.size()) != cells) bad("takeWholeCells: a part's sample counts, one a cell");
  if (cells == 0) return TimeCellPairFlow{};                                               // nothing in this part: the carried cell waits for the next
  if (carried.power.cols() == 1 && carried.firstCell != part.firstCell)
    bad("takeWholeCells: the carried cell " + std::to_string(carried.firstCell) + " is not continued by the next chunk (its first cell " + std::to_string(part.firstCell) +
        "): a chunk is missing from the line or the chunks are out of order");
  TimeCellPairFlow merged = part;                                                          // the carried cell added to the part's first cell (the same cell)
  if (carried.power.cols() == 1) {
    merged.power.col(0) += carried.power.col(0);
    merged.pairReal.col(0) += carried.pairReal.col(0);
    merged.pairImaginary.col(0) += carried.pairImaginary.col(0);
    if (merged.vertexCurrent.size()) merged.vertexCurrent.col(0) += carried.vertexCurrent.col(0);
    merged.samples[0] += carried.samples[0];
  }
  carried = TimeCellPairFlow{};
  if (cells > 0 && merged.samples.back() < cellSamples) {                                  // the last cell, cut by this chunk's end: carried to the next chunk
    const Eigen::Index last = cells - 1;
    carried.firstCell = merged.firstCell + last;
    carried.power = merged.power.col(last);
    carried.pairReal = merged.pairReal.col(last);
    carried.pairImaginary = merged.pairImaginary.col(last);
    if (merged.vertexCurrent.size()) carried.vertexCurrent = merged.vertexCurrent.col(last);
    carried.samples = {merged.samples.back()};
  }
  std::vector<Eigen::Index> whole;
  for (Eigen::Index cell = 0; cell < cells; ++cell) if (merged.samples[static_cast<std::size_t>(cell)] == cellSamples) whole.push_back(cell);
  TimeCellPairFlow out;
  out.firstCell = whole.empty() ? 0 : merged.firstCell + whole.front();
  const Eigen::Index count = static_cast<Eigen::Index>(whole.size());
  out.power.resize(merged.power.rows(), count);
  out.pairReal.resize(merged.pairReal.rows(), count);
  out.pairImaginary.resize(merged.pairImaginary.rows(), count);
  if (merged.vertexCurrent.size()) out.vertexCurrent.resize(merged.vertexCurrent.rows(), count);
  for (Eigen::Index j = 0; j < count; ++j) {
    const Eigen::Index cell = whole[static_cast<std::size_t>(j)];
    if (cell != whole.front() + j) bad("takeWholeCells: the whole cells are not contiguous (a chunk missing from the line)");
    out.power.col(j) = merged.power.col(cell);
    out.pairReal.col(j) = merged.pairReal.col(cell);
    out.pairImaginary.col(j) = merged.pairImaginary.col(cell);
    if (merged.vertexCurrent.size()) out.vertexCurrent.col(j) = merged.vertexCurrent.col(cell);
    out.samples.push_back(cellSamples);
  }
  return out;
}

FlowSplit flowSplit(const IntrinsicComplex& c, const Modes& m, const Eigen::MatrixXd& edgeCurrent, const OneFormRemainderSplit* remainderSplit) {
  if (edgeCurrent.rows() != c.nE) bad("flowSplit: an edge current on the " + std::to_string(c.nE) + " edges");
  FlowSplit out;
  out.coefficients = projectOneForm(c, m, edgeCurrent);
  out.potential = exactPotential(m, out.coefficients);
  out.stream = coexactStream(c, m, out.coefficients);
  out.flowEnergy = edgeCurrent.cwiseAbs2().transpose() * c.star1;
  if (!remainderSplit) return out;
  const Eigen::MatrixXd remainder = edgeCurrent - synthesizeOneForm(c, m, out.coefficients);   // the flow beyond the kept modes
  out.remainderEnergy = remainder.cwiseAbs2().transpose() * c.star1;
  out.remainderPotential.resize(c.nV, edgeCurrent.cols());
  out.remainderStream.resize(c.nV, edgeCurrent.cols());
  for (Eigen::Index column = 0; column < edgeCurrent.cols(); ++column) {
    const OneFormRemainderSplit::Parts parts = remainderSplit->split(remainder.col(column));
    out.remainderPotential.col(column) = parts.potential;
    out.remainderStream.col(column) = parts.stream;
  }
  return out;
}

OneFormRemainderSplit::OneFormRemainderSplit(const IntrinsicComplex& c) : complex_(&c) {
  Eigen::SparseMatrix<double> vertexLaplacian = c.d0.transpose() * c.star1.asDiagonal() * c.d0;
  const double vertexRidge = 1e-10 * vertexLaplacian.diagonal().mean() / c.star0.mean();
  for (int v = 0; v < c.nV; ++v) vertexLaplacian.coeffRef(v, v) += vertexRidge * c.star0(v);
  vertexLaplacian_.compute(vertexLaplacian);
  if (vertexLaplacian_.info() != Eigen::Success) throw std::runtime_error("OneFormRemainderSplit: the vertex Laplacian could not be factored");
  Eigen::SparseMatrix<double> faceLaplacian = c.d1 * c.star1.cwiseInverse().asDiagonal() * c.d1.transpose();
  const double faceRidge = 1e-10 * faceLaplacian.diagonal().mean();
  for (int f = 0; f < c.nF; ++f) faceLaplacian.coeffRef(f, f) += faceRidge;
  faceLaplacian_.compute(faceLaplacian);
  if (faceLaplacian_.info() != Eigen::Success) throw std::runtime_error("OneFormRemainderSplit: the face Laplacian could not be factored");
}

OneFormRemainderSplit::Parts OneFormRemainderSplit::split(const Eigen::VectorXd& remainder) const {
  const IntrinsicComplex& c = *complex_;
  if (remainder.size() != c.nE) bad("OneFormRemainderSplit: a 1-form on the " + std::to_string(c.nE) + " edges");
  Parts parts;
  // the potential: d0ᵀ ⋆1 d0 alpha = d0ᵀ ⋆1 r (its exact part is d0 alpha)
  const Eigen::VectorXd alpha = vertexLaplacian_.solve(c.d0.transpose() * c.star1.cwiseProduct(remainder));
  parts.potential = alpha.array() - alpha.dot(c.star0) / c.star0.sum();
  // the stream: the 2-form potential beta with ⋆1⁻¹ d1ᵀ ⋆2 beta = the coexact part, so d1 ⋆1⁻¹ d1ᵀ (⋆2 beta) = d1 r (d1 d0 = 0); drawn at the vertices as coexactStream
  const Eigen::VectorXd starredBeta = faceLaplacian_.solve(c.d1 * remainder);
  const Eigen::VectorXd beta = starredBeta.cwiseQuotient(c.star2);
  const Eigen::VectorXd stream = vertexDensity(c, beta);
  parts.stream = stream.array() - stream.dot(c.star0) / c.star0.sum();
  return parts;
}

}  // namespace nxr::signal
