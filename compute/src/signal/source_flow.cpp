// nxr::signal — the flow of a weighted cross-spectrum read through a linear readout onto a surface (see the header).
#include "nxr/signal/source_flow.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
using ComplexValue = std::complex<double>;
using RowMajorComplex = Eigen::Matrix<ComplexValue, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

Eigen::MatrixXd selectVertexRows(const Eigen::MatrixXd& perVertexRows, const std::vector<std::int64_t>& sourceVertex, int components) {
  if (components < 1 || perVertexRows.rows() % components != 0) bad("selectVertexRows: the rows are not whole vertices of " + std::to_string(components) + " components");
  const std::int64_t sourceVertices = perVertexRows.rows() / components;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(sourceVertex.size()) * components, perVertexRows.cols());
  for (std::size_t target = 0; target < sourceVertex.size(); ++target) {
    const std::int64_t source = sourceVertex[target];
    if (source < 0) continue;                                                              // no source vertex (the glue's bridge): zero rows, never a blend
    if (source >= sourceVertices) bad("selectVertexRows: a source vertex " + std::to_string(source) + " outside the " + std::to_string(sourceVertices) + " vertices");
    out.middleRows(static_cast<Eigen::Index>(target) * components, components) = perVertexRows.middleRows(source * components, components);
  }
  return out;
}

Eigen::MatrixXd synthesisFromAnalysis(const Eigen::MatrixXd& analysisRows, const Eigen::VectorXd& vertexMass) {
  if (analysisRows.cols() != vertexMass.size()) bad("synthesisFromAnalysis: the analysis rows read " + std::to_string(analysisRows.cols()) + " vertices, the mass has " + std::to_string(vertexMass.size()));
  if (!(vertexMass.minCoeff() > 0)) bad("synthesisFromAnalysis: a vertex mass is not positive");
  return vertexMass.cwiseInverse().asDiagonal() * analysisRows.transpose();
}

Eigen::MatrixXd componentMajor(const Eigen::MatrixXd& kernelInColumns, int components) {
  if (components < 1 || kernelInColumns.rows() % components != 0) bad("componentMajor: the rows are not whole vertices");
  const Eigen::Index vertices = kernelInColumns.rows() / components;
  Eigen::MatrixXd out(kernelInColumns.rows(), kernelInColumns.cols());
  for (Eigen::Index v = 0; v < vertices; ++v)
    for (int c = 0; c < components; ++c) out.row(c * vertices + v) = kernelInColumns.row(components * v + c);
  return out;
}

std::vector<Eigen::MatrixXd> bandReadouts(const Eigen::MatrixXd& modeOperator, const Eigen::MatrixXd& modeSynthesis, const Eigen::MatrixXd& bandGains,
                                          const Eigen::MatrixXd& kernelInColumns, int components) {
  const Eigen::Index modes = modeSynthesis.cols(), vertices = modeSynthesis.rows(), bands = bandGains.rows(), columns = modeOperator.cols();
  if (modeOperator.rows() != components * modes || bandGains.cols() != modes || kernelInColumns.rows() != components * vertices || kernelInColumns.cols() != columns || bands < 1)
    bad("bandReadouts: a mode operator [components K x columns], a mode synthesis [V x K], band gains [bands x K] and a kernel [components V x columns] of one K, V and columns");
  std::vector<Eigen::MatrixXd> out;
  Eigen::MatrixXd complement = componentMajor(kernelInColumns, components);
  for (Eigen::Index q = 0; q + 1 < bands; ++q) {
    Eigen::MatrixXd readout(components * vertices, columns);
    // compose first: the band's synthesis [V x K] on the operator's K rows of each component (never the vertex x vertex filter)
    const Eigen::MatrixXd bandSynthesis = modeSynthesis * bandGains.row(q).transpose().asDiagonal();
    for (int c = 0; c < components; ++c) readout.middleRows(c * vertices, vertices) = bandSynthesis * modeOperator.middleRows(c * modes, modes);
    complement -= readout;
    out.push_back(std::move(readout));
  }
  out.push_back(std::move(complement));                                                    // the last band: the kernel minus every other band (its own gain and everything outside the modes)
  return out;
}

BandPairFlow readoutPairFlow(const Eigen::MatrixXcd& Y, const std::vector<Eigen::MatrixXd>& readouts, int components, const std::vector<std::int32_t>& edges,
                             const Eigen::SparseMatrix<double>& sharpWeights) {
  const Eigen::Index bands = static_cast<Eigen::Index>(readouts.size()), columns = Y.rows(), draws = Y.cols();
  if (bands < 1 || components < 1 || readouts.front().rows() % components != 0) bad("readoutPairFlow: at least one readout of whole vertices");
  const Eigen::Index vertices = readouts.front().rows() / components, edgeCount = static_cast<Eigen::Index>(edges.size() / 2);
  for (const auto& readout : readouts) if (readout.rows() != components * vertices || readout.cols() != columns) bad("readoutPairFlow: every readout [components V x columns], columns those of the weighted columns");
  if (sharpWeights.rows() != 3 * vertices || sharpWeights.cols() != edgeCount) bad("readoutPairFlow: the sharp weights are [3V x E]");
  for (const auto v : edges) if (v < 0 || v >= vertices) bad("readoutPairFlow: an edge's vertex is outside the readout's vertices");
  BandPairFlow out;
  out.power = Eigen::MatrixXd::Zero(vertices, bands * bands);
  out.edgeCurrent = Eigen::MatrixXd::Zero(edgeCount, bands * bands);
  out.vertexCurrent = Eigen::MatrixXd::Zero(3 * vertices, bands * bands);
  if (draws == 0) return out;
  // the cheaper order (compose first): dense conj(C) [columns x columns] once, or the factored conj(Y) (Yᵀ ·) when Y has fewer draws than columns
  const bool factored = 2 * draws < columns;
  Eigen::MatrixXd crossSpectrumReal, crossSpectrumImaginaryConjugate;                       // conj(C) = Re C - i Im C, applied to a REAL readout as two real products
  if (!factored) { const Eigen::MatrixXcd C = Y * Y.adjoint(); crossSpectrumReal = C.real(); crossSpectrumImaginaryConjugate = -C.imag(); }
  std::vector<Eigen::MatrixXd> transposed;                                                 // U_pᵀ [columns x components V]: each vertex's row a contiguous column
  for (const auto& readout : readouts) transposed.push_back(readout.transpose());
  const Eigen::SparseMatrix<double, Eigen::RowMajor> sharpByEdge = Eigen::SparseMatrix<double, Eigen::RowMajor>(sharpWeights.transpose());   // [E x 3V]: each edge's (at most 6) weights
  for (Eigen::Index q = 0; q < bands; ++q) {
    Eigen::MatrixXcd right;                                                                 // conj(C) U_qᵀ [columns x components V]
    if (factored) right = Y.conjugate() * (Y.transpose() * transposed[static_cast<std::size_t>(q)].cast<ComplexValue>());
    else {
      right.resize(columns, transposed[static_cast<std::size_t>(q)].cols());
      right.real() = crossSpectrumReal * transposed[static_cast<std::size_t>(q)];
      right.imag() = crossSpectrumImaginaryConjugate * transposed[static_cast<std::size_t>(q)];
    }
    for (Eigen::Index p = 0; p < bands; ++p) {
      const Eigen::MatrixXd& left = transposed[static_cast<std::size_t>(p)];
      const Eigen::Index pair = p * bands + q;
      auto entry = [&](int c, Eigen::Index a, Eigen::Index b) { return left.col(c * vertices + a).cast<ComplexValue>().dot(right.col(c * vertices + b)); };   // U_p,a conj(C) U_q,bᵀ (dot is linear in its 2nd argument: U real)
      Eigen::VectorXd diagonalImaginary = Eigen::VectorXd::Zero(vertices);
      for (int c = 0; c < components; ++c)
        for (Eigen::Index v = 0; v < vertices; ++v) {
          const ComplexValue z = entry(c, v, v);
          out.power(v, pair) += z.real();
          diagonalImaginary(v) += z.imag();
        }
      for (Eigen::Index e = 0; e < edgeCount; ++e) {
        const Eigen::Index tail = edges[static_cast<std::size_t>(2 * e)], tip = edges[static_cast<std::size_t>(2 * e + 1)];
        double forward = 0, backward = 0;
        for (int c = 0; c < components; ++c) { forward += entry(c, tail, tip).imag(); backward += entry(c, tip, tail).imag(); }
        out.edgeCurrent(e, pair) = forward;
        // the vertex phase current Σ_e sharp(v, e) (Im Z(v, tip_e) - Im Z(v, tail_e)): at the tail Im Z(tail, tip) - Im Z(tail, tail); at the tip Im Z(tip, tip) - Im Z(tip, tail)
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(sharpByEdge, e); it; ++it) {
          const Eigen::Index vertex = it.col() / 3;
          if (vertex == tail) out.vertexCurrent(it.col(), pair) += it.value() * (forward - diagonalImaginary(tail));
          else out.vertexCurrent(it.col(), pair) += it.value() * (diagonalImaginary(tip) - backward);
        }
      }
    }
  }
  return out;
}

CellFlow readoutTimeCellFlow(const Eigen::MatrixXcd& X, std::int64_t firstBin, std::int64_t N, std::int64_t ownFirst, std::int64_t ownSamples, std::int64_t cellSamples,
                             const Eigen::MatrixXd& readout, int components, const std::vector<std::int32_t>& edges, const Eigen::SparseMatrix<double>& sharpWeights) {
  const Eigen::Index columns = X.rows(), binCount = X.cols();
  if (components < 1 || readout.rows() % components != 0 || readout.cols() != columns) bad("readoutTimeCellFlow: a readout [components V x columns] on the spectrum's rows");
  const Eigen::Index vertices = readout.rows() / components, edgeCount = static_cast<Eigen::Index>(edges.size() / 2);
  if (sharpWeights.rows() != 3 * vertices || sharpWeights.cols() != edgeCount) bad("readoutTimeCellFlow: the sharp weights are [3V x E]");
  if (firstBin < 0 || firstBin + binCount > N) bad("readoutTimeCellFlow: the bins fall outside the transform");
  if (ownFirst < 0 || ownSamples < 1 || ownFirst + ownSamples > N) bad("readoutTimeCellFlow: the chunk's own samples fall outside the transform");
  if (cellSamples < 1) bad("readoutTimeCellFlow: a cell holds at least one sample");
  for (const auto v : edges) if (v < 0 || v >= vertices) bad("readoutTimeCellFlow: an edge's vertex is outside the readout's vertices");
  const std::int64_t cells = ownSamples / cellSamples;
  CellFlow out;
  out.edgeCurrent = Eigen::MatrixXd::Zero(edgeCount, cells);
  out.vertexCurrent = Eigen::MatrixXd::Zero(3 * vertices, cells);
  out.power = Eigen::MatrixXd::Zero(vertices, cells);
  if (cells == 0) return out;
  // the inverse transform on the columns' side (columns x N, not vertices x N), the own samples kept
  std::vector<ComplexValue> buffer(static_cast<std::size_t>(columns * N), ComplexValue(0, 0));
  for (Eigen::Index r = 0; r < columns; ++r) for (Eigen::Index k = 0; k < binCount; ++k) buffer[static_cast<std::size_t>(r * N + firstBin + k)] = X(r, k);
  c2c(buffer.data(), buffer.data(), {static_cast<std::size_t>(columns), static_cast<std::size_t>(N)}, 1, false);
  const Eigen::Map<const RowMajorComplex> inTime(buffer.data(), columns, N);
  const Eigen::MatrixXcd complexReadout = readout.cast<ComplexValue>();
  for (std::int64_t cell = 0; cell < cells; ++cell) {
    const RowMajorComplex z = complexReadout * inTime.middleCols(ownFirst + cell * cellSamples, cellSamples);   // [components V x cell samples]
    for (int c = 0; c < components; ++c) {
      for (Eigen::Index v = 0; v < vertices; ++v) out.power(v, cell) += z.row(c * vertices + v).squaredNorm();
      for (Eigen::Index e = 0; e < edgeCount; ++e) {
        const Eigen::Index tail = edges[static_cast<std::size_t>(2 * e)], tip = edges[static_cast<std::size_t>(2 * e + 1)];
        out.edgeCurrent(e, cell) += z.row(c * vertices + tail).dot(z.row(c * vertices + tip)).imag();   // Σ_t Im(conj(z_tail) z_tip) (dot conjugates its first argument)
      }
    }
  }
  // the vertex phase current Σ_t Im(conj(z_v) (grad z)_v) = Σ_e sharp(v, e) Σ_t Im(conj(z_v) (z_tip - z_tail)): the edge sum at both ends (Im(conj(z_v) z_v) = 0)
  out.vertexCurrent = sharpWeights * out.edgeCurrent;
  return out;
}

}  // namespace nxr::signal
