// nxr::signal — the flow readout (see the header).
#include "nxr/signal/flow_readout.h"

#include <complex>
#include <stdexcept>
#include <string>

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
using ComplexValue = std::complex<double>;
}  // namespace

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
