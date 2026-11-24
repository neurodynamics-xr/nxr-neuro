// nxr::signal — the flow readout (see the header).
#include "nxr/signal/flow_readout.h"

#include <stdexcept>
#include <string>

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& message) { throw std::invalid_argument(message); }
}  // namespace

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
