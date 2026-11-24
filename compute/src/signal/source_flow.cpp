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

}  // namespace nxr::signal
