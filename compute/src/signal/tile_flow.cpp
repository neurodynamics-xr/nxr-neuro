#include "nxr/signal/tile_flow.h"

#include <Eigen/SVD>
#include <stdexcept>
#include <string>

#include "nxr/signal/fourier.h"

namespace nxr::signal {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }
using RowMajorComplex = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

ServedSynthesis servedSynthesis(const Eigen::MatrixXd& serving) {
  if (serving.rows() < 1 || serving.cols() < 1) bad("servedSynthesis: an empty serving operator");
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(serving, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd s = svd.singularValues();
  ServedSynthesis out;
  Eigen::VectorXd inverse = Eigen::VectorXd::Zero(s.size());
  for (Eigen::Index i = 0; i < s.size(); ++i) if (s(i) > 1e-10 * s(0)) { inverse(i) = 1.0 / s(i); ++out.rank; }
  out.synthesis = svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose();
  return out;
}

Eigen::MatrixXd tileOperator(const Eigen::MatrixXd& kernelRows, const std::vector<std::int32_t>& tileOfVertex, std::int64_t tiles, const Eigen::MatrixXd& synthesis) {
  const Eigen::Index V = static_cast<Eigen::Index>(tileOfVertex.size()), C = kernelRows.cols();
  if (kernelRows.rows() != 3 * V) bad("tileOperator: the kernel's rows are three components of each vertex of the tiles' surface (" + std::to_string(kernelRows.rows()) + " rows, " + std::to_string(V) + " vertices)");
  if (synthesis.rows() != C) bad("tileOperator: the synthesis's rows are the kernel's channels");
  if (tiles < 1) bad("tileOperator: no tile");
  Eigen::MatrixXd averaged = Eigen::MatrixXd::Zero(3 * tiles, C);
  std::vector<std::int64_t> count(static_cast<std::size_t>(tiles), 0);
  for (Eigen::Index v = 0; v < V; ++v) {
    const std::int32_t t = tileOfVertex[static_cast<std::size_t>(v)];
    if (t < 0) continue;
    if (t >= tiles) bad("tileOperator: a vertex's tile is outside the tiles");
    averaged.middleRows(3 * t, 3) += kernelRows.middleRows(3 * v, 3);
    ++count[static_cast<std::size_t>(t)];
  }
  for (std::int64_t t = 0; t < tiles; ++t) {
    if (count[static_cast<std::size_t>(t)] == 0) bad("tileOperator: tile " + std::to_string(t) + " holds no vertex");
    averaged.middleRows(3 * t, 3) /= static_cast<double>(count[static_cast<std::size_t>(t)]);
  }
  return averaged * synthesis;                                                               // compose first: [3T x C] [C x K]
}

}  // namespace nxr::signal
