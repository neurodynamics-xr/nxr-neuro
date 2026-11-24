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

void accumulateTileCrossSpectra(const Eigen::MatrixXcd& Z, const Eigen::MatrixXd& gains, std::vector<Eigen::MatrixXcd>& sums) {
  if (Z.rows() % 3 != 0) bad("accumulateTileCrossSpectra: the tile coefficients are three components of each tile");
  if (gains.cols() != Z.cols()) bad("accumulateTileCrossSpectra: one gain per bin");
  const Eigen::Index T = Z.rows() / 3, A = gains.rows();
  if (sums.empty()) sums.assign(static_cast<std::size_t>(A), Eigen::MatrixXcd::Zero(T, T));
  if (static_cast<Eigen::Index>(sums.size()) != A) bad("accumulateTileCrossSpectra: one sum per atom");
  for (Eigen::Index a = 0; a < A; ++a) {
    std::vector<Eigen::Index> bins;
    for (Eigen::Index k = 0; k < Z.cols(); ++k) if (gains(a, k) != 0) bins.push_back(k);
    if (bins.empty()) continue;
    for (int c = 0; c < 3; ++c) {
      Eigen::MatrixXcd weighted(T, static_cast<Eigen::Index>(bins.size()));
      for (std::size_t q = 0; q < bins.size(); ++q)
        for (Eigen::Index t = 0; t < T; ++t) weighted(t, static_cast<Eigen::Index>(q)) = gains(a, bins[q]) * Z(3 * t + c, bins[q]);
      sums[static_cast<std::size_t>(a)].noalias() += weighted * weighted.adjoint();
    }
  }
}

TileCells tileTimeCells(const Eigen::MatrixXcd& X, std::int64_t firstBin, std::int64_t N, std::int64_t ownFirst, std::int64_t ownSamples, std::int64_t lineFirst,
                        std::int64_t cellSamples, const Eigen::MatrixXd& M) {
  const Eigen::Index K = X.rows(), B = X.cols();
  if (M.cols() != K || M.rows() % 3 != 0) bad("tileTimeCells: the tile operator is [3T x K] of the spectrum's K coordinates");
  if (firstBin < 0 || firstBin + B > N) bad("tileTimeCells: the bins fall outside the transform");
  if (ownFirst < 0 || ownSamples < 1 || ownFirst + ownSamples > N) bad("tileTimeCells: the chunk's own samples fall outside the transform");
  if (cellSamples < 1 || lineFirst < 0) bad("tileTimeCells: a cell holds at least one sample; the line starts at sample 0");
  const Eigen::Index T = M.rows() / 3;
  const std::int64_t P = T * (T - 1) / 2;
  TileCells out;
  out.firstCell = lineFirst / cellSamples;
  const std::int64_t cells = (lineFirst + ownSamples - 1) / cellSamples - out.firstCell + 1;
  out.pairReal = Eigen::MatrixXd::Zero(P, cells);
  out.pairImaginary = Eigen::MatrixXd::Zero(P, cells);
  out.power = Eigen::MatrixXd::Zero(T, cells);
  out.samples.assign(static_cast<std::size_t>(cells), 0);
  for (std::int64_t c = 0; c < cells; ++c) {
    const std::int64_t lo = std::max(lineFirst, (out.firstCell + c) * cellSamples), hi = std::min(lineFirst + ownSamples, (out.firstCell + c + 1) * cellSamples);
    out.samples[static_cast<std::size_t>(c)] = hi - lo;
  }
  // the atom's bins only (the gains are in X): drop the bins it does not reach before the operator is applied
  std::vector<Eigen::Index> bins;
  for (Eigen::Index k = 0; k < B; ++k) if (X.col(k).squaredNorm() > 0) bins.push_back(k);
  Eigen::MatrixXcd reached(K, static_cast<Eigen::Index>(bins.size()));
  for (std::size_t q = 0; q < bins.size(); ++q) reached.col(static_cast<Eigen::Index>(q)) = X.col(bins[q]);
  for (int c = 0; c < 3; ++c) {
    Eigen::MatrixXd rows(T, K);
    for (Eigen::Index t = 0; t < T; ++t) rows.row(t) = M.row(3 * t + c);
    const Eigen::MatrixXcd spectral = rows.cast<std::complex<double>>() * reached;               // [T x the atom's bins]
    std::vector<std::complex<double>> buffer(static_cast<std::size_t>(T * N), std::complex<double>(0, 0));
    for (Eigen::Index t = 0; t < T; ++t)
      for (std::size_t q = 0; q < bins.size(); ++q) buffer[static_cast<std::size_t>(t * N + firstBin + bins[q])] = spectral(t, static_cast<Eigen::Index>(q));
    c2c(buffer.data(), buffer.data(), {static_cast<std::size_t>(T), static_cast<std::size_t>(N)}, 1, false);
    Eigen::MatrixXcd z(T, ownSamples);
    for (Eigen::Index t = 0; t < T; ++t) for (std::int64_t s = 0; s < ownSamples; ++s) z(t, s) = buffer[static_cast<std::size_t>(t * N + ownFirst + s)];
    buffer.clear(); buffer.shrink_to_fit();
    std::int64_t start = 0;
    for (std::int64_t cell = 0; cell < cells; ++cell) {
      const std::int64_t width = out.samples[static_cast<std::size_t>(cell)];
      const auto block = z.middleCols(start, width);
      start += width;
      const Eigen::MatrixXcd sums = block * block.adjoint();                                      // [T x T]: Σ_t z_i conj(z_j) over the cell's samples (every pair is an element)
      for (Eigen::Index i = 0; i < T; ++i) {
        out.power(i, cell) += sums(i, i).real();
        for (Eigen::Index j = i + 1; j < T; ++j) {
          const std::int64_t q = i * T - i * (i + 1) / 2 + (j - i - 1);
          out.pairReal(q, cell) += sums(i, j).real();
          out.pairImaginary(q, cell) += sums(i, j).imag();
        }
      }
    }
  }
  return out;
}

}  // namespace nxr::signal
