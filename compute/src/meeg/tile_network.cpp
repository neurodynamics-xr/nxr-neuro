#include "nxr/meeg/tile_network.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace nxr::meeg {

namespace {
[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }

template <class Scalar>
std::vector<double> lengthsOf(const std::vector<Scalar>& points, std::int64_t pointsPerFiber) {
  if (pointsPerFiber < 2) bad("fiberLengths: a fibre has at least two points");
  const std::int64_t rows = static_cast<std::int64_t>(points.size()) / 3;
  if (static_cast<std::int64_t>(points.size()) != 3 * rows || rows % pointsPerFiber != 0) bad("fiberLengths: points are [nF·nP x 3]");
  const std::int64_t fibres = rows / pointsPerFiber;
  std::vector<double> out(static_cast<std::size_t>(fibres), 0.0);
  for (std::int64_t f = 0; f < fibres; ++f) {
    double sum = 0;
    for (std::int64_t p = 1; p < pointsPerFiber; ++p) {
      const std::size_t a = static_cast<std::size_t>(3 * (f * pointsPerFiber + p - 1)), b = a + 3;
      double s = 0;
      for (int c = 0; c < 3; ++c) { const double d = static_cast<double>(points[b + c]) - static_cast<double>(points[a + c]); s += d * d; }
      sum += std::sqrt(s);
    }
    out[static_cast<std::size_t>(f)] = sum;
  }
  return out;
}

double medianOf(std::vector<double> v) {                         // numpy's median: the mean of the two middle values for an even count
  const std::size_t n = v.size();
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 2), v.end());
  const double upper = v[n / 2];
  if (n % 2 == 1) return upper;
  return 0.5 * (upper + *std::max_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(n / 2)));
}
}  // namespace

std::vector<double> fiberLengths(const std::vector<double>& points, std::int64_t pointsPerFiber) { return lengthsOf(points, pointsPerFiber); }
std::vector<double> fiberLengths(const std::vector<float>& points, std::int64_t pointsPerFiber) { return lengthsOf(points, pointsPerFiber); }

std::vector<std::int64_t> tileCodesAtLevel(const std::vector<std::int64_t>& finestCodes, int depth, int level) {
  if (level < 0 || level > depth) bad("tileCodesAtLevel: the level " + std::to_string(level) + " is outside the grid's levels 0 .. " + std::to_string(depth));
  std::vector<std::int64_t> out(finestCodes.size());
  for (std::size_t v = 0; v < finestCodes.size(); ++v) out[v] = finestCodes[v] < 0 ? -1 : finestCodes[v] >> (depth - level);
  return out;
}

TileNetwork tilePairs(const std::vector<std::int64_t>& tileCodes, const Eigen::MatrixXd& positions, int level, const std::vector<std::int32_t>& endVertex,
                      const std::vector<std::uint8_t>& keep, const std::vector<double>& lengths) {
  const std::int64_t V = static_cast<std::int64_t>(tileCodes.size());
  if (positions.rows() != V || positions.cols() != 3) bad("tilePairs: positions are [V x 3] of the tiles' vertices");
  if (endVertex.size() % 2 != 0) bad("tilePairs: the end vertices are [2nF] in block order");
  const std::int64_t fibres = static_cast<std::int64_t>(endVertex.size() / 2);
  if (!keep.empty() && static_cast<std::int64_t>(keep.size()) != fibres) bad("tilePairs: keep is [nF]");
  if (!lengths.empty() && static_cast<std::int64_t>(lengths.size()) != fibres) bad("tilePairs: lengths are [nF]");
  TileNetwork out;
  std::map<std::int64_t, std::int32_t> indexOf;
  for (const std::int64_t c : tileCodes) if (c >= 0) indexOf.emplace(c, 0);
  if (indexOf.size() < 2) bad("tilePairs: fewer than two tiles");
  for (auto& [code, index] : indexOf) { index = static_cast<std::int32_t>(out.codes.size()); out.codes.push_back(code); out.pieces.push_back(code >> level); }
  const std::int64_t T = static_cast<std::int64_t>(out.codes.size());
  std::vector<std::int32_t> tileOf(static_cast<std::size_t>(V), -1);
  out.centroids = Eigen::MatrixXd::Zero(T, 3);
  out.vertexCount.assign(static_cast<std::size_t>(T), 0);
  for (std::int64_t v = 0; v < V; ++v)
    if (tileCodes[static_cast<std::size_t>(v)] >= 0) {
      const std::int32_t t = indexOf.at(tileCodes[static_cast<std::size_t>(v)]);
      tileOf[static_cast<std::size_t>(v)] = t;
      out.centroids.row(t) += positions.row(v);
      ++out.vertexCount[static_cast<std::size_t>(t)];
    }
  for (std::int64_t t = 0; t < T; ++t) out.centroids.row(t) /= static_cast<double>(out.vertexCount[static_cast<std::size_t>(t)]);
  const std::int64_t P = T * (T - 1) / 2;
  out.first.resize(static_cast<std::size_t>(P)); out.second.resize(static_cast<std::size_t>(P));
  out.count = Eigen::VectorXd::Zero(P); out.medianLength = Eigen::VectorXd::Constant(P, std::numeric_limits<double>::quiet_NaN());
  out.distance.resize(P); out.across.resize(P);
  for (std::int64_t i = 0; i < T; ++i)
    for (std::int64_t j = i + 1; j < T; ++j) {
      const std::int64_t p = pairIndex(i, j, T);
      out.first[static_cast<std::size_t>(p)] = static_cast<std::int32_t>(i); out.second[static_cast<std::size_t>(p)] = static_cast<std::int32_t>(j);
      out.distance(p) = (out.centroids.row(i) - out.centroids.row(j)).norm();
      out.across(p) = out.pieces[static_cast<std::size_t>(i)] != out.pieces[static_cast<std::size_t>(j)] ? 1.0 : 0.0;
    }
  std::vector<std::vector<double>> pairLengths(static_cast<std::size_t>(P));
  for (std::int64_t f = 0; f < fibres; ++f) {
    if (!keep.empty() && !keep[static_cast<std::size_t>(f)]) { ++out.gatedOut; continue; }
    const std::int32_t a = endVertex[static_cast<std::size_t>(f)], b = endVertex[static_cast<std::size_t>(fibres + f)];
    const std::int32_t ta = a >= 0 && a < V ? tileOf[static_cast<std::size_t>(a)] : -1, tb = b >= 0 && b < V ? tileOf[static_cast<std::size_t>(b)] : -1;
    if (ta < 0 || tb < 0) { ++out.endsOutside; continue; }
    if (ta == tb) { ++out.sameTile; continue; }
    const std::int64_t p = pairIndex(std::min(ta, tb), std::max(ta, tb), T);
    out.count(p) += 1.0;
    if (!lengths.empty()) pairLengths[static_cast<std::size_t>(p)].push_back(lengths[static_cast<std::size_t>(f)]);
  }
  for (std::int64_t p = 0; p < P; ++p) if (!pairLengths[static_cast<std::size_t>(p)].empty()) out.medianLength(p) = medianOf(std::move(pairLengths[static_cast<std::size_t>(p)]));
  return out;
}

LaplacianModes laplacianEigenmodes(const Eigen::VectorXd& pairWeights, std::int64_t tiles, double nullTolerance) {
  if (tiles < 2 || pairWeights.size() != tiles * (tiles - 1) / 2) bad("laplacianEigenmodes: the weights are one per pair i < j of the tiles");
  if (!(nullTolerance >= 0)) bad("laplacianEigenmodes: the null tolerance is >= 0");
  LaplacianModes out;
  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(tiles, tiles);
  for (std::int64_t i = 0; i < tiles; ++i)
    for (std::int64_t j = i + 1; j < tiles; ++j) {
      const double w = pairWeights(pairIndex(i, j, tiles));
      if (w < 0) bad("laplacianEigenmodes: a negative weight");
      W(i, j) = w; W(j, i) = w;
    }
  out.laplacian = Eigen::MatrixXd(W.rowwise().sum().asDiagonal()) - W;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(out.laplacian);
  if (solver.info() != Eigen::Success) bad("laplacianEigenmodes: the eigensolve did not converge");
  out.eigenvalues = solver.eigenvalues();
  out.eigenvectors = solver.eigenvectors();
  const double top = out.eigenvalues.cwiseAbs().maxCoeff();
  for (Eigen::Index k = 0; k < out.eigenvalues.size(); ++k) out.nullModes += out.eigenvalues(k) <= nullTolerance * top;
  return out;
}

NetworkFlow networkFlow(const Eigen::VectorXd& pairFlow, const Eigen::VectorXd& pairWeights, const LaplacianModes& modes) {
  const std::int64_t T = modes.eigenvalues.size();
  if (pairFlow.size() != T * (T - 1) / 2 || pairWeights.size() != pairFlow.size()) bad("networkFlow: one flow and one weight per pair i < j of the network's tiles");
  NetworkFlow out;
  out.divergence = Eigen::VectorXd::Zero(T);
  Eigen::VectorXd weightedDivergence = Eigen::VectorXd::Zero(T);               // Bᵀ W F
  for (std::int64_t i = 0; i < T; ++i)
    for (std::int64_t j = i + 1; j < T; ++j) {
      const std::int64_t p = pairIndex(i, j, T);
      const double w = pairWeights(p);
      if (!(w > 0)) continue;
      const double f = pairFlow(p);
      if (!std::isfinite(f)) bad("networkFlow: an edge's flow is not finite");
      out.divergence(i) -= f; out.divergence(j) += f;
      weightedDivergence(i) -= w * f; weightedDivergence(j) += w * f;
      out.weightedFlowSquares += w * f * f;
    }
  // the Poisson solve as a filter: the 1/eigenvalue gain on the complete eigenmodes, the null modes (one constant per connected component) left out
  const Eigen::VectorXd projected = modes.eigenvectors.transpose() * weightedDivergence;
  out.coefficients = Eigen::VectorXd::Zero(T);
  for (std::int64_t k = modes.nullModes; k < T; ++k) out.coefficients(k) = projected(k) / modes.eigenvalues(k);
  out.potential = modes.eigenvectors * out.coefficients;
  for (std::int64_t i = 0; i < T; ++i)
    for (std::int64_t j = i + 1; j < T; ++j) {
      const double w = pairWeights(pairIndex(i, j, T));
      if (!(w > 0)) continue;
      const double g = out.potential(j) - out.potential(i);
      out.weightedGradientSquares += w * g * g;
    }
  return out;
}

}  // namespace nxr::meeg
