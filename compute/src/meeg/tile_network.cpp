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

GraphModes graphHodgeDirac(const std::vector<std::int32_t>& edges, std::int64_t nodes, const Eigen::VectorXd& nodeWeights, const Eigen::VectorXd& edgeWeights, double nullTolerance) {
  const std::int64_t V = nodes, E = static_cast<std::int64_t>(edges.size() / 2);
  if (V < 1 || edges.size() % 2) bad("graphHodgeDirac: edges are [E x 2] on a graph of at least one vertex");
  if (nodeWeights.size() != V || edgeWeights.size() != E) bad("graphHodgeDirac: one weight per vertex (⋆0) and one per edge (⋆1)");
  if (!(nullTolerance >= 0)) bad("graphHodgeDirac: the null tolerance is >= 0");
  for (std::int64_t v = 0; v < V; ++v) if (!(nodeWeights(v) > 0) || !std::isfinite(nodeWeights(v))) bad("graphHodgeDirac: a vertex weight (⋆0) is not positive");
  for (std::int64_t e = 0; e < E; ++e) if (!(edgeWeights(e) > 0) || !std::isfinite(edgeWeights(e))) bad("graphHodgeDirac: an edge weight (⋆1) is not positive");
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(E, V);                              // d0: -1 at the tail, +1 at the tip
  for (std::int64_t e = 0; e < E; ++e) {
    const std::int32_t a = edges[static_cast<std::size_t>(2 * e)], b = edges[static_cast<std::size_t>(2 * e + 1)];
    if (a < 0 || a >= V || b < 0 || b >= V) bad("graphHodgeDirac: an edge names a vertex outside the graph");
    if (a == b) bad("graphHodgeDirac: an edge is a loop");
    B(e, a) -= 1; B(e, b) += 1;
  }
  GraphModes out;
  out.nodes = V; out.edges = E;
  // the exact modes: the generalized symmetric problem Bᵀ⋆1 B ψ = λ ⋆0 ψ (ψᵀ⋆0ψ = 1), λ ascending; the null ones (one per component) out
  const Eigen::MatrixXd L = B.transpose() * edgeWeights.asDiagonal() * B;
  const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(L, Eigen::MatrixXd(nodeWeights.asDiagonal()));
  if (solver.info() != Eigen::Success) bad("graphHodgeDirac: the eigensolve did not converge");
  const Eigen::VectorXd lambda = solver.eigenvalues();
  const double top = lambda.size() ? lambda.cwiseAbs().maxCoeff() : 0.0;
  std::vector<Eigen::Index> exact;
  for (Eigen::Index k = 0; k < lambda.size(); ++k) { if (lambda(k) <= nullTolerance * top) ++out.components; else exact.push_back(k); }
  // the cycle modes: the kernel of Bᵀ⋆1, in ⋆1^½ coordinates the orthogonal complement of the range of ⋆1^½ B (a rank-revealing QR), back to h = ⋆1^-½ y (hᵀ⋆1h = 1)
  Eigen::MatrixXd cycles(E, 0);
  if (E > 0) {
    const Eigen::VectorXd root = edgeWeights.array().sqrt();
    const Eigen::MatrixXd M = root.asDiagonal() * B;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(M);
    const Eigen::Index rank = qr.rank();
    if (rank != V - out.components) bad("graphHodgeDirac: the incidence's rank (" + std::to_string(rank) + ") is not the vertices less the components (" + std::to_string(V - out.components) + "): raise or lower the null tolerance");
    const Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(E, E);
    cycles = root.cwiseInverse().asDiagonal() * Q.rightCols(E - rank);
  }
  out.cycles = cycles.cols();
  const std::int64_t K = out.cycles + static_cast<std::int64_t>(exact.size());
  out.mu = Eigen::VectorXd::Zero(K);
  out.forms = Eigen::MatrixXd::Zero(V + E, K);
  out.kind.assign(static_cast<std::size_t>(K), 1);
  const double half = std::sqrt(0.5);
  for (std::int64_t c = 0; c < out.cycles; ++c) out.forms.block(V, c, E, 1) = cycles.col(c);
  for (std::size_t i = 0; i < exact.size(); ++i) {
    const std::int64_t k = out.cycles + static_cast<std::int64_t>(i);
    Eigen::VectorXd psi = solver.eigenvectors().col(exact[i]);
    Eigen::Index at = 0;
    psi.cwiseAbs().maxCoeff(&at);
    if (psi(at) < 0) psi = -psi;                                                // one sign: the largest entry positive
    const double mu = std::sqrt(lambda(exact[i]));
    out.mu(k) = mu;
    out.kind[static_cast<std::size_t>(k)] = 0;
    out.forms.block(0, k, V, 1) = half * psi;
    if (E > 0) out.forms.block(V, k, E, 1) = (half / mu) * (B * psi);
  }
  return out;
}

NetworkFlow networkFlow(const Eigen::VectorXd& edgeFlow, const std::vector<std::int32_t>& edges, const Eigen::VectorXd& edgeWeights, const GraphModes& modes) {
  const std::int64_t V = modes.nodes, E = modes.edges, K = modes.mu.size();
  if (static_cast<std::int64_t>(edges.size()) != 2 * E || edgeFlow.size() != E || edgeWeights.size() != E) bad("networkFlow: one flow and one weight per edge of the graph the modes were solved on");
  if (modes.forms.rows() != V + E || modes.forms.cols() != K) bad("networkFlow: the modes' forms are not [(V + E) x K]");
  NetworkFlow out;
  out.divergence = Eigen::VectorXd::Zero(V);
  for (std::int64_t e = 0; e < E; ++e) {
    const double f = edgeFlow(e);
    if (!std::isfinite(f)) bad("networkFlow: an edge's flow is not finite");
    out.divergence(edges[static_cast<std::size_t>(2 * e)]) -= f;
    out.divergence(edges[static_cast<std::size_t>(2 * e + 1)]) += f;
    out.weightedFlowSquares += edgeWeights(e) * f * f;
  }
  // the coefficients: the projection on each mode's 1-form part φ_k (⋆1-norm² 1/2 on an exact mode, 1 on a cycle mode): c_k = ⟨F, φ_k⟩⋆1 / ⟨φ_k, φ_k⟩⋆1; inner products, no solve
  const Eigen::VectorXd weighted = edgeWeights.cwiseProduct(edgeFlow);
  out.coefficients = Eigen::VectorXd::Zero(K);
  out.potential = Eigen::VectorXd::Zero(V);
  for (std::int64_t k = 0; k < K; ++k) {
    const auto phi = modes.forms.block(V, k, E, 1);
    const double norm = phi.col(0).dot(edgeWeights.cwiseProduct(phi.col(0)));
    if (!(norm > 0)) continue;
    const double c = phi.col(0).dot(weighted) / norm;
    out.coefficients(k) = c;
    if (modes.kind[static_cast<std::size_t>(k)] == 0) {
      out.potential += (c / modes.mu(k)) * modes.forms.block(0, k, V, 1);          // Σ c_k (ψ_k/√2) / μ_k: its d0 is the exact part Σ c_k φ_k (intrinsic_forms' exactPotential)
      out.weightedGradientSquares += c * c * norm;
    } else {
      out.weightedCycleSquares += c * c * norm;
    }
  }
  return out;
}

}  // namespace nxr::meeg
