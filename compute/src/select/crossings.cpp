// nxr::select::track_crossings (crossings.h): nxr `Tiles.crossings`.
#include "nxr/select/track_crossings.h"
#include "nxr/geometry/surface_maps.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nxr::select::track_crossings {


std::vector<std::int32_t> nearestVertex(const Eigen::MatrixXd& positions, const Eigen::MatrixXd& query) {
  if (positions.cols() != 3 || (query.rows() > 0 && query.cols() != 3)) throw std::invalid_argument("nearestVertex: points are [n x 3]");
  if (positions.rows() == 0) throw std::invalid_argument("nearestVertex: no vertex");
  // the one nearest-vertex search (the k-d tree of geometry::nearest; the lowest index on a tie), on flat row-major copies
  const auto flat = [](const Eigen::MatrixXd& m) { std::vector<double> v(static_cast<std::size_t>(m.size())); for (Eigen::Index i = 0; i < m.rows(); ++i) for (int c = 0; c < 3; ++c) v[static_cast<std::size_t>(3 * i + c)] = m(i, c); return v; };
  return nxr::geometry::nearest::nearestVertices<double>(flat(positions), flat(query)).vertex;
}

Result crossings(const Steps& s, const std::vector<std::uint8_t>& use, const Surface& S, int tiles) {
  const std::int64_t n = s.size(), V = S.positions.rows();
  if (s.from.rows() != n || s.disp.rows() != n || static_cast<std::int64_t>(s.track.size()) != n || (n > 0 && (s.from.cols() != 3 || s.disp.cols() != 3))) throw std::invalid_argument("crossings: the steps' arrays are for different numbers of steps");
  if (!use.empty() && static_cast<std::int64_t>(use.size()) != n) throw std::invalid_argument("crossings: `use` is for another number of steps");
  if (V == 0 || S.positions.cols() != 3 || S.normals.rows() != V || S.e1.rows() != V || S.e2.rows() != V || S.rho.size() != V || static_cast<std::int64_t>(S.node.size()) != V) throw std::invalid_argument("crossings: the surface's arrays are for different numbers of vertices");
  if (tiles < 0) throw std::invalid_argument("crossings: tiles >= 0");
  for (const auto sg : s.sense) if (sg != 1 && sg != -1) throw std::invalid_argument("crossings: a sense is +1 or -1");
  for (const auto t : S.node) if (t >= tiles) throw std::invalid_argument("crossings: a vertex's tile " + std::to_string(t) + " is outside 0 … " + std::to_string(tiles - 1));
  Result r;
  r.sum = Eigen::MatrixXcd::Zero(tiles, 2); r.n = Eigen::MatrixXi::Zero(tiles, 2);
  // the steps asked for, placed on their nearest vertex
  std::vector<std::int64_t> pick;
  for (std::int64_t i = 0; i < n; ++i) if (use.empty() || use[static_cast<std::size_t>(i)]) pick.push_back(i);
  r.steps_in = static_cast<std::int64_t>(pick.size());
  Eigen::MatrixXd q(static_cast<Eigen::Index>(pick.size()), 3);
  for (std::size_t k = 0; k < pick.size(); ++k) q.row(static_cast<Eigen::Index>(k)) = s.from.row(pick[k]);
  const std::vector<std::int32_t> near = nearestVertex(S.positions, q);
  // one crossing a (tile, sense, track): the carried unit vectors of its steps added; the bridge's in 3-D
  std::map<std::tuple<std::int32_t, int, std::int32_t>, std::complex<double>> tile;
  std::map<std::tuple<int, std::int32_t>, Eigen::Vector3d> bridge;
  for (std::size_t k = 0; k < pick.size(); ++k) {
    const std::int64_t i = pick[k];
    const std::int32_t v = near[k];
    const Eigen::Vector3d nrm = S.normals.row(v).transpose(), d0 = s.disp.row(i).transpose();
    const Eigen::Vector3d d = d0 - d0.dot(nrm) * nrm;
    const double ln = d.norm();
    if (!(ln > 1e-9)) { ++r.no_direction; continue; }
    const std::int32_t nd = S.node[static_cast<std::size_t>(v)];
    const int sx = s.sense[static_cast<std::size_t>(i)] > 0 ? 0 : 1;
    // emplace the zero vector: a map's operator[] leaves an Eigen fixed-size vector uninitialised (zero only by luck of the allocator)
    if (nd == kBridge) { bridge.emplace(std::make_tuple(sx, s.track[static_cast<std::size_t>(i)]), Eigen::Vector3d::Zero().eval()).first->second += d / ln; ++r.steps_used; continue; }
    if (nd < 0) { ++r.off_tiles; continue; }
    const Eigen::Vector3d u = d / ln;
    const double c1 = u.dot(S.e1.row(v).transpose()), c2 = u.dot(S.e2.row(v).transpose());
    std::complex<double> w = 0;
    if (std::isfinite(c1) && std::isfinite(c2) && std::isfinite(S.rho[v])) w = std::complex<double>(c1, c2) * std::exp(std::complex<double>(0, -S.rho[v]));
    tile[{nd, sx, s.track[static_cast<std::size_t>(i)]}] += w;
    ++r.steps_used;
  }
  for (const auto& [key, w] : tile) {
    const auto [nd, sx, trk] = key;
    (void)trk;
    r.sum(nd, sx) += w / std::max(std::abs(w), 1e-12);
    ++r.n(nd, sx);
  }
  for (const auto& [key, w] : bridge) {
    const int sx = std::get<0>(key);
    r.bridge.row(sx) += w.transpose() / std::max(w.norm(), 1e-12);
    ++r.bridge_n[sx];
  }
  return r;
}

Eigen::MatrixXd embedAtCentres(const Eigen::VectorXcd& w, const std::vector<int>& centres, const geometry::Mat3& e1, const geometry::Mat3& e2) {
  if (static_cast<std::size_t>(w.size()) != centres.size()) throw std::invalid_argument("embedAtCentres: one centre a tile");
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(w.size(), 3);
  for (Eigen::Index t = 0; t < w.size(); ++t) {
    const int c = centres[static_cast<std::size_t>(t)];
    if (c < 0 || !std::isfinite(w[t].real()) || !std::isfinite(w[t].imag())) continue;
    const Eigen::Vector3d a = w[t].real() * e1.row(c).transpose() + w[t].imag() * e2.row(c).transpose();
    if (a.allFinite()) out.row(t) = a.transpose();
  }
  return out;
}

Pooled poolSenses(const Eigen::MatrixXd& plus, const Eigen::VectorXd& nPlus, const Eigen::MatrixXd& minus, const Eigen::VectorXd& nMinus) {
  if (plus.rows() != minus.rows() || plus.cols() != 3 || minus.cols() != 3 || nPlus.size() != plus.rows() || nMinus.size() != minus.rows()) throw std::invalid_argument("poolSenses: the senses' sums are [tiles x 3] and their counts [tiles], the same tiles");
  return Pooled{plus + minus, nPlus + nMinus};
}

Repeat repeat(const Pooled& a, const Pooled& b, int minCrossings) {
  if (a.sum.rows() != b.sum.rows() || a.sum.cols() != 3 || b.sum.cols() != 3 || a.n.size() != a.sum.rows() || b.n.size() != b.sum.rows()) throw std::invalid_argument("repeat: the two runs' sums are [tiles x 3] and counts [tiles], the same tiles");
  if (minCrossings < 1) throw std::invalid_argument("repeat: minCrossings >= 1");
  const Eigen::Index T = a.sum.rows();
  Repeat r;
  r.dot = Eigen::VectorXd::Constant(T, std::numeric_limits<double>::quiet_NaN());
  double num = 0, den = 0;
  for (Eigen::Index t = 0; t < T; ++t) {
    if (!(a.n[t] >= minCrossings && b.n[t] >= minCrossings)) continue;
    r.dot[t] = (a.sum.row(t) / a.n[t]).dot(b.sum.row(t) / b.n[t]);
    const double na = a.sum.row(t).norm(), nb = b.sum.row(t).norm();
    if (!(na > 0 && nb > 0)) continue;
    const double w = std::min(a.n[t], b.n[t]);
    num += a.sum.row(t).dot(b.sum.row(t)) / (na * nb) * w; den += w;
    ++r.tilesUsed;
  }
  if (den > 0) r.cosine = num / den;
  return r;
}

}  // namespace nxr::select::track_crossings
