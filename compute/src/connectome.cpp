// nxr::connectome — the structural connectome's pure functions (connectome.h).
#include "nxr/connectome.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#include "nxr/congruence.h"
#include "nxr/nearest.h"

namespace nxr::connectome {

namespace {
[[noreturn]] void refuse(const std::string& m) { throw std::invalid_argument("connectome: " + m); }
int ix(std::int64_t v) { return static_cast<int>(v); }
}  // namespace

template <class Scalar>
std::vector<Scalar> fiberEnds(const std::vector<Scalar>& points, std::int64_t nP) {
  if (nP < 2) refuse("a fiber has at least 2 points");
  if (points.size() % 3 != 0) refuse("points are [n x 3]");
  const std::int64_t n = static_cast<std::int64_t>(points.size() / 3);
  if (n % nP != 0) refuse("the number of points (" + std::to_string(n) + ") is not a multiple of the points per fiber (" + std::to_string(nP) + ")");
  const std::int64_t nF = n / nP;
  std::vector<Scalar> out(static_cast<std::size_t>(6 * nF));
  for (std::int64_t f = 0; f < nF; ++f)
    for (int k = 0; k < 3; ++k) {
      out[static_cast<std::size_t>(3 * f + k)] = points[static_cast<std::size_t>(3 * (f * nP) + k)];
      out[static_cast<std::size_t>(3 * (nF + f) + k)] = points[static_cast<std::size_t>(3 * (f * nP + nP - 1) + k)];
    }
  return out;
}

template <class Item>
std::vector<Item> interleaveEnds(const std::vector<Item>& block) {
  if (block.size() % 2 != 0) refuse("the ends of a bundle are two a fiber");
  const std::size_t nF = block.size() / 2;
  std::vector<Item> out(block.size());
  for (std::size_t f = 0; f < nF; ++f) { out[2 * f] = block[f]; out[2 * f + 1] = block[nF + f]; }
  return out;
}
template <class Item>
std::vector<Item> blockEnds(const std::vector<Item>& inter) {
  if (inter.size() % 2 != 0) refuse("the ends of a bundle are two a fiber");
  const std::size_t nF = inter.size() / 2;
  std::vector<Item> out(inter.size());
  for (std::size_t f = 0; f < nF; ++f) { out[f] = inter[2 * f]; out[nF + f] = inter[2 * f + 1]; }
  return out;
}

template <class Scalar>
std::vector<std::uint8_t> withinGate(const std::vector<Scalar>& d, Scalar maxDistance) {
  if (!(maxDistance >= Scalar(0))) refuse("the gate is a distance >= 0 (no gate: keepAll)");
  if (d.size() % 2 != 0) refuse("the ends of a bundle are two a fiber");
  const std::size_t nF = d.size() / 2;
  std::vector<std::uint8_t> keep(nF);
  for (std::size_t f = 0; f < nF; ++f) keep[f] = (d[f] <= maxDistance && d[nF + f] <= maxDistance) ? 1 : 0;   // a NaN distance is not within
  return keep;
}

template <class Scalar>
std::vector<std::int32_t> strongestDestination(const nxr::registration::Csr<Scalar>& w) {
  std::vector<std::int32_t> dest(static_cast<std::size_t>(w.cols), -1);
  std::vector<Scalar> best(static_cast<std::size_t>(w.cols), -std::numeric_limits<Scalar>::infinity());
  for (std::int64_t i = 0; i < w.rows; ++i)   // rows ascend: a tie keeps the lowest
    for (std::int32_t k = w.indptr[static_cast<std::size_t>(i)]; k < w.indptr[static_cast<std::size_t>(i) + 1]; ++k) {
      const std::size_t j = static_cast<std::size_t>(w.indices[static_cast<std::size_t>(k)]);
      if (w.data[static_cast<std::size_t>(k)] > best[j]) { best[j] = w.data[static_cast<std::size_t>(k)]; dest[j] = static_cast<std::int32_t>(i); }
    }
  return dest;
}

template <class Scalar>
std::vector<std::int32_t> registrationMatch(const std::vector<Scalar>& tSphere, const std::vector<nxr::registration::Piece>& tPieces, const std::vector<Scalar>& sSphere,
                                            const std::vector<nxr::registration::Piece>& sPieces, int neighbours, MatchReport* report) {
  nxr::registration::Options opt;
  opt.method = nxr::registration::Method::Shepard;
  opt.neighbours = neighbours;
  const auto w = nxr::registration::resample_operator<Scalar>(tSphere, {}, tPieces, sSphere, sPieces, opt);
  std::vector<std::int32_t> match = strongestDestination(w);
  // a template vertex no subject vertex takes weight from: its nearest subject vertex on the (unit-normalised) sphere, within its piece
  std::int64_t empty = 0;
  auto unit = [](const Scalar* p, std::vector<Scalar>& out) {
    const Scalar n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    for (int k = 0; k < 3; ++k) out.push_back(n > Scalar(0) ? p[k] / n : p[k]);
  };
  for (std::size_t k = 0; k < tPieces.size(); ++k) {
    std::vector<std::int64_t> todo;
    for (std::int64_t v = tPieces[k].first; v < tPieces[k].first + tPieces[k].count; ++v)
      if (match[static_cast<std::size_t>(v)] < 0) todo.push_back(v);
    if (todo.empty()) continue;
    empty += static_cast<std::int64_t>(todo.size());
    std::vector<Scalar> dst, q;
    for (std::int64_t v = sPieces[k].first; v < sPieces[k].first + sPieces[k].count; ++v) unit(&sSphere[static_cast<std::size_t>(3 * v)], dst);
    for (const std::int64_t v : todo) unit(&tSphere[static_cast<std::size_t>(3 * v)], q);
    const auto near = nxr::nearest::nearestVertices<Scalar>(dst, q);
    for (std::size_t i = 0; i < todo.size(); ++i) match[static_cast<std::size_t>(todo[i])] = static_cast<std::int32_t>(sPieces[k].first) + near.vertex[i];
  }
  if (report) report->empty_columns = empty;
  return match;
}

std::vector<std::int32_t> carryEnds(const std::vector<std::int32_t>& endVertex, const std::vector<std::int32_t>& match) {
  std::vector<std::int32_t> out(endVertex.size());
  for (std::size_t i = 0; i < endVertex.size(); ++i) {
    if (endVertex[i] < 0 || static_cast<std::size_t>(endVertex[i]) >= match.size()) refuse("an end's vertex " + std::to_string(endVertex[i]) + " is outside the registration's " + std::to_string(match.size()) + " vertices");
    out[i] = match[static_cast<std::size_t>(endVertex[i])];
  }
  return out;
}

template <class Scalar>
Sparse<Scalar> smoothingStep(const std::vector<std::int32_t>& faces, std::int64_t nVertices) {
  if (faces.size() % 3 != 0) refuse("faces are [m x 3]");
  std::vector<std::vector<std::int32_t>> nb(static_cast<std::size_t>(nVertices));
  for (std::size_t i = 0; i < faces.size(); i += 3)
    for (int a = 0; a < 3; ++a) {
      const std::int32_t u = faces[i + static_cast<std::size_t>(a)], v = faces[i + static_cast<std::size_t>((a + 1) % 3)];
      if (u < 0 || v < 0 || u >= nVertices || v >= nVertices) refuse("a face names a vertex outside 0 ... " + std::to_string(nVertices - 1));
      nb[static_cast<std::size_t>(u)].push_back(v);
      nb[static_cast<std::size_t>(v)].push_back(u);
    }
  std::vector<Eigen::Triplet<Scalar>> t;
  for (std::int64_t u = 0; u < nVertices; ++u) {
    auto& n = nb[static_cast<std::size_t>(u)];
    n.push_back(static_cast<std::int32_t>(u));
    std::sort(n.begin(), n.end());
    n.erase(std::unique(n.begin(), n.end()), n.end());
    const Scalar wgt = Scalar(1) / static_cast<Scalar>(n.size());
    for (const auto v : n) t.emplace_back(ix(u), v, wgt);
  }
  Sparse<Scalar> m(ix(nVertices), ix(nVertices));
  m.setFromTriplets(t.begin(), t.end());
  return m;
}

template <class Scalar>
Sparse<Scalar> smoothingOperator(const std::vector<std::int32_t>& faces, std::int64_t nVertices, int hops) {
  if (hops < 0) refuse("hops >= 0");
  Sparse<Scalar> p(ix(nVertices), ix(nVertices));
  p.setIdentity();
  if (hops == 0) return p;
  const Sparse<Scalar> step = smoothingStep<Scalar>(faces, nVertices);
  for (int h = 0; h < hops; ++h) p = (p * step).eval();
  return p;
}

template <class Scalar>
Sparse<Scalar> memberships(const std::vector<std::int32_t>& label, std::int64_t nRegions) {
  std::vector<Eigen::Triplet<Scalar>> t;
  for (std::size_t v = 0; v < label.size(); ++v) {
    if (label[v] < 0) continue;
    if (label[v] >= nRegions) refuse("a label " + std::to_string(label[v]) + " is outside the " + std::to_string(nRegions) + " regions");
    t.emplace_back(static_cast<int>(v), label[v], Scalar(1));
  }
  Sparse<Scalar> m(static_cast<int>(label.size()), ix(nRegions));
  m.setFromTriplets(t.begin(), t.end());
  return m;
}

template <class Scalar>
Sparse<Scalar> smoothMemberships(const Sparse<Scalar>& step, const Sparse<Scalar>& m, int hops) {
  if (hops < 0) refuse("hops >= 0");
  if (step.rows() != step.cols() || step.cols() != m.rows()) refuse("the smoothing step and the memberships are for different numbers of vertices");
  const Sparse<Scalar> adjoint = step.transpose();   // the adjoint of the smoothing: counting with these equals summing the smoothed vertex matrix over the partition
  Sparse<Scalar> out = m;
  for (int h = 0; h < hops; ++h) out = (adjoint * out).eval();
  return out;
}

template <class Scalar>
Sparse<Scalar> partitionCounts(const std::vector<std::int32_t>& endVertex, const std::vector<std::uint8_t>& keep, const Sparse<Scalar>& membershipOfVertex) {
  if (endVertex.size() % 2 != 0) refuse("the ends of a bundle are two a fiber");
  const std::size_t nF = endVertex.size() / 2;
  if (!keep.empty() && keep.size() != nF) refuse("the gate is for " + std::to_string(keep.size()) + " fibers, the ends for " + std::to_string(nF));
  std::vector<Eigen::Triplet<Scalar>> a, b;
  int row = 0;
  for (std::size_t f = 0; f < nF; ++f) {
    if (!keep.empty() && !keep[f]) continue;
    const std::int32_t u = endVertex[f], v = endVertex[nF + f];
    if (u < 0 || v < 0 || u >= membershipOfVertex.rows() || v >= membershipOfVertex.rows()) refuse("an end's vertex is outside the memberships' " + std::to_string(membershipOfVertex.rows()) + " vertices");
    a.emplace_back(row, u, Scalar(1));
    b.emplace_back(row, v, Scalar(1));
    ++row;
  }
  Sparse<Scalar> first(row, static_cast<int>(membershipOfVertex.rows())), second(row, static_cast<int>(membershipOfVertex.rows()));
  first.setFromTriplets(a.begin(), a.end());
  second.setFromTriplets(b.begin(), b.end());
  const Sparse<Scalar> m1 = first * membershipOfVertex, m2 = second * membershipOfVertex;
  const Sparse<Scalar> c = m1.transpose() * m2;
  Sparse<Scalar> out = c + Sparse<Scalar>(c.transpose());
  out.prune(Scalar(0), Scalar(0));
  return out;
}

template <class Scalar>
Sparse<Scalar> endConnectome(const std::vector<std::int32_t>& endVertex, const std::vector<std::uint8_t>& keep, std::int64_t nVertices) {
  if (endVertex.size() % 2 != 0) refuse("the ends of a bundle are two a fiber");
  const std::size_t nF = endVertex.size() / 2;
  if (!keep.empty() && keep.size() != nF) refuse("the gate is for " + std::to_string(keep.size()) + " fibers, the ends for " + std::to_string(nF));
  std::vector<Eigen::Triplet<Scalar>> t;
  for (std::size_t f = 0; f < nF; ++f) {
    if (!keep.empty() && !keep[f]) continue;
    const std::int32_t u = endVertex[f], v = endVertex[nF + f];
    if (u < 0 || v < 0 || u >= nVertices || v >= nVertices) refuse("an end's vertex is outside 0 ... " + std::to_string(nVertices - 1));
    if (u == v) continue;
    t.emplace_back(u, v, Scalar(1));
    t.emplace_back(v, u, Scalar(1));
  }
  Sparse<Scalar> m(ix(nVertices), ix(nVertices));
  m.setFromTriplets(t.begin(), t.end());   // duplicates are summed
  return m;
}

namespace {
template <class Scalar>
Sparse<Scalar> withoutDiagonal(const Sparse<Scalar>& m) {
  std::vector<Eigen::Triplet<Scalar>> t;
  for (int k = 0; k < m.outerSize(); ++k)
    for (typename Sparse<Scalar>::InnerIterator it(m, k); it; ++it)
      if (it.row() != it.col() && it.value() != Scalar(0)) t.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
  Sparse<Scalar> out(m.rows(), m.cols());
  out.setFromTriplets(t.begin(), t.end());
  return out;
}
}  // namespace

template <class Scalar>
Sparse<Scalar> connectionWeights(const Sparse<Scalar>& counts, const std::vector<std::int32_t>& faces, int hops) {
  if (counts.rows() != counts.cols()) refuse("the fiber counts are square");
  const Sparse<Scalar> p = smoothingOperator<Scalar>(faces, counts.rows(), hops);
  return withoutDiagonal<Scalar>(nxr::algebra::sparseCongruence<Scalar>(p, counts));
}

template <class Scalar>
Scalar strongest(const Sparse<Scalar>& m) {
  Scalar best = Scalar(0);
  for (int k = 0; k < m.outerSize(); ++k)
    for (typename Sparse<Scalar>::InnerIterator it(m, k); it; ++it) best = std::max(best, it.value());
  return best;
}

template <class Scalar>
Sparse<Scalar> pruneBelow(const Sparse<Scalar>& m, Scalar fraction) {
  if (!(fraction >= Scalar(0))) refuse("the pruning fraction is >= 0");
  const Scalar t = fraction * strongest(m);
  std::vector<Eigen::Triplet<Scalar>> keep;
  for (int k = 0; k < m.outerSize(); ++k)
    for (typename Sparse<Scalar>::InnerIterator it(m, k); it; ++it)
      if (it.value() != Scalar(0) && !(it.value() < t)) keep.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
  Sparse<Scalar> out(m.rows(), m.cols());
  out.setFromTriplets(keep.begin(), keep.end());
  return out;
}

template <class Scalar>
std::vector<std::int32_t> componentsOf(const Sparse<Scalar>& m) {
  if (m.rows() != m.cols()) refuse("the connection weights are square");
  const std::int32_t n = static_cast<std::int32_t>(m.rows());
  std::vector<std::int32_t> parent(static_cast<std::size_t>(n));
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](std::int32_t x) {
    while (parent[static_cast<std::size_t>(x)] != x) { parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])]; x = parent[static_cast<std::size_t>(x)]; }
    return x;
  };
  for (int k = 0; k < m.outerSize(); ++k)
    for (typename Sparse<Scalar>::InnerIterator it(m, k); it; ++it) {
      if (it.value() == Scalar(0)) continue;
      const std::int32_t a = find(static_cast<std::int32_t>(it.row())), b = find(static_cast<std::int32_t>(it.col()));
      if (a != b) parent[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b);   // the root is the component's lowest vertex
    }
  std::vector<std::int32_t> label(static_cast<std::size_t>(n)), ordinal(static_cast<std::size_t>(n), -1);
  std::int32_t next = 0;
  for (std::int32_t v = 0; v < n; ++v) {
    const std::int32_t r = find(v);
    if (ordinal[static_cast<std::size_t>(r)] < 0) ordinal[static_cast<std::size_t>(r)] = next++;
    label[static_cast<std::size_t>(v)] = ordinal[static_cast<std::size_t>(r)];
  }
  return label;
}

std::vector<std::int32_t> largestComponent(const std::vector<std::int32_t>& label, std::vector<std::int64_t>* sizes) {
  std::int32_t k = -1;
  for (const auto l : label) k = std::max(k, l);
  std::vector<std::int64_t> n(static_cast<std::size_t>(k + 1), 0);
  for (const auto l : label) if (l >= 0) ++n[static_cast<std::size_t>(l)];
  std::int32_t best = -1;
  for (std::int32_t c = 0; c <= k; ++c) if (best < 0 || n[static_cast<std::size_t>(c)] > n[static_cast<std::size_t>(best)]) best = c;
  std::vector<std::int32_t> out;
  for (std::size_t v = 0; v < label.size(); ++v) if (label[v] == best) out.push_back(static_cast<std::int32_t>(v));
  if (sizes) *sizes = n;
  return out;
}

#define NXR_CONNECTOME_INSTANTIATE(Scalar)                                                                                                                                      \
  template std::vector<Scalar> fiberEnds(const std::vector<Scalar>&, std::int64_t);                                                                                                  \
  template std::vector<std::uint8_t> withinGate(const std::vector<Scalar>&, Scalar);                                                                                                 \
  template std::vector<std::int32_t> strongestDestination(const nxr::registration::Csr<Scalar>&);                                                                               \
  template std::vector<std::int32_t> registrationMatch(const std::vector<Scalar>&, const std::vector<nxr::registration::Piece>&, const std::vector<Scalar>&,                         \
                                                       const std::vector<nxr::registration::Piece>&, int, MatchReport*);                                                   \
  template Sparse<Scalar> smoothingStep(const std::vector<std::int32_t>&, std::int64_t);                                                                                        \
  template Sparse<Scalar> smoothingOperator(const std::vector<std::int32_t>&, std::int64_t, int);                                                                               \
  template Sparse<Scalar> memberships(const std::vector<std::int32_t>&, std::int64_t);                                                                                          \
  template Sparse<Scalar> smoothMemberships(const Sparse<Scalar>&, const Sparse<Scalar>&, int);                                                                                           \
  template Sparse<Scalar> partitionCounts(const std::vector<std::int32_t>&, const std::vector<std::uint8_t>&, const Sparse<Scalar>&);                                                \
  template Sparse<Scalar> endConnectome(const std::vector<std::int32_t>&, const std::vector<std::uint8_t>&, std::int64_t);                                                      \
  template Sparse<Scalar> connectionWeights(const Sparse<Scalar>&, const std::vector<std::int32_t>&, int);                                                                           \
  template Scalar strongest(const Sparse<Scalar>&);                                                                                                                                  \
  template Sparse<Scalar> pruneBelow(const Sparse<Scalar>&, Scalar);                                                                                                                      \
  template std::vector<std::int32_t> componentsOf(const Sparse<Scalar>&);
NXR_CONNECTOME_INSTANTIATE(float)
NXR_CONNECTOME_INSTANTIATE(double)
template std::vector<std::int32_t> interleaveEnds(const std::vector<std::int32_t>&);
template std::vector<float> interleaveEnds(const std::vector<float>&);
template std::vector<double> interleaveEnds(const std::vector<double>&);
template std::vector<std::int32_t> blockEnds(const std::vector<std::int32_t>&);
template std::vector<float> blockEnds(const std::vector<float>&);
template std::vector<double> blockEnds(const std::vector<double>&);

}  // namespace nxr::connectome
