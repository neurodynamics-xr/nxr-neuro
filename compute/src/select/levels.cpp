// nxr::select — the levels of one manifold (levels.h; spec docs/specs/2026-11-26-levels-of-one-manifold.md).
#include "nxr/select/levels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "nxr/measurement/measures.h"
#include "nxr/measurement/roll_up.h"
#include "nxr/selection/resolved.h"
#include "nxr/selection/selection.h"

namespace nxr::select {

namespace {

using Triplet = Eigen::Triplet<double>;

[[noreturn]] void refuse(const std::string& why) { throw std::invalid_argument("levels: " + why); }

std::string pair_text(std::int64_t a, std::int64_t b) { return "(" + std::to_string(a) + ", " + std::to_string(b) + ")"; }

// the tail and tip of each fine edge from d0 (one -1 and one +1 a row)
void edge_ends(const SparseMatrix& d0, std::vector<std::int32_t>& tail, std::vector<std::int32_t>& tip) {
  const auto E = d0.rows();
  tail.assign(static_cast<std::size_t>(E), -1); tip.assign(static_cast<std::size_t>(E), -1);
  for (int c = 0; c < d0.outerSize(); ++c)
    for (SparseMatrix::InnerIterator it(d0, c); it; ++it) {
      if (it.value() < 0) tail[static_cast<std::size_t>(it.row())] = static_cast<std::int32_t>(it.col());
      else if (it.value() > 0) tip[static_cast<std::size_t>(it.row())] = static_cast<std::int32_t>(it.col());
    }
  for (Eigen::Index e = 0; e < E; ++e)
    if (tail[static_cast<std::size_t>(e)] < 0 || tip[static_cast<std::size_t>(e)] < 0) refuse("fine edge " + std::to_string(e) + " of d0 is not one -1 and one +1 (a loop or an empty row)");
}

// each fine face's corners in its orientation, from d1's signed edges (a step along an edge with +1 goes tail -> tip, with -1 tip -> tail; the steps chain)
std::vector<std::array<std::int32_t, 3>> face_corners(const SparseMatrix& d1, const std::vector<std::int32_t>& tail, const std::vector<std::int32_t>& tip) {
  const auto F = d1.rows();
  std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> steps(static_cast<std::size_t>(F));
  for (int c = 0; c < d1.outerSize(); ++c)
    for (SparseMatrix::InnerIterator it(d1, c); it; ++it) {
      const auto e = static_cast<std::size_t>(it.col());
      if (it.value() > 0) steps[static_cast<std::size_t>(it.row())].push_back({tail[e], tip[e]});
      else if (it.value() < 0) steps[static_cast<std::size_t>(it.row())].push_back({tip[e], tail[e]});
    }
  std::vector<std::array<std::int32_t, 3>> out(static_cast<std::size_t>(F));
  for (Eigen::Index f = 0; f < F; ++f) {
    auto& s = steps[static_cast<std::size_t>(f)];
    if (s.size() != 3) refuse("fine face " + std::to_string(f) + " of d1 has " + std::to_string(s.size()) + " edges, not 3");
    std::array<std::int32_t, 3> v{s[0].first, s[0].second, -1};
    for (int k = 1; k < 3; ++k) if (s[static_cast<std::size_t>(k)].first == v[1]) v[2] = s[static_cast<std::size_t>(k)].second;
    if (v[2] < 0) refuse("fine face " + std::to_string(f) + "'s edges do not chain into a cycle");
    out[static_cast<std::size_t>(f)] = v;
  }
  return out;
}

void build_adjacency(Level& L) {
  const std::int64_t E = L.edges.rows();
  std::vector<std::vector<std::pair<std::int64_t, std::int64_t>>> nb(static_cast<std::size_t>(L.cells));
  for (std::int64_t e = 0; e < E; ++e) {
    nb[static_cast<std::size_t>(L.edges(e, 0))].push_back({L.edges(e, 1), e});
    nb[static_cast<std::size_t>(L.edges(e, 1))].push_back({L.edges(e, 0), e});
  }
  L.adjacencyOffsets.assign(1, 0); L.adjacencyCells.clear(); L.adjacencyEdges.clear();
  for (auto& n : nb) {
    std::sort(n.begin(), n.end());
    for (const auto& [c, e] : n) { L.adjacencyCells.push_back(c); L.adjacencyEdges.push_back(e); }
    L.adjacencyOffsets.push_back(static_cast<std::int64_t>(L.adjacencyCells.size()));
  }
}

// the halfedges of the junction faces, checked: every edge in one face (boundary) or two of opposite orientation, every non-empty cell's faces one fan, V - E + F the fine one's
void build_halfedges(Level& L, const std::vector<char>& nonEmpty) {
  const std::int64_t F = L.faces.rows(), E = L.edges.rows();
  auto fail = [&](const std::string& why) { L.valid = false; L.why = why; L.heTwin.clear(); L.heEdge.clear(); };
  std::int64_t V = 0;
  for (const char c : nonEmpty) V += c ? 1 : 0;
  L.euler = V - E + F;
  std::unordered_map<std::int64_t, std::int64_t> edgeOf;
  for (std::int64_t e = 0; e < E; ++e) edgeOf[L.edges(e, 0) * L.cells + L.edges(e, 1)] = e;
  std::vector<std::int32_t> he(static_cast<std::size_t>(3 * F)), twin(static_cast<std::size_t>(3 * F), -1);
  std::vector<std::vector<std::int64_t>> onEdge(static_cast<std::size_t>(E));
  for (std::int64_t f = 0; f < F; ++f)
    for (int i = 0; i < 3; ++i) {
      const std::int64_t a = L.faces(f, i), b = L.faces(f, (i + 1) % 3);
      const std::int64_t e = edgeOf.at(std::min(a, b) * L.cells + std::max(a, b));
      he[static_cast<std::size_t>(3 * f + i)] = static_cast<std::int32_t>(e);
      onEdge[static_cast<std::size_t>(e)].push_back(3 * f + i);
    }
  for (std::int64_t e = 0; e < E; ++e) {
    const auto& h = onEdge[static_cast<std::size_t>(e)];
    const std::string ab = pair_text(L.edges(e, 0), L.edges(e, 1));
    if (h.empty()) return fail("edge " + ab + " bounds no junction face (two cells whose boundary meets no third cell): the level is a graph here, with no halfedges");
    if (h.size() > 2) return fail("edge " + ab + " bounds " + std::to_string(h.size()) + " junction faces (the two cells share their boundary in several pieces): not a surface complex");
    if (h.size() == 2) {
      const auto f0 = h[0] / 3, i0 = h[0] % 3, f1 = h[1] / 3, i1 = h[1] % 3;
      if (L.faces(f0, i0) != L.faces(f1, (i1 + 1) % 3)) return fail("the two faces on edge " + ab + " run it the same way: the level's faces are not consistently oriented");
      twin[static_cast<std::size_t>(h[0])] = static_cast<std::int32_t>(h[1]); twin[static_cast<std::size_t>(h[1])] = static_cast<std::int32_t>(h[0]);
    }
  }
  // the fan round every cell: from an outgoing halfedge h, the next outgoing one is twin(prev(h)); backward, next(twin(h))
  std::vector<std::vector<std::int64_t>> out(static_cast<std::size_t>(L.cells));
  for (std::int64_t h = 0; h < 3 * F; ++h) out[static_cast<std::size_t>(L.faces(h / 3, h % 3))].push_back(h);
  auto prev = [](std::int64_t h) { return 3 * (h / 3) + (h % 3 + 2) % 3; };
  auto nxt = [](std::int64_t h) { return 3 * (h / 3) + (h % 3 + 1) % 3; };
  for (std::int64_t c = 0; c < L.cells; ++c) {
    const auto& o = out[static_cast<std::size_t>(c)];
    if (o.empty()) continue;
    std::size_t seen = 1;
    std::int64_t h = o[0];
    bool closed = false;
    for (;;) {
      const std::int64_t t = twin[static_cast<std::size_t>(prev(h))];
      if (t < 0) break;
      if (t == o[0]) { closed = true; break; }
      h = t; ++seen;
      if (seen > o.size()) break;
    }
    if (!closed) {
      h = o[0];
      for (;;) {
        const std::int64_t t = twin[static_cast<std::size_t>(h)];
        if (t < 0) break;
        h = nxt(t); ++seen;
        if (seen > o.size()) break;
      }
    }
    if (seen != o.size()) return fail("cell " + std::to_string(c) + ": its junction faces form more than one fan (it touches itself, or the cells round it are not a disc)");
  }
  if (L.euler != L.fineEuler)
    return fail("V - E + F is " + std::to_string(L.euler) + " over the level's cells, " + std::to_string(L.fineEuler) + " on the fine mesh: a cell is not a disc, or a junction is missing");
  L.valid = true; L.why.clear();
  L.heTwin = std::move(twin); L.heEdge = std::move(he);
}

void need_valid(const Level& L) { if (!L.valid) refuse("the level has no halfedges: " + (L.why.empty() ? std::string("it has no faces") : L.why)); }
void need_cell(const Level& L, std::int64_t c) { if (c < 0 || c >= L.cells) refuse("cell " + std::to_string(c) + " is outside 0 .. " + std::to_string(L.cells - 1)); }

bool surface_ladder(const Level& L) { return L.ladder == Level::Ladder::vertices || L.ladder == Level::Ladder::dyadic; }

SparseMatrix map_matrix(const std::vector<std::int64_t>& map, std::int64_t to) {   // P [from x to]
  std::vector<Triplet> t;
  for (std::size_t i = 0; i < map.size(); ++i) if (map[i] >= 0) t.emplace_back(static_cast<int>(i), static_cast<int>(map[i]), 1.0);
  SparseMatrix P(static_cast<Eigen::Index>(map.size()), to);
  P.setFromTriplets(t.begin(), t.end());
  return P;
}
Eigen::VectorXd inverse_or_zero(const Eigen::VectorXd& w) {
  Eigen::VectorXd r(w.size());
  for (Eigen::Index i = 0; i < w.size(); ++i) r(i) = w(i) != 0 ? 1.0 / w(i) : 0.0;
  return r;
}

}  // namespace

// ── the quotient ──────────────────────────────────────────────────────────────────────────
Level levelComplex(const FineComplex& fine, const std::vector<std::int32_t>& cellOf, std::int64_t cells) {
  const auto V = fine.d0.cols(), Ef = fine.d0.rows();
  if (static_cast<Eigen::Index>(cellOf.size()) != V) refuse("cellOf has " + std::to_string(cellOf.size()) + " vertices, d0 " + std::to_string(V));
  if (fine.star0.size() != V) refuse("star0 has " + std::to_string(fine.star0.size()) + " values, the fine complex " + std::to_string(V) + " vertices");
  if (fine.star1.size() != Ef) refuse("star1 has " + std::to_string(fine.star1.size()) + " values, the fine complex " + std::to_string(Ef) + " edges");
  if (fine.d1.rows() > 0 && fine.d1.cols() != Ef) refuse("d1 has " + std::to_string(fine.d1.cols()) + " columns, d0 " + std::to_string(Ef) + " rows");
  if (cells < 1) refuse("a level has at least one cell");
  for (const auto c : cellOf) if (c < -1 || c >= cells) refuse("a vertex's cell " + std::to_string(c) + " is outside 0 .. " + std::to_string(cells - 1));
  std::vector<std::int32_t> tail, tip;
  edge_ends(fine.d0, tail, tip);

  Level L;
  L.cells = cells;
  L.cellOf = cellOf;
  L.star0 = Eigen::VectorXd::Zero(cells);
  std::vector<char> nonEmpty(static_cast<std::size_t>(cells), 0);
  std::int64_t Vin = 0;
  for (Eigen::Index v = 0; v < V; ++v) {
    const auto c = cellOf[static_cast<std::size_t>(v)];
    if (c < 0) continue;
    L.star0(c) += fine.star0(v); nonEmpty[static_cast<std::size_t>(c)] = 1; ++Vin;
  }
  // the level's edges: the pairs a < b joined by a fine edge, in pair order; the Galerkin conductance and the crossing map
  std::map<std::pair<std::int32_t, std::int32_t>, std::int64_t> pairs;
  std::int64_t EfIn = 0;
  for (Eigen::Index e = 0; e < Ef; ++e) {
    const auto a = cellOf[static_cast<std::size_t>(tail[static_cast<std::size_t>(e)])], b = cellOf[static_cast<std::size_t>(tip[static_cast<std::size_t>(e)])];
    if (a < 0 || b < 0) continue;
    ++EfIn;
    if (a != b) pairs.emplace(std::make_pair(std::min(a, b), std::max(a, b)), 0);
  }
  std::int64_t E = 0;
  for (auto& [ab, idx] : pairs) idx = E++;
  L.edges.resize(E, 2);
  for (const auto& [ab, idx] : pairs) { L.edges(idx, 0) = ab.first; L.edges(idx, 1) = ab.second; }
  L.star1 = Eigen::VectorXd::Zero(E);
  std::vector<Triplet> cross;
  for (Eigen::Index e = 0; e < Ef; ++e) {
    const auto a = cellOf[static_cast<std::size_t>(tail[static_cast<std::size_t>(e)])], b = cellOf[static_cast<std::size_t>(tip[static_cast<std::size_t>(e)])];
    if (a < 0 || b < 0 || a == b) continue;
    const std::int64_t idx = pairs.at({std::min(a, b), std::max(a, b)});
    L.star1(idx) += fine.star1(e);
    cross.emplace_back(static_cast<int>(e), static_cast<int>(idx), a < b ? 1.0 : -1.0);
  }
  L.crossing.resize(Ef, E);
  L.crossing.setFromTriplets(cross.begin(), cross.end());
  for (std::int64_t e = 0; e < E; ++e) if (L.star1(e) < 0) ++L.negativeConductances;
  std::vector<Triplet> d0;
  for (std::int64_t e = 0; e < E; ++e) { d0.emplace_back(static_cast<int>(e), L.edges(e, 0), -1.0); d0.emplace_back(static_cast<int>(e), L.edges(e, 1), 1.0); }
  L.d0.resize(E, cells);
  L.d0.setFromTriplets(d0.begin(), d0.end());
  build_adjacency(L);

  // the junction faces: a fine face whose corners lie in three different cells
  std::int64_t FfIn = 0;
  if (fine.d1.rows() > 0) {
    const auto corners = face_corners(fine.d1, tail, tip);
    std::vector<std::array<std::int32_t, 3>> faces;
    for (std::size_t f = 0; f < corners.size(); ++f) {
      const std::int32_t a = cellOf[static_cast<std::size_t>(corners[f][0])], b = cellOf[static_cast<std::size_t>(corners[f][1])], c = cellOf[static_cast<std::size_t>(corners[f][2])];
      if (a < 0 || b < 0 || c < 0) continue;
      ++FfIn;
      if (a != b && b != c && a != c) { faces.push_back({a, b, c}); L.junctions.push_back(static_cast<std::int64_t>(f)); }
    }
    const std::int64_t F = static_cast<std::int64_t>(faces.size());
    L.faces.resize(F, 3);
    std::vector<Triplet> d1;
    for (std::int64_t f = 0; f < F; ++f)
      for (int i = 0; i < 3; ++i) {
        L.faces(f, i) = faces[static_cast<std::size_t>(f)][static_cast<std::size_t>(i)];
        const std::int32_t a = faces[static_cast<std::size_t>(f)][static_cast<std::size_t>(i)], b = faces[static_cast<std::size_t>(f)][static_cast<std::size_t>((i + 1) % 3)];
        d1.emplace_back(static_cast<int>(f), static_cast<int>(pairs.at({std::min(a, b), std::max(a, b)})), a < b ? 1.0 : -1.0);
      }
    L.d1.resize(F, E);
    L.d1.setFromTriplets(d1.begin(), d1.end());
  } else {
    L.faces.resize(0, 3);
  }
  L.fineEuler = Vin - EfIn + FfIn;
  if (fine.d1.rows() > 0) build_halfedges(L, nonEmpty);
  else { L.euler = Vin - E; L.why = "the fine complex has no faces"; }
  return L;
}

Level vertexLevel(const FineComplex& fine, const std::vector<std::int32_t>& codes, int depth, std::int64_t slots) {
  std::vector<std::int32_t> self(static_cast<std::size_t>(fine.d0.cols()));
  for (std::size_t v = 0; v < self.size(); ++v) self[v] = static_cast<std::int32_t>(v);
  if (!codes.empty() && codes.size() != self.size()) refuse("codes has " + std::to_string(codes.size()) + " vertices, the fine complex " + std::to_string(self.size()));
  Level L = levelComplex(fine, self, static_cast<std::int64_t>(self.size()));
  L.ladder = Level::Ladder::vertices; L.level = kVertexLevel; L.codes = codes; L.depth = depth; L.slots = slots;
  return L;
}

Level dyadicLevel(const FineComplex& fine, const std::vector<std::int32_t>& codes, int depth, std::int64_t slots, int level) {
  if (level < 0 || level > depth) refuse("level " + std::to_string(level) + " is outside the grid's 0 .. " + std::to_string(depth));
  (void)selection::dyadic_grid(codes, depth, slots);   // checks the codes against depth and slots (a code outside 0 .. (slots << depth) - 1 is refused)
  std::vector<std::int32_t> cellOf(codes.size());
  for (std::size_t v = 0; v < codes.size(); ++v) cellOf[v] = codes[v] < 0 ? -1 : codes[v] >> (depth - level);
  Level L = levelComplex(fine, cellOf, slots << level);
  L.ladder = Level::Ladder::dyadic; L.level = level; L.depth = depth; L.slots = slots; L.codes = codes;
  return L;
}

Level lineLevel(std::int64_t samples, double rate, int k) {
  if (samples < 1) refuse("a line has at least one sample");
  if (!(rate > 0) || !std::isfinite(rate)) refuse("the rate must be positive");
  if (k < 0 || k > 62) refuse("time level " + std::to_string(k) + " is outside 0 .. 62");
  Level L;
  L.ladder = Level::Ladder::time; L.level = k; L.samples = samples; L.rate = rate;
  const std::int64_t w = std::int64_t(1) << k;
  L.cells = (samples + w - 1) / w;
  L.star0.resize(L.cells);
  for (std::int64_t m = 0; m < L.cells; ++m) L.star0(m) = static_cast<double>(std::min((m + 1) * w, samples) - m * w) / rate;
  const std::int64_t E = L.cells - 1;
  L.edges.resize(E, 2);
  L.star1 = Eigen::VectorXd::Constant(E, rate);
  std::vector<Triplet> d0;
  for (std::int64_t e = 0; e < E; ++e) { L.edges(e, 0) = static_cast<std::int32_t>(e); L.edges(e, 1) = static_cast<std::int32_t>(e + 1); d0.emplace_back(static_cast<int>(e), static_cast<int>(e), -1.0); d0.emplace_back(static_cast<int>(e), static_cast<int>(e + 1), 1.0); }
  L.d0.resize(E, L.cells);
  L.d0.setFromTriplets(d0.begin(), d0.end());
  L.faces.resize(0, 3);
  L.why = "a line has no faces: its neighbours are the tiles before and after";
  L.fineEuler = L.euler = 1;
  build_adjacency(L);
  return L;
}

// ── on one level ──────────────────────────────────────────────────────────────────────────
std::vector<std::int64_t> neighbours(const Level& L, std::int64_t c) {
  need_cell(L, c);
  return {L.adjacencyCells.begin() + L.adjacencyOffsets[static_cast<std::size_t>(c)], L.adjacencyCells.begin() + L.adjacencyOffsets[static_cast<std::size_t>(c + 1)]};
}
std::vector<std::int64_t> incidentEdges(const Level& L, std::int64_t c) {
  need_cell(L, c);
  return {L.adjacencyEdges.begin() + L.adjacencyOffsets[static_cast<std::size_t>(c)], L.adjacencyEdges.begin() + L.adjacencyOffsets[static_cast<std::size_t>(c + 1)]};
}
std::vector<std::int64_t> outgoing(const Level& L, std::int64_t c) {
  need_valid(L); need_cell(L, c);
  std::vector<std::int64_t> o;
  for (std::int64_t h = 0; h < 3 * L.faces.rows(); ++h) if (L.faces(h / 3, h % 3) == c) o.push_back(h);
  return o;
}
std::int64_t twin(const Level& L, std::int64_t h) {
  need_valid(L);
  if (h < 0 || h >= 3 * L.faces.rows()) refuse("halfedge " + std::to_string(h) + " is outside 0 .. " + std::to_string(3 * L.faces.rows() - 1));
  return L.heTwin[static_cast<std::size_t>(h)];
}
std::int64_t next(const Level& L, std::int64_t h) {
  need_valid(L);
  if (h < 0 || h >= 3 * L.faces.rows()) refuse("halfedge " + std::to_string(h) + " is outside 0 .. " + std::to_string(3 * L.faces.rows() - 1));
  return 3 * (h / 3) + (h % 3 + 1) % 3;
}
LevelAddress tail(const Level& L, std::int64_t h) {
  if (h < 0 || h >= 3 * L.faces.rows()) refuse("halfedge " + std::to_string(h) + " is outside the level's faces");
  return {L.level, L.faces(h / 3, h % 3)};
}
LevelAddress tip(const Level& L, std::int64_t h) {
  if (h < 0 || h >= 3 * L.faces.rows()) refuse("halfedge " + std::to_string(h) + " is outside the level's faces");
  return {L.level, L.faces(h / 3, (h % 3 + 1) % 3)};
}
LevelAddress parent(const Level& L, std::int64_t c) {
  need_cell(L, c);
  switch (L.ladder) {
    case Level::Ladder::vertices: {
      if (L.codes.empty()) return {};
      const auto code = L.codes[static_cast<std::size_t>(c)];
      return code < 0 ? LevelAddress{} : LevelAddress{L.depth, code};
    }
    case Level::Ladder::dyadic: return L.level == 0 ? LevelAddress{} : LevelAddress{L.level - 1, c >> 1};
    case Level::Ladder::time: return L.level >= 62 ? LevelAddress{} : LevelAddress{L.level + 1, c >> 1};
  }
  return {};
}
Children children(const Level& L, std::int64_t c) {
  need_cell(L, c);
  Children r;
  switch (L.ladder) {
    case Level::Ladder::vertices: return r;
    case Level::Ladder::dyadic:
      if (L.level < L.depth) { r.level = L.level + 1; r.codes = {2 * c, 2 * c + 1}; return r; }
      r.level = kVertexLevel;
      for (std::size_t v = 0; v < L.codes.size(); ++v) if (L.codes[v] == c) r.codes.push_back(static_cast<std::int64_t>(v));
      return r;
    case Level::Ladder::time: {
      if (L.level == 0) return r;
      r.level = L.level - 1;
      const std::int64_t finer = (L.samples + (std::int64_t(1) << r.level) - 1) >> r.level;
      for (const std::int64_t k : {2 * c, 2 * c + 1}) if (k < finer) r.codes.push_back(k);
      return r;
    }
  }
  return r;
}
double area(const Level& L, std::int64_t c) { need_cell(L, c); return L.star0(c); }
std::int64_t crossingEdge(const Level& L, std::int64_t e) {
  if (L.ladder != Level::Ladder::time) refuse("crossingEdge is a time level's (a surface level holds its crossing map)");
  if (e < 0 || e >= L.edges.rows()) refuse("edge " + std::to_string(e) + " is outside the level's edges");
  return (std::int64_t(1) << L.level) * (e + 1) - 1;
}
SparseMatrix laplacian(const Level& L) {
  SparseMatrix S = L.star1.asDiagonal() * L.d0;
  return SparseMatrix(L.d0.transpose() * S);
}

// ── between two levels ────────────────────────────────────────────────────────────────────
std::vector<std::int64_t> cellMap(const Level& from, const Level& to) {
  std::vector<std::int64_t> m(static_cast<std::size_t>(from.cells), -1);
  if (from.ladder == Level::Ladder::time) {
    if (to.ladder != Level::Ladder::time || to.samples != from.samples) refuse("a time level maps only to a coarser level of the same line");
    if (to.level < from.level) refuse("time level " + std::to_string(to.level) + " is not coarser than " + std::to_string(from.level));
    for (std::int64_t c = 0; c < from.cells; ++c) m[static_cast<std::size_t>(c)] = c >> (to.level - from.level);
    return m;
  }
  if (to.ladder != Level::Ladder::dyadic) refuse("a surface level maps only to a dyadic level of its grid");
  if (from.ladder == Level::Ladder::vertices) {
    if (from.codes.size() != static_cast<std::size_t>(from.cells) || to.codes != from.codes) refuse("the vertices carry no codes of this grid (vertexLevel with the grid's codes)");
    for (std::int64_t v = 0; v < from.cells; ++v) { const auto code = from.codes[static_cast<std::size_t>(v)]; m[static_cast<std::size_t>(v)] = code < 0 ? -1 : code >> (to.depth - to.level); }
    return m;
  }
  if (to.codes != from.codes || to.depth != from.depth) refuse("the two dyadic levels are not of one grid");
  if (to.level > from.level) refuse("dyadic level " + std::to_string(to.level) + " is not coarser than " + std::to_string(from.level));
  for (std::int64_t c = 0; c < from.cells; ++c) m[static_cast<std::size_t>(c)] = c >> (from.level - to.level);
  return m;
}

SparseMatrix reductionOperator(const Level& from, const Level& to, const std::string& rule) {
  const SparseMatrix P = map_matrix(cellMap(from, to), to.cells);
  if (rule == "sum") return SparseMatrix(P.transpose());
  if (rule == "mean") {
    SparseMatrix R = inverse_or_zero(to.star0).asDiagonal() * SparseMatrix(P.transpose());
    return SparseMatrix(R * from.star0.asDiagonal());
  }
  if (rule == "max" || rule == "min") refuse("'" + rule + "' is an order statistic: it has no matrix (use reduce)");
  refuse("'" + rule + "' is never reduced (the merge rules that reduce: sum, mean, max, min)");
}
SparseMatrix expansionOperator(const Level& coarse, const Level& fine, const std::string& rule) {
  const SparseMatrix P = map_matrix(cellMap(fine, coarse), coarse.cells);
  if (rule == "mean") return P;
  if (rule == "sum") {
    SparseMatrix X = fine.star0.asDiagonal() * P;
    return SparseMatrix(X * inverse_or_zero(coarse.star0).asDiagonal());
  }
  if (rule == "max" || rule == "min") refuse("'" + rule + "' is an order statistic: it has no expansion");
  refuse("'" + rule + "' has no expansion (the rules that expand: mean, sum)");
}
Eigen::MatrixXd reduce(const Level& from, const Level& to, const std::string& rule, const Eigen::MatrixXd& values) {
  if (values.cols() != from.cells) refuse("values have " + std::to_string(values.cols()) + " columns, the level " + std::to_string(from.cells) + " cells");
  const auto map = cellMap(from, to);
  selection::Elements groups;
  groups.size = from.cells;
  std::vector<std::vector<std::int64_t>> members(static_cast<std::size_t>(to.cells));
  for (std::size_t i = 0; i < map.size(); ++i) if (map[i] >= 0) members[static_cast<std::size_t>(map[i])].push_back(static_cast<std::int64_t>(i));
  for (const auto& m : members) { groups.elements.insert(groups.elements.end(), m.begin(), m.end()); groups.offsets.push_back(static_cast<std::int64_t>(groups.elements.size())); }
  measurement::MeasureSpec spec;
  spec.name = "value"; spec.merge = measurement::merge_from(rule); spec.weight = "star0";
  if (spec.merge == measurement::Merge::none) refuse("'" + rule + "' is never reduced");
  const measurement::Registry registry({spec});
  const Eigen::MatrixXd w = from.star0.transpose();
  return measurement::roll_values(registry, "value", values, spec.merge == measurement::Merge::mean ? &w : nullptr, groups);
}

// ── the product ───────────────────────────────────────────────────────────────────────────
ProductLevel productLevel(Level space, Level time, Coupling coupling) {
  if (!surface_ladder(space)) refuse("the product's first factor is a surface level (vertices or dyadic)");
  if (time.ladder != Level::Ladder::time) refuse("the product's second factor is a time level");
  if (coupling.kind == Coupling::Kind::dispersion)
    refuse("a product coupled by a dispersion relation is not built yet: its levels are chosen independently (separable) in this step");
  return ProductLevel{std::move(space), std::move(time), coupling};
}
std::int64_t jointCode(const ProductLevel& p, std::int64_t s, std::int64_t t) { return selection::product_code({p.space.cells, p.time.cells}, {s, t}); }
JointAddress jointAddress(const ProductLevel& p, std::int64_t code) {
  const auto c = selection::product_codes({p.space.cells, p.time.cells}, code);
  return {{p.space.level, c[0]}, {p.time.level, c[1]}};
}
std::vector<std::int64_t> productNeighbours(const ProductLevel& p, std::int64_t code) {
  const auto a = jointAddress(p, code);
  std::vector<std::int64_t> n;
  for (const auto s : neighbours(p.space, a.space.code)) n.push_back(jointCode(p, s, a.time.code));
  for (const auto t : neighbours(p.time, a.time.code)) n.push_back(jointCode(p, a.space.code, t));
  std::sort(n.begin(), n.end());
  return n;
}
ProductOperators productOperators(const ProductLevel& p) {
  const std::int64_t S = p.space.cells, T = p.time.cells, Es = p.space.edges.rows(), Et = p.time.edges.rows();
  ProductOperators o;
  o.star0.resize(S * T);
  for (std::int64_t t = 0; t < T; ++t) for (std::int64_t s = 0; s < S; ++s) o.star0(s + S * t) = p.space.star0(s) * p.time.star0(t);
  o.star1.resize(Es * T + S * Et);
  std::vector<Triplet> d0;
  for (std::int64_t t = 0; t < T; ++t)
    for (std::int64_t e = 0; e < Es; ++e) {
      const std::int64_t row = e + Es * t;
      o.star1(row) = p.space.star1(e) * p.time.star0(t);
      d0.emplace_back(static_cast<int>(row), static_cast<int>(p.space.edges(e, 0) + S * t), -1.0);
      d0.emplace_back(static_cast<int>(row), static_cast<int>(p.space.edges(e, 1) + S * t), 1.0);
    }
  for (std::int64_t e = 0; e < Et; ++e)
    for (std::int64_t s = 0; s < S; ++s) {
      const std::int64_t row = Es * T + s + S * e;
      o.star1(row) = p.space.star0(s) * p.time.star1(e);
      d0.emplace_back(static_cast<int>(row), static_cast<int>(s + S * p.time.edges(e, 0)), -1.0);
      d0.emplace_back(static_cast<int>(row), static_cast<int>(s + S * p.time.edges(e, 1)), 1.0);
    }
  o.d0.resize(Es * T + S * Et, S * T);
  o.d0.setFromTriplets(d0.begin(), d0.end());
  return o;
}

}  // namespace nxr::select
