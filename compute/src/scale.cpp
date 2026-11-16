// nxr::scale — see scale.h. Pure functions of arrays.
#include "nxr/scale.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nxr::scale {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
struct Cell { std::int32_t tile, frame; };
// the cells of each event of a pair, in column-major order of the labelling (time tile, then tile)
std::vector<std::vector<Cell>> cellsOf(const Pair& p) {
  std::vector<std::vector<Cell>> out(static_cast<std::size_t>(p.events));
  for (Eigen::Index f = 0; f < p.label.cols(); ++f)
    for (Eigen::Index s = 0; s < p.label.rows(); ++s) {
      const std::int32_t k = p.label(s, f);
      if (k >= 0) out[static_cast<std::size_t>(k)].push_back({static_cast<std::int32_t>(s), static_cast<std::int32_t>(f)});
    }
  return out;
}
struct Dsu {
  std::vector<int> p;
  explicit Dsu(int n) : p(static_cast<std::size_t>(n)) { std::iota(p.begin(), p.end(), 0); }
  int find(int x) { while (p[static_cast<std::size_t>(x)] != x) { p[static_cast<std::size_t>(x)] = p[static_cast<std::size_t>(p[static_cast<std::size_t>(x)])]; x = p[static_cast<std::size_t>(x)]; } return x; }
  void unite(int a, int b) { a = find(a); b = find(b); if (a != b) p[static_cast<std::size_t>(std::max(a, b))] = std::min(a, b); }   // the lowest index is the root
};
}  // namespace

Response parseResponse(const std::string& name) {
  if (name == "peak") return Response::Peak;
  if (name == "mean") return Response::Mean;
  throw std::invalid_argument("response '" + name + "': peak (the largest z over the event's cells) or mean (their mean z)");
}
const char* responseName(Response r) { return r == Response::Peak ? "peak" : "mean"; }

bool nests(std::int64_t s, int levelA, std::int64_t t, int levelB) {
  if (s < 0 || t < 0 || levelA < 0 || levelB < 0) throw std::invalid_argument("nests: tiles and levels are not negative");
  if (levelA >= levelB) return (s >> (levelA - levelB)) == t;
  return (t >> (levelB - levelA)) == s;
}
bool overlaps(std::int64_t f, std::int64_t windowA, std::int64_t originA, std::int64_t g, std::int64_t windowB, std::int64_t originB) {
  if (windowA < 1 || windowB < 1) throw std::invalid_argument("overlaps: a time tile holds at least one sample");
  const std::int64_t a0 = originA + f * windowA, a1 = a0 + windowA, b0 = originB + g * windowB, b1 = b0 + windowB;
  return a0 < b1 && b0 < a1;
}

std::vector<EventFacts> eventFacts(const Pair& p) {
  if (p.events < 0) throw std::invalid_argument("eventFacts: the number of events is not negative");
  if (p.window < 1) throw std::invalid_argument("eventFacts: a time tile holds at least one sample");
  if (p.z.rows() != p.label.rows() || p.z.cols() != p.label.cols()) throw std::invalid_argument("eventFacts: z is " + std::to_string(p.z.rows()) + " x " + std::to_string(p.z.cols()) + ", the labelling " + std::to_string(p.label.rows()) + " x " + std::to_string(p.label.cols()));
  for (Eigen::Index i = 0; i < p.label.size(); ++i) if (p.label.data()[i] >= p.events) throw std::invalid_argument("eventFacts: a cell is labelled event " + std::to_string(p.label.data()[i]) + " of " + std::to_string(p.events));
  std::vector<EventFacts> out(static_cast<std::size_t>(p.events));
  std::vector<double> sum(out.size(), 0.0);
  std::vector<int> n(out.size(), 0);
  std::vector<std::int64_t> lo(out.size(), std::numeric_limits<std::int64_t>::max()), hi(out.size(), -1);
  for (auto& e : out) e.peak = kNaN;
  for (Eigen::Index f = 0; f < p.label.cols(); ++f)
    for (Eigen::Index s = 0; s < p.label.rows(); ++s) {
      const std::int32_t k = p.label(s, f);
      if (k < 0) continue;
      const auto u = static_cast<std::size_t>(k);
      ++out[u].cells;
      lo[u] = std::min<std::int64_t>(lo[u], f);
      hi[u] = std::max<std::int64_t>(hi[u], f);
      const double v = p.z(s, f);
      if (!std::isfinite(v)) continue;
      out[u].peak = std::isfinite(out[u].peak) ? std::max(out[u].peak, v) : v;
      sum[u] += v;
      ++n[u];
    }
  for (std::size_t k = 0; k < out.size(); ++k) {
    out[k].mean = n[k] > 0 ? sum[k] / n[k] : kNaN;
    if (out[k].cells > 0) { out[k].first = p.origin + lo[k] * p.window; out[k].stop = p.origin + (hi[k] + 1) * p.window; }
  }
  return out;
}

Selection selectScale(const std::vector<Pair>& pairs, Response response, int minLink) {
  if (pairs.empty()) throw std::invalid_argument("selectScale: at least one pair");
  if (minLink < 1) throw std::invalid_argument("selectScale: minLink is at least 1");
  const int P = static_cast<int>(pairs.size());
  std::vector<std::vector<EventFacts>> facts;
  std::vector<std::vector<std::vector<Cell>>> cells;
  std::vector<int> offset(static_cast<std::size_t>(P) + 1, 0);
  for (int a = 0; a < P; ++a) {
    facts.push_back(eventFacts(pairs[static_cast<std::size_t>(a)]));
    cells.push_back(cellsOf(pairs[static_cast<std::size_t>(a)]));
    offset[static_cast<std::size_t>(a) + 1] = offset[static_cast<std::size_t>(a)] + pairs[static_cast<std::size_t>(a)].events;
  }
  const int N = offset.back();
  Dsu dsu(N);
  // the links: an event of one pair with an event of a LATER pair (the relation is symmetric); the sample bounds first (cheap), then the cells that nest and overlap, counted up to minLink
  for (int a = 0; a < P; ++a)
    for (int b = a + 1; b < P; ++b) {
      const Pair &A = pairs[static_cast<std::size_t>(a)], &B = pairs[static_cast<std::size_t>(b)];
      for (int ea = 0; ea < A.events; ++ea) {
        const EventFacts& fa = facts[static_cast<std::size_t>(a)][static_cast<std::size_t>(ea)];
        if (fa.cells == 0) continue;
        for (int eb = 0; eb < B.events; ++eb) {
          const EventFacts& fb = facts[static_cast<std::size_t>(b)][static_cast<std::size_t>(eb)];
          if (fb.cells == 0 || fa.first >= fb.stop || fb.first >= fa.stop) continue;
          if (dsu.find(offset[static_cast<std::size_t>(a)] + ea) == dsu.find(offset[static_cast<std::size_t>(b)] + eb)) continue;   // already one event
          int count = 0;
          for (const Cell& ca : cells[static_cast<std::size_t>(a)][static_cast<std::size_t>(ea)]) {
            for (const Cell& cb : cells[static_cast<std::size_t>(b)][static_cast<std::size_t>(eb)])
              if (nests(ca.tile, A.level, cb.tile, B.level) && overlaps(ca.frame, A.window, A.origin, cb.frame, B.window, B.origin) && ++count >= minLink) break;
            if (count >= minLink) break;
          }
          if (count >= minLink) dsu.unite(offset[static_cast<std::size_t>(a)] + ea, offset[static_cast<std::size_t>(b)] + eb);
        }
      }
    }
  // the components, numbered by their earliest sample, then their lowest member
  std::vector<std::int64_t> cstart(static_cast<std::size_t>(N), std::numeric_limits<std::int64_t>::max()), cstop(static_cast<std::size_t>(N), 0);
  std::vector<std::pair<int, int>> owner(static_cast<std::size_t>(N));
  for (int a = 0; a < P; ++a)
    for (int e = 0; e < pairs[static_cast<std::size_t>(a)].events; ++e) {
      const int id = offset[static_cast<std::size_t>(a)] + e, r = dsu.find(id);
      owner[static_cast<std::size_t>(id)] = {a, e};
      const EventFacts& f = facts[static_cast<std::size_t>(a)][static_cast<std::size_t>(e)];
      if (f.cells == 0) continue;
      cstart[static_cast<std::size_t>(r)] = std::min(cstart[static_cast<std::size_t>(r)], f.first);
      cstop[static_cast<std::size_t>(r)] = std::max(cstop[static_cast<std::size_t>(r)], f.stop);
    }
  std::vector<int> roots;
  for (int id = 0; id < N; ++id) if (dsu.find(id) == id) roots.push_back(id);
  std::sort(roots.begin(), roots.end(), [&](int x, int y) { return cstart[static_cast<std::size_t>(x)] != cstart[static_cast<std::size_t>(y)] ? cstart[static_cast<std::size_t>(x)] < cstart[static_cast<std::size_t>(y)] : x < y; });
  std::vector<int> ordinal(static_cast<std::size_t>(N), -1);
  for (std::size_t k = 0; k < roots.size(); ++k) ordinal[static_cast<std::size_t>(roots[k])] = static_cast<int>(k);
  const int E = static_cast<int>(roots.size());
  Selection out;
  out.events = E;
  out.cross.resize(static_cast<std::size_t>(P));
  out.response = Eigen::MatrixXd::Constant(P, E, kNaN);
  out.best = Eigen::VectorXi::Constant(E, -1);
  out.peak = Eigen::VectorXd::Constant(E, kNaN);
  out.pairs = Eigen::VectorXi::Zero(E);
  out.start = Eigen::VectorXd::Zero(E);
  out.stop = Eigen::VectorXd::Zero(E);
  out.members.resize(static_cast<std::size_t>(E));
  for (int a = 0; a < P; ++a) {
    out.cross[static_cast<std::size_t>(a)].assign(static_cast<std::size_t>(pairs[static_cast<std::size_t>(a)].events), -1);
    for (int e = 0; e < pairs[static_cast<std::size_t>(a)].events; ++e) {
      const int id = offset[static_cast<std::size_t>(a)] + e, k = ordinal[static_cast<std::size_t>(dsu.find(id))];
      out.cross[static_cast<std::size_t>(a)][static_cast<std::size_t>(e)] = k;
      out.members[static_cast<std::size_t>(k)].emplace_back(a, e);
      const EventFacts& f = facts[static_cast<std::size_t>(a)][static_cast<std::size_t>(e)];
      const double v = response == Response::Peak ? f.peak : f.mean;
      if (std::isfinite(v) && !(out.response(a, k) >= v)) out.response(a, k) = v;
    }
  }
  for (int k = 0; k < E; ++k) {
    const int r = roots[static_cast<std::size_t>(k)];
    out.start(k) = static_cast<double>(cstart[static_cast<std::size_t>(r)]);
    out.stop(k) = static_cast<double>(cstop[static_cast<std::size_t>(r)]);
    for (int a = 0; a < P; ++a) {
      const double v = out.response(a, k);
      if (!std::isfinite(v)) continue;
      ++out.pairs(k);
      if (out.best(k) < 0 || v > out.peak(k)) { out.best(k) = a; out.peak(k) = v; }      // strictly larger: a tie keeps the first pair
    }
  }
  return out;
}

}  // namespace nxr::scale
