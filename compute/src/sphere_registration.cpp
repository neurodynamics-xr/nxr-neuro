#include "nxr/sphere_registration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace nxr::registration {

namespace {

using P3 = std::array<double, 3>;

P3 unit(const P3& p) {
  const double n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  if (!(n > 0)) throw std::invalid_argument("a registered sphere has a vertex at its centre");
  return {p[0] / n, p[1] / n, p[2] / n};
}
double d2(const P3& a, const P3& b) {
  const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
  return x * x + y * y + z * z;
}

// A k-d tree (Bentley 1975) over the unit points of one piece; ties go to the lowest index.
class Tree {
 public:
  explicit Tree(const std::vector<P3>& pts) : pts_(pts), idx_(pts.size()), split_(pts.size(), 0.0), axis_(pts.size(), 0) {
    std::iota(idx_.begin(), idx_.end(), 0);
    build(0, static_cast<std::int64_t>(idx_.size()), 0);
  }
  // the k nearest as (squared distance, index), nearest first
  std::vector<std::pair<double, std::int64_t>> knn(const P3& q, std::size_t k) const {
    std::priority_queue<std::pair<double, std::int64_t>> heap;   // max-heap on (distance, index): the worst on top
    search(0, static_cast<std::int64_t>(idx_.size()), 0, q, k, heap);
    std::vector<std::pair<double, std::int64_t>> out;
    while (!heap.empty()) { out.push_back(heap.top()); heap.pop(); }
    std::reverse(out.begin(), out.end());
    return out;
  }

 private:
  static constexpr std::int64_t kLeaf = 8;
  const std::vector<P3>& pts_;
  std::vector<std::int64_t> idx_;
  std::vector<double> split_;       // its splitting coordinate (the child ranges are reordered afterwards, so it is kept here)
  std::vector<std::int8_t> axis_;   // the split axis of the internal node whose right child starts at this index (unique per node)

  void build(std::int64_t lo, std::int64_t hi, int) {
    if (hi - lo <= kLeaf) return;
    P3 mn = pts_[static_cast<std::size_t>(idx_[static_cast<std::size_t>(lo)])], mx = mn;
    for (std::int64_t i = lo; i < hi; ++i)
      for (int a = 0; a < 3; ++a) {
        mn[static_cast<std::size_t>(a)] = std::min(mn[static_cast<std::size_t>(a)], pts_[static_cast<std::size_t>(idx_[static_cast<std::size_t>(i)])][static_cast<std::size_t>(a)]);
        mx[static_cast<std::size_t>(a)] = std::max(mx[static_cast<std::size_t>(a)], pts_[static_cast<std::size_t>(idx_[static_cast<std::size_t>(i)])][static_cast<std::size_t>(a)]);
      }
    int ax = 0;
    for (int a = 1; a < 3; ++a) if (mx[static_cast<std::size_t>(a)] - mn[static_cast<std::size_t>(a)] > mx[static_cast<std::size_t>(ax)] - mn[static_cast<std::size_t>(ax)]) ax = a;
    const std::int64_t mid = (lo + hi) / 2;
    std::nth_element(idx_.begin() + lo, idx_.begin() + mid, idx_.begin() + hi, [&](std::int64_t a, std::int64_t b) {
      const double pa = pts_[static_cast<std::size_t>(a)][static_cast<std::size_t>(ax)], pb = pts_[static_cast<std::size_t>(b)][static_cast<std::size_t>(ax)];
      return pa < pb || (pa == pb && a < b);
    });
    axis_[static_cast<std::size_t>(mid)] = static_cast<std::int8_t>(ax);
    split_[static_cast<std::size_t>(mid)] = pts_[static_cast<std::size_t>(idx_[static_cast<std::size_t>(mid)])][static_cast<std::size_t>(ax)];
    build(lo, mid, 0);
    build(mid, hi, 0);
  }
  void search(std::int64_t lo, std::int64_t hi, int, const P3& q, std::size_t k, std::priority_queue<std::pair<double, std::int64_t>>& heap) const {
    if (hi - lo <= kLeaf) {
      for (std::int64_t i = lo; i < hi; ++i) {
        const std::int64_t j = idx_[static_cast<std::size_t>(i)];
        const std::pair<double, std::int64_t> c{d2(q, pts_[static_cast<std::size_t>(j)]), j};
        if (heap.size() < k) heap.push(c);
        else if (c < heap.top()) { heap.pop(); heap.push(c); }
      }
      return;
    }
    const std::int64_t mid = (lo + hi) / 2;
    const int ax = axis_[static_cast<std::size_t>(mid)];
    const double split = split_[static_cast<std::size_t>(mid)];
    const double diff = q[static_cast<std::size_t>(ax)] - split;
    const bool left_first = diff < 0;
    if (left_first) search(lo, mid, 0, q, k, heap); else search(mid, hi, 0, q, k, heap);
    if (heap.size() < k || diff * diff <= heap.top().first) {
      if (left_first) search(mid, hi, 0, q, k, heap); else search(lo, mid, 0, q, k, heap);
    }
  }
};

std::vector<P3> unit_points(const std::vector<double>& pos, const Piece& piece) {
  std::vector<P3> out(static_cast<std::size_t>(piece.count));
  for (std::int64_t i = 0; i < piece.count; ++i) {
    const std::size_t o = static_cast<std::size_t>(3 * (piece.first + i));
    out[static_cast<std::size_t>(i)] = unit({pos[o], pos[o + 1], pos[o + 2]});
  }
  return out;
}

void check_pieces(const std::vector<Piece>& pieces, std::int64_t n, const char* what) {
  std::int64_t at = 0;
  for (const Piece& p : pieces) {
    if (p.first != at || p.count <= 0) throw std::invalid_argument(std::string("the pieces of the ") + what + " sphere do not tile it in order");
    at += p.count;
  }
  if (at != n) throw std::invalid_argument(std::string("the pieces of the ") + what + " sphere hold " + std::to_string(at) + " vertices, the sphere " + std::to_string(n));
}

// spherical barycentric coordinates of unit p in triangle (a, b, c): p proportional to u a + v b + w c
bool barycentric(const P3& p, const P3& a, const P3& b, const P3& c, std::array<double, 3>& w) {
  const double m[3][3] = {{a[0], b[0], c[0]}, {a[1], b[1], c[1]}, {a[2], b[2], c[2]}};
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (std::abs(det) < 1e-300) return false;
  auto d3 = [&](int col) {
    double t[3][3];
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) t[i][j] = j == col ? p[static_cast<std::size_t>(i)] : m[i][j];
    return t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1]) - t[0][1] * (t[1][0] * t[2][2] - t[1][2] * t[2][0]) + t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0]);
  };
  w = {d3(0) / det, d3(1) / det, d3(2) / det};
  return true;
}

}  // namespace

Method parse_method(const std::string& name) {
  if (name == "shepard") return Method::Shepard;
  if (name == "nearest") return Method::Nearest;
  if (name == "barycentric") return Method::Barycentric;
  throw std::invalid_argument("registration method '" + name + "': choose shepard, nearest or barycentric");
}
std::string method_name(Method m) { return m == Method::Shepard ? "shepard" : m == Method::Nearest ? "nearest" : "barycentric"; }

std::vector<Piece> contiguous_pieces(const std::vector<std::int32_t>& faces, std::int64_t n) {
  std::vector<std::int64_t> parent(static_cast<std::size_t>(n));
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](std::int64_t x) {
    while (parent[static_cast<std::size_t>(x)] != x) x = parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    return x;
  };
  std::vector<std::uint8_t> used(static_cast<std::size_t>(n), 0);
  for (std::size_t f = 0; f + 2 < faces.size(); f += 3) {
    for (int k = 0; k < 3; ++k) {
      if (faces[f + static_cast<std::size_t>(k)] < 0 || faces[f + static_cast<std::size_t>(k)] >= n) throw std::invalid_argument("a face names a vertex outside the surface");
      used[static_cast<std::size_t>(faces[f + static_cast<std::size_t>(k)])] = 1;
    }
    for (int k = 1; k < 3; ++k) parent[static_cast<std::size_t>(find(faces[f]))] = find(faces[f + static_cast<std::size_t>(k)]);
  }
  std::vector<Piece> out;
  std::vector<std::int64_t> roots;
  for (std::int64_t v = 0; v < n; ++v) {
    if (!used[static_cast<std::size_t>(v)]) throw std::invalid_argument("vertex " + std::to_string(v) + " is on no face: the sphere's pieces cannot be found");
    const std::int64_t r = find(v);
    if (!roots.empty() && roots.back() == r) { ++out.back().count; continue; }
    if (std::find(roots.begin(), roots.end(), r) != roots.end()) throw std::invalid_argument("a piece of the sphere is not one run of vertices (vertex " + std::to_string(v) + ")");
    roots.push_back(r);
    out.push_back({v, 1});
  }
  return out;
}

template <class S>
Csr<S> resample_operator(const std::vector<S>& from_pos, const std::vector<std::int32_t>& from_faces, const std::vector<Piece>& from_pieces, const std::vector<S>& to_pos,
                         const std::vector<Piece>& to_pieces, const Options& opt, Report* report) {
  if (from_pos.size() % 3 || to_pos.size() % 3) throw std::invalid_argument("sphere positions are [n x 3]");
  const std::int64_t nf = static_cast<std::int64_t>(from_pos.size() / 3), nt = static_cast<std::int64_t>(to_pos.size() / 3);
  check_pieces(from_pieces, nf, "source");
  check_pieces(to_pieces, nt, "destination");
  if (from_pieces.size() != to_pieces.size())
    throw std::invalid_argument("the spheres have " + std::to_string(from_pieces.size()) + " and " + std::to_string(to_pieces.size()) + " pieces: pieces are matched one to one");
  if (opt.method == Method::Shepard && opt.neighbours < 1) throw std::invalid_argument("shepard needs at least one neighbour");
  const std::vector<double> fp(from_pos.begin(), from_pos.end()), tp(to_pos.begin(), to_pos.end());

  // faces around each source vertex (barycentric)
  std::vector<std::vector<std::int32_t>> around;
  if (opt.method == Method::Barycentric) {
    around.resize(static_cast<std::size_t>(nf));
    for (std::size_t f = 0; f + 2 < from_faces.size(); f += 3)
      for (int k = 0; k < 3; ++k) {
        const std::int32_t v = from_faces[f + static_cast<std::size_t>(k)];
        if (v < 0 || v >= nf) throw std::invalid_argument("a source face names a vertex outside the sphere");
        around[static_cast<std::size_t>(v)].push_back(static_cast<std::int32_t>(f / 3));
      }
  }

  Csr<S> w;
  w.rows = nt;
  w.cols = nf;
  w.indptr.assign(static_cast<std::size_t>(nt) + 1, 0);
  Report rep;
  std::vector<std::uint8_t> taken(static_cast<std::size_t>(nf), 0);
  std::vector<std::vector<std::pair<std::int32_t, double>>> rows(static_cast<std::size_t>(nt));

  for (std::size_t pc = 0; pc < from_pieces.size(); ++pc) {
    const Piece &fpc = from_pieces[pc], &tpc = to_pieces[pc];
    const std::vector<P3> src = unit_points(fp, fpc), dst = unit_points(tp, tpc);
    const Tree tree(src);
    const std::size_t K = std::min<std::size_t>(opt.method == Method::Shepard ? static_cast<std::size_t>(opt.neighbours) : (opt.method == Method::Barycentric ? 8 : 1), src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
      const auto nn = tree.knn(dst[i], K);
      rep.max_chord = std::max(rep.max_chord, std::sqrt(nn.front().first));
      auto& row = rows[static_cast<std::size_t>(tpc.first) + i];
      const auto global = [&](std::int64_t local) { return static_cast<std::int32_t>(fpc.first + local); };
      auto nearest_only = [&] { row.assign(1, {global(nn.front().second), 1.0}); };
      if (opt.method == Method::Nearest || nn.size() == 1) { nearest_only(); }
      else if (opt.method == Method::Shepard) {
        const double dK = nn.back().first;
        const double tiny = std::numeric_limits<S>::min();
        std::vector<double> u(nn.size() - 1);
        double umax = 0;
        for (std::size_t k = 0; k + 1 < nn.size(); ++k) {
          const double dk = std::max(nn[k].first, tiny);
          u[k] = dK > 0 ? std::max(0.0, (dK - dk) / (dK * dk)) : 0.0;
          umax = std::max(umax, u[k]);
        }
        if (!(umax > 0)) { nearest_only(); }
        else {
          double sum = 0;
          for (double& x : u) { x /= umax; x *= x; sum += x; }   // scaled before squaring: the same weights, no overflow when a distance is the type's smallest
          for (std::size_t k = 0; k < u.size(); ++k) if (u[k] > 0) row.push_back({global(nn[k].second), u[k] / sum});
        }
      } else {
        bool found = false;
        double best = -std::numeric_limits<double>::infinity();
        std::array<double, 3> bw{};
        std::array<std::int32_t, 3> bv{};
        for (const auto& cand : nn) {
          for (std::int32_t f : around[static_cast<std::size_t>(global(cand.second))]) {
            const std::int32_t a = from_faces[static_cast<std::size_t>(3 * f)], b = from_faces[static_cast<std::size_t>(3 * f + 1)], c = from_faces[static_cast<std::size_t>(3 * f + 2)];
            std::array<double, 3> bc;
            if (!barycentric(dst[i], src[static_cast<std::size_t>(a - fpc.first)], src[static_cast<std::size_t>(b - fpc.first)], src[static_cast<std::size_t>(c - fpc.first)], bc)) continue;
            const double m = std::min({bc[0], bc[1], bc[2]});
            if (m > best) { best = m; bw = bc; bv = {a, b, c}; }
          }
          if (best >= -1e-12) { found = true; break; }
        }
        if (!found) { ++rep.fallback_to_nearest; nearest_only(); }
        else {
          double sum = 0;
          for (double& x : bw) { x = std::max(0.0, x); sum += x; }
          for (int k = 0; k < 3; ++k) if (bw[static_cast<std::size_t>(k)] > 0) row.push_back({bv[static_cast<std::size_t>(k)], bw[static_cast<std::size_t>(k)] / sum});
        }
      }
    }
  }
  for (std::int64_t i = 0; i < nt; ++i) {
    auto& row = rows[static_cast<std::size_t>(i)];
    std::sort(row.begin(), row.end());
    for (const auto& e : row) { w.indices.push_back(e.first); w.data.push_back(static_cast<S>(e.second)); taken[static_cast<std::size_t>(e.first)] = 1; }
    w.indptr[static_cast<std::size_t>(i) + 1] = static_cast<std::int32_t>(w.indices.size());
  }
  rep.empty_columns = std::count(taken.begin(), taken.end(), 0);
  if (report) *report = rep;
  return w;
}

template <class S>
std::vector<S> carry(const Csr<S>& w, const std::vector<S>& x) {
  if (static_cast<std::int64_t>(x.size()) != w.cols) throw std::invalid_argument("a field of " + std::to_string(x.size()) + " values for an operator of " + std::to_string(w.cols) + " columns");
  std::vector<S> y(static_cast<std::size_t>(w.rows), S(0));
  for (std::int64_t i = 0; i < w.rows; ++i) {
    double acc = 0;
    for (std::int32_t k = w.indptr[static_cast<std::size_t>(i)]; k < w.indptr[static_cast<std::size_t>(i) + 1]; ++k) acc += static_cast<double>(w.data[static_cast<std::size_t>(k)]) * static_cast<double>(x[static_cast<std::size_t>(w.indices[static_cast<std::size_t>(k)])]);
    y[static_cast<std::size_t>(i)] = static_cast<S>(acc);
  }
  return y;
}

template Csr<float> resample_operator<float>(const std::vector<float>&, const std::vector<std::int32_t>&, const std::vector<Piece>&, const std::vector<float>&, const std::vector<Piece>&, const Options&, Report*);
template Csr<double> resample_operator<double>(const std::vector<double>&, const std::vector<std::int32_t>&, const std::vector<Piece>&, const std::vector<double>&, const std::vector<Piece>&, const Options&, Report*);
template std::vector<float> carry<float>(const Csr<float>&, const std::vector<float>&);
template std::vector<double> carry<double>(const Csr<double>&, const std::vector<double>&);

}  // namespace nxr::registration
