// nxr::operators::registration on synthetic spheres (subdivided icosahedra): the three methods, the weights' rules, the pieces, the precision, the refusals.
#include "nxr/operators/resample.h"

#include <cmath>
#include <iostream>
#include <functional>
#include <map>
#include <type_traits>
#include <stdexcept>

using namespace nxr::operators::registration;

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

struct Mesh { std::vector<double> p; std::vector<std::int32_t> f; };

static Mesh icosphere(int level) {
  const double t = (1 + std::sqrt(5.0)) / 2;
  Mesh m;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto& b : base) { const double n = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]); for (int k = 0; k < 3; ++k) m.p.push_back(b[k] / n); }
  m.f = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
  for (int l = 0; l < level; ++l) {
    std::map<std::pair<int, int>, int> mid;
    auto get = [&](int a, int b) {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      if (const auto it = mid.find(key); it != mid.end()) return it->second;
      double q[3], n = 0;
      for (int k = 0; k < 3; ++k) { q[k] = m.p[static_cast<std::size_t>(3 * a + k)] + m.p[static_cast<std::size_t>(3 * b + k)]; n += q[k] * q[k]; }
      n = std::sqrt(n);
      for (int k = 0; k < 3; ++k) m.p.push_back(q[k] / n);
      return mid[key] = static_cast<int>(m.p.size() / 3 - 1);
    };
    std::vector<std::int32_t> nf;
    for (std::size_t i = 0; i < m.f.size(); i += 3) {
      const int a = m.f[i], b = m.f[i + 1], c = m.f[i + 2], ab = get(a, b), bc = get(b, c), ca = get(c, a);
      for (int v : {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca}) nf.push_back(v);
    }
    m.f = nf;
  }
  return m;
}
// the same sphere turned about z by `angle` (and optionally scaled: the radius is irrelevant)
static std::vector<double> turned(const std::vector<double>& p, double angle, double scale = 1) {
  std::vector<double> q(p.size());
  for (std::size_t i = 0; i < p.size(); i += 3) {
    q[i] = scale * (std::cos(angle) * p[i] - std::sin(angle) * p[i + 1]);
    q[i + 1] = scale * (std::sin(angle) * p[i] + std::cos(angle) * p[i + 1]);
    q[i + 2] = scale * p[i + 2];
  }
  return q;
}
static Mesh glued(const Mesh& a, const Mesh& b) {   // two pieces, left then right
  Mesh m = a;
  const std::int32_t n = static_cast<std::int32_t>(a.p.size() / 3);
  m.p.insert(m.p.end(), b.p.begin(), b.p.end());
  for (auto f : b.f) m.f.push_back(f + n);
  return m;
}
static bool refused(const std::function<void()>& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }

int main() {
  const Mesh coarse = icosphere(3), fine = icosphere(4);   // 642 and 2562 vertices
  const std::vector<Piece> one_c = {{0, 642}}, one_f = {{0, 2562}};

  std::cout << "=== every row is a weighting that sums to 1; a smooth field is carried, on the coarse sphere to the fine and back ===\n";
  for (Method m : {Method::Shepard, Method::Nearest, Method::Barycentric}) {
    Options o;
    o.method = m;
    Report rep;
    const Csr<double> w = resample_operator<double>(coarse.p, coarse.f, one_c, fine.p, one_f, o, &rep);
    EXPECT(w.rows == 2562 && w.cols == 642 && w.indptr.size() == 2563, method_name(m) << ": shape");
    bool sums = true, sorted = true;
    for (std::int64_t i = 0; i < w.rows; ++i) {
      double s = 0;
      for (int k = w.indptr[static_cast<std::size_t>(i)]; k < w.indptr[static_cast<std::size_t>(i) + 1]; ++k) { s += w.data[static_cast<std::size_t>(k)]; if (k > w.indptr[static_cast<std::size_t>(i)] && w.indices[static_cast<std::size_t>(k)] <= w.indices[static_cast<std::size_t>(k) - 1]) sorted = false; }
      if (std::abs(s - 1) > 1e-12) sums = false;
    }
    EXPECT(sums, method_name(m) << ": rows sum to 1");
    EXPECT(sorted, method_name(m) << ": indices ascend within a row");
    std::vector<double> x(642), want(2562);
    for (std::size_t i = 0; i < 642; ++i) x[i] = coarse.p[3 * i + 2] + 0.5 * coarse.p[3 * i];
    for (std::size_t i = 0; i < 2562; ++i) want[i] = fine.p[3 * i + 2] + 0.5 * fine.p[3 * i];
    const std::vector<double> y = carry(w, x);
    double err = 0;
    for (std::size_t i = 0; i < 2562; ++i) err = std::max(err, std::abs(y[i] - want[i]));
    const double tol = m == Method::Nearest ? 0.2 : 0.05;
    EXPECT(err < tol, method_name(m) << ": a linear field is carried within " << tol << " (error " << err << ")");
    std::cout << "  " << method_name(m) << ": max error " << err << ", nnz " << w.nnz() << ", empty columns " << rep.empty_columns << "\n";
    if (m == Method::Nearest) EXPECT(w.nnz() == 2562, "nearest: one entry per row");
    if (m == Method::Barycentric) EXPECT(rep.fallback_to_nearest == 0 && w.nnz() <= 3 * 2562, "barycentric: every point on a triangle, at most three entries");
    if (m == Method::Shepard) EXPECT(w.nnz() <= 7 * 2562, "shepard: at most K-1 entries per row");
  }

  std::cout << "=== the same sphere: each vertex takes itself (shepard, nearest, barycentric); a turned sphere takes the nearest ===\n";
  for (Method m : {Method::Shepard, Method::Nearest, Method::Barycentric}) {
    Options o;
    o.method = m;
    const Csr<double> w = resample_operator<double>(coarse.p, coarse.f, one_c, coarse.p, one_c, o);
    bool id = true;
    for (std::int64_t i = 0; i < 642; ++i) {
      const int a = w.indptr[static_cast<std::size_t>(i)], b = w.indptr[static_cast<std::size_t>(i) + 1];
      double self = 0;
      for (int k = a; k < b; ++k) if (w.indices[static_cast<std::size_t>(k)] == i) self = w.data[static_cast<std::size_t>(k)];
      if (std::abs(self - 1) > 1e-9) id = false;
    }
    EXPECT(id, method_name(m) << ": identity on the same sphere");
  }
  {
    Options o;
    o.method = Method::Nearest;
    const Csr<double> w = resample_operator<double>(coarse.p, coarse.f, one_c, turned(coarse.p, 1e-3, 100.0), one_c, o);   // a radius of 100 changes nothing
    bool id = true;
    for (std::int64_t i = 0; i < 642; ++i) if (w.indices[static_cast<std::size_t>(w.indptr[static_cast<std::size_t>(i)])] != i) id = false;
    EXPECT(id, "a slightly turned sphere of another radius: nearest is the identity");
  }

  std::cout << "=== pieces are matched one to one: a destination vertex of the first piece takes only from the first ===\n";
  {
    const Mesh L = glued(coarse, coarse), R = glued(fine, fine);
    const std::vector<Piece> pc = contiguous_pieces(L.f, 1284), pf = contiguous_pieces(R.f, 5124);
    EXPECT(pc.size() == 2 && pc[0].first == 0 && pc[0].count == 642 && pc[1].first == 642 && pc[1].count == 642, "contiguous_pieces: two runs");
    EXPECT(pf.size() == 2 && pf[1].first == 2562, "contiguous_pieces: fine");
    Options o;
    const Csr<double> w = resample_operator<double>(L.p, L.f, pc, R.p, pf, o);
    bool apart = true;
    for (std::int64_t i = 0; i < w.rows; ++i)
      for (int k = w.indptr[static_cast<std::size_t>(i)]; k < w.indptr[static_cast<std::size_t>(i) + 1]; ++k) if ((i < 2562) != (w.indices[static_cast<std::size_t>(k)] < 642)) apart = false;
    EXPECT(apart, "left stays with left, right with right");
    EXPECT(refused([&] { resample_operator<double>(L.p, L.f, pc, fine.p, one_f, o); }), "two pieces against one are refused");
    EXPECT(refused([&] { resample_operator<double>(L.p, L.f, {{0, 600}}, R.p, pf, o); }), "pieces that do not tile their sphere are refused");
    std::vector<std::int32_t> bad = coarse.f;
    EXPECT(refused([&] { contiguous_pieces(bad, 700); }), "a vertex on no face is refused");
  }

  std::cout << "=== precision follows the input; the K-th neighbour takes no weight; options are named ===\n";
  {
    const std::vector<float> cf(coarse.p.begin(), coarse.p.end()), ff(fine.p.begin(), fine.p.end());
    Options o;
    const Csr<float> w = resample_operator<float>(cf, coarse.f, one_c, ff, one_f, o);
    static_assert(std::is_same_v<decltype(w.data)::value_type, float>, "float in, float out");
    EXPECT(w.nnz() > 0, "float32 operator built");
    o.neighbours = 3;
    const Csr<double> w3 = resample_operator<double>(coarse.p, coarse.f, one_c, fine.p, one_f, o);
    EXPECT(w3.nnz() <= 2 * 2562, "K = 3: two entries at most");
    o.neighbours = 1;
    const Csr<double> w1 = resample_operator<double>(coarse.p, coarse.f, one_c, fine.p, one_f, o);
    EXPECT(w1.nnz() == 2562, "K = 1 is the nearest vertex");
  }
  EXPECT(parse_method("barycentric") == Method::Barycentric && refused([] { parse_method("linear"); }), "method names");
  if (fails) { std::cout << fails << " failure(s)\n"; return 1; }
  std::cout << "OK\n";
  return 0;
}
