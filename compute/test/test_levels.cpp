// nxr::select levels (spec docs/specs/2026-11-26-levels-of-one-manifold.md) on synthetic meshes only: a sensor helmet (points on a spherical cap meshed by pointSetMesh, an
// open disc) and a closed icosphere, their dyadic grids (subjectTree), and a time line. The level is the quotient of the fine complex: the cochain map d0_f P = E d0_l, the
// boundary on the crossing edges (Pᵀ d0_fᵀ F = d0_lᵀ Eᵀ F), the projected Laplacian and mass, d1_f E = J d1_l, the input mesh's edges = tileAdjacency, halfedges (twin, next,
// Euler characteristic) or the refusal, parent and children, the reductions against roll_values, the expansions, the time line against levelComplex, the product.
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "nxr/geometry/point_set_mesh.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dec.h"
#include "nxr/select/dyadic.h"
#include "nxr/select/extent.h"
#include "nxr/select/levels.h"

static int failures = 0;
#define EXPECT(cond, msg) do { if (cond) std::cout << "  [PASS] " << msg << "\n"; else { std::cout << "  [FAIL] " << msg << "\n"; ++failures; } } while (0)
using namespace nxr::select;
using Mat = Eigen::MatrixXd;

template <class F> bool throws(F f, const std::string& part = "") {
  try { f(); } catch (const std::invalid_argument& e) { return part.empty() || std::string(e.what()).find(part) != std::string::npos; }
  return false;
}
double maxAbs(const Mat& m) { return m.size() ? m.cwiseAbs().maxCoeff() : 0.0; }

struct TestMesh { std::vector<double> p; std::vector<std::int32_t> f; int nV() const { return static_cast<int>(p.size() / 3); } int nF() const { return static_cast<int>(f.size() / 3); } };

TestMesh icosphere(int subdivisions) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  TestMesh m;
  const double raw[] = {-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1};
  for (int i = 0; i < 36; i += 3) { const double l = std::sqrt(raw[i] * raw[i] + raw[i + 1] * raw[i + 1] + raw[i + 2] * raw[i + 2]); for (int k = 0; k < 3; ++k) m.p.push_back(raw[i + k] / l); }
  m.f = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
  for (int s = 0; s < subdivisions; ++s) {
    std::map<std::pair<int, int>, int> mid;
    auto midpoint = [&](int a, int b) {
      const auto key = std::make_pair(std::min(a, b), std::max(a, b));
      if (auto it = mid.find(key); it != mid.end()) return it->second;
      double x[3], l = 0;
      for (int k = 0; k < 3; ++k) { x[k] = m.p[3 * a + k] + m.p[3 * b + k]; l += x[k] * x[k]; }
      l = std::sqrt(l);
      for (int k = 0; k < 3; ++k) m.p.push_back(x[k] / l);
      return mid[key] = m.nV() - 1;
    };
    std::vector<std::int32_t> f;
    for (std::size_t i = 0; i < m.f.size(); i += 3) {
      const int a = m.f[i], b = m.f[i + 1], c = m.f[i + 2], ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      f.insert(f.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
    }
    m.f = f;
  }
  return m;
}
// a sensor helmet: points on a cap of a sphere (radius 0.1 m, polar angle up to 100 degrees), on rings, meshed as the sensor surface is
TestMesh helmet() {
  TestMesh m;
  m.p = {0, 0, 0.1};
  for (int r = 1; r <= 9; ++r) {
    const double th = r * (100.0 / 9.0) * M_PI / 180.0;
    const int n = 6 * r;
    for (int k = 0; k < n; ++k) { const double ph = 2 * M_PI * (k + 0.5 * (r % 2)) / n; m.p.insert(m.p.end(), {0.1 * std::sin(th) * std::cos(ph), 0.1 * std::sin(th) * std::sin(ph), 0.1 * std::cos(th)}); }
  }
  m.f = nxr::geometry::pointset::pointSetMesh(m.p).faces;
  return m;
}

FineComplex fineOf(const nxr::operators::DECOperators& dec, const Eigen::VectorXd& star0) {
  FineComplex f;
  f.d0 = dec.d0; f.d1 = dec.d1; f.star0 = star0; f.star1 = dec.hodge1.diagonal();
  return f;
}
SparseMatrix prolong(const Level& L, std::int64_t V) {
  std::vector<Eigen::Triplet<double>> t;
  for (std::int64_t v = 0; v < V; ++v) if (L.cellOf[static_cast<std::size_t>(v)] >= 0) t.emplace_back(static_cast<int>(v), L.cellOf[static_cast<std::size_t>(v)], 1.0);
  SparseMatrix P(V, L.cells);
  P.setFromTriplets(t.begin(), t.end());
  return P;
}
std::vector<std::int32_t> asInt32(const std::vector<std::int64_t>& v) { return {v.begin(), v.end()}; }

void surfaceChecks(const char* name, const TestMesh& m, int expectEuler) {
  std::cout << name << " (" << m.nV() << " vertices, " << m.nF() << " faces)\n";
  const auto S = nxr::geometry::surfaceArrays(m.p.data(), m.nV(), m.f.data(), m.nF());
  const auto dec = nxr::operators::assembleDECOperators(S);
  nxr::select::bisection::Mesh bm; bm.positions = m.p; bm.faces = m.f; bm.n = m.nV();
  const std::vector<double> bary = nxr::select::bisection::vertexAreas(bm);
  const Eigen::VectorXd star0 = Eigen::Map<const Eigen::VectorXd>(bary.data(), m.nV());
  const FineComplex fine = fineOf(dec, star0);
  const Tree tree = subjectTree(m.p.data(), m.nV(), m.f.data(), m.nF());
  const int D = tree.depth;
  const SparseMatrix Lf = SparseMatrix(fine.d0.transpose() * fine.star1.asDiagonal() * fine.d0);
  std::mt19937 rng(7);
  std::normal_distribution<double> g;
  Eigen::VectorXd F(fine.d0.rows());
  for (Eigen::Index i = 0; i < F.size(); ++i) F(i) = g(rng);
  bool cochain = true, boundary = true, galerkin = true, mass = true, chain1 = true, d1d0 = true, w = true;
  int validLevels = 0, refusedLevels = 0;
  bool halfedgesOk = true, eulerOk = true;
  std::string aRefusal;
  for (int l = 0; l <= D; ++l) {
    const Level L = dyadicLevel(fine, tree.codes, D, tree.slots, l);
    const SparseMatrix P = prolong(L, m.nV());
    cochain = cochain && maxAbs(Mat(fine.d0 * P - L.crossing * L.d0)) < 1e-12;
    boundary = boundary && maxAbs(Mat(P.transpose() * (fine.d0.transpose() * F) - L.d0.transpose() * (L.crossing.transpose() * F))) < 1e-9;
    galerkin = galerkin && maxAbs(Mat(P.transpose() * Lf * P) - Mat(laplacian(L))) < 1e-9;
    mass = mass && maxAbs(Mat(P.transpose() * star0.asDiagonal() * P) - Mat(L.star0.asDiagonal())) < 1e-15;
    // the level's ⋆0 is the stored grid's `w` (the barycentric areas summed per tile: tiles_steps)
    std::vector<double> wl(static_cast<std::size_t>(L.cells), 0.0);
    const auto at = tree.at(l);
    for (int v = 0; v < m.nV(); ++v) if (at[static_cast<std::size_t>(v)] >= 0) wl[static_cast<std::size_t>(at[static_cast<std::size_t>(v)])] += bary[static_cast<std::size_t>(v)];
    for (std::int64_t c = 0; c < L.cells; ++c) w = w && std::abs(wl[static_cast<std::size_t>(c)] - L.star0(c)) < 1e-15;
    if (L.faces.rows()) {
      SparseMatrix J(fine.d1.rows(), L.faces.rows());
      std::vector<Eigen::Triplet<double>> t;
      for (std::size_t f = 0; f < L.junctions.size(); ++f) t.emplace_back(static_cast<int>(L.junctions[f]), static_cast<int>(f), 1.0);
      J.setFromTriplets(t.begin(), t.end());
      chain1 = chain1 && maxAbs(Mat(fine.d1 * L.crossing - J * L.d1)) < 1e-12;
      d1d0 = d1d0 && maxAbs(Mat(L.d1 * L.d0)) == 0;
    }
    if (L.valid) {
      ++validLevels;
      for (std::int64_t h = 0; h < 3 * L.faces.rows(); ++h) {
        const auto tw = twin(L, h);
        if (tw >= 0) halfedgesOk = halfedgesOk && twin(L, tw) == h && tail(L, tw).code == tip(L, h).code && tip(L, tw).code == tail(L, h).code;
        halfedgesOk = halfedgesOk && next(L, next(L, next(L, h))) == h && tip(L, h).code == tail(L, next(L, h)).code;
      }
      eulerOk = eulerOk && L.euler == expectEuler;
    } else {
      ++refusedLevels; if (aRefusal.empty()) aRefusal = L.why;
      halfedgesOk = halfedgesOk && throws([&] { twin(L, 0); }, "no halfedges");
    }
    // the input mesh: the level's edges are the stored adjacency (tileAdjacency)
    const auto emb = nxr::operators::embeddedDEC(S);
    const Level Li = dyadicLevel(fineOf(emb, star0), tree.codes, D, tree.slots, l);
    const auto adj = tileAdjacency(m.f.data(), m.nF(), at);
    bool same = static_cast<std::int64_t>(adj.size()) == Li.edges.rows();
    for (std::size_t e = 0; same && e < adj.size(); ++e) same = adj[e].first == Li.edges(static_cast<Eigen::Index>(e), 0) && adj[e].second == Li.edges(static_cast<Eigen::Index>(e), 1);
    EXPECT(same, std::string("level ") + std::to_string(l) + ": on the input mesh the edges are tileAdjacency's pairs, in their order (" + std::to_string(adj.size()) + ")");
  }
  EXPECT(cochain, "every level: d0_fine P = E d0_level (the crossing map is a cochain map)");
  EXPECT(boundary, "every level: Pᵀ d0_fineᵀ F = d0_levelᵀ Eᵀ F for a random edge flow (the boundary commutes on the crossing edges)");
  EXPECT(galerkin, "every level: d0ᵀ ⋆1 d0 = Pᵀ L_fine P (the projected Laplacian)");
  EXPECT(mass, "every level: Pᵀ ⋆0 P = ⋆0_level");
  EXPECT(w, "every level: ⋆0 is the tiles' barycentric areas (the stored `w`)");
  EXPECT(chain1, "every level with junctions: d1_fine E = J d1_level");
  EXPECT(d1d0, "every level with junctions: d1 d0 = 0");
  EXPECT(halfedgesOk, "valid levels: twin(twin(h)) = h with swapped ends, next^3 = id, tip(h) = tail(next(h)); refused levels refuse twin naming why");
  EXPECT(eulerOk && validLevels > 0, std::to_string(validLevels) + " valid levels, each with V - E + F = " + std::to_string(expectEuler));
  std::cout << "    " << refusedLevels << " levels refused, e.g.: " << aRefusal << "\n";

  // navigation: parent and children against the codes; the finest level's children are its vertices
  const Level Lv = vertexLevel(fine, tree.codes, D, tree.slots);
  const Level LD = dyadicLevel(fine, tree.codes, D, tree.slots, D), L2 = dyadicLevel(fine, tree.codes, D, tree.slots, 2), L3 = dyadicLevel(fine, tree.codes, D, tree.slots, 3);
  bool fam = true;
  for (std::int64_t c = 0; c < L3.cells; ++c) {
    const auto p = parent(L3, c);
    fam = fam && p.level == 2 && p.code == c >> 1;
    const auto ch = children(L2, c >> 1);
    fam = fam && ch.level == 3 && std::find(ch.codes.begin(), ch.codes.end(), c) != ch.codes.end();
  }
  for (int v = 0; v < m.nV(); ++v) {
    fam = fam && parent(Lv, v).level == D && parent(Lv, v).code == tree.codes[static_cast<std::size_t>(v)];
    const auto ch = children(LD, tree.codes[static_cast<std::size_t>(v)]);
    fam = fam && ch.level == kVertexLevel && std::find(ch.codes.begin(), ch.codes.end(), v) != ch.codes.end();
  }
  fam = fam && parent(dyadicLevel(fine, tree.codes, D, tree.slots, 0), 0).level == kNoLevel;
  EXPECT(fam, "parent (l - 1, c >> 1) and children (2c, 2c + 1); a vertex's parent is its code; the finest level's children are its vertices; level 0 has no parent");
  bool nb = true;
  for (std::int64_t c = 0; c < L3.cells; ++c)
    for (const auto n : neighbours(L3, c)) { const auto back = neighbours(L3, n); nb = nb && std::find(back.begin(), back.end(), c) != back.end(); }
  EXPECT(nb, "neighbours are symmetric");

  // reductions: the matrices equal roll_values (reduce) for sum and mean; max and min through reduce; expansions
  Mat x(2, m.nV());
  for (Eigen::Index i = 0; i < x.size(); ++i) x(i) = g(rng);
  const Mat sumR = Mat(x * Mat(reductionOperator(Lv, L3, "sum")).transpose()), meanR = Mat(x * Mat(reductionOperator(Lv, L3, "mean")).transpose());
  const Mat sumV = reduce(Lv, L3, "sum", x), meanV = reduce(Lv, L3, "mean", x);
  bool meanOk = true;
  for (Eigen::Index r = 0; r < 2; ++r) for (std::int64_t c = 0; c < L3.cells; ++c) meanOk = meanOk && (L3.star0(c) == 0 ? std::isnan(meanV(r, c)) : std::abs(meanV(r, c) - meanR(r, c)) < 1e-12);
  EXPECT(maxAbs(sumR - sumV) < 1e-12 && meanOk, "the vertices to level 3: the sum and mean reduction operators equal roll_values with the same rule (mean weighted by ⋆0; an empty tile NaN)");
  const Mat maxV = reduce(Lv, L3, "max", x), minV = reduce(Lv, L3, "min", x);
  bool order = true;
  for (int v = 0; v < m.nV(); ++v) { const auto c = L3.cellOf[static_cast<std::size_t>(v)]; order = order && maxV(0, c) >= x(0, v) && minV(0, c) <= x(0, v); }
  EXPECT(order, "max and min through roll_values bound every vertex of their tile");
  const Mat twoStep = reduce(L3, L2, "sum", sumV), direct = reduce(Lv, L2, "sum", x);
  EXPECT(maxAbs(twoStep - direct) < 1e-12, "level to level: the sum at level 3 rolled to level 2 = the vertices' sum at level 2");
  Eigen::VectorXd y = Eigen::VectorXd::Random(L3.cells);
  const Eigen::VectorXd pm = expansionOperator(L3, Lv, "mean") * y, ps = expansionOperator(L3, Lv, "sum") * y;
  bool pc = true;
  double total = 0, totalBack = 0;
  for (int v = 0; v < m.nV(); ++v) pc = pc && pm(v) == y(L3.cellOf[static_cast<std::size_t>(v)]);
  for (std::int64_t c = 0; c < L3.cells; ++c) if (L3.star0(c) > 0) total += y(c);
  totalBack = ps.sum();
  EXPECT(pc && std::abs(total - totalBack) < 1e-12, "expansion: mean is piecewise constant, sum keeps the total (split by area)");
  const Eigen::VectorXd roundTrip = reductionOperator(Lv, L3, "mean") * pm;
  bool rt = true;
  for (std::int64_t c = 0; c < L3.cells; ++c) if (L3.star0(c) > 0) rt = rt && std::abs(roundTrip(c) - y(c)) < 1e-12;
  EXPECT(rt, "the mean reduction undoes the mean expansion");
  EXPECT(throws([&] { expansionOperator(L3, Lv, "max"); }, "no expansion") && throws([&] { reductionOperator(Lv, L3, "none"); }, "never reduced") &&
             throws([&] { reduce(Lv, L3, "none", x); }) && throws([&] { reductionOperator(L2, L3, "sum"); }, "not coarser"),
         "refused: an order statistic's expansion, a `none` reduction, a reduction to a finer level");
}

int main() {
  std::cout << "test_levels\n";
  surfaceChecks("the sensor helmet (an open disc)", helmet(), 1);
  surfaceChecks("the icosphere (closed)", icosphere(3), 2);

  std::cout << "the time line\n";
  {
    const std::int64_t n = 37;
    const double fs = 200;
    FineComplex line;
    std::vector<Eigen::Triplet<double>> t;
    for (std::int64_t i = 0; i + 1 < n; ++i) { t.emplace_back(static_cast<int>(i), static_cast<int>(i), -1.0); t.emplace_back(static_cast<int>(i), static_cast<int>(i + 1), 1.0); }
    line.d0.resize(n - 1, n); line.d0.setFromTriplets(t.begin(), t.end());
    line.star0 = Eigen::VectorXd::Constant(n, 1 / fs); line.star1 = Eigen::VectorXd::Constant(n - 1, fs);
    bool same = true, cross = true;
    for (int k = 0; k <= 6; ++k) {
      const Level T = lineLevel(n, fs, k);
      std::vector<std::int32_t> of(static_cast<std::size_t>(n));
      for (std::int64_t i = 0; i < n; ++i) of[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i >> k);
      const Level Q = levelComplex(line, of, (n + (1 << k) - 1) >> k);
      same = same && T.cells == Q.cells && T.edges == Q.edges && maxAbs(T.star0 - Q.star0) < 1e-15 && maxAbs(T.star1 - Q.star1) < 1e-9 && maxAbs(Mat(T.d0) - Mat(Q.d0)) == 0;
      for (std::int64_t e = 0; e < T.edges.rows(); ++e) cross = cross && Q.crossing.coeff(crossingEdge(T, e), e) == 1.0 && Q.crossing.col(e).nonZeros() == 1;
    }
    EXPECT(same, "lineLevel(37, 200 Hz, k) = levelComplex of the samples with i >> k, k = 0 .. 6 (cells, edges, ⋆0 the tiles' durations, ⋆1 = rate, d0)");
    EXPECT(cross, "each time tile boundary is crossed by one fine edge, crossingEdge's");
    const Level T2 = lineLevel(n, fs, 2), T3 = lineLevel(n, fs, 3), T0 = lineLevel(n, fs, 0);
    EXPECT(T2.cells == 10 && std::abs(T2.star0(9) - 1 / fs) < 1e-15, "the last tile is as long as the line (37 samples at level 2: 10 tiles, the last of one sample)");
    bool fam = parent(T2, 9).level == 3 && parent(T2, 9).code == 4 && children(T3, 4).codes == std::vector<std::int64_t>{8, 9} && children(T3, 4).level == 2 &&
               children(T2, 9).codes == std::vector<std::int64_t>{18} && children(T0, 3).codes.empty() && neighbours(T2, 0) == std::vector<std::int64_t>{1} &&
               neighbours(T2, 5) == std::vector<std::int64_t>{4, 6};
    EXPECT(fam, "time: parent (k + 1, m >> 1), children inside the line, none at the samples; neighbours m - 1 and m + 1");
    Mat x = Mat::Random(3, n);
    const Mat r = reduce(T0, T3, "mean", x), rm = x * Mat(reductionOperator(T0, T3, "mean")).transpose();
    const Mat s = reduce(T0, T3, "sum", x), sm = x * Mat(reductionOperator(T0, T3, "sum")).transpose();
    EXPECT(maxAbs(r - rm) < 1e-12 && maxAbs(s - sm) < 1e-12, "time: the mean (duration-weighted) and sum reductions equal roll_values");
    EXPECT(!T2.valid && throws([&] { next(T2, 0); }, "no faces"), "a line has no halfedges: next is refused with why");
    EXPECT(throws([&] { lineLevel(0, fs, 0); }) && throws([&] { lineLevel(n, 0, 0); }) && throws([&] { cellMap(T3, T2); }, "not coarser"), "refused: no samples, no rate, a finer target");
  }

  std::cout << "the product: sensor helmet level x time level\n";
  {
    const TestMesh m = helmet();
    const auto S = nxr::geometry::surfaceArrays(m.p.data(), m.nV(), m.f.data(), m.nF());
    const auto dec = nxr::operators::assembleDECOperators(S);
    const FineComplex fine = fineOf(dec, dec.hodge0.diagonal());
    const Tree tree = subjectTree(m.p.data(), m.nV(), m.f.data(), m.nF());
    const Level Ls = dyadicLevel(fine, tree.codes, tree.depth, tree.slots, 3), Lt = lineLevel(50, 100, 2);
    const ProductLevel P = productLevel(Ls, Lt);
    bool rt = true;
    for (std::int64_t c = 0; c < P.cells(); ++c) { const auto a = jointAddress(P, c); rt = rt && jointCode(P, a.space.code, a.time.code) == c && a.space.level == 3 && a.time.level == 2; }
    EXPECT(rt, "joint codes round-trip: jointCode(jointAddress(c)) = c, the address ((surface, 3, code), (time, 2, code))");
    EXPECT(jointCode(P, 5, 2) == 5 + Ls.cells * 2, "the joint code is space fastest (a [space x time] column-major array's index)");
    Pairs tilePairs;
    for (std::int64_t e = 0; e < Ls.edges.rows(); ++e) tilePairs.push_back({Ls.edges(e, 0), Ls.edges(e, 1)});
    const Pairs st = spaceTimePairs(tilePairs, static_cast<int>(Ls.cells), static_cast<int>(Lt.cells));
    std::set<std::pair<std::int64_t, std::int64_t>> fromPairs, fromLevel;
    for (const auto& [a, b] : st) fromPairs.insert({std::min(a, b), std::max(a, b)});
    for (std::int64_t c = 0; c < P.cells(); ++c) for (const auto n : productNeighbours(P, c)) fromLevel.insert({std::min(c, n), std::max(c, n)});
    EXPECT(fromPairs == fromLevel, "the product's neighbours are spaceTimePairs's graph (" + std::to_string(fromPairs.size()) + " pairs)");
    // the product Laplacian is the Galerkin projection of the fine product's (helmet vertices x samples)
    const Level Lv = vertexLevel(fine), Ls0 = lineLevel(50, 100, 0);
    const ProductOperators fo = productOperators(productLevel(Lv, Ls0)), co = productOperators(P);
    const SparseMatrix Lf = SparseMatrix(fo.d0.transpose() * fo.star1.asDiagonal() * fo.d0), Lc = SparseMatrix(co.d0.transpose() * co.star1.asDiagonal() * co.d0);
    std::vector<Eigen::Triplet<double>> t;
    for (std::int64_t s = 0; s < Lv.cells; ++s) for (std::int64_t i = 0; i < Ls0.cells; ++i) if (Ls.cellOf[static_cast<std::size_t>(s)] >= 0) t.emplace_back(static_cast<int>(s + Lv.cells * i), static_cast<int>(Ls.cellOf[static_cast<std::size_t>(s)] + Ls.cells * (i >> 2)), 1.0);
    SparseMatrix Pp(Lv.cells * Ls0.cells, P.cells());
    Pp.setFromTriplets(t.begin(), t.end());
    EXPECT(maxAbs(Mat(Pp.transpose() * Lf * Pp) - Mat(Lc)) < 1e-6 * maxAbs(Mat(Lc)), "the product Laplacian = Pᵀ L_fine-product P with P = P_space x P_time");
    EXPECT(maxAbs(Mat(Pp.transpose() * fo.star0.asDiagonal() * Pp) - Mat(co.star0.asDiagonal())) < 1e-15, "the product ⋆0 = Pᵀ ⋆0_fine-product P");
    Coupling c; c.kind = Coupling::Kind::dispersion; c.speed = 3;
    EXPECT(throws([&] { productLevel(Ls, Lt, c); }, "dispersion") && throws([&] { productLevel(Lt, Ls); }, "first factor"), "refused: a dispersion coupling (not built yet), the factors swapped");
  }
  std::cout << (failures ? "FAILED: " : "all passed") << (failures ? std::to_string(failures) : "") << "\n";
  return failures ? 1 : 0;
}
