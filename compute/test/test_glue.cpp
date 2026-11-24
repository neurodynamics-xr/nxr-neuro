// nxr::glue: parity with the Studio's glueMesh and its sources (golden/glue_golden.txt from gen_glue_golden.mts: two bumpy ellipsoids with a medial-wall
// cap and a stray wall vertex; equal and unequal meshes) — the faces, the kept vertices and the bridge's positions identical, the bridge's statistics, and
// a Field carried onto the glued surface as the Studio carries it (a bridge vertex: the inverse-distance blend of its 3 nearest cut wall vertices, in
// float32 as the Studio accumulates). The reference keeps the stray wall vertex as cortex (it cuts only the largest wall piece); nxr's glue removes exactly
// the wall, so the parity is checked with the stray vertex off the wall (the cap alone: its exact cut is one simple loop, where both agree), and the golden's
// own wall (two pieces) is refused. Then the exact cut: every vertex off the wall kept, every wall vertex removed; a cortex vertex whose faces all touch the
// wall (a pinch in the rim: the spike u -> v -> u) kept and stitched; a wall enclosing cortex refused; and the refusals (no wall, not two pieces).
#include "nxr/geometry/surface_maps.h"
#include "golden_io.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

template <class T> static std::vector<T> flat(const Eigen::MatrixXd& M) {
  std::vector<T> v(static_cast<size_t>(M.size()));
  for (Eigen::Index i = 0; i < M.rows(); ++i) for (Eigen::Index j = 0; j < M.cols(); ++j) v[static_cast<size_t>(i * M.cols() + j)] = static_cast<T>(M(i, j));
  return v;
}

int main() {
  std::cout << "=== parity with the Studio's glueMesh (golden/glue_golden.txt) ===\n";
  Golden g(kDir + "/glue_golden.txt");
  EXPECT(g.good(), "glue golden opens");
  g.tok(); const int cases = static_cast<int>(g.num());
  std::vector<double> lastP; std::vector<std::int32_t> lastF; std::vector<std::uint8_t> lastW;
  for (int c = 0; c < cases; ++c) {
    const auto P = flat<double>(g.mat("positions"));
    const auto F = flat<std::int32_t>(g.mat("faces"));
    const auto goldenWall = flat<std::uint8_t>(g.mat("wall"));
    const auto wantP = flat<double>(g.mat("out_positions"));
    const auto wantF = flat<std::int32_t>(g.mat("out_faces"));
    const auto wantK = flat<std::int32_t>(g.mat("out_keep"));
    const Eigen::MatrixXd br = g.mat("bridge");
    const auto carried = flat<double>(g.mat("carried"));
    // the golden's wall is the cap and a stray vertex the reference kept: refused here (two pieces); the parity runs on the cap alone
    std::string twoPieces;
    try { nxr::geometry::glue::glueMesh(P, F, goldenWall); } catch (const std::runtime_error& e) { twoPieces = e.what(); }
    EXPECT(twoPieces.find("2 pieces") != std::string::npos, "case " << c << ": a wall of two pieces refused, naming them (" << twoPieces << ")");
    std::vector<std::uint8_t> W = goldenWall;
    for (std::int32_t k : wantK) if (k >= 0 && W[static_cast<size_t>(k)]) W[static_cast<size_t>(k)] = 0;
    const nxr::geometry::glue::Glued out = nxr::geometry::glue::glueMesh(P, F, W);
    EXPECT(out.faces == wantF, "case " << c << ": faces differ (" << out.faces.size() / 3 << " vs " << wantF.size() / 3 << ")");
    EXPECT(out.keep == wantK, "case " << c << ": kept vertices differ");
    { std::vector<std::int32_t> offWall; for (size_t v = 0; v < W.size(); ++v) if (!W[v]) offWall.push_back(static_cast<std::int32_t>(v));
      EXPECT(std::vector<std::int32_t>(out.keep.begin(), out.keep.begin() + static_cast<long>(offWall.size())) == offWall && (out.keep.size() == offWall.size() || out.keep[offWall.size()] < 0),
             "case " << c << ": the kept vertices are exactly those off the wall, ascending"); }
    double dp = out.positions.size() == wantP.size() ? 0 : 1e9;
    if (dp == 0) for (size_t i = 0; i < wantP.size(); ++i) dp = std::max(dp, std::abs(out.positions[i] - wantP[i]));
    EXPECT(dp == 0, "case " << c << ": positions differ by " << dp);
    EXPECT(out.bridge.vertices == static_cast<std::int64_t>(br(0, 0)) && out.bridge.rings == static_cast<std::int64_t>(br(0, 1)), "case " << c << ": bridge size");
    EXPECT(std::abs(out.bridge.area - br(0, 2)) < 1e-15 * br(0, 2) && std::abs(out.bridge.longestEdge - br(0, 3)) < 1e-15 * br(0, 3), "case " << c << ": bridge area / longest edge");
    // the sources: f(v) = v + 1 carried as the Studio's gather does (float32: a weight 1 is copied; otherwise each term accumulated in float32)
    const nxr::geometry::glue::Sources s = nxr::geometry::glue::sourcesOf(P, out);
    double worst = 0; int bridge_vertices = 0;
    for (size_t i = 0; i < out.keep.size(); ++i) {
      float acc = 0;
      for (int k = 0; k < s.k; ++k) {
        const std::int32_t r = s.idx[i * static_cast<size_t>(s.k) + static_cast<size_t>(k)];
        if (r < 0) continue;
        const float wt = s.w[i * static_cast<size_t>(s.k) + static_cast<size_t>(k)], v = static_cast<float>(r + 1);
        acc = wt == 1.0f ? v : static_cast<float>(static_cast<double>(acc) + static_cast<double>(wt) * static_cast<double>(v));
      }
      if (out.keep[i] < 0) ++bridge_vertices;
      worst = std::max(worst, std::abs(static_cast<double>(acc) - carried[i]));
    }
    EXPECT(worst == 0, "case " << c << ": a carried value differs by " << worst);
    std::cout << "  case " << c << ": " << P.size() / 3 << " -> " << out.keep.size() << " vertices (" << bridge_vertices << " on the bridge, " << out.bridge.rings << " rings), "
              << out.faces.size() / 3 << " faces; positions max diff " << dp << ", carried values max diff " << worst << "\n";
    lastP = P; lastF = F; lastW = W;
  }
  std::cout << "=== the exact cut: a pinch kept, cortex enclosed by the wall refused ===\n";
  {
    const std::size_t n = lastW.size();
    std::vector<std::vector<std::int32_t>> neighbours(n);
    for (size_t f = 0; f < lastF.size(); f += 3) for (int k = 0; k < 3; ++k) neighbours[static_cast<size_t>(lastF[f + static_cast<size_t>(k)])].push_back(lastF[f + static_cast<size_t>((k + 1) % 3)]);
    // a PINCH: a cortex vertex next to the wall whose other neighbours but one join the wall; its faces then all touch the wall, it is kept and reached along its one edge
    bool pinchGlued = false;
    for (std::size_t v = 0; v < n && !pinchGlued; ++v) {
      if (lastW[v]) continue;
      const auto& around = neighbours[v];
      if (std::none_of(around.begin(), around.end(), [&](std::int32_t x) { return lastW[static_cast<size_t>(x)] != 0; })) continue;
      for (std::int32_t stay : around) {
        if (lastW[static_cast<size_t>(stay)]) continue;
        std::vector<std::uint8_t> wall = lastW;
        for (std::int32_t x : around) if (x != stay) wall[static_cast<size_t>(x)] = 1;
        try {
          const nxr::geometry::glue::Glued out = nxr::geometry::glue::glueMesh(lastP, lastF, wall);
          std::vector<std::int32_t> offWall; for (size_t u = 0; u < n; ++u) if (!wall[u]) offWall.push_back(static_cast<std::int32_t>(u));
          const bool exact = std::vector<std::int32_t>(out.keep.begin(), out.keep.begin() + static_cast<long>(offWall.size())) == offWall && out.keep[offWall.size()] < 0;
          bool vOnKeptFace = false;     // the glued faces made only of kept vertices: v is on none (all its faces touched the wall), only on the bridge's
          const auto vg = static_cast<std::int32_t>(std::lower_bound(offWall.begin(), offWall.end(), static_cast<std::int32_t>(v)) - offWall.begin());
          for (size_t f = 0; f < out.faces.size(); f += 3) {
            bool allKept = true, hasV = false;
            for (int k = 0; k < 3; ++k) { const auto x = out.faces[f + static_cast<size_t>(k)]; allKept = allKept && x < static_cast<std::int32_t>(offWall.size()); hasV = hasV || x == vg; }
            vOnKeptFace = vOnKeptFace || (allKept && hasV);
          }
          EXPECT(exact, "pinch: the kept vertices are exactly those off the wall");
          EXPECT(!vOnKeptFace, "pinch: the vertex is on no cortex face (all its faces touched the wall), only on the bridge's");
          std::cout << "  pinch at vertex " << v << " (its one edge to the cortex: " << stay << "): glued, " << out.keep.size() << " vertices, every vertex off the wall kept\n";
          pinchGlued = true;
          break;
        } catch (const std::runtime_error&) {}
      }
    }
    EXPECT(pinchGlued, "a wall leaving a pinch (a cortex vertex reached along one edge) is glued");
    // cortex ENCLOSED by the wall: refused. One wall vertex all of whose neighbours are wall taken off it: a cortex vertex with no edge to the rest. A wall
    // vertex and its neighbours taken off it, all theirs wall: an island of cortex, the cut leaving two rim loops in that hemisphere.
    auto allWallAround = [&](std::size_t v) { return std::all_of(neighbours[v].begin(), neighbours[v].end(), [&](std::int32_t x) { return lastW[static_cast<size_t>(x)] != 0; }); };
    std::string lone, island;
    for (std::size_t v = 0; v < n && (lone.empty() || island.empty()); ++v) {
      if (!lastW[v] || !allWallAround(v)) continue;
      auto attempt = [&](const std::vector<std::uint8_t>& wall) { try { nxr::geometry::glue::glueMesh(lastP, lastF, wall); return std::string("glued"); } catch (const std::runtime_error& e) { return std::string(e.what()); } };
      if (lone.empty()) { std::vector<std::uint8_t> wall = lastW; wall[v] = 0; lone = attempt(wall); }
      if (island.empty() && std::all_of(neighbours[v].begin(), neighbours[v].end(), [&](std::int32_t x) { return allWallAround(static_cast<size_t>(x)); })) {
        std::vector<std::uint8_t> wall = lastW; wall[v] = 0; for (std::int32_t x : neighbours[v]) wall[static_cast<size_t>(x)] = 0;
        island = attempt(wall);
      }
    }
    std::cout << "  a cortex vertex enclosed by the wall: " << lone.substr(0, 240) << "\n  an island of cortex in the wall: " << island.substr(0, 240) << "\n";
    EXPECT(lone.find("1 cortex vertices enclosed by the wall") != std::string::npos, "a cortex vertex enclosed by the wall: refused, naming it");
    EXPECT(island.find("2 rim loops") != std::string::npos, "an island of cortex in the wall: refused, naming the rim loops");
  }
  std::cout << "=== refusals ===\n";
  auto throws = [](auto&& f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; };
  std::vector<std::uint8_t> none(lastW.size(), 0);
  EXPECT(throws([&] { nxr::geometry::glue::glueMesh(lastP, lastF, none); }), "no medial wall: refused");
  std::vector<std::int32_t> oneF(lastF.begin(), lastF.begin() + static_cast<long>(lastF.size() / 2));
  std::vector<double> P3 = lastP; std::vector<std::int32_t> F3 = lastF;
  const auto n = static_cast<std::int32_t>(lastP.size() / 3);
  for (size_t i = 0; i < lastF.size() / 2; ++i) F3.push_back(lastF[i] + n);                     // a third piece
  P3.insert(P3.end(), lastP.begin(), lastP.begin() + static_cast<long>(3 * static_cast<size_t>(n)));
  std::vector<std::uint8_t> W3 = lastW; W3.insert(W3.end(), lastW.begin(), lastW.end());
  EXPECT(throws([&] { nxr::geometry::glue::glueMesh(P3, F3, W3); }), "three pieces: refused");
  EXPECT(throws([&] { nxr::geometry::glue::glueMesh(lastP, lastF, std::vector<std::uint8_t>(3, 1)); }), "a wall of the wrong length: refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
