// nxr::glue: parity with the Studio's glueMesh and its sources (golden/glue_golden.txt from gen_glue_golden.mts: two bumpy ellipsoids with a medial-wall
// cap and a stray wall vertex; equal and unequal meshes) — the faces, the kept vertices and the bridge's positions identical, the bridge's statistics, and
// a Field carried onto the glued surface as the Studio carries it (a bridge vertex: the inverse-distance blend of its 3 nearest cut wall vertices, in
// float32 as the Studio accumulates); and the refusals (no wall, not two pieces).
#include "nxr/glue.h"
#include "golden_io.h"
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
    const auto W = flat<std::uint8_t>(g.mat("wall"));
    const auto wantP = flat<double>(g.mat("out_positions"));
    const auto wantF = flat<std::int32_t>(g.mat("out_faces"));
    const auto wantK = flat<std::int32_t>(g.mat("out_keep"));
    const Eigen::MatrixXd br = g.mat("bridge");
    const auto carried = flat<double>(g.mat("carried"));
    const nxr::glue::Glued out = nxr::glue::glueMesh(P, F, W);
    EXPECT(out.faces == wantF, "case " << c << ": faces differ (" << out.faces.size() / 3 << " vs " << wantF.size() / 3 << ")");
    EXPECT(out.keep == wantK, "case " << c << ": kept vertices differ");
    double dp = out.positions.size() == wantP.size() ? 0 : 1e9;
    if (dp == 0) for (size_t i = 0; i < wantP.size(); ++i) dp = std::max(dp, std::abs(out.positions[i] - wantP[i]));
    EXPECT(dp == 0, "case " << c << ": positions differ by " << dp);
    EXPECT(out.bridge.vertices == static_cast<std::int64_t>(br(0, 0)) && out.bridge.rings == static_cast<std::int64_t>(br(0, 1)), "case " << c << ": bridge size");
    EXPECT(std::abs(out.bridge.area - br(0, 2)) < 1e-15 * br(0, 2) && std::abs(out.bridge.longestEdge - br(0, 3)) < 1e-15 * br(0, 3), "case " << c << ": bridge area / longest edge");
    // the sources: f(v) = v + 1 carried as the Studio's gather does (float32: a weight 1 is copied; otherwise each term accumulated in float32)
    const nxr::glue::Sources s = nxr::glue::sourcesOf(P, out);
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
  std::cout << "=== refusals ===\n";
  auto throws = [](auto&& f) { try { f(); } catch (const std::runtime_error&) { return true; } return false; };
  std::vector<std::uint8_t> none(lastW.size(), 0);
  EXPECT(throws([&] { nxr::glue::glueMesh(lastP, lastF, none); }), "no medial wall: refused");
  std::vector<std::int32_t> oneF(lastF.begin(), lastF.begin() + static_cast<long>(lastF.size() / 2));
  std::vector<double> P3 = lastP; std::vector<std::int32_t> F3 = lastF;
  const auto n = static_cast<std::int32_t>(lastP.size() / 3);
  for (size_t i = 0; i < lastF.size() / 2; ++i) F3.push_back(lastF[i] + n);                     // a third piece
  P3.insert(P3.end(), lastP.begin(), lastP.begin() + static_cast<long>(3 * static_cast<size_t>(n)));
  std::vector<std::uint8_t> W3 = lastW; W3.insert(W3.end(), lastW.begin(), lastW.end());
  EXPECT(throws([&] { nxr::glue::glueMesh(P3, F3, W3); }), "three pieces: refused");
  EXPECT(throws([&] { nxr::glue::glueMesh(lastP, lastF, std::vector<std::uint8_t>(3, 1)); }), "a wall of the wrong length: refused");
  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
