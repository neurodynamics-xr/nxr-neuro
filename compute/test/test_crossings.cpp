// nxr::crossings::crossings against nxr's `Tiles.crossings` (golden: test/golden/crossings_golden.txt, gen_crossings_golden.py; two icosphere hemispheres with PLANTED MOVING VORTICES of both senses, nxr's surface ladder at
// level 3, a stage's subset of the frames): the crossing COUNTS per tile and sense are nxr's exactly, and the sums (each crossing's unit direction carried to its tile's centre, added, embedded as a 3-D tangent vector at the
// centre) agree with nxr's to the difference of the two transports (the Studio's intrinsic-Delaunay vector heat method against nxr's geometry-central one: a fraction of a degree on a sphere).
// The RUN-TO-RUN REPEAT (piece 8) against nxr's `repeat` and `tile_dot` (golden: test/golden/repeat_golden.txt, gen_repeat_golden.py; planted tile sums of two runs, most tiles agreeing, three opposed, a few with too few crossings): the
// per-tile dot products and the count-weighted mean cosine are nxr's, the senses pooled. Synthetic pairs: the same motion twice repeats (cosine 1, dots > 0), the reversed motion gives -1 and negative dots, tiles below the crossing
// minimum are left out, no qualifying tile gives NaN; refusals.
// Hand-made cases: the nearest vertex, a crossing is ONE TRACK'S steps in one tile (the same track's steps add before they are made a unit vector, so a track that crosses a tile in two steps counts once), opposite senses
// are separate, a too-short step has no direction, the bridge, the `use` mask; refusals.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/chart.h"
#include "nxr/select/track_crossings.h"
#include "nxr/frames.h"
#include "nxr/select/dyadic.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
template <class F> bool throws(F f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; }
nxr::crossings::Surface flat_surface(int V, std::vector<std::int32_t> node) {   // vertices along x, the plane z = 0, e1 = x, e2 = y, normal z, no rotation
  nxr::crossings::Surface S;
  S.positions = Eigen::MatrixXd::Zero(V, 3); S.normals = Eigen::MatrixXd::Zero(V, 3); S.e1 = Eigen::MatrixXd::Zero(V, 3); S.e2 = Eigen::MatrixXd::Zero(V, 3);
  for (int i = 0; i < V; ++i) { S.positions(i, 0) = i; S.normals(i, 2) = 1; S.e1(i, 0) = 1; S.e2(i, 1) = 1; }
  S.rho = Eigen::VectorXd::Zero(V); S.node = std::move(node);
  return S;
}
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/crossings_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  int nv = 0, nf = 0, n0 = 0, level = 0, ntile = 0, nsteps = 0, nframes = 0;
  std::vector<double> pos, normals, steps, sums;
  std::vector<int> faces, node, anchors, frames, ncount;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag, a;
    s >> tag;
    if (tag == "nv") { std::string b, c, d, e, f; s >> nv >> a >> nf >> b >> n0 >> c >> level >> d >> ntile >> e >> nsteps >> f >> nframes; }
    else if (tag == "positions") pos = nums(s);
    else if (tag == "faces") for (double f : nums(s)) faces.push_back(static_cast<int>(f));
    else if (tag == "normals") normals = nums(s);
    else if (tag == "node") for (double f : nums(s)) node.push_back(static_cast<int>(f));
    else if (tag == "anchors") for (double f : nums(s)) anchors.push_back(static_cast<int>(f));
    else if (tag == "steps") steps = nums(s);
    else if (tag == "frames") for (double f : nums(s)) frames.push_back(static_cast<int>(f));
    else if (tag == "sums") sums = nums(s);
    else if (tag == "n") for (double f : nums(s)) ncount.push_back(static_cast<int>(f));
  }
  // the surface: the frames (the signpost basis), the transport of each tile centre's frame (the Levi-Civita rho), the golden's anchors and normals
  nxr::chart::ChartKit kit(pos.data(), nv, faces.data(), nf);
  const nxr::chart::Basis B = kit.basis();
  nxr::crossings::Surface S;
  S.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(pos.data(), nv, 3);
  S.normals = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(normals.data(), nv, 3);
  S.e1.resize(nv, 3); S.e2.resize(nv, 3);
  for (int i = 0; i < nv; ++i) for (int k = 0; k < 3; ++k) { S.e1(i, k) = B.e1[static_cast<std::size_t>(3 * i + k)]; S.e2(i, k) = B.e2[static_cast<std::size_t>(3 * i + k)]; }
  const Eigen::VectorXd gauge = nxr::frames::gaugeAngle(kit, S.e1);
  const int tiles = *std::max_element(node.begin(), node.end()) + 1;
  S.node.assign(node.begin(), node.end());
  const nxr::ladder::LeviCivita lc = nxr::ladder::leviCivita(kit, S.e1, S.e2, gauge, S.node, tiles, anchors);
  S.rho = lc.rho;
  nxr::crossings::Steps st;
  st.from.resize(nsteps, 3); st.disp.resize(nsteps, 3);
  for (int i = 0; i < nsteps; ++i) {
    const double* r = steps.data() + 9 * i;
    st.sense.push_back(static_cast<std::int8_t>(r[0] > 0 ? 1 : -1));
    for (int k = 0; k < 3; ++k) { st.from(i, k) = r[1 + k]; st.disp(i, k) = r[4 + k]; }
    st.track.push_back(static_cast<std::int32_t>(r[7]));
  }
  std::vector<std::uint8_t> use(static_cast<std::size_t>(nsteps), 0);
  for (int i = 0; i < nsteps; ++i) { const int f = static_cast<int>(steps[static_cast<std::size_t>(9 * i + 8)]); use[static_cast<std::size_t>(i)] = std::find(frames.begin(), frames.end(), f) != frames.end(); }
  const nxr::crossings::Result R = nxr::crossings::crossings(st, use, S, tiles);
  // the 3-D sums at the centres, both senses (nxr's tile-sense index 2 code + (sense > 0))
  const Eigen::MatrixXd plus = nxr::crossings::embedAtCentres(R.sum.col(0), anchors, S.e1, S.e2), minus = nxr::crossings::embedAtCentres(R.sum.col(1), anchors, S.e1, S.e2);
  bool counts = static_cast<int>(ncount.size()) == 2 * tiles;
  double dmax = 0, dang = 0, smax = 0;
  int cells = 0;
  for (int t = 0; counts && t < tiles; ++t)
    for (int sx = 0; sx < 2; ++sx) {
      const int idx = 2 * t + (sx == 0 ? 1 : 0), mine = R.n(t, sx);
      counts = counts && mine == ncount[static_cast<std::size_t>(idx)];
      const Eigen::Vector3d a = (sx == 0 ? plus : minus).row(t).transpose(), b = Eigen::Map<const Eigen::Vector3d>(sums.data() + 3 * idx);
      dmax = std::max(dmax, (a - b).norm()); smax = std::max(smax, b.norm());
      if (mine >= 3 && a.norm() > 0.5 && b.norm() > 0.5) { dang = std::max(dang, std::acos(std::min(1.0, a.dot(b) / (a.norm() * b.norm())))); ++cells; }
    }
  std::cout << "  " << nsteps << " steps, " << R.steps_in << " in the stage, " << R.steps_used << " used; " << ncount.size() << " tile-senses; counts " << (counts ? "equal nxr's" : "DIFFER") << "; sums differ by at most " << dmax
            << " of " << smax << " (largest |sum|), directions by at most " << dang * 180 / M_PI << " deg over " << cells << " tile-senses\n";
  EXPECT(counts && R.steps_in < nsteps && R.steps_in > 0, "the crossing counts are nxr's exactly");
  EXPECT(dmax < 0.01 * smax && dang < 0.02, "the carried sums are nxr's up to the transports' difference");
  EXPECT(R.bridge_n.sum() == 0, "no bridge on this surface");
  // ── THE REPEAT: nxr's golden ──
  {
    std::ifstream rin(std::string(NXR_TEST_GOLDEN_DIR) + "/repeat_golden.txt");
    if (!rin) { std::cout << "no repeat golden\n"; return 1; }
    int T = 0, nmin = 0;
    std::vector<double> sa, sb, dot;
    std::vector<int> na, nb;
    double cosine = 0;
    std::string l2;
    while (std::getline(rin, l2)) {
      std::istringstream q(l2);
      std::string tag, a;
      q >> tag;
      if (tag == "tiles") { std::string b; q >> T >> b >> nmin; }
      else if (tag == "sums_a") sa = nums(q); else if (tag == "sums_b") sb = nums(q);
      else if (tag == "n_a") for (double v : nums(q)) na.push_back(static_cast<int>(v));
      else if (tag == "n_b") for (double v : nums(q)) nb.push_back(static_cast<int>(v));
      else if (tag == "dot") { std::string w; while (q >> w) dot.push_back(w == "nan" ? std::nan("") : std::stod(w)); }
      else if (tag == "cosine") q >> cosine;
    }
    auto pooled = [&](const std::vector<double>& s, const std::vector<int>& n) {
      Eigen::MatrixXd plus(T, 3), minus(T, 3);
      Eigen::VectorXd np(T), nm(T);
      for (int t = 0; t < T; ++t) { for (int k = 0; k < 3; ++k) { minus(t, k) = s[static_cast<std::size_t>(3 * (2 * t) + k)]; plus(t, k) = s[static_cast<std::size_t>(3 * (2 * t + 1) + k)]; } nm[t] = n[static_cast<std::size_t>(2 * t)]; np[t] = n[static_cast<std::size_t>(2 * t + 1)]; }
      return nxr::crossings::poolSenses(plus, np, minus, nm);
    };
    const auto A = pooled(sa, na), B = pooled(sb, nb);
    const auto rr = nxr::crossings::repeat(A, B, nmin);
    double dd = 0;
    bool nanmatch = static_cast<int>(dot.size()) == T;
    int compared = 0;
    for (int t = 0; nanmatch && t < T; ++t) {
      nanmatch = nanmatch && (std::isnan(dot[static_cast<std::size_t>(t)]) == std::isnan(rr.dot[t]));
      if (!std::isnan(dot[static_cast<std::size_t>(t)])) { dd = std::max(dd, std::abs(dot[static_cast<std::size_t>(t)] - rr.dot[t])); ++compared; }
    }
    std::cout << "  repeat vs nxr: " << compared << " of " << T << " tiles compared, dots differ by " << dd << ", cosine " << rr.cosine << " (nxr " << cosine << ")\n";
    EXPECT(nanmatch && dd < 1e-12 && std::abs(rr.cosine - cosine) < 1e-12 && rr.tilesUsed == compared, "the per-tile dots and the whole recording's cosine are nxr's");
    // the same motion twice: 1; the reverse: -1; unrelated tiles: about 0
    const auto same = nxr::crossings::repeat(A, A, nmin);
    EXPECT(std::abs(same.cosine - 1) < 1e-12 && same.dot[0] >= 0, "the same motion repeats: cosine 1");
    nxr::crossings::Pooled Rv = A; Rv.sum = -A.sum;
    const auto rev = nxr::crossings::repeat(A, Rv, nmin);
    bool negative = true;
    for (int t = 0; t < T; ++t) if (!std::isnan(rev.dot[t])) negative = negative && rev.dot[t] <= 0;
    EXPECT(std::abs(rev.cosine + 1) < 1e-12 && negative, "the reversed motion: cosine -1, dots negative");
    nxr::crossings::Pooled few = A; few.n.setConstant(nmin - 1);
    const auto none = nxr::crossings::repeat(few, A, nmin);
    EXPECT(std::isnan(none.cosine) && none.tilesUsed == 0 && std::isnan(none.dot[0]), "too few crossings in a run: no tile qualifies");
    EXPECT(throws([&] { nxr::crossings::repeat(A, nxr::crossings::Pooled{Eigen::MatrixXd::Zero(T - 1, 3), Eigen::VectorXd::Zero(T - 1)}, nmin); }) && throws([&] { nxr::crossings::repeat(A, A, 0); }), "other tiles, or a minimum of 0, are refused");
  }
  // hand-made on a flat strip: vertices 0..9 along x, tiles 0 (0-4) and 1 (5-9), the bridge vertex 10
  { nxr::crossings::Surface F = flat_surface(11, {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, nxr::crossings::kBridge});
    nxr::crossings::Steps s;
    auto add = [&](int sense, double x, double dx, double dy, int track) { const int i = static_cast<int>(s.sense.size()); s.sense.push_back(static_cast<std::int8_t>(sense)); s.track.push_back(track); s.from.conservativeResize(i + 1, 3); s.disp.conservativeResize(i + 1, 3); s.from.row(i) << x, 0, 0; s.disp.row(i) << dx, dy, 0.5; };
    add(1, 1.2, 1, 0, 7);            // track 7 in tile 0: +x
    add(1, 2.4, 1, 0, 7);            // the same track, the same tile: one crossing with the first
    add(1, 3.1, 0, 2, 8);            // track 8 in tile 0: +y
    add(-1, 1.0, 0, 3, 9);           // the other sense: its own
    add(1, 7.0, -1, 0, 7);           // track 7 again, in tile 1: -x
    add(1, 8.0, 0, 0, 11);           // no direction (the displacement is along the normal, 0.5 in z, 0 in the plane)
    add(1, 10.1, 1, 1, 12);          // on the bridge: a 3-D unit vector
    const auto r = nxr::crossings::crossings(s, {}, F, 2);
    EXPECT(r.n(0, 0) == 2 && r.n(1, 0) == 1 && r.n(0, 1) == 1 && r.n(1, 1) == 0, "a crossing is one track's steps in one tile and sense: tile 0 holds two (tracks 7 and 8), the other sense one");
    EXPECT(std::abs(r.sum(0, 0) - std::complex<double>(1, 1)) < 1e-12 && std::abs(r.sum(1, 0) - std::complex<double>(-1, 0)) < 1e-12 && std::abs(r.sum(0, 1) - std::complex<double>(0, 1)) < 1e-12, "the sums are of unit vectors: (1,0) + (0,1), (-1,0), (0,1)");
    EXPECT(r.no_direction == 1 && r.bridge_n[0] == 1 && std::abs(r.bridge(0, 0) - 1 / std::sqrt(2.0)) < 1e-12 && std::abs(r.bridge(0, 1) - 1 / std::sqrt(2.0)) < 1e-12 && r.steps_used == 6, "a step with no tangent part has no direction; the bridge sums in 3-D, normalised");
    std::vector<std::uint8_t> mask = {1, 0, 1, 0, 0, 0, 0};
    const auto m = nxr::crossings::crossings(s, mask, F, 2);
    EXPECT(m.steps_in == 2 && m.n(0, 0) == 2 && m.n.sum() == 2, "the mask leaves steps out");
    // the transport: a vertex whose frame is turned by rho carries the direction by e^{-i rho}
    F.rho(1) = M_PI / 2;
    nxr::crossings::Steps one; one.sense = {1}; one.track = {1}; one.from.resize(1, 3); one.disp.resize(1, 3); one.from << 1.1, 0, 0; one.disp << 1, 0, 0;
    const auto t = nxr::crossings::crossings(one, {}, F, 2);
    EXPECT(std::abs(t.sum(0, 0) - std::complex<double>(0, -1)) < 1e-12, "a direction is carried by e^{-i rho}");
    const auto near = nxr::crossings::nearestVertex(F.positions, (Eigen::MatrixXd(2, 3) << 2.4, 0, 0, 7.6, 3, 0).finished());
    EXPECT(near == std::vector<std::int32_t>({2, 8}), "the nearest vertex");
  }
  // refusals
  { auto F = flat_surface(3, {0, 0, 0});
    nxr::crossings::Steps s; s.sense = {0}; s.track = {1}; s.from = Eigen::MatrixXd::Zero(1, 3); s.disp = Eigen::MatrixXd::Zero(1, 3);
    EXPECT(throws([&] { nxr::crossings::crossings(s, {}, F, 1); }), "a sense of 0 is refused");
    s.sense = {1};
    EXPECT(throws([&] { nxr::crossings::crossings(s, {}, F, 0); }), "a vertex's tile outside the tiles is refused");
    EXPECT(throws([&] { nxr::crossings::crossings(s, {1, 1}, F, 1); }), "a mask of another size is refused");
    s.track = {};
    EXPECT(throws([&] { nxr::crossings::crossings(s, {}, F, 1); }), "arrays that disagree are refused"); }
  std::cout << (fails ? "FAILED\n" : "test_crossings: all passed\n");
  return fails ? 1 : 0;
}
