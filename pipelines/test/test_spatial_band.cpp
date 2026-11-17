// The spatial band of a tile level: EIGENMODE FIRST, CHEBYSHEV SECOND. Synthetic surfaces only:
//   - no stored modes: `auto` COMPUTES them (eigen_solve), STORES them as `eigen solve` does and SAYS so (progress and the result's messages); the band equals the dense exact one
//   - stored modes that reach the level: used (no computation, no message), the same band; the cache key (the plan's) names the transform and differs from the series'
//   - a level beyond the stored modes: the series, ANNOUNCED (a warning naming the level, the band's top and the modes' top, in the result and its info); forced `eigen` refuses with the reason
//   - nothing can be computed (no medial-wall atlas on two pieces): auto falls back, announced; eigen refuses; chebyshev forced is silent and writes nothing
//   - a cortex of two hemispheres glued by anatomy: the stored modes live on the glued surface; the band read back on the hemispheres equals the dense band of the glued surface
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>

#include <Eigen/Eigenvalues>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/glue.h"
#include "nxr/io.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
bool has(const std::string& s, const std::string& t) { return s.find(t) != std::string::npos; }
bool any_has(const std::vector<std::string>& v, const std::string& t) { for (const auto& s : v) if (has(s, t)) return true; return false; }
double rel(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); }
// a smooth-plus-rough field on the vertices
Eigen::MatrixXd field(const SurfaceInput& S, int cols) {
  const std::int64_t n = S.n_vertices();
  Eigen::MatrixXd F(n, cols);
  for (std::int64_t i = 0; i < n; ++i) {
    const double x = S.vertices[3 * i], y = S.vertices[3 * i + 1], z = S.vertices[3 * i + 2];
    for (int c = 0; c < cols; ++c) F(i, c) = std::sin((7 + 3 * c) * x + 2 * c) * std::cos(9 * y - c) + std::sin(15 * z + 0.3 * c) + 0.2 * std::cos(25 * (x + y + z));
  }
  return F;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_spatial_band_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  std::vector<double> U, V;                       // the unit icosphere (the registered pre-image) and a hemisphere-sized one: radius 0.08 m
  std::vector<std::int32_t> F;
  icosphere(3, U, F);                             // 642 vertices
  V = U;
  for (auto& x : V) x *= 0.08;
  DatasetWriter ds = create_dataset(root, "demo");

  // ── S1: ONE closed piece, no package ──
  {
    SubjectWriter s = SubjectWriter::create(ds, "S1");
    s.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = U; add_surface(s, so, V, F); });
  }
  SubjectWriter s1 = SubjectWriter::open(ds, "S1");
  nxrtest::make_tiles(ds, s1);
  const SurfaceInput S = read_surface(s1, std::nullopt, true);
  const int L = 3;
  const SurfaceLevel SL = read_surface_level(s1, S, L);
  const Eigen::MatrixXd X = field(S, 3);
  SpatialBandOptions dense; dense.transform = "exact";
  const Eigen::MatrixXd want = make_spatial_band(S, SL, dense).apply(X);       // the dense modes of the mesh itself: the reference
  CHECK(want.cwiseAbs().maxCoeff() > 1e-3);

  // the plan reads and writes nothing: no package yet, so it would COMPUTE; the key names the transform and the package
  SpatialBandOptions o; o.modes = 100;            // 100 modes: ~50 exact, plenty for level 3 (Weyl: about 30) on this small mesh
  const SpatialBandPlan p0 = plan_spatial_band(s1, S, SL, o);
  CHECK(p0.transform == "eigen" && p0.source == "computed" && p0.package == "hodgeDirac" && p0.compute_modes == 100 && !p0.messages.empty() && p0.warnings.empty());
  CHECK(!s1.find("operator", "hodgeDirac_eigenvectors").has_value());          // planning stored nothing
  CHECK(p0.key == Json({{"transform", "eigen"}, {"package", "hodgeDirac"}}));
  SpatialBandOptions oc = o; oc.transform = "chebyshev";
  const SpatialBandPlan pc = plan_spatial_band(s1, S, SL, oc);
  CHECK(pc.transform == "chebyshev" && pc.key["transform"] == "chebyshev" && pc.key != p0.key && pc.warnings.empty());   // forced: silent, no warning
  SpatialBandOptions oe = o; oe.transform = "exact";
  CHECK(plan_spatial_band(s1, S, SL, oe).key != p0.key);
  CHECK(code_of([&] { SpatialBandOptions b; b.transform = "fourier"; plan_spatial_band(s1, S, SL, b); }) == std::string(code::invalid_argument));

  // `auto` with no stored modes: computed, stored, said
  std::vector<Json> said;
  const SpatialBand b1 = make_spatial_band(ds, s1, S, SL, o, [&](const Json& j) { said.push_back(j); });
  CHECK(s1.find("operator", "hodgeDirac_eigenvectors").has_value() && s1.find("manifold", "eigenvalue/hodgeDirac").has_value());   // stored as `eigen solve` stores it
  CHECK(b1.info["transform"] == "eigen" && b1.info["source"] == "computed" && b1.info["requested"] == "auto" && b1.warnings.empty());
  CHECK(b1.info["package"]["name"] == "hodgeDirac" && b1.info["package"]["lambda_max"].get<double>() >= SL.bank.members[L].hi * std::pow(2.0, 1.0 / 3.0) && b1.info["modes"].get<int>() >= 5);
  CHECK(b1.info["package"]["computed"]["modes"].get<int>() >= 50 && b1.info["package"]["bridge_vertices"] == 0);
  CHECK(any_has(b1.messages, "eigenmodes computed and stored") && any_has(b1.messages, "name hodgeDirac") && any_has(b1.messages, "largest exact eigenvalue"));
  bool told = false;
  for (const Json& j : said) told = told || (j.contains("message") && has(j["message"].get<std::string>(), "eigenmodes computed and stored"));
  CHECK(told);                                                                  // in the progress, not only the result
  CHECK(b1.info.contains("messages"));                                          // and in the provenance
  const double e1 = rel(b1.apply(X), want);
  std::printf("  computed modes vs dense exact: max relative difference %.2e\n", e1);
  CHECK(e1 < 2e-3);

  // stored modes that reach the level: used, no computation, no message; the same band
  said.clear();
  const SpatialBandPlan p1 = plan_spatial_band(s1, S, SL, o);
  CHECK(p1.transform == "eigen" && p1.source == "stored" && p1.compute_modes == 0 && p1.messages.empty() && p1.key == p0.key);   // same key before and after the modes exist: a hit stays a hit
  const SpatialBand b2 = make_spatial_band(ds, s1, S, SL, o, [&](const Json& j) { said.push_back(j); });
  CHECK(b2.info["source"] == "stored" && b2.messages.empty() && said.empty() && b2.info["package"]["computed"].is_null());
  CHECK(rel(b2.apply(X), b1.apply(X)) < 1e-12);
  // forced eigen uses them too (the default and the forced request agree)
  SpatialBandOptions of = o; of.transform = "eigen";
  CHECK(rel(make_spatial_band(ds, s1, S, SL, of).apply(X), b1.apply(X)) < 1e-12);
  // THE RIESZ PART of the band (the spatial analytic signal, nxr/monogenic.h): R_l = ∇ Δ^{-1/2} F_l; stored eigenmodes = the dense exact reference; the series follows; absent unless asked
  {
    CHECK(!b1.apply_riesz && !make_spatial_band(S, SL, dense).apply_riesz);
    SpatialBandOptions dr = dense; dr.riesz = true;
    const SpatialBand er = make_spatial_band(S, SL, dr);
    const Eigen::MatrixXd Rw = er.apply_riesz(X);
    CHECK(Rw.rows() == 3 * S.n_vertices() && Rw.cols() == X.cols() && Rw.cwiseAbs().maxCoeff() > 1e-3);
    CHECK(rel(er.apply(X), want) < 1e-14);                                        // the band itself is unchanged by asking for the Riesz part
    SpatialBandOptions orr = o; orr.riesz = true;
    const SpatialBand sr = make_spatial_band(ds, s1, S, SL, orr);
    CHECK(sr.info["transform"] == "eigen" && sr.info["source"] == "stored" && sr.info["riesz"]["faces_used"].get<int>() == 1280);
    const double ee = rel(sr.apply_riesz(X), Rw);
    SpatialBandOptions cr = o; cr.transform = "chebyshev"; cr.riesz = true;
    const SpatialBand chr = make_spatial_band(S, SL, cr);
    const double ec = rel(chr.apply_riesz(X), Rw);
    std::printf("  Riesz part: stored eigenmodes vs dense exact %.2e; the series (degree %d) %.2e\n", ee, chr.info["riesz"]["order"].get<int>(), ec);
    CHECK(ee < 2e-3 && ec < 2e-2);
    // a level-0 (scaling) Chebyshev Riesz part is refused: its gain is not 0 at lambda = 0, g/sqrt(lambda) is singular
    const SurfaceLevel L0r = read_surface_level(s1, S, 0);
    CHECK(code_of([&] { make_spatial_band(S, L0r, cr); }) == std::string(code::invalid_argument));
  }
  // the scaling member (level 0) holds the constant, which no package stores: it is added exactly, so the band of a constant is the constant
  {
    const SurfaceLevel L0 = read_surface_level(s1, S, 0);
    const SpatialBand b0 = make_spatial_band(ds, s1, S, L0, o);
    const Eigen::MatrixXd one = Eigen::MatrixXd::Ones(S.n_vertices(), 1);
    CHECK(b0.info["transform"] == "eigen" && rel(b0.apply(one), make_spatial_band(S, L0, dense).apply(one)) < 2e-3);
  }

  // a level BEYOND the stored modes: the series, announced; forced eigen refuses with the reason
  const int Lf = 6;
  const SurfaceLevel SLf = read_surface_level(s1, S, Lf);
  said.clear();
  const SpatialBand bf = make_spatial_band(ds, s1, S, SLf, o, [&](const Json& j) { said.push_back(j); });
  CHECK(bf.info["transform"] == "chebyshev" && bf.info["requested"] == "auto" && bf.info["order"].get<int>() >= 40 && bf.warnings.size() == 1);
  CHECK(has(bf.warnings[0], "CHEBYSHEV fallback") && has(bf.warnings[0], "level 6") && has(bf.warnings[0], "do not reach") && has(bf.warnings[0], "hodgeDirac") && has(bf.warnings[0], "largest eigenvalue"));
  CHECK(bf.info["warnings"].size() == 1 && bf.info["warnings"][0] == bf.warnings[0]);                       // in the provenance
  bool warned = false;
  for (const Json& j : said) warned = warned || (j.contains("warning") && j["warning"] == bf.warnings[0]);
  CHECK(warned);                                                                // and in the progress
  std::printf("  fallback: %s\n", bf.warnings[0].c_str());
  const std::string why = what_of([&] { SpatialBandOptions b = o; b.transform = "eigen"; make_spatial_band(ds, s1, S, SLf, b); });
  CHECK(code_of([&] { SpatialBandOptions b = o; b.transform = "eigen"; make_spatial_band(ds, s1, S, SLf, b); }) == std::string(code::invalid_argument));
  CHECK(has(why, "--transform eigen") && has(why, "do not reach") && has(why, "level 6"));
  // the key of the fallback is the series' (what it computes), not the eigenmodes'
  CHECK(plan_spatial_band(s1, S, SLf, o).key == plan_spatial_band(s1, S, SLf, oc).key);

  // ── S2: no package and nothing that can be computed: too few modes asked, then two pieces with no medial-wall atlas ──
  {
    SubjectWriter s = SubjectWriter::create(ds, "S2");
    s.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = U; add_surface(s, so, V, F); });
  }
  SubjectWriter s2 = SubjectWriter::open(ds, "S2");
  nxrtest::make_tiles(ds, s2);
  const SurfaceInput S2 = read_surface(s2, std::nullopt, true);
  const SurfaceLevel SL2 = read_surface_level(s2, S2, L);
  {
    SpatialBandOptions few; few.modes = 8;      // about 4 exact modes: the level needs about 30 (Weyl)
    const SpatialBandPlan pf = plan_spatial_band(s2, S2, SL2, few);
    CHECK(pf.transform == "chebyshev" && pf.warnings.size() == 1 && has(pf.warnings[0], "cannot be computed") && has(pf.warnings[0], "Weyl"));
    few.transform = "eigen";
    CHECK(code_of([&] { plan_spatial_band(s2, S2, SL2, few); }) == std::string(code::invalid_argument));
    SpatialBandOptions cap; cap.max_modes = 50;
    CHECK(plan_spatial_band(s2, S2, SL2, cap).transform == "chebyshev" && has(plan_spatial_band(s2, S2, SL2, cap).warnings[0], "cap"));
    // forced chebyshev builds the series and writes NOTHING
    SpatialBandOptions fc; fc.transform = "chebyshev";
    const SpatialBand bc = make_spatial_band(ds, s2, S2, SL2, fc);
    CHECK(bc.warnings.empty() && bc.info["transform"] == "chebyshev" && !s2.find("operator", "hodgeDirac_eigenvectors").has_value());
  }
  {
    std::vector<double> W = V;
    std::vector<std::int32_t> G = F;
    const std::int32_t n1 = static_cast<std::int32_t>(V.size() / 3);
    for (std::size_t i = 0; i < V.size(); i += 3) { W.push_back(V[i] + 0.2); W.push_back(V[i + 1]); W.push_back(V[i + 2]); }
    for (auto f : F) G.push_back(f + n1);
    std::vector<double> UU = U; UU.insert(UU.end(), U.begin(), U.end());
    std::vector<std::int32_t> dk(static_cast<std::size_t>(2 * n1), 1);
    std::vector<std::uint8_t> wall(dk.size(), 0);
    for (std::int32_t i = 0; i < 2 * n1; ++i)
      if ((i < n1 && W[3 * i] > 0.05) || (i >= n1 && W[3 * i] - 0.2 < -0.05)) { dk[static_cast<std::size_t>(i)] = 0; wall[static_cast<std::size_t>(i)] = 1; }
    {
      SubjectWriter a = SubjectWriter::create(ds, "S3");     // two pieces, no atlas
      a.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = UU; add_surface(a, so, W, G); });
      SubjectWriter b = SubjectWriter::create(ds, "S4");     // two pieces, the Desikan-Killiany atlas with its wall
      SurfaceOptions so{.name = "cortex"};
      so.sphere = UU;
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      b.compose([&] { add_surface(b, so, W, G); });
    }
    // no atlas: not computable; auto falls back (announced), eigen refuses naming the atlas
    SubjectWriter s3 = SubjectWriter::open(ds, "S3");
    nxrtest::make_tiles(ds, s3);
    const SurfaceInput S3 = read_surface(s3, std::nullopt, true);
    const SurfaceLevel SL3 = read_surface_level(s3, S3, L);
    const SpatialBandPlan p3 = plan_spatial_band(s3, S3, SL3, o);
    CHECK(p3.transform == "chebyshev" && p3.warnings.size() == 1 && has(p3.warnings[0], "Desikan-Killiany"));
    SpatialBandOptions o3 = o; o3.transform = "eigen";
    CHECK(has(what_of([&] { plan_spatial_band(s3, S3, SL3, o3); }), "Desikan-Killiany"));

    // THE GLUE: the atlas gives the medial wall; the modes are computed on the glued surface and stored; the band read back on the hemispheres
    SubjectWriter s4 = SubjectWriter::open(ds, "S4");
    nxrtest::make_tiles(ds, s4);
    const SurfaceInput S4 = read_surface(s4, std::nullopt, true);
    const SurfaceLevel SL4 = read_surface_level(s4, S4, L);
    SpatialBandOptions o4 = o; o4.modes = 160;
    const SpatialBand b4 = make_spatial_band(ds, s4, S4, SL4, o4);
    CHECK(b4.info["transform"] == "eigen" && b4.info["source"] == "computed" && b4.info["package"]["bridge_vertices"].get<std::int64_t>() > 0 && b4.info["package"]["glued_surface"] == "cortex_glued");
    CHECK(b4.info["package"]["kept_vertices"].get<std::int64_t>() < S4.n_vertices());          // the medial wall is cut out
    // the reference: the dense band of the glued surface (the same glue, the stored mesh options), zero on the bridge, read on the kept vertices (the wall's vertices get 0)
    std::vector<double> W32(W.size());
    for (std::size_t i = 0; i < W.size(); ++i) W32[i] = static_cast<double>(static_cast<float>(W[i]));
    const nxr::glue::Glued gm = nxr::glue::glueMesh(W32, G, wall);
    const int Vg = static_cast<int>(gm.keep.size());
    nxr::manifold::IntrinsicSurface isurf(gm.positions.data(), Vg, gm.faces.data(), static_cast<int>(gm.faces.size() / 3));
    const std::vector<double> dm_mass = isurf.inputMixedVoronoiMass();
    Eigen::MatrixXd Ld(isurf.cotanLaplacian()), Md = Eigen::MatrixXd::Zero(Vg, Vg);
    for (int i = 0; i < Vg; ++i) Md(i, i) = dm_mass[static_cast<std::size_t>(i)];
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(Ld, Md);
    const Eigen::VectorXd lam = es.eigenvalues().cwiseMax(0.0);
    const Eigen::MatrixXd X4 = field(S4, 2);
    Eigen::MatrixXd Xg = Eigen::MatrixXd::Zero(Vg, X4.cols());
    for (int i = 0; i < Vg; ++i) if (gm.keep[static_cast<std::size_t>(i)] >= 0) Xg.row(i) = X4.row(gm.keep[static_cast<std::size_t>(i)]);
    Eigen::VectorXd g(lam.size());
    for (Eigen::Index j = 0; j < lam.size(); ++j) g(j) = SL4.bank.members[static_cast<std::size_t>(L)].gain(lam(j));
    const Eigen::MatrixXd Yg = es.eigenvectors() * g.asDiagonal() * es.eigenvectors().transpose() * Md * Xg;
    Eigen::MatrixXd ref = Eigen::MatrixXd::Zero(S4.n_vertices(), X4.cols());
    for (int i = 0; i < Vg; ++i) if (gm.keep[static_cast<std::size_t>(i)] >= 0) ref.row(gm.keep[static_cast<std::size_t>(i)]) = Yg.row(i);
    const double e4 = rel(b4.apply(X4), ref);
    std::printf("  glued cortex: stored modes (%lld glued vertices, %lld bridge) vs dense glued band: max relative difference %.2e\n", static_cast<long long>(b4.info["package"]["glued_vertices"].get<std::int64_t>()),
                static_cast<long long>(b4.info["package"]["bridge_vertices"].get<std::int64_t>()), e4);
    CHECK(ref.cwiseAbs().maxCoeff() > 1e-3 && e4 < 3e-3);
    for (std::int64_t i = 0; i < S4.n_vertices(); ++i) if (wall[static_cast<std::size_t>(i)]) CHECK(b4.apply(X4).row(i).isZero());   // the cut wall carries no band
  }
  return nxrtest::finish("test_spatial_band") || g_failures ? 1 : 0;
}
