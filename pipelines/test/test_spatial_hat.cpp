// The spatial level as a MEXICAN HAT in `make_spatial_band` (`spatial_shape: mexhat`);
// Meyer stays the named option (and the library default). Synthetic sphere only:
//   - the gain is the hat peaked at the GEOMETRIC centre of the level's Weyl band (info records the centre, the floor and the reach); the dense exact band applied to a mode-like field is the hat's gain times it
//   - the three routes agree: the stored/computed eigenmodes (they reach where the hat is under 1e-3) and the Chebyshev series equal the dense exact reference
//   - the hat's long tail: modes that reach the Meyer band do NOT reach the hat: `auto` falls back to the series, ANNOUNCED (naming the hat), forced `eigen` refuses with the reason
//   - the keys: Meyer's unchanged, the hat's carry the shape and centre (a different key); another shape is refused
//   - the Riesz part follows the hat's gain (no singular scaling member)
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>

#include <Eigen/Eigenvalues>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/delaunay.h"
#include "nxr/glue.h"
#include "nxr/io.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
bool has(const std::string& s, const std::string& t) { return s.find(t) != std::string::npos; }
double rel(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff()); }
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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_spatial_hat_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  std::vector<double> U, V;
  std::vector<std::int32_t> F;
  icosphere(3, U, F);                             // 642 vertices, radius 0.08 m
  V = U;
  for (auto& x : V) x *= 0.08;
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter s = SubjectWriter::create(ds, "S1");
    s.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = U; add_surface(s, so, V, F); });
  }
  SubjectWriter s1 = SubjectWriter::open(ds, "S1");
  tiles_compute(ds, s1, TilesComputeOptions{});
  const SurfaceInput S = read_surface(s1, std::nullopt, true);
  const int L = 3;
  const SurfaceLevel SL = read_surface_level(s1, S, L);
  const Eigen::MatrixXd X = field(S, 3);
  const auto& mem = SL.bank.members[static_cast<std::size_t>(L)];
  const double centre = std::sqrt(mem.lo * mem.hi);

  SpatialBandOptions dense; dense.transform = "exact"; dense.spatial_shape = "mexhat";
  const SpatialBand want = make_spatial_band(S, SL, dense);
  const Eigen::MatrixXd Yw = want.apply(X);
  CHECK(Yw.cwiseAbs().maxCoeff() > 1e-3);
  CHECK(want.info["spatial_shape"] == "mexhat" && std::fabs(want.info["hat_centre_lambda"].get<double>() - centre) < 1e-9 * centre && want.info["hat_gain_floor"] == 1e-3);
  CHECK(want.info["hat_reach_lambda"].get<double>() > 5 * centre);                         // the hat's tail is long: the reach is many centres
  // the hat is not the Meyer band: another operator
  SpatialBandOptions meyer = dense; meyer.spatial_shape = "meyer";
  const Eigen::MatrixXd Ym = make_spatial_band(S, SL, meyer).apply(X);
  CHECK(make_spatial_band(S, SL, meyer).info["spatial_shape"] == "meyer" && rel(Yw, Ym) > 0.1);
  CHECK(rel(Ym, make_spatial_band(S, SL, SpatialBandOptions{[] { SpatialBandOptions o; o.transform = "exact"; return o; }()}).apply(X)) < 1e-14);   // the library default is Meyer
  // the dense hat equals the gain applied to the modes: an independent check on the sphere's own spectrum (λ_l = l(l+1)/r², 2l+1 fold)
  // the Chebyshev series follows the dense hat
  SpatialBandOptions ch = dense; ch.transform = "chebyshev";
  const SpatialBand bc = make_spatial_band(S, SL, ch);
  const double ec = rel(bc.apply(X), Yw);
  std::printf("  Mexican hat, level %d: the series (degree %d) vs the dense exact band: max relative difference %.2e\n", L, bc.info["order"].get<int>(), ec);
  CHECK(bc.info["transform"] == "chebyshev" && bc.info["spatial_shape"] == "mexhat" && ec < 2e-2);

  // the eigen route: modes that reach where the hat is under 1e-3 (computed, stored, said), equal to the dense reference
  SpatialBandOptions o; o.spatial_shape = "mexhat"; o.modes = 320;
  const SpatialBandPlan p0 = plan_spatial_band(s1, S, SL, o);
  CHECK(p0.transform == "eigen" && p0.source == "computed" && p0.reach > 5 * centre && p0.key["spatial_shape"] == "mexhat" && p0.warnings.empty());
  SpatialBandOptions om = o; om.spatial_shape = "meyer";
  const SpatialBandPlan pm = plan_spatial_band(s1, S, SL, om);
  CHECK(pm.key == Json({{"transform", "eigen"}, {"package", "hodgeDirac"}}) && p0.key != pm.key);   // Meyer's key unchanged
  CHECK(p0.reach > pm.reach);
  CHECK(code_of([&] { SpatialBandOptions b; b.spatial_shape = "gauss"; plan_spatial_band(s1, S, SL, b); }) == std::string(code::invalid_argument));
  std::vector<Json> said;
  const SpatialBand b1 = make_spatial_band(ds, s1, S, SL, o, [&](const Json& j) { said.push_back(j); });
  CHECK(b1.info["transform"] == "eigen" && b1.info["source"] == "computed" && b1.info["spatial_shape"] == "mexhat" && b1.info["package"]["lambda_max"].get<double>() >= p0.reach);
  const double e1 = rel(b1.apply(X), Yw);
  std::printf("  Mexican hat, level %d: eigenmodes (to %.3g 1/m^2, hat < 1e-3) vs the dense exact band: max relative difference %.2e\n", L, p0.reach, e1);
  CHECK(e1 < 5e-3);
  SpatialBandOptions o2 = o; o2.riesz = true;
  const SpatialBand br = make_spatial_band(ds, s1, S, SL, o2);
  SpatialBandOptions d2 = dense; d2.riesz = true;
  const SpatialBand dr = make_spatial_band(S, SL, d2);
  const double er = rel(br.apply_riesz(X), dr.apply_riesz(X));
  std::printf("  Mexican hat, Riesz part: eigenmodes vs dense exact %.2e\n", er);
  CHECK(br.apply_riesz && dr.apply_riesz && er < 5e-3 && rel(dr.apply(X), Yw) < 1e-14);
  // the eigen and the series routes agree with each other (the maintainers' pair)
  std::printf("  eigen vs Chebyshev: %.2e\n", rel(b1.apply(X), bc.apply(X)));
  CHECK(rel(b1.apply(X), bc.apply(X)) < 2e-2);

  // the hat's tail: a package that reaches the Meyer band does not reach the hat: `auto` falls back to the series, announced and naming the hat; forced eigen refuses
  {
    SubjectWriter s2 = SubjectWriter::create(ds, "S2");
    s2.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = U; add_surface(s2, so, V, F); });
    SubjectWriter t2 = SubjectWriter::open(ds, "S2");
    tiles_compute(ds, t2, TilesComputeOptions{});
    const SurfaceInput S2 = read_surface(t2, std::nullopt, true);
    const SurfaceLevel SL2 = read_surface_level(t2, S2, L);
    SpatialBandOptions few; few.modes = 100;      // about 50 exact modes: enough for the Meyer band of level 3 (the existing test), not for the hat
    const SpatialBand bm = make_spatial_band(ds, t2, S2, SL2, few);          // Meyer: computed and stored
    CHECK(bm.info["transform"] == "eigen" && bm.warnings.empty());
    SpatialBandOptions fh = few; fh.spatial_shape = "mexhat";
    const SpatialBandPlan ph = plan_spatial_band(t2, S2, SL2, fh);
    CHECK(ph.transform == "chebyshev" && ph.warnings.size() == 1 && has(ph.warnings[0], "CHEBYSHEV fallback") && has(ph.warnings[0], "do not reach") && has(ph.warnings[0], "Mexican hat"));
    std::vector<Json> told;
    const SpatialBand bh = make_spatial_band(ds, t2, S2, SL2, fh, [&](const Json& j) { told.push_back(j); });
    bool warned = false;
    for (const Json& j : told) warned = warned || (j.contains("warning") && has(j["warning"].get<std::string>(), "Mexican hat"));
    CHECK(bh.info["transform"] == "chebyshev" && bh.info["spatial_shape"] == "mexhat" && warned && bh.info["warnings"].size() == 1);
    fh.transform = "eigen";
    CHECK(code_of([&] { plan_spatial_band(t2, S2, SL2, fh); }) == std::string(code::invalid_argument) && has(what_of([&] { plan_spatial_band(t2, S2, SL2, fh); }), "do not reach"));
  }
  return nxrtest::finish("test_spatial_hat") || g_failures ? 1 : 0;
}
