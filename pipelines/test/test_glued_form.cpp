// apply_kernel's `form` ON THE GLUED SURFACE: the form's band (the Mexican hat) is evaluated on EVERY vertex of the glued surface, the corpus callosum's bridge included, the form
// being zero on the bridge before the band and the band filling it, exactly as nxr's `vortex.Cortex` does (golden: test/golden/gen_glued_hat_golden.py, a SYNTHETIC glued pair of spheres, never a real dataset).
//   - the glue: the glued surface the eigen solve stores is nxr's `glue_mesh` (the same vertices, the same faces)
//   - the band on every glued vertex equals nxr's: the stored eigenmodes (truncated where the hat's gain is under 1e-3), the dense modes of the glued mesh, the Chebyshev series of the glued mesh
//   - the bridge carries the band (not 0), the native read-back is the kept vertices of it, the cut medial wall carries none
//   - eigenmode first, Chebyshev an ANNOUNCED fallback: a level beyond the stored modes falls back on the glued mesh's series, with a warning, never silently
//   - one closed piece: its own glued domain (no bridge); the plan key names the glued evaluation; a cortex of two pieces with no stored glue is refused with what to run
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "check.h"
#include "hemispheres.h"
#include "mesh_helpers.h"
#include "nxr/signal/forms.h"
#include "nxr/io.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
bool has(const std::string& s, const std::string& t) { return s.find(t) != std::string::npos; }
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
double rel(const Eigen::MatrixXd& a, const std::vector<double>& ref, int cols) {
  double w = 0, m = 0;
  for (Eigen::Index v = 0; v < a.rows(); ++v) for (int c = 0; c < cols; ++c) { m = std::max(m, std::abs(ref[static_cast<std::size_t>(v * cols + c)])); w = std::max(w, std::abs(a(v, c) - ref[static_cast<std::size_t>(v * cols + c)])); }
  return w / m;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/glued_hat_golden.txt");
  if (!in) { std::cout << "no golden\n"; return 1; }
  int nv = 0, nf = 0, nG = 0, cols = 0;
  double area = 0;
  std::vector<double> pos, q, curl, gpos;
  std::vector<std::int32_t> faces, wallv, gfaces, keep;
  std::map<int, std::vector<double>> hat;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string tag;
    s >> tag;
    if (tag == "nv") { std::string a, b, c, d, e; s >> nv >> a >> nf >> b >> nG >> c >> cols >> d >> area; }
    else if (tag == "positions") pos = nums(s);
    else if (tag == "faces") { for (double f : nums(s)) faces.push_back(static_cast<std::int32_t>(f)); }
    else if (tag == "wall") { for (double f : nums(s)) wallv.push_back(static_cast<std::int32_t>(f)); }
    else if (tag == "glued_positions") gpos = nums(s);
    else if (tag == "glued_faces") { for (double f : nums(s)) gfaces.push_back(static_cast<std::int32_t>(f)); }
    else if (tag == "keep") { for (double f : nums(s)) keep.push_back(static_cast<std::int32_t>(f)); }
    else if (tag == "q") q = nums(s);
    else if (tag == "curl") curl = nums(s);
    else if (tag == "hat") { int lv; double peak; s >> lv >> peak; hat[lv] = nums(s); }
  }
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_glued_form_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  std::vector<double> U;
  std::vector<std::int32_t> F0;
  icosphere(3, U, F0);
  std::vector<double> UU = U;
  UU.insert(UU.end(), U.begin(), U.end());
  CHECK(static_cast<int>(UU.size() / 3) == nv && static_cast<int>(faces.size() / 3) == nf);
  std::vector<std::int32_t> dk(static_cast<std::size_t>(nv), 1);
  for (int i = 0; i < nv; ++i) if (wallv[static_cast<std::size_t>(i)]) dk[static_cast<std::size_t>(i)] = 0;
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter a = SubjectWriter::create(ds, "G1");       // two pieces, the Desikan-Killiany atlas with its medial wall
    SurfaceOptions so{.name = "cortex"};
    so.sphere = UU;
    so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
    so.atlases.push_back(nxrtest::structures_of(pos, faces));   // the hemispheres, as Brainstorm's Structures (the glue reads them)
    a.compose([&] { add_surface(a, so, pos, faces); });
    SubjectWriter b = SubjectWriter::create(ds, "G2");       // one closed piece
    b.compose([&] { SurfaceOptions o2; o2.name = "cortex"; o2.sphere = U; std::vector<double> p1(pos.begin(), pos.begin() + static_cast<std::ptrdiff_t>(3 * U.size() / 3)); add_surface(b, o2, p1, F0); });
  }
  SubjectWriter s1 = SubjectWriter::open(ds, "G1");
  nxrtest::make_tiles(ds, s1);
  const SurfaceInput S = read_surface(s1, std::string("cortex"), true);
  // the free currents' curl on the native mesh (tested against nxr in test_form_columns), zero on the wall's vertices of the glued domain by construction (the support excludes them)
  std::vector<float> P32(pos.begin(), pos.end());
  const auto op = nxr::signal::assembleFaceGradient(P32.data(), faces.data(), nv, nf);
  Eigen::MatrixXd K(3 * nv, cols);
  for (int r = 0; r < 3 * nv; ++r) for (int c = 0; c < cols; ++c) K(r, c) = q[static_cast<std::size_t>(r * cols + c)];
  const Eigen::MatrixXd X = nxr::signal::formOfFreeColumns(nxr::signal::Form::curl_tan, K, op);
  CHECK(rel(X, curl, cols) < 1e-4);

  // a cortex of two pieces with no stored glue: refused, saying what to run
  {
    const SurfaceLevel SL = read_surface_level(s1, S, 3);
    const internal::GluedDomain none = internal::glued_domain(s1, S);
    CHECK(!none.found && has(none.why, "run glue"));
    SpatialBandOptions o; o.glued = true; o.transform = "chebyshev";
    CHECK(has(what_of([&] { make_spatial_band(ds, s1, S, SL, o); }), "run glue"));
    CHECK(has(what_of([&] { make_spatial_band(S, SL, o); }), "subject"));              // without a subject there is no glued surface
  }

  // THE EIGENMODES, computed and stored (the glued surface with them); the plan key names the glued evaluation
  SpatialBandOptions o; o.glued = true; o.spatial_shape = "mexhat"; o.modes = 500;
  SpatialBandOptions og = o; og.glued = false;
  const SurfaceLevel SL3 = read_surface_level(s1, S, 3);
  CHECK(plan_spatial_band(s1, S, SL3, o).key["glued"] == true && !plan_spatial_band(s1, S, SL3, og).key.contains("glued"));
  const SpatialBand b3 = make_spatial_band(ds, s1, S, SL3, o);
  CHECK(b3.info["transform"] == "eigen" && b3.apply_glued && b3.glued_surface == "cortex_glued" && b3.glued_vertices == nG && b3.info["glued"]["bridge_vertices"].get<std::int64_t>() > 0);
  // the glue is nxr's
  const SurfaceInput GS = read_surface(s1, "cortex_glued", true);
  CHECK(GS.n_vertices() == nG && static_cast<int>(GS.faces.size() / 3) == static_cast<int>(gfaces.size() / 3));
  double dp = 0;
  for (std::size_t i = 0; i < gpos.size() && i < GS.vertices.size(); ++i) dp = std::max(dp, std::abs(GS.vertices[i] - gpos[i]));
  bool same_faces = GS.faces == gfaces;
  std::printf("  glue: %d glued vertices (%lld bridge); positions vs nxr's glue_mesh max %.1e; faces %s\n", nG, static_cast<long long>(b3.info["glued"]["bridge_vertices"].get<std::int64_t>()), dp, same_faces ? "identical" : "DIFFER");
  CHECK(dp < 1e-6 && same_faces);
  const Eigen::MatrixXd Y3 = b3.apply_glued(X);
  CHECK(Y3.rows() == nG && Y3.cols() == cols);
  const double e_eig = rel(Y3, hat.at(3), cols);
  std::printf("  level 3, stored eigenmodes (%d) vs nxr's hat on every glued vertex: max relative difference %.2e\n", b3.info["modes"].get<int>(), e_eig);
  CHECK(e_eig < 5e-3);
  // the bridge carries the band; the native read-back is its kept vertices; the cut wall carries none
  double bmax = 0, top = 0;
  for (int i = 0; i < nG; ++i) { double m = 0; for (int c = 0; c < cols; ++c) m = std::max(m, std::abs(Y3(i, c))); top = std::max(top, m); if (keep[static_cast<std::size_t>(i)] < 0) bmax = std::max(bmax, m); }
  std::printf("  the bridge: largest |band| there %.3g of the surface's %.3g\n", bmax, top);
  CHECK(bmax > 0.05 * top);
  const Eigen::MatrixXd R3 = b3.apply(X);
  bool readback = R3.rows() == nv;
  for (int i = 0; i < nG && readback; ++i) if (keep[static_cast<std::size_t>(i)] >= 0) readback = (R3.row(keep[static_cast<std::size_t>(i)]) - Y3.row(i)).cwiseAbs().maxCoeff() < 1e-12;
  CHECK(readback);
  for (int i = 0; i < nv; ++i) if (wallv[static_cast<std::size_t>(i)]) CHECK(R3.row(i).isZero());
  // the field is linear: the operator folded once equals the operator on the sum
  CHECK((b3.apply_glued(X.col(0) + X.col(1)) - (Y3.col(0) + Y3.col(1))).cwiseAbs().maxCoeff() < 1e-9 * (1 + top));
  // level 2
  { const SurfaceLevel SL2 = read_surface_level(s1, S, 2); const double e = rel(make_spatial_band(ds, s1, S, SL2, o).apply_glued(X), hat.at(2), cols); std::printf("  level 2, stored eigenmodes vs nxr's hat: %.2e\n", e); CHECK(e < 5e-3); }
  // THE DENSE MODES and the SERIES of the glued mesh (the fallback routes) equal it too
  { SpatialBandOptions ox = o; ox.transform = "exact"; const double e = rel(make_spatial_band(ds, s1, S, SL3, ox).apply_glued(X), hat.at(3), cols); std::printf("  level 3, dense modes of the glued mesh vs nxr's: %.2e\n", e); CHECK(e < 1e-6); }
  { SpatialBandOptions oc = o; oc.transform = "chebyshev"; const SpatialBand c = make_spatial_band(ds, s1, S, SL3, oc); const double e = rel(c.apply_glued(X), hat.at(3), cols);
    std::printf("  level 3, Chebyshev series (degree %d) of the glued mesh vs nxr's: %.2e\n", c.info["order"].get<int>(), e); CHECK(e < 2e-2 && c.warnings.empty()); }
  // A LEVEL BEYOND THE STORED MODES: the series of the GLUED mesh, announced
  {
    const SurfaceLevel SL6 = read_surface_level(s1, S, 6);
    std::vector<Json> said;
    const SpatialBand f = make_spatial_band(ds, s1, S, SL6, o, [&](const Json& j) { said.push_back(j); });
    CHECK(f.info["transform"] == "chebyshev" && f.warnings.size() == 1 && has(f.warnings[0], "CHEBYSHEV fallback") && f.apply_glued && f.glued_vertices == nG && f.apply_glued(X).rows() == nG);
    bool warned = false;
    for (const Json& j : said) warned = warned || (j.contains("warning") && j["warning"] == f.warnings[0]);
    CHECK(warned);
    SpatialBandOptions oe = o; oe.transform = "eigen";
    CHECK(has(what_of([&] { make_spatial_band(ds, s1, S, SL6, oe); }), "do not reach"));
    SpatialBandOptions orz = o; orz.riesz = true;
    CHECK(has(what_of([&] { make_spatial_band(ds, s1, S, SL3, orz); }), "Riesz"));
  }
  // ONE CLOSED PIECE: its own glued domain, no bridge
  {
    SubjectWriter s2 = SubjectWriter::open(ds, "G2");
    nxrtest::make_tiles(ds, s2);
    const SurfaceInput S2 = read_surface(s2, std::string("cortex"), true);
    const SurfaceLevel L2 = read_surface_level(s2, S2, 3);
    const internal::GluedDomain d = internal::glued_domain(s2, S2);
    CHECK(d.found && d.surface.n_vertices() == S2.n_vertices() && d.support.size() == static_cast<std::size_t>(S2.n_vertices()));
    SpatialBandOptions o2; o2.glued = true; o2.transform = "exact";
    const SpatialBand b = make_spatial_band(ds, s2, S2, L2, o2);
    const Eigen::MatrixXd Z = X.topRows(S2.n_vertices());
    CHECK(b.glued_vertices == S2.n_vertices() && b.apply_glued(Z).rows() == S2.n_vertices() && (b.apply_glued(Z) - b.apply(Z)).cwiseAbs().maxCoeff() < 1e-12);
  }
  return nxrtest::finish("test_glued_form") || g_failures ? 1 : 0;
}
