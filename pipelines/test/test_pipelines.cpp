// The cache and eigen_solve on a synthetic icosphere (no real data): a key, a miss then a hit with identical arrays, a changed K or mesh misses,
// a stale package is refused without --replace and moved aside with it; a surface of two pieces (two spheres, a cap of each facing the other as its
// 'unassigned' Desikan-Killiany wall) is GLUED by anatomy and solved on the glued surface, stored in the layout nxr stores one in (the glued surface, its
// bridge, the support = the kept vertices); without the atlas it is refused by name.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

#include <Eigen/Sparse>

#include "check.h"
#include "run_definition.h"
#include "nxr/io/task/task.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/subject_api.h"
#include "nxr/operators/dirac.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/surface_maps.h"
#include "nxr/pipelines/eigen.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
void icosphere(int levels, double radius, std::vector<double>& V, std::vector<std::int32_t>& F) {
  const double t = (1 + std::sqrt(5.0)) / 2;
  const double base[12][3] = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  const int tri[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                          {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9}, {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
  V.clear(); F.clear();
  for (auto& b : base) V.insert(V.end(), b, b + 3);
  for (auto& f : tri) F.insert(F.end(), f, f + 3);
  for (int l = 0; l < levels; ++l) {
    std::map<std::pair<int, int>, int> mid;
    auto midpoint = [&](int a, int b) {
      auto k = std::make_pair(std::min(a, b), std::max(a, b));
      auto it = mid.find(k);
      if (it != mid.end()) return it->second;
      const int id = static_cast<int>(V.size() / 3);
      for (int c = 0; c < 3; ++c) V.push_back((V[3 * a + c] + V[3 * b + c]) / 2);
      return mid[k] = id;
    };
    std::vector<std::int32_t> G;
    for (std::size_t f = 0; f < F.size(); f += 3) {
      const int a = F[f], b = F[f + 1], c = F[f + 2], ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      for (int q : {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca}) G.push_back(q);
    }
    F = G;
  }
  for (std::size_t i = 0; i < V.size(); i += 3) {
    const double n = std::sqrt(V[i] * V[i] + V[i + 1] * V[i + 1] + V[i + 2] * V[i + 2]);
    for (int c = 0; c < 3; ++c) V[i + static_cast<std::size_t>(c)] *= radius / n;
  }
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_io_test_pipelines_" + std::to_string(::getpid()));
  fs::remove_all(root);

  // the keys: nxr's mesh key is "<nF>:<8 hex>"; the cache key's JSON is the four parts
  CHECK_EQ(mesh_key({0, 1, 2}), std::string("1:") + mesh_key({0, 1, 2}).substr(2));
  CHECK(mesh_key({0, 1, 2}) != mesh_key({0, 2, 1}));
  CacheKey k1{"p", Json{{"modes", 10}}, Json{{"mesh", "a"}}, "v1"};
  CacheKey k2 = k1;
  CHECK(k1 == k2);
  k2.params["modes"] = 12;
  CHECK(!(k1 == k2));

  std::vector<double> V;
  std::vector<std::int32_t> F;
  icosphere(2, 0.08, V, F);                                   // 162 vertices
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] { add_surface(sub, {.name = "cortex"}, V, F); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  EigenSolveOptions opt;
  opt.modes = 20;
  const EigenSolveResult a = eigen_solve(ds, sub, opt);
  CHECK(!a.hit);
  CHECK(a.modes.n_modes() >= 10);
  CHECK(a.modes.n_exact > 0 && a.modes.n_coexact > 0);
  for (std::size_t i = 1; i < a.modes.mu.size(); ++i) CHECK(a.modes.mu[i] >= a.modes.mu[i - 1]);
  const EigenSolveResult b = eigen_solve(ds, sub, opt);
  CHECK(b.hit);
  CHECK(b.modes.mu == a.modes.mu);
  CHECK(b.modes.kind == a.modes.kind);
  CHECK(b.modes.eigenvectors == a.modes.eigenvectors);        // the stored float32 bytes, back
  CHECK(b.seconds < a.seconds);

  // (the cache key is the definition runner's now: its hit and miss, another K and --replace are what is checked below; a hand-made key is no longer the store's lookup)

  // another K: a miss; the stored package is refused without --replace
  opt.modes = 24;
  CHECK_EQ(([&] { try { eigen_solve(ds, sub, opt); } catch (const Error& e) { return e.code(); } return std::string(); })(), std::string(code::already_exists));
  opt.replace = true;
  const EigenSolveResult c = eigen_solve(ds, sub, opt);
  CHECK(!c.hit);
  CHECK(c.info["replaced"].get<bool>());
  CHECK(c.modes.n_modes() > a.modes.n_modes());
  CHECK(fs::exists(ds.root() / ("_replaced-" + now_utc(false).substr(0, 10))));
  CHECK(eigen_solve(ds, sub, opt).hit);

  // a surface of TWO pieces: two spheres 0.2 m apart, each with a cap facing the other as its medial wall
  std::vector<double> W = V;
  std::vector<std::int32_t> G = F;
  const std::int32_t n1 = static_cast<std::int32_t>(V.size() / 3);
  for (std::size_t i = 0; i < V.size(); i += 3) { W.push_back(V[i] + 0.2); W.push_back(V[i + 1]); W.push_back(V[i + 2]); }
  for (auto f : F) G.push_back(f + n1);
  std::vector<std::int32_t> dk(static_cast<std::size_t>(2 * n1), 1);           // level 1: cortex; level 0: unassigned (the wall)
  std::vector<std::uint8_t> wall(dk.size(), 0);
  for (std::int32_t i = 0; i < 2 * n1; ++i)
    if ((i < n1 && W[3 * i] > 0.05) || (i >= n1 && W[3 * i] - 0.2 < -0.05)) { dk[static_cast<std::size_t>(i)] = 0; wall[static_cast<std::size_t>(i)] = 1; }
  {
    SubjectWriter s2 = SubjectWriter::create(ds, "S2");     // no atlas: refused by name
    s2.compose([&] { add_surface(s2, {.name = "cortex"}, W, G); });
    SubjectWriter s3 = SubjectWriter::create(ds, "S3");     // the Desikan-Killiany atlas with its wall
    SurfaceOptions so{.name = "cortex"};
    so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
    s3.compose([&] { add_surface(s3, so, W, G); });
  }
  SubjectWriter s2 = SubjectWriter::open(ds, "S2");
  CHECK_EQ(([&] { try { eigen_solve(ds, s2, opt); } catch (const Error& e) { return e.code(); } return std::string(); })(), std::string(code::no_such_row));
  SubjectWriter s3 = SubjectWriter::open(ds, "S3");
  opt.replace = false;
  // two identical spheres make exactly cocircular quads: flipped to Delaunay their cotan weights are 0 up to rounding (about -1e-17), which the 2-form ladder inverts: the named *1 floor keeps it defined, and the
  // stored *1 is the floored one (the Gram check below reads it)
  opt.method.star1_floor = 1e-9;
  opt.surface = "cortex";                                     // the glued surface is another surface of the subject now: the caller names the one to solve
  const EigenSolveResult g = eigen_solve(ds, s3, opt);
  // the glue the solve used is nxr-compute's glueMesh on the positions as the Studio receives them (float32)
  std::vector<double> W32(W.size());
  for (std::size_t i = 0; i < W.size(); ++i) W32[i] = static_cast<double>(static_cast<float>(W[i]));
  const nxr::geometry::glue::Glued gm = nxr::geometry::glue::glueMesh(W32, G, wall);
  const std::int64_t Vg = static_cast<std::int64_t>(gm.keep.size()), kept = std::count_if(gm.keep.begin(), gm.keep.end(), [](std::int32_t k) { return k >= 0; });
  CHECK(!g.hit && g.info["glued"].get<bool>());
  CHECK(gm.bridge.vertices > 0 && Vg == kept + gm.bridge.vertices);
  CHECK_EQ(g.info["rows"].get<std::int64_t>(), Vg);
  CHECK_EQ(g.modes.n_vertices, Vg);
  CHECK(static_cast<std::int64_t>(g.modes.eigenvectors.size()) == Vg * g.modes.n_modes());
  CHECK(g.modes.n_exact > 0 && g.modes.n_coexact > 0);
  {
    const auto gs = s3.find("manifold", "source/cortex_glued");
    const auto br = s3.find("selection", "source/cortex_glued_bridge");
    const auto op = s3.find("operator", "hodgeDirac__cortex_glued_eigenvectors");
    const auto tri = s3.find("manifold", "source/cortex_glued_intrinsic");
    CHECK(gs && br && op && tri);
    // ONE operator from the eigenvalue line to the intrinsic triangulation (made from the glued surface, its positions shared), cell key 0+1+2: rows V + E + F
    CHECK_EQ(text(*op, "to_manifold_id"), text(*tri, "id"));
    CHECK_EQ(text(*tri, "derived_from_id"), text(*gs, "id"));
    CHECK(opt_text(*op, "to_cell") == std::optional<std::string>("0+1+2") && opt_text(*op, "from_cell") == std::optional<std::string>("0"));
    CHECK_EQ(integer(*gs, "n_vertices"), Vg);
    CHECK_EQ(integer(*op, "n_rows"), Vg + integer(*tri, "n_edges") + integer(*tri, "n_faces"));
    {   // the triangulation reads the glued surface's positions (the same array, not copied), through a geometry of its own
      const auto gg = nxr::io::manifold_row_named(s3, "cortex_glued"), tg = nxr::io::manifold_row_named(s3, "cortex_glued_intrinsic");
      CHECK(gg && tg && text(*gg, "pp") == text(*tg, "pp") && text(*gs, "geometry_id") != text(*tri, "geometry_id"));
    }
    // d0, d1 and the stars are ANALYTIC operators on the triangulation (method dec, no bytes), their shapes in its cells
    for (const char* nm : {"cortex_glued_intrinsic_d0", "cortex_glued_intrinsic_d1", "cortex_glued_intrinsic_star0", "cortex_glued_intrinsic_star1", "cortex_glued_intrinsic_star2"}) {
      nxr::io::task::Want w{.cls = nxr::io::task::Cls::operator_}; w.name = std::string(nm);
      const auto a = nxr::io::task::Subject::of(s3).find(w);
      CHECK(a && a->row.value("layout", std::string()) == "analytic" && a->method == std::optional<std::string>("dec") && !a->path && a->from_manifold_id == std::optional<std::string>(text(*tri, "id")));
    }
    CHECK(!s3.find("operator", "hodgeDirac__cortex_glued_forms1") && !s3.find("operator", "hodgeDirac__cortex_glued_forms2"));
    const std::vector<std::uint8_t> raw = ArrayReader::open(s3.store(), ("source/cortex_glued/faces")).read();
    CHECK(raw.size() == gm.faces.size() * sizeof(std::int32_t) && std::memcmp(raw.data(), gm.faces.data(), raw.size()) == 0);   // the glued faces, as glueMesh gives them
    const std::vector<std::uint8_t> sup = ArrayReader::open(s3.store(), (text(*s3.find("selection", "source/hodgeDirac__cortex_glued_support"), "array_path"))).read();
    // the support: the glued vertices that are cortex (the bridge is not), as original vertex numbers, in order
    CHECK(sup.size() == static_cast<std::size_t>(kept) * sizeof(std::int32_t) && std::memcmp(sup.data(), gm.keep.data(), sup.size()) == 0);
  }
  // THE FULL FORMS: the intrinsic complex (d0, d1, the stars) and the 2-forms, read back; built from them alone, every mode's 1-form part has energy
  // λ/2 in ⋆1 and the 1-form parts are mutually orthogonal (the Dirac eigenvectors' orthonormality): reading a 1-form into the modes is inner products
  {
    auto arr = [&](const std::string& rel) {
      ArrayReader a = ArrayReader::open(s3.store(), (rel));
      const auto raw = a.read();
      return std::make_pair(raw, a.dtype());
    };
    auto as_d = [&](const std::string& rel) {
      const auto [raw, dt] = arr(rel);
      std::vector<double> v;
      if (dt == DType::Float64) { v.resize(raw.size() / 8); std::memcpy(v.data(), raw.data(), raw.size()); }
      else if (dt == DType::Float32) { std::vector<float> f(raw.size() / 4); std::memcpy(f.data(), raw.data(), raw.size()); v.assign(f.begin(), f.end()); }
      else if (dt == DType::Int64) { std::vector<std::int64_t> f(raw.size() / 8); std::memcpy(f.data(), raw.data(), raw.size()); v.assign(f.begin(), f.end()); }
      else { std::vector<std::int32_t> f(raw.size() / 4); std::memcpy(f.data(), raw.data(), raw.size()); v.assign(f.begin(), f.end()); }
      return v;
    };
    const auto im = s3.find("manifold", "source/cortex_glued_intrinsic");
    CHECK(im.has_value());
    const std::int64_t nE = integer(*im, "n_edges"), nF = integer(*im, "n_faces");
    CHECK_EQ(2 * nE, 3 * nF);
    // the analytic operators evaluated as their method says (the construction on the glued surface, the solve's floor): d0, d1 and the stars
    const SurfaceInput glued = read_surface(s3, std::string("cortex_glued"));
    nxr::geometry::DelaunayOptions dopt;
    const nxr::geometry::SurfaceArrays sa = nxr::geometry::surfaceArrays(glued.vertices.data(), static_cast<int>(Vg), glued.faces.data(), static_cast<int>(glued.n_faces()), nxr::geometry::intrinsicOptions(true, dopt));
    const nxr::operators::hodgedirac::HodgeDirac hd = nxr::operators::hodgedirac::assemble(sa, 1e-9, 1e-6, 1e-6);
    CHECK(hd.nE == nE && hd.nF == nF);
    {
      const std::vector<std::uint8_t> raw = ArrayReader::open(s3.store(), text(*im, "path") + "/faces").read();   // its stored faces are the construction's
      std::vector<std::int32_t> fs(raw.size() / 4); std::memcpy(fs.data(), raw.data(), raw.size());
      CHECK(fs == hd.complex.faces);
    }
    const Eigen::SparseMatrix<double> d0 = hd.d0, d1 = hd.d1;
    CHECK(Eigen::SparseMatrix<double>(d1 * d0).norm() == 0.0);
    const std::vector<double> st0(hd.star0.data(), hd.star0.data() + Vg), st1(hd.star1.data(), hd.star1.data() + nE), st2(hd.star2.data(), hd.star2.data() + nF);
    CHECK(*std::min_element(st1.begin(), st1.end()) >= 0);                     // Delaunay: every cotan weight non-negative (after the floor)
    const auto mu = as_d(text(*s3.find("manifold", "eigenvalue/hodgeDirac__cortex_glued"), "path"));
    // the kinds: a labelling of the eigenvalue line (label 0 exact, 2 coexact)
    const auto ks = nxr::io::selection::named_row(s3, "hodgeDirac__cortex_glued_kinds");
    CHECK(ks.has_value());
    const nxr::io::selection::ElementMembers em = nxr::io::selection::read_element_members(s3, text(*ks, "id"));
    std::vector<std::uint8_t> kind(em.member_of.size());
    for (std::size_t k = 0; k < kind.size(); ++k) kind[k] = static_cast<std::uint8_t>(em.member_codes.empty() ? em.member_of[k] : em.member_codes[static_cast<std::size_t>(em.member_of[k])]);
    CHECK(kind.size() == mu.size());
    // the mixed forms: rows 0-forms [0, V) · 1-forms [V, V+E) · 2-forms [V+E, V+E+F)
    const auto ALL = as_d(text(*s3.find("operator", "hodgeDirac__cortex_glued_eigenvectors"), "path"));
    const std::int64_t Kall = static_cast<std::int64_t>(mu.size());
    CHECK(static_cast<std::int64_t>(ALL.size()) == (Vg + nE + nF) * Kall);
    const std::vector<double> P(ALL.begin(), ALL.begin() + Vg * Kall), X1(ALL.begin() + Vg * Kall, ALL.begin() + (Vg + nE) * Kall), X2(ALL.begin() + (Vg + nE) * Kall, ALL.end());
    const std::int64_t K = static_cast<std::int64_t>(mu.size()), Kc = std::min<std::int64_t>(K, 24);
    Eigen::Map<const Eigen::VectorXd> s0(st0.data(), Vg), s1(st1.data(), nE), s2(st2.data(), nF);
    Eigen::MatrixXd W(nE, Kc);                                                   // the first Kc modes' 1-form parts
    double norm02 = 0;
    for (std::int64_t k = 0; k < Kc; ++k) {
      Eigen::VectorXd x;
      if (kind[k] == 0) {
        Eigen::VectorXd p(Vg); for (std::int64_t v = 0; v < Vg; ++v) p(v) = P[v * K + k];
        norm02 = std::max(norm02, std::abs(p.dot(s0.asDiagonal() * p) - 0.5));
        x = d0 * p;                                                              // dψ/√2
      } else {
        Eigen::VectorXd c(nF); for (std::int64_t f = 0; f < nF; ++f) c(f) = X2[f * K + k];
        norm02 = std::max(norm02, std::abs(c.dot(s2.asDiagonal() * c) - 0.5));
        x = s1.cwiseInverse().asDiagonal() * (d1.transpose() * (s2.asDiagonal() * c));   // ⋆1⁻¹ d1ᵀ ⋆2 χ/√2
      }
      W.col(k) = x / mu[k];
    }
    const Eigen::MatrixXd Gm = W.transpose() * s1.asDiagonal() * W;            // should be I/2
    const double diagErr = (Gm.diagonal().array() - 0.5).abs().maxCoeff(), off = (Gm - Gm.diagonal().asDiagonal().toDenseMatrix()).cwiseAbs().maxCoeff();
    std::printf("  full forms: %lld edges, %lld faces; |0/2-form norm − 1/2| %.1e; 1-form Gram: |diag − 1/2| %.1e, off-diagonal %.1e\n", (long long)nE, (long long)nF, norm02, diagErr, off);
    CHECK(norm02 < 1e-4);
    CHECK(diagErr < 1e-4);
    CHECK(off < 1e-4);
    {   // the stored 1-form rows (the solve's own) are the parts rebuilt here from the 0- and 2-form rows
      CHECK(static_cast<std::int64_t>(X1.size()) == nE * K);
      // A coexact part is ⋆1⁻¹ d1ᵀ ⋆2 χ: on an edge whose ⋆1 is ~0 (the glued bridge) the sum cancels to the same order as ⋆1, so rebuilding it from the float32 2-form parts and ⋆1 loses it;
      // the solve's own double values (stored) are the accurate ones. Compared on the edges with a ⋆1 of at least 1e-3 of the mean (every edge for an exact mode); the others are counted.
      const double s1mean = s1.mean();
      std::int64_t ill = 0;
      for (std::int64_t e = 0; e < nE; ++e) ill += s1(e) < 1e-3 * s1mean;
      double e1 = 0, sc = 0;
      for (std::int64_t k = 0; k < Kc; ++k) for (std::int64_t e = 0; e < nE; ++e) {
        sc = std::max(sc, std::abs(W(e, k)));
        if (kind[k] == 0 || s1(e) >= 1e-3 * s1mean) e1 = std::max(e1, std::abs(X1[e * K + k] - W(e, k)));
      }
      std::printf("  edges with a star1 under 1e-3 of the mean (a coexact part rebuilt from float32 parts is lost there): %lld of %lld\n", (long long)ill, (long long)nE);
      std::printf("  stored 1-form parts vs rebuilt: max |difference| %.1e (scale %.1e)\n", e1, sc);
      CHECK(e1 < 1e-5 * sc);
    }
  }
  CHECK(eigen_solve(ds, s3, opt).hit);
  // a re-solve with --replace whose glue is the same surface KEEPS the glued surface: a field another run stored on it stays
  {
    const std::string gid = text(*s3.find("manifold", "source/cortex_glued"), "id");
    std::vector<float> ones(static_cast<std::size_t>(Vg), 1.0f);
    s3.compose([&] {
      FieldSpec f; f.name = "other_run"; f.path = nxr::io::task::Subject::of(s3).node_path(gid, "other_run"); f.kind = "test"; f.manifold_id = gid; f.value_type = "scalar";
      f.data = view(ones, {Vg}); s3.field(std::move(f));
    });
    EigenSolveOptions o2 = opt; o2.modes = 26; o2.replace = true;
    const EigenSolveResult r2 = eigen_solve(ds, s3, o2);
    CHECK(!r2.hit && r2.info["replaced"].get<bool>());
    CHECK_EQ(text(*s3.find("manifold", "source/cortex_glued"), "id"), gid);
    CHECK(s3.find("field", nxr::io::task::Subject::of(s3).node_path(gid, "other_run")).has_value());
    CHECK(opt_text(*s3.find("operator", "hodgeDirac__cortex_glued_eigenvectors"), "to_cell") == std::optional<std::string>("0+1+2"));
  }
  // the glued sphere pair is ONE closed surface of genus 0: two zero modes dropped, the first exact mode non-constant across the bridge
  CHECK(g.modes.mu.front() > 0);

  // the default subject's Hodge–Dirac basis (`dataset create --modes K`): the same solve on the template subject's pial, stored as any subject's
  {
    DatasetWriter dd = create_dataset(root / "tmpl", "demo");
    SubjectOptions so; so.kind = "template";
    SubjectWriter def = SubjectWriter::create(dd, "@default_subject", so);
    def.compose([&] { SurfaceOptions s; s.name = "cortex_pial"; s.primary = true; add_surface(def, s, V, F); });
    SubjectWriter dsub = SubjectWriter::open(dd, "@default_subject");
    EigenSolveOptions eo; eo.modes = 20; eo.surface = "cortex_pial";
    const EigenSolveResult r = eigen_solve(dd, dsub, eo);
    CHECK(!r.hit && r.modes.n_exact > 0 && r.modes.n_coexact > 0);
    CHECK(dsub.find("operator", "hodgeDirac_eigenvectors").has_value());
    CHECK(dsub.find("manifold", "source/cortex_pial_intrinsic").has_value());
    CHECK(opt_text(*dsub.find("operator", "hodgeDirac_eigenvectors"), "to_cell") == std::optional<std::string>("0+1+2"));
    CHECK(dsub.find("manifold", "eigenvalue/hodgeDirac").has_value());
    dd.close();
  }

  ds.close();
  fs::remove_all(root);
  std::cout << "test_pipelines: ok\n";
  return 0;
}
