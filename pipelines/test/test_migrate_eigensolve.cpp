// `nxr-neuro migrate eigensolve` (pipelines/src/eigen_migrate.cpp) on a synthetic sphere (never a real dataset): a solve stored in the EARLIER form (three operators _eigenvectors -> the surface,
// _forms2 -> the triangulation's faces, no _forms1; a kind field; the triangulation with its own positions; d0, d1 as csr; the stars and signposts as fields), written here through io's task API
// from a tiny solve made by the test, is converted in place:
//   - ONE eigenvectors operator, line -> the triangulation '0+1+2': its 0- and 2-form rows the earlier bytes exactly, its 1-form rows made from the stored parts (no solve) = the solve's own 1-form parts;
//   - the kinds a labelling of the line, the eigenvalues a field of the new operator, the triangulation on the surface's positions, d0, d1 and the stars analytic (method dec), the signposts kept;
//   - the reader (`stored_eigensolve`) reads it; a second migrate is a no-op; a stem of a form it does not know is refused, and nothing of it changes.
// and `nxr-neuro migrate glue`: a glued cortex of two spheres stored in the earlier form (the field <glued>_map, the indices <glued>_bridge) becomes the hemispheres labelling of the glued cortex
// (Structures carried over, `bridge` added) and the original's kept vertices; the correspondence read back equals the glue's; a labelling that disagrees with the pieces is refused; idempotent.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unistd.h>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/io.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/task/task.h"
#include "nxr/io/subject_api.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/eigen_migrate.h"
#include "nxr/pipelines/input.h"
#include "nxr/geometry/surface_maps.h"
#include "hemispheres.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
template <class T>
std::vector<T> bytes_as(SubjectWriter& sub, const std::string& path) {
  const auto raw = ArrayReader::open(sub.store(), path).read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}
std::string refusal(const std::function<void()>& f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_migrate_eigensolve_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::vector<double> Vx; std::vector<std::int32_t> Fx;
  icosphere(2, Vx, Fx, 0.08);
  for (auto& x : Vx) x = static_cast<double>(static_cast<float>(x));     // the positions as the solve receives them
  DatasetWriter ds = create_dataset(root, "demo");
  { SubjectWriter s = SubjectWriter::create(ds, "S1"); s.compose([&] { add_surface(s, {.name = "cortex", .primary = true}, Vx, Fx); }); }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  const std::string surf_id = *tk::Subject::of(sub).manifold_named("cortex");

  // a tiny solve, stored in the EARLIER form
  HodgeDirac h = solve_hodge_dirac(Vx, Fx, 20);
  const std::int64_t V = h.n_vertices, K = h.n_modes(), E = static_cast<std::int64_t>(h.lengths.size()), F = static_cast<std::int64_t>(h.faces.size() / 3);
  const std::string stem = "hodgeDirac";
  {
    const tk::Subject ts = tk::Subject::of(sub);
    tk::Unit u = ts.unit(tk::By{"earlier eigenmodes", Json{{"solve", Json{{"options", Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}}}}}}});
    tk::PointsOut p; p.name = stem; p.output = "modes"; p.cs_key = "eigenvalue:" + stem; p.operator_name = "hodgeDirac"; p.derived_from_id = surf_id;
    p.positions.shape = {K}; p.positions.f64 = h.mu;
    const std::string line = u.points(p);
    tk::ComplexOut c; c.name = "cortex_intrinsic"; c.output = "complex"; c.derived_from_id = surf_id; c.edges = h.edges; c.faces = h.faces;
    c.vertices.shape = {V, 3}; c.vertices.f64 = Vx; c.producer = Json{{"delaunay", Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}}}};
    const std::string tri = u.cell_complex(c);
    auto dense = [&](const std::string& name, const std::string& to, const char* to_cell, std::int64_t rows, const std::vector<float>& v) {
      tk::OperatorOut o; o.name = name; o.output = name; o.kind = "eigenvectors"; o.from_manifold_id = line; o.to_manifold_id = to; o.from_cell = "0"; o.to_cell = to_cell;
      o.data.shape = {rows, K}; o.data.f32 = v;
      return u.operator_(o);
    };
    const std::string evid = dense(stem + "_eigenvectors", surf_id, "0", V, h.eigenvectors);
    dense(stem + "_forms2", tri, "2", F, h.forms2);
    tk::SelectionOut sp; sp.name = stem + "_support"; sp.output = "support"; sp.type = "indices"; sp.manifold_id = surf_id;
    for (std::int64_t v = 0; v < V; ++v) sp.codes.push_back(v);
    u.selection(sp);                    // no _forms1: a solve made before they were stored
    tk::FieldOut kf; kf.name = stem + "_kind"; kf.output = "kind"; kf.kind = "form degree"; kf.manifold_id = line; kf.of_operator_id = evid; kf.value_type = "integer";
    kf.data.shape = {K}; kf.data.u8 = h.kind; u.field(kf);
    // d0 [E x V], d1 [F x E] as csr; the stars and signposts as fields
    std::vector<std::int64_t> ip0, ip1; std::vector<std::int32_t> ix0, ix1; std::vector<double> dv0, dv1;
    for (std::int64_t e = 0; e < E; ++e) {
      ip0.push_back(2 * e);
      const std::int32_t a = h.edges[2 * e], b = h.edges[2 * e + 1];
      ix0.push_back(std::min(a, b)); dv0.push_back(a < b ? -1.0 : 1.0);
      ix0.push_back(std::max(a, b)); dv0.push_back(a < b ? 1.0 : -1.0);
    }
    ip0.push_back(2 * E);
    for (std::int64_t f = 0; f < F; ++f) {
      ip1.push_back(3 * f);
      std::vector<std::pair<std::int32_t, double>> row;
      for (int k = 0; k < 3; ++k) { const std::int32_t se = h.face_edges[3 * f + k]; row.push_back({std::abs(se) - 1, se > 0 ? 1.0 : -1.0}); }
      std::sort(row.begin(), row.end());
      for (const auto& [e, sgn] : row) { ix1.push_back(e); dv1.push_back(sgn); }
    }
    ip1.push_back(3 * F);
    auto csr = [&](const char* name, std::int64_t rows, std::int64_t cols, const std::vector<std::int64_t>& ip, const std::vector<std::int32_t>& ix, const std::vector<double>& dv, const char* fc, const char* tcell) {
      tk::OperatorOut o; o.name = std::string("cortex_intrinsic_") + name; o.output = name; o.kind = "exterior derivative"; o.from_manifold_id = tri; o.to_manifold_id = tri; o.from_cell = fc; o.to_cell = tcell;
      o.sparse = tk::OperatorOut::Sparse{"csr", rows, cols, ip, ix}; o.data.shape = {static_cast<std::int64_t>(dv.size())}; o.data.f64 = dv;
      u.operator_(o);
    };
    csr("d0", E, V, ip0, ix0, dv0, "0", "1");
    csr("d1", F, E, ip1, ix1, dv1, "1", "2");
    auto field = [&](const char* name, const char* cell, const std::vector<double>& v) {
      tk::FieldOut f; f.name = std::string("cortex_intrinsic_") + name; f.output = name; f.kind = "intrinsic complex"; f.manifold_id = tri; f.cell = cell;
      f.data.shape = {static_cast<std::int64_t>(v.size())}; f.data.f64 = v; u.field(f);
    };
    field("length", "1", h.lengths); field("star0", "0", h.star0); field("star1", "1", h.star1); field("star2", "2", h.star2); field("angle_sum", "0", h.angle_sum);
    u.commit();
  }
  // a second stem of a form it does not know: eigenvectors with no 2-form parts
  {
    const tk::Subject ts = tk::Subject::of(sub);
    tk::Unit u = ts.unit(tk::By{"broken", Json()});
    tk::PointsOut p; p.name = "broken"; p.output = "modes"; p.cs_key = "eigenvalue:broken"; p.operator_name = "hodgeDirac"; p.derived_from_id = surf_id;
    p.positions.shape = {K}; p.positions.f64 = h.mu;
    const std::string line = u.points(p);
    tk::OperatorOut o; o.name = "broken_eigenvectors"; o.output = "eigenvectors"; o.kind = "eigenvectors"; o.from_manifold_id = line; o.to_manifold_id = surf_id;
    o.data.shape = {V, K}; o.data.f32 = h.eigenvectors; u.operator_(o);
    u.commit();
  }
  const std::string why = refusal([&] { migrate_eigensolve(sub, std::string("broken")); });
  std::printf("  an unknown form: %s\n", why.substr(0, 160).c_str());
  CHECK(why.find("not a stored eigensolve of a form") != std::string::npos);
  CHECK(sub.find("operator", "broken_eigenvectors").has_value());

  const std::vector<float> ev_before = h.eigenvectors, f2_before = h.forms2;
  const Json r = migrate_eigensolve(sub, stem);
  std::printf("  migrated: %s\n", r.dump().substr(0, 200).c_str());
  CHECK(r["stems"][0]["status"] == "migrated");

  // the new form, read back
  const StoredEigensolve e = stored_eigensolve(sub, stem);
  CHECK(e.V == V && e.E == E && e.F == F && e.K == K);
  CHECK(e.kind == h.kind && e.mu == h.mu);
  CHECK(opt_text(e.eigenvectors, "to_cell") == std::optional<std::string>("0+1+2"));
  CHECK_EQ(text(e.surface, "id"), surf_id);
  const std::vector<float> all = bytes_as<float>(sub, text(e.eigenvectors, "path"));
  CHECK(static_cast<std::int64_t>(all.size()) == (V + E + F) * K);
  CHECK(std::equal(ev_before.begin(), ev_before.end(), all.begin()));                         // the 0-form rows: the earlier bytes
  CHECK(std::equal(f2_before.begin(), f2_before.end(), all.begin() + (V + E) * K));            // the 2-form rows: the earlier bytes
  double d1max = 0, scale = 0;                                                                  // the 1-form rows made from the stored parts = the solve's own (to float32)
  for (std::int64_t i = 0; i < E * K; ++i) { d1max = std::max(d1max, std::abs(static_cast<double>(all[V * K + i]) - h.forms1[i])); scale = std::max(scale, std::abs(static_cast<double>(h.forms1[i]))); }
  std::printf("  1-form rows made from the stored parts vs the solve's: max |difference| %.1e (scale %.1e)\n", d1max, scale);
  CHECK(d1max < 1e-4 * scale);
  CHECK(!sub.find("operator", stem + "_forms2") && !sub.find("operator", stem + "_forms1"));
  for (const char* nm : {"cortex_intrinsic_d0", "cortex_intrinsic_d1", "cortex_intrinsic_star0", "cortex_intrinsic_star1", "cortex_intrinsic_star2"}) {
    tk::Want w{.cls = tk::Cls::operator_}; w.name = std::string(nm);
    const auto a = tk::Subject::of(sub).find(w);
    CHECK(a && a->row.value("layout", std::string()) == "analytic" && a->method == std::optional<std::string>("dec") && !a->path);
  }
  auto has_field = [&](const std::string& name) { tk::Want w{.cls = tk::Cls::field}; w.name = name; return tk::Subject::of(sub).find(w).has_value(); };
  CHECK(!has_field("cortex_intrinsic_star1"));
  CHECK(has_field("cortex_intrinsic_length"));
  CHECK(!has_field("hodgeDirac_kind"));
  {
    tk::Want lw{.cls = tk::Cls::field}; lw.name = std::string("hodgeDirac_eigenvalues");
    const auto ef = tk::Subject::of(sub).find(lw);
    CHECK(ef && ef->row.value("of_operator_id", std::string()) == text(e.eigenvectors, "id"));
    const auto tg = manifold_row_named(sub, "cortex_intrinsic"), sg = manifold_row_named(sub, "cortex");
    CHECK(tg && sg && text(*tg, "pp") == text(*sg, "pp"));
    const Json pj = Json::parse(opt_text(e.eigenvectors, "producer_json").value_or("{}"));
    CHECK(pj.contains("migrated_from") && pj["migrated_from"]["items"].contains("hodgeDirac_forms2"));
  }
  // the readers: the basis and the forms of the new form
  const HodgePackageData d = hodge_package_data(sub, stem + "_eigenvectors", std::nullopt);
  CHECK(d.basis.P.rows() == K && d.basis.P.cols() == V);
  const HodgePackageForms fm = hodge_package_forms(sub, d);
  CHECK(fm.nE == E && fm.nF == F && fm.forms2 == f2_before);

  // idempotent
  const Json again = migrate_eigensolve(sub, stem);
  CHECK(again["stems"][0]["status"] == "already in the new form");

  // THE GLUE of the earlier form, converted
  {
    std::vector<double> W = Vx; std::vector<std::int32_t> G = Fx;
    const std::int32_t n1 = static_cast<std::int32_t>(Vx.size() / 3);
    for (std::size_t i = 0; i < Vx.size(); i += 3) { W.push_back(Vx[i] + 0.2); W.push_back(Vx[i + 1]); W.push_back(Vx[i + 2]); }
    for (auto f : Fx) G.push_back(f + n1);
    std::vector<std::int32_t> dk(static_cast<std::size_t>(2 * n1), 1);
    std::vector<std::uint8_t> wall(dk.size(), 0);
    for (std::int32_t i = 0; i < 2 * n1; ++i) if ((i < n1 && W[3 * i] > 0.05) || (i >= n1 && W[3 * i] - 0.2 < -0.05)) { dk[static_cast<std::size_t>(i)] = 0; wall[static_cast<std::size_t>(i)] = 1; }
    AtlasMember bad = nxrtest::structures_of(W, G);
    bad.family = "Bad";
    for (std::int32_t i = 0; i < n1; ++i) if (W[3 * i + 2] > 0.06) bad.codes[static_cast<std::size_t>(i)] = 1;
    { SubjectWriter s = SubjectWriter::create(ds, "S2");
      SurfaceOptions so{.name = "cortex"};
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      so.atlases.push_back(nxrtest::structures_of(W, G));
      so.atlases.push_back(bad);
      s.compose([&] { add_surface(s, so, W, G); }); }
    SubjectWriter s2 = SubjectWriter::open(ds, "S2");
    const std::string orig = *tk::Subject::of(s2).manifold_named("cortex");
    std::vector<double> W32(W.size());
    for (std::size_t i = 0; i < W.size(); ++i) W32[i] = static_cast<double>(static_cast<float>(W[i]));
    const nxr::geometry::glue::Glued gm = nxr::geometry::glue::glueMesh(W32, G, wall);
    const std::int64_t Vg = static_cast<std::int64_t>(gm.keep.size());
    std::int64_t m = 0;
    for (const auto k : gm.keep) m += k >= 0;
    {
      const tk::Subject ts = tk::Subject::of(s2);
      tk::Unit u = ts.unit(tk::By{"earlier glue", Json()});
      tk::SurfaceOut so; so.name = "cortex_glued"; so.output = "glued"; so.derived_from_id = orig; so.faces = gm.faces;
      so.vertices.shape = {Vg, 3}; so.vertices.f64 = gm.positions; so.producer = Json{{"glued_from", "cortex"}, {"glue", Json{{"original", orig}}}};
      const std::string gid = u.surface(so);
      tk::FieldOut mf; mf.name = "cortex_glued_map"; mf.output = "map"; mf.kind = "glue map"; mf.manifold_id = gid; mf.value_type = "integer";
      mf.data.shape = {Vg}; mf.data.i32 = gm.keep; u.field(mf);
      tk::SelectionOut bs; bs.name = "cortex_glued_bridge"; bs.output = "bridge"; bs.type = "indices"; bs.manifold_id = gid;
      for (std::int64_t i = m; i < Vg; ++i) bs.codes.push_back(i);
      u.selection(bs);
      u.commit();
    }
    const std::string gid = *tk::Subject::of(s2).manifold_named("cortex_glued");
    const std::string earlier = refusal([&] { glue_map_of(s2, gid); });
    CHECK(earlier.find("migrate glue") != std::string::npos);                                     // the readers name the migration
    const std::string disagree = refusal([&] { migrate_glue(s2, std::string("cortex_glued"), "Bad", "Cortex L", "Cortex R"); });
    std::printf("  a hemisphere labelling that disagrees: %s\n", disagree.substr(0, 160).c_str());
    CHECK(disagree.find("Bad:") != std::string::npos);
    CHECK(s2.find("selection", "source/cortex_glued_bridge").has_value());
    const Json gr = migrate_glue(s2, std::nullopt, "Structures", "Cortex L", "Cortex R");
    std::printf("  glue migrated: %s\n", gr.dump().substr(0, 200).c_str());
    CHECK(gr["glued"][0]["status"] == "migrated");
    const auto gmap = glue_map_of(s2, gid);
    CHECK(gmap && gmap->original_of == gm.keep && gmap->original_id == orig);
    auto sel_named = [&](const std::string& name) { tk::Want w{.cls = tk::Cls::selection}; w.name = name; return tk::Subject::of(s2).find(w); };
    const auto hl = sel_named("cortex_glued_hemispheres");
    CHECK(hl.has_value() && !sel_named("cortex_glued_bridge") && !sel_named("cortex_glued_kept"));
    if (hl) {
      const StoredLabelling H = stored_labelling(s2, hl->id);
      const auto L = H.code_of("Cortex L"), R = H.code_of("Cortex R"), B = H.code_of("bridge");
      bool ok = L && R && B && static_cast<std::int64_t>(H.labels.size()) == Vg;
      for (std::int64_t i = 0; ok && i < Vg; ++i) ok = H.labels[static_cast<std::size_t>(i)] == (i >= m ? *B : gm.keep[static_cast<std::size_t>(i)] < n1 ? *L : *R);
      CHECK(ok);
    }
    CHECK(migrate_glue(s2, std::nullopt, "Structures", "Cortex L", "Cortex R")["glued"][0]["status"] == "already in the new form");
    // the form with the original's kept vertices `<glued>_kept` beside the labelling: the readers refuse it and name the migration, which checks the kept vertices are exactly those off the medial
    // wall and moves them aside (redundant); kept vertices that miss one vertex off the wall (a glue that cut more than the wall) are refused, nothing changed
    std::vector<std::int32_t> off_wall;
    for (std::size_t v = 0; v < wall.size(); ++v) if (!wall[v]) off_wall.push_back(static_cast<std::int32_t>(v));
    auto store_kept = [&](const std::vector<std::int32_t>& kept) {
      const tk::Subject ts = tk::Subject::of(s2);
      tk::Unit u = ts.unit(tk::By{"earlier glue", Json()});
      tk::SelectionOut ks; ks.name = "cortex_glued_kept"; ks.output = "kept"; ks.type = "indices"; ks.manifold_id = orig;
      for (const auto k : kept) ks.codes.push_back(k);
      u.selection(ks);
      u.commit();
    };
    store_kept(off_wall);
    CHECK(refusal([&] { glue_map_of(s2, gid); }).find("migrate glue") != std::string::npos);
    const Json kr = migrate_glue(s2, std::nullopt, "Structures", "Cortex L", "Cortex R");
    CHECK(kr["glued"][0]["status"] == "migrated" && !sel_named("cortex_glued_kept"));
    { const auto again_map = glue_map_of(s2, gid); CHECK(again_map && again_map->original_of == gm.keep); }
    std::vector<std::int32_t> widened = off_wall;
    widened.erase(widened.begin() + static_cast<long>(widened.size() / 2));
    store_kept(widened);
    const std::string more = refusal([&] { migrate_glue(s2, std::nullopt, "Structures", "Cortex L", "Cortex R"); });
    std::printf("  a stored glue that cut more than the wall: %s\n", more.substr(0, 200).c_str());
    CHECK(more.find("the stored glue removed more than the medial wall; re-run glue") != std::string::npos && sel_named("cortex_glued_kept").has_value());
  }
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_migrate_eigensolve");
}
