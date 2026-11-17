#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/cache.h"

#include <algorithm>
#include <cmath>
#include <array>
#include <chrono>
#include <cstring>
#include <numeric>

#include "nxr/compute.h"
#include "nxr/glue.h"
#include "nxr/io/hash.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif
#ifndef NXR_IO_VERSION
#define NXR_IO_VERSION "0.0.0"
#endif

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {

struct Names { std::string stem, axis, kind, op, support, glued, bridge, forms; };
Names package_names(const std::string& surface, bool primary) {
  const std::string stem = primary ? "hodgeDirac" : "hodgeDirac__" + surface;
  const std::string eigen = folder_for("points", "", std::nullopt, std::string("hodgeDirac"));
  return {stem, eigen + "/" + stem, eigen + "/" + stem + "_kind", stem + "_eigenvectors", stem + "_support", "source/" + surface + "_glued", "source/" + surface + "_glued_bridge", stem + "_forms2"};
}

template <class T>
std::vector<T> read_vector(const nxr::io::Store& store, const std::string& key) {
  ArrayReader r = ArrayReader::open(store, key);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}

}  // namespace

HodgeDirac solve_hodge_dirac(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, int modes, const SolveMethod& method) {
  const auto t0 = std::chrono::steady_clock::now();
  const int nV = static_cast<int>(vertices.size() / 3), nF = static_cast<int>(faces.size() / 3);
  // the Hodge–Dirac's triangulation: THE one Delaunay construction (mollify, the mixed-Voronoi mass of the mollified input recorded, flip to Delaunay) —
  // exactly as nxr and the Studio prepare a surface; `mollify` and `flip_eps` are the named options
  nxr::manifold::delaunay::DelaunayOptions dopt;
  dopt.mollify = method.mollify; dopt.flipEps = method.flip_eps;
  nxr::manifold::Manifold m(vertices.data(), nV, faces.data(), nF, /*intrinsicDelaunay=*/true, dopt);
  const nxr::manifold::ops::hodgedirac::HodgeDirac hd =
      nxr::manifold::ops::hodgedirac::assemble(m, method.star1_floor.value_or(0.0), method.mollify, method.flip_eps);
  const int nE = hd.nE;
  const std::int64_t clamped = hd.star1Clamped;

  // *1 = the cotan weights (nxr: none floored; Rheome: floor 1e-6; here the option `star1_floor`, unset = none), and its inverse
  const Eigen::VectorXd& s1 = hd.star1;
  const Eigen::VectorXd s1_inv = s1.cwiseInverse();

  // the exact ladder: (d0^T *1 d0, *0 = the input's mixed Voronoi mass)
  Eigen::SparseMatrix<double> M0(nV, nV);
  for (int i = 0; i < nV; ++i) M0.insert(i, i) = hd.star0(i);
  M0.makeCompressed();
  const Eigen::SparseMatrix<double> L0 = hd.d0.transpose() * s1.asDiagonal() * hd.d0;
  // the coexact ladder: (*2 d1 *1^-1 d1^T *2, *2)
  Eigen::SparseMatrix<double> star2(nF, nF);
  for (int f = 0; f < nF; ++f) star2.insert(f, f) = hd.star2(f);
  star2.makeCompressed();
  const Eigen::SparseMatrix<double> B2 = star2 * hd.d1;
  const Eigen::SparseMatrix<double> S2 = B2 * s1_inv.asDiagonal() * B2.transpose();

  // both ladders, the constants out, ONE lambda cut, merged by mu: nxr-compute's hodgeLadders (the reference rule)
  const nxr::manifold::solve::HodgeLadders lad = nxr::manifold::solve::hodgeLadders(L0, M0, S2, star2, modes, method.solve);
  const Eigen::MatrixXd& psi = lad.exact.eigenvectors;
  const Eigen::MatrixXd& chi = lad.coexact.eigenvectors;

  // W: <phi2, w_v> = the sum over the faces round v of *2 phi2, divided by their count (nxr.pipelines.hodge._W)
  std::vector<int> count(static_cast<std::size_t>(nV), 0);
  std::vector<std::array<int, 3>> fv(static_cast<std::size_t>(nF));
  for (int f = 0; f < nF; ++f)
    for (int k = 0; k < 3; ++k) {
      fv[static_cast<std::size_t>(f)][static_cast<std::size_t>(k)] = hd.faces(f, k);
      ++count[static_cast<std::size_t>(hd.faces(f, k))];
    }
  const double r2 = std::sqrt(2.0);                                  // divided by, as nxr does (not multiplied by 1/sqrt 2)
  const std::size_t K = lad.modes.size(), V = static_cast<std::size_t>(nV);
  HodgeDirac out;
  out.n_vertices = nV;
  out.gauge = Json{{"gauge", hd.record.gauge}, {"gauge_record", Json{{"mollify", hd.record.mollify}, {"epsilon", hd.record.epsilon}, {"flipEps", hd.record.flipEps}, {"mass", hd.record.mass},
                                                                      {"boundary_vertices", hd.record.boundaryVertices}, {"fingerprint", hd.record.fingerprint}}}};
  out.mu.resize(K);
  out.kind.resize(K);
  out.eigenvectors.assign(V * K, 0.0f);
  out.top = lad.top;
  out.star1_clamped = clamped;
  out.mollify_epsilon = hd.record.epsilon;
  out.flips = hd.record.flips;
  std::vector<double> acc(V);
  for (std::size_t j = 0; j < K; ++j) {
    const nxr::manifold::solve::HodgeLadders::Mode& md = lad.modes[j];
    out.mu[j] = md.mu;
    out.kind[j] = static_cast<std::uint8_t>(md.kind);
    if (md.kind == 0) {
      ++out.n_exact;
      for (std::size_t v = 0; v < V; ++v) out.eigenvectors[v * K + j] = static_cast<float>(psi(static_cast<Eigen::Index>(v), md.col) / r2);
    } else {
      ++out.n_coexact;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int f = 0; f < nF; ++f) {
        const double c = hd.star2(f) * chi(f, md.col);
        for (int k = 0; k < 3; ++k) acc[static_cast<std::size_t>(fv[static_cast<std::size_t>(f)][static_cast<std::size_t>(k)])] += c;
      }
      for (std::size_t v = 0; v < V; ++v) out.eigenvectors[v * K + j] = static_cast<float>(acc[v] / std::max(1, count[v]) / r2);
    }
  }
  // THE FULL FORMS: the intrinsic complex (the operator mesh the pencil was built on) and each coexact mode's 2-form χ/√2
  {
    const auto& cx = hd.complex;   // the intrinsic complex of geometry-central's triangulation, in the stored layout
    if (static_cast<int>(cx.faces.size()) != 3 * nF || static_cast<int>(cx.length.size()) != nE) throw nxr::io::Error(nxr::io::code::invalid_argument, "solve_hodge_dirac: the operator mesh and the pencil disagree in size");
    out.edges = cx.edges;
    out.lengths = cx.length;
    out.faces = cx.faces;
    out.face_edges = cx.faceEdges;
    // the stored orientation must be the pencil's: d0 (−1 tail, +1 tip) and d1 (the face's signed edges)
    {
      const Eigen::SparseMatrix<double, Eigen::RowMajor> D0(hd.d0), D1(hd.d1);
      for (int e = 0; e < nE; ++e) {
        double tail = 0, tip = 0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D0, e); it; ++it) {
          if (it.col() == out.edges[2 * static_cast<std::size_t>(e)]) tail = it.value();
          if (it.col() == out.edges[2 * static_cast<std::size_t>(e) + 1]) tip = it.value();
        }
        if (tail != -1.0 || tip != 1.0) throw nxr::io::Error(nxr::io::code::invalid_argument, "solve_hodge_dirac: d0's orientation is not the operator mesh's at edge " + std::to_string(e));
      }
      for (int f = 0; f < nF; ++f) {
        double sum = 0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(D1, f); it; ++it) {
          double want = 0;
          for (int k = 0; k < 3; ++k) { const int se = out.face_edges[3 * static_cast<std::size_t>(f) + static_cast<std::size_t>(k)]; if (std::abs(se) - 1 == it.col()) want += se > 0 ? 1.0 : -1.0; }
          sum += std::abs(it.value() - want);
        }
        if (sum > 0) throw nxr::io::Error(nxr::io::code::invalid_argument, "solve_hodge_dirac: d1's signs are not the operator mesh's at face " + std::to_string(f));
      }
    }
    out.star0.assign(hd.star0.data(), hd.star0.data() + hd.star0.size());
    out.star1.assign(hd.star1.data(), hd.star1.data() + hd.star1.size());
    out.star2.assign(hd.star2.data(), hd.star2.data() + hd.star2.size());
    out.angle = cx.angle; out.angle_sum = cx.angleSum; out.reference = cx.reference;
    const std::size_t F_ = static_cast<std::size_t>(nF);
    out.forms2.assign(F_ * K, 0.0f);
    for (std::size_t j = 0; j < K; ++j) {
      const nxr::manifold::solve::HodgeLadders::Mode& md = lad.modes[j];
      if (md.kind != 2) continue;
      for (std::size_t f = 0; f < F_; ++f) out.forms2[f * K + j] = static_cast<float>(chi(static_cast<Eigen::Index>(f), md.col) / r2);
    }
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

EigenSolveResult eigen_solve(DatasetWriter& ds, SubjectWriter& sub, const EigenSolveOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.modes < 2) throw Error(code::invalid_argument, "eigen_solve: modes must be at least 2");
  const SurfaceInput in = read_surface(sub, opt.surface);
  const std::int64_t V = in.n_vertices();
  const Names P = package_names(in.name, in.primary);
  const int pieces = connected_pieces(in.faces, V);
  if (pieces > 2)
    throw Error(code::unsupported, in.name + " is " + std::to_string(pieces) + " pieces: the Hodge–Dirac solve takes a surface of one closed piece, or two (a cortex's hemispheres, glued by anatomy)");
  // TWO PIECES (a cortex's hemispheres): glued BY ANATOMY, as nxr and the Studio solve a cortex (nxr.pipelines.hodge.solve; glue.ts glueMesh): each
  // medial wall (the Desikan-Killiany atlas' `unassigned`) cut out as a disk, a bridge grown between the rims; the solve is on the glued surface, whose
  // first vertices are the kept ones (ascending) and the rest the bridge's. ONE piece (a mesh, a sphere) is solved as it is.
  const bool glue = pieces == 2;
  std::vector<std::uint8_t> wall;
  nxr::glue::Glued G;
  if (glue) {
    wall = medial_wall(sub, in);
    G = nxr::glue::glueMesh(in.vertices, in.faces, wall);
  } else {
    G.positions = in.vertices;
    G.faces = in.faces;
    G.keep.resize(static_cast<std::size_t>(V));
    std::iota(G.keep.begin(), G.keep.end(), 0);
  }
  const std::int64_t Vg = static_cast<std::int64_t>(G.keep.size());          // the rows of the eigenvectors: the solved surface's vertices
  std::vector<std::int32_t> kept;
  for (auto k : G.keep) if (k >= 0) kept.push_back(k);
  const std::string mesh = mesh_key(in.faces), solved_mesh = mesh_key(G.faces);
  const Json glue_record = glue ? Json{{"by", "anatomy: the Desikan-Killiany medial wall cut, a bridge grown"},
                                       {"bridge", Json{{"vertices", G.bridge.vertices}, {"rings", G.bridge.rings}, {"area", G.bridge.area}, {"longestEdge", G.bridge.longestEdge}}},
                                       {"nV", Vg}, {"positions", "float32, as the Studio receives them"}, {"reference", "the Studio's glue.ts glueMesh (nxr-compute glue.h, bit-exact)"}}
                                : Json{{"by", "none: one closed piece"}, {"nV", Vg}};
  // THE INTRINSIC COMPLEX the modes are solved on (stored with the full forms): its manifold and the nodes on its cells
  const std::string iname = (glue ? in.name + "_glued" : in.name) + "_intrinsic", ipath = "source/" + iname;
  const std::vector<std::string> ifields = {iname + "_length", iname + "_star0", iname + "_star1", iname + "_star2", iname + "_angle_tail", iname + "_angle_tip",
                                            iname + "_angle_sum", iname + "_reference"};
  const std::vector<std::string> iops = {iname + "_d0", iname + "_d1"};
  if (glue) say(Json{{"stage", "glued"}, {"surface", in.name}, {"nV", Vg}, {"nF", static_cast<std::int64_t>(G.faces.size() / 3)}, {"mesh", solved_mesh}, {"bridge", glue_record["bridge"]}});

  CacheKey key;
  key.pipeline = "eigen_solve";
  const SolveMethod& sm = opt.method;
  const nxr::manifold::solve::HodgeSolveOptions& so = sm.solve;
  const Json method = Json{{"shift", so.shift == nxr::manifold::solve::HodgeShift::Absolute ? "absolute" : "relative"},
                           {"sigma", so.shift == nxr::manifold::solve::HodgeShift::Absolute && so.slice == 0 ? Json(so.sigma) : Json()},
                           {"slice", so.slice}, {"normalize", so.normalize},
                           {"star1_floor", sm.star1_floor ? Json(*sm.star1_floor) : Json()},
                           {"mollify", sm.mollify}, {"flip_eps", sm.flip_eps}};
  key.params = Json{{"modes", opt.modes}, {"method", method}, {"forms", "full: the intrinsic complex (d0, d1, stars, signposts) and each mode's 2-form"}};
  key.input = Json{{"mesh", mesh}, {"positions", positions_key(in.vertices)}};
  if (glue) {                                              // the wall that was cut belongs to the input: another atlas, another glued surface
    std::string bytes(wall.begin(), wall.end());
    for (auto& c : bytes) c = static_cast<char>(c + '0');
    key.input["medial_wall"] = sha256_hex(bytes).substr(0, 16);
    key.input["glue"] = "anatomy";
  }
  key.compute = NXR_COMPUTE_VERSION;

  std::string moved_aside;
  bool replaced = false;
  std::string operator_id;

  // HIT: the stored arrays
  auto load = [&](const Row& op) {
    HodgeDirac h;
    const auto axis = sub.find("manifold", P.axis);
    const auto kind = sub.find("field", P.kind);
    if (!axis || !kind) throw Error(code::no_such_row, in.name + ": the stored Hodge–Dirac package is incomplete (" + (axis ? P.kind : P.axis) + " is missing)");
    h.mu = read_vector<double>(sub.store(), P.axis);
    h.kind = read_vector<std::uint8_t>(sub.store(), P.kind);
    h.eigenvectors = read_vector<float>(sub.store(), text(op, "path"));
    h.n_vertices = Vg;
    for (auto k : h.kind) (k == 0 ? h.n_exact : h.n_coexact)++;
    if (static_cast<std::int64_t>(h.eigenvectors.size()) != Vg * h.n_modes()) throw Error(code::invalid_argument, in.name + ": the stored eigenvectors do not fit the surface");
    return h;
  };

  // MISS: solve, then (in one composition) store
  auto compute = [&]() {
    const std::pair<const char*, const std::string*> nodes[] = {{"operator", &P.op}, {"field", &P.kind}, {"selection", &P.support}, {"manifold", &P.axis}};
    const bool have = std::any_of(std::begin(nodes), std::end(nodes), [&](const auto& t) { return sub.find(t.first, *t.second).has_value(); });
    if (have && !opt.replace)
      throw Error(code::already_exists, sub.name() + ": " + in.name + " already holds a Hodge–Dirac package (" + P.axis + ") made by another job (other modes, mesh or nxr-compute); pass --replace");
    if (glue)
      for (const auto& [t, pth] : std::vector<std::pair<const char*, std::string>>{{"selection", P.bridge}, {"manifold", P.glued}})
        if (sub.find(t, pth) && !opt.replace)
          throw Error(code::already_exists, sub.name() + ": " + in.name + " already holds a glued surface (" + pth + ") made by another job; pass --replace");
    say(Json{{"stage", "cache"}, {"hit", false}, {"surface", in.name}, {"nV", V}, {"nF", in.n_faces()}, {"mesh", mesh}});
    HodgeDirac h = solve_hodge_dirac(G.positions, G.faces, opt.modes, sm);
    say(Json{{"stage", "solved"}, {"exact", h.n_exact}, {"coexact", h.n_coexact}, {"seconds", std::round(h.seconds * 10) / 10}});
    return h;
  };

  auto store = [&](const HodgeDirac& h) {
    const std::string stamp_utc = now_utc(false);
    const std::int64_t K = h.n_modes();
    // a replaced package's arrays are MOVED ASIDE, never deleted
    struct Have { const char* table; const std::string* path; };
    std::vector<std::pair<std::string, std::string>> order = {{"operator", P.forms}, {"operator", P.op}, {"field", P.kind}, {"selection", P.support}, {"manifold", P.axis}};
    if (const auto im = sub.find("manifold", ipath)) {                 // the intrinsic complex: its fields and operators, then itself
      for (const auto& n : ifields) order.push_back({"field", node_path(sub.db(), text(*im, "id"), n)});
      for (const auto& n : iops) order.push_back({"operator", node_path(sub.db(), text(*im, "id"), n)});
      order.push_back({"manifold", ipath});
    }
    // the glued surface (and its bridge) is KEPT when the new glue is the same surface (the same vertices, the same faces): nodes other runs stored on it stay
    std::optional<std::string> kept_glued;
    if (glue)
      if (const auto gm = sub.find("manifold", P.glued)) {
        const auto pos = read_vector<double>(sub.store(), P.glued + "/vertices");
        const auto fac = read_vector<std::int32_t>(sub.store(), P.glued + "/faces");
        if (fac == G.faces && pos.size() == G.positions.size() && std::equal(pos.begin(), pos.end(), G.positions.begin()) && sub.find("selection", P.bridge))
          kept_glued = text(*gm, "id");
      }
    if (!kept_glued) {
      order.push_back({"selection", P.bridge});
      order.push_back({"manifold", P.glued});
    }
    for (const auto& [table, path] : order) {
      struct { const char* table; const std::string* path; } t{table.c_str(), &path};
      const auto r = sub.find(t.table, *t.path);
      if (!r) continue;
      replaced = true;
      const std::string loc = std::string(t.table) == "selection" ? text(*r, "array_path") : text(*r, "path");
      if (!loc.empty() && sub.store().exists_prefix(loc)) {
        const AsideFolder aside = replaced_aside(ds, sub, stamp_utc.substr(0, 10), 1);
        sub.datastore().rename_prefix(nxr::io::join_key(sub.prefix(), loc), nxr::io::join_key(aside.base, loc));
        moved_aside = aside.on_disk;
      }
      sub.remove(t.table, text(*r, "id"));
    }
    const Json prov = [&] {
      Json p = Json::object();
      p["producer"] = std::string("nxr-io ") + NXR_IO_VERSION + " eigen solve (nxr-compute " + NXR_COMPUTE_VERSION + ")";
      p["mesh"] = solved_mesh;                               // the solved surface's (the glued one for a cortex), as nxr records it
      p["subject"] = sub.name();
      p["dataset"] = nullptr;
      p["job"] = nullptr;
      p["rheome_commit"] = nullptr;
      p["source"] = nullptr;
      p["glue"] = glue_record;
      const bool ref = so.shift == nxr::manifold::solve::HodgeShift::Absolute && so.slice == 0 && !so.normalize;
      p["solve"] = Json{{"route", "split: Laplace–Beltrami (d₀ᵀ⋆₁d₀, ⋆₀) and 2-form (⋆₂d₁⋆₁⁻¹d₁ᵀ⋆₂, ⋆₂) ladders, shift-invert eigensolve, one λ cut"},
                        {"reference", ref ? "nxr solve_hodge / Rheome hodge_dirac: one solve per ladder, absolute sigma, no re-orthonormalisation" : "not the reference method (an option differs)"},
                        {"requested", opt.modes}, {"exact", h.n_exact}, {"coexact", h.n_coexact}, {"seconds", std::round(h.seconds * 10) / 10},
                        {"options", method}, {"lambda_top", h.top}, {"star1_clamped", h.star1_clamped},
                        {"delaunay", Json{{"construction", "nxr-compute delaunay.h: the Studio's intrinsicDelaunay.ts (mollify, mixed-Voronoi mass of the mollified input, flip to Delaunay)"},
                                          {"mollify", sm.mollify}, {"flip_eps", sm.flip_eps}, {"epsilon", h.mollify_epsilon}, {"flips", h.flips}}},
                        {"tolerance", so.slice == 0 && so.shift == nxr::manifold::solve::HodgeShift::Absolute ? Json(so.tol) : Json()}};
      p["checks"] = nullptr;
      p["mixing_max"] = 0.0;
      p["cache"] = key.json();
      return p;
    }();
    const std::string cs = "eigenvalue:" + P.stem;
    const std::string axis_id = uuid7();

    ManifoldSpec ms;
    ms.id = axis_id;
    ms.name = P.stem;
    ms.path = P.axis;
    ms.type = "points";
    ms.n_vertices = K;
    ms.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec g;
    g.form = "stored";
    g.cs_key = cs;
    g.positions_path = P.axis;
    ms.geometry = g;
    ms.created_utc = stamp_utc;
    ms.cols.set("derived_from_id", in.id).set("operator", "hodgeDirac");
    {
      Json pj = Json{{"variant", "Hodge-Dirac (d + δ, glued)"}, {"fieldType", "potentials"}, {"kept", "μ > 0"}};
      for (auto it = prov.begin(); it != prov.end(); ++it) pj[it.key()] = it.value();
      pj.erase("cache");                                   // the cache stamp is carried by the main product, the operator
      ms.cols.json("producer_json", pj);
    }
    ms.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array(P.axis, view(h.mu, {K}), {}, std::nullopt, "manifold", axis_id); };
    sub.manifold(std::move(ms));

    const std::vector<std::int32_t>& support = kept;           // the input's vertices the solved surface keeps (all of them for one piece)
    SelectionSpec ss;
    ss.name = P.stem + "_support";
    ss.path = P.support;
    ss.type = "indices";
    ss.manifold_id = in.id;
    ss.data = view(support, {static_cast<std::int64_t>(support.size())});
    ss.picked = true;
    ss.cols.set("description", "the " + in.name + " vertices the glued Hodge–Dirac surface keeps (the medial cut is outside)");
    const std::string sel_id = sub.selection(std::move(ss));

    // THE GLUED SURFACE (a cortex): the kept vertices (in support order), then the bridge's; and the bridge as a selection — stored as nxr's
    // store_package stores one, so nxr and `eigen solve` give the same layout
    std::string target = kept_glued ? *kept_glued : in.id;
    std::vector<std::int32_t> bridge;
    if (glue && !kept_glued) {
      const std::int64_t gf = static_cast<std::int64_t>(G.faces.size() / 3);
      std::optional<std::string> csk;
      if (!is_null(in.manifold, "geometry_id"))
        if (auto c = sub.db().one("SELECT c.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id WHERE g.id = ?", {nxr::io::V(text(in.manifold, "geometry_id"))})) csk = text(*c, "name");
      ManifoldSpec gm;
      gm.name = in.name + "_glued";
      gm.path = P.glued;
      gm.type = "surface";
      gm.n_vertices = Vg;
      gm.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, P.glued + "/faces"}}};
      GeometrySpec gg;
      gg.form = "stored";
      gg.cs_key = csk;
      gg.positions_path = P.glued + "/vertices";
      gm.geometry = gg;
      gm.created_utc = stamp_utc;
      gm.cols.set("n_faces", gf).set("derived_from_id", in.id)
          .set("comment", in.name + " glued by anatomy: its medial walls cut (the Desikan–Killiany atlas), a bridge grown between the rims — vertex i < " +
                              std::to_string(kept.size()) + " is " + P.stem + "_support[i], the rest the bridge")
          .json("producer_json", Json{{"glued_from", in.name}, {"mesh", solved_mesh}, {"glue", glue_record}});
      gm.populate = [&](nxr::io::Store&, const std::string&) {
        sub.write_array(P.glued + "/vertices", view(G.positions, {Vg, 3}));
        sub.write_array(P.glued + "/faces", view(G.faces, {gf, 3}));
      };
      target = sub.manifold(std::move(gm));
      for (std::int64_t i = static_cast<std::int64_t>(kept.size()); i < Vg; ++i) bridge.push_back(static_cast<std::int32_t>(i));
      SelectionSpec bs;
      bs.name = in.name + "_glued_bridge";
      bs.path = P.bridge;
      bs.type = "indices";
      bs.manifold_id = target;
      bs.data = view(bridge, {static_cast<std::int64_t>(bridge.size())});
      bs.picked = true;
      bs.cols.set("description", "the medial BRIDGE (not cortex): to mask it — no sources, a forward zero there");
      sub.selection(std::move(bs));
    }

    FieldSpec fs_;
    fs_.name = P.stem + "_kind";
    fs_.path = P.kind;
    fs_.kind = "form degree";
    fs_.manifold_id = axis_id;
    fs_.value_type = "integer";
    fs_.data = view(h.kind, {K});
    fs_.cols.set("description", "per mode: 0 exact (a 0-form potential A) · 2 coexact (a 2-form potential S)");
    sub.field(std::move(fs_));

    OperatorSpec os;
    os.name = P.op;
    os.path = P.op;
    os.kind = "eigenvectors";
    os.from_manifold_id = axis_id;
    os.layout = "dense";
    os.n_rows = Vg;
    os.n_cols = K;
    ArrayOptions ao;
    ao.chunks = Shape{Vg, 1};
    ao.shards = Shape{Vg, std::min<std::int64_t>(K, 64)};
    ao.compress = false;                                   // uncompressed: the Studio's ranged reader merges a block into a few requests
    os.meta = array_meta(DType::Float32, {Vg, K}, ao);
    os.populate = [&](nxr::io::Store& store, const std::string& at) { populate_array(store, at, view(h.eigenvectors, {Vg, K})); };
    os.created_utc = stamp_utc;
    os.cols.set("to_manifold_id", target).set("from_value_type", "scalar").set("to_value_type", "scalar");
    os.cols.set("description", "Hodge–Dirac modes as vertex potentials: A (exact) or S (coexact) per column; 1-form (∇A + n×∇S)/μ");
    os.cols.json("params_json", Json{{"potentials", "A = ψ/√2 (kind 0) · S = ⟨χ/√2, ω_v⟩ (kind 2)"}, {"support", sel_id}, {"solved", "K x = μ M x, d + δ on the glued intrinsic-Delaunay complex"}});
    os.cols.json("producer_json", prov);
    operator_id = sub.operator_(std::move(os));

    // THE FULL FORMS. The intrinsic complex the pencil was built on, as a manifold of its own (the same vertices; its edges and faces as
    // its cells: two edges may join one pair of vertices, so the incidence is read from d0 and d1, never from vertex pairs), d0 and d1 as sparse operators
    // between its cells, the edge lengths, the three stars and the signposts as fields on its cells; and each mode's 2-form χ/√2 on its faces. With the
    // stored 0-form part ψ/√2 (the eigenvectors above) both 1-form parts follow exactly: dψ/(μ√2), ⋆1⁻¹ d1ᵀ ⋆2 χ/(μ√2).
    const std::int64_t nEi = static_cast<std::int64_t>(h.lengths.size()), nFi = static_cast<std::int64_t>(h.faces.size() / 3);
    {
      std::optional<std::string> csk2;
      if (!is_null(in.manifold, "geometry_id"))
        if (auto c = sub.db().one("SELECT c.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id WHERE g.id = ?", {nxr::io::V(text(in.manifold, "geometry_id"))})) csk2 = text(*c, "name");
      Json rec = Json{{"construction", "geometry-central's signpost intrinsic triangulation (flipToDelaunay)"}, {"mollify", sm.mollify}, {"flip_eps", sm.flip_eps},
                            {"epsilon", h.mollify_epsilon}, {"flips", h.flips}, {"star1_floor", sm.star1_floor ? Json(*sm.star1_floor) : Json()}, {"star1_clamped", h.star1_clamped}};
      if (h.gauge.is_object()) { rec["gauge"] = h.gauge["gauge"]; rec["gauge_surface"] = target; rec["gauge_record"] = h.gauge["gauge_record"]; }   // per intrinsic edge or face: the gauge (G6)
      ManifoldSpec im;
      im.name = iname;
      im.path = ipath;
      im.type = "surface";
      im.n_vertices = Vg;
      im.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{1, ipath + "/edges"}, {2, ipath + "/faces"}}};
      GeometrySpec ig;
      ig.form = "stored";
      ig.cs_key = csk2;
      ig.positions_path = ipath + "/vertices";
      im.geometry = ig;
      im.created_utc = stamp_utc;
      im.cols.set("n_edges", nEi).set("n_faces", nFi).set("derived_from_id", target)
          .set("comment", "the intrinsic (mollified) Delaunay triangulation of " + (glue ? in.name + "_glued" : in.name) + " the Hodge–Dirac modes " + P.stem +
                              " are solved on: the same vertices, its own edges (lengths along the surface) and faces; read its incidence from " + iops[0] + " and " + iops[1])
          .json("producer_json", Json{{"intrinsic_of", glue ? in.name + "_glued" : in.name}, {"mesh", solved_mesh}, {"delaunay", rec}});
      im.populate = [&](nxr::io::Store&, const std::string&) {
        sub.write_array(ipath + "/vertices", view(G.positions, {Vg, 3}));
        sub.write_array(ipath + "/edges", view(h.edges, {nEi, 2}));
        sub.write_array(ipath + "/faces", view(h.faces, {nFi, 3}));
      };
      const std::string iid = sub.manifold(std::move(im));
      const Json fpj = Json{{"producer", prov["producer"]}, {"package", P.stem}, {"delaunay", rec}, {"cache", key.json()}};
      // d0 [nE x nV]: −1 at an edge's tail, +1 at its tip; d1 [nF x nE]: each face's edges with their signs
      std::vector<std::int64_t> ip0(static_cast<std::size_t>(nEi + 1)), ip1(static_cast<std::size_t>(nFi + 1));
      std::vector<std::int32_t> ix0(static_cast<std::size_t>(2 * nEi)), ix1(static_cast<std::size_t>(3 * nFi));
      std::vector<float> dv0(static_cast<std::size_t>(2 * nEi)), dv1(static_cast<std::size_t>(3 * nFi));
      for (std::int64_t e = 0; e <= nEi; ++e) ip0[static_cast<std::size_t>(e)] = 2 * e;
      for (std::int64_t e = 0; e < nEi; ++e) {
        const std::int32_t a = h.edges[static_cast<std::size_t>(2 * e)], b = h.edges[static_cast<std::size_t>(2 * e + 1)];
        const bool order_ab = a < b;                         // CSR: columns ascending (a self-edge is refused by the construction)
        ix0[static_cast<std::size_t>(2 * e)] = order_ab ? a : b; dv0[static_cast<std::size_t>(2 * e)] = order_ab ? -1.0f : 1.0f;
        ix0[static_cast<std::size_t>(2 * e + 1)] = order_ab ? b : a; dv0[static_cast<std::size_t>(2 * e + 1)] = order_ab ? 1.0f : -1.0f;
      }
      for (std::int64_t f = 0; f <= nFi; ++f) ip1[static_cast<std::size_t>(f)] = 3 * f;
      for (std::int64_t f = 0; f < nFi; ++f) {
        std::array<std::pair<std::int32_t, float>, 3> row;
        for (int k = 0; k < 3; ++k) { const std::int32_t se = h.face_edges[static_cast<std::size_t>(3 * f + k)]; row[static_cast<std::size_t>(k)] = {std::abs(se) - 1, se > 0 ? 1.0f : -1.0f}; }
        std::sort(row.begin(), row.end());
        for (int k = 0; k < 3; ++k) { ix1[static_cast<std::size_t>(3 * f + k)] = row[static_cast<std::size_t>(k)].first; dv1[static_cast<std::size_t>(3 * f + k)] = row[static_cast<std::size_t>(k)].second; }
      }
      auto put_op = [&](const std::string& name, const char* from_cell, const char* to_cell, std::int64_t rows, std::int64_t cols, std::vector<std::int64_t>& ip,
                        std::vector<std::int32_t>& ix, std::vector<float>& dv, const std::string& what) {
        OperatorSpec od;
        od.name = name;
        od.path = node_path(sub.db(), iid, name);
        od.kind = "exterior derivative";
        od.from_manifold_id = iid;
        od.layout = "csr";
        od.n_rows = rows;
        od.n_cols = cols;
        od.sparse = SparseArrays{{{"indptr", view(ip, {rows + 1})}, {"indices", view(ix, {static_cast<std::int64_t>(ix.size())})}, {"data", view(dv, {static_cast<std::int64_t>(dv.size())})}}};
        od.created_utc = stamp_utc;
        od.cols.set("nnz", static_cast<std::int64_t>(dv.size())).set("data_type", "float32");
        od.cols.set("to_manifold_id", iid).set("from_cell", from_cell).set("to_cell", to_cell).set("from_value_type", "scalar").set("to_value_type", "scalar")
            .set("description", what).json("producer_json", fpj);
        sub.operator_(std::move(od));
      };
      put_op(iops[0], "0", "1", nEi, Vg, ip0, ix0, dv0, "d0 of the intrinsic complex: vertices → edges (−1 at an edge's tail, +1 at its tip)");
      put_op(iops[1], "1", "2", nFi, nEi, ip1, ix1, dv1, "d1 of the intrinsic complex: edges → faces (each face's three edges, +1 along the face's winding)");
      // the signposts per edge: the angle of the edge at its tail and at its tip (its twin's angle), each in its vertex's tangent plane
      std::vector<double> atail(static_cast<std::size_t>(nEi), NAN), atip(static_cast<std::size_t>(nEi), NAN);
      if (!h.angle.empty())
        for (std::int64_t hh = 0; hh < 3 * nFi; ++hh) {
          const std::int32_t se = h.face_edges[static_cast<std::size_t>(hh)];
          (se > 0 ? atail : atip)[static_cast<std::size_t>(std::abs(se) - 1)] = h.angle[static_cast<std::size_t>(hh)];
        }
      auto put_field = [&](const std::string& name, const char* cell, const char* vt, const ArrayView& data, const std::optional<std::string>& unit, const std::string& what) {
        FieldSpec f;
        f.name = name;
        f.path = node_path(sub.db(), iid, name);
        f.kind = "intrinsic complex";
        f.manifold_id = iid;
        f.value_type = vt;
        f.data = data;
        f.created_utc = stamp_utc;
        f.cols.set("cell", cell).set("description", what);
        if (unit) f.cols.set("unit", *unit);
        f.cols.json("producer_json", fpj);
        sub.field(std::move(f));
      };
      std::vector<std::int32_t> ref = h.reference.empty() ? std::vector<std::int32_t>(static_cast<std::size_t>(Vg), -1) : h.reference;
      std::vector<double> asum = h.angle_sum.empty() ? std::vector<double>(static_cast<std::size_t>(Vg), NAN) : h.angle_sum;
      put_field(ifields[0], "1", "scalar", view(h.lengths, {nEi}), std::nullopt, "each intrinsic edge's length along the surface (the input's units)");
      put_field(ifields[1], "0", "scalar", view(h.star0, {Vg}), std::nullopt, "⋆0: each vertex's mixed-Voronoi mass of the (mollified) input, as the pencil used it");
      put_field(ifields[2], "1", "scalar", view(h.star1, {nEi}), std::nullopt, "⋆1: each intrinsic edge's cotangent weight (≥ 0: Delaunay), as the pencil used it");
      put_field(ifields[3], "2", "scalar", view(h.star2, {nFi}), std::nullopt, "⋆2: 1 / each intrinsic face's area");
      put_field(ifields[4], "1", "scalar", view(atail, {nEi}), std::string("rad"), "the signpost: each edge's angle at its tail, in the tail's tangent plane (from its reference neighbour)");
      put_field(ifields[5], "1", "scalar", view(atip, {nEi}), std::string("rad"), "the signpost: each edge's angle at its tip, in the tip's tangent plane");
      put_field(ifields[6], "0", "scalar", view(asum, {Vg}), std::string("rad"), "each vertex's cone angle (the sum of its corners)");
      put_field(ifields[7], "0", "integer", view(ref, {Vg}), std::nullopt, "each vertex's reference neighbour: the head of its angle-0 edge (an input edge)");
      // each mode's 2-form χ/√2 on the intrinsic faces: eigen axis → faces
      OperatorSpec of;
      of.name = P.forms;
      of.path = P.forms;
      of.kind = "eigenvectors";
      of.from_manifold_id = axis_id;
      of.layout = "dense";
      of.n_rows = nFi;
      of.n_cols = K;
      ArrayOptions fo;
      fo.chunks = Shape{nFi, 1};
      fo.shards = Shape{nFi, std::min<std::int64_t>(K, 64)};
      fo.compress = false;
      of.meta = array_meta(DType::Float32, {nFi, K}, fo);
      of.populate = [&](nxr::io::Store& store, const std::string& at) { populate_array(store, at, view(h.forms2, {nFi, K})); };
      of.created_utc = stamp_utc;
      of.cols.set("to_manifold_id", iid).set("to_cell", "2").set("from_value_type", "scalar").set("to_value_type", "scalar")
          .set("description", "Hodge–Dirac modes' 2-form parts χ/√2 on the intrinsic faces (0 on an exact mode); with " + P.op + " (the 0-form parts) the full forms");
      of.cols.json("params_json", Json{{"forms", "2-form part χ/√2 of each Dirac eigenvector (0, δχ/μ, χ)/√2; ⋆2-normalised: Σ ⋆2 (χ/√2)² = 1/2"}, {"complex", iname}});
      of.cols.json("producer_json", fpj);
      sub.operator_(std::move(of));
    }
  };

  const Outcome<HodgeDirac> o = cached<HodgeDirac>(sub, "operator", P.op, key, load, compute, store);
  EigenSolveResult r;
  r.hit = o.hit;
  r.seconds = o.seconds;
  r.modes = o.value;
  const HodgeDirac& h = r.modes;
  if (o.hit) say(Json{{"stage", "cache"}, {"hit", true}, {"surface", in.name}, {"nV", V}, {"nF", in.n_faces()}, {"mesh", mesh}});
  r.info = Json{{"axis", P.axis}, {"surface", in.name}, {"glued", glue}, {"rows", Vg}, {"moved_aside", moved_aside.empty() ? Json() : Json(moved_aside)}, {"n_modes", h.n_modes()},
                {"n_exact", h.n_exact}, {"n_coexact", h.n_coexact}, {"mu", Json::array({h.mu.front(), h.mu.back()})}, {"support", static_cast<std::int64_t>(kept.size())}, {"of", V}, {"position_err_m", 0.0},
                {"replaced", replaced}, {"operator_id", o.hit ? o.node_id : operator_id}, {"mesh", solved_mesh}, {"bridge", glue ? glue_record["bridge"] : Json()}, {"cache", o.hit ? "hit" : "miss"}};
  return r;
}

}  // namespace nxr::pipelines
