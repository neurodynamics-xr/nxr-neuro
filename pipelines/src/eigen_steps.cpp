// nxr-pipelines — THE HODGE-DIRAC EIGENMODES AS A REGISTERED STEP (the hand-wired `eigen_solve` it replaces): a surface's modes, solved on its intrinsic Delaunay complex, with the full forms.
//
//   hodge_dirac   setup   surface (a stored surface of one closed piece; a cortex of two hemispheres is the glued surface `surface.glue` made) -> the modes and everything that lies on them (below)
//   surface.glue  setup   a cortex of two hemispheres -> its glued surface, the map glued vertex -> original (-1 on the bridge) and the bridge (rule 28, rule 33)
//
// THE GRAPH (written first, docs/specs/2026-11-17-eigenmodes-as-definition.md; the Studio's chain: glue.ts glueMesh -> intrinsicDelaunay.ts -> the Laplace-Beltrami and 2-form ladders -> the package):
//   surface ──[intrinsic Delaunay: mollify, flip]──> the cell complex (vertices, edges, faces) with d0, d1, the three stars, the signposts
//           ──[split solve: LB ladder (exact modes) + 2-form ladder (coexact modes), one lambda cut, the constants out]──> mu, kind, the 0-form parts psi/sqrt2 (or <chi/sqrt2, w_v>), the 2-form parts chi/sqrt2
// The step reads no file and no row: the surface is read by the runner (Bound) and handed over; everything it makes is DESCRIBED (meta.describe) and
// stored by io from the definition's declared outputs (eigenmodes.yaml): a mode axis (points, the eigenvalues mu as coordinates: the operator's by-product, `of_operator`), its kind field, the
// eigenvectors and the 2-form parts as operators from that axis, the intrinsic complex (a cell complex manifold) with d0 and d1 as sparse operators and the stars and signposts as fields on
// its cells ('0' vertices, '1' edges, '2' faces).
//
// The method rule: the default is the REFERENCE method of nxr and Rheome (see eigen.h: SolveMethod); the other methods are named options, every option is in the run's provenance and cache key.
// What it cannot resolve: the modes are the discrete operator's on the intrinsic triangulation (mollified, flipped to Delaunay): a mode above the mesh's resolution is not a property of the cortex;
// a surface of two or more pieces is refused (a cortex of two hemispheres is glued by `surface.glue` first, and the glued surface named as the input): the bridge is not cortex.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "nxr/compute.h"
#include "nxr/glue.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {

namespace hs = nxr::manifold::solve;

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

std::string stem_of(const std::string& surface, bool primary) { return primary ? "hodgeDirac" : "hodgeDirac__" + surface; }

// the wall's identity for the cache key (FNV-1a over its 0/1 bytes): another atlas, another glued surface
std::string wall_key(const std::vector<std::uint8_t>& wall) {
  std::uint64_t h = 1469598103934665603ULL;
  for (const std::uint8_t b : wall) { h ^= static_cast<std::uint64_t>(b); h *= 1099511628211ULL; }
  char buf[24];
  std::snprintf(buf, sizeof buf, "%zu:%016llx", wall.size(), static_cast<unsigned long long>(h));
  return buf;
}

Array made(std::int64_t rows, std::int64_t cols, std::vector<double> data, Json describe) {
  Array a;
  a.rows = rows; a.cols = cols;
  a.data = std::move(data);
  a.meta["describe"] = std::move(describe);
  return a;
}
Array absent() {                                                       // an optional output this run does not make
  Array a;
  a.meta["absent"] = true;
  return a;
}
template <class T>
std::vector<double> widened(const std::vector<T>& v) { return std::vector<double>(v.begin(), v.end()); }

struct HodgeState {
  std::string surface;
  int modes = 3000;
  SolveMethod method;
};

Json method_json(const SolveMethod& sm) {
  const hs::HodgeSolveOptions& so = sm.solve;
  return Json{{"shift", so.shift == hs::HodgeShift::Absolute ? "absolute" : "relative"},
              {"sigma", so.shift == hs::HodgeShift::Absolute && so.slice == 0 ? Json(so.sigma) : Json()},
              {"slice", so.slice}, {"normalize", so.normalize},
              {"star1_floor", sm.star1_floor ? Json(*sm.star1_floor) : Json()},
              {"mollify", sm.mollify}, {"flip_eps", sm.flip_eps}};
}

SolveMethod method_of(const Json& p) {
  SolveMethod sm;
  const std::string shift = p.value("shift", std::string("absolute"));
  need(shift == "absolute" || shift == "relative", "hodge_dirac: shift is absolute or relative, not " + shift);
  sm.solve.shift = shift == "absolute" ? hs::HodgeShift::Absolute : hs::HodgeShift::RelativeLadder;
  if (p.contains("sigma") && p["sigma"].is_number()) sm.solve.sigma = p["sigma"].get<double>();
  if (p.contains("slice") && p["slice"].is_number_integer()) sm.solve.slice = p["slice"].get<int>();
  sm.solve.normalize = p.value("normalize", false);
  if (p.contains("star1_floor") && p["star1_floor"].is_number() && p["star1_floor"].get<double>() > 0) sm.star1_floor = p["star1_floor"].get<double>();
  sm.mollify = p.value("mollify", 1e-6);
  sm.flip_eps = p.value("flip_eps", 1e-6);
  return sm;
}

const char* const kOutputs[] = {"modes", "kind", "eigenvalues", "support", "eigenvectors", "forms2", "complex", "d0", "d1",
                                "length", "star0", "star1", "star2", "angle_tail", "angle_tip", "angle_sum", "reference"};

StepSpec hodge_dirac_step() {
  StepSpec s;
  s.name = "hodge_dirac";
  s.kind = "setup";
  s.inputs = {"surface"};
  for (const char* o : kOutputs) s.outputs.push_back(o);
  s.parameters = Json{{"modes", 3000}, {"shift", "absolute"}, {"sigma", nullptr}, {"slice", 0}, {"normalize", false}, {"star1_floor", nullptr}, {"mollify", 1e-6}, {"flip_eps", 1e-6}};
  s.matches = "nxr.eigen.hodge_dirac.solve_hodge and nxr.pipelines.hodge.solved_package; Rheome rheome.eigen.hodge_dirac; the Studio's glue.ts glueMesh and intrinsicDelaunay.ts";
  s.description =
      "The Hodge-Dirac eigenmodes of a surface: the Laplace-Beltrami ladder (exact modes) and the 2-form ladder (coexact modes) of the surface's intrinsic Delaunay complex, cut at one lambda, the constants out. "
      "The surface is ONE closed piece: a cortex of two hemispheres is solved on its glued surface (made by `surface.glue`, named as the input). Makes the mode axis (the eigenvalues mu, ascending), each mode's kind (0 exact, 2 coexact), "
      "the eigenvectors (per mode one vertex potential) and the 2-form parts (the faces of the complex), the complex itself with d0 and d1 and the stars and signposts on its cells.";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface: positions as the Studio receives them (float32 widened), double");

  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("surface") != 0 && ins.at("surface").row, "hodge_dirac: the input `surface` is missing");
    need(p.value("modes", 3000) >= 2, "hodge_dirac: modes must be at least 2");
    method_of(p);                                                     // the options refused here, before anything is read
    const auto& row = *ins.at("surface").row;
    const std::string name = nxr::io::text(row, "name"), stem = stem_of(name, nxr::io::integer(row, "is_primary") != 0);
    const std::string solved = name;
    const std::string sid = ins.at("surface").manifolds.empty() ? std::string() : ins.at("surface").manifolds[0];
    Values out;
    for (const char* o : kOutputs) {
      Value v;
      const std::string port = o;
      v.cls = (port == "modes" || port == "complex") ? "manifold"
              : port == "support" ? "selection"
              : (port == "eigenvectors" || port == "forms2" || port == "d0" || port == "d1") ? "operator" : "field";
      if (v.cls == "operator") v.manifolds = {sid, sid};                  // the places themselves are declared (from, to)
      else v.manifolds = {sid};
      if (port == "kind" || port == "reference") v.value_type = "integer";
      v.facts["names"] = Json{{"stem", stem}, {"solved", solved}};
      out[port] = std::move(v);
    }
    return out;
  };

  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    HodgeState st;
    st.surface = nxr::io::text(*ins.at("surface").row, "name");
    st.modes = p.value("modes", 3000);
    st.method = method_of(p);
    return std::any(std::move(st));
  };

  // what the request resolves to on what is stored: the surface's mesh and positions (nxr's mesh key) and, glued, its medial wall: they belong to the cache key
  s.resolve = [](const Values& ins, const Json& p, const Bound& bound) {
    const SurfaceInput S = bound.surface(nxr::io::text(*ins.at("surface").row, "name"));
    Json r = Json{{"mesh", mesh_key(S.faces)}, {"positions", positions_key(S.vertices)}, {"forms", "full: the intrinsic complex (d0, d1, stars, signposts) and each mode's 2-form"}};
    return r;
  };

  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const HodgeState& st = std::any_cast<const HodgeState&>(state);
    auto say = [&](const Json& j) { if (ctx.say) ctx.say(j); };
    const SurfaceInput S = ctx.bound.surface(st.surface);               // float32 widened, as the Studio's GPU-side pipelines receive them
    const std::int64_t V = S.n_vertices();
    const int pieces = connected_pieces(S.faces, V);
    need(pieces == 1, st.surface + " is " + std::to_string(pieces) + " pieces: the Hodge-Dirac solve takes a surface of one closed piece (a cortex's two hemispheres are glued first by `run glue`, and the glued surface named as the input)");
    nxr::glue::Glued G;
    G.positions = S.vertices;
    G.faces = S.faces;
    G.keep.resize(static_cast<std::size_t>(V));
    std::iota(G.keep.begin(), G.keep.end(), 0);
    const std::int64_t Vg = static_cast<std::int64_t>(G.keep.size());
    std::vector<std::int32_t> kept;
    for (const auto k : G.keep) if (k >= 0) kept.push_back(k);
    const std::string cs = ctx.bound.coordinate_system_key(S.id);
    const std::string stem = stem_of(S.name, S.primary), solved = S.name, iname = solved + "_intrinsic";
    const std::string solved_mesh = mesh_key(G.faces);
    const Json glue_record = Json{{"by", "none: one closed piece"}, {"nV", Vg}};

    say(Json{{"stage", "solve"}, {"surface", S.name}, {"nV", V}, {"nF", S.n_faces()}, {"mesh", mesh_key(S.faces)}});
    const HodgeDirac h = solve_hodge_dirac(G.positions, G.faces, st.modes, st.method);
    say(Json{{"stage", "solved"}, {"exact", h.n_exact}, {"coexact", h.n_coexact}, {"seconds", std::round(h.seconds * 10) / 10}});

    const std::int64_t K = h.n_modes(), nE = static_cast<std::int64_t>(h.lengths.size()), nF = static_cast<std::int64_t>(h.faces.size() / 3);
    const hs::HodgeSolveOptions& so = st.method.solve;
    const Json method = method_json(st.method);
    const bool ref = so.shift == hs::HodgeShift::Absolute && so.slice == 0 && !so.normalize;
    const Json delaunay = Json{{"construction", "nxr-compute delaunay.h: the Studio's intrinsicDelaunay.ts (mollify, mixed-Voronoi mass of the mollified input, flip to Delaunay)"},
                               {"mollify", st.method.mollify}, {"flip_eps", st.method.flip_eps}, {"epsilon", h.mollify_epsilon}, {"flips", h.flips}};
    Json solve = Json{{"route", "split: Laplace-Beltrami (d0^T *1 d0, *0) and 2-form (*2 d1 *1^-1 d1^T *2, *2) ladders, shift-invert eigensolve, one lambda cut"},
                      {"reference", ref ? "nxr solve_hodge / Rheome hodge_dirac: one solve per ladder, absolute sigma, no re-orthonormalisation" : "not the reference method (an option differs)"},
                      {"requested", st.modes}, {"exact", h.n_exact}, {"coexact", h.n_coexact}, {"seconds", std::round(h.seconds * 10) / 10},
                      {"options", method}, {"lambda_top", h.top}, {"star1_clamped", h.star1_clamped}, {"delaunay", delaunay},
                      {"tolerance", so.slice == 0 && so.shift == hs::HodgeShift::Absolute ? Json(so.tol) : Json()}};
    const Json prov = Json{{"mesh", solved_mesh}, {"gluing", glue_record},   // (not `glue`: that key marks a glued surface, {original, map_path})
                           {"solve", solve}, {"mixing_max", 0.0}};
    Json complex_rec = delaunay;
    complex_rec["star1_floor"] = st.method.star1_floor ? Json(*st.method.star1_floor) : Json();
    complex_rec["star1_clamped"] = h.star1_clamped;
    if (h.gauge.is_object()) { complex_rec["gauge"] = h.gauge["gauge"]; complex_rec["gauge_record"] = h.gauge["gauge_record"]; }
    const Json fpj = Json{{"package", stem}, {"delaunay", complex_rec}};

    Arrays out;
    // THE MODE AXIS (points; coordinate mu = sqrt(lambda), ascending) and what lies on it
    out["modes"] = made(K, 1, h.mu, Json{{"cs", "eigenvalue:" + stem}, {"operator", "hodgeDirac"}, {"description", "the eigenvalues mu = sqrt(lambda) > 0 of the Hodge-Dirac operator d + delta on " + solved + ", ascending: the mode axis"},
                                         {"producer", [&] { Json j = Json{{"variant", "Hodge-Dirac (d + delta)"}, {"fieldType", "potentials"}, {"kept", "mu > 0"}}; for (auto it = prov.begin(); it != prov.end(); ++it) j[it.key()] = it.value(); return j; }()}});
    {   // the eigenvalues lambda = mu^2 as a FIELD of the operator (rule 31), beside the mode axis's coordinates mu
      std::vector<double> lam(h.mu.size());
      for (std::size_t i = 0; i < lam.size(); ++i) lam[i] = h.mu[i] * h.mu[i];
      out["eigenvalues"] = made(K, 1, std::move(lam), Json{{"value_type", "scalar"}, {"precision", "float64"}, {"description", "per mode: the eigenvalue lambda = mu^2 of the squared operator"}});
    }
    out["kind"] = made(K, 1, widened(h.kind), Json{{"dtype", "u8"}, {"value_type", "integer"}, {"shape", Json::array({K})},
                                                   {"description", "per mode: 0 exact (a 0-form potential A) · 2 coexact (a 2-form potential S)"}});
    out["support"] = made(static_cast<std::int64_t>(kept.size()), 1, widened(kept), Json{{"description", "the " + S.name + " vertices the " + "Hodge-Dirac surface keeps (all of them: one piece)"}});
    const Json solve_params = Json{{"potentials", "A = psi/sqrt2 (kind 0) · S = <chi/sqrt2, w_v> (kind 2)"}, {"solved", "K x = mu M x, d + delta on the intrinsic-Delaunay complex of " + solved}};
    out["eigenvectors"] = made(Vg, K, widened(h.eigenvectors), Json{{"precision", "float32"}, {"chunks", Json::array({Vg, 1})}, {"shards", Json::array({Vg, std::min<std::int64_t>(K, 64)})},
                               {"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"params", solve_params}, {"producer", prov},
                               {"description", "Hodge-Dirac modes as vertex potentials: A (exact) or S (coexact) per column; 1-form (grad A + n x grad S)/mu"}});
    out["forms2"] = made(nF, K, widened(h.forms2), Json{{"precision", "float32"}, {"chunks", Json::array({nF, 1})}, {"shards", Json::array({nF, std::min<std::int64_t>(K, 64)})},
                          {"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"to_cell", "2"}, {"producer", fpj},
                          {"params", Json{{"forms", "2-form part chi/sqrt2 of each Dirac eigenvector (0, delta chi/mu, chi)/sqrt2; *2-normalised: sum *2 (chi/sqrt2)^2 = 1/2"}, {"complex", iname}}},
                          {"description", "Hodge-Dirac modes' 2-form parts chi/sqrt2 on the intrinsic faces (0 on an exact mode); with " + stem + "_eigenvectors (the 0-form parts) the full forms"}});

    // THE INTRINSIC COMPLEX the modes were solved on (the same vertices, its own edges and faces; two edges may join one pair of vertices: read the incidence from d0 and d1)
    out["complex"] = made(Vg, 3, G.positions, Json{{"edges", h.edges}, {"faces", h.faces}, {"cs", cs}, {"producer", Json{{"intrinsic_of", solved}, {"mesh", solved_mesh}, {"delaunay", complex_rec}}},
                          {"description", "the intrinsic (mollified) Delaunay triangulation of " + solved + " the Hodge-Dirac modes " + stem + " are solved on: the same vertices, its own edges (lengths along the surface) and faces; read its incidence from " + iname + "_d0 and " + iname + "_d1"}});
    // d0 [nE x nV]: -1 at an edge's tail, +1 at its tip (csr: the columns ascending); d1 [nF x nE]: each face's edges with their signs
    {
      std::vector<std::int64_t> ip0(static_cast<std::size_t>(nE + 1)), ip1(static_cast<std::size_t>(nF + 1));
      std::vector<std::int32_t> ix0(static_cast<std::size_t>(2 * nE)), ix1(static_cast<std::size_t>(3 * nF));
      std::vector<double> dv0(static_cast<std::size_t>(2 * nE)), dv1(static_cast<std::size_t>(3 * nF));
      for (std::int64_t e = 0; e <= nE; ++e) ip0[static_cast<std::size_t>(e)] = 2 * e;
      for (std::int64_t e = 0; e < nE; ++e) {
        const std::int32_t a = h.edges[static_cast<std::size_t>(2 * e)], b = h.edges[static_cast<std::size_t>(2 * e + 1)];
        const bool order_ab = a < b;
        ix0[static_cast<std::size_t>(2 * e)] = order_ab ? a : b; dv0[static_cast<std::size_t>(2 * e)] = order_ab ? -1.0 : 1.0;
        ix0[static_cast<std::size_t>(2 * e + 1)] = order_ab ? b : a; dv0[static_cast<std::size_t>(2 * e + 1)] = order_ab ? 1.0 : -1.0;
      }
      for (std::int64_t f = 0; f <= nF; ++f) ip1[static_cast<std::size_t>(f)] = 3 * f;
      for (std::int64_t f = 0; f < nF; ++f) {
        std::array<std::pair<std::int32_t, double>, 3> row;
        for (int k = 0; k < 3; ++k) { const std::int32_t se = h.face_edges[static_cast<std::size_t>(3 * f + k)]; row[static_cast<std::size_t>(k)] = {std::abs(se) - 1, se > 0 ? 1.0 : -1.0}; }
        std::sort(row.begin(), row.end());
        for (int k = 0; k < 3; ++k) { ix1[static_cast<std::size_t>(3 * f + k)] = row[static_cast<std::size_t>(k)].first; dv1[static_cast<std::size_t>(3 * f + k)] = row[static_cast<std::size_t>(k)].second; }
      }
      auto sparse = [&](std::int64_t rows, std::int64_t cols, const std::vector<std::int64_t>& indptr, const std::vector<std::int32_t>& indices, const char* from_cell, const char* to_cell, const std::string& what) {
        return Json{{"precision", "float32"}, {"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"from_cell", from_cell}, {"to_cell", to_cell}, {"producer", fpj}, {"description", what},
                    {"sparse", Json{{"layout", "csr"}, {"rows", rows}, {"cols", cols}, {"indptr", indptr}, {"indices", indices}}}};
      };
      out["d0"] = made(2 * nE, 1, std::move(dv0), sparse(nE, Vg, ip0, ix0, "0", "1", "d0 of the intrinsic complex: vertices -> edges (-1 at an edge's tail, +1 at its tip)"));
      out["d1"] = made(3 * nF, 1, std::move(dv1), sparse(nF, nE, ip1, ix1, "1", "2", "d1 of the intrinsic complex: edges -> faces (each face's three edges, +1 along the face's winding)"));
    }
    // the signposts per edge: each edge's angle at its tail and at its tip (its twin's angle), each in its vertex's tangent plane
    std::vector<double> atail(static_cast<std::size_t>(nE), NAN), atip(static_cast<std::size_t>(nE), NAN);
    if (!h.angle.empty())
      for (std::int64_t k = 0; k < 3 * nF; ++k) {
        const std::int32_t se = h.face_edges[static_cast<std::size_t>(k)];
        (se > 0 ? atail : atip)[static_cast<std::size_t>(std::abs(se) - 1)] = h.angle[static_cast<std::size_t>(k)];
      }
    const std::vector<double> asum = h.angle_sum.empty() ? std::vector<double>(static_cast<std::size_t>(Vg), NAN) : h.angle_sum;
    const std::vector<std::int32_t> ref_nb = h.reference.empty() ? std::vector<std::int32_t>(static_cast<std::size_t>(Vg), -1) : h.reference;
    auto cells_field = [&](const char* cell, std::int64_t n, std::vector<double> v, const std::string& what, const char* unit = nullptr, const char* dtype = nullptr) {
      Json d = Json{{"cell", cell}, {"shape", Json::array({n})}, {"description", what}, {"producer", fpj}};
      if (unit) d["unit"] = unit;
      if (dtype) { d["dtype"] = dtype; d["value_type"] = "integer"; }
      return made(n, 1, std::move(v), d);
    };
    out["length"] = cells_field("1", nE, h.lengths, "each intrinsic edge's length along the surface (the input's units)");
    out["star0"] = cells_field("0", Vg, h.star0, "star0: each vertex's mixed-Voronoi mass of the (mollified) input, as the pencil used it");
    out["star1"] = cells_field("1", nE, h.star1, "star1: each intrinsic edge's cotangent weight (>= 0: Delaunay), as the pencil used it");
    out["star2"] = cells_field("2", nF, h.star2, "star2: 1 / each intrinsic face's area");
    out["angle_tail"] = cells_field("1", nE, std::move(atail), "the signpost: each edge's angle at its tail, in the tail's tangent plane (from its reference neighbour)", "rad");
    out["angle_tip"] = cells_field("1", nE, std::move(atip), "the signpost: each edge's angle at its tip, in the tip's tangent plane", "rad");
    out["angle_sum"] = cells_field("0", Vg, asum, "each vertex's cone angle (the sum of its corners)", "rad");
    out["reference"] = cells_field("0", Vg, widened(ref_nb), "each vertex's reference neighbour: the head of its angle-0 edge (an input edge)", nullptr, "i32");
    return out;
  };

  s.record = [](const std::any& state) {
    const HodgeState& st = std::any_cast<const HodgeState&>(state);
    return Json{{"modes", st.modes}, {"method", method_json(st.method)}};
  };
  return s;
}


// ── surface.glue ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE GLUED CORTEX (rule 33), made by its own definition (`glue`): the two hemispheres' medial walls (the Desikan-Killiany atlas' `unassigned`) cut as disks and a BRIDGE grown between the rims (the Studio's
// glue.ts glueMesh, nxr-compute glue.h, bit-exact). Makes the glued surface (the kept vertices in ascending original index, then the bridge's), the MAP glued vertex -> original vertex (-1 on the bridge: not
// cortex) as a field of the glued surface, and the bridge as an indices selection. The surface's producer record carries {glue: {original}}; the map is the field `<glued name>_map` beside it; a result on the glued surface is restricted back to the original.
struct GlueState { std::string surface; };
StepSpec surface_glue_step() {
  StepSpec s;
  s.name = "surface.glue";
  s.kind = "setup";
  s.inputs = {"surface"};
  s.outputs = {"glued", "map", "bridge"};
  s.parameters = Json::object();
  s.matches = "the Studio's glue.ts glueMesh (nxr-compute glue.h; nxr geom.glue)";
  s.description = "a cortex of two hemispheres glued by anatomy into ONE closed surface: each medial wall cut out as a disk, a bridge of new vertices grown between the two rims. Makes the glued surface, the map "
                  "glued vertex -> original vertex (-1 on the bridge) and the bridge selection; refused when the pieces do not match";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface: positions as the Studio receives them (float32 widened), double");
  s.plan = [](const Values& ins, const Json&, const PlanContext&) {
    need(ins.count("surface") != 0 && ins.at("surface").row, "surface.glue: the input `surface` is missing");
    const std::string sid = ins.at("surface").manifolds.empty() ? std::string() : ins.at("surface").manifolds[0];
    Values out;
    for (const char* port : {"glued", "map", "bridge"}) {
      Value v;
      v.cls = std::string(port) == "glued" ? "manifold" : std::string(port) == "bridge" ? "selection" : "field";
      v.manifolds = {sid};
      if (std::string(port) == "map") v.value_type = "integer";
      out[port] = std::move(v);
    }
    return out;
  };
  s.prepare = [](const Values& ins, const Json&, ExecContext&) { return std::any(GlueState{nxr::io::text(*ins.at("surface").row, "name")}); };
  s.resolve = [](const Values& ins, const Json&, const Bound& bound) {
    const SurfaceInput S = bound.surface(nxr::io::text(*ins.at("surface").row, "name"));
    return Json{{"mesh", mesh_key(S.faces)}, {"positions", positions_key(S.vertices)}, {"medial_wall", wall_key(bound.medial_wall(S))}, {"glue", "anatomy"}};
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const GlueState& st = std::any_cast<const GlueState&>(state);
    const SurfaceInput S = ctx.bound.surface(st.surface);
    const int pieces = connected_pieces(S.faces, S.n_vertices());
    need(pieces == 2, st.surface + " is " + std::to_string(pieces) + " piece(s): `glue` is for a cortex of two pieces (its hemispheres)");
    const std::vector<std::uint8_t> wall = ctx.bound.medial_wall(S);
    const nxr::glue::Glued G = nxr::glue::glueMesh(S.vertices, S.faces, wall);
    const std::int64_t Vg = static_cast<std::int64_t>(G.keep.size());
    std::int64_t kept = 0;
    for (const auto k : G.keep) if (k >= 0) ++kept;
    const Json bridge = Json{{"vertices", G.bridge.vertices}, {"rings", G.bridge.rings}, {"area", G.bridge.area}, {"longestEdge", G.bridge.longestEdge}};
    if (ctx.say) ctx.say(Json{{"stage", "glued"}, {"surface", S.name}, {"nV", Vg}, {"nF", static_cast<std::int64_t>(G.faces.size() / 3)}, {"mesh", mesh_key(G.faces)}, {"bridge", bridge}});
    const Json glue_record = Json{{"by", "anatomy: the Desikan-Killiany medial wall cut, a bridge grown"}, {"bridge", bridge}, {"nV", Vg}, {"positions", "float32, as the Studio receives them"},
                                  {"reference", "the Studio's glue.ts glueMesh (nxr-compute glue.h, bit-exact)"}};
    Arrays out;
    // the producer's `glue` marks a glued surface: the original manifold and the map's field (named, found beside the surface); results on it are restricted back through the map
    out["glued"] = made(Vg, 3, G.positions, Json{{"faces", G.faces}, {"cs", ctx.bound.coordinate_system_key(S.id)},
                        {"description", S.name + " glued by anatomy: its medial walls cut (the Desikan-Killiany atlas), a bridge grown between the rims; vertex i < " + std::to_string(kept) + " is a kept vertex of " + S.name + ", the rest the bridge"},
                        {"producer", Json{{"glued_from", S.name}, {"mesh", mesh_key(G.faces)}, {"gluing", glue_record}, {"glue", Json{{"original", S.id}}}}}});
    out["map"] = made(Vg, 1, widened(G.keep), Json{{"dtype", "i32"}, {"value_type", "integer"}, {"shape", Json::array({Vg})}, {"glue_map", true},
                      {"description", "per glued vertex: its original vertex in " + S.name + ", -1 on the bridge (not cortex)"}});
    std::vector<double> br;
    for (std::int64_t i = kept; i < Vg; ++i) br.push_back(static_cast<double>(i));
    const std::int64_t nb = static_cast<std::int64_t>(br.size());
    out["bridge"] = made(nb, 1, std::move(br), Json{{"glue_map", true}, {"description", "the medial BRIDGE (not cortex): label 0 'bridge: not cortex'; to mask it, no sources, a forward zero there"}});
    return out;
  };
  return s;
}

}  // namespace

void add_eigen_steps(std::vector<StepSpec>& r) { r.push_back(hodge_dirac_step()); r.push_back(surface_glue_step()); }

}  // namespace nxr::pipelines::graph
