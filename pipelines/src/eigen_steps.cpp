// nxr-pipelines — THE HODGE-DIRAC EIGENSOLVE AND ITS INTRINSIC TRIANGULATION AS REGISTERED STEPS, stored in the four classes on the schema's chain-complex design (D92, D93).
//
//   geometry.intrinsic   setup   surface (one closed piece) -> its intrinsic triangulation: a manifold with the surface's positions and its own edges and faces, d0, d1 and the three stars as ANALYTIC
//                                operators on it (method 'dec', no bytes), its edge lengths and signposts as fields on its cells
//   hodge_dirac          setup   surface + its intrinsic triangulation -> the eigenvalue line, the eigenvalues, the kinds labelling and ONE eigenvectors operator (the mixed forms)
//   geometry.glue        setup   a cortex of two hemispheres -> its glued surface and the labelling of its vertices (left, right, bridge)
//
// THE GRAPH (written first, docs/specs/2026-11-17-eigenmodes-as-definition.md; the Studio's chain: glue.ts glueMesh -> intrinsicDelaunay.ts -> hodgeDirac.ts dec(m) -> the two ladders):
//   surface ──[intrinsic Delaunay: mollify, flip]──> the intrinsic triangulation (cells 0, 1, 2) with d0, d1, the stars (analytic), the lengths and signposts      (`run intrinsic`)
//   surface + triangulation ──[split solve: LB ladder (exact modes) + 2-form ladder (coexact modes), one lambda cut, the constants out]──> mu, the kinds, each mode's mixed form   (`run eigenmodes`)
// The steps read no file and no row: the surfaces are read by the runner (Bound) and handed over; everything they make is DESCRIBED (meta.describe) and stored by io from the definitions' declared
// outputs (intrinsic.yaml, eigenmodes.yaml). The eigensolve checks that the triangulation it is handed is the one the construction makes from the surface with the triangulation's recorded options.
//
// The method rule: the default is the REFERENCE method of nxr and Rheome (see eigen.h: SolveMethod); the other methods are named options, every option is in the run's provenance and cache key.
// What it cannot resolve: the modes are the discrete operator's on the intrinsic triangulation (mollified, flipped to Delaunay): a mode above the mesh's resolution is not a property of the cortex;
// a surface of two or more pieces is refused (a cortex of two hemispheres is glued by `geometry.glue` first, and the glued surface named as the input): the bridge is not cortex.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <set>
#include <tuple>

#include "nxr/geometry/curvature.h"
#include "nxr/operators/factorization.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/geometry/surface_maps.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/registry.h"

namespace nxr::pipelines::graph {

namespace {

namespace hs = nxr::operators;

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
  std::string surface, intrinsic;
  std::int64_t n_edges = 0;
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
  return sm;
}

// ── geometry.intrinsic ──────────────────────────────────────────────────────────────────────────────────────────
// THE INTRINSIC TRIANGULATION of a surface, by its own definition (`intrinsic`): the surface's GEOMETRY (its positions, the same array) with its OWN TOPOLOGY (the flipped edges and faces are its cells).
// d0, d1 and the stars are ANALYTIC operators on it (layout analytic, method 'dec', D93: evaluated from the triangulation as the Studio's hodgeDirac.ts dec(m) assembles them at setup); the edge lengths
// (its metric: a flipped edge's length is not a distance between positions) and the signposts are fields on its cells.
struct IntrinsicState { std::string surface; double mollify = 1e-6, flip_eps = 1e-6; };
const char* const kIntrinsicOutputs[] = {"triangulation", "d0", "d1", "star0", "star1", "star2", "length", "angle_tail", "angle_tip", "angle_sum", "reference"};

Json triangulation_record(const IntrinsicTriangulation& t) {
  Json r = Json{{"construction", "nxr-compute delaunay.h: the Studio's intrinsicDelaunay.ts (mollify, mixed-Voronoi mass of the mollified input, flip to Delaunay)"},
                {"mollify", t.mollify}, {"flip_eps", t.flip_eps}, {"epsilon", t.mollify_epsilon}, {"flips", t.flips}};
  if (t.gauge.is_object()) { r["gauge"] = t.gauge["gauge"]; r["gauge_record"] = t.gauge["gauge_record"]; }
  return r;
}

StepSpec intrinsic_step() {
  StepSpec s;
  s.name = "geometry.intrinsic";
  s.kind = "setup";
  s.inputs = {"surface"};
  for (const char* o : kIntrinsicOutputs) s.outputs.push_back(o);
  s.parameters = Json{{"mollify", 1e-6}, {"flip_eps", 1e-6}};
  s.matches = "the Studio's intrinsicDelaunay.ts (intrinsicMesh, mollify, flipToDelaunay) and hodgeDirac.ts dec(m); nxr-compute delaunay.h";
  s.description = "the intrinsic Delaunay triangulation of a surface of one closed piece: its own edges and faces on the surface's positions, d0, d1 and the Hodge stars as analytic operators (method dec), "
                  "the edge lengths and the signposts as fields on its cells";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface: positions as the Studio receives them (float32 widened), double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("surface") != 0 && ins.at("surface").row, "geometry.intrinsic: the input `surface` is missing");
    need(p.value("mollify", 1e-6) >= 0 && p.value("flip_eps", 1e-6) >= 0, "geometry.intrinsic: mollify and flip_eps are >= 0");
    const std::string sid = ins.at("surface").manifolds.empty() ? std::string() : ins.at("surface").manifolds[0];
    Values out;
    for (const char* o : kIntrinsicOutputs) {
      const std::string port = o;
      Value v;
      v.cls = port == "triangulation" ? "manifold" : (port == "d0" || port == "d1" || port.rfind("star", 0) == 0) ? "operator" : "field";
      v.manifolds = v.cls == "operator" ? std::vector<std::string>{sid, sid} : std::vector<std::string>{sid};
      if (port == "reference") v.value_type = "integer";
      out[port] = std::move(v);
    }
    return out;
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    return std::any(IntrinsicState{nxr::io::text(*ins.at("surface").row, "name"), p.value("mollify", 1e-6), p.value("flip_eps", 1e-6)});
  };
  s.resolve = [](const Values& ins, const Json&, const Bound& bound) {
    const SurfaceInput S = bound.surface(nxr::io::text(*ins.at("surface").row, "name"));
    return Json{{"mesh", mesh_key(S.faces)}, {"positions", positions_key(S.vertices)}};
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const IntrinsicState& st = std::any_cast<const IntrinsicState&>(state);
    const SurfaceInput S = ctx.bound.surface(st.surface);
    const std::int64_t V = S.n_vertices();
    const int pieces = connected_pieces(S.faces, V);
    need(pieces == 1, st.surface + " is " + std::to_string(pieces) + " pieces: the intrinsic triangulation is made on a surface of one closed piece (a cortex's two hemispheres are glued first by `run glue`, and the glued surface named)");
    const IntrinsicTriangulation t = intrinsic_triangulation(S.vertices, S.faces, st.mollify, st.flip_eps);
    if (ctx.say) ctx.say(Json{{"stage", "triangulated"}, {"surface", S.name}, {"nV", V}, {"nE", t.n_edges()}, {"nF", t.n_faces()}, {"flips", t.flips}});
    const std::int64_t nE = t.n_edges(), nF = t.n_faces();
    const std::string iname = S.name + "_intrinsic";
    const Json rec = triangulation_record(t);
    const Json producer = Json{{"intrinsic_of", S.name}, {"mesh", mesh_key(S.faces)}, {"delaunay", rec}};
    Arrays out;
    // THE TRIANGULATION: the surface's positions (shared: no vertices of its own), its own edges and faces (two edges may join one pair of vertices: read the incidence from face_edges, d0, d1)
    {
      Array a;
      a.rows = V; a.cols = 3;
      a.meta["describe"] = Json{{"edges", t.edges}, {"faces", t.faces}, {"cs", ctx.bound.coordinate_system_key(S.id)}, {"producer", producer},
                                {"description", "the intrinsic (mollified) Delaunay triangulation of " + S.name + ": its positions, its own edges (lengths along the surface) and faces"}};
      out["triangulation"] = std::move(a);
    }
    // the ANALYTIC operators (no bytes): their shape in cells, how they are evaluated in params
    auto analytic = [&](const char* op, std::int64_t rows, std::int64_t cols, const char* from_cell, const char* to_cell, const std::string& what) {
      Array a;
      a.meta["describe"] = Json{{"analytic", Json{{"method", "dec"}, {"rows", rows}, {"cols", cols}}}, {"from_cell", from_cell}, {"to_cell", to_cell}, {"description", what},
                                {"params", Json{{"operator", op}, {"mollify", t.mollify}, {"flip_eps", t.flip_eps},
                                                {"evaluated", "from the triangulation (its cells, its lengths field) as the Studio's hodgeDirac.ts dec(m) and nxr-compute hodgedirac::assemble build it"}}},
                                {"producer", producer}};
      return a;
    };
    out["d0"] = analytic("d0", nE, V, "0", "1", "d0 of " + iname + ": vertices -> edges (-1 at an edge's tail, +1 at its tip)");
    out["d1"] = analytic("d1", nF, nE, "1", "2", "d1 of " + iname + ": edges -> faces (each face's three edges, +1 along the face's winding)");
    out["star0"] = analytic("star0", V, V, "0", "0*", "the Hodge star *0 of " + iname + ": each vertex's mixed-Voronoi mass of the mollified input (diagonal)");
    out["star1"] = analytic("star1", nE, nE, "1", "1*", "the Hodge star *1 of " + iname + ": each edge's cotangent weight (diagonal; a solve's floor is the solve's option)");
    out["star2"] = analytic("star2", nF, nF, "2", "2*", "the Hodge star *2 of " + iname + ": 1 / each face's area (diagonal)");
    // the signposts per edge: each edge's angle at its tail and at its tip (its twin's angle), each in its vertex's tangent plane
    std::vector<double> atail(static_cast<std::size_t>(nE), NAN), atip(static_cast<std::size_t>(nE), NAN);
    for (std::int64_t k = 0; k < 3 * nF && !t.angle.empty(); ++k) {
      const std::int32_t se = t.face_edges[static_cast<std::size_t>(k)];
      (se > 0 ? atail : atip)[static_cast<std::size_t>(std::abs(se) - 1)] = t.angle[static_cast<std::size_t>(k)];
    }
    auto cells_field = [&](const char* cell, std::int64_t n, std::vector<double> v, const std::string& what, const char* unit = nullptr, const char* dtype = nullptr) {
      Json d = Json{{"cell", cell}, {"shape", Json::array({n})}, {"description", what}, {"producer", producer}};
      if (unit) d["unit"] = unit;
      if (dtype) { d["dtype"] = dtype; d["value_type"] = "integer"; }
      return made(n, 1, std::move(v), d);
    };
    out["length"] = cells_field("1", nE, t.lengths, "each intrinsic edge's length along the surface (the input's units)");
    out["angle_tail"] = cells_field("1", nE, std::move(atail), "the signpost: each edge's angle at its tail, in the tail's tangent plane (from its reference neighbour)", "rad");
    out["angle_tip"] = cells_field("1", nE, std::move(atip), "the signpost: each edge's angle at its tip, in the tip's tangent plane", "rad");
    out["angle_sum"] = cells_field("0", V, t.angle_sum.empty() ? std::vector<double>(static_cast<std::size_t>(V), NAN) : t.angle_sum, "each vertex's cone angle (the sum of its corners)", "rad");
    out["reference"] = cells_field("0", V, widened(t.reference.empty() ? std::vector<std::int32_t>(static_cast<std::size_t>(V), -1) : t.reference),
                                   "each vertex's reference neighbour: the head of its angle-0 edge (an input edge)", nullptr, "i32");
    return out;
  };
  s.record = [](const std::any& state) {
    const IntrinsicState& st = std::any_cast<const IntrinsicState&>(state);
    return Json{{"mollify", st.mollify}, {"flip_eps", st.flip_eps}};
  };
  return s;
}

// ── hodge_dirac ─────────────────────────────────────────────────────────────────────────────────────────────────
const char* const kOutputs[] = {"modes", "eigenvalues", "kinds", "eigenvectors"};

// the construction's options a stored triangulation records (its producer's `delaunay`)
std::pair<double, double> triangulation_options(const nxr::io::Row& row) {
  const Json pj = Json::parse(nxr::io::opt_text(row, "producer_json").value_or("{}"), nullptr, false);
  need(pj.is_object() && pj.contains("delaunay") && pj["delaunay"].is_object() && pj["delaunay"].contains("mollify") && pj["delaunay"].contains("flip_eps"),
       nxr::io::text(row, "name") + ": the intrinsic triangulation records no construction options (made by `run intrinsic`)");
  return {pj["delaunay"]["mollify"].get<double>(), pj["delaunay"]["flip_eps"].get<double>()};
}

StepSpec hodge_dirac_step() {
  StepSpec s;
  s.name = "operators.hodge_dirac";
  s.kind = "setup";
  s.inputs = {"surface", "intrinsic"};
  for (const char* o : kOutputs) s.outputs.push_back(o);
  s.parameters = Json{{"modes", 3000}, {"shift", "absolute"}, {"sigma", nullptr}, {"slice", 0}, {"normalize", false}, {"star1_floor", nullptr}};
  s.matches = "nxr.eigen.hodge_dirac.solve_hodge and nxr.pipelines.hodge.solved_package; Rheome rheome.eigen.hodge_dirac; the Studio's hodgeDirac.ts (the mixed forms, rows 0-forms · 1-forms · 2-forms)";
  s.description =
      "The Hodge-Dirac eigenmodes of a surface on its intrinsic triangulation (made first by `geometry.intrinsic`): the Laplace-Beltrami ladder (exact modes) and the 2-form ladder (coexact modes), cut at one lambda, "
      "the constants out. Makes the eigenvalue line (mu, ascending), the eigenvalues, the kinds labelling (exact, coexact) and ONE eigenvectors operator from the line to the triangulation's cells "
      "0, 1 and 2: each mode's mixed form (the vertex potential, the 1-form part, the 2-form part), rows stacked rank by rank.";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface: positions as the Studio receives them (float32 widened), double");

  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("surface") != 0 && ins.at("surface").row, "hodge_dirac: the input `surface` is missing");
    need(ins.count("intrinsic") != 0 && ins.at("intrinsic").row, "hodge_dirac: the input `intrinsic` (the surface's intrinsic triangulation) is missing");
    need(p.value("modes", 3000) >= 2, "hodge_dirac: modes must be at least 2");
    method_of(p);                                                     // the options refused here, before anything is read
    triangulation_options(*ins.at("intrinsic").row);
    const auto& row = *ins.at("surface").row;
    const std::string name = nxr::io::text(row, "name"), stem = stem_of(name, nxr::io::integer(row, "is_primary") != 0);
    const std::string sid = ins.at("surface").manifolds.empty() ? std::string() : ins.at("surface").manifolds[0];
    const std::string tid = ins.at("intrinsic").manifolds.empty() ? std::string() : ins.at("intrinsic").manifolds[0];
    Values out;
    for (const char* o : kOutputs) {
      Value v;
      const std::string port = o;
      v.cls = port == "modes" ? "manifold" : port == "kinds" ? "selection" : port == "eigenvectors" ? "operator" : "field";
      if (v.cls == "operator") v.manifolds = {sid, tid};                  // the places themselves are declared (from, to)
      else v.manifolds = {sid};
      v.facts["names"] = Json{{"stem", stem}, {"solved", name}};
      out[port] = std::move(v);
    }
    return out;
  };

  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    HodgeState st;
    st.surface = nxr::io::text(*ins.at("surface").row, "name");
    st.intrinsic = nxr::io::text(*ins.at("intrinsic").row, "name");
    st.n_edges = nxr::io::integer(*ins.at("intrinsic").row, "n_edges");
    st.modes = p.value("modes", 3000);
    st.method = method_of(p);
    std::tie(st.method.mollify, st.method.flip_eps) = triangulation_options(*ins.at("intrinsic").row);   // the triangulation's own options: the solve is on it
    return std::any(std::move(st));
  };

  // what the request resolves to on what is stored: the surface's mesh and positions (nxr's mesh key); the triangulation is an input (its id is in the key)
  s.resolve = [](const Values& ins, const Json&, const Bound& bound) {
    const SurfaceInput S = bound.surface(nxr::io::text(*ins.at("surface").row, "name"));
    return Json{{"mesh", mesh_key(S.faces)}, {"positions", positions_key(S.vertices)}, {"forms", "the mixed forms: one operator to the intrinsic triangulation's cells 0+1+2"}};
  };

  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const HodgeState& st = std::any_cast<const HodgeState&>(state);
    auto say = [&](const Json& j) { if (ctx.say) ctx.say(j); };
    const SurfaceInput S = ctx.bound.surface(st.surface);               // float32 widened, as the Studio's GPU-side pipelines receive them
    const SurfaceInput T = ctx.bound.surface(st.intrinsic);             // the triangulation: the same positions, its own faces
    const std::int64_t V = S.n_vertices();
    const int pieces = connected_pieces(S.faces, V);
    need(pieces == 1, st.surface + " is " + std::to_string(pieces) + " pieces: the Hodge-Dirac solve takes a surface of one closed piece (a cortex's two hemispheres are glued first by `run glue`, and the glued surface named as the input)");
    need(T.n_vertices() == V, st.intrinsic + " has " + std::to_string(T.n_vertices()) + " vertices, " + st.surface + " " + std::to_string(V) + ": not its intrinsic triangulation");
    const std::string stem = stem_of(S.name, S.primary), solved = S.name;
    const std::string solved_mesh = mesh_key(S.faces);
    const Json glue_record = Json{{"by", "none: one closed piece"}, {"nV", V}};

    say(Json{{"stage", "solve"}, {"surface", S.name}, {"intrinsic", st.intrinsic}, {"nV", V}, {"nF", S.n_faces()}, {"mesh", solved_mesh}});
    HodgeDirac h = solve_hodge_dirac(S.vertices, S.faces, st.modes, st.method);
    say(Json{{"stage", "solved"}, {"exact", h.n_exact}, {"coexact", h.n_coexact}, {"seconds", std::round(h.seconds * 10) / 10}});
    // the triangulation it was handed is the one the solve is on (the construction with the triangulation's options): its faces and its edge count
    need(h.faces == T.faces && static_cast<std::int64_t>(h.lengths.size()) == st.n_edges,
         st.intrinsic + " is not the intrinsic triangulation the construction makes from " + S.name + " with its recorded options (made from another surface or by another construction): make it again with `run intrinsic --replace`");

    const std::int64_t K = h.n_modes(), nE = static_cast<std::int64_t>(h.lengths.size()), nF = static_cast<std::int64_t>(h.faces.size() / 3), N = V + nE + nF;
    const hs::HodgeSolveOptions& so = st.method.solve;
    const Json method = method_json(st.method);
    const bool ref = so.shift == hs::HodgeShift::Absolute && so.slice == 0 && !so.normalize;
    Json solve = Json{{"route", "split: Laplace-Beltrami (d0^T *1 d0, *0) and 2-form (*2 d1 *1^-1 d1^T *2, *2) ladders, shift-invert eigensolve, one lambda cut"},
                      {"reference", ref ? "nxr solve_hodge / Rheome hodge_dirac: one solve per ladder, absolute sigma, no re-orthonormalisation" : "not the reference method (an option differs)"},
                      {"requested", st.modes}, {"exact", h.n_exact}, {"coexact", h.n_coexact}, {"seconds", std::round(h.seconds * 10) / 10},
                      {"options", method}, {"lambda_top", h.top}, {"star1_clamped", h.star1_clamped}, {"intrinsic", st.intrinsic},
                      {"tolerance", so.slice == 0 && so.shift == hs::HodgeShift::Absolute ? Json(so.tol) : Json()}};
    const Json prov = Json{{"mesh", solved_mesh}, {"gluing", glue_record}, {"solve", solve}, {"mixing_max", 0.0}};

    Arrays out;
    // THE EIGENVALUE LINE (points; coordinate mu = sqrt(lambda), ascending) and what lies on it
    out["modes"] = made(K, 1, h.mu, Json{{"cs", "eigenvalue:" + stem}, {"operator", "hodgeDirac"}, {"description", "the eigenvalues mu = sqrt(lambda) > 0 of the Hodge-Dirac operator d + delta on " + solved + ", ascending: the eigenvalue line"},
                                         {"producer", [&] { Json j = Json{{"variant", "Hodge-Dirac (d + delta)"}, {"fieldType", "potentials"}, {"kept", "mu > 0"}}; for (auto it = prov.begin(); it != prov.end(); ++it) j[it.key()] = it.value(); return j; }()}});
    {   // the eigenvalues lambda = mu^2 as a FIELD of the operator, beside the line's coordinates mu
      std::vector<double> lam(h.mu.size());
      for (std::size_t i = 0; i < lam.size(); ++i) lam[i] = h.mu[i] * h.mu[i];
      out["eigenvalues"] = made(K, 1, std::move(lam), Json{{"value_type", "scalar"}, {"precision", "float64"}, {"description", "per mode: the eigenvalue lambda = mu^2 of the squared operator"}});
    }
    // THE KINDS: a labelling of the eigenvalue line by the Hodge part (the label is the potential's form degree: 0 exact, 2 coexact; no harmonic part on a closed surface of genus 0)
    out["kinds"] = made(K, 1, widened(h.kind), Json{{"entries", Json::array({"exact", "harmonic", "coexact"})},
                                                    {"description", "the kind of each mode: exact (label 0, a 0-form potential) or coexact (label 2, a 2-form potential); no mode is harmonic on a closed surface of genus 0"}});
    // THE EIGENVECTORS: ONE operator, the eigenvalue line -> the triangulation's cells 0+1+2, each mode's mixed form stacked rank by rank (the Studio's Hodge-Dirac order). Built mode-major per block, the
    // solve's float parts released as they are copied (no second copy of all three held at once)
    {
      std::vector<double> stacked(static_cast<std::size_t>(N * K));
      auto put = [&](std::vector<float>& part, std::int64_t r0, std::int64_t rows) {
        for (std::int64_t r = 0; r < rows; ++r) for (std::int64_t k = 0; k < K; ++k) stacked[static_cast<std::size_t>((r0 + r) * K + k)] = part[static_cast<std::size_t>(r * K + k)];
        std::vector<float>().swap(part);
      };
      put(h.eigenvectors, 0, V);
      put(h.forms1, V, nE);
      put(h.forms2, V + nE, nF);
      const Json solve_params = Json{{"rows", "0-forms [0, " + std::to_string(V) + ") · 1-forms [" + std::to_string(V) + ", " + std::to_string(V + nE) + ") · 2-forms [" + std::to_string(V + nE) + ", " + std::to_string(N) + ")"},
                                     {"forms", "each mode's mixed form, orthonormal under the stars *0, *1, *2: the 0-form part psi/sqrt2 (0 on a coexact mode); the 1-form part d psi/(mu sqrt2) or *1^-1 d1^T *2 chi/(mu sqrt2) (*1-orthogonal, energy 1/2); the 2-form part chi/sqrt2 (0 on an exact mode). A coexact mode's stream function at the vertices is derived from its 2-form part (each vertex the mean over its faces of *2 chi/sqrt2), never stored"},
                                     {"solved", "K x = mu M x, d + delta on the intrinsic triangulation " + st.intrinsic + " of " + solved},
                                     {"analysis", "the line <- triangulation map is this operator's D106 pair (its M-adjoint), not stored"}};
      out["eigenvectors"] = made(N, K, std::move(stacked), Json{{"precision", "float32"}, {"chunks", Json::array({V, 1})}, {"shards", Json::array({V, std::min<std::int64_t>(K, 64)})},
                                 {"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"from_cell", "0"}, {"to_cell", "0+1+2"}, {"params", solve_params}, {"producer", prov},
                                 {"description", "Hodge-Dirac modes as mixed forms on " + st.intrinsic + ": rows 0-forms (the vertices), 1-forms (the edges), 2-forms (the faces), one column a mode"}});
    }
    return out;
  };

  s.record = [](const std::any& state) {
    const HodgeState& st = std::any_cast<const HodgeState&>(state);
    return Json{{"modes", st.modes}, {"method", method_json(st.method)}};
  };
  return s;
}


// ── geometry.glue ────────────────────────────────────────────────────────────────────────────────────────────────
// THE GLUED CORTEX, made by its own definition (`glue`): the two hemispheres' medial walls (the Desikan-Killiany atlas' `unassigned`) cut out EXACTLY (the wall's vertices and the faces touching
// them, nothing more) and a BRIDGE grown between the rims (the Studio's glue.ts glueMesh, nxr-compute glue.h; bit-exact where the exact cut is one simple loop). The hemispheres are READ, never found: the caller names the original's HEMISPHERE LABELLING (an atlas: Brainstorm's `Structures`, `Cortex L` /
// `Cortex R` / `unassigned`) and which of its entries are the left and the right; a labelling whose left or right member is not one connected piece, that leaves cortex outside its own `unassigned`
// unlabelled, or that puts one face in both hemispheres is refused with the reason. Makes the glued surface (every vertex off the wall in ascending original index, then the bridge's) and ONE
// LABELLING of it (the input's entries, names, codes and colours, and one more: `bridge`): the correspondence glued -> original is its non-bridge vertices in order <-> the original's vertices off the wall.
struct GlueState { std::string surface, hemispheres; std::string left, right; };

StepSpec surface_glue_step() {
  StepSpec s;
  s.name = "geometry.glue";
  s.kind = "setup";
  s.inputs = {"surface", "hemispheres"};
  s.outputs = {"glued", "labels"};
  s.parameters = Json{{"left", "Cortex L"}, {"right", "Cortex R"}};
  s.matches = "the Studio's glue.ts glueMesh (nxr-compute glue.h; nxr geom.glue)";
  s.description = "a cortex of two hemispheres glued by anatomy into ONE closed surface: each medial wall cut out exactly (nothing more), a bridge of new vertices grown between the two rims. The hemispheres are the caller's "
                  "labelling of the original (left and right named); makes the glued surface and its hemispheres labelling (the input's entries and `bridge`); refused when the "
                  "labelling and the surface disagree";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface: positions as the Studio receives them (float32 widened), double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("surface") != 0 && ins.at("surface").row, "geometry.glue: the input `surface` is missing");
    need(ins.count("hemispheres") != 0 && ins.at("hemispheres").row, "geometry.glue: the input `hemispheres` (the original's hemisphere labelling) is missing");
    need(p.value("left", std::string()) != p.value("right", std::string()) && !p.value("left", std::string()).empty(), "geometry.glue: left and right name two entries of the hemisphere labelling");
    const std::string sid = ins.at("surface").manifolds.empty() ? std::string() : ins.at("surface").manifolds[0];
    Values out;
    for (const char* port : {"glued", "labels"}) {
      Value v;
      v.cls = std::string(port) == "glued" ? "manifold" : "selection";
      v.manifolds = {sid};
      out[port] = std::move(v);
    }
    return out;
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    return std::any(GlueState{nxr::io::text(*ins.at("surface").row, "name"), nxr::io::text(*ins.at("hemispheres").row, "id"), p.value("left", std::string("Cortex L")), p.value("right", std::string("Cortex R"))});
  };
  s.resolve = [](const Values& ins, const Json&, const Bound& bound) {
    const SurfaceInput S = bound.surface(nxr::io::text(*ins.at("surface").row, "name"));
    return Json{{"mesh", mesh_key(S.faces)}, {"positions", positions_key(S.vertices)}, {"medial_wall", wall_key(bound.medial_wall(S))}, {"glue", "anatomy"}};
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const GlueState& st = std::any_cast<const GlueState&>(state);
    const SurfaceInput S = ctx.bound.surface(st.surface);
    const StoredLabelling H = ctx.bound.labelling(st.hemispheres);
    const std::int64_t V = S.n_vertices();
    need(static_cast<std::int64_t>(H.labels.size()) == V, H.name + " labels " + std::to_string(H.labels.size()) + " vertices, " + S.name + " has " + std::to_string(V) + ": not a labelling of it");
    const std::string disagree = hemispheres_disagree(S.faces, H, st.left, st.right);   // each one connected piece, every vertex outside its own `unassigned` in one, no face in both
    need(disagree.empty(), disagree + " (" + S.name + ")");
    const std::vector<std::uint8_t> wall = ctx.bound.medial_wall(S);
    const nxr::geometry::glue::Glued G = nxr::geometry::glue::glueMesh(S.vertices, S.faces, wall);
    const std::int64_t Vg = static_cast<std::int64_t>(G.keep.size());
    std::vector<std::int32_t> kept;
    for (const auto k : G.keep) if (k >= 0) kept.push_back(k);
    const std::int64_t m = static_cast<std::int64_t>(kept.size());
    std::vector<std::int32_t> off_wall;
    for (std::int64_t v = 0; v < V; ++v) if (!wall[static_cast<std::size_t>(v)]) off_wall.push_back(static_cast<std::int32_t>(v));
    need(kept == off_wall && std::all_of(G.keep.begin() + m, G.keep.end(), [](std::int32_t k) { return k < 0; }),
         "glue: the glued vertices are not every vertex off the medial wall in ascending order, then the bridge's");
    const Json bridge = Json{{"vertices", G.bridge.vertices}, {"rings", G.bridge.rings}, {"area", G.bridge.area}, {"longestEdge", G.bridge.longestEdge}};
    if (ctx.say) ctx.say(Json{{"stage", "glued"}, {"surface", S.name}, {"nV", Vg}, {"nF", static_cast<std::int64_t>(G.faces.size() / 3)}, {"mesh", mesh_key(G.faces)}, {"bridge", bridge}});
    const Json glue_record = Json{{"by", "anatomy: exactly the Desikan-Killiany medial wall cut, a bridge grown"}, {"bridge", bridge}, {"nV", Vg}, {"positions", "float32, as the Studio receives them"},
                                  {"reference", "the Studio's glue.ts glueMesh (nxr-compute glue.h), the cut never widened beyond the wall"},
                                  {"hemispheres", Json{{"labelling", H.name}, {"left", st.left}, {"right", st.right}}}};
    // the glued labelling: each kept vertex its input label, the bridge one more entry (a code after the input's)
    std::int64_t bcode = 0;
    Json lookup = Json::array();
    for (const auto& e : H.entries) {
      bcode = std::max(bcode, e.code + 1);
      Json x = Json{{"code", e.code}, {"name", e.name}};
      if (e.color) x["colour"] = Json::array({(*e.color)[0], (*e.color)[1], (*e.color)[2]});
      lookup.push_back(x);
    }
    lookup.push_back(Json{{"code", bcode}, {"name", "bridge"}, {"colour", Json::array({128, 128, 128})}});
    std::vector<double> glued_labels(static_cast<std::size_t>(Vg));
    for (std::int64_t i = 0; i < Vg; ++i) glued_labels[static_cast<std::size_t>(i)] = i < m ? static_cast<double>(H.labels[static_cast<std::size_t>(kept[static_cast<std::size_t>(i)])]) : static_cast<double>(bcode);
    Arrays out;
    // the producer's `glue` marks a glued surface and STATES THE RELATION (input.h glue_relation): the original manifold, the hemispheres labelling of the glued cortex (the output `labels`, its stored
    // name filled in by the runner), the medial wall's selection on the original, the counts and the rule; results on it are restricted back through the labelling and the original's wall
    out["glued"] = made(Vg, 3, G.positions, Json{{"faces", G.faces}, {"edges_count", G.edges}, {"closed", true}, {"cs", ctx.bound.coordinate_system_key(S.id)},
                        {"description", S.name + " glued by anatomy: exactly its medial walls cut (the Desikan-Killiany atlas), a bridge grown between the rims; vertex i < " + std::to_string(m) + " is the i-th vertex of " + S.name + " off the wall, the rest the bridge"},
                        {"producer", Json{{"glued_from", S.name}, {"mesh", mesh_key(G.faces)}, {"gluing", glue_record},
                                          {"glue", glue_relation(S.id, S.name, Json{{"output", "labels"}}, ctx.bound.medial_wall_record(S), m, Vg)}}}});
    out["labels"] = made(Vg, 1, std::move(glued_labels), Json{{"lookup", lookup}, {"glue_map", true},
                              {"description", "the hemispheres of the glued cortex: " + H.name + " of " + S.name + " carried over (its entries kept), and the bridge (not cortex: no sources, a forward zero there)"}});
    return out;
  };
  return s;
}

}  // namespace

void add_eigen_steps(std::vector<StepSpec>& r) { r.push_back(intrinsic_step()); r.push_back(hodge_dirac_step()); r.push_back(surface_glue_step()); }

}  // namespace nxr::pipelines::graph
