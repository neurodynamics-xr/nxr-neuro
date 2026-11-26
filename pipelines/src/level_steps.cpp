// nxr-pipelines — THE LEVEL COMPLEX of a surface's dyadic grid as a registered step (spec docs/specs/2026-11-26-levels-of-one-manifold.md, section 7):
//
//   select.level_complex   setup   surface (vertices, faces) + its dyadic grid at the level a stated scale sets --> the level's navigation operators: d0 [E x cells], star1 [E x E] (diagonal), the
//                                  crossing map [fine edges x E], d1 [F x E] (the junctions), as SPARSE operators keyed by (the surface, the grid, the level, the mesh, the metric)
//
// The level is not a manifold row: its operators are stored on the surface (from_selection = the grid, the level in params), found next time by their key. ⋆0 is the grid's own `w` (the
// tiles' barycentric areas) and is not stored again. Compute: nxr::select::levelComplex on the fine DEC (nxr::operators::assembleDECOperators on the operator mesh, embeddedDEC on the
// input) and the barycentric vertex areas (nxr::select::bisection::vertexAreas). The node reads no recording and no database: the surface and the level's tiles come from the binding.
#include <cmath>
#include <string>
#include <vector>

#include "nxr/geometry/surface_arrays.h"
#include "nxr/operators/dec.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/select_nodes.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/select/dyadic.h"
#include "nxr/select/levels.h"

namespace nxr::pipelines::graph {

namespace {
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
struct LevelState { std::string surface, mesh, grid_id, grid_name; int level = 0; Json scale; };

// a sparse operator as the declared operator output stores it: the nnz values in the array, the CSR structure in its description
Array sparse_out(const nxr::select::SparseMatrix& m, const Json& describe) {
  const Eigen::SparseMatrix<double, Eigen::RowMajor> r(m);
  Array a;
  a.rows = static_cast<std::int64_t>(r.nonZeros()); a.cols = 1;
  Json indptr = Json::array(), indices = Json::array();
  for (Eigen::Index i = 0; i <= r.rows(); ++i) indptr.push_back(static_cast<std::int64_t>(r.outerIndexPtr()[i]));
  for (Eigen::Index k = 0; k < r.nonZeros(); ++k) { indices.push_back(r.innerIndexPtr()[k]); a.data.push_back(r.valuePtr()[k]); }
  Json d = describe;
  d["sparse"] = Json{{"layout", "csr"}, {"rows", static_cast<std::int64_t>(r.rows())}, {"cols", static_cast<std::int64_t>(r.cols())}, {"indptr", indptr}, {"indices", indices}};
  a.meta["describe"] = d;
  return a;
}
const char* const kOutputs[] = {"d0", "star1", "crossing", "d1"};
}  // namespace

void add_level_steps(std::vector<StepSpec>& r) {
  StepSpec s;
  s.name = "select.level_complex";
  s.kind = "setup";
  s.inputs = {"surface", "tiles"};
  for (const char* o : kOutputs) s.outputs.push_back(o);
  s.parameters = Json{{"spatial_scale", nullptr}, {"mesh", "operator"}};
  s.matches = "nxr::select::levelComplex (the quotient of the fine complex by the grid's tiles at a level, Galerkin metric; spec docs/specs/2026-11-26-levels-of-one-manifold.md)";
  s.description = "the navigation operators of one level of a surface's dyadic grid, a level that is no manifold of its own: d0 of its edges (the pairs of tiles a fine edge joins, tail < tip), its edge "
                  "conductances star1 (the summed fine cotan weights across each boundary: the level's Laplacian is the projected fine one), the crossing map of the fine edges onto its edges, and d1 of "
                  "its junction faces; the record says whether the level is a valid surface complex (halfedges) and why not";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface and its grid: no time axis, double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext& c) {
    need(ins.count("surface") != 0 && ins.at("surface").row, "level_complex: the input `surface` is missing");
    const Value& t = ins.at("tiles");
    need(t.cls == "selection" && t.row && nxr::io::text(*t.row, "type") == "ladder", "level_complex: tiles must be a surface's dyadic grid (`run select_dyadic_surface` first)");
    need(p["spatial_scale"].is_string(), "level_complex: spatial_scale is a quantity (\"parts 16\", a tile diameter \"20 mm\", an area \"area 400 mm2\") naming the level; a bare level index is never a parameter");
    need(p["mesh"] == "operator" || p["mesh"] == "input", "level_complex: mesh is operator (the intrinsic Delaunay triangulation, the fine operators' mesh) or input (the stored triangulation)");
    const int level = level_of_ladder(c.bound, t, p["spatial_scale"], "level_complex");
    const std::string sid = ins.at("surface").manifolds.empty() ? std::string() : ins.at("surface").manifolds[0];
    Values out;
    for (const char* o : kOutputs) { Value v; v.cls = "operator"; v.manifolds = {sid, sid}; v.facts["names"] = Json{{"level", std::to_string(level)}}; out[o] = std::move(v); }   // `{level}` in the outputs' names
    return out;
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    LevelState st;
    const Value& t = ins.at("tiles");
    st.surface = nxr::io::text(*ins.at("surface").row, "name");
    st.mesh = p["mesh"].get<std::string>();
    st.grid_id = nxr::io::text(*t.row, "id"); st.grid_name = nxr::io::text(*t.row, "name");
    Json snapped;
    st.level = level_of_ladder(c.bound, t, p["spatial_scale"], "level_complex", &snapped);
    st.scale = Json{{"stated", p["spatial_scale"]}, {"level", st.level}, {"snapped", snapped}};
    return std::any(std::move(st));
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const LevelState& st = std::any_cast<const LevelState&>(state);
    const SurfaceInput S = ctx.bound.surface(st.surface, true);
    const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
    const SurfaceLevel SL = ctx.bound.surface_level(S, st.level);              // the level's tile of every vertex (the grid's codes >> (depth - level)) and its tiles' areas
    const auto arrays = nxr::geometry::surfaceArrays(S.vertices.data(), nV, S.faces.data(), nF, nxr::geometry::intrinsicOptions(st.mesh == "operator"));
    const nxr::operators::DECOperators dec = st.mesh == "operator" ? nxr::operators::assembleDECOperators(arrays) : nxr::operators::embeddedDEC(arrays);
    nxr::select::bisection::Mesh mesh; mesh.n = nV; mesh.positions = S.vertices; mesh.faces = S.faces;
    const std::vector<double> area = nxr::select::bisection::vertexAreas(mesh);
    nxr::select::FineComplex fine;
    fine.d0 = dec.d0; fine.d1 = dec.d1; fine.star1 = dec.hodge1.diagonal();
    fine.star0 = Eigen::Map<const Eigen::VectorXd>(area.data(), nV);
    nxr::select::Level L;
    try { L = nxr::select::levelComplex(fine, SL.tile, SL.nodes); } catch (const std::invalid_argument& e) { refuse(std::string("level_complex: ") + e.what()); }
    const Json params = Json{{"grid", st.grid_name}, {"grid_id", st.grid_id}, {"level", st.level}, {"mesh", st.mesh}, {"metric", "galerkin"}};
    const Json producer = Json{{"surface", S.name}, {"grid", st.grid_name}, {"scale", st.scale}, {"level", st.level}, {"mesh", st.mesh}, {"metric", "galerkin"},
                               {"cells", L.cells}, {"edges", static_cast<std::int64_t>(L.edges.rows())}, {"faces", static_cast<std::int64_t>(L.faces.rows())},
                               {"fine_edges", static_cast<std::int64_t>(fine.d0.rows())}, {"negative_conductances", L.negativeConductances},
                               {"valid_complex", L.valid}, {"why", L.why}, {"euler", L.euler}, {"fine_euler", L.fineEuler},
                               {"star0", "the grid's own `w` at this level (the tiles' barycentric areas), not stored again"}};
    if (ctx.say) ctx.say(Json{{"stage", "level complex"}, {"surface", S.name}, {"level", st.level}, {"cells", L.cells}, {"edges", L.edges.rows()}, {"faces", L.faces.rows()}, {"valid", L.valid}});
    const std::string at = "level " + std::to_string(st.level) + " of " + st.grid_name;
    auto describe = [&](const char* from_cell, const char* to_cell, const std::string& what) {
      return Json{{"from_cell", from_cell}, {"to_cell", to_cell}, {"params", params}, {"producer", producer}, {"description", what}};
    };
    Arrays out;
    out["d0"] = sparse_out(L.d0, describe("0", "1", "d0 of " + at + ": its cells -> its edges (-1 at an edge's tail, +1 at its tip; edges the pairs of tiles a fine edge joins, tail < tip)"));
    nxr::select::SparseMatrix star1(L.star1.size(), L.star1.size());
    { std::vector<Eigen::Triplet<double>> t; for (Eigen::Index e = 0; e < L.star1.size(); ++e) t.emplace_back(static_cast<int>(e), static_cast<int>(e), L.star1(e)); star1.setFromTriplets(t.begin(), t.end()); }
    out["star1"] = sparse_out(star1, describe("1", "1*", "the Hodge star *1 of " + at + ": each edge's Galerkin conductance, the summed fine cotan weights of the fine edges crossing its boundary (diagonal)"));
    out["crossing"] = sparse_out(L.crossing, describe("1", "1", "the crossing map of " + at + ": the surface's fine edges -> the level's edges (+1 along, -1 against; 0 inside a tile): d0_fine P = crossing d0"));
    out["d1"] = sparse_out(L.d1, describe("1", "2", "d1 of " + at + ": its edges -> its junction faces (a fine face whose corners lie in three tiles), each face's three edges with +1 along its winding"));
    return out;
  };
  r.push_back(std::move(s));
}

}  // namespace nxr::pipelines::graph
