// nxr-pipelines — THE DYADIC TILES AS REGISTERED STEPS: the tile GRID of a surface (compute's dyadic builder), the group tree (the same grid on the default subject, with the atlas definitions and
// the Levi-Civita connection), and the group tree carried onto a subject (the nearest vertex on the registered spheres). Each step makes ONE grid and DESCRIBES it (meta.describe, the tile code of
// every vertex per level as its data); the definition (tiles.yaml, group_tree.yaml, group_tiles.yaml) says where it is stored, and io's save_grid writes it. A step reads no file and no row: what
// it needs beside its inputs (the surface, the stored frames, the template) is handed over by the runner (Bound).
//
// THE GRAPHS (docs/specs/2026-11-17-tile-grids.md has the Studio's side by side):
//   tiles         surface ─▶ tile_grid ─▶ tiles   (grid, `<surface> tiles`)
//   group_tree    surface, frames? ─▶ tile_grid {tree: group} ─▶ tiles   (grid, `<surface> tiles` with tree group; the Levi-Civita connection on its levels when frames are stored)
//   group_tiles   surface ─▶ group_tile_grid ─▶ tiles   (grid, `<surface>_group_tiles`: the default subject's codes through the registered spheres)
//
//   tile_grid        setup   surface -> tiles    nxr::selection::dyadic: equal-area bisection, 2^l tiles a piece, the member number = the finest tile code h 2^D + t, level l = code >> (D - l)
//
// These two steps are the RULES of the maker `select.dyadic_surface` (select_nodes.cpp: rule subject / group / transfer), no longer registered under their own names. The finest tile is a
// QUANTITY (`finest`: a tile diameter, area, wavelength or parts per hemisphere), resolved to the depth through the grid's MEASURED level diameters (spec 3.3), never a bare depth. The
// Levi-Civita connection between neighbouring tiles is not part of the grid: it is a measurement over the product of a level with itself (`reduce`, measure levi_civita; rule 26).
//   group_tile_grid  setup   surface          -> tiles    nxr::ladder::groupTreeFor: each vertex takes the code of the nearest vertex of the default subject's cortex, piece by piece
//
// The Studio's design is kept with its names (backend/src/db/surface-ladder.ts, studio/src/engine/ladder.ts: `surfaceLadder`, the levels, `tileAreas`, `tileCentres`, Weyl's law).
// What they cannot resolve: a tile is as equal in area as the cut across its geodesic diameter allows (the level's area spread is recorded in the Studio, not here); the Levi-Civita connection between
// distant tiles depends on the path (its record says so); the group tree puts a vertex where the registered sphere puts it (the match's median and maximum chord are recorded).
#include <algorithm>
#include <cmath>

#include "nxr/chart.h"
#include "nxr/frames.h"
#include "nxr/ladder.h"
#include "nxr/measurement/reduce.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/select_nodes.h"
#include "nxr/quantity.h"
#include "nxr/selection/grid.h"
#include "nxr/selection/partition.h"

namespace nxr::pipelines::graph {

namespace lad = nxr::ladder;

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }

Array filled(const std::vector<std::vector<std::int32_t>>& per_level, std::int64_t n, const Json& describe) {
  Array a;
  a.rows = static_cast<std::int64_t>(per_level.size());
  a.cols = n;
  a.data.reserve(static_cast<std::size_t>(a.rows * a.cols));
  for (const auto& l : per_level) for (const std::int32_t c : l) a.data.push_back(c);
  a.meta["describe"] = describe;
  return a;
}
Json array_f64(const std::string& suffix, const std::vector<std::int64_t>& shape, const std::vector<double>& v) {
  return Json{{"suffix", suffix}, {"dtype", "f64"}, {"shape", shape}, {"values", v}};
}
Json array_i32(const std::string& suffix, const std::vector<std::int64_t>& shape, const std::vector<std::int32_t>& v) {
  return Json{{"suffix", suffix}, {"dtype", "i32"}, {"shape", shape}, {"values", v}};
}
// a level's tile of every vertex: the finest code shifted (the closed form of the hierarchy), -1 where no tile
std::vector<std::int32_t> level_of(const std::vector<std::int32_t>& finest_codes, int depth, int level) {
  std::vector<std::int32_t> out(finest_codes.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = finest_codes[i] >= 0 ? (finest_codes[i] >> (depth - level)) : -1;
  return out;
}
std::string tiles_name(const std::string& surface, int l) { return surface + " tiles L" + std::to_string(l); }

// ── tile_grid ───────────────────────────────────────────────────────────────────────────────────────────────────
struct TileGridState {
  std::string surface, tree;
  Json finest;                                  // the finest tile as a quantity (null: the finest every piece reaches; the group grid names it)
  std::string resolution = "fsaverage5";        // rule group: recorded in the atlas definitions
  Json atlas_finest;                            // rule group: the finest tile of the dataset's atlases as a quantity (null: the grid's finest)
  double frames_per_cycle = 0;                  // rule group: the default frame density recorded in the atlas definitions (PI rule 34: an option, written by the definition)
};
// THE ATLAS DEFINITIONS of the group tree (D131): what the dataset's atlases and group analyses default to. Every scale is a quantity resolved in closed form (the atlas's leaf level is `atlas_depth`, the SNAPPED level of
// `atlas_finest`); the frame density is frames per cycle of a band's centre (36: the 10 degree step of phase work), no tower and no day-anchored frame level.
Json group_tree_definitions(const std::string& resolution, int atlas_depth, const Json& atlas_snapped, double frames_per_cycle) {
  Json bands = Json::array();
  for (int b = 0; b < 6; ++b) bands.push_back(Json::array({1.0 * (1 << b), 2.0 * (1 << b)}));   // nxr's DEFAULT_BANDS: octaves 1-2 ... 32-64 Hz
  return Json{{"default", true}, {"tree", "group"}, {"depth", atlas_depth}, {"atlas_finest", atlas_snapped}, {"resolution", resolution}, {"frames_per_cycle", frames_per_cycle}, {"bands_hz", bands}};
}
// "36 frames/cycle" (or a bare 36) as the number of frames to a cycle
double frames_per_cycle_of(const Json& j) {
  need(j.is_string() || j.is_number(), "tiles: frame_rate is a frame density (\"36 frames/cycle\")");
  if (j.is_number()) return j.get<double>();
  const std::string t = j.get<std::string>();
  std::size_t used = 0;
  double v = 0;
  try { v = std::stod(t, &used); } catch (const std::exception&) { refuse("tiles: frame_rate " + j.dump() + ": a number of frames to a cycle (\"36 frames/cycle\")"); }
  const std::string rest = t.substr(used);
  need(rest.find("frames/cycle") != std::string::npos && v >= 1, "tiles: frame_rate " + j.dump() + ": a number of frames to a cycle, at least 1 (\"36 frames/cycle\")");
  return v;
}

// A tile's area, per node: the `sum` of the vertex areas over the partition's members, by compute's one reduction (the weights of a member with no values).
std::vector<double> tile_areas(const std::vector<std::int32_t>& node, int nodes, const std::vector<double>& area) {
  nxr::selection::MemberOfElement members;
  members.member_of = node;
  members.members = nodes;
  nxr::measurement::Ingredients acc(1, nodes);
  nxr::measurement::Request request;
  request.weighting = nxr::measurement::Weighting::given;
  nxr::measurement::reduce(acc, static_cast<const double*>(nullptr), 1, static_cast<std::int64_t>(node.size()), members, area.data(), request);
  return std::vector<double>(acc.weight.begin(), acc.weight.begin() + nodes);
}

Json level_facts(const nxr::selection::Selection& grid, int l, std::int64_t tiles) {
  const auto& level = grid.members.at(static_cast<std::size_t>(l));
  const auto& band = *level.band;
  Json j{{"level", l}, {"tiles", tiles}, {"lambda_min", band.first}, {"lambda_max", band.second}, {"wavelength_m", lad::wavelengthOf(band.second)}, {"tau", 1 / band.second}, {"modes_per_piece", 1 << l}};
  if (level.diameter) j["diameter"] = *level.diameter;                      // the MEASURED mean tile diameter (the surface's unit): the level dictionary of jointcell's rule
  return j;
}

StepSpec tile_grid_step() {
  StepSpec s;
  s.name = "tile_grid";
  s.kind = "setup";
  s.inputs = {"surface"};
  s.outputs = {"tiles"};
  s.parameters = Json{{"finest", nullptr}, {"tree", "subject"}, {"resolution", "fsaverage5"}, {"atlas_finest", nullptr}, {"frame_rate", nullptr}};
  s.matches = "the Cortical Flow Studio engine/ladder.ts (surfaceLadder, Weyl pairing); nxr geom.ladder / geom.trees.subject_tree; Rheome +geom/surfaceladder.m";
  s.description =
      "The dyadic equal-area tiles of a surface as ONE grid with levels: each tile is halved by equal mass across its geodesic diameter, every tile one piece; level l has 2^l tiles a piece, paired with the "
      "band it adds by Weyl's law. The member number is the finest tile code h 2^D + t (h the connected piece), level l's member that code >> (D - l): the hierarchy is a rule, nothing per level is held. "
      "finest: the finest tile as a QUANTITY (a tile diameter in mm, an area in mm2, a wavelength, or parts per hemisphere), resolved to the depth through the grid's measured level diameters "
      "(default for the subject's own grid: the finest every piece reaches). tree group (the default subject's only): the grid also carries the dataset's atlas definitions (resolution, atlas_finest, frame_rate; "
      "its leaf depth is the finest, which the group grid must name).";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface: positions as stored, double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("surface") != 0, "tile_grid: the input `surface` is missing");
    const std::string tree = p.value("tree", std::string("subject"));
    need(tree == "subject" || tree == "group", "tiles: tree must be subject or group, not " + tree);
    need(p["finest"].is_null() || p["finest"].is_string(), "tiles: finest is a quantity with a unit (e.g. \"20 mm\", \"parts 16\"), never a bare level");
    need(tree != "group" || p["finest"].is_string(), "tiles: the group tree names its finest tile (finest: a quantity); it is the leaf depth of every subject's atlas");
    Value v;
    v.cls = "selection";
    v.manifolds = ins.at("surface").manifolds;
    return Values{{"tiles", v}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    TileGridState st;
    st.surface = nxr::io::text(*ins.at("surface").row, "name");
    st.tree = p.value("tree", std::string("subject"));
    st.finest = p["finest"];
    if (st.tree == "group") {
      need(ctx.bound.subject_kind() == "template", "the group tree is the default subject's (a template), not a " + ctx.bound.subject_kind() + "'s");
      st.resolution = p.value("resolution", std::string("fsaverage5"));
      st.atlas_finest = p["atlas_finest"].is_null() ? p["finest"] : p["atlas_finest"];
      need(p["frame_rate"].is_string() || p["frame_rate"].is_number(), "tiles: the group tree states its default frame density (frame_rate: \"36 frames/cycle\"); the definition writes it, the node has none");
      st.frames_per_cycle = frames_per_cycle_of(p["frame_rate"]);
    }
    return std::any(std::move(st));
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const TileGridState& st = std::any_cast<const TileGridState&>(state);
    const SurfaceInput S = ctx.bound.surface(st.surface, true);               // float64 as stored: the Studio's stored ladder and nxr read it so
    const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
    std::string unit = ctx.bound.coordinate_unit(S.id);
    if (unit.empty()) unit = "m";                                             // a surface of the lab's datasets is stored in metres
    nxr::selection::Surface surf;
    surf.unit = unit; surf.positions = S.vertices.data(); surf.vertices = nV; surf.faces = S.faces.data(); surf.face_count = nF;
    nxr::selection::Dyadic method;
    nxr::selection::Selection grid;
    try { grid = nxr::selection::dyadic(surf, method); } catch (const std::invalid_argument& e) { refuse(std::string("tiles: ") + e.what()); }
    Json snapped = Json::object();
    if (!st.finest.is_null()) {                                            // the stated finest tile, resolved on the grid's MEASURED diameters; the grid is rebuilt to that depth
      const int want = level_of_quantity(grid, st.finest, unit, "tiles", snapped);
      need(want >= 0 && want <= grid.grid->depth, "tiles: finest " + st.finest.get<std::string>() + " is level " + std::to_string(want) + "; the surface's grid reaches level " + std::to_string(grid.grid->depth));
      if (want < grid.grid->depth) { method.depth = want; try { grid = nxr::selection::dyadic(surf, method); } catch (const std::invalid_argument& e) { refuse(std::string("tiles: ") + e.what()); } }
    }
    const int D = grid.grid->depth;
    const int finest = grid.grid->finest >= 0 ? grid.grid->finest : D;      // the deepest level every piece reaches (compute's `dyadic` returns it)
    const int slots = static_cast<int>(grid.grid->slots);
    const auto runs = lad::pieceRuns(S.faces.data(), nF, nV);
    const std::int64_t pieces = static_cast<std::int64_t>(runs.size());
    lad::Mesh whole;
    whole.n = nV; whole.positions = S.vertices; whole.faces = S.faces;
    const std::vector<double> area = lad::vertexAreas(whole);
    double total = 0;
    for (const double a : area) total += a;
    if (ctx.say) ctx.say(Json{{"stage", "ladder"}, {"surface", st.surface}, {"pieces", pieces}, {"slots", slots}, {"depth", D}});

    // the dictionary of the levels: one entry per level, code = the level, measurement = the level's wavelength
    Json ladder_entries = Json::array();
    for (int l = 0; l <= D; ++l) {
      const std::int64_t tiles = static_cast<std::int64_t>(slots) << l;
      const Json lj = level_facts(grid, l, tiles);
      ladder_entries.push_back(Json{{"code", l}, {"name", l == 0 ? std::string("the whole surface") : std::to_string(tiles) + " tiles"}, {"measurement", lj["wavelength_m"]}, {"measurement_unit", "m"}, {"attributes", lj}});
    }
    Json atlas;
    if (st.tree == "group") {
      Json atlas_snapped = Json::object();
      const int atlas_depth = st.atlas_finest.is_string() ? level_of_quantity(grid, st.atlas_finest, unit, "tiles", atlas_snapped) : D;   // the atlas's leaf level IS the stated finest tile, resolved
      need(atlas_depth >= 0 && atlas_depth <= D, "tiles: atlas_finest is level " + std::to_string(atlas_depth) + "; the grid reaches level " + std::to_string(D));
      atlas = group_tree_definitions(st.resolution, atlas_depth, atlas_snapped, st.frames_per_cycle);
    }

    Json levels = Json::array();
    std::vector<std::vector<std::int32_t>> per_level;
    for (int l = 0; l <= D; ++l) {
      const std::vector<std::int32_t> node = level_of(grid.member_of, D, l);
      const int nodes = slots << l;
      const std::vector<double> w = tile_areas(node, nodes, area);
      const std::vector<int> cen = lad::tileCentres(S.vertices.data(), node, nodes, area);       // the vertex nearest each tile's area-weighted centroid (-1 empty)
      const std::vector<std::int32_t> cen32(cen.begin(), cen.end());
      const auto adj = lad::tileAdjacency(S.faces.data(), nF, node);
      std::vector<std::int32_t> adj32;
      for (const auto& e : adj) { adj32.push_back(e.first); adj32.push_back(e.second); }
      Json lj = level_facts(grid, l, nodes);
      {                                                                     // the level attributes the grid carries (spec 1.1): the mean tile area, and the measured mean tile diameter in mm
        double sum_area = 0; int filled_tiles = 0;
        for (const double a : w) if (a > 0) { sum_area += a; ++filled_tiles; }
        lj["area_m2"] = filled_tiles ? sum_area / filled_tiles : 0.0;
        if (lj.contains("diameter")) lj["diameter_mm"] = lj["diameter"].get<double>() * (unit == "m" ? 1e3 : unit == "cm" ? 1e1 : 1.0);
        lj["weyl_band_per_m2"] = Json::array({lj["lambda_min"], lj["lambda_max"]});
      }
      lj["tree"] = st.tree; lj["depth"] = D; lj["finest"] = finest; lj["summaries"] = Json::array();
      // the tiles' names, as the Studio writes them (entries up to 1024 tiles)
      Json entries = Json::array();
      if (nodes <= 1024) {
        const int per = 1 << l;
        for (int c = 0; c < nodes; ++c)
          entries.push_back(Json{{"code", c}, {"name", (pieces > 1 ? "piece " + std::to_string(c / per) + " tile " : std::string("tile ")) + std::to_string(c % per)}, {"measurement", w[static_cast<std::size_t>(c)]},
                                 {"measurement_unit", "m2"}, {"attributes", Json{{"level", l}, {"tile", c % per}, {"piece", c / per}}}});
      }
      Json lv = Json{{"name", tiles_name(st.surface, l)}, {"tiles", nodes}, {"params", lj}, {"label", l == 0 ? std::string("the whole surface") : std::to_string(nodes) + " tiles"},
                     {"description", "level " + std::to_string(l) + " of the dyadic tiling: " + std::to_string(nodes) + " tiles, tile code h 2^" + std::to_string(l) + " + t (h the connected piece, a hemisphere); the tile code is the member number"},
                     {"dictionary", Json{{"name", tiles_name(st.surface, l)}, {"description", "the tiles of level " + std::to_string(l) + " of the dyadic tiling, every connected piece"}, {"entries", entries}}},
                     {"arrays", Json::array({array_f64("w", {nodes}, w), array_i32("lc/L" + std::to_string(l) + "/centre", {nodes}, cen32),
                                             array_i32("lc/L" + std::to_string(l) + "/adjacency", {static_cast<std::int64_t>(adj.size()), 2}, adj32)})}};
      levels.push_back(std::move(lv));
      per_level.push_back(node);
    }

    // the ladder's record: the rule (`grid`: the hierarchy is the closed form, `slots` the piece slots the codes reserve, `pieces` the connected pieces), the area, the atlas definitions
    Json params = Json{{"weyl", "4pi m/A"}, {"tree", st.tree}, {"area_m2", total}, {"length_unit", unit}, {"pieces", pieces}, {"depth", D}, {"finest", finest},
                       {"grid", Json{{"kind", "dyadic"}, {"depth", D}, {"slots", slots}, {"pieces", pieces}, {"finest", finest}}}};
    if (!atlas.is_null()) params["atlas"] = atlas;
    if (!st.finest.is_null()) params["finest_quantity"] = Json{{"stated", st.finest}, {"snapped", snapped}};   // what was asked, and what the level actually is (spec 3.1, item 3)
    Json describe = Json{{"label", "dyadic tiles, " + std::to_string(D + 1) + " levels, 1 ... " + std::to_string(static_cast<std::int64_t>(slots) << D) + " tiles"}, {"unit", "m"},
                         {"description", "the dyadic tiling of the surface (Cortical Flow Studio engine/ladder.ts, nxr geom.ladder / geom.trees): each tile halved by equal mass across its geodesic diameter, every tile one piece; "
                                         "level l has 2^l tiles per connected piece, paired with scale by Weyl's law"},
                         {"params", params},
                         {"dictionary", Json{{"name", st.surface + " tiles"}, {"description", "the levels of the dyadic tiling '" + st.surface + " tiles': code = the level, 2^level equal-area tiles per connected piece paired by Weyl's law (N(lambda) = A lambda / 4 pi) with the band their count fills"}, {"entries", ladder_entries}}},
                         {"levels", levels}};
    Arrays out;
    out["tiles"] = filled(per_level, nV, describe);
    return out;
  };
  return s;
}

// ── group_tile_grid ─────────────────────────────────────────────────────────────────────────────────────────────
struct GroupGridState { std::string surface; };

StepSpec group_tile_grid_step() {
  StepSpec s;
  s.name = "group_tile_grid";
  s.kind = "setup";
  s.inputs = {"surface"};
  s.outputs = {"tiles"};
  s.parameters = Json::object();
  s.matches = "nxr geom.trees Template.tree_for (D127/D128): the subject's vertices on the group tree through the registered sphere";
  s.description =
      "A subject's vertices on the dataset's GROUP tree: each vertex takes the code of the nearest vertex of the default subject's primary cortex on the registered sphere (`<surface>_sphere` of both), "
      "hemisphere (piece) by piece, in the default subject's group ladder: the leaves of the group tree at depth D are the SAME tiles in every subject, and what is measured on them adds cell for cell "
      "across subjects. The grid keeps the template's codes (a tile no vertex reaches is an empty member) and its level dictionaries; level l is the finest code >> (finest - l).";
  s.traits = node_traits(TimeBehaviour::pointwise, Precision::float64, OutputPrecision::float64, "runs once on the surface and the registered spheres: double");
  s.plan = [](const Values& ins, const Json&, const PlanContext&) {
    need(ins.count("surface") != 0, "group_tile_grid: the input `surface` is missing");
    Value v;
    v.cls = "selection";
    v.manifolds = ins.at("surface").manifolds;
    return Values{{"tiles", v}};
  };
  s.prepare = [](const Values& ins, const Json&, ExecContext& ctx) {
    need(ctx.bound.subject_kind() != "template", "the default subject's group tree is its own ladder (`run tiles`, made by `dataset create`)");
    GroupGridState st;
    st.surface = nxr::io::text(*ins.at("surface").row, "name");
    return std::any(std::move(st));
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const GroupGridState& st = std::any_cast<const GroupGridState&>(state);
    const SurfaceInput S = ctx.bound.surface(st.surface, true);
    const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
    const GroupTemplate T = ctx.bound.group_template(st.surface);
    std::string unit = ctx.bound.coordinate_unit(S.id);
    if (unit.empty()) unit = "m";
    if (ctx.say) ctx.say(Json{{"stage", "group tiles"}, {"surface", st.surface}, {"template", T.subject}, {"finest", T.finest}});
    // the template's grid (its codes at its finest level) carried through the registered spheres: compute's `transfer`, which also gives the sphere match's median and maximum chord
    const nxr::selection::Selection from = nxr::selection::dyadic_grid(T.codes, T.finest, T.slots);
    nxr::selection::Surface surf;
    surf.unit = unit; surf.positions = S.vertices.data(); surf.vertices = nV; surf.faces = S.faces.data(); surf.face_count = nF;
    nxr::selection::Transfer method;
    method.from = &from; method.from_sphere = T.sphere.data(); method.from_faces = T.faces.data();
    method.from_vertices = static_cast<std::int32_t>(T.n_vertices); method.from_face_count = static_cast<std::int32_t>(T.faces.size() / 3);
    method.sphere = T.subject_sphere.data();
    nxr::selection::Selection grid;
    try { grid = nxr::selection::transfer(surf, method); } catch (const std::invalid_argument& e) { refuse(st.surface + ": " + e.what()); }
    const double med = *grid.grid->match_median, mx = *grid.grid->match_max;
    const std::int64_t pieces = static_cast<std::int64_t>(lad::pieceRuns(S.faces.data(), nF, nV).size());
    lad::Tree tree;
    tree.codes = grid.member_of;
    lad::Mesh mesh;
    mesh.n = nV; mesh.positions = S.vertices; mesh.faces = S.faces;
    const std::vector<double> area = lad::vertexAreas(mesh);
    const int finest = T.finest;
    Json levels = Json::array();
    std::vector<std::vector<std::int32_t>> per_level;
    for (int l = 0; l <= finest; ++l) {
      const std::vector<std::int32_t> node = level_of(tree.codes, finest, l);
      const std::int64_t nodes = static_cast<std::int64_t>(T.slots) << l;
      const std::vector<double> w = tile_areas(node, static_cast<int>(nodes), area);
      levels.push_back(Json{{"name", st.surface + " group tiles L" + std::to_string(l)}, {"tiles", nodes}, {"dictionary_id", T.level_dictionary_ids.at(static_cast<std::size_t>(l))},
                            {"params", Json{{"tree", "group"}, {"level", l}, {"tiles", nodes}, {"finest", finest}}},
                            {"description", "level " + std::to_string(l) + " of the GROUP tree: the default subject's tile of the nearest vertex on the registered sphere, tile code h 2^" + std::to_string(l) + " + t"},
                            {"arrays", Json::array({array_f64("w", {nodes}, w)})}});
      per_level.push_back(node);
    }
    Json describe = Json{{"description", "this subject's vertices on the dataset's GROUP tree (the default subject's tiles, through the registered sphere); member L<k> codes h 2^k + t"}, {"unit", "m"},
                         {"dictionary_id", T.ladder_dictionary_id},
                         {"params", Json{{"tree", "group"}, {"template", T.subject}, {"finest", finest}, {"sphere_match_median", med}, {"sphere_match_max", mx},
                                         {"grid", Json{{"kind", "dyadic"}, {"depth", finest}, {"slots", T.slots}, {"pieces", pieces}, {"finest", finest}}}}},   // the hierarchy is a rule (closed form), never parent arrays
                         {"levels", levels}};
    Arrays out;
    out["tiles"] = filled(per_level, nV, describe);
    return out;
  };
  return s;
}

}  // namespace

void add_tile_steps(std::vector<StepSpec>& r) {
  r.push_back(tile_grid_step());
  r.push_back(group_tile_grid_step());
}

}  // namespace nxr::pipelines::graph
