// nxr-pipelines — the registered step of the WHERE of a global burst (spec docs/specs/2026-11-13-cortical-events.md, phase 2b): `cell_extent` (the events of each burst on the joint cells), an `after` step on arrays;
// the events' measures are `reduce.members`, a merge of their cells' ingredients. The maths is nxr::select (compute/), the runner binds and stores.
#include <cmath>
#include <limits>
#include <memory>

#include "nxr/select/extent.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/spatial_band.h"

namespace nxr::pipelines::graph {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
Value made(const Value& x) { Value v = x; v.row.reset(); return v; }

// THE PRODUCT CELLS (a selection on the product of a surface grid's level and a time grid; members c = s + S t, the spatial factor fastest) as `reduce` recorded them in the cells' meta (`over`, `radices`):
// the surface and the level the spatial factor names, and the geometry the extent needs: tile areas (the cells' weights: barycentric vertex areas summed by tile)
struct Geometry {
  Json over;
  nxr::pipelines::SurfaceInput surface;
  nxr::pipelines::SurfaceLevel level;
  std::vector<double> vertex_area;
  Eigen::VectorXd tile_area;
  nxr::select::Pairs pairs;
  std::int64_t S = 0, T = 0;                 // the spatial tiles and the time tiles
  double width_s = 0;                        // a time tile's width in seconds
};
Geometry load_geometry(ExecContext& c, const Json& cells_meta) {
  Geometry g;
  g.over = cells_meta.value("over", Json::object());
  need(g.over.is_object() && g.over.value("kind", std::string()) == "product" && g.over.contains("factors") && g.over["factors"].size() == 2, "the cells are not a measurement over a product of a surface grid and a time grid");
  const Json &sp = g.over["factors"][0], &tm = g.over["factors"][1];
  need(sp.value("kind", std::string()) == "dyadic" && tm.value("kind", std::string()) == "time", "the cells' product must be (a surface grid's level) x (a time grid): its factors are " + sp.value("kind", std::string("?")) + " x " + tm.value("kind", std::string("?")));
  g.surface = c.bound.surface(sp["surface"].get<std::string>(), true);
  g.level = c.bound.surface_level(g.surface, sp["level"].get<int>());
  g.S = g.level.nodes;
  g.T = tm.value("count", std::int64_t(0));
  const double rate = cells_meta.value("rate", 0.0);
  need(g.T > 0 && rate > 0, "the cells' time grid records no tiles or the cells no rate");
  g.width_s = tm.value("width", std::int64_t(0)) / rate;
  need(cells_meta.contains("radices") && cells_meta["radices"].size() == 2 && cells_meta["radices"][0].get<std::int64_t>() == g.S && cells_meta["radices"][1].get<std::int64_t>() == g.T,
       "the cells' radices are not (spatial tiles " + std::to_string(g.S) + ", time tiles " + std::to_string(g.T) + ")");
  nxr::select::Mesh mesh; mesh.n = static_cast<int>(g.surface.n_vertices()); mesh.positions = g.surface.vertices; mesh.faces = g.surface.faces;
  g.vertex_area = nxr::select::vertexAreas(mesh);
  g.tile_area = Eigen::VectorXd::Zero(g.level.nodes);
  for (std::size_t v = 0; v < g.level.tile.size(); ++v) if (g.level.tile[v] >= 0) g.tile_area(g.level.tile[v]) += g.vertex_area[v];
  for (const auto& [a, b] : g.level.pairs) g.pairs.emplace_back(a, b);
  return g;
}
// the cells of one measure of the cells' values [1 x S T x C] as [S x T]
Eigen::MatrixXd cell_matrix(const Array& a, const Geometry& g, std::int64_t column) {
  const std::int64_t C = std::max<std::int64_t>(1, a.comps);
  need(a.cols == g.S * g.T, "the cells have " + std::to_string(a.cols) + " members, the product " + std::to_string(g.S) + " x " + std::to_string(g.T));
  Eigen::MatrixXd M(g.S, g.T);
  for (std::int64_t t = 0; t < g.T; ++t) for (std::int64_t s = 0; s < g.S; ++s) M(s, t) = a.data[static_cast<std::size_t>((s + g.S * t) * C + column)];
  return M;
}
std::int64_t column_of(const Array& a, const std::string& measure) {
  std::int64_t c = 0;
  for (const auto& m : a.meta.value("measures", Json::array())) {
    if (m.value("measure", std::string()) == measure) return c;
    c += m.value("components", 1);
  }
  refuse("the cells hold no measure " + measure);
  return -1;
}
nxr::select::Labels labels_of(const Array& a, const Geometry& g) {
  need(static_cast<std::int64_t>(a.data.size()) == g.S * g.T, "the labelling has " + std::to_string(a.data.size()) + " cells, the product " + std::to_string(g.S * g.T));
  nxr::select::Labels L(g.S, g.T);
  for (std::int64_t t = 0; t < g.T; ++t) for (std::int64_t s = 0; s < g.S; ++s) L(s, t) = static_cast<std::int32_t>(a.data[static_cast<std::size_t>(s + g.S * t)]);
  return L;
}
struct ExtentState { Json p; std::string groups; };
}  // namespace

void add_extent_steps(std::vector<StepSpec>& r) {
  r.push_back({"cell_extent", "after", {"x", "baseline", "events"}, {}, {"labels"}, Json{{"enter_sd", 1.5}, {"exit_sd", 1.0}, {"min_cells", 1}, {"min_area", 0.0}},
               "docs/specs/2026-11-13-cortical-events.md, Phase 2b; compute nxr::select (windowRegions on nxr::select::detectRegions, unchanged) with the tile graph of nxr::select::tileAdjacency",
               "the SPACE x TIME EXTENT of each event of a time span set (the global bursts): on the PRODUCT cells (a surface grid's level x a time grid) of the burst's own time tiles (and, per hemisphere, of its row's tiles), each cell's amplitude "
               "standardized against its spatial tile's baseline (the median and robust SD of a statistic of the cells across the run's time tiles: `summarize` with by_factor 0), hysteresis regions on the graph of touching tiles within a time "
               "tile and a tile with itself in the next; all regions are events, numbered by burst and then by their lowest cell. Out: a labelling of the product's cells (code = the event's ordinal, -1 none), each event a member described by its "
               "CENTRE and EXTENT (time centre and duration, the vertex nearest its area-weighted centroid, its tiles' union area) and the path of its centres",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value &x = ins.at("x"), &b = ins.at("baseline"), &ev = ins.at("events");
                 need(x.cls == "measurement" && x.facts.contains("over") && x.facts["over"].contains("selection_id") && !x.facts["over"]["selection_id"].is_null(),
                      "cell_extent: x must be a measurement over a stored product of a surface grid and a time grid (the cells `reduce` made over `select_product`'s product)");
                 bool has_amp = x.facts.value("measure", std::string()) == "amplitude";
                 for (const auto& m : x.facts.value("measures", Json::array())) has_amp = has_amp || m == "amplitude";
                 need(has_amp, "cell_extent: x must hold the `amplitude` of the cells");
                 const std::string pid = x.facts["over"]["selection_id"].get<std::string>();
                 need(b.cls == "statistic" && b.facts.contains("summaries") && b.facts.value("by_factor", Json(nullptr)).is_number_integer() && b.facts["by_factor"].get<int>() == 0 && b.facts.contains("over") &&
                      b.facts["over"].value("selection_id", std::string("?")) == pid && b.facts.value("measure", std::string()) == "amplitude",
                      "cell_extent: baseline must be the statistic of the same cells' amplitude summarised across the time tiles of each spatial tile (`summarize`, measure amplitude, by_factor 0): a median and a robust SD per spatial tile");
                 bool med = false, mad = false;
                 for (const auto& n : b.facts["summaries"]) { med = med || n == "median"; mad = mad || n == "mad"; }
                 need(med && mad, "cell_extent: the baseline must hold median and mad");
                 const auto prow = c.bound.selection(pid);
                 need(prow.has_value(), "cell_extent: the product is not among the subject's rows");
                 need(ev.cls == "spans" && ev.facts.contains("detect") && ev.facts.contains("over") && ev.facts.contains("rolled_to"),
                      "cell_extent: events must be the spans `detect_spans` found on the trace `roll_up` made from the same cells");
                 need(ev.facts["over"].value("selection_id", std::string("?")) == pid, "cell_extent: the events were found on " + ev.facts["over"].value("selection", std::string("another product")) + ", the cells' product is another: the bursts' time tiles are the cells'");
                 for (const char* k : {"enter_sd", "exit_sd", "min_area"}) need(p[k].is_number(), std::string("cell_extent: ") + k + " is a number");
                 need(p["min_cells"].is_number_integer() && p["min_cells"].get<int>() >= 1, "cell_extent: min_cells is a whole number, at least 1");
                 need(p["enter_sd"].get<double>() >= p["exit_sd"].get<double>(), "cell_extent: enter_sd is at least exit_sd (enter high, extend low)");
                 need(p["min_area"].get<double>() >= 0, "cell_extent: min_area is at least 0 (m2)");
                 Value y = made(x);
                 y.cls = "labelling";
                 y.manifolds = {nxr::io::text(*prow, "manifold_id")};
                 y.chunked = false;
                 y.unit.reset();
                 y.facts = Json{{"cells", x.facts["over"]}, {"extent", p}, {"groups", ev.facts["rolled_to"]}};
                 return Values{{"labels", y}};
               }});

  // ── execution ──
  auto at = [&](const char* n) -> StepSpec& { for (auto& s : r) if (s.name == n) return s; refuse(std::string("no step ") + n); };
  StepSpec& ce = at("cell_extent");
  ce.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64,
                          "grows hysteresis regions over the graph of touching tiles within a time tile and from each time tile to the next: the labelling is carried forward in time; a region's extent is not defined per timestep: stateful, confirmed", {});
  ce.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    ExtentState st;
    st.p = p;
    st.groups = ins.at("events").facts["rolled_to"].get<std::string>();
    return std::any(std::move(st));
  };
  ce.run = [](const std::any& st_, const Arrays& ins, ExecContext& c) {
    const ExtentState& st = std::any_cast<const ExtentState&>(st_);
    const Array &x = ins.at("x"), &base = ins.at("baseline"), &ev = ins.at("events");
    const Geometry g = load_geometry(c, x.meta);
    const int S = static_cast<int>(g.S), T = static_cast<int>(g.T), l = g.level.l;
    if (st.groups == "pieces" && (g.level.nodes >> l) != 2) refuse("cell_extent: pieces needs a surface of two pieces (the ladder's level has " + std::to_string(g.level.nodes >> l) + ")");
    need(base.rows == S, "cell_extent: the baseline has " + std::to_string(base.rows) + " rows, the cells " + std::to_string(S) + " spatial tiles");
    int im = -1, is = -1;
    for (std::size_t k = 0; k < base.meta["summaries"].size(); ++k) { const auto n = base.meta["summaries"][k].value("name", std::string()); if (n == "median") im = static_cast<int>(k); if (n == "mad") is = static_cast<int>(k); }
    need(im >= 0 && is >= 0, "cell_extent: the baseline holds no median and mad");
    Eigen::VectorXd med(S), sd(S);
    for (int s = 0; s < S; ++s) { med(s) = base.data[static_cast<std::size_t>(s * base.cols + im)]; sd(s) = base.data[static_cast<std::size_t>(s * base.cols + is)]; }
    int skipped = 0;
    const Eigen::MatrixXd z = nxr::select::standardize(cell_matrix(x, g, column_of(x, "amplitude")), med, sd, &skipped);
    const double enter = st.p["enter_sd"].get<double>(), exit = st.p["exit_sd"].get<double>(), min_area = st.p["min_area"].get<double>();
    const int min_cells = st.p["min_cells"].get<int>();
    nxr::select::Labels lab = nxr::select::Labels::Constant(S, T, -1);
    Json bursts = Json::array();
    std::int32_t next = 0;
    std::vector<std::int32_t> burst_of;
    std::vector<std::int64_t> row_of;
    for (std::int64_t b = 0; b < ev.rows; ++b) {
      const double* e = ev.data.data() + b * 4;
      const std::int64_t first = static_cast<std::int64_t>(e[0]), stop = std::min<std::int64_t>(static_cast<std::int64_t>(e[1]), T), row = static_cast<std::int64_t>(e[2]);
      if (first < 0 || stop <= first) { bursts.push_back(Json{{"burst", b}, {"first_tile", first}, {"stop_tile", stop}, {"events", Json::array()}, {"skipped", "outside the cells' time tiles"}}); continue; }
      Eigen::MatrixXd w = z.middleCols(first, stop - first);
      if (st.groups == "pieces" && row >= 0)
        for (int s = 0; s < S; ++s) if ((s >> l) != row) w.row(s).setConstant(kNaN);
      const auto R = nxr::select::windowRegions(w, g.pairs, g.tile_area, enter, exit, min_cells, min_area);
      Json evs = Json::array();
      for (std::size_t k = 0; k < R.regions.size(); ++k) {
        for (const auto n : R.regions[k].nodes) lab(n % S, first + n / S) = next + static_cast<std::int32_t>(k);
        evs.push_back(next + static_cast<std::int64_t>(k));
        burst_of.push_back(static_cast<std::int32_t>(b));
        row_of.push_back(row);
      }
      next += static_cast<std::int32_t>(R.regions.size());
      bursts.push_back(Json{{"burst", b}, {"row", row}, {"first_tile", first}, {"stop_tile", stop}, {"events", evs}});
    }
    Array out{1, g.S * g.T, std::vector<double>(static_cast<std::size_t>(S) * static_cast<std::size_t>(T), -1.0)};
    for (int f = 0; f < T; ++f) for (int s = 0; s < S; ++s) out.data[static_cast<std::size_t>(s + S * f)] = lab(s, f);
    // EACH EVENT IS A MEMBER DESCRIBED BY ITS CENTRE AND EXTENT (the Studio's span: centre + extent, here in space and time): the time centre and the duration, the vertex nearest the area-weighted
    // centroid of its tiles' union and that union's area, the vertex of its centre in each time tile and the chord path of those centres (reductions across the cells, `reduce.members`, merge the cells' ingredients)
    const int E = static_cast<int>(next);
    Eigen::Matrix<std::int32_t, Eigen::Dynamic, Eigen::Dynamic> centres;
    Eigen::VectorXd path;
    if (E > 0) { centres = nxr::select::centres(lab, E, g.level.tile, g.surface.vertices, g.vertex_area); path = nxr::select::pathLength(centres, g.surface.vertices); }
    const double rate = x.meta.value("rate", 0.0), origin = x.meta.value("origin", 0.0), tile_s = g.width_s;
    const std::int64_t nV = g.surface.n_vertices();
    Json list = Json::array();
    for (std::int32_t k = 0; k < next; ++k) {
      std::int64_t f0 = T, f1 = -1, n_cells = 0;
      std::vector<char> tiles(static_cast<std::size_t>(S), 0);
      for (int f = 0; f < T; ++f) for (int s = 0; s < S; ++s) if (lab(s, f) == k) { f0 = std::min<std::int64_t>(f0, f); f1 = std::max<std::int64_t>(f1, f); ++n_cells; tiles[static_cast<std::size_t>(s)] = 1; }
      double area = 0, w = 0, c3[3] = {0, 0, 0};
      std::int64_t n_tiles = 0;
      for (int s = 0; s < S; ++s) if (tiles[static_cast<std::size_t>(s)]) { area += g.tile_area(s); ++n_tiles; }
      for (std::int64_t v = 0; v < nV; ++v) {
        const int s = g.level.tile[static_cast<std::size_t>(v)];
        if (s < 0 || !tiles[static_cast<std::size_t>(s)]) continue;
        const double a = g.vertex_area[static_cast<std::size_t>(v)];
        w += a;
        for (int d = 0; d < 3; ++d) c3[d] += a * g.surface.vertices[static_cast<std::size_t>(v) * 3 + d];
      }
      std::int64_t centre_vertex = -1;
      double best = std::numeric_limits<double>::infinity();
      for (std::int64_t v = 0; v < nV && w > 0; ++v) {
        const int s = g.level.tile[static_cast<std::size_t>(v)];
        if (s < 0 || !tiles[static_cast<std::size_t>(s)]) continue;
        double d2 = 0;
        for (int d = 0; d < 3; ++d) { const double dd = g.surface.vertices[static_cast<std::size_t>(v) * 3 + d] - c3[d] / w; d2 += dd * dd; }
        if (d2 < best) { best = d2; centre_vertex = v; }
      }
      Json per_tile = Json::array(), area_tile = Json::array();
      for (std::int64_t f = f0; f <= f1; ++f) {
        per_tile.push_back(centres(k, f));
        double a_f = 0;
        for (int s = 0; s < S; ++s) if (lab(s, f) == k) a_f += g.tile_area(s);
        area_tile.push_back(a_f);
      }
      list.push_back(Json{{"ordinal", k}, {"burst", burst_of[static_cast<std::size_t>(k)]}, {"row", row_of[static_cast<std::size_t>(k)]}, {"first_tile", f0}, {"last_tile", f1}, {"n_cells", n_cells}, {"n_tiles", n_tiles},
                          {"centre", Json{{"time_s", origin + 0.5 * static_cast<double>(f0 + f1 + 1) * tile_s}, {"vertex", centre_vertex}}},
                          {"extent", Json{{"duration_s", static_cast<double>(f1 - f0 + 1) * tile_s}, {"area_m2", area}}},
                          {"area", area}, {"duration", static_cast<double>(f1 - f0 + 1) * tile_s}, {"path_length", path(k)}, {"centres_by_tile", per_tile}, {"area_by_tile", area_tile}});
    }
    (void)rate;
    const int bits = nxr::select::bitsFor(T);
    out.meta = Json{{"events", static_cast<std::int64_t>(next)}, {"bursts", bursts}, {"event_list", list}, {"described_by", "centre + extent"}, {"cells", x.meta}, {"bursts_tiles_of", ev.meta["tiles_of"]}, {"groups", st.groups},
                    {"space_level", l}, {"tiles", S}, {"frames", T}, {"bits", bits}, {"width_s", tile_s}, {"layout", "[1 x spatial tile + S time tile]: member c = s + S t (the product's code, the spatial factor fastest); code = the event's ordinal"},
                    {"thresholds", Json{{"enter_sd", enter}, {"exit_sd", exit}, {"min_cells", min_cells}, {"min_area", min_area}, {"standardized_by", "per spatial tile: (amplitude - median) / robust SD of the baseline statistic, across the run's time tiles"},
                                        {"tiles_without_spread", skipped}}},
                    {"baseline", Json{{"summaries", base.meta["summaries"]}, {"members_used", base.meta.value("members_used", Json(nullptr))}}},
                    {"graph", "a cell touches the cells of the level's touching tiles in its time tile (nxr::select::tileAdjacency) and its own tile in the next time tile; detection confined to the burst's time tiles"},
                    {"warnings", Json::array({"an extent is a set of cells of one pair of levels: its size is quantized to the level's tiles and time tiles; a pattern wider than a tile is placed to within the band's scale",
                                              "a region is a set of cells above their own tile's baseline, not a propagating oscillation: speed and direction are measurements of phase"})}};
    return Arrays{{"labels", std::move(out)}};
  };

}

}  // namespace nxr::pipelines::graph
