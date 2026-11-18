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
struct ExtentState { Json p; std::shared_ptr<Geometry> g; std::string groups; };
}  // namespace

void add_extent_steps(std::vector<StepSpec>& r) {
  r.push_back({"cell_extent", "after", {"x", "baseline", "events"}, {}, {"labels"}, Json{{"enter_sd", 1.5}, {"exit_sd", 1.0}, {"min_cells", 1}, {"min_area", 0.0}},
               "docs/specs/2026-11-13-cortical-events.md, Phase 2b; compute nxr::select (windowRegions on nxr::select::detectRegions, unchanged) with the tile graph of nxr::select::tileAdjacency",
               "the SPACE × TIME EXTENT of each event of a time span set (the global bursts): on the joint cells of the burst's own time tiles (and, per hemisphere, of its row's tiles), each cell's amplitude standardized "
               "against its spatial tile's baseline (the median and robust SD of a statistic across the run's time tiles), hysteresis regions on the graph of touching tiles within a time tile and a tile with itself in the next; "
               "all regions are events, numbered by burst and then by their lowest cell. Out: a labelling of the joint cells (code = the event's ordinal, -1 none)",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value &x = ins.at("x"), &b = ins.at("baseline"), &ev = ins.at("events");
                 need(x.cls == "measurement" && x.facts.contains("over") && x.facts["over"].value("joint", false) && x.facts.value("measure", std::string()) == "amplitude",
                      "cell_extent: x must be the `amplitude` of the joint cells (a measurement over a joint partition)");
                 const std::string jid = x.facts["over"]["selection_id"].get<std::string>();
                 need(b.cls == "statistic" && b.facts.contains("summaries") && b.facts.value("combine", std::string()) == "each" && b.facts.contains("over") && b.facts["over"].value("selection_id", std::string("?")) == jid &&
                      b.facts.value("measure", std::string()) == "amplitude",
                      "cell_extent: baseline must be the statistic (`summarize`, combine each) of the same cells' amplitude: a median and a robust SD per spatial tile");
                 bool med = false, mad = false;
                 for (const auto& n : b.facts["summaries"]) { med = med || n == "median"; mad = mad || n == "mad"; }
                 need(med && mad, "cell_extent: the baseline must hold median and mad");
                 const auto jrow = c.bound.selection(jid);
                 need(jrow.has_value(), "cell_extent: the joint partition is not among the subject's rows");
                 const Json pj = Json::parse(nxr::io::opt_text(*jrow, "params_json").value_or("{}"), nullptr, false);
                 need(pj.is_object() && pj.contains("joint"), "cell_extent: " + nxr::io::text(*jrow, "name") + " records no joint partition");
                 need(ev.cls == "spans" && ev.facts.contains("detect") && ev.facts.contains("over") && ev.facts.contains("reduced"),
                      "cell_extent: events must be the spans `detect_spans` found on the trace `reduce_cells` made from the same cells");
                 need(ev.facts["over"].value("selection_id", std::string("?")) == pj["joint"]["time"].get<std::string>(),
                      "cell_extent: the events were found on " + ev.facts["over"].value("selection", std::string("another level")) + ", the cells' time tiles are another level's: the bursts' time tiles are the cells'");
                 need(ev.facts["reduced"].value("space_level", -1) == pj["joint"]["space_level"].get<int>(), "cell_extent: the events' trace is of spatial level " + ev.facts["reduced"]["space_level"].dump() + ", the cells of level " + pj["joint"]["space_level"].dump());
                 for (const char* k : {"enter_sd", "exit_sd", "min_area"}) need(p[k].is_number(), std::string("cell_extent: ") + k + " is a number");
                 need(p["min_cells"].is_number_integer() && p["min_cells"].get<int>() >= 1, "cell_extent: min_cells is a whole number, at least 1");
                 need(p["enter_sd"].get<double>() >= p["exit_sd"].get<double>(), "cell_extent: enter_sd is at least exit_sd (enter high, extend low)");
                 need(p["min_area"].get<double>() >= 0, "cell_extent: min_area is at least 0 (m2)");
                 Value y = made(x);
                 y.cls = "labelling";
                 y.manifolds = {nxr::io::text(*jrow, "manifold_id")};
                 y.chunked = false;
                 y.unit.reset();
                 y.facts = Json{{"cells", x.facts["over"]}, {"extent", p}, {"groups", ev.facts["reduced"]["groups"]}};
                 return Values{{"labels", y}};
               }});

  // ── execution ──
  auto at = [&](const char* n) -> StepSpec& { for (auto& s : r) if (s.name == n) return s; refuse(std::string("no step ") + n); };
  StepSpec& ce = at("cell_extent");
  ce.traits = node_traits(TimeBehaviour::stateful, Precision::float64, OutputPrecision::float64,
                          "grows hysteresis regions over the graph of touching tiles within a time tile and from each time tile to the next: the labelling is carried forward in time; a region's extent is not defined per timestep: stateful, confirmed", {});
  ce.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    ExtentState st;
    st.p = p;
    st.g = std::make_shared<Geometry>(load_geometry(c, ins.at("x").facts["over"]["selection_id"].get<std::string>()));
    st.groups = ins.at("events").facts["reduced"]["groups"].get<std::string>();
    if (st.groups == "per_hemisphere" && (st.g->level.nodes >> st.g->level.l) != 2) refuse("cell_extent: per_hemisphere needs a surface of two pieces (the ladder's level has " + std::to_string(st.g->level.nodes >> st.g->level.l) + ")");
    return std::any(std::move(st));
  };
  ce.run = [](const std::any& st_, const Arrays& ins, ExecContext&) {
    const ExtentState& st = std::any_cast<const ExtentState&>(st_);
    const Geometry& g = *st.g;
    const Array &x = ins.at("x"), &base = ins.at("baseline"), &ev = ins.at("events");
    const int S = static_cast<int>(x.rows), T = static_cast<int>(x.cols), l = g.level.l;
    need(S == g.level.nodes, "cell_extent: the cells have " + std::to_string(S) + " spatial tiles, level " + std::to_string(l) + " has " + std::to_string(g.level.nodes));
    need(base.rows == S, "cell_extent: the baseline has " + std::to_string(base.rows) + " rows, the cells " + std::to_string(S) + " spatial tiles");
    const int im = summary_index(base.meta["summaries"], "median"), is = summary_index(base.meta["summaries"], "mad");
    Eigen::VectorXd med(S), sd(S);
    for (int s = 0; s < S; ++s) { med(s) = base.data[static_cast<std::size_t>(s * base.cols + im)]; sd(s) = base.data[static_cast<std::size_t>(s * base.cols + is)]; }
    int skipped = 0;
    const Eigen::MatrixXd z = nxr::select::standardize(matrix_of(x), med, sd, &skipped);
    const double enter = st.p["enter_sd"].get<double>(), exit = st.p["exit_sd"].get<double>(), min_area = st.p["min_area"].get<double>();
    const int min_cells = st.p["min_cells"].get<int>();
    Array out{S, T, std::vector<double>(static_cast<std::size_t>(S) * static_cast<std::size_t>(T), -1.0)};
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
      if (st.groups == "per_hemisphere" && row >= 0)
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
    for (int s = 0; s < S; ++s) for (int f = 0; f < T; ++f) out.data[static_cast<std::size_t>(s) * static_cast<std::size_t>(T) + static_cast<std::size_t>(f)] = lab(s, f);
    // each event's extent in time tiles and its number of cells, counted from the labelling (the measures over its cells are `reduce.members`, a merge of the cells' ingredients: never a bespoke roll-up here)
    std::vector<std::int64_t> first_tile(static_cast<std::size_t>(next), T), last_tile(static_cast<std::size_t>(next), -1), n_cells(static_cast<std::size_t>(next), 0);
    for (int s = 0; s < S; ++s)
      for (int f = 0; f < T; ++f) {
        const std::int32_t k = lab(s, f);
        if (k < 0) continue;
        first_tile[static_cast<std::size_t>(k)] = std::min<std::int64_t>(first_tile[static_cast<std::size_t>(k)], f);
        last_tile[static_cast<std::size_t>(k)] = std::max<std::int64_t>(last_tile[static_cast<std::size_t>(k)], f);
        ++n_cells[static_cast<std::size_t>(k)];
      }
    Json list = Json::array();
    for (std::int32_t k = 0; k < next; ++k)
      list.push_back(Json{{"ordinal", k}, {"burst", burst_of[static_cast<std::size_t>(k)]}, {"row", row_of[static_cast<std::size_t>(k)]}, {"first_tile", first_tile[static_cast<std::size_t>(k)]}, {"last_tile", last_tile[static_cast<std::size_t>(k)]}, {"n_cells", n_cells[static_cast<std::size_t>(k)]}});
    const int bits = nxr::select::bitsFor(T);
    out.meta = Json{{"events", static_cast<std::int64_t>(next)}, {"bursts", bursts}, {"event_list", list}, {"cells", x.meta}, {"bursts_tiles_of", ev.meta["tiles_of"]}, {"groups", st.groups},
                    {"space_level", l}, {"tiles", S}, {"frames", T}, {"bits", bits}, {"layout", "[spatial tile x time tile]; code = (tile << bits) | time tile"},
                    {"thresholds", Json{{"enter_sd", enter}, {"exit_sd", exit}, {"min_cells", min_cells}, {"min_area", min_area}, {"standardized_by", "per spatial tile: (amplitude - median) / robust SD of the baseline statistic, across the run's time tiles"},
                                        {"tiles_without_spread", skipped}}},
                    {"baseline", Json{{"summaries", base.meta["summaries"]}, {"members_used", base.meta["members_used"]}}},
                    {"graph", "a cell touches the cells of the level's touching tiles in its time tile (nxr::select::tileAdjacency) and its own tile in the next time tile; detection confined to the burst's time tiles"},
                    {"warnings", Json::array({"an extent is a set of cells of one pair of levels: its size is quantized to the level's tiles and time tiles; a pattern wider than a tile is placed to within the band's scale",
                                              "a region is a set of cells above their own tile's baseline, not a propagating oscillation: speed and direction are measurements of phase"})}};
    return Arrays{{"labels", std::move(out)}};
  };

}

}  // namespace nxr::pipelines::graph
