// nxr-pipelines — the registered steps of the WHERE of a global burst (spec docs/specs/2026-11-13-cortical-events.md, phase 2b): `cell_extent` (the events of each burst on the joint cells) and `extent_measure`
// (each event's measures by the exact roll-up of its cells). Both are `after` steps on arrays; the maths is nxr::extent (compute/), the runner binds and stores.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include "nxr/extent.h"
#include "nxr/io/errors.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/spatial_band.h"

namespace nxr::pipelines::graph {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
Value made(const Value& x) { Value v = x; v.row.reset(); return v; }

// the joint partition's record, the surface and the level it names, and the geometry the extent needs: tile areas (the cells' weights: barycentric vertex areas summed by tile)
struct Geometry {
  Json joint;
  nxr::pipelines::SurfaceInput surface;
  nxr::pipelines::SurfaceLevel level;
  std::vector<double> vertex_area;
  Eigen::VectorXd tile_area;
  nxr::extent::Pairs pairs;
};
Geometry load_geometry(ExecContext& c, const std::string& joint_selection_id) {
  Geometry g;
  const auto row = c.sub.db().read("selection", joint_selection_id);
  need(row.has_value(), "the joint partition is not among the subject's rows");
  g.joint = Json::parse(nxr::io::text(*row, "params_json"))["joint"];
  g.surface = nxr::pipelines::read_surface(c.sub, g.joint["surface"].get<std::string>(), true);
  g.level = nxr::pipelines::read_surface_level(c.sub, g.surface, g.joint["space_level"].get<int>());
  if (g.level.nodes != g.joint["tiles"].get<std::int64_t>()) refuse("the joint partition records " + g.joint["tiles"].dump() + " tiles, level " + std::to_string(g.level.l) + " of " + g.surface.name + " has " + std::to_string(g.level.nodes));
  nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(g.surface.n_vertices()); mesh.positions = g.surface.vertices; mesh.faces = g.surface.faces;
  g.vertex_area = nxr::ladder::vertexAreas(mesh);
  g.tile_area = Eigen::VectorXd::Zero(g.level.nodes);
  for (std::size_t v = 0; v < g.level.tile.size(); ++v) if (g.level.tile[v] >= 0) g.tile_area(g.level.tile[v]) += g.vertex_area[v];
  for (const auto& [a, b] : g.level.pairs) g.pairs.emplace_back(a, b);
  return g;
}
Eigen::MatrixXd matrix_of(const Array& a) {
  Eigen::MatrixXd M(a.rows, a.cols);
  for (std::int64_t r = 0; r < a.rows; ++r) for (std::int64_t c = 0; c < a.cols; ++c) M(r, c) = a.data[static_cast<std::size_t>(r * a.cols + c)];
  return M;
}
nxr::extent::Labels labels_of(const Array& a) {
  nxr::extent::Labels L(a.rows, a.cols);
  for (std::int64_t r = 0; r < a.rows; ++r) for (std::int64_t c = 0; c < a.cols; ++c) L(r, c) = static_cast<std::int32_t>(a.data[static_cast<std::size_t>(r * a.cols + c)]);
  return L;
}
Eigen::VectorXd samples_of(const Json& counts, std::int64_t T) {
  Eigen::VectorXd s = Eigen::VectorXd::Zero(T);
  for (const auto& c : counts) { const auto t = c[0].get<std::int64_t>(); if (t >= 0 && t < T) s(t) = c[1].get<double>(); }
  return s;
}
int summary_index(const Json& summaries, const char* name) {
  for (std::size_t k = 0; k < summaries.size(); ++k) if (summaries[k]["name"] == name) return static_cast<int>(k);
  return -1;
}
struct ExtentState { Json p; std::shared_ptr<Geometry> g; std::string groups; };
}  // namespace

void add_extent_steps(std::vector<StepSpec>& r) {
  r.push_back({"cell_extent", "after", {"x", "baseline", "events"}, {}, {"labels"}, Json{{"enter_sd", 1.5}, {"exit_sd", 1.0}, {"min_cells", 1}, {"min_area", 0.0}},
               "docs/specs/2026-11-13-cortical-events.md, Phase 2b; compute nxr::extent (windowRegions on nxr::regions::detectRegions, unchanged) with the tile graph of nxr::ladder::tileAdjacency",
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
                 const auto jrow = c.db.read("selection", jid);
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
  r.push_back({"extent_measure", "after", {"labels", "x"}, {"x"}, {"y"}, Json{{"measure", nullptr}, {"per", "event"}},
               "docs/specs/2026-11-13-cortical-events.md, Phase 2b; compute nxr::extent (rollup, areaByFrame, centres, pathLength): the registry's reductions over the labelling's members (sum, mean, max), the centre a place",
               "a measure of each event of a labelling of the joint cells, by the exact roll-up of its cells. per event: n_cells, n_tiles (distinct spatial tiles), area (their union, m2), amplitude (the area- and sample-weighted mean "
               "of the cells' amplitude), envelope_max (the largest cell peak), first_tile, last_tile, duration (s), burst_ordinal, path_length (the chord of the centres' path, m). per time_tile (rows = the events, members = "
               "the time level's tiles; NaN where the event has no cell): area (m2) and centre_vertex",
               [](const Values& ins, const Json& p, const PlanContext& c) {
                 const Value& lb = ins.at("labels");
                 need(lb.cls == "labelling" && lb.facts.contains("cells"), "extent_measure: labels must be the labelling `cell_extent` made");
                 need(p["measure"].is_string(), "extent_measure: measure is one of the event measures (n_cells, n_tiles, area, amplitude, envelope_max, first_tile, last_tile, duration, burst_ordinal, path_length) or, per time_tile, area or centre_vertex");
                 const std::string m = p["measure"].get<std::string>(), per = p["per"].is_string() ? p["per"].get<std::string>() : std::string();
                 need(per == "event" || per == "time_tile", "extent_measure: per " + p["per"].dump() + ": event or time_tile");
                 static const char* ev_measures[] = {"n_cells", "n_tiles", "area", "amplitude", "envelope_max", "first_tile", "last_tile", "duration", "burst_ordinal", "path_length"};
                 bool known = false;
                 if (per == "event") for (const char* k : ev_measures) known = known || m == k;
                 else known = m == "area" || m == "centre_vertex";
                 need(known, "extent_measure: measure " + p["measure"].dump() + " per " + per + (per == "event" ? ": n_cells, n_tiles, area, amplitude, envelope_max, first_tile, last_tile, duration, burst_ordinal or path_length" : ": area or centre_vertex"));
                 if (per == "event" && (m == "amplitude" || m == "envelope_max")) {
                   need(ins.count("x") != 0, "extent_measure: " + m + " rolls up the cells' " + (m == "amplitude" ? "amplitude" : "envelope_max") + ": give x (the measurement over the same joint partition)");
                   const Value& x = ins.at("x");
                   need(x.cls == "measurement" && x.facts.contains("over") && x.facts["over"].value("joint", false) && x.facts["over"].value("selection_id", std::string("a")) == lb.facts["cells"].value("selection_id", std::string("b")) &&
                        x.facts.value("measure", std::string()) == (m == "amplitude" ? "amplitude" : "envelope_max"),
                        "extent_measure: x must be the `" + std::string(m == "amplitude" ? "amplitude" : "envelope_max") + "` over the labelling's joint partition");
                 }
                 Value y = made(lb);
                 y.cls = "measurement";
                 y.chunked = false;
                 y.facts = Json::object();
                 y.facts["measure"] = m;
                 y.unit.reset();
                 if (m == "area") y.unit = "m\xc2\xb2";
                 if (m == "path_length") y.unit = "m";
                 if (m == "duration") y.unit = "s";
                 if (per == "event") {
                   y.facts["over"] = Json{{"events", true}, {"events_of", "labelling"}, {"selection", "the events of this run"}, {"tile_level", lb.facts["cells"]}};
                 } else {
                   const auto jrow = c.db.read("selection", lb.facts["cells"]["selection_id"].get<std::string>());
                   const Json j = Json::parse(nxr::io::text(*jrow, "params_json"))["joint"];
                   const auto tl = c.db.read("selection", j["time"].get<std::string>());
                   need(tl.has_value(), "extent_measure: the joint partition's time level is not among the subject's rows");
                   Json over = lb.facts["cells"];
                   for (const char* k : {"joint", "space_level", "space_tiles"}) over.erase(k);
                   over["selection_id"] = nxr::io::text(*tl, "id"); over["selection"] = nxr::io::text(*tl, "name");
                   over["rows"] = "the events of the labelling"; over["rows_events_of"] = "labelling";
                   y.facts["over"] = over;
                 }
                 y.facts["per"] = per;
                 return Values{{"y", y}};
               }});

  // ── execution ──
  auto at = [&](const char* n) -> StepSpec& { for (auto& s : r) if (s.name == n) return s; refuse(std::string("no step ") + n); };
  StepSpec& ce = at("cell_extent");
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
    const Eigen::MatrixXd z = nxr::extent::standardize(matrix_of(x), med, sd, &skipped);
    const double enter = st.p["enter_sd"].get<double>(), exit = st.p["exit_sd"].get<double>(), min_area = st.p["min_area"].get<double>();
    const int min_cells = st.p["min_cells"].get<int>();
    Array out{S, T, std::vector<double>(static_cast<std::size_t>(S) * static_cast<std::size_t>(T), -1.0)};
    nxr::extent::Labels lab = nxr::extent::Labels::Constant(S, T, -1);
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
      const auto R = nxr::extent::windowRegions(w, g.pairs, g.tile_area, enter, exit, min_cells, min_area);
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
    const auto ru = nxr::extent::rollup(lab, next, g.tile_area, samples_of(x.meta["counts"], T), nullptr, nullptr);
    Json list = Json::array();
    for (std::int32_t k = 0; k < next; ++k)
      list.push_back(Json{{"ordinal", k}, {"burst", burst_of[static_cast<std::size_t>(k)]}, {"row", row_of[static_cast<std::size_t>(k)]}, {"first_tile", ru.first(k)}, {"last_tile", ru.last(k)}, {"n_cells", ru.cells(k)}});
    const int bits = nxr::extent::bitsFor(T);
    out.meta = Json{{"events", static_cast<std::int64_t>(next)}, {"bursts", bursts}, {"event_list", list}, {"cells", x.meta}, {"bursts_tiles_of", ev.meta["tiles_of"]}, {"groups", st.groups},
                    {"space_level", l}, {"tiles", S}, {"frames", T}, {"bits", bits}, {"layout", "[spatial tile x time tile]; code = (tile << bits) | time tile"},
                    {"thresholds", Json{{"enter_sd", enter}, {"exit_sd", exit}, {"min_cells", min_cells}, {"min_area", min_area}, {"standardized_by", "per spatial tile: (amplitude - median) / robust SD of the baseline statistic, across the run's time tiles"},
                                        {"tiles_without_spread", skipped}}},
                    {"baseline", Json{{"summaries", base.meta["summaries"]}, {"members_used", base.meta["members_used"]}}},
                    {"graph", "a cell touches the cells of the level's touching tiles in its time tile (nxr::ladder::tileAdjacency) and its own tile in the next time tile; detection confined to the burst's time tiles"},
                    {"warnings", Json::array({"an extent is a set of cells of one pair of levels: its size is quantized to the level's tiles and time tiles; a pattern wider than a tile is placed to within the band's scale",
                                              "a region is a set of cells above their own tile's baseline, not a propagating oscillation: speed and direction are measurements of phase"})}};
    return Arrays{{"labels", std::move(out)}};
  };

  StepSpec& em = at("extent_measure");
  em.prepare = [](const Values& ins, const Json& p, ExecContext& c) {
    ExtentState st;
    st.p = p;
    st.g = std::make_shared<Geometry>(load_geometry(c, ins.at("labels").facts["cells"]["selection_id"].get<std::string>()));
    return std::any(std::move(st));
  };
  em.run = [](const std::any& st_, const Arrays& ins, ExecContext&) {
    const ExtentState& st = std::any_cast<const ExtentState&>(st_);
    const Geometry& g = *st.g;
    const Array& lb = ins.at("labels");
    const std::string m = st.p["measure"].get<std::string>(), per = st.p["per"].get<std::string>();
    const int S = static_cast<int>(lb.rows), T = static_cast<int>(lb.cols), E = lb.meta["events"].get<int>();
    const nxr::extent::Labels lab = labels_of(lb);
    const Json& cells = lb.meta["cells"];
    const Eigen::VectorXd samples = samples_of(cells["counts"], T);
    const double width_s = cells.value("width_s", 0.0), origin = cells.value("origin", 0.0);
    const std::int64_t W = cells.value("window_samples", std::int64_t(0));
    if (per == "time_tile") {
      Array y{E, E == 0 ? 0 : T, std::vector<double>(static_cast<std::size_t>(E) * static_cast<std::size_t>(T), kNaN)};
      if (E > 0) {
        if (m == "area") { const Eigen::MatrixXd a = nxr::extent::areaByFrame(lab, E, g.tile_area); for (int e = 0; e < E; ++e) for (int f = 0; f < T; ++f) y.data[static_cast<std::size_t>(e) * static_cast<std::size_t>(T) + static_cast<std::size_t>(f)] = a(e, f); }
        else {
          const auto C = nxr::extent::centres(lab, E, g.level.tile, g.surface.vertices, g.vertex_area);
          for (int e = 0; e < E; ++e) for (int f = 0; f < T; ++f) if (C(e, f) >= 0) y.data[static_cast<std::size_t>(e) * static_cast<std::size_t>(T) + static_cast<std::size_t>(f)] = C(e, f);
        }
      }
      Json meta = cells;
      meta["measure"] = m; meta["rows_are"] = "the events of the labelling (ordinal order)"; meta["time_tile_of"] = "the events' labelling";
      for (const char* k : {"joint", "space_tiles"}) meta.erase(k);
      y.meta = meta;
      return Arrays{{"y", std::move(y)}};
    }
    Array y{1, E, std::vector<double>(static_cast<std::size_t>(E), kNaN)};
    const nxr::extent::Rollup* ru = nullptr;
    nxr::extent::Rollup rollup;
    Eigen::MatrixXd amp, peak;
    if (m == "amplitude") amp = matrix_of(ins.at("x"));
    if (m == "envelope_max") peak = matrix_of(ins.at("x"));
    rollup = nxr::extent::rollup(lab, E, g.tile_area, samples, m == "amplitude" ? &amp : nullptr, m == "envelope_max" ? &peak : nullptr);
    ru = &rollup;
    Eigen::VectorXd path;
    if (m == "path_length") path = nxr::extent::pathLength(nxr::extent::centres(lab, E, g.level.tile, g.surface.vertices, g.vertex_area), g.surface.vertices);
    const Eigen::MatrixXd byframe = nxr::extent::areaByFrame(lab, E, g.tile_area);
    Json counts = Json::array();
    for (int e = 0; e < E; ++e) {
      double n = 0;
      for (int f = 0; f < T; ++f) if (!std::isnan(byframe(e, f))) n += samples(f);
      counts.push_back(Json::array({e, n}));
      double v = kNaN;
      if (m == "n_cells") v = ru->cells(e);
      else if (m == "n_tiles") v = ru->tiles(e);
      else if (m == "area") v = ru->area(e);
      else if (m == "amplitude") v = ru->amplitude(e);
      else if (m == "envelope_max") v = ru->peak(e);
      else if (m == "first_tile") v = ru->first(e);
      else if (m == "last_tile") v = ru->last(e);
      else if (m == "duration") v = (ru->last(e) - ru->first(e) + 1) * width_s;
      else if (m == "burst_ordinal") v = lb.meta["event_list"][static_cast<std::size_t>(e)]["burst"].get<double>();
      else if (m == "path_length") v = path(e);
      y.data[static_cast<std::size_t>(e)] = v;
    }
    (void)S;
    y.meta = Json{{"measure", m}, {"events", true}, {"events_of", "labelling"}, {"tiles", E}, {"counts", counts}, {"whole_tiles", E}, {"tiles_measured", nullptr}, {"selection", "the events of this run"},
                  {"window_samples", W}, {"width_s", width_s}, {"origin", origin}, {"tile_level", Json{{"selection_id", cells.value("selection_id", std::string())}, {"selection", cells.value("selection", std::string())}}}};
    if (m == "path_length") y.meta["path"] = Json{{"between", "the centre vertices of consecutive time tiles"}, {"distance", "straight (chord), the surface's units"}};
    return Arrays{{"y", std::move(y)}};
  };
}

}  // namespace nxr::pipelines::graph
