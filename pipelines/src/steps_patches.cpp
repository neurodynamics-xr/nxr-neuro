// nxr-pipelines — MORE VERBS AS REGISTERED STEPS (the pattern of verb_steps.cpp: setup steps that read stored data only through the run's bound services and hand back described arrays the
// definition stores). One step a verb; the maths are the pure functions of compute/.
//
//   run_repeat      setup  two recordings' tile crossings over one tile level  ->  tiles, whole: does each tile's direction of core movement repeat from one run to the next?
//   cellquadrature  setup  an analytic cross-spectrum + kernel + joint cells (+ bursts)  ->  the quadrature ingredients per spatial tile (and tile pair), per time tile and per burst
//   patches         setup  a scalar map on a surface + the tiles of a level  ->  its band, the tiles' measurements, the baseline, the patches and their measurements and summaries
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

#include <Eigen/Dense>

#include "nxr/select/track_crossings.h"
#include "nxr/select/events.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/run_repeat.h"
#include "nxr/pipelines/select_nodes.h"
#include "nxr/quantity/quantity.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/core/algebra.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/geometry/mesh.h"
#include "nxr/signal/complex_signal.h"
#include "nxr/signal/crossspectral.h"
#include "nxr/signal/crossspectral_decomposition.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/stats/circular.h"
#include "nxr/measurement/congruence_norms.h"
#include "nxr/select/regions.h"
#include "nxr/stats/summary.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using nxr::io::text;

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::invalid_argument(m); }
void need(bool ok, const std::string& m) { if (!ok) fail(m); }
Value value_of_class(const char* cls) { Value v; v.cls = cls; return v; }
const double kNaN = std::numeric_limits<double>::quiet_NaN();

// ── run_repeat ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE RUN-TO-RUN REPEAT of the vortex cores' crossings (the hand-wired `repeat compute` it replaces; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 8): per tile of one level the dot
// product of the two runs' mean resultant crossing vectors (the senses pooled), for the tiles with at least `min_crossings` crossings in both runs; for the whole recording the count-weighted
// mean cosine of the tiles' directions (nxr `repeat`, `tile_dot`). Read from the two runs' stored `tile_crossings` measurements over the level's tiles; never a recording.
struct RunData { nxr::io::MeasurementValues x, y, z, n; };
// the pooled senses of stage `stage`: rows 2s (sense +1) and 2s+1 (sense -1); the members' last is the bridge, not a tile
nxr::select::track_crossings::Pooled pooled_of(const RunData& d, int stage, std::int64_t tiles) {
  const std::int64_t M = d.n.members, r0 = 2 * static_cast<std::int64_t>(stage), r1 = r0 + 1;
  Eigen::MatrixXd plus(tiles, 3), minus(tiles, 3);
  Eigen::VectorXd np(tiles), nm(tiles);
  for (std::int64_t t = 0; t < tiles; ++t) {
    const std::size_t a = static_cast<std::size_t>(r0 * M + t), b = static_cast<std::size_t>(r1 * M + t);
    plus.row(t) << d.x.values[a], d.y.values[a], d.z.values[a];
    minus.row(t) << d.x.values[b], d.y.values[b], d.z.values[b];
    np[t] = d.n.values[a]; nm[t] = d.n.values[b];
  }
  return nxr::select::track_crossings::poolSenses(plus, np, minus, nm);
}

struct RepeatState { Value a, b, over; Json p; };
StepSpec run_repeat_step() {
  StepSpec s;
  s.name = "select.run_repeat";
  s.kind = "setup";
  s.inputs = {"a", "b", "over"};
  s.outputs = {"tiles", "whole"};
  s.parameters = Json{{"spatial_scale", nullptr}, {"stage", 0}, {"stage_name", nullptr}, {"min_crossings", 10}};
  s.matches = "nxr vortex_cores.repeat, tile_dot (NMIN_TILE 10); compute nxr::select::track_crossings::repeat";
  s.description = "per tile of one level, the dot product of two runs' mean resultant crossing vectors, and the count-weighted mean cosine of the directions over the tiles both runs cross enough";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double from stored measurements");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    for (const char* k : {"a", "b", "over"}) need(ins.count(k) && ins.at(k).row, std::string("run_repeat: ") + k + " is not given");
    need(text(*ins.at("a").row, "id") != text(*ins.at("b").row, "id"), "run_repeat: the two runs are the same recording (" + text(*ins.at("a").row, "name") + "): a recording repeats itself");
    need(!(p["stage_name"].is_string() && p["stage_name"].get<std::string>().empty()), "run_repeat: stage_name is empty");
    need(p.value("stage", 0) >= 0, "run_repeat: stage is a whole number >= 0 (the stage's index in the run's `stages`)");
    need(p.value("min_crossings", 10) >= 1, "run_repeat: min_crossings is at least 1 (nxr: 10)");
    return Values{{"tiles", value_of_class("statistic")}, {"whole", value_of_class("statistic")}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(RepeatState{ins.at("a"), ins.at("b"), ins.at("over"), p}); };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const RepeatState& st = std::any_cast<const RepeatState&>(state);
    const Json& p = st.p;
    const Bound& b = ctx.bound;
    const std::string name_a = text(*st.a.row, "name"), name_b = text(*st.b.row, "name"), sel_id = text(*st.over.row, "id");
    const int min_crossings = p.value("min_crossings", 10);
    // `spatial_scale`: the tiles the crossings were counted in, as a QUANTITY; the `over` level must be the level it resolves to on its ladder (never a bare index)
    Json scale_snapped = nullptr;
    if (p["spatial_scale"].is_string()) {
      need(!nxr::io::is_null(*st.over.row, "member_of_id"), "run_repeat: " + text(*st.over.row, "name") + " is not a level of a tile ladder: spatial_scale cannot be checked against it");
      Value ladder;
      ladder.row = b.selection(text(*st.over.row, "member_of_id"));
      Json sn;
      const int l = level_of_ladder(b, ladder, p["spatial_scale"], "run_repeat", &sn);
      need(nxr::io::integer(*st.over.row, "member_ordinal") == l, "run_repeat: spatial_scale " + p["spatial_scale"].dump() + " is level " + std::to_string(l) + " of the ladder, `over` (" + text(*st.over.row, "name") + ") is level " + std::to_string(nxr::io::integer(*st.over.row, "member_ordinal")));
      scale_snapped = sn;
    }
    const std::optional<std::string> stage_name = p["stage_name"].is_string() ? std::optional<std::string>(p["stage_name"].get<std::string>()) : std::nullopt;
    const int stage = p.value("stage", 0);
    auto read = [&](const Value& f) {
      const std::string fid = text(*f.row, "id");
      auto get = [&](const char* m) { return b.measurement_values(sel_id, fid, std::nullopt, m); };
      return RunData{get("crossing_sum_x"), get("crossing_sum_y"), get("crossing_sum_z"), get("crossings")};
    };
    const RunData A = read(st.a), B = read(st.b);
    const std::int64_t tiles = nxr::io::is_null(*st.over.row, "n_members") ? A.n.members - 1 : nxr::io::integer(*st.over.row, "n_members");
    // THE STAGE, by name in each run's own rows (they may list their stages in another order) or by index
    const std::vector<std::string> names_a = crossing_stage_names(A.n.computed_by), names_b = crossing_stage_names(B.n.computed_by);
    const int sa = stage_name ? crossing_stage_index(name_a, names_a, *stage_name) : stage, sb = stage_name ? crossing_stage_index(name_b, names_b, *stage_name) : stage;
    for (const auto& [d, sg] : {std::pair<const RunData*, int>{&A, sa}, std::pair<const RunData*, int>{&B, sb}})
      need(d->n.members == tiles + 1 && d->n.rows >= 2 * static_cast<std::int64_t>(sg) + 2,
           "run_repeat: the stored crossings are [" + std::to_string(d->n.rows) + " rows x " + std::to_string(d->n.members) + " members]; the level has " + std::to_string(tiles) + " tiles (and the bridge) and stage " + std::to_string(sg) +
               " needs rows " + std::to_string(2 * sg) + " and " + std::to_string(2 * sg + 1));
    const nxr::select::track_crossings::Pooled pa = pooled_of(A, sa, tiles), pb = pooled_of(B, sb, tiles);
    const nxr::select::track_crossings::Repeat rep = nxr::select::track_crossings::repeat(pa, pb, min_crossings);
    const Json warnings = Json::array({"a repeat is two runs of one subject agreeing: they share the cortex, the kernel and the instrument's leakage, so it is evidence only against the lag-free null (piece 10), whose p-value this statistic waits for",
                                       "a tile counts only with at least min_crossings crossings in both runs (the senses pooled); the whole recording's cosine is the count-weighted mean over those tiles"});
    Json pj = Json{{"warnings", warnings}, {"reference", "nxr vortex_cores.repeat, tile_dot (NMIN_TILE 10)"}};
    pj["rows"] = std::string("stage ") + (stage_name ? *stage_name : std::to_string(stage)) + ": rows " + std::to_string(2 * sa) + "/" + std::to_string(2 * sa + 1) + " of " + name_a + ", " + std::to_string(2 * sb) + "/" + std::to_string(2 * sb + 1) + " of " + name_b + " (sense +1 and -1, pooled)";
    const Json opts = Json{{"other_field_id", text(*st.b.row, "id")}, {"recording_a", name_a}, {"recording_b", name_b}, {"stage", stage_name ? Json(*stage_name) : Json(stage)}, {"stage_a", sa}, {"stage_b", sb},
                           {"min_crossings", min_crossings}, {"spatial_scale", Json{{"stated", p["spatial_scale"]}, {"snapped", scale_snapped}}}, {"tiles_used", rep.tilesUsed}, {"tiles", tiles}, {"crossings_a", pa.n.sum()}, {"crossings_b", pb.n.sum()}};
    Array t;
    t.rows = tiles; t.cols = 1;
    t.data.assign(rep.dot.data(), rep.dot.data() + rep.dot.size());
    Json tile_values = Json::array();                                       // the entries of the axis: the level's tiles by number (a statistic's axis names what it runs along)
    for (std::int64_t k = 0; k < tiles; ++k) tile_values.push_back(Json{{"name", std::to_string(k)}});
    const Json tile_axis = Json{{"name", "tile"}, {"size", tiles}, {"values", tile_values}};
    t.meta["describe"] = Json{{"measure", "repeat_dot"}, {"method", "run_repeat"}, {"precision", "float64"}, {"n_members", tiles}, {"options", opts}, {"producer", pj},
                              {"axes", Json::array({tile_axis})},
                              {"description", "per tile of " + text(*st.over.row, "name") + ": the dot product of the mean resultant crossing vectors of " + name_a + " and " + name_b + " (senses pooled; NaN with fewer than " + std::to_string(min_crossings) + " crossings in either)"}};
    Array w;
    w.rows = 1; w.cols = 1;
    w.data = {rep.cosine};
    w.meta["describe"] = Json{{"measure", "repeat_cosine"}, {"method", "run_repeat"}, {"precision", "float64"}, {"n_members", rep.tilesUsed}, {"options", opts}, {"producer", pj},
                              {"axes", Json::array({Json{{"name", "whole"}, {"size", 1}, {"values", Json::array({Json{{"name", "count_weighted_mean_cosine"}}})}}})},
                              {"description", "the count-weighted mean cosine of the tiles' directions between " + name_a + " and " + name_b + " (tiles with at least " + std::to_string(min_crossings) + " crossings in both)"}};
    return Arrays{{"tiles", std::move(t)}, {"whole", std::move(w)}};
  };
  return s;
}

// ── cellquadrature ───────────────────────────────────────────────────────────────────────────────────────────────
// QUADRATURE PER CELL AND PER BURST (the hand-wired `cellquadrature compute` it replaces; spec docs/specs/2026-11-13-cortical-events.md, phase 2d): travelling or standing. T = B C B^T per spatial tile of
// a level (B the tile's rows of the kernel with the level's Weyl band, all components; C a time tile's analytic cross-spectrum on the kernel's channels), stored as the two additive ingredients
// sum |T|^2 and sum |Im T|^2 (the quadrature fraction is derived at read), per time tile of the joint cells and per burst (C summed over the burst's tiles FIRST: products add), and with
// `tile_pairs` the cross-tile blocks of the level's touching tiles. Weights: area (a vertex counts by its barycentric area) or none.
const char* const kTotal = "cross_spectrum_norm2";
const char* const kQuad = "cross_spectrum_quadrature_norm2";
const char* const kPairTotal = "tile_pair_cross_spectrum_norm2";
const char* const kPairQuad = "tile_pair_cross_spectrum_quadrature_norm2";

struct CellState { Value cross_spectrum, recording, kernel, joint; std::optional<Value> bursts; Json p; };
SpatialBandOptions band_options(const Json& p) {
  SpatialBandOptions bo;
  bo.transform = p.value("transform", std::string("auto"));
  if (!p["order"].is_null()) bo.order = p["order"].get<int>();
  if (!p["tolerance"].is_null()) bo.tolerance = p["tolerance"].get<double>();
  if (!p["modes"].is_null()) bo.modes = p["modes"].get<int>();
  return bo;
}
// an area given as a quantity ("400 mm^2", "area 100-400 mm^2"; a limit is its upper end) in m^2; null stays null. A bare number is refused: an area is a quantity with a unit.
std::optional<double> area_m2_of(const Json& j, const std::string& who, const std::string& what) {
  if (j.is_null()) return std::nullopt;
  need(j.is_string(), who + ": " + what + " is a quantity with a unit (\"400 mm^2\"), not " + j.dump());
  try {
    const nxr::quantity::Quantity q = nxr::quantity::parse(j.get<std::string>());
    need(q.chart == "area", who + ": " + what + " " + j.dump() + " is not an area (mm^2, m^2)");
    const double v = std::isfinite(q.hi) ? q.hi : q.lo;
    return v * nxr::quantity::unit_factor(q.unit, "m^2");
  } catch (const std::invalid_argument& e) { fail(who + ": " + what + ": " + e.what()); }
}
StepSpec cellquadrature_step() {
  StepSpec s;
  s.name = "measurement.cellquadrature";
  s.kind = "setup";
  s.inputs = {"cross_spectrum", "recording", "kernel", "joint", "bursts"};
  s.optional = {"bursts"};
  s.outputs = {"cells", "cell_pairs", "bursts", "burst_pairs"};
  s.parameters = Json{{"spatial_scale", nullptr}, {"weights", "area"}, {"tile_pairs", true}, {"transform", "auto"}, {"modes", nullptr}, {"order", nullptr}, {"tolerance", nullptr}};
  s.matches = "compute nxr::measurement::groupedCongruenceNorms, pairCongruenceNorms; the merge rule sum (nxr::measurement)";
  s.description = "the quadrature ingredients of the kernel's spatial tiles (and touching tile pairs) per time tile of the joint cells and per burst, from an analytic cross-spectrum";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double from stored windows");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    for (const char* k : {"cross_spectrum", "recording", "kernel", "joint"}) need(ins.count(k) && ins.at(k).row, std::string("cellquadrature: ") + k + " is not given");
    need(p["spatial_scale"].is_string(), "cellquadrature: spatial_scale is a quantity (a tile diameter \"20 mm\", \"parts 8\", a wavelength) stating the surface's tiles; a bare level index is never a parameter");
    const std::string w = p.value("weights", std::string("area"));
    need(w == "area" || w == "none", "cellquadrature: weights is area (the vertices count by their area) or none (every vertex counts equally)");
    const std::string t = p.value("transform", std::string("auto"));
    need(t == "auto" || t == "eigen" || t == "chebyshev" || t == "exact", "cellquadrature: transform is auto (eigenmodes first, chebyshev an announced fallback), eigen, chebyshev or exact");
    need(p["order"].is_null() || p["order"].get<int>() >= 0, "cellquadrature: order is at least 0 (0: chosen from the tolerance)");
    need(p["tolerance"].is_null() || p["tolerance"].get<double>() > 0, "cellquadrature: tolerance is above 0");
    need(ins.at("kernel").manifolds.size() == 2, "cellquadrature: the kernel is an operator from the channels to a surface");
    if (ins.count("bursts") && ins.at("bursts").row) need(ins.at("recording").time && ins.at("bursts").manifolds.size() == 1 && ins.at("bursts").manifolds[0] == *ins.at("recording").time, "cellquadrature: the bursts must be events of the cross-spectrum's recording (on its time line)");
    return Values{{"cells", value_of_class("measurement")}, {"cell_pairs", value_of_class("measurement")}, {"bursts", value_of_class("measurement")}, {"burst_pairs", value_of_class("measurement")}};
  };
  s.resolve = [](const Values& ins, const Json& p, const Bound& bound) -> Json {            // the spatial band's choice (eigenmodes stored or computable, else the series) enters the run's cache key
    if (!ins.count("kernel") || p["spatial_scale"].is_null()) return nullptr;
    const auto man = bound.manifold(ins.at("kernel").manifolds[1]);
    if (!man) return nullptr;
    const SurfaceInput S = bound.surface(text(*man, "name"), true);
    return bound.plan_spatial_band(S, bound.surface_level(S, level_of_surface(bound, S, p["spatial_scale"], "cellquadrature")), band_options(p)).key;
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) {
    CellState st{ins.at("cross_spectrum"), ins.at("recording"), ins.at("kernel"), ins.at("joint"), std::nullopt, p};
    if (ins.count("bursts") && ins.at("bursts").row) st.bursts = ins.at("bursts");
    return std::any(std::move(st));
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const CellState& st = std::any_cast<const CellState&>(state);
    const Json& p = st.p;
    const Bound& b = ctx.bound;
    const std::string weights = p.value("weights", std::string("area"));
    const bool tile_pairs = p.value("tile_pairs", true);
    const std::string csname = text(*st.cross_spectrum.row, "name"), recording = text(*st.recording.row, "name");
    // the cross-spectrum: an analytic series on time tiles, its channels' manifold, its provenance
    // (the live analytic cross-spectrum is the complex measurement `sum` over a band's time grid, `run measure --set analytic_cross_spectrum_tiles`; its grid IS the tiles)
    const Json cpj = Json::parse(nxr::io::opt_text(*st.cross_spectrum.row, "params_json").value_or("{}"), nullptr, false);
    need(cpj.is_object() && cpj.contains("grid") && cpj["grid"].value("width", std::int64_t(0)) > 0, "cellquadrature: " + csname + " is not a time grid of tiles (`run measure --set analytic_cross_spectrum_tiles` first: its band's time grid holds the cross-spectrum)");
    need(!st.recording.time.has_value() || text(*st.cross_spectrum.row, "manifold_id") == *st.recording.time, "cellquadrature: " + csname + " is a grid of another time line than " + recording + "'s");
    const std::int64_t window_samples = cpj["grid"]["width"].get<std::int64_t>();
    int window_level = -1;
    for (int k = 0; k < 40; ++k) if ((std::int64_t(1) << k) == window_samples) window_level = k;
    need(window_level >= 0, "cellquadrature: " + csname + ": its tiles are " + std::to_string(window_samples) + " samples, not a level of the dyadic grid (a power of two)");
    need(st.recording.time.has_value() && st.recording.rate > 0, "cellquadrature: " + recording + ": it has no time line");
    // the spatial scale is a QUANTITY resolved on the kernel's surface's ladder (compute's level_of_*; the snapped value into the provenance)
    Json scale_snapped;
    const int l = [&] {
      const auto kman = b.manifold(text(*st.kernel.row, "to_manifold_id"));
      need(kman.has_value(), "cellquadrature: " + text(*st.kernel.row, "name") + ": its surface is not stored");
      return level_of_surface(b, b.surface(text(*kman, "name"), true), p["spatial_scale"], "cellquadrature", &scale_snapped);
    }();
    // the joint partition of the pair (space level l, the time level whose window IS the cross-spectrum's tile level)
    const Json jp = Json::parse(nxr::io::opt_text(*st.joint.row, "params_json").value_or("{}"), nullptr, false);
    const Json jj = product_facts(jp, [&](const std::string& id) { return b.selection(id); }, [&](const std::string& id) { return b.manifold(id); }, "cellquadrature: " + text(*st.joint.row, "name"));
    need((!jj.contains("recording") || jj["recording"].get<std::string>() == recording) && jj.value("space_level", -1) == l, "cellquadrature: the joint '" + text(*st.joint.row, "name") + "' is not of " + recording + " at spatial level " + std::to_string(l));
    need(jj.value("window_samples", std::int64_t(0)) == window_samples,
         "cellquadrature: the joint '" + text(*st.joint.row, "name") + "' has time tiles of " + std::to_string(jj.value("window_samples", std::int64_t(0))) + " samples, the cross-spectrum '" + csname + "' of " + std::to_string(window_samples) +
             " (level " + std::to_string(window_level) + "): the tile level of the cross-spectrum must equal the pair's time level");
    const std::int64_t frames = jj["frames"].get<std::int64_t>();
    // the kernel
    const nxr::io::Row& kern = *st.kernel.row;
    const std::string kname = text(kern, "name");
    const auto man = b.manifold(text(kern, "to_manifold_id"));
    need(man.has_value(), "cellquadrature: " + kname + ": its surface is not stored");
    const SurfaceInput S = b.surface(text(*man, "name"), true);
    need(jj["surface"].get<std::string>() == S.name, "cellquadrature: " + kname + " maps to another surface than the joint's (" + jj["surface"].get<std::string>() + ")");
    const std::int64_t nV = S.n_vertices();
    const SurfaceLevel SL = b.surface_level(S, l);
    need(!jj.contains("space") || SL.level.id == jj["space"].get<std::string>(), "cellquadrature: the joint '" + text(*st.joint.row, "name") + "' was made on another tiling of level " + std::to_string(l) + " than the surface's now (make the joint again)");
    const std::int64_t tiles = SL.nodes;
    need(!nxr::io::is_null(kern, "from_selection_id"), "cellquadrature: " + kname + ": its channels (from_selection) are not stored");
    const auto ksel = b.selection(text(kern, "from_selection_id"));
    need(ksel.has_value() && !nxr::io::is_null(*ksel, "array_path"), "cellquadrature: " + kname + ": its channels (from_selection) are not stored");
    const std::vector<std::int32_t> chans = b.array_int32(text(*ksel, "array_path"));
    const std::int64_t m = static_cast<std::int64_t>(chans.size());
    const double dt = 1.0 / st.recording.rate, t0 = st.recording.origin;
    std::vector<std::pair<double, double>> spans;
    if (st.bursts) {
      spans = b.spans(text(*st.bursts->row, "id"));
      need(text(*st.bursts->row, "type") == "spans", "cellquadrature: " + text(*st.bursts->row, "name") + " is a " + text(*st.bursts->row, "type") + " Selection, not spans (events with a start and a stop)");
    }
    const SpatialBandOptions bo = band_options(p);
    need(!(bo.transform == "exact" && nV > 4096), "cellquadrature: transform exact is the dense eigen-decomposition: at most 4096 vertices, this surface has " + std::to_string(nV) + "; use chebyshev");
    if (ctx.say) ctx.say(Json{{"stage", "prepared"}, {"cross_spectrum", csname}, {"recording", recording}, {"joint", text(*st.joint.row, "name")}, {"tile_level", window_level}, {"tiles", tiles}, {"time_tiles", frames}, {"channels", m},
                              {"bursts", st.bursts ? Json(static_cast<std::int64_t>(spans.size())) : Json(nullptr)}, {"weights", weights}, {"tile_pairs", tile_pairs ? Json(static_cast<std::int64_t>(SL.pairs.size())) : Json(nullptr)}});
    // LOADING: the tiles of the cross-spectrum on the kernel's channels, in its order (refused for a channel it does not cover)
    const CrossSpectrumWindows win = b.cross_spectrum_windows(csname, chans);
    need(win.level == window_level, "cellquadrature: " + csname + ": read at level " + std::to_string(win.level) + ", recorded at " + std::to_string(window_level));
    need(!win.sum.empty(), "cellquadrature: " + csname + ": it stores no tile");
    for (const std::int64_t k : win.index) need(k >= 0 && k < frames, "cellquadrature: " + csname + ": a stored tile (" + std::to_string(k) + ") is outside the joint's " + std::to_string(frames) + " time tiles");
    // M_l = (F_l (x) I) K on the kernel's channels, bound ONCE (the level's band applied to every component of the kernel's rows; a linear map before any non-linear step)
    std::vector<std::int64_t> kshape;
    const std::vector<float> kf = b.array_float(text(kern, "path"), &kshape);
    need(kshape.size() == 2 && kshape[1] == m, "cellquadrature: " + kname + ": its array is not [rows x " + std::to_string(m) + " channels]");
    need(kshape[0] % nV == 0, "cellquadrature: " + kname + ": its " + std::to_string(kshape[0]) + " rows are not a whole number of components for the " + std::to_string(nV) + " vertices of " + S.name);
    const std::int64_t comps = kshape[0] / nV;
    const SpatialBand band = b.make_spatial_band(S, SL, bo);
    Eigen::MatrixXd A(kshape[0], m);
    for (std::int64_t comp = 0; comp < comps; ++comp) {
      Eigen::MatrixXd Kc(nV, m);
      for (std::int64_t v = 0; v < nV; ++v) for (std::int64_t j = 0; j < m; ++j) Kc(v, j) = kf[static_cast<std::size_t>((v * comps + comp) * m + j)];
      const Eigen::MatrixXd Mc = band.apply(Kc);
      for (std::int64_t v = 0; v < nV; ++v) A.row(v * comps + comp) = Mc.row(v);
    }
    std::vector<int> row_tile(static_cast<std::size_t>(kshape[0]));
    for (std::int64_t v = 0; v < nV; ++v) for (std::int64_t c = 0; c < comps; ++c) row_tile[static_cast<std::size_t>(v * comps + c)] = SL.tile[static_cast<std::size_t>(v)];
    // the area weights: a row (vertex, component) weighs its vertex's barycentric area (the tiles' areas are sums of these); none: no weights
    Eigen::VectorXd row_w;
    if (weights == "area") {
      nxr::select::Mesh whole; whole.n = static_cast<int>(nV); whole.positions = S.vertices; whole.faces = S.faces;
      const std::vector<double> va = nxr::select::vertexAreas(whole);
      row_w.resize(kshape[0]);
      for (std::int64_t v = 0; v < nV; ++v) for (std::int64_t c = 0; c < comps; ++c) row_w[v * comps + c] = va[static_cast<std::size_t>(v)];
    }
    if (ctx.say) ctx.say(Json{{"stage", "spatial_band"}, {"transform", band.info.value("transform", std::string())}, {"source", band.info.value("source", std::string())}, {"rows", kshape[0]}, {"components", comps}});
    Json warnings = Json::array();
    for (const auto& w : band.warnings) warnings.push_back(w);
    const Json recipe = Json{{"pipeline", "cellquadrature"}, {"cross_spectrum", csname}, {"joint", text(*st.joint.row, "name")}, {"kernel", kname}, {"spatial_scale", Json{{"stated", p["spatial_scale"]}, {"level", l}, {"snapped", scale_snapped}}}, {"window_level", window_level}, {"spatial_band", band.info}, {"weights", weights},
                             {"note", "T = B C B^T per spatial tile of the level; B the tile's rows of the kernel with the level's Weyl band, all components; stored: sum |T|^2 and sum |Im T|^2; the quadrature fraction is derived at read; weights: " + weights +
                                          " (area: sums of a_i a_j |T_ij|^2 with a the vertex's barycentric area; none: every vertex counts equally)"}};
    // THE GROUPING IS A SELECTION in its resolved form (spec 2026-11-17-selection-module, section 8): the spatial tiles of the level (a GRID: a level of the surface's dyadic grid) as the member number of each row
    nxr::selection::MemberOfElement tile_members;
    tile_members.members = tiles;
    tile_members.member_of.assign(row_tile.begin(), row_tile.end());
    struct Norms { Eigen::MatrixXd total, quad, ptotal, pquad; };
    auto norms_of = [&](const std::vector<Eigen::MatrixXcd>& Cs) {
      Norms n;
      if (tile_pairs) {
        const auto r = nxr::measurement::pairCongruenceNorms(A, tile_members, SL.pairs, Cs, row_w);
        n.total = r.tiles.total; n.quad = r.tiles.quadrature; n.ptotal = r.pairTotal; n.pquad = r.pairQuadrature;
      } else {
        const auto r = nxr::measurement::groupedCongruenceNorms(A, tile_members, Cs, row_w);
        n.total = r.total; n.quad = r.quadrature;
      }
      return n;
    };
    // [rows x frames] with the stored tiles at their ordinals, NaN elsewhere
    auto place = [&](const Eigen::MatrixXd& mm) {
      Eigen::MatrixXd o = Eigen::MatrixXd::Constant(mm.rows(), frames, kNaN);
      for (std::size_t w = 0; w < win.sum.size(); ++w) o.col(win.index[w]) = mm.col(static_cast<Eigen::Index>(w));
      return o;
    };
    // [rows x members x 2] as one array: the two ingredients of a product, a measurement each
    auto pack = [&](const Eigen::MatrixXd& total, const Eigen::MatrixXd& quad, const char* mt, const char* mq, const std::string& over_name, const std::string& what) {
      const std::int64_t R = total.rows(), N = total.cols();
      Array a;
      a.rows = R; a.cols = N * 2;
      a.data.resize(static_cast<std::size_t>(R * N * 2));
      for (std::int64_t r = 0; r < R; ++r) for (std::int64_t n = 0; n < N; ++n) { a.data[static_cast<std::size_t>((r * N + n) * 2)] = total(r, n); a.data[static_cast<std::size_t>((r * N + n) * 2 + 1)] = quad(r, n); }
      a.meta["describe"] = Json{{"shape", Json::array({R, N, 2})}, {"precision", "float64"},
                                {"measures", Json::array({Json{{"measure", mt}, {"unit", ""}}, Json{{"measure", mq}, {"unit", ""}}})},
                                {"description", what + " of " + over_name + " (" + csname + ", " + kname + ", spatial level " + std::to_string(l) + "): " + mt + " and " + mq + ", [" + std::to_string(R) + " rows x " + std::to_string(N) + " members]"}, {"producer", recipe}, {"warnings", warnings}};
      return a;
    };
    Arrays out;
    // the cells: per spatial tile (and tile pair) and time tile
    {
      const Norms n = norms_of(win.sum);
      out["cells"] = pack(place(n.total), place(n.quad), kTotal, kQuad, text(*st.joint.row, "name"), "the quadrature ingredients per spatial tile and time tile");
      if (tile_pairs) out["cell_pairs"] = pack(place(n.ptotal), place(n.pquad), kPairTotal, kPairQuad, text(*st.joint.row, "name"), "the quadrature ingredients per pair of touching tiles (nxr::select::tileAdjacency order) and time tile");
      if (ctx.say) ctx.say(Json{{"stage", "cells"}, {"rows", tiles}, {"members", frames}});
    }
    // the bursts: C summed over each burst's time tiles FIRST (products add: exact), then the ingredients per spatial tile (and pair)
    if (st.bursts) {
      const std::int64_t B = static_cast<std::int64_t>(spans.size());
      const double eps = 0.5 * dt, width = static_cast<double>(window_samples) * dt;
      std::map<std::int64_t, std::size_t> at;                                    // tile ordinal -> position in win
      for (std::size_t w = 0; w < win.index.size(); ++w) at[win.index[w]] = w;
      std::vector<Eigen::MatrixXcd> sums(static_cast<std::size_t>(B));
      Json report = Json::array();
      for (std::int64_t k = 0; k < B; ++k) {
        const double a0 = spans[static_cast<std::size_t>(k)].first, z0 = spans[static_cast<std::size_t>(k)].second;
        std::int64_t k0 = static_cast<std::int64_t>(std::ceil((a0 - t0 - eps) / width)), k1 = static_cast<std::int64_t>(std::floor((z0 - t0 + eps) / width)) - 1;   // the tiles WHOLLY inside the event
        k0 = std::max<std::int64_t>(k0, 0);
        std::vector<Eigen::MatrixXcd> parts;
        std::int64_t absent = 0;
        for (std::int64_t t = k0; t <= k1; ++t) { const auto it = at.find(t); if (it == at.end()) ++absent; else parts.push_back(win.sum[it->second]); }
        const bool whole = k1 >= k0 && absent == 0;
        if (whole) {                                                          // the event's value is the SUM of its whole tiles' sums (the merge rule `sum`: additive ingredients); an event with a missing tile stays empty (NaN)
          Eigen::MatrixXcd total = parts.front();
          for (std::size_t q = 1; q < parts.size(); ++q) total += parts[q];
          sums[static_cast<std::size_t>(k)] = std::move(total);
        }
        report.push_back(Json{{"burst", k}, {"start", a0}, {"stop", z0}, {"tiles", k1 >= k0 ? k1 - k0 + 1 : 0}, {"first_tile", k0}, {"missing_tiles", absent}, {"complete", whole}});
      }
      const Norms n = norms_of(sums);
      const std::string bname = text(*st.bursts->row, "name");
      out["bursts"] = pack(n.total, n.quad, kTotal, kQuad, bname, "the quadrature ingredients per spatial tile and burst");
      if (tile_pairs) out["burst_pairs"] = pack(n.ptotal, n.pquad, kPairTotal, kPairQuad, bname, "the quadrature ingredients per pair of touching tiles (nxr::select::tileAdjacency order) and burst");
      if (ctx.say) ctx.say(Json{{"stage", "bursts"}, {"events_read", report}});
    }
    for (const char* o : {"cells", "cell_pairs", "bursts", "burst_pairs"}) if (!out.count(o)) { Array none; none.meta["absent"] = true; out[o] = std::move(none); }   // a port this run makes none of (no pairs, no bursts)
    return out;
  };
  return s;
}

// ── patches ──────────────────────────────────────────────────────────────────────────────────────────────────────
// PATCHES ON A STATIC CORTICAL MAP (the hand-wired `patches compute` it replaces; spec docs/specs/2026-11-13-cortical-events.md, phase 1): the map filtered to a level's Weyl band (the band, a field), the band's
// mean / max / min / abs_max per tile of the level (barycentric weights), the median and robust SD of the tile means (the baseline), the regions of the signed tile means that reach median + enter_sd SDs and
// extend over tiles at median + exit_sd SDs (hysteresis over the tile graph: the patches, a set Selection), the band's values over the patches, the patches' own measures (area, tiles, centre), and the
// summaries across the patches of each. It builds no instrument limit into the detection: the caller's min_area, named. It does not merge patches that do not touch.
struct PatchState { Value map, tiles; Json p; };
StepSpec patches_step() {
  StepSpec s;
  s.name = "select.patch_summaries";
  s.kind = "setup";
  s.inputs = {"map", "tiles"};
  s.outputs = {"band", "tile_measures", "baseline", "patches", "patch_measures", "patch_own", "area", "n_tiles", "mean", "peak"};
  s.parameters = Json{{"spatial_scale", nullptr}, {"sign", "positive"}, {"enter_sd", 1.5}, {"exit_sd", 1.0}, {"min_area", nullptr}, {"max_area", nullptr}, {"transform", "auto"}, {"modes", nullptr}, {"order", nullptr}, {"tolerance", nullptr},
                      {"summaries", Json::array({"count", "sum", "median", "mean", "sd", "percentile:25", "percentile:75", "percentile:90"})}};
  s.matches = "nxr patches; compute nxr::select::detectRegions, nxr::select::median/mad, nxr::select";
  s.description = "the patches of a scalar map: its band at a level of the surface's ladder, the tiles' measurements and baseline, the regions above it, their measurements and summaries";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    need(ins.count("map") && ins.at("map").row, "patches: map is the stored Field to find patches in");
    need(ins.count("tiles") && ins.at("tiles").row, "patches: tiles is the level's partition (the surface's ladder: `tiles` first)");
    need(p["spatial_scale"].is_string(), "patches: spatial_scale is a quantity (a tile diameter \"20 mm\", \"parts 16\", a wavelength): the scale whose Weyl band the map is filtered to and whose tiles are the partition; a bare level index is never a parameter");
    try { (void)nxr::select::parseSign(p.value("sign", std::string("positive"))); } catch (const std::invalid_argument& e) { fail(std::string("patches: ") + e.what()); }
    need(p.value("exit_sd", 1.0) <= p.value("enter_sd", 1.5), "patches: exit_sd (" + std::to_string(p.value("exit_sd", 1.0)) + ") is at most enter_sd (" + std::to_string(p.value("enter_sd", 1.5)) + ")");
    { const auto lo = area_m2_of(p["min_area"], "patches", "min_area"), hi = area_m2_of(p["max_area"], "patches", "max_area");
      need(!lo || *lo >= 0, "patches: min_area is at least 0"); need(!hi || *hi >= lo.value_or(0.0), "patches: max_area is at least min_area"); }
    const std::string t = p.value("transform", std::string("auto"));
    need(t == "auto" || t == "eigen" || t == "chebyshev" || t == "exact", "patches: transform is auto (eigenmodes first, chebyshev an announced fallback), eigen, chebyshev or exact");
    need(p["order"].is_null() || p["order"].get<int>() >= 0, "patches: order is at least 0 (0: chosen from the tolerance)");
    need(p["tolerance"].is_null() || p["tolerance"].get<double>() > 0, "patches: tolerance is above 0");
    try { (void)nxr::stats::parseSummaries(p["summaries"].get<std::vector<std::string>>()); } catch (const std::exception& e) { fail(std::string("patches: ") + e.what()); }
    Value band = value_of_class("field");
    band.manifolds = ins.at("map").manifolds;
    return Values{{"band", band}, {"tile_measures", value_of_class("measurement")}, {"baseline", value_of_class("statistic")}, {"patches", value_of_class("selection")}, {"patch_measures", value_of_class("measurement")},
                  {"patch_own", value_of_class("measurement")}, {"area", value_of_class("statistic")}, {"n_tiles", value_of_class("statistic")}, {"mean", value_of_class("statistic")}, {"peak", value_of_class("statistic")}};
  };
  s.resolve = [](const Values& ins, const Json& p, const Bound& bound) -> Json {
    if (!ins.count("map") || ins.at("map").manifolds.empty() || p["spatial_scale"].is_null()) return nullptr;
    const auto man = bound.manifold(ins.at("map").manifolds[0]);
    if (!man) return nullptr;
    const SurfaceInput S = bound.surface(text(*man, "name"), true);
    return bound.plan_spatial_band(S, bound.surface_level(S, level_of_ladder(bound, ins.at("tiles"), p["spatial_scale"], "patches")), band_options(p)).key;
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext&) { return std::any(PatchState{ins.at("map"), ins.at("tiles"), p}); };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const PatchState& st = std::any_cast<const PatchState&>(state);
    const Json& p = st.p;
    const Bound& b = ctx.bound;
    Json scale_snapped;
    const int l = level_of_ladder(b, st.tiles, p["spatial_scale"], "patches", &scale_snapped);
    const std::string sign_name = p.value("sign", std::string("positive")), mapname = text(*st.map.row, "name");
    const nxr::select::Sign sign = nxr::select::parseSign(sign_name);
    const double enter_sd = p.value("enter_sd", 1.5), exit_sd = p.value("exit_sd", 1.0);
    const std::optional<double> min_area = area_m2_of(p["min_area"], "patches", "min_area"), max_area = area_m2_of(p["max_area"], "patches", "max_area");
    // the map: a scalar value per vertex of a surface, every one finite
    need(nxr::io::opt_text(*st.map.row, "value_type").value_or("") == "scalar", "patches: " + mapname + ": patches are found in a scalar map (one value per vertex)");
    const auto man = b.manifold(text(*st.map.row, "manifold_id"));
    need(man.has_value() && text(*man, "type") == "surface", "patches: " + mapname + ": its manifold is not a surface (a recording, a sensor array or a frequency axis has no tiles)");
    const SurfaceInput S = b.surface(text(*man, "name"), true);
    const std::int64_t nV = S.n_vertices();
    need(!(p.value("transform", std::string("auto")) == "exact" && nV > 4096), "patches: transform exact is the dense eigen-decomposition: at most 4096 vertices, this surface has " + std::to_string(nV) + "; use chebyshev");
    std::vector<std::int64_t> shape;
    const std::vector<double> xv = b.array_double(text(*st.map.row, "path"), &shape);
    need(static_cast<std::int64_t>(xv.size()) == nV, "patches: " + mapname + ": not one scalar value per vertex of " + S.name + " (" + std::to_string(nV) + ")");
    std::int64_t nonfinite = 0;
    for (const double v : xv) if (!std::isfinite(v)) ++nonfinite;
    need(nonfinite == 0, "patches: " + mapname + ": " + std::to_string(nonfinite) + " vertices have no value (NaN or infinite); a band-pass of a map with holes is not defined: fill or mask them first");
    const Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(xv.data(), static_cast<Eigen::Index>(nV));
    const SurfaceLevel SL = b.surface_level(S, l);                   // the ladder, the level, its tiles and graph, the Weyl pairing checked
    need(SL.ladder.id == text(*st.tiles.row, "id"), "patches: the tiles found are not the grid of " + S.name + " (the ladder whose level " + std::to_string(l) + " is " + SL.tiling + ")");
    const std::int64_t nodes = SL.nodes;
    const nxr::signal::graphfilterbank::Member& member = SL.bank.members[static_cast<std::size_t>(l)];
    nxr::select::Mesh whole; whole.n = static_cast<int>(nV); whole.positions = S.vertices; whole.faces = S.faces;
    const std::vector<double> vertex_area = nxr::select::vertexAreas(whole);
    if (ctx.say) ctx.say(Json{{"stage", "prepared"}, {"map", mapname}, {"surface", S.name}, {"level", l}, {"tiles", nodes}, {"tile_pairs", SL.pairs.size()}, {"band_lambda", Json::array({member.lo, member.hi})}});
    Json warnings = Json::array();
    // the band-passed map
    SpatialBandOptions bo = band_options(p);
    const SpatialBand sb = b.make_spatial_band(S, SL, bo);
    const Eigen::VectorXd band = sb.apply(Eigen::MatrixXd(x)).col(0);
    for (const auto& w : sb.warnings) warnings.push_back(w);
    if (ctx.say) ctx.say(Json{{"stage", "spatial_band"}, {"transform", sb.info.value("transform", std::string())}, {"source", sb.info.value("source", std::string())}, {"order", sb.info.value("order", 0)}, {"series_error", sb.info.value("series_error", 0.0)}});
    Array band_a;
    band_a.rows = nV; band_a.cols = 1;
    band_a.data.assign(band.data(), band.data() + band.size());
    band_a.meta["describe"] = Json{{"value_type", "scalar"}, {"precision", "float64"}, {"shape", Json::array({nV})},
                                   {"description", "'" + mapname + "' filtered to the Weyl band of level " + std::to_string(l) + " of " + S.name + "'s tile ladder (graph wavelet, Meyer edges +-1/3 octave, on the cotangent Laplacian pencil of the intrinsic Delaunay mesh): signed, local to the band's wavelength; a tile mean of it is a height relative to the surroundings at that scale, not the map's own value"},
                                   {"unit", nxr::io::opt_text(*st.map.row, "unit") ? Json(*nxr::io::opt_text(*st.map.row, "unit")) : Json()}};
    // THE TILES' MEASUREMENTS of the band (barycentric weights): mean, max, min, abs_max
    const std::vector<std::int32_t>& tile = SL.tile;
    auto measure_over = [&](const std::vector<std::int32_t>& code, std::int64_t members, std::vector<std::array<double, 4>>& out) {
      out.assign(static_cast<std::size_t>(members), std::array<double, 4>{kNaN, kNaN, kNaN, kNaN});
      std::vector<double> sw(static_cast<std::size_t>(members), 0.0), sx(static_cast<std::size_t>(members), 0.0);
      for (std::int64_t v = 0; v < nV; ++v) {
        const std::int32_t c = code[static_cast<std::size_t>(v)];
        if (c < 0 || c >= members) continue;
        auto& o = out[static_cast<std::size_t>(c)];
        const double xv2 = band[v], w = vertex_area[static_cast<std::size_t>(v)];
        sw[static_cast<std::size_t>(c)] += w; sx[static_cast<std::size_t>(c)] += w * xv2;
        o[1] = std::isnan(o[1]) ? xv2 : std::max(o[1], xv2);
        o[2] = std::isnan(o[2]) ? xv2 : std::min(o[2], xv2);
        o[3] = std::isnan(o[3]) ? std::abs(xv2) : std::max(o[3], std::abs(xv2));
      }
      for (std::int64_t c = 0; c < members; ++c) if (sw[static_cast<std::size_t>(c)] > 0) out[static_cast<std::size_t>(c)][0] = sx[static_cast<std::size_t>(c)] / sw[static_cast<std::size_t>(c)];
    };
    const Json measures4 = Json::array({Json{{"measure", "mean"}, {"unit", ""}}, Json{{"measure", "max"}, {"unit", ""}}, Json{{"measure", "min"}, {"unit", ""}}, Json{{"measure", "abs_max"}, {"unit", ""}}});
    const std::string map_unit = nxr::io::opt_text(*st.map.row, "unit").value_or("");
    Json measures_u = Json::array();
    for (const auto& m4 : measures4) measures_u.push_back(Json{{"measure", m4["measure"]}, {"unit", map_unit}});
    auto four = [&](const std::vector<std::array<double, 4>>& v, const std::string& over_name, const Json& over_member) {
      Array a;
      a.rows = static_cast<std::int64_t>(v.size()); a.cols = 4;
      for (const auto& r : v) a.data.insert(a.data.end(), r.begin(), r.end());
      a.meta["describe"] = Json{{"measures", measures_u}, {"precision", "float64"}, {"over_member", over_member}, {"description", "the band's mean (area-weighted by the vertices' barycentric areas), max, min and abs_max over the members of " + over_name}};
      return a;
    };
    std::vector<std::array<double, 4>> tile_m;
    measure_over(tile, nodes, tile_m);
    Arrays out;
    out["band"] = std::move(band_a);
    out["tile_measures"] = four(tile_m, SL.tiling, SL.level.id);
    // THE BASELINE across the level's tiles: the median and the robust SD of the band's tile means
    Eigen::VectorXd mean = Eigen::VectorXd::Constant(nodes, kNaN);
    std::vector<double> tmean(static_cast<std::size_t>(nodes));
    for (std::int64_t t = 0; t < nodes; ++t) { mean(t) = tile_m[static_cast<std::size_t>(t)][0]; tmean[static_cast<std::size_t>(t)] = mean(t); }
    const std::vector<nxr::stats::Summary> bsums = nxr::stats::parseSummaries({"median", "mad"});
    const std::vector<double> bvals = nxr::stats::summarizeRows(tmean, 1, nodes, bsums, "each", 1, nullptr);
    need(bvals.size() == 2, "patches: expected a median and a robust SD of the tile means, found " + std::to_string(bvals.size()) + " values");
    double med = bvals[0], sd = bvals[1];
    {
      Array a;
      a.rows = 1; a.cols = 2; a.data = bvals;
      a.meta["describe"] = Json{{"measure", "mean"}, {"method", "summary"}, {"precision", "float64"}, {"unit", map_unit.empty() ? Json() : Json(map_unit)}, {"n_members", nodes},
                                {"axes", Json::array({Json{{"name", "row"}, {"size", 1}, {"values", Json::array({Json{{"name", "all"}}})}}, Json{{"name", "summary"}, {"size", 2}, {"values", Json::array({Json{{"name", "median"}}, Json{{"name", "mad"}}})}}})},
                                {"options", Json{{"summaries", Json::array({"median", "mad"})}, {"combine", "each"}, {"ddof", 1}}},
                                {"description", "the median and robust SD of the band's tile means over the tiles of " + SL.tiling + " (the baseline the patches' thresholds are relative to)"}};
      out["baseline"] = std::move(a);
    }
    Json baseline = Json{{"median", med}, {"sd", sd}, {"over", "the level's tiles, of the band's tile means"}};
    // the sign: the caller's, before the thresholds (a baseline of the SIGNED values)
    const Eigen::VectorXd signed_v = nxr::select::applySign(mean, sign);
    if (sign == nxr::select::Sign::Negative) med = -med;
    if (sign == nxr::select::Sign::Magnitude) {
      std::vector<double> a;
      for (Eigen::Index t = 0; t < signed_v.size(); ++t) if (std::isfinite(signed_v(t))) a.push_back(signed_v(t));
      const Eigen::Map<const Eigen::VectorXd> av(a.data(), static_cast<Eigen::Index>(a.size()));
      med = nxr::select::median(av); sd = nxr::select::mad(av);
      baseline["note"] = "magnitude: the median and robust SD are of the absolute tile means, computed with the same functions (the stored statistic is of the signed means)";
    }
    baseline["median_used"] = med; baseline["sd_used"] = sd;
    const double scale = x.cwiseAbs().maxCoeff();
    need(std::isfinite(sd) && sd > 1e-9 * scale, "patches: " + mapname + ": the robust SD of the level " + std::to_string(l) + " band's tile means is " + std::to_string(sd) + ", nothing beyond round-off (a constant map, or a level the map has nothing at): thresholds relative to it are not defined");
    const double enter = med + enter_sd * sd, exit = med + exit_sd * sd;
    // THE PATCHES: hysteresis over the tile graph
    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(SL.tile_area.data(), static_cast<Eigen::Index>(SL.tile_area.size()));
    nxr::select::Regions R;
    try {
      R = nxr::select::detectRegions(signed_v, SL.pairs, w, enter, exit, min_area.value_or(0.0), max_area.value_or(std::numeric_limits<double>::infinity()));
    } catch (const std::invalid_argument& e) { fail(std::string("patches: ") + e.what()); }
    const std::int64_t n_patches = static_cast<std::int64_t>(R.regions.size());
    std::vector<std::int32_t> pcode(static_cast<std::size_t>(nV), -1);
    for (std::int64_t v = 0; v < nV; ++v) { const std::int32_t t = tile[static_cast<std::size_t>(v)]; if (t >= 0) pcode[static_cast<std::size_t>(v)] = R.label[static_cast<std::size_t>(t)]; }
    std::vector<int> pcode_int(pcode.begin(), pcode.end());
    const std::vector<int> centres = nxr::select::tileCentres(S.vertices.data(), pcode_int, static_cast<int>(n_patches), vertex_area);
    Json entries = Json::array();
    for (std::int64_t k = 0; k < n_patches; ++k) entries.push_back("patch " + std::to_string(k));
    Json pparams = Json{{"pipeline", "patches"}, {"map", mapname}, {"level", l}, {"spatial_scale", Json{{"stated", p["spatial_scale"]}, {"snapped", scale_snapped}}}, {"tiling", SL.tiling}, {"band", sb.info}, {"sign", sign_name}, {"enter_sd", enter_sd}, {"exit_sd", exit_sd},
                        {"thresholds", Json{{"enter", enter}, {"exit", exit}, {"unit", "of the signed band-passed tile mean"}}}, {"min_area_m2", min_area ? Json(*min_area) : Json(nullptr)}, {"max_area_m2", max_area ? Json(*max_area) : Json(nullptr)},
                        {"baseline", baseline}, {"merge", "none"}, {"weight", "barycentric"}, {"graph", "tiles of the level joined by a mesh edge (stored adjacency), hemispheres never touch"},
                        {"patch_tiles", Json::array()}};
    for (std::int64_t k = 0; k < n_patches; ++k) pparams["patch_tiles"].push_back(Json{{"patch", k}, {"tiles", R.regions[static_cast<std::size_t>(k)].nodes}, {"peak_tile", R.regions[static_cast<std::size_t>(k)].peak}});
    {
      Array a;
      a.rows = nV; a.cols = 1;
      a.data.assign(pcode.begin(), pcode.end());
      a.meta["describe"] = Json{{"params", pparams}, {"entries", entries}, {"description", "the patches of '" + mapname + "' (" + sign_name + ") at level " + std::to_string(l) + " of " + S.name + "'s tile ladder (code = the patch's ordinal, by its lowest tile)"}};
      out["patches"] = std::move(a);
    }
    if (ctx.say) ctx.say(Json{{"stage", "detect_regions"}, {"patches", n_patches}, {"enter", enter}, {"exit", exit}});
    const std::string over_name = mapname + " patches L" + std::to_string(l) + " " + sign_name;
    if (n_patches > 0) {
      std::vector<std::array<double, 4>> patch_m;
      measure_over(pcode, n_patches, patch_m);
      out["patch_measures"] = four(patch_m, over_name, nullptr);
      // the patches' own measures: area (m2, the tiles' areas summed), tiles, centre vertex (a count and a place)
      Array own;
      own.rows = n_patches; own.cols = 3;
      for (std::int64_t k = 0; k < n_patches; ++k) {
        const auto& g = R.regions[static_cast<std::size_t>(k)];
        own.data.push_back(g.weight); own.data.push_back(static_cast<double>(g.nodes.size())); own.data.push_back(static_cast<double>(centres[static_cast<std::size_t>(k)]));
      }
      own.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "area"}, {"unit", "m2"}}, Json{{"measure", "n_tiles"}, {"unit", ""}}, Json{{"measure", "centre_vertex"}, {"unit", ""}}})}, {"precision", "float64"},
                                  {"description", "the patches' own measures: area (the tiles' areas summed), the number of tiles and the centre vertex"}};
      out["patch_own"] = std::move(own);
      // the summaries across the patches of area, tiles, mean and peak (max for positive, min for negative, abs_max for magnitude)
      const std::vector<nxr::stats::Summary> sums = nxr::stats::parseSummaries(p["summaries"].get<std::vector<std::string>>());
      Json sjson = Json::array();
      for (const auto& sm : sums) sjson.push_back(sm.name == "percentile" ? Json{{"name", "percentile"}, {"p", sm.p}} : Json{{"name", sm.name}});
      const int peak_col = sign == nxr::select::Sign::Positive ? 1 : sign == nxr::select::Sign::Negative ? 2 : 3;
      auto summarize = [&](const std::vector<double>& vals, const std::string& measure, const std::string& unit, const std::string& what) {
        const std::vector<double> sv = nxr::stats::summarizeRows(vals, 1, n_patches, sums, "each", 1, nullptr);
        Array a;
        a.rows = 1; a.cols = static_cast<std::int64_t>(sums.size());
        a.data = sv;
        a.meta["describe"] = Json{{"measure", measure}, {"method", "summary"}, {"precision", "float64"}, {"unit", unit.empty() ? Json() : Json(unit)}, {"n_members", n_patches},
                                  {"axes", Json::array({Json{{"name", "row"}, {"size", 1}, {"values", Json::array({Json{{"name", "all"}}})}}, Json{{"name", "summary"}, {"size", static_cast<std::int64_t>(sums.size())}, {"values", sjson}}})},
                                  {"options", Json{{"summaries", sjson}, {"combine", "each"}, {"ddof", 1}, {"own", what == "area" || what == "n_tiles"}}},
                                  {"description", "the " + measure + " over the members of " + over_name + ", summarised (each)"}};
        return a;
      };
      std::vector<double> va, vn, vm, vp;
      for (std::int64_t k = 0; k < n_patches; ++k) {
        va.push_back(R.regions[static_cast<std::size_t>(k)].weight);
        vn.push_back(static_cast<double>(R.regions[static_cast<std::size_t>(k)].nodes.size()));
        vm.push_back(patch_m[static_cast<std::size_t>(k)][0]);
        vp.push_back(patch_m[static_cast<std::size_t>(k)][static_cast<std::size_t>(peak_col)]);
      }
      out["area"] = summarize(va, "area", "m2", "area");
      out["n_tiles"] = summarize(vn, "n_tiles", "", "n_tiles");
      out["mean"] = summarize(vm, "mean", map_unit, "mean");
      out["peak"] = summarize(vp, peak_col == 1 ? "max" : peak_col == 2 ? "min" : "abs_max", map_unit, "peak");
    }
    for (const char* o : {"patch_measures", "patch_own", "area", "n_tiles", "mean", "peak"}) if (!out.count(o)) { Array none; none.meta["absent"] = true; out[o] = std::move(none); }   // no patch: nothing to measure or summarise
    return out;
  };
  return s;
}

}  // namespace

void append_patches_steps(std::vector<StepSpec>& r) {
  r.push_back(run_repeat_step());
  r.push_back(cellquadrature_step());
  r.push_back(patches_step());
}

}  // namespace nxr::pipelines::graph
