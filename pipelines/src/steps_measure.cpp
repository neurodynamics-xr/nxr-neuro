// nxr-pipelines — THE MEASUREMENT VERBS AS REGISTERED STEPS: what the hand-wired `measure compute` (over a time tiling, over a partition of a surface) and `scaleselection compute` computed, as steps that read through
// the run's bound services and hand back described arrays the definitions store (declared outputs: kind measurement, selection, statistic). Their maths is the pure functions of compute/ (nxr::select,
// nxr::select, nxr::select::scale_selection); the YAML definitions in pipelines/definitions/ (measure_over_time, measure, scaleselection) are the verbs.
//
//   (window_measures and partition_measures are retired: the one `reduce` node, measurement_nodes.cpp, over a time grid or a surface grid)
//   scale_selection    setup       a recording and the stored pairs of cortical_event_pairs  ->  events (a spans Selection), scale (their measurements), links (a statistic)
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>

#include <Eigen/Dense>

#include "nxr/select/extent.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/io/finder.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/registry.h"
#include "nxr/pipelines/select_nodes.h"
#include "nxr/quantity/quantity.h"
#include "nxr/select/grid.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/select/scale_selection.h"

namespace nxr::pipelines::graph {

using nxr::io::Json;
using nxr::io::text;

namespace {

void need(bool ok, const std::string& msg) { if (!ok) refuse(msg); }
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Value measurement_value(const Value& over, const std::string& measure, const std::vector<std::string>& manifolds) {
  Value v;
  v.cls = "measurement";
  v.manifolds = manifolds;
  v.facts = Json{{"measure", measure}, {"over", Json{{"selection", over.row ? text(*over.row, "name") : std::string("the partition")}}}};
  return v;
}
std::vector<std::string> strings_of(const Json& j, const std::string& what) {
  need(j.is_array(), what + " is a list of names");
  std::vector<std::string> out;
  for (const auto& x : j) { need(x.is_string(), what + " is a list of names"); out.push_back(x.get<std::string>()); }
  return out;
}

// ── scale_selection ─────────────────────────────────────────────────────────────────────────────────────────────
// EACH EVENT'S OWN SIZE AND RHYTHM across a declared set of (spatial level, time level) pairs (the hand-wired `scaleselection compute` it replaces; spec docs/specs/2026-11-13-cortical-events.md, "Phase 2e"). From STORED
// rows only, no recording pass: what `run cortical_event_pairs` stored for every pair of the product `space_levels` x `time_levels` (the cells' amplitude on the pair's joint Selection, the cells' per-tile baseline, the
// labelling of the pair's events on the cells), read by the runner (Bound::scale_inputs). z = (amplitude - median) / SD per tile, the standardization the extents were found on (nxr::select::standardize); events of
// different pairs are linked by the ladders' exact nesting in space and overlap in time (nxr::select::scale_selection::selectScale); the connected components are the cross-pair events: each one's response at every pair (its footprint),
// best pair, the response there.
//
//   events   a spans Selection of the cross-pair events on the recording's time line (their span in time; one event per cross-pair event)
//   scale    measurements of the recording over it, one value an event: scale_space_level, scale_time_level, scale_peak, scale_tile_area (m2), scale_band_lo, scale_band_hi (Hz), scale_pairs, scale_edge (a bit mask),
//            and the footprint scale_response_s<space level>_t<time level>, a column a pair (NaN where the event has no member at that pair)
//   links    a statistic of the recording [pairs x events of the pair]: the cross-pair event of each pair's event (NaN beyond a pair's events)
struct ScaleState {
  Value recording;
  Json p;
  ScaleInputs in;
  Json pairs_json = Json::array();
};

StepSpec scale_selection_step() {
  StepSpec s;
  s.name = "select.scale_selection";
  s.kind = "setup";
  s.inputs = {"recording"};
  s.outputs = {"events", "scale", "links"};
  s.parameters = Json{{"spatial_scales", nullptr}, {"bands", nullptr}, {"source", "cortical_event_pairs"}, {"response", "peak"}, {"min_link", 1}, {"surface", nullptr}};
  s.matches = "Rheome detect.blobscale (the scale of the largest normalised response); compute nxr::select::scale_selection::selectScale, nxr::select::standardize";
  s.description = "each event's own size and rhythm across the product of two lists of levels, from the stored cells, baselines and labellings of the pairs: the cross-pair events, each one's footprint and best pair";
  s.traits = nxr::graph::node_traits(nxr::graph::TimeBehaviour::pointwise, nxr::graph::Precision::float64, nxr::graph::OutputPrecision::float64, "in double from stored cells");
  s.plan = [](const Values& ins, const Json& p, const PlanContext&) {
    const Value& rec = ins.at("recording");
    need(rec.cls == "field" && rec.row && rec.time && rec.rate > 0, "scaleselection: the recording has no time line");
    for (const char* key : {"spatial_scales", "bands"}) {
      need(p[key].is_array() && !p[key].empty() && std::all_of(p[key].begin(), p[key].end(), [](const Json& e) { return e.is_string(); }),
           std::string("scaleselection: ") + key + " is a list of quantities (" + (std::string(key) == "bands" ? "bands \"8-12 Hz\"" : "tile diameters \"20 mm\", \"parts 16\", wavelengths") + "); the pair set is the product of the two lists; a bare level index is never a parameter");
      std::set<std::string> seen;
      for (const auto& x : p[key]) need(seen.insert(x.get<std::string>()).second, "scaleselection: " + x.dump() + " is listed twice");
    }
    try { (void)nxr::select::scale_selection::parseResponse(p["response"].get<std::string>()); } catch (const std::invalid_argument& e) { refuse(std::string("scaleselection: ") + e.what()); }
    need(p["min_link"].is_number_integer() && p["min_link"].get<int>() >= 1, "scaleselection: min_link is at least 1");
    Value ev;
    ev.cls = "selection";
    ev.manifolds = {*rec.time};
    Value links;
    links.cls = "statistic";
    return Values{{"events", ev}, {"scale", measurement_value(ev, "scale_selection", {*rec.time})}, {"links", links}};
  };
  s.prepare = [](const Values& ins, const Json& p, ExecContext& ctx) {
    auto st = std::make_shared<ScaleState>();
    st->recording = ins.at("recording");
    st->p = p;
    // the levels are RESOLVED from the quantities (compute's level_of_*, time_level_of_band): the space levels on the surface's tile ladder, the time levels from the bands' centres on the recording's rate
    std::vector<int> sl, tl;
    {
      const SurfaceInput S0 = ctx.bound.surface(p["surface"].is_string() ? std::optional<std::string>(p["surface"].get<std::string>()) : std::nullopt, true);
      for (const auto& q : p["spatial_scales"]) sl.push_back(level_of_surface(ctx.bound, S0, q, "scaleselection"));
      for (const auto& q : p["bands"]) {
        const nxr::quantity::Quantity bq = nxr::quantity::parse(q.get<std::string>());
        need(bq.unit == "Hz", "scaleselection: band " + q.dump() + " is in Hz");
        tl.push_back(nxr::select::time_level_of_band(bq.lo, bq.hi, st->recording.rate, nxr::select::kJointCycles));
      }
    }
    st->in = ctx.bound.scale_inputs(text(*st->recording.row, "id"), text(*st->recording.row, "name"), sl, tl, p["source"].get<std::string>(),
                                    p["surface"].is_string() ? std::optional<std::string>(p["surface"].get<std::string>()) : std::nullopt);
    return std::any(st);
  };
  s.record = [](const std::any& state) {
    const ScaleState& st = *std::any_cast<const std::shared_ptr<ScaleState>&>(state);
    Json pairs = Json::array();
    for (const auto& q : st.in.pairs)
      pairs.push_back(Json{{"pair", static_cast<std::int64_t>(pairs.size())}, {"space_level", q.space_level}, {"time_level", q.time_level}, {"window_samples", q.window}, {"band_hz", q.joint["band"]}, {"tiles", q.tiles},
                           {"time_tiles", q.frames}, {"events", q.events}, {"joint", Json{{"id", q.joint_id}, {"name", q.joint_name}}}, {"labelling", Json{{"id", q.labelling_id}, {"name", q.labelling_name}}}, {"baseline", q.baseline_name}});
    return Json{{"scale_pairs", pairs}, {"surface", st.in.surface}};
  };
  s.run = [](const std::any& state, const Arrays&, ExecContext& ctx) {
    const ScaleState& st = *std::any_cast<const std::shared_ptr<ScaleState>&>(state);
    const Json& p = st.p;
    const auto& pr = st.in.pairs;
    const std::int64_t P = static_cast<std::int64_t>(pr.size());
    const double dt = 1.0 / st.recording.rate, t0 = st.recording.origin;
    // each pair: the tiles' area, z = (amplitude - median) / SD per tile
    const SurfaceInput S = ctx.bound.surface(st.in.surface, true);
    std::vector<nxr::select::scale_selection::Pair> pairs;
    std::vector<double> tile_area;
    Json pairs_json = Json::array();
    std::set<int> levels_seen;
    std::set<std::int64_t> windows_seen;
    for (const ScalePairInput& q : pr) {
      const SurfaceLevel SL = ctx.bound.surface_level(S, q.space_level);
      need(SL.nodes == q.tiles, "scaleselection: level " + std::to_string(q.space_level) + " of " + st.in.surface + " has " + std::to_string(SL.nodes) + " tiles, the joint Selection '" + q.joint_name + "' records " + std::to_string(q.tiles));
      double area = 0;
      for (const double a : SL.tile_area) area += a;
      tile_area.push_back(area / static_cast<double>(SL.nodes));
      Eigen::MatrixXd amp(q.tiles, q.frames);
      for (std::int64_t a = 0; a < q.tiles; ++a) for (std::int64_t f = 0; f < q.frames; ++f) amp(a, f) = q.amplitude[static_cast<std::size_t>(a * q.frames + f)];
      Eigen::VectorXd med(q.tiles), sd(q.tiles);
      for (std::int64_t a = 0; a < q.tiles; ++a) { med(a) = q.median[static_cast<std::size_t>(a)]; sd(a) = q.sd[static_cast<std::size_t>(a)]; }
      nxr::select::scale_selection::Pair sp;
      sp.level = q.space_level; sp.window = q.window; sp.origin = 0; sp.events = static_cast<int>(q.events);
      sp.z = nxr::select::standardize(amp, med, sd);
      sp.label.resize(q.tiles, q.frames);
      for (std::int64_t a = 0; a < q.tiles; ++a) for (std::int64_t f = 0; f < q.frames; ++f) sp.label(a, f) = q.labels[static_cast<std::size_t>(a * q.frames + f)];
      pairs.push_back(std::move(sp));
      levels_seen.insert(q.space_level);
      windows_seen.insert(q.window);
      pairs_json.push_back(Json{{"pair", static_cast<std::int64_t>(pairs_json.size())}, {"space_level", q.space_level}, {"time_level", q.time_level}, {"window_samples", q.window}, {"band_hz", q.joint["band"]},
                                {"tiles", q.tiles}, {"time_tiles", q.frames}, {"tile_area_m2", tile_area.back()}, {"events", q.events}, {"labelling", q.labelling_name}, {"baseline", q.baseline_name}});
    }
    {   // a pair's time tiles are the joint's: two pairs of one window and one level would be the same pair
      std::set<std::pair<int, std::int64_t>> uniq;
      for (const ScalePairInput& q : pr) need(uniq.insert({q.space_level, q.window}).second, "scaleselection: two time levels have time tiles of " + std::to_string(q.window) + " samples (the same pair twice)");
    }
    const nxr::select::scale_selection::Response response = nxr::select::scale_selection::parseResponse(p["response"].get<std::string>());
    nxr::select::scale_selection::Selection sel;
    try { sel = nxr::select::scale_selection::selectScale(pairs, response, p["min_link"].get<int>()); } catch (const std::invalid_argument& e) { refuse(std::string("scaleselection: ") + e.what()); }
    const std::int64_t E = sel.events;
    need(E > 0, "scaleselection: no event in any pair: nothing is stored");
    const int min_level = *levels_seen.begin(), max_level = *levels_seen.rbegin();
    const std::int64_t min_window = *windows_seen.begin(), max_window = *windows_seen.rbegin();
    const Json warnings = Json::array({
        "the response is a STANDARDIZED amplitude (z against each tile's own history at that pair), comparable across pairs as such; a pair with more cells has more chances for a large noise z, so the peak is selection-biased towards fine levels and short windows (the `mean` response is the less biased alternative)",
        "the size and rhythm are the best pair of the DECLARED set, quantized to its levels and bands, not a fitted scale; a selection on the set's edge (scale_edge) may lie beyond it",
        "a coarse level's footprint is wide: its tiles hold the pattern of a neighbourhood (phase 2a); an event is placed to within the level's scale"});
    // THE EVENTS' SPANS and their measurements [events x (8 + pairs)]
    constexpr int K = 8;
    const std::size_t Cn = static_cast<std::size_t>(K + P);
    Array ev, sc, lk;
    Json spans = Json::array();
    sc.rows = E; sc.cols = static_cast<std::int64_t>(Cn);
    sc.data.assign(static_cast<std::size_t>(E) * Cn, kNaN);
    for (std::int64_t k = 0; k < E; ++k) {
      const int b = sel.best(k);
      double* e = sc.data.data() + static_cast<std::size_t>(k) * Cn;
      int edge = 0;
      if (b >= 0) {
        const ScalePairInput& q = pr[static_cast<std::size_t>(b)];
        if (levels_seen.size() >= 2) { if (q.space_level == max_level) edge |= 1; if (q.space_level == min_level) edge |= 2; }
        if (windows_seen.size() >= 2) { if (q.window == min_window) edge |= 4; if (q.window == max_window) edge |= 8; }
        e[0] = q.space_level; e[1] = q.time_level; e[2] = sel.peak(k); e[3] = tile_area[static_cast<std::size_t>(b)]; e[4] = q.joint["band"][0].get<double>(); e[5] = q.joint["band"][1].get<double>();
      }
      e[6] = sel.pairs(k); e[7] = edge;
      for (std::int64_t a = 0; a < P; ++a) e[K + a] = sel.response(a, k);
      spans.push_back(Json{{"start", t0 + sel.start(k) * dt}, {"stop", t0 + sel.stop(k) * dt}});
    }
    Json ms = Json::array({Json{{"measure", "scale_space_level"}}, Json{{"measure", "scale_time_level"}}, Json{{"measure", "scale_peak"}}, Json{{"measure", "scale_tile_area"}, {"unit", "m\xc2\xb2"}},
                           Json{{"measure", "scale_band_lo"}, {"unit", "Hz"}}, Json{{"measure", "scale_band_hi"}, {"unit", "Hz"}}, Json{{"measure", "scale_pairs"}}, Json{{"measure", "scale_edge"}}});
    for (const ScalePairInput& q : pr) ms.push_back(Json{{"measure", "scale_response_s" + std::to_string(q.space_level) + "_t" + std::to_string(q.time_level)}});
    const Json params = Json{{"scale", Json{{"pairs", pairs_json}, {"response", p["response"]}, {"min_link", p["min_link"]}, {"source", p["source"]}, {"order", "space level ascending, then time level ascending"}, {"events", E}, {"warnings", warnings}}}};
    const std::string desc = "the cross-pair events of " + text(*st.recording.row, "name") + " (one event seen at several level pairs, matched by the ladders' nesting): each one's span in time; its size and rhythm are the measurements scale_*";
    ev.rows = E; ev.cols = 1;
    ev.meta["describe"] = Json{{"spans", spans}, {"params", params}, {"description", desc}};
    sc.meta["describe"] = Json{{"measures", ms}, {"n_members", E}, {"description", "per cross-pair event: its best pair's size and rhythm, the pairs that hold it, and its response at every pair"}};
    // THE LINKS: the cross-pair event of each pair's event
    std::int64_t maxe = 0;
    for (const ScalePairInput& q : pr) maxe = std::max(maxe, q.events);
    lk.rows = P; lk.cols = maxe;
    lk.data.assign(static_cast<std::size_t>(P * maxe), kNaN);
    for (std::int64_t a = 0; a < P; ++a)
      for (std::int64_t e = 0; e < pr[static_cast<std::size_t>(a)].events; ++e) lk.data[static_cast<std::size_t>(a * maxe + e)] = sel.cross[static_cast<std::size_t>(a)][static_cast<std::size_t>(e)];
    Json prs = Json::array();
    for (const ScalePairInput& q : pr) prs.push_back(Json{{"name", "s" + std::to_string(q.space_level) + "_t" + std::to_string(q.time_level)}, {"labelling", q.labelling_name}});
    lk.meta["describe"] = Json{{"measure", "scale_event"}, {"method", "scale_selection"}, {"precision", "float64"}, {"n_members", E},
                               {"axes", Json::array({Json{{"name", "pair"}, {"size", P}, {"values", prs}}, Json{{"name", "event"}, {"size", maxe}}})},
                               {"options", Json{{"response", p["response"]}, {"min_link", p["min_link"]}, {"source", p["source"]}}}, {"producer", Json{{"pairs", pairs_json}, {"warnings", warnings}}},
                               {"description", "for each pair's event (a labelling's member), the cross-pair event it belongs to; NaN beyond the pair's events"}};
    return Arrays{{"events", std::move(ev)}, {"scale", std::move(sc)}, {"links", std::move(lk)}};
  };
  return s;
}

}  // namespace

void append_measure_steps(std::vector<StepSpec>& r) {
  r.push_back(scale_selection_step());
}

}  // namespace nxr::pipelines::graph
