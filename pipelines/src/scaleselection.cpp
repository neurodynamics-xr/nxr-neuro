// nxr-pipelines — scale selection (scaleselection.h). Preparation reads the stored nodes and type-checks them; the nodes are the pure functions of compute/ (nxr::extent::standardize, nxr::scale);
// the results go back to io as a spans Selection and measurements.
#include "nxr/pipelines/scaleselection.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>

#include "nxr/bst/entities.h"
#include "nxr/extent.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/finder.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/scale.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

template <class T> std::vector<T> read_array(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}
std::string made_by(const CacheKey& key) { return "pipeline " + key.pipeline + " \xc2\xb7 cache " + sha256_hex(key.json().dump()).substr(0, 16) + " \xc2\xb7 nxr-neuro " + NXR_NEURO_VERSION; }

// the stored node of one pair
struct PairRows {
  int space_level = 0, time_level = 0;
  Row joint, labelling;
  Json jj, lab_params;
  std::int64_t tiles = 0, frames = 0, window = 0, events = 0;
  std::string amp_path;
  std::int64_t amp_component = 0, amp_stride = 2;            // the amplitude's component of its (packed) array [tiles x frames x components]
  Row baseline;
  double tile_area = kNaN;
  std::vector<std::int32_t> labels;
  std::vector<float> amplitude;               // [tiles x frames x 2]
  Eigen::MatrixXd z;
};
std::string list_of(const std::vector<int>& v) { std::string s; for (const int x : v) s += (s.empty() ? "" : ",") + std::to_string(x); return s; }
}  // namespace

ScaleSelectionResult scaleselection_compute(DatasetWriter& ds, SubjectWriter& sub, const ScaleSelectionOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  // ── the request, checked before anything is read ──
  if (opt.recording.empty()) invalid("scaleselection: --recording names the recording whose pairs are selected over");
  if (opt.space_levels.empty()) invalid("scaleselection: --space-levels lists the surface ladder's levels (e.g. 4,5)");
  if (opt.time_levels.empty()) invalid("scaleselection: --time-levels lists the recording's constant-Q levels (the pair set is the product of the two lists)");
  for (const auto* v : {&opt.space_levels, &opt.time_levels}) {
    std::set<int> seen;
    for (const int x : *v) { if (x < 0) invalid("scaleselection: levels are 0, 1, 2, \xe2\x80\xa6"); if (!seen.insert(x).second) invalid("scaleselection: level " + std::to_string(x) + " is listed twice"); }
  }
  nxr::scale::Response response;
  try { response = nxr::scale::parseResponse(opt.response); } catch (const std::invalid_argument& e) { invalid(std::string("scaleselection: --") + e.what()); }
  if (opt.min_link < 1) invalid("scaleselection: --min-link is at least 1");
  const auto rec = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND kind = 'recording' AND status = 'complete'", {V(sub.id()), V(opt.recording)});
  if (!rec) throw Error(code::no_such_row, sub.name() + ": no recording named '" + opt.recording + "'");
  const std::string fid = text(*rec, "id");
  const nxr::io::finder::Value rv = nxr::io::finder::value_of(sub.db(), "field", *rec);
  if (!rv.time) invalid(opt.recording + ": it has no time line");
  const std::string tid = *rv.time;
  const auto ld = sub.db().one("SELECT d.spacing AS spacing, d.origin AS origin FROM geometry_dimension d JOIN manifold mm ON mm.geometry_id = d.geometry_id WHERE mm.id = ? AND d.ordinal = 0", {V(tid)});
  if (!ld || is_null(*ld, "spacing")) invalid(opt.recording + ": its time line is not a closed-form line (no sampling step)");
  const double dt = std::stod(text(*ld, "spacing")), t0 = is_null(*ld, "origin") ? 0.0 : std::stod(text(*ld, "origin"));

  // ── PREPARATION: every pair's rows, found and checked; every missing pair named ──
  std::vector<PairRows> pr;
  std::string missing_joint, missing_run;
  std::vector<Row> joints;
  for (const Row& r : sub.db().all("SELECT * FROM selection WHERE subject_id = ? AND type = 'set' AND status = 'complete' AND params_json LIKE '%\"joint\"%' ORDER BY name", {V(sub.id())})) joints.push_back(r);
  std::vector<Row> labellings;
  for (const Row& r : sub.db().all("SELECT * FROM selection WHERE subject_id = ? AND type = 'set' AND status = 'complete' AND params_json LIKE '%\"cells_of\"%' ORDER BY name", {V(sub.id())})) labellings.push_back(r);
  const auto stats = statistics_of(sub.db(), sub.id(), fid, std::string("amplitude"));
  for (const int l : opt.space_levels)
    for (const int m : opt.time_levels) {
      PairRows p;
      p.space_level = l; p.time_level = m;
      bool found = false;
      for (const Row& r : joints) {
        const Json jp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
        if (!jp.is_object() || !jp.contains("joint")) continue;
        const Json& j = jp["joint"];
        if (j.value("recording", std::string()) == opt.recording && j.value("space_level", -1) == l && j.value("time_level", -1) == m) { p.joint = r; p.jj = j; found = true; break; }
      }
      const std::string pair = "space level " + std::to_string(l) + " x time level " + std::to_string(m);
      if (!found) { missing_joint += (missing_joint.empty() ? "" : "; ") + pair + ": nxr-neuro joint compute --recording " + opt.recording + " --space-level " + std::to_string(l) + " --time-level " + std::to_string(m); continue; }
      const std::string jid = text(p.joint, "id");
      std::vector<Row> labs;
      for (const Row& r : labellings) {
        const Json lp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
        if (!lp.is_object() || !lp.contains("cells_of") || lp["cells_of"].value("selection_id", std::string()) != jid) continue;
        if (!lp.contains("pipeline") || !lp["pipeline"].is_object() || lp["pipeline"].value("name", std::string()) != opt.source) continue;
        labs.push_back(r);
      }
      if (labs.empty()) { missing_run += (missing_run.empty() ? "" : "; ") + pair; continue; }
      if (labs.size() > 1) { std::string n; for (const Row& r : labs) n += (n.empty() ? "" : ", ") + text(r, "name"); invalid("scaleselection: " + std::to_string(labs.size()) + " labellings of " + text(p.joint, "name") + " were made by " + opt.source + " (" + n + "): remove all but one"); }
      p.labelling = labs[0];
      p.lab_params = Json::parse(text(p.labelling, "params_json"));
      p.tiles = p.jj["tiles"].get<std::int64_t>(); p.frames = p.jj["frames"].get<std::int64_t>(); p.window = p.jj.value("window_samples", std::int64_t(0));
      if (p.window < 1) invalid("scaleselection: the joint Selection '" + text(p.joint, "name") + "' records no time tile width (made before `window_samples`: `joint compute --replace`)");
      p.events = integer(p.labelling, "n_members");
      if (is_null(p.labelling, "array_path")) invalid(text(p.labelling, "name") + ": it stores no array of cells");
      const auto am = sub.db().one("SELECT array_path, component FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'amplitude'", {V(jid), V(fid)});
      if (!am) { missing_run += (missing_run.empty() ? "" : "; ") + pair + " (no cell amplitude)"; continue; }
      p.amp_path = text(*am, "array_path");
      p.amp_component = nxr::io::integer(*am, "component");
      std::vector<Row> bs;
      for (const Row& r : stats) {
        if (text(r, "method") != "summary" || is_null(r, "partition_selection_id") || text(r, "partition_selection_id") != jid) continue;
        const Json pj = Json::parse(opt_text(r, "producer_json").value_or("{}"), nullptr, false);
        if (pj.is_object() && pj.contains("pipeline") && pj["pipeline"].is_object() && pj["pipeline"].value("name", std::string()) == opt.source) bs.push_back(r);
      }
      if (bs.empty()) { missing_run += (missing_run.empty() ? "" : "; ") + pair + " (no cell baseline)"; continue; }
      if (bs.size() > 1) invalid("scaleselection: " + std::to_string(bs.size()) + " baselines of " + text(p.joint, "name") + " were made by " + opt.source + ": remove all but one");
      p.baseline = bs[0];
      pr.push_back(std::move(p));
    }
  if (!missing_joint.empty()) invalid("scaleselection: " + opt.recording + " has no joint Selection for: " + missing_joint);
  if (!missing_run.empty())
    invalid("scaleselection: no stored cells, baseline and labelling made by " + opt.source + " for: " + missing_run + " (nxr-neuro run " + opt.source + " --set 'space_levels=[" + list_of(opt.space_levels) + "]' --set 'bands=[the bands of the time levels]' --chunks all)");
  // one surface and one product manifold for every pair
  const std::string surface_name = pr[0].jj["surface"].get<std::string>(), mid = text(pr[0].joint, "manifold_id");
  for (const PairRows& p : pr) {
    if (p.jj["surface"].get<std::string>() != surface_name) invalid("scaleselection: the joint Selections are on different surfaces (" + surface_name + ", " + p.jj["surface"].get<std::string>() + ")");
    if (text(p.joint, "manifold_id") != mid) invalid("scaleselection: the joint Selections are on different manifolds");
  }
  if (opt.surface && *opt.surface != surface_name) invalid("scaleselection: --surface " + *opt.surface + " is not the joint Selections' (" + surface_name + ")");
  const SurfaceInput S = read_surface(sub, surface_name, true);
  // read and check each pair's arrays; z = (amplitude − median) / SD per tile (the standardization the extents were found on)
  Json pairs_json = Json::array();
  std::vector<nxr::scale::Pair> sp;
  std::set<int> levels_seen;
  std::set<std::int64_t> windows_seen;
  for (PairRows& p : pr) {
    const SurfaceLevel SL = read_surface_level(sub, S, p.space_level);
    if (SL.nodes != p.tiles) invalid("scaleselection: level " + std::to_string(p.space_level) + " of " + surface_name + " has " + std::to_string(SL.nodes) + " tiles, the joint Selection '" + text(p.joint, "name") + "' records " + std::to_string(p.tiles));
    double area = 0;
    for (const double a : SL.tile_area) area += a;
    p.tile_area = area / static_cast<double>(SL.nodes);
    std::vector<std::int64_t> shape;
    p.labels = read_array<std::int32_t>(sub, text(p.labelling, "array_path"), &shape);
    if (shape != std::vector<std::int64_t>{p.tiles, p.frames}) invalid("scaleselection: the labelling '" + text(p.labelling, "name") + "' is not [" + std::to_string(p.tiles) + " tiles x " + std::to_string(p.frames) + " time tiles] (the joint Selection's cells)");
    p.amplitude = read_array<float>(sub, p.amp_path, &shape);
    if (shape.size() != 3 || shape[0] != p.tiles || shape[1] != p.frames || p.amp_component < 0 || p.amp_component >= shape[2]) invalid("scaleselection: the cell amplitude of '" + text(p.joint, "name") + "' is not [" + std::to_string(p.tiles) + " x " + std::to_string(p.frames) + " x components]");
    p.amp_stride = shape[2];
    const StoredStatistic bs = read_statistic(sub, text(p.baseline, "name"));
    int im = -1, is = -1;
    if (bs.axes.is_array() && bs.axes.size() == 2 && bs.axes[1].contains("values"))
      for (std::size_t k = 0; k < bs.axes[1]["values"].size(); ++k) {
        const Json& v = bs.axes[1]["values"][k];
        const std::string nm = v.is_string() ? v.get<std::string>() : v.value("name", std::string());
        if (nm == "median") im = static_cast<int>(k);
        if (nm == "mad") is = static_cast<int>(k);
      }
    if (im < 0 || is < 0 || bs.shape.size() != 2 || bs.shape[0] != p.tiles) invalid("scaleselection: the baseline '" + text(p.baseline, "name") + "' does not hold a median and a mad per spatial tile of '" + text(p.joint, "name") + "'");
    const std::int64_t K = bs.shape[1];
    Eigen::VectorXd med(p.tiles), sd(p.tiles);
    for (std::int64_t s = 0; s < p.tiles; ++s) { med(s) = bs.values[static_cast<std::size_t>(s * K + im)]; sd(s) = bs.values[static_cast<std::size_t>(s * K + is)]; }
    Eigen::MatrixXd amp(p.tiles, p.frames);
    for (std::int64_t s = 0; s < p.tiles; ++s) for (std::int64_t f = 0; f < p.frames; ++f) amp(s, f) = static_cast<double>(p.amplitude[static_cast<std::size_t>((s * p.frames + f) * p.amp_stride + p.amp_component)]);
    p.z = nxr::extent::standardize(amp, med, sd);
    for (const std::int32_t c : p.labels) if (c >= p.events) invalid("scaleselection: the labelling '" + text(p.labelling, "name") + "' labels an event beyond its " + std::to_string(p.events));
    levels_seen.insert(p.space_level);
    windows_seen.insert(p.window);
    pairs_json.push_back(Json{{"pair", static_cast<std::int64_t>(pairs_json.size())}, {"space_level", p.space_level}, {"time_level", p.time_level}, {"window_samples", p.window}, {"band_hz", p.jj["band"]}, {"tiles", p.tiles},
                              {"time_tiles", p.frames}, {"tile_area_m2", p.tile_area}, {"events", p.events}, {"joint", Json{{"id", text(p.joint, "id")}, {"name", text(p.joint, "name")}}},
                              {"labelling", Json{{"id", text(p.labelling, "id")}, {"name", text(p.labelling, "name")}}}, {"baseline", text(p.baseline, "name")}});
  }
  {   // a pair's time tiles are the joint's: two pairs of one window and one level would be the same pair
    std::set<std::pair<int, std::int64_t>> uniq;
    for (const PairRows& p : pr) if (!uniq.insert({p.space_level, p.window}).second) invalid("scaleselection: two time levels have time tiles of " + std::to_string(p.window) + " samples (the same pair twice)");
  }
  const std::int64_t P = static_cast<std::int64_t>(pr.size());

  // ── THE CACHE KEY: the options, every pair's identity (a Selection's modified time moves when measurements are written over it, so the identity is its id and a hash of its record) ──
  const std::string name = opt.name.value_or(opt.recording + "_scale_selection");
  const std::string path = node_path(sub.db(), mid, safe_name(name));
  CacheKey key;
  key.pipeline = "scaleselection compute";
  key.params = Json{{"source", opt.source}, {"response", opt.response}, {"min_link", opt.min_link}, {"space_levels", opt.space_levels}, {"time_levels", opt.time_levels}};
  Json ins = Json::array();
  for (const PairRows& p : pr) {
    const auto amp_by = sub.db().one("SELECT computed_by FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'amplitude'", {V(text(p.joint, "id")), V(fid)});
    ins.push_back(Json{{"joint", Json{{"id", text(p.joint, "id")}, {"layout", sha256_hex(text(p.joint, "params_json")).substr(0, 16)}}}, {"labelling", Json{{"id", text(p.labelling, "id")}, {"record", sha256_hex(text(p.labelling, "params_json")).substr(0, 16)}}},
                       {"amplitude", amp_by ? opt_text(*amp_by, "computed_by").value_or("") : std::string()}, {"baseline", Json{{"id", text(p.baseline, "id")}, {"record", sha256_hex(opt_text(p.baseline, "producer_json").value_or("")).substr(0, 16)}}}});
  }
  key.input = Json{{"recording", Json{{"id", fid}}}, {"pairs", ins}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string by = made_by(key);
  ScaleSelectionResult res;
  if (const auto hit = find_cached(sub.db(), sub.id(), "selection", key, path)) {
    res.hit = true;
    res.info = Json{{"selection", text(*hit, "name")}, {"id", text(*hit, "id")}, {"events", integer(*hit, "n_members")}, {"pairs", pairs_json}, {"cache", "hit"}};
    say(res.info);
    return res;
  }
  const auto have = sub.find("selection", path);
  if (have && !opt.replace) throw Error(code::already_exists, sub.name() + ": scaleselection: the Selection '" + name + "' exists, made with other inputs or options; pass --replace (or --name another)");
  // another job's rows over the pairs' labellings (scale_event) are refused too
  if (!opt.replace)
    for (const PairRows& p : pr)
      if (sub.db().one("SELECT 1 AS x FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'scale_event'", {V(text(p.labelling, "id")), V(fid)}))
        throw Error(code::already_exists, sub.name() + ": scaleselection: '" + text(p.labelling, "name") + "' already holds scale_event rows (another selection's); pass --replace");

  // ── THE GRAPH (pure nodes of compute/): events' facts, the links, the selection ──
  std::vector<nxr::scale::Pair> pairs;
  for (const PairRows& p : pr) {
    nxr::scale::Pair q;
    q.level = p.space_level; q.window = p.window; q.origin = 0; q.events = static_cast<int>(p.events);
    q.z = p.z;
    q.label.resize(p.tiles, p.frames);
    for (std::int64_t s = 0; s < p.tiles; ++s) for (std::int64_t f = 0; f < p.frames; ++f) q.label(s, f) = p.labels[static_cast<std::size_t>(s * p.frames + f)];
    pairs.push_back(std::move(q));
  }
  nxr::scale::Selection sel;
  try { sel = nxr::scale::selectScale(pairs, response, opt.min_link); } catch (const std::invalid_argument& e) { invalid(std::string("scaleselection: ") + e.what()); }
  const std::int64_t E = sel.events;
  // the edge flags: only for a dimension with at least two values
  const int min_level = *levels_seen.begin(), max_level = *levels_seen.rbegin();
  const std::int64_t min_window = *windows_seen.begin(), max_window = *windows_seen.rbegin();
  Json warnings = Json::array({
      "the response is a STANDARDIZED amplitude (z against each tile's own history at that pair), comparable across pairs as such; a pair with more cells has more chances for a large noise z, so the peak is selection-biased towards fine levels and short windows (the `mean` response is the less biased alternative)",
      "the size and rhythm are the best pair of the DECLARED set, quantized to its levels and bands, not a fitted scale; a selection on the set's edge (scale_edge) may lie beyond it",
      "a coarse level's footprint is wide: its tiles hold the pattern of a neighbourhood (phase 2a); an event is placed to within the level's scale"});
  say(Json{{"stage", "prepared"}, {"recording", opt.recording}, {"pairs", P}, {"events_in_pairs", [&] { std::int64_t n = 0; for (const PairRows& p : pr) n += p.events; return n; }()}, {"cross_pair_events", E}, {"cache", "miss"}});

  // ── READBACK ──
  Json table = Json::array();
  constexpr int K = 8;
  std::vector<double> ev(static_cast<std::size_t>(E * K), kNaN), foot(static_cast<std::size_t>(P * E), kNaN);
  std::vector<nxr::bst::TimeSpan> spans;
  for (std::int64_t k = 0; k < E; ++k) {
    const int b = sel.best(k);
    double* e = ev.data() + k * K;
    int edge = 0;
    if (b >= 0) {
      const PairRows& p = pr[static_cast<std::size_t>(b)];
      if (levels_seen.size() >= 2) { if (p.space_level == max_level) edge |= 1; if (p.space_level == min_level) edge |= 2; }
      if (windows_seen.size() >= 2) { if (p.window == min_window) edge |= 4; if (p.window == max_window) edge |= 8; }
      e[0] = p.space_level; e[1] = p.time_level; e[2] = sel.peak(k); e[3] = p.tile_area; e[4] = p.jj["band"][0].get<double>(); e[5] = p.jj["band"][1].get<double>();
    }
    e[6] = sel.pairs(k); e[7] = edge;
    for (std::int64_t a = 0; a < P; ++a) foot[static_cast<std::size_t>(a * E + k)] = sel.response(a, k);
    spans.push_back({t0 + sel.start(k) * dt, t0 + sel.stop(k) * dt, 0});
    if (E <= 200) table.push_back(Json{{"event", k}, {"best_pair", b}, {"space_level", std::isnan(e[0]) ? Json() : Json(e[0])}, {"time_level", std::isnan(e[1]) ? Json() : Json(e[1])}, {"peak", std::isnan(e[2]) ? Json() : Json(e[2])},
                                       {"band_hz", b >= 0 ? pr[static_cast<std::size_t>(b)].jj["band"] : Json()}, {"pairs", sel.pairs(k)}, {"edge", edge}, {"start_s", spans.back().start}, {"stop_s", spans.back().stop}, {"members", sel.members[static_cast<std::size_t>(k)].size()}});
  }
  res.info = Json{{"selection", name}, {"recording", opt.recording}, {"surface", surface_name}, {"pairs", pairs_json}, {"response", opt.response}, {"min_link", opt.min_link}, {"events", E}, {"warnings", warnings}, {"cache", "miss"}};
  if (E <= 200) res.info["table"] = table;
  if (E == 0) { res.info["note"] = "no event in any pair: nothing is stored (a rerun recomputes)"; say(res.info); return res; }
  const std::string producer = internal::producer_string("scaleselection compute");
  bool replaced = false;
  sub.compose([&] {
    if (have) { clear_stored(ds, sub, {{"selection", path}}, true, "scaleselection " + name, replaced); }
    for (const PairRows& p : pr) delete_measurements(sub.db(), Json{{"selectionId", text(p.labelling, "id")}, {"ofFieldId", fid}, {"measure", "scale_event"}});
    // the recipe (every option, the inputs, the pair table) as an entry of the dataset's `recipes` dictionary
    DictionaryEntry de;
    de.name = by.substr(0, by.find(" \xc2\xb7 nxr-neuro"));
    de.cols.set("function", std::string("pipeline"));
    de.cols.set("params_json", jdump(Json{{"pipeline", "scaleselection compute"}, {"producer", producer}, {"parameters", key.params}, {"inputs", key.input}, {"pairs", pairs_json}, {"warnings", warnings}, {"cache", key.json()}, {"nxr_neuro", NXR_NEURO_VERSION},
                                         {"note", "events of different pairs linked by the ladders' nesting in space and overlap in time (>= min_link cell pairs); the best pair is the pair of the largest response (peak or mean standardized cell amplitude)"}}));
    DictionarySpec ds_spec;
    ds_spec.name = "recipes"; ds_spec.scope = "dataset"; ds_spec.description = "the recipes of this dataset: one entry a version, the recipe in params_json"; ds_spec.entries = std::vector<DictionaryEntry>{de};
    sub.dictionary(ds_spec);
    nxr::bst::TimeSpansOptions so;
    so.line_id = tid; so.of_field_id = fid;
    if (const auto aid = opt_text(*rec, "acquisition_id")) so.acquisition_id = *aid;
    so.description = "the cross-pair events of " + opt.recording + " (one event seen at several level pairs, matched by the ladders' nesting): each one's span in time; its size and rhythm are the measurements scale_*";
    so.params = Json{{"scale", Json{{"pairs", pairs_json}, {"response", opt.response}, {"min_link", opt.min_link}, {"source", opt.source}, {"order", "space level ascending, then time level ascending"}, {"events", E}, {"warnings", warnings}}},
                     {"pipeline", "scaleselection compute"}, {"producer", producer}, {"cache", key.json()}};
    const std::string sid = nxr::bst::time_spans(sub, path, spans, {"scale_event"}, so);
    auto write = [&](const std::string& selection, const std::string& stem, const std::vector<double>& data, std::int64_t R, std::int64_t M, std::int64_t C, const std::vector<std::string>& measures) {
      const std::string ap = text(*rec, "path") + "_" + safe_name(stem);
      sub.write_array(ap, view(data, {R, M, C}));
      Json rows = Json::array();
      for (std::size_t c = 0; c < measures.size(); ++c) rows.push_back(Json{{"selection_id", selection}, {"of_field_id", fid}, {"measure", measures[c]}, {"array_path", ap}, {"component", static_cast<std::int64_t>(c)}, {"computed_by", by}});
      write_measurements(sub.db(), rows, by);
    };
    write(sid, "scaleselection." + name + ".events", ev, 1, E, K, {"scale_space_level", "scale_time_level", "scale_peak", "scale_tile_area", "scale_band_lo", "scale_band_hi", "scale_pairs", "scale_edge"});
    write(sid, "scaleselection." + name + ".response", foot, P, E, 1, {"scale_response"});
    for (std::size_t a = 0; a < pr.size(); ++a) {
      const PairRows& p = pr[a];
      if (p.events == 0) continue;
      std::vector<double> cx(static_cast<std::size_t>(p.events));
      for (std::int64_t e = 0; e < p.events; ++e) cx[static_cast<std::size_t>(e)] = sel.cross[a][static_cast<std::size_t>(e)];
      write(text(p.labelling, "id"), "scaleselection." + name + ".event." + text(p.labelling, "name"), cx, 1, p.events, 1, {"scale_event"});
    }
    res.info["id"] = sid;
    res.info["path"] = path;
  });
  res.info["replaced"] = replaced;
  say(Json{{"stage", "stored"}, {"selection", name}, {"events", E}, {"replaced", replaced}});
  return res;
}

}  // namespace nxr::pipelines
