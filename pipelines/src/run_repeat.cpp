// nxr-pipelines — the run-to-run repeat (run_repeat.h).
#include "nxr/pipelines/run_repeat.h"

#include <cmath>
#include <set>

#include "nxr/crossings.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/cache.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }
// the stage names the rows carry (runner `store_crossings`: `... \xc2\xb7 stages burst,onset`); empty for crossings stored before they were named
}  // namespace
std::vector<std::string> crossing_stage_names(const std::vector<std::string>& computed_by) {
  std::vector<std::string> out;
  if (computed_by.empty()) return out;
  const std::string tag = " \xc2\xb7 stages ";
  const std::size_t at = computed_by[0].find(tag);
  if (at == std::string::npos) return out;
  const std::string rest = computed_by[0].substr(at + tag.size()) + ",";
  std::string cur;
  for (const char c : rest) { if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += c; }
  return out;
}
int crossing_stage_index(const std::string& which, const std::vector<std::string>& names, const std::string& name) {
  for (std::size_t i = 0; i < names.size(); ++i) if (names[i] == name) return static_cast<int>(i);
  std::string have;
  for (const auto& n : names) have += (have.empty() ? "" : ", ") + n;
  invalid("repeat compute: " + which + (names.empty() ? " does not name its stages (crossings stored before stage names: run vortex_cores again with --replace, or give --stage as an index)" : " has no stage '" + name + "' (it has: " + have + ")"));
}
namespace {
struct RunData { nxr::io::MeasurementValues x, y, z, n; };
RunData read_run(nxr::io::SubjectWriter& sub, const std::string& sel, const std::string& field) {
  auto get = [&](const char* m) { return nxr::io::measurement_values(sub, sel, field, std::nullopt, std::nullopt, m); };
  return RunData{get("crossing_sum_x"), get("crossing_sum_y"), get("crossing_sum_z"), get("crossings")};
}
// the pooled senses of stage `stage`: rows 2s (sense +1) and 2s+1 (sense -1); the members' last is the bridge, not a tile
nxr::crossings::Pooled pooled_of(const RunData& d, int stage, std::int64_t tiles) {
  const std::int64_t M = d.n.members, r0 = 2 * static_cast<std::int64_t>(stage), r1 = r0 + 1;
  Eigen::MatrixXd plus(tiles, 3), minus(tiles, 3);
  Eigen::VectorXd np(tiles), nm(tiles);
  for (std::int64_t t = 0; t < tiles; ++t) {
    const std::size_t a = static_cast<std::size_t>(r0 * M + t), b = static_cast<std::size_t>(r1 * M + t);
    plus.row(t) << d.x.values[a], d.y.values[a], d.z.values[a];
    minus.row(t) << d.x.values[b], d.y.values[b], d.z.values[b];
    np[t] = d.n.values[a]; nm[t] = d.n.values[b];
  }
  return nxr::crossings::poolSenses(plus, np, minus, nm);
}
}  // namespace

RunRepeatResult run_repeat_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const RunRepeatOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.recording_a.empty() || opt.recording_b.empty()) invalid("repeat compute: name the two recordings (--recording-a, --recording-b)");
  if (opt.recording_a == opt.recording_b) invalid("repeat compute: the two runs are the same recording (" + opt.recording_a + "): a recording repeats itself");
  if (opt.stage_name && opt.stage_name->empty()) invalid("repeat compute: --stage NAME is empty");
  if (opt.stage < 0) invalid("repeat compute: --stage is a whole number >= 0 (the stage's index in the run's `stages`)");
  if (opt.min_crossings < 1) invalid("repeat compute: --min-crossings is at least 1 (nxr: 10)");
  std::vector<Row> f;
  for (const std::string* nm : {&opt.recording_a, &opt.recording_b}) {
    const auto r = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(*nm)});
    if (!r) missing(sub.name() + ": no recording (Field) named '" + *nm + "'");
    f.push_back(*r);
  }
  // THE TILE LEVEL both were measured over
  std::string sel_id;
  if (opt.over) {
    const auto s = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.over)});
    if (!s) missing(sub.name() + ": no Selection named '" + *opt.over + "'");
    sel_id = text(*s, "id");
  } else {
    std::set<std::string> both;
    for (std::size_t k = 0; k < 2; ++k) {
      std::set<std::string> have;
      for (const Row& r : sub.db().all("SELECT DISTINCT selection_id FROM selection_measurement WHERE of_field_id = ? AND measure = 'crossings'", {V(text(f[k], "id"))})) have.insert(text(r, "selection_id"));
      if (have.empty()) missing(sub.name() + ": " + text(f[k], "name") + " has no stored crossings (`run vortex_cores` first)");
      if (k == 0) both = have;
      else { std::set<std::string> x; for (const auto& s : both) if (have.count(s)) x.insert(s); both = x; }
    }
    if (both.empty()) missing(sub.name() + ": " + opt.recording_a + " and " + opt.recording_b + " were measured over different tile levels (name --over)");
    if (both.size() > 1) invalid(sub.name() + ": both runs have crossings over " + std::to_string(both.size()) + " tile levels; name one with --over");
    sel_id = *both.begin();
  }
  const auto sel = sub.db().read("selection", sel_id);
  const std::int64_t tiles = nxr::io::integer(*sel, "n_members");
  const RunData A = read_run(sub, sel_id, text(f[0], "id")), B = read_run(sub, sel_id, text(f[1], "id"));
  // THE STAGE, by name in each run's own rows (they may list their stages in another order) or by index
  const std::vector<std::string> names_a = crossing_stage_names(A.n.computed_by), names_b = crossing_stage_names(B.n.computed_by);
  const int sa = opt.stage_name ? crossing_stage_index(opt.recording_a, names_a, *opt.stage_name) : opt.stage, sb = opt.stage_name ? crossing_stage_index(opt.recording_b, names_b, *opt.stage_name) : opt.stage;
  for (const auto& [d, st] : {std::pair<const RunData*, int>{&A, sa}, std::pair<const RunData*, int>{&B, sb}})
    if (d->n.members != tiles + 1 || d->n.rows < 2 * static_cast<std::int64_t>(st) + 2)
      invalid("repeat compute: the stored crossings are [" + std::to_string(d->n.rows) + " rows x " + std::to_string(d->n.members) + " members]; the level has " + std::to_string(tiles) + " tiles (and the bridge) and stage " + std::to_string(st) + " needs rows " +
              std::to_string(2 * st) + " and " + std::to_string(2 * st + 1));
  const std::string stage_tag = opt.stage_name ? (sa == 0 && sb == 0 ? std::string() : "." + *opt.stage_name) : (opt.stage ? ".stage" + std::to_string(opt.stage) : std::string());
  const std::string base = opt.name ? *opt.name : "vortex_cores.run_repeat." + opt.recording_a + "." + opt.recording_b + stage_tag;
  const std::string pt = "statistics/" + nxr::io::safe_name(base + ".tiles"), pw = "statistics/" + nxr::io::safe_name(base + ".whole");
  CacheKey key;
  key.pipeline = "repeat compute";
  key.params = Json{{"name", base}, {"stage_a", sa}, {"stage_b", sb}, {"min_crossings", opt.min_crossings}};
  key.input = Json{{"a", text(f[0], "id")}, {"b", text(f[1], "id")}, {"over", sel_id}, {"computed_by", Json{{"a", A.n.computed_by}, {"b", B.n.computed_by}}}};
  key.compute = NXR_COMPUTE_VERSION;
  RunRepeatResult res;
  if (const auto hit = find_cached(sub.db(), sub.id(), "statistic", key, pt)) {
    res.hit = true;
    res.info = Json{{"name", base}, {"tiles", pt}, {"whole", pw}, {"cache", "hit"}};
    say(Json{{"stage", "cache"}, {"hit", true}, {"statistic", base}});
    return res;
  }
  const nxr::crossings::Pooled pa = pooled_of(A, sa, tiles), pb = pooled_of(B, sb, tiles);
  const nxr::crossings::Repeat rep = nxr::crossings::repeat(pa, pb, opt.min_crossings);
  std::vector<double> dots(rep.dot.data(), rep.dot.data() + rep.dot.size()), whole = {rep.cosine};
  bool replaced = false;
  clear_stored(ds, sub, {{"statistic", pt}, {"statistic", pw}}, opt.replace, base, replaced);
  const Json warnings = Json::array({"a repeat is two runs of one subject agreeing: they share the cortex, the kernel and the instrument's leakage, so it is evidence only against the lag-free null (piece 10), whose p-value this statistic waits for",
                                     "a tile counts only with at least min_crossings crossings in both runs (the senses pooled); the whole recording's cosine is the count-weighted mean over those tiles"});
  Json pj = stamp(key);
  pj["by"] = "nxr-neuro repeat compute"; pj["warnings"] = warnings; pj["reference"] = "nxr vortex_cores.repeat, tile_dot (NMIN_TILE 10)";
  pj["rows"] = std::string("stage ") + (opt.stage_name ? *opt.stage_name : std::to_string(opt.stage)) + ": rows " + std::to_string(2 * sa) + "/" + std::to_string(2 * sa + 1) + " of " + opt.recording_a + ", " + std::to_string(2 * sb) + "/" + std::to_string(2 * sb + 1) + " of " + opt.recording_b + " (sense +1 and -1, pooled)";
  Json opts = Json{{"other_field_id", text(f[1], "id")}, {"recording_a", opt.recording_a}, {"recording_b", opt.recording_b}, {"stage", opt.stage_name ? Json(*opt.stage_name) : Json(opt.stage)}, {"stage_a", sa}, {"stage_b", sb}, {"min_crossings", opt.min_crossings}, {"tiles_used", rep.tilesUsed}, {"tiles", tiles}};
  std::string id_t, id_w;
  sub.compose([&] {
    nxr::io::StatisticSpec st;
    st.name = base + ".tiles"; st.path = pt; st.measure = "repeat_dot"; st.method = "run_repeat";
    st.of_field_id = text(f[0], "id"); st.partition_selection_id = sel_id; st.n_members = tiles;
    st.axes = Json::array({Json{{"name", "tile"}, {"selection_id", sel_id}, {"size", tiles}}});
    st.data = nxr::io::view(dots, {tiles});
    st.description = "per tile of " + text(*sel, "name") + ": the dot product of the mean resultant crossing vectors of " + opt.recording_a + " and " + opt.recording_b + " (senses pooled; NaN with fewer than " + std::to_string(opt.min_crossings) + " crossings in either)";
    st.options = opts; st.producer = pj;
    id_t = sub.statistic(std::move(st));
    nxr::io::StatisticSpec sw;
    sw.name = base + ".whole"; sw.path = pw; sw.measure = "repeat_cosine"; sw.method = "run_repeat";
    sw.of_field_id = text(f[0], "id"); sw.partition_selection_id = sel_id; sw.n_members = rep.tilesUsed;
    sw.axes = Json::array({Json{{"name", "whole"}, {"values", Json::array({Json{{"name", "count_weighted_mean_cosine"}}})}, {"size", 1}}});
    sw.data = nxr::io::view(whole, {std::int64_t(1)});
    sw.description = "the count-weighted mean cosine of the tiles' directions between " + opt.recording_a + " and " + opt.recording_b + " (tiles with at least " + std::to_string(opt.min_crossings) + " crossings in both)";
    sw.options = opts; sw.producer = pj;
    id_w = sub.statistic(std::move(sw));
  });
  res.info = Json{{"name", base}, {"tiles", pt}, {"whole", pw}, {"id_tiles", id_t}, {"id_whole", id_w}, {"cosine", std::isnan(rep.cosine) ? Json() : Json(rep.cosine)}, {"tiles_used", rep.tilesUsed}, {"tiles_total", tiles},
                  {"crossings_a", pa.n.sum()}, {"crossings_b", pb.n.sum()}, {"min_crossings", opt.min_crossings}, {"stage", opt.stage_name ? Json(*opt.stage_name) : Json(opt.stage)}, {"stage_a", sa}, {"stage_b", sb}, {"replaced", replaced}, {"cache", "miss"}, {"warnings", warnings}};
  return res;
}

}  // namespace nxr::pipelines
