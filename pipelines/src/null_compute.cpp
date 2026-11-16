// nxr-pipelines — the lag-free null's tests (null_compute.h).
#include "nxr/pipelines/null_compute.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "nxr/crossings.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/run_repeat.h"
#include "nxr/pipelines/runner.h"
#include "nxr/surrogate.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }
const double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int64_t kSeedStride = 100000;   // run k's draw i is seed i + k * stride

// the pooled senses of stage `stage` from [rows x members x 4] values (x y z count), the bridge member (the last) left out
nxr::crossings::Pooled pooled_of(const std::vector<double>& v, std::int64_t members, int stage) {
  const std::int64_t tiles = members - 1;
  Eigen::MatrixXd plus(tiles, 3), minus(tiles, 3);
  Eigen::VectorXd np(tiles), nm(tiles);
  for (std::int64_t t = 0; t < tiles; ++t) {
    const std::size_t a = static_cast<std::size_t>(((2 * stage) * members + t) * 4), b = static_cast<std::size_t>(((2 * stage + 1) * members + t) * 4);
    plus.row(t) << v[a], v[a + 1], v[a + 2]; np[t] = v[a + 3];
    minus.row(t) << v[b], v[b + 1], v[b + 2]; nm[t] = v[b + 3];
  }
  return nxr::crossings::poolSenses(plus, np, minus, nm);
}
// the unbiased rho2 of a tile's pooled sum (NaN with fewer than `minn` crossings)
Eigen::VectorXd rho2(const nxr::crossings::Pooled& p, int minn) {
  Eigen::VectorXd r(p.n.size());
  for (Eigen::Index t = 0; t < p.n.size(); ++t) { const double n = p.n[t]; r[t] = n >= std::max(minn, 2) ? (p.sum.row(t).squaredNorm() - n) / (n * (n - 1)) : kNaN; }
  return r;
}
struct Run {
  std::string name, field_id;
  Row field;
  std::int64_t members = 0;
  std::vector<double> data;                  // [rows x members x 4] interleaved (x y z count), as a draw's statistic
  std::vector<std::string> computed_by;
  int stage = 0;
  nxr::crossings::Pooled pooled;
  Eigen::VectorXd cons;                      // the data's unbiased rho2 per tile
  Eigen::VectorXd exceed, null_n;            // draws at least as large; the null's crossings summed over draws
  Eigen::MatrixXd null_sum;                  // [tiles x 3] the null's mean resultants summed over draws
};
}  // namespace

NullResult null_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const NullOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.recording_a.empty()) invalid("null compute: name the recording (--recording-a)");
  if (opt.recording_b && *opt.recording_b == opt.recording_a) invalid("null compute: the two runs are the same recording (" + opt.recording_a + ")");
  if (opt.batch < 1 || opt.min_draws < 1 || opt.max_draws < opt.min_draws || opt.h < 1 || opt.min_crossings < 1) invalid("null compute: --batch, --min-draws (<= --max-draws), --h and --min-crossings are whole numbers >= 1");
  if (opt.max_draws > 100000) invalid("null compute: --max-draws is at most 100000 (the two runs' seeds are 100000 apart)");
  if (opt.stage < 0) invalid("null compute: --stage is a stage's name or a whole number >= 0");
  const graph::Definition dnull = graph::load("vortex_null");
  std::vector<Run> runs(opt.recording_b ? 2 : 1);
  runs[0].name = opt.recording_a;
  if (opt.recording_b) runs[1].name = *opt.recording_b;
  for (Run& r : runs) {
    const auto f = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(r.name)});
    if (!f) missing(sub.name() + ": no recording (Field) named '" + r.name + "'");
    r.field = *f; r.field_id = text(*f, "id");
  }
  // THE TILE LEVEL the crossings were measured over (both runs')
  std::string sel_id;
  if (opt.over) {
    const auto s = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.over)});
    if (!s) missing(sub.name() + ": no Selection named '" + *opt.over + "'");
    sel_id = text(*s, "id");
  } else {
    std::set<std::string> both;
    bool first = true;
    for (const Run& r : runs) {
      std::set<std::string> have;
      for (const Row& x : sub.db().all("SELECT DISTINCT selection_id FROM selection_measurement WHERE of_field_id = ? AND measure = 'crossings'", {V(r.field_id)})) have.insert(text(x, "selection_id"));
      if (have.empty()) missing(sub.name() + ": " + r.name + " has no stored crossings (`run vortex_cores` first)");
      if (first) both = have; else { std::set<std::string> x; for (const auto& s : both) if (have.count(s)) x.insert(s); both = x; }
      first = false;
    }
    if (both.empty()) missing(sub.name() + ": the two recordings were measured over different tile levels (name --over)");
    if (both.size() > 1) invalid(sub.name() + ": the crossings are over " + std::to_string(both.size()) + " tile levels; name one with --over");
    sel_id = *both.begin();
  }
  const auto sel = sub.db().read("selection", sel_id);
  const std::int64_t tiles = nxr::io::integer(*sel, "n_members");
  // THE DATA'S CROSSINGS and their stage
  for (Run& r : runs) {
    auto get = [&](const char* m) { return nxr::io::measurement_values(sub, sel_id, r.field_id, std::nullopt, std::nullopt, m); };
    const auto x = get("crossing_sum_x"), y = get("crossing_sum_y"), z = get("crossing_sum_z"), n = get("crossings");
    r.members = n.members; r.computed_by = n.computed_by;
    if (n.members != tiles + 1) invalid("null compute: " + r.name + "'s stored crossings have " + std::to_string(n.members) + " members; the level has " + std::to_string(tiles) + " tiles (and the bridge)");
    r.stage = opt.stage_name ? crossing_stage_index(r.name, crossing_stage_names(r.computed_by), *opt.stage_name) : opt.stage;
    if (n.rows < 2 * static_cast<std::int64_t>(r.stage) + 2) invalid("null compute: " + r.name + "'s crossings have " + std::to_string(n.rows) + " rows; stage " + std::to_string(r.stage) + " needs rows " + std::to_string(2 * r.stage) + " and " + std::to_string(2 * r.stage + 1));
    r.data.resize(static_cast<std::size_t>(n.rows * n.members * 4));
    for (std::size_t i = 0; i < n.values.size(); ++i) { r.data[4 * i] = x.values[i]; r.data[4 * i + 1] = y.values[i]; r.data[4 * i + 2] = z.values[i]; r.data[4 * i + 3] = n.values[i]; }
    r.pooled = pooled_of(r.data, r.members, r.stage);
    r.cons = rho2(r.pooled, opt.min_crossings);
    r.exceed = Eigen::VectorXd::Zero(tiles); r.null_n = Eigen::VectorXd::Zero(tiles); r.null_sum = Eigen::MatrixXd::Zero(tiles, 3);
  }
  const bool two = runs.size() == 2;
  // the result names
  auto nm = [](const std::string& r, const char* what) { return "vortex_cores.null." + r + "." + what; };
  const std::string repeat_base = two ? "vortex_cores.run_repeat." + runs[0].name + "." + runs[1].name + (opt.stage_name && !(runs[0].stage == 0 && runs[1].stage == 0) ? "." + *opt.stage_name : opt.stage ? ".stage" + std::to_string(opt.stage) : std::string()) : std::string();
  // THE CACHE: the same data, options and draws' definition make the same tests
  CacheKey key;
  key.pipeline = "null compute";
  key.params = Json{{"stage", opt.stage_name ? Json(*opt.stage_name) : Json(opt.stage)}, {"batch", opt.batch}, {"min_draws", opt.min_draws}, {"max_draws", opt.max_draws}, {"h", opt.h}, {"min_crossings", opt.min_crossings}, {"set", opt.set}};
  key.input = Json{{"recordings", [&] { Json a = Json::array(); for (const Run& r : runs) a.push_back(Json{{"id", r.field_id}, {"computed_by", r.computed_by}}); return a; }()}, {"over", sel_id}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string p_cons0 = "statistics/" + nxr::io::safe_name(nm(runs[0].name, "consistency_p"));
  NullResult res;
  if (!opt.replace)
    if (const auto hit = find_cached(sub.db(), sub.id(), "statistic", key, p_cons0)) {
      res.hit = true;
      res.info = Json{{"cache", "hit"}, {"consistency_p", nm(runs[0].name, "consistency_p")}};
      say(Json{{"stage", "cache"}, {"hit", true}});
      return res;
    }
  // THE DRAWS, in batches, until every tile is decided
  Eigen::VectorXd dot_data, dot_exceed = Eigen::VectorXd::Zero(tiles);
  double cos_data = kNaN, cos_exceed = 0;
  if (two) { const auto rp = nxr::crossings::repeat(runs[0].pooled, runs[1].pooled, opt.min_crossings); dot_data = rp.dot; cos_data = rp.cosine; }
  graph::RunOptions ro;
  ro.chunks = opt.chunks;
  ro.start = opt.start; ro.stop = opt.stop;
  ro.replace = opt.replace;                    // a draw stored with other parameters is moved aside (never deleted); a draw that is the same is a hit
  std::int64_t nd = 0;
  std::string stopped = "max_draws";
  while (nd < opt.max_draws) {
    const std::int64_t upto = std::min<std::int64_t>(nd + opt.batch, opt.max_draws);
    for (std::int64_t i = nd; i < upto; ++i) {
      std::vector<nxr::crossings::Pooled> draw;
      for (std::size_t k = 0; k < runs.size(); ++k) {
        Run& r = runs[k];
        const std::int64_t seed = i + static_cast<std::int64_t>(k) * kSeedStride;     // the two runs' draws are independent: the white noise is a function of the seed, so run B's seeds are offset
        Json set = opt.set;
        set["recording"] = r.name; set["seed"] = seed;
        const auto out = graph::run(ds, sub, dnull, set, ro);
        if (out.empty()) invalid("null compute: the draw " + std::to_string(i) + " of " + r.name + " made no run");
        const nxr::io::StoredStatistic D = nxr::io::read_statistic(sub, "vortex_null.draw" + std::to_string(seed) + ".null_crossings." + r.name);
        if (D.shape.size() != 3 || D.shape[1] != r.members || D.shape[2] != 4 || D.shape[0] < 2 * static_cast<std::int64_t>(r.stage) + 2) invalid("null compute: the draw's crossings do not match the data's rows and tiles (" + r.name + ")");
        draw.push_back(pooled_of(D.values, r.members, r.stage));
        const nxr::crossings::Pooled& P = draw.back();
        const Eigen::VectorXd c = rho2(P, 2);
        for (std::int64_t t = 0; t < tiles; ++t) {
          if (std::isfinite(r.cons[t]) && std::isfinite(c[t]) && c[t] >= r.cons[t]) r.exceed[t] += 1;
          r.null_n[t] += P.n[t];
          if (P.n[t] > 0) r.null_sum.row(t) += P.sum.row(t) / P.n[t];
        }
      }
      if (two) {
        const auto rp = nxr::crossings::repeat(draw[0], draw[1], opt.min_crossings);
        for (std::int64_t t = 0; t < tiles; ++t) if (std::isfinite(dot_data[t]) && std::isfinite(rp.dot[t]) && rp.dot[t] >= dot_data[t]) dot_exceed[t] += 1;
        if (std::isfinite(cos_data) && std::isfinite(rp.cosine) && rp.cosine >= cos_data) cos_exceed += 1;
      }
    }
    nd = upto;
    // decided: every tile with a statistic has h exceedances (a tile that never will goes to max_draws, which is the Besag-Clifford rule)
    bool decided = nd >= opt.min_draws;
    for (std::int64_t t = 0; decided && t < tiles; ++t) {
      for (const Run& r : runs) if (std::isfinite(r.cons[t]) && r.exceed[t] < opt.h) decided = false;
      if (two && std::isfinite(dot_data[t]) && dot_exceed[t] < opt.h) decided = false;
    }
    if (two && std::isfinite(cos_data) && cos_exceed < opt.h) decided = false;
    say(Json{{"stage", "draws"}, {"draws", nd}, {"decided", decided}});
    if (decided) { stopped = "decided"; break; }
  }
  // THE P-VALUES, q-values, ratios and the null's resultants
  const nxr::io::Row& selrow = *sel;
  auto sequential = [&](const Eigen::VectorXd& data, const Eigen::VectorXd& e) {
    Eigen::VectorXd p(tiles);
    for (std::int64_t t = 0; t < tiles; ++t) p[t] = std::isfinite(data[t]) ? nxr::surrogate::besagCliffordP(static_cast<std::int64_t>(e[t]), nd, opt.h) : kNaN;
    return p;
  };
  auto store = [&](const std::string& name, const char* measure, const char* method, const Row& of, std::vector<double> values, std::vector<std::int64_t> shape, const Json& axes, const Json& components, const std::string& desc, const Json& options, const Json& warnings, bool first) {
    const std::string path = "statistics/" + nxr::io::safe_name(name);
    nxr::io::StatisticSpec st;
    st.name = name; st.path = path; st.measure = measure; st.method = method;
    st.of_field_id = text(of, "id"); st.partition_selection_id = sel_id; st.n_members = tiles;
    st.axes = axes; if (!components.is_null()) st.components = components;
    st.data = nxr::io::view(values, shape);
    st.description = desc;
    st.options = options;
    Json pj = first ? stamp(key) : Json::object();
    pj["by"] = "nxr-neuro null compute"; pj["warnings"] = warnings; pj["reference"] = "nxr vortex.envelope_matched_null; Besag and Clifford 1991; Benjamini-Hochberg";
    st.producer = pj;
    return sub.statistic(std::move(st));
  };
  const Json warnings = Json::array({"the null has the data's zero-lag covariance, spectrum and envelope and nothing else: a tile beyond it is beyond what mixing and amplitude modulation produce, not a proof of a wave",
                                     "the threshold and the burst frames are the data's: only the signal is replaced",
                                     "a null tile with too few crossings has no statistic and is not an exceedance: read the crossing ratio beside the p-values"});
  const Json tile_axis = Json::array({Json{{"name", "tile"}, {"selection_id", sel_id}, {"size", tiles}}});
  auto axes_with = [&](const char* name, std::vector<const char*> comps) {
    Json c = Json::array(), v = Json::array();
    for (const char* k : comps) v.push_back(Json{{"name", k}});
    Json a = tile_axis; a.push_back(Json{{"name", name}, {"size", static_cast<std::int64_t>(comps.size())}});
    return std::make_pair(a, v);
  };
  Json info = Json{{"draws", nd}, {"stopped", stopped}, {"tiles", tiles}, {"max_draws", opt.max_draws}, {"h", opt.h}};
  Json made = Json::array();
  {
    std::vector<std::pair<const char*, std::string>> nodes;
    std::vector<std::string> keep;                                                  // the paths must outlive the call
    for (const Run& r : runs) for (const char* w : {"consistency_p", "crossing_ratio", "resultant"}) keep.push_back("statistics/" + nxr::io::safe_name(nm(r.name, w)));
    if (two) { keep.push_back("statistics/" + nxr::io::safe_name(repeat_base + ".p")); keep.push_back("statistics/" + nxr::io::safe_name(repeat_base + ".whole_p")); }
    for (const auto& k : keep) nodes.emplace_back("statistic", k);
    bool replaced = false;
    clear_stored(ds, sub, nodes, opt.replace, "null compute", replaced);
    info["replaced"] = replaced;
  }
  sub.compose([&] {
    bool first = true;
    for (Run& r : runs) {
      const Eigen::VectorXd p = sequential(r.cons, r.exceed), q = nxr::surrogate::benjaminiHochberg(p);
      std::vector<double> pq(static_cast<std::size_t>(tiles * 3));
      for (std::int64_t t = 0; t < tiles; ++t) { pq[static_cast<std::size_t>(3 * t)] = p[t]; pq[static_cast<std::size_t>(3 * t + 1)] = q[t]; pq[static_cast<std::size_t>(3 * t + 2)] = r.exceed[t]; }
      const auto [ax, comps] = axes_with("test", {"p", "q", "exceedances"});
      const Json opts = Json{{"draws", nd}, {"h", opt.h}, {"stopped", stopped}, {"min_crossings", opt.min_crossings}, {"stage", r.stage}, {"statistic", "the unbiased rho2 of the tile's pooled crossings (senses pooled), one-sided"}, {"q", "Benjamini-Hochberg over the tiles with a statistic"}, {"set", opt.set}};
      made.push_back(store(nm(r.name, "consistency_p"), "null_p", "besag_clifford", r.field, pq, {tiles, 3}, ax, comps, "per tile of " + text(selrow, "name") + ": the sequential p, BH q and exceedances of the crossing consistency (unbiased rho2) of " + r.name + " against " + std::to_string(nd) + " lag-free null draws", opts, warnings, first));
      first = false;
      std::vector<double> ratio(static_cast<std::size_t>(tiles)), res4(static_cast<std::size_t>(tiles * 4));
      for (std::int64_t t = 0; t < tiles; ++t) {
        const double mean_n = r.null_n[t] / static_cast<double>(nd);
        ratio[static_cast<std::size_t>(t)] = mean_n > 0 ? r.pooled.n[t] / mean_n : kNaN;
        for (int k = 0; k < 3; ++k) res4[static_cast<std::size_t>(4 * t + k)] = r.null_sum(t, k) / static_cast<double>(nd);
        res4[static_cast<std::size_t>(4 * t + 3)] = mean_n;
      }
      made.push_back(store(nm(r.name, "crossing_ratio"), "crossing_ratio", "lagfree_null", r.field, ratio, {tiles}, tile_axis, Json(), "per tile of " + text(selrow, "name") + ": the data's crossings of " + r.name + " over the lag-free null's mean number (" + std::to_string(nd) + " draws)", Json{{"draws", nd}, {"stage", r.stage}}, warnings, false));
      const auto [ax4, comps4] = axes_with("component", {"resultant_x", "resultant_y", "resultant_z", "crossings"});
      made.push_back(store(nm(r.name, "resultant"), "null_resultant", "lagfree_null", r.field, res4, {tiles, 4}, ax4, comps4, "per tile of " + text(selrow, "name") + ": the lag-free null's mean resultant crossing vector (senses pooled) and mean crossings for " + r.name + " over " + std::to_string(nd) + " draws: what a group map subtracts", Json{{"draws", nd}, {"stage", r.stage}}, warnings, false));
      info["consistency"].push_back(Json{{"recording", r.name}, {"tiles_tested", [&] { std::int64_t k = 0; for (std::int64_t t = 0; t < tiles; ++t) k += std::isfinite(r.cons[t]); return k; }()},
                                         {"tiles_q05", [&] { std::int64_t k = 0; for (std::int64_t t = 0; t < tiles; ++t) k += std::isfinite(q[t]) && q[t] < 0.05; return k; }()}});
    }
    if (two) {
      const Eigen::VectorXd p = sequential(dot_data, dot_exceed), q = nxr::surrogate::benjaminiHochberg(p);
      std::vector<double> pq(static_cast<std::size_t>(tiles * 3));
      for (std::int64_t t = 0; t < tiles; ++t) { pq[static_cast<std::size_t>(3 * t)] = p[t]; pq[static_cast<std::size_t>(3 * t + 1)] = q[t]; pq[static_cast<std::size_t>(3 * t + 2)] = dot_exceed[t]; }
      const auto [ax, comps] = axes_with("test", {"p", "q", "exceedances"});
      const Json opts = Json{{"draws", nd}, {"h", opt.h}, {"stopped", stopped}, {"min_crossings", opt.min_crossings}, {"recording_a", runs[0].name}, {"recording_b", runs[1].name}, {"statistic", "the dot product of the two runs' mean resultants (nxr tile_dot), draw i of one run against draw i of the other"}};
      made.push_back(store(repeat_base + ".p", "null_p", "besag_clifford", runs[0].field, pq, {tiles, 3}, ax, comps, "per tile of " + text(selrow, "name") + ": the sequential p, BH q and exceedances of the repeat (dot product of the mean resultants) of " + runs[0].name + " and " + runs[1].name + " against " + std::to_string(nd) + " null draws", opts, warnings, false));
      const double pw = std::isfinite(cos_data) ? nxr::surrogate::besagCliffordP(static_cast<std::int64_t>(cos_exceed), nd, opt.h) : kNaN;
      Json axw = Json::array({Json{{"name", "whole"}, {"values", Json::array({Json{{"name", "p"}}, Json{{"name", "exceedances"}}, Json{{"name", "draws"}}})}, {"size", 3}}});
      made.push_back(store(repeat_base + ".whole_p", "null_p", "besag_clifford", runs[0].field, {pw, cos_exceed, static_cast<double>(nd)}, {3}, axw, Json(), "the sequential p of the count-weighted mean cosine of the tiles' directions between " + runs[0].name + " and " + runs[1].name + " against the null's", opts, warnings, false));
      info["repeat"] = Json{{"cosine", std::isfinite(cos_data) ? Json(cos_data) : Json()}, {"p", std::isfinite(pw) ? Json(pw) : Json()}, {"tiles_q05", [&] { std::int64_t k = 0; for (std::int64_t t = 0; t < tiles; ++t) k += std::isfinite(q[t]) && q[t] < 0.05; return k; }()}};
    }
  });
  info["statistics"] = made;
  res.info = info;
  return res;
}

}  // namespace nxr::pipelines
