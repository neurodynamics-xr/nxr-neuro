// nxr-pipelines — the surrogate comparisons (surrogate_compare.h). Orchestration only: the draws are runs of typed definitions, read through io; the maths are nxr::stats and nxr::select::track_crossings.
#include "nxr/pipelines/surrogate_compare.h"

#include "nxr/pipelines/task.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "nxr/select/track_crossings.h"
#include "nxr/io/hash.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/stats/nulltest.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/run_repeat.h"
#include "nxr/pipelines/runner.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;
namespace nt = nxr::stats;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }
const double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int64_t kSeedStride = 100000;   // run k's draw i is seed i + k * stride

// THE GENERATOR a draw must come from: the draw definition the comparison runs (its name and its text's hash, from the draw's producer record). A draw made by another definition is refused.
Json generator_of(const graph::Definition& d) { return Json{{"pipeline", d.name}, {"version", d.version}, {"sha256", nxr::io::sha256_hex(d.text)}}; }
void same_generator(const nxr::io::StoredStatistic& D, const graph::Definition& d, const std::string& what) {
  const Json pl = D.producer.value("pipeline", Json::object());
  std::string name = pl.value("name", std::string());
  const std::string sha = pl.value("sha256", std::string());
  if (const auto at = name.find(".draw"); at != std::string::npos) name.resize(at);   // a draw's run is recorded under `<pipeline>.draw<seed>` (the definition's `draw:`): the pipeline is what precedes it
  if (name != d.name || sha != nxr::io::sha256_hex(d.text))
    invalid(what + " was made by " + (name.empty() ? std::string("no recorded pipeline") : name + " (" + sha.substr(0, std::min<std::size_t>(12, sha.size())) + ")") + ", not by " + d.name + " as it is now: rerun the draws (--replace)");
}
// the draw line's record: the seeds and the batches (spec section 7)
Json seeds_of(std::int64_t draws, int batch, bool two) {
  Json s = Json{{"first", 0}, {"draws", draws}, {"batch", batch}, {"line", "draw i = seed i; a batch of `batch` draws is a member of the draw line's grid"}};
  if (two) s["second_run_offset"] = kSeedStride;
  return s;
}

// ── the vortex chain ──
// the pooled senses of stage `stage` from [rows x members x 4] values (x y z count), the bridge member (the last) left out: a selection of the stored array's rows, then nxr::select::track_crossings::poolSenses
nxr::select::track_crossings::Pooled pooled_of(const std::vector<double>& v, std::int64_t members, int stage) {
  const std::int64_t tiles = members - 1;
  Eigen::MatrixXd plus(tiles, 3), minus(tiles, 3);
  Eigen::VectorXd np(tiles), nm(tiles);
  for (std::int64_t t = 0; t < tiles; ++t) {
    const std::size_t a = static_cast<std::size_t>(((2 * stage) * members + t) * 4), b = static_cast<std::size_t>(((2 * stage + 1) * members + t) * 4);
    plus.row(t) << v[a], v[a + 1], v[a + 2]; np[t] = v[a + 3];
    minus.row(t) << v[b], v[b + 1], v[b + 2]; nm[t] = v[b + 3];
  }
  return nxr::select::track_crossings::poolSenses(plus, np, minus, nm);
}
struct Run {
  std::string name, field_id;
  task::Node field;
  std::int64_t members = 0;
  std::vector<double> data;                  // [rows x members x 4] interleaved (x y z count), as a draw's statistic
  std::vector<std::string> computed_by;
  int stage = 0;
  nxr::select::track_crossings::Pooled pooled;
  Eigen::VectorXd cons;                      // the data's unbiased rho2 per tile
  nt::Tally consistency, counts, rx, ry, rz; // the draws' rho2 against the data's; their crossing counts; their mean resultants' components
};

// ── the spatial events ──
// the summary's entries that are SHARES of the patches (level_*, area_bin_*), and where the patch and frame counts are: read from the statistic's axis (the names it stores)
struct Entries { std::vector<std::string> names; std::vector<std::size_t> at; std::size_t patches = 0, frames = 0; bool has_patches = false, has_frames = false; };
Entries entries_of(const nxr::io::StoredStatistic& S) {
  Entries e;
  const Json& v = S.axes[0]["values"];
  for (std::size_t k = 0; k < v.size(); ++k) {
    const std::string n = v[k]["name"].get<std::string>();
    if (n == "patches") { e.patches = k; e.has_patches = true; }
    if (n == "frames") { e.frames = k; e.has_frames = true; }
    if (n.rfind("level_", 0) == 0 || n.rfind("area_bin_", 0) == 0) { e.names.push_back(n); e.at.push_back(k); }
  }
  e.names.push_back("patches_per_frame");
  return e;
}
Eigen::VectorXd shares_of(const nxr::io::StoredStatistic& S, const Entries& e) {
  Eigen::VectorXd counts(static_cast<Eigen::Index>(e.at.size()));
  for (std::size_t k = 0; k < e.at.size(); ++k) counts[static_cast<Eigen::Index>(k)] = S.values[e.at[k]];
  return nt::patchShares(counts, e.has_patches ? S.values[e.patches] : 0.0, e.has_frames ? S.values[e.frames] : 0.0);
}
}  // namespace

SurrogateCompareResult surrogate_lagfree_vortex_compare(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurrogateVortexCompareOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  const std::string who = "surrogate_lagfree_vortex_compare";
  if (opt.recording_a.empty()) invalid(who + ": name the recording (--recording-a)");
  if (opt.recording_b && *opt.recording_b == opt.recording_a) invalid(who + ": the two runs are the same recording (" + opt.recording_a + ")");
  if (opt.batch < 1 || opt.min_draws < 1 || opt.max_draws < opt.min_draws || opt.h < 1 || opt.min_crossings < 1) invalid(who + ": --batch, --min-draws (<= --max-draws), --h and --min-crossings are whole numbers >= 1");
  if (opt.max_draws > kSeedStride) invalid(who + ": --max-draws is at most 100000 (the two runs' seeds are 100000 apart)");
  if (opt.stage < 0) invalid(who + ": --stage is a stage's name or a whole number >= 0");
  const graph::Definition ddraw = graph::load("surrogate_lagfree_vortex");
  std::vector<Run> runs(opt.recording_b ? 2 : 1);
  runs[0].name = opt.recording_a;
  if (opt.recording_b) runs[1].name = *opt.recording_b;
  const task::Subject subj = task::Subject::of(sub);
  for (Run& r : runs) {
    const auto f = subj.find({.cls = task::Cls::field, .name = r.name});
    if (!f) missing(sub.name() + ": no recording (Field) named '" + r.name + "'");
    r.field = *f; r.field_id = f->id;
  }
  // THE SELECTION the data's crossings are over (both runs'): the draws must be over the same one
  std::string sel_id;
  if (opt.over) {
    const auto s = subj.find({.cls = task::Cls::selection, .name = *opt.over});
    if (!s) missing(sub.name() + ": no Selection named '" + *opt.over + "'");
    sel_id = s->id;
  } else {
    std::set<std::string> both;
    bool first = true;
    for (const Run& r : runs) {
      std::set<std::string> have;
      for (const std::string& x : subj.measured_selections(r.field_id, "crossings")) have.insert(x);
      if (have.empty()) missing(sub.name() + ": " + r.name + " has no stored crossings (`run vortex_cores` first)");
      if (first) both = have; else { std::set<std::string> x; for (const auto& s : both) if (have.count(s)) x.insert(s); both = x; }
      first = false;
    }
    if (both.empty()) missing(sub.name() + ": the two recordings were measured over different tile levels (name --over)");
    if (both.size() > 1) invalid(sub.name() + ": the crossings are over " + std::to_string(both.size()) + " tile levels; name one with --over");
    sel_id = *both.begin();
  }
  const auto sel = subj.require_id(task::Cls::selection, sel_id);
  const std::int64_t tiles = sel.n_members.value_or(0);
  // THE DATA'S CROSSINGS over that selection, and their stage
  for (Run& r : runs) {
    auto get = [&](const char* m) { return nxr::io::measurement_values(sub, sel_id, r.field_id, std::nullopt, std::nullopt, m); };
    const auto x = get("crossing_sum_x"), y = get("crossing_sum_y"), z = get("crossing_sum_z"), n = get("crossings");
    r.members = n.members; r.computed_by = n.computed_by;
    if (n.members != tiles + 1) invalid(who + ": " + r.name + "'s stored crossings have " + std::to_string(n.members) + " members; the selection " + sel.name + " has " + std::to_string(tiles) + " tiles (and the bridge)");
    try { r.stage = opt.stage_name ? crossing_stage_index(r.name, crossing_stage_names(r.computed_by), *opt.stage_name) : opt.stage; }
    catch (const std::invalid_argument& e) { invalid(e.what()); }                  // a stage the run does not have: refused as every other argument of the comparison
    if (n.rows < 2 * static_cast<std::int64_t>(r.stage) + 2) invalid(who + ": " + r.name + "'s crossings have " + std::to_string(n.rows) + " rows; stage " + std::to_string(r.stage) + " needs rows " + std::to_string(2 * r.stage) + " and " + std::to_string(2 * r.stage + 1));
    r.data.resize(static_cast<std::size_t>(n.rows * n.members * 4));
    for (std::size_t i = 0; i < n.values.size(); ++i) { r.data[4 * i] = x.values[i]; r.data[4 * i + 1] = y.values[i]; r.data[4 * i + 2] = z.values[i]; r.data[4 * i + 3] = n.values[i]; }
    r.pooled = pooled_of(r.data, r.members, r.stage);
    r.cons = nt::unbiasedConsistency(r.pooled.sum, r.pooled.n, opt.min_crossings);
    r.consistency = r.counts = r.rx = r.ry = r.rz = nt::emptyTally(tiles);
  }
  const bool two = runs.size() == 2;
  const Eigen::VectorXd none = Eigen::VectorXd::Constant(tiles, kNaN);           // no data statistic: a tally of the draws' values only (their counts and resultants)
  auto nm = [&](const std::string& r, const char* what) { return who + "." + r + "." + what; };
  const std::string repeat_base = two ? "vortex_cores.run_repeat." + runs[0].name + "." + runs[1].name + (opt.stage_name && !(runs[0].stage == 0 && runs[1].stage == 0) ? "." + *opt.stage_name : opt.stage ? ".stage" + std::to_string(opt.stage) : std::string()) : std::string();
  // THE CACHE: the same data, options and draw definition make the same comparison
  CacheKey key;
  key.pipeline = who;
  key.params = Json{{"stage", opt.stage_name ? Json(*opt.stage_name) : Json(opt.stage)}, {"batch", opt.batch}, {"min_draws", opt.min_draws}, {"max_draws", opt.max_draws}, {"h", opt.h}, {"min_crossings", opt.min_crossings}, {"set", opt.set},
                    {"generator", generator_of(ddraw)}};
  key.input = Json{{"recordings", [&] { Json a = Json::array(); for (const Run& r : runs) a.push_back(Json{{"id", r.field_id}, {"computed_by", r.computed_by}}); return a; }()}, {"over", sel_id}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string p_cons0 = "statistics/" + nxr::io::safe_name(nm(runs[0].name, "consistency_p"));
  SurrogateCompareResult res;
  if (!opt.replace)
    if (const auto hit = subj.cached(task::Cls::statistic, key.json(), p_cons0)) {
      res.hit = true;
      res.info = Json{{"cache", "hit"}, {"consistency_p", nm(runs[0].name, "consistency_p")}};
      say(Json{{"stage", "cache"}, {"hit", true}});
      return res;
    }
  // THE DRAWS, in batches, until every tile is decided (the stopping rule: nxr::stats::undecidedMembers)
  Eigen::VectorXd dot_data = none;
  Eigen::VectorXd cos_data = Eigen::VectorXd::Constant(1, kNaN);
  if (two) { const auto rp = nxr::select::track_crossings::repeat(runs[0].pooled, runs[1].pooled, opt.min_crossings); dot_data = rp.dot; cos_data[0] = rp.cosine; }
  nt::Tally dot_tally = nt::emptyTally(tiles), cos_tally = nt::emptyTally(1);
  graph::RunOptions ro;
  ro.chunks = opt.chunks;
  ro.start = opt.start; ro.stop = opt.stop;
  ro.replace = opt.replace;                    // a draw stored with other parameters is moved aside (never deleted); a draw that is the same is a hit
  std::int64_t nd = 0;
  std::string stopped = "max_draws";
  while (nd < opt.max_draws) {
    const std::int64_t upto = std::min<std::int64_t>(nd + opt.batch, opt.max_draws);
    for (std::int64_t i = nd; i < upto; ++i) {
      std::vector<nxr::select::track_crossings::Pooled> draw;
      for (std::size_t k = 0; k < runs.size(); ++k) {
        Run& r = runs[k];
        const std::int64_t seed = i + static_cast<std::int64_t>(k) * kSeedStride;     // the two runs' draws are independent: the white noise is a function of the seed, so run B's seeds are offset
        Json set = opt.set;
        set["recording"] = r.name; set["seed"] = seed;
        const auto out = graph::run(ds, sub, ddraw, set, ro);
        if (out.empty()) invalid(who + ": the draw " + std::to_string(i) + " of " + r.name + " made no run");
        const std::string dname = ddraw.name + ".draw" + std::to_string(seed) + ".null_crossings." + r.name;
        const nxr::io::StoredStatistic D = nxr::io::read_statistic(sub, dname);
        // THE SAME SELECTION and the same generator (spec section 7)
        same_generator(D, ddraw, who + ": the draw " + dname);
        const auto over = nxr::io::opt_text(D.row, "partition_selection_id");
        if (!over) invalid(who + ": the draw " + dname + " records no selection it is over; the data's crossings are over " + sel.name + " (rerun the draws: --replace)");
        if (*over != sel_id) invalid(who + ": the draw " + dname + " is over another selection (" + *over + ") than the data's crossings (" + sel.name + ", " + sel_id + "): set the draws' `space_level` to the data's level");
        if (D.shape.size() != 3 || D.shape[1] != r.members || D.shape[2] != 4 || D.shape[0] < 2 * static_cast<std::int64_t>(r.stage) + 2)
          invalid(who + ": the draw " + dname + " is not [stage x sense, " + std::to_string(r.members) + " members, 4] with the data's stage " + std::to_string(r.stage));
        draw.push_back(pooled_of(D.values, r.members, r.stage));
        const nxr::select::track_crossings::Pooled& P = draw.back();
        r.consistency = nt::tallyDraw(r.consistency, r.cons, nt::unbiasedConsistency(P.sum, P.n, 2));
        r.counts = nt::tallyDraw(r.counts, none, P.n);
        const Eigen::MatrixXd mr = nt::meanResultants(P.sum, P.n);
        r.rx = nt::tallyDraw(r.rx, none, mr.col(0)); r.ry = nt::tallyDraw(r.ry, none, mr.col(1)); r.rz = nt::tallyDraw(r.rz, none, mr.col(2));
      }
      if (two) {
        const auto rp = nxr::select::track_crossings::repeat(draw[0], draw[1], opt.min_crossings);
        dot_tally = nt::tallyDraw(dot_tally, dot_data, rp.dot);
        cos_tally = nt::tallyDraw(cos_tally, cos_data, Eigen::VectorXd::Constant(1, rp.cosine));
      }
    }
    nd = upto;
    std::int64_t undecided = 0;
    for (const Run& r : runs) undecided += nt::undecidedMembers(r.cons, r.consistency, opt.h);
    if (two) undecided += nt::undecidedMembers(dot_data, dot_tally, opt.h) + nt::undecidedMembers(cos_data, cos_tally, opt.h);
    const bool decided = nd >= opt.min_draws && undecided == 0;
    say(Json{{"stage", "draws"}, {"draws", nd}, {"undecided", undecided}, {"decided", decided}});
    if (decided) { stopped = "decided"; break; }
  }
  // THE P-VALUES, q-values, ratios and the surrogate's means (nxr::stats)
  const Eigen::VectorXd all_draws = Eigen::VectorXd::Constant(tiles, static_cast<double>(nd));
  const Json generator = generator_of(ddraw), seeds = seeds_of(nd, opt.batch, two), over_record = Json{{"selection_id", sel_id}, {"selection", sel.name}};
  auto store = [&](const std::string& name, const char* measure, const char* method, const task::Node& of, std::vector<double> values, std::vector<std::int64_t> shape, const Json& axes, const Json& components, const std::string& desc, Json options, const Json& warnings, bool first) {
    const std::string path = "statistics/" + nxr::io::safe_name(name);
    nxr::io::StatisticSpec st;
    st.name = name; st.path = path; st.measure = measure; st.method = method;
    st.of_field_id = of.id; st.partition_selection_id = sel_id; st.n_members = tiles;
    st.axes = axes; if (!components.is_null()) st.components = components;
    st.data = nxr::io::view(values, shape);
    st.description = desc;
    options["generator"] = generator; options["seeds"] = seeds; options["over"] = over_record;
    st.options = options;
    Json pj = first ? stamp(key) : Json::object();
    pj["by"] = "nxr-neuro run " + who; pj["warnings"] = warnings; pj["generator"] = generator;
    pj["reference"] = "Rheome lagfreenull, envelopenull, tilepq, bh; Besag and Clifford 1991; Benjamini-Hochberg";
    st.producer = pj;
    return sub.statistic(std::move(st));
  };
  const Json warnings = Json::array({"the surrogate has the data's zero-lag covariance, spectrum and envelope and nothing else: a tile beyond it is beyond what mixing and amplitude modulation produce, not a proof of a wave",
                                     "the threshold and the burst frames are the data's: only the signal is replaced",
                                     "a surrogate tile with too few crossings has no statistic and is not an exceedance: read the crossing ratio beside the p-values"});
  const Json tile_axis = Json::array({Json{{"name", "tile"}, {"selection_id", sel_id}, {"size", tiles}}});
  auto axes_with = [&](const char* name, std::vector<const char*> comps) {
    Json v = Json::array();
    for (const char* k : comps) v.push_back(Json{{"name", k}});
    Json a = tile_axis; a.push_back(Json{{"name", name}, {"size", static_cast<std::int64_t>(comps.size())}});
    return std::make_pair(a, v);
  };
  auto finite_count = [](const Eigen::VectorXd& v, double below = std::numeric_limits<double>::infinity()) { std::int64_t k = 0; for (Eigen::Index i = 0; i < v.size(); ++i) k += std::isfinite(v[i]) && v[i] < below; return k; };
  Json info = Json{{"draws", nd}, {"stopped", stopped}, {"tiles", tiles}, {"max_draws", opt.max_draws}, {"h", opt.h}, {"generator", generator}, {"over", over_record}};
  Json made = Json::array();
  {
    std::vector<std::pair<const char*, std::string>> nodes;
    std::vector<std::string> keep;                                                  // the paths must outlive the call
    for (const Run& r : runs) for (const char* w : {"consistency_p", "crossing_ratio", "resultant"}) keep.push_back("statistics/" + nxr::io::safe_name(nm(r.name, w)));
    if (two) { keep.push_back("statistics/" + nxr::io::safe_name(repeat_base + ".p")); keep.push_back("statistics/" + nxr::io::safe_name(repeat_base + ".whole_p")); }
    for (const auto& k : keep) nodes.emplace_back("statistic", k);
    bool replaced = false;
    clear_stored(ds, sub, nodes, opt.replace, who, replaced);
    info["replaced"] = replaced;
  }
  sub.compose([&] {
    bool first = true;
    for (Run& r : runs) {
      const Eigen::VectorXd p = nt::sequentialPs(r.cons, r.consistency.upper, all_draws, opt.h), q = nt::benjaminiHochbergQ(p);
      std::vector<double> pq(static_cast<std::size_t>(tiles * 3));
      for (std::int64_t t = 0; t < tiles; ++t) { pq[static_cast<std::size_t>(3 * t)] = p[t]; pq[static_cast<std::size_t>(3 * t + 1)] = q[t]; pq[static_cast<std::size_t>(3 * t + 2)] = r.consistency.upper[t]; }
      const auto [ax, comps] = axes_with("test", {"p", "q", "exceedances"});
      const Json opts = Json{{"draws", nd}, {"h", opt.h}, {"stopped", stopped}, {"min_crossings", opt.min_crossings}, {"stage", r.stage}, {"statistic", "the unbiased rho2 of the tile's pooled crossings (senses pooled), one-sided"}, {"q", "Benjamini-Hochberg over the tiles with a statistic"}, {"set", opt.set}};
      made.push_back(store(nm(r.name, "consistency_p"), "null_p", "besag_clifford", r.field, pq, {tiles, 3}, ax, comps, "per tile of " + sel.name + ": the sequential p, BH q and exceedances of the crossing consistency (unbiased rho2) of " + r.name + " against " + std::to_string(nd) + " lag-free surrogate draws", opts, warnings, first));
      first = false;
      const Eigen::VectorXd ratio = nt::countRatio(r.pooled.n, r.counts.sum, nd), mean_n = nt::nullMean(r.counts), mx = nt::nullMean(r.rx), my = nt::nullMean(r.ry), mz = nt::nullMean(r.rz);
      std::vector<double> ratio_v(ratio.data(), ratio.data() + ratio.size()), res4(static_cast<std::size_t>(tiles * 4));
      for (std::int64_t t = 0; t < tiles; ++t) { res4[static_cast<std::size_t>(4 * t)] = mx[t]; res4[static_cast<std::size_t>(4 * t + 1)] = my[t]; res4[static_cast<std::size_t>(4 * t + 2)] = mz[t]; res4[static_cast<std::size_t>(4 * t + 3)] = mean_n[t]; }
      made.push_back(store(nm(r.name, "crossing_ratio"), "crossing_ratio", "lagfree_null", r.field, ratio_v, {tiles}, tile_axis, Json(), "per tile of " + sel.name + ": the data's crossings of " + r.name + " over the lag-free surrogate's mean number (" + std::to_string(nd) + " draws)", Json{{"draws", nd}, {"stage", r.stage}}, warnings, false));
      const auto [ax4, comps4] = axes_with("component", {"resultant_x", "resultant_y", "resultant_z", "crossings"});
      made.push_back(store(nm(r.name, "resultant"), "null_resultant", "lagfree_null", r.field, res4, {tiles, 4}, ax4, comps4, "per tile of " + sel.name + ": the lag-free surrogate's mean resultant crossing vector (senses pooled) and mean crossings for " + r.name + " over " + std::to_string(nd) + " draws: what a group map subtracts", Json{{"draws", nd}, {"stage", r.stage}}, warnings, false));
      info["consistency"].push_back(Json{{"recording", r.name}, {"tiles_tested", finite_count(r.cons)}, {"tiles_q05", finite_count(q, 0.05)}});
    }
    if (two) {
      const Eigen::VectorXd p = nt::sequentialPs(dot_data, dot_tally.upper, all_draws, opt.h), q = nt::benjaminiHochbergQ(p);
      std::vector<double> pq(static_cast<std::size_t>(tiles * 3));
      for (std::int64_t t = 0; t < tiles; ++t) { pq[static_cast<std::size_t>(3 * t)] = p[t]; pq[static_cast<std::size_t>(3 * t + 1)] = q[t]; pq[static_cast<std::size_t>(3 * t + 2)] = dot_tally.upper[t]; }
      const auto [ax, comps] = axes_with("test", {"p", "q", "exceedances"});
      const Json opts = Json{{"draws", nd}, {"h", opt.h}, {"stopped", stopped}, {"min_crossings", opt.min_crossings}, {"recording_a", runs[0].name}, {"recording_b", runs[1].name}, {"statistic", "the dot product of the two runs' mean resultants, draw i of one run against draw i of the other"}};
      made.push_back(store(repeat_base + ".p", "null_p", "besag_clifford", runs[0].field, pq, {tiles, 3}, ax, comps, "per tile of " + sel.name + ": the sequential p, BH q and exceedances of the repeat (dot product of the mean resultants) of " + runs[0].name + " and " + runs[1].name + " against " + std::to_string(nd) + " surrogate draws", opts, warnings, false));
      const double pw = nt::sequentialPs(cos_data, cos_tally.upper, Eigen::VectorXd::Constant(1, static_cast<double>(nd)), opt.h)[0];
      Json axw = Json::array({Json{{"name", "whole"}, {"values", Json::array({Json{{"name", "p"}}, Json{{"name", "exceedances"}}, Json{{"name", "draws"}}})}, {"size", 3}}});
      made.push_back(store(repeat_base + ".whole_p", "null_p", "besag_clifford", runs[0].field, {pw, cos_tally.upper[0], static_cast<double>(nd)}, {3}, axw, Json(), "the sequential p of the count-weighted mean cosine of the tiles' directions between " + runs[0].name + " and " + runs[1].name + " against the surrogate's", opts, warnings, false));
      info["repeat"] = Json{{"cosine", std::isfinite(cos_data[0]) ? Json(cos_data[0]) : Json()}, {"p", std::isfinite(pw) ? Json(pw) : Json()}, {"tiles_q05", finite_count(q, 0.05)}};
    }
  });
  info["statistics"] = made;
  res.info = info;
  return res;
}

SurrogateCompareResult surrogate_lagfree_spatial_events_compare(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SurrogateSpatialEventsCompareOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  const std::string who = "surrogate_lagfree_spatial_events_compare";
  if (opt.recording.empty()) invalid(who + ": name the recording (--recording)");
  if (opt.batch < 1 || opt.min_draws < 1 || opt.max_draws < opt.min_draws || opt.h < 1) invalid(who + ": --batch, --min-draws (<= --max-draws) and --h are whole numbers >= 1");
  const graph::Definition ddraw = graph::load("surrogate_lagfree_spatial_events");
  const task::Subject subj = task::Subject::of(sub);
  const auto f = subj.find({.cls = task::Cls::field, .name = opt.recording});
  if (!f) throw Error(nxr::io::code::no_such_row, sub.name() + ": no recording (Field) named '" + opt.recording + "'");
  const std::string home_name = opt.events + ".home." + opt.recording, summary_name = opt.events + ".summary." + opt.recording;
  const nxr::io::StoredStatistic H = nxr::io::read_statistic(sub, home_name), SD = nxr::io::read_statistic(sub, summary_name);
  if (H.shape.size() != 2 || H.shape[1] != 2) invalid(who + ": " + home_name + " is not [vertices x (frames in a patch, frames)]");
  const std::int64_t V0 = H.shape[0];
  const Json vertex_axis = H.axes[0];
  const Json vertex_manifold = vertex_axis.value("manifold_id", Json());
  const Eigen::VectorXd data_home = nt::occupancyFraction(H.values, true);
  const Entries ent = entries_of(SD);
  const Eigen::VectorXd data_size = shares_of(SD, ent);
  const Eigen::Index NE = data_size.size();
  CacheKey key;
  key.pipeline = who;
  key.params = Json{{"batch", opt.batch}, {"min_draws", opt.min_draws}, {"max_draws", opt.max_draws}, {"h", opt.h}, {"set", opt.set}, {"events", opt.events}, {"generator", generator_of(ddraw)}};
  key.input = Json{{"recording", f->id}, {"home", text(H.row, "id")}, {"summary", text(SD.row, "id")}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string base = who + "." + opt.recording;
  const std::string p_home = "statistics/" + nxr::io::safe_name(base + ".home_p");
  SurrogateCompareResult res;
  if (!opt.replace)
    if (const auto hit = subj.cached(task::Cls::statistic, key.json(), p_home)) { res.hit = true; res.info = Json{{"cache", "hit"}, {"home_p", base + ".home_p"}}; say(Json{{"stage", "cache"}, {"hit", true}}); return res; }
  graph::RunOptions ro;
  ro.chunks = opt.chunks; ro.start = opt.start; ro.stop = opt.stop; ro.replace = opt.replace;
  nt::Tally home = nt::emptyTally(V0), size = nt::emptyTally(NE);
  std::int64_t nd = 0;
  std::string stopped = "max_draws";
  while (nd < opt.max_draws) {
    const std::int64_t upto = std::min<std::int64_t>(nd + opt.batch, opt.max_draws);
    for (std::int64_t i = nd; i < upto; ++i) {
      Json set = opt.set;
      set["recording"] = opt.recording; set["seed"] = i;
      if (graph::run(ds, sub, ddraw, set, ro).empty()) invalid(who + ": the draw " + std::to_string(i) + " made no run");
      const std::string hname = ddraw.name + ".draw" + std::to_string(i) + ".home." + opt.recording, sname = ddraw.name + ".draw" + std::to_string(i) + ".summary." + opt.recording;
      const nxr::io::StoredStatistic Hd = nxr::io::read_statistic(sub, hname), Sd = nxr::io::read_statistic(sub, sname);
      // THE SAME SELECTION and the same generator (spec section 7): the home map on the data's surface and vertices, the summary with the data's entries
      same_generator(Hd, ddraw, who + ": the draw " + hname);
      same_generator(Sd, ddraw, who + ": the draw " + sname);
      if (Hd.shape != H.shape) invalid(who + ": the draw " + hname + " has " + std::to_string(Hd.shape.empty() ? 0 : Hd.shape[0]) + " vertices; the data's home map " + std::to_string(V0) + ": not the same surface");
      if (Hd.axes[0].value("manifold_id", Json()) != vertex_manifold) invalid(who + ": the draw " + hname + " lies on another surface (" + Hd.axes[0].value("manifold_id", Json()).dump() + ") than the data's home map (" + vertex_manifold.dump() + ")");
      const Entries de = entries_of(Sd);
      if (de.names != ent.names) invalid(who + ": the draw " + sname + "'s patch summary names other entries than the data's (other levels or area bins): set the draws' `levels` and the area bins to the data's");
      home = nt::tallyDraw(home, data_home, nt::occupancyFraction(Hd.values, false));
      size = nt::tallyDraw(size, data_size, shares_of(Sd, de));
    }
    nd = upto;
    const std::int64_t undecided = nt::undecidedMembers(data_home, home, opt.h);
    const bool decided = nd >= opt.min_draws && undecided == 0;
    say(Json{{"stage", "draws"}, {"draws", nd}, {"undecided", undecided}, {"decided", decided}});
    if (decided) { stopped = "decided"; break; }
  }
  // THE P-VALUES (nxr::stats): home one-sided over every draw; the size shares two-sided over the draws that made patches
  const Eigen::VectorXd p = nt::sequentialPs(data_home, home.upper, Eigen::VectorXd::Constant(V0, static_cast<double>(nd)), opt.h), q = nt::benjaminiHochbergQ(p);
  const Eigen::VectorXd pu = nt::sequentialPs(data_size, size.upper, size.finite, opt.h), pl = nt::sequentialPs(data_size, size.lower, size.finite, opt.h);
  const Eigen::VectorXd qu = nt::benjaminiHochbergQ(pu), ql = nt::benjaminiHochbergQ(pl), home_null = nt::nullMean(home), size_null = nt::nullMean(size);
  std::vector<std::pair<const char*, std::string>> nodes;
  std::vector<std::string> keep;
  for (const char* w : {"home_p", "home_null", "size_null"}) keep.push_back("statistics/" + nxr::io::safe_name(base + "." + w));
  for (const auto& k : keep) nodes.emplace_back("statistic", k);
  bool replaced = false;
  clear_stored(ds, sub, nodes, opt.replace, who, replaced);
  const Json warnings = Json::array({"the surrogate has the data's zero-lag covariance, spectrum and envelope and nothing else: a vertex beyond it is occupied more than mixing and amplitude modulation under the same threshold make it, not evidence of a travelling wave",
                                     "the baseline, the threshold and the burst frames are the data's: only the signal is replaced",
                                     "a vertex the data never puts in a patch has no test (NaN); the size shares are of the patches a draw makes (a draw with none has no shares)"});
  const Json generator = generator_of(ddraw), seeds = seeds_of(nd, opt.batch, false), over_record = Json{{"vertices", vertex_axis}, {"entries", ent.names}};
  auto store = [&](const std::string& name, const char* measure, const char* method, std::vector<double> values, std::vector<std::int64_t> shape, const Json& axes, const Json& components, const std::string& desc, bool first) {
    nxr::io::StatisticSpec st;
    st.name = name; st.path = "statistics/" + nxr::io::safe_name(name); st.measure = measure; st.method = method;
    st.of_field_id = f->id;
    st.axes = axes; if (!components.is_null()) st.components = components;
    st.data = nxr::io::view(values, shape);
    st.description = desc;
    st.options = Json{{"draws", nd}, {"h", opt.h}, {"stopped", stopped}, {"set", opt.set}, {"data_home", home_name}, {"data_summary", summary_name}, {"generator", generator}, {"seeds", seeds}, {"over", over_record}};
    Json pj = first ? stamp(key) : Json::object();
    pj["by"] = "nxr-neuro run " + who; pj["warnings"] = warnings; pj["generator"] = generator;
    pj["reference"] = "Rheome lagfreenull, envelopenull, bh; Besag and Clifford 1991; Benjamini-Hochberg";
    st.producer = pj;
    return sub.statistic(std::move(st));
  };
  Json made = Json::array();
  sub.compose([&] {
    std::vector<double> pq(static_cast<std::size_t>(V0 * 3)), nf(home_null.data(), home_null.data() + home_null.size());
    for (std::int64_t v = 0; v < V0; ++v) { pq[static_cast<std::size_t>(3 * v)] = p[v]; pq[static_cast<std::size_t>(3 * v + 1)] = q[v]; pq[static_cast<std::size_t>(3 * v + 2)] = home.upper[v]; }
    Json ax3 = Json::array({vertex_axis, Json{{"name", "test"}, {"size", 3}}});
    made.push_back(store(base + ".home_p", "null_p", "besag_clifford", pq, {V0, 3}, ax3, Json::array({Json{{"name", "p"}}, Json{{"name", "q"}}, Json{{"name", "exceedances"}}}),
                         "per glued vertex: the sequential p, BH q and exceedances of the fraction of burst frames the vertex lies in a patch, against " + std::to_string(nd) + " lag-free surrogate draws", true));
    made.push_back(store(base + ".home_null", "null_fraction", "lagfree_null", nf, {V0}, Json::array({vertex_axis}), Json(), "per glued vertex: the lag-free surrogate's mean fraction of burst frames in a patch over " + std::to_string(nd) + " draws", false));
    Json en = Json::array();
    for (const auto& n : ent.names) en.push_back(Json{{"name", n}});
    std::vector<double> sz(static_cast<std::size_t>(NE * 6));
    for (Eigen::Index k = 0; k < NE; ++k) {
      double* r = &sz[static_cast<std::size_t>(6 * k)];
      r[0] = data_size[k]; r[1] = size_null[k]; r[2] = pu[k]; r[3] = pl[k]; r[4] = qu[k]; r[5] = ql[k];
    }
    made.push_back(store(base + ".size_null", "null_size", "besag_clifford", sz, {static_cast<std::int64_t>(NE), 6},
                         Json::array({Json{{"name", "entry"}, {"values", en}, {"size", static_cast<std::int64_t>(NE)}}, Json{{"name", "test"}, {"size", 6}}}),
                         Json::array({Json{{"name", "data"}}, Json{{"name", "null_mean"}}, Json{{"name", "p_upper"}}, Json{{"name", "p_lower"}}, Json{{"name", "q_upper"}}, Json{{"name", "q_lower"}}}),
                         "the size distribution of the patches (the share at each characteristic level and in each area bin, and the patches per frame): the data's, the lag-free surrogate's mean over " + std::to_string(nd) + " draws, and the two one-sided p and BH q", false));
  });
  std::int64_t tested = 0, sig = 0;
  for (std::int64_t v = 0; v < V0; ++v) { tested += std::isfinite(p[v]); sig += std::isfinite(q[v]) && q[v] < 0.05; }
  res.info = Json{{"draws", nd}, {"stopped", stopped}, {"vertices", V0}, {"vertices_tested", tested}, {"vertices_q05", sig}, {"entries", static_cast<std::int64_t>(NE)}, {"statistics", made}, {"replaced", replaced}, {"generator", generator}};
  return res;
}

}  // namespace nxr::pipelines
