// nxr-pipelines — the lag-free null of the spatial events (spatial_null.h).
#include "nxr/pipelines/spatial_null.h"

#include "nxr/pipelines/task.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "nxr/io/statistic.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/definition.h"
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
const double kNaN = std::numeric_limits<double>::quiet_NaN();
// the summary's entries that are SHARES of the patches (level_*, area_bin_*) and the rate (patches per frame): from the stored counts
struct Entries { std::vector<std::string> names; std::vector<std::string> kinds; };
std::vector<double> shares_of(const nxr::io::StoredStatistic& S, Entries* names = nullptr) {
  const Json& v = S.axes[0]["values"];
  double patches = 0, frames = 0;
  for (std::size_t k = 0; k < v.size(); ++k) { if (v[k]["name"] == "patches") patches = S.values[k]; if (v[k]["name"] == "frames") frames = S.values[k]; }
  std::vector<double> out;
  for (std::size_t k = 0; k < v.size(); ++k) {
    const std::string n = v[k]["name"].get<std::string>();
    if (n.rfind("level_", 0) == 0 || n.rfind("area_bin_", 0) == 0) { out.push_back(patches > 0 ? S.values[k] / patches : kNaN); if (names) names->names.push_back(n); }
  }
  out.push_back(frames > 0 ? patches / frames : kNaN);
  if (names) names->names.push_back("patches_per_frame");
  return out;
}
}  // namespace

SpatialNullResult spatial_null_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const SpatialNullOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.recording.empty()) invalid("spatialnull compute: name the recording (--recording)");
  if (opt.batch < 1 || opt.min_draws < 1 || opt.max_draws < opt.min_draws || opt.h < 1) invalid("spatialnull compute: --batch, --min-draws (<= --max-draws) and --h are whole numbers >= 1");
  const graph::Definition dnull = graph::load("spatial_events_null");
  const task::Subject subj = task::Subject::of(sub);
  const auto f = subj.find({.cls = task::Cls::field, .name = opt.recording});
  if (!f) throw Error(nxr::io::code::no_such_row, sub.name() + ": no recording (Field) named '" + opt.recording + "'");
  const nxr::io::StoredStatistic H = nxr::io::read_statistic(sub, opt.events + ".home." + opt.recording), SD = nxr::io::read_statistic(sub, opt.events + ".summary." + opt.recording);
  if (H.shape.size() != 2 || H.shape[1] != 2) invalid("spatialnull compute: " + opt.events + ".home." + opt.recording + " is not [vertices x (frames in a patch, frames)]");
  const std::int64_t V0 = H.shape[0];
  Eigen::VectorXd data_home(V0);
  for (std::int64_t v = 0; v < V0; ++v) data_home[v] = H.values[static_cast<std::size_t>(2 * v + 1)] > 0 && H.values[static_cast<std::size_t>(2 * v)] > 0 ? H.values[static_cast<std::size_t>(2 * v)] / H.values[static_cast<std::size_t>(2 * v + 1)] : kNaN;
  Entries ent;
  const std::vector<double> data_size = shares_of(SD, &ent);
  const std::size_t NE = data_size.size();
  CacheKey key;
  key.pipeline = "spatialnull compute";
  key.params = Json{{"batch", opt.batch}, {"min_draws", opt.min_draws}, {"max_draws", opt.max_draws}, {"h", opt.h}, {"set", opt.set}, {"events", opt.events}};
  key.input = Json{{"recording", f->id}, {"home", text(H.row, "id")}, {"summary", text(SD.row, "id")}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string base = "spatial_events.null." + opt.recording;
  const std::string p_home = "statistics/" + nxr::io::safe_name(base + ".home_p");
  SpatialNullResult res;
  if (!opt.replace)
    if (const auto hit = subj.cached(task::Cls::statistic, key.json(), p_home)) { res.hit = true; res.info = Json{{"cache", "hit"}, {"home_p", base + ".home_p"}}; say(Json{{"stage", "cache"}, {"hit", true}}); return res; }
  graph::RunOptions ro;
  ro.chunks = opt.chunks; ro.start = opt.start; ro.stop = opt.stop; ro.replace = opt.replace;
  Eigen::VectorXd ex_home = Eigen::VectorXd::Zero(V0), sum_home = Eigen::VectorXd::Zero(V0);
  std::vector<double> ex_up(NE, 0.0), ex_lo(NE, 0.0), sum_size(NE, 0.0), n_size(NE, 0.0);
  std::int64_t nd = 0;
  std::string stopped = "max_draws";
  while (nd < opt.max_draws) {
    const std::int64_t upto = std::min<std::int64_t>(nd + opt.batch, opt.max_draws);
    for (std::int64_t i = nd; i < upto; ++i) {
      Json set = opt.set;
      set["recording"] = opt.recording; set["seed"] = i;
      if (graph::run(ds, sub, dnull, set, ro).empty()) invalid("spatialnull compute: the draw " + std::to_string(i) + " made no run");
      const nxr::io::StoredStatistic Hd = nxr::io::read_statistic(sub, "spatial_events_null.draw" + std::to_string(i) + ".home." + opt.recording), Sd = nxr::io::read_statistic(sub, "spatial_events_null.draw" + std::to_string(i) + ".summary." + opt.recording);
      if (Hd.shape != H.shape) invalid("spatialnull compute: the draw's home map is not the data's shape");
      for (std::int64_t v = 0; v < V0; ++v) {
        const double fr = Hd.values[static_cast<std::size_t>(2 * v + 1)], d = fr > 0 ? Hd.values[static_cast<std::size_t>(2 * v)] / fr : 0.0;
        sum_home[v] += d;
        if (std::isfinite(data_home[v]) && d >= data_home[v]) ex_home[v] += 1;
      }
      const std::vector<double> sz = shares_of(Sd);
      for (std::size_t k = 0; k < NE; ++k) {
        if (!std::isfinite(sz[k]) || !std::isfinite(data_size[k])) continue;
        sum_size[k] += sz[k]; n_size[k] += 1;
        if (sz[k] >= data_size[k]) ex_up[k] += 1;
        if (sz[k] <= data_size[k]) ex_lo[k] += 1;
      }
    }
    nd = upto;
    bool decided = nd >= opt.min_draws;
    for (std::int64_t v = 0; decided && v < V0; ++v) if (std::isfinite(data_home[v]) && ex_home[v] < opt.h) decided = false;
    say(Json{{"stage", "draws"}, {"draws", nd}, {"decided", decided}});
    if (decided) { stopped = "decided"; break; }
  }
  // THE P-VALUES
  Eigen::VectorXd p(V0);
  for (std::int64_t v = 0; v < V0; ++v) p[v] = std::isfinite(data_home[v]) ? nxr::surrogate::besagCliffordP(static_cast<std::int64_t>(ex_home[v]), nd, opt.h) : kNaN;
  const Eigen::VectorXd q = nxr::surrogate::benjaminiHochberg(p);
  Eigen::VectorXd pu(NE), pl(NE);
  for (std::size_t k = 0; k < NE; ++k) {
    const std::int64_t n = static_cast<std::int64_t>(n_size[k]);
    pu[static_cast<Eigen::Index>(k)] = n > 0 ? nxr::surrogate::besagCliffordP(static_cast<std::int64_t>(ex_up[k]), n, opt.h) : kNaN;
    pl[static_cast<Eigen::Index>(k)] = n > 0 ? nxr::surrogate::besagCliffordP(static_cast<std::int64_t>(ex_lo[k]), n, opt.h) : kNaN;
  }
  const Eigen::VectorXd qu = nxr::surrogate::benjaminiHochberg(pu), ql = nxr::surrogate::benjaminiHochberg(pl);
  std::vector<std::pair<const char*, std::string>> nodes;
  std::vector<std::string> keep;
  for (const char* w : {"home_p", "home_null", "size_null"}) keep.push_back("statistics/" + nxr::io::safe_name(base + "." + w));
  for (const auto& k : keep) nodes.emplace_back("statistic", k);
  bool replaced = false;
  clear_stored(ds, sub, nodes, opt.replace, "spatialnull compute", replaced);
  const Json warnings = Json::array({"the null has the data's zero-lag covariance, spectrum and envelope and nothing else: a vertex beyond it is occupied more than mixing and amplitude modulation under the same threshold make it, not evidence of a travelling wave",
                                     "the baseline, the threshold and the burst frames are the data's: only the signal is replaced",
                                     "a vertex the data never puts in a patch has no test (NaN); the size shares are of the patches a draw makes (a draw with none has no shares)"});
  const Json vertex_axis = H.axes[0];
  auto store = [&](const std::string& name, const char* measure, const char* method, std::vector<double> values, std::vector<std::int64_t> shape, const Json& axes, const Json& components, const std::string& desc, bool first) {
    nxr::io::StatisticSpec st;
    st.name = name; st.path = "statistics/" + nxr::io::safe_name(name); st.measure = measure; st.method = method;
    st.of_field_id = f->id;
    st.axes = axes; if (!components.is_null()) st.components = components;
    st.data = nxr::io::view(values, shape);
    st.description = desc;
    st.options = Json{{"draws", nd}, {"h", opt.h}, {"stopped", stopped}, {"set", opt.set}, {"data_home", opt.events + ".home." + opt.recording}, {"data_summary", opt.events + ".summary." + opt.recording}};
    Json pj = first ? stamp(key) : Json::object();
    pj["by"] = "nxr-neuro spatialnull compute"; pj["warnings"] = warnings; pj["reference"] = "nxr vortex.envelope_matched_null; Besag and Clifford 1991; Benjamini-Hochberg";
    st.producer = pj;
    return sub.statistic(std::move(st));
  };
  Json made = Json::array();
  sub.compose([&] {
    std::vector<double> pq(static_cast<std::size_t>(V0 * 3)), nf(static_cast<std::size_t>(V0));
    for (std::int64_t v = 0; v < V0; ++v) { pq[static_cast<std::size_t>(3 * v)] = p[v]; pq[static_cast<std::size_t>(3 * v + 1)] = q[v]; pq[static_cast<std::size_t>(3 * v + 2)] = ex_home[v]; nf[static_cast<std::size_t>(v)] = sum_home[v] / static_cast<double>(nd); }
    Json ax3 = Json::array({vertex_axis, Json{{"name", "test"}, {"size", 3}}});
    made.push_back(store(base + ".home_p", "null_p", "besag_clifford", pq, {V0, 3}, ax3, Json::array({Json{{"name", "p"}}, Json{{"name", "q"}}, Json{{"name", "exceedances"}}}),
                         "per glued vertex: the sequential p, BH q and exceedances of the fraction of burst frames the vertex lies in a patch, against " + std::to_string(nd) + " lag-free null draws", true));
    made.push_back(store(base + ".home_null", "null_fraction", "lagfree_null", nf, {V0}, Json::array({vertex_axis}), Json(), "per glued vertex: the lag-free null's mean fraction of burst frames in a patch over " + std::to_string(nd) + " draws", false));
    Json en = Json::array();
    for (const auto& n : ent.names) en.push_back(Json{{"name", n}});
    std::vector<double> sz(NE * 6);
    for (std::size_t k = 0; k < NE; ++k) {
      sz[6 * k] = data_size[k]; sz[6 * k + 1] = n_size[k] > 0 ? sum_size[k] / n_size[k] : kNaN;
      sz[6 * k + 2] = pu[static_cast<Eigen::Index>(k)]; sz[6 * k + 3] = pl[static_cast<Eigen::Index>(k)]; sz[6 * k + 4] = qu[static_cast<Eigen::Index>(k)]; sz[6 * k + 5] = ql[static_cast<Eigen::Index>(k)];
    }
    made.push_back(store(base + ".size_null", "null_size", "besag_clifford", sz, {static_cast<std::int64_t>(NE), 6},
                         Json::array({Json{{"name", "entry"}, {"values", en}, {"size", static_cast<std::int64_t>(NE)}}, Json{{"name", "test"}, {"size", 6}}}),
                         Json::array({Json{{"name", "data"}}, Json{{"name", "null_mean"}}, Json{{"name", "p_upper"}}, Json{{"name", "p_lower"}}, Json{{"name", "q_upper"}}, Json{{"name", "q_lower"}}}),
                         "the size distribution of the patches (the share at each characteristic level and in each area bin, and the patches per frame): the data's, the lag-free null's mean over " + std::to_string(nd) + " draws, and the two one-sided p and BH q", false));
  });
  std::int64_t tested = 0, sig = 0;
  for (std::int64_t v = 0; v < V0; ++v) { tested += std::isfinite(p[v]); sig += std::isfinite(q[v]) && q[v] < 0.05; }
  res.info = Json{{"draws", nd}, {"stopped", stopped}, {"vertices", V0}, {"vertices_tested", tested}, {"vertices_q05", sig}, {"entries", static_cast<std::int64_t>(NE)}, {"statistics", made}, {"replaced", replaced}};
  return res;
}

}  // namespace nxr::pipelines
