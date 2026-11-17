// nxr-pipelines — statistic summaries (statistics.h).
#include "nxr/pipelines/statistics.h"

#include "nxr/pipelines/task.h"

#include "nxr/io/finder.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/cache.h"
#include "nxr/stats.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }
}  // namespace

StatisticComputeResult statistic_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const StatisticComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  std::vector<nxr::stats::Summary> sums;
  try { sums = nxr::stats::parseSummaries(opt.summaries); } catch (const std::invalid_argument& e) { invalid(std::string("statistic compute: ") + e.what()); }
  if (opt.combine != "each" && opt.combine != "pool" && opt.combine != "mean") invalid("statistic compute: --combine is each, pool or mean");
  if (opt.ddof < 0) invalid("statistic compute: --ddof is at least 0");
  // THE PARTITION and WHAT WAS MEASURED: a Field, or a band of a map statistic
  const task::Subject subj = task::Subject::of(sub);
  const auto sel = subj.find({.cls = task::Cls::selection, .name = opt.over});
  if (!sel) missing(sub.name() + ": no Selection named '" + opt.over + "'");
  std::optional<task::Node> field = subj.find({.cls = task::Cls::field, .name = opt.of});
  std::optional<task::MapSlice> slice;
  if (!field) slice = subj.map_slice(opt.of);
  if (!field && !slice) missing(sub.name() + ": nothing named '" + opt.of + "' (a Field, or a band of a map statistic)");
  // `own`: the measure is the members' OWN (rows of no Field: area, n_elements, n_tiles); the Field is then only the statistic's parent
  const auto mv = nxr::io::measurement_values(sub, sel->id, field && !opt.own ? std::optional<std::string>(field->id) : std::nullopt,
                                              slice && !opt.own ? std::optional<std::string>(slice->statistic_id) : std::nullopt,
                                              slice && !opt.own ? std::optional<std::int64_t>(slice->index) : std::nullopt, opt.measure);
  const std::string name = opt.name ? *opt.name : nxr::io::safe_name(opt.of + "_" + opt.measure + "_" + opt.over);
  const std::string path = (field && field->path ? *field->path + "_statistics/" : std::string("statistics/")) + nxr::io::safe_name(name);
  Json sjson = Json::array();
  for (const auto& sm : sums) sjson.push_back(sm.name == "percentile" ? Json{{"name", "percentile"}, {"p", sm.p}} : Json{{"name", sm.name}});
  CacheKey key;
  key.pipeline = "statistic compute";
  key.params = Json{{"name", name}, {"summaries", sjson}, {"combine", opt.combine}, {"ddof", opt.ddof}};
  key.input = Json{{"over", sel->id}, {"of", field ? field->id : slice->statistic_id}, {"of_index", slice ? Json(slice->index) : Json()},
                   {"measure", opt.measure}, {"computed_by", mv.computed_by}, {"own", opt.own}};
  key.compute = NXR_COMPUTE_VERSION;
  StatisticComputeResult res;
  if (const auto hit = subj.cached(task::Cls::statistic, key.json(), path)) {
    res.hit = true;
    res.info = Json{{"name", name}, {"id", hit->id}, {"path", path}, {"cache", "hit"}};
    say(Json{{"stage", "cache"}, {"hit", true}, {"statistic", name}});
    return res;
  }
  std::vector<std::int64_t> used;
  const std::vector<double> out = nxr::stats::summarizeRows(mv.values, mv.rows, mv.members, sums, opt.combine, opt.ddof, &used);
  const std::int64_t R = opt.combine == "each" ? mv.rows : 1, S = static_cast<std::int64_t>(sums.size());
  // the rows' axis: the measured Field's first factor (its channels, its vertices) when per row of an array; else what the row is
  Json rows_axis;
  if (opt.combine == "each" && field && mv.array_backed) {
    const auto f = subj.factors(field->manifold_id.value_or(std::string()));
    rows_axis = Json{{"name", "row"}, {"manifold_id", f[0]}, {"size", R}};
  } else {
    rows_axis = Json{{"name", "row"}, {"values", Json::array({Json{{"name", opt.combine == "each" ? opt.of : opt.combine}}})}, {"size", R}};
  }
  std::int64_t least = mv.members;
  for (auto u : used) least = std::min(least, u);
  bool replaced = false;
  clear_stored(ds, sub, {{"statistic", path}}, opt.replace, name, replaced);
  std::string id;
  sub.compose([&] {
    nxr::io::StatisticSpec st;
    st.name = name; st.path = path; st.measure = opt.measure; st.method = "summary";
    if (field) st.of_field_id = field->id; else st.of_statistic_id = slice->statistic_id;
    st.partition_selection_id = sel->id;
    st.n_members = mv.members; st.members_skipped = mv.members - least;
    st.axes = Json::array({rows_axis, Json{{"name", "summary"}, {"values", sjson}, {"size", S}}});
    st.data = nxr::io::view(out, {R, S});
    st.unit = mv.unit;
    st.description = "the " + opt.measure + " of " + opt.of + " over the members of " + opt.over + ", summarised (" + opt.combine + ")";
    st.options = Json{{"summaries", sjson}, {"combine", opt.combine}, {"ddof", opt.ddof}, {"of_index", slice ? Json(slice->index) : Json()}, {"own", opt.own}};
    Json pj = stamp(key);
    pj["by"] = "nxr-neuro statistic compute"; pj["members_used"] = used; pj["array_backed"] = mv.array_backed;
    pj["unit_note"] = "the unit is the measurement's: kurtosis and skewness are unitless";
    st.producer = pj;
    id = sub.statistic(std::move(st));
  });
  res.info = Json{{"name", name}, {"id", id}, {"path", path}, {"rows", R}, {"members", mv.members}, {"members_used", used}, {"summaries", sjson}, {"replaced", replaced}, {"cache", "miss"}};
  return res;
}

}  // namespace nxr::pipelines
