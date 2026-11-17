// nxr-pipelines — the pca statistic (pca.h).
#include "nxr/pipelines/pca.h"

#include <cstdlib>

#include "nxr/io/finder.h"
#include "nxr/io/layout.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/pca.h"
#include "nxr/pipelines/cache.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }

// The table to decompose and what the stored result says about where it came from.
struct Table {
  MatrixXd X;                                  // [members x features]
  Json member_axis;                            // the axis the members lie on (copied from the input, or the partition's)
  Json feature_axes = Json::array();           // the axes the features lie on, in order (their product is the feature count)
  std::vector<std::int64_t> feature_shape;
  std::optional<std::string> unit;
  bool single = false;                         // the input is stored in float32: the result follows
  std::optional<std::string> of_field_id, of_statistic_id, partition_selection_id;
  Json input;                                  // what was read, for the provenance and the cache key
};

bool stored_single(nxr::io::SubjectWriter& sub, const Row& node) {
  if (nxr::io::is_null(node, "path")) return false;
  return nxr::io::ArrayReader::open(sub.store(), text(node, "path")).dtype() != nxr::io::DType::Float64;
}

Table table_of_statistic(nxr::io::SubjectWriter& sub, const PcaComputeOptions& opt) {
  const nxr::io::StoredStatistic st = nxr::io::read_statistic(sub, opt.of);
  if (text(st.row, "value_type") != "scalar") invalid("pca: the statistic '" + opt.of + "' holds " + text(st.row, "value_type") + " values; only real (scalar) values are decomposed. Decompose its real and imaginary parts, or a derived real quantity, as separate tables");
  if (!st.axes.is_array() || st.axes.size() != st.shape.size()) invalid("pca: the statistic '" + opt.of + "' has " + std::to_string(st.shape.size()) + " array dimensions but " + std::to_string(st.axes.is_array() ? st.axes.size() : 0) + " axes");
  const std::size_t nd = st.shape.size();
  // the members' axis, by position or by name (a name that two axes share is ambiguous: refused with the axes listed)
  std::string listing;
  for (std::size_t i = 0; i < nd; ++i) listing += (i ? ", " : "") + std::to_string(i) + " " + st.axes[i].value("name", std::string("?")) + " (" + std::to_string(st.shape[i]) + ")";
  std::size_t a = nd;
  const std::string m = *opt.members;
  char* end = nullptr;
  const long pos = std::strtol(m.c_str(), &end, 10);
  if (!m.empty() && end && *end == '\0') {
    if (pos < 0 || static_cast<std::size_t>(pos) >= nd) invalid("pca: --members " + m + " is not an axis of '" + opt.of + "': its axes are " + listing);
    a = static_cast<std::size_t>(pos);
  } else {
    for (std::size_t i = 0; i < nd; ++i)
      if (st.axes[i].value("name", std::string()) == m) {
        if (a != nd) invalid("pca: --members " + m + " names more than one axis of '" + opt.of + "'; give its position: " + listing);
        a = i;
      }
    if (a == nd) invalid("pca: '" + opt.of + "' has no axis named " + m + ": its axes are " + listing);
  }
  if (nd < 2) invalid("pca: the statistic '" + opt.of + "' has one axis: a table needs the members and at least one feature axis");
  Table t;
  t.member_axis = st.axes[a];
  std::int64_t n = st.shape[a], p = 1;
  std::vector<std::size_t> others;
  for (std::size_t i = 0; i < nd; ++i) if (i != a) { others.push_back(i); t.feature_axes.push_back(st.axes[i]); t.feature_shape.push_back(st.shape[i]); p *= st.shape[i]; }
  // X[member, feature]: the member axis moved first, the other axes in order (C order)
  std::vector<std::int64_t> stride(nd, 1);
  for (std::size_t i = nd; i-- > 1;) stride[i - 1] = stride[i] * st.shape[i];
  t.X.resize(n, p);
  std::vector<std::int64_t> idx(others.size(), 0);
  for (std::int64_t f = 0; f < p; ++f) {
    std::int64_t base = 0;
    for (std::size_t j = 0; j < others.size(); ++j) base += idx[j] * stride[others[j]];
    for (std::int64_t i = 0; i < n; ++i) t.X(i, f) = st.values[static_cast<std::size_t>(base + i * stride[a])];
    for (std::size_t j = others.size(); j-- > 0;) { if (++idx[j] < st.shape[others[j]]) break; idx[j] = 0; }
  }
  t.unit = nxr::io::opt_text(st.row, "unit");
  t.single = stored_single(sub, st.row);
  t.of_statistic_id = text(st.row, "id");
  t.input = Json{{"form", "statistic"}, {"statistic", opt.of}, {"statistic_id", text(st.row, "id")}, {"members_axis", static_cast<std::int64_t>(a)}, {"shape", st.shape}};
  return t;
}

Table table_of_measurement(nxr::io::SubjectWriter& sub, const PcaComputeOptions& opt) {
  if (opt.measure.empty()) invalid("pca: --over needs --measure: the stored measure whose members' values are decomposed");
  const auto sel = sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.over)});
  if (!sel) missing(sub.name() + ": no Selection named '" + *opt.over + "'");
  std::optional<Row> field = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(opt.of)});
  std::optional<nxr::io::MapSlice> slice;
  if (!field)
    for (const Row& st : nxr::io::statistics_of(sub.db(), sub.id())) {
      const Json axes = Json::parse(text(st, "axes_json"), nullptr, false);
      if (!axes.is_array() || axes.empty() || !axes[0].contains("manifold_id")) continue;
      for (const auto& ms : nxr::io::map_slices(sub.db(), sub.id(), axes[0]["manifold_id"].get<std::string>()))
        if (ms.name == opt.of && text(ms.statistic, "id") == text(st, "id")) slice = ms;
    }
  if (!field && !slice) missing(sub.name() + ": nothing named '" + opt.of + "' (a Field, or a band of a map statistic)");
  const auto mv = nxr::io::measurement_values(sub, text(*sel, "id"), field && !opt.own ? std::optional<std::string>(text(*field, "id")) : std::nullopt,
                                              slice && !opt.own ? std::optional<std::string>(text(slice->statistic, "id")) : std::nullopt,
                                              slice && !opt.own ? std::optional<std::int64_t>(slice->index) : std::nullopt, opt.measure);
  Table t;
  t.X.resize(mv.members, mv.rows);              // measurement_values is [rows x members]; the members are the observations
  for (std::int64_t r = 0; r < mv.rows; ++r)
    for (std::int64_t m = 0; m < mv.members; ++m) t.X(m, r) = mv.values[static_cast<std::size_t>(r * mv.members + m)];
  t.member_axis = Json{{"name", "member"}, {"selection_id", text(*sel, "id")}, {"size", mv.members}};
  if (field && mv.array_backed && !opt.own) {
    const auto f = nxr::io::finder::factors(sub.db(), text(*field, "manifold_id"));
    t.feature_axes.push_back(Json{{"name", "row"}, {"manifold_id", f[0]}, {"size", mv.rows}});
  } else {
    t.feature_axes.push_back(Json{{"name", "row"}, {"values", Json::array({Json{{"name", opt.of}}})}, {"size", mv.rows}});
  }
  t.feature_shape = {mv.rows};
  t.unit = mv.unit;
  t.single = field ? stored_single(sub, *field) : stored_single(sub, slice->statistic);
  if (field) t.of_field_id = text(*field, "id"); else t.of_statistic_id = text(slice->statistic, "id");
  t.partition_selection_id = text(*sel, "id");
  t.input = Json{{"form", "measurement"}, {"over", text(*sel, "id")}, {"of", field ? text(*field, "id") : text(slice->statistic, "id")}, {"of_index", slice ? Json(slice->index) : Json()},
                 {"measure", opt.measure}, {"computed_by", mv.computed_by}, {"own", opt.own}};
  return t;
}

template <class T> std::vector<T> cast(const std::vector<double>& v) { return std::vector<T>(v.begin(), v.end()); }
// the values as the input's stored precision (float32 input: float32), or float64
nxr::io::ArrayView values_view(const std::vector<double>& v, bool single, std::vector<float>& f32, const nxr::io::Shape& shape) {
  if (single) { f32 = cast<float>(v); return nxr::io::view(f32, shape); }
  return nxr::io::view(v, shape);
}
std::vector<double> flat(const MatrixXd& A) {       // C order, row-major
  std::vector<double> out(static_cast<std::size_t>(A.size()));
  for (Eigen::Index i = 0; i < A.rows(); ++i) for (Eigen::Index j = 0; j < A.cols(); ++j) out[static_cast<std::size_t>(i * A.cols() + j)] = A(i, j);
  return out;
}
}  // namespace

PcaComputeResult pca_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PcaComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.of.empty()) invalid("pca: --of names what is decomposed");
  if (opt.scale != "none" && opt.scale != "unit-variance") invalid("pca: --scale is none or unit-variance (got '" + opt.scale + "')");
  if (opt.members.has_value() == opt.over.has_value()) invalid("pca: name the table's members one way: --members AXIS (of a stored statistic), or --over PARTITION with --measure (a stored measurement); " + std::string(opt.members ? "both were given" : "neither was given"));
  Table t = opt.members ? table_of_statistic(sub, opt) : table_of_measurement(sub, opt);
  const std::string name = opt.name ? *opt.name : nxr::io::safe_name(opt.of + "_pca");
  const std::string n_scores = name + "_scores", n_var = name + "_variance", n_cen = name + "_centring";
  auto path_of = [](const std::string& n) { return "statistics/" + nxr::io::safe_name(n); };
  CacheKey key;
  key.pipeline = "pca compute";
  key.params = Json{{"name", name}, {"scale", opt.scale}};
  key.input = t.input;
  key.compute = NXR_COMPUTE_VERSION;
  PcaComputeResult res;
  if (const auto hit = find_cached(sub.db(), sub.id(), "statistic", key, path_of(name))) {
    res.hit = true;
    res.info = Json{{"name", name}, {"id", text(*hit, "id")}, {"path", path_of(name)}, {"cache", "hit"}};
    say(Json{{"stage", "cache"}, {"hit", true}, {"statistic", name}});
    return res;
  }
  nxr::pca::PcaOptions po;
  po.scale = opt.scale == "unit-variance" ? nxr::pca::Scale::UnitVariance : nxr::pca::Scale::None;
  nxr::pca::PcaResult r;
  try { r = nxr::pca::pca(t.X, po); } catch (const std::invalid_argument& e) { invalid(std::string("pca of '") + opt.of + "': " + e.what()); }
  const std::int64_t n = t.X.rows(), p = t.X.cols(), k = r.rank;
  bool replaced = false;
  std::string moved = clear_stored(ds, sub, {{"statistic", path_of(name)}, {"statistic", path_of(n_scores)}, {"statistic", path_of(n_var)}, {"statistic", path_of(n_cen)}}, opt.replace, name, replaced);
  // the axes: the components' (k, then the features), the scores' (the members, then the components) ...
  Json comp_index = Json::array();                                  // the component axis is its numbers, 1 .. k (the order of decreasing variance)
  for (std::int64_t c = 1; c <= k; ++c) comp_index.push_back(c);
  const Json comp_axis = Json{{"name", "component"}, {"size", k}, {"values", comp_index}};
  auto with_features = [&](const Json& first) { Json a = Json::array({first}); for (const auto& f : t.feature_axes) a.push_back(f); return a; };
  nxr::io::Shape comp_shape{k}, cen_shape{2};
  for (auto s : t.feature_shape) { comp_shape.push_back(s); cen_shape.push_back(s); }
  const Json options = Json{{"scale", opt.scale}, {"input", t.input}, {"sign", "the entry of largest absolute loading of each component is positive"}, {"rank", k},
                            {"statistics", Json{{"components", name}, {"scores", n_scores}, {"variance", n_var}, {"centring", n_cen}}}, {"members", n}, {"features", p}};
  Json pj = stamp(key);
  pj["by"] = "nxr-neuro pca compute";
  pj["producer"] = "nxr-neuro pca compute";
  pj["layout"] = "components [component x features]: unit rows; scores [member x component] = U S; variance [component x (singular value, explained variance S^2/(n-1), ratio)]; centring [(mean, divisor) x features]; reconstruction at k: scores[:, :k] components[:k] x divisor + mean";
  std::string id;
  std::vector<float> f32a, f32b, f32c;
  const std::vector<double> d_comp = flat(r.components), d_scores = flat(r.scores);   // held until the composition has written them (a view does not own its data)
  sub.compose([&] {
    auto base = [&](const std::string& nm, const char* measure, const char* what) {
      nxr::io::StatisticSpec st;
      st.name = nm; st.path = path_of(nm); st.measure = measure; st.method = "pca";
      st.of_field_id = t.of_field_id; st.of_statistic_id = t.of_statistic_id; st.partition_selection_id = t.partition_selection_id;
      st.n_members = n;
      st.options = options;
      st.producer = pj;
      st.description = std::string(what) + " of " + opt.of + " across its " + std::to_string(n) + " members (" + std::to_string(p) + " features, scale " + opt.scale + ")";
      return st;
    };
    {   // the components
      nxr::io::StatisticSpec st = base(name, "principal_components", "the principal components");
      st.axes = with_features(comp_axis);
      st.data = values_view(d_comp, t.single, f32a, comp_shape);
      id = sub.statistic(std::move(st));
    }
    {   // the scores
      nxr::io::StatisticSpec st = base(n_scores, "principal_scores", "the scores on the principal components");
      st.axes = Json::array({t.member_axis, comp_axis});
      st.data = values_view(d_scores, t.single, f32b, {n, k});
      if (t.unit && opt.scale == "none") st.unit = t.unit;
      sub.statistic(std::move(st));
    }
    {   // the variance table (float64)
      nxr::io::StatisticSpec st = base(n_var, "explained_variance", "the variance explained by each principal component");
      MatrixXd v(k, 3);
      v.col(0) = r.singularValues; v.col(1) = r.explainedVariance; v.col(2) = r.explainedVarianceRatio;
      st.axes = Json::array({comp_axis, Json{{"name", "quantity"}, {"values", Json::array({Json{{"name", "singular_value"}}, Json{{"name", "explained_variance"}}, Json{{"name", "explained_variance_ratio"}}})}, {"size", 3}}});
      const std::vector<double> vf = flat(v);
      st.data = nxr::io::view(vf, {k, 3});
      sub.statistic(std::move(st));
    }
    {   // the centring: the mean and the divisor of each feature
      nxr::io::StatisticSpec st = base(n_cen, "pca_centring", "the centring of the table");
      MatrixXd c(2, p);
      c.row(0) = r.mean.transpose(); c.row(1) = r.scale.transpose();
      Json qa = Json{{"name", "quantity"}, {"values", Json::array({Json{{"name", "mean"}}, Json{{"name", "divisor"}}})}, {"size", 2}};
      st.axes = with_features(qa);
      const std::vector<double> dc = flat(c);
      st.data = values_view(dc, t.single, f32c, cen_shape);
      if (t.unit) st.unit = t.unit;
      sub.statistic(std::move(st));
    }
  });
  res.info = Json{{"name", name}, {"id", id}, {"path", path_of(name)}, {"statistics", Json::array({name, n_scores, n_var, n_cen})}, {"members", n}, {"features", p}, {"rank", k},
                  {"explained_variance_ratio", Json(std::vector<double>(r.explainedVarianceRatio.data(), r.explainedVarianceRatio.data() + std::min<std::int64_t>(k, 10)))},
                  {"scale", opt.scale}, {"replaced", replaced}, {"moved_aside", moved.empty() ? Json() : Json(moved)}, {"cache", "miss"}};
  return res;
}

StoredPca pca_read(nxr::io::SubjectWriter& sub, const std::string& name) {
  const nxr::io::StoredStatistic comp = nxr::io::read_statistic(sub, name);
  if (text(comp.row, "method") != "pca" || text(comp.row, "measure") != "principal_components") invalid("'" + name + "' is not the components of a pca (`compute pca`): its method is " + text(comp.row, "method") + ", its measure " + text(comp.row, "measure"));
  const nxr::io::StoredStatistic scores = nxr::io::read_statistic(sub, name + "_scores"), var = nxr::io::read_statistic(sub, name + "_variance"), cen = nxr::io::read_statistic(sub, name + "_centring");
  StoredPca s;
  const std::int64_t k = comp.shape.at(0);
  std::int64_t p = 1;
  for (std::size_t i = 1; i < comp.shape.size(); ++i) { s.feature_shape.push_back(comp.shape[i]); p *= comp.shape[i]; }
  const std::int64_t n = scores.shape.at(0);
  if (scores.shape.at(1) != k || var.shape.at(0) != k || var.shape.at(1) != 3 || cen.shape.at(0) != 2) invalid("'" + name + "': the stored scores, variance and centring do not match the components (" + std::to_string(k) + ")");
  auto mat = [](const std::vector<double>& v, std::int64_t r, std::int64_t c) {
    MatrixXd M(r, c);
    for (std::int64_t i = 0; i < r; ++i) for (std::int64_t j = 0; j < c; ++j) M(i, j) = v[static_cast<std::size_t>(i * c + j)];
    return M;
  };
  s.components = mat(comp.values, k, p);
  s.scores = mat(scores.values, n, k);
  const MatrixXd v = mat(var.values, k, 3), c = mat(cen.values, 2, p);
  s.singular_values = v.col(0); s.explained_variance = v.col(1); s.explained_variance_ratio = v.col(2);
  s.mean = c.row(0).transpose(); s.scale = c.row(1).transpose();
  s.options = comp.options;
  return s;
}

}  // namespace nxr::pipelines
