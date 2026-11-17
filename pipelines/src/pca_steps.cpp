// nxr-pipelines — THE PCA NODE: the principal components ACROSS the members of a table (members x features), as a pure node of nxr::graph::registry() (grouped name `statistic.pca`).
// Spec docs/specs/2026-11-16-meg-fingerprinting.md, 4.4. The maths is `nxr::pca` (compute/include/nxr/pca.h: a thin SVD of the centred table, never the features x features
// covariance); this node reads the stored statistic it is handed as a table (type-checked, refused with its reason), calls it and DESCRIBES the four statistics it made; the
// definition (`pca.yaml`) says where they are stored.
//
//   statistic.pca  array  x (a stored statistic of real values, two axes: the members and the features)  ->  components, scores, variance, centring
//
// The four outputs, every component up to the table's rank (so any k is available; reconstruction at k is derived at read, never stored):
//   components   measure principal_components   [component x feature]   unit-length loadings, the sign convention of nxr::pca (largest loading positive)
//   scores       measure principal_scores       [member x component]     the table's rows on the components, U S
//   variance     measure explained_variance     [component x 3]          singular value, explained variance (S^2 / (n - 1)), its ratio over the total
//   centring     measure pca_centring           [2 x feature]            the mean removed from each feature, and the divisor after it (1 unless unit-variance)
// Precision follows the input's stored precision (float32 in: components, scores and centring in float32); the variance table is float64.
//
// What it cannot show: a component is a direction of variance in THIS sample; with n members at most n - 1 components exist, and the leading ones of a small sample are not stable
// estimates of a population's. A statistic of more than two axes (several feature axes) is reduced to a table by a step before it, not here.
#include "nxr/pipelines/pca_steps.h"

#include <cstdlib>
#include <string>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/pca.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

ng::PortType statistic_out() {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = "statistic";
  return t;
}

// the axis whose entries are the members: its name (when unique among the axes) or its position, from 0
std::size_t members_axis(const Json& axes, const std::string& m, const std::string& statistic) {
  std::string listing;
  for (std::size_t i = 0; i < axes.size(); ++i) listing += (i ? ", " : "") + std::to_string(i) + " " + axes[i].value("name", std::string("?"));
  char* end = nullptr;
  const long pos = std::strtol(m.c_str(), &end, 10);
  if (!m.empty() && end && *end == '\0') {
    if (pos < 0 || static_cast<std::size_t>(pos) >= axes.size()) ng::refuse("statistic.pca: members " + m + " is not an axis of '" + statistic + "': its axes are " + listing);
    return static_cast<std::size_t>(pos);
  }
  std::size_t found = axes.size();
  for (std::size_t i = 0; i < axes.size(); ++i)
    if (axes[i].value("name", std::string()) == m) {
      if (found != axes.size()) ng::refuse("statistic.pca: members " + m + " names more than one axis of '" + statistic + "'; give its position: " + listing);
      found = i;
    }
  if (found == axes.size()) ng::refuse("statistic.pca: '" + statistic + "' has no axis named " + m + ": its axes are " + listing);
  return found;
}

ng::Array table_array(const Eigen::MatrixXd& M) {
  ng::Array a;
  a.rows = M.rows(); a.cols = M.cols();
  a.data.resize(static_cast<std::size_t>(M.size()));
  for (Eigen::Index i = 0; i < M.rows(); ++i) for (Eigen::Index j = 0; j < M.cols(); ++j) a.data[static_cast<std::size_t>(i * M.cols() + j)] = M(i, j);
  return a;
}
Json axis_copy(const Json& ax, const std::string& name, std::int64_t size) {   // an input axis under a new name, its entries kept (values, a selection, ...)
  Json out = Json{{"name", name}, {"size", size}};
  for (const char* key : {"values", "edges", "scale", "unit", "selection_id", "bands", "members"}) if (ax.contains(key)) out[key] = ax[key];
  return out;
}

ng::NodeSpec statistic_pca() {
  ng::NodeSpec n;
  n.name = "statistic.pca";
  n.kind = ng::Kind::array;
  n.inputs = {{"x", "a stored statistic of real values with two axes: the members (the observations) and the features", false}};
  n.outputs = {{"components", "the principal components: [component x feature], unit rows", false},
               {"scores", "the members' scores on the components, U S: [member x component]", false},
               {"variance", "per component its singular value, explained variance and the ratio over the total: [component x 3]", false},
               {"centring", "the mean removed from each feature and the divisor applied after it: [2 x feature]", false}};
  n.parameters = {{"members", Json(), "the axis whose entries are the members (the observations): its name (when unique) or its position from 0; no default: the caller names it", "-", {}, false},
                  {"scale", Json("none"), "none = the features keep their units (the covariance's components) | unit-variance = each divided by its standard deviation (the correlation's)", "-", {"none", "unit-variance"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "a decomposition across the members of a table: it needs every member, and is not defined per timestep; computed in double");
  n.matches = "scikit-learn's PCA (svd_solver full); nxr::pca";
  n.description = "the principal components across the members of a table of measurements, by the thin SVD of the centred table";
  n.plan = [](const ng::Types& in, const Json& p) {
    const auto x = in.find("x");
    if (x == in.end() || x->second.cls != ng::PortClass::constant) ng::refuse("statistic.pca: x is a stored statistic");
    if (!p.contains("members") || !p["members"].is_string() || p["members"].get<std::string>().empty()) ng::refuse("statistic.pca: name the axis whose entries are the members (members: an axis name or position)");
    const std::string s = p.value("scale", std::string("none"));
    if (s != "none" && s != "unit-variance") ng::refuse("statistic.pca: scale is none or unit-variance (got '" + s + "')");
    ng::Planned out;
    for (const char* o : {"components", "scores", "variance", "centring"}) out.out[o] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    const auto it = c.find("x");
    if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse("statistic.pca: the statistic's values were not handed over");
    ng::Arrays a{{"x", it->second}};
    a["x"].meta["members"] = p["members"];
    a["x"].meta["scale"] = p.value("scale", std::string("none"));
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Array& X = std::any_cast<const ng::Arrays&>(prepared).at("x");
    const std::string name = X.meta.value("name", std::string());
    if (X.meta.value("value_type", std::string("scalar")) != "scalar")
      ng::refuse("statistic.pca: '" + name + "' holds " + X.meta.value("value_type", std::string()) + " values; only real (scalar) values are decomposed: decompose its real and imaginary parts, or a derived real quantity, as separate tables");
    const Json axes = X.meta.value("axes", Json::array());
    const Json shape = X.meta.value("shape", Json::array());
    if (!axes.is_array() || axes.size() != shape.size()) ng::refuse("statistic.pca: '" + name + "' has " + std::to_string(shape.size()) + " array dimensions but " + std::to_string(axes.is_array() ? axes.size() : 0) + " axes");
    if (axes.size() != 2) ng::refuse("statistic.pca: '" + name + "' has " + std::to_string(axes.size()) + " axes: a table is the members and ONE feature axis (reduce the others first)");
    const std::size_t a = members_axis(axes, X.meta["members"].get<std::string>(), name), f = 1 - a;
    const std::int64_t n_members = shape[a].get<std::int64_t>(), p = shape[f].get<std::int64_t>();
    Eigen::MatrixXd T(n_members, p);                                       // [member x feature]; the stored values are C order over the axes
    for (std::int64_t i = 0; i < n_members; ++i)
      for (std::int64_t j = 0; j < p; ++j) T(i, j) = X.data[static_cast<std::size_t>(a == 0 ? i * p + j : j * n_members + i)];
    const bool unit_variance = X.meta["scale"].get<std::string>() == "unit-variance";
    nxr::pca::PcaOptions po;
    po.scale = unit_variance ? nxr::pca::Scale::UnitVariance : nxr::pca::Scale::None;
    nxr::pca::PcaResult r;
    try { r = nxr::pca::pca(T, po); } catch (const std::invalid_argument& e) { ng::refuse("pca of '" + name + "': " + e.what()); }
    const std::int64_t k = r.rank;
    Json comp_values = Json::array();                                      // the component axis is its numbers, 1 .. k (the order of decreasing variance)
    for (std::int64_t c = 1; c <= k; ++c) comp_values.push_back(c);
    const Json comp_axis = Json{{"name", "component"}, {"size", k}, {"values", comp_values}};
    const Json feature_axis = axis_copy(axes[f], "feature", p), member_axis = axis_copy(axes[a], "member", n_members);
    const std::string precision = X.meta.value("stored_precision", std::string("float64"));
    const Json options = Json{{"scale", unit_variance ? "unit-variance" : "none"}, {"input", Json{{"statistic", name}, {"statistic_id", X.meta.value("id", std::string())}, {"members_axis", static_cast<std::int64_t>(a)},
                                                                                              {"axes", axes}, {"shape", shape}}},
                              {"sign", "the entry of largest absolute loading of each component is positive"}, {"rank", k}, {"members", n_members}, {"features", p},
                              {"feature_axis_name", axes[f].value("name", std::string())}, {"member_axis_name", axes[a].value("name", std::string())}};
    const Json producer = Json{{"stage", "pca"}, {"layout", "components [component x feature]: unit rows; scores [member x component] = U S; variance [component x (singular value, explained variance S^2/(n-1), ratio)]; "
                                                          "centring [(mean, divisor) x feature]; reconstruction at k: scores[:, :k] components[:k] x divisor + mean"}};
    const std::string unit = X.meta.contains("unit") && X.meta["unit"].is_string() ? X.meta["unit"].get<std::string>() : std::string();
    auto describe = [&](ng::Array& arr, const char* measure, const char* what, Json axes_out, const std::string& precision_out, const std::string& unit_out) {
      Json d = Json{{"measure", measure}, {"method", "pca"}, {"label", "pca"}, {"precision", precision_out}, {"axes", std::move(axes_out)}, {"n_members", n_members}, {"options", options}, {"producer", producer},
                    {"description", std::string(what) + " of " + name + " across its " + std::to_string(n_members) + " members (" + std::to_string(p) + " features, scale " + (unit_variance ? "unit-variance" : "none") + ")"}};
      if (!unit_out.empty()) d["unit"] = unit_out;
      arr.meta["describe"] = std::move(d);
    };
    ng::Array comp = table_array(r.components), scores = table_array(r.scores), cen;
    describe(comp, "principal_components", "the principal components", Json::array({comp_axis, feature_axis}), precision, std::string());
    describe(scores, "principal_scores", "the scores on the principal components", Json::array({member_axis, comp_axis}), precision, unit_variance ? std::string() : unit);
    Eigen::MatrixXd v(k, 3);
    v.col(0) = r.singularValues; v.col(1) = r.explainedVariance; v.col(2) = r.explainedVarianceRatio;
    ng::Array var = table_array(v);
    describe(var, "explained_variance", "the variance explained by each principal component",
             Json::array({comp_axis, Json{{"name", "quantity"}, {"size", 3}, {"values", Json::array({Json{{"name", "singular_value"}}, Json{{"name", "explained_variance"}}, Json{{"name", "explained_variance_ratio"}}})}}}),
             "float64", std::string());
    Eigen::MatrixXd c(2, p);
    c.row(0) = r.mean.transpose(); c.row(1) = r.scale.transpose();
    cen = table_array(c);
    describe(cen, "pca_centring", "the centring of the table",
             Json::array({Json{{"name", "quantity"}, {"size", 2}, {"values", Json::array({Json{{"name", "mean"}}, Json{{"name", "divisor"}}})}}, feature_axis}), precision, unit);
    return ng::Arrays{{"components", std::move(comp)}, {"scores", std::move(scores)}, {"variance", std::move(var)}, {"centring", std::move(cen)}};
  };
  return n;
}

}  // namespace

void register_pca_nodes() {
  ng::Registry& r = ng::registry();
  ng::NodeSpec n = statistic_pca();
  if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
