// nxr-pipelines — THE IDENTIFICATION NODES (brain fingerprinting's cohort statistics; spec docs/specs/2026-11-16-meg-fingerprinting.md, 11.4 steps 20-25), pure nodes of nxr::graph::registry().
// The maths is nxr::stats (compute/include/nxr/stats/identification.h, and pca.h for the reconstruction); these nodes check the tables they are handed, call it and DESCRIBE the statistics they
// made; the definition (`fingerprint.yaml`) says where they are stored. The members' stored features reach them already read and stacked by the runner (a group run's per-subject statistic
// inputs, [members x values], one shape and axes in every member): the nodes read no store.
//
//   stats.featureTable         target, database (stacked)                   ->  table  [2n x p]: the n target vectors, then the n database vectors (flatten | upper-triangle)
//   stats.identification       table                                        ->  correlation [n x n], self [n x 2], accuracy [3], edgewise [p x 3]
//   stats.identificationCurve  components, scores, centring (stats.pca)     ->  curve [steps x 5], reconstructed_self [n x 2], reconstructed_edgewise [p x 3]
//
// MEMORY: the stacked inputs and the table, 2n x p doubles each (158 members x 20 468 features: ~52 MB a copy), the correlation n x n; the curve reconstructs one 2n x p table at a time.
// Every statistic is float64 (a summary across the members, whatever the features' precision). Every description carries the method's citation (producer.references).
//
// What it cannot show: why a person's two measurements are alike (anatomy, the head's position, the kernel, recurring artefacts and the day's noise are individual too); accuracy falls as
// the cohort grows and is not comparable between cohorts; the curve and its best k are fitted on the data they score; the edgewise statistics are on features z-scored within each vector.
#include "nxr/pipelines/identification_steps.h"

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/stats/identification.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

const Json& references() {
  static const Json r = Json::array({
      "da Silva Castanheira J, Orozco Perez HD, Misic B, Baillet S (2021) Brief segments of neurophysiological activity enable individual differentiation. Nat Commun 12:5713, doi:10.1038/s41467-021-25895-8",
      "software: megFingerprinting v1.0.0 (06fdba8), github.com/neurohazardous/megFingerprinting; written from the paper and the notebooks' documented behaviour, its code run only to make the goldens",
      "the identification statistics: Finn ES et al. (2015) Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. Nat Neurosci 18:1664",
      "Amico E, Goni J (2018) The quest for identifiability in human functional connectomes. Sci Rep 8:8254"});
  return r;
}

ng::PortType statistic_out() {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = "statistic";
  return t;
}
void need_constant(const ng::Types& in, const char* port, const std::string& node) {
  const auto x = in.find(port);
  if (x == in.end() || x->second.cls != ng::PortClass::constant) ng::refuse(node + ": " + port + " is a stored statistic or a table made before it");
}
const ng::Array& given(const ng::Arrays& c, const char* port, const std::string& node) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(node + ": the values of " + port + " were not handed over");
  return it->second;
}
Eigen::MatrixXd matrix(const ng::Array& a) {
  Eigen::MatrixXd M(a.rows, a.cols);
  for (std::int64_t i = 0; i < a.rows; ++i) for (std::int64_t j = 0; j < a.cols; ++j) M(i, j) = a.data[static_cast<std::size_t>(i * a.cols + j)];
  return M;
}
ng::Array array_of(const Eigen::MatrixXd& M) {
  ng::Array a;
  a.rows = M.rows(); a.cols = M.cols();
  a.data.resize(static_cast<std::size_t>(M.size()));
  for (Eigen::Index i = 0; i < M.rows(); ++i) for (Eigen::Index j = 0; j < M.cols(); ++j) a.data[static_cast<std::size_t>(i * M.cols() + j)] = M(i, j);
  return a;
}
Json names_axis(const std::string& axis, const std::vector<std::string>& names) {
  Json v = Json::array();
  for (const auto& n : names) v.push_back(Json{{"name", n}});
  return Json{{"name", axis}, {"size", static_cast<std::int64_t>(names.size())}, {"values", v}};
}
Json quantity_axis(const std::vector<std::string>& names) { return names_axis("quantity", names); }
// the feature axis: each feature's place in the stored feature, its indices along the stored axes ([region, bin] when flattened; [row, column, slice] for an upper triangle)
Json feature_axis(std::int64_t p, const Json& from, const std::vector<std::int64_t>& shape = {}, bool upper = false) {
  Json a = Json{{"name", "feature"}, {"size", p}};
  Json v = Json::array();
  if (!shape.empty() && !upper) {
    std::vector<std::int64_t> at(shape.size(), 0);
    for (std::int64_t k = 0; k < p; ++k) {
      v.push_back(at);
      for (std::size_t d = shape.size(); d-- > 0;) { if (++at[d] < shape[d]) break; at[d] = 0; }
    }
  } else if (!shape.empty()) {
    const std::int64_t r = shape[0], slices = shape.size() == 3 ? shape[2] : 1;
    for (std::int64_t s = 0; s < slices; ++s)
      for (std::int64_t i = 0; i < r; ++i)
        for (std::int64_t j = i; j < r; ++j) v.push_back(shape.size() == 3 ? Json::array({i, j, s}) : Json::array({i, j}));
  } else {
    for (std::int64_t k = 0; k < p; ++k) v.push_back(Json::array({k}));
  }
  a["values"] = std::move(v);
  if (from.is_object()) a["features"] = from;   // what a feature is: the stored statistic's axes and the layout of the vector
  return a;
}
nxr::stats::SelfReference self_reference(const Json& p, const std::string& node) {
  const std::string s = p.value("self_identifiability", std::string("all"));
  if (s == "all") return nxr::stats::SelfReference::All;
  if (s == "others") return nxr::stats::SelfReference::Others;
  ng::refuse(node + ": self_identifiability is all (the release's notebooks: each correlation row z-scored over all its entries) or others (the paper's formula), got '" + s + "'");
}
void describe(ng::Array& a, const char* measure, const std::string& what, Json axes, std::int64_t members, const Json& options, const char* method = "fingerprint") {
  a.meta["describe"] = Json{{"measure", measure}, {"method", method}, {"label", method}, {"precision", "float64"}, {"axes", std::move(axes)}, {"n_members", members}, {"options", options},
                            {"producer", Json{{"stage", "fingerprint"}, {"references", references()}}}, {"description", what}};
}

// ── stats.featureTable ──
ng::NodeSpec feature_table() {
  ng::NodeSpec n;
  n.name = "stats.featureTable";
  n.kind = ng::Kind::array;
  n.inputs = {{"target", "each member's target (validation) feature, its values a row: [members x values], the members' shape and axes in meta", false},
              {"database", "each member's database (training) feature, the same members in the same order and the same shape", false}};
  n.outputs = {{"table", "the feature vectors, the n target rows then the n database rows: [2n x features]", false}};
  n.parameters = {{"vector", Json("flatten"), "flatten = the values as stored, row-major (a region x bin spectrum) | upper-triangle = per trailing slice (a band) the upper triangle WITH the diagonal of the leading square (a connectome), slices concatenated", "-", {"flatten", "upper-triangle"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "a table across the members of a cohort: it needs every member and is not defined per timestep; its values are the stored ones, laid out");
  n.matches = "the release's feature vectors (megFingerprinting 4_PSD_ and 4_AEC_fingerprinting_*, cell 4); nxr::stats::featureTable";
  n.description = "the members' feature vectors from their stored features, the target's then the database's";
  n.plan = [](const ng::Types& in, const Json& p) {
    need_constant(in, "target", "stats.featureTable");
    need_constant(in, "database", "stats.featureTable");
    const std::string v = p.value("vector", std::string("flatten"));
    if (v != "flatten" && v != "upper-triangle") ng::refuse("stats.featureTable: vector is flatten or upper-triangle (got '" + v + "')");
    ng::Planned out;
    out.out["table"] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ng::Arrays a{{"target", given(c, "target", "stats.featureTable")}, {"database", given(c, "database", "stats.featureTable")}};
    a["target"].meta["vector"] = p.value("vector", std::string("flatten"));
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Arrays& in = std::any_cast<const ng::Arrays&>(prepared);
    const ng::Array &T = in.at("target"), &D = in.at("database");
    const std::string vector = T.meta["vector"].get<std::string>();
    const std::string tn = T.meta.value("name", std::string("target")), dn = D.meta.value("name", std::string("database"));
    for (const auto* x : {&T, &D})
      if (x->meta.value("value_type", std::string("scalar")) != "scalar")
        ng::refuse("stats.featureTable: '" + x->meta.value("name", std::string()) + "' holds " + x->meta.value("value_type", std::string()) + " values; a feature vector is of real values");
    if (T.meta.value("subjects", Json()) != D.meta.value("subjects", Json())) ng::refuse("stats.featureTable: the target and the database are not of the same members in the same order");
    // the type check of the two features: one shape, and the same axes (names, sizes, entries; the ids of what they lie on differ between packages and are not compared), and one unit
    const Json ts = T.meta.value("member_shape", Json::array()), ds = D.meta.value("member_shape", Json::array());
    if (ts != ds) ng::refuse("stats.featureTable: '" + tn + "' has the shape " + ts.dump() + " in each member, '" + dn + "' " + ds.dump() + ": the two measurements are compared feature by feature");
    if (T.meta.value("axes_compared", Json()) != D.meta.value("axes_compared", Json()))
      ng::refuse("stats.featureTable: '" + tn + "' and '" + dn + "' lie on different axes (" + T.meta.value("axes_compared", Json()).dump() + " and " + D.meta.value("axes_compared", Json()).dump() + ")");
    if (T.meta.value("unit", Json()) != D.meta.value("unit", Json()))
      ng::refuse("stats.featureTable: '" + tn + "' is in " + T.meta.value("unit", Json()).dump() + ", '" + dn + "' in " + D.meta.value("unit", Json()).dump());
    std::vector<std::int64_t> shape;
    for (const auto& s : ts) shape.push_back(s.get<std::int64_t>());
    const auto layout = vector == "upper-triangle" ? nxr::stats::FeatureLayout::UpperTriangle : nxr::stats::FeatureLayout::Flatten;
    Eigen::MatrixXd FT, FD;
    try { FT = nxr::stats::featureTable(matrix(T), shape, layout); FD = nxr::stats::featureTable(matrix(D), shape, layout); }
    catch (const std::invalid_argument& e) { ng::refuse(std::string("stats.featureTable: ") + e.what()); }
    const std::int64_t n_members = FT.rows(), p = FT.cols();
    Eigen::MatrixXd X(2 * n_members, p);
    X.topRows(n_members) = FT; X.bottomRows(n_members) = FD;
    ng::Array out = array_of(X);
    std::vector<std::string> subjects;
    for (const auto& s : T.meta.value("subjects", Json::array())) subjects.push_back(s.get<std::string>());
    Json rows = Json::array();
    for (const char* which : {"target", "database"}) for (const auto& s : subjects) rows.push_back(Json{{"name", s}, {"measurement", which}});
    const Json features = Json{{"vector", vector}, {"of_axes", T.meta.value("axes", Json::array())}, {"member_shape", ts}};
    out.meta["name"] = tn + " ; " + dn;
    out.meta["value_type"] = "scalar";
    out.meta["shape"] = Json::array({2 * n_members, p});
    out.meta["axes"] = Json::array({Json{{"name", "measurement"}, {"size", 2 * n_members}, {"values", rows}}, feature_axis(p, features, shape, vector == "upper-triangle")});
    out.meta["stored_precision"] = T.meta.value("stored_precision", std::string("float64"));
    if (T.meta.contains("unit")) out.meta["unit"] = T.meta["unit"];
    out.meta["subjects"] = subjects;
    out.meta["n_target"] = n_members;
    out.meta["inputs"] = Json{{"target", tn}, {"database", dn}, {"vector", vector}};
    return ng::Arrays{{"table", std::move(out)}};
  };
  return n;
}

// the members' names and the n of a stacked table [target; database]
std::vector<std::string> members_of(const ng::Array& table) {
  std::vector<std::string> s;
  for (const auto& x : table.meta.value("subjects", Json::array())) s.push_back(x.get<std::string>());
  return s;
}
void put_identification(const nxr::stats::Identification& id, const std::vector<std::string>& members, std::int64_t p, const Json& feature_ax, const Json& options, const std::string& of,
                        ng::Array* correlation, ng::Array& self, ng::Array* accuracy, ng::Array& edgewise, const std::string& prefix) {
  const std::int64_t n = static_cast<std::int64_t>(members.size());
  if (correlation) {
    *correlation = array_of(id.correlation);
    Json t = names_axis("target_member", members), d = names_axis("database_member", members);
    describe(*correlation, "identification_correlation", prefix + "Pearson's r between each member's target and every member's database feature vector (" + of + ")", Json::array({t, d}), n, options);
  }
  Eigen::MatrixXd s(n, 2);
  s.col(0) = id.selfCorrelation; s.col(1) = id.selfIdentifiability;
  self = array_of(s);
  describe(self, "self_identifiability", prefix + "each member's self-correlation r_ii and self-identifiability D_self (" + of + ")", Json::array({names_axis("member", members), quantity_axis({"self_correlation", "self_identifiability"})}), n, options);
  if (accuracy) {
    Eigen::MatrixXd a(3, 1);
    a(0, 0) = id.accuracy.accuracyByRow; a(1, 0) = id.accuracy.accuracyByColumn; a(2, 0) = id.accuracy.chance;
    *accuracy = array_of(a);
    describe(*accuracy, "identification_accuracy", "the share of members identified, target from database (argmax per row) and database from target (argmax per column), and chance (1/n) (" + of + ")",
             Json::array({quantity_axis({"target_from_database", "database_from_target", "chance"})}), n, options);
  }
  Eigen::MatrixXd e(p, 3);
  e.col(0) = id.icc; e.col(1) = id.differentialPower; e.col(2) = id.groupConsistency;
  edgewise = array_of(e);
  describe(edgewise, "edgewise_reliability", prefix + "per feature, on the z-scored feature vectors: ICC(1,1) (k = 2), Finn's differential power and the group consistency (" + of + ")",
           Json::array({feature_ax, quantity_axis({"icc", "differential_power", "group_consistency"})}), n, options);
}

// ── stats.identification ──
ng::NodeSpec identification() {
  ng::NodeSpec n;
  n.name = "stats.identification";
  n.kind = ng::Kind::array;
  n.inputs = {{"table", "the feature vectors, the n target rows then the n database rows, row i and row n + i the same member (stats.featureTable)", false}};
  n.outputs = {{"correlation", "Pearson's r, target member x database member: [n x n]", false},
               {"self", "per member its self-correlation r_ii and D_self: [n x 2]", false},
               {"accuracy", "target from database, database from target, chance: [3]", false},
               {"edgewise", "per feature ICC(1,1), differential power, group consistency on the z-scored vectors: [features x 3]", false}};
  n.parameters = {{"self_identifiability", Json("all"), "all = each correlation row z-scored over all its entries, the member included (the release's notebooks) | others = the paper's formula, over the other members", "-", {"all", "others"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64,
                             "statistics across the members of a cohort: they need every member and are not defined per timestep; computed and stored in double");
  n.matches = "the release's notebooks (megFingerprinting 4_PSD_fingerprinting_within_session, cells 6, 13, 16, 20); nxr::stats::identification";
  n.description = "the identification of a cohort's members from two feature vectors each: correlation, accuracy both ways, self-identifiability and the edgewise reliability";
  n.plan = [](const ng::Types& in, const Json& p) {
    need_constant(in, "table", "stats.identification");
    self_reference(p, "stats.identification");
    ng::Planned out;
    for (const char* o : {"correlation", "self", "accuracy", "edgewise"}) out.out[o] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ng::Arrays a{{"table", given(c, "table", "stats.identification")}};
    a["table"].meta["self_identifiability"] = p.value("self_identifiability", std::string("all"));
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Array& X = std::any_cast<const ng::Arrays&>(prepared).at("table");
    const std::int64_t n_t = X.meta.value("n_target", std::int64_t(0));
    if (n_t < 2 || X.rows != 2 * n_t) ng::refuse("stats.identification: the table is the target's rows then the database's, at least two members each");
    const Eigen::MatrixXd M = matrix(X);
    const auto ref = self_reference(X.meta, "stats.identification");
    nxr::stats::Identification id;
    try { id = nxr::stats::identification(M.topRows(n_t), M.bottomRows(n_t), ref); }
    catch (const std::invalid_argument& e) { ng::refuse(std::string("stats.identification: ") + e.what()); }
    const Json options = Json{{"self_identifiability", X.meta["self_identifiability"]}, {"zscore", "each feature vector across its features, ddof 0"}, {"correlation", "pearson"},
                              {"inputs", X.meta.value("inputs", Json::object())}, {"members", X.meta.value("subjects", Json::array())}, {"features", X.cols}};
    const Json fa = X.meta["axes"][1];
    ng::Array c, s, a, e;
    put_identification(id, members_of(X), X.cols, fa, options, X.meta.value("name", std::string()), &c, s, &a, e, "");
    return ng::Arrays{{"correlation", std::move(c)}, {"self", std::move(s)}, {"accuracy", std::move(a)}, {"edgewise", std::move(e)}};
  };
  return n;
}

// ── stats.identificationCurve ──
ng::NodeSpec identification_curve() {
  ng::NodeSpec n;
  n.name = "stats.identificationCurve";
  n.kind = ng::Kind::array;
  n.inputs = {{"components", "the principal components of the stacked table [target; database] (stats.pca): [component x feature]", false},
              {"scores", "the stacked table's scores on them: [2n x component]", false},
              {"centring", "the mean removed from each feature and the divisor: [2 x feature]", false}};
  n.outputs = {{"curve", "per step: the components asked and used, accuracy both ways and the mean D_self of the reconstruction: [steps x 5]", false},
               {"reconstructed_self", "at the best k: per member r_ii and D_self: [n x 2]", false},
               {"reconstructed_edgewise", "at the best k: per feature ICC(1,1), differential power, group consistency: [features x 3]", false}};
  n.parameters = {{"steps", Json("notebook"), "the numbers of components: notebook = 2, 5, then 10, 20, ... below 2n (the release's) | a list of integers, each from 1 to min(2n, features)", "components", {}, false},
                  {"self_identifiability", Json("all"), "all = each correlation row z-scored over all its entries (the release's) | others = the paper's formula, over the other members", "-", {"all", "others"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64,
                             "statistics across the members of a cohort, from its PCA: not defined per timestep; computed and stored in double");
  n.matches = "the release's notebooks (megFingerprinting 4_PSD_fingerprinting_within_session, cells 8, 10, 13, 16-20) with an exact SVD in place of scikit-learn's randomised solver; nxr::stats::identificationCurve";
  n.description = "the identification of a cohort's PCA reconstructions at increasing numbers of components, and the best one's edgewise reliability";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* port : {"components", "scores", "centring"}) need_constant(in, port, "stats.identificationCurve");
    self_reference(p, "stats.identificationCurve");
    const Json s = p.value("steps", Json("notebook"));
    if (!(s.is_string() && s.get<std::string>() == "notebook") && !s.is_array()) ng::refuse("stats.identificationCurve: steps is notebook or a list of integers (got " + s.dump() + ")");
    if (s.is_array()) {
      if (s.empty()) ng::refuse("stats.identificationCurve: steps lists at least one number of components");
      for (const auto& k : s) if (!k.is_number_integer() || k.get<std::int64_t>() < 1) ng::refuse("stats.identificationCurve: each step is a whole number of components, at least 1 (got " + k.dump() + ")");
    }
    ng::Planned out;
    for (const char* o : {"curve", "reconstructed_self", "reconstructed_edgewise"}) out.out[o] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ng::Arrays a;
    for (const char* port : {"components", "scores", "centring"}) a[port] = given(c, port, "stats.identificationCurve");
    a["scores"].meta["steps"] = p.value("steps", Json("notebook"));
    a["scores"].meta["self_identifiability"] = p.value("self_identifiability", std::string("all"));
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const ng::Arrays& in = std::any_cast<const ng::Arrays&>(prepared);
    const ng::Array &C = in.at("components"), &S = in.at("scores"), &Z = in.at("centring");
    if (Z.rows != 2 || Z.cols != C.cols || S.cols != C.rows) ng::refuse("stats.identificationCurve: the components, scores and centring are not of one decomposition");
    const Eigen::MatrixXd centring = matrix(Z);
    const Eigen::VectorXd mean = centring.row(0).transpose(), scale = centring.row(1).transpose();
    const Json& sp = S.meta["steps"];
    const std::vector<std::int64_t> steps = sp.is_string() ? nxr::stats::identificationSteps(S.rows) : sp.get<std::vector<std::int64_t>>();
    if (steps.empty()) ng::refuse("stats.identificationCurve: the notebook's steps (2, 5, 10, ... below 2n) leave none below 2n = " + std::to_string(S.rows) + "; list the steps");
    const auto ref = self_reference(S.meta, "stats.identificationCurve");
    nxr::stats::IdentificationCurve c;
    try { c = nxr::stats::identificationCurve(matrix(S), matrix(C), mean, scale, steps, ref); }
    catch (const std::invalid_argument& e) { ng::refuse(std::string("stats.identificationCurve: ") + e.what()); }
    // the members and the feature axis, as stats.pca carried them from the table
    const Json d_scores = S.meta.value("describe", Json::object()), d_comp = C.meta.value("describe", Json::object());
    std::vector<std::string> members;
    const std::int64_t n_members = S.rows / 2;
    if (d_scores.contains("axes") && d_scores["axes"][0].contains("values"))
      for (std::int64_t i = 0; i < n_members; ++i) members.push_back(d_scores["axes"][0]["values"][static_cast<std::size_t>(i)].value("name", std::to_string(i)));
    else for (std::int64_t i = 0; i < n_members; ++i) members.push_back(std::to_string(i));
    const Json fa = d_comp.contains("axes") ? d_comp["axes"][1] : feature_axis(C.cols, Json());
    const Json pca_options = d_comp.value("options", Json::object());
    const std::int64_t k_best = c.steps[static_cast<std::size_t>(c.best)];
    const Json options = Json{{"steps", c.steps}, {"used", c.used}, {"best_k", k_best}, {"best_rule", "the first step of the largest mean reconstructed self-identifiability"},
                              {"self_identifiability", S.meta["self_identifiability"]}, {"pca", "an exact thin SVD of the centred stacked table (nxr::stats::pca); the release's scikit-learn PCA(k) is randomised at this size"},
                              {"input", pca_options.value("input", Json::object())}, {"members", members}};
    Eigen::MatrixXd curve(static_cast<Eigen::Index>(steps.size()), 5);
    for (std::size_t s = 0; s < steps.size(); ++s)
    {
      const auto r = static_cast<Eigen::Index>(s);
      curve(r, 0) = static_cast<double>(c.steps[s]); curve(r, 1) = static_cast<double>(c.used[s]);
      curve(r, 2) = c.accuracyByRow[r]; curve(r, 3) = c.accuracyByColumn[r]; curve(r, 4) = c.meanSelfIdentifiability[r];
    }
    ng::Array cu = array_of(curve);
    Json step_values = Json::array();
    for (const auto k : c.steps) step_values.push_back(k);
    describe(cu, "identification_curve", "the identification of the PCA reconstructions of the stacked target and database at each number of components: accuracy both ways and the mean self-identifiability (best k " +
             std::to_string(k_best) + ")", Json::array({Json{{"name", "step"}, {"size", static_cast<std::int64_t>(steps.size())}, {"values", step_values}},
                                                          quantity_axis({"components", "components_used", "target_from_database", "database_from_target", "mean_self_identifiability"})}), n_members, options);
    ng::Array s, e;
    put_identification(c.atBest, members, C.cols, fa, options, "the reconstruction at " + std::to_string(k_best) + " components", nullptr, s, nullptr, e, "at the best k: ");
    return ng::Arrays{{"curve", std::move(cu)}, {"reconstructed_self", std::move(s)}, {"reconstructed_edgewise", std::move(e)}};
  };
  return n;
}

}  // namespace

void register_identification_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {feature_table(), identification(), identification_curve()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
