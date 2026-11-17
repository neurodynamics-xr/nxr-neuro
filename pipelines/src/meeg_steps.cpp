// nxr-pipelines — THE M/EEG NODES: the source-imaging steps as pure nodes of nxr::graph::registry() (grouped names `meeg.*`), each the wrapper of the maths of
// meeg/ (nxr::meeg): the same function a test calls directly. A node reads what it is handed (the streamed recording, stored arrays as constants, the facts of
// its ports: a recording's channel types and quality) and describes what it made (`Array::meta["describe"]`); the definition says where it is stored.
//
//   meeg.noise_covariance  accumulate  x [channels x samples]  ->  channels (the good channels of the imaging types, indices), covariance [M x M] (a statistic)
//   meeg.spherical_head_model  array   channels (sensor geometry) + sources + meg/eeg surfaces  ->  gain (an operator), grid (the source positions)
//   meeg.minimum_norm      array       forward (a head model) + covariance (a statistic on a pick)  ->  kernel, channels, covariance, whitener
#include "nxr/pipelines/meeg_steps.h"

#include <algorithm>
#include <optional>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/meeg/covariance.h"
#include "nxr/meeg/inverse.h"
#include "nxr/meeg/forward.h"
#include "nxr/meeg/sphere.h"
#include "nxr/bst/inverse.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

using RowMajorF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

Eigen::MatrixXd own_rows(const ng::Block& b, const std::vector<std::int32_t>& rows) {
  const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
  Eigen::MatrixXd x(static_cast<Eigen::Index>(rows.size()), b.own());
  for (std::size_t i = 0; i < rows.size(); ++i) x.row(static_cast<Eigen::Index>(i)) = all.row(rows[i]).segment(b.left, b.own()).cast<double>();
  return x;
}
const ng::PortType& streamed(const ng::Types& in, const char* port, const char* who) {
  const auto it = in.find(port);
  if (it == in.end()) ng::refuse(std::string(who) + ": port '" + port + "' is not given");
  const ng::PortType& t = it->second;
  if (!t.streamed() || t.axis != ng::Axis::samples) ng::refuse(std::string(who) + ": '" + port + "' is a recording's samples (a streamed field)");
  if (!t.sfreq || !(*t.sfreq > 0)) ng::refuse(std::string(who) + ": '" + port + "' has no sampling rate");
  return t;
}
ng::PortType constant_port(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}
// the channels of a recording a node keeps: good ones (its flags, else the montage's quality) of the types asked; from its port's facts
std::vector<std::int32_t> good_channels(const ng::PortType& x, const Json& types, const char* who) {
  if (!x.facts.contains("channels")) ng::refuse(std::string(who) + ": the recording's channels (types, quality) were not handed over");
  const Json& c = x.facts["channels"];
  std::set<std::string> want;
  for (const auto& t : types) want.insert(t.get<std::string>());
  std::vector<std::int32_t> out;
  const auto& ty = c["types"];
  const auto& q = c["quality"];
  for (std::size_t i = 0; i < ty.size(); ++i)
    if ((want.empty() || want.count(ty[i].get<std::string>())) && (i >= q.size() || q[i].get<std::string>() == "good")) out.push_back(static_cast<std::int32_t>(i));
  if (out.empty()) ng::refuse(std::string(who) + ": no good channel of the types asked");
  return out;
}

// ── meeg.noise_covariance ───────────────────────────────────────────────────────────────────────────────────────
// THE NOISE COVARIANCE IS A STATISTIC: the zero-lag second moment of the good channels of the imaging types over the span the caller states, [M x M]; the blocks are its
// members (each with its own mean removed when remove_mean), the divisor its degrees of freedom. The whitener derived from it is an operator (made by the inverse).
struct CovPrepared {
  std::vector<std::int32_t> channels;
  nxr::meeg::CovarianceOptions opt;
  std::int64_t block = 0;   // samples a block; 0: the span is one block
  double sfreq = 0;
  std::string divisor;
  Json types;
  std::string unit;
};
struct CovState {
  std::optional<nxr::meeg::CovarianceAccumulator> acc;
  std::int64_t in_block = 0, first = -1, samples = 0;
};

ng::NodeSpec noise_covariance() {
  ng::NodeSpec n;
  n.name = "meeg.noise_covariance";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed); its channels' types and quality are the port's facts", false}};
  n.outputs = {{"channels", "the channels covered, in order: the good ones of the types asked (an indices selection of the recording's channels)", false},
               {"covariance", "the covariance of those channels over the span, channels x channels (a statistic)", false}};
  n.parameters = {
      {"channel_types", Json::array({"MEG", "MEG MAG", "MEG GRAD", "EEG", "SEEG", "ECOG"}), "the channel types covered (only good channels of them)", "-", {}, false},
      {"block_seconds", Json(), "the length of a block (its own mean removed); null: the span is one block", "s", {}, true},
      {"remove_mean", Json(true), "remove each block's mean", "-", {}, false},
      {"divisor", Json("samples_minus_blocks"), "the divisor: samples_minus_blocks | samples | samples_minus_one", "-", {"samples_minus_blocks", "samples", "samples_minus_one"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "the blocks' sums are carried across chunks in double");
  n.matches = "MNE compute_raw_covariance (method empirical); Brainstorm's noise covariance (remove DC offset per file); meeg nxr::meeg::CovarianceAccumulator";
  n.description = "the noise covariance of a recording's good imaging channels over a span, as blocks with their own mean removed";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed(in, "x", "meeg.noise_covariance");
    good_channels(x, p.value("channel_types", Json::array()), "meeg.noise_covariance");
    ng::Planned out;
    out.out["channels"] = constant_port("selection");
    out.out["covariance"] = constant_port("statistic");
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json& p, ng::Shared&) {
    const ng::PortType& x = in.at("x");
    CovPrepared pr;
    pr.types = p.value("channel_types", Json::array());
    pr.channels = good_channels(x, pr.types, "meeg.noise_covariance");
    pr.sfreq = *x.sfreq;
    pr.unit = x.unit;
    pr.opt.removeMean = p.value("remove_mean", true);
    pr.divisor = p.value("divisor", std::string("samples_minus_blocks"));
    pr.opt.divisor = pr.divisor == "samples" ? nxr::meeg::CovarianceDivisor::Samples
                     : pr.divisor == "samples_minus_one" ? nxr::meeg::CovarianceDivisor::SamplesMinusOne : nxr::meeg::CovarianceDivisor::SamplesMinusBlocks;
    if (p.contains("block_seconds") && !p["block_seconds"].is_null())
      pr.block = std::max<std::int64_t>(2, static_cast<std::int64_t>(std::llround(p["block_seconds"].get<double>() * pr.sfreq)));
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(CovState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const CovPrepared& pr = std::any_cast<const CovPrepared&>(prepared);
    CovState& s = std::any_cast<CovState&>(state);
    const ng::Block& b = in.at("x");
    if (!s.acc) s.acc.emplace(static_cast<int>(pr.channels.size()), pr.opt);
    if (s.first < 0) s.first = at.first;
    const Eigen::MatrixXd x = own_rows(b, pr.channels);
    for (Eigen::Index j = 0; j < x.cols();) {                     // a block boundary splits the samples handed over
      Eigen::Index k = x.cols() - j;
      if (pr.block > 0) {
        if (s.in_block == pr.block) { s.acc->beginBlock(); s.in_block = 0; }
        k = std::min<Eigen::Index>(k, pr.block - s.in_block);
      }
      s.acc->add(Eigen::MatrixXd(x.middleCols(j, k)));
      s.in_block += k;
      j += k;
    }
    s.samples += x.cols();
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const CovPrepared& pr = std::any_cast<const CovPrepared&>(prepared);
    const CovState& s = std::any_cast<const CovState&>(state);
    if (!s.acc || s.samples < 2) ng::refuse("meeg.noise_covariance: the span holds fewer than 2 samples");
    const Eigen::MatrixXd C = s.acc->covariance();
    const std::int64_t M = C.rows(), N = s.acc->samples(), B = s.acc->blocks();
    const std::int64_t dof = !pr.opt.removeMean || pr.opt.divisor != nxr::meeg::CovarianceDivisor::SamplesMinusBlocks
                                 ? (pr.opt.divisor == nxr::meeg::CovarianceDivisor::Samples ? N : N - 1) : N - B;
    ng::Array ch;
    ch.rows = M; ch.cols = 1;
    for (const auto c : pr.channels) ch.data.push_back(c);
    ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the noise covariance covers, in order"}};
    ng::Array a;
    a.rows = M; a.cols = M;
    a.data.resize(static_cast<std::size_t>(M * M));
    for (std::int64_t i = 0; i < M; ++i) for (std::int64_t j = 0; j < M; ++j) a.data[static_cast<std::size_t>(i * M + j)] = C(i, j);
    const double t0 = static_cast<double>(s.first) / pr.sfreq, t1 = static_cast<double>(s.first + s.samples) / pr.sfreq;
    a.meta["describe"] = Json{
        {"measure", "covariance"}, {"method", "sample"}, {"label", "sample"}, {"precision", "float64"}, {"dof", dof},
        {"unit", pr.unit.empty() ? std::string("channel unit\xc2\xb2") : pr.unit + "\xc2\xb2"},
        {"axes", Json::array({Json{{"name", "channel"}, {"size", M}}, Json{{"name", "channel"}, {"size", M}}})},
        {"n_members", B},
        {"options", Json{{"remove_mean", pr.opt.removeMean}, {"divisor", pr.divisor}, {"block_samples", pr.block > 0 ? Json(pr.block) : Json()}, {"n_samples", N}, {"n_blocks", B},
                         {"channel_types", pr.types}, {"n_channels", M}}},
        {"producer", Json{{"stage", "noise covariance"}, {"n_samples", N}, {"n_blocks", B}, {"start_sample", s.first}, {"stop_sample", s.first + s.samples}}},
        {"description", "the noise covariance over [" + std::to_string(t0) + ", " + std::to_string(t1) + ") s (the recording's line from its first sample), on " + std::to_string(M) + " channels"}};
    return ng::Arrays{{"channels", std::move(ch)}, {"covariance", std::move(a)}};
  };
  return n;
}


// ── meeg.minimum_norm ───────────────────────────────────────────────────────────────────────────────────────────
// THE MINIMUM-NORM KERNEL (free orientation) from a head model's gain G [channels x 3N] and a noise covariance C on a pick of its channels: G restricted to the pick, the
// average reference of EEG, sEEG and ECoG applied to both (G := A G, C := A C A^T, per family, unless average_reference is false), then nxr::meeg::unconstrainedKernel.
// Made: the kernel [3N x M] on the pick, the pick, the covariance it was computed with (re-referenced) and its symmetric whitener [M x M] (a PCA whitener is not stored).
struct MinNormPrepared {
  Eigen::MatrixXd G, C;
  std::vector<std::int32_t> channels;
  std::vector<std::string> types;
  nxr::meeg::KernelOptions opt;
  bool averaged = false;
  std::string forward, covariance, conventions;
};

nxr::meeg::KernelOptions kernel_options(const Json& p) {
  using namespace nxr::meeg;
  const std::string conv = p.value("conventions", std::string("brainstorm"));
  if (conv != "brainstorm" && conv != "mne") ng::refuse("meeg.minimum_norm: conventions is brainstorm or mne");
  KernelOptions k = conv == "mne" ? mneKernelOptions() : KernelOptions{};
  auto given = [&](const char* key) { return p.contains(key) && !p[key].is_null(); };
  auto pick = [&](const char* key, std::initializer_list<const char*> allowed) {
    const std::string v = p[key].get<std::string>();
    for (const char* a : allowed) if (v == a) return v;
    ng::refuse(std::string("meeg.minimum_norm: ") + key + " is not one of its choices");
  };
  if (given("depth")) k.depth.enabled = p["depth"].get<bool>();
  if (given("depth_exponent")) k.depth.exponent = p["depth_exponent"].get<double>();
  if (given("depth_limit")) k.depth.limit = p["depth_limit"].get<double>();
  if (given("depth_norm")) k.depth.norm = pick("depth_norm", {"frobenius", "spectral"}) == "spectral" ? DepthNorm::Spectral : DepthNorm::Frobenius;
  if (given("depth_limit_on")) k.depth.limitOn = pick("depth_limit_on", {"weight", "norm"}) == "norm" ? DepthLimitOn::Norm : DepthLimitOn::Weight;
  if (given("depth_channels")) k.depth.channels = pick("depth_channels", {"all", "one_type"}) == "one_type" ? DepthChannels::OneType : DepthChannels::All;
  if (given("snr")) k.snr = p["snr"].get<double>();
  if (given("snr_normalisation")) k.snrNormalisation = pick("snr_normalisation", {"all_channels", "whitener_rank"}) == "whitener_rank" ? SnrNormalisation::WhitenerRank : SnrNormalisation::AllChannels;
  if (given("regularisation")) {
    const std::string r = pick("regularisation", {"ridge", "diagonal", "none"});
    k.whitener.regularisation = r == "ridge" ? Regularisation::Ridge : r == "diagonal" ? Regularisation::Diagonal : Regularisation::None;
  }
  if (given("ridge_fraction")) k.whitener.ridgeFraction = p["ridge_fraction"].get<double>();
  if (given("whitener_form")) k.whitener.form = pick("whitener_form", {"symmetric", "pca"}) == "pca" ? WhitenerForm::Pca : WhitenerForm::Symmetric;
  if (given("rank_tolerance")) k.whitener.rankTolerance = pick("rank_tolerance", {"single", "double"}) == "double" ? RankTolerance::DoublePrecision : RankTolerance::SinglePrecision;
  if (given("joint_types")) k.whitener.jointTypes = p["joint_types"].get<bool>();
  if (given("ridge_over_rank")) k.whitener.ridgeOverRank = p["ridge_over_rank"].get<bool>();
  if (given("detect_diagonal")) k.whitener.detectDiagonal = p["detect_diagonal"].get<bool>();
  return k;
}
Json kernel_options_record(const nxr::meeg::KernelOptions& o) {
  using namespace nxr::meeg;
  auto reg = [](Regularisation r) { return r == Regularisation::Ridge ? "ridge" : r == Regularisation::Diagonal ? "diagonal" : "none"; };
  return Json{{"whitener", Json{{"regularisation", reg(o.whitener.regularisation)}, {"ridge_fraction", o.whitener.ridgeFraction},
                                {"rank_tolerance", o.whitener.rankTolerance == RankTolerance::SinglePrecision ? "single" : "double"},
                                {"detect_diagonal", o.whitener.detectDiagonal}, {"joint_types", o.whitener.jointTypes}, {"ridge_over_rank", o.whitener.ridgeOverRank},
                                {"form", o.whitener.form == WhitenerForm::Symmetric ? "symmetric" : "pca"}}},
              {"depth", Json{{"enabled", o.depth.enabled}, {"exponent", o.depth.exponent}, {"limit", o.depth.limit},
                             {"norm", o.depth.norm == DepthNorm::Frobenius ? "frobenius" : "spectral"}, {"limit_on", o.depth.limitOn == DepthLimitOn::Weight ? "weight" : "norm"},
                             {"channels", o.depth.channels == DepthChannels::All ? "all" : "one type"}}},
              {"snr", o.snr}, {"snr_normalisation", o.snrNormalisation == SnrNormalisation::AllChannels ? "all channels" : "whitener rank"}};
}

ng::NodeSpec minimum_norm() {
  ng::NodeSpec n;
  n.name = "meeg.minimum_norm";
  n.kind = ng::Kind::array;
  n.inputs = {{"forward", "the head model: a stored operator, its gain channels x 3 sources (x y z)", false},
              {"covariance", "the noise covariance: a stored statistic over a pick of the head model's channels", false}};
  n.outputs = {{"kernel", "the inverse kernel, 3 rows a source x the pick's channels (an operator from the channels to the sources, vector3)", false},
               {"channels", "the channels the kernel maps, in column order (the covariance's pick)", false},
               {"covariance", "the covariance the kernel was computed with (re-referenced when the average reference applies), on the pick", false},
               {"whitener", "the symmetric whitener of that covariance, channels x channels (absent for a PCA whitener)", true}};
  n.parameters = {
      {"conventions", Json("brainstorm"), "the defaults every option below starts from: brainstorm | mne (joint MEG types, ridge over the rank, PCA whitener, depth 0.8 by the spectral norm)", "-", {"brainstorm", "mne"}, false},
      {"average_reference", Json(true), "apply the average reference of EEG, sEEG and ECoG to the gain and the covariance, per family", "-", {}, false},
      {"depth", Json(), "depth weighting on or off; null: the conventions'", "-", {}, true},
      {"depth_exponent", Json(), "the depth weighting's exponent; null: the conventions'", "-", {}, true},
      {"depth_limit", Json(), "the largest ratio of the depth weights; null: the conventions'", "-", {}, true},
      {"depth_norm", Json(), "frobenius | spectral; null: the conventions'", "-", {"frobenius", "spectral"}, true},
      {"depth_limit_on", Json(), "weight | norm; null: the conventions'", "-", {"weight", "norm"}, true},
      {"depth_channels", Json(), "all | one_type; null: the conventions'", "-", {"all", "one_type"}, true},
      {"snr", Json(), "the amplitude signal-to-noise ratio (lambda^2 = 1/snr^2); null: the conventions'", "-", {}, true},
      {"snr_normalisation", Json(), "all_channels | whitener_rank; null: the conventions'", "-", {"all_channels", "whitener_rank"}, true},
      {"regularisation", Json(), "the noise covariance's regularisation: ridge | diagonal | none; null: the conventions'", "-", {"ridge", "diagonal", "none"}, true},
      {"ridge_fraction", Json(), "the ridge's fraction of the mean eigenvalue; null: the conventions'", "fraction", {}, true},
      {"whitener_form", Json(), "symmetric | pca; null: the conventions'", "-", {"symmetric", "pca"}, true},
      {"rank_tolerance", Json(), "single | double; null: the conventions'", "-", {"single", "double"}, true},
      {"joint_types", Json(), "whiten the types of a modality jointly (MEG MAG with MEG GRAD); null: the conventions'", "-", {}, true},
      {"ridge_over_rank", Json(), "the ridge's mean eigenvalue over the rank, not the channel count; null: the conventions'", "-", {}, true},
      {"detect_diagonal", Json(), "whiten a diagonal covariance by its diagonal; null: the conventions'", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "solved in double; the kernel is stored as float32 like every operator");
  n.matches = "Brainstorm's minimum norm (bst_inverse_linear_2018, free orientation); MNE-Python make_inverse_operator + apply (MNE, loose 1); meeg nxr::meeg::unconstrainedKernel";
  n.description = "the free-orientation minimum-norm kernel from a head model and a noise covariance, with the whitener it used";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("forward") || in.at("forward").cls != ng::PortClass::operator_) ng::refuse("meeg.minimum_norm: forward is a stored operator (a head model)");
    if (!in.count("covariance")) ng::refuse("meeg.minimum_norm: covariance is a stored statistic");
    kernel_options(p);
    ng::Planned out;
    out.out["kernel"] = constant_port("operator");
    out.out["channels"] = constant_port("selection");
    out.out["covariance"] = constant_port("statistic");
    out.out["whitener"] = constant_port("operator");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    MinNormPrepared pr;
    const auto f = constants.find("forward"), c = constants.find("covariance");
    if (f == constants.end() || c == constants.end() || f->second.meta.value("placeholder", false) || c->second.meta.value("placeholder", false))
      ng::refuse("meeg.minimum_norm: the head model and the covariance were not handed over");
    const ng::Array& F = f->second;
    const ng::Array& S = c->second;
    pr.forward = F.meta.value("name", std::string());
    pr.covariance = S.meta.value("name", std::string());
    if (F.meta.value("to_manifold_id", std::string()) != S.meta.value("channels_manifold_id", std::string()))
      ng::refuse("meeg.minimum_norm: " + pr.covariance + " is on other channels than " + pr.forward);
    if (F.cols % 3 != 0) ng::refuse("meeg.minimum_norm: " + pr.forward + ": the gain is not channels x 3N");
    pr.channels = S.meta.value("channels", std::vector<std::int32_t>());
    pr.types = S.meta.value("channel_types", std::vector<std::string>());
    const std::int64_t M = static_cast<std::int64_t>(pr.channels.size());
    if (M == 0 || S.rows != M || S.cols != M) ng::refuse("meeg.minimum_norm: " + pr.covariance + ": its array does not fit its channels");
    pr.C = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(S.data.data(), M, M);
    pr.G.resize(M, F.cols);
    for (std::int64_t i = 0; i < M; ++i) {
      const std::int32_t ch = pr.channels[static_cast<std::size_t>(i)];
      if (ch < 0 || ch >= F.rows) ng::refuse("meeg.minimum_norm: a channel of " + pr.covariance + " is outside the gain's channels");
      for (std::int64_t j = 0; j < F.cols; ++j) pr.G(i, j) = F.data[static_cast<std::size_t>(ch * F.cols + j)];
    }
    pr.opt = kernel_options(p);
    pr.conventions = p.value("conventions", std::string("brainstorm"));
    if (p.value("average_reference", true)) {
      Eigen::MatrixXd A = Eigen::MatrixXd::Identity(M, M);
      for (const char* fam : {"EEG", "SEEG", "ECOG"}) {
        std::vector<int> idx;
        for (int i = 0; i < static_cast<int>(M); ++i) if (pr.types[static_cast<std::size_t>(i)] == fam) idx.push_back(i);
        if (idx.size() < 2) continue;
        for (int a : idx) for (int b : idx) A(a, b) -= 1.0 / static_cast<double>(idx.size());
        pr.averaged = true;
      }
      if (pr.averaged) { pr.G = A * pr.G; pr.C = A * pr.C * A.transpose(); }
    }
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const MinNormPrepared& pr = std::any_cast<const MinNormPrepared&>(prepared);
    const nxr::meeg::Kernel ker = nxr::meeg::unconstrainedKernel(pr.G, pr.C, pr.types, pr.opt);
    const std::int64_t M = static_cast<std::int64_t>(pr.channels.size()), R = ker.K.rows(), N = R / 3;
    const Json options = kernel_options_record(pr.opt);
    const auto& w = pr.opt.whitener;
    // the producer record in the importer's keys (nxr::bst::inverse_method_of reads it the same), plus what a computed kernel adds
    Json producer = Json{{"method", "MNE"}, {"method_comment", "MN"}, {"units", "A.m"}, {"n_channels", M}, {"n_sources", R}, {"n_components", 3},
                         {"source_ori", "free"}, {"inverse_method", "minnorm"}, {"inverse_measure", "amplitude"}, {"source_orient", Json::array({"free"})},
                         {"use_depth", pr.opt.depth.enabled}, {"noise_method", w.regularisation == nxr::meeg::Regularisation::Ridge ? "reg" : w.regularisation == nxr::meeg::Regularisation::Diagonal ? "diag" : "none"},
                         {"snr_method", "fixed"}, {"snr_fixed", pr.opt.snr}, {"whitener_rank", ker.whitenerRank}, {"lambda", ker.lambda},
                         {"computed", Json{{"options", options}, {"conventions", pr.conventions}, {"average_reference", pr.averaged}, {"forward", pr.forward}, {"noise_cov", pr.covariance}, {"notes", ker.notes}}}};
    if (pr.opt.depth.enabled) { producer["weight_exp"] = pr.opt.depth.exponent; producer["weight_limit"] = pr.opt.depth.limit; }
    if (w.regularisation == nxr::meeg::Regularisation::Ridge) producer["noise_reg"] = w.ridgeFraction;
    const std::optional<Json> inverse = nxr::bst::inverse_method_of(producer);   // the importer's shape, so a computed kernel is found as an imported one is
    ng::Array K;
    K.rows = R; K.cols = M;
    K.data.resize(static_cast<std::size_t>(R * M));
    for (std::int64_t i = 0; i < R; ++i) for (std::int64_t j = 0; j < M; ++j) K.data[static_cast<std::size_t>(i * M + j)] = ker.K(i, j);
    K.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "vector3"}, {"precision", "float32"}, {"params", inverse ? Json{{"inverse", *inverse}} : Json()}, {"producer", producer},
                              {"description", "sensor field to source field on the cortex (minimum norm, free orientation), " + std::to_string(N) + " sources"}};
    ng::Array ch;
    ch.rows = M; ch.cols = 1;
    for (const auto c : pr.channels) ch.data.push_back(c);
    ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the kernel maps, in column order"}};
    ng::Array C;
    C.rows = M; C.cols = M;
    C.data.resize(static_cast<std::size_t>(M * M));
    for (std::int64_t i = 0; i < M; ++i) for (std::int64_t j = 0; j < M; ++j) C.data[static_cast<std::size_t>(i * M + j)] = pr.C(i, j);
    C.meta["describe"] = Json{{"measure", "covariance"}, {"method", "empirical"}, {"precision", "float64"},
                              {"axes", Json::array({Json{{"name", "channel"}, {"size", M}}, Json{{"name", "channel"}, {"size", M}}})},
                              {"options", Json{{"noise_cov", pr.covariance}, {"average_reference", pr.averaged}}},
                              {"description", "the noise covariance the kernel was computed with, on its " + std::to_string(M) + " channels"}};
    ng::Arrays out{{"kernel", std::move(K)}, {"channels", std::move(ch)}, {"covariance", std::move(C)}};
    if (ker.W.rows() == ker.W.cols()) {
      ng::Array W;
      W.rows = M; W.cols = M;
      W.data.resize(static_cast<std::size_t>(M * M));
      for (std::int64_t i = 0; i < M; ++i) for (std::int64_t j = 0; j < M; ++j) W.data[static_cast<std::size_t>(i * M + j)] = ker.W(i, j);
      W.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"precision", "float64"}, {"producer", Json{{"rank", ker.whitenerRank}}},
                                {"description", "the whitener of the kernel's noise covariance, on its " + std::to_string(M) + " channels"}};
      out["whitener"] = std::move(W);
    }
    return out;
  };
  return n;
}


// ── meeg.spherical_head_model ───────────────────────────────────────────────────────────────────────────────────
// THE HEAD MODEL (the gain, sources to channels) in spheres: MEG by Sarvas in one sphere fitted to the MEG surface (single_sphere) or one sphere a sensor (overlapping_spheres,
// Brainstorm's default), EEG in three concentric shells fitted to the EEG surface (Berg-Scherg or the series). The sources are the vertices of the source surface (x y z
// moments); the sensors are the channels' coils with their weights when stored, else one point a channel. Channels of no computed type keep zero rows.
using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
Json sphere_record(const nxr::meeg::Sphere& s) { return Json{{"centre", Json::array({s.centre(0), s.centre(1), s.centre(2)})}, {"radius", s.radius}}; }
Json minimiser_record(const nxr::meeg::NelderMeadOptions& m) {
  return Json{{"reflection", m.reflection}, {"expansion", m.expansion}, {"contraction", m.contraction}, {"shrink", m.shrink}, {"relative_step", m.relativeStep}, {"zero_step", m.zeroStep},
              {"x_tolerance", m.xTolerance}, {"f_tolerance", m.fTolerance}, {"max_evaluations", m.maxEvaluations}};
}
Json fit_record(const nxr::meeg::SphereFitOptions& f) { return Json{{"cost", f.cost == nxr::meeg::SphereCost::Geometric ? "geometric" : "algebraic"}, {"minimiser", minimiser_record(f.minimiser)}}; }
Json vector_record(const Eigen::VectorXd& v) { return Json(std::vector<double>(v.data(), v.data() + v.size())); }

struct HeadPrepared {
  ng::Array channels, sources;
  std::optional<ng::Array> meg_surface, eeg_surface;
  std::string meg = "overlapping_spheres";
  bool eeg = true, project = true;
  nxr::meeg::SarvasOptions sarvas;
  nxr::meeg::SphereFitOptions fit;
  nxr::meeg::SensorSphereOptions sensor;
  nxr::meeg::EegSphereOptions shells;
};

ng::NodeSpec spherical_head_model() {
  ng::NodeSpec n;
  n.name = "meeg.spherical_head_model";
  n.kind = ng::Kind::array;
  n.inputs = {{"channels", "the channels: a points manifold (positions, the 'type' of each, orientations and coils when stored)", false},
              {"sources", "the source surface: its vertices are the sources", false},
              {"meg_surface", "the surface the MEG spheres are fitted to (the inner skull)", true},
              {"eeg_surface", "the surface the EEG sphere is fitted to (the scalp)", true}};
  n.outputs = {{"gain", "the gain, channels x 3 sources (x y z moments): an operator from the sources (vector3) to the channels", false},
               {"grid", "the source positions, sources x 3 (m): a field of the gain on the source surface", false}};
  n.parameters = {
      {"meg", Json("overlapping_spheres"), "the MEG model: overlapping_spheres | single_sphere | none", "-", {"overlapping_spheres", "single_sphere", "none"}, false},
      {"eeg", Json(true), "compute the EEG channels (three shells)", "-", {}, false},
      {"project_electrodes", Json(true), "move the electrodes radially onto the fitted EEG sphere (the formula puts them on the outer shell)", "-", {}, false},
      {"sphere_cost", Json("geometric"), "the sphere fit's cost: geometric | algebraic", "-", {"geometric", "algebraic"}, false},
      {"sensor_weight_exponent", Json(4.0), "the overlapping spheres' weight exponent p (w = sum |weight| d^-p)", "-", {}, false},
      {"eeg_layers", Json("brainstorm"), "the shells' radii and conductivities: brainstorm | mne", "-", {"brainstorm", "mne"}, false},
      {"eeg_radii", Json(), "the shells' relative radii, innermost to outermost; null: the layers'", "-", {}, true},
      {"eeg_conductivities", Json(), "the shells' conductivities, innermost to outermost; null: the layers'", "S/m", {}, true},
      {"eeg_approximation", Json("berg_scherg"), "berg_scherg | series", "-", {"berg_scherg", "series"}, false},
      {"eeg_series_terms", Json(), "the series' terms; null: the default", "-", {}, true},
      {"berg_dipoles", Json(), "the Berg-Scherg dipoles; null: the default", "-", {}, true},
      {"berg_terms", Json(), "the series terms the Berg-Scherg fit matches; null: the default", "-", {}, true},
      {"berg_starts", Json(), "the Berg-Scherg fit's simplex starts; null: the default", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "solved in double; the gain is stored as float32 like every operator");
  n.matches = "Brainstorm's bst_headmodeler (os_meg, meg_sphere, eeg_3sphereberg); meeg nxr::meeg::sarvasGain, eegSphereGain";
  n.description = "the gain of a spherical head model (Sarvas MEG in one or overlapping spheres, three-shell EEG) on a source surface";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"channels", "sources"}) if (!in.count(k)) ng::refuse(std::string("meeg.spherical_head_model: ") + k + " is not given");
    const bool meg = p.value("meg", std::string("overlapping_spheres")) != "none" && in.count("meg_surface"), eeg = p.value("eeg", true) && in.count("eeg_surface");
    if (!meg && !eeg) ng::refuse("meeg.spherical_head_model: no model to compute: name the meg_surface (the inner skull) for MEG, the eeg_surface (the scalp) for EEG");
    ng::Planned out;
    out.out["gain"] = constant_port("operator");
    out.out["grid"] = constant_port("field");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    using namespace nxr::meeg;
    HeadPrepared pr;
    auto take = [&](const char* k) -> std::optional<ng::Array> { const auto it = c.find(k); if (it == c.end() || it->second.meta.value("placeholder", false)) return std::nullopt; return it->second; };
    const auto ch = take("channels"), src = take("sources");
    if (!ch || !src) ng::refuse("meeg.spherical_head_model: the channels and the source surface were not handed over");
    pr.channels = *ch; pr.sources = *src;
    pr.meg_surface = take("meg_surface"); pr.eeg_surface = take("eeg_surface");
    pr.meg = p.value("meg", std::string("overlapping_spheres"));
    pr.eeg = p.value("eeg", true);
    pr.project = p.value("project_electrodes", true);
    const std::string cs = pr.channels.meta.value("cs_id", std::string());
    for (const ng::Array* s : {&pr.sources, pr.meg_surface ? &*pr.meg_surface : nullptr, pr.eeg_surface ? &*pr.eeg_surface : nullptr})
      if (s && !cs.empty() && !s->meta.value("cs_id", std::string()).empty() && s->meta.value("cs_id", std::string()) != cs)
        ng::refuse("meeg.spherical_head_model: " + s->meta.value("name", std::string()) + " and " + pr.channels.meta.value("name", std::string()) + " are not in the same coordinate system");
    const SphereCost cost = p.value("sphere_cost", std::string("geometric")) == "algebraic" ? SphereCost::Algebraic : SphereCost::Geometric;
    pr.fit.cost = cost; pr.sensor.fit.cost = cost;
    pr.sensor.weightExponent = p.value("sensor_weight_exponent", 4.0);
    if (p.value("eeg_layers", std::string("brainstorm")) == "mne") pr.shells = mneEegSphereOptions();
    auto vec = [&](const char* k, Eigen::VectorXd& v) {
      if (!p.contains(k) || p[k].is_null()) return;
      const std::vector<double> x = p[k].get<std::vector<double>>();
      v = Eigen::Map<const Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(x.size()));
    };
    vec("eeg_radii", pr.shells.radius);
    vec("eeg_conductivities", pr.shells.conductivity);
    if (p.value("eeg_approximation", std::string("berg_scherg")) == "series") pr.shells.approximation = EegApproximation::Series;
    if (p.contains("eeg_series_terms") && !p["eeg_series_terms"].is_null()) pr.shells.seriesTerms = p["eeg_series_terms"].get<int>();
    if (p.contains("berg_dipoles") && !p["berg_dipoles"].is_null()) pr.shells.berg.dipoles = p["berg_dipoles"].get<int>();
    if (p.contains("berg_terms") && !p["berg_terms"].is_null()) pr.shells.berg.terms = p["berg_terms"].get<int>();
    if (p.contains("berg_starts") && !p["berg_starts"].is_null()) pr.shells.berg.starts = p["berg_starts"].get<int>();
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    using namespace nxr::meeg;
    const HeadPrepared& pr = std::any_cast<const HeadPrepared&>(prepared);
    const ng::Array& CH = pr.channels;
    const std::string chname = CH.meta.value("name", std::string());
    const std::vector<std::string> types = CH.meta.value("types", std::vector<std::string>());
    const std::int64_t M = CH.rows, N = pr.sources.rows;
    const MatRM ch_pos = Eigen::Map<const MatRM>(CH.data.data(), M, 3);
    const std::optional<std::vector<double>> ch_ori = CH.meta.contains("orientation") ? std::optional<std::vector<double>>(CH.meta["orientation"].get<std::vector<double>>()) : std::nullopt;
    const Eigen::MatrixXd src = Eigen::Map<const MatRM>(pr.sources.data.data(), N, 3);
    const bool have_coils = CH.meta.contains("coils");
    std::vector<double> coil_pos, coil_ori;
    std::vector<std::int32_t> indptr, indices;
    std::vector<float> cw;
    if (have_coils) {
      const Json& k = CH.meta["coils"];
      coil_pos = k["positions"].get<std::vector<double>>(); coil_ori = k["orientation"].get<std::vector<double>>();
      indptr = k["indptr"].get<std::vector<std::int32_t>>(); indices = k["indices"].get<std::vector<std::int32_t>>(); cw = k["weights"].get<std::vector<float>>();
    }
    std::vector<int> meg_ch, eeg_ch;
    for (std::int64_t i = 0; i < M; ++i) {
      const std::string& t = types[static_cast<std::size_t>(i)];
      if (t == "MEG" || t == "MEG MAG" || t == "MEG GRAD") meg_ch.push_back(static_cast<int>(i));
      else if (t == "EEG") eeg_ch.push_back(static_cast<int>(i));
    }
    const bool do_meg = pr.meg != "none" && pr.meg_surface && !meg_ch.empty(), do_eeg = pr.eeg && pr.eeg_surface && !eeg_ch.empty();
    if (!do_meg && !do_eeg) ng::refuse("meeg.spherical_head_model: " + chname + ": no MEG or EEG channel to compute");
    auto points = [](const ng::Array& s) { return Eigen::MatrixXd(Eigen::Map<const MatRM>(s.data.data(), s.rows, 3)); };
    auto coils_of = [&](const std::vector<int>& chans, bool need_orientation) {
      Coils k;
      std::vector<Eigen::Vector3d> pos, ori;
      std::vector<double> w;
      for (std::size_t c = 0; c < chans.size(); ++c) {
        const int ch = chans[c];
        const std::size_t before = pos.size();
        if (have_coils)
          for (std::int32_t q = indptr[static_cast<std::size_t>(ch)]; q < indptr[static_cast<std::size_t>(ch) + 1]; ++q) {
            const std::size_t ci = static_cast<std::size_t>(indices[static_cast<std::size_t>(q)]);
            pos.emplace_back(coil_pos[3 * ci], coil_pos[3 * ci + 1], coil_pos[3 * ci + 2]);
            ori.emplace_back(coil_ori[3 * ci], coil_ori[3 * ci + 1], coil_ori[3 * ci + 2]);
            w.push_back(cw[static_cast<std::size_t>(q)]);
            k.channel.push_back(static_cast<int>(c));
          }
        if (pos.size() == before) {                                 // no coils stored for this channel: its one point
          pos.push_back(ch_pos.row(ch).transpose());
          const std::size_t o = 3 * static_cast<std::size_t>(ch);
          ori.push_back(ch_ori ? Eigen::Vector3d((*ch_ori)[o], (*ch_ori)[o + 1], (*ch_ori)[o + 2]) : Eigen::Vector3d::Constant(std::nan("")));
          w.push_back(1.0);
          k.channel.push_back(static_cast<int>(c));
        }
        if (need_orientation)
          for (std::size_t q = before; q < pos.size(); ++q)
            if (!ori[q].allFinite() || ori[q].norm() < 0.5)
              ng::refuse("meeg.spherical_head_model: " + chname + ": channel " + std::to_string(ch) + " (" + types[static_cast<std::size_t>(ch)] +
                         ") has a sensor point with no orientation; the MEG model needs " + chname + "_orientation or " + chname + "_coils_orientation");
      }
      k.channels = static_cast<int>(chans.size());
      k.position.resize(static_cast<Eigen::Index>(pos.size()), 3);
      k.orientation.resize(static_cast<Eigen::Index>(pos.size()), 3);
      k.weight.resize(static_cast<Eigen::Index>(pos.size()));
      for (std::size_t q = 0; q < pos.size(); ++q) {
        k.position.row(static_cast<Eigen::Index>(q)) = pos[q].transpose();
        k.orientation.row(static_cast<Eigen::Index>(q)) = (need_orientation ? Eigen::Vector3d(ori[q].normalized()) : Eigen::Vector3d(Eigen::Vector3d::Zero())).transpose();
        k.weight(static_cast<Eigen::Index>(q)) = w[q];
      }
      return k;
    };
    const char* meg_name = pr.meg == "single_sphere" ? "single sphere" : pr.meg == "overlapping_spheres" ? "overlapping spheres" : "none";
    Eigen::MatrixXd gain = Eigen::MatrixXd::Zero(M, 3 * N);
    Json computed = Json::object();
    if (do_meg) {
      const Eigen::MatrixXd P = points(*pr.meg_surface);
      const Coils k = coils_of(meg_ch, true);
      Eigen::MatrixXd Gm;
      Json mj = Json{{"model", meg_name}, {"surface", pr.meg_surface->meta.value("name", std::string())}, {"channels", static_cast<int>(meg_ch.size())}};
      if (pr.meg == "single_sphere") {
        const Sphere s = fitSphere(P, Eigen::VectorXd(), pr.fit);
        Gm = sarvasGain(src, k, s.centre, pr.sarvas);
        mj["sphere"] = sphere_record(s);
        int outside = 0;
        for (std::int64_t i = 0; i < N; ++i) if ((src.row(i).transpose() - s.centre).norm() > s.radius) ++outside;
        mj["sources_outside_sphere"] = outside;
      } else {
        const std::vector<Sphere> sp = sensorSpheres(P, k.position, k.weight, k.channel, k.channels, pr.sensor);
        std::vector<Eigen::Vector3d> centres;
        Json sj = Json::array();
        int outside = 0;
        for (const Sphere& s : sp) centres.push_back(s.centre);
        for (std::size_t c = 0; c < sp.size(); ++c) {
          sj.push_back(Json{{"channel", meg_ch[c]}, {"centre", Json::array({sp[c].centre(0), sp[c].centre(1), sp[c].centre(2)})}, {"radius", sp[c].radius}});
          for (std::int64_t i = 0; i < N; ++i) if ((src.row(i).transpose() - sp[c].centre).norm() > sp[c].radius) { ++outside; break; }
        }
        Gm = sarvasGain(src, k, centres, pr.sarvas);
        mj["spheres"] = sj;
        mj["channels_with_a_source_outside_their_sphere"] = outside;
      }
      for (std::size_t c = 0; c < meg_ch.size(); ++c) gain.row(meg_ch[c]) = Gm.row(static_cast<Eigen::Index>(c));
      computed["meg"] = mj;
    }
    if (do_eeg) {
      const Eigen::MatrixXd P = points(*pr.eeg_surface);
      const Sphere s = fitSphere(P, Eigen::VectorXd(), pr.fit);
      Coils k = coils_of(eeg_ch, false);
      if (pr.project)
        for (Eigen::Index q = 0; q < k.position.rows(); ++q) {
          const Eigen::Vector3d d = k.position.row(q).transpose() - s.centre;
          if (d.norm() == 0) ng::refuse("meeg.spherical_head_model: an electrode sits at the centre of the fitted sphere");
          k.position.row(q) = (s.centre + s.radius * d.normalized()).transpose();
        }
      BergScherg used;
      const Eigen::MatrixXd Ge = eegSphereGain(src, k, s.centre, pr.shells, &used);
      for (std::size_t c = 0; c < eeg_ch.size(); ++c) gain.row(eeg_ch[c]) = Ge.row(static_cast<Eigen::Index>(c));
      int outside_inner = 0, outside = 0;
      for (std::int64_t i = 0; i < N; ++i) {
        const double d = (src.row(i).transpose() - s.centre).norm();
        if (d > s.radius) ++outside;
        if (d > s.radius * pr.shells.radius(0)) ++outside_inner;
      }
      Json ej = Json{{"model", "three shells"}, {"surface", pr.eeg_surface->meta.value("name", std::string())}, {"channels", static_cast<int>(eeg_ch.size())}, {"sphere", sphere_record(s)},
                     {"sources_outside_innermost_shell", outside_inner}, {"sources_outside_sphere", outside}};
      if (pr.shells.approximation == EegApproximation::BergScherg)
        ej["berg_scherg"] = Json{{"eccentricity", vector_record(used.eccentricity)}, {"magnitude", vector_record(used.magnitude)}, {"residual", used.residual}};
      computed["eeg"] = ej;
    }
    if (!gain.allFinite()) ng::refuse("meeg.spherical_head_model: the gain holds a NaN (a sensor point on a source?)");
    const auto& el = pr.shells;
    const Json options = Json{{"meg", do_meg ? meg_name : "none"}, {"eeg", do_eeg ? "three shells" : "none"}, {"sarvas", Json{{"permeability", pr.sarvas.permeability}}},
                              {"sphere_fit", fit_record(pr.fit)}, {"sensor_sphere", Json{{"weight_exponent", pr.sensor.weightExponent}, {"fit", fit_record(pr.sensor.fit)}}},
                              {"eeg_sphere", Json{{"radius", vector_record(el.radius)}, {"conductivity", vector_record(el.conductivity)},
                                                  {"approximation", el.approximation == EegApproximation::BergScherg ? "berg-scherg" : "series"},
                                                  {"berg", Json{{"dipoles", el.berg.dipoles}, {"terms", el.berg.terms}, {"starts", el.berg.starts}, {"minimiser", minimiser_record(el.berg.minimiser)}}},
                                                  {"series_terms", el.seriesTerms}, {"centre_tolerance", el.centreTolerance}}},
                              {"project_electrodes", pr.project}};
    // the importer's keys (`export_forward`), then what a computed head model adds
    Json producer = Json{{"units", "m"}, {"n_sources", N}, {"has_source_grid", true}, {"head_model_type", "surface"}, {"bst_surface_file", pr.sources.meta.value("name", std::string())},
                         {"computed", computed}, {"gain_units", Json{{"meg", "T/(A.m)"}, {"eeg", "V/(A.m)"}}}, {"average_reference", false}};
    if (do_meg) producer["meg_method"] = pr.meg == "single_sphere" ? "meg_sphere" : "os_meg";
    if (do_eeg) producer["eeg_method"] = "eeg_3sphereberg";
    ng::Array G;
    G.rows = M; G.cols = 3 * N;
    G.data.resize(static_cast<std::size_t>(M * 3 * N));
    for (std::int64_t i = 0; i < M; ++i) for (std::int64_t j = 0; j < 3 * N; ++j) G.data[static_cast<std::size_t>(i * 3 * N + j)] = gain(i, j);
    G.meta["describe"] = Json{{"from_value_type", "vector3"}, {"to_value_type", "scalar"}, {"precision", "float32"}, {"producer", producer},
                              {"params", Json{{"forward", Json{{"options", options}, {"computed", computed}}}}},
                              {"description", std::string("the leadfield, source field to sensor field (computed: ") + (do_meg ? meg_name : "no MEG") + (do_eeg ? ", three-shell EEG)" : ")")}};
    ng::Array L;
    L.rows = N; L.cols = 3;
    L.data = pr.sources.data;
    L.meta["describe"] = Json{{"value_type", "vector3"}, {"unit", "m"}, {"precision", "float32"}, {"description", "the source positions of the head model"}};
    return ng::Arrays{{"gain", std::move(G)}, {"grid", std::move(L)}};
  };
  return n;
}

}  // namespace

void register_meeg_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {noise_covariance(), minimum_norm(), spherical_head_model()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
