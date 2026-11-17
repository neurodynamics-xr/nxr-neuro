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
#include <complex>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/io/constant_q.h"
#include "nxr/meeg/covariance.h"
#include "nxr/filter.h"
#include "nxr/hodge_weights.h"
#include "nxr/ladder.h"
#include "nxr/meeg/wave.h"
#include "nxr/partition.h"
#include "nxr/pipelines/spectral_record.h"
#include "nxr/timebank.h"
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


// ── timebank.analytic_bands · analytic.tile_cross_spectrum ─────────────────────────────────────────────────────
// THE ANALYTIC CROSS-SPECTRUM (the hand-wired `crossspectrum --estimator analytic` it replaces; spec docs/specs/2026-11-13-cycle-pattern.md, stage 1, the time tiles): each band's
// analytic signal (a Meyer band, gain 2; a chunk node with the band's halo), then per band: decimated by the constant-Q ladder's stride ON THE LINE's grid and z z^H summed per
// tile of `level` (an accumulate node). A tile is kept when it is whole for every band and free of bad samples.
std::vector<std::int32_t> channels_picked(const ng::PortType& x, const ng::Arrays& constants, const Json& p, const char* who) {
  if (const auto it = constants.find("channels_of"); it != constants.end() && !it->second.meta.value("placeholder", false))
    return it->second.meta.value("channels", std::vector<std::int32_t>());
  return good_channels(x, p.value("channel_types", Json::array()), who);
}
struct AnalyticBands {
  nxr::timebank::Design design{nxr::timebank::Family::Meyer, 2.0, nxr::timebank::Sides::Analytic};
  std::vector<std::int32_t> channels;
  double sfreq = 0;
};
std::int64_t analytic_halo(const Json& p, double sfreq) {
  nxr::timebank::Design d(nxr::timebank::Family::Meyer, 2.0, nxr::timebank::Sides::Analytic);
  for (const auto& b : p.value("bands", Json::array())) d.bands.push_back({b.value("lo", b.value("lo_hz", 0.0)), b.value("hi", b.value("hi_hz", 0.0))});
  const nxr::timebank::TimeBank bank = nxr::timebank::design(d, std::size_t(1) << 15, sfreq);
  std::size_t H = 0;
  for (int m = 0; m < static_cast<int>(d.bands.size()); ++m) H = std::max(H, nxr::timebank::haloOf(bank, m));
  return static_cast<std::int64_t>(H);
}
ng::NodeSpec analytic_bands() {
  ng::NodeSpec n;
  n.name = "timebank.analytic_bands";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"channels_of", "an operator whose channels are the ones kept, in order; absent: the good channels of channel_types", true}};
  n.outputs = {{"z", "the analytic signal of each band on the channels kept: bands x channels rows (band-major), complex, at the recording's rate", false}};
  n.parameters = {{"bands", Json::array(), "the bands: a list of {name, lo, hi} in Hz (a Meyer band each, gain 2)", "Hz", {}, false},
                  {"channel_types", Json::array({"MEG", "MEG MAG", "MEG GRAD", "EEG", "SEEG", "ECOG"}), "the channel types kept when channels_of is absent (good channels only)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input, "the block is transformed in double; the output block follows the input (float32 blocks)",
                             "the halo is the length of the narrowest band's response (nxr::timebank::haloOf), samples on each side");
  n.matches = "the Studio's Signal chain meyerBand (toolbox/wavelets/dyadicFrame.ts); compute nxr::timebank::analysisOfBlock";
  n.description = "the analytic signal of each band (Meyer bands), with the halo the narrowest band needs";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed(in, "x", "timebank.analytic_bands");
    const Json bands = p.value("bands", Json::array());
    if (!bands.is_array() || bands.empty()) ng::refuse("timebank.analytic_bands: at least one band");
    for (const auto& b : bands) {
      const double lo = b.value("lo", b.value("lo_hz", 0.0)), hi = b.value("hi", b.value("hi_hz", 0.0));
      if (!(hi > lo && lo > 0)) ng::refuse("timebank.analytic_bands: band " + b.value("name", std::string()) + " needs 0 < lo < hi");
      if (hi >= *x.sfreq / 2) ng::refuse("timebank.analytic_bands: band " + b.value("name", std::string()) + " reaches past Nyquist");
    }
    ng::Planned out;
    ng::PortType z = x;
    z.value = ng::ValueType::complex;
    z.rows = 0;
    z.facts.erase("channels");
    out.out["z"] = z;
    out.halo = analytic_halo(p, *x.sfreq);
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    AnalyticBands a;
    a.sfreq = *in.at("x").sfreq;
    for (const auto& b : p["bands"]) a.design.bands.push_back({b.value("lo", b.value("lo_hz", 0.0)), b.value("hi", b.value("hi_hz", 0.0))});
    a.channels = channels_picked(in.at("x"), constants, p, "timebank.analytic_bands");
    return std::any(std::move(a));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place&) {
    const AnalyticBands& a = std::any_cast<const AnalyticBands&>(prepared);
    const ng::Block& b = in.at("x");
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    const std::int64_t M = static_cast<std::int64_t>(a.channels.size());
    Eigen::MatrixXd block(M, b.width);
    for (std::int64_t i = 0; i < M; ++i) block.row(i) = all.row(a.channels[static_cast<std::size_t>(i)]).cast<double>();
    const std::int64_t left = b.left - keep, own = b.own() + 2 * keep;
    const auto t = nxr::timebank::analysisOfBlock(a.design, block, a.sfreq, static_cast<std::size_t>(left), static_cast<std::size_t>(own));
    const std::int64_t B = static_cast<std::int64_t>(t.own.size());
    ng::Block z;
    z.rows = B * M; z.width = own; z.left = keep; z.right = keep;
    z.re.resize(static_cast<std::size_t>(z.rows * own)); z.im.resize(z.re.size());
    for (std::int64_t k = 0; k < B; ++k)
      for (std::int64_t i = 0; i < M; ++i)
        for (std::int64_t c = 0; c < own; ++c) {
          const std::complex<double> v = t.own[static_cast<std::size_t>(k)](i, c);
          const std::size_t at = static_cast<std::size_t>((k * M + i) * own + c);
          z.re[at] = static_cast<float>(v.real()); z.im[at] = static_cast<float>(v.imag());
        }
    return ng::Blocks{{"z", std::move(z)}};
  };
  return n;
}

struct TilePrepared {
  std::vector<std::int32_t> channels;
  std::vector<std::int64_t> stride;
  int level = 0;
  std::int64_t tile = 0;
  Json bands = Json::array();
  std::string unit;
  double sfreq = 0;
};
struct TileState {
  std::int64_t m0 = -1;
  std::map<std::int64_t, std::vector<Eigen::MatrixXcd>> sum;     // per tile (absolute index on the line), per band
  std::map<std::int64_t, std::vector<std::int64_t>> count;
  std::set<std::int64_t> bad;
  std::int64_t last = -1;                                          // the last sample read + 1
};
ng::NodeSpec tile_cross_spectrum() {
  ng::NodeSpec n;
  n.name = "analytic.tile_cross_spectrum";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"z", "the analytic signal of each band: bands x channels rows (band-major), complex", false},
              {"x", "the recording z was made from (its channels' facts name the channels kept)", false},
              {"channels_of", "the operator whose channels z keeps (as timebank.analytic_bands was given)", true},
              {"exclude", "the recording's bad segments (spans): a tile touching one is left out when bad_segments is exclude", true}};
  n.outputs = {{"channels", "the channels covered, in order", false},
               {"spectrum", "per band and kept tile: the sum of z z^H, band x window x channel x channel (complex)", false},
               {"windows", "the tiles of the level over the span (closed form); the spectrum's windows are the kept ones", false}};
  n.parameters = {{"bands", Json::array(), "the bands, as timebank.analytic_bands was given them", "Hz", {}, false},
                  {"channel_types", Json::array({"MEG", "MEG MAG", "MEG GRAD", "EEG", "SEEG", "ECOG"}), "as timebank.analytic_bands was given them", "-", {}, false},
                  {"level", Json(), "the tile level: a tile is 2^level samples (0 to 15); null: from window_seconds, else the finest of the bands' constant-Q windows", "level", {}, true},
                  {"window_seconds", Json(), "the tile's length in seconds (snapped to a level); null: see level", "s", {}, true},
                  {"bad_segments", Json("exclude"), "exclude = a tile touching one of the recording's bad segments is left out | ignore = every whole tile is kept", "-", {"exclude", "ignore"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "per-tile sums carried across blocks, in double");
  n.matches = "the Studio's envelope tiles (D96); compute nxr::partition::crossSpectrum";
  n.description = "per band and tile of a level, the sum of z z^H of the band's analytic signal decimated on the line's grid";
  auto prep = [](const ng::Types& in, const ng::Arrays* constants, const Json& p) {
    namespace cq = nxr::io::constantq;
    TilePrepared t;
    const ng::PortType& x = streamed(in, "x", "analytic.tile_cross_spectrum");
    t.sfreq = *x.sfreq;
    t.unit = x.unit.empty() ? std::string("channel unit") : x.unit;
    int level = 15;
    for (const auto& b : p.value("bands", Json::array())) {
      const double lo = b.value("lo", b.value("lo_hz", 0.0)), hi = b.value("hi", b.value("hi_hz", 0.0));
      const std::int64_t d = cq::strideFor(t.sfreq, lo);
      if (hi >= t.sfreq / (2.0 * static_cast<double>(d)))
        ng::refuse("analytic.tile_cross_spectrum: band " + b.value("name", std::string()) + " reaches its decimated Nyquist rate (the ladder's stride is sized from the low edge: a band is at most 12 times its low edge wide)");
      t.stride.push_back(d);
      level = std::min(level, cq::levelOfWidth(cq::windowFor(t.sfreq, lo)));
      t.bands.push_back(Json{{"name", b.value("name", std::string())}, {"lo_hz", lo}, {"hi_hz", hi}, {"decimation", d}});
    }
    if (t.stride.empty()) ng::refuse("analytic.tile_cross_spectrum: at least one band");
    const bool lv = p.contains("level") && !p["level"].is_null(), ws = p.contains("window_seconds") && !p["window_seconds"].is_null();
    if (lv && ws) ng::refuse("analytic.tile_cross_spectrum: level and window_seconds are alternatives");
    if (lv) level = p["level"].get<int>();
    else if (ws) level = cq::levelOf(p["window_seconds"].get<double>(), t.sfreq);
    if (level < 0 || level > 15) ng::refuse("analytic.tile_cross_spectrum: the tile level is 0 to 15, got " + std::to_string(level));
    t.level = level;
    t.tile = std::int64_t(1) << level;
    for (std::size_t b = 0; b < t.stride.size(); ++b) {
      if (t.stride[b] > t.tile) ng::refuse("analytic.tile_cross_spectrum: band " + t.bands[b]["name"].get<std::string>() + "'s stride is longer than a level-" + std::to_string(level) + " tile");
      t.bands[b]["window_samples"] = t.tile / t.stride[b];
    }
    if (constants) {
      t.channels = channels_picked(x, *constants, p, "analytic.tile_cross_spectrum");
      // each band's scaling record (its noise bandwidth): a reader converts the per-window sums to density or power from it
      nxr::timebank::Design d(nxr::timebank::Family::Meyer, 2.0, nxr::timebank::Sides::Analytic);
      for (const auto& b : t.bands) d.bands.push_back({b["lo_hz"].get<double>(), b["hi_hz"].get<double>()});
      const nxr::timebank::TimeBank bank = nxr::timebank::design(d, std::size_t(1) << 15, t.sfreq);
      for (std::size_t b = 0; b < t.bands.size(); ++b)
        t.bands[b]["spectral_scaling"] = nxr::pipelines::analytic_scaling_record(nxr::timebank::noiseBandwidth(bank, static_cast<int>(b)), t.unit + "\xc2\xb2");
    }
    return t;
  };
  n.plan = [prep](const ng::Types& in, const Json& p) {
    const TilePrepared t = prep(in, nullptr, p);
    ng::Planned out;
    out.out["channels"] = constant_port("selection");
    out.out["spectrum"] = constant_port("statistic");
    out.out["windows"] = constant_port("selection");
    out.align = t.tile;
    return out;
  };
  n.prepare = [prep](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) { return std::any(prep(in, &constants, p)); };
  n.start = [](const std::any&) { return std::any(TileState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const TilePrepared& t = std::any_cast<const TilePrepared&>(prepared);
    TileState& s = std::any_cast<TileState&>(state);
    const ng::Block& b = in.at("z");
    const std::int64_t M = static_cast<std::int64_t>(t.channels.size()), B = static_cast<std::int64_t>(t.stride.size()), own = b.own();
    if (b.rows != B * M) ng::refuse("analytic.tile_cross_spectrum: z has " + std::to_string(b.rows) + " rows, not bands x channels");
    if (s.m0 < 0) s.m0 = at.first >> t.level;
    if (at.good) for (std::int64_t j = 0; j < own; ++j) if (!(*at.good)[static_cast<std::size_t>(j)]) s.bad.insert((at.first + j) >> t.level);
    for (std::int64_t k = 0; k < B; ++k) {
      const std::int64_t d = t.stride[static_cast<std::size_t>(k)];
      const std::int64_t offset = (d - at.first % d) % d;                     // the LINE's grid: the samples kept are multiples of d
      const std::int64_t cols = offset < own ? (own - offset + d - 1) / d : 0;
      Eigen::MatrixXcd Z(M, cols);
      for (std::int64_t i = 0; i < M; ++i)
        for (std::int64_t c = 0; c < cols; ++c) {
          const std::size_t q = static_cast<std::size_t>((k * M + i) * b.width + b.left + offset + c * d);
          Z(i, c) = std::complex<double>(b.re[q], b.im[q]);
        }
      for (Eigen::Index col = 0; col < Z.cols();) {                            // one sum per tile: m = line sample >> level
        const std::int64_t x = at.first + offset + col * d, m = x >> t.level;
        const std::int64_t end = (m + 1) << t.level;
        const Eigen::Index len = static_cast<Eigen::Index>(std::min<std::int64_t>(Z.cols() - col, (end - x + d - 1) / d));
        const auto cs = nxr::partition::crossSpectrum(Z.middleCols(col, len), static_cast<std::size_t>(len), 1);
        auto& sv = s.sum[m];
        auto& cv = s.count[m];
        if (sv.empty()) { sv.assign(static_cast<std::size_t>(B), Eigen::MatrixXcd::Zero(M, M)); cv.assign(static_cast<std::size_t>(B), 0); }
        sv[static_cast<std::size_t>(k)] += cs.sum[0];
        cv[static_cast<std::size_t>(k)] += cs.count[0];
        col += len;
      }
    }
    s.last = at.first + own;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const TilePrepared& t = std::any_cast<const TilePrepared&>(prepared);
    const TileState& s = std::any_cast<const TileState&>(state);
    if (s.m0 < 0) ng::refuse("analytic.tile_cross_spectrum: no sample was read");
    const std::int64_t M = static_cast<std::int64_t>(t.channels.size()), B = static_cast<std::int64_t>(t.stride.size());
    const std::int64_t W = (s.last >> t.level) - s.m0;                        // the whole tiles of the span (a partial last one is left out)
    if (W < 1) ng::refuse("analytic.tile_cross_spectrum: the span holds no whole tile of level " + std::to_string(t.level) + " (" + std::to_string(t.tile) + " samples)");
    std::vector<std::int64_t> kept;
    for (std::int64_t m = s.m0; m < s.m0 + W; ++m) {
      if (s.bad.count(m) || !s.count.count(m)) continue;
      bool whole = true;
      for (std::int64_t k = 0; k < B; ++k) whole = whole && s.count.at(m)[static_cast<std::size_t>(k)] == t.tile / t.stride[static_cast<std::size_t>(k)];
      if (whole) kept.push_back(m);
    }
    if (kept.empty()) ng::refuse("analytic.tile_cross_spectrum: no whole tile of " + std::to_string(t.tile) + " samples is free of the recording's bad segments");
    const std::int64_t K = static_cast<std::int64_t>(kept.size());
    ng::Array ch;
    ch.rows = M; ch.cols = 1;
    for (const auto c : t.channels) ch.data.push_back(c);
    ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the cross-spectrum covers, in order"}};
    ng::Array S;
    S.rows = B; S.cols = K * M * M * 2;
    S.data.resize(static_cast<std::size_t>(B * K * M * M * 2));
    for (std::int64_t k = 0; k < B; ++k)
      for (std::int64_t w = 0; w < K; ++w)
        for (std::int64_t i = 0; i < M; ++i)
          for (std::int64_t j = 0; j < M; ++j) {
            const std::complex<double> v = s.sum.at(kept[static_cast<std::size_t>(w)])[static_cast<std::size_t>(k)](i, j);
            const std::size_t at = static_cast<std::size_t>((((k * K + w) * M + i) * M + j) * 2);
            S.data[at] = v.real(); S.data[at + 1] = v.imag();
          }
    const Json tiling = Json{{"unit", "samples"}, {"origin", s.m0 << t.level}, {"width", t.tile}, {"hop", t.tile}, {"count", K}, {"skipped", W - K}, {"end", (s.m0 + W) << t.level}};
    S.meta["describe"] = Json{{"measure", "cross_spectrum_sum"}, {"method", "analytic"}, {"label", "analytic"}, {"precision", "float64"}, {"value_type", "complex"},
                              {"shape", Json::array({B, K, M, M, 2})}, {"unit", t.unit + "\xc2\xb2"},
                              {"axes", Json::array({Json{{"name", "band"}, {"size", B}, {"values", t.bands}}, Json{{"name", "window"}, {"size", K}, {"members", kept}},
                                                    Json{{"name", "channel"}, {"size", M}}, Json{{"name", "channel"}, {"size", M}}})},
                              {"tiling", tiling}, {"n_members", K}, {"members_skipped", W - K},
                              {"producer", Json{{"stage", "analytic cross-spectrum"}, {"window_level", t.level}, {"window_index", kept}, {"family", "meyer"}, {"gain", 2.0}, {"grid", "line"}}},
                              {"options", Json{{"level", t.level}, {"bands", t.bands}}},
                              {"description", "per band and tile of level " + std::to_string(t.level) + " (" + std::to_string(t.tile) + " samples), the sum of z z^H on " + std::to_string(M) + " channels"}};
    ng::Array Wn;
    Wn.rows = 0; Wn.cols = 1;
    Wn.meta["describe"] = Json{{"tiling", tiling}, {"params", Json{{"level", t.level}}},
                               {"description", "the tiles of level " + std::to_string(t.level) + " (" + std::to_string(t.tile) + " samples) of the span; " + std::to_string(K) + " of them kept"}};
    return ng::Arrays{{"channels", std::move(ch)}, {"spectrum", std::move(S)}, {"windows", std::move(Wn)}};
  };
  return n;
}


// ── meeg.phase_gradient ─────────────────────────────────────────────────────────────────────────────────────────
// THE BAND'S PHASE GRADIENT AS THE PYRAMID'S LEAF INGREDIENTS (the hand-wired `phasegradient` it replaces; spec docs/specs/2026-11-14-measurement-pyramid.md, step 5): from a
// stored band cross-spectrum C and a free kernel K, the signal's rows A (the normal current, or each ambient component) and their tangent P1 gradients give per vertex P (power),
// H (the phase current, x y z) and Q (the gradient power) (nxr::meeg::phaseSums); H is read in the canonical frames; per leaf, the sums weighted by the vertex's lumped area:
// phase_power, phase_gradient_power, phase_current_1/2, phase_current_magnitude. Ratios (the gradient, the travelling index, the speed) are derived at read.
struct StatBand { Eigen::MatrixXd S, Sq; std::vector<std::int32_t> channels; std::string name; double lo = 0, hi = 0; bool density = true; };
// one band of a stored cross-spectrum statistic (band x channel x channel, or channel x channel for a statistic of one band), its channels in order
StatBand band_of(const ng::Array& C, const Json& band, const char* who) {
  StatBand out;
  const Json axes = C.meta.value("axes", Json::array());
  const std::vector<std::int64_t> shape = C.meta.value("shape", std::vector<std::int64_t>());
  out.channels = C.meta.value("channels", std::vector<std::int32_t>());
  const std::int64_t M = static_cast<std::int64_t>(out.channels.size());
  out.density = C.meta.value("measure", std::string()) == "cross_spectral_density";
  std::size_t b = 0;
  if (!axes.empty() && axes[0].value("name", std::string()) == "band") {
    const Json& entries = axes[0]["values"];
    if (band.is_null()) { if (entries.size() != 1) ng::refuse(std::string(who) + ": the cross-spectrum holds " + std::to_string(entries.size()) + " bands; name one (band)"); }
    else {
      b = entries.size();
      for (std::size_t k = 0; k < entries.size(); ++k) if (entries[k].value("name", std::string()) == band.get<std::string>()) b = k;
      if (b == entries.size()) ng::refuse(std::string(who) + ": the cross-spectrum has no band '" + band.get<std::string>() + "'");
    }
    out.name = entries[b].value("name", std::string()); out.lo = entries[b].value("lo_hz", 0.0); out.hi = entries[b].value("hi_hz", 0.0);
  } else {
    const Json pj = C.meta.value("producer", Json::object());
    out.name = pj.value("band_name", std::string()); out.lo = pj.value("band_lo_hz", 0.0); out.hi = pj.value("band_hi_hz", 0.0);
    if (!(out.hi > 0)) ng::refuse(std::string(who) + ": the cross-spectrum does not name its band's edges");
  }
  if (static_cast<std::int64_t>(C.data.size()) % (M * M * 2) != 0 || M == 0) ng::refuse(std::string(who) + ": the cross-spectrum's array does not fit its channels");
  (void)shape;
  out.S.resize(M, M); out.Sq.resize(M, M);
  const std::size_t base = b * static_cast<std::size_t>(M * M * 2);
  for (std::int64_t i = 0; i < M; ++i)
    for (std::int64_t j = 0; j < M; ++j) {
      out.S(i, j) = C.data[base + static_cast<std::size_t>((i * M + j) * 2)];
      out.Sq(i, j) = C.data[base + static_cast<std::size_t>((i * M + j) * 2 + 1)];
    }
  return out;
}
// C reordered to `want` (the kernel's channels)
void reorder(StatBand& c, const std::vector<std::int32_t>& want, const char* who) {
  std::map<std::int32_t, Eigen::Index> where;
  for (std::size_t i = 0; i < c.channels.size(); ++i) where[c.channels[i]] = static_cast<Eigen::Index>(i);
  std::vector<Eigen::Index> at;
  for (const auto ch : want) {
    const auto it = where.find(ch);
    if (it == where.end()) ng::refuse(std::string(who) + ": the cross-spectrum does not cover channel " + std::to_string(ch) + " of the kernel (compute it with channels_of the kernel)");
    at.push_back(it->second);
  }
  const Eigen::Index m = static_cast<Eigen::Index>(at.size());
  Eigen::MatrixXd S(m, m), Q(m, m);
  for (Eigen::Index i = 0; i < m; ++i) for (Eigen::Index j = 0; j < m; ++j) { S(i, j) = c.S(at[i], at[j]); Q(i, j) = c.Sq(at[i], at[j]); }
  c.S = std::move(S); c.Sq = std::move(Q); c.channels = want;
}
struct PhasePrepared { ng::Arrays c; Json band; std::string signal; };
ng::NodeSpec phase_gradient() {
  ng::NodeSpec n;
  n.name = "meeg.phase_gradient";
  n.kind = ng::Kind::array;
  n.inputs = {{"cross_spectrum", "the band cross-spectrum: a stored statistic (band x channel x channel, or one band's channel x channel)", false},
              {"kernel", "a free-orientation inverse kernel on the cross-spectrum's channels", false},
              {"surface", "the kernel's surface", false},
              {"frames", "its canonical frames: a field, vertices x 9 (e1, e2, n)", false},
              {"leaves", "the leaves: a partition of the surface's vertices (a level of its ladder)", false}};
  n.outputs = {{"field", "the field the leaves' measurements are of (no array: its values are the measurements)", false},
               {"measurements", "per leaf: phase_power, phase_gradient_power, phase_current_1, phase_current_2, phase_current_magnitude", false}};
  n.parameters = {{"band", Json(), "the band of the cross-spectrum (its name); null: its one band", "-", {}, true},
                  {"signal", Json("normal"), "normal = the current along the surface's normal | ambient = each of its x y z components", "-", {"normal", "ambient"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "in double from a stored cross-spectrum (no time axis)");
  n.matches = "nxr.detect.core_phase PhaseReader.read_out (grad phi = Im(conj z grad z) / |z|^2), power-weighted; meeg nxr::meeg::phaseSums";
  n.description = "a band's phase gradient from its cross-spectrum through a free kernel, as the additive ingredients of the surface's leaves";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"cross_spectrum", "kernel", "surface", "frames", "leaves"}) if (!in.count(k)) ng::refuse(std::string("meeg.phase_gradient: ") + k + " is not given");
    if (p.value("signal", std::string("normal")) != "normal" && p.value("signal", std::string("normal")) != "ambient") ng::refuse("meeg.phase_gradient: signal is normal or ambient");
    ng::Planned out;
    out.out["field"] = constant_port("field");
    ng::PortType m = constant_port("measurement");
    m.facts["measure"] = "phase_power";
    m.facts["over"] = Json{{"selection", "leaves"}};
    out.out["measurements"] = m;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    PhasePrepared pr;
    for (const char* k : {"cross_spectrum", "kernel", "surface", "frames", "leaves"}) {
      const auto it = c.find(k);
      if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string("meeg.phase_gradient: ") + k + " was not handed over");
      pr.c[k] = it->second;
    }
    pr.band = p.contains("band") ? p["band"] : Json();
    pr.signal = p.value("signal", std::string("normal"));
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const PhasePrepared& pr = std::any_cast<const PhasePrepared&>(prepared);
    const ng::Array& KA = pr.c.at("kernel");
    const ng::Array& SA = pr.c.at("surface");
    const ng::Array& FA = pr.c.at("frames");
    const ng::Array& LA = pr.c.at("leaves");
    if (KA.meta.value("to_value_type", std::string("vector3")) == "scalar") ng::refuse("meeg.phase_gradient: a fixed-orientation kernel: the phase gradient reads the current's components");
    const std::vector<std::int32_t> chans = KA.meta.value("channels", std::vector<std::int32_t>());
    const std::int64_t M = static_cast<std::int64_t>(chans.size()), N = KA.rows / 3;
    if (KA.cols != M || KA.rows % 3 != 0) ng::refuse("meeg.phase_gradient: the kernel is not 3 sources x its channels");
    if (SA.rows != N) ng::refuse("meeg.phase_gradient: the surface has " + std::to_string(SA.rows) + " vertices, the kernel " + std::to_string(N) + " sources");
    const std::string unit = SA.meta.value("unit", std::string());
    if (unit != "m" && unit != "mm") ng::refuse("meeg.phase_gradient: the surface's coordinates are in '" + unit + "', not m or mm: the speed needs a length unit");
    const double metres = unit == "mm" ? 1e-3 : 1.0;
    StatBand C = band_of(pr.c.at("cross_spectrum"), pr.band, "meeg.phase_gradient");
    reorder(C, chans, "meeg.phase_gradient");
    const std::vector<std::int32_t> faces = SA.meta.value("faces", std::vector<std::int32_t>());
    const nxr::hodge::RowMat3 P = Eigen::Map<const nxr::hodge::RowMat3>(SA.data.data(), N, 3);
    const nxr::hodge::FaceMat F = Eigen::Map<const nxr::hodge::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
    const nxr::hodge::FaceOperators ops(P, F, false);
    const Eigen::MatrixXd K = Eigen::Map<const MatRM>(KA.data.data(), KA.rows, M);
    nxr::meeg::PhaseSums sums;
    auto add_signal = [&](const Eigen::MatrixXd& A) {
      Eigen::MatrixXd G[3];
      ops.vertexGradient(A, G);
      for (std::int64_t v = 0; v < N; ++v) {
        const Eigen::RowVectorXd dot = G[0].row(v) * ops.normals(v, 0) + G[1].row(v) * ops.normals(v, 1) + G[2].row(v) * ops.normals(v, 2);
        for (int d = 0; d < 3; ++d) G[d].row(v) -= dot * ops.normals(v, d);
      }
      const nxr::meeg::PhaseSums one = nxr::meeg::phaseSums(A, G, C.S, C.Sq);
      if (sums.power.size() == 0) sums = one; else nxr::meeg::addPhaseSums(sums, one);
    };
    if (pr.signal == "normal") {
      Eigen::MatrixXd A(N, M);
      for (std::int64_t v = 0; v < N; ++v) A.row(v) = ops.normals(v, 0) * K.row(3 * v) + ops.normals(v, 1) * K.row(3 * v + 1) + ops.normals(v, 2) * K.row(3 * v + 2);
      add_signal(A);
    } else {
      for (int c = 0; c < 3; ++c) {
        Eigen::MatrixXd A(N, M);
        for (std::int64_t v = 0; v < N; ++v) A.row(v) = K.row(3 * v + c);
        add_signal(A);
      }
    }
    if (static_cast<std::int64_t>(FA.data.size()) != N * 9) ng::refuse("meeg.phase_gradient: the frames hold " + std::to_string(FA.data.size()) + " values for " + std::to_string(N) + " vertices");
    Eigen::MatrixXd e1(N, 3), e2(N, 3);
    for (std::int64_t i = 0; i < N; ++i) for (int k = 0; k < 3; ++k) { e1(i, k) = FA.data[static_cast<std::size_t>(i * 9 + k)]; e2(i, k) = FA.data[static_cast<std::size_t>(i * 9 + 3 + k)]; }
    const nxr::meeg::PhaseCurrentFrames jf = nxr::meeg::phaseCurrentInFrames(sums.phase, e1, e2);
    nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(N); mesh.positions = SA.data; mesh.faces = faces;
    const std::vector<double> area = nxr::ladder::vertexAreas(mesh);
    std::vector<double> wv(static_cast<std::size_t>(N)), vp(wv.size()), vq(wv.size()), v1(wv.size()), v2(wv.size()), vm(wv.size());
    std::int64_t singular = 0;
    for (std::int64_t v = 0; v < N; ++v) {
      const std::size_t i = static_cast<std::size_t>(v);
      const bool ok = std::isfinite(jf.j1(v)) && std::isfinite(jf.j2(v)) && std::isfinite(sums.power(v)) && std::isfinite(sums.gradientPower(v));
      if (!ok) ++singular;
      wv[i] = ok ? area[i] : 0.0;
      vp[i] = ok ? sums.power(v) : 0.0;
      vq[i] = ok ? sums.gradientPower(v) / (metres * metres) : 0.0;
      v1[i] = ok ? jf.j1(v) / metres : 0.0;
      v2[i] = ok ? jf.j2(v) / metres : 0.0;
      vm[i] = ok ? jf.magnitude(v) / metres : 0.0;
    }
    std::vector<std::int32_t> codes;
    for (const double c : LA.data) codes.push_back(static_cast<std::int32_t>(c));
    if (static_cast<std::int64_t>(codes.size()) != N) ng::refuse("meeg.phase_gradient: the leaves are not a partition of the surface's vertices");
    const int L = LA.meta.contains("n_members") ? LA.meta["n_members"].get<int>() : (codes.empty() ? 0 : *std::max_element(codes.begin(), codes.end()) + 1);
    const std::vector<std::vector<double>> cols = {vp, vq, v1, v2, vm};
    ng::Array Mz;
    Mz.rows = L; Mz.cols = static_cast<std::int64_t>(cols.size());
    Mz.data.assign(static_cast<std::size_t>(L) * cols.size(), 0.0);
    for (std::size_t c = 0; c < cols.size(); ++c) {
      const auto sum = nxr::partition::accumulate(codes.data(), codes.size(), L, cols[c].data(), wv.data()).sum;
      for (int l = 0; l < L; ++l) Mz.data[static_cast<std::size_t>(l) * cols.size() + c] = sum[static_cast<std::size_t>(l)];
    }
    const std::string per_hz = C.density ? "/Hz" : "";
    const double omega = M_PI * (C.lo + C.hi);
    Mz.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "phase_power"}, {"unit", std::string("A\xc2\xb2\xc2\xb7m\xc2\xb2") + per_hz}},
                                                         Json{{"measure", "phase_gradient_power"}, {"unit", std::string("A\xc2\xb2") + per_hz}},
                                                         Json{{"measure", "phase_current_1"}, {"unit", "rad/m x power"}}, Json{{"measure", "phase_current_2"}, {"unit", "rad/m x power"}},
                                                         Json{{"measure", "phase_current_magnitude"}, {"unit", "rad/m x power"}}})},
                               {"band", Json{{"name", C.name}, {"lo_hz", C.lo}, {"hi_hz", C.hi}}}};
    ng::Array Fd;
    Fd.meta["describe"] = Json{{"no_array", true}, {"value_type", "vector3"},
                               {"description", "the phase gradient of the " + pr.signal + " current by " + KA.meta.value("name", std::string()) + " \xe2\x80\x94 " + C.name +
                                               ", stored as its additive ingredients per leaf of " + LA.meta.value("name", std::string())},
                               {"producer", Json{{"band", Json{{"name", C.name}, {"lo_hz", C.lo}, {"hi_hz", C.hi}}}, {"omega_rad_s", omega}, {"signal", pr.signal}, {"surface_unit", unit},
                                                 {"leaves", Json{{"selection", LA.meta.value("name", std::string())}, {"count", L}, {"vertices_without_frame", singular}, {"weights", "the vertex's lumped area"}}},
                                                 {"derived_at_read", "phase_gradient_1/2 = j / P, phase_gradient_magnitude = |sum j| / P, travelling_index = |sum j| / sqrt(P Q), phase_coherence = |sum j| / sum |j|; the speed is omega over the gradient's magnitude"},
                                                 {"studio", Json{{"type", "complex"}}}}}};
    return ng::Arrays{{"field", std::move(Fd)}, {"measurements", std::move(Mz)}};
  };
  return n;
}


// ── filter.channel_mean · filter.apply ──────────────────────────────────────────────────────────────────────────
// BRAINSTORM'S FILTERS AS NODES (the hand-wired lowpass, highpass, bandpass, bandstop and notch they replace): a windowed chunk node, each block filtered with the halo the design
// needs (look-behind, look-ahead), the record's ends padded as the option set says. A FIR filter with remove_mean subtracts the record's channel means first (a statistic made by
// `filter.channel_mean` in its own run: a first pass) and adds them back when the filter has no high-pass part.
nxr::filter::FilterOptions filter_options(const Json& p, double sfreq) {
  using namespace nxr::filter;
  FilterOptions o = brainstormFilterOptions(sfreq);
  const std::string kind = p.value("kind", std::string());
  o.kind = kind == "lowpass" ? Kind::Lowpass : kind == "highpass" ? Kind::Highpass : kind == "bandpass" ? Kind::Bandpass : kind == "bandstop" ? Kind::Bandstop : kind == "notch" ? Kind::Notch
           : (ng::refuse("filter: kind is lowpass, highpass, bandpass, bandstop or notch"), Kind::Lowpass);
  auto num = [&](const char* k) -> std::optional<double> { return p.contains(k) && !p[k].is_null() ? std::optional<double>(p[k].get<double>()) : std::nullopt; };
  if (const auto v = num("lower")) o.lowerEdge = *v;
  if (const auto v = num("upper")) o.upperEdge = *v;
  if (p.contains("frequencies") && p["frequencies"].is_array()) o.frequencies = p["frequencies"].get<std::vector<double>>();
  if (p.contains("method") && !p["method"].is_null()) o.method = p["method"].get<std::string>() == "iir" ? Method::Iir : Method::Fir;
  if (const auto v = num("ripple")) o.ripple = *v;
  if (const auto v = num("transition")) o.transitionBandwidth = *v;
  if (p.contains("edge") && !p["edge"].is_null()) o.edge = p["edge"].get<std::string>() == "zeros" ? Edge::Zeros : Edge::MeanOfTransient;
  if (const auto v = num("transient_energy")) o.transientEnergy = *v;
  if (const auto v = num("transient_samples")) o.transientSamples = static_cast<std::size_t>(*v);
  if (p.contains("remove_mean") && !p["remove_mean"].is_null()) o.removeMean = p["remove_mean"].get<bool>();
  if (const auto v = num("notch_bandwidth")) o.notchBandwidth = *v;
  if (const auto v = num("bandstop_order")) o.bandstopOrder = static_cast<int>(*v);
  if (const auto v = num("tolerance")) o.tolerance = *v;
  const int harmonics = p.contains("harmonics") && !p["harmonics"].is_null() ? p["harmonics"].get<int>() : 0;
  if (harmonics != 0 && o.kind == Kind::Notch && o.frequencies.size() == 1) {
    const double line = o.frequencies[0], bw = o.notchBandwidth.value_or(1.0);
    o.frequencies.clear();
    for (int k = 1; (harmonics < 0 || k <= harmonics) && k * line < sfreq / 2 - bw; ++k) o.frequencies.push_back(k * line);
  }
  return o;
}
const std::vector<ng::ParamSpec>& filter_parameters() {
  static const std::vector<ng::ParamSpec> ps = {
      {"kind", Json(), "lowpass | highpass | bandpass | bandstop | notch", "-", {"lowpass", "highpass", "bandpass", "bandstop", "notch"}, false},
      {"lower", Json(), "the lower edge: highpass edge, bandpass lower passband edge, bandstop lower stopband edge", "Hz", {}, true},
      {"upper", Json(), "the upper edge: lowpass edge, bandpass upper passband edge, bandstop upper stopband edge", "Hz", {}, true},
      {"frequencies", Json(), "notch: the line frequency (and its harmonics)", "Hz", {}, true},
      {"harmonics", Json(), "notch: with one frequency, its harmonics below Nyquist up to this many (-1: all); null: none", "-", {}, true},
      {"method", Json(), "fir | iir; null: the reference's (FIR, but IIR for bandstop and notch)", "-", {"fir", "iir"}, true},
      {"ripple", Json(), "FIR: the passband and stopband deviation (linear); null: 1e-3", "-", {}, true},
      {"transition", Json(), "FIR: each transition band; null: the reference's rule", "Hz", {}, true},
      {"edge", Json(), "FIR: the record's ends padded with the mean of the transient (mean_of_transient) or zeros", "-", {"mean_of_transient", "zeros"}, true},
      {"transient_energy", Json(), "FIR: the transient holds this fraction of the response's energy; null: 0.99", "-", {}, true},
      {"transient_samples", Json(), "FIR: the transient's length stated", "samples", {}, true},
      {"remove_mean", Json(), "FIR: subtract the record's channel means first (added back without a high-pass part); null: true", "-", {}, true},
      {"notch_bandwidth", Json(), "IIR notch: each notch's 3-dB bandwidth; null: 1 Hz", "Hz", {}, true},
      {"bandstop_order", Json(), "IIR bandstop: the Butterworth prototype's order; null: 4", "-", {}, true},
      {"tolerance", Json(), "IIR: the decayed tail the look-behind / look-ahead may leave out; null: 1e-15", "-", {}, true}};
  return ps;
}

ng::NodeSpec channel_mean() {
  ng::NodeSpec n;
  n.name = "filter.channel_mean";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}};
  n.outputs = {{"mean", "each channel's mean over the span (a statistic over the channels)", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "the sums are carried across blocks in double");
  n.matches = "Brainstorm's process_bandpass (remove the mean before a FIR filter); compute nxr::filter::RunningMean";
  n.description = "each channel's mean over the span (the first pass of a FIR filter that removes the record's mean)";
  n.plan = [](const ng::Types& in, const Json&) {
    streamed(in, "x", "filter.channel_mean");
    ng::Planned out;
    out.out["mean"] = constant_port("statistic");
    return out;
  };
  n.start = [](const std::any&) { return std::any(std::optional<nxr::filter::RunningMean>{}); };
  n.step = [](const std::any&, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    auto& rm = std::any_cast<std::optional<nxr::filter::RunningMean>&>(state);
    const ng::Block& b = in.at("x");
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    if (!rm) rm.emplace(static_cast<int>(b.rows));
    rm->add(all.middleCols(b.left, b.own()).cast<double>());
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any&, std::any state) {
    const auto& rm = std::any_cast<const std::optional<nxr::filter::RunningMean>&>(state);
    if (!rm || rm->samples() == 0) ng::refuse("filter.channel_mean: no sample was read");
    const Eigen::VectorXd m = rm->mean();
    ng::Array a;
    a.rows = m.size(); a.cols = 1;
    a.data.assign(m.data(), m.data() + m.size());
    a.meta["describe"] = Json{{"measure", "mean"}, {"method", "sample"}, {"precision", "float64"}, {"axes", Json::array({Json{{"name", "channel"}, {"size", m.size()}}})},
                              {"n_members", rm->samples()}, {"description", "each channel's mean over " + std::to_string(rm->samples()) + " samples"}};
    return ng::Arrays{{"mean", std::move(a)}};
  };
  return n;
}

struct FilterPrepared { std::optional<nxr::filter::Filter> f; std::int64_t halo = 0; };
ng::NodeSpec filter_apply() {
  ng::NodeSpec n;
  n.name = "filter.apply";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false},
              {"mean", "the record's channel means (filter.channel_mean): needed by a FIR filter that removes the mean", true}};
  n.outputs = {{"y", "the filtered recording, channels x samples", false}};
  n.parameters = filter_parameters();
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input, "each block filtered in double; the output follows the input",
                             "the halo is the design's look-behind and look-ahead (the FIR's half length; the IIR's decayed response at `tolerance`), samples on each side");
  n.matches = "Brainstorm's process_bandpass / process_notch (bst_bandpass_hfilter, kaiserord+fir1; filtfilt IIR); compute nxr::filter";
  n.description = "a zero-phase filter (lowpass, highpass, bandpass, bandstop, notch) on each block, with the halo its design needs";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed(in, "x", "filter.apply");
    nxr::filter::Design d;
    try { d = nxr::filter::design(filter_options(p, *x.sfreq)); } catch (const std::invalid_argument& e) { ng::refuse(std::string("filter.apply: ") + e.what()); }
    ng::Planned out;
    ng::PortType y = x;
    y.facts.erase("channels");
    out.out["y"] = y;
    out.halo = static_cast<std::int64_t>(std::max(d.lookBehind, d.lookAhead));
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json& p, ng::Shared&) {
    FilterPrepared pr;
    const ng::PortType& x = in.at("x");
    const nxr::filter::FilterOptions o = filter_options(p, *x.sfreq);
    const nxr::filter::Design d = nxr::filter::design(o);
    Eigen::VectorXd mean;
    if (d.method == nxr::filter::Method::Fir && o.removeMean) {
      const auto it = c.find("mean");
      if (it == c.end() || it->second.meta.value("placeholder", false))
        ng::refuse("filter.apply: a FIR filter that removes the record's mean needs the channel means (`nxr-neuro run channel_mean` first), or remove_mean false");
      mean = Eigen::Map<const Eigen::VectorXd>(it->second.data.data(), static_cast<Eigen::Index>(it->second.data.size()));
    }
    std::int64_t rows = x.rows;
    if (rows <= 0 && x.facts.contains("channels")) rows = x.facts["channels"].value("n_channels", std::int64_t(0));
    if (rows <= 0) rows = mean.size();
    if (rows <= 0) ng::refuse("filter.apply: the recording's channel count is not known");
    pr.f.emplace(static_cast<int>(rows), o, mean);
    pr.halo = static_cast<std::int64_t>(std::max(d.lookBehind, d.lookAhead));
    return std::any(std::move(pr));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place& at) {
    const FilterPrepared& pr = std::any_cast<const FilterPrepared&>(prepared);
    const ng::Block& b = in.at("x");
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    const bool at_start = at.first - b.left == 0 && b.left < pr.halo + keep, at_end = b.right < pr.halo + keep;
    const std::int64_t left = b.left - keep, own = b.own() + 2 * keep;
    const Eigen::MatrixXd yb = pr.f->block(all.cast<double>(), static_cast<std::size_t>(left), static_cast<std::size_t>(own), at_start, at_end);
    ng::Block y;
    y.rows = b.rows; y.width = own; y.left = keep; y.right = keep;
    y.re.resize(static_cast<std::size_t>(b.rows * own));
    for (std::int64_t r = 0; r < b.rows; ++r) for (std::int64_t c = 0; c < own; ++c) y.re[static_cast<std::size_t>(r * own + c)] = static_cast<float>(yb(r, c));
    return ng::Blocks{{"y", std::move(y)}};
  };
  return n;
}


// ── hodge.weights ───────────────────────────────────────────────────────────────────────────────────────────────
// THE FUSED HODGE-DIRAC WEIGHTS B [2K x channels] (nxr.inverse.fused fused_weights; nxr-compute hodgeWeights): a free kernel carried onto the package's glued surface, then each
// mode's coefficients a (normal-current, rows 0..K) and b (flow: an exact mode's divergence potential, a coexact mode's stream function, rows K..2K) as linear maps of the
// channels. Handed on as an operator from the kernel's channels: a node that applies a kernel applies it (band power, cross-spectra of the modes).
nxr::hodge::Basis basis_of(const ng::Array& H, std::optional<std::int64_t> modes) {
  nxr::hodge::Basis b;
  const std::int64_t Ktot = H.rows, Vg = H.cols, K = modes ? std::min(*modes, Ktot) : Ktot;
  b.P.resize(K, Vg);
  for (std::int64_t k = 0; k < K; ++k) for (std::int64_t v = 0; v < Vg; ++v) b.P(k, v) = H.data[static_cast<std::size_t>(k * Vg + v)];
  const std::vector<double> mu = H.meta.at("mu").get<std::vector<double>>();
  const std::vector<int> kind = H.meta.at("kind").get<std::vector<int>>();
  b.mu = Eigen::Map<const Eigen::VectorXd>(mu.data(), K);
  b.kind.assign(kind.begin(), kind.begin() + K);
  const std::vector<double> mass = H.meta.at("mass").get<std::vector<double>>();
  b.mass = Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size()));
  const std::vector<double> pos = H.meta.at("positions").get<std::vector<double>>();
  b.positions = Eigen::Map<const nxr::hodge::RowMat3>(pos.data(), Vg, 3);
  const std::vector<std::int32_t> faces = H.meta.at("faces").get<std::vector<std::int32_t>>();
  b.faces = Eigen::Map<const nxr::hodge::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
  b.keep = H.meta.at("keep").get<std::vector<std::int64_t>>();
  return b;
}
struct WeightsPrepared { ng::Array K, H; std::optional<std::int64_t> modes; };
ng::NodeSpec hodge_weights() {
  ng::NodeSpec n;
  n.name = "hodge.weights";
  n.kind = ng::Kind::array;
  n.inputs = {{"kernel", "a free-orientation inverse kernel on the package's input surface", false}, {"package", "a Hodge-Dirac package (its eigenvectors, as the runner hands them over)", false}};
  n.outputs = {{"weights", "B [2K x the kernel's channels]: rows 0..K each mode's normal-current coefficient a, rows K..2K its flow coefficient b (an operator from the channels)", false}};
  n.parameters = {{"modes", Json(), "the first K modes of the package; null: all", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "solved in double, once a run");
  n.matches = "nxr.inverse.fused fused_weights; compute nxr::hodge::gluedKernel, hodgeWeights";
  n.description = "the fused Hodge-Dirac weights of a free kernel: each mode's normal-current and flow coefficients as linear maps of the channels";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"kernel", "package"}) if (!in.count(k)) ng::refuse(std::string("hodge.weights: ") + k + " is not given");
    ng::Planned out;
    ng::PortType t;
    t.cls = ng::PortClass::operator_;
    t.facts["cls"] = "operator";
    out.out["weights"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    for (const char* k : {"kernel", "package"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("hodge.weights: ") + k + " was not handed over");
    if (!c.at("package").meta.value("hodge", false)) ng::refuse("hodge.weights: package is not a Hodge-Dirac package (an operator of kind eigenvectors, <stem>_eigenvectors)");
    WeightsPrepared w{c.at("kernel"), c.at("package"), std::nullopt};
    if (p.contains("modes") && !p["modes"].is_null()) w.modes = p["modes"].get<std::int64_t>();
    return std::any(std::move(w));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const WeightsPrepared& w = std::any_cast<const WeightsPrepared&>(prepared);
    const ng::Array& KA = w.K;
    const std::vector<std::int32_t> chans = KA.meta.value("channels", std::vector<std::int32_t>());
    const std::int64_t M = static_cast<std::int64_t>(chans.size());
    const std::vector<double> pin = w.H.meta.at("input_positions").get<std::vector<double>>();
    const std::int64_t Vin = static_cast<std::int64_t>(pin.size() / 3);
    if (KA.meta.value("to_value_type", std::string("vector3")) == "scalar" || KA.rows != 3 * Vin || KA.cols != M)
      ng::refuse("hodge.weights: " + KA.meta.value("name", std::string()) + " is not a free-orientation kernel of the package's input surface (" + std::to_string(KA.rows) + " rows for " + std::to_string(Vin) + " sources)");
    const nxr::hodge::Basis basis = basis_of(w.H, w.modes);
    if (basis.mu.size() < 2) ng::refuse("hodge.weights: at least 2 modes");
    const Eigen::MatrixXd W = Eigen::Map<const MatRM>(KA.data.data(), KA.rows, M);
    const Eigen::MatrixXd Wg = nxr::hodge::gluedKernel(W, basis, Eigen::Map<const nxr::hodge::RowMat3>(pin.data(), Vin, 3));
    const Eigen::MatrixXd B = nxr::hodge::hodgeWeights(basis, Wg);
    ng::Array a;
    a.rows = B.rows(); a.cols = B.cols();
    a.data.resize(static_cast<std::size_t>(a.rows * a.cols));
    for (Eigen::Index i = 0; i < B.rows(); ++i) for (Eigen::Index j = 0; j < B.cols(); ++j) a.data[static_cast<std::size_t>(i * a.cols + j)] = B(i, j);
    a.meta["channels"] = chans;
    a.meta["name"] = KA.meta.value("name", std::string()) + "+" + w.H.meta.value("name", std::string());
    a.meta["to_value_type"] = "scalar";
    a.meta["unit"] = "mode weight unit";
    a.meta["modes"] = basis.mu.size();
    a.meta["mu"] = std::vector<double>(basis.mu.data(), basis.mu.data() + basis.mu.size());
    a.meta["kind"] = basis.kind;
    a.meta["axis_id"] = w.H.meta.value("axis_id", std::string());
    return ng::Arrays{{"weights", std::move(a)}};
  };
  return n;
}

}  // namespace

void register_meeg_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {noise_covariance(), minimum_norm(), spherical_head_model(), analytic_bands(), tile_cross_spectrum(), phase_gradient(), channel_mean(), filter_apply(), hodge_weights()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
