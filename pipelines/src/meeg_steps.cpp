// nxr-pipelines — THE M/EEG NODES: the source-imaging steps as pure nodes of nxr::graph::registry() (grouped names `meeg.*`), each the wrapper of the maths of
// meeg/ (nxr::meeg): the same function a test calls directly. A node reads what it is handed (the streamed recording, stored arrays as constants, the facts of
// its ports: a recording's channel types and quality) and describes what it made (`Array::meta["describe"]`); the definition says where it is stored.
//
//   meeg.noise_covariance  accumulate  x [channels x samples]  ->  channels (the good channels of the imaging types, indices), covariance [M x M] (a statistic)
//   meeg.spherical_head_model  array   channels (sensor geometry) + sources + meg/eeg surfaces  ->  gain (an operator), grid (the source positions)
//   meeg.minimum_norm      array       forward (a head model) + covariance (a statistic on a pick)  ->  kernel, channels, covariance, whitener
//   meeg.lcmv              array       forward + data covariance + noise covariance + source surface  ->  kernel (fixed, scalar), channels, whitener, data whitener
//   meeg.region_pca_basis  array       kernel (fixed) + the span's covariance + partition + source surface  ->  basis (each region's component and peak source as channel weights)
//   meeg.region_pca        accumulate  x + basis  ->  map (regions x channels), channels: each component signed by the span's extremes (the offsets in its provenance)
#include "nxr/pipelines/meeg_steps.h"
#include "nxr/pipelines/select_nodes.h"
#include "nxr/selection/selection.h"

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
#include "nxr/select/constant_q.h"
#include "nxr/stats/covariance.h"
#include "nxr/signal/filters.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/signal/speedspectrum.h"
#include "nxr/select/dyadic.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/measurement/measures.h"
#include "nxr/measurement/reduce.h"
#include "nxr/pipelines/spectral_record.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"
#include "nxr/meeg/inverse.h"
#include "nxr/meeg/beamformer.h"
#include "nxr/meeg/scouts.h"
#include "nxr/geometry/frames.h"
#include "nxr/meeg/forward.h"
#include "nxr/meeg/sphere.h"
#include "nxr/bst/inverse.h"
#include "nxr/bst/projectors.h"

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
  if (!want.empty() && !c.contains("types")) ng::refuse(std::string(who) + ": its channels of the types asked cannot be told: the recording's channels have no 'type' labelling");
  const auto& q = c["quality"];
  const std::size_t n = static_cast<std::size_t>(c.value("n_channels", std::int64_t(0)));
  for (std::size_t i = 0; i < n; ++i)
    if ((want.empty() || want.count(c["types"][i].get<std::string>())) && (i >= q.size() || q[i].get<std::string>() == "good")) out.push_back(static_cast<std::int32_t>(i));
  if (out.empty()) ng::refuse(std::string(who) + ": no good channel of the types asked");
  return out;
}

// ── meeg.noise_covariance ───────────────────────────────────────────────────────────────────────────────────────
// THE NOISE COVARIANCE IS A STATISTIC: the zero-lag second moment of the good channels of the imaging types over the span the caller states, [M x M]; the blocks are its
// members (each with its own mean removed when remove_mean), the divisor its degrees of freedom. The whitener derived from it is an operator (made by the inverse).
struct CovPrepared {
  std::vector<std::int32_t> channels;
  nxr::stats::CovarianceOptions opt;
  std::int64_t block = 0;   // samples a block; 0: the span is one block
  double sfreq = 0;
  std::string divisor;
  Json types;
  std::string unit;
};
struct CovState {
  std::optional<nxr::stats::CovarianceEstimate> acc;
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
  n.matches = "MNE compute_raw_covariance (method empirical); Brainstorm's noise covariance (remove DC offset per file); meeg nxr::stats::CovarianceEstimate";
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
    pr.opt.divisor = pr.divisor == "samples" ? nxr::stats::CovarianceDivisor::Samples
                     : pr.divisor == "samples_minus_one" ? nxr::stats::CovarianceDivisor::SamplesMinusOne : nxr::stats::CovarianceDivisor::SamplesMinusBlocks;
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
    const std::int64_t dof = !pr.opt.removeMean || pr.opt.divisor != nxr::stats::CovarianceDivisor::SamplesMinusBlocks
                                 ? (pr.opt.divisor == nxr::stats::CovarianceDivisor::Samples ? N : N - 1) : N - B;
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
  std::optional<Eigen::MatrixXd> right;      // MNE-Python: the projector the kernel ends with (inv['proj']), [M x M]
  std::vector<std::int32_t> channels;
  std::vector<std::string> types;
  nxr::meeg::KernelOptions opt;
  bool averaged = false;
  std::string forward, covariance, conventions;
  Json projectors = Json::object();          // which projectors went where (the provenance)
  Json channels_matched;                     // the covariance's channel set matched to the head model's by name (the preparation's record), null: on the head model's own set
};

// the projectors handed with the covariance (the runner reads its channels' projector Operators, those the caller chose with the run's projectors option: `meta.projectors`)
std::vector<nxr::bst::StoredProjector> projectors_of(const Json& meta) {
  std::vector<nxr::bst::StoredProjector> out;
  if (!meta.contains("projectors") || !meta["projectors"].is_array()) return out;
  for (const auto& j : meta["projectors"]) {
    nxr::bst::StoredProjector p;
    p.name = j.value("name", std::string());
    p.kind = j.value("kind", std::string("ssp"));
    p.method = j.value("method", std::string());
    p.order = j.value("order", 0);
    p.status = j.value("status", 0);
    p.m = j.value("m", std::size_t(0));
    if (j.contains("vectors")) for (const auto& v : j["vectors"]) p.vectors.push_back(v.get<std::vector<double>>());
    if (j.contains("matrix")) p.matrix = j["matrix"].get<std::vector<double>>();
    out.push_back(std::move(p));
  }
  return out;
}
// the projectors a covariance's recording was read with (its run's provenance: producer.run.projectors.applied)
std::vector<std::string> covariance_projectors_of(const Json& meta) {
  std::vector<std::string> out;
  if (meta.contains("producer") && meta["producer"].is_object() && meta["producer"].contains("run") && meta["producer"]["run"].is_object()) {
    const Json& r = meta["producer"]["run"];
    if (r.contains("projectors") && r["projectors"].is_object() && r["projectors"].contains("applied")) out = r["projectors"]["applied"].get<std::vector<std::string>>();
  }
  return out;
}
Eigen::MatrixXd dense(const std::vector<double>& m, std::size_t n) {
  if (m.empty()) return Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
  return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(m.data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
}

}  // namespace

MinNormInputs minimum_norm_inputs(const Eigen::MatrixXd& F, const Eigen::MatrixXd& C, const std::vector<std::int32_t>& pick, const std::vector<std::string>& types,
                                  const std::vector<nxr::bst::StoredProjector>& projectors, const std::vector<std::string>& covariance_projectors, const std::string& conventions,
                                  bool average_reference) {
  const Eigen::Index M = static_cast<Eigen::Index>(pick.size()), m = F.rows();
  if (C.rows() != M || C.cols() != M || static_cast<Eigen::Index>(types.size()) != M) ng::refuse("meeg.minimum_norm: the covariance, its channels and their types do not agree");
  for (const auto& q : projectors) if (static_cast<Eigen::Index>(q.m) != m) ng::refuse("meeg.minimum_norm: the projector " + q.name + " is on " + std::to_string(q.m) + " channels, the gain on " + std::to_string(m));
  MinNormInputs out;
  Json rec = Json::object();
  // the average reference's families among the pick (EEG, sEEG, ECoG with at least two channels)
  std::vector<std::vector<int>> families;
  if (average_reference)
    for (const char* fam : {"EEG", "SEEG", "ECOG"}) {
      std::vector<int> idx;
      for (int i = 0; i < static_cast<int>(M); ++i) if (types[static_cast<std::size_t>(i)] == fam) idx.push_back(i);
      if (idx.size() >= 2) families.push_back(idx);
    }
  out.averaged = !families.empty();
  if (conventions == "mne") {
    // MNE-Python (make_inverse_operator: _prepare_forward, prepare_noise_cov / _smart_eigh, prepare_inverse_operator): the projector is made on the picked channels from
    // EVERY projector of the info (active or not) with each vector cut to the pick and renormalised (_make_projector); the average reference is one of them (an EEG
    // average-reference projector). The covariance is P C P^T with P from the info's projectors and the covariance's own; the gain is used as it is (the whitener's rows lie in
    // the projected space, and the depth prior is computed on the gain as it is); the kernel ends with the info's projector (inv['proj']).
    std::vector<nxr::bst::StoredProjector> info = projectors;
    int order = 0;
    for (const auto& q : info) order = std::max(order, q.order);
    for (const auto& idx : families) {
      nxr::bst::StoredProjector a;
      a.name = "average reference (" + types[static_cast<std::size_t>(idx.front())] + ")";
      a.kind = "ssp"; a.status = 1; a.order = ++order; a.m = static_cast<std::size_t>(m);
      std::vector<double> v(static_cast<std::size_t>(m), 0.0);
      for (int i : idx) v[static_cast<std::size_t>(pick[static_cast<std::size_t>(i)])] = 1.0;
      a.vectors.push_back(v);
      info.push_back(std::move(a));
    }
    std::vector<nxr::bst::StoredProjector> with_cov = info;
    for (const auto& name : covariance_projectors)
      for (const auto& q : projectors) if (q.name == name) { nxr::bst::StoredProjector c = q; c.order = ++order; with_cov.push_back(std::move(c)); }
    nxr::bst::ProjectorRule rule;
    rule.statuses = {0, 1, 2};
    for (const auto c : pick) rule.channels.push_back(static_cast<std::size_t>(c));
    rule.renormalise = true;
    rule.zero_tiny = false;
    std::vector<std::string> used_info, used_cov;
    const Eigen::MatrixXd Pk = dense(nxr::bst::joint_projector(info, rule, &used_info), static_cast<std::size_t>(M));
    const Eigen::MatrixXd Pc = dense(nxr::bst::joint_projector(with_cov, rule, &used_cov), static_cast<std::size_t>(M));
    out.G.resize(M, F.cols());
    for (Eigen::Index i = 0; i < M; ++i) out.G.row(i) = F.row(pick[static_cast<std::size_t>(i)]);
    out.C = Pc * C * Pc.transpose();
    if (!used_info.empty()) out.right = Pk;
    out.info_projectors = info;
    rec = Json{{"rule", "MNE-Python make_inverse_operator: every projector of the info (and the covariance's) on the picked channels, each vector renormalised; P C P^T; the kernel ends with P"},
               {"covariance", used_cov}, {"kernel", used_info}};
  } else {
    // Brainstorm (process_inverse_2018): the channel file's projectors of status 1 and 2 (BuildProjector(Projector, [1 2])) on ALL channels, applied to the gain's rows that
    // hold no NaN and to both sides of the noise covariance; then the good channels; then the average reference per family, on the gain and both sides of the covariance.
    nxr::bst::ProjectorRule rule;
    rule.statuses = {1, 2};
    std::vector<std::string> used;
    const std::vector<double> pv = nxr::bst::joint_projector(projectors, rule, &used);
    std::vector<Eigen::Index> ok;
    for (Eigen::Index r = 0; r < m; ++r) if (!F.row(r).hasNaN()) ok.push_back(r);
    out.G.resize(M, F.cols());
    if (pv.empty()) {
      for (Eigen::Index i = 0; i < M; ++i) out.G.row(i) = F.row(pick[static_cast<std::size_t>(i)]);
      out.C = C;
    } else {
      const Eigen::MatrixXd P = dense(pv, static_cast<std::size_t>(m));
      Eigen::MatrixXd Pp(M, M);
      for (Eigen::Index i = 0; i < M; ++i) {
        const Eigen::Index r = pick[static_cast<std::size_t>(i)];
        if (F.row(r).hasNaN()) { out.G.row(i) = F.row(r); }
        else { out.G.row(i).setZero(); for (const Eigen::Index l : ok) if (P(r, l) != 0.0) out.G.row(i) += P(r, l) * F.row(l); }
        for (Eigen::Index j = 0; j < M; ++j) Pp(i, j) = P(r, pick[static_cast<std::size_t>(j)]);
      }
      out.C = Pp * C * Pp.transpose();
    }
    if (out.averaged) {
      Eigen::MatrixXd A = Eigen::MatrixXd::Identity(M, M);
      for (const auto& idx : families) for (int a : idx) for (int b : idx) A(a, b) -= 1.0 / static_cast<double>(idx.size());
      out.G = A * out.G;
      out.C = A * out.C * A.transpose();
    }
    rec = Json{{"rule", "Brainstorm process_inverse_2018: the projectors of status 1 and 2 on all channels, applied to the gain and both sides of the noise covariance"}, {"applied", used}};
  }
  out.record = rec;
  return out;
}

namespace {

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
      ng::refuse("meeg.minimum_norm: " + pr.covariance + " is on other channels than " + pr.forward + " (a covariance of another channel set is matched to the head model's by name when it is prepared)");
    if (F.cols % 3 != 0) ng::refuse("meeg.minimum_norm: " + pr.forward + ": the gain is not channels x 3N");
    pr.channels = S.meta.value("channels", std::vector<std::int32_t>());
    pr.types = S.meta.value("channel_types", std::vector<std::string>());
    const std::int64_t M = static_cast<std::int64_t>(pr.channels.size());
    if (M == 0 || S.rows != M || S.cols != M) ng::refuse("meeg.minimum_norm: " + pr.covariance + ": its array does not fit its channels");
    const Eigen::MatrixXd C = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(S.data.data(), M, M);
    for (const std::int32_t ch : pr.channels) if (ch < 0 || ch >= F.rows) ng::refuse("meeg.minimum_norm: a channel of " + pr.covariance + " is outside the gain's channels");
    const Eigen::MatrixXd Fall = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(F.data.data(), F.rows, F.cols);
    pr.opt = kernel_options(p);
    pr.conventions = p.value("conventions", std::string("brainstorm"));
    const std::vector<std::string> cov_projs = covariance_projectors_of(S.meta);
    const MinNormInputs mi = minimum_norm_inputs(Fall, C, pr.channels, pr.types, projectors_of(S.meta), cov_projs, pr.conventions, p.value("average_reference", true));
    pr.G = mi.G; pr.C = mi.C; pr.right = mi.right; pr.averaged = mi.averaged; pr.projectors = mi.record;
    if (S.meta.contains("projectors_choice")) pr.projectors["choice"] = S.meta["projectors_choice"];
    if (S.meta.contains("channels_matched")) pr.channels_matched = S.meta["channels_matched"];   // a covariance of another channel set, matched to the head model's by name   // which projectors the caller had handed (the run's --projectors), for the provenance
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const MinNormPrepared& pr = std::any_cast<const MinNormPrepared&>(prepared);
    nxr::meeg::Kernel ker = nxr::meeg::unconstrainedKernel(pr.G, pr.C, pr.types, pr.opt);
    if (pr.right) ker.K = ker.K * *pr.right;                            // MNE-Python's kernel ends with the projector (trans = eigen_fields . whitener . proj)
    const std::int64_t M = static_cast<std::int64_t>(pr.channels.size()), R = ker.K.rows(), N = R / 3;
    const Json options = kernel_options_record(pr.opt);
    const auto& w = pr.opt.whitener;
    // the producer record in the importer's keys (nxr::bst::inverse_method_of reads it the same), plus what a computed kernel adds
    Json producer = Json{{"method", "MNE"}, {"method_comment", "MN"}, {"units", "A.m"}, {"n_channels", M}, {"n_sources", R}, {"n_components", 3},
                         {"source_ori", "free"}, {"inverse_method", "minnorm"}, {"inverse_measure", "amplitude"}, {"source_orient", Json::array({"free"})},
                         {"use_depth", pr.opt.depth.enabled}, {"noise_method", w.regularisation == nxr::meeg::Regularisation::Ridge ? "reg" : w.regularisation == nxr::meeg::Regularisation::Diagonal ? "diag" : "none"},
                         {"snr_method", "fixed"}, {"snr_fixed", pr.opt.snr}, {"whitener_rank", ker.whitenerRank}, {"lambda", ker.lambda},
                         {"computed", Json{{"options", options}, {"conventions", pr.conventions}, {"average_reference", pr.averaged}, {"projectors", pr.projectors}, {"forward", pr.forward}, {"noise_cov", pr.covariance}, {"notes", ker.notes}}}};
    if (!pr.channels_matched.is_null()) producer["computed"]["noise_cov_channels"] = pr.channels_matched;
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
    } else {
      ng::Array none; none.meta["absent"] = true;                          // a PCA whitener is not stored: an optional output this run does not make
      out["whitener"] = std::move(none);
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
Json minimiser_record(const nxr::optimisation::NelderMeadOptions& m) {
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
  nxr::signal::timefilterbank::Design design{nxr::signal::timefilterbank::Family::Meyer, 2.0, nxr::signal::timefilterbank::Sides::Analytic};
  std::vector<std::int32_t> channels;
  double sfreq = 0;
};
std::int64_t analytic_halo(const Json& p, double sfreq) {
  nxr::signal::timefilterbank::Design d(nxr::signal::timefilterbank::Family::Meyer, 2.0, nxr::signal::timefilterbank::Sides::Analytic);
  for (const auto& b : p.value("bands", Json::array())) d.bands.push_back({b.value("lo", b.value("lo_hz", 0.0)), b.value("hi", b.value("hi_hz", 0.0))});
  const nxr::signal::timefilterbank::TimeFilterbank bank = nxr::signal::timefilterbank::timeBank(d, std::size_t(1) << 15, sfreq);
  std::size_t H = 0;
  for (int m = 0; m < static_cast<int>(d.bands.size()); ++m) H = std::max(H, nxr::signal::timefilterbank::haloOfTaps(bank, m));
  return static_cast<std::int64_t>(H);
}
ng::NodeSpec analytic_bands() {
  ng::NodeSpec n;
  n.name = "signal.analytic_bands";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"channels_of", "an operator whose channels are the ones kept, in order; absent: the good channels of channel_types", true}};
  n.outputs = {{"z", "the analytic signal of each band on the channels kept: bands x channels rows (band-major), complex, at the recording's rate", false}};
  n.parameters = {{"bands", Json::array(), "the bands: a list of {name, lo, hi} in Hz (a Meyer band each, gain 2)", "Hz", {}, false},
                  {"channel_types", Json::array({"MEG", "MEG MAG", "MEG GRAD", "EEG", "SEEG", "ECOG"}), "the channel types kept when channels_of is absent (good channels only)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input, "the block is transformed in double; the output block follows the input (float32 blocks)",
                             "the halo is the length of the narrowest band's response (nxr::signal::timefilterbank::haloOf), samples on each side");
  n.matches = "the Studio's Signal chain meyerBand (toolbox/wavelets/dyadicFrame.ts); compute nxr::signal::wtBlock";
  n.description = "the analytic signal of each band (Meyer bands), with the halo the narrowest band needs";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed(in, "x", "signal.analytic_bands");
    const Json bands = p.value("bands", Json::array());
    if (!bands.is_array() || bands.empty()) ng::refuse("signal.analytic_bands: at least one band");
    for (const auto& b : bands) {
      const double lo = b.value("lo", b.value("lo_hz", 0.0)), hi = b.value("hi", b.value("hi_hz", 0.0));
      if (!(hi > lo && lo > 0)) ng::refuse("signal.analytic_bands: band " + b.value("name", std::string()) + " needs 0 < lo < hi");
      if (hi >= *x.sfreq / 2) ng::refuse("signal.analytic_bands: band " + b.value("name", std::string()) + " reaches past Nyquist");
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
    a.channels = channels_picked(in.at("x"), constants, p, "signal.analytic_bands");
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
    const auto t = nxr::signal::wtBlock(a.design, block, a.sfreq, static_cast<std::size_t>(left), static_cast<std::size_t>(own));
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

// (analytic.tile_cross_spectrum is retired: `reduce` of the `outer` form of field.form over a time grid of the band, the preset analytic_cross_spectrum_tiles of the definition measure)

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
struct PhasePrepared { ng::Arrays c; Json band; std::string signal; Json finest; };
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
                  {"signal", Json(), "normal = the current along the surface's normal | ambient = each of its x y z components. NO DEFAULT (an orientation prior is never a default): the caller states it", "-", {"normal", "ambient"}, false},
                  {"finest", Json(), "the leaves measured, as a quantity (a tile diameter, an area, parts per hemisphere), resolved on the grid's measured level diameters; null: the grid's own depth", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "in double from a stored cross-spectrum (no time axis)");
  n.matches = "nxr.detect.core_phase PhaseReader.read_out (grad phi = Im(conj z grad z) / |z|^2), power-weighted; meeg nxr::signal::phaseSums";
  n.description = "a band's phase gradient from its cross-spectrum through a free kernel, as the additive ingredients of the surface's leaves";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"cross_spectrum", "kernel", "surface", "frames", "leaves"}) if (!in.count(k)) ng::refuse(std::string("meeg.phase_gradient: ") + k + " is not given");
    if (!p.contains("signal") || !p["signal"].is_string() || (p["signal"] != "normal" && p["signal"] != "ambient"))
      ng::refuse("meeg.phase_gradient: signal is normal (the current along the surface's normal) or ambient (its x y z components): it has no default, the caller states it (an orientation prior is never a default)");
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
    if (!p.contains("signal") || !p["signal"].is_string() || (p["signal"] != "normal" && p["signal"] != "ambient"))
      ng::refuse("meeg.phase_gradient: signal is normal or ambient (no default: the caller states it)");
    pr.signal = p["signal"].get<std::string>();
    pr.finest = p.contains("finest") ? p["finest"] : Json();
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
    const nxr::operators::RowMat3 P = Eigen::Map<const nxr::operators::RowMat3>(SA.data.data(), N, 3);
    const nxr::operators::FaceMat F = Eigen::Map<const nxr::operators::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
    const nxr::operators::FaceOperators ops(P, F, false);
    const Eigen::MatrixXd K = Eigen::Map<const MatRM>(KA.data.data(), KA.rows, M);
    nxr::signal::PhaseSums sums;
    auto add_signal = [&](const Eigen::MatrixXd& A) {
      Eigen::MatrixXd G[3];
      ops.meanFaceGradient(A, G);
      for (std::int64_t v = 0; v < N; ++v) {
        const Eigen::RowVectorXd dot = G[0].row(v) * ops.normals(v, 0) + G[1].row(v) * ops.normals(v, 1) + G[2].row(v) * ops.normals(v, 2);
        for (int d = 0; d < 3; ++d) G[d].row(v) -= dot * ops.normals(v, d);
      }
      const nxr::signal::PhaseSums one = nxr::signal::phaseSums(A, G, C.S, C.Sq);
      if (sums.power.size() == 0) sums = one; else nxr::signal::addPhaseSums(sums, one);
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
    const nxr::signal::PhaseCurrentFrames jf = nxr::signal::phaseCurrentInFrames(sums.phase, e1, e2);
    nxr::select::Mesh mesh; mesh.n = static_cast<int>(N); mesh.positions = SA.data; mesh.faces = faces;
    const std::vector<double> area = nxr::select::vertexAreas(mesh);
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
    if (static_cast<std::int64_t>(codes.size()) != N) ng::refuse("meeg.phase_gradient: the leaves are not a partition of the surface's vertices (" + std::to_string(codes.size()) + " codes for " + std::to_string(N) + " vertices)");
    Json finest_snapped = nullptr;
    int at_level = LA.meta.value("depth", -1);                              // the level of the grid the sums are at (the finest, or the one `finest` resolves to)
    int L = LA.meta.contains("n_members") ? LA.meta["n_members"].get<int>() : (codes.empty() ? 0 : *std::max_element(codes.begin(), codes.end()) + 1);
    if (pr.finest.is_string()) {
      // `finest` is a QUANTITY resolved on the grid's measured level dictionary (compute's level_of_*, via the one resolver); the leaves are that level's tiles: the finest codes shifted (closed form)
      const Json levels = LA.meta.value("levels", Json::array());
      const int depth = LA.meta.value("depth", -1);
      if (!levels.is_array() || levels.empty() || depth < 0 || static_cast<int>(levels.size()) != depth + 1)
        ng::refuse("meeg.phase_gradient: finest " + pr.finest.dump() + " needs the grid's level dictionary (depth and each level's measured diameter); the leaves handed over record none");
      nxr::selection::Selection grid;
      grid.type = nxr::selection::Type::set;
      for (const auto& lj : levels) {
        nxr::selection::Selection lv;
        if (lj.contains("diameter")) lv.diameter = lj["diameter"].get<double>();
        if (lj.contains("lambda_min")) lv.band = std::make_pair(lj["lambda_min"].get<double>(), lj["lambda_max"].get<double>());
        grid.members.push_back(std::move(lv));
      }
      const int level = nxr::pipelines::graph::level_of_quantity(grid, pr.finest, unit, "meeg.phase_gradient", finest_snapped);
      if (level < 0 || level > depth) ng::refuse("meeg.phase_gradient: finest " + pr.finest.dump() + " is level " + std::to_string(level) + "; the grid has levels 0 .. " + std::to_string(depth));
      for (auto& c : codes) if (c >= 0) c >>= (depth - level);
      at_level = level;
      const std::int64_t slots = LA.meta.value("slots", std::int64_t(1));
      L = static_cast<int>(slots << level);
    }
    const std::vector<std::vector<double>> cols = {vp, vq, v1, v2, vm};
    ng::Array Mz;
    Mz.rows = L; Mz.cols = static_cast<std::int64_t>(cols.size());
    Mz.data.assign(static_cast<std::size_t>(L) * cols.size(), 0.0);
    {
      // the leaves' sums: Σ a_v x_v over each leaf's vertices (the weights a_v the barycentric areas), the ONE reduction (nxr::measurement::reduce) over the leaves as the members of a surface grid
      nxr::selection::MemberOfElement leaves;
      leaves.member_of.assign(codes.begin(), codes.end());
      leaves.members = L;
      nxr::measurement::Request req;
      req.measures = {"sum"};
      req.weighting = nxr::measurement::Weighting::given;
      for (std::size_t c = 0; c < cols.size(); ++c) {
        nxr::measurement::Ingredients ing(1, L);
        nxr::measurement::reduce(ing, cols[c].data(), 1, static_cast<std::int64_t>(codes.size()), leaves, wv.data(), req);   // one row (the sums), the vertices the elements
        for (int l = 0; l < L; ++l) Mz.data[static_cast<std::size_t>(l) * cols.size() + c] = nxr::measurement::value(ing, 0, l, "sum");
      }
    }
    const std::string per_hz = C.density ? "/Hz" : "";
    const double omega = M_PI * (C.lo + C.hi);
    Mz.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "phase_power"}, {"unit", std::string("A\xc2\xb2\xc2\xb7m\xc2\xb2") + per_hz}},
                                                         Json{{"measure", "phase_gradient_power"}, {"unit", std::string("A\xc2\xb2") + per_hz}},
                                                         Json{{"measure", "phase_current_1"}, {"unit", "rad/m x power"}}, Json{{"measure", "phase_current_2"}, {"unit", "rad/m x power"}},
                                                         Json{{"measure", "phase_current_magnitude"}, {"unit", "rad/m x power"}}})},
                               {"band", Json{{"name", C.name}, {"lo_hz", C.lo}, {"hi_hz", C.hi}}}};
    if (at_level >= 0) Mz.meta["describe"]["level"] = at_level;             // the rows are one LEVEL of the grid: its members are that level's tiles
    ng::Array Fd;
    Fd.meta["describe"] = Json{{"no_array", true}, {"value_type", "vector3"},
                               {"description", "the phase gradient of the " + pr.signal + " current by " + KA.meta.value("name", std::string()) + " \xe2\x80\x94 " + C.name +
                                               ", stored as its additive ingredients per leaf of " + LA.meta.value("name", std::string())},
                               {"producer", Json{{"band", Json{{"name", C.name}, {"lo_hz", C.lo}, {"hi_hz", C.hi}}}, {"omega_rad_s", omega}, {"signal", pr.signal}, {"surface_unit", unit},
                                                 {"leaves", Json{{"selection", LA.meta.value("name", std::string())}, {"count", L}, {"finest", pr.finest}, {"snapped", finest_snapped}, {"vertices_without_frame", singular}, {"weights", "the vertex's lumped area"}}},
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
nxr::signal::filters::FilterOptions filter_options(const Json& p, double sfreq) {
  using namespace nxr::signal::filters;
  FilterOptions o = brainstormFilterOptions(sfreq);
  const std::string kind = p.value("kind", std::string());
  o.kind = kind == "lowpass" ? Kind::Lowpass : kind == "highpass" ? Kind::Highpass : kind == "bandpass" ? Kind::Bandpass : kind == "bandstop" ? Kind::Bandstop : kind == "notch" ? Kind::Notch
           : (ng::refuse("filter: kind is lowpass, highpass, bandpass, bandstop or notch"), Kind::Lowpass);
  auto num = [&](const char* k) -> std::optional<double> { return p.contains(k) && !p[k].is_null() ? std::optional<double>(p[k].get<double>()) : std::nullopt; };
  if (const auto v = num("lower")) o.lowerEdge = *v;
  if (const auto v = num("upper")) o.upperEdge = *v;
  if (p.contains("frequencies") && p["frequencies"].is_array()) o.frequencies = p["frequencies"].get<std::vector<double>>();
  else if (p.contains("frequencies") && p["frequencies"].is_number()) o.frequencies = {p["frequencies"].get<double>()};   // one frequency, as `--frequencies 50`
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

// (filter.channel_mean is retired: `reduce` measure mean over the run's span, the preset channel_span_mean of the definition measure)

struct FilterPrepared { nxr::signal::filters::FilterOptions options; nxr::signal::filters::Design design; Eigen::VectorXd mean; std::int64_t halo = 0; };
ng::NodeSpec filter_apply() {
  ng::NodeSpec n;
  n.name = "signal.filter";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false},
              {"mean", "the record's channel means (filter.channel_mean): needed by a FIR filter that removes the mean", true}};
  n.outputs = {{"y", "the filtered recording, channels x samples", false}};
  n.parameters = filter_parameters();
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input, "each block filtered in double; the output follows the input (float32 stored for float32 input)",
                             "the halo is the design's look-behind and look-ahead (the FIR's half length; the IIR's decayed response at `tolerance`), samples on each side");
  n.matches = "Brainstorm's process_bandpass / process_notch (bst_bandpass_hfilter, kaiserord+fir1; filtfilt IIR); compute nxr::signal::filter";
  n.description = "a zero-phase filter (lowpass, highpass, bandpass, bandstop, notch) on each block, with the halo its design needs";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed(in, "x", "signal.filter");
    nxr::signal::filters::Design d;
    try { d = nxr::signal::filters::design(filter_options(p, *x.sfreq)); } catch (const std::invalid_argument& e) { ng::refuse(std::string("signal.filter: ") + e.what()); }
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
    const nxr::signal::filters::FilterOptions o = filter_options(p, *x.sfreq);
    const nxr::signal::filters::Design d = nxr::signal::filters::design(o);
    Eigen::VectorXd mean;
    if (d.method == nxr::signal::filters::Method::Fir && o.removeMean) {
      const auto it = c.find("mean");
      if (it == c.end() || it->second.meta.value("placeholder", false))
        ng::refuse("signal.filter: a FIR filter that removes the record's mean needs the channel means (`nxr-neuro run channel_mean` first), or remove_mean false");
      mean = Eigen::Map<const Eigen::VectorXd>(it->second.data.data(), static_cast<Eigen::Index>(it->second.data.size()));
    }
    pr.options = o;
    pr.design = d;
    pr.mean = mean;
    pr.halo = static_cast<std::int64_t>(std::max(d.lookBehind, d.lookAhead));
    return std::any(std::move(pr));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place& at) {
    const FilterPrepared& pr = std::any_cast<const FilterPrepared&>(prepared);
    const ng::Block& b = in.at("x");
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    const bool at_start = at.first - b.left == 0 && b.left < pr.halo + keep, at_end = b.right < pr.halo + keep;
    const std::int64_t left = b.left - keep, own = b.own() + 2 * keep;
    const Eigen::MatrixXd yb = nxr::signal::filters::filter(all.cast<double>(), pr.options, pr.design, pr.mean, static_cast<std::size_t>(left), static_cast<std::size_t>(own), at_start, at_end);
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
nxr::operators::Basis basis_of(const ng::Array& H, std::optional<std::int64_t> modes) {
  nxr::operators::Basis b;
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
  b.positions = Eigen::Map<const nxr::operators::RowMat3>(pos.data(), Vg, 3);
  const std::vector<std::int32_t> faces = H.meta.at("faces").get<std::vector<std::int32_t>>();
  b.faces = Eigen::Map<const nxr::operators::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
  b.keep = H.meta.at("keep").get<std::vector<std::int64_t>>();
  return b;
}
struct WeightsPrepared { ng::Array K, H; std::optional<std::int64_t> modes; };
ng::NodeSpec hodge_weights() {
  ng::NodeSpec n;
  n.name = "operators.hodge_weights";
  n.kind = ng::Kind::array;
  n.inputs = {{"kernel", "a free-orientation inverse kernel on the package's input surface", false}, {"package", "a Hodge-Dirac package (its eigenvectors, as the runner hands them over)", false}};
  n.outputs = {{"weights", "B [2K x the kernel's channels]: rows 0..K each mode's normal-current coefficient a, rows K..2K its flow coefficient b (an operator from the channels)", false}};
  n.parameters = {{"modes", Json(), "the first K modes of the package; null: all", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "solved in double, once a run");
  n.matches = "nxr.inverse.fused fused_weights; compute nxr::operators::gluedKernel, hodgeWeights";
  n.description = "the fused Hodge-Dirac weights of a free kernel: each mode's normal-current and flow coefficients as linear maps of the channels";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"kernel", "package"}) if (!in.count(k)) ng::refuse(std::string("operators.hodge_weights: ") + k + " is not given");
    ng::Planned out;
    ng::PortType t;
    t.cls = ng::PortClass::operator_;
    t.facts["cls"] = "operator";
    out.out["weights"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    for (const char* k : {"kernel", "package"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("operators.hodge_weights: ") + k + " was not handed over");
    if (!c.at("package").meta.value("hodge", false)) ng::refuse("operators.hodge_weights: package is not a Hodge-Dirac package (an operator of kind eigenvectors, <stem>_eigenvectors)");
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
      ng::refuse("operators.hodge_weights: " + KA.meta.value("name", std::string()) + " is not a free-orientation kernel of the package's input surface (" + std::to_string(KA.rows) + " rows for " + std::to_string(Vin) + " sources)");
    const nxr::operators::Basis basis = basis_of(w.H, w.modes);
    if (basis.mu.size() < 2) ng::refuse("operators.hodge_weights: at least 2 modes");
    const Eigen::MatrixXd W = Eigen::Map<const MatRM>(KA.data.data(), KA.rows, M);
    const Eigen::MatrixXd Wg = nxr::operators::gluedKernel(W, basis, Eigen::Map<const nxr::operators::RowMat3>(pin.data(), Vin, 3));
    const Eigen::MatrixXd B = nxr::operators::hodgeWeights(basis, Wg);
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


// ── hodge.speed_spectrum ────────────────────────────────────────────────────────────────────────────────────────
// THE SPEED SPECTRUM (the hand-wired `speedspectrum` it replaces): the modes' cross-spectra (bin by bin) read through a joint bank, the separable product of a graph bank over
// lambda = mu^2 and a time bank over f: per tile (graph member x time member) its energy, its speed (a wave of phase speed c lies on 2 pi f = c mu) and its quadrature fraction,
// and the energy's histogram over log-spaced speed bins; exact (divergence) and coexact (rotation) modes apart. Four statistics, kind x graph member x time member (speed bin).
struct SpeedPrepared { ng::Array spectra, weights; Json p; };
ng::NodeSpec speed_spectrum() {
  ng::NodeSpec n;
  n.name = "signal.speed_spectrum";
  n.kind = ng::Kind::array;
  n.inputs = {{"spectra", "the modes' cross-spectra bin by bin (signal.mode_cross_spectra on the flow weights)", false}, {"weights", "the fused Hodge-Dirac weights (their modes' mu and kind)", false}};
  n.outputs = {{"energy", "per kind and tile, the energy", false}, {"tile_speed", "per kind and tile, its speed", false},
               {"quadrature", "per kind and tile, its quadrature fraction", false}, {"histogram", "per kind and graph member, the energy over the speed bins", false}};
  n.parameters = {
      {"graph_family", Json(), "the graph bank over lambda = mu^2: mexhat | heat | itersine | logitersine (no default: the caller names it)", "-", {"mexhat", "heat", "itersine", "logitersine"}, false},
      {"graph_voices", Json(3), "graph members per octave", "-", {}, false},
      {"graph_filters", Json(0), "graph member count; 0: from the voices", "-", {}, false},
      {"family", Json(), "the time bank: itersine | meyer | morlet | morse | amor | bump (no default)", "-", {"itersine", "meyer", "morlet", "morse", "amor", "bump"}, false},
      {"gain", Json(), "the time bank's peak in-band gain (no default)", "-", {}, false},
      {"sides", Json(), "analytic | two_sided (no default)", "-", {"analytic", "two_sided"}, false},
      {"centres", Json::array(), "Morlet, Morse, Amor, Bump: the members' centres", "Hz", {}, false},
      {"cycles", Json(6.0), "Morlet: the width as centre / cycles", "-", {}, false},
      {"voices", Json(4), "Itersine: voices per octave", "-", {}, false},
      {"anchor", Json(1.0), "Itersine: the anchor frequency", "Hz", {}, false},
      {"limits", Json(), "Itersine, Meyer: the bank's frequency limits [lo, hi]", "Hz", {}, true},
      {"half_width", Json(1.0 / 3.0), "Meyer: the transition half-width", "octaves", {}, false},
      {"gamma", Json(3.0), "Morse: gamma", "-", {}, false},
      {"time_bandwidth", Json(60.0), "Morse: the time-bandwidth product", "-", {}, false},
      {"speed_bins", Json(40), "log-spaced speed bins", "-", {}, false},
      {"speed_min", Json(0.0), "the lowest speed; 0: from the spectrum", "length/s", {}, false},
      {"speed_max", Json(0.0), "the highest speed; 0: from the spectrum", "length/s", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "from the accumulated cross-spectra, in double");
  n.matches = "nxr.spectral speed spectrum; compute nxr::signal::speedspectrum::compute";
  n.description = "the speed spectrum of the Hodge-Dirac modes: a joint (wavenumber x frequency) bank's tiles, their energy, speed and quadrature fraction";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "weights"}) if (!in.count(k)) ng::refuse(std::string("signal.speed_spectrum: ") + k + " is not given");
    for (const char* k : {"graph_family", "family", "gain", "sides"}) if (!p.contains(k) || p[k].is_null()) ng::refuse(std::string("signal.speed_spectrum: ") + k + " has no default: name it");
    ng::Planned out;
    for (const char* o : {"energy", "tile_speed", "quadrature", "histogram"}) out.out[o] = constant_port("statistic");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    for (const char* k : {"spectra", "weights"}) if (!c.count(k) || c.at(k).meta.value("placeholder", false)) ng::refuse(std::string("signal.speed_spectrum: ") + k + " was not handed over");
    return std::any(SpeedPrepared{c.at("spectra"), c.at("weights"), p});
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    namespace tb = nxr::signal::timefilterbank;
    namespace fb = nxr::signal::graphfilterbank;
    const SpeedPrepared& sp = std::any_cast<const SpeedPrepared&>(prepared);
    const Json& p = sp.p;
    const std::vector<double> mu_all = sp.weights.meta.at("mu").get<std::vector<double>>();
    const std::vector<int> kind = sp.weights.meta.at("kind").get<std::vector<int>>();
    const std::int64_t K = sp.spectra.meta.value("rows", std::int64_t(0));
    if (K != static_cast<std::int64_t>(mu_all.size())) ng::refuse("signal.speed_spectrum: the spectra are of " + std::to_string(K) + " rows, the weights of " + std::to_string(mu_all.size()) + " modes (use the flow rows)");
    nxr::signal::speedspectrum::ModeSpectra ms;
    const std::vector<double> hz = sp.spectra.meta.at("frequencies").get<std::vector<double>>();
    ms.frequencies = Eigen::Map<const Eigen::VectorXd>(hz.data(), static_cast<Eigen::Index>(hz.size()));
    for (std::size_t g = 0; g < hz.size(); ++g) {
      Eigen::MatrixXd S(K, K), Q(K, K);
      for (std::int64_t i = 0; i < K; ++i) for (std::int64_t j = 0; j < K; ++j) {
        const std::size_t at = static_cast<std::size_t>(((static_cast<std::int64_t>(g) * K + i) * K + j) * 2);
        S(i, j) = sp.spectra.data[at]; Q(i, j) = sp.spectra.data[at + 1];
      }
      ms.S.push_back(std::move(S)); ms.Sq.push_back(std::move(Q));
    }
    const Eigen::VectorXd mu = Eigen::Map<const Eigen::VectorXd>(mu_all.data(), K);
    const std::string gf = p["graph_family"].get<std::string>(), tf = p["family"].get<std::string>();
    const fb::Family gfam = gf == "mexhat" ? fb::Family::MexHat : gf == "heat" ? fb::Family::Heat : gf == "itersine" ? fb::Family::Itersine : fb::Family::LogItersine;
    const tb::Family tfam = tf == "itersine" ? tb::Family::Itersine : tf == "meyer" ? tb::Family::Meyer : tf == "morlet" ? tb::Family::Morlet : tf == "morse" ? tb::Family::Morse : tf == "amor" ? tb::Family::Amor : tb::Family::Bump;
    tb::Design d(tfam, p["gain"].get<double>(), p["sides"].get<std::string>() == "analytic" ? tb::Sides::Analytic : tb::Sides::TwoSided);
    d.voices = p.value("voices", 4); d.anchor = p.value("anchor", 1.0); d.halfWidth = p.value("half_width", 1.0 / 3.0); d.cycles = p.value("cycles", 6.0);
    d.gamma = p.value("gamma", 3.0); d.timeBandwidth = p.value("time_bandwidth", 60.0); d.centres = p.value("centres", std::vector<double>());
    if (p.contains("limits") && p["limits"].is_array() && p["limits"].size() == 2) d.limits = std::make_pair(p["limits"][0].get<double>(), p["limits"][1].get<double>());
    const Json prod = sp.spectra.meta.value("producer", Json::object());
    const std::size_t nfft = static_cast<std::size_t>(prod.value("segment_samples", std::int64_t(0)));
    const double fs = prod.value("sfreq", 0.0);
    if (nfft < 2 || !(fs > 0)) ng::refuse("signal.speed_spectrum: the spectra do not record their segment and rate");
    fb::DesignOptions dop; dop.voices = p.value("graph_voices", 3); dop.numFilters = p.value("graph_filters", 0);
    const fb::GraphFilterbank graph = fb::design(gfam, fb::Range::fromSpectrum(mu.array().square().matrix()), dop);
    tb::TimeFilterbank time;
    try { time = tb::timeBank(d, nfft, fs); } catch (const std::invalid_argument& e) { ng::refuse(std::string("signal.speed_spectrum: the time bank: ") + e.what()); }
    nxr::signal::speedspectrum::Options so; so.speedBins = p.value("speed_bins", 40); so.speedMin = p.value("speed_min", 0.0); so.speedMax = p.value("speed_max", 0.0);
    const nxr::signal::speedspectrum::Result R = nxr::signal::speedspectrum::compute(graph, time, mu, std::vector<int>(kind.begin(), kind.begin() + K), ms, so);
    const std::int64_t P = static_cast<std::int64_t>(R.parts.size()), G = R.graphCentres.size(), T = R.timeCentres.size(), NB = R.speedCentres.size();
    auto vec = [](const Eigen::VectorXd& v) { return Json(std::vector<double>(v.data(), v.data() + v.size())); };
    Json kinds = Json::array();
    for (const auto& pt : R.parts) kinds.push_back(Json{{"name", pt.kind == 0 ? "exact" : "coexact"}, {"modes", pt.modes}, {"power", pt.power}});
    const Json producer = Json{{"graph_centre_wavenumbers", vec(R.graphCentres)}, {"time_centre_frequencies_hz", vec(R.timeCentres)}, {"speed_edges", vec(R.speedEdges)},
                               {"speed_centres", vec(R.speedCentres)}, {"spectra", prod},
                               {"fused", "nxr-compute hodgeWeights: B [2K x C] = [a; b]; the flow weights b feed CrossWelch (one-bin groups, Re and Im)"}};
    auto stat = [&](const char* part, const char* measure, const Json& unit, bool hist) {
      const std::int64_t C = hist ? NB : T;
      ng::Array a;
      a.rows = P; a.cols = G * C;
      a.data.resize(static_cast<std::size_t>(P * G * C));
      for (std::int64_t k = 0; k < P; ++k) {
        const auto& pt = R.parts[static_cast<std::size_t>(k)];
        const Eigen::MatrixXd& M = std::string(part) == "energy" ? pt.tileEnergy : std::string(part) == "tile_speed" ? pt.tileSpeed : std::string(part) == "quadrature" ? pt.quadrature : pt.histogram;
        for (std::int64_t g = 0; g < G; ++g) for (std::int64_t c = 0; c < C; ++c) a.data[static_cast<std::size_t>((k * G + g) * C + c)] = M(g, c);
      }
      Json pj = producer; pj["part"] = part;
      a.meta["describe"] = Json{{"measure", measure}, {"method", "welch"}, {"precision", "float32"}, {"unit", unit},
                                {"axes", Json::array({Json{{"name", "kind"}, {"size", P}, {"values", kinds}}, Json{{"name", "graph_member"}, {"size", G}, {"values", vec(R.graphCentres)}},
                                                      Json{{"name", hist ? "speed_bin" : "time_member"}, {"size", C}, {"values", hist ? vec(R.speedCentres) : vec(R.timeCentres)}}})},
                                {"tiling", sp.spectra.meta.value("tiling", Json())}, {"n_members", sp.spectra.meta.value("n_members", Json())},
                                {"members_skipped", sp.spectra.meta.value("members_skipped", Json())}, {"options", sp.spectra.meta.value("options", Json())}, {"producer", pj},
                                {"description", std::string("the speed spectrum (") + part + ") of the Hodge-Dirac modes: joint (eigenvalue x frequency) bank tiles; a wave of phase speed c lies on 2 pi f = c mu"}};
      return a;
    };
    return ng::Arrays{{"energy", stat("energy", "power", Json(), false)}, {"tile_speed", stat("tile_speed", "speed", Json("length/s"), false)},
                      {"quadrature", stat("quadrature", "quadrature_fraction", Json(), false)}, {"histogram", stat("histogram", "power", Json(), true)}};
  };
  return n;
}

// ── the source model of MEG fingerprinting (docs/specs/2026-11-16-meg-fingerprinting.md, 11.2, steps 14 and 15) ──────────────────────────────────────────────────
// The citation every stored result of these nodes carries (spec 11.5), with the method's own references.
const Json& source_model_references() {
  static const Json r = Json::array({
      "Van Veen BD, van Drongelen W, Yuchtman M, Suzuki A (1997) Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. IEEE Trans Biomed Eng 44:867",
      "Tadel F, Baillet S, Mosher JC, Pantazis D, Leahy RM (2011) Brainstorm: a user-friendly application for MEG/EEG analysis. Comput Intell Neurosci 2011:879716; the 2016 source estimation "
      "(bst_inverse_linear_2016: LCMV, pseudo neural activity index) and the scout function PCA (bst_scout_value) of the release of 2018-10-12, written from their documented behaviour, "
      "their code run only to make the goldens",
      "da Silva Castanheira J, Orozco Perez HD, Misic B, Baillet S (2021) Brief segments of neurophysiological activity enable individual differentiation. Nat Commun 12:5713, "
      "doi:10.1038/s41467-021-25895-8; software: megFingerprinting v1.0.0 (06fdba8)"});
  return r;
}
ng::Array matrix_array(const Eigen::MatrixXd& m) {
  ng::Array a;
  a.rows = m.rows(); a.cols = m.cols();
  a.data.resize(static_cast<std::size_t>(m.size()));
  for (Eigen::Index i = 0; i < m.rows(); ++i) for (Eigen::Index j = 0; j < m.cols(); ++j) a.data[static_cast<std::size_t>(i * m.cols() + j)] = m(i, j);
  return a;
}
Eigen::MatrixXd matrix_of(const ng::Array& a) { return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(a.data.data(), a.rows, a.cols); }
const ng::Array& handed(const ng::Arrays& constants, const char* port, const char* who, const char* what) {
  const auto it = constants.find(port);
  if (it == constants.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + what + " was not handed over");
  return it->second;
}
// the unit vertex normals of a surface handed as a constant (vertices [n x 3], faces in meta): the orientation of each source of a fixed-orientation model. `normals`: brainstorm (the
// reference's VertNormals, nxr::geometry::vertexNormalsPatch) | corner_angle (nxr::geometry::vertexNormals)
std::string normals_choice(const Json& p, const char* who) {
  const std::string c = p.value("normals", std::string("brainstorm"));
  if (c != "brainstorm" && c != "corner_angle") ng::refuse(std::string(who) + ": normals is brainstorm or corner_angle (not " + c + ")");
  return c;
}
Eigen::MatrixXd surface_normals(const ng::Array& S, const char* who, const std::string& choice) {
  const std::vector<std::int32_t> faces = S.meta.value("faces", std::vector<std::int32_t>());
  if (S.cols != 3 || faces.empty() || faces.size() % 3) ng::refuse(std::string(who) + ": the source surface (" + S.meta.value("name", std::string()) + ") has no faces to orient its vertices");
  const int nV = static_cast<int>(S.rows), nF = static_cast<int>(faces.size() / 3);
  const nxr::geometry::Mat3 n = choice == "corner_angle" ? nxr::geometry::vertexNormals(S.data.data(), nV, faces.data(), nF) : nxr::geometry::vertexNormalsPatch(S.data.data(), nV, faces.data(), nF);
  Eigen::MatrixXd out(n.rows(), 3);
  for (Eigen::Index i = 0; i < n.rows(); ++i) {
    const Eigen::Vector3d v(n(i, 0), n(i, 1), n(i, 2));
    if (!(v.norm() > 0) || !v.allFinite()) ng::refuse(std::string(who) + ": vertex " + std::to_string(i) + " of " + S.meta.value("name", std::string()) + " has no normal (it is in no face)");
    out.row(i) = v.normalized().transpose();
  }
  return out;
}

// ── meeg.lcmv ──────────────────────────────────────────────────────────────────────────────────────────────────
// THE LCMV BEAMFORMER of a head model, from a DATA covariance (a statistic of the run: `run noisecov` on the recording itself, or on its band-passed copy) and a NOISE covariance (`run noisecov`
// on the empty room), both over picks of the head model's channels; the kernel's channels are the data covariance's pick, the noise covariance is read on them in their order. Two conventions:
//  - brainstorm (the default): Brainstorm process_inverse_2016 with LCMV, measure NAI, orientation fixed (nxr::meeg::lcmvKernel). The inputs are prepared as minimum_norm_inputs prepares them
//    under Brainstorm's conventions (the projectors of status 1 and 2 on the gain and both sides of each covariance, then the average reference per family); the sources are oriented along
//    the source surface's vertex normals. Made: the kernel [N x M], its channels, the noise whitener (Brainstorm's `Whitener`) and the data whitener the kernel used.
//  - mne: MNE-Python's make_lcmv (nxr::meeg::rankWhitener, nxr::meeg::mneLcmvKernel). The noise covariance is optional (none: MNE-Python's ad hoc identity covariance); minimum_norm_inputs
//    under MNE-Python's conventions gives the projected noise covariance (every projector of the data's recording, with the average reference of EEG, sEEG and ECoG, and the noise
//    covariance's own), its whitener keeps per modality the rank 'info' (the channels less the projectors on them) or the rank given. Made: the kernel [N x M] or [3N x M] (free: x y z per
//    source), its channels and the noise whitener the kernel ends with (none without a noise covariance: the whitener is then the projector); no data whitener.
struct LcmvPrepared {
  Eigen::MatrixXd G, Cd, Cn, orientations;
  std::vector<std::int32_t> channels;
  std::vector<std::string> types;
  nxr::meeg::LcmvOptions opt;
  bool averaged = false;
  std::string forward, data_cov, noise_cov, surface, regularisation;
  Json projectors = Json::object();
  std::string normals;                         // brainstorm | corner_angle
  Json channels_matched = Json::object();      // each covariance of another channel set matched to the head model's by name (the preparation's record)
  std::string conventions = "brainstorm";
  // mne
  nxr::meeg::MneLcmvOptions mopt;
  Eigen::MatrixXd Wn;                          // the noise whitener (MNE-Python's filters['whitener'])
  int rank = 0;
  Json ranks = Json::object();                 // the rank kept per modality, and where it came from
};
nxr::meeg::LcmvOptions lcmv_options(const Json& p, std::string* name = nullptr) {
  const std::string r = p.value("regularisation", std::string("median"));
  nxr::meeg::LcmvOptions o;
  using R = nxr::meeg::TruncatedRegularisation;
  if (r == "median") o.covariance.regularisation = R::Median;
  else if (r == "ridge") o.covariance.regularisation = R::Ridge;
  else if (r == "diagonal") o.covariance.regularisation = R::Diagonal;
  else if (r == "none") o.covariance.regularisation = R::None;
  else ng::refuse("meeg.lcmv: regularisation is median, ridge, diagonal or none (not " + r + ")");
  o.covariance.ridgeFraction = p.value("ridge_fraction", 0.1);
  if (!(o.covariance.ridgeFraction >= 0 && o.covariance.ridgeFraction <= 1)) ng::refuse("meeg.lcmv: ridge_fraction is in [0, 1]");
  o.detectDiagonal = p.value("detect_diagonal", true);
  if (name) *name = r;
  return o;
}
std::string lcmv_conventions(const Json& p) {
  const std::string c = p.value("conventions", std::string("brainstorm"));
  if (c != "brainstorm" && c != "mne") ng::refuse("meeg.lcmv: conventions is brainstorm or mne (not " + c + ")");
  // each convention's own options: the other's are refused when given, never silently ignored
  auto given = [&](const char* k) { return p.contains(k) && !p[k].is_null(); };
  if (c == "brainstorm") {
    for (const char* k : {"reg", "pick_ori", "weight_norm", "rank"})
      if (given(k)) ng::refuse(std::string("meeg.lcmv: ") + k + " is an option of conventions mne (MNE-Python's make_lcmv)");
  } else {
    if (p.value("regularisation", std::string("median")) != "median" || p.value("ridge_fraction", 0.1) != 0.1 || !p.value("detect_diagonal", true))
      ng::refuse("meeg.lcmv: regularisation, ridge_fraction and detect_diagonal are options of conventions brainstorm; under mne the data covariance is loaded by reg");
  }
  return c;
}
// MNE-Python's options: reg (0.05), pick_ori (free), weight_norm (unit-noise-gain-invariant); null: MNE-Python's default
nxr::meeg::MneLcmvOptions mne_lcmv_options(const Json& p) {
  nxr::meeg::MneLcmvOptions o;
  auto given = [&](const char* k) { return p.contains(k) && !p[k].is_null(); };
  if (given("reg")) {
    if (!p["reg"].is_number() || !(p["reg"].get<double>() >= 0)) ng::refuse("meeg.lcmv: reg is a number >= 0 (a fraction of the mean singular value of the whitened data covariance)");
    o.reg = p["reg"].get<double>();
  }
  if (given("pick_ori")) {
    const std::string v = p["pick_ori"].get<std::string>();
    if (v == "free") o.orientation = nxr::meeg::LcmvOrientation::Free;
    else if (v == "normal") o.orientation = nxr::meeg::LcmvOrientation::Normal;
    else if (v == "max-power") o.orientation = nxr::meeg::LcmvOrientation::MaxPower;
    else ng::refuse("meeg.lcmv: pick_ori is free, normal or max-power (not " + v + ")");
  }
  if (given("weight_norm")) {
    const std::string v = p["weight_norm"].get<std::string>();
    using W = nxr::meeg::LcmvWeightNorm;
    if (v == "none") o.weightNorm = W::None;
    else if (v == "unit-noise-gain") o.weightNorm = W::UnitNoiseGain;
    else if (v == "unit-noise-gain-invariant") o.weightNorm = W::UnitNoiseGainInvariant;
    else if (v == "nai") o.weightNorm = W::Nai;
    else ng::refuse("meeg.lcmv: weight_norm is none, unit-noise-gain, unit-noise-gain-invariant or nai (not " + v + ")");
  }
  if (given("rank") && !(p["rank"].is_string() && p["rank"].get<std::string>() == "info") && !(p["rank"].is_number_integer() && p["rank"].get<std::int64_t>() > 0))
    ng::refuse("meeg.lcmv: rank is info (the channels less the projectors on them) or a positive integer");
  return o;
}
const char* orientation_name(nxr::meeg::LcmvOrientation o) { return o == nxr::meeg::LcmvOrientation::Free ? "free" : o == nxr::meeg::LcmvOrientation::Normal ? "normal" : "max-power"; }
const char* weight_norm_name(nxr::meeg::LcmvWeightNorm w) {
  using W = nxr::meeg::LcmvWeightNorm;
  return w == W::None ? "none" : w == W::UnitNoiseGain ? "unit-noise-gain" : w == W::UnitNoiseGainInvariant ? "unit-noise-gain-invariant" : "nai";
}
const Json& mne_lcmv_references() {
  static const Json r = Json::array({
      "Van Veen BD, van Drongelen W, Yuchtman M, Suzuki A (1997) Localization of brain electrical activity via linearly constrained minimum variance spatial filtering. IEEE Trans Biomed Eng 44:867",
      "Sekihara K, Nagarajan SS (2008) Adaptive spatial filters for electromagnetic brain imaging. Springer (the weight normalisations and the max-power orientation)",
      "Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Parkkonen L, Hamalainen MS (2014) MNE software for processing MEG and EEG data. NeuroImage 86:446; "
      "mne.beamformer.make_lcmv of MNE-Python 1.13.2, written from its documented behaviour, run only to make the goldens"});
  return r;
}

// Brainstorm's 2016 LCMV (nxr::meeg::lcmvKernel)
ng::Arrays lcmv_run_brainstorm(const LcmvPrepared& pr) {
  nxr::meeg::LcmvKernel ker;
  try { ker = nxr::meeg::lcmvKernel(pr.G, pr.orientations, pr.Cd, pr.Cn, pr.types, pr.opt); } catch (const std::invalid_argument& e) { ng::refuse(std::string("meeg.lcmv: ") + e.what()); }
  const std::int64_t M = static_cast<std::int64_t>(pr.channels.size()), N = ker.K.rows();
  const std::string rule = ker.applied == pr.opt.covariance.regularisation ? pr.regularisation : std::string("diagonal");   // the diagonal rule of a diagonal noise covariance
  // the producer record in the importer's keys (nxr::bst::inverse_method_of reads it the same: measure LCMV, solver lcmv, fixed), plus what a computed kernel adds
  Json producer = Json{{"method", "LCMV"}, {"method_comment", "PNAI"}, {"units", "PNAI"}, {"n_channels", M}, {"n_sources", N}, {"n_components", 1},
                       {"source_ori", "fixed"}, {"inverse_method", "lcmv"}, {"inverse_measure", "nai"}, {"source_orient", Json::array({"fixed"})}, {"function", "lcmvp"},
                       {"use_depth", false}, {"noise_method", pr.regularisation}, {"noise_reg", pr.opt.covariance.ridgeFraction},
                       {"references", source_model_references()},
                       {"computed", Json{{"reference", "Brainstorm bst_inverse_linear_2016 (process_inverse_2016), the release of 2018-10-12"},
                                         {"options", Json{{"regularisation", pr.regularisation}, {"ridge_fraction", pr.opt.covariance.ridgeFraction}, {"detect_diagonal", pr.opt.detectDiagonal}}},
                                         {"rule_applied", rule}, {"average_reference", pr.averaged}, {"projectors", pr.projectors}, {"channels_matched", pr.channels_matched},
                                         {"forward", pr.forward}, {"data_cov", pr.data_cov}, {"noise_cov", pr.noise_cov},
                                         {"orientation", "fixed: the unit vertex normals of " + pr.surface}, {"normals", pr.normals},
                                         {"data_ranks", ker.dataRanks}, {"noise_ranks", ker.noiseRanks}, {"notes", ker.notes},
                                         {"warnings", Json::array({"an index, not a current: each source's output has unit variance on the data covariance it was made from (when nothing is regularised)",
                                                                   "the noise covariance does not enter the kernel (the reference's 2016 LCMV); only its whitener is stored",
                                                                   "correlated sources partly cancel (the beamformer's blind spot)"})}}}};
  const std::optional<Json> inverse = nxr::bst::inverse_method_of(producer);
  ng::Array K = matrix_array(ker.K);
  K.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"precision", "float32"}, {"params", inverse ? Json{{"inverse", *inverse}} : Json()}, {"producer", producer},
                            {"description", "sensor field to source field on the cortex (LCMV beamformer, pseudo neural activity index, fixed orientation), " + std::to_string(N) + " sources"}};
  ng::Array ch;
  ch.rows = M; ch.cols = 1;
  for (const auto c : pr.channels) ch.data.push_back(c);
  ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the beamformer maps, in column order"}};
  ng::Array W = matrix_array(ker.noiseWhitener), D = matrix_array(ker.dataWhitener);
  W.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"precision", "float64"}, {"producer", Json{{"rule", rule}, {"ranks", ker.noiseRanks}, {"references", source_model_references()}}},
                            {"description", "the whitener of the beamformer's noise covariance (" + rule + "), on its " + std::to_string(M) + " channels"}};
  D.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"precision", "float64"}, {"producer", Json{{"rule", rule}, {"ranks", ker.dataRanks}, {"references", source_model_references()}}},
                            {"description", "the whitener of the beamformer's data covariance (" + rule + "), on its " + std::to_string(M) + " channels: the one the kernel is made with"}};
  return ng::Arrays{{"kernel", std::move(K)}, {"channels", std::move(ch)}, {"whitener", std::move(W)}, {"data_whitener", std::move(D)}};
}

// MNE-Python's make_lcmv (nxr::meeg::mneLcmvKernel): the operator apply_lcmv applies to the channels as recorded, filters['weights'] x filters['whitener']
ng::Arrays lcmv_run_mne(const LcmvPrepared& pr) {
  nxr::meeg::MneLcmvKernel ker;
  try { ker = nxr::meeg::mneLcmvKernel(pr.G, pr.orientations, pr.Cd, pr.Wn, pr.rank, pr.mopt); } catch (const std::invalid_argument& e) { ng::refuse(std::string("meeg.lcmv: ") + e.what()); }
  const std::int64_t M = static_cast<std::int64_t>(pr.channels.size()), R = ker.K.rows();
  const bool free = pr.mopt.orientation == nxr::meeg::LcmvOrientation::Free;
  const std::int64_t N = free ? R / 3 : R;
  const std::string wn = weight_norm_name(pr.mopt.weightNorm), ori = orientation_name(pr.mopt.orientation);
  const bool unit_gain = pr.mopt.weightNorm == nxr::meeg::LcmvWeightNorm::None;
  Json warnings = Json::array({"correlated sources partly cancel (the beamformer's blind spot)", "the diagonal loading (reg) trades spatial resolution for stability"});
  if (!unit_gain) warnings.push_back("weight-normalised (" + wn + "): an index in units of the projected noise, not a current; its scale depends on the data covariance's span");
  if (free && pr.mopt.weightNorm == nxr::meeg::LcmvWeightNorm::UnitNoiseGain) warnings.push_back("unit-noise-gain with a free orientation depends on the frame of the x y z rows (the invariant form does not)");
  if (pr.noise_cov.empty()) warnings.push_back("no noise covariance: the channels are whitened by an identity covariance (MNE-Python's ad hoc), so their units must be comparable");
  Json computed{{"reference", "MNE-Python 1.13.2 mne.beamformer.make_lcmv (rank 'info', depth None, inversion 'matrix', reduce_rank False); the kernel is filters['weights'] @ filters['whitener']"},
                {"options", Json{{"reg", pr.mopt.reg}, {"pick_ori", ori}, {"weight_norm", wn}, {"rank", pr.ranks}}},
                {"loading", ker.loading}, {"rank", pr.rank}, {"average_reference", pr.averaged}, {"projectors", pr.projectors}, {"channels_matched", pr.channels_matched},
                {"forward", pr.forward}, {"data_cov", pr.data_cov}, {"noise_cov", pr.noise_cov.empty() ? Json("none: an identity covariance on the projectors (MNE-Python's ad hoc)") : Json(pr.noise_cov)},
                {"orientation", free ? std::string("free: x y z rows of the head model's frame") : pr.mopt.orientation == nxr::meeg::LcmvOrientation::Normal ? "the unit vertex normals of " + pr.surface
                                                                                                                                                              : "max-power, its sign along the unit vertex normals of " + pr.surface},
                {"normals", pr.normals}, {"notes", ker.notes}, {"warnings", warnings}};
  if (pr.mopt.weightNorm == nxr::meeg::LcmvWeightNorm::Nai) computed["noise_level"] = ker.noise;
  Json producer = Json{{"method", "LCMV"}, {"method_comment", "LCMV (MNE-Python, " + wn + ")"}, {"units", unit_gain ? "A.m" : wn}, {"n_channels", M}, {"n_sources", R}, {"n_components", free ? 3 : 1},
                       {"source_ori", free ? "free" : "fixed"}, {"inverse_method", "lcmv"}, {"inverse_measure", wn}, {"source_orient", Json::array({free ? "free" : "fixed"})}, {"function", "make_lcmv"},
                       {"use_depth", false}, {"noise_method", "mne"}, {"noise_reg", pr.mopt.reg}, {"references", mne_lcmv_references()}, {"computed", computed}};
  const std::optional<Json> inverse = nxr::bst::inverse_method_of(producer);
  ng::Array K = matrix_array(ker.K);
  K.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", free ? "vector3" : "scalar"}, {"precision", "float32"}, {"params", inverse ? Json{{"inverse", *inverse}} : Json()}, {"producer", producer},
                            {"description", "sensor field to source field on the cortex (LCMV beamformer, MNE-Python's conventions, " + ori + ", " + wn + "), " + std::to_string(N) + " sources"}};
  ng::Array ch;
  ch.rows = M; ch.cols = 1;
  for (const auto c : pr.channels) ch.data.push_back(c);
  ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the beamformer maps, in column order"}};
  ng::Arrays out{{"kernel", std::move(K)}, {"channels", std::move(ch)}};
  if (!pr.noise_cov.empty()) {
    ng::Array W = matrix_array(pr.Wn);
    W.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"precision", "float64"}, {"producer", Json{{"rule", "MNE-Python: rank per modality, no regularisation"}, {"ranks", pr.ranks}, {"references", mne_lcmv_references()}}},
                              {"description", "the whitener of the beamformer's noise covariance (MNE-Python's, the projectors in it), on its " + std::to_string(M) + " channels: the one the kernel ends with"}};
    out["whitener"] = std::move(W);
  } else {
    ng::Array none; none.meta["absent"] = true;                          // no noise covariance: the whitener is the projector, not a statistic's
    out["whitener"] = std::move(none);
  }
  ng::Array none; none.meta["absent"] = true;                            // MNE-Python does not whiten by the data covariance
  out["data_whitener"] = std::move(none);
  return out;
}

ng::NodeSpec lcmv() {
  ng::NodeSpec n;
  n.name = "meeg.lcmv";
  n.kind = ng::Kind::array;
  n.inputs = {{"forward", "the head model: a stored operator, its gain channels x 3 sources (x y z)", false},
              {"data_covariance", "the data covariance: a stored statistic of the recording the kernel is for, over a pick of the head model's channels", false},
              {"noise_covariance", "the noise covariance: a stored statistic (the empty room) over the same pick; optional under conventions mne (none: an identity covariance)", true},
              {"sources", "the source surface (the head model's sources are its vertices): its vertex normals orient the sources", false}};
  n.outputs = {{"kernel", "the beamformer, one row a source (three, x y z, for a free orientation) x the pick's channels (an operator from the channels to the sources)", false},
               {"channels", "the channels the kernel maps, in column order (the covariances' pick)", false},
               {"whitener", "the noise covariance's whitener (brainstorm: under the same rule, not used by the kernel; mne: the one the kernel ends with; absent without a noise covariance)", true},
               {"data_whitener", "the data covariance's whitener, the one the kernel is made with (brainstorm only)", true}};
  n.parameters = {
      {"conventions", Json("brainstorm"), "brainstorm (Brainstorm 2016: fixed orientation, the pseudo neural activity index) | mne (MNE-Python's make_lcmv)", "-", {"brainstorm", "mne"}, false},
      {"regularisation", Json("median"), "brainstorm: how each covariance is regularised after truncation to its rank, type by type: median (the eigenvalue tail set to the median) | ridge (a fraction of the largest eigenvalue) | diagonal | none", "-", {"median", "ridge", "diagonal", "none"}, false},
      {"ridge_fraction", Json(0.1), "brainstorm: the ridge's fraction of the largest eigenvalue (used by ridge only; the reference passes 0.1 with median, where it has no effect)", "fraction", {}, false},
      {"detect_diagonal", Json(true), "brainstorm: a diagonal noise covariance whitens both covariances by their diagonal (the reference's rule)", "-", {}, false},
      {"reg", Json(), "mne: the diagonal loading of the whitened data covariance, a fraction of its mean singular value (MNE-Python's reg); null: 0.05", "fraction", {}, true},
      {"pick_ori", Json(), "mne: free (three rows a source, x y z) | normal (along the source surface's normals) | max-power (each source's orientation of most power); null: free (MNE-Python's None)", "-", {"free", "normal", "max-power"}, true},
      {"weight_norm", Json(), "mne: none (unit gain, A.m) | unit-noise-gain | unit-noise-gain-invariant | nai; null: unit-noise-gain-invariant (MNE-Python's default)", "-", {"none", "unit-noise-gain", "unit-noise-gain-invariant", "nai"}, true},
      {"rank", Json(), "mne: the rank of the covariances, info (per modality, the channels less the projectors on them) or an integer (one modality only); null: info", "-", {}, true},
      {"average_reference", Json(true), "apply the average reference of EEG, sEEG and ECoG (brainstorm: to the gain and both covariances, per family; mne: as a projector)", "-", {}, false},
      {"normals", Json("brainstorm"), "the sources' orientations: brainstorm (the reference's VertNormals: minus the sum of the incident faces' unit normals, normalised) | corner_angle (geometry-central's corner-angle weighted normals)", "-", {"brainstorm", "corner_angle"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "solved in double; the kernel is stored as float32 like every operator");
  n.matches = "Brainstorm's LCMV beamformer (process_inverse_2016 / bst_inverse_linear_2016, measure NAI, fixed orientation, the release of 2018-10-12), meeg nxr::meeg::lcmvKernel; "
              "MNE-Python's mne.beamformer.make_lcmv (1.13.2), meeg nxr::meeg::rankWhitener + mneLcmvKernel";
  n.description = "the LCMV beamformer from a head model, a data covariance and a noise covariance: Brainstorm's 2016 pseudo neural activity index, or MNE-Python's make_lcmv";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("forward") || in.at("forward").cls != ng::PortClass::operator_) ng::refuse("meeg.lcmv: forward is a stored operator (a head model)");
    const std::string conv = lcmv_conventions(p);
    for (const char* c : {"data_covariance", "sources"}) if (!in.count(c)) ng::refuse(std::string("meeg.lcmv: ") + c + " is not given");
    if (conv == "brainstorm" && !in.count("noise_covariance")) ng::refuse("meeg.lcmv: the noise covariance is required under conventions brainstorm (give noise_cov)");
    if (conv == "brainstorm") lcmv_options(p); else mne_lcmv_options(p);
    ng::Planned out;
    out.out["kernel"] = constant_port("operator");
    out.out["channels"] = constant_port("selection");
    out.out["whitener"] = constant_port("operator");
    out.out["data_whitener"] = constant_port("operator");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    const char* who = "meeg.lcmv";
    LcmvPrepared pr;
    pr.conventions = lcmv_conventions(p);
    const bool mne = pr.conventions == "mne";
    const ng::Array& F = handed(constants, "forward", who, "the head model");
    const ng::Array& Sd = handed(constants, "data_covariance", who, "the data covariance");
    const auto nit = constants.find("noise_covariance");
    const ng::Array* Sn = nit == constants.end() || nit->second.meta.value("placeholder", false) ? nullptr : &nit->second;
    if (!Sn && !mne) ng::refuse("meeg.lcmv: the noise covariance is required under conventions brainstorm (give noise_cov)");
    const ng::Array& S = handed(constants, "sources", who, "the source surface");
    pr.forward = F.meta.value("name", std::string());
    pr.data_cov = Sd.meta.value("name", std::string());
    pr.noise_cov = Sn ? Sn->meta.value("name", std::string()) : std::string();
    pr.surface = S.meta.value("name", std::string());
    for (const ng::Array* c : {&Sd, Sn})
      if (c && F.meta.value("to_manifold_id", std::string()) != c->meta.value("channels_manifold_id", std::string()))
        ng::refuse("meeg.lcmv: " + c->meta.value("name", std::string()) + " is on other channels than " + pr.forward + " (a covariance of another channel set is matched to the head model's by name when it is prepared)");
    if (F.cols % 3 != 0) ng::refuse("meeg.lcmv: " + pr.forward + ": the gain is not channels x 3N");
    // the kernel's channels are the data covariance's pick (the good channels of the run); the noise covariance is read on them, in that order (Brainstorm indexes both by the run's good channels)
    pr.channels = Sd.meta.value("channels", std::vector<std::int32_t>());
    pr.types = Sd.meta.value("channel_types", std::vector<std::string>());
    const std::int64_t M = static_cast<std::int64_t>(pr.channels.size()), N = F.cols / 3;
    const std::vector<std::int32_t> nch = Sn ? Sn->meta.value("channels", std::vector<std::int32_t>()) : std::vector<std::int32_t>();
    for (const ng::Array* c : {&Sd, Sn})
      if (c && (M == 0 || c->rows != c->cols || c->rows != static_cast<std::int64_t>(c == &Sd ? pr.channels.size() : nch.size())))
        ng::refuse("meeg.lcmv: " + c->meta.value("name", std::string()) + ": its array does not fit its channels");
    for (const std::int32_t ch : pr.channels) if (ch < 0 || ch >= F.rows) ng::refuse("meeg.lcmv: a channel of " + pr.data_cov + " is outside the gain's channels");
    Eigen::MatrixXd Cn = Eigen::MatrixXd::Identity(M, M);              // no noise covariance (mne only): MNE-Python's ad hoc covariance, an identity (std 1 for every type)
    if (Sn) {
      std::map<std::int32_t, Eigen::Index> in_noise;
      for (std::size_t i = 0; i < nch.size(); ++i) in_noise[nch[i]] = static_cast<Eigen::Index>(i);
      std::vector<Eigen::Index> at;
      for (const std::int32_t ch : pr.channels) {
        const auto it = in_noise.find(ch);
        if (it == in_noise.end()) ng::refuse("meeg.lcmv: channel " + std::to_string(ch) + " of " + pr.data_cov + " is not covered by " + pr.noise_cov + " (both covariances are read on the run's good channels)");
        at.push_back(it->second);
      }
      const Eigen::MatrixXd Snall = matrix_of(*Sn);
      for (Eigen::Index i = 0; i < M; ++i) for (Eigen::Index j = 0; j < M; ++j) Cn(i, j) = Snall(at[static_cast<std::size_t>(i)], at[static_cast<std::size_t>(j)]);
    }
    if (S.rows != N) ng::refuse("meeg.lcmv: " + pr.surface + " has " + std::to_string(S.rows) + " vertices, the head model " + std::to_string(N) + " sources");
    pr.normals = normals_choice(p, who);
    pr.orientations = surface_normals(S, who, pr.normals);
    const Eigen::MatrixXd Fall = matrix_of(F);
    const bool avg = p.value("average_reference", true);
    if (!mne) {
      std::string reg;
      pr.opt = lcmv_options(p, &reg);
      pr.regularisation = reg;
      const MinNormInputs d = minimum_norm_inputs(Fall, matrix_of(Sd), pr.channels, pr.types, projectors_of(Sd.meta), {}, "brainstorm", avg);
      const MinNormInputs z = minimum_norm_inputs(Fall, Cn, pr.channels, pr.types, projectors_of(Sn->meta), {}, "brainstorm", avg);
      pr.G = d.G; pr.Cd = d.C; pr.Cn = z.C; pr.averaged = d.averaged;
      pr.projectors = Json{{"data_covariance", d.record}, {"noise_covariance", z.record}};
    } else {
      // MNE-Python (make_lcmv): the info's projectors are the data's recording's (those the caller handed with the data covariance), with the average reference as a projector; the noise
      // covariance is projected by them and by its own (its recording's), and whitened per modality on its rank; the gain and the data covariance are used as they are (the whitener holds
      // the projectors). Without a noise covariance MNE-Python makes an identity covariance of all types, which it allows for one type only.
      pr.mopt = mne_lcmv_options(p);
      if (!Sn) {
        std::set<std::string> kinds;
        for (const auto& t : pr.types) kinds.insert(t == "MEG GRAD" ? "grad" : t == "EEG" ? "eeg" : t.rfind("MEG", 0) == 0 ? "mag" : t);
        int sensor = 0;
        for (const char* k : {"mag", "grad", "eeg"}) sensor += static_cast<int>(kinds.count(k));
        if (sensor > 1) ng::refuse("meeg.lcmv: several sensor types (magnetometers, gradiometers, EEG) need a noise covariance to be whitened (give noise_cov), as MNE-Python requires");
      }
      const MinNormInputs z = minimum_norm_inputs(Fall, Cn, pr.channels, pr.types, projectors_of(Sd.meta), Sn ? covariance_projectors_of(Sn->meta) : std::vector<std::string>(), "mne", avg);
      pr.G = z.G; pr.Cd = matrix_of(Sd); pr.Cn = z.C; pr.averaged = z.averaged;
      // the rank per modality: 'info' (its channels less the dimension of the info's projectors restricted to them, each vector renormalised: MNE-Python's make_projector), or the caller's
      const auto blocks = nxr::meeg::modalityBlocks(pr.types);
      std::vector<int> ranks;
      const bool given = p.contains("rank") && p["rank"].is_number_integer();
      if (given && blocks.size() != 1) ng::refuse("meeg.lcmv: rank as an integer is for one modality; these channels have " + std::to_string(blocks.size()));
      for (const auto& idx : blocks) {
        const int n = static_cast<int>(idx.size());
        if (given) { ranks.push_back(static_cast<int>(p["rank"].get<std::int64_t>())); continue; }
        nxr::bst::ProjectorRule rule;
        rule.statuses = {0, 1, 2};
        for (const auto i : idx) rule.channels.push_back(static_cast<std::size_t>(pr.channels[static_cast<std::size_t>(i)]));
        rule.renormalise = true;
        rule.zero_tiny = false;
        const std::vector<double> pv = nxr::bst::joint_projector(z.info_projectors, rule);
        ranks.push_back(pv.empty() ? n : static_cast<int>(std::lround(dense(pv, idx.size()).trace())));
      }
      nxr::meeg::RankWhitening w;
      try { w = nxr::meeg::rankWhitener(pr.Cn, pr.types, ranks); } catch (const std::invalid_argument& e) { ng::refuse(std::string("meeg.lcmv: ") + e.what()); }
      pr.Wn = w.W;
      pr.rank = 0;
      for (std::size_t b = 0; b < ranks.size(); ++b) { pr.ranks[w.modalities[b]] = ranks[b]; pr.rank += ranks[b]; }
      pr.ranks["from"] = given ? "given" : "info: the channels less the projectors on them";
      pr.projectors = Json{{"noise_covariance", z.record}};
    }
    for (const auto& [key, c] : {std::pair<const char*, const ng::Array*>{"data_covariance", &Sd}, {"noise_covariance", Sn}}) {
      if (!c) continue;
      if (c->meta.contains("projectors_choice")) pr.projectors[key]["choice"] = c->meta["projectors_choice"];   // which projectors the caller had handed (the run's --projectors)
      if (c->meta.contains("channels_matched")) pr.channels_matched[key] = c->meta["channels_matched"];        // a covariance of another channel set (the empty room's), matched by name
    }
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const LcmvPrepared& pr = std::any_cast<const LcmvPrepared&>(prepared);
    return pr.conventions == "mne" ? lcmv_run_mne(pr) : lcmv_run_brainstorm(pr);
  };
  return n;
}

// ── meeg.region_pca_basis ─────────────────────────────────────────────────────────────────────────────────────
// THE FIRST PRINCIPAL COMPONENT OF EACH REGION, from statistics made first (Brainstorm's scout function PCA with the sign flip, applied before the spectrum): the region's sources are the rows
// of a fixed-orientation kernel K_R, their covariance over the span is K_R C K_R^T with C the channels' covariance over that span (a stored statistic: `run noisecov` over the span, one block,
// its mean removed), never formed for the whole cortex. Each source's sign is flipped by its orientation (the source surface's normals) as the reference does. Made: a constant [2R x M] (the
// R components as channel weights, then the R peak sources' flipped kernel rows), what the next node measures over the span to set each component's sign.
struct RegionBasisPrepared {
  Eigen::MatrixXd basis;
  Json meta;
};
ng::NodeSpec region_pca_basis() {
  ng::NodeSpec n;
  n.name = "meeg.region_pca_basis";
  n.kind = ng::Kind::array;
  n.inputs = {{"kernel", "a fixed-orientation inverse kernel (one row a source), a stored operator", false},
              {"covariance", "the channels' covariance over the span the components are for (a statistic: one block, its mean removed), on a pick holding the kernel's channels", false},
              {"partition", "the regions: a partition (a set) of the kernel's sources", false},
              {"sources", "the kernel's source surface: its vertex normals set the sign flip", false}};
  n.outputs = {{"basis", "per region its first principal component as channel weights, then the flipped kernel row of its peak source: [2 regions x channels]", false}};
  n.parameters = {{"sign_flip", Json(true), "flip each source's sign by its orientation before the component, as the reference's scouts do (false: the series as the kernel gives them)", "-", {}, false},
                  {"normals", Json("brainstorm"), "the sources' orientations: brainstorm (the reference's VertNormals: minus the sum of the incident faces' unit normals, normalised) | corner_angle (geometry-central's corner-angle weighted normals)", "-", {"brainstorm", "corner_angle"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a constant made once from stored statistics, in double");
  n.matches = "Brainstorm's scout function PCA with the sign flip (bst_scout_value 'pca', isSignFlip, the release of 2018-10-12): its component from the region's covariance; meeg nxr::meeg::regionComponent";
  n.description = "each region's first principal component (a linear map of the channels) from a kernel, the channels' covariance over a span, a partition and the sources' normals";
  n.plan = [](const ng::Types& in, const Json&) {
    if (!in.count("kernel") || in.at("kernel").cls != ng::PortClass::operator_) ng::refuse("meeg.region_pca_basis: kernel is a stored operator (an inverse kernel)");
    for (const char* c : {"covariance", "partition", "sources"}) if (!in.count(c)) ng::refuse(std::string("meeg.region_pca_basis: ") + c + " is not given");
    ng::Planned out;
    out.out["basis"] = constant_port("statistic");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    const char* who = "meeg.region_pca_basis";
    const ng::Array& K = handed(constants, "kernel", who, "the kernel");
    const ng::Array& S = handed(constants, "covariance", who, "the covariance");
    const ng::Array& P = handed(constants, "partition", who, "the partition");
    const ng::Array& surf = handed(constants, "sources", who, "the source surface");
    const std::string kname = K.meta.value("name", std::string()), cname = S.meta.value("name", std::string()), pname = P.meta.value("name", std::string());
    if (K.meta.value("to_value_type", std::string("vector3")) != "scalar")
      ng::refuse("meeg.region_pca_basis: " + kname + " is a free-orientation kernel: the reference's component is of a fixed-orientation kernel's sources (one row a source)");
    const std::int64_t N = K.rows, M = K.cols;
    if (P.rows != N) ng::refuse("meeg.region_pca_basis: " + pname + " covers " + std::to_string(P.rows) + " elements, " + kname + " has " + std::to_string(N) + " sources");
    if (surf.rows != N) ng::refuse("meeg.region_pca_basis: the source surface has " + std::to_string(surf.rows) + " vertices, " + kname + " " + std::to_string(N) + " sources");
    // the covariance: one block over the span, its mean removed (the component is of the centred series over the span)
    const Json prod = S.meta.value("producer", Json::object());
    if (S.meta.value("measure", std::string()) != "covariance") ng::refuse("meeg.region_pca_basis: " + cname + " is not a covariance");
    if (prod.value("n_blocks", std::int64_t(1)) != 1)
      ng::refuse("meeg.region_pca_basis: " + cname + " has " + std::to_string(prod.value("n_blocks", std::int64_t(1))) + " blocks: the component needs the span's covariance as ONE block (run noisecov without block_seconds)");
    if (!prod.contains("start_sample") || !prod.contains("stop_sample")) ng::refuse("meeg.region_pca_basis: " + cname + " does not say its span (a covariance made by noisecov does)");
    // the kernel's channels in the covariance's pick
    const std::vector<std::int32_t> kch = K.meta.value("channels", std::vector<std::int32_t>()), sch = S.meta.value("channels", std::vector<std::int32_t>());
    std::map<std::int32_t, Eigen::Index> at;
    for (std::size_t i = 0; i < sch.size(); ++i) at[sch[i]] = static_cast<Eigen::Index>(i);
    std::vector<Eigen::Index> pick;
    for (const auto c : kch) {
      const auto it = at.find(c);
      if (it == at.end()) ng::refuse("meeg.region_pca_basis: channel " + std::to_string(c) + " of " + kname + " is not covered by " + cname);
      pick.push_back(it->second);
    }
    if (static_cast<std::int64_t>(pick.size()) != M || S.rows != static_cast<std::int64_t>(sch.size())) ng::refuse("meeg.region_pca_basis: " + cname + ": its array does not fit its channels");
    const Eigen::MatrixXd Call = matrix_of(S);
    Eigen::MatrixXd C(M, M);
    for (Eigen::Index i = 0; i < M; ++i) for (Eigen::Index j = 0; j < M; ++j) C(i, j) = Call(pick[static_cast<std::size_t>(i)], pick[static_cast<std::size_t>(j)]);
    const Eigen::MatrixXd Km = matrix_of(K);
    const Eigen::MatrixXd normals = surface_normals(surf, who, normals_choice(p, who));
    const bool flip_signs = p.value("sign_flip", true);
    // the regions: the partition's members in order (an element of no member is left out)
    const std::int64_t R = P.meta.value("n_members", std::int64_t(0));
    if (R <= 0) ng::refuse("meeg.region_pca_basis: " + pname + " has no member");
    std::vector<std::vector<Eigen::Index>> members(static_cast<std::size_t>(R));
    for (std::int64_t v = 0; v < N; ++v) {
      const std::int64_t m = static_cast<std::int64_t>(P.data[static_cast<std::size_t>(v)]);
      if (m >= R) ng::refuse("meeg.region_pca_basis: " + pname + ": element " + std::to_string(v) + " is in member " + std::to_string(m) + " of " + std::to_string(R));
      if (m >= 0) members[static_cast<std::size_t>(m)].push_back(v);
    }
    const Json codes = P.meta.value("member_codes", Json::array());
    RegionBasisPrepared pr;
    pr.basis.resize(2 * R, M);
    Json regions = Json::array();
    for (std::int64_t r = 0; r < R; ++r) {
      const auto& iv = members[static_cast<std::size_t>(r)];
      const Json label = r < static_cast<std::int64_t>(codes.size()) ? codes[static_cast<std::size_t>(r)] : Json(r);
      if (iv.empty()) ng::refuse("meeg.region_pca_basis: region " + label.dump() + " of " + pname + " holds no source");
      Eigen::MatrixXd Kr(static_cast<Eigen::Index>(iv.size()), M), Or(static_cast<Eigen::Index>(iv.size()), 3);
      for (std::size_t i = 0; i < iv.size(); ++i) { Kr.row(static_cast<Eigen::Index>(i)) = Km.row(iv[i]); Or.row(static_cast<Eigen::Index>(i)) = normals.row(iv[i]); }
      nxr::meeg::RegionComponent rc;
      Eigen::VectorXd flip;
      try {
        flip = flip_signs ? nxr::meeg::scoutSignFlip(Or) : Eigen::VectorXd(Eigen::VectorXd::Ones(Kr.rows()));
        rc = nxr::meeg::regionComponent(Kr, C, flip);
      } catch (const std::invalid_argument& e) { ng::refuse("meeg.region_pca_basis: region " + label.dump() + ": " + e.what()); }
      pr.basis.row(r) = rc.component;
      pr.basis.row(R + r) = rc.peak;
      regions.push_back(Json{{"member", r}, {"code", label}, {"n_sources", static_cast<std::int64_t>(iv.size())}, {"flipped", static_cast<std::int64_t>((flip.array() < 0).count())},
                             {"peak_source", iv[static_cast<std::size_t>(rc.peakSource)]}, {"explained", rc.explained}});
    }
    pr.meta = Json{{"channels", kch}, {"kernel", kname}, {"normals", normals_choice(p, who)}, {"covariance", cname}, {"partition", pname}, {"regions", regions}, {"n_regions", R}, {"sign_flip", flip_signs},
                   {"start_sample", prod["start_sample"]}, {"stop_sample", prod["stop_sample"]}};
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const RegionBasisPrepared& pr = std::any_cast<const RegionBasisPrepared&>(prepared);
    ng::Array a = matrix_array(pr.basis);
    a.meta = pr.meta;
    return ng::Arrays{{"basis", std::move(a)}};
  };
  return n;
}

// ── meeg.region_pca ───────────────────────────────────────────────────────────────────────────────────────────
// THE SIGN OF EACH REGION'S COMPONENT, set by the data as the reference sets it, and the region map: the component z = w^T x and the peak source p of each region are measured over the span
// (their maximum, minimum and mean: nxr::measurement::reduce over the span as one member of a time rule); then sign = nxr::meeg::regionComponentSign, and the region's signal is
// y = sign w^T x + (mean p - sign mean z): a linear map of the channels [R x M] (stored as an operator) and a constant per region (its `offset`, recorded per region in the map's provenance:
// a value per member of the partition would be a measurement, and it changes only the signal's mean). The span is the covariance's: a run over another span is refused.
struct RegionPcaPrepared {
  Eigen::MatrixXd basis;                    // [2R x M]
  std::vector<std::int32_t> channels;
  std::int64_t R = 0, start = 0, stop = 0;
  nxr::selection::Rule span;
  Json meta;
};
struct RegionPcaState {
  nxr::measurement::Ingredients acc;
  std::int64_t first = -1, samples = 0;
};
ng::NodeSpec region_pca() {
  ng::NodeSpec n;
  n.name = "meeg.region_pca";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed), over the span the components were made for", false},
              {"basis", "the regions' components and peak sources (meeg.region_pca_basis)", false}};
  n.outputs = {{"map", "each region's signal as channel weights: [regions x channels] (an operator from the channels to the regions)", false},
               {"channels", "the channels the map reads, in column order (the kernel's)", false},
               };
  n.parameters = {};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64,
                             "the components' maximum, minimum and mean over the span are measurement ingredients folded across chunks (max, min, sum), in double");
  n.matches = "Brainstorm's scout function PCA (bst_scout_value 'pca' PcaFirstMode: the sign from the data's extremes, the peak source's mean added back), the release of 2018-10-12; meeg nxr::meeg::regionComponentSign";
  n.description = "each region's first principal component signed by the data over the span: the region map (channels to regions), each region's offset in its provenance";
  n.plan = [](const ng::Types& in, const Json&) {
    streamed(in, "x", "meeg.region_pca");
    if (!in.count("basis")) ng::refuse("meeg.region_pca: basis is not given (meeg.region_pca_basis)");
    ng::Planned out;
    out.out["map"] = constant_port("operator");
    out.out["channels"] = constant_port("selection");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json&, ng::Shared&) {
    const ng::Array& B = handed(constants, "basis", "meeg.region_pca", "the basis");
    RegionPcaPrepared pr;
    pr.basis = matrix_of(B);
    pr.meta = B.meta;
    pr.channels = B.meta.value("channels", std::vector<std::int32_t>());
    pr.R = B.meta.value("n_regions", std::int64_t(0));
    pr.start = B.meta.value("start_sample", std::int64_t(0));
    pr.stop = B.meta.value("stop_sample", std::int64_t(0));
    if (pr.R <= 0 || pr.basis.rows() != 2 * pr.R || pr.basis.cols() != static_cast<Eigen::Index>(pr.channels.size())) ng::refuse("meeg.region_pca: the basis does not fit its regions and channels");
    if (pr.stop <= pr.start) ng::refuse("meeg.region_pca: the basis's span is empty");
    pr.span = nxr::selection::Rule{pr.start, pr.stop - pr.start, pr.stop - pr.start, 1, pr.stop};
    return std::any(std::move(pr));
  };
  n.start = [](const std::any& prepared) {
    const RegionPcaPrepared& pr = std::any_cast<const RegionPcaPrepared&>(prepared);
    RegionPcaState s;
    s.acc = nxr::measurement::Ingredients(2 * pr.R, 1);
    return std::any(std::move(s));
  };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const RegionPcaPrepared& pr = std::any_cast<const RegionPcaPrepared&>(prepared);
    RegionPcaState& s = std::any_cast<RegionPcaState&>(state);
    const ng::Block& b = in.at("x");
    for (const auto c : pr.channels) if (c < 0 || c >= b.rows) ng::refuse("meeg.region_pca: channel " + std::to_string(c) + " of the basis is not in the recording");
    const Eigen::MatrixXd x = own_rows(b, pr.channels);
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> z = pr.basis * x;   // [2R x samples], one row a series
    nxr::measurement::Request req;
    req.measures = {"max", "min", "mean"};
    req.per_row = true;
    nxr::measurement::reduce(s.acc, z.data(), z.rows(), z.cols(), at.first, pr.span, req);
    if (s.first < 0) s.first = at.first;
    s.samples += x.cols();
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const RegionPcaPrepared& pr = std::any_cast<const RegionPcaPrepared&>(prepared);
    const RegionPcaState& s = std::any_cast<const RegionPcaState&>(state);
    if (s.first != pr.start || s.first + s.samples != pr.stop)
      ng::refuse("meeg.region_pca: the components were made over samples [" + std::to_string(pr.start) + ", " + std::to_string(pr.stop) + ") (" + pr.meta.value("covariance", std::string()) +
                 "), this run reads [" + std::to_string(s.first) + ", " + std::to_string(s.first + s.samples) + "): state the same span");
    const std::int64_t R = pr.R, M = pr.basis.cols();
    Eigen::MatrixXd W(R, M);
    Json regions = pr.meta.value("regions", Json::array());
    for (std::int64_t r = 0; r < R; ++r) {
      auto v = [&](std::int64_t row, const char* m) { return nxr::measurement::value(s.acc, row, 0, m); };
      const double sg = nxr::meeg::regionComponentSign(v(R + r, "max"), v(R + r, "min"), v(R + r, "mean"), v(r, "max"), v(r, "min"), v(r, "mean"));
      W.row(r) = sg * pr.basis.row(r);
      if (r < static_cast<std::int64_t>(regions.size())) {
        regions[static_cast<std::size_t>(r)]["sign"] = sg;
        regions[static_cast<std::size_t>(r)]["offset"] = v(R + r, "mean") - sg * v(r, "mean");   // the peak source's mean, less the signed component's: signal = map x + offset
      }
    }
    const Json producer = Json{{"stage", "region pca"}, {"kernel", pr.meta.value("kernel", std::string())}, {"covariance", pr.meta.value("covariance", std::string())},
                               {"partition", pr.meta.value("partition", std::string())}, {"sign_flip", pr.meta.value("sign_flip", true)}, {"normals", pr.meta.value("normals", std::string())}, {"start_sample", pr.start}, {"stop_sample", pr.stop},
                               {"regions", regions}, {"references", source_model_references()},
                               {"reference", "Brainstorm bst_scout_value 'pca' with the sign flip, applied before the spectrum (the release of 2018-10-12)"},
                               {"warnings", Json::array({"the component is the direction of most variance within each region over THIS span: it can change between spans and people",
                                                         "its share of the region's signal is `explained`; the rest of the region is left out",
                                                         "its sign is a convention set by the span's extremes"})}};
    ng::Array map = matrix_array(W);
    map.meta["describe"] = Json{{"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"precision", "float32"}, {"producer", producer},
                                {"description", "the channels to the first principal component of each of " + std::to_string(R) + " regions of " + pr.meta.value("partition", std::string()) +
                                                    " through " + pr.meta.value("kernel", std::string()) + " (signal = map x + each region's offset, in the provenance)"}};
    ng::Array ch;
    ch.rows = M; ch.cols = 1;
    for (const auto c : pr.channels) ch.data.push_back(c);
    ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the region map reads, in column order"}};
    return ng::Arrays{{"map", std::move(map)}, {"channels", std::move(ch)}};
  };
  return n;
}

}  // namespace

void register_meeg_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {noise_covariance(), minimum_norm(), spherical_head_model(), analytic_bands(), phase_gradient(), filter_apply(), hodge_weights(), speed_spectrum(),
                         lcmv(), region_pca_basis(), region_pca()})
    if (!r.find(n.name)) r.add(std::move(n));
  register_quadrature_nodes();
}

}  // namespace nxr::pipelines::graph
