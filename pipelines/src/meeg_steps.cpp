// nxr-pipelines — THE M/EEG NODES: the source-imaging steps as pure nodes of nxr::graph::registry() (grouped names `meeg.*`), each the wrapper of the maths of
// meeg/ (nxr::meeg): the same function a test calls directly. A node reads what it is handed (the streamed recording, stored arrays as constants, the facts of
// its ports: a recording's channel types and quality) and describes what it made (`Array::meta["describe"]`); the definition says where it is stored.
//
//   meeg.noise_covariance  accumulate  x [channels x samples]  ->  channels (the good channels of the imaging types, indices), covariance [M x M] (a statistic)
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

}  // namespace

void register_meeg_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {noise_covariance()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
