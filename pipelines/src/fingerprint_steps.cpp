// nxr-pipelines — THE PER-SUBJECT FEATURES OF MEG FINGERPRINTING (spec docs/specs/2026-11-16-meg-fingerprinting.md, 11.3 steps 17-19, its amendment of 2026-11-19 and section 12): pure nodes of
// nxr::graph::registry(), each a thin wrapper of compute (nxr::signal, nxr::stats); the definitions `fingerprint_band` and `fingerprint_features` wire them, io stores what they describe.
//
//   signal.hilbert             chunk       x [rows x samples] real  ->  z = x + i H x, complex, or its envelope |z| (a type III FIR Hilbert transformer, Kaiser-windowed; nxr::signal::hilbert, method Fir)
//   stats.cov                  accumulate  x [rows x samples]       ->  covariance [rows x rows] over the run's span (one block, its mean removed; nxr::stats::CovarianceEstimate): a statistic over the
//                                                                      span's time partition
//   stats.correlation          array       covariance               ->  correlation [rows x rows] (Pearson: the ratio of the covariance's entries, nxr::stats::correlation)
//   stats.spectralFeature      array       spectra (a list, one per band) + the band set  ->  [regions x bins]: each band's spectrum cut to its band, side by side (nxr::stats::spectralFeature)
//   stats.connectomeFeature    array       matrices (a list, one per band) + the band set ->  [regions x regions x bands] (nxr::stats::connectomeFeature)
//
// MEMORY: signal.hilbert holds one block of rows x (block + 2 halfLength) doubles (68 regions x (2^15 + 2 x 3066) x 8 bytes: ~21 MB at 600 Hz for a 1 Hz band); stats.cov rows^2 doubles; the
// features regions x bins (68 x 301) and regions^2 x bands (68 x 68 x 6) doubles.
// Every description carries the method's citation (producer.references).
#include "nxr/pipelines/fingerprint_steps.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/signal/opticalflow.h"
#include "nxr/stats/covariance.h"
#include "nxr/stats/identification.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;

namespace {

using RowMajorF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

const Json& references() {
  static const Json r = Json::array({
      "da Silva Castanheira J, Orozco Perez HD, Misic B, Baillet S (2021) Brief segments of neurophysiological activity enable individual differentiation. Nat Commun 12:5713, doi:10.1038/s41467-021-25895-8",
      "software: megFingerprinting v1.0.0 (06fdba8), github.com/neurohazardous/megFingerprinting (2_prerprocessing_PSD.m, 2_preprocessing_AEC_ortho.m); written from the paper and the code's documented behaviour, its code run only to make the goldens",
      "Brainstorm, release of 2018-10-12 (process_psd / bst_psd; process_aec1n / bst_connectivity: process_bandpass, hilbert, bst_corrn), the version the paper used"});
  return r;
}
ng::PortType statistic_out() {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = "statistic";
  return t;
}
const ng::PortType& streamed_series(const ng::Types& in, const char* port, const char* who) {
  const auto it = in.find(port);
  if (it == in.end()) ng::refuse(std::string(who) + ": port '" + port + "' is not given");
  const ng::PortType& t = it->second;
  if (!t.streamed() || t.axis != ng::Axis::samples) ng::refuse(std::string(who) + ": '" + port + "' is a series of samples (a streamed value)");
  if (!t.sfreq || !(*t.sfreq > 0)) ng::refuse(std::string(who) + ": '" + port + "' has no sampling rate");
  return t;
}
const ng::Array& given(const ng::Arrays& c, const std::string& port, const std::string& node) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(node + ": the values of " + port + " were not handed over");
  return it->second;
}
Eigen::MatrixXd matrix(const ng::Array& a) {
  Eigen::MatrixXd M(a.rows, a.cols);
  for (std::int64_t i = 0; i < a.rows; ++i) for (std::int64_t j = 0; j < a.cols; ++j) M(i, j) = a.at(static_cast<std::size_t>(i * a.cols + j));
  return M;
}
// the members of the band set a `bands` port carries (select.bands): name, lo, hi, in the set's order
struct Band { std::string name; double lo = 0, hi = 0; };
std::vector<Band> band_set(const ng::Types& in, const std::string& who) {
  const auto it = in.find("bands");
  if (it == in.end()) ng::refuse(who + ": bands is the band set select.bands made: wire it to this port");
  const ng::PortType& b = it->second;
  if (b.cls != ng::PortClass::constant || !b.facts.contains("set") || b.facts["set"].value("type", std::string()) != "spans" || b.facts["set"].value("unit", std::string()) != "Hz")
    ng::refuse(who + ": bands is not a set of spans in Hz (select.bands makes one)");
  std::vector<Band> out;
  for (const auto& m : b.facts["set"]["members"]) out.push_back({m["name"].get<std::string>(), m["lo"].get<double>(), m["hi"].get<double>()});
  if (out.empty()) ng::refuse(who + ": the band set has no member");
  return out;
}
// a list port's wired values (`<port>.0`, `<port>.1`, ...), one per band of the set, in its order
std::vector<std::string> listed(const ng::Types& in, const std::string& port, std::size_t bands, const std::string& who) {
  std::vector<std::string> keys;
  for (const auto& it : ng::many_inputs(in, port)) keys.push_back(it->first);
  if (keys.size() != bands) ng::refuse(who + ": " + port + " lists " + std::to_string(keys.size()) + " value(s) for the " + std::to_string(bands) + " band(s) of the set: one a band, in the set's order");
  for (const auto& k : keys) if (in.at(k).cls != ng::PortClass::constant) ng::refuse(who + ": " + k + " is a stored statistic or one made before it");
  return keys;
}
// the region axis a statistic's stored axes give (its first), else one of `size` rows
Json first_axis(const ng::Array& a, std::int64_t size, const char* name) {
  const Json axes = a.meta.value("axes", Json::array());
  if (axes.is_array() && !axes.empty() && axes[0].is_object()) {
    Json ax = axes[0];                                        // as stored: what it runs along (the partition: selection_id) kept
    ax["size"] = size;
    return ax;
  }
  return Json{{"name", name}, {"size", size}};
}

// a region map's rows named as the members of its partition (as signal.psd names them), else plain rows
Json rows_axis(const std::string& name, std::int64_t n, const std::string& partition) {
  Json a = Json{{"name", name}, {"size", n}};
  if (partition.empty()) return a;
  Json v = Json::array();
  for (std::int64_t k = 0; k < n; ++k) v.push_back(Json{{"name", "member " + std::to_string(k)}, {"member", k}});   // the partition's member number (its id differs between packages: compared across a cohort, it is left out)
  a["values"] = std::move(v);
  return a;
}

// ── signal.hilbert ──────────────────────────────────────────────────────────────────────────────────────────────
// THE ANALYTIC SIGNAL OF A BAND-PASSED SERIES, STREAMED: z = x + i (h * x), h a type III Hilbert transformer (h_k = 2 / (pi k), odd k) windowed by Kaiser(beta), of half length L chosen for the band:
// its transition at DC (and Nyquist) spans `lowest` / 2 Hz, so a series band-limited above lowest / 4 is transformed to the window's ripple (beta 8: about 1e-4). The reference transform (MATLAB's
// hilbert, Brainstorm's hilbert_fcn) is the FFT of the whole window it is handed (a circular response): equal to this one away from the window's ends, within about L samples of them it differs.
// A block is read with L samples of halo on each side; where the recording ends (or a filter before it made the span its record: 0 outside), the halo is zeros. Pointwise in the result's place: any
// cut of the recording gives the same values.
std::size_t hilbert_half_length(double sfreq, double lowest, double beta) {
  const double A = beta > 4.5513 ? beta / 0.1102 + 8.7 : 50.0;     // the attenuation a Kaiser window of shape beta gives (the inverse of kaiserord's beta for A > 50)
  const double dw = 2 * M_PI * (lowest / 2) / sfreq;
  return static_cast<std::size_t>(std::ceil((A - 7.95) / (2.285 * dw) / 2));
}
struct HilbertPrepared { std::size_t half = 0; double beta = 8; bool envelope = false; };
ng::NodeSpec hilbert() {
  ng::NodeSpec n;
  n.name = "signal.hilbert";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "a real series, rows x samples (streamed): a band-passed signal", false}};
  n.outputs = {{"z", "its analytic signal z = x + i H x (complex), or its amplitude envelope |z| (real), rows x samples", false}};
  n.parameters = {{"lowest", Json(), "the lowest frequency of the series' band (its lower passband edge): the transformer's transition at DC spans half of it", "Hz", {}, false},
                  {"beta", Json(8.0), "the Kaiser window's shape (8: a ripple of about 1e-4)", "-", {}, false},
                  {"transform", Json("analytic"), "what is handed on (the Studio chain's transform stage): analytic = z, complex | envelope = |z|, the amplitude envelope (non-linear)", "-", {"analytic", "envelope"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input, "a FIR Hilbert transformer summed in double, stored as float32 for float32 input",
                             "the transformer's half length L each side (from `lowest` and `beta`); zeros beyond the recording's ends");
  n.matches = "MATLAB hilbert / Brainstorm's hilbert_fcn of 2018-10-12 (the release's AEC: bst_connectivity), away from the window's ends; compute nxr::signal::hilbert (HilbertMethod::Fir)";
  n.description = "the analytic signal of a band-passed series: a Kaiser-windowed FIR Hilbert transformer whose length follows the band's lower edge";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed_series(in, "x", "signal.hilbert");
    if (x.value != ng::ValueType::real) ng::refuse("signal.hilbert: x is a real series");
    if (!p.contains("lowest") || !p["lowest"].is_number() || !(p["lowest"].get<double>() > 0)) ng::refuse("signal.hilbert: lowest (the band's lower edge, Hz) is positive");
    if (!(p.value("beta", 8.0) > 0)) ng::refuse("signal.hilbert: beta is positive");
    ng::Planned out;
    const std::string tr = p.value("transform", std::string("analytic"));
    if (tr != "analytic" && tr != "envelope") ng::refuse("signal.hilbert: transform is analytic or envelope, got '" + tr + "'");
    ng::PortType z = x;
    z.value = tr == "envelope" ? ng::ValueType::real : ng::ValueType::complex;
    z.nonlinear = x.nonlinear || tr == "envelope";
    out.out["z"] = z;
    out.halo = static_cast<std::int64_t>(hilbert_half_length(*x.sfreq, p["lowest"].get<double>(), p.value("beta", 8.0)));
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json& p, ng::Shared&) {
    HilbertPrepared pr;
    pr.beta = p.value("beta", 8.0);
    pr.half = hilbert_half_length(*in.at("x").sfreq, p["lowest"].get<double>(), pr.beta);
    pr.envelope = p.value("transform", std::string("analytic")) == "envelope";
    return std::any(pr);
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place&) {
    const HilbertPrepared& pr = std::any_cast<const HilbertPrepared&>(prepared);
    const ng::Block& b = in.at("x");
    const std::int64_t L = static_cast<std::int64_t>(pr.half), W = b.own() + 2 * keep, first = b.left - keep - L;   // the output's columns, and the first column read
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(b.rows, W + 2 * L);                       // zeros where the block holds nothing (the recording's ends)
    const std::int64_t a = std::max<std::int64_t>(0, first), e = std::min<std::int64_t>(b.width, first + W + 2 * L);
    if (e > a) X.middleCols(a - first, e - a) = all.middleCols(a, e - a).cast<double>();
    const Eigen::MatrixXcd Z = nxr::signal::hilbert(X, nxr::signal::HilbertMethod::Fir, pr.half, pr.beta);
    ng::Block z{b.rows, W, keep, keep, std::vector<float>(static_cast<std::size_t>(b.rows * W)), pr.envelope ? std::vector<float>() : std::vector<float>(static_cast<std::size_t>(b.rows * W))};
    for (std::int64_t r = 0; r < b.rows; ++r)
      for (std::int64_t t = 0; t < W; ++t) {
        if (pr.envelope) { z.re[static_cast<std::size_t>(r * W + t)] = static_cast<float>(std::abs(Z(r, L + t))); continue; }   // |z| in double, then stored
        z.re[static_cast<std::size_t>(r * W + t)] = static_cast<float>(Z(r, L + t).real());
        z.im[static_cast<std::size_t>(r * W + t)] = static_cast<float>(Z(r, L + t).imag());
      }
    return ng::Blocks{{"z", std::move(z)}};
  };
  return n;
}

// ── stats.cov ───────────────────────────────────────────────────────────────────────────────────────────────────
// THE COVARIANCE OF A SERIES' ROWS OVER THE RUN'S SPAN (a statistic over the span's time partition): one block, its mean removed, summed in double chunk by chunk (nxr::stats::CovarianceEstimate).
struct CovPrepared { nxr::stats::CovarianceOptions opt; std::string divisor, unit, axis, regions; double sfreq = 0; };
struct CovState { std::optional<nxr::stats::CovarianceEstimate> acc; std::int64_t first = -1, samples = 0; };
ng::NodeSpec cov() {
  ng::NodeSpec n;
  n.name = "stats.cov";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "a real series, rows x samples (streamed): every row is covered", false}};
  n.outputs = {{"covariance", "the rows' covariance over the run's span, rows x rows (a statistic)", false}};
  n.parameters = {{"remove_mean", Json(true), "remove the span's mean", "-", {}, false},
                  {"divisor", Json("samples_minus_one"), "the divisor: samples_minus_one | samples", "-", {"samples_minus_one", "samples"}, false},
                  {"axis", Json("row"), "the name of the rows' axis (region, channel, ...)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "the sums are carried across chunks in double, in time order");
  n.matches = "MATLAB cov / numpy cov of the span's samples (observations in the columns); compute nxr::stats::CovarianceEstimate";
  n.description = "the covariance of a series' rows over the run's span (one block, its mean removed)";
  n.plan = [](const ng::Types& in, const Json&) {
    const ng::PortType& x = streamed_series(in, "x", "stats.cov");
    if (x.value != ng::ValueType::real) ng::refuse("stats.cov: x is a real series (take a modulus first)");
    ng::Planned out;
    out.out["covariance"] = statistic_out();
    out.out["covariance"].rows = x.rows;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json& p, ng::Shared&) {
    CovPrepared pr;
    pr.opt.removeMean = p.value("remove_mean", true);
    pr.divisor = p.value("divisor", std::string("samples_minus_one"));
    pr.opt.divisor = pr.divisor == "samples" ? nxr::stats::CovarianceDivisor::Samples : nxr::stats::CovarianceDivisor::SamplesMinusOne;
    pr.unit = in.at("x").unit;
    pr.sfreq = *in.at("x").sfreq;
    pr.axis = p.value("axis", std::string("row"));
    const ng::PortType& x = in.at("x");
    if (x.facts.contains("operator") && x.facts["operator"].is_object() && x.facts["operator"].value("kind", std::string()) == "region map") pr.regions = x.facts["operator"].value("to_selection_id", std::string());
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(CovState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const CovPrepared& pr = std::any_cast<const CovPrepared&>(prepared);
    CovState& s = std::any_cast<CovState&>(state);
    const ng::Block& b = in.at("x");
    if (!s.acc) s.acc.emplace(static_cast<int>(b.rows), pr.opt);
    if (s.first < 0) s.first = at.first;
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    s.acc->add(Eigen::MatrixXd(all.middleCols(b.left, b.own()).cast<double>()));
    s.samples += b.own();
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const CovPrepared& pr = std::any_cast<const CovPrepared&>(prepared);
    const CovState& s = std::any_cast<const CovState&>(state);
    if (!s.acc || s.samples < 2) ng::refuse("stats.cov: the span holds fewer than 2 samples");
    const Eigen::MatrixXd C = s.acc->covariance();
    const std::int64_t R = C.rows();
    ng::Array a;
    a.rows = R; a.cols = R;
    a.data.resize(static_cast<std::size_t>(R * R));
    for (std::int64_t i = 0; i < R; ++i) for (std::int64_t j = 0; j < R; ++j) a.data[static_cast<std::size_t>(i * R + j)] = C(i, j);
    a.meta["describe"] = Json{
        {"measure", "covariance"}, {"method", "sample"}, {"label", "sample"}, {"precision", "float64"},
        {"unit", (pr.unit.empty() ? std::string("unit") : pr.unit) + "\xc2\xb2"},
        {"axes", Json::array({rows_axis(pr.axis, R, pr.regions), rows_axis(pr.axis, R, pr.regions)})},
        {"n_members", 1},
        {"options", Json{{"remove_mean", pr.opt.removeMean}, {"divisor", pr.divisor}, {"n_samples", s.samples}}},
        {"producer", Json{{"stage", "covariance"}, {"start_sample", s.first}, {"stop_sample", s.first + s.samples}, {"sfreq", pr.sfreq}}},
        {"description", "the covariance of " + std::to_string(R) + " rows over the run's span (" + std::to_string(s.samples) + " samples, one block)"}};
    return ng::Arrays{{"covariance", std::move(a)}};
  };
  return n;
}

// ── stats.correlation ───────────────────────────────────────────────────────────────────────────────────────────
// PEARSON'S CORRELATION from a covariance made before it (a ratio of the stored statistic's entries, derived after the reduction, never a reduction of ratios).
ng::NodeSpec correlation() {
  ng::NodeSpec n;
  n.name = "stats.correlation";
  n.kind = ng::Kind::array;
  n.inputs = {{"covariance", "a covariance, rows x rows (stats.cov over the span, or a stored one)", false}};
  n.outputs = {{"correlation", "Pearson's correlation of every pair of rows, rows x rows, its diagonal 1", false}};
  n.parameters = {{"measure", Json("correlation"), "the name the correlation is stored under (the amplitude envelope correlation of MEG fingerprinting: amplitude_envelope_correlation)", "-", {}, false},
                  {"warnings", Json::array(), "what the chain before it cannot resolve, recorded in the statistic's producer (producer_json.warnings)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a ratio of the covariance's entries, once a run");
  n.matches = "MATLAB corrcoef, Brainstorm's bst_corrn (of 2018-10-12), numpy corrcoef; compute nxr::stats::correlation";
  n.description = "Pearson's correlation of every pair of rows, from their covariance";
  n.plan = [](const ng::Types& in, const Json&) {
    const auto c = in.find("covariance");
    if (c == in.end() || c->second.cls != ng::PortClass::constant) ng::refuse("stats.correlation: covariance is a statistic (stats.cov's, or a stored covariance)");
    ng::Planned out;
    out.out["correlation"] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays&, const Json& p, ng::Shared&) { return std::any(std::make_pair(p.value("measure", std::string("correlation")), p.value("warnings", Json::array()))); };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const ng::Array& c = given(in, "covariance", "stats.correlation");
    Eigen::MatrixXd R;
    try { R = nxr::stats::correlation(matrix(c)); } catch (const std::invalid_argument& e) { ng::refuse(std::string("stats.correlation: ") + e.what()); }
    const std::int64_t n = R.rows();
    ng::Array a;
    a.rows = n; a.cols = n;
    a.data.resize(static_cast<std::size_t>(n * n));
    for (std::int64_t i = 0; i < n; ++i) for (std::int64_t j = 0; j < n; ++j) a.data[static_cast<std::size_t>(i * n + j)] = R(i, j);
    const Json cd = c.meta.value("describe", Json::object());
    a.meta["describe"] = Json{
        {"measure", std::any_cast<const std::pair<std::string, Json>&>(prepared).first}, {"method", "pearson"}, {"label", "pearson"}, {"precision", "float64"}, {"unit", "1"},
        {"axes", cd.value("axes", Json::array({Json{{"name", "row"}, {"size", n}}, Json{{"name", "row"}, {"size", n}}}))},
        {"n_members", 1}, {"options", Json{{"of", "covariance"}, {"covariance", cd.value("options", Json::object())}}},
        {"producer", Json{{"stage", "correlation"}, {"covariance", cd.value("producer", Json::object())}, {"references", references()}, {"warnings", std::any_cast<const std::pair<std::string, Json>&>(prepared).second}}},
        {"description", "Pearson's correlation of every pair of " + std::to_string(n) + " rows over the run's span (from their covariance)"}};
    return ng::Arrays{{"correlation", std::move(a)}};
  };
  return n;
}

// ── stats.spectralFeature ───────────────────────────────────────────────────────────────────────────────────────
struct FeaturePrepared { std::vector<Band> bands; std::vector<std::string> keys; };
ng::NodeSpec spectral_feature() {
  ng::NodeSpec n;
  n.name = "stats.spectralFeature";
  n.kind = ng::Kind::array;
  n.inputs = {{"spectra", "one spectrum a band, [regions x bins] each (the same resolution and regions), in the band set's order", false, true},
              {"bands", "the band set (select.bands): each member the bins its spectrum gives, lo to hi INCLUSIVE", false}};
  n.outputs = {{"feature", "[regions x bins kept]: each band's spectrum cut to its band, the bands side by side", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a cut of stored statistics, once a run (no value is changed)");
  n.matches = "the MEG fingerprinting release's spectral feature (68 regions x 301 bins, 0-150 Hz, its notebooks' tables; spec section 12.1); compute nxr::stats::spectralFeature";
  n.description = "the spectral feature of a person: each band's region spectrum cut to its band, the bands side by side (regions x bins)";
  n.plan = [](const ng::Types& in, const Json&) {
    const std::vector<Band> bands = band_set(in, "stats.spectralFeature");
    listed(in, "spectra", bands.size(), "stats.spectralFeature");
    ng::Planned out;
    out.out["feature"] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json&, ng::Shared&) {
    FeaturePrepared pr;
    pr.bands = band_set(in, "stats.spectralFeature");
    pr.keys = listed(in, "spectra", pr.bands.size(), "stats.spectralFeature");
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const FeaturePrepared& pr = std::any_cast<const FeaturePrepared&>(prepared);
    std::vector<Eigen::MatrixXd> spectra;
    std::vector<std::array<double, 2>> edges;
    double df = 0;
    Json inputs = Json::array();
    for (std::size_t b = 0; b < pr.keys.size(); ++b) {
      const ng::Array& s = given(in, pr.keys[b], "stats.spectralFeature");
      const Json producer = s.meta.value("producer", Json::object());
      const double d = producer.value("frequency_resolution_hz", 0.0);
      if (!(d > 0)) ng::refuse("stats.spectralFeature: " + s.meta.value("name", pr.keys[b]) + " records no frequency resolution (a Welch spectrum's producer.frequency_resolution_hz)");
      if (b == 0) df = d;
      else if (std::abs(d - df) > 1e-9 * df) ng::refuse("stats.spectralFeature: " + s.meta.value("name", pr.keys[b]) + " has a resolution of " + std::to_string(d) + " Hz, the first " + std::to_string(df) + " Hz");
      spectra.push_back(matrix(s));
      edges.push_back({pr.bands[b].lo, pr.bands[b].hi});
      inputs.push_back(Json{{"band", pr.bands[b].name}, {"lo_hz", pr.bands[b].lo}, {"hi_hz", pr.bands[b].hi}, {"spectrum", s.meta.value("name", std::string())}, {"spectrum_id", s.meta.value("id", std::string())}});
    }
    nxr::stats::SpectralFeature f;
    try { f = nxr::stats::spectralFeature(spectra, df, 0.0, edges); } catch (const std::invalid_argument& e) { ng::refuse(std::string("stats.spectralFeature: ") + e.what()); }
    const std::int64_t R = f.values.rows(), K = f.values.cols();
    ng::Array a;
    a.rows = R; a.cols = K;
    a.data.resize(static_cast<std::size_t>(R * K));
    for (std::int64_t i = 0; i < R; ++i) for (std::int64_t k = 0; k < K; ++k) a.data[static_cast<std::size_t>(i * K + k)] = f.values(i, k);
    Json freq = Json::array(), counts = Json::object();
    for (const double v : f.frequencies) freq.push_back(Json{{"name", std::to_string(v) + " Hz"}, {"hz", v}});
    for (std::size_t b = 0; b < pr.bands.size(); ++b) counts[pr.bands[b].name] = f.counts[b];
    const ng::Array& s0 = given(in, pr.keys[0], "stats.spectralFeature");
    a.meta["describe"] = Json{
        {"measure", s0.meta.value("measure", std::string("power"))}, {"method", "welch"}, {"label", "fingerprint"}, {"precision", "float64"},
        {"unit", s0.meta.value("unit", std::string())},
        {"axes", Json::array({first_axis(s0, R, "region"), Json{{"name", "frequency"}, {"size", K}, {"values", freq}}})},
        {"n_members", 1}, {"options", Json{{"bands", inputs}, {"bins", counts}, {"frequency_resolution_hz", df}, {"cut", "per band the bins lo <= f <= hi of the band's own spectrum, the bands side by side"}}},
        {"producer", Json{{"stage", "spectral feature"}, {"inputs", inputs},
                          {"inference", "the release does not hold this export; the cut (each band's spectrum cut to its band, side by side) is the only reading that reproduces its notebooks' table sizes (spec 12.1)"},
                          {"references", references()}}},
        {"description", "the spectral feature: " + std::to_string(pr.bands.size()) + " bands' region spectra cut to their bands, " + std::to_string(R) + " regions x " + std::to_string(K) + " bins"}};
    return ng::Arrays{{"feature", std::move(a)}};
  };
  return n;
}

// ── stats.connectomeFeature ─────────────────────────────────────────────────────────────────────────────────────
ng::NodeSpec connectome_feature() {
  ng::NodeSpec n;
  n.name = "stats.connectomeFeature";
  n.kind = ng::Kind::array;
  n.inputs = {{"matrices", "one [regions x regions] matrix a band (the amplitude envelope correlation), in the band set's order", false, true},
              {"bands", "the band set (select.bands): the band axis's names", false}};
  n.outputs = {{"feature", "[regions x regions x bands]: the bands' matrices stacked on a trailing band axis", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a stack of stored statistics, once a run (no value is changed)");
  n.matches = "the MEG fingerprinting release's connectome (68 x 68 a band, six bands; read with the upper triangle and its diagonal, 4_AEC_fingerprinting_*); compute nxr::stats::connectomeFeature";
  n.description = "the connectome feature of a person: the bands' region x region matrices stacked on a band axis";
  n.plan = [](const ng::Types& in, const Json&) {
    const std::vector<Band> bands = band_set(in, "stats.connectomeFeature");
    listed(in, "matrices", bands.size(), "stats.connectomeFeature");
    ng::Planned out;
    out.out["feature"] = statistic_out();
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json&, ng::Shared&) {
    FeaturePrepared pr;
    pr.bands = band_set(in, "stats.connectomeFeature");
    pr.keys = listed(in, "matrices", pr.bands.size(), "stats.connectomeFeature");
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const FeaturePrepared& pr = std::any_cast<const FeaturePrepared&>(prepared);
    std::vector<Eigen::MatrixXd> M;
    Json inputs = Json::array(), names = Json::array();
    for (std::size_t b = 0; b < pr.keys.size(); ++b) {
      const ng::Array& s = given(in, pr.keys[b], "stats.connectomeFeature");
      M.push_back(matrix(s));
      inputs.push_back(Json{{"band", pr.bands[b].name}, {"matrix", s.meta.value("name", std::string())}, {"matrix_id", s.meta.value("id", std::string())}});
      names.push_back(Json{{"name", pr.bands[b].name}});
    }
    std::vector<double> v;
    try { v = nxr::stats::connectomeFeature(M); } catch (const std::invalid_argument& e) { ng::refuse(std::string("stats.connectomeFeature: ") + e.what()); }
    const std::int64_t r = M[0].rows(), B = static_cast<std::int64_t>(M.size());
    ng::Array a;
    a.rows = r; a.cols = r * B;
    a.data = std::move(v);
    const ng::Array& s0 = given(in, pr.keys[0], "stats.connectomeFeature");
    const Json ax = first_axis(s0, r, "region");
    a.meta["shape"] = Json::array({r, r, B});
    a.meta["describe"] = Json{
        {"measure", s0.meta.value("measure", std::string("correlation"))}, {"method", "pearson"}, {"label", "fingerprint"}, {"precision", "float64"}, {"unit", "1"},
        {"axes", Json::array({ax, ax, Json{{"name", "band"}, {"size", B}, {"values", names}}})},
        {"n_members", 1}, {"options", Json{{"bands", inputs}}},
        {"producer", Json{{"stage", "connectome feature"}, {"inputs", inputs}, {"references", references()}}},
        {"description", "the connectome feature: " + std::to_string(B) + " bands' " + std::to_string(r) + " x " + std::to_string(r) + " matrices on a band axis"}};
    return ng::Arrays{{"feature", std::move(a)}};
  };
  return n;
}

}  // namespace

void register_fingerprint_feature_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {hilbert(), cov(), correlation(), spectral_feature(), connectome_feature()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
