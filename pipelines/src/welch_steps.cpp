// nxr-pipelines — THE WELCH NODES (docs/specs/2026-11-17-definition-outputs.md, section 5): the spectra as pure nodes of nxr::graph::registry(), grouped names, each a thin
// wrapper of the maths of compute/ (nxr::welch, nxr::rotary): the same function a test calls directly.
//
//   welch.psd          accumulate  x [channels x samples] (+ `exclude`: the recording's bad segments)  ->  psd [channels x bins], one-sided, described as a statistic
//   rotary.pair        chunk       x + a stored [2 x C] operator (its rows a_x, a_y)  ->  z = a_x.X + i a_y.X, one complex series (pointwise, a linear map)
//   welch.complex_psd  accumulate  z  ->  psd [1 x n], TWO-SIDED on the signed frequency line (+f: the pair turning from x towards y)
//
// A node describes what its values ARE (`Array::meta["describe"]`: measure, method, axes with the frequency line, the segment tiling, the scaling record, the unit, the options it
// resolved); the definition says where they are stored (`outputs:`) and io writes them. A node knows no id, name or path: what it needs of the operator it is handed as an array
// (the runner reads it) and as the facts of its port.
#include "nxr/pipelines/welch_steps.h"

#include <cmath>
#include <complex>
#include <optional>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/pipelines/spectral_record.h"
#include "nxr/meeg/sources.h"
#include "nxr/rotary.h"
#include "nxr/welch.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
namespace nw = nxr::welch;
using nxr::graph::Json;

namespace {
constexpr const char* kSq = "\xc2\xb2";   // ²

using RowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

nw::Window window_of(const std::string& s) {
  if (s == "hann") return nw::Window::Hann;
  if (s == "hamming") return nw::Window::Hamming;
  if (s == "blackman") return nw::Window::Blackman;
  if (s == "rectangular") return nw::Window::Rectangular;
  ng::refuse("window is hann, hamming, blackman or rectangular (got '" + s + "')");
}
nw::Detrend detrend_of(const std::string& s) {
  if (s == "none") return nw::Detrend::None;
  if (s == "constant") return nw::Detrend::Constant;
  if (s == "linear") return nw::Detrend::Linear;
  ng::refuse("detrend is none, constant or linear (got '" + s + "')");
}
nw::Scaling scaling_of(const std::string& s) {
  if (s == "density") return nw::Scaling::Density;
  if (s == "spectrum") return nw::Scaling::Spectrum;
  if (s == "chain") return nw::Scaling::Chain;
  ng::refuse("scaling is density, spectrum or chain (got '" + s + "')");
}
std::string window_name(nw::Window w) { return w == nw::Window::Hann ? "hann" : w == nw::Window::Hamming ? "hamming" : w == nw::Window::Blackman ? "blackman" : "rectangular"; }
std::string detrend_name(nw::Detrend d) { return d == nw::Detrend::None ? "none" : d == nw::Detrend::Constant ? "constant" : "linear"; }
std::string scaling_name(nw::Scaling s) { return s == nw::Scaling::Density ? "density" : s == nw::Scaling::Spectrum ? "spectrum" : "chain"; }

// the parameters, each with its meaning, default and unit (the definition's `parameters:` repeat them; `run psd --help` lists them)
std::vector<ng::ParamSpec> welch_parameters() {
  return {
      {"segment_seconds", Json(4.0), "the segment's length in seconds (under segment_rule tile it snaps to the nearest 2^k samples)", "s", {}, false},
      {"segment_samples", Json(), "the segment's length in samples (wins over segment_seconds; under tile a power of two)", "samples", {}, true},
      {"segment_level", Json(), "under segment_rule tile: the segment is 2^segment_level samples (wins over segment_seconds)", "level", {}, true},
      {"segment_rule", Json("tile"), "how the segment's length becomes samples: tile = a tile of the line (2^k samples; the span snaps to its grid) | round = round(seconds x fs) from the span's start (Brainstorm, scipy) | even = as round, made even (nxr's chain)", "-", {"tile", "round", "even"}, false},
      {"overlap", Json(0.5), "the fraction of a segment shared with the next, in [0, 1): the hop is n - floor(n x overlap)", "fraction", {}, false},
      {"window", Json("hamming"), "the taper of each segment", "-", {"hann", "hamming", "blackman", "rectangular"}, false},
      {"periodic", Json(false), "the DFT-even form of the window (scipy, nxr) rather than the symmetric one (MATLAB's hamming(n), Brainstorm)", "-", {}, false},
      {"detrend", Json("constant"), "what is removed from each segment before windowing", "-", {"none", "constant", "linear"}, false},
      {"scaling", Json("density"), "density = power per Hz | spectrum = power | chain = per sample, per bin (nxr's spectral chain)", "-", {"density", "spectrum", "chain"}, false},
      {"bad_segments", Json("exclude"), "exclude = a segment touching one of the recording's bad segments is left out | ignore = every segment is kept", "-", {"exclude", "ignore"}, false},
  };
}

std::optional<double> opt_double(const Json& p, const char* k) { return p.contains(k) && p[k].is_number() ? std::optional<double>(p[k].get<double>()) : std::nullopt; }
std::optional<std::int64_t> opt_int(const Json& p, const char* k) { return p.contains(k) && p[k].is_number() ? std::optional<std::int64_t>(p[k].get<std::int64_t>()) : std::nullopt; }

// what the parameters resolve to at a sampling rate: the Welch options and the grid the span snaps to
struct Resolved {
  nw::WelchOptions wo;
  int level = 0;
  std::string rule;
  bool exclude = true;
};
Resolved resolve(const Json& p, double sfreq) {
  Resolved r;
  r.wo.sfreq = sfreq;
  r.wo.overlap = p.value("overlap", 0.5);
  r.wo.window = window_of(p.value("window", std::string("hamming")));
  r.wo.periodic = p.value("periodic", false);
  r.wo.detrend = detrend_of(p.value("detrend", std::string("constant")));
  r.wo.scaling = scaling_of(p.value("scaling", std::string("density")));
  r.rule = p.value("segment_rule", std::string("tile"));
  const std::string bs = p.value("bad_segments", std::string("exclude"));
  if (bs != "exclude" && bs != "ignore") ng::refuse("bad_segments is exclude or ignore (got '" + bs + "')");
  r.exclude = bs == "exclude";
  const nw::Segment seg = nw::segmentFor(r.rule, opt_double(p, "segment_seconds"), opt_int(p, "segment_samples"), p.contains("segment_level") && p["segment_level"].is_number() ? std::optional<int>(p["segment_level"].get<int>()) : std::nullopt, sfreq, r.wo.overlap);
  r.wo.segmentSamples = seg.samples;
  r.level = seg.level;
  if (r.wo.segmentSamples < 2) ng::refuse("a segment holds at least 2 samples");
  return r;
}
std::int64_t hop_of(const nw::WelchOptions& wo) {
  const std::int64_t n = static_cast<std::int64_t>(wo.segmentSamples);
  return n - static_cast<std::int64_t>(std::floor(static_cast<double>(n) * wo.overlap));
}

// the own columns of a block as [rows x own] doubles
Eigen::MatrixXd own_real(const ng::Block& b) {
  const Eigen::Map<const RowMajor> all(b.re.data(), b.rows, b.width);
  return all.middleCols(b.left, b.own()).cast<double>();
}
Eigen::MatrixXcd own_complex(const ng::Block& b) {
  const Eigen::Map<const RowMajor> re(b.re.data(), b.rows, b.width), im(b.im.data(), b.rows, b.width);
  Eigen::MatrixXcd z(b.rows, b.own());
  for (Eigen::Index i = 0; i < b.rows; ++i)
    for (Eigen::Index t = 0; t < b.own(); ++t) z(i, t) = std::complex<double>(re(i, b.left + t), b.complex() ? im(i, b.left + t) : 0.0f);
  return z;
}
bool all_good(const ng::Place& at) {
  if (!at.good || at.good->empty()) return true;
  for (const auto g : *at.good) if (!g) return false;
  return true;
}

// the segment tiling a Welch run made: in samples of the recording's line, `first` the span's first sample
Json tiling_of(const nw::WelchOptions& wo, std::int64_t first, std::int64_t samples, std::int64_t segments, std::int64_t skipped) {
  return Json{{"unit", "samples"}, {"origin", first}, {"width", static_cast<std::int64_t>(wo.segmentSamples)}, {"hop", hop_of(wo)}, {"count", segments}, {"skipped", skipped}, {"end", first + samples}};
}
Json options_of(const Resolved& r) {
  return Json{{"segment_rule", r.rule}, {"segment_samples", static_cast<std::int64_t>(r.wo.segmentSamples)}, {"hop_samples", hop_of(r.wo)}, {"overlap", r.wo.overlap},
              {"window", window_name(r.wo.window)}, {"window_form", r.wo.periodic ? "periodic" : "symmetric"}, {"detrend", detrend_name(r.wo.detrend)},
              {"scaling", scaling_name(r.wo.scaling)}, {"exclude_bad_segments", r.exclude}};
}
// what a spectrum's readers ask of its provenance: the resolution, the rate, the segments, and the scaling record a conversion needs
Json producer_of(const Resolved& r, std::int64_t segments, std::int64_t skipped, std::int64_t samples, std::int64_t bad, double df, const Json& scaling) {
  return Json{{"stage", "welch"}, {"segment_samples", static_cast<std::int64_t>(r.wo.segmentSamples)}, {"window_s", static_cast<double>(r.wo.segmentSamples) / r.wo.sfreq}, {"hop_samples", hop_of(r.wo)},
              {"overlap", r.wo.overlap}, {"segments", segments}, {"segments_skipped", skipped}, {"sfreq", r.wo.sfreq}, {"frequency_resolution_hz", df},
              {"excluded", r.exclude ? "bad segments" : "none"}, {"excluded_fraction", samples > 0 ? static_cast<double>(bad) / static_cast<double>(samples) : 0.0}, {"spectral_scaling", scaling}};
}
[[noreturn]] void no_segment(const Resolved& r, std::int64_t samples, std::int64_t skipped) {
  if (samples < static_cast<std::int64_t>(r.wo.segmentSamples))
    ng::refuse("the span holds " + std::to_string(samples) + " samples, fewer than one segment (" + std::to_string(r.wo.segmentSamples) + ")");
  ng::refuse("no whole segment of " + std::to_string(r.wo.segmentSamples) + " samples is free of the recording's bad segments (" + std::to_string(skipped) + " left out)");
}

ng::PortType statistic_port() {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = "statistic";
  return t;
}
const ng::PortType& streamed_real(const ng::Types& in, const char* port, const char* who) {
  const auto it = in.find(port);
  if (it == in.end()) ng::refuse(std::string(who) + ": port '" + port + "' is not given");
  const ng::PortType& t = it->second;
  if (!t.streamed() || t.axis != ng::Axis::samples) ng::refuse(std::string(who) + ": '" + port + "' is a recording's samples (a streamed field), not " + (t.streamed() ? "frames" : "an array"));
  if (!t.sfreq || !(*t.sfreq > 0)) ng::refuse(std::string(who) + ": '" + port + "' has no sampling rate");
  return t;
}

// ── welch.psd ────────────────────────────────────────────────────────────────────────────────────────────────────

struct PsdPrepared {
  Resolved r;
  std::string unit;   // the recording's unit; empty: "channel unit"
};
struct PsdState {
  std::optional<nw::Welch> w;   // made with the first block: its rows are the channels
  std::int64_t first = -1, samples = 0, bad = 0;
};

ng::NodeSpec welch_psd() {
  ng::NodeSpec n;
  n.name = "welch.psd";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed, one chunk at a time)", false}, {"exclude", "the recording's bad segments (spans): a segment touching one is left out when bad_segments is exclude", true}};
  n.outputs = {{"psd", "the one-sided Welch power spectral density, channels x bins, a statistic over the segment partition", false}};
  n.parameters = welch_parameters();
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64,
                             "segments are carried across blocks, so blocks meet in time order; the sum over the segments is accumulated in double and finished once");
  n.matches = "scipy.signal.welch, Brainstorm's bst_psd and nxr's spectral chain (the named presets of the pipeline psd); compute nxr::welch::Welch";
  n.description = "the Welch power spectral density of every channel, streaming: segments (a tile of 2^k samples, or Brainstorm's or scipy's rule) are detrended, windowed, transformed and averaged; a segment touching a bad sample is left out";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed_real(in, "x", "welch.psd");
    if (x.value != ng::ValueType::real) ng::refuse("welch.psd: x is real (a complex series is welch.complex_psd's)");
    const Resolved r = resolve(p, *x.sfreq);
    ng::Planned out;
    out.out["psd"] = statistic_port();
    out.out["psd"].rows = x.rows;
    out.align = r.level > 0 ? (std::int64_t(1) << r.level) : 0;   // a tile segment starts on the line's grid: the span snaps to level k
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json& p, ng::Shared&) {
    const ng::PortType& x = in.at("x");
    PsdPrepared pr;
    pr.r = resolve(p, *x.sfreq);
    try { nw::Welch probe(1, pr.r.wo); } catch (const std::invalid_argument& e) { ng::refuse(e.what()); }
    pr.unit = x.unit;
    return std::any(std::move(pr));
  };
  n.start = [](const std::any& prepared) {
    (void)prepared;
    return std::any(PsdState{});
  };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const PsdPrepared& pr = std::any_cast<const PsdPrepared&>(prepared);
    PsdState& s = std::any_cast<PsdState&>(state);
    const ng::Block& b = in.at("x");
    const Eigen::MatrixXd chunk = own_real(b);
    if (s.first < 0) s.first = at.first;
    if (!s.w) s.w.emplace(static_cast<int>(b.rows), pr.r.wo);
    if (!pr.r.exclude || all_good(at)) s.w->add(chunk);
    else s.w->add(chunk, *at.good);
    s.samples += b.own();
    if (pr.r.exclude && at.good) for (const auto g : *at.good) if (!g) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const PsdPrepared& pr = std::any_cast<const PsdPrepared&>(prepared);
    const PsdState& s = std::any_cast<const PsdState&>(state);
    if (!s.w) ng::refuse("welch.psd: no sample was read");
    const nw::WelchResult res = s.w->result();
    if (res.segments == 0) no_segment(pr.r, s.samples, res.segmentsSkipped);
    const std::int64_t C = res.psd.rows(), bins = res.psd.cols();
    const double df = bins > 1 ? res.frequencies[1] - res.frequencies[0] : 0.0;
    const bool density = pr.r.wo.scaling == nw::Scaling::Density;
    const std::string base = (pr.unit.empty() ? std::string("channel unit") : pr.unit) + kSq;
    ng::Array a;
    a.rows = C; a.cols = bins;
    a.data.resize(static_cast<std::size_t>(C * bins));
    for (std::int64_t c = 0; c < C; ++c) for (std::int64_t k = 0; k < bins; ++k) a.data[static_cast<std::size_t>(c * bins + k)] = res.psd(c, k);
    Json o = options_of(pr.r);
    a.meta["describe"] = Json{
        {"measure", density ? "power_density" : "power"}, {"method", "welch"}, {"label", "welch"}, {"precision", "float32"},
        {"unit", density ? base + "/Hz" : base},
        {"axes", Json::array({Json{{"name", "channel"}, {"size", C}}, Json{{"name", "frequency"}, {"size", bins}, {"line", Json{{"spacing", df}, {"origin", 0.0}, {"unit", "Hz"}, {"dimension", "frequency"}}}}})},
        {"tiling", tiling_of(pr.r.wo, s.first, s.samples, res.segments, res.segmentsSkipped)},
        {"n_members", res.segments}, {"members_skipped", res.segmentsSkipped},
        {"scaling", welch_scaling_record(pr.r.wo, df, base)},
        {"producer", producer_of(pr.r, res.segments, res.segmentsSkipped, s.samples, s.bad, df, welch_scaling_record(pr.r.wo, df, base))},
        {"options", o},
        {"description", "the sensor PSD (Welch, " + window_name(pr.r.wo.window) + " windows of " + std::to_string(pr.r.wo.segmentSamples) + " samples, overlap " + std::to_string(pr.r.wo.overlap) +
                        (pr.r.exclude ? "; the recording's bad segments left out" : "") + ")"},
        {"bad_samples", s.bad}};
    return ng::Arrays{{"psd", std::move(a)}};
  };
  return n;
}

// ── rotary.pair ──────────────────────────────────────────────────────────────────────────────────────────────────

struct PairPrepared {
  Eigen::MatrixXd rows;                // [2 x M]: a_x, a_y
  std::vector<std::int32_t> channels;  // the recording's channels the operator's columns are over
};

ng::NodeSpec rotary_pair() {
  ng::NodeSpec n;
  n.name = "rotary.pair";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"operator", "a stored 2-row operator over the channels (its rows a_x and a_y: the plane's x and y)", false}};
  n.outputs = {{"z", "z = a_x.X + i a_y.X: one complex series, +f the pair turning from x towards y", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input, "each sample is mapped on its own: a linear map of the channels, so any block split gives the same values");
  n.linear_map = true;
  n.matches = "nxr.timefreq (the complex series of a rotary pair); compute nxr::rotary::pairSeries";
  n.description = "the complex series of a rotary pair: the operator's two rows applied to the channels, x + i y";
  n.plan = [](const ng::Types& in, const Json&) {
    const ng::PortType& x = streamed_real(in, "x", "rotary.pair");
    if (x.value != ng::ValueType::real) ng::refuse("rotary.pair: x is a real series");
    const auto op = in.find("operator");
    if (op == in.end() || op->second.cls != ng::PortClass::operator_) ng::refuse("rotary.pair: operator is a stored operator");
    ng::Planned out;
    ng::PortType z = x;
    z.value = ng::ValueType::complex;
    z.rows = 1;
    z.unit = "(operator output)";
    z.on = "pair";
    z.facts = Json::object();
    z.facts["pair"] = op->second.facts.contains("value") ? op->second.facts["value"]["row"] : Json();   // the operator's row, for the node after it to describe its spectrum
    out.out["z"] = z;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json&, ng::Shared&) {
    const auto it = constants.find("operator");
    if (it == constants.end() || it->second.meta.value("placeholder", false)) ng::refuse("rotary.pair: the operator's matrix was not handed over");
    const ng::Array& a = it->second;
    if (a.rows != 2) ng::refuse("rotary.pair: a pair is two rows (the operator has " + std::to_string(a.rows) + ")");
    PairPrepared p;
    p.rows = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(a.data.data(), a.rows, a.cols);
    for (const auto& c : a.meta.value("channels", Json::array())) p.channels.push_back(c.get<std::int32_t>());
    if (static_cast<std::int64_t>(p.channels.size()) != a.cols) ng::refuse("rotary.pair: the operator's channels (" + std::to_string(p.channels.size()) + ") are not its columns (" + std::to_string(a.cols) + ")");
    return std::any(std::move(p));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place&) {
    const PairPrepared& p = std::any_cast<const PairPrepared&>(prepared);
    const ng::Block& b = in.at("x");
    const std::int64_t kl = std::min(keep, b.left), kr = std::min(keep, b.right), w = b.own() + kl + kr, c0 = b.left - kl;
    const Eigen::Map<const RowMajor> all(b.re.data(), b.rows, b.width);
    Eigen::MatrixXd X(static_cast<Eigen::Index>(p.channels.size()), w);
    for (std::size_t i = 0; i < p.channels.size(); ++i) {
      if (p.channels[i] < 0 || p.channels[i] >= b.rows) ng::refuse("rotary.pair: a channel of the operator is outside the recording's channels");
      X.row(static_cast<Eigen::Index>(i)) = all.row(p.channels[i]).segment(c0, w).cast<double>();
    }
    const Eigen::MatrixXcd z = nxr::rotary::pairSeries(p.rows, X);
    ng::Block o{1, w, kl, kr, std::vector<float>(static_cast<std::size_t>(w)), std::vector<float>(static_cast<std::size_t>(w))};
    for (std::int64_t t = 0; t < w; ++t) { o.re[static_cast<std::size_t>(t)] = static_cast<float>(z(0, t).real()); o.im[static_cast<std::size_t>(t)] = static_cast<float>(z(0, t).imag()); }
    return ng::Blocks{{"z", std::move(o)}};
  };
  return n;
}

// ── welch.complex_psd ────────────────────────────────────────────────────────────────────────────────────────────

struct CPrepared {
  Resolved r;
  Json pair;   // the operator's row (name, id, params_json), from the port's facts
};
struct CState {
  nw::ComplexWelch w;
  std::int64_t first = -1, samples = 0, bad = 0;
};

ng::NodeSpec welch_complex_psd() {
  ng::NodeSpec n;
  n.name = "welch.complex_psd";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"z", "a complex series (one row): a rotary pair, streamed", false}, {"exclude", "the recording's bad segments (spans)", true}};
  n.outputs = {{"psd", "the two-sided Welch power spectral density of z on the signed frequency line, [1 x n], a statistic over the segment partition", false}};
  n.parameters = welch_parameters();
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64,
                             "segments are carried across blocks, so blocks meet in time order; accumulated in double and finished once");
  n.matches = "scipy.signal.welch(z, return_onesided=False), the frequencies ascending (nxr.timefreq.psd of a complex series); compute nxr::welch::ComplexWelch";
  n.description = "the two-sided Welch power spectral density of a complex series, streaming: no bin is doubled; +f is the pair turning from x towards y";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& z = streamed_real(in, "z", "welch.complex_psd");
    if (z.value != ng::ValueType::complex || z.rows != 1) ng::refuse("welch.complex_psd: z is one complex series");
    const Resolved r = resolve(p, *z.sfreq);
    ng::Planned out;
    out.out["psd"] = statistic_port();
    out.out["psd"].rows = 1;
    out.align = r.level > 0 ? (std::int64_t(1) << r.level) : 0;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json& p, ng::Shared&) {
    const ng::PortType& z = in.at("z");
    CPrepared pr;
    pr.r = resolve(p, *z.sfreq);
    pr.pair = z.facts.value("pair", Json());
    try { nw::ComplexWelch probe(1, pr.r.wo); } catch (const std::invalid_argument& e) { ng::refuse(e.what()); }
    return std::any(std::move(pr));
  };
  n.start = [](const std::any& prepared) { return std::any(CState{nw::ComplexWelch(1, std::any_cast<const CPrepared&>(prepared).r.wo)}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const CPrepared& pr = std::any_cast<const CPrepared&>(prepared);
    CState& s = std::any_cast<CState&>(state);
    const ng::Block& b = in.at("z");
    const Eigen::MatrixXcd z = own_complex(b);
    if (s.first < 0) s.first = at.first;
    if (!pr.r.exclude || all_good(at)) s.w.add(z);
    else s.w.add(z, *at.good);
    s.samples += b.own();
    if (pr.r.exclude && at.good) for (const auto g : *at.good) if (!g) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const CPrepared& pr = std::any_cast<const CPrepared&>(prepared);
    const CState& s = std::any_cast<const CState&>(state);
    const nw::ComplexWelchResult res = s.w.result();
    if (res.segments == 0) no_segment(pr.r, s.samples, res.segmentsSkipped);
    const std::int64_t bins = res.psd.cols();
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples), f0 = res.frequencies[0];
    const bool density = pr.r.wo.scaling == nw::Scaling::Density;
    const std::string base = std::string("(operator output)") + kSq;
    ng::Array a;
    a.rows = 1; a.cols = bins;
    a.data.resize(static_cast<std::size_t>(bins));
    for (std::int64_t k = 0; k < bins; ++k) a.data[static_cast<std::size_t>(k)] = res.psd(0, k);
    const std::string oname = pr.pair.is_object() ? pr.pair.value("name", std::string()) : std::string();
    Json pair_value = Json{{"name", oname}};
    Json options = options_of(pr.r);
    Json band;
    if (pr.pair.is_object() && pr.pair.contains("params_json") && pr.pair["params_json"].is_string()) {
      const Json pj = Json::parse(pr.pair["params_json"].get<std::string>(), nullptr, false);
      if (pj.is_object() && pj.contains("band")) band = pj["band"];
    }
    if (pr.pair.is_object() && pr.pair.contains("id")) pair_value["operator_id"] = pr.pair["id"];
    options["input"] = Json{{"operator", oname}, {"operator_id", pair_value.value("operator_id", Json())}, {"rows", Json::array({"x", "y"})}, {"band", band}};
    options["sides"] = "two";
    a.meta["describe"] = Json{
        {"measure", density ? "power_density" : "power"}, {"method", "welch"}, {"label", "welch"}, {"precision", "float32"},
        {"unit", density ? base + "/Hz" : base},
        {"axes", Json::array({Json{{"name", "pair"}, {"size", 1}, {"values", Json::array({pair_value})}},
                              Json{{"name", "frequency"}, {"size", bins}, {"line", Json{{"spacing", df}, {"origin", f0}, {"unit", "Hz"}, {"dimension", "frequency"}}}}})},
        {"tiling", tiling_of(pr.r.wo, s.first, s.samples, res.segments, res.segmentsSkipped)},
        {"n_members", res.segments}, {"members_skipped", res.segmentsSkipped},
        {"scaling", welch_scaling_record(pr.r.wo, df, base)},
        {"producer", producer_of(pr.r, res.segments, res.segmentsSkipped, s.samples, s.bad, df, welch_scaling_record(pr.r.wo, df, base))},
        {"options", options},
        {"description", "the rotary spectrum through " + oname + ": the two-sided Welch PSD of z = x + i y (the operator's two rows applied to the channels), +f the pair turning from x towards y (" +
                        window_name(pr.r.wo.window) + " windows of " + std::to_string(pr.r.wo.segmentSamples) + " samples, overlap " + std::to_string(pr.r.wo.overlap) + (pr.r.exclude ? "; the recording's bad segments left out" : "") + ")"},
        {"bad_samples", s.bad}};
    return ng::Arrays{{"psd", std::move(a)}};
  };
  return n;
}

// ── welch.source_band_power ─────────────────────────────────────────────────────────────────────────────────────
// The band power of a recording's SOURCES (the hand-wired `sourcepsd` it replaces): the channels' cross-spectral density accumulated per band (bins lo..hi Hz
// inclusive, mean over the band's bins), then diag(K Re(S) K^T) through the inverse kernel K, the orientations' power summed (nxr::meeg::sourcePower). With
// normalisation relative, each band over the total (all bins, summed). Exactly Welch run on K x, at the cost of channels^2 per band.
struct SourceBandPrepared {
  Resolved r;
  Eigen::MatrixXd K;
  std::vector<std::int32_t> channels;
  int orientations = 3;
  std::vector<nw::BinGroup> groups;
  Json bands = Json::array();
  bool relative = false;
  std::string kernel;
};
struct SourceBandState {
  std::optional<nw::CrossWelch> w;
  std::int64_t first = -1, samples = 0, bad = 0;
};

ng::NodeSpec welch_source_band_power() {
  ng::NodeSpec n;
  n.name = "welch.source_band_power";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"kernel", "the inverse kernel: a stored operator, sources x the recording's channels", false},
              {"exclude", "the recording's bad segments (spans): a segment touching one is left out when bad_segments is exclude", true}};
  n.outputs = {{"power", "the band power of every source, sources x bands, a statistic over the segment partition", false}};
  n.parameters = welch_parameters();
  n.parameters.push_back({"bands", Json::array(), "the bands: a list of {name, lo, hi} in Hz (a band holds the bins from lo to hi inclusive)", "Hz", {}, false});
  n.parameters.push_back({"normalisation", Json("none"), "none = the band power | relative = each band over the source's total power (all bins)", "-", {"none", "relative"}, false});
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64,
                             "segments are carried across blocks; the cross-spectra are accumulated in double and the kernel applied once at the end");
  n.matches = "Brainstorm's band power of sources (bst_psd through the kernel); meeg nxr::meeg::sourcePower";
  n.description = "the band power of the sources through an inverse kernel, from the channels' cross-spectral density per band (Welch), orientations summed";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed_real(in, "x", "welch.source_band_power");
    const auto k = in.find("kernel");
    if (k == in.end() || k->second.cls != ng::PortClass::operator_) ng::refuse("welch.source_band_power: kernel is a stored operator (an inverse kernel)");
    if (!p.contains("bands") || !p["bands"].is_array() || p["bands"].empty()) ng::refuse("welch.source_band_power: at least one band");
    const Resolved r = resolve(p, *x.sfreq);
    ng::Planned out;
    out.out["power"] = statistic_port();
    out.align = r.level > 0 ? (std::int64_t(1) << r.level) : 0;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    SourceBandPrepared pr;
    pr.r = resolve(p, *in.at("x").sfreq);
    const auto it = constants.find("kernel");
    if (it == constants.end() || it->second.meta.value("placeholder", false)) ng::refuse("welch.source_band_power: the kernel's matrix was not handed over");
    const ng::Array& a = it->second;
    pr.K = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(a.data.data(), a.rows, a.cols);
    for (const auto& c : a.meta.value("channels", Json::array())) pr.channels.push_back(c.get<std::int32_t>());
    pr.orientations = a.meta.value("to_value_type", std::string("vector3")) == "scalar" ? 1 : 3;
    pr.kernel = a.meta.value("name", std::string());
    if (pr.K.rows() % pr.orientations != 0) ng::refuse("welch.source_band_power: the kernel's rows are not a multiple of " + std::to_string(pr.orientations));
    const std::string norm = p.value("normalisation", std::string("none"));
    if (norm != "none" && norm != "relative") ng::refuse("welch.source_band_power: normalisation is none or relative");
    pr.relative = norm == "relative";
    const std::size_t nbins = pr.r.wo.segmentSamples / 2 + 1;
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    for (const auto& b : p["bands"]) {
      const std::string name = b.value("name", std::string());
      const double lo = b.value("lo", b.value("lo_hz", 0.0)), hi = b.value("hi", b.value("hi_hz", 0.0));
      if (!(hi >= lo)) ng::refuse("welch.source_band_power: band " + name + " has hi < lo");
      const double first = std::ceil(lo / df - 1e-9), last = std::floor(hi / df + 1e-9);
      if (first > last || last < 0 || first >= static_cast<double>(nbins)) ng::refuse("welch.source_band_power: band " + name + " holds no frequency bin at " + std::to_string(df) + " Hz resolution");
      pr.groups.push_back({static_cast<std::size_t>(first), static_cast<std::size_t>(std::min(last, static_cast<double>(nbins - 1))), true});
      pr.bands.push_back(Json{{"name", name}, {"lo_hz", lo}, {"hi_hz", hi}, {"bins", Json::array({pr.groups.back().first, pr.groups.back().last})}, {"function", "mean"}});
    }
    if (pr.relative) pr.groups.push_back({0, nbins - 1, false});
    try { nw::CrossWelch probe(1, pr.r.wo, pr.groups); } catch (const std::invalid_argument& e) { ng::refuse(e.what()); }
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(SourceBandState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const SourceBandPrepared& pr = std::any_cast<const SourceBandPrepared&>(prepared);
    SourceBandState& s = std::any_cast<SourceBandState&>(state);
    const ng::Block& b = in.at("x");
    const Eigen::MatrixXd all = own_real(b);
    Eigen::MatrixXd chunk(static_cast<Eigen::Index>(pr.channels.size()), all.cols());
    for (std::size_t i = 0; i < pr.channels.size(); ++i) chunk.row(static_cast<Eigen::Index>(i)) = all.row(pr.channels[i]);
    if (s.first < 0) s.first = at.first;
    if (!s.w) s.w.emplace(static_cast<int>(pr.channels.size()), pr.r.wo, pr.groups);
    if (!pr.r.exclude || all_good(at)) s.w->add(chunk); else s.w->add(chunk, *at.good);
    s.samples += b.own();
    if (pr.r.exclude && at.good) for (const auto g : *at.good) if (!g) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const SourceBandPrepared& pr = std::any_cast<const SourceBandPrepared&>(prepared);
    const SourceBandState& s = std::any_cast<const SourceBandState&>(state);
    if (!s.w) ng::refuse("welch.source_band_power: no sample was read");
    const nw::CrossWelchResult cw = s.w->result();
    if (cw.segments == 0) no_segment(pr.r, s.samples, cw.segmentsSkipped);
    const std::size_t B = pr.bands.size();
    Eigen::MatrixXd maps = nxr::meeg::sourcePower(pr.K, std::vector<Eigen::MatrixXd>(cw.S.begin(), cw.S.begin() + static_cast<std::ptrdiff_t>(B)), pr.orientations, nxr::meeg::Orientation::Norm);
    if (pr.relative) maps = nxr::meeg::relativePower(maps, nxr::meeg::sourcePower(pr.K, {cw.S.back()}, pr.orientations, nxr::meeg::Orientation::Norm).col(0));
    const std::int64_t N = maps.rows();
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    const bool density = pr.r.wo.scaling == nw::Scaling::Density;
    const std::string base = std::string("A") + kSq + "\xc2\xb7m" + kSq;
    ng::Array a;
    a.rows = N; a.cols = static_cast<std::int64_t>(B);
    a.data.resize(static_cast<std::size_t>(N) * B);
    for (std::int64_t i = 0; i < N; ++i) for (std::size_t b = 0; b < B; ++b) a.data[static_cast<std::size_t>(i) * B + b] = maps(i, static_cast<Eigen::Index>(b));
    Json entries = pr.bands;
    if (!pr.relative)
      for (std::size_t b = 0; b < B; ++b) entries[b]["spectral_scaling"] = welch_scaling_record(pr.r.wo, static_cast<double>(pr.groups[b].last - pr.groups[b].first + 1) * df, base);
    Json o = options_of(pr.r);
    o["normalisation"] = pr.relative ? "relative" : "none";
    o["relative_total"] = pr.relative ? Json("sum of the source's PSD over all bins, 0 to Nyquist") : Json();
    o["orientation_combination"] = pr.orientations == 3 ? "sum of the three orientations' power" : "fixed orientation";
    o["kernel"] = pr.kernel;
    a.meta["describe"] = Json{
        {"measure", pr.relative ? "relative_power" : density ? "power_density" : "power"}, {"method", "welch"}, {"label", "welch"}, {"precision", "float32"},
        {"unit", pr.relative ? Json() : Json(density ? base + "/Hz" : base)},
        {"axes", Json::array({Json{{"name", "vertex"}, {"size", N}}, Json{{"name", "band"}, {"size", static_cast<std::int64_t>(B)}, {"values", entries}}})},
        {"tiling", tiling_of(pr.r.wo, s.first, s.samples, cw.segments, cw.segmentsSkipped)},
        {"n_members", cw.segments}, {"members_skipped", cw.segmentsSkipped},
        {"producer", producer_of(pr.r, cw.segments, cw.segmentsSkipped, s.samples, s.bad, df, Json())},
        {"options", o},
        {"description", std::string("the ") + (pr.relative ? "relative " : "") + "band power of the sources through " + pr.kernel + ", per band"},
        {"bad_samples", s.bad}};
    return ng::Arrays{{"power", std::move(a)}};
  };
  return n;
}
}  // namespace

void register_welch_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {welch_psd(), rotary_pair(), welch_complex_psd(), welch_source_band_power()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
