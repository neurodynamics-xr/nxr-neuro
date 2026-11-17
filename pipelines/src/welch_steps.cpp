// nxr-pipelines — THE WELCH NODES (docs/specs/2026-11-17-definition-outputs.md, section 5): the spectra as pure nodes of nxr::graph::registry(), grouped names, each a thin
// wrapper of the maths of compute/ (nxr::welch, nxr::rotary): the same function a test calls directly.
//
//   welch.psd          accumulate  x [channels x samples] (+ `exclude`: the recording's bad segments)  ->  psd [channels x bins], one-sided, described as a statistic
//   rotary.pair        chunk       x + a stored [2 x C] operator (its rows a_x, a_y)  ->  z = a_x.X + i a_y.X, one complex series (pointwise, a linear map)
//   fft.spectrum       chunk       x -> the block's one-sided spectrum, complex channels x bins (axis block)
//   measure.band_amplitude  accumulate  spectrum + bands (Hz) -> the bands as spans, the frequency line, the mean |X| per band
//   welch.complex_psd  accumulate  z  ->  psd [1 x n], TWO-SIDED on the signed frequency line (+f: the pair turning from x towards y)
//
// A node describes what its values ARE (`Array::meta["describe"]`: measure, method, axes with the frequency line, the segment tiling, the scaling record, the unit, the options it
// resolved); the definition says where they are stored (`outputs:`) and io writes them. A node knows no id, name or path: what it needs of the operator it is handed as an array
// (the runner reads it) and as the facts of its port.
#include "nxr/pipelines/welch_steps.h"

#include <cmath>
#include <complex>
#include <optional>
#include <set>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/pipelines/spectral_record.h"
#include <algorithm>
#include <array>
#include <limits>
#include "nxr/meeg/sources.h"
#include "nxr/meeg/tensor.h"
#include "nxr/rotary.h"
#include "nxr/welch.h"
#include "nxr/fft.h"
#include "nxr/band_amplitude.h"
#include "nxr/selection/selection.h"

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
  std::string kernel, row_axis = "vertex", unit = std::string("A\xc2\xb2\xc2\xb7m\xc2\xb2");
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
  n.parameters.push_back({"row_axis", Json("vertex"), "the name of the rows' axis (vertex for a kernel's sources; mode_part for the fused Hodge-Dirac weights)", "-", {}, false});
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
    pr.row_axis = p.value("row_axis", std::string("vertex"));
    if (a.meta.contains("unit") && a.meta["unit"].is_string()) pr.unit = a.meta["unit"].get<std::string>() + kSq;
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
    const std::string base = pr.unit;
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
        {"axes", Json::array({Json{{"name", pr.row_axis}, {"size", N}}, Json{{"name", "band"}, {"size", static_cast<std::int64_t>(B)}, {"values", entries}}})},
        {"tiling", tiling_of(pr.r.wo, s.first, s.samples, cw.segments, cw.segmentsSkipped)},
        {"n_members", cw.segments}, {"members_skipped", cw.segmentsSkipped},
        {"producer", producer_of(pr.r, cw.segments, cw.segmentsSkipped, s.samples, s.bad, df, Json())},
        {"options", o},
        {"description", std::string("the ") + (pr.relative ? "relative " : "") + "band power of the rows of " + pr.kernel + " (" + pr.row_axis + "), per band"},
        {"bad_samples", s.bad}};
    return ng::Arrays{{"power", std::move(a)}};
  };
  return n;
}

// ── welch.source_tensor ─────────────────────────────────────────────────────────────────────────────────────────
// The band cross-spectral TENSOR of the current at every source (the hand-wired `sourcetensor` it replaces): the channels' cross-spectrum and quadrature
// accumulated per band, then T = K_v S K_v^H per source (3 x 3 Hermitian, the kernel's x y z), and from it the tangential Stokes parameters in the canonical frames,
// the power along the normal and its share of the trace. Four statistics, sources x bands, over one segment partition.
struct SourceTensorPrepared {
  SourceBandPrepared b;
  Eigen::MatrixXd e1, e2, normal;   // [N x 3]; e1/e2 NaN at a singular vertex; normal: e1 x e2, or the area-weighted mesh normal where the frames are singular
  std::int64_t singular = 0;
  std::string frames;
};

ng::NodeSpec welch_source_tensor() {
  ng::NodeSpec n = welch_source_band_power();
  n.name = "welch.source_tensor";
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"kernel", "a free-orientation inverse kernel: 3 rows a source x the recording's channels", false},
              {"frames", "the canonical frames of the kernel's surface: a field, vertices x 9 (e1, e2, n)", false},
              {"surface", "the kernel's surface: its normals where the frames are singular", false},
              {"exclude", "the recording's bad segments (spans): a segment touching one is left out when bad_segments is exclude", true}};
  n.outputs = {{"tensor", "the band cross-spectral tensor of the current (hermitian3), sources x bands", false},
               {"stokes", "the tangential Stokes parameters [S0 S1 S2 S3] in the canonical frames (quaternion), sources x bands", false},
               {"normal_power", "the power along the normal, sources x bands", false},
               {"normal_share", "the normal power over the trace, sources x bands", false}};
  n.parameters.erase(std::remove_if(n.parameters.begin(), n.parameters.end(), [](const auto& q) { return q.name == "normalisation"; }), n.parameters.end());
  n.matches = "meeg nxr::meeg::sourceTensor, stokes, normalPower";
  n.description = "the band cross-spectral tensor of the current through a free-orientation kernel, its tangential Stokes parameters, its normal power and share";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed_real(in, "x", "welch.source_tensor");
    const auto k = in.find("kernel");
    if (k == in.end() || k->second.cls != ng::PortClass::operator_) ng::refuse("welch.source_tensor: kernel is a stored operator (an inverse kernel)");
    if (!p.contains("bands") || !p["bands"].is_array() || p["bands"].empty()) ng::refuse("welch.source_tensor: at least one band");
    const Resolved r = resolve(p, *x.sfreq);
    ng::Planned out;
    for (const char* o : {"tensor", "stokes", "normal_power", "normal_share"}) out.out[o] = statistic_port();
    out.align = r.level > 0 ? (std::int64_t(1) << r.level) : 0;
    return out;
  };
  const auto base_prepare = n.prepare;
  n.prepare = [base_prepare](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared& sh) {
    Json q = p; q["normalisation"] = "none";
    SourceTensorPrepared pr;
    pr.b = std::any_cast<SourceBandPrepared>(base_prepare(in, constants, q, sh));
    if (pr.b.orientations != 3) ng::refuse("welch.source_tensor: " + pr.b.kernel + " is a fixed-orientation kernel (one orientation a source): it has no tensor");
    const std::int64_t N = pr.b.K.rows() / 3;
    const auto fr = constants.find("frames"), su = constants.find("surface");
    if (fr == constants.end() || su == constants.end()) ng::refuse("welch.source_tensor: the frames and the surface were not handed over");
    const ng::Array& F = fr->second;
    if (static_cast<std::int64_t>(F.data.size()) != N * 9) ng::refuse("welch.source_tensor: the frames hold " + std::to_string(F.data.size()) + " values for " + std::to_string(N) + " sources (9 each)");
    const ng::Array& S = su->second;
    if (S.rows != N) ng::refuse("welch.source_tensor: the surface has " + std::to_string(S.rows) + " vertices, the kernel " + std::to_string(N) + " sources");
    pr.frames = F.meta.value("name", std::string());
    Eigen::MatrixXd mesh = Eigen::MatrixXd::Zero(N, 3);
    const std::vector<std::int32_t> faces = S.meta.value("faces", std::vector<std::int32_t>());
    for (std::size_t f = 0; f + 2 < faces.size(); f += 3) {
      Eigen::Vector3d v[3];
      for (int j = 0; j < 3; ++j) for (int c = 0; c < 3; ++c) v[j][c] = S.data[static_cast<std::size_t>(faces[f + j]) * 3 + c];
      const Eigen::Vector3d fn = (v[1] - v[0]).cross(v[2] - v[0]);
      for (int j = 0; j < 3; ++j) mesh.row(faces[f + j]) += fn.transpose();
    }
    pr.e1.resize(N, 3); pr.e2.resize(N, 3); pr.normal.resize(N, 3);
    for (std::int64_t i = 0; i < N; ++i) {
      for (int c = 0; c < 3; ++c) { pr.e1(i, c) = F.data[static_cast<std::size_t>(i * 9 + c)]; pr.e2(i, c) = F.data[static_cast<std::size_t>(i * 9 + 3 + c)]; }
      if (std::isfinite(pr.e1(i, 0)) && std::isfinite(pr.e2(i, 0))) pr.normal.row(i) = Eigen::Vector3d(pr.e1.row(i)).cross(Eigen::Vector3d(pr.e2.row(i))).transpose();
      else { pr.normal.row(i) = mesh.row(i).normalized(); ++pr.singular; }
    }
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(SourceBandState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const SourceBandPrepared& pr = std::any_cast<const SourceTensorPrepared&>(prepared).b;
    SourceBandState& s = std::any_cast<SourceBandState&>(state);
    const ng::Block& b = in.at("x");
    const Eigen::MatrixXd all = own_real(b);
    Eigen::MatrixXd chunk(static_cast<Eigen::Index>(pr.channels.size()), all.cols());
    for (std::size_t i = 0; i < pr.channels.size(); ++i) chunk.row(static_cast<Eigen::Index>(i)) = all.row(pr.channels[i]);
    if (s.first < 0) s.first = at.first;
    if (!s.w) s.w.emplace(static_cast<int>(pr.channels.size()), pr.r.wo, pr.groups, true);
    if (!pr.r.exclude || all_good(at)) s.w->add(chunk); else s.w->add(chunk, *at.good);
    s.samples += b.own();
    if (pr.r.exclude && at.good) for (const auto g : *at.good) if (!g) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const SourceTensorPrepared& tp = std::any_cast<const SourceTensorPrepared&>(prepared);
    const SourceBandPrepared& pr = tp.b;
    const SourceBandState& s = std::any_cast<const SourceBandState&>(state);
    if (!s.w) ng::refuse("welch.source_tensor: no sample was read");
    const nw::CrossWelchResult cw = s.w->result();
    if (cw.segments == 0) no_segment(pr.r, s.samples, cw.segmentsSkipped);
    const std::size_t B = pr.bands.size();
    const std::int64_t N = pr.K.rows() / 3;
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    const bool density = pr.r.wo.scaling == nw::Scaling::Density;
    const std::string base = std::string("A") + kSq + "\xc2\xb7m" + kSq;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    ng::Array T, Q, P, H;
    T.rows = Q.rows = P.rows = H.rows = N;
    T.cols = static_cast<std::int64_t>(B) * 18; Q.cols = static_cast<std::int64_t>(B) * 4; P.cols = H.cols = static_cast<std::int64_t>(B);
    T.data.resize(static_cast<std::size_t>(N * T.cols)); Q.data.resize(static_cast<std::size_t>(N * Q.cols)); P.data.resize(static_cast<std::size_t>(N) * B); H.data.resize(static_cast<std::size_t>(N) * B);
    Json entries = pr.bands;
    for (std::size_t b = 0; b < B; ++b) {
      const nxr::meeg::SourceTensors t = nxr::meeg::sourceTensor(pr.K, cw.S[b], cw.Sq[b]);
      entries[b]["spectral_scaling"] = welch_scaling_record(pr.r.wo, static_cast<double>(pr.groups[b].last - pr.groups[b].first + 1) * df, base);
      for (std::int64_t v = 0; v < N; ++v) {
        const std::size_t vb = static_cast<std::size_t>(v) * B + b;
        for (int k = 0; k < 9; ++k) { T.data[18 * vb + 2 * k] = t.reOf(v)[k]; T.data[18 * vb + 2 * k + 1] = t.imOf(v)[k]; }
        const bool ok = std::isfinite(tp.e1(v, 0)) && std::isfinite(tp.e2(v, 0));
        std::array<double, 3> e1{}, e2{}, nn{};
        for (int k = 0; k < 3; ++k) { e1[k] = tp.e1(v, k); e2[k] = tp.e2(v, k); nn[k] = tp.normal(v, k); }
        const nxr::meeg::Stokes st = ok ? nxr::meeg::stokes(t.reOf(v), t.imOf(v), e1, e2) : nxr::meeg::Stokes{nan, nan, nan, nan};
        Q.data[4 * vb] = st.s0; Q.data[4 * vb + 1] = st.s1; Q.data[4 * vb + 2] = st.s2; Q.data[4 * vb + 3] = st.s3;
        const double np = nxr::meeg::normalPower(t.reOf(v), nn), tr = nxr::meeg::tracePower(t.reOf(v));
        P.data[vb] = np;
        H.data[vb] = tr > 0 ? np / tr : nan;
      }
    }
    Json o = options_of(pr.r);
    o["kernel"] = pr.kernel; o["frames"] = tp.frames;
    const Json axes = Json::array({Json{{"name", "vertex"}, {"size", N}}, Json{{"name", "band"}, {"size", static_cast<std::int64_t>(B)}, {"values", entries}}});
    Json producer = producer_of(pr.r, cw.segments, cw.segmentsSkipped, s.samples, s.bad, df, Json());
    producer["singular_vertices"] = tp.singular;
    auto describe = [&](ng::Array& a, const char* measure, const char* value_type, std::vector<std::int64_t> shape, const Json& unit, const std::string& what) {
      a.meta["describe"] = Json{{"measure", measure}, {"method", "welch"}, {"label", "welch"}, {"precision", "float32"}, {"value_type", value_type}, {"shape", shape}, {"unit", unit},
                                {"axes", axes}, {"tiling", tiling_of(pr.r.wo, s.first, s.samples, cw.segments, cw.segmentsSkipped)}, {"n_members", cw.segments},
                                {"members_skipped", cw.segmentsSkipped}, {"producer", producer}, {"options", o}, {"bad_samples", s.bad},
                                {"description", what + " of the sources through " + pr.kernel + ", per band"}};
    };
    const Json u = density ? Json(base + "/Hz") : Json(base);
    const std::int64_t Bi = static_cast<std::int64_t>(B);
    describe(T, density ? "cross_spectral_density" : "cross_spectrum", "hermitian3", {N, Bi, 3, 3, 2}, u, "the band cross-spectral tensor of the current");
    describe(Q, "stokes", "quaternion", {N, Bi, 4}, u, "the tangential Stokes parameters (canonical frames: e1 north, e2 west)");
    describe(P, density ? "power_density" : "power", "scalar", {N, Bi}, u, "the power along the normal");
    describe(H, "share", "scalar", {N, Bi}, Json(), "the share of the power along the normal (normal power over the trace)");
    return ng::Arrays{{"tensor", std::move(T)}, {"stokes", std::move(Q)}, {"normal_power", std::move(P)}, {"normal_share", std::move(H)}};
  };
  return n;
}

// ── welch.cross_spectrum ────────────────────────────────────────────────────────────────────────────────────────
// THE SENSOR CROSS-SPECTRUM PER BAND (the hand-wired `crossspectrum` it replaces): C[a, b] = <X_a conj(X_b)> over the segments, the mean over the band's bins, on the channels of
// an operator (channels_of: a kernel's) or the good channels of the imaging types. One statistic, band x channel x channel (complex); with resolve bins also every bin of each
// band, frequency x channel x channel, its frequency axis naming each band's bins (the band statistic is the mean of them).
struct CrossPrepared {
  Resolved r;
  std::vector<std::int32_t> channels;
  std::vector<nw::BinGroup> groups;
  std::size_t band_groups = 0;
  Json bands = Json::array();
  bool bins = false;
  std::string unit;
};

std::vector<std::int32_t> picked_channels(const ng::PortType& x, const ng::Arrays& constants, const Json& p, const char* who) {
  if (const auto it = constants.find("channels_of"); it != constants.end() && !it->second.meta.value("placeholder", false))
    return it->second.meta.value("channels", std::vector<std::int32_t>());
  if (!x.facts.contains("channels")) ng::refuse(std::string(who) + ": the recording's channels (types, quality) were not handed over");
  const Json& c = x.facts["channels"];
  std::set<std::string> want;
  for (const auto& t : p.value("channel_types", Json::array())) want.insert(t.get<std::string>());
  std::vector<std::int32_t> out;
  for (std::size_t i = 0; i < c["types"].size(); ++i)
    if ((want.empty() || want.count(c["types"][i].get<std::string>())) && (i >= c["quality"].size() || c["quality"][i].get<std::string>() == "good")) out.push_back(static_cast<std::int32_t>(i));
  if (out.empty()) ng::refuse(std::string(who) + ": no good channel of the types asked");
  return out;
}

ng::NodeSpec welch_cross_spectrum() {
  ng::NodeSpec n;
  n.name = "welch.cross_spectrum";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed); its channels' types and quality are the port's facts", false},
              {"channels_of", "an operator whose channels (its from_selection) are the ones covered, in order (a kernel's); absent: the good channels of channel_types", true},
              {"exclude", "the recording's bad segments (spans): a segment touching one is left out when bad_segments is exclude", true}};
  n.outputs = {{"channels", "the channels covered, in order (indices of the recording's channels)", false},
               {"spectrum", "the cross-spectrum per band, band x channel x channel (complex), a statistic over the segment partition", false},
               {"bins", "resolve bins: every bin of each band, frequency x channel x channel (complex)", true}};
  n.parameters = welch_parameters();
  n.parameters.push_back({"bands", Json::array(), "the bands: a list of {name, lo, hi} in Hz (a band holds the bins from lo to hi inclusive)", "Hz", {}, false});
  n.parameters.push_back({"channel_types", Json::array({"MEG", "MEG MAG", "MEG GRAD", "EEG", "SEEG", "ECOG"}), "the channel types covered when channels_of is absent (good channels only)", "-", {}, false});
  n.parameters.push_back({"resolve", Json("band"), "band = each band one member (its bins averaged) | bins = also every bin of each band", "-", {"band", "bins"}, false});
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "segments are carried across blocks; the cross-spectra are accumulated in double");
  n.matches = "MNE csd_array_fourier / scipy.signal.csd (Welch, averaged over the band's bins); meeg nxr::welch::CrossWelch";
  n.description = "the sensor cross-spectral density per band (Welch), on a kernel's channels or the good imaging channels";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed_real(in, "x", "welch.cross_spectrum");
    if (!p.contains("bands") || !p["bands"].is_array() || p["bands"].empty()) ng::refuse("welch.cross_spectrum: at least one band");
    const Resolved r = resolve(p, *x.sfreq);
    ng::Planned out;
    out.out["channels"] = ng::PortType{};
    out.out["channels"].cls = ng::PortClass::constant; out.out["channels"].facts["cls"] = "selection";
    out.out["spectrum"] = statistic_port();
    out.out["bins"] = statistic_port();
    out.align = r.level > 0 ? (std::int64_t(1) << r.level) : 0;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    CrossPrepared pr;
    const ng::PortType& x = in.at("x");
    pr.r = resolve(p, *x.sfreq);
    pr.channels = picked_channels(x, constants, p, "welch.cross_spectrum");
    pr.unit = x.unit.empty() ? std::string("channel unit") : x.unit;
    const std::size_t nbins = pr.r.wo.segmentSamples / 2 + 1;
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    for (const auto& b : p["bands"]) {
      const std::string name = b.value("name", std::string());
      const double lo = b.value("lo", b.value("lo_hz", 0.0)), hi = b.value("hi", b.value("hi_hz", 0.0));
      if (!(hi >= lo)) ng::refuse("welch.cross_spectrum: band " + name + " has hi < lo");
      const double first = std::ceil(lo / df - 1e-9), last = std::floor(hi / df + 1e-9);
      if (first > last || last < 0 || first >= static_cast<double>(nbins)) ng::refuse("welch.cross_spectrum: band " + name + " holds no frequency bin at " + std::to_string(df) + " Hz resolution");
      pr.groups.push_back({static_cast<std::size_t>(first), static_cast<std::size_t>(std::min(last, static_cast<double>(nbins - 1))), true});
      pr.bands.push_back(Json{{"name", name}, {"lo_hz", lo}, {"hi_hz", hi}, {"bins", Json::array({pr.groups.back().first, pr.groups.back().last})}, {"function", "mean"}});
    }
    pr.band_groups = pr.groups.size();
    pr.bins = p.value("resolve", std::string("band")) == "bins";
    if (pr.bins) for (std::size_t b = 0; b < pr.band_groups; ++b) for (std::size_t k = pr.groups[b].first; k <= pr.groups[b].last; ++k) pr.groups.push_back({k, k, true});
    try { nw::CrossWelch probe(1, pr.r.wo, pr.groups, true); } catch (const std::invalid_argument& e) { ng::refuse(e.what()); }
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(SourceBandState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const CrossPrepared& pr = std::any_cast<const CrossPrepared&>(prepared);
    SourceBandState& s = std::any_cast<SourceBandState&>(state);
    const ng::Block& b = in.at("x");
    const Eigen::MatrixXd all = own_real(b);
    Eigen::MatrixXd chunk(static_cast<Eigen::Index>(pr.channels.size()), all.cols());
    for (std::size_t i = 0; i < pr.channels.size(); ++i) chunk.row(static_cast<Eigen::Index>(i)) = all.row(pr.channels[i]);
    if (s.first < 0) s.first = at.first;
    if (!s.w) s.w.emplace(static_cast<int>(pr.channels.size()), pr.r.wo, pr.groups, true);
    if (!pr.r.exclude || all_good(at)) s.w->add(chunk); else s.w->add(chunk, *at.good);
    s.samples += b.own();
    if (pr.r.exclude && at.good) for (const auto g : *at.good) if (!g) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const CrossPrepared& pr = std::any_cast<const CrossPrepared&>(prepared);
    const SourceBandState& s = std::any_cast<const SourceBandState&>(state);
    if (!s.w) ng::refuse("welch.cross_spectrum: no sample was read");
    const nw::CrossWelchResult cw = s.w->result();
    if (cw.segments == 0) no_segment(pr.r, s.samples, cw.segmentsSkipped);
    const std::int64_t M = static_cast<std::int64_t>(pr.channels.size()), B = static_cast<std::int64_t>(pr.band_groups);
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    const bool density = pr.r.wo.scaling == nw::Scaling::Density;
    const std::string sq = pr.unit + kSq;
    const Json unit = density ? Json(sq + "/Hz") : Json(sq);
    Json o = options_of(pr.r);
    o["resolve"] = pr.bins ? "bins" : "band";
    const Json producer = producer_of(pr.r, cw.segments, cw.segmentsSkipped, s.samples, s.bad, df, welch_scaling_record(pr.r.wo, df, sq));
    const Json tiling = tiling_of(pr.r.wo, s.first, s.samples, cw.segments, cw.segmentsSkipped);
    ng::Array ch;
    ch.rows = M; ch.cols = 1;
    for (const auto c : pr.channels) ch.data.push_back(c);
    ch.meta["describe"] = Json{{"description", "the " + std::to_string(M) + " channels the cross-spectrum covers, in order"}};
    auto fill = [&](ng::Array& a, std::size_t g0, std::int64_t G) {
      a.rows = G; a.cols = M * M * 2;
      a.data.resize(static_cast<std::size_t>(G * M * M * 2));
      for (std::int64_t g = 0; g < G; ++g)
        for (std::int64_t i = 0; i < M; ++i)
          for (std::int64_t j = 0; j < M; ++j) {
            const std::size_t at = static_cast<std::size_t>(((g * M + i) * M + j) * 2);
            a.data[at] = cw.S[g0 + static_cast<std::size_t>(g)](i, j);
            a.data[at + 1] = cw.Sq[g0 + static_cast<std::size_t>(g)](i, j);
          }
    };
    Json entries = pr.bands;
    for (std::int64_t b = 0; b < B; ++b)
      entries[static_cast<std::size_t>(b)]["spectral_scaling"] = welch_scaling_record(pr.r.wo, static_cast<double>(pr.groups[b].last - pr.groups[b].first + 1) * df, sq);
    ng::Array S;
    fill(S, 0, B);
    S.meta["describe"] = Json{{"measure", density ? "cross_spectral_density" : "cross_spectrum"}, {"method", "welch"}, {"label", "welch"}, {"precision", "float64"}, {"value_type", "complex"},
                              {"shape", Json::array({B, M, M, 2})}, {"unit", unit},
                              {"axes", Json::array({Json{{"name", "band"}, {"size", B}, {"values", entries}}, Json{{"name", "channel"}, {"size", M}}, Json{{"name", "channel"}, {"size", M}}})},
                              {"tiling", tiling}, {"n_members", cw.segments}, {"members_skipped", cw.segmentsSkipped}, {"producer", producer}, {"options", o}, {"bad_samples", s.bad},
                              {"description", "the sensor cross-spectral density per band on " + std::to_string(M) + " channels (the mean over each band's bins)"}};
    ng::Arrays out{{"channels", std::move(ch)}, {"spectrum", std::move(S)}};
    if (pr.bins) {
      const std::int64_t F = static_cast<std::int64_t>(pr.groups.size() - pr.band_groups);
      Json hz = Json::array(), spans = Json::array();
      std::int64_t first = 0;
      for (std::int64_t b = 0; b < B; ++b) {
        const std::int64_t count = static_cast<std::int64_t>(pr.groups[b].last - pr.groups[b].first + 1);
        for (std::size_t k = pr.groups[b].first; k <= pr.groups[b].last; ++k) hz.push_back(static_cast<double>(k) * df);
        spans.push_back(Json{{"name", pr.bands[b]["name"]}, {"lo_hz", pr.bands[b]["lo_hz"]}, {"hi_hz", pr.bands[b]["hi_hz"]}, {"first", first}, {"count", count}});
        first += count;
      }
      ng::Array Bn;
      fill(Bn, pr.band_groups, F);
      Json ob = o; ob["band_function"] = "none";
      Bn.meta["describe"] = Json{{"measure", density ? "cross_spectral_density" : "cross_spectrum"}, {"method", "welch"}, {"label", "welch"}, {"precision", "float64"}, {"value_type", "complex"},
                                 {"shape", Json::array({F, M, M, 2})}, {"unit", unit}, {"scaling", welch_scaling_record(pr.r.wo, df, sq)},
                                 {"axes", Json::array({Json{{"name", "frequency"}, {"size", F}, {"unit", "Hz"}, {"values", hz}, {"bands", spans}}, Json{{"name", "channel"}, {"size", M}}, Json{{"name", "channel"}, {"size", M}}})},
                                 {"tiling", tiling}, {"n_members", cw.segments}, {"members_skipped", cw.segmentsSkipped}, {"producer", producer}, {"options", ob},
                                 {"description", "the sensor cross-spectral density bin by bin within each band, on " + std::to_string(M) + " channels"}};
      out["bins"] = std::move(Bn);
    }
    return out;
  };
  return n;
}

// ── welch.mode_cross_spectra ────────────────────────────────────────────────────────────────────────────────────
// THE CROSS-SPECTRA OF THE ROWS OF A KERNEL, BIN BY BIN (the Hodge spectra of `speedspectrum`): the rows (all, or the flow or current half of the fused Hodge-Dirac weights)
// applied to each chunk, then Welch per bin of [fmin, fmax] with the quadrature part: S[b] (real) and Sq[b] (imaginary), [rows x rows] per bin. Handed on (not stored): a node
// after the loop reads them.
struct ModePrepared { Resolved r; Eigen::MatrixXd B; std::vector<std::int32_t> channels; std::vector<nw::BinGroup> groups; };
ng::NodeSpec welch_mode_cross_spectra() {
  ng::NodeSpec n;
  n.name = "welch.mode_cross_spectra";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"weights", "an operator from the recording's channels: its rows are the series (the fused Hodge-Dirac weights)", false},
              {"exclude", "the recording's bad segments (spans): a segment touching one is left out when bad_segments is exclude", true}};
  n.outputs = {{"spectra", "per bin of [fmin, fmax]: the rows' cross-spectrum, bins x rows x rows x 2 (real, imaginary), with the bins' frequencies", false}};
  n.parameters = welch_parameters();
  n.parameters.push_back({"fmin", Json(1.0), "the lowest frequency kept (the first bin at or above it, never the DC bin)", "Hz", {}, false});
  n.parameters.push_back({"fmax", Json(40.0), "the highest frequency kept (never the Nyquist bin)", "Hz", {}, false});
  n.parameters.push_back({"rows", Json("flow"), "the weights' rows used: flow (the second half: the modes' flow coefficients) | current (the first half) | all", "-", {"flow", "current", "all"}, false});
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "segments carried across blocks; the cross-spectra accumulated in double");
  n.matches = "nxr.spectral.chain mode cross-spectra (CrossWelch on the fused weights b, one-bin groups, Re and Im)";
  n.description = "the cross-spectra of a kernel's rows applied to a recording, bin by bin, with their quadrature part";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed_real(in, "x", "welch.mode_cross_spectra");
    if (!in.count("weights")) ng::refuse("welch.mode_cross_spectra: weights is not given");
    const Resolved r = resolve(p, *x.sfreq);
    ng::Planned out;
    out.out["spectra"] = statistic_port();
    out.align = r.level > 0 ? (std::int64_t(1) << r.level) : 0;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ModePrepared pr;
    pr.r = resolve(p, *in.at("x").sfreq);
    const auto it = c.find("weights");
    if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse("welch.mode_cross_spectra: the weights were not handed over");
    const ng::Array& W = it->second;
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> all(W.data.data(), W.rows, W.cols);
    const std::string rows = p.value("rows", std::string("flow"));
    pr.B = rows == "all" ? Eigen::MatrixXd(all) : rows == "current" ? Eigen::MatrixXd(all.topRows(W.rows / 2)) : Eigen::MatrixXd(all.bottomRows(W.rows / 2));
    pr.channels = W.meta.value("channels", std::vector<std::int32_t>());
    if (static_cast<std::int64_t>(pr.channels.size()) != W.cols) ng::refuse("welch.mode_cross_spectra: the weights' channels do not fit their columns");
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    const std::size_t nbins = pr.r.wo.segmentSamples / 2 + 1;
    const double fmin = p.value("fmin", 1.0), fmax = p.value("fmax", 40.0);
    if (!(fmin >= 0) || !(fmax > fmin)) ng::refuse("welch.mode_cross_spectra: 0 <= fmin < fmax");
    const double first = std::max(1.0, std::ceil(fmin / df - 1e-9)), last = std::min(static_cast<double>(nbins) - 2, std::floor(fmax / df + 1e-9));
    if (first > last) ng::refuse("welch.mode_cross_spectra: the range holds no frequency bin at " + std::to_string(df) + " Hz resolution");
    for (std::size_t b = static_cast<std::size_t>(first); b <= static_cast<std::size_t>(last); ++b) pr.groups.push_back({b, b, true});
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(SourceBandState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const ModePrepared& pr = std::any_cast<const ModePrepared&>(prepared);
    SourceBandState& s = std::any_cast<SourceBandState&>(state);
    const ng::Block& b = in.at("x");
    const Eigen::MatrixXd all = own_real(b);
    Eigen::MatrixXd chunk(static_cast<Eigen::Index>(pr.channels.size()), all.cols());
    for (std::size_t i = 0; i < pr.channels.size(); ++i) chunk.row(static_cast<Eigen::Index>(i)) = all.row(pr.channels[i]);
    const Eigen::MatrixXd modes = pr.B * chunk;
    if (s.first < 0) s.first = at.first;
    if (!s.w) s.w.emplace(static_cast<int>(pr.B.rows()), pr.r.wo, pr.groups, true);
    if (!pr.r.exclude || all_good(at)) s.w->add(modes); else s.w->add(modes, *at.good);
    s.samples += b.own();
    if (pr.r.exclude && at.good) for (const auto g : *at.good) if (!g) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const ModePrepared& pr = std::any_cast<const ModePrepared&>(prepared);
    const SourceBandState& s = std::any_cast<const SourceBandState&>(state);
    if (!s.w) ng::refuse("welch.mode_cross_spectra: no sample was read");
    const nw::CrossWelchResult cw = s.w->result();
    if (cw.segments == 0) no_segment(pr.r, s.samples, cw.segmentsSkipped);
    const std::int64_t K = pr.B.rows(), Bn = static_cast<std::int64_t>(pr.groups.size());
    const double df = pr.r.wo.sfreq / static_cast<double>(pr.r.wo.segmentSamples);
    ng::Array a;
    a.rows = Bn; a.cols = K * K * 2;
    a.data.resize(static_cast<std::size_t>(Bn * K * K * 2));
    Json hz = Json::array();
    for (std::int64_t g = 0; g < Bn; ++g) {
      hz.push_back(static_cast<double>(pr.groups[static_cast<std::size_t>(g)].first) * df);
      for (std::int64_t i = 0; i < K; ++i)
        for (std::int64_t j = 0; j < K; ++j) {
          const std::size_t at = static_cast<std::size_t>(((g * K + i) * K + j) * 2);
          a.data[at] = cw.S[static_cast<std::size_t>(g)](i, j);
          a.data[at + 1] = cw.Sq[static_cast<std::size_t>(g)](i, j);
        }
    }
    a.meta["frequencies"] = hz;
    a.meta["rows"] = K;
    a.meta["tiling"] = tiling_of(pr.r.wo, s.first, s.samples, cw.segments, cw.segmentsSkipped);
    a.meta["n_members"] = cw.segments; a.meta["members_skipped"] = cw.segmentsSkipped;
    a.meta["options"] = options_of(pr.r);
    a.meta["producer"] = producer_of(pr.r, cw.segments, cw.segmentsSkipped, s.samples, s.bad, df, Json());
    return ng::Arrays{{"spectra", std::move(a)}};
  };
  return n;
}

// ── fft.spectrum and measure.band_amplitude ──────────────────────────────────────────────────────────────────────

ng::NodeSpec fft_spectrum() {
  ng::NodeSpec n;
  n.name = "fft.spectrum";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed, one chunk at a time)", false}};
  n.outputs = {{"spectrum", "the one-sided Fourier transform of every channel of the block, complex channels x bins, made once per block (axis block)", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "a change of basis of one block: nothing is carried from block to block; the transform runs in double, the value is handed on in float32");
  n.matches = "numpy.fft.rfft along time (norm backward: unscaled); compute nxr::fft::rfftColumns";
  n.description = "the one-sided FFT of every channel of the block as given: no window, no taper, no detrend, no scaling (X[k] = sum_j x[j] exp(-2 pi i j k / n), n = the block's samples); "
                  "its frequency line is bin k at k x sfreq / n Hz (spacing sfreq / n, origin 0, unit Hz), bins n/2 + 1";
  n.plan = [](const ng::Types& in, const Json&) {
    const ng::PortType& x = streamed_real(in, "x", "fft.spectrum");
    if (x.value != ng::ValueType::real) ng::refuse("fft.spectrum: x is real");
    ng::Planned out;
    ng::PortType s;
    s.cls = ng::PortClass::field; s.value = ng::ValueType::complex; s.axis = ng::Axis::block;
    s.rows = x.rows; s.unit = x.unit; s.sfreq = x.sfreq;
    out.out["spectrum"] = s;
    return out;
  };
  n.chunk = [](const std::any&, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    const ng::Block& b = in.at("x");
    const Eigen::MatrixXd x = own_real(b);                                   // channels x n
    const Eigen::MatrixXcd X = nxr::fft::rfftColumns(x.transpose()).transpose();   // channels x bins
    ng::Block o;
    o.rows = X.rows(); o.width = X.cols(); o.left = 0; o.right = 0;
    o.re.resize(static_cast<std::size_t>(o.rows * o.width)); o.im.resize(o.re.size());
    for (Eigen::Index i = 0; i < X.rows(); ++i)
      for (Eigen::Index k = 0; k < X.cols(); ++k) { o.re[static_cast<std::size_t>(i * X.cols() + k)] = static_cast<float>(X(i, k).real()); o.im[static_cast<std::size_t>(i * X.cols() + k)] = static_cast<float>(X(i, k).imag()); }
    return ng::Blocks{{"spectrum", std::move(o)}};
  };
  return n;
}

struct BandAmpPrepared {
  std::vector<nxr::selection::SpanRow> rows;
  nxr::selection::Lookup lookup;
  double sfreq = 0;
  std::string unit;
};
struct BandAmpState {
  std::int64_t bins = -1, channels = 0, chunks = 0;
  std::vector<double> amplitude;                      // chunk-major: chunk x channel x band
  std::vector<nxr::BinRange> ranges;
  nxr::selection::Selection bands;
  double df = 0;
};

ng::NodeSpec measure_band_amplitude() {
  ng::NodeSpec n;
  n.name = "measure.band_amplitude";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"spectrum", "a block's complex spectrum, channels x bins (fft.spectrum)", false}};
  n.outputs = {{"freq", "the spectrum's frequency line (count, step, start, unit)", false},
               {"bands", "the bands as a set of spans on the frequency line, their names the lookup", false},
               {"measurement", "the amplitude (mean |X| over the band's bins) of every channel of every chunk in every band, over the bands", false}};
  n.parameters = {{"bands", Json(), "the bands: a list of {name, lo, hi} in Hz, each the span [lo, hi) of the spectrum's frequency line (no default: the caller chooses)", "Hz", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::float64, "the rows of each block are kept in block order; nothing is combined across blocks");
  n.matches = "the mean of |numpy.fft.rfft| over the bins of each band";
  n.description = "the amplitude of every channel in named frequency bands: the bands are a set of spans in Hz (nxr::selection::set_of_spans), each resolved to the spectrum's bins [first, last) "
                  "(first = ceil(lo / spacing), last = ceil(hi / spacing)), the amplitude the mean |X| over them; rows are chunk x channel";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("spectrum") || in.at("spectrum").axis != ng::Axis::block || in.at("spectrum").value != ng::ValueType::complex) ng::refuse("measure.band_amplitude: spectrum is a complex block value (fft.spectrum)");
    if (!in.at("spectrum").sfreq || !(*in.at("spectrum").sfreq > 0)) ng::refuse("measure.band_amplitude: the spectrum has no sampling rate");
    if (!p.contains("bands") || !p["bands"].is_array() || p["bands"].empty()) ng::refuse("measure.band_amplitude: bands is required: a list of {name, lo, hi} in Hz (no default)");
    ng::Planned out;
    for (const char* k : {"freq", "bands", "measurement"}) { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = std::string(k) == "freq" ? "manifold" : std::string(k) == "bands" ? "selection" : "measurement"; if (std::string(k) == "measurement") { t.facts["measure"] = "amplitude"; t.facts["over"] = Json{{"selection", "the bands of this run"}}; } out.out[k] = t; }
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json& p, ng::Shared&) {
    BandAmpPrepared pr;
    pr.sfreq = *in.at("spectrum").sfreq;
    pr.unit = in.at("spectrum").unit;
    std::int64_t code = 0;
    for (const auto& b : p["bands"]) {
      const std::string name = b.value("name", std::string());
      if (name.empty() || !b.contains("lo") || !b.contains("hi")) ng::refuse("measure.band_amplitude: each band is {name, lo, hi}");
      const double lo = b["lo"].get<double>(), hi = b["hi"].get<double>();
      if (!(hi > lo) || lo < 0) ng::refuse("measure.band_amplitude: band " + name + " needs 0 <= lo < hi");
      pr.rows.push_back({lo, hi, code});
      pr.lookup.push_back({code, name, std::nullopt});
      ++code;
    }
    try { nxr::selection::check(pr.lookup); } catch (const std::invalid_argument& e) { ng::refuse(e.what()); }
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(BandAmpState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    const BandAmpPrepared& pr = std::any_cast<const BandAmpPrepared&>(prepared);
    BandAmpState& s = std::any_cast<BandAmpState&>(state);
    const ng::Block& b = in.at("spectrum");
    const Eigen::MatrixXcd X = own_complex(b);
    if (s.bins < 0) {
      s.bins = b.width; s.channels = b.rows;
      const double len = 2.0 * static_cast<double>(b.width - 1);              // the block's samples (an even number: the bins do not tell an odd one)
      s.df = pr.sfreq / len;
      s.bands = nxr::selection::set_of_spans(pr.rows, "Hz", pr.lookup);
      for (const auto& m : s.bands.members) {
        const double lo = m.extents.at(0).start, hi = m.extents.at(0).stop;
        const std::int64_t first = static_cast<std::int64_t>(std::ceil(lo / s.df - 1e-9)), last = static_cast<std::int64_t>(std::ceil(hi / s.df - 1e-9));
        if (last <= first || first >= s.bins) ng::refuse("measure.band_amplitude: the band [" + std::to_string(lo) + ", " + std::to_string(hi) + ") Hz holds no bin at " + std::to_string(s.df) + " Hz spacing");
        s.ranges.push_back({first, std::min(last, s.bins)});
      }
    } else if (b.width != s.bins || b.rows != s.channels) {
      ng::refuse("measure.band_amplitude: blocks of different lengths (" + std::to_string(b.width) + " and " + std::to_string(s.bins) + " bins): the bands' bins would differ; use whole chunks");
    }
    const Eigen::MatrixXd A = nxr::bandAmplitude(X, s.ranges);                 // channels x bands
    for (Eigen::Index c = 0; c < A.rows(); ++c) for (Eigen::Index k = 0; k < A.cols(); ++k) s.amplitude.push_back(A(c, k));
    ++s.chunks;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const BandAmpPrepared& pr = std::any_cast<const BandAmpPrepared&>(prepared);
    const BandAmpState& s = std::any_cast<const BandAmpState&>(state);
    if (s.chunks == 0) ng::refuse("measure.band_amplitude: no block was read");
    const std::int64_t nb = static_cast<std::int64_t>(s.ranges.size());
    ng::Array freq;
    freq.rows = 1; freq.cols = 1; freq.data = {0.0};
    freq.meta["describe"] = Json{{"count", s.bins}, {"step", s.df}, {"start", 0.0}, {"unit", "Hz"}, {"dimension", "frequency"}};
    ng::Array bands;
    bands.rows = nb; bands.cols = 1;
    Json spans = Json::array(), entries = Json::array();
    for (std::size_t k = 0; k < s.bands.members.size(); ++k) {
      const auto& m = s.bands.members[k];
      spans.push_back(Json{{"start", m.extents.at(0).start}, {"stop", m.extents.at(0).stop}, {"code", *m.code}});
      entries.push_back(nxr::selection::entry(pr.lookup, *m.code).name);
      bands.data.push_back(static_cast<double>(*m.code));
    }
    bands.meta["describe"] = Json{{"spans", spans}, {"entries", entries}, {"description", "named frequency bands: spans on the frequency line, [lo, hi) in Hz"}};
    ng::Array a;
    a.rows = s.chunks * s.channels * nb; a.cols = 1;
    a.data = s.amplitude;
    a.meta["describe"] = Json{{"shape", Json::array({s.chunks * s.channels, nb, 1})}, {"values_shape", Json::array({s.chunks * s.channels, nb})}, {"values_f32", true},
                              {"measures", Json::array({Json{{"measure", "amplitude"}, {"unit", pr.unit.empty() ? std::string("channel unit") : pr.unit}}})},
                              {"description", "mean |X| over each band's bins; rows chunk x channel, members the bands"}};
    return ng::Arrays{{"freq", std::move(freq)}, {"bands", std::move(bands)}, {"measurement", std::move(a)}};
  };
  return n;
}
}  // namespace

void register_welch_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {welch_psd(), rotary_pair(), welch_complex_psd(), welch_source_band_power(), welch_source_tensor(), welch_cross_spectrum(), welch_mode_cross_spectra(), fft_spectrum(), measure_band_amplitude()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
