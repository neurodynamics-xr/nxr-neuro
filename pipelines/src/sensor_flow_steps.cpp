// nxr-pipelines — THE FLOW OF A WEIGHTED CROSS-SPECTRUM ON A SURFACE'S FIELD COORDINATES as graph nodes (spec docs/specs/2026-11-24-sensor-flow.md; definitions `sensor_flow`).
//
// THE GRAPH. before: eigenvectors + channel map --[signal.mode_analysis, modes exact_and_constant]--> the serving operator [K x channels], bound once.
// chunks: recording (each chunk with `overlap_samples` on each side from its neighbours, reflected at the recording's ends) --[signal.overlapped_spectrum]--> the chunk's spectrum [K x B]
//         once a block --[signal.chunk_spectra]--> [chunks x B x K]: the cross-spectrum in its exact factored form (one value per member of the chunk grid; nothing summed).
// after:  chunk spectra + eigenvectors + a frequency weighting (a kernel set: the atoms; or every bin) + eigenvalue bands --[signal.spectral_flow]--> per frequency member the whole field's
//         and every band pair's power, phase currents, flow coefficients, potential and stream;
//         --[signal.atom_time_cells]--> per atom and band, the time cells' sums; --[stats.sign_flip_consistency]--> R1 and its sign-flip null per atom and band;
//         --[signal.sensor_power]--> each sensor's power per bin --[stats.aperiodic_fit]--> its aperiodic background (an option of the definition).
// Every node is a function of arrays: no file, no subject, no loop over a recording (the chunk service streams; the stored chunk spectra are an array handed over).
#include "nxr/pipelines/sensor_flow_steps.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <complex>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "flow_measures.h"
#include "nxr/pipelines/form_steps.h"
#include "chunk_spectra_readers.h"
#include "nxr/measurement/roll_up.h"
#include "nxr/signal/aperiodic.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/stats/sign_flip.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using cd = std::complex<double>;

namespace {

// a stated suffix on every measure an output describes (e.g. "_left", "_right"): two measurements of one recording over one selection, made on two surfaces
// (the hemispheres), stay two (the store's key has no surface: #264). Empty: the names as they are.
void suffix_measures(ng::Array& a, const std::string& suffix) {
  if (suffix.empty() || !a.meta.contains("describe") || !a.meta["describe"].contains("measures")) return;
  for (auto& m : a.meta["describe"]["measures"]) m["measure"] = m["measure"].get<std::string>() + suffix;
}
ng::PortType constant_of(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}
std::int64_t integer_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_number_integer()) ng::refuse(std::string(who) + ": " + key + " has no default: name it (an integer)");
  return p[key].get<std::int64_t>();
}
std::string text_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_string()) ng::refuse(std::string(who) + ": " + key + " has no default: name it");
  return p[key].get<std::string>();
}
std::pair<double, double> span_param(const Json& p, const char* who) {
  if (!p.contains("frequency_span") || !p["frequency_span"].is_array() || p["frequency_span"].size() != 2) ng::refuse(std::string(who) + ": frequency_span [lo, hi) has no default: name it");
  const double lo = p["frequency_span"][0].get<double>(), hi = p["frequency_span"][1].get<double>();
  if (!(lo >= 0) || !(hi > lo)) ng::refuse(std::string(who) + ": frequency_span is [lo, hi) with 0 <= lo < hi");
  return {lo, hi};
}
// the bins [first, first + count) of a transform of n samples at fs that lie in [lo, hi)
std::pair<std::int64_t, std::int64_t> bins_of(double lo, double hi, double fs, std::int64_t n, const char* who) {
  const double df = fs / static_cast<double>(n);
  const std::int64_t first = static_cast<std::int64_t>(std::ceil(lo / df - 1e-9));
  const std::int64_t end = std::min<std::int64_t>(n / 2 + 1, static_cast<std::int64_t>(std::ceil(hi / df - 1e-9)));
  if (end - first < 1) ng::refuse(std::string(who) + ": the frequency span holds no bin (spacing " + std::to_string(df) + " Hz)");
  return {first, end - first};
}
const ng::Array& constant_input(const ng::Arrays& c, const char* port, const char* who) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + port + " was not handed over");
  return it->second;
}

// an axis described by its entries (indices or names): the runner stores it as the values it runs along
Json indexed_axis(const char* name, std::int64_t size, const Json& values = Json()) {
  Json v = values;
  if (v.is_null()) { v = Json::array(); for (std::int64_t i = 0; i < size; ++i) v.push_back(i); }
  return Json{{"name", name}, {"size", size}, {"values", v}};
}

// ── signal.overlapped_spectrum ──────────────────────────────────────────────────────────────────────────────────
const char* const kServe = "signal.overlapped_spectrum";
struct ServePrepared { Eigen::MatrixXd op; std::vector<std::int32_t> channels; std::int64_t overlap = 0, transform = 0, first = 0, bins = 0; };

ng::NodeSpec overlapped_spectrum() {
  ng::NodeSpec n;
  n.name = kServe;
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples, each block a chunk with its overlap on each side (streamed)", false},
              {"operator", "the serving operator from the recording's channels (signal.mode_analysis: the field coordinates' analysis rows composed with the channel map)", false}};
  n.outputs = {{"spectrum", "the chunk's spectrum in the operator's rows: [rows x bins] complex at the bins of the frequency span of a transform of transform_samples, once a block", false}};
  n.parameters = {
      {"overlap_samples", Json(), "the samples the chunk's transform takes in on its left from the neighbouring chunk (and on its right, up to transform_samples); reflected where the recording ends (no default; the flow analysis: half a chunk, 16384)", "samples", {}, false},
      {"transform_samples", Json(), "the length of each chunk's one transform: the chunk, its overlaps and, for a part chunk at the recording's end, the reflection (no default; the flow analysis: 65536)", "samples", {}, false},
      {"frequency_span", Json(), "[lo, hi): the bins kept of the transform's frequency line (no default; the flow analysis: [0, 60])", "Hz", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "a change of basis of one chunk with its overlap: the operator on the samples, then one transform in double; float32 out for float32 input",
                             "overlap_samples on each side of the chunk, taken from the neighbouring chunks and reflected about the recording's first or last sample where it ends");
  n.linear_map = true;
  n.matches = "numpy: rfft of P x over the chunk and its overlaps, np.pad mode reflect at the recording's ends, no window (the flow analysis's serving step); compute nxr::signal::overlappedSpectrum";
  n.description = "each chunk's spectrum in a surface's field coordinates: the chunk with its overlap read through the serving operator and transformed once (no window, no taper, no zero padding), the bins of a span kept";
  n.plan = [](const ng::Types& in, const Json& p) {
    const auto it = in.find("x");
    if (it == in.end() || !it->second.streamed() || it->second.axis != ng::Axis::samples) ng::refuse(std::string(kServe) + ": x is a recording's samples (a streamed field)");
    const ng::PortType& x = it->second;
    if (x.value != ng::ValueType::real) ng::refuse(std::string(kServe) + ": x is real");
    if (!x.sfreq || !(*x.sfreq > 0)) ng::refuse(std::string(kServe) + ": x has no sampling rate");
    if (!in.count("operator")) ng::refuse(std::string(kServe) + ": operator is not given");
    const std::int64_t overlap = integer_param(p, "overlap_samples", kServe), transform = integer_param(p, "transform_samples", kServe);
    if (overlap < 0 || transform <= 2 * overlap) ng::refuse(std::string(kServe) + ": 0 <= overlap_samples and 2 x overlap_samples < transform_samples");
    const auto [lo, hi] = span_param(p, kServe);
    bins_of(lo, hi, *x.sfreq, transform, kServe);
    ng::Planned out;
    ng::PortType s;
    s.cls = ng::PortClass::field; s.value = ng::ValueType::complex; s.axis = ng::Axis::none; s.unit = x.unit; s.sfreq = x.sfreq;
    const ng::ManifoldRef rows = ng::manifold_id("coordinates", "operator");
    s.domain.factors = {rows, ng::made_line("frequency", "signal.overlapped_spectrum.spectrum", "Hz", 0.0, x.sfreq, x.domain.factors.empty() ? ng::ManifoldRef() : x.domain.factors.back())};
    s.domain.cells = "0,1";
    s.facts["overlap_samples"] = overlap; s.facts["transform_samples"] = transform;   // the spectrum carries how it was made: signal.chunk_spectra reads them, never asked twice
    s.facts["frequency_span"] = Json::array({lo, hi});
    out.out["spectrum"] = s;
    out.halo = overlap;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ServePrepared pr;
    const ng::Array& O = constant_input(c, "operator", kServe);
    const std::vector<double> ov = O.values();
    pr.op = Eigen::Map<const RowMajor>(ov.data(), O.rows, O.cols);
    pr.channels = O.meta.value("channels", std::vector<std::int32_t>());
    if (static_cast<std::int64_t>(pr.channels.size()) != O.cols) ng::refuse(std::string(kServe) + ": the operator's channels do not fit its columns");
    pr.overlap = p["overlap_samples"].get<std::int64_t>();
    pr.transform = p["transform_samples"].get<std::int64_t>();
    const auto [lo, hi] = span_param(p, kServe);
    std::tie(pr.first, pr.bins) = bins_of(lo, hi, *in.at("x").sfreq, pr.transform, kServe);
    return std::any(std::move(pr));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    const ServePrepared& pr = std::any_cast<const ServePrepared&>(prepared);
    const ng::Block& b = in.at("x");
    Eigen::MatrixXd samples(static_cast<Eigen::Index>(pr.channels.size()), b.width);
    for (std::size_t i = 0; i < pr.channels.size(); ++i) {
      const std::int32_t ch = pr.channels[i];
      if (ch < 0 || ch >= b.rows) ng::refuse(std::string(kServe) + ": a channel of the operator is outside the recording's channels");
      for (std::int64_t t = 0; t < b.width; ++t) samples(static_cast<Eigen::Index>(i), t) = b.re[static_cast<std::size_t>(ch * b.width + t)];
    }
    const Eigen::MatrixXcd X = nxr::signal::overlappedSpectrum(pr.op, samples, b.left, b.right, pr.overlap, pr.transform, pr.first, pr.bins);
    ng::Block o;
    o.rows = X.rows(); o.width = X.cols(); o.line_samples = pr.transform;
    o.re.resize(static_cast<std::size_t>(o.rows * o.width)); o.im.resize(o.re.size());
    for (Eigen::Index r = 0; r < X.rows(); ++r)
      for (Eigen::Index k = 0; k < X.cols(); ++k) { o.re[static_cast<std::size_t>(r * X.cols() + k)] = static_cast<float>(X(r, k).real()); o.im[static_cast<std::size_t>(r * X.cols() + k)] = static_cast<float>(X(r, k).imag()); }
    return ng::Blocks{{"spectrum", std::move(o)}};
  };
  n.record = [](const std::any& prepared) {
    const ServePrepared& pr = std::any_cast<const ServePrepared&>(prepared);
    return Json{{"overlap_samples", pr.overlap}, {"transform_samples", pr.transform}, {"first_bin", pr.first}, {"bins", pr.bins},
                {"transform", "one-sided, unscaled, no window, no taper, no zero padding; reflected (numpy reflect) at the recording's ends"}};
  };
  return n;
}

// ── signal.chunk_spectra ─────────────────────────────────────────────────────────────────────────────────────────
const char* const kGather = "signal.chunk_spectra";
struct GatherPrepared { Json op_meta; std::int64_t overlap = 0, transform = 0, first = 0, bins = 0; double sfreq = 0; };
struct GatherState { std::vector<float> values; std::vector<std::int64_t> firsts, owns; std::int64_t rows = -1; };

ng::NodeSpec chunk_spectra() {
  ng::NodeSpec n;
  n.name = kGather;
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"spectrum", "each chunk's spectrum (signal.overlapped_spectrum), once a block", false},
              {"operator", "the serving operator the spectra were read through (its rows: the coordinates and their eigenvalues)", false}};
  n.outputs = {{"spectra", "[chunks x bins x rows] complex: each chunk's spectrum, one value per member of the chunk grid (the cross-spectrum in its exact factored form)", false}};
  n.parameters = {};   // the overlap, the transform's length and the span are the spectrum's own (its port's facts, from signal.overlapped_spectrum): stated once
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float32, ng::OutputPrecision::follows_input,
                             "each block's spectrum kept in time order (a measurement over the chunk partition: one value a member, nothing summed)");
  n.matches = "the flow analysis's served chunk spectra (each chunk's rfft in the field coordinates, stored); no reduction";
  n.description = "the chunk spectra of a run gathered into one statistic: the cross-spectrum per bin is the sum over its chunks of each spectrum times its conjugate";
  n.plan = [](const ng::Types& in, const Json&) {
    const auto it = in.find("spectrum");
    if (it == in.end() || !it->second.per_block()) ng::refuse(std::string(kGather) + ": spectrum is a value made once per block (signal.overlapped_spectrum)");
    if (!in.count("operator")) ng::refuse(std::string(kGather) + ": operator is not given");
    for (const char* k : {"overlap_samples", "transform_samples", "frequency_span"})
      if (!it->second.facts.contains(k)) ng::refuse(std::string(kGather) + ": spectrum does not say its " + k + " (made by signal.overlapped_spectrum)");
    ng::Planned out;
    out.out["spectra"] = constant_of("statistic");
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json&, ng::Shared&) {
    GatherPrepared pr;
    pr.op_meta = constant_input(c, "operator", kGather).meta;
    pr.op_meta.erase("channels");
    const auto it = in.find("spectrum");
    if (it == in.end() || !it->second.sfreq) ng::refuse(std::string(kGather) + ": spectrum has no sampling rate");
    const Json& made = it->second.facts;
    pr.overlap = made.at("overlap_samples").get<std::int64_t>(); pr.transform = made.at("transform_samples").get<std::int64_t>();
    pr.sfreq = *it->second.sfreq;
    const double lo = made.at("frequency_span")[0].get<double>(), hi = made.at("frequency_span")[1].get<double>();
    std::tie(pr.first, pr.bins) = bins_of(lo, hi, pr.sfreq, pr.transform, kGather);
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(GatherState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const GatherPrepared& pr = std::any_cast<const GatherPrepared&>(prepared);
    GatherState& s = std::any_cast<GatherState&>(state);
    const ng::Block& b = in.at("spectrum");
    if (b.width != pr.bins || !b.complex()) ng::refuse(std::string(kGather) + ": a block's spectrum holds " + std::to_string(b.width) + " bins, not " + std::to_string(pr.bins));
    if (s.rows < 0) s.rows = b.rows;
    if (b.rows != s.rows) ng::refuse(std::string(kGather) + ": the blocks' spectra have different rows");
    for (std::int64_t k = 0; k < b.width; ++k)                                            // bin-major then row
      for (std::int64_t r = 0; r < b.rows; ++r) { s.values.push_back(b.re[static_cast<std::size_t>(r * b.width + k)]); s.values.push_back(b.im[static_cast<std::size_t>(r * b.width + k)]); }
    s.firsts.push_back(at.first); s.owns.push_back(at.n);
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const GatherPrepared& pr = std::any_cast<const GatherPrepared&>(prepared);
    GatherState& s = std::any_cast<GatherState&>(state);
    if (s.firsts.empty()) ng::refuse(std::string(kGather) + ": no chunk was read");
    const std::int64_t T = static_cast<std::int64_t>(s.firsts.size()), R = s.rows;
    ng::Array a;
    a.rows = T; a.cols = pr.bins * R * 2;
    a.single = std::move(s.values);
    const double df = pr.sfreq / static_cast<double>(pr.transform);
    Json chunks = Json::array();
    for (std::size_t i = 0; i < s.firsts.size(); ++i)
      chunks.push_back(Json{{"first", s.firsts[i]}, {"samples", s.owns[i]}, {"span", Json::array({static_cast<double>(s.firsts[i]) / pr.sfreq, static_cast<double>(s.firsts[i] + s.owns[i]) / pr.sfreq})}});
    Json bin_members = Json::array();
    for (std::int64_t k = 0; k < pr.bins; ++k) bin_members.push_back(pr.first + k);
    const Json producer = Json{{"sfreq", pr.sfreq}, {"transform_samples", pr.transform}, {"overlap_samples", pr.overlap}, {"first_bin", pr.first}, {"bins", pr.bins},
                               {"frequency_resolution_hz", df}, {"modes", pr.op_meta.value("modes", Json::array())}, {"mu", pr.op_meta.value("mu", Json::array())},
                               {"normalisation", pr.op_meta.value("rows", Json::object()).value("normalisation", Json())}, {"components", 1}, {"chunks", chunks}};
    a.meta["shape"] = Json::array({T, pr.bins, R, 2});
    a.meta["producer"] = producer;
    a.meta["describe"] = Json{{"measure", "fourier_coefficient"}, {"method", "chunk_spectra"}, {"label", "chunk_spectra"}, {"precision", "float32"}, {"value_type", "complex"},
                              {"shape", Json::array({T, pr.bins, R, 2})}, {"unit", "(operator output) x samples"},
                              {"axes", Json::array({Json{{"name", "chunk"}, {"size", T}, {"values", chunks}},
                                                    Json{{"name", "frequency"}, {"size", pr.bins}, {"members", bin_members}, {"line", Json{{"spacing", df}, {"origin", static_cast<double>(pr.first) * df}, {"unit", "Hz"}, {"dimension", "frequency"},
                                                                                                             {"fft", Json{{"n", pr.transform}, {"fs", pr.sfreq}}}}}},
                                                    Json{{"name", "coordinate"}, {"size", R}, {"members", pr.op_meta.value("modes", Json::array())}}})},
                              {"options", Json{{"overlap_samples", pr.overlap}, {"transform_samples", pr.transform}, {"operator", pr.op_meta},
                                               {"transform", "one-sided, unscaled, no window, no taper, no zero padding; reflected at the recording's ends"}}},
                              {"producer", producer},
                              {"description", "each chunk's spectrum in the field coordinates, the cross-spectrum in its exact factored form: [chunks x bins x coordinates] (complex)"}};
    return ng::Arrays{{"spectra", std::move(a)}};
  };
  return n;
}

}  // namespace

// ── what the readers of the chunk spectra share (declared in chunk_spectra_readers.h: the source-flow nodes read them too) ────────────────────────────
namespace spectra_readers {
Spectra spectra_of(const ng::Array& A, const char* who) {
  const Json& d = A.meta.contains("shape") ? A.meta : A.meta.value("describe", Json::object());
  const Json pr = A.meta.contains("producer") ? A.meta["producer"] : d.value("producer", Json::object());
  if (!d.contains("shape") || d["shape"].size() != 4 || d["shape"][3] != 2) ng::refuse(std::string(who) + ": spectra are not [chunks x bins x coordinates x 2] (signal.chunk_spectra)");
  Spectra s;
  s.chunks = d["shape"][0]; s.bins = d["shape"][1]; s.rows = d["shape"][2];
  for (const char* k : {"sfreq", "transform_samples", "first_bin", "modes"}) if (!pr.contains(k)) ng::refuse(std::string(who) + ": the spectra do not record " + k);
  s.sfreq = pr["sfreq"]; s.transform = pr["transform_samples"]; s.first = pr["first_bin"]; s.overlap = pr.value("overlap_samples", std::int64_t(0));
  s.modes = pr["modes"].get<std::vector<std::int64_t>>();
  s.unit_mass = pr.value("normalisation", Json("unit_mass")) != Json("as_stored");
  for (const auto& c : pr.value("chunks", Json::array())) { s.owns.push_back(c.value("samples", std::int64_t(0))); s.firsts.push_back(c.value("first", std::int64_t(0))); }
  if (static_cast<std::int64_t>(s.modes.size()) != s.rows) ng::refuse(std::string(who) + ": the spectra's coordinates are not their modes");
  s.array = &A;
  return s;
}
// the field coordinates' synthesis Psi [S x K] (the constant at unit star0 norm; each mode at unit star0 norm under unit_mass) and their eigenvalues
std::pair<Eigen::MatrixXd, Eigen::VectorXd> synthesis_of(const Spectra& s, const nxr::signal::Modes& m, const Eigen::VectorXd& mass, const char* who) {
  const Eigen::Index S = m.P.rows(), K = static_cast<Eigen::Index>(s.modes.size());
  if (mass.size() != S) ng::refuse(std::string(who) + ": the package's vertex mass does not fit its vertices");
  Eigen::MatrixXd Psi(S, K);
  Eigen::VectorXd mu(K);
  for (Eigen::Index j = 0; j < K; ++j) {
    const std::int64_t k = s.modes[static_cast<std::size_t>(j)];
    if (k < 0) { Psi.col(j).setConstant(1.0 / std::sqrt(mass.sum())); mu(j) = 0.0; continue; }
    if (k >= m.P.cols()) ng::refuse(std::string(who) + ": a coordinate's mode is outside the package's modes");
    Psi.col(j) = m.P.col(k);
    if (s.unit_mass) Psi.col(j) /= std::sqrt(Psi.col(j).cwiseAbs2().dot(mass));
    mu(j) = m.mu(k);
  }
  return {Psi, mu};
}
// THE ONE READER of a stored kernel set: the members select.kernels wrote, as a typed selection (the only place their stored form is read)
nxr::selection::Selection kernelSetOf(const ng::Array& stored, const char* who, const char* port) {
  const Json p = stored.meta.value("params", Json::object());
  if (!p.contains("kernels") || !p["kernels"].contains("members")) ng::refuse(std::string(who) + ": " + port + " is not a kernel set (`nxr-neuro run select_kernels`)");
  std::vector<nxr::selection::Selection> members;
  for (const Json& k : p["kernels"]["members"]) {
    for (const char* key : {"start", "stop", "family", "chart", "gain"}) if (!k.contains(key)) ng::refuse(std::string(who) + ": a member of " + port + " does not record " + key);
    nxr::selection::Profile profile;
    profile.family = k.at("family").get<std::string>();
    profile.chart = k.at("chart").get<std::string>();
    profile.gain = k.at("gain").get<double>();
    const Json params = k.value("params", Json::object());   // held: the loop below iterates over it (a temporary would be gone before the body)
    for (const auto& [name, value] : params.items()) {
      if (name == "width_rule")
        ng::refuse(std::string(who) + ": " + port + " was made before the proportional width was stored as fwhm_per_centre: re-make it with `select_kernels`");
      if (value.is_number()) profile.params[name] = value.get<double>();   // a text entry is a description (a partition's), not a number of the profile
    }
    nxr::selection::Selection member = nxr::selection::span({{0, k.at("start").get<double>(), k.at("stop").get<double>()}}, k.value("code", static_cast<std::int64_t>(members.size())));
    member.profile = profile;
    members.push_back(std::move(member));
  }
  try {
    return nxr::selection::set(std::move(members));
  } catch (const std::invalid_argument& e) {
    ng::refuse(std::string(who) + ": " + port + ": " + e.what());
  }
}
nxr::signal::KernelGains kernel_gains_of(const ng::Array& stored, const Eigen::VectorXd& line, const char* who, const char* port) {
  try {
    return nxr::signal::kernelGains(kernelSetOf(stored, who, port), line, nxr::signal::NonPositiveCoordinates::zero);
  } catch (const std::invalid_argument& e) {
    ng::refuse(std::string(who) + ": " + port + ": " + e.what());
  }
}
// the frequency weighting's members: each a gain per bin (the atoms; normalised as the parameter says) or every bin alone
Eigen::MatrixXd frequency_gains(const Spectra& s, const ng::Arrays& c, const Json& p, const char* who, std::vector<double>* standardDeviations) {
  const std::string weighting = text_param(p, "frequency_weighting", who);
  Eigen::VectorXd f(s.bins);
  for (std::int64_t k = 0; k < s.bins; ++k) f(k) = s.frequency(k);
  if (weighting == "every_bin") {
    if (standardDeviations) ng::refuse(std::string(who) + ": every_bin has no atom (no time width)");
    if (c.count("frequency") && !c.at("frequency").meta.value("placeholder", false))
      ng::refuse(std::string(who) + ": frequency_weighting every_bin reads every bin alone: the frequency kernel set handed to it would be ignored (leave it out, or choose kernels)");
    return Eigen::MatrixXd::Identity(s.bins, s.bins);
  }
  if (weighting != "kernels") ng::refuse(std::string(who) + ": frequency_weighting is kernels or every_bin");
  const nxr::signal::KernelGains atoms = kernel_gains_of(constant_input(c, "frequency", who), f, who, "frequency");
  Eigen::MatrixXd g = atoms.gains;
  if (standardDeviations) {
    if (atoms.family != "gaussian" || atoms.chart != "linear") ng::refuse(std::string(who) + ": the time cells need Gaussian atoms on a linear chart");
    for (Eigen::Index m = 0; m < atoms.fullWidths.size(); ++m) standardDeviations->push_back(atoms.fullWidths(m) / (2.0 * std::sqrt(2.0 * std::log(2.0))));
  }
  const std::string normalisation = text_param(p, "frequency_normalisation", who);
  if (normalisation == "sum_of_squared_gains") {
    if (!p.contains("support_floor") || !p["support_floor"].is_number()) ng::refuse(std::string(who) + ": support_floor has no default: name it");
    g = nxr::signal::normaliseBySumOfSquaredGains(g, p["support_floor"].get<double>());
  } else if (normalisation != "none") ng::refuse(std::string(who) + ": frequency_normalisation is sum_of_squared_gains or none");
  return g;
}
// one frequency member's weighted columns Y [K x chunks.bins] (each bin's column times its gain; the cross-spectrum it weights is Y Yᴴ)
Eigen::MatrixXcd weighted_columns(const Spectra& s, const Eigen::RowVectorXd& gain, double floorFraction) {
  const double top = gain.cwiseAbs().maxCoeff();
  std::vector<std::int64_t> bins;
  for (std::int64_t k = 0; k < s.bins; ++k) if (std::abs(gain(k)) > floorFraction * top && gain(k) != 0) bins.push_back(k);
  Eigen::MatrixXcd Y(s.rows, s.chunks * static_cast<std::int64_t>(bins.size()));
  for (std::int64_t t = 0; t < s.chunks; ++t)
    for (std::size_t q = 0; q < bins.size(); ++q)
      for (std::int64_t r = 0; r < s.rows; ++r) Y(r, t * static_cast<std::int64_t>(bins.size()) + static_cast<std::int64_t>(q)) = gain(bins[q]) * s.at(t, bins[q], r);
  return Y;
}

// ── the time cells on a stored time grid (chunk_spectra_readers.h) ──────────────────────────────────────────────────────────────────────────────────
Json TimeGrid::record() const {
  return Json{{"name", name}, {"level", level}, {"tile_samples", width}, {"tile_seconds", static_cast<double>(width) / rate}, {"tiles", count}, {"line_samples", size}, {"rate", rate}};
}
TimeGrid time_grid_of(const ng::Array& grid, const Spectra& s, const char* who) {
  const Json prm = grid.meta.contains("describe") && grid.meta["describe"].is_object() && grid.meta["describe"].contains("params") ? grid.meta["describe"]["params"] : grid.meta.value("params", Json::object());
  const Json g = prm.is_object() ? prm.value("grid", Json::object()) : Json::object();
  const std::string name = grid.meta.value("name", std::string("the time grid"));
  if (g.value("kind", std::string()) != "time" || !g.contains("width") || !g.contains("count") || !g.contains("level"))
    ng::refuse(std::string(who) + ": " + name + " is not a time grid (`nxr-neuro run select_time_grid --rule tiles --duration D` first)");
  TimeGrid t;
  t.name = name;
  t.id = grid.meta.value("id", std::string());
  t.width = g["width"]; t.count = g["count"]; t.level = g["level"]; t.size = g.value("size", std::int64_t(0)); t.rate = g.value("rate", 0.0);
  if (g.value("hop", t.width) != t.width || t.width != (std::int64_t(1) << t.level) || g.value("origin", std::int64_t(0)) != 0)
    ng::refuse(std::string(who) + ": " + name + " is not one level of the line's dyadic tiling from its first sample (rule tiles, origin start): the cells are its tiles");
  if (std::abs(t.rate - s.sfreq) > 1e-9 * s.sfreq) ng::refuse(std::string(who) + ": " + name + " lies on a line of " + std::to_string(t.rate) + " Hz, the spectra's on one of " + std::to_string(s.sfreq) + " Hz");
  std::int64_t samples = 0;
  for (std::size_t c = 0; c < s.owns.size(); ++c) samples = std::max(samples, s.firsts[c] + s.owns[c]);
  if (t.size == 0) t.size = samples;
  if (samples != t.size) ng::refuse(std::string(who) + ": " + name + " covers " + std::to_string(t.size) + " samples, the spectra's line " + std::to_string(samples) + ": not the recording's grid");
  if (t.count != (t.size + t.width - 1) / t.width) ng::refuse(std::string(who) + ": " + name + "'s tiles do not cover its line");
  return t;
}
int own_level(double cellWidth, double frequencySd, const TimeGrid& g, const char* who) {
  if (!(cellWidth > 0)) ng::refuse(std::string(who) + ": cell_width > 0 (the atom's own window in its time standard deviations)");
  if (!std::isfinite(frequencySd) || !(frequencySd > 0)) ng::refuse(std::string(who) + ": an atom's own window needs Gaussian atoms on a linear chart (their time width)");
  const int k = static_cast<int>(std::lround(std::log2(cellWidth / (2.0 * M_PI * frequencySd) * g.rate)));
  if (k < g.level)
    ng::refuse(std::string(who) + ": an atom's own window is 2^" + std::to_string(k) + " samples, finer than the time grid " + g.name + " (2^" + std::to_string(g.level) +
               "): store the cells on a grid of level " + std::to_string(k) + " or finer (`select_time_grid --rule tiles --duration " + std::to_string(static_cast<double>(std::int64_t(1) << k) / g.rate) + " s`)");
  return k;
}
void check_cells_product(const ng::Array& product, const TimeGrid& g, const std::vector<std::int64_t>& members, const char* who) {
  const Json prm = product.meta.contains("describe") && product.meta["describe"].is_object() && product.meta["describe"].contains("params") ? product.meta["describe"]["params"] : product.meta.value("params", Json::object());
  // made in this run (the node's record, params.product) or stored (the row's params: layout and radices; the factors' levels in their own rows)
  const Json pr = !prm.is_object() ? Json::object() : prm.contains("product") && prm["product"].is_object() ? prm["product"] : prm;
  std::vector<std::int64_t> want = {g.count};
  want.insert(want.end(), members.begin(), members.end());
  const std::string name = product.meta.value("name", std::string("the cells' product"));
  if (!pr.contains("radices") || pr["radices"].get<std::vector<std::int64_t>>() != want || pr.value("layout", std::string("product")) != "product")
    ng::refuse(std::string(who) + ": " + name + " is not the product of " + g.name + " (first, fastest) with the members the cells are of: its radices " + pr.value("radices", Json()).dump() +
               ", wanted " + Json(want).dump() + " (`nxr-neuro run select_product --time-grid ... --kernels atoms[_scales] --atoms ... [--scales ...]`)");
  const Json f0 = pr.contains("factors") && pr["factors"].is_array() && !pr["factors"].empty() ? pr["factors"][0] : Json::object();
  if (!f0.empty() && (f0.value("kind", std::string()) != "time" || f0.value("level", -1) != g.level)) ng::refuse(std::string(who) + ": " + name + "'s first factor is not the time grid " + g.name);
  // stored: each factor is a selection by id; the first must be the grid itself
  if (prm.is_object() && prm.contains("product") && prm["product"].is_array() && !prm["product"].empty() && !g.id.empty() && prm["product"][0].value("selection", std::string()) != g.id)
    ng::refuse(std::string(who) + ": " + name + "'s first factor is not the time grid " + g.name);
}
TileSums tile_sums(const Spectra& s, const Eigen::RowVectorXd& gains, const TimeGrid& g, const Eigen::MatrixXd& readout, int components, const std::vector<std::int32_t>& pairs,
                   const Eigen::SparseMatrix<double>* sharp) {
  TileSums out;
  const Eigen::Index V = readout.rows() / components, P = static_cast<Eigen::Index>(pairs.size() / 2);
  out.power = Eigen::MatrixXd::Zero(V, g.count);
  out.pairReal = Eigen::MatrixXd::Zero(P, g.count);
  out.pairImaginary = Eigen::MatrixXd::Zero(P, g.count);
  if (sharp) out.vertexCurrent = Eigen::MatrixXd::Zero(3 * V, g.count);
  out.samples.assign(static_cast<std::size_t>(g.count), 0);
  for (std::int64_t t = 0; t < s.chunks; ++t) {
    Eigen::MatrixXcd X(s.rows, s.bins);
    for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t r = 0; r < s.rows; ++r) X(r, k) = gains(k) * s.at(t, k, r);
    const nxr::signal::TimeCellPairFlow part = nxr::signal::timeCellPairFlow(X, s.first, s.transform, s.overlap, s.owns[static_cast<std::size_t>(t)], s.firsts[static_cast<std::size_t>(t)],
                                                                             g.width, readout, components, pairs, sharp);
    for (Eigen::Index j = 0; j < part.power.cols(); ++j) {                  // a tile cut by the chunk's edge: its parts added (each sample once)
      const std::int64_t tile = part.firstCell + j;
      if (tile < 0 || tile >= g.count) ng::refuse("time cells: a chunk's samples lie outside the time grid's tiles");
      out.power.col(tile) += part.power.col(j);
      out.pairReal.col(tile) += part.pairReal.col(j);
      out.pairImaginary.col(tile) += part.pairImaginary.col(j);
      if (sharp) out.vertexCurrent.col(tile) += part.vertexCurrent.col(j);
      out.samples[static_cast<std::size_t>(tile)] += part.samples[static_cast<std::size_t>(j)];
    }
  }
  return out;
}
ng::Array cells_measurement(const std::vector<std::vector<Eigen::MatrixXd>>& values, const std::vector<std::string>& measures, const TimeGrid& g, const Json& groups, const char* rowsAre,
                            const Json& producer) {
  const std::int64_t G = static_cast<std::int64_t>(values.size()), K = static_cast<std::int64_t>(measures.size()), N = g.count;
  const std::int64_t rows = G > 0 && K > 0 ? values[0][0].rows() : 0, members = N * G;
  ng::Array a;
  a.rows = rows * members; a.cols = K;
  a.data.assign(static_cast<std::size_t>(rows * members * K), 0.0);
  for (std::int64_t q = 0; q < G; ++q)
    for (std::int64_t k = 0; k < K; ++k) {
      const Eigen::MatrixXd& v = values[static_cast<std::size_t>(q)][static_cast<std::size_t>(k)];
      if (v.rows() != rows || v.cols() != N) ng::refuse("time cells: a member's sums do not fit the grid");
      for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t n = 0; n < N; ++n)                                 // stored float32: the run reads what the store holds
          a.data[static_cast<std::size_t>((r * members + n + N * q) * K + k)] = static_cast<double>(static_cast<float>(v(r, n)));
    }
  Json ms = Json::array();
  for (const auto& m : measures) ms.push_back(Json{{"measure", m}});
  a.meta["producer"] = producer;
  a.meta["describe"] = Json{{"method", "time_cells"}, {"measures", ms}, {"shape", Json::array({rows, members, K})}, {"values_shape", Json::array({rows, members})}, {"values_f32", true},
                            {"producer", producer},
                            {"attributes", Json{{"rows", rowsAre}, {"members", "the product of the time grid's tiles (fastest) with the groups: member n + tiles x group"},
                                                {"grid", g.record()}, {"groups", groups},
                                                {"sum_over", "a tile's samples (each sample once; a tile cut by a chunk's edge completed from the next chunk; the line's last tile as long as it is)"}}}};
  declare_flow_measures(a.meta["describe"]["measures"]);
  return a;
}
StoredCells stored_cells(const ng::Array& cells, const char* who) {
  const Json d = cells.meta.value("describe", Json::object());
  const Json at = d.value("attributes", Json::object());
  if (!d.contains("shape") || d["shape"].size() != 3 || !at.contains("grid") || !at.contains("groups"))
    ng::refuse(std::string(who) + ": cells are not time cells on a stored grid (a flow's cells measurement over the time grid's product)");
  StoredCells c;
  c.array = &cells;
  c.rows = d["shape"][0]; c.members = d["shape"][1]; c.K = d["shape"][2];
  c.tiles = at["grid"]["tiles"]; c.width = at["grid"]["tile_samples"]; c.size = at["grid"]["line_samples"]; c.level = at["grid"]["level"];
  c.groups = at["groups"]; c.measures = d.value("measures", Json::array());
  if (c.tiles * static_cast<std::int64_t>(c.groups.size()) != c.members) ng::refuse(std::string(who) + ": the cells' members are not the grid's tiles times their groups");
  return c;
}
std::int64_t StoredCells::measure_index(const std::string& name, const char* who) const {
  for (std::size_t k = 0; k < measures.size(); ++k) if (measures[k].value("measure", std::string()) == name) return static_cast<std::int64_t>(k);
  ng::refuse(std::string(who) + ": the cells hold no measure '" + name + "'");
}
Eigen::MatrixXd StoredCells::rolled(std::int64_t group, std::int64_t measure) const {
  const int own = groups[static_cast<std::size_t>(group)].value("own_level", level);
  nxr::selection::Selection grid;                                           // the stored grid as a closed-form tiling: its roll-up to the atom's own level (sums of 2^(own - level) tiles)
  grid.type = nxr::selection::Type::set;
  grid.family = nxr::selection::Family::grid;
  grid.tiling = nxr::selection::Tiling{0.0, static_cast<double>(width), static_cast<double>(width), tiles};
  const nxr::selection::Elements cells = nxr::measurement::coarser_groups(grid, level, own);
  std::int64_t whole = 0;                                                    // whole cells only: a cell holding samples past the line's end (its last part tile) is left out
  while (whole < cells.members() && ((whole + 1) << own) <= size) ++whole;
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(whole, rows);
  for (std::int64_t c = 0; c < whole; ++c)
    for (std::int64_t e = cells.offsets[static_cast<std::size_t>(c)]; e < cells.offsets[static_cast<std::size_t>(c) + 1]; ++e) {
      const std::int64_t member = cells.elements[static_cast<std::size_t>(e)] + tiles * group;
      for (std::int64_t r = 0; r < rows; ++r) out(c, r) += array->at(static_cast<std::size_t>((r * members + member) * K + measure));
    }
  return out;
}
}  // namespace spectra_readers
using namespace spectra_readers;

namespace {

struct FlowPrepared {
  nxr::pipelines::graph::BoundPackage package;
  Eigen::MatrixXd frequency, scale;          // [members x bins], [bands x K] (scale empty: no bands)
  std::vector<double> standardDeviations;
  Json p;
  ng::Array grid, product;                   // the time cells' stored grid and the product they are measured over (signal.atom_time_cells)
};
FlowPrepared prepare_flow(const ng::Arrays& c, const Json& p, const char* who, bool needAtoms) {
  FlowPrepared pr;
  pr.package = bind_hodge_package(constant_input(c, "package", who), who);
  const Spectra s = spectra_of(constant_input(c, "spectra", who), who);   // read in place: the run reads the same stored array (its `in`), never a copy
  pr.frequency = frequency_gains(s, c, p, who, needAtoms ? &pr.standardDeviations : nullptr);
  const ng::Array& H = constant_input(c, "package", who);
  const std::vector<double> mass = H.meta.at("mass").get<std::vector<double>>();
  const auto [Psi, mu] = synthesis_of(s, *pr.package.modes, Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size())), who);
  (void)Psi;
  if (c.count("scale") && !c.at("scale").meta.value("placeholder", false)) pr.scale = kernel_gains_of(c.at("scale"), mu, who, "scale").gains;
  if (pr.scale.rows() > 0)                                                                   // the constant (mu 0) lies in the coarsest band of a partition
    for (Eigen::Index j = 0; j < mu.size(); ++j) if (!(mu(j) > 0) && pr.scale.col(j).sum() == 0) pr.scale(0, j) = 1.0;
  pr.p = p;
  return pr;
}

// ── signal.spectral_flow ─────────────────────────────────────────────────────────────────────────────────────────
const char* const kFlow = "signal.spectral_flow";
const char* const kVertexMeasures[] = {"power", "exact_potential", "coexact_stream", "vertex_phase_current_x", "vertex_phase_current_y", "vertex_phase_current_z"};

ng::NodeSpec spectral_flow() {
  ng::NodeSpec n;
  n.name = kFlow;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the field coordinates (signal.chunk_spectra)", false},
              {"package", "the Hodge-Dirac package of the surface the coordinates are on (its eigenvectors operator, with its form parts)", false},
              {"frequency", "the frequency weighting's kernel set on the spectra's frequency line (select_kernels): the atoms. Absent with frequency_weighting every_bin", true},
              {"scale", "the eigenvalue bands, a kernel set on the eigenvalue line (select_kernels; an amplitude partition: cos2_partition). Absent: the whole field only", true}};
  n.outputs = {{"whole_vertices", "per frequency member, per vertex: power, exact_potential, coexact_stream, vertex_phase_current x y z (the whole field)", false},
               {"whole_edges", "per frequency member, per intrinsic edge: the phase current (a 1-form)", false},
               {"whole_modes", "per frequency member, per Hodge-Dirac mode: the phase current's coefficient (exact: travel; coexact: rotation)", false},
               {"pair_vertices", "per frequency member x band x band (the member fastest), per vertex: the six vertex measures of the band pair", false},
               {"pair_edges", "per frequency member x band x band, per edge: the band pair's phase current", false},
               {"pair_modes", "per frequency member x band x band, per mode: the band pair's flow coefficient", false}};
  n.parameters = {
      {"frequency_weighting", Json(), "kernels (the frequency kernel set: atoms) | every_bin (each bin of the spectra alone, weight 1: the whole-recording flow per bin) (no default)", "-", {"kernels", "every_bin"}, false},
      {"frequency_normalisation", Json(), "sum_of_squared_gains (the kernels divided at each bin by the square root of their squared gains' sum: they add back to the whole span) | none (no default with kernels)", "-", {"sum_of_squared_gains", "none"}, true},
      {"support_floor", Json(), "with sum_of_squared_gains: a bin whose squared gains' sum is below this fraction of its largest is outside the set (gain 0) (no default)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run: per frequency member one weighted cross-spectrum (a sum over the chunks and the member's bins), then per band the readings' right factor, per pair only the entries read");
  n.matches = "the flow analysis's per-atom readout (dev check: Im(Psi_tail C Psi_tipᵀ), its projection on the Hodge-Dirac edge rows; nxr's flow analysis's P = grad k Im(C) kᵀ); compute nxr::signal::pairFlow, flowSplit";
  n.description = "the flow of a weighted cross-spectrum in a surface's field coordinates: per frequency member (an atom, or a bin), the whole field's and every band pair's power, phase current per edge and per vertex, "
                  "Hodge-Dirac flow coefficients, potential and stream";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "package"}) if (!in.count(k)) ng::refuse(std::string(kFlow) + ": " + k + " is not given");
    const std::string w = text_param(p, "frequency_weighting", kFlow);
    if (w == "kernels" && !in.count("frequency")) ng::refuse(std::string(kFlow) + ": frequency_weighting kernels needs the frequency kernel set");
    if (w == "kernels") text_param(p, "frequency_normalisation", kFlow);
    ng::Planned out;
    // per member of the frequency kernel set (and of the band pairs): MEASUREMENTS over those members; every_bin's axis is the frequency line's bins (no members): statistics
    const char* kind = w == "kernels" ? "measurement" : "statistic";
    for (const char* o : {"whole_vertices", "whole_edges", "whole_modes", "pair_vertices", "pair_edges", "pair_modes"}) out.out[o] = constant_of(kind);
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p, kFlow, false)); };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(constant_input(in, "spectra", kFlow), kFlow);
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const nxr::signal::Modes& m = *pr.package.modes;
    const Eigen::Map<const Eigen::VectorXd> mass(cx.star0.data(), cx.star0.size());
    const auto [Psi, mu] = synthesis_of(s, m, cx.star0, kFlow);
    const Eigen::SparseMatrix<double> sharpWeights = nxr::signal::sharpOperator(cx);         // the vertex phase current from the edge sums: no gradient matrix formed
    const Eigen::Index S = Psi.rows(), K = Psi.cols(), E = cx.nE, M = m.mu.size(), A = pr.frequency.rows(), Q = pr.scale.rows();
    const Eigen::Index pairs = Q * Q;
    auto measured = [](Eigen::Index rows, Eigen::Index cells, Eigen::Index measures) { ng::Array a; a.rows = rows * cells; a.cols = measures; a.data.assign(static_cast<std::size_t>(rows * cells * measures), 0.0); return a; };
    ng::Array wv = measured(S, A, 6), we = measured(E, A, 1), wm = measured(M, A, 1), pv = measured(S, A * pairs, 6), pe = measured(E, A * pairs, 1), pm = measured(M, A * pairs, 1);
    // one reading's columns j stored as cells cellOf(j): power, potential, stream, vertex phase current per vertex; phase current per edge; flow coefficient per mode
    auto store = [&](ng::Array& vertices, ng::Array& edges, ng::Array& modes, Eigen::Index cells, const std::function<Eigen::Index(Eigen::Index)>& cellOf, const nxr::signal::PairFlow& flow) {
      const nxr::signal::FlowSplit split = nxr::signal::flowSplit(cx, m, flow.pairImaginary, nullptr);   // the complete sensor basis: no remainder
      for (Eigen::Index j = 0; j < flow.pairImaginary.cols(); ++j) {
        const Eigen::Index cell = cellOf(j);
        for (Eigen::Index v = 0; v < S; ++v) {
          const std::size_t r = static_cast<std::size_t>((v * cells + cell) * 6);
          vertices.data[r] = flow.power(v, j); vertices.data[r + 1] = split.potential(v, j); vertices.data[r + 2] = split.stream(v, j);
          for (int comp = 0; comp < 3; ++comp) vertices.data[r + 3 + comp] = flow.vertexCurrent(3 * v + comp, j);
        }
        for (Eigen::Index e = 0; e < E; ++e) edges.data[static_cast<std::size_t>(e * cells + cell)] = flow.pairImaginary(e, j);
        for (Eigen::Index k = 0; k < M; ++k) modes.data[static_cast<std::size_t>(k * cells + cell)] = split.coefficients(k, j);
      }
    };
    std::vector<Eigen::MatrixXd> bandReadouts;                                                 // Psi diag(h_q): each band's readout
    for (Eigen::Index q = 0; q < Q; ++q) bandReadouts.push_back(Psi * pr.scale.row(q).transpose().asDiagonal());
    const std::vector<Eigen::MatrixXd> wholeReadout{Psi};
    Json members = Json::array();
    for (Eigen::Index a = 0; a < A; ++a) {
      const Eigen::MatrixXcd Y = weighted_columns(s, pr.frequency.row(a));                    // the member's weighting of the cross-spectrum, factored: C = Y Yᴴ, a SUM over chunks and bins
      members.push_back(Json{{"member", a}, {"columns", Y.cols()}});
      if (Y.cols() == 0) continue;
      store(wv, we, wm, A, [&](Eigen::Index) { return a; }, nxr::signal::pairFlow(Y, wholeReadout, 1, cx.edges, &sharpWeights));
      if (Q == 0) continue;
      store(pv, pe, pm, A * pairs, [&](Eigen::Index pq) { return a + A * pq; }, nxr::signal::pairFlow(Y, bandReadouts, 1, cx.edges, &sharpWeights));
    }
    Json vertexMeasures = Json::array();
    for (const char* name : kVertexMeasures) vertexMeasures.push_back(Json{{"measure", name}});
    const Json attrs = Json{{"members", "the frequency weighting's members"}, {"frequency_members", members}, {"boundary", pr.package.boundary},
                            {"frequency_weighting", pr.p.value("frequency_weighting", Json())}, {"frequency_normalisation", pr.p.value("frequency_normalisation", Json())},
                            {"sum_over", "the chunks and the member's bins (each sample counted twice with an overlap of half a chunk)"},
                            {"field_coordinates", K}, {"flow_modes", M}, {"sign", "phase current J = Im(conj(z_tail) z_tip): travel from tail to tip gives J < 0"}};
    auto describe = [&](ng::Array& a, const Json& measures, Eigen::Index rows, Eigen::Index cells, const char* rowsAre, const char* membersAre) {
      Json at = attrs; at["rows"] = rowsAre; if (membersAre) at["members"] = membersAre;
      a.meta["describe"] = Json{{"method", "spectral_flow"}, {"measure", "flow"}, {"shape", Json::array({rows, cells, measures.size()})},
                                {"axes", Json::array({indexed_axis("element", rows), indexed_axis("member", cells), indexed_axis("measure", static_cast<std::int64_t>(measures.size()), [&] { Json n = Json::array(); for (const auto& m : measures) n.push_back(m["measure"]); return n; }())})}, {"measures", measures}, {"values_shape", Json::array({rows, cells})}, {"attributes", at}};
      declare_flow_measures(a.meta["describe"]["measures"]);   // the flow measures it writes, declared (flow_measures.h)
    };
    const char* pairsAre = "frequency member x band p x band q, the frequency member fastest (the phase current of the pair: conj(z_p) at the tail, z_q at the tip)";
    describe(wv, vertexMeasures, S, A, "the vertices of the surface", nullptr);
    describe(we, Json::array({Json{{"measure", "phase_current"}}}), E, A, "the edges of the intrinsic complex (tail -> tip)", nullptr);
    describe(wm, Json::array({Json{{"measure", "flow_coefficient"}}}), M, A, "the Hodge-Dirac modes (exact: travel; coexact: rotation)", nullptr);
    describe(pv, vertexMeasures, S, A * pairs, "the vertices of the surface", pairsAre);
    describe(pe, Json::array({Json{{"measure", "phase_current"}}}), E, A * pairs, "the edges of the intrinsic complex (tail -> tip)", pairsAre);
    describe(pm, Json::array({Json{{"measure", "flow_coefficient"}}}), M, A * pairs, "the Hodge-Dirac modes", pairsAre);
    if (Q == 0) for (ng::Array* none : {&pv, &pe, &pm}) { *none = ng::Array{}; none->meta["absent"] = true; }   // no bands wired: no band pairs (the runner stores none)
    return ng::Arrays{{"whole_vertices", std::move(wv)}, {"whole_edges", std::move(we)}, {"whole_modes", std::move(wm)},
                      {"pair_vertices", std::move(pv)}, {"pair_edges", std::move(pe)}, {"pair_modes", std::move(pm)}};
  };
  n.record = [](const std::any& prepared) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    return Json{{"boundary", pr.package.boundary}, {"frequency_members", pr.frequency.rows()}, {"bands", pr.scale.rows()}};
  };
  return n;
}

// ── signal.atom_time_cells ───────────────────────────────────────────────────────────────────────────────────────
// THE TIME CELLS ON A STORED GRID (G7): every atom and band summed over the tiles of the stored time grid the definition names (one level of the line's dyadic tiling), stored as
// MEASUREMENTS over the product grid x atoms x bands; the sign-flip test reads them rolled up to each atom's own window (chunk_spectra_readers.h).
const char* const kCells = "signal.atom_time_cells";
ng::NodeSpec atom_time_cells() {
  ng::NodeSpec n;
  n.name = kCells;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the field coordinates (signal.chunk_spectra)", false},
              {"package", "the Hodge-Dirac package of the surface the coordinates are on", false},
              {"frequency", "the atoms: a kernel set of Gaussians on the spectra's frequency line (their time width sets each atom's own window)", false},
              {"scale", "the eigenvalue bands (a kernel set on the eigenvalue line); each band's field is read on its own", false},
              {"grid", "the stored time grid of the recording (`select_time_grid --rule tiles`): its tiles are the cells", false},
              {"product", "the product the cells are measured over: the grid x the atoms x the bands (`select_product --time-grid --atoms --scales`)", false}};
  n.outputs = {{"edge_cells", "a MEASUREMENT over grid x atoms x bands: per edge the phase current summed over each tile", false},
               {"vertex_cells", "a MEASUREMENT over grid x atoms x bands: per vertex the phase current x y z and the power, summed over each tile", false}};
  n.parameters = {
      {"frequency_weighting", Json("kernels"), "kernels (the only weighting with atoms: the time cells are the atoms')", "-", {"kernels"}, false},
      {"frequency_normalisation", Json(), "sum_of_squared_gains | none: as signal.spectral_flow's (no default)", "-", {"sum_of_squared_gains", "none"}, false},
      {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true},
      {"cell_width", Json(), "each atom's own window (a box): this many of the atom's time standard deviations (1 / (2 pi sigma_f)), snapped to the nearest dyadic level, not finer than the grid's (no default; the flow analysis: 2)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the atom's band signal back in time (the synthesis composed on the atom's bins, one inverse transform), summed over the grid's tiles");
  n.matches = "the flow analysis's time-cell check (dev: z = Psi a per sample, Im(conj z_tail z_tip) summed per cell); compute nxr::signal::timeCellPairFlow; the cells are the stored grid's tiles";
  n.description = "the flow of each atom and band summed over the tiles of a stored time grid: per tile the sums of the edge and vertex phase currents and of the power (measurements over the grid x atoms x bands)";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "package", "frequency", "scale", "grid", "product"}) if (!in.count(k)) ng::refuse(std::string(kCells) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kCells);
    if (!p.contains("cell_width") || !p["cell_width"].is_number() || !(p["cell_width"].get<double>() > 0)) ng::refuse(std::string(kCells) + ": cell_width has no default: name it (> 0)");
    ng::Planned out;
    out.out["edge_cells"] = constant_of("measurement");
    out.out["vertex_cells"] = constant_of("measurement");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    FlowPrepared pr = prepare_flow(c, p, kCells, true);
    pr.grid = constant_input(c, "grid", kCells);
    pr.product = constant_input(c, "product", kCells);
    return std::any(std::move(pr));
  };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(constant_input(in, "spectra", kCells), kCells);
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const auto [Psi, mu] = synthesis_of(s, *pr.package.modes, cx.star0, kCells);
    const Eigen::SparseMatrix<double> sharpWeights = nxr::signal::sharpOperator(cx);
    const Eigen::Index S = Psi.rows(), A = pr.frequency.rows(), Q = pr.scale.rows();
    if (Q == 0) ng::refuse(std::string(kCells) + ": scale is a kernel set with members");
    if (static_cast<std::int64_t>(s.owns.size()) != s.chunks || static_cast<std::int64_t>(s.firsts.size()) != s.chunks) ng::refuse(std::string(kCells) + ": the spectra do not record each chunk's samples");
    const TimeGrid g = time_grid_of(pr.grid, s, kCells);
    check_cells_product(pr.product, g, {A, Q}, kCells);
    const double width = pr.p["cell_width"].get<double>();
    std::vector<int> own(static_cast<std::size_t>(A));
    for (Eigen::Index a = 0; a < A; ++a) own[static_cast<std::size_t>(a)] = own_level(width, pr.standardDeviations[static_cast<std::size_t>(a)], g, kCells);
    std::vector<std::vector<Eigen::MatrixXd>> edges, vertices;
    Json groups = Json::array();
    for (Eigen::Index q = 0; q < Q; ++q) {
      const Eigen::MatrixXd bandReadout = Psi * pr.scale.row(q).transpose().asDiagonal();     // Psi diag(h_q), once per band
      for (Eigen::Index a = 0; a < A; ++a) {
        TileSums t = tile_sums(s, pr.frequency.row(a), g, bandReadout, 1, cx.edges, &sharpWeights);
        Eigen::MatrixXd x(S, g.count), y(S, g.count), z(S, g.count);
        for (Eigen::Index v = 0; v < S; ++v) { x.row(v) = t.vertexCurrent.row(3 * v); y.row(v) = t.vertexCurrent.row(3 * v + 1); z.row(v) = t.vertexCurrent.row(3 * v + 2); }
        edges.push_back({std::move(t.pairImaginary)});
        vertices.push_back({std::move(x), std::move(y), std::move(z), std::move(t.power)});
        const int k = own[static_cast<std::size_t>(a)];
        groups.push_back(Json{{"atom", a}, {"band", q}, {"own_level", k}, {"own_samples", std::int64_t(1) << k}, {"own_seconds", static_cast<double>(std::int64_t(1) << k) / s.sfreq}});
      }
    }
    const Json producer = Json{{"grid", g.record()}, {"cell_width", width}, {"own_window", "a box of cell_width time standard deviations of the atom, snapped to the nearest dyadic level"}};
    return ng::Arrays{{"edge_cells", cells_measurement(edges, {"phase_current"}, g, groups, "the edges of the intrinsic complex", producer)},
                      {"vertex_cells", cells_measurement(vertices, {"vertex_phase_current_x", "vertex_phase_current_y", "vertex_phase_current_z", "power_sum"}, g, groups, "the vertices of the surface", producer)}};
  };
  return n;
}

// ── stats.sign_flip_consistency ──────────────────────────────────────────────────────────────────────────────────
const char* const kFlip = "stats.sign_flip_consistency";
ng::NodeSpec sign_flip_consistency() {
  ng::NodeSpec n;
  n.name = kFlip;
  n.kind = ng::Kind::array;
  n.inputs = {{"edge_cells", "the stored time cells of the edges: a measurement over the time grid x the members (signal.atom_time_cells, signal.cortical_time_cells): read with elements edges", true},
              {"vertex_cells", "the stored time cells of the vertices (phase current x y z): read with elements vertices", true},
              {"whole_edge_cells", "the whole field's stored time cells of the edges, over the time grid x the atoms (signal.cortical_time_cells): tested first, as band -1", true},
              {"whole_vertex_cells", "the whole field's stored time cells of the vertices", true}};
  n.outputs = {{"consistency", "a MEASUREMENT over the members of atoms x bands (the band groups, the atom fastest): per element its R1, p and family-wise p", false},
               {"consistency_summary", "a MEASUREMENT over the same members, one row: the threshold, the median R1, the median's null 95th percentile, its p, the neighbouring cells' correlation, the cells, the groups flipped", false},
               {"whole_consistency", "as consistency, over the atoms, for the whole field's groups (band -1); absent when the cells have none", false},
               {"whole_consistency_summary", "as consistency_summary, over the atoms, for the whole field's groups; absent when the cells have none", false}};
  n.parameters = {
      {"measure_suffix", Json(""), "a suffix on every measure name (e.g. _left, _right: per-hemisphere runs of one recording over one selection); empty: none", "-", {}, false},
      {"elements", Json(), "which values are tested: edges (the edge phase current, one component) | vertices (the vertex phase current, three components) (no default)", "-", {"edges", "vertices"}, false},
      {"patterns", Json(), "the random sign patterns over the cells (no default; nxr's flow analysis: 500 for its spectrum, 2000 for its maps)", "-", {}, false},
      {"seed", Json(), "the seed of numpy's default_rng, created once a run and drawn in the groups' order (no default; nxr's flow analysis: 0)", "-", {}, false},
      {"cells_per_flip", Json(), "consecutive cells one sign flips together (no default)", "cells", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a statistic of the cells: unit vectors, resultants, sign patterns");
  n.matches = "nxr's flow maps flip_test (nxr's flow analysis: R1, per-vertex p, family-wise maximum, numpy default_rng); compute nxr::stats::signFlipConsistency, NumpyGenerator";
  n.description = "the sign-flip (time-reversal) test of the direction's consistency over each atom's own time cells (the stored tiles rolled up to the atom's own window), per element, with a family-wise maximum";
  n.plan = [](const ng::Types& in, const Json& p) {
    const std::string elements = text_param(p, "elements", kFlip);
    const bool e = elements == "edges";
    if (!in.count(e ? "edge_cells" : "vertex_cells") && !in.count(e ? "whole_edge_cells" : "whole_vertex_cells"))
      ng::refuse(std::string(kFlip) + ": elements " + elements + " reads " + (e ? "edge_cells" : "vertex_cells") + " (or the whole field's), which is not given"); integer_param(p, "patterns", kFlip); integer_param(p, "seed", kFlip); integer_param(p, "cells_per_flip", kFlip);
    ng::Planned out;
    for (const char* o : {"consistency", "consistency_summary", "whole_consistency", "whole_consistency_summary"}) out.out[o] = constant_of("measurement");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    const bool edges = p["elements"].get<std::string>() == "edges";
    std::vector<ng::Array> cells;                                           // the whole field's cells first (band -1), then the bands': the groups' order of the generator
    for (const char* port : {edges ? "whole_edge_cells" : "whole_vertex_cells", edges ? "edge_cells" : "vertex_cells"})
      if (c.count(port) && !c.at(port).meta.value("placeholder", false) && !c.at(port).meta.value("absent", false)) cells.push_back(c.at(port));
    if (cells.empty()) ng::refuse(std::string(kFlip) + ": no stored time cells were handed over");
    return std::any(std::make_pair(std::move(cells), p));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [cellSets, p] = std::any_cast<const std::pair<std::vector<ng::Array>, Json>&>(prepared);
    // THE STORED CELLS (G7): per member, the grid's tiles rolled up to the atom's own window (sums of consecutive tiles; whole cells only), then tested as before
    const bool edges = p["elements"].get<std::string>() == "edges";
    const int components = edges ? 1 : 3;
    nxr::stats::NumpyGenerator generator(static_cast<std::uint64_t>(p["seed"].get<std::int64_t>()));
    std::int64_t elements = -1;
    std::vector<nxr::stats::SignFlipConsistency> tests;
    Json members = Json::array();
    for (const ng::Array& cells : cellSets) {
      const StoredCells sc = stored_cells(cells, kFlip);
      std::vector<std::int64_t> picked_measures;
      if (edges) picked_measures = {sc.measure_index("phase_current", kFlip)};
      else for (const char* m : {"vertex_phase_current_x", "vertex_phase_current_y", "vertex_phase_current_z"}) picked_measures.push_back(sc.measure_index(m, kFlip));
      if (elements >= 0 && elements != sc.rows) ng::refuse(std::string(kFlip) + ": the whole field's cells and the bands' are not on the same elements");
      elements = sc.rows;
      const std::int64_t width = elements * components;
      for (std::size_t gi = 0; gi < sc.groups.size(); ++gi) {
        const Json& g = sc.groups[gi];
        std::vector<Eigen::MatrixXd> parts;
        for (const auto m : picked_measures) parts.push_back(sc.rolled(static_cast<std::int64_t>(gi), m));
        const std::int64_t count = parts[0].rows();
        if (count < 2) ng::refuse(std::string(kFlip) + ": an atom x band has fewer than two whole cells of its own window");
        Eigen::MatrixXd values(count, width);
        for (std::int64_t r = 0; r < count; ++r)
          for (std::int64_t e = 0; e < elements; ++e)
            for (int k = 0; k < components; ++k) values(r, e * components + k) = parts[static_cast<std::size_t>(k)](r, e);
        tests.push_back(nxr::stats::signFlipConsistency(values, components, p["patterns"].get<std::int64_t>(), generator, p["cells_per_flip"].get<std::int64_t>()));
        members.push_back(Json{{"atom", g["atom"]}, {"band", g["band"]}, {"own_level", g.value("own_level", sc.level)}, {"cells", count}});
      }
    }
    // MEASUREMENTS over the members, rows x members x measures, value (r members + m) measures + k: the band groups (band >= 0: atom x band, the atom fastest, the
    // groups' order) and the whole field's groups (band -1: the atoms), each its own pair of outputs (absent when the cells have none)
    const Json options = Json{{"elements", p["elements"]}, {"patterns", p["patterns"]}, {"seed", p["seed"]}, {"cells_per_flip", p["cells_per_flip"]}, {"generator", "numpy default_rng (SeedSequence, PCG64), random() < 0.5 -> -1"}};
    auto measured = [&](bool whole) {
      std::vector<std::size_t> picked;
      Json picked_members = Json::array();
      for (std::size_t m = 0; m < tests.size(); ++m)
        if ((members[m].value("band", 0) < 0) == whole) { picked.push_back(m); picked_members.push_back(members[m]); }
      const std::int64_t G = static_cast<std::int64_t>(picked.size());
      ng::Array perElement, summary;
      if (G == 0) { perElement.meta["absent"] = true; summary.meta["absent"] = true; return std::make_pair(perElement, summary); }
      perElement.rows = elements * G; perElement.cols = 3;
      perElement.data.assign(static_cast<std::size_t>(elements * G * 3), 0.0);
      for (std::int64_t e = 0; e < elements; ++e)
        for (std::int64_t j = 0; j < G; ++j) {
          const auto& t = tests[picked[static_cast<std::size_t>(j)]];
          const std::size_t at = static_cast<std::size_t>((e * G + j) * 3);
          perElement.data[at] = t.consistency(e); perElement.data[at + 1] = t.p(e); perElement.data[at + 2] = t.familywiseP(e);
        }
      summary.rows = G; summary.cols = 7;
      for (const std::size_t m : picked) {
        const auto& t = tests[m];
        for (const double v : {t.threshold, t.median, t.medianNull95, t.medianP, t.neighbourCorrelation, static_cast<double>(t.cells), static_cast<double>(t.groups)}) summary.data.push_back(v);
      }
      auto describe = [&](ng::Array& a, const std::vector<const char*>& names, std::int64_t rows, const char* rowsAre) {
        Json measures = Json::array();
        for (const char* m : names) measures.push_back(Json{{"measure", m}});
        a.meta["describe"] = Json{{"method", "sign_flip_consistency"}, {"measures", measures}, {"shape", Json::array({rows, G, static_cast<std::int64_t>(names.size())})},
                                  {"values_shape", Json::array({rows, G})}, {"attributes", Json{{"rows", rowsAre}, {"members", picked_members}, {"options", options}}}};
        declare_flow_measures(a.meta["describe"]["measures"]);   // the flow measures it writes, declared (flow_measures.h)
      };
      describe(perElement, {"consistency", "p", "familywise_p"}, elements, edges ? "the edges of the intrinsic complex" : "the vertices of the surface");
      describe(summary, {"threshold", "median_consistency", "median_null_95", "median_p", "neighbour_correlation", "cells", "groups_flipped"}, 1, "one row: the member's summary");
      return std::make_pair(perElement, summary);
    };
    auto [banded, bandedSummary] = measured(false);
    auto [whole, wholeSummary] = measured(true);
    for (ng::Array* a : {&banded, &bandedSummary, &whole, &wholeSummary}) suffix_measures(*a, p.value("measure_suffix", std::string()));
    return ng::Arrays{{"consistency", std::move(banded)}, {"consistency_summary", std::move(bandedSummary)}, {"whole_consistency", std::move(whole)}, {"whole_consistency_summary", std::move(wholeSummary)}};
  };
  return n;
}

// ── signal.sensor_power and stats.aperiodic_fit ──────────────────────────────────────────────────────────────────
const char* const kPower = "signal.sensor_power";
ng::NodeSpec sensor_power() {
  ng::NodeSpec n;
  n.name = kPower;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the field coordinates (signal.chunk_spectra)", false}, {"package", "the Hodge-Dirac package of the surface the coordinates are on", false}};
  n.outputs = {{"power", "each vertex's (sensor's) power per bin: the mean over the chunks of |Psi x|^2 [vertices x bins]", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a measurement over the chunk partition (the mean), bin by bin");
  n.matches = "the diagonal of Psi C_bin Psiᵀ / chunks, made from the coordinates (never the S x S matrix)";
  n.description = "each sensor's power spectrum from the chunk spectra: the field synthesised at the vertices, its squared magnitude averaged over the chunks";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"spectra", "package"}) if (!in.count(k)) ng::refuse(std::string(kPower) + ": " + k + " is not given");
    ng::Planned out;
    out.out["power"] = constant_of("statistic");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json&, ng::Shared&) {
    return std::any(std::make_pair(bind_hodge_package(constant_input(c, "package", kPower), kPower), constant_input(c, "spectra", kPower)));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [package, A] = std::any_cast<const std::pair<BoundPackage, ng::Array>&>(prepared);
    const Spectra s = spectra_of(A, kPower);
    const auto [Psi, mu] = synthesis_of(s, *package.modes, package.complex->star0, kPower);
    const Eigen::Index S = Psi.rows();
    ng::Array out;
    out.rows = S; out.cols = s.bins;
    out.data.assign(static_cast<std::size_t>(S * s.bins), 0.0);
    for (std::int64_t t = 0; t < s.chunks; ++t) {
      Eigen::MatrixXcd X(s.rows, s.bins);
      for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t r = 0; r < s.rows; ++r) X(r, k) = s.at(t, k, r);
      const Eigen::MatrixXd P = (Psi.cast<cd>() * X).cwiseAbs2();
      for (Eigen::Index v = 0; v < S; ++v) for (std::int64_t k = 0; k < s.bins; ++k) out.data[static_cast<std::size_t>(v * s.bins + k)] += P(v, k) / static_cast<double>(s.chunks);
    }
    std::vector<double> f;
    for (std::int64_t k = 0; k < s.bins; ++k) f.push_back(s.frequency(k));
    out.meta["frequencies"] = f;
    out.meta["describe"] = Json{{"statistic", "power"}, {"method", "sensor_power"}, {"measure", "power"}, {"shape", Json::array({S, s.bins})}, {"axes", Json::array({indexed_axis("vertex", S), indexed_axis("frequency", s.bins, f)})}, {"frequencies", f}, {"chunks", s.chunks},
                                {"unit", "(the recording's unit x samples)^2: the unscaled transform's squared magnitude, mean over the chunks"}};
    return ng::Arrays{{"power", std::move(out)}};
  };
  return n;
}

const char* const kAperiodic = "stats.aperiodic_fit";
ng::NodeSpec aperiodic_fit() {
  ng::NodeSpec n;
  n.name = kAperiodic;
  n.kind = ng::Kind::array;
  n.inputs = {{"power", "spectra [rows x bins] with their frequencies (signal.sensor_power)", false}};
  n.outputs = {{"fit", "per row: offset, exponent, knee, r^2, points used (the aperiodic background; the periodic share 1 - background / power is derived at read)", false}};
  n.parameters = {{"knee", Json(), "true: offset - log10(knee + f^exponent); false: offset - exponent log10 f (no default; nxr's flow analysis ran both)", "-", {}, false},
                  {"fit_range", Json(), "[low, high] the frequencies fitted (no default; nxr's flow analysis: [1, 45])", "Hz", {}, false},
                  {"iterations", Json(), "the fit's passes, points above the fit dropped between them (no default; specparam's port: 3)", "-", {}, false},
                  {"threshold", Json(), "a point above the fit by more than this many standard deviations is dropped (no default; specparam's port: 2)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a statistic of each spectrum");
  n.matches = "nxr nxr.spectral.aperiodic.fit (the lab's specparam port); compute nxr::signal::fitAperiodic";
  n.description = "the aperiodic background of each spectrum (specparam's aperiodic model, with or without a knee)";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("power")) ng::refuse(std::string(kAperiodic) + ": power is not given");
    if (!p.contains("knee") || !p["knee"].is_boolean()) ng::refuse(std::string(kAperiodic) + ": knee has no default: name it (true or false)");
    if (!p.contains("fit_range") || !p["fit_range"].is_array() || p["fit_range"].size() != 2) ng::refuse(std::string(kAperiodic) + ": fit_range [low, high] has no default: name it");
    integer_param(p, "iterations", kAperiodic);
    if (!p.contains("threshold") || !p["threshold"].is_number()) ng::refuse(std::string(kAperiodic) + ": threshold has no default: name it");
    ng::Planned out;
    out.out["fit"] = constant_of("statistic");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(constant_input(c, "power", kAperiodic), p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [P, p] = std::any_cast<const std::pair<ng::Array, Json>&>(prepared);
    const std::vector<double> fv = P.meta.value("frequencies", std::vector<double>());
    if (static_cast<std::int64_t>(fv.size()) != P.cols) ng::refuse(std::string(kAperiodic) + ": power does not record its frequencies");
    const Eigen::Map<const Eigen::VectorXd> f(fv.data(), P.cols);
    ng::Array out;
    out.rows = P.rows; out.cols = 5;
    for (std::int64_t r = 0; r < P.rows; ++r) {
      Eigen::VectorXd row(P.cols);
      for (std::int64_t k = 0; k < P.cols; ++k) row(k) = P.at(static_cast<std::size_t>(r * P.cols + k));
      const nxr::signal::AperiodicFit a = nxr::signal::fitAperiodic(row, f, p["knee"].get<bool>(), static_cast<int>(p["iterations"].get<std::int64_t>()), p["threshold"].get<double>(),
                                                                    p["fit_range"][0].get<double>(), p["fit_range"][1].get<double>());
      for (const double v : {a.offset, a.exponent, a.knee, a.r2, static_cast<double>(a.pointsUsed)}) out.data.push_back(v);
    }
    out.meta["describe"] = Json{{"statistic", "aperiodic_fit"}, {"method", "aperiodic_fit"}, {"measure", "aperiodic"}, {"shape", Json::array({out.rows, 5})}, {"axes", Json::array({indexed_axis("row", out.rows), indexed_axis("parameter", 5, Json::array({"offset", "exponent", "knee", "r2", "points_used"}))})}, {"columns", Json::array({"offset", "exponent", "knee", "r2", "points_used"})},
                                {"options", p}, {"model", p["knee"].get<bool>() ? "offset - log10(knee + f^exponent)" : "offset - exponent log10 f"}};
    return ng::Arrays{{"fit", std::move(out)}};
  };
  return n;
}

}  // namespace

void register_sensor_flow_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {overlapped_spectrum(), chunk_spectra(), spectral_flow(), atom_time_cells(), sign_flip_consistency(), sensor_power(), aperiodic_fit()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
