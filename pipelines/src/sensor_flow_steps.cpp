// nxr-pipelines — THE FLOW OF A WEIGHTED CROSS-SPECTRUM ON A SURFACE'S FIELD COORDINATES as graph nodes (spec docs/specs/2026-11-24-sensor-flow.md; definitions `sensor_flow`).
//
// THE GRAPH. before: eigenvectors + channel map --[signal.mode_analysis, modes exact_and_constant]--> the serving operator [K x channels], bound once.
// chunks: recording (each chunk with `overlap_samples` on each side from its neighbours, reflected at the recording's ends) --[signal.overlapped_spectrum]--> the chunk's spectrum [K x B]
//         once a block --[signal.chunk_spectra]--> [chunks x B x K]: the cross-spectrum in its exact factored form (one value per member of the chunk grid; nothing summed).
// after:  chunk spectra + eigenvectors + a frequency weighting (a kernel set: the atoms; or every bin) + eigenvalue bands --[signal.spectral_flow]--> per frequency member the whole field's
//         and every band pair's power, phase currents, flow coefficients, potential and stream;
//         --[signal.atom_time_cells]--> per atom and band, the time cells' sums; --[stats.sign_flip_consistency]--> R1 and its sign-flip null per atom and band;
//         --[signal.sensor_power]--> each sensor's power per bin --[stats.aperiodic_fit]--> its aperiodic background (an option of the definition);
//         a control (the empty room on the same sensor surface): the same chain, then --[stats.control_comparison]--> its sign-flip summary beside the run's, length-matched.
// Every node is a function of arrays: no file, no subject, no loop over a recording (the chunk service streams; the stored chunk spectra are an array handed over).
#include "nxr/pipelines/sensor_flow_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/pipelines/form_steps.h"
#include "nxr/signal/aperiodic.h"
#include "nxr/signal/intrinsic_forms.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/stats/sign_flip.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using cd = std::complex<double>;

namespace {

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
  n.parameters = {
      {"overlap_samples", Json(), "the overlap the spectra were made with (as signal.overlapped_spectrum's)", "samples", {}, false},
      {"transform_samples", Json(), "the transform's length the spectra were made with (as signal.overlapped_spectrum's)", "samples", {}, false},
      {"frequency_span", Json(), "[lo, hi) the spectra keep (as signal.overlapped_spectrum's)", "Hz", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float32, ng::OutputPrecision::follows_input,
                             "each block's spectrum kept in time order (a measurement over the chunk partition: one value a member, nothing summed)");
  n.matches = "the flow analysis's served chunk spectra (each chunk's rfft in the field coordinates, stored); no reduction";
  n.description = "the chunk spectra of a run gathered into one statistic: the cross-spectrum per bin is the sum over its chunks of each spectrum times its conjugate";
  n.plan = [](const ng::Types& in, const Json& p) {
    const auto it = in.find("spectrum");
    if (it == in.end() || !it->second.per_block()) ng::refuse(std::string(kGather) + ": spectrum is a value made once per block (signal.overlapped_spectrum)");
    if (!in.count("operator")) ng::refuse(std::string(kGather) + ": operator is not given");
    integer_param(p, "overlap_samples", kGather); integer_param(p, "transform_samples", kGather); span_param(p, kGather);
    ng::Planned out;
    out.out["spectra"] = constant_of("statistic");
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json& p, ng::Shared&) {
    GatherPrepared pr;
    pr.op_meta = constant_input(c, "operator", kGather).meta;
    pr.op_meta.erase("channels");
    pr.overlap = p["overlap_samples"].get<std::int64_t>(); pr.transform = p["transform_samples"].get<std::int64_t>();
    const auto it = in.find("spectrum");
    if (it == in.end() || !it->second.sfreq) ng::refuse(std::string(kGather) + ": spectrum has no sampling rate");
    pr.sfreq = *it->second.sfreq;
    const auto [lo, hi] = span_param(p, kGather);
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
    for (std::int64_t k = 0; k < b.width; ++k)                                            // bin-major then row, as the coefficients of `spectral_coefficients`
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

// ── what the readers of the chunk spectra share ─────────────────────────────────────────────────────────────────
struct Spectra {
  std::int64_t chunks = 0, bins = 0, rows = 0, first = 0, transform = 0, overlap = 0;
  double sfreq = 0;
  std::vector<std::int64_t> modes, owns;
  bool unit_mass = true;
  const ng::Array* array = nullptr;
  cd at(std::int64_t t, std::int64_t k, std::int64_t r) const {
    const std::size_t i = static_cast<std::size_t>(((t * bins + k) * rows + r) * 2);
    return cd(array->at(i), array->at(i + 1));
  }
  double frequency(std::int64_t k) const { return static_cast<double>(first + k) * sfreq / static_cast<double>(transform); }
};
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
  for (const auto& c : pr.value("chunks", Json::array())) s.owns.push_back(c.value("samples", std::int64_t(0)));
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
// a kernel set's members (select.kernels), and their gains on points of the line's coordinate (Hz on a frequency line; mu on an eigenvalue line)
std::vector<Json> members_of(const ng::Array& sel, const char* who, const char* port) {
  const Json p = sel.meta.value("params", Json::object());
  if (!p.contains("kernels") || !p["kernels"].contains("members")) ng::refuse(std::string(who) + ": " + port + " is not a kernel set (`nxr-neuro run select_kernels`)");
  return p["kernels"]["members"].get<std::vector<Json>>();
}
Eigen::MatrixXd gains_of(const std::vector<Json>& members, const Eigen::VectorXd& x, const char* who) {
  const Eigen::Index M = static_cast<Eigen::Index>(members.size());
  if (M == 0) ng::refuse(std::string(who) + ": a kernel set with no member");
  const std::string family = members.front().value("family", std::string());
  for (const auto& k : members) if (k.value("family", std::string()) != family) ng::refuse(std::string(who) + ": a kernel set's members have one profile");
  if (family == "cos2_partition") {
    std::vector<double> centres;
    for (const auto& k : members) centres.push_back(std::log2(k.at("start").get<double>()));
    Eigen::VectorXd lx(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) lx(i) = x(i) > 0 ? std::log2(x(i)) : -std::numeric_limits<double>::infinity();
    return nxr::signal::cosineSquaredPartition(lx, centres);
  }
  if (family == "mexican_hat_partition") {                                                  // the graph bank's hats in lambda = mu², divided by their sum (x: the eigenvalue line's mu)
    std::vector<double> centres;
    for (const auto& k : members) centres.push_back(k.at("start").get<double>());
    return nxr::signal::mexicanHatPartition(x.cwiseMax(0.0), centres);
  }
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(M, x.size());
  for (Eigen::Index m = 0; m < M; ++m) {
    const Json& k = members[static_cast<std::size_t>(m)];
    const double c = k.at("start").get<double>(), gain = k.value("gain", 1.0);
    const std::string chart = k.value("chart", std::string("linear"));
    const Json params = k.value("params", Json::object());
    double fwhm = params.value("fwhm", 0.0);
    if (params.value("width_rule", std::string("absolute")) == "proportional") fwhm *= c;
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      if (!(x(i) > 0)) continue;                                                            // one-sided: nothing at 0 (a frequency's mean, an eigenvalue's constant)
      const double u = chart == "log2" ? std::log2(x(i)) - std::log2(c) : x(i) - c;
      if (family == "gaussian") { const double sd = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0))); g(m, i) = gain * std::exp(-u * u / (2.0 * sd * sd)); }
      else if (family == "box") g(m, i) = std::abs(u) <= fwhm / 2.0 ? gain : 0.0;
      else ng::refuse(std::string(who) + ": a kernel's profile is gaussian, box, cos2_partition or mexican_hat_partition, not " + family);
    }
  }
  return g;
}
// the frequency weighting's members: each a gain per bin (the atoms; normalised as the parameter says) or every bin alone
Eigen::MatrixXd frequency_gains(const Spectra& s, const ng::Arrays& c, const Json& p, const char* who, std::vector<double>* standardDeviations = nullptr) {
  const std::string weighting = text_param(p, "frequency_weighting", who);
  Eigen::VectorXd f(s.bins);
  for (std::int64_t k = 0; k < s.bins; ++k) f(k) = s.frequency(k);
  if (weighting == "every_bin") {
    if (standardDeviations) ng::refuse(std::string(who) + ": every_bin has no atom (no time width)");
    return Eigen::MatrixXd::Identity(s.bins, s.bins);
  }
  if (weighting != "kernels") ng::refuse(std::string(who) + ": frequency_weighting is kernels or every_bin");
  const std::vector<Json> members = members_of(constant_input(c, "frequency", who), who, "frequency");
  Eigen::MatrixXd g = gains_of(members, f, who);
  if (standardDeviations)
    for (const auto& k : members) {
      const Json params = k.value("params", Json::object());
      double fwhm = params.value("fwhm", 0.0);
      if (params.value("width_rule", std::string("absolute")) == "proportional") fwhm *= k.at("start").get<double>();
      if (k.value("family", std::string()) != "gaussian" || k.value("chart", std::string()) != "linear") ng::refuse(std::string(who) + ": the time cells need Gaussian atoms on a linear chart");
      standardDeviations->push_back(fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0))));
    }
  const std::string normalisation = text_param(p, "frequency_normalisation", who);
  if (normalisation == "sum_of_squared_gains") {
    if (!p.contains("support_floor") || !p["support_floor"].is_number()) ng::refuse(std::string(who) + ": support_floor has no default: name it");
    g = nxr::signal::normaliseBySumOfSquaredGains(g, p["support_floor"].get<double>());
  } else if (normalisation != "none") ng::refuse(std::string(who) + ": frequency_normalisation is sum_of_squared_gains or none");
  return g;
}
// one frequency member's weighted columns Y [K x chunks.bins] (each bin's column times its gain; the cross-spectrum it weights is Y Yᴴ)
Eigen::MatrixXcd weighted_columns(const Spectra& s, const Eigen::RowVectorXd& gain, double floorFraction = 1e-12) {
  const double top = gain.cwiseAbs().maxCoeff();
  std::vector<std::int64_t> bins;
  for (std::int64_t k = 0; k < s.bins; ++k) if (std::abs(gain(k)) > floorFraction * top && gain(k) != 0) bins.push_back(k);
  Eigen::MatrixXcd Y(s.rows, s.chunks * static_cast<std::int64_t>(bins.size()));
  for (std::int64_t t = 0; t < s.chunks; ++t)
    for (std::size_t q = 0; q < bins.size(); ++q)
      for (std::int64_t r = 0; r < s.rows; ++r) Y(r, t * static_cast<std::int64_t>(bins.size()) + static_cast<std::int64_t>(q)) = gain(bins[q]) * s.at(t, bins[q], r);
  return Y;
}

struct FlowPrepared {
  nxr::pipelines::graph::BoundPackage package;
  const ng::Array* spectra_array = nullptr;
  ng::Array spectra_copy;
  Eigen::MatrixXd frequency, scale;          // [members x bins], [bands x K] (scale empty: no bands)
  std::vector<double> standardDeviations;
  Json p;
};
FlowPrepared prepare_flow(const ng::Arrays& c, const Json& p, const char* who, bool needAtoms) {
  FlowPrepared pr;
  pr.package = bind_hodge_package(constant_input(c, "package", who), who);
  pr.spectra_copy = constant_input(c, "spectra", who);
  const Spectra s = spectra_of(pr.spectra_copy, who);
  pr.frequency = frequency_gains(s, c, p, who, needAtoms ? &pr.standardDeviations : nullptr);
  const ng::Array& H = constant_input(c, "package", who);
  const std::vector<double> mass = H.meta.at("mass").get<std::vector<double>>();
  const auto [Psi, mu] = synthesis_of(s, *pr.package.modes, Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size())), who);
  (void)Psi;
  if (c.count("scale") && !c.at("scale").meta.value("placeholder", false)) pr.scale = gains_of(members_of(c.at("scale"), who, "scale"), mu, who);
  if (pr.scale.rows() > 0)                                                                   // the constant (mu 0) lies in the coarsest band of a partition
    for (Eigen::Index j = 0; j < mu.size(); ++j) if (!(mu(j) > 0) && pr.scale.col(j).sum() == 0) pr.scale(0, j) = 1.0;
  pr.p = p;
  return pr;
}

// ── signal.spectral_flow ─────────────────────────────────────────────────────────────────────────────────────────
const char* const kFlow = "signal.spectral_flow";
// the vertex measures: power; the split's potential and stream (of the split input); the vertex phase current; the parity split's potential and stream (of the other 1-form)
const char* const kVertexMeasures[] = {"power", "exact_potential", "coexact_stream", "vertex_phase_current_x", "vertex_phase_current_y", "vertex_phase_current_z",
                                       "parity_exact_potential", "parity_coexact_stream"};
constexpr int kVertexMeasureCount = 8;
const char* const kModeMeasures[] = {"flow_coefficient", "parity_flow_coefficient"};
constexpr int kModeMeasureCount = 2;
// the 1-forms the split may read: the edge phase current itself, or the vertex phase current carried to the edges by `flat` (the field integrated along each edge)
bool split_reads_edges(const Json& p, const char* who) {
  const std::string input = text_param(p, "split_input", who);
  if (input != "edge_phase_current" && input != "vertex_phase_current") ng::refuse(std::string(who) + ": split_input is edge_phase_current or vertex_phase_current, not " + input);
  return input == "edge_phase_current";
}

ng::NodeSpec spectral_flow() {
  ng::NodeSpec n;
  n.name = kFlow;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the field coordinates (signal.chunk_spectra)", false},
              {"package", "the Hodge-Dirac package of the surface the coordinates are on (its eigenvectors operator, with its form parts)", false},
              {"frequency", "the frequency weighting's kernel set on the spectra's frequency line (select_kernels): the atoms. Absent with frequency_weighting every_bin", true},
              {"scale", "the eigenvalue bands, a kernel set on the eigenvalue line (select_kernels; an amplitude partition: cos2_partition). Absent: the whole field only", true}};
  n.outputs = {{"whole_vertices", "per frequency member, per vertex: power, exact_potential, coexact_stream (of the split input), vertex_phase_current x y z, parity_exact_potential, parity_coexact_stream (of the other 1-form) (the whole field)", false},
               {"whole_edges", "per frequency member, per intrinsic edge: the phase current (a 1-form)", false},
               {"whole_modes", "per frequency member, per Hodge-Dirac mode: the split input's coefficient (exact: travel; coexact: rotation) and the other 1-form's (parity)", false},
               {"pair_vertices", "per frequency member x band x band (the member fastest), per vertex: the eight vertex measures of the band pair", false},
               {"pair_edges", "per frequency member x band x band, per edge: the band pair's phase current", false},
               {"pair_modes", "per frequency member x band x band, per mode: the band pair's two flow coefficients", false}};
  n.parameters = {
      {"frequency_weighting", Json(), "kernels (the frequency kernel set: atoms) | every_bin (each bin of the spectra alone, weight 1: the whole-recording flow per bin) (no default)", "-", {"kernels", "every_bin"}, false},
      {"frequency_normalisation", Json(), "sum_of_squared_gains (the kernels divided at each bin by the square root of their squared gains' sum: they add back to the whole span) | none (no default with kernels)", "-", {"sum_of_squared_gains", "none"}, true},
      {"support_floor", Json(), "with sum_of_squared_gains: a bin whose squared gains' sum is below this fraction of its largest is outside the set (gain 0) (no default)", "-", {}, true},
      {"split_input", Json(), "the 1-form the Hodge-Dirac split reads: edge_phase_current (the phase current on the intrinsic edges, what the modes' edge rows act on) | vertex_phase_current "
                              "(the vertex phase current carried to the edges by flat, nxr's flow analysis's |w|² grad theta); the other is split too and stored as the parity measures (no default)",
       "-", {"edge_phase_current", "vertex_phase_current"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run: per frequency member one weighted cross-spectrum (a sum over the chunks and the member's bins), then per band the readings' right factor, per pair only the entries read");
  n.matches = "the flow analysis's per-atom readout (dev check: Im(Psi_tail C Psi_tipᵀ), its projection on the Hodge-Dirac edge rows; nxr's flow analysis's P = grad k Im(C) kᵀ); compute nxr::signal::bandPairFlow, projectOneForm, exactPotential, coexactStream";
  n.description = "the flow of a weighted cross-spectrum in a surface's field coordinates: per frequency member (an atom, or a bin), the whole field's and every band pair's power, phase current per edge and per vertex, "
                  "Hodge-Dirac flow coefficients, potential and stream";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "package"}) if (!in.count(k)) ng::refuse(std::string(kFlow) + ": " + k + " is not given");
    const std::string w = text_param(p, "frequency_weighting", kFlow);
    if (w == "kernels" && !in.count("frequency")) ng::refuse(std::string(kFlow) + ": frequency_weighting kernels needs the frequency kernel set");
    if (w == "kernels") text_param(p, "frequency_normalisation", kFlow);
    split_reads_edges(p, kFlow);
    ng::Planned out;
    for (const char* o : {"whole_vertices", "whole_edges", "whole_modes", "pair_vertices", "pair_edges", "pair_modes"}) { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "statistic"; out.out[o] = t; }
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p, kFlow, false)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(pr.spectra_copy, kFlow);
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const nxr::signal::Modes& m = *pr.package.modes;
    const Eigen::Map<const Eigen::VectorXd> mass(cx.star0.data(), cx.star0.size());
    const auto [Psi, mu] = synthesis_of(s, m, cx.star0, kFlow);
    const Eigen::MatrixXd G = nxr::signal::sharp(cx, Eigen::MatrixXd(cx.d0 * Psi));            // [3S x K] each coordinate's tangent gradient
    const Eigen::Index S = Psi.rows(), K = Psi.cols(), E = cx.nE, M = m.mu.size(), A = pr.frequency.rows(), Q = pr.scale.rows();
    const Eigen::Index pairs = Q * Q;
    auto measured = [](Eigen::Index rows, Eigen::Index cells, Eigen::Index measures) { ng::Array a; a.rows = rows * cells; a.cols = measures; a.data.assign(static_cast<std::size_t>(rows * cells * measures), 0.0); return a; };
    const bool splitEdges = split_reads_edges(pr.p, kFlow);
    ng::Array wv = measured(S, A, kVertexMeasureCount), we = measured(E, A, 1), wm = measured(M, A, kModeMeasureCount),
              pv = measured(S, A * pairs, kVertexMeasureCount), pe = measured(E, A * pairs, 1), pm = measured(M, A * pairs, kModeMeasureCount);
    // the split of both 1-forms: the one the caller names (split_input) is THE flow; the other is kept for parity (both cost one projection on the modes' edge rows)
    auto store = [&](ng::Array& vertices, ng::Array& edges, ng::Array& modes, Eigen::Index cells, Eigen::Index cell, const Eigen::VectorXd& power, const Eigen::VectorXd& edge, const Eigen::VectorXd& vertex) {
      const Eigen::MatrixXd edgeCoefficients = nxr::signal::projectOneForm(cx, m, edge);
      const Eigen::MatrixXd vertexCoefficients = nxr::signal::projectOneForm(cx, m, nxr::signal::flat(cx, vertex));
      const Eigen::MatrixXd& coef = splitEdges ? edgeCoefficients : vertexCoefficients;
      const Eigen::MatrixXd& parity = splitEdges ? vertexCoefficients : edgeCoefficients;
      const Eigen::MatrixXd potential = nxr::signal::exactPotential(m, coef), stream = nxr::signal::coexactStream(cx, m, coef);
      const Eigen::MatrixXd parityPotential = nxr::signal::exactPotential(m, parity), parityStream = nxr::signal::coexactStream(cx, m, parity);
      for (Eigen::Index v = 0; v < S; ++v) {
        const std::size_t r = static_cast<std::size_t>((v * cells + cell) * kVertexMeasureCount);
        vertices.data[r] = power(v); vertices.data[r + 1] = potential(v, 0); vertices.data[r + 2] = stream(v, 0);
        for (int comp = 0; comp < 3; ++comp) vertices.data[r + 3 + comp] = vertex(3 * v + comp);
        vertices.data[r + 6] = parityPotential(v, 0); vertices.data[r + 7] = parityStream(v, 0);
      }
      for (Eigen::Index e = 0; e < E; ++e) edges.data[static_cast<std::size_t>(e * cells + cell)] = edge(e);
      for (Eigen::Index k = 0; k < M; ++k) {
        const std::size_t r = static_cast<std::size_t>((k * cells + cell) * kModeMeasureCount);
        modes.data[r] = coef(k, 0); modes.data[r + 1] = parity(k, 0);
      }
    };
    Json members = Json::array();
    for (Eigen::Index a = 0; a < A; ++a) {
      const Eigen::MatrixXcd Y = weighted_columns(s, pr.frequency.row(a));
      members.push_back(Json{{"member", a}, {"columns", Y.cols()}});
      if (Y.cols() == 0) continue;
      const Eigen::MatrixXcd C = Y * Y.adjoint();                                             // the member's weighting of the cross-spectrum: a SUM over chunks and bins
      const nxr::signal::BandPairFlow whole = nxr::signal::bandPairFlow(C, Psi, G, cx.edges, Eigen::MatrixXd::Ones(1, K));
      store(wv, we, wm, A, a, whole.power.col(0), whole.edgeCurrent.col(0), whole.vertexCurrent.col(0));
      if (Q == 0) continue;
      const nxr::signal::BandPairFlow bp = nxr::signal::bandPairFlow(C, Psi, G, cx.edges, pr.scale);
      for (Eigen::Index pq = 0; pq < pairs; ++pq) store(pv, pe, pm, A * pairs, a + A * pq, bp.power.col(pq), bp.edgeCurrent.col(pq), bp.vertexCurrent.col(pq));
    }
    Json vertexMeasures = Json::array();
    for (const char* name : kVertexMeasures) vertexMeasures.push_back(Json{{"measure", name}});
    const Json attrs = Json{{"members", "the frequency weighting's members"}, {"frequency_members", members}, {"boundary", pr.package.boundary},
                            {"frequency_weighting", pr.p.value("frequency_weighting", Json())}, {"frequency_normalisation", pr.p.value("frequency_normalisation", Json())},
                            {"sum_over", "the chunks and the member's bins (each sample counted twice with an overlap of half a chunk)"},
                            {"field_coordinates", K}, {"flow_modes", M}, {"sign", "phase current J = Im(conj(z_tail) z_tip): travel from tail to tip gives J < 0"},
                            {"split_input", splitEdges ? "edge_phase_current" : "vertex_phase_current"},
                            {"parity_input", splitEdges ? "vertex_phase_current (carried to the edges by flat)" : "edge_phase_current"}};
    auto describe = [&](ng::Array& a, const Json& measures, Eigen::Index rows, Eigen::Index cells, const char* rowsAre, const char* membersAre) {
      Json at = attrs; at["rows"] = rowsAre; if (membersAre) at["members"] = membersAre;
      a.meta["describe"] = Json{{"method", "spectral_flow"}, {"measure", "flow"}, {"shape", Json::array({rows, cells, measures.size()})},
                                {"axes", Json::array({indexed_axis("element", rows), indexed_axis("member", cells), indexed_axis("measure", static_cast<std::int64_t>(measures.size()), [&] { Json n = Json::array(); for (const auto& m : measures) n.push_back(m["measure"]); return n; }())})}, {"measures", measures}, {"values_shape", Json::array({rows, cells})}, {"attributes", at}};
    };
    const char* pairsAre = "frequency member x band p x band q, the frequency member fastest (the phase current of the pair: conj(z_p) at the tail, z_q at the tip)";
    describe(wv, vertexMeasures, S, A, "the vertices of the surface", nullptr);
    describe(we, Json::array({Json{{"measure", "phase_current"}}}), E, A, "the edges of the intrinsic complex (tail -> tip)", nullptr);
    Json modeMeasures = Json::array();
    for (const char* name : kModeMeasures) modeMeasures.push_back(Json{{"measure", name}});
    describe(wm, modeMeasures, M, A, "the Hodge-Dirac modes (exact: travel; coexact: rotation)", nullptr);
    describe(pv, vertexMeasures, S, A * pairs, "the vertices of the surface", pairsAre);
    describe(pe, Json::array({Json{{"measure", "phase_current"}}}), E, A * pairs, "the edges of the intrinsic complex (tail -> tip)", pairsAre);
    describe(pm, modeMeasures, M, A * pairs, "the Hodge-Dirac modes", pairsAre);
    return ng::Arrays{{"whole_vertices", std::move(wv)}, {"whole_edges", std::move(we)}, {"whole_modes", std::move(wm)},
                      {"pair_vertices", std::move(pv)}, {"pair_edges", std::move(pe)}, {"pair_modes", std::move(pm)}};
  };
  n.record = [](const std::any& prepared) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    return Json{{"boundary", pr.package.boundary}, {"frequency_members", pr.frequency.rows()}, {"bands", pr.scale.rows()}, {"split_input", pr.p.value("split_input", Json())}};
  };
  return n;
}

// ── signal.atom_time_cells ───────────────────────────────────────────────────────────────────────────────────────
const char* const kCells = "signal.atom_time_cells";
ng::NodeSpec atom_time_cells() {
  ng::NodeSpec n;
  n.name = kCells;
  n.kind = ng::Kind::array;
  n.reads_package_forms = true;
  n.inputs = {{"spectra", "the chunk spectra in the field coordinates (signal.chunk_spectra)", false},
              {"package", "the Hodge-Dirac package of the surface the coordinates are on", false},
              {"frequency", "the atoms: a kernel set of Gaussians on the spectra's frequency line (their time width sets the cells)", false},
              {"scale", "the eigenvalue bands (a kernel set on the eigenvalue line); each band's field is read on its own", false}};
  n.outputs = {{"cells", "per atom x band (the atom fastest; each band alone, then the whole field: every coordinate, no eigenvalue gain), per time cell of the atom's own width: the edge phase "
                         "currents, the vertex phase currents (x y z), the power per vertex (sums over the cell)", false}};
  n.parameters = {
      {"frequency_weighting", Json("kernels"), "kernels (the only weighting with atoms: the time cells are the atoms')", "-", {"kernels"}, false},
      {"frequency_normalisation", Json(), "sum_of_squared_gains | none: as signal.spectral_flow's (no default)", "-", {"sum_of_squared_gains", "none"}, false},
      {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true},
      {"cell_width", Json(), "a time cell's width in the atom's time standard deviations (1 / (2 pi sigma_f)); the cell is the power of two of samples nearest it (no default; the flow analysis: 2)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the atom's band signal back in time (the synthesis composed on the atom's bins, one inverse transform), summed over the cells");
  n.matches = "the flow analysis's time-cell check (dev: z = Psi a per sample, Im(conj z_tail z_tip) summed per cell); compute nxr::signal::timeCellFlow";
  n.description = "the flow of each atom and band in the atom's own time cells: per cell the sums of the edge and vertex phase currents and of the power (a measurement over the cells of a dyadic level of the time line)";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "package", "frequency", "scale"}) if (!in.count(k)) ng::refuse(std::string(kCells) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kCells);
    if (!p.contains("cell_width") || !p["cell_width"].is_number() || !(p["cell_width"].get<double>() > 0)) ng::refuse(std::string(kCells) + ": cell_width has no default: name it (> 0)");
    ng::Planned out;
    ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "statistic"; out.out["cells"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p, kCells, true)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(pr.spectra_copy, kCells);
    const nxr::signal::IntrinsicComplex& cx = *pr.package.complex;
    const auto [Psi, mu] = synthesis_of(s, *pr.package.modes, cx.star0, kCells);
    const Eigen::MatrixXd G = nxr::signal::sharp(cx, Eigen::MatrixXd(cx.d0 * Psi));
    const Eigen::Index S = Psi.rows(), E = cx.nE, A = pr.frequency.rows(), Q = pr.scale.rows();
    if (Q == 0) ng::refuse(std::string(kCells) + ": scale is a kernel set with members");
    if (static_cast<std::int64_t>(s.owns.size()) != s.chunks) ng::refuse(std::string(kCells) + ": the spectra do not record each chunk's own samples");
    const double width = pr.p["cell_width"].get<double>();
    ng::Array out;
    out.cols = E + 4 * S;
    Json groups = Json::array();
    std::int64_t row = 0;
    const Eigen::VectorXd wholeField = Eigen::VectorXd::Ones(Psi.cols());                      // the whole field: every coordinate at gain one (the bands' partition sums to it)
    for (Eigen::Index q = 0; q <= Q; ++q)                                                       // each band alone (nxr's flow analysis's cells), then the whole field
      for (Eigen::Index a = 0; a < A; ++a) {
        const double sigmaT = 1.0 / (2.0 * M_PI * pr.standardDeviations[static_cast<std::size_t>(a)]);
        const std::int64_t cell = std::int64_t(1) << static_cast<int>(std::lround(std::log2(width * sigmaT * s.sfreq)));
        std::int64_t cells = 0;
        for (std::int64_t t = 0; t < s.chunks; ++t) {
          Eigen::MatrixXcd X(s.rows, s.bins);
          for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t r = 0; r < s.rows; ++r) X(r, k) = pr.frequency(a, k) * s.at(t, k, r);
          const nxr::signal::CellFlow f = nxr::signal::timeCellFlow(X, s.first, s.transform, s.overlap, s.owns[static_cast<std::size_t>(t)], cell, Psi, G, cx.edges,
                                                                    q < Q ? Eigen::VectorXd(pr.scale.row(q).transpose()) : wholeField);
          for (Eigen::Index j = 0; j < f.edgeCurrent.cols(); ++j) {
            for (Eigen::Index e = 0; e < E; ++e) out.single.push_back(static_cast<float>(f.edgeCurrent(e, j)));
            for (Eigen::Index r = 0; r < 3 * S; ++r) out.single.push_back(static_cast<float>(f.vertexCurrent(r, j)));
            for (Eigen::Index v = 0; v < S; ++v) out.single.push_back(static_cast<float>(f.power(v, j)));
          }
          cells += f.edgeCurrent.cols();
        }
        groups.push_back(Json{{"atom", a}, {"band", q < Q ? Json(q) : Json("whole")}, {"field", q < Q ? "band" : "whole"}, {"first_row", row}, {"cells", cells}, {"cell_samples", cell},
                              {"cell_seconds", static_cast<double>(cell) / s.sfreq}});
        row += cells;
      }
    out.rows = row;
    out.meta["describe"] = Json{{"method", "atom_time_cells"}, {"measure", "time_cell_flow"}, {"shape", Json::array({row, E + 4 * S})}, {"axes", Json::array({indexed_axis("cell", row), indexed_axis("column", E + 4 * S)})}, {"measures", Json::array({Json{{"measure", "phase_current"}, {"columns", Json::array({0, E})}}, Json{{"measure", "vertex_phase_current"}, {"columns", Json::array({E, E + 3 * S})}},
                                                         Json{{"measure", "power"}, {"columns", Json::array({E + 3 * S, E + 4 * S})}}})},
                                {"values_shape", Json::array({row, E + 4 * S})}, {"precision", "float32"},
                                {"attributes", Json{{"groups", groups}, {"rows", "the time cells of each atom x band (each band alone, then the whole field), in the groups' order; within a group chunk by chunk, a trailing part cell of a chunk left out"},
                                                    {"columns", "the edges' phase currents, then each vertex's x y z phase current, then each vertex's power"},
                                                    {"cell_width", width}, {"sum_over", "the cell's samples (each sample once)"}}}};
    return ng::Arrays{{"cells", std::move(out)}};
  };
  return n;
}

// ── stats.sign_flip_consistency ──────────────────────────────────────────────────────────────────────────────────
const char* const kFlip = "stats.sign_flip_consistency";
constexpr std::int64_t kFlipSummaryColumns = 11;   // after the three per-element blocks: threshold, median, its null, its p, neighbour correlation, cells, groups, cells a flip, two correlations, reached
ng::NodeSpec sign_flip_consistency() {
  ng::NodeSpec n;
  n.name = kFlip;
  n.kind = ng::Kind::array;
  n.inputs = {{"cells", "the time cells of each atom x band (signal.atom_time_cells)", false}};
  n.outputs = {{"consistency", "per atom x band (and per atom on the whole field): R1 per element, its p and family-wise p, the threshold, the median and its null, the neighbouring cells' "
                               "correlation, and the grouping (cells a flip, the group sums' correlation measured on the atom's whole field and on the row, whether the rule was reached)", false}};
  n.parameters = {
      {"elements", Json(), "which values are tested: edges (the edge phase current, one component) | vertices (the vertex phase current, three components) (no default)", "-", {"edges", "vertices"}, false},
      {"patterns", Json(), "the random sign patterns over the cells (no default; nxr's flow analysis: 500 for its spectrum, 2000 for its maps)", "-", {}, false},
      {"seed", Json(), "the seed of numpy's default_rng, created once a run and drawn in the groups' order (no default; nxr's flow analysis: 0)", "-", {}, false},
      {"cells_per_flip", Json(), "consecutive cells one sign flips together: a whole number, or measured (THE GROUPING RULE: per atom, the fewest consecutive cells whose group sums' "
                                 "neighbour correlation, measured on the atom's whole-field cells, is below decorrelation_threshold; every band of the atom flips in groups of that many) (no default)", "cells", {}, false},
      {"decorrelation_threshold", Json(), "with cells_per_flip measured: the neighbour correlation of the group sums below which they count as practically uncorrelated (no default)", "-", {}, true},
      {"minimum_groups", Json(), "with cells_per_flip measured: the search stops before fewer whole groups would remain; not reached, the largest grouping tried is used and flagged (no default)", "groups", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a statistic of the cells: unit vectors, resultants, sign patterns");
  n.matches = "nxr's flow maps flip_test (nxr's flow analysis: R1, per-vertex p, family-wise maximum, numpy default_rng); compute nxr::stats::signFlipConsistency, NumpyGenerator";
  n.description = "the sign-flip (time-reversal) test of the direction's consistency over each atom's time cells, per element, with a family-wise maximum";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("cells")) ng::refuse(std::string(kFlip) + ": cells is not given");
    text_param(p, "elements", kFlip); integer_param(p, "patterns", kFlip); integer_param(p, "seed", kFlip);
    if (p.contains("cells_per_flip") && p["cells_per_flip"].is_string()) {
      if (p["cells_per_flip"].get<std::string>() != "measured") ng::refuse(std::string(kFlip) + ": cells_per_flip is a whole number or measured");
      if (!p.contains("decorrelation_threshold") || !p["decorrelation_threshold"].is_number()) ng::refuse(std::string(kFlip) + ": with cells_per_flip measured, decorrelation_threshold has no default: name it");
      if (integer_param(p, "minimum_groups", kFlip) < 2) ng::refuse(std::string(kFlip) + ": minimum_groups is at least 2");
    } else if (integer_param(p, "cells_per_flip", kFlip) < 1) ng::refuse(std::string(kFlip) + ": cells_per_flip is at least 1");
    ng::Planned out;
    out.out["consistency"] = constant_of("statistic");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(constant_input(c, "cells", kFlip), p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [cells, p] = std::any_cast<const std::pair<ng::Array, Json>&>(prepared);
    const Json d = cells.meta.value("describe", Json::object());
    const Json groups = d.value("attributes", Json::object()).value("groups", Json::array());
    const Json measures = d.value("measures", Json::array());
    if (groups.empty() || measures.size() < 2) ng::refuse(std::string(kFlip) + ": cells are not time cells (signal.atom_time_cells)");
    const bool edges = p["elements"].get<std::string>() == "edges";
    const std::vector<std::int64_t> columns = measures[edges ? 0 : 1].at("columns").get<std::vector<std::int64_t>>();
    const int components = edges ? 1 : 3;
    nxr::stats::NumpyGenerator generator(static_cast<std::uint64_t>(p["seed"].get<std::int64_t>()));
    ng::Array out;
    const std::int64_t width = columns[1] - columns[0], elements = width / components;
    out.cols = 3 * elements + kFlipSummaryColumns;
    auto values_of = [&](const Json& g) {
      const std::int64_t first = g["first_row"], count = g["cells"];
      if (count < 2) ng::refuse(std::string(kFlip) + ": an atom x band has fewer than two cells");
      Eigen::MatrixXd values(count, width);
      for (std::int64_t r = 0; r < count; ++r) for (std::int64_t j = 0; j < width; ++j) values(r, j) = cells.at(static_cast<std::size_t>((first + r) * cells.cols + columns[0] + j));
      return values;
    };
    // THE GROUPING RULE (cells_per_flip measured): per atom, on its whole-field cells, the fewest consecutive cells whose group sums are practically uncorrelated
    const bool measured = p["cells_per_flip"].is_string();
    std::map<std::int64_t, nxr::stats::DecorrelatedGrouping> rule;
    if (measured) {
      for (const auto& g : groups)
        if (g.value("field", std::string()) == "whole")
          rule[g["atom"].get<std::int64_t>()] = nxr::stats::fewestDecorrelatedCells(values_of(g), components, p["decorrelation_threshold"].get<double>(), p["minimum_groups"].get<std::int64_t>());
      for (const auto& g : groups)
        if (!rule.count(g["atom"].get<std::int64_t>())) ng::refuse(std::string(kFlip) + ": cells_per_flip measured reads each atom's whole-field cells, and atom " + g["atom"].dump() + " has none (signal.atom_time_cells)");
    }
    Json rows = Json::array();
    for (const auto& g : groups) {
      const Eigen::MatrixXd values = values_of(g);
      const std::int64_t atom = g["atom"].get<std::int64_t>();
      const std::int64_t perFlip = measured ? rule.at(atom).cellsPerFlip : p["cells_per_flip"].get<std::int64_t>();
      const nxr::stats::SignFlipConsistency t = nxr::stats::signFlipConsistency(values, components, p["patterns"].get<std::int64_t>(), generator, perFlip);
      const double atomCorrelation = measured ? rule.at(atom).correlation : std::numeric_limits<double>::quiet_NaN();
      const double rowCorrelation = nxr::stats::groupNeighbourCorrelation(values, components, perFlip);
      const double reached = measured ? (rule.at(atom).reached ? 1.0 : 0.0) : std::numeric_limits<double>::quiet_NaN();
      for (std::int64_t e = 0; e < elements; ++e) out.data.push_back(t.consistency(e));
      for (std::int64_t e = 0; e < elements; ++e) out.data.push_back(t.p(e));
      for (std::int64_t e = 0; e < elements; ++e) out.data.push_back(t.familywiseP(e));
      for (const double v : {t.threshold, t.median, t.medianNull95, t.medianP, t.neighbourCorrelation, static_cast<double>(t.cells), static_cast<double>(t.groups),
                             static_cast<double>(perFlip), atomCorrelation, rowCorrelation, reached})
        out.data.push_back(v);
      Json r = Json{{"atom", g["atom"]}, {"band", g["band"]}, {"cells_per_flip", perFlip}, {"flip_seconds", static_cast<double>(perFlip) * g.value("cell_seconds", 0.0)}};
      if (measured) r["correlations_tried"] = rule.at(atom).correlations;
      rows.push_back(std::move(r));
    }
    out.rows = static_cast<std::int64_t>(rows.size());
    Json options = Json{{"patterns", p["patterns"]}, {"seed", p["seed"]}, {"cells_per_flip", p["cells_per_flip"]}, {"generator", "numpy default_rng (SeedSequence, PCG64), random() < 0.5 -> -1"}};
    if (measured) {
      options["decorrelation_threshold"] = p["decorrelation_threshold"]; options["minimum_groups"] = p["minimum_groups"];
      options["grouping_rule"] = "per atom: the fewest consecutive cells whose group sums' neighbour correlation (the mean cosine of neighbouring groups' sums of unit fields), measured on the atom's whole-field cells, "
                                 "is below decorrelation_threshold, searched while at least minimum_groups whole groups remain; every band of the atom flips in groups of that many";
    }
    out.meta["describe"] = Json{{"statistic", "sign_flip_consistency"}, {"method", "sign_flip_consistency"}, {"measure", "consistency"}, {"axes", Json::array({indexed_axis("atom_band", out.rows, rows), indexed_axis("column", out.cols)})}, {"elements", p["elements"]}, {"shape", Json::array({out.rows, out.cols})},
                                {"columns", "R1 per element, p per element, family-wise p per element, then threshold, median R1, the median's null 95th percentile, its p, the neighbouring cells' correlation, cells, groups flipped, "
                                            "cells a flip, the group sums' neighbour correlation on the atom's whole field (NaN: fixed), the same on the row, the rule reached (1, 0; NaN: fixed)"},
                                {"rows", rows}, {"options", options}};
    return ng::Arrays{{"consistency", std::move(out)}};
  };
  return n;
}

// ── stats.control_comparison ─────────────────────────────────────────────────────────────────────────────────────
// THE CONTROL BESIDE THE RUN IT CONTROLS: two sign-flip statistics of the same atoms and bands (a run's, and a control recording's, e.g. the empty room on the same sensor surface),
// LENGTH-MATCHED: refused unless every row holds the same number of time cells in both (the same duration at the same cell width). Per row, for each: the median R1, its p, the share of
// elements with p below `significance`, the share with family-wise p below it, the cells a flip. No new statistic: the same numbers side by side.
const char* const kCompare = "stats.control_comparison";
ng::NodeSpec control_comparison() {
  ng::NodeSpec n;
  n.name = kCompare;
  n.kind = ng::Kind::array;
  n.inputs = {{"consistency", "the control's sign-flip statistic (stats.sign_flip_consistency)", false},
              {"compared", "the sign-flip statistic of the run the control is compared with (stored, by name): the same atoms, bands and elements", false}};
  n.outputs = {{"comparison", "per atom x band: the run's then the control's median R1, its p, the share of elements below the significance, the same family-wise, the cells a flip; cells", false}};
  n.parameters = {{"significance", Json(), "the p below which an element counts as consistent in the shares (no default)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a table of two statistics' summaries, row by row");
  n.matches = "nxr's flow analysis's rotation control (the same test on the empty-room recording for the same duration as the rest span compared)";
  n.description = "a control recording's sign-flip test beside the run it controls, length-matched (the same cells per atom and band): the median R1, its p and the shares of consistent elements";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"consistency", "compared"}) if (!in.count(k)) ng::refuse(std::string(kCompare) + ": " + k + " is not given");
    if (!p.contains("significance") || !p["significance"].is_number() || !(p["significance"].get<double>() > 0 && p["significance"].get<double>() < 1)) ng::refuse(std::string(kCompare) + ": significance has no default: name it (between 0 and 1)");
    ng::Planned out;
    out.out["comparison"] = constant_of("statistic");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    return std::any(std::make_tuple(constant_input(c, "consistency", kCompare), constant_input(c, "compared", kCompare), p));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [control, compared, p] = std::any_cast<const std::tuple<ng::Array, ng::Array, Json>&>(prepared);
    if (control.rows != compared.rows || control.cols != compared.cols)
      ng::refuse(std::string(kCompare) + ": the two statistics are not of the same atoms, bands and elements (" + std::to_string(compared.rows) + " x " + std::to_string(compared.cols) + " and " +
                 std::to_string(control.rows) + " x " + std::to_string(control.cols) + ")");
    if ((control.cols - kFlipSummaryColumns) % 3 != 0 || control.cols <= kFlipSummaryColumns) ng::refuse(std::string(kCompare) + ": the statistics are not sign-flip statistics (stats.sign_flip_consistency)");
    const std::int64_t elements = (control.cols - kFlipSummaryColumns) / 3, summary = 3 * elements;
    const double alpha = p["significance"].get<double>();
    ng::Array out;
    out.rows = control.rows; out.cols = 11;
    for (std::int64_t r = 0; r < control.rows; ++r) {
      auto at = [&](const ng::Array& a, std::int64_t c) { return a.at(static_cast<std::size_t>(r * a.cols + c)); };
      if (at(control, summary + 5) != at(compared, summary + 5))
        ng::refuse(std::string(kCompare) + ": row " + std::to_string(r) + " holds " + std::to_string(static_cast<std::int64_t>(at(compared, summary + 5))) + " cells in the run and " +
                   std::to_string(static_cast<std::int64_t>(at(control, summary + 5))) + " in the control: not length-matched (name the run's span to the control's duration)");
      for (const ng::Array* a : {&compared, &control}) {
        double below = 0, belowFamily = 0;
        for (std::int64_t e = 0; e < elements; ++e) { below += at(*a, elements + e) < alpha; belowFamily += at(*a, 2 * elements + e) < alpha; }
        for (const double v : {at(*a, summary + 1), at(*a, summary + 3), below / static_cast<double>(elements), belowFamily / static_cast<double>(elements), at(*a, summary + 7)}) out.data.push_back(v);
      }
      out.data.push_back(at(control, summary + 5));
    }
    const Json columns = Json::array({"run_median_r1", "run_median_p", "run_share_below", "run_familywise_share_below", "run_cells_per_flip",
                                      "control_median_r1", "control_median_p", "control_share_below", "control_familywise_share_below", "control_cells_per_flip", "cells"});
    Json rows = control.meta.value("describe", Json::object()).value("rows", Json::array());
    out.meta["describe"] = Json{{"statistic", "control_comparison"}, {"method", "control_comparison"}, {"measure", "consistency"}, {"shape", Json::array({out.rows, out.cols})},
                                {"axes", Json::array({indexed_axis("atom_band", out.rows, rows.size() == static_cast<std::size_t>(out.rows) ? rows : Json()), indexed_axis("column", out.cols, columns)})},
                                {"columns", columns}, {"significance", alpha}, {"compared", compared.meta.value("name", std::string())},
                                {"length_matched", "every row the same number of time cells in the run and the control"}};
    return ng::Arrays{{"comparison", std::move(out)}};
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
  for (ng::NodeSpec n : {overlapped_spectrum(), chunk_spectra(), spectral_flow(), atom_time_cells(), sign_flip_consistency(), control_comparison(), sensor_power(), aperiodic_fit()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
