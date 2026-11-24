// nxr-pipelines — the mode-space Fourier coefficients as graph nodes (the definition `spectral_coefficients`).
//
// THE GRAPH. before: package (+ a free kernel, or the sensor surface's vertex-to-channel map) --[signal.mode_analysis]--> the composed spatial operator, rows x channels, bound once.
// chunks: recording (streamed, the caller's span) --[signal.spectral_coefficients: per time tile of `tile_samples`, the one-sided transform of the tile and the operator applied at the bins of
// the caller's frequency span; one slice a tile, nothing summed]--> A [tiles x bins x rows], complex, handed to io as a measurement of the recording over its tile grid (one value per member).
// The transform and the operator are linear and commute (the operator acts on the channels, the transform on time), so vertex x time is never formed.
#include "nxr/pipelines/spectral_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/mode_spectrum.h"
#include "nxr/signal/source_flow.h"
#include "source_flow_readers.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace {

ng::PortType constant_of(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}
std::string text_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_string()) ng::refuse(std::string(who) + ": " + key + " has no default: name it");
  return p[key].get<std::string>();
}

// ── signal.mode_analysis ────────────────────────────────────────────────────────────────────────────────────────
const char* const kAnalysis = "signal.mode_analysis";
struct AnalysisPrepared { ng::Array package, kernel, channel_map; bool has_kernel = false, has_map = false; Json p; };

nxr::operators::Basis basis_from(const ng::Array& H) {
  nxr::operators::Basis b;
  const std::int64_t K = H.rows, V = H.cols;
  const std::vector<double> P = H.values();
  b.P = Eigen::Map<const RowMajor>(P.data(), K, V);
  const std::vector<double> mu = H.meta.at("mu").get<std::vector<double>>();
  b.mu = Eigen::Map<const Eigen::VectorXd>(mu.data(), K);
  b.kind = H.meta.at("kind").get<std::vector<int>>();
  const std::vector<double> mass = H.meta.at("mass").get<std::vector<double>>();
  b.mass = Eigen::Map<const Eigen::VectorXd>(mass.data(), static_cast<Eigen::Index>(mass.size()));
  const std::vector<double> pos = H.meta.at("positions").get<std::vector<double>>();
  b.positions = Eigen::Map<const nxr::operators::RowMat3>(pos.data(), V, 3);
  const std::vector<std::int32_t> faces = H.meta.at("faces").get<std::vector<std::int32_t>>();
  b.faces = Eigen::Map<const nxr::operators::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
  b.keep = H.meta.at("keep").get<std::vector<std::int64_t>>();
  return b;
}

ng::NodeSpec mode_analysis() {
  ng::NodeSpec n;
  n.name = kAnalysis;
  n.kind = ng::Kind::array;
  n.inputs = {{"package", "a Hodge-Dirac package (its eigenvectors, as the runner hands them over): the modes the coefficients are on", false},
              {"kernel", "a free-orientation inverse kernel (channels -> vertices, x y z) on the package's input surface or on the cortex a hemisphere's package is a part of (its rows selected): the cortex. Exactly one of kernel, channel_map", true},
              {"channel_map", "the sensor surface's vertex-to-channel map (`<surface>_map`): the helmet, whose vertices are channels. Exactly one of kernel, channel_map", true}};
  n.outputs = {{"operator", "the composed spatial operator, rows x the channels it reads: the modes' analysis rows (the selected modes' 0-form parts, normalised as stated, times the vertex mass), "
                            "composed with the kernel carried onto the package's surface (one block of rows per Cartesian component) or with the vertex-to-channel map", false}};
  n.parameters = {
      {"modes", Json(), "which modes: exact | coexact | all | exact_and_constant (the constant, index -1 and mu 0, then the exact modes: a field's coordinates on the surface) (no default: the caller names it; the nxr flow scripts expand on the exact modes)", "-", {"exact", "coexact", "all", "exact_and_constant"}, false},
      {"normalisation", Json(), "unit_mass (each mode rescaled to unit vertex-mass norm: rows x synthesis = I, the nxr flow scripts) | as_stored (the package's parts as stored) (no default)", "-", {"unit_mass", "as_stored"}, false},
      {"mode_count", Json(), "the first modes of the package read; null: all", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "made in double, once a run, from the package and the kernel");
  n.matches = "the nxr flow scripts' expansion of each Cartesian current component on the exact modes; compute nxr::signal::modeAnalysis, modeAnalysisOfComponents, selectVertexRows (the kernel's rows selected onto the package's surface: a hemisphere's range, or the glued surface with the bridge zero)";
  n.description = "the composed spatial operator of the mode-space coefficients: the selected modes' analysis rows composed with a free kernel (each component) or with the sensor surface's channel map";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("package")) ng::refuse(std::string(kAnalysis) + ": package is not given");
    if (in.count("kernel") == in.count("channel_map")) ng::refuse(std::string(kAnalysis) + ": exactly one of kernel (the cortex) and channel_map (the sensor surface)");
    text_param(p, "modes", kAnalysis);
    text_param(p, "normalisation", kAnalysis);
    ng::Planned out;
    ng::PortType t;
    t.cls = ng::PortClass::operator_;
    t.facts["cls"] = "operator";
    // its TARGET is the package's mode axis (the eigenvectors' source manifold): the rows it gives are modes of that stored axis (a subset: the coefficients name the members)
    if (const auto pk = in.find("package"); pk != in.end() && !pk->second.domain.factors.empty()) { t.typed = true; t.target = pk->second.domain; }
    out.out["operator"] = t;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    AnalysisPrepared a;
    if (!c.count("package") || c.at("package").meta.value("placeholder", false)) ng::refuse(std::string(kAnalysis) + ": package was not handed over");
    a.package = c.at("package");
    if (!a.package.meta.value("hodge", false)) ng::refuse(std::string(kAnalysis) + ": package is not a Hodge-Dirac package (an operator of kind eigenvectors, <stem>_eigenvectors)");
    if (c.count("kernel") && !c.at("kernel").meta.value("placeholder", false)) { a.kernel = c.at("kernel"); a.has_kernel = true; }
    if (c.count("channel_map") && !c.at("channel_map").meta.value("placeholder", false)) { a.channel_map = c.at("channel_map"); a.has_map = true; }
    if (a.has_kernel == a.has_map) ng::refuse(std::string(kAnalysis) + ": exactly one of kernel and channel_map was handed over");
    a.p = p;
    return std::any(std::move(a));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const AnalysisPrepared& a = std::any_cast<const AnalysisPrepared&>(prepared);
    const std::string modes = a.p["modes"].get<std::string>(), norm = a.p["normalisation"].get<std::string>();
    nxr::operators::Basis basis = basis_from(a.package);
    std::int64_t K = basis.P.rows();
    if (a.p.contains("mode_count") && !a.p["mode_count"].is_null()) K = std::min<std::int64_t>(K, a.p["mode_count"].get<std::int64_t>());
    const std::vector<int> kind(basis.kind.begin(), basis.kind.begin() + K);
    const nxr::signal::ModeAnalysis ma = nxr::signal::modeAnalysis(basis.P.topRows(K).transpose(), basis.mass, kind,
                                                                   modes == "exact" ? nxr::signal::ModeKinds::exact : modes == "coexact" ? nxr::signal::ModeKinds::coexact
                                                                   : modes == "exact_and_constant" ? nxr::signal::ModeKinds::exact_and_constant : nxr::signal::ModeKinds::all,
                                                                   norm == "unit_mass" ? nxr::signal::ModeNormalisation::unit_mass : nxr::signal::ModeNormalisation::as_stored);
    Eigen::MatrixXd O;
    std::vector<std::int32_t> channels;
    int components = 1;
    if (a.has_kernel) {
      const ng::Array& KA = a.kernel;
      channels = KA.meta.value("channels", std::vector<std::int32_t>());
      // the kernel's rows SELECTED onto the package's surface (the glue is a selection): the input surface's own rows, a hemisphere's vertex range of the cortex, or the glued
      // surface's kept vertices with the bridge rows zero; never a blend
      const KernelSelection selection = kernel_rows_for_package(KA, a.package, kAnalysis);
      const std::vector<double> kv = KA.values();
      const Eigen::MatrixXd W = Eigen::Map<const RowMajor>(kv.data(), KA.rows, KA.cols);
      O = nxr::signal::modeAnalysisOfComponents(ma.rows, nxr::signal::selectVertexRows(W, selection.source_vertex, 3), 3);
      components = 3;
    } else {
      const std::vector<double> mv = a.channel_map.values();
      const std::int64_t V = basis.P.cols();
      if (static_cast<std::int64_t>(mv.size()) != V) ng::refuse(std::string(kAnalysis) + ": the channel map has " + std::to_string(mv.size()) + " vertices, the package's surface " + std::to_string(V));
      for (std::int64_t v = 0; v < V; ++v) {
        if (basis.keep[static_cast<std::size_t>(v)] != v) ng::refuse(std::string(kAnalysis) + ": the package's surface is not the channel map's (a glued or re-ordered surface)");
        if (!(mv[static_cast<std::size_t>(v)] >= 0)) ng::refuse(std::string(kAnalysis) + ": vertex " + std::to_string(v) + " of the channel map has no channel");
        channels.push_back(static_cast<std::int32_t>(mv[static_cast<std::size_t>(v)]));
      }
      O = ma.rows;
    }
    ng::Array out;
    out.rows = O.rows(); out.cols = O.cols();
    out.data.resize(static_cast<std::size_t>(out.rows * out.cols));
    Eigen::Map<RowMajor>(out.data.data(), out.rows, out.cols) = O;
    out.meta["channels"] = channels;
    out.meta["components"] = components;
    out.meta["modes"] = ma.modes;
    std::vector<double> mu;
    for (const auto k : ma.modes) mu.push_back(k < 0 ? 0.0 : basis.mu(k));
    out.meta["mu"] = mu;
    out.meta["name"] = a.package.meta.value("name", std::string()) + (a.has_kernel ? "+" + a.kernel.meta.value("name", std::string()) : std::string("+channels"));
    out.meta["axis_id"] = a.package.meta.value("axis_id", std::string());
    out.meta["to_value_type"] = "scalar";
    out.meta["unit"] = "mode coefficient";
    out.meta["rows"] = Json{{"order", components == 3 ? "component-major: rows c*K .. (c+1)*K - 1 are Cartesian component c (x, y, z) of mode modes[k]" : "mode modes[k]"},
                            {"selection", modes}, {"normalisation", norm}};
    return ng::Arrays{{"operator", std::move(out)}};
  };
  n.record = [](const std::any& prepared) {
    const AnalysisPrepared& a = std::any_cast<const AnalysisPrepared&>(prepared);
    return Json{{"modes", a.p.value("modes", Json())}, {"normalisation", a.p.value("normalisation", Json())}, {"source", a.has_kernel ? "kernel" : "channel_map"}};
  };
  return n;
}

// ── signal.spectral_coefficients ────────────────────────────────────────────────────────────────────────────────
const char* const kCoefficients = "signal.spectral_coefficients";
struct CoefPrepared {
  Eigen::MatrixXd op;
  std::vector<std::int32_t> channels;
  std::int64_t tile = 0, first_bin = 0, bins = 0;
  double sfreq = 0, lo = 0, hi = 0;
  bool exclude = true;
  Json op_meta;
};
struct CoefState {
  std::int64_t origin = -1;                       // the first sample of the tile being filled
  Eigen::MatrixXd buffer;                         // [channels x tile]
  std::int64_t filled = 0;
  bool tile_bad = false;
  std::vector<std::int64_t> tiles, left_out;      // the tiles kept and those touching a bad sample (members of the tile grid: sample / tile)
  std::vector<float> values;                      // re, im of each value, tile-major then bin then row
};

ng::NodeSpec spectral_coefficients() {
  ng::NodeSpec n;
  n.name = kCoefficients;
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed, the caller's span)", false},
              {"operator", "the composed spatial operator (signal.mode_analysis, or any operator from the recording's channels)", false},
              {"exclude", "the recording's bad segments (spans): with bad_segments exclude, a tile touching one is left out", true}};
  n.outputs = {{"coefficients", "per time tile, at the bins of the frequency span, the operator's rows: [tiles x bins x rows] complex, one value per member of the tile grid (a measurement)", false}};
  n.parameters = {
      {"tile_samples", Json(), "the time tile, in samples: a power of two, the tile grid the span snaps to (the flow analysis: 2^15, the chunk grid) (no default)", "samples", {}, false},
      {"frequency_span", Json(), "[lo, hi): the bins k with lo <= k fs / n < hi of each tile's frequency line (no default; the flow analysis: [0, 60])", "Hz", {}, false},
      {"bad_segments", Json(), "exclude = a tile touching one of the recording's bad samples is left out (listed) | ignore = every tile is kept (no default)", "-", {"exclude", "ignore"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "the tile is filled across blocks in time order (the only state), transformed in double when whole; each tile's values are kept (nothing summed), in float32 for float32 input");
  n.matches = "numpy.fft.rfft per tile (norm backward, no window) then the operator's rows (the nxr flow scripts' per-chunk FFT and mode expansion); compute nxr::signal::spectralCoefficients";
  n.description = "the mode-space Fourier coefficients per time tile: each tile's one-sided transform at the bins of a frequency span, read through a spatial operator (one value per tile, per bin, per row)";
  n.plan = [](const ng::Types& in, const Json& p) {
    const auto it = in.find("x");
    if (it == in.end() || !it->second.streamed() || it->second.axis != ng::Axis::samples) ng::refuse(std::string(kCoefficients) + ": x is a recording's samples (a streamed field)");
    if (it->second.value != ng::ValueType::real) ng::refuse(std::string(kCoefficients) + ": x is real");
    if (!it->second.sfreq || !(*it->second.sfreq > 0)) ng::refuse(std::string(kCoefficients) + ": x has no sampling rate");
    if (!in.count("operator")) ng::refuse(std::string(kCoefficients) + ": operator is not given");
    if (!p.contains("tile_samples") || !p["tile_samples"].is_number_integer()) ng::refuse(std::string(kCoefficients) + ": tile_samples has no default: name it");
    const std::int64_t t = p["tile_samples"].get<std::int64_t>();
    if (t < 2 || (t & (t - 1)) != 0) ng::refuse(std::string(kCoefficients) + ": tile_samples is a power of two (the tile grid)");
    if (!p.contains("frequency_span") || !p["frequency_span"].is_array() || p["frequency_span"].size() != 2) ng::refuse(std::string(kCoefficients) + ": frequency_span [lo, hi) has no default: name it");
    text_param(p, "bad_segments", kCoefficients);
    // A STATISTIC of the run whose axes the definition declares and the runner links (the four classes): the tiles (positions on the recording's time line), the bins kept of the
    // frequency line a tile's transform makes (keyed by (time line, n, fs)), the modes kept of the package's mode axis (one block a Cartesian component)
    ng::Planned out;
    out.out["coefficients"] = constant_of("statistic");
    out.align = t;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& c, const Json& p, ng::Shared&) {
    CoefPrepared pr;
    const auto it = c.find("operator");
    if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(kCoefficients) + ": the operator was not handed over");
    const ng::Array& O = it->second;
    const std::vector<double> ov = O.values();
    pr.op = Eigen::Map<const RowMajor>(ov.data(), O.rows, O.cols);
    pr.channels = O.meta.value("channels", std::vector<std::int32_t>());
    if (static_cast<std::int64_t>(pr.channels.size()) != O.cols) ng::refuse(std::string(kCoefficients) + ": the operator's channels do not fit its columns");
    pr.op_meta = O.meta;
    pr.op_meta.erase("channels");
    pr.sfreq = *in.at("x").sfreq;
    pr.tile = p["tile_samples"].get<std::int64_t>();
    pr.lo = p["frequency_span"][0].get<double>(); pr.hi = p["frequency_span"][1].get<double>();
    if (!(pr.lo >= 0) || !(pr.hi > pr.lo)) ng::refuse(std::string(kCoefficients) + ": frequency_span is [lo, hi) with 0 <= lo < hi");
    const double df = pr.sfreq / static_cast<double>(pr.tile);
    const std::int64_t nb = pr.tile / 2 + 1;
    pr.first_bin = static_cast<std::int64_t>(std::ceil(pr.lo / df - 1e-9));
    const std::int64_t end = std::min<std::int64_t>(nb, static_cast<std::int64_t>(std::ceil(pr.hi / df - 1e-9)));
    pr.bins = end - pr.first_bin;
    if (pr.bins < 1) ng::refuse(std::string(kCoefficients) + ": the frequency span holds no bin of a tile's line (spacing " + std::to_string(df) + " Hz)");
    pr.exclude = p["bad_segments"].get<std::string>() == "exclude";
    return std::any(std::move(pr));
  };
  n.start = [](const std::any&) { return std::any(CoefState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const CoefPrepared& pr = std::any_cast<const CoefPrepared&>(prepared);
    CoefState& s = std::any_cast<CoefState&>(state);
    const ng::Block& b = in.at("x");
    const std::int64_t own = b.own();
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    for (const std::int32_t ch : pr.channels) if (ch < 0 || ch >= b.rows) ng::refuse(std::string(kCoefficients) + ": a channel of the operator is outside the recording's channels");
    for (std::int64_t t = 0; t < own; ++t) {
      const std::int64_t sample = at.first + t;
      if (s.filled == 0) {
        if (sample % pr.tile != 0) ng::refuse(std::string(kCoefficients) + ": the span does not start on the tile grid (sample " + std::to_string(sample) + ")");
        s.origin = sample;
        s.buffer.resize(static_cast<Eigen::Index>(pr.channels.size()), pr.tile);
        s.tile_bad = false;
      }
      for (std::size_t i = 0; i < pr.channels.size(); ++i) s.buffer(static_cast<Eigen::Index>(i), s.filled) = all(pr.channels[i], b.left + t);
      if (at.good && static_cast<std::int64_t>(at.good->size()) == own && !(*at.good)[static_cast<std::size_t>(t)]) s.tile_bad = true;
      if (++s.filled == pr.tile) {
        const std::int64_t member = s.origin / pr.tile;
        if (pr.exclude && s.tile_bad) {
          s.left_out.push_back(member);
        } else {
          const Eigen::MatrixXcd A = nxr::signal::spectralCoefficients(pr.op, s.buffer, pr.first_bin, pr.bins);   // [rows x bins]
          for (Eigen::Index k = 0; k < A.cols(); ++k)
            for (Eigen::Index r = 0; r < A.rows(); ++r) { s.values.push_back(static_cast<float>(A(r, k).real())); s.values.push_back(static_cast<float>(A(r, k).imag())); }
          s.tiles.push_back(member);
        }
        s.filled = 0;
      }
    }
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const CoefPrepared& pr = std::any_cast<const CoefPrepared&>(prepared);
    CoefState& s = std::any_cast<CoefState&>(state);
    if (s.filled != 0)
      ng::refuse(std::string(kCoefficients) + ": the span ends inside a tile (" + std::to_string(s.filled) + " of " + std::to_string(pr.tile) +
                 " samples): a part tile lies on another frequency line; state a span of whole tiles");
    if (s.tiles.empty()) ng::refuse(std::string(kCoefficients) + ": no tile was kept (" + std::to_string(s.left_out.size()) + " left out by the bad segments)");
    const std::int64_t T = static_cast<std::int64_t>(s.tiles.size()), R = pr.op.rows();
    ng::Array a;
    a.rows = T * pr.bins * R; a.cols = 2;
    a.single = std::move(s.values);                                          // float32 values (precision follows the input), moved: no double copy
    const double df = pr.sfreq / static_cast<double>(pr.tile);
    Json tiles = Json::array();
    for (const auto m : s.tiles) tiles.push_back(Json{{"member", m}, {"span", Json::array({static_cast<double>(m * pr.tile) / pr.sfreq, static_cast<double>((m + 1) * pr.tile) / pr.sfreq})}});
    Json bins = Json::array();
    for (std::int64_t k = 0; k < pr.bins; ++k) bins.push_back(pr.first_bin + k);
    Json mode = Json{{"name", "mode"}, {"size", R}, {"members", pr.op_meta.value("modes", Json::array())}};
    const std::int64_t components = pr.op_meta.value("components", std::int64_t(1));
    if (components > 1) { Json parts = Json::array({"x", "y", "z"}); parts.erase(parts.begin() + components, parts.end()); mode["parts"] = parts; }
    a.rows = T; a.cols = pr.bins * R * 2;
    a.meta["describe"] = Json{{"measure", "fourier_coefficient"}, {"method", "spectral_coefficients"}, {"label", "spectral_coefficients"}, {"precision", "float32"}, {"value_type", "complex"},
                              {"shape", Json::array({T, pr.bins, R, 2})}, {"unit", "(operator output) x samples"},
                              {"axes", Json::array({Json{{"name", "tile"}, {"size", T}, {"values", tiles}},
                                                    Json{{"name", "frequency"}, {"size", pr.bins}, {"members", bins},
                                                         {"line", Json{{"spacing", df}, {"origin", 0.0}, {"unit", "Hz"}, {"dimension", "frequency"}, {"fft", Json{{"n", pr.tile}, {"fs", pr.sfreq}}}}}},
                                                    mode})},
                              {"options", Json{{"tiles_left_out", s.left_out}, {"tile_samples", pr.tile}, {"frequency_span", Json::array({pr.lo, pr.hi})},
                                               {"transform", "one-sided, unscaled, no window (numpy rfft norm backward)"}, {"operator", pr.op_meta}}},
                              {"producer", Json{{"sfreq", pr.sfreq}, {"tile_samples", pr.tile}, {"first_bin", pr.first_bin}, {"bins", pr.bins}, {"frequency_resolution_hz", df},
                                                {"modes", pr.op_meta.value("modes", Json::array())}, {"mu", pr.op_meta.value("mu", Json::array())}, {"components", components},
                                                {"normalisation", pr.op_meta.value("normalisation", Json())}, {"tiles", s.tiles}}},
                              {"description", "the mode-space Fourier coefficients per time tile: [tiles x bins x modes] (complex)"}};
    return ng::Arrays{{"coefficients", std::move(a)}};
  };
  return n;
}

}  // namespace

void register_spectral_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {mode_analysis(), spectral_coefficients()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
