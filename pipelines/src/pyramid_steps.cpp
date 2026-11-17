// nxr-pipelines — THE MEASUREMENT PYRAMID AS NODES (the hand-wired `pyramid build` and `pyramid spatial-bands` they replace; spec docs/specs/2026-11-17-pyramid-grid.md).
//
// The pyramid is a GRID on the product manifold cortex × time: the cells of the joint grid (Studio D129, `joint.cells`) are a dyadic spatial tile (a leaf: `<surface> tiles L<D>`) × a time tile
// (a frame: a cycle of ONE tower level, `frame_tiling`). Every number it stores is a MEASUREMENT over the members of that grid; the roll-ups (coarser leaves, coarser frames) are the grids'
// closed-form hierarchy (code >> shift) at read, never stored. The nodes, one band at a time (the definition repeats them over the bands, `for: {band: $bands}`):
//
//   pyramid.band_analytic   chunk (windowed)  x [channels x samples] (+ the kernel's channels)  ->  z: the band's analytic signal at the channels, zero phase (the block's spectrum × the band's gain, the
//                                              positive frequencies doubled); the halo is the filter's circular edge. A function of matrices (nxr::pyramid::spectrumOf, bandGains, analyticBand).
//   pyramid.band_sums       accumulate         z + the kernel + the leaves + the frame grid  ->  the band's measurements over the joint cells: power_sum (Σ a|J|²), envelope_sum (Σ a|J|), envelope_max
//                                              (max |J|), J = K z (the inverse is linear, so the filter comes first); a FREE kernel (3 rows a source) gives the Hermitian orientation tensor's ten sums instead
//                                              (nxr::pyramid::bandSums, tensorBandSums over the RESOLVED forms: the leaves a MemberOfElement, the frames' member of each sample). Mergeable: sums add, maxima merge.
//   pyramid.frame_samples   accumulate         the recording + the frame grid  ->  the good samples of each frame (`count`, the roll-up weight on the frame tiling)
//   pyramid.estimate        array              the source estimate the sums are of: a Field with no bytes (its reductions are the pyramid), with its provenance
//   pyramid.bands           array              the band bank: spans on the frequency line, code = the band's index (nxr's `atlas bands`, D158)
//   pyramid.spatial_bands   array              the spatial band bank: one span per level of a surface's tile ladder on the eigenvalue line (the level IS its Weyl band)
//
// The Studio's tower (studio/src/engine/toolbox/tower/tower.ts) and Rheome's jointladder (+geom/jointladder.m) keep their names: a frame's tower level, the tower code `frame_code0 + k`, a leaf's
// depth, `tile`, `frame`. A node knows no file, chunk or subject: it reads what the runner hands it (constants) and describes what it made (`meta.describe`); the definition places, io writes.
//
// MEMORY (stated): `pyramid.band_analytic` holds the block (one chunk and its two halos) of the kernel's channels in double and its spectrum and the band's analytic signal (complex, the same size);
// `pyramid.band_sums` the kernel (rows × channels float64) and the projection's sub-block (rows × `block_samples` × 4 float64, 3 more for a free kernel), and its sums are 3 float64 (10 for a free kernel) a leaf of
// every frame the run's span touches (the store writes the recording's whole frame axis: frames outside the span hold 0, −∞ for the maxima, and a count of 0). Never the whole recording.
//
// WHAT IT CANNOT RESOLVE (recorded in the estimate's producer.warnings): the minimum-norm current is smooth (its spatial resolution is the kernel's point-spread, not the leaves'); a band's analytic
// amplitude means something only for a band narrow against its centre; the filter is circular over the block, so the first and last `halo` seconds of a recording carry its edge effect; a Morlet band
// overlaps its neighbours, so the band powers do not add to broadband (`power_partition` does); the principal axis of a free kernel's tensor is partly the kernel's own prior.
#include "nxr/pipelines/partition_steps.h"
#include "nxr/pipelines/pyramid_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/frame_grid.h"
#include "nxr/graph/node.h"
#include "nxr/ladder.h"
#include "nxr/parallel.h"
#include "nxr/partition.h"
#include "nxr/pyramid.h"
#include "nxr/selection/resolved.h"
#include "nxr/selection/selection.h"
#include "nxr/tower.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
namespace py = nxr::pyramid;
using nxr::graph::Json;

namespace {

using RowMajorF = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorD = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::string g(double v) { std::ostringstream s; s << v; return s.str(); }

ng::PortType constant_port(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}
const ng::PortType& streamed(const ng::Types& in, const char* port, const char* who) {
  const auto it = in.find(port);
  if (it == in.end()) ng::refuse(std::string(who) + ": port '" + port + "' is not given");
  const ng::PortType& t = it->second;
  if (!t.streamed() || t.axis != ng::Axis::samples) ng::refuse(std::string(who) + ": '" + port + "' is a recording's samples (a streamed field)");
  if (!t.sfreq || !(*t.sfreq > 0)) ng::refuse(std::string(who) + ": '" + port + "' has no sampling rate");
  return t;
}
const ng::Array& constant(const ng::Arrays& c, const char* port, const char* who) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + port + " was not handed over");
  return it->second;
}

// a band as [lo, hi] (a pair, or {lo, hi}): Hz
std::pair<double, double> band_of(const Json& b, const char* who) {
  if (b.is_array() && b.size() == 2 && b[0].is_number() && b[1].is_number()) return {b[0].get<double>(), b[1].get<double>()};
  if (b.is_object() && b.contains("lo") && b.contains("hi")) return {b["lo"].get<double>(), b["hi"].get<double>()};
  ng::refuse(std::string(who) + ": a band is [lo, hi] in Hz");
}
py::Bands bank_of(const Json& bands, const char* who) {
  py::Bands out;
  if (bands.is_null() || (bands.is_array() && bands.empty())) return py::octaveBands();         // nxr DEFAULT_BANDS: the octaves 1-2 ... 32-64 Hz
  if (!bands.is_array()) ng::refuse(std::string(who) + ": bands is a list of [lo, hi] in Hz");
  for (const auto& b : bands) out.push_back(band_of(b, who));
  return out;
}
py::Shape shape_of(const Json& p, const char* who) {
  try { return py::shapeOf(p.value("shape", std::string("morlet"))); } catch (const std::invalid_argument& e) { ng::refuse(std::string(who) + ": " + e.what()); }
}
std::size_t index_in(const py::Bands& bank, const std::pair<double, double>& b, const char* who) {
  for (std::size_t k = 0; k < bank.size(); ++k)
    if (std::abs(bank[k].first - b.first) < 1e-9 && std::abs(bank[k].second - b.second) < 1e-9) return k;
  ng::refuse(std::string(who) + ": the band [" + g(b.first) + ", " + g(b.second) + ") is not in the bank (`bands`)");
}
Json band_record(const std::pair<double, double>& b, std::size_t code) {
  return Json{{"name", g(b.first) + "\xe2\x80\x93" + g(b.second) + " Hz"}, {"lo_hz", b.first}, {"hi_hz", b.second}, {"code", static_cast<std::int64_t>(code)}};
}

// the frames of a stored tiling record, rebuilt as the tower's FrameTiling (tower.h) and resolved on the recording's line: the grid every sum is a measurement over
nxr::tower::FrameTiling tower_of(const ng::Array& frames, const char* who) {
  const Json p = frames.meta.value("params", Json::object());
  if (!p.is_object() || !p.contains("atlas") || !p.contains("frame_tiling") || !p["atlas"].contains("frame_level"))
    ng::refuse(std::string(who) + ": " + frames.meta.value("name", std::string("frames")) + " is not a frame tiling of the tower (`run frametiling`)");
  const Json& a = p["atlas"];
  try {
    return nxr::tower::makeFrameTiling(p["frame_tiling"]["sfreq"].get<double>(), p["frame_tiling"]["samples"].get<std::int64_t>(), a["frame_level"].get<int>(),
                                       a.contains("start_s") && a["start_s"].is_number() ? std::optional<double>(a["start_s"].get<double>()) : std::nullopt);
  } catch (const std::invalid_argument& e) { ng::refuse(std::string(who) + ": " + e.what()); }
}

Eigen::MatrixXcd own_columns_transposed(const ng::Block& b) {          // [own samples x rows]
  const std::int64_t n = b.own();
  Eigen::MatrixXcd A(n, b.rows);
  for (std::int64_t r = 0; r < b.rows; ++r)
    for (std::int64_t t = 0; t < n; ++t) {
      const std::size_t at = static_cast<std::size_t>(r * b.width + b.left + t);
      A(t, r) = std::complex<double>(b.re[at], b.complex() ? b.im[at] : 0.0);
    }
  return A;
}

// ── pyramid.band_analytic ───────────────────────────────────────────────────────────────────────────────────────────
struct AnalyticPrepared {
  std::vector<std::int32_t> channels;
  double sfreq = 0;
  py::Shape shape = py::Shape::Morlet;
  py::Bands bank;
  std::size_t index = 0;
};

ng::NodeSpec band_analytic() {
  ng::NodeSpec n;
  n.name = "pyramid.band_analytic";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"channels_of", "an operator whose channels (its from_selection) are the ones kept, in its order (the inverse kernel)", false}};
  n.outputs = {{"z", "the band's analytic signal on the kept channels: channels rows, complex, at the recording's rate", false}};
  n.parameters = {{"band", Json(), "this band [lo, hi) in Hz (one of `bands`)", "Hz", {}, false},
                  {"bands", Json::array(), "the bank [[lo, hi], ...] the band belongs to (it sets the power partition's crossovers); empty: the octaves 1-2 ... 32-64 Hz", "Hz", {}, false},
                  {"shape", Json("morlet"), "the band filter: morlet (a Gaussian centred on (lo+hi)/2, FWHM hi-lo) | power_partition (nxr's bank: band powers add to broadband)", "-", {"morlet", "power_partition"}, false},
                  {"halo_seconds", Json(8.0), "the context read on each side of a chunk for the filter (the filter's edge effect is inside it)", "s", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "the block (chunk plus halo) is transformed in double and handed on in float32; the filter is circular over the block",
                             "the halo is `halo_seconds` on each side: the filter is circular over the block read, so its edge effect lies inside the halo");
  n.matches = "nxr analytic_bands (nxr/pipelines/atlas/build.py: the whole-record FFT filter, here per block with a halo); compute nxr::pyramid::spectrumOf, bandGains, analyticBand";
  n.description = "the analytic signal of one band of the recording's channels, zero phase: the block's spectrum times the band's amplitude gain, the positive frequencies doubled and the rest dropped";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = streamed(in, "x", "pyramid.band_analytic");
    if (x.value != ng::ValueType::real) ng::refuse("pyramid.band_analytic: x is a real recording");
    if (!in.count("channels_of")) ng::refuse("pyramid.band_analytic: channels_of (the kernel) names the channels kept");
    const py::Bands bank = bank_of(p["bands"], "pyramid.band_analytic");
    const auto b = band_of(p["band"], "pyramid.band_analytic");
    index_in(bank, b, "pyramid.band_analytic");
    try { py::bandGains(Eigen::VectorXd::Ones(1), bank, shape_of(p, "pyramid.band_analytic")); } catch (const std::invalid_argument& e) { ng::refuse(std::string("pyramid.band_analytic: ") + e.what()); }
    if (!(p["halo_seconds"].is_number() && p["halo_seconds"].get<double>() >= 0)) ng::refuse("pyramid.band_analytic: halo_seconds is 0 or more");
    ng::Planned out;
    ng::PortType z = x;
    z.value = ng::ValueType::complex;
    z.rows = 0;
    z.facts.erase("channels");
    z.facts["band"] = Json::array({b.first, b.second});
    out.out["z"] = z;
    out.halo = static_cast<std::int64_t>(std::ceil(p["halo_seconds"].get<double>() * *x.sfreq));
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    AnalyticPrepared a;
    a.sfreq = *in.at("x").sfreq;
    a.shape = shape_of(p, "pyramid.band_analytic");
    a.bank = bank_of(p["bands"], "pyramid.band_analytic");
    a.index = index_in(a.bank, band_of(p["band"], "pyramid.band_analytic"), "pyramid.band_analytic");
    a.channels = constant(constants, "channels_of", "pyramid.band_analytic").meta.value("channels", std::vector<std::int32_t>());
    if (a.channels.empty()) ng::refuse("pyramid.band_analytic: the kernel names no channel");
    return std::any(std::move(a));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t keep, const ng::Place&) {
    const AnalyticPrepared& a = std::any_cast<const AnalyticPrepared&>(prepared);
    const ng::Block& b = in.at("x");
    const Eigen::Map<const RowMajorF> all(b.re.data(), b.rows, b.width);
    const std::int64_t M = static_cast<std::int64_t>(a.channels.size());
    Eigen::MatrixXd block(b.width, M);                                       // [samples x channels], the whole block with its halo
    for (std::int64_t i = 0; i < M; ++i) {
      if (a.channels[static_cast<std::size_t>(i)] < 0 || a.channels[static_cast<std::size_t>(i)] >= b.rows) ng::refuse("pyramid.band_analytic: a channel of the kernel is outside the recording's channels");
      block.col(i) = all.row(a.channels[static_cast<std::size_t>(i)]).cast<double>().transpose();
    }
    const py::Spectrum spec = py::spectrumOf(block, a.sfreq);
    const Eigen::MatrixXd gains = py::bandGains(spec.freq, a.bank, a.shape);
    const Eigen::MatrixXcd A = py::analyticBand(spec, gains.row(static_cast<Eigen::Index>(a.index)).transpose());   // [samples x channels]
    const std::int64_t kl = std::min(keep, b.left), kr = std::min(keep, b.right), w = b.own() + kl + kr, c0 = b.left - kl;
    ng::Block z;
    z.rows = M; z.width = w; z.left = kl; z.right = kr;
    z.re.resize(static_cast<std::size_t>(M * w)); z.im.resize(z.re.size());
    for (std::int64_t i = 0; i < M; ++i)
      for (std::int64_t t = 0; t < w; ++t) {
        const std::complex<double> v = A(c0 + t, i);
        z.re[static_cast<std::size_t>(i * w + t)] = static_cast<float>(v.real());
        z.im[static_cast<std::size_t>(i * w + t)] = static_cast<float>(v.imag());
      }
    return ng::Blocks{{"z", std::move(z)}};
  };
  return n;
}

// ── pyramid.band_sums ───────────────────────────────────────────────────────────────────────────────────────────────
// The band's MEASUREMENTS OVER THE JOINT CELLS (leaf × frame): mergeable ingredients only (sums, maxima); means, densities and every ratio are derived at read. A bad sample carries no current.
struct SumsPrepared {
  Eigen::MatrixXd K;                         // [rows x channels]
  bool is_free = false;
  py::FrameKernel fk;                        // a free kernel in the canonical frames
  nxr::selection::MemberOfElement leaves;    // the leaf (member number) of each source row (-1: none)
  std::vector<double> area, w_tensor;        // [sources] the weights: the vertex area, and the area of a vertex with a frame (0 at a singular one)
  nxr::selection::Intervals frames;          // the frame grid resolved on the recording's line: one interval a frame
  std::int64_t L = 0, NF = 0;
  std::int64_t block_samples = 0;
  int threads = 1;
  Json band;                                 // {name, lo_hz, hi_hz, code}
  std::string kernel;
  std::int64_t channels = 0;
  bool exclude = true;                       // a sample in a bad span carries no current (bad_segments: exclude)
};
struct SumsState {
  std::int64_t frame0 = -1, frames = 0;      // the first frame the run's samples touch and how many it has seen
  nxr::partition::AnalyticSums acc;
  nxr::partition::TensorSums tacc;
  std::int64_t samples = 0, bad = 0;
};

struct OutSpec { const char* measure; double fill; };
const std::vector<OutSpec>& constrained_measures() {
  static const std::vector<OutSpec> m = {{"power_sum", 0.0}, {"envelope_sum", 0.0}, {"envelope_max", -std::numeric_limits<double>::infinity()}};
  return m;
}
const std::vector<OutSpec>& tensor_measures() {
  static const std::vector<OutSpec> m = {{"tensor_11", 0.0}, {"tensor_22", 0.0}, {"tensor_nn", 0.0}, {"tensor_12_re", 0.0}, {"tensor_12_im", 0.0},
                                         {"tensor_1n_re", 0.0}, {"tensor_1n_im", 0.0}, {"tensor_2n_re", 0.0}, {"tensor_2n_im", 0.0}, {"tensor_total", 0.0}};
  return m;
}

// the sums grown to hold `frames` frames (frames are folded in time order, so a run's span grows at the end: [frame x leaf], frame-major as the accumulators lay them out)
void grow(SumsState& s, bool is_free, std::int64_t L, std::int64_t frames) {
  if (is_free) {
    if (s.tacc.members == 0) s.tacc = nxr::partition::TensorSums(static_cast<int>(L), 0);
    if (frames <= s.tacc.tiles) return;
    for (auto* a : s.tacc.arrays()) a->resize(static_cast<std::size_t>(frames * L), 0.0);
    s.tacc.tiles = frames;
  } else {
    if (s.acc.members == 0) s.acc = nxr::partition::AnalyticSums(static_cast<int>(L), 0);
    if (frames <= s.acc.tiles) return;
    const std::size_t at = s.acc.power.size();
    s.acc.power.resize(static_cast<std::size_t>(frames * L), 0.0);
    s.acc.env.resize(static_cast<std::size_t>(frames * L), 0.0);
    s.acc.envmax.resize(static_cast<std::size_t>(frames * L), -std::numeric_limits<double>::infinity());
    (void)at;
    s.acc.tiles = frames;
  }
}

ng::NodeSpec band_sums() {
  ng::NodeSpec n;
  n.name = "pyramid.band_sums";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"z", "the band's analytic signal on the kernel's channels (pyramid.band_analytic), streamed", false},
              {"kernel", "the inverse kernel: a stored operator, sources x the recording's channels (3 rows a source: free orientation)", false},
              {"leaves", "the spatial grid: the leaves of the surface's tile ladder at one level (a set of the surface's vertices, one member a leaf)", false},
              {"frames", "the time grid: the frame tiling of the recording on the tower", false},
              {"surface", "the kernel's surface: its vertices and faces give the area weights", false},
              {"canonical", "the surface's canonical frames (a field, vertices x 9: e1, e2, n): required by a free-orientation kernel", true},
              {"exclude", "the recording's bad segments (spans): a bad sample carries no current", true}};
  n.outputs = {{"sums", "the band's mergeable measurements over the joint cells (leaf x frame): the values [leaves x frames], one array a measure", false}};
  n.parameters = {{"band", Json(), "this band [lo, hi) in Hz", "Hz", {}, false},
                  {"bands", Json::array(), "the bank the band belongs to; its index is the band's code (`atlas bands`)", "Hz", {}, false},
                  {"block_samples", Json(0), "samples projected through the kernel at a time (memory only: the result does not depend on it); 0: 512 constrained (rows x block x 4 float64), 128 free", "samples", {}, false},
                  {"bad_segments", Json("exclude"), "exclude = a sample in one of the recording's bad segments carries no current | ignore = every sample counts", "-", {"exclude", "ignore"}, false},
                  {"threads", Json(0), "workers of the kernel products (nxr::parallel): 0 = the machine's performance cores, 1 = serial; the result is bit-identical for every count", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "adds each good sample's current into its cell's sums: sums add and maxima merge, whatever the order of the blocks (up to rounding)");
  n.matches = "nxr measure.vectors.band_tensor and pipelines/atlas/build.py (the leaf x frame sums of a band's current); compute nxr::pyramid::bandSums, tensorBandSums";
  n.description = "a band's measurements over the cells of the joint grid (leaves x frames): per cell the sums of the area-weighted power, the envelope, the envelope's maximum of the current J = K z "
                  "(or, for a free-orientation kernel, the orientation tensor's ten sums in the surface's canonical frames), each a reduction over the cell's samples and sources";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& z = streamed(in, "z", "pyramid.band_sums");
    if (z.value != ng::ValueType::complex) ng::refuse("pyramid.band_sums: z is the band's analytic signal (complex): `pyramid.band_analytic` first");
    for (const char* k : {"kernel", "leaves", "frames", "surface"}) if (!in.count(k)) ng::refuse(std::string("pyramid.band_sums: ") + k + " is not given");
    if (in.at("kernel").cls != ng::PortClass::operator_) ng::refuse("pyramid.band_sums: kernel is a stored operator (an inverse kernel)");
    const py::Bands bank = bank_of(p["bands"], "pyramid.band_sums");
    index_in(bank, band_of(p["band"], "pyramid.band_sums"), "pyramid.band_sums");
    if (!(p["block_samples"].is_number_integer() && p["block_samples"].get<std::int64_t>() >= 0)) ng::refuse("pyramid.band_sums: block_samples is at least 1 (0: the default of the kernel's orientation)");
    if (!(p["threads"].is_number_integer() && p["threads"].get<int>() >= 0)) ng::refuse("pyramid.band_sums: threads is at least 1 (0: the machine's performance cores)");
    if (p["bad_segments"] != "exclude" && p["bad_segments"] != "ignore") ng::refuse("pyramid.band_sums: bad_segments is exclude or ignore");
    ng::Planned out;
    out.out["sums"] = constant_port("measurement");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    SumsPrepared s;
    const ng::Array& k = constant(constants, "kernel", "pyramid.band_sums");
    s.is_free = k.meta.value("to_value_type", std::string("scalar")) == "vector3";
    s.kernel = k.meta.value("name", std::string());
    s.K = Eigen::Map<const RowMajorD>(k.data.data(), k.rows, k.cols);
    s.channels = k.cols;
    const std::int64_t N = s.is_free ? k.rows / 3 : k.rows;
    if (s.is_free && k.rows % 3 != 0) ng::refuse("pyramid.band_sums: " + s.kernel + ": a free kernel has 3 rows a source, not " + std::to_string(k.rows));
    const ng::Array& surf = constant(constants, "surface", "pyramid.band_sums");
    if (surf.rows != N) ng::refuse("pyramid.band_sums: " + s.kernel + " has " + std::to_string(N) + " sources, its surface " + std::to_string(surf.rows) + " vertices");
    // the leaves: the member number of each source (a dyadic tile code), -1 in none
    const ng::Array& lv = constant(constants, "leaves", "pyramid.band_sums");
    if (lv.rows != N) ng::refuse("pyramid.band_sums: the leaves " + lv.meta.value("name", std::string()) + " hold " + std::to_string(lv.rows) + " vertices, the surface " + std::to_string(N));
    s.L = lv.meta.value("n_members", std::int64_t(0));
    if (s.L < 1) ng::refuse("pyramid.band_sums: the leaves name no member count");
    s.leaves.member_of.resize(static_cast<std::size_t>(N));
    for (std::int64_t v = 0; v < N; ++v) s.leaves.member_of[static_cast<std::size_t>(v)] = static_cast<std::int32_t>(lv.data[static_cast<std::size_t>(v)]);
    s.leaves.members = s.L;
    // the area weights: the mesh's barycentric vertex areas (nxr::ladder)
    nxr::ladder::Mesh mesh;
    mesh.n = static_cast<int>(N);
    mesh.positions = surf.data;
    for (const std::int32_t f : surf.meta.value("faces", std::vector<std::int32_t>())) mesh.faces.push_back(f);
    s.area = nxr::ladder::vertexAreas(mesh);
    // the frames: the tower's tiling of the recording as a grid, resolved on its line
    const nxr::tower::FrameTiling ft = tower_of(constant(constants, "frames", "pyramid.band_sums"), "pyramid.band_sums");
    s.frames = nxr::tower::frameIntervals(ft);
    s.NF = ft.count;
    if ((s.L << 1) >= (std::int64_t(1) << 31)) ng::refuse("pyramid.band_sums: " + std::to_string(s.L) + " leaves do not fit a 32-bit joint code; use a smaller depth");
    s.block_samples = p["block_samples"].get<std::int64_t>() > 0 ? p["block_samples"].get<std::int64_t>() : (s.is_free ? 128 : 512);
    s.threads = nxr::parallel::resolveThreads(p["threads"].get<int>());
    s.exclude = p["bad_segments"] == "exclude";
    const py::Bands bank = bank_of(p["bands"], "pyramid.band_sums");
    const auto b = band_of(p["band"], "pyramid.band_sums");
    s.band = band_record(b, index_in(bank, b, "pyramid.band_sums"));
    if (s.is_free) {                                                         // the kernel in the canonical frames (a vertex with no frame is out of the nine sums and in the total)
      const auto fr = constants.find("canonical");
      if (fr == constants.end() || fr->second.meta.value("placeholder", false)) ng::refuse("pyramid.band_sums: " + surf.meta.value("name", std::string("the surface")) + " has no canonical frames: the orientation tensor is measured in them (`nxr-neuro run frames` first)");
      const ng::Array& F = fr->second;
      if (static_cast<std::int64_t>(F.data.size()) != N * 9) ng::refuse("pyramid.band_sums: the canonical frames hold " + std::to_string(F.data.size()) + " values for " + std::to_string(N) + " vertices (9 each)");
      Eigen::MatrixXd e1(N, 3), e2(N, 3), nn(N, 3);
      std::vector<char> sing(static_cast<std::size_t>(N), 0);
      s.w_tensor = s.area;
      for (std::int64_t v = 0; v < N; ++v) {
        bool bad_row = false;
        for (int c = 0; c < 3; ++c) {
          e1(v, c) = F.data[static_cast<std::size_t>(v * 9 + c)]; e2(v, c) = F.data[static_cast<std::size_t>(v * 9 + 3 + c)]; nn(v, c) = F.data[static_cast<std::size_t>(v * 9 + 6 + c)];
          bad_row = bad_row || !std::isfinite(e1(v, c)) || !std::isfinite(e2(v, c));
        }
        sing[static_cast<std::size_t>(v)] = bad_row;
        if (bad_row) s.w_tensor[static_cast<std::size_t>(v)] = 0.0;
      }
      try { s.fk = py::frameKernel(s.K, e1, e2, nn, sing, s.threads); } catch (const std::invalid_argument& e) { ng::refuse(s.kernel + ": " + e.what()); }
    }
    return std::any(std::move(s));
  };
  n.start = [](const std::any&) { return std::any(SumsState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const SumsPrepared& pr = std::any_cast<const SumsPrepared&>(prepared);
    SumsState& s = std::any_cast<SumsState&>(state);
    const ng::Block& b = in.at("z");
    if (b.rows != pr.channels) ng::refuse("pyramid.band_sums: z has " + std::to_string(b.rows) + " channels, the kernel " + std::to_string(pr.channels));
    const std::int64_t n_own = b.own();
    if (n_own <= 0) return std::make_pair(ng::Blocks{}, std::move(state));
    std::vector<std::int32_t> tile = py::frameOfSamples(pr.frames, at.first, n_own, 0);          // the frame of each sample (grid member numbers)
    if (s.frame0 < 0) s.frame0 = tile.front();
    if (tile.front() < s.frame0) ng::refuse("pyramid.band_sums: blocks must meet in time order");
    for (auto& t : tile) t -= static_cast<std::int32_t>(s.frame0);
    grow(s, pr.is_free, pr.L, static_cast<std::int64_t>(tile.back()) + 1);
    const std::uint8_t* good = pr.exclude && at.good && !at.good->empty() ? at.good->data() : nullptr;
    const Eigen::MatrixXcd A = own_columns_transposed(b);                                       // [samples x channels]
    if (pr.is_free) py::tensorBandSums(s.tacc, pr.fk, A, good, pr.leaves, pr.w_tensor.data(), pr.area.data(), tile, pr.block_samples, pr.threads);
    else py::bandSums(s.acc, pr.K, A, good, pr.leaves, pr.area.data(), tile, pr.block_samples, pr.threads);
    s.samples += n_own;
    if (good) for (std::int64_t i = 0; i < n_own; ++i) if (!good[i]) ++s.bad;
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const SumsPrepared& pr = std::any_cast<const SumsPrepared&>(prepared);
    const SumsState& s = std::any_cast<const SumsState&>(state);
    if (s.frame0 < 0) ng::refuse("pyramid.band_sums: no sample was read");
    const std::vector<OutSpec>& ms = pr.is_free ? tensor_measures() : constrained_measures();
    const std::int64_t C = static_cast<std::int64_t>(ms.size()), NF = pr.NF, L = pr.L;
    // gather: [leaf x frame] over the recording's whole frame axis, a column a measure; frames outside the run's span hold the measure's empty value
    ng::Array a;
    a.rows = L * NF; a.cols = C;
    a.data.resize(static_cast<std::size_t>(a.rows * C));
    for (std::int64_t c = 0; c < C; ++c) {
      const std::vector<double>* from = nullptr;
      std::vector<std::vector<double>*> ta;
      if (pr.is_free) { ta = const_cast<nxr::partition::TensorSums&>(s.tacc).arrays(); from = ta[static_cast<std::size_t>(c)]; }
      else from = c == 0 ? &s.acc.power : c == 1 ? &s.acc.env : &s.acc.envmax;
      for (std::int64_t l = 0; l < L; ++l)
        for (std::int64_t f = 0; f < NF; ++f) {
          const std::int64_t fr = f - s.frame0;
          const bool in_span = fr >= 0 && fr < (pr.is_free ? s.tacc.tiles : s.acc.tiles);
          a.data[static_cast<std::size_t>((l * NF + f) * C + c)] = in_span ? (*from)[static_cast<std::size_t>(fr * L + l)] : ms[static_cast<std::size_t>(c)].fill;
        }
    }
    Json measures = Json::array();
    for (const auto& m : ms) measures.push_back(Json{{"measure", m.measure}, {"unit", Json()}});
    a.meta["describe"] = Json{{"measures", measures}, {"values_shape", Json::array({L, NF})}, {"band", pr.band}, {"within_code", pr.band["code"]},
                              {"layout", Json::array({"band", "tile", "frame"})}, {"bad_samples", s.bad}, {"first_frame", s.frame0},
                              {"description", std::string("the band ") + pr.band["name"].get<std::string>() + " of " + pr.kernel + " over the joint cells (leaf x frame): mergeable sums and maxima, ratios derived at read"}};
    return ng::Arrays{{"sums", std::move(a)}};
  };
  return n;
}

// ── pyramid.frame_samples ───────────────────────────────────────────────────────────────────────────────────────────
struct SamplesPrepared { nxr::selection::Intervals frames; std::int64_t NF = 0; bool exclude = true; };
struct SamplesState { std::int64_t frame0 = -1; std::vector<std::int64_t> counts; };

ng::NodeSpec frame_samples() {
  ng::NodeSpec n;
  n.name = "pyramid.frame_samples";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"x", "the recording: channels x samples (streamed): only its good samples are counted", false}, {"frames", "the time grid: the frame tiling of the recording on the tower", false},
              {"exclude", "the recording's bad segments (spans)", true}};
  n.outputs = {{"count", "the good samples of each frame: a measurement over the frame grid, the roll-up weight of every mean", false}};
  n.parameters = {{"bad_segments", Json("exclude"), "exclude = a sample in one of the recording's bad segments is not counted | ignore = every sample counts", "-", {"exclude", "ignore"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "adds each good sample to its frame's count: additive over blocks");
  n.matches = "nxr pipelines/atlas/build.py (the per-frame `samples`); compute nxr::pyramid::addSampleCounts, tower::sampleCounts";
  n.description = "the good samples of each frame of the frame grid (a bad sample carries no current and no count)";
  n.plan = [](const ng::Types& in, const Json&) {
    streamed(in, "x", "pyramid.frame_samples");
    if (!in.count("frames")) ng::refuse("pyramid.frame_samples: frames is not given");
    ng::Planned out;
    out.out["count"] = constant_port("measurement");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    SamplesPrepared s;
    s.exclude = p.value("bad_segments", std::string("exclude")) == "exclude";
    const nxr::tower::FrameTiling ft = tower_of(constant(constants, "frames", "pyramid.frame_samples"), "pyramid.frame_samples");
    s.frames = nxr::tower::frameIntervals(ft);
    s.NF = ft.count;
    return std::any(std::move(s));
  };
  n.start = [](const std::any&) { return std::any(SamplesState{}); };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const SamplesPrepared& pr = std::any_cast<const SamplesPrepared&>(prepared);
    SamplesState& s = std::any_cast<SamplesState&>(state);
    const ng::Block& b = in.at("x");
    const std::int64_t n_own = b.own();
    if (n_own <= 0) return std::make_pair(ng::Blocks{}, std::move(state));
    const std::int64_t f0 = py::frameOfSamples(pr.frames, at.first, 1, 0).front(), f1 = py::frameOfSamples(pr.frames, at.first + n_own - 1, 1, 0).front();
    if (s.frame0 < 0) s.frame0 = f0;
    if (f0 < s.frame0) ng::refuse("pyramid.frame_samples: blocks must meet in time order");
    if (static_cast<std::int64_t>(s.counts.size()) < f1 - s.frame0 + 1) s.counts.resize(static_cast<std::size_t>(f1 - s.frame0 + 1), 0);
    py::addSampleCounts(pr.frames, at.first, n_own, pr.exclude && at.good && !at.good->empty() ? at.good->data() : nullptr, s.frame0, s.counts);
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const SamplesPrepared& pr = std::any_cast<const SamplesPrepared&>(prepared);
    const SamplesState& s = std::any_cast<const SamplesState&>(state);
    if (s.frame0 < 0) ng::refuse("pyramid.frame_samples: no sample was read");
    ng::Array a;
    a.rows = pr.NF; a.cols = 1;
    a.data.assign(static_cast<std::size_t>(pr.NF), 0.0);
    for (std::size_t f = 0; f < s.counts.size(); ++f) a.data[static_cast<std::size_t>(s.frame0) + f] = static_cast<double>(s.counts[f]);
    a.meta["describe"] = Json{{"measures", Json::array({Json{{"measure", "count"}, {"unit", Json()}}})},
                              {"description", "the good samples of each frame of the recording (the roll-up weight)"}};
    return ng::Arrays{{"count", std::move(a)}};
  };
  return n;
}

// ── pyramid.estimate ────────────────────────────────────────────────────────────────────────────────────────────────
// THE SOURCE ESTIMATE the sums are of: a Field with no bytes (kind `source estimate`; its reductions ARE the pyramid, never K x itself), with the provenance of how it was made
struct EstimatePrepared { ng::Array kernel, frames, leaves; Json params; };

ng::NodeSpec estimate() {
  ng::NodeSpec n;
  n.name = "pyramid.estimate";
  n.kind = ng::Kind::array;
  n.inputs = {{"kernel", "the inverse kernel the estimate is made through", false}, {"frames", "the frame tiling of the recording", false}, {"leaves", "the leaves of the surface's tile ladder", false}};
  n.outputs = {{"estimate", "the byte-less source estimate (a field whose values are its measurements)", false}};
  n.parameters = {{"bands", Json::array(), "the bank [[lo, hi], ...]; empty: the octaves 1-2 ... 32-64 Hz", "Hz", {}, false},
                  {"shape", Json("morlet"), "the band filter", "-", {"morlet", "power_partition"}, false},
                  {"halo_seconds", Json(8.0), "the filter's context on each side of a chunk", "s", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a description: no data flows");
  n.matches = "nxr pipelines/atlas/build.py (the source estimate and its record)";
  n.description = "the byte-less source estimate of a recording through a kernel: its kind, orientation, bands and the warnings of what it cannot resolve";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"kernel", "frames", "leaves"}) if (!in.count(k)) ng::refuse(std::string("pyramid.estimate: ") + k + " is not given");
    shape_of(p, "pyramid.estimate");
    bank_of(p["bands"], "pyramid.estimate");
    ng::Planned out;
    out.out["estimate"] = constant_port("field");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    EstimatePrepared e;
    e.kernel = constant(constants, "kernel", "pyramid.estimate");
    e.kernel.data.clear();                                                  // the matrix is not needed to describe it
    e.frames = constant(constants, "frames", "pyramid.estimate");
    e.leaves = constant(constants, "leaves", "pyramid.estimate");
    e.leaves.data.clear();
    e.params = p;
    return std::any(std::move(e));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const EstimatePrepared& c = std::any_cast<const EstimatePrepared&>(prepared);
    const Json& p = c.params;
    const std::string kname = c.kernel.meta.value("name", std::string());
    const bool is_free = c.kernel.meta.value("to_value_type", std::string("scalar")) == "vector3";
    const py::Shape shape = shape_of(p, "pyramid.estimate");
    const py::Bands bank = bank_of(p["bands"], "pyramid.estimate");
    Json bj = Json::array();
    for (const auto& b : bank) bj.push_back(Json::array({b.first, b.second}));
    // nxr's key `<recording>__<method>_<stamp>` (a Brainstorm-made kernel is named `<method>_KERNEL_<stamp>`); another name stands in whole
    std::string method, stamp;
    if (const auto at = kname.find("_KERNEL_"); at != std::string::npos) { method = kname.substr(0, at); stamp = kname.substr(at + 8, 11); }
    const nxr::tower::FrameTiling ft = tower_of(c.frames, "pyramid.estimate");
    const Json warnings = Json::array({
        "the minimum-norm current is smooth: its spatial resolution is the kernel's point-spread, not the leaves'; a leaf finer than it carries no independent information",
        "a band's analytic amplitude is meaningful for a band narrow against its centre; octave bands are wide, read the powers as band-limited energy, not as an oscillation's amplitude",
        "the band filter is circular over the block read (chunk + halo): the first and last `halo` seconds of a recording carry its edge effect, as nxr's whole-record filter does at both ends",
        shape == py::Shape::Morlet ? "Morlet bands overlap their neighbours: band powers do not add to broadband (shape power_partition does)" : "power_partition crossovers are half an octave wide: a tone near an edge splits between two bands",
        is_free ? "free orientation: the tensor's principal axis is the dominant orientation of the current as the kernel's inverse made it; for a minimum-norm kernel that is partly its own prior (the kernel's orientation prior is not a measurement); a vertex with no canonical frame is in the total with its tangential power under an arbitrary tangent basis and out of the nine sums"
                : "constrained orientation only: the current is the kernel's normal component; the free-orientation tensor is not stored (a free kernel gives it)"});
    Json producer = Json{{"by", "nxr-neuro run pyramid"}, {"kernel", kname}, {"method", method.empty() ? Json() : Json(method)}, {"stamp", stamp.empty() ? Json() : Json(stamp)},
                         {"orientation", is_free ? "free" : "constrained"}, {"shape", py::shapeName(shape)}, {"bands_hz", bj}, {"halo_seconds", p["halo_seconds"]},
                         {"frame_level", ft.level}, {"frame_s", ft.frame_s()}, {"leaves", c.leaves.meta.value("n_members", std::int64_t(0))},
                         {"filterbank", shape == py::Shape::Morlet ? "Morlet: Gaussian gain, unit peak, centred on (lo+hi)/2, FWHM hi-lo (analytic: positive frequencies doubled)" : "octave power partition (cos/sin crossover, half an octave), nxr band_gains"},
                         {"warnings", warnings}};
    ng::Array a;
    a.meta["describe"] = Json{{"value_type", is_free ? "vector3" : "scalar"}, {"no_array", true}, {"producer", producer},
                              {"description", "the kernel applied to the recording, stored as its pyramid reductions only, never K x itself"}};
    return ng::Arrays{{"estimate", std::move(a)}};
  };
  return n;
}

// ── pyramid.bands ───────────────────────────────────────────────────────────────────────────────────────────────────
// THE BAND BANK (nxr D158 `atlas bands`): spans [lo, hi) on the frequency line, code = the band's index, so a measurement's `within` key is the band's index in the bank
ng::NodeSpec bands_bank() {
  ng::NodeSpec n;
  n.name = "pyramid.bands";
  n.kind = ng::Kind::array;
  n.outputs = {{"freq", "the frequency line the bands are spans of (0-500 Hz in 0.5 Hz steps)", false}, {"bands", "the bank: a span [lo, hi) of the line a band, code = its index, named", false}};
  n.parameters = {{"bands", Json::array(), "the bank [[lo, hi], ...] in Hz; empty: the octaves 1-2 ... 32-64 Hz (nxr DEFAULT_BANDS)", "Hz", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a description: no data flows");
  n.matches = "nxr convert.frequency and the pyramid's `atlas bands` (D158)";
  n.description = "the pyramid's frequency band bank as a set of spans on the frequency line, code = the band's index";
  n.plan = [](const ng::Types&, const Json& p) {
    const py::Bands bank = bank_of(p["bands"], "pyramid.bands");
    for (std::size_t k = 0; k < bank.size(); ++k) if (!(bank[k].second > bank[k].first && bank[k].first >= 0)) ng::refuse("pyramid.bands: band " + std::to_string(k) + " needs 0 <= lo < hi");
    ng::Planned out;
    out.out["freq"] = constant_port("manifold");
    out.out["bands"] = constant_port("selection");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays&, const Json& p, ng::Shared&) { return std::any(bank_of(p["bands"], "pyramid.bands")); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const py::Bands bank = std::any_cast<const py::Bands&>(prepared);
    ng::Array freq;
    freq.rows = 1; freq.cols = 1; freq.data = {0.0};
    freq.meta["describe"] = Json{{"count", std::int64_t(1001)}, {"step", 0.5}, {"start", 0.0}, {"unit", "Hz"}, {"dimension", "frequency"}};
    ng::Array bands;
    bands.rows = static_cast<std::int64_t>(bank.size()); bands.cols = 1;
    Json spans = Json::array(), entries = Json::array();
    for (std::size_t k = 0; k < bank.size(); ++k) {
      spans.push_back(Json{{"start", bank[k].first}, {"stop", bank[k].second}, {"code", static_cast<std::int64_t>(k)}});
      entries.push_back(g(bank[k].first) + "\xe2\x80\x93" + g(bank[k].second) + " Hz");
      bands.data.push_back(static_cast<double>(k));
    }
    bands.meta["describe"] = Json{{"spans", spans}, {"entries", entries}, {"description", "the pyramid's frequency bands: a span [lo, hi) of the Frequency line each, code = the band's index"}};
    return ng::Arrays{{"freq", std::move(freq)}, {"bands", std::move(bands)}};
  };
  return n;
}

// ── pyramid.spatial_bands ───────────────────────────────────────────────────────────────────────────────────────────
// THE SPATIAL BAND BANK of a surface: one span [lambda_lo, lambda_hi) of the Eigenvalue line a level of its tile ladder, code = the level (the level IS its Weyl band: Weyl's law on the ladder's mean piece area
// A, level l the band [4 pi 2^(l-1) / A, 4 pi 2^l / A), level 0 from 0; nxr::selection::weyl_level is its inverse). A measurement's spatial band is its `within` key.
struct SpatialPrepared { std::int64_t levels = 0; double piece_area = 0; std::string surface; };

ng::NodeSpec spatial_bands() {
  ng::NodeSpec n;
  n.name = "pyramid.spatial_bands";
  n.kind = ng::Kind::array;
  n.inputs = {{"tiles", "the surface's tile ladder (its record: the levels, the surface's area and pieces)", false}};
  n.outputs = {{"eigenvalue", "the Laplace-Beltrami eigenvalue line (0 to 4.1e6 1/m^2 in steps of 1000)", false}, {"bands", "a span of the line a level of the ladder, code = the level", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a description from the ladder's record: no data flows");
  n.matches = "nxr::filterbank::weyl (Weyl's law on the mean piece area) and the Studio's engine/ladder.ts levels' bands";
  n.description = "the spatial bands of a surface's tile ladder as a set of spans on the eigenvalue line, one a level, code = the level";
  n.plan = [](const ng::Types& in, const Json&) {
    if (!in.count("tiles")) ng::refuse("pyramid.spatial_bands: tiles (the surface's ladder) is not given");
    ng::Planned out;
    out.out["eigenvalue"] = constant_port("manifold");
    out.out["bands"] = constant_port("selection");
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& constants, const Json&, ng::Shared&) {
    const ng::Array& t = constant(constants, "tiles", "pyramid.spatial_bands");
    const Json gp = t.meta.value("params", Json::object());
    if (!gp.is_object() || !gp.contains("depth") || !gp.contains("area_m2") || !gp.contains("pieces"))
      ng::refuse("pyramid.spatial_bands: " + t.meta.value("name", std::string("the ladder")) + " records no depth, area or pieces (a ladder made by another producer)");
    SpatialPrepared s;
    s.levels = gp["depth"].get<std::int64_t>() + 1;
    s.piece_area = gp["area_m2"].get<double>() / std::max(1, gp["pieces"].get<int>());
    s.surface = t.meta.value("name", std::string());
    if (!(s.piece_area > 0)) ng::refuse("pyramid.spatial_bands: the ladder's pieces have no area");
    return std::any(std::move(s));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const SpatialPrepared& s = std::any_cast<const SpatialPrepared&>(prepared);
    ng::Array line;
    line.rows = 1; line.cols = 1; line.data = {0.0};
    line.meta["describe"] = Json{{"count", std::int64_t(4096)}, {"step", 1000.0}, {"start", 0.0}, {"unit", "1/m\xc2\xb2"}, {"dimension", "eigenvalue"}, {"folder", "eigen"}, {"cs", "eigenvalue:Laplace-Beltrami"}};
    ng::Array bands;
    bands.rows = s.levels; bands.cols = 1;
    Json spans = Json::array(), entries = Json::array();
    const double four_pi_over_a = 4.0 * M_PI / s.piece_area;
    for (std::int64_t l = 0; l < s.levels; ++l) {
      const double hi = four_pi_over_a * std::ldexp(1.0, static_cast<int>(l)), lo = l == 0 ? 0.0 : hi / 2;
      spans.push_back(Json{{"start", lo}, {"stop", hi}, {"code", l}});
      entries.push_back("level " + std::to_string(l));
      bands.data.push_back(static_cast<double>(l));
    }
    bands.meta["describe"] = Json{{"spans", spans}, {"entries", entries}, {"unit", "1/m\xc2\xb2"},
                                  {"params", Json{{"spatial_bands", Json{{"levels", s.levels}, {"unit", "1/m^2"}, {"piece_area_m2", s.piece_area}}}}},
                                  {"description", "the spatial bands of the surface: a span [lambda_min, lambda_max) of the Eigenvalue line per level of its tile ladder (the level is its Weyl band), code = the level"}};
    return ng::Arrays{{"eigenvalue", std::move(line)}, {"bands", std::move(bands)}};
  };
  return n;
}

}  // namespace

void register_pyramid_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {band_analytic(), band_sums(), frame_samples(), estimate(), bands_bank(), spatial_bands()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
