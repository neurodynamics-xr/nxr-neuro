// nxr-pipelines — THE APPARENT MOTION OF A SOURCE MAP AS NODES (the chain of the hand-wired `opticalflow` verb; spec docs/specs/2026-11-17-flow-verbs.md, section 2).
//
//   recording --timebank.member_analytic--> z [channels x frames, complex]      RETIMING: one filter-bank member at its OWN rate, the halo declared (its margin)
//             --flow.activation----------> a [N x frames]                       pointwise: J = K z (the linear map before the non-linear step, D106), then |J|
//             --flow.max_amplitude-------> the window's maximum |a|              accumulate: stored by its own definition (`flow_amplitude`), read BY NAME by the flow run (rule 13)
//             --flow.velocity------------> v [3N x frames]                       stateful: Horn-Schunck between consecutive frames; the previous map is its state; the unit amplitude is the stored maximum
//             --flow.hodge_modes---------> b [K x frames]                        pointwise: the velocity carried to the glued surface, read in the package's modes (exact / coexact)
//             --flow.series--------------> the flow weights as an array [K x steps] and the line they lie on
//
// Every node is a function of the block it is handed. The neighbourhood is explicit: the member's coefficients at the frames inside a block's own samples are computed from the block and its
// halo (the filter's margin plus the lattice period), so a result does not depend on the chunk size beyond the filter's own tail, which the margin bounds. The frame lattice is the member's:
// frame g sits at g p/q samples from sample 0 of the recording, the same for every chunk. A reduction over space or time of the velocity (divergence, curl, speed over a leaf and a frame) is a
// measurement over a grid selection: it is NOT a node here (the grids are being redesigned as plain selections; see the spec).
//
// WHAT IT CANNOT RESOLVE: kinematic, not physical (where activity is created or destroyed the residual appears as divergence); aperture-limited (Horn-Schunck recovers the velocity along the
// intensity gradient only; alpha sets how much is data); per frame, not per second; the filter's margin at a recording's ends is zero padding.
#include "nxr/pipelines/flow_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/signal/fourier.h"
#include "nxr/graph/node.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/meeg/sources.h"
#include "nxr/signal/opticalflow.h"
#include "nxr/signal/opticalflow_methods.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
namespace tb = nxr::signal::timefilterbank;
using nxr::graph::Json;

namespace {

using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

const ng::Array& constant(const ng::Arrays& c, const char* port, const char* who) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + port + " was not handed over");
  return it->second;
}
const ng::PortType& need_port(const ng::Types& in, const char* port, const char* who) {
  const auto it = in.find(port);
  if (it == in.end()) ng::refuse(std::string(who) + ": port '" + port + "' is not given");
  return it->second;
}
ng::PortType constant_port(const char* cls) {
  ng::PortType t;
  t.cls = ng::PortClass::constant;
  t.facts["cls"] = cls;
  return t;
}
// the value of a column of a block's own samples, [rows x own] as a double matrix
Eigen::MatrixXd own_real(const ng::Block& b) {
  const std::int64_t n = b.own();
  Eigen::MatrixXd A(b.rows, n);
  for (std::int64_t r = 0; r < b.rows; ++r)
    for (std::int64_t t = 0; t < n; ++t) A(r, t) = b.re[static_cast<std::size_t>(r * b.width + b.left + t)];
  return A;
}
Eigen::MatrixXcd own_complex(const ng::Block& b) {
  const std::int64_t n = b.own();
  Eigen::MatrixXcd A(b.rows, n);
  for (std::int64_t r = 0; r < b.rows; ++r)
    for (std::int64_t t = 0; t < n; ++t) {
      const std::size_t at = static_cast<std::size_t>(r * b.width + b.left + t);
      A(r, t) = std::complex<double>(b.re[at], b.complex() ? b.im[at] : 0.0);
    }
  return A;
}
// a block of the own columns only (no halo): what a node after the retiming node hands on
ng::Block block_of(const Eigen::MatrixXd& A) {
  ng::Block b;
  b.rows = A.rows(); b.width = A.cols();
  b.re.resize(static_cast<std::size_t>(b.rows * b.width));
  for (Eigen::Index r = 0; r < A.rows(); ++r)
    for (Eigen::Index t = 0; t < A.cols(); ++t) b.re[static_cast<std::size_t>(r * b.width + t)] = static_cast<float>(A(r, t));
  return b;
}
ng::Block block_of(const Eigen::MatrixXcd& A) {
  ng::Block b = block_of(Eigen::MatrixXd(A.real()));
  b.im.resize(b.re.size());
  for (Eigen::Index r = 0; r < A.rows(); ++r)
    for (Eigen::Index t = 0; t < A.cols(); ++t) b.im[static_cast<std::size_t>(r * b.width + t)] = static_cast<float>(A(r, t).imag());
  return b;
}
std::int64_t ceil_div(std::int64_t a, std::int64_t b) { return a >= 0 ? (a + b - 1) / b : -((-a) / b); }

// ── the member's filter bank (timebank.member_analytic) ────────────────────────────────────────────────────────────
struct BankChoice { tb::TimeFilterbank bank; nxr::signal::Options wo; int member = 0; double centre = 0; };

tb::Family family_of(const std::string& f, const char* who) {
  if (f == "itersine") return tb::Family::Itersine;
  if (f == "meyer") return tb::Family::Meyer;
  if (f == "morlet") return tb::Family::Morlet;
  if (f == "morse") return tb::Family::Morse;
  if (f == "amor") return tb::Family::Amor;
  if (f == "bump") return tb::Family::Bump;
  ng::refuse(std::string(who) + ": family is itersine, meyer, morlet, morse, amor or bump (no default)");
}
BankChoice choose_bank(const Json& p, double sfreq, int channels, const char* who) {
  if (!p.contains("family") || !p["family"].is_string()) ng::refuse(std::string(who) + ": the filter-bank family is required (itersine, meyer, morlet, morse, amor or bump)");
  if (!p.contains("gain") || !p["gain"].is_number() || !(p["gain"].get<double>() > 0)) ng::refuse(std::string(who) + ": the filter-bank gain is required and positive (2 makes |c| the physical amplitude)");
  if (!p.contains("freq") || !p["freq"].is_number() || !(p["freq"].get<double>() > 0) || !(p["freq"].get<double>() < sfreq / 2))
    ng::refuse(std::string(who) + ": freq must satisfy 0 < freq < Nyquist (" + std::to_string(sfreq / 2) + " Hz)");
  const double resolution = p["resolution"].get<double>();
  if (!(resolution > 0)) ng::refuse(std::string(who) + ": resolution must be positive");
  const tb::Family fam = family_of(p["family"].get<std::string>(), who);
  tb::Design d(fam, p["gain"].get<double>(), tb::Sides::Analytic);
  if (p.contains("centres") && p["centres"].is_array()) d.centres = p["centres"].get<std::vector<double>>();
  if (fam == tb::Family::Morlet && d.centres.empty()) d.centres = {p["freq"].get<double>()};
  d.cycles = p["cycles"].get<double>(); d.voices = p["voices"].get<int>(); d.anchor = p["anchor"].get<double>();
  if (p.contains("limits") && p["limits"].is_array() && p["limits"].size() == 2) d.limits = std::make_pair(p["limits"][0].get<double>(), p["limits"][1].get<double>());
  d.halfWidth = p["half_width"].get<double>(); d.gamma = p["gamma"].get<double>(); d.timeBandwidth = p["time_bandwidth"].get<double>();
  BankChoice c;
  const std::size_t nfft = nxr::signal::next_fast_len(static_cast<std::size_t>(std::ceil(sfreq / resolution)), true);
  try { c.bank = tb::design(d, nfft, sfreq); } catch (const std::invalid_argument& e) { ng::refuse(std::string(who) + ": the filter bank: " + e.what()); }
  c.wo.oversample = p["oversample"].get<double>();
  c.wo.tolerance = p.contains("tolerance") && p["tolerance"].is_number() ? p["tolerance"].get<double>() : 1e-7;   // 1e-7: float32 recordings (the blocks are float32); name it for float64
  if (!(c.wo.tolerance > 0) || !(c.wo.tolerance < 1)) ng::refuse(std::string(who) + ": tolerance must be in (0, 1)");
  const Eigen::VectorXd centres = tb::centerFrequencies(c.bank);
  for (int m = 1; m < c.bank.size(); ++m) if (std::abs(centres[m] - p["freq"].get<double>()) < std::abs(centres[c.member] - p["freq"].get<double>())) c.member = m;
  c.centre = centres[c.member];
  (void)channels;
  return c;
}

struct MemberPrepared {
  nxr::signal::Wt wt;                 // the template: its plan; a block copies it (no state is held between blocks)
  int member = 0;
  std::int64_t p = 1, q = 1, P = 1, margin = 0;
  std::int64_t channels = 0, n_samples = 0;
  std::vector<std::int32_t> picked;
  double rate = 0, centre = 0;
  MemberPrepared(const BankChoice& c, int ch) : wt(c.bank, ch, c.wo) {}
};

ng::NodeSpec member_analytic() {
  ng::NodeSpec n;
  n.name = "signal.member_analytic";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"x", "the recording: channels x samples (streamed)", false}, {"channels_of", "an operator whose channels (its from_selection) are the ones kept, in its order (the inverse kernel)", false}};
  n.outputs = {{"z", "the member's coefficients on the kept channels at the member's OWN rate: channels rows, complex, axis frames", false}};
  n.parameters = {{"family", Json(), "the filter bank: itersine | meyer | morlet | morse | amor | bump (no default)", "-", {"itersine", "meyer", "morlet", "morse", "amor", "bump"}, false},
                  {"gain", Json(), "the peak in-band response of each member (no default; 2 makes |c| the physical amplitude of a real recording)", "-", {}, false},
                  {"freq", Json(), "the frequency whose nearest member is analysed", "Hz", {}, false},
                  {"centres", Json::array(), "Morlet, Morse, Amor, Bump: the members' centres; for Morlet, empty means the single member at freq", "Hz", {}, false},
                  {"cycles", Json(6.0), "Morlet: the width is centre / cycles", "cycles", {}, false},
                  {"voices", Json(4), "Itersine: members per octave", "voices/octave", {}, false},
                  {"anchor", Json(1.0), "Itersine: the anchor frequency", "Hz", {}, false},
                  {"limits", Json(), "Itersine, Meyer: the bank's frequency limits [lo, hi]", "Hz", {}, true},
                  {"half_width", Json(1.0 / 3.0), "Meyer: the transition half-width", "octaves", {}, false},
                  {"gamma", Json(3.0), "Morse: gamma", "-", {}, false},
                  {"time_bandwidth", Json(60.0), "Morse: the time-bandwidth product", "-", {}, false},
                  {"resolution", Json(0.5), "the frequency resolution of the grid the bank is designed on", "Hz", {}, false},
                  {"oversample", Json(2.0), "the member's own rate is oversample x its bins' rate", "-", {}, false},
                  {"tolerance", Json(), "the impulse response's tail the margin may leave out, relative to its norm; null: 1e-7 (float32 blocks)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::windowed, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "each block (own samples and halo) is filtered in double (nxr::signal, overlap-save) and the frames inside the own samples are handed on in float32",
                             "the halo is the filter's margin (tolerance-bound) plus the lattice period on each side; the record's ends are zero padded");
  n.matches = "Rheome timebank.analysisOwnRate / @timefilterbank wt (the coefficients at the member's own rate); compute nxr::signal::Wt";
  n.description = "one filter-bank member's analytic coefficients of the recording's channels at the member's own rate (frames on the lattice g p/q samples), a retiming node";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& x = need_port(in, "x", "signal.member_analytic");
    if (!x.streamed() || x.axis != ng::Axis::samples || !x.sfreq || !(*x.sfreq > 0)) ng::refuse("signal.member_analytic: x is a recording's samples (a streamed field with a rate)");
    if (x.value != ng::ValueType::real) ng::refuse("signal.member_analytic: x is a real recording");
    if (!in.count("channels_of")) ng::refuse("signal.member_analytic: channels_of (the kernel) names the channels kept");
    const BankChoice c = choose_bank(p, *x.sfreq, 0, "signal.member_analytic");
    const nxr::signal::Wt w(c.bank, 1, c.wo);
    const nxr::signal::Plan& plan = w.plan();
    ng::Planned out;
    ng::PortType z = x;
    z.value = ng::ValueType::complex;
    z.rows = 0;
    z.axis = ng::Axis::frames;
    z.sfreq = plan.rate[c.member];
    z.facts.erase("channels");
    z.facts["step_p"] = plan.p[c.member]; z.facts["step_q"] = plan.q[c.member];
    z.facts["centre_hz"] = c.centre; z.facts["rate_hz"] = plan.rate[c.member];
    out.out["z"] = z;
    out.halo = plan.margin + plan.P;
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays& constants, const Json& p, ng::Shared&) {
    const double fs = *in.at("x").sfreq;
    const std::vector<std::int32_t> chans = constant(constants, "channels_of", "signal.member_analytic").meta.value("channels", std::vector<std::int32_t>());
    if (chans.empty()) ng::refuse("signal.member_analytic: the kernel names no channel");
    const BankChoice c = choose_bank(p, fs, static_cast<int>(chans.size()), "signal.member_analytic");
    MemberPrepared m(c, static_cast<int>(chans.size()));
    const nxr::signal::Plan& plan = m.wt.plan();
    m.member = c.member; m.p = plan.p[c.member]; m.q = plan.q[c.member]; m.P = plan.P; m.margin = plan.margin;
    m.channels = static_cast<std::int64_t>(chans.size()); m.picked = chans; m.rate = plan.rate[c.member]; m.centre = c.centre;
    return std::any(std::move(m));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    const MemberPrepared& m = std::any_cast<const MemberPrepared&>(prepared);
    const ng::Block& b = in.at("x");
    // the lattice is global: frame g sits at g p / q samples from sample 0 of the recording. The block is fed from the first sample at or after its start that is a multiple of P (every
    // member's lattice then starts there), so the frames are the same whatever the chunk size.
    const std::int64_t a_blk = at.first - b.left;                                  // the block's first sample, in the recording
    const std::int64_t a_feed = ceil_div(a_blk, m.P) * m.P;                        // the first sample fed (a multiple of P)
    const std::int64_t skip = a_feed - a_blk;
    const std::int64_t width = b.width - skip;
    if (width <= 0) ng::refuse("signal.member_analytic: the block is shorter than the lattice period");
    Eigen::MatrixXd X(m.channels, width);
    for (std::int64_t i = 0; i < m.channels; ++i) {
      const std::int32_t ch = m.picked[static_cast<std::size_t>(i)];
      if (ch < 0 || ch >= b.rows) ng::refuse("signal.member_analytic: a channel of the kernel is outside the recording's channels");
      for (std::int64_t t = 0; t < width; ++t) X(i, t) = b.re[static_cast<std::size_t>(ch * b.width + skip + t)];
    }
    nxr::signal::Wt w = m.wt;                                                         // a copy of the template: no state is shared between blocks
    std::vector<nxr::signal::Block> got = w.add(X);
    std::vector<nxr::signal::Block> rest = w.finish();
    // the frames g (global) with at.first <= g p/q < at.first + at.n; local instant k = g - a_feed q / p (a_feed is a multiple of p: P is a multiple of every p)
    const std::int64_t g0 = ceil_div(at.first * m.q, m.p), g1 = ceil_div((at.first + at.n) * m.q, m.p);
    const std::int64_t offset = a_feed / m.p * m.q;
    Eigen::MatrixXcd Z = Eigen::MatrixXcd::Constant(m.channels, std::max<std::int64_t>(0, g1 - g0), std::complex<double>(0, 0));
    std::vector<char> have(static_cast<std::size_t>(std::max<std::int64_t>(0, g1 - g0)), 0);
    for (const auto* blocks : {&got, &rest}) {
      const nxr::signal::Block& mb = (*blocks)[static_cast<std::size_t>(m.member)];
      for (Eigen::Index r = 0; r < mb.coef.rows(); ++r) {
        const std::int64_t g = mb.first + r + offset;
        if (g < g0 || g >= g1) continue;
        Z.col(g - g0) = mb.coef.row(r).transpose();
        have[static_cast<std::size_t>(g - g0)] = 1;
      }
    }
    for (const char h : have) if (!h) ng::refuse("signal.member_analytic: the filter returned no coefficient for a frame inside the block (the halo is shorter than the filter's margin)");
    return ng::Blocks{{"z", block_of(Z)}};
  };
  n.frame_place = [](const std::any& prepared, const ng::Place& at) {
    const MemberPrepared& m = std::any_cast<const MemberPrepared&>(prepared);
    ng::FramePlace f;
    f.first = ceil_div(at.first * m.q, m.p);
    f.n = ceil_div((at.first + at.n) * m.q, m.p) - f.first;
    return f;
  };
  n.record = [](const std::any& prepared) {
    const MemberPrepared& m = std::any_cast<const MemberPrepared&>(prepared);
    return Json{{"member", m.member}, {"centre_hz", m.centre}, {"rate_hz", m.rate}, {"step_samples", static_cast<double>(m.p) / static_cast<double>(m.q)}, {"margin_samples", m.margin}, {"lattice_period", m.P}};
  };
  return n;
}

// ── flow.activation ──────────────────────────────────────────────────────────────────────────────────────────────────
struct ActivationPrepared { Eigen::MatrixXd K; int orientations = 3; bool signed_input = false; };

ng::NodeSpec activation() {
  ng::NodeSpec n;
  n.name = "signal.flow_activation";
  n.kind = ng::Kind::chunk;
  n.linear_map = false;
  n.inputs = {{"z", "the member's coefficients on the kernel's channels (signal.member_analytic), frames", false}, {"kernel", "the inverse kernel: a stored operator, sources x channels (3 rows a source: free orientation)", false}};
  n.outputs = {{"a", "the activation map of each frame: one non-negative number a source (amplitude |J|), or the signed value for a fixed kernel with input signed", false}};
  n.parameters = {{"input", Json("amplitude"), "amplitude = |J| of the kernel applied to the complex coefficients | signed = the real part (fixed-orientation kernels only)", "-", {"amplitude", "signed"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input, "the kernel is applied in double to each frame on its own, then the amplitude; handed on in float32");
  n.matches = "Rheome flow.apparent (the amplitude |J| of the kernel applied to the wavelet coefficients); meeg nxr::meeg::applyKernel, amplitude";
  n.description = "J = K z (the linear map before the non-linear step) then the amplitude per source: the map the flow is computed on";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& z = need_port(in, "z", "signal.flow_activation");
    if (z.axis != ng::Axis::frames || z.value != ng::ValueType::complex) ng::refuse("signal.flow_activation: z is the member's coefficients on frames (complex): `signal.member_analytic` first");
    if (!in.count("kernel")) ng::refuse("signal.flow_activation: kernel is not given");
    if (p["input"] != "amplitude" && p["input"] != "signed") ng::refuse("signal.flow_activation: input is amplitude or signed");
    ng::Planned out;
    ng::PortType a = z;
    a.value = ng::ValueType::real; a.rows = 0;
    out.out["a"] = a;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    ActivationPrepared a;
    const ng::Array& k = constant(c, "kernel", "signal.flow_activation");
    a.orientations = k.meta.value("to_value_type", std::string("scalar")) == "vector3" ? 3 : 1;
    a.K = Eigen::Map<const MatRM>(k.data.data(), k.rows, k.cols);
    a.signed_input = p["input"] == "signed";
    if (a.signed_input && a.orientations != 1)
      ng::refuse("signal.flow_activation: input signed needs a fixed-orientation kernel (one value per source); " + k.meta.value("name", std::string()) + " has free orientations, whose signed sum has no meaning");
    return std::any(std::move(a));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    const ActivationPrepared& a = std::any_cast<const ActivationPrepared&>(prepared);
    const Eigen::MatrixXcd Z = own_complex(in.at("z"));
    if (Z.rows() != a.K.cols()) ng::refuse("signal.flow_activation: z has " + std::to_string(Z.rows()) + " channels, the kernel " + std::to_string(a.K.cols()));
    const Eigen::MatrixXcd J = nxr::meeg::applyKernel(a.K, Z);
    const Eigen::MatrixXd A = a.signed_input ? Eigen::MatrixXd(J.real()) : nxr::meeg::amplitude(J, a.orientations);
    return ng::Blocks{{"a", block_of(A)}};
  };
  return n;
}

// ── flow.max_amplitude ───────────────────────────────────────────────────────────────────────────────────────────────
// A REDUCTION the flow needs as a constant (Rheome scales the map to unit amplitude over the whole window): the window's maximum |a|, stored by its own definition (`flow_amplitude`) and read by
// name by the flow run (rule 13 of the abc round), never a second pass inside a run.
struct MaxState { double max = 0; std::int64_t frames = 0; };

ng::NodeSpec max_amplitude() {
  ng::NodeSpec n;
  n.name = "signal.flow_max_amplitude";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"a", "the activation map of each frame (signal.flow_activation)", false}};
  n.outputs = {{"maximum", "the largest absolute value of the map over the run's frames: one number", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a running maximum: it merges whatever the order or the blocks");
  n.matches = "Rheome flow.apparent (the window's unit amplitude); compute nxr::signal::FlowEngine::observe";
  n.description = "the maximum of |a| over every frame and source of the run";
  n.plan = [](const ng::Types& in, const Json&) {
    if (need_port(in, "a", "signal.flow_max_amplitude").axis != ng::Axis::frames) ng::refuse("signal.flow_max_amplitude: a is the activation on frames");
    ng::Planned out;
    out.out["maximum"] = constant_port("statistic");
    return out;
  };
  n.start = [](const std::any&) { return std::any(MaxState{}); };
  n.step = [](const std::any&, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    MaxState& s = std::any_cast<MaxState&>(state);
    const ng::Block& b = in.at("a");
    for (std::int64_t r = 0; r < b.rows; ++r)
      for (std::int64_t t = 0; t < b.own(); ++t) {
        const float v = b.re[static_cast<std::size_t>(r * b.width + b.left + t)];
        if (std::isfinite(v)) s.max = std::max(s.max, static_cast<double>(std::abs(v)));
      }
    s.frames += b.own();
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any&, std::any state) {
    const MaxState& s = std::any_cast<const MaxState&>(state);
    if (s.frames == 0) ng::refuse("signal.flow_max_amplitude: no frame was read");
    ng::Array a;
    a.rows = 1; a.cols = 1; a.data = {s.max};
    a.meta["describe"] = Json{{"shape", Json::array({1})}, {"axes", Json::array()}, {"frames", s.frames}, {"measure", "maximum_amplitude"}, {"value_type", "scalar"},
                              {"description", "the largest absolute value of the activation map over the window's frames: the unit amplitude of the optical flow"}};
    return ng::Arrays{{"maximum", std::move(a)}};
  };
  return n;
}

// ── flow.velocity ────────────────────────────────────────────────────────────────────────────────────────────────────
struct VelocityPrepared {
  nxr::signal::FlowMethodOptions fo;
  nxr::operators::RowMat3 P;
  nxr::operators::FaceMat F;
  double alpha = 0.01, scale = 0;
  std::int64_t N = 0;
};
struct VelocityState {
  std::shared_ptr<nxr::signal::FlowEngine> engine;
  Eigen::VectorXd prev;
  bool have_prev = false;
};

ng::NodeSpec velocity() {
  ng::NodeSpec n;
  n.name = "signal.flow_velocity";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"a", "the activation map of each frame (signal.flow_activation)", false}, {"surface", "the kernel's surface: its vertices and faces", false},
              {"scale", "the window's maximum amplitude: the statistic `flow_amplitude` stored (the unit amplitude of the solve)", false}};
  n.outputs = {{"v", "the velocity between a frame and the one before it, embedded 3 numbers a vertex (3N rows); the run's first frame has none: NaN", false}};
  n.parameters = {{"method", Json("pcg"), "pcg | direct (perscale is a set of spatial bands: not built here)", "-", {"pcg", "direct"}, false},
                  {"regulariser", Json("connection"), "connection (the connection Laplacian on the signpost gauge) | ambient (Lefevre and Baillet 2008)", "-", {"connection", "ambient"}, false},
                  {"datafit", Json("auto"), "auto | gauge | triangle", "-", {}, false},
                  {"alpha", Json(), "the smoothness weight; null: the regulariser's own (connection 0.01, ambient 1)", "-", {}, true},
                  {"flow_tolerance", Json(1e-8), "relative residual at which the iterative solver stops", "-", {}, false},
                  {"max_iterations", Json(500), "most iterations of the iterative solver for one frame", "-", {}, false},
                  {"preconditioner", Json("ldlt"), "ldlt | ic", "-", {"ldlt", "ic"}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::stateful, ng::Precision::float64, ng::OutputPrecision::follows_input,
                             "the previous frame's map is the state: blocks must meet in time order; each solve is in double, the velocity stored as float32 for float32 input", "");
  n.matches = "Rheome opticalflow_scalar (Horn-Schunck on the manifold, frame pairs); compute nxr::signal::FlowEngine";
  n.description = "Horn-Schunck optical flow of the activation map between consecutive frames, the unit amplitude the stored window maximum";
  n.plan = [](const ng::Types& in, const Json& p) {
    const ng::PortType& a = need_port(in, "a", "signal.flow_velocity");
    if (a.axis != ng::Axis::frames || a.value != ng::ValueType::real) ng::refuse("signal.flow_velocity: a is the activation on frames (real)");
    for (const char* k : {"surface", "scale"}) if (!in.count(k)) ng::refuse(std::string("signal.flow_velocity: ") + k + " is not given");
    try {
      const auto r = nxr::signal::parseRegulariser(p["regulariser"].get<std::string>());
      nxr::signal::resolveDataFit(r, nxr::signal::parseDataFit(p["datafit"].get<std::string>()));
      nxr::signal::parseFlowMethod(p["method"].get<std::string>());
    } catch (const std::invalid_argument& e) { ng::refuse(std::string("signal.flow_velocity: ") + e.what()); }
    if (p["method"] == "perscale") ng::refuse("signal.flow_velocity: perscale is one flow per spatial band (a band set): not built here");
    if (p.contains("alpha") && p["alpha"].is_number() && !(p["alpha"].get<double>() > 0)) ng::refuse("signal.flow_velocity: alpha must be positive");
    ng::Planned out;
    ng::PortType v = a;
    v.value = ng::ValueType::real; v.rows = 0;
    out.out["v"] = v;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    VelocityPrepared v;
    const ng::Array& S = constant(c, "surface", "signal.flow_velocity");
    v.N = S.rows;
    v.P = Eigen::Map<const nxr::operators::RowMat3>(S.data.data(), S.rows, 3);
    const std::vector<std::int32_t> faces = S.meta.value("faces", std::vector<std::int32_t>());
    v.F = Eigen::Map<const nxr::operators::FaceMat>(faces.data(), static_cast<Eigen::Index>(faces.size() / 3), 3);
    const ng::Array& sc = constant(c, "scale", "signal.flow_velocity");
    if (sc.data.empty() || !(sc.data[0] > 0)) ng::refuse("signal.flow_velocity: the stored amplitude scale '" + sc.meta.value("name", std::string()) + "' holds no positive maximum (`run flow_amplitude` first)");
    v.scale = sc.data[0];
    v.fo.method = nxr::signal::parseFlowMethod(p["method"].get<std::string>());
    v.fo.regulariser = nxr::signal::parseRegulariser(p["regulariser"].get<std::string>());
    v.fo.dataFit = nxr::signal::resolveDataFit(v.fo.regulariser, nxr::signal::parseDataFit(p["datafit"].get<std::string>()));
    v.alpha = p.contains("alpha") && p["alpha"].is_number() ? p["alpha"].get<double>() : (v.fo.regulariser == nxr::signal::Regulariser::Connection ? 0.01 : 1.0);
    v.fo.tolerance = p["flow_tolerance"].get<double>(); v.fo.maxIterations = p["max_iterations"].get<int>();
    v.fo.preconditioner = p["preconditioner"] == "ic" ? nxr::signal::PcgOptions::Preconditioner::IncompleteCholesky : nxr::signal::PcgOptions::Preconditioner::Ldlt;
    return std::any(std::move(v));
  };
  n.start = [](const std::any& prepared) {
    const VelocityPrepared& v = std::any_cast<const VelocityPrepared&>(prepared);
    VelocityState s;
    s.engine = std::make_shared<nxr::signal::FlowEngine>(v.fo, v.P, v.F, nxr::signal::vertexNormals(v.P, v.F), v.alpha);
    s.engine->observe(Eigen::VectorXd::Constant(1, v.scale));                      // the window's unit amplitude: the stored maximum, never a second pass
    return std::any(std::move(s));
  };
  n.step = [](const std::any& prepared, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    const VelocityPrepared& pr = std::any_cast<const VelocityPrepared&>(prepared);
    VelocityState& s = std::any_cast<VelocityState&>(state);
    const ng::Block& b = in.at("a");
    if (b.rows != pr.N) ng::refuse("signal.flow_velocity: the map has " + std::to_string(b.rows) + " sources, the surface " + std::to_string(pr.N) + " vertices");
    const Eigen::MatrixXd A = own_real(b);
    Eigen::MatrixXd V = Eigen::MatrixXd::Constant(3 * pr.N, A.cols(), std::numeric_limits<double>::quiet_NaN());
    for (Eigen::Index t = 0; t < A.cols(); ++t) {
      const Eigen::VectorXd next = A.col(t);
      if (s.have_prev) V.col(t) = s.engine->step(s.prev, next).velocity[0];
      s.prev = next;
      s.have_prev = true;
    }
    return std::make_pair(ng::Blocks{{"v", block_of(V)}}, std::move(state));
  };
  return n;
}

// ── flow.hodge_modes · flow.series ───────────────────────────────────────────────────────────────────────────────────
nxr::operators::Basis basis_of(const ng::Array& H) {
  nxr::operators::Basis b;
  const std::int64_t K = H.rows, Vg = H.cols;
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
struct ModesPrepared {
  std::shared_ptr<nxr::operators::GluedCarry> carry;
  std::shared_ptr<nxr::operators::WeightsReader> reader;
  std::int64_t K = 0, N = 0;
  std::vector<int> kind;
  std::vector<double> mu;
};

ng::NodeSpec hodge_modes() {
  ng::NodeSpec n;
  n.name = "signal.flow_hodge_modes";
  n.kind = ng::Kind::chunk;
  n.inputs = {{"v", "the velocity of each frame (signal.flow_velocity): 3N rows", false}, {"package", "a Hodge-Dirac package of the kernel's surface (its eigenvectors, as the runner hands them over)", false}};
  n.outputs = {{"b", "the flow weights [K x frames]: an exact mode's divergence potential (sources and sinks), a coexact mode's stream function (rotation); NaN where the velocity is", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::follows_input, "each frame on its own: the carry to the glued surface and the fused read are linear; the weights are stored as float32 for float32 input");
  n.matches = "Rheome flow.apparent (Hodge-Dirac flow weights of the velocity, fused weights); compute nxr::operators::GluedCarry, WeightsReader";
  n.description = "the velocity carried to the package's glued surface and read in its modes: the flow weights of each frame";
  n.plan = [](const ng::Types& in, const Json&) {
    const ng::PortType& v = need_port(in, "v", "signal.flow_hodge_modes");
    if (v.axis != ng::Axis::frames) ng::refuse("signal.flow_hodge_modes: v is the velocity on frames");
    if (!in.count("package")) ng::refuse("signal.flow_hodge_modes: package is not given");
    ng::Planned out;
    ng::PortType b = v;
    b.rows = 0;
    out.out["b"] = b;
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json&, ng::Shared&) {
    const ng::Array& H = constant(c, "package", "signal.flow_hodge_modes");
    if (!H.meta.value("hodge", false)) ng::refuse("signal.flow_hodge_modes: package is not a Hodge-Dirac package (an operator of kind eigenvectors, <stem>_eigenvectors)");
    const nxr::operators::Basis basis = basis_of(H);
    if (basis.mu.size() < 2) ng::refuse("signal.flow_hodge_modes: at least 2 modes");
    const std::vector<double> pin = H.meta.at("input_positions").get<std::vector<double>>();
    const std::int64_t N = static_cast<std::int64_t>(pin.size() / 3);
    ModesPrepared m;
    m.carry = std::make_shared<nxr::operators::GluedCarry>(basis, Eigen::Map<const nxr::operators::RowMat3>(pin.data(), N, 3));
    m.reader = std::make_shared<nxr::operators::WeightsReader>(basis);
    m.K = basis.mu.size(); m.N = N; m.kind = basis.kind;
    m.mu.assign(basis.mu.data(), basis.mu.data() + basis.mu.size());
    return std::any(std::move(m));
  };
  n.chunk = [](const std::any& prepared, const ng::Blocks& in, std::int64_t, const ng::Place&) {
    const ModesPrepared& m = std::any_cast<const ModesPrepared&>(prepared);
    const Eigen::MatrixXd V = own_real(in.at("v"));
    if (V.rows() != 3 * m.N) ng::refuse("signal.flow_hodge_modes: the velocity has " + std::to_string(V.rows()) + " rows, the package's input surface 3 x " + std::to_string(m.N));
    Eigen::MatrixXd B = Eigen::MatrixXd::Constant(m.K, V.cols(), std::numeric_limits<double>::quiet_NaN());
    for (Eigen::Index t = 0; t < V.cols(); ++t) {
      if (!std::isfinite(V(0, t))) continue;                                       // the run's first frame has no velocity
      B.col(t) = m.reader->flow(m.carry->apply(V.col(t))).col(0);
    }
    return ng::Blocks{{"b", block_of(B)}};
  };
  return n;
}

struct SeriesPrepared { std::int64_t K = 0; double rate = 0; std::vector<int> kind; std::vector<double> mu; };
struct SeriesState { std::vector<std::vector<float>> cols; std::int64_t first = -1, last = -1; };

ng::NodeSpec series() {
  ng::NodeSpec n;
  n.name = "signal.flow_series";
  n.kind = ng::Kind::accumulate;
  n.inputs = {{"b", "the flow weights of each frame (signal.flow_hodge_modes)", false}};
  n.outputs = {{"coefficients", "the flow weights [K x steps]: the series on the package's modes", false}, {"time", "the line the steps lie on: a step sits between two frames", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "materialises each frame's weights in time order (a series, no reduction)");
  n.matches = "Rheome flow.apparent (flow weights per frame)";
  n.description = "the flow weights of the run's steps gathered as a series on the package's modes, with the line they lie on (the first frame, which has no velocity, is not a step)";
  n.plan = [](const ng::Types& in, const Json&) {
    const ng::PortType& b = need_port(in, "b", "signal.flow_series");
    if (b.axis != ng::Axis::frames || !b.sfreq) ng::refuse("signal.flow_series: b is the flow weights on frames (with the frame rate)");
    ng::Planned out;
    out.out["coefficients"] = constant_port("field");
    out.out["time"] = constant_port("manifold");
    return out;
  };
  n.prepare = [](const ng::Types& in, const ng::Arrays&, const Json&, ng::Shared&) {
    SeriesPrepared s;
    s.rate = *in.at("b").sfreq;
    return std::any(std::move(s));
  };
  n.start = [](const std::any&) { return std::any(SeriesState{}); };
  n.step = [](const std::any&, std::any state, const ng::Blocks& in, std::int64_t, const ng::Place& at) {
    SeriesState& s = std::any_cast<SeriesState&>(state);
    const ng::Block& b = in.at("b");
    if (s.cols.empty()) s.cols.assign(static_cast<std::size_t>(b.rows), {});
    for (std::int64_t t = 0; t < b.own(); ++t) {
      const float probe = b.re[static_cast<std::size_t>(b.left + t)];
      if (!std::isfinite(probe)) continue;                                          // the run's first frame
      if (s.first < 0) s.first = at.first + t;
      s.last = at.first + t;
      for (std::int64_t r = 0; r < b.rows; ++r) s.cols[static_cast<std::size_t>(r)].push_back(b.re[static_cast<std::size_t>(r * b.width + b.left + t)]);
    }
    return std::make_pair(ng::Blocks{}, std::move(state));
  };
  n.finish = [](const std::any& prepared, std::any state) {
    const SeriesPrepared& pr = std::any_cast<const SeriesPrepared&>(prepared);
    const SeriesState& s = std::any_cast<const SeriesState&>(state);
    if (s.first < 0 || s.last - s.first + 1 < 7) ng::refuse("signal.flow_series: the window holds fewer than 8 frames at the member's rate (" + std::to_string(pr.rate) + " Hz); the flow needs at least 8");
    const std::int64_t K = static_cast<std::int64_t>(s.cols.size()), steps = static_cast<std::int64_t>(s.cols[0].size());
    ng::Array a;
    a.rows = K; a.cols = steps;
    a.data.resize(static_cast<std::size_t>(K * steps));
    for (std::int64_t k = 0; k < K; ++k) for (std::int64_t t = 0; t < steps; ++t) a.data[static_cast<std::size_t>(k * steps + t)] = s.cols[static_cast<std::size_t>(k)][static_cast<std::size_t>(t)];
    a.meta["describe"] = Json{{"shape", Json::array({K, steps})}, {"value_type", "scalar"}, {"first_frame", s.first}, {"steps", steps}, {"rate_hz", pr.rate},
                              {"description", "the Hodge-Dirac flow weights (exact modes: divergence potential, sources and sinks; coexact: stream function, rotation) per step"}};
    // the line: a step sits between two frames, so its time is (frame - 1/2) / rate from the recording's first sample; the definition adds the recording's origin
    ng::Array line;
    line.rows = 1; line.cols = 1; line.data = {0.0};
    line.meta["describe"] = Json{{"count", steps}, {"step", 1.0 / pr.rate}, {"start", (static_cast<double>(s.first) - 0.5) / pr.rate}, {"unit", "s"}, {"dimension", "time"}, {"folder", "time"}, {"cs", "clock"}};
    return ng::Arrays{{"coefficients", std::move(a)}, {"time", std::move(line)}};
  };
  return n;
}

}  // namespace

void register_flow_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {member_analytic(), activation(), max_amplitude(), velocity(), hodge_modes(), series()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
