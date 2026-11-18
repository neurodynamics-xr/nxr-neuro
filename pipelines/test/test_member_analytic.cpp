// The node `signal.member_analytic` (pipelines/src/flow_steps.cpp): one filter-bank member's analytic signal read at the frames of a frame grid, through the graph's dispatch point, on a synthetic recording:
//   - its frames equal the compute function it wraps, wt(bank, member, window, instants), on the whole record (float32 result);
//   - the result does not depend on the chunk size: chunks of 1024 and 700 own samples, each with the halo the node declares (the member's taps), agree with the whole run to the tail the taps leave out;
//   - the frames are the grid's (a frame belongs to the chunk holding its start; the chunks' frames tile the grid) and the output is complex, on the kept channels, at the frame rate.
// Synthetic data only (compute/test/frames_wt_inputs.h).
#include <cmath>
#include <cstdio>

#include "check.h"
#include "frames_wt_inputs.h"
#include "nxr/graph/dispatch.h"
#include "nxr/graph/node.h"
#include "nxr/pipelines/flow_steps.h"
#include "nxr/signal/fourier.h"
#include "nxr/signal/timefilterbank.h"
#include "nxr/signal/wavelet.h"

namespace ng = nxr::graph;
namespace tb = nxr::signal::timefilterbank;
using ng::Json;

int main() {
  nxr::pipelines::graph::register_flow_nodes();
  const ng::NodeSpec& node = ng::registry().at("signal.member_analytic");
  CHECK(node.traits.declared() && node.frame_place);
  const std::int64_t N = fwi::kN;
  const std::int64_t count = static_cast<std::int64_t>(std::ceil(static_cast<double>(N) / fwi::kHop - 1e-9));
  const std::vector<std::int32_t> kept = {2, 0};                      // the kernel's channels, in its order

  Json p = Json::object();
  for (const auto& ps : node.parameters) p[ps.name] = ps.def;
  p["family"] = "morlet"; p["gain"] = 2.0; p["freq"] = 10.0;
  const Json record = Json{{"grid", Json{{"kind", "time"}, {"origin", 0}, {"hop", 3}, {"hop_exact", fwi::kHop}, {"count", count}, {"rate", fwi::kFs}}}};

  ng::Types in;
  ng::PortType x; x.cls = ng::PortClass::field; x.value = ng::ValueType::real; x.axis = ng::Axis::samples; x.sfreq = fwi::kFs;
  in["x"] = x;
  ng::PortType k; k.cls = ng::PortClass::operator_;
  in["channels_of"] = k;
  ng::PortType t; t.cls = ng::PortClass::selection;
  t.facts["value"] = Json{{"row", Json{{"params_json", record.dump()}}}};
  in["tiling"] = t;
  const ng::Planned planned = node.plan(in, p);
  const ng::PortType& z = planned.out.at("z");
  CHECK(z.axis == ng::Axis::frames && z.value == ng::ValueType::complex);
  CHECK(z.sfreq && std::fabs(*z.sfreq - fwi::kFs / fwi::kHop) < 1e-9);
  const std::int64_t H = planned.halo;
  std::printf("  halo %lld samples, frame rate %.4f Hz\n", static_cast<long long>(H), *z.sfreq);
  CHECK(H > 0);

  ng::Arrays constants;
  constants["channels_of"].meta["channels"] = kept;
  constants["tiling"].meta["params"] = record;
  ng::Shared shared;
  ng::NodeCall prep = ng::NodeCall::prepare(in, constants, p, shared);
  ng::dispatch(node, prep);

  const Eigen::MatrixXd rec = fwi::recording();
  const std::int64_t C = static_cast<std::int64_t>(kept.size()), R = fwi::kC;
  auto run = [&](std::int64_t own) {                       // the frames of every chunk of `own` samples, with the declared halo, gathered
    std::vector<std::complex<double>> all(static_cast<std::size_t>(C * count));
    std::int64_t frames_seen = 0;
    for (std::int64_t first = 0; first < N; first += own) {
      const std::int64_t n = std::min(own, N - first), left = std::min(H, first), right = std::min(H, N - first - n), width = left + n + right;
      ng::Block b; b.rows = R; b.width = width; b.left = left; b.right = right; b.re.resize(static_cast<std::size_t>(R * width));
      for (std::int64_t r = 0; r < R; ++r) for (std::int64_t c = 0; c < width; ++c) b.re[static_cast<std::size_t>(r * width + c)] = static_cast<float>(rec(r, first - left + c));
      const ng::Blocks bi{{"x", b}};
      const ng::Place place{first, n, nullptr};
      ng::NodeCall call = ng::NodeCall::chunk(prep.made, bi, 0, place);
      ng::dispatch(node, call);
      const ng::Block& y = call.made_blocks.at("z");
      CHECK(y.complex() && y.rows == C);
      ng::NodeCall fp = ng::NodeCall::frame_place(prep.made, place);
      ng::dispatch(node, fp);
      CHECK(fp.made_frames.n == y.width);
      for (std::int64_t r = 0; r < C; ++r) for (std::int64_t j = 0; j < y.width; ++j)
        all[static_cast<std::size_t>(r * count + fp.made_frames.first + j)] = {y.re[static_cast<std::size_t>(r * y.width + j)], y.im[static_cast<std::size_t>(r * y.width + j)]};
      frames_seen += y.width;
    }
    CHECK(frames_seen == count);
    return all;
  };
  const auto whole = run(N), a = run(1024), b2 = run(700);

  // the compute function it wraps, on the whole record, the bank designed as the node does
  tb::Design d(tb::Family::Morlet, 2.0, tb::Sides::Analytic); d.centres = {10.0}; d.cycles = 6.0;
  const tb::TimeFilterbank bank = tb::design(d, nxr::signal::next_fast_len(static_cast<std::size_t>(std::ceil(fwi::kFs / 0.5)), true), fwi::kFs);
  Eigen::VectorXd at(count); for (std::int64_t j = 0; j < count; ++j) at(j) = static_cast<double>(j) * fwi::kHop + static_cast<double>(H);
  Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(C, N + 2 * H);                     // zero outside the record, the member's reach each side
  for (std::int64_t r = 0; r < C; ++r) padded.row(r).segment(H, N) = rec.row(kept[static_cast<std::size_t>(r)]).cast<float>().cast<double>();
  const Eigen::MatrixXcd direct = nxr::signal::wt(bank, 0, padded, at);
  double e0 = 0, e1 = 0, e2 = 0, peak = 0;
  for (std::int64_t r = 0; r < C; ++r) for (std::int64_t j = 0; j < count; ++j) {
    const auto w = whole[static_cast<std::size_t>(r * count + j)];
    e0 = std::max(e0, std::abs(w - direct(r, j)));
    e1 = std::max(e1, std::abs(a[static_cast<std::size_t>(r * count + j)] - w));
    e2 = std::max(e2, std::abs(b2[static_cast<std::size_t>(r * count + j)] - w));
    peak = std::max(peak, std::abs(w));
  }
  std::printf("  whole run vs wt: %.2e; chunks of 1024 vs whole: %.2e; of 700: %.2e (peak %.3g)\n", e0, e1, e2, peak);
  CHECK(e0 < 1e-5 * peak);
  CHECK(e1 < 1e-3 * peak && e2 < 1e-3 * peak);          // the halo is the member's taps (1e-6 of the response's energy left out): chunked == whole to that tail
  return nxrtest::finish("member_analytic");
}
