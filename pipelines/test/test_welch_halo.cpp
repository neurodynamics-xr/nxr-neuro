// test_welch_halo: Welch's accumulating node (signal.psd), run through the executor block by block the way the runner feeds it (each block owns its columns and carries the right halo of n - 1
// samples, read past the span's end the node is not to read), gives the result of the whole record BIT FOR BIT, whatever the cut: chunks smaller than, equal to and larger than a segment (a segment
// over three chunks, a chunk holding no whole segment), a bad span inside a halo, a span that begins off the chunk grid and one that ends mid-segment. The reference is the compute function `psd` on
// the span's samples. A real recording and one complex series. Synthetic data only.
#include <cmath>
#include <cstdio>
#include <vector>

#include "nxr/graph/execution.h"
#include "nxr/graph/graph.h"
#include "nxr/pipelines/welch_steps.h"
#include "nxr/signal/spectral.h"

namespace g = nxr::graph;
namespace sp = nxr::signal::spectral;
using g::Json;

namespace {
int failures = 0, checked = 0;
#define CHECK(cond) do { ++checked; if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)

constexpr std::int64_t C = 3, N = 3001, kSeg = 256;
constexpr double kFs = 100.0;

Eigen::MatrixXd recording() {
  Eigen::MatrixXd x(C, N);
  for (std::int64_t c = 0; c < C; ++c) for (std::int64_t i = 0; i < N; ++i) {
    double v = 0; for (int k = 0; k < 7; ++k) v += std::cos(2 * M_PI * (1.1 + 3.7 * k + 0.6 * c) * i / kFs + 0.8 * k + c) / (1.0 + 0.3 * k);
    x(c, i) = static_cast<float>(v + 0.1 * std::sin(0.31 * i * (1 + c)));   // float32 values: what the loop's blocks hold
  }
  return x;
}
sp::WelchOptions options() {
  sp::WelchOptions o = sp::brainstormPsdOptions(kFs, 2.56);
  o.segmentSamples = kSeg;
  return o;
}
g::Graph psd_graph() {
  g::Graph gr; gr.name = "psd"; gr.inputs = {"x"}; gr.outputs = {"psd"};
  g::Use u; u.node = "signal.psd"; u.in = {{"x", "x"}}; u.out = {"psd"};
  u.with = Json{{"segment_rule", "round"}, {"segment_samples", kSeg}, {"overlap", 0.5}, {"window", "hamming"}, {"detrend", "constant"}, {"scaling", "density"}};
  gr.chunks = {u};
  return gr;
}
g::PortType stream_type(std::int64_t rows, bool complex) {
  g::PortType t; t.value = complex ? g::ValueType::complex : g::ValueType::real; t.rows = rows; t.axis = g::Axis::samples; t.sfreq = kFs; t.unit = "T"; return t;
}

// the runner's loop (io's haloed_loop): chunks of `W` on the recording's grid, the span's part of each, the halo read from the recording on each side (not clipped at the span), the flags with it
g::Array run(const g::Checked& c, const Eigen::MatrixXd& re, const Eigen::MatrixXd* im, std::int64_t s0, std::int64_t s1, std::int64_t W, const std::vector<std::uint8_t>& bad_flags) {
  g::Execution ex(c, {}, {}, {});
  const std::int64_t H = ex.halo();
  for (std::int64_t k = s0 / W; k * W < s1; ++k) {
    const std::int64_t own0 = std::max(s0, k * W), own1 = std::min(s1, (k + 1) * W), n = own1 - own0;
    const std::int64_t left = std::min(H, own0), right = std::min(H, N - own1);
    g::Block b; b.rows = re.rows(); b.width = left + n + right; b.left = left; b.right = right;
    b.re.resize(static_cast<std::size_t>(b.rows * b.width));
    if (im) b.im.resize(b.re.size());
    for (std::int64_t r = 0; r < b.rows; ++r) for (std::int64_t j = 0; j < b.width; ++j) {
      b.re[static_cast<std::size_t>(r * b.width + j)] = static_cast<float>(re(r, own0 - left + j));
      if (im) b.im[static_cast<std::size_t>(r * b.width + j)] = static_cast<float>((*im)(r, own0 - left + j));
    }
    std::vector<std::uint8_t> good(bad_flags.begin() + own0, bad_flags.begin() + own1), after(bad_flags.begin() + own1, bad_flags.begin() + own1 + right);
    g::Place place{own0, n, &good};
    place.origin = s0; place.end = s1; place.good_after = &after;
    ex.block({{"x", std::move(b)}}, place);
  }
  return ex.finish().at("psd");
}

void one(bool complex) {
  g::Types types{{"x", stream_type(complex ? 1 : C, complex)}};
  nxr::pipelines::graph::register_welch_nodes();
  const g::Checked c = g::check(psd_graph(), types);
  CHECK(c.halo == kSeg - 1);
  CHECK(c.nodes[c.loop[0]].planned.halo == kSeg - 1);
  Eigen::MatrixXd re = recording(), imm;
  Eigen::MatrixXcd z;
  if (complex) {   // one complex series
    imm = re.row(1); re = re.row(0).eval();
    z.resize(1, N); for (std::int64_t i = 0; i < N; ++i) z(0, i) = {re(0, i), imm(0, i)};
  }
  std::vector<std::uint8_t> good(N, 1);
  for (int i = 700; i < 760; ++i) good[i] = 0;
  for (int i = 1500; i < 1503; ++i) good[i] = 0;
  for (int i = 2490; i < 2510; ++i) good[i] = 0;            // inside the right halo of the chunks before it, and near a span's end
  const std::vector<std::uint8_t> clean(N, 1);
  struct Span { std::int64_t s0, s1; };
  for (const Span sp_ : {Span{0, N}, Span{0, 2500}, Span{333, N}, Span{700, 1900}}) {
    for (const bool masked : {false, true}) {
      const std::vector<std::uint8_t>& flags = masked ? good : clean;
      const std::int64_t len = sp_.s1 - sp_.s0;
      std::vector<std::uint8_t> sl(flags.begin() + sp_.s0, flags.begin() + sp_.s1);
      Eigen::MatrixXd want;
      std::int64_t segs = 0;
      if (complex) { const auto r = sp::psd(Eigen::MatrixXcd(z.middleCols(sp_.s0, len)), options(), &sl); want = r.psd; segs = r.segments; }
      else { const auto r = sp::psd(Eigen::MatrixXd(re.middleCols(sp_.s0, len)), options(), &sl); want = r.psd; segs = r.segments; }
      CHECK(segs > 0);
      for (const std::int64_t W : std::vector<std::int64_t>{1, 100, 255, 256, 257, 512, 1000, N, 5000}) {
        if (W == 1 && sp_.s1 - sp_.s0 > 1500) continue;   // one-sample blocks: the same rule, spared the time on the long spans
        const g::Array got = run(c, re, complex ? &imm : nullptr, sp_.s0, sp_.s1, W, flags);
        bool same = got.data.size() == static_cast<std::size_t>(want.size());
        for (std::int64_t r = 0; same && r < want.rows(); ++r) for (std::int64_t k = 0; same && k < want.cols(); ++k) same = got.data[static_cast<std::size_t>(r * want.cols() + k)] == want(r, k);
        if (!same) std::printf("  differs: %s span [%lld, %lld) %s chunk %lld\n", complex ? "complex" : "real", (long long)sp_.s0, (long long)sp_.s1, masked ? "masked" : "clean", (long long)W);
        CHECK(same);
        CHECK(got.meta["describe"]["n_members"].get<std::int64_t>() == segs);
      }
    }
  }
}
}  // namespace

int main() {
  one(false);
  one(true);
  std::printf("%d checks, %d failed\n%s\n", checked, failures, failures ? "FAILED" : "all passed");
  return failures ? 1 : 0;
}
