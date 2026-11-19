// The frames step through wt (`signal.analytic_frames`, pipelines/src/registry.cpp) run through the dispatch point on a synthetic recording:
//   - its frames equal the compute function it wraps, wt(bank, member, window, instants) of the whole record (float32 result);
//   - the result does not depend on the chunk size: the same frames from chunks of 1024 and 700 own samples, each with the halo the step declares, agree with the whole run to the float32 storage (the halo is the band's impulse response down to 1e-12 of its L1 norm);
//   - a frame belongs to one chunk (the frames of the chunks tile the whole run's), and the output is complex, its columns the frames.
// Synthetic data only (compute/test/frames_wt_inputs.h).
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>

#include "check.h"
#include "frames_wt_inputs.h"
#include "nxr/io/subject_package.h"
#include "nxr/pipelines/dispatch.h"
#include "nxr/pipelines/registry.h"
#include "nxr/signal/wavelet.h"

namespace g = nxr::pipelines::graph;
using nxr::io::Json;

namespace {
struct Frames { std::vector<std::complex<double>> v; std::int64_t first = 0; };
}  // namespace

int main() {
  const std::filesystem::path root = std::filesystem::temp_directory_path() / ("nxr_pipelines_test_analytic_frames_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  nxr::io::CreatePackageOptions opt;
  opt.home_dataset = "demo";
  nxr::io::DatasetWriter pkg = nxr::io::create_package(root / "S.nxr.zarr", "S", opt);
  nxr::io::SubjectWriter sub = nxr::io::SubjectWriter::open(pkg, "S");
  g::Bound db(sub, &pkg);

  const g::StepSpec& node = g::step("signal.analytic_frames");
  CHECK(node.traits.declared() && node.chunk_at && node.frame_place && node.halo);
  const std::int64_t N = fwi::kN, C = fwi::kC;
  const std::int64_t count = static_cast<std::int64_t>(std::ceil(static_cast<double>(N) / fwi::kHop - 1e-9));
  g::Values ins;
  g::Value x; x.cls = "field"; x.manifolds = {"s", "t"}; x.time = "t"; x.rate = fwi::kFs; x.origin = 0; x.chunked = true;
  ins["x"] = x;
  g::Value t; t.cls = "selection"; t.manifolds = {"t"};
  ins["tiling"] = t;
  g::Arrays made;
  made["tiling"].meta = Json{{"describe", Json{{"params", Json{{"grid", Json{{"kind", "time"}, {"origin", 0}, {"hop", 3}, {"hop_exact", fwi::kHop}, {"count", count}, {"rate", fwi::kFs}}}}}}}};
  g::ExecContext ctx{db, [](const Json&) {}, {}, &made};
  Json p = node.parameters;
  p["band"] = Json::array({8.0, 12.0});
  g::NodeCall prep = g::NodeCall::prepare(ins, p, ctx);
  g::dispatch(node, prep);
  const std::int64_t H = node.halo(prep.made_state);
  std::printf("  halo %lld samples\n", static_cast<long long>(H));
  CHECK(H > 0);

  const Eigen::MatrixXd rec = fwi::recording();
  auto run = [&](std::int64_t own) {                       // the frames of every chunk of `own` samples, with the declared halo, gathered
    std::vector<std::complex<double>> all(static_cast<std::size_t>(C * count));
    std::int64_t frames_seen = 0;
    for (std::int64_t first = 0; first < N; first += own) {
      const std::int64_t n = std::min(own, N - first), left = std::min(H, first), right = std::min(H, N - first - n), width = left + n + right;
      g::Block b; b.rows = C; b.width = width; b.left = left; b.right = right; b.re.resize(static_cast<std::size_t>(C * width));
      for (std::int64_t r = 0; r < C; ++r) for (std::int64_t k = 0; k < width; ++k) b.re[static_cast<std::size_t>(r * width + k)] = static_cast<float>(rec(r, first - left + k));
      const g::Blocks bi{{"x", b}};
      const g::Place place{first, n, nullptr};
      g::NodeCall call = g::NodeCall::chunk(prep.made_state, bi, 0, place);
      g::dispatch(node, call);
      const g::Block& y = call.made_blocks.at("y");
      CHECK(y.complex() && y.rows == C);
      const g::FramePlace fp = node.frame_place(prep.made_state, place);
      CHECK(fp.n == y.width);
      for (std::int64_t r = 0; r < C; ++r) for (std::int64_t j = 0; j < y.width; ++j)
        all[static_cast<std::size_t>(r * count + fp.first + j)] = {y.re[static_cast<std::size_t>(r * y.width + j)], y.im[static_cast<std::size_t>(r * y.width + j)]};
      frames_seen += y.width;
    }
    CHECK(frames_seen == count);
    return all;
  };
  const auto whole = run(N), a = run(1024), b2 = run(700);
  // the compute function it wraps, on the whole record
  namespace tb = nxr::signal::timefilterbank;
  tb::Design d(tb::Family::Morlet, 2.0, tb::Sides::Analytic); d.bands = {{8.0, 12.0}}; d.edgeBins = tb::EdgeBins::Drop;
  Eigen::VectorXd at(count); for (std::int64_t j = 0; j < count; ++j) at(j) = static_cast<double>(j) * fwi::kHop + static_cast<double>(H);
  Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(C, N + 2 * H); padded.middleCols(H, N) = rec.cast<float>().cast<double>();   // zero outside the record, the reach of the band each side
  const Eigen::MatrixXcd direct = nxr::signal::wt(tb::timeBank(d, std::size_t(1) << 16, fwi::kFs), 0, padded, at);
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
  CHECK(e1 < 2e-6 * peak && e2 < 2e-6 * peak);          // the declared halo is the impulse response down to 1e-6 of its L1 norm: chunked == whole to the float32 storage
  std::filesystem::remove_all(root);
  return nxrtest::finish("analytic_frames_step");
}
