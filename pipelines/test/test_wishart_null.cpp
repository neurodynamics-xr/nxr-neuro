// THE WISHART NULL (`wishart compute`, pipelines/include/nxr/pipelines/wishart_null.h; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10) on a synthetic 6-channel recording (never a real dataset): a narrow 8-12 Hz band process reaching the channels with NO lag
// in bursts 0 and 2, and with a lag of 6 samples a channel (24 ms: 87 degrees a step at 10 Hz) in bursts 1 and 3. Shown:
//   - the stored analytic windows summed over each burst's tiles are the burst's second moment; the lagged share of the lagged bursts is far above the lag-free ones'
//   - the Wishart null calls them: the lagged bursts' p is at most 0.01 (400 draws); the lag-free bursts' are not at that level; the run's mean share is beyond its null
//   - the effective bandwidth of the band (the PSD through the analytic gain) and the degrees of freedom (duration x bandwidth) are stored; a draw is a function of the seed (the same seed twice is the same result); a rerun is a hit
//   - the refusals: a recording, windows, bursts and PSD that do not exist, a band that is not LO < HI, too few draws
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "psd_run.h"
#include "check.h"
#include "mesh_helpers.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/constant_q.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/wishart_null.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_wishart_null_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const double sfreq = 250.0, tile_s = 256.0 / sfreq;
  const int M = 6, T = 8192, lag = 6;
  // the sources: 8-12 Hz sums of cosines of random phase (two independent sources, mixed without lag)
  std::mt19937 rng(13);
  std::normal_distribution<double> nd;
  std::uniform_real_distribution<double> ph(0, 2 * M_PI);
  auto source = [&](int k) {
    std::vector<double> f, p;
    for (double fr = 8.0; fr <= 12.0; fr += 0.1) { f.push_back(fr); p.push_back(ph(rng)); }
    return [f, p, k](double t) { double s = 0; for (std::size_t i = 0; i < f.size(); ++i) s += std::cos(2 * M_PI * f[i] * t + p[i]); return s / std::sqrt(static_cast<double>(f.size()) / 2); };
  };
  const auto s0 = source(0), s1 = source(1);
  std::vector<std::vector<double>> mix(M, std::vector<double>(2));
  for (auto& r : mix) { r[0] = nd(rng); r[1] = nd(rng); }
  const std::vector<std::pair<int, bool>> bursts = {{2, false}, {10, true}, {18, false}, {26, true}};            // first tile, lagged (4 tiles each)
  std::vector<float> x(static_cast<std::size_t>(M) * T);
  for (int c = 0; c < M; ++c)
    for (int t = 0; t < T; ++t) {
      const int tile = static_cast<int>((t * 1.0 / 256.0));
      bool in = false, lagged = false;
      for (const auto& [k, l] : bursts) if (tile >= k && tile < k + 4) { in = true; lagged = l; }
      const double tt = (t - (in && lagged ? lag * c : 0)) / sfreq;
      x[static_cast<std::size_t>(c) * T + t] = static_cast<float>((in ? 3.0 * (mix[c][0] * s0(tt) + mix[c][1] * s1(tt)) : 0.0) + 0.5 * nd(rng));
    }
  DatasetWriter ds = create_dataset(root, "demo");
  std::string tid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      const std::vector<std::int64_t> zeros(static_cast<std::size_t>(M), 0);
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", zeros, {"MEG MAG"}, {}}, {"quality", zeros, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(x, {M, T}); rec.array.chunks = Shape{M, 4096}; rec.array.compress = false; rec.cols.set("unit", "T");
      const std::string rid = sub.field(std::move(rec));
      std::vector<std::pair<double, double>> sp;
      nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "the bursts";
      std::vector<nxr::bst::TimeSpan> spans;
      for (const auto& [k, l] : bursts) { (void)l; spans.push_back({k * tile_s, (k + 4) * tile_s, 0}); }
      nxr::bst::time_spans(sub, "time/bursts", spans, {"burst"}, bo);
    });
    sub.compose([&] { nxr::io::constantq::write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::run_psd(ds, sub, "rest", Json{{"segment_level", 8}});
  { CrossSpectrumOptions co; co.recording = "rest"; co.chunks = "all"; co.bands = {{"alpha", 8.0, 12.0}}; co.estimator = "analytic"; co.level = 8; crossspectrum_compute(ds, sub, co); }
  WishartNullOptions wo;
  wo.recording = "rest"; wo.windows = "rest_crossspectrum_alpha_windows"; wo.bursts = "bursts"; wo.psd = "rest_psd_welch"; wo.band = {8.0, 12.0}; wo.draws = 400; wo.seed = 3;
  const auto r = wishart_null_compute(ds, sub, wo);
  const StoredStatistic W = read_statistic(sub, "wishart.rest_crossspectrum_alpha_windows.bursts"), Wr = read_statistic(sub, "wishart.rest_crossspectrum_alpha_windows.bursts.run");
  std::printf("  effective bandwidth %.2f Hz; %lld bursts used\n", r.info["effective_bandwidth_hz"].get<double>(), static_cast<long long>(r.info["bursts_used"].get<std::int64_t>()));
  double mn_free = 0, mn_lag = 0;
  for (int b = 0; b < 4; ++b) {
    const double sh = W.values[static_cast<std::size_t>(5 * b)], nm = W.values[static_cast<std::size_t>(5 * b + 1)], p = W.values[static_cast<std::size_t>(5 * b + 2)], q = W.values[static_cast<std::size_t>(5 * b + 3)], dof = W.values[static_cast<std::size_t>(5 * b + 4)];
    std::printf("  burst %d (%s): lagged share %.3f, null mean %.3f, p %.4f, q %.4f, dof %.1f\n", b, bursts[static_cast<std::size_t>(b)].second ? "lagged" : "lag-free", sh, nm, p, q, dof);
    (bursts[static_cast<std::size_t>(b)].second ? mn_lag : mn_free) += sh / 2;
  }
  CHECK(text(W.row, "method") == "wishart_null" && text(W.row, "measure") == "null_p" && W.shape == std::vector<std::int64_t>({4, 5}) && Wr.shape == std::vector<std::int64_t>({4}));
  CHECK(mn_lag > 3 * mn_free);                                                    // the lagged bursts are far more in quadrature
  for (int b = 0; b < 4; ++b) {
    const double p = W.values[static_cast<std::size_t>(5 * b + 2)];
    if (bursts[static_cast<std::size_t>(b)].second) CHECK(p <= 0.01);                                              // at most 3 of 400 draws reach the lagged share
    else CHECK(p > 0.01);                                                          // a lag-free burst is not called at the 1 % level (fixed seeds)
  }
  CHECK(Wr.values[3] <= 3.0 / 401 + 1e-12 && Wr.values[0] > Wr.values[1] + 3 * Wr.values[2]);   // the run's mean share is beyond its null
  // the same seed is the same result; a rerun is a hit; another seed is another set of draws
  { WishartNullOptions again = wo; CHECK(wishart_null_compute(ds, sub, again).hit);
    WishartNullOptions other = wo; other.seed = 99; other.replace = true;
    wishart_null_compute(ds, sub, other);
    const StoredStatistic W2 = read_statistic(sub, "wishart.rest_crossspectrum_alpha_windows.bursts");
    CHECK(W2.values != W.values && W2.values[0] == W.values[0]);                  // the data's share is the same, the null's mean differs
    WishartNullOptions back = wo; back.replace = true; wishart_null_compute(ds, sub, back);
    CHECK(read_statistic(sub, "wishart.rest_crossspectrum_alpha_windows.bursts").values == W.values); }
  // refusals
  CHECK(what_of([&] { WishartNullOptions b = wo; b.recording = "no_such"; wishart_null_compute(ds, sub, b); }).find("no_such") != std::string::npos);
  CHECK(!what_of([&] { WishartNullOptions b = wo; b.windows = "no_such"; wishart_null_compute(ds, sub, b); }).empty());
  CHECK(what_of([&] { WishartNullOptions b = wo; b.bursts = "no_such"; wishart_null_compute(ds, sub, b); }).find("no_such") != std::string::npos);
  CHECK(what_of([&] { WishartNullOptions b = wo; b.band = {12.0, 8.0}; wishart_null_compute(ds, sub, b); }).find("LO < HI") != std::string::npos);
  CHECK(what_of([&] { WishartNullOptions b = wo; b.draws = 5; wishart_null_compute(ds, sub, b); }).find("at least 20") != std::string::npos);
  return nxrtest::finish("test_wishart_null") ? 1 : 0;
}
