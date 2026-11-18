// THE LAG-FREE PAIRS SURROGATE (the definition `surrogate_lagfree_pairs`, was `wishart compute`; spec docs/specs/2026-11-14-vortex-cores-port.md, piece 10) on a synthetic 6-channel recording (never a real dataset): a narrow 8-12 Hz band process reaching the channels with NO lag
// in bursts 0 and 2, and with a lag of 6 samples a channel (24 ms: 87 degrees a step at 10 Hz) in bursts 1 and 3. Shown:
//   - the band's analytic windows are the COMPLEX measurement `sum` of `measure --set analytic_cross_spectrum_tiles` (a complex reduce of z z^H over the band's time grid); summed over each burst's tiles are the burst's second moment; the lagged share of the lagged bursts is far above the lag-free ones'
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
#include "nxr/io/testing/probe.h"
#include "nxr/io/array_reader.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/runner.h"
#include "make_grids.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

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
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  namespace g = nxr::pipelines::graph;
  nxrtest::run_psd(ds, sub, "rest", Json{{"segment_samples", 256}});
  const std::string windows = "rest time grid band 8-12 Hz";
  { g::RunOptions po; po.chunks = "all"; po.preset = "analytic_cross_spectrum_tiles";
    const auto rm = g::run(ds, sub, g::load("measure"), Json{{"recording", "rest"}, {"band", "8-12 Hz"}}, po);
    CHECK(rm.size() == 1 && rm[0]["hit"] == false); }
  // THE COMPLEX MEASUREMENT: a window's sum of z z^H is Hermitian (rows a*M+b, b*M+a conjugate), its diagonal real and positive, and the stored value type is complex
  {
    const CrossSpectrumWindows win = crossspectrum_windows_read(sub, windows);
    CHECK(win.level == 7 && win.channels.size() == static_cast<std::size_t>(M) && win.sum.size() >= 60);
    double herm = 0, diag_im = 0, big = 0;
    for (const auto& S : win.sum) { herm = std::max(herm, (S - S.adjoint()).cwiseAbs().maxCoeff()); diag_im = std::max(diag_im, S.diagonal().imag().cwiseAbs().maxCoeff()); big = std::max(big, S.cwiseAbs().maxCoeff()); for (int c = 0; c < M; ++c) CHECK(S(c, c).real() > 0); }
    std::printf("  windows: %zu whole tiles of 2^%d samples; Hermitian error %.2e, diagonal imaginary %.2e (of %.2f)\n", win.sum.size(), win.level, herm, diag_im, big);
    CHECK(herm < 1e-4 * big && diag_im < 1e-4 * big);
    // the stored array says what it is
    const auto sums = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"measure", V(std::string("sum"))}});
    CHECK(sums.size() == 1);
    ArrayReader rd = ArrayReader::open(sub.store(), text(sums[0], "array_path"));
    CHECK(rd.shape() == std::vector<std::int64_t>({M * M, rd.shape()[1], 2}));
  }
  // a complex field is reduced by the measures linear in its value; the others refuse with their reason
  {
    const auto why = what_of([&] { g::RunOptions po; po.chunks = "all"; po.preset = "analytic_cross_spectrum_tiles"; po.replace = true;
                                   g::run(ds, sub, g::load("measure"), Json{{"recording", "rest"}, {"band", "8-12 Hz"}, {"measures", Json::array({"max"})}}, po); });
    CHECK(why.find("complex") != std::string::npos && why.find("needs a real quantity") != std::string::npos);
  }
  Json set = Json{{"recording", "rest"}, {"windows", windows}, {"bursts", "bursts"}, {"psd", "rest_psd_welch"}, {"band", Json::array({8.0, 12.0})}, {"draws", 400}, {"seed", 3}};
  auto run_pairs = [&](Json st, bool replace) { g::RunOptions ro; ro.chunks = "all"; ro.replace = replace; return g::run(ds, sub, g::load("surrogate_lagfree_pairs"), st, ro); };
  const auto r = run_pairs(set, false);
  const std::string nm = "surrogate_lagfree_pairs.bursts";
  // the per-burst results are a MEASUREMENT of the recording over the bursts (5 measures); the run's, a statistic
  const std::string bursts_id = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = "bursts", .any_status = true})->id, rec_id = text(*find_field(sub, FieldQuery{"rest", std::nullopt, std::nullopt, true}), "id");
  struct PerBurst { std::vector<double> values; Row row; };                                  // values [burst x 5]: lagged_share, null_mean, p, q, dof
  auto read_pairs = [&]() {
    PerBurst out;
    std::vector<std::vector<double>> cols;
    for (const char* m : {"lagged_share", "lagfree_null_mean", "lagfree_p", "lagfree_q", "lagfree_dof"}) {
      const MeasurementValues mv = measurement_values(sub, bursts_id, rec_id, std::nullopt, std::nullopt, m);
      cols.push_back(mv.values);
    }
    for (std::size_t b = 0; b < cols[0].size(); ++b) for (std::size_t k = 0; k < 5; ++k) out.values.push_back(cols[k][b]);
    return out;
  };
  const PerBurst W = read_pairs();
  const StoredStatistic Wr = read_statistic(sub, nm + ".run");
  std::printf("  %zu run(s)\n", r.size());
  double mn_free = 0, mn_lag = 0;
  for (int b = 0; b < 4; ++b) {
    const double sh = W.values[static_cast<std::size_t>(5 * b)], nm_ = W.values[static_cast<std::size_t>(5 * b + 1)], p = W.values[static_cast<std::size_t>(5 * b + 2)], q = W.values[static_cast<std::size_t>(5 * b + 3)], dof = W.values[static_cast<std::size_t>(5 * b + 4)];
    std::printf("  burst %d (%s): lagged share %.3f, null mean %.3f, p %.4f, q %.4f, dof %.1f\n", b, bursts[static_cast<std::size_t>(b)].second ? "lagged" : "lag-free", sh, nm_, p, q, dof);
    (bursts[static_cast<std::size_t>(b)].second ? mn_lag : mn_free) += sh / 2;
  }
  CHECK(W.values.size() == 20 && Wr.shape == std::vector<std::int64_t>({4}));
  CHECK(mn_lag > 3 * mn_free);                                                    // the lagged bursts are far more in quadrature
  for (int b = 0; b < 4; ++b) {
    const double p = W.values[static_cast<std::size_t>(5 * b + 2)];
    if (bursts[static_cast<std::size_t>(b)].second) CHECK(p <= 0.01);                                              // at most 3 of 400 draws reach the lagged share
    else CHECK(p > 0.01);                                                          // a lag-free burst is not called at the 1 % level (fixed seeds)
  }
  CHECK(Wr.values[3] <= 3.0 / 401 + 1e-12 && Wr.values[0] > Wr.values[1] + 3 * Wr.values[2]);   // the run's mean share is beyond its null
  // the same seed is the same result; a rerun is a hit; another seed is another set of draws
  { const auto again = run_pairs(set, false); CHECK(again[0]["hit"] == true);
    Json other = set; other["seed"] = 99;
    run_pairs(other, true);
    const PerBurst W2 = read_pairs();
    CHECK(W2.values != W.values && W2.values[0] == W.values[0]);                  // the data's share is the same, the null's mean differs
    run_pairs(set, true);
    CHECK(read_pairs().values == W.values); }
  // refusals
  CHECK(!what_of([&] { Json b = set; b["recording"] = "no_such"; run_pairs(b, false); }).empty());
  CHECK(!what_of([&] { Json b = set; b["windows"] = "no_such"; run_pairs(b, false); }).empty());
  CHECK(!what_of([&] { Json b = set; b["bursts"] = "no_such"; run_pairs(b, false); }).empty());
  CHECK(what_of([&] { Json b = set; b["band"] = Json::array({12.0, 8.0}); run_pairs(b, false); }).find("lo < hi") != std::string::npos);
  CHECK(what_of([&] { Json b = set; b["draws"] = 5; run_pairs(b, false); }).find("at least 20") != std::string::npos);
  return nxrtest::finish("test_wishart_null") ? 1 : 0;
}
