// sourcepsd compute --resolve bins (spec docs/specs/2026-11-16-meg-fingerprinting.md, 3.1) on a small synthetic dataset (four sensors, three sources, a free kernel; one sinusoid plus noise):
// the bins of each band are stored as a statistic `<stem>_<band>_bins` (vertex x frequency); the band statistic is their mean, to rounding, and equals a `--resolve band` run's (the sensor-side
// accumulation) to rounding; the bins equal Welch on K x directly, over a stated time range; the result does not depend on the chunk size; the band run's cache key is the one from before the
// option; the stored precision follows the recording's (float32 in, float32 out; float64 in, float64 out); refusals.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>

#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io/measures.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/welch.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

static std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
static double rel(const MatrixXd& a, const MatrixXd& b) { return (a - b).cwiseAbs().maxCoeff() / b.cwiseAbs().maxCoeff(); }

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_sourcepsd_bins_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  const std::int64_t M = 4, N = 3, T = 2400;
  const double sfreq = 100.0;
  MatrixXd X(M, T), K(3 * N, M);
  for (auto& v : X.reshaped()) v = static_cast<float>(nd(rng));
  for (int t = 0; t < T; ++t) X.col(t) += Eigen::VectorXd::Constant(M, 3.0 * std::sin(2 * M_PI * 10.0 * t / sfreq));
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  for (auto& v : K.reshaped()) v = static_cast<float>(nd(rng));
  K = K.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M * 3));
      for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 0, 0}, {"MEG MAG"}, {}}, {"quality", {0, 0, 0, 0}, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = lattice_counts({T});
      if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = T; d.spacing = 1.0 / sfreq; d.origin = 0.0; d.unit = "s";
      tg.dimensions.push_back(d);
      tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      std::vector<float> xf(static_cast<std::size_t>(M * T));
      std::vector<double> xd(static_cast<std::size_t>(M * T));
      for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) { xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t)); xd[static_cast<std::size_t>(i * T + t)] = X(i, t); }
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(700)), std::make_pair(std::string("rest_b"), std::int64_t(1777)), std::make_pair(std::string("rest_c"), std::int64_t(128))}) {
        FieldSpec rec;
        rec.name = name; rec.path = node_path(sub.db(), prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
      }
      {                                                         // the same samples stored in double
        FieldSpec rec;
        rec.name = "rest64"; rec.path = node_path(sub.db(), prod, "rest64"); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xd, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
      }
      const std::string sid = uuid7();
      std::vector<float> sp(static_cast<std::size_t>(N * 3));
      for (auto& p : sp) p = static_cast<float>(nd(rng));
      ManifoldSpec sm;
      sm.id = sid; sm.name = "cortex"; sm.path = "source/cortex"; sm.type = "points"; sm.n_vertices = N;
      sm.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec sg; sg.form = "stored"; sg.cs_key = "scs"; sg.positions_path = "source/cortex"; sm.geometry = sg;
      sm.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("source/cortex", view(sp, {N, 3}), {}, std::nullopt, "manifold", sid); };
      surf = sub.manifold(std::move(sm));
      std::vector<std::int32_t> chans = {0, 1, 2, 3};
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      std::vector<float> kf(static_cast<std::size_t>(3 * N * M));
      for (int i = 0; i < 3 * N; ++i) for (int j = 0; j < M; ++j) kf[static_cast<std::size_t>(i * M + j)] = static_cast<float>(K(i, j));
      OperatorSpec op;
      op.name = "kern"; op.path = "kern"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
      op.data = view(kf, {3 * N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
      sub.operator_(std::move(op));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  SourcePsdOptions o;
  o.recording = "rest";
  o.kernel = "kern";
  o.chunks = "all";
  o.option_set = "brainstorm";                                  // Brainstorm's exact 4 s: 400 samples, 0.25 Hz bins
  o.segment_seconds = 4.0;
  o.bands = {{"alpha", 8, 12}, {"beta", 15, 29}};
  auto wo = nxr::welch::brainstormPsdOptions(sfreq, 4.0);
  const double df = sfreq / wo.segmentSamples;

  // the band run first (the sensor-side accumulation), under its own name
  SourcePsdOptions ob = o;
  ob.name = "pa";
  const SourcePsdResult band = sourcepsd_compute(ds, sub, ob);
  CHECK(!band.hit && band.bins.empty());

  // --resolve bins
  SourcePsdOptions ox = o;
  ox.name = "pb";
  ox.resolve = "bins";
  const SourcePsdResult r = sourcepsd_compute(ds, sub, ox);
  CHECK(!r.hit && r.bins.size() == 2 && r.bins[0].rows() == N && r.bins[0].cols() == 17 && r.bins[1].cols() == 57);   // alpha 8-12 Hz at 0.25 Hz: 17 bins; beta 15-29: 57
  {
    const StoredStatistic sb = read_statistic(sub, "pb_alpha_bins");
    CHECK(text(sb.row, "measure") == "power_density" && text(sb.row, "method") == "welch" && sb.shape == (std::vector<std::int64_t>{N, 17}));
    CHECK(sb.axes.size() == 2 && sb.axes[0]["manifold_id"] == surf && sb.axes[1]["name"] == "frequency" && sb.axes[1]["values"].size() == 17 && sb.axes[1]["bins"] == Json::array({32, 48}));
    CHECK(std::abs(sb.axes[1]["values"][0].get<double>() - 32 * df) < 1e-12 && std::abs(sb.axes[1]["values"][16].get<double>() - 48 * df) < 1e-12);
    CHECK(sb.options["resolve"] == "bins" && sb.producer["resolve"] == "bins" && !is_null(sb.row, "partition_selection_id") && integer(sb.row, "n_members") > 0);
    // the stored bins are the result's (float32 storage)
    double worst = 0;
    for (int i = 0; i < N; ++i) for (int k = 0; k < 17; ++k) worst = std::max(worst, std::abs(sb.values[static_cast<std::size_t>(i * 17 + k)] / r.bins[0](i, k) - 1));
    CHECK(worst < 1e-6);
    CHECK(ArrayReader::open(sub.store(), text(sb.row, "path")).dtype() == nxr::io::DType::Float32);   // the recording is float32
  }
  // the bins' mean is the band statistic (stored and in the result) exactly, and the band run's to rounding
  {
    MatrixXd mean(N, 2);
    for (int b = 0; b < 2; ++b) mean.col(b) = r.bins[static_cast<std::size_t>(b)].rowwise().mean();
    const StoredStatistic st = read_statistic(sub, "pb"), sa = read_statistic(sub, "pa");
    double stored = 0, vs_band = 0;
    for (int i = 0; i < N; ++i) for (int b = 0; b < 2; ++b) {
      stored = std::max(stored, std::abs(st.values[static_cast<std::size_t>(i * 2 + b)] / mean(i, b) - 1));
      vs_band = std::max(vs_band, std::abs(st.values[static_cast<std::size_t>(i * 2 + b)] / sa.values[static_cast<std::size_t>(i * 2 + b)] - 1));
    }
    std::printf("  bins: mean over the bins vs the band statistic %.2e (float32); vs a band-resolution run %.2e; maps vs that run %.2e\n", stored, vs_band, rel(r.maps, band.maps));
    CHECK(rel(r.maps, mean) == 0.0);                            // the map IS the mean of the bins
    CHECK(stored < 1e-6 && vs_band < 1e-6 && rel(r.maps, band.maps) < 1e-10);
  }
  // the bins equal Welch on K x directly
  {
    const auto direct = nxr::welch::welch(K * X, wo);
    double worst = 0;
    const std::size_t first[2] = {32, 60}, last[2] = {48, 116};
    for (int b = 0; b < 2; ++b) {
      CHECK(r.bins[static_cast<std::size_t>(b)].cols() == static_cast<Eigen::Index>(last[b] - first[b] + 1));
      MatrixXd want(N, last[b] - first[b] + 1);
      for (int s = 0; s < N; ++s) want.row(s) = direct.psd.block(3 * s, first[b], 1, last[b] - first[b] + 1) + direct.psd.block(3 * s + 1, first[b], 1, last[b] - first[b] + 1) + direct.psd.block(3 * s + 2, first[b], 1, last[b] - first[b] + 1);
      worst = std::max(worst, rel(r.bins[static_cast<std::size_t>(b)], want));
    }
    std::printf("  bins vs Welch on K x: %.2e\n", worst);
    CHECK(worst < 1e-10);
  }
  // a stated time range: 4 s to 20 s is samples 400 .. 2000
  {
    SourcePsdOptions oy = ox;
    oy.name = "pr"; oy.chunks = std::nullopt; oy.start = 4.0; oy.stop = 20.0;
    const SourcePsdResult t = sourcepsd_compute(ds, sub, oy);
    const auto direct = nxr::welch::welch(K * X.middleCols(400, 1600), wo);
    MatrixXd want(N, 17);
    for (int s = 0; s < N; ++s) want.row(s) = direct.psd.block(3 * s, 32, 1, 17) + direct.psd.block(3 * s + 1, 32, 1, 17) + direct.psd.block(3 * s + 2, 32, 1, 17);
    std::printf("  bins over 4-20 s vs Welch on that range: %.2e\n", rel(t.bins[0], want));
    CHECK(rel(t.bins[0], want) < 1e-10);
  }
  // the result does not depend on the chunk size (stored 700, 1777, 128 wide; the whole span read)
  {
    for (const char* other : {"rest_b", "rest_c"}) {
      SourcePsdOptions oz = ox;
      oz.recording = other; oz.name = std::string("pb_") + other;
      const SourcePsdResult z = sourcepsd_compute(ds, sub, oz);
      double w = 0;
      for (int b = 0; b < 2; ++b) w = std::max(w, rel(z.bins[static_cast<std::size_t>(b)], r.bins[static_cast<std::size_t>(b)]));
      std::printf("  bins, chunk width 700 vs %s: %.2e\n", other, w);
      CHECK(w < 1e-12);
    }
  }
  // the band run's cache key is the one from before the option (no `resolve`); the bins run adds it. A re-run is a hit; the other resolution under the same name is refused
  {
    const Json kb = read_statistic(sub, "pa").producer["cache"]["params"], kx = read_statistic(sub, "pb").producer["cache"]["params"];
    std::vector<std::string> names;
    for (auto it = kb.begin(); it != kb.end(); ++it) names.push_back(it.key());
    std::sort(names.begin(), names.end());
    CHECK((names == std::vector<std::string>{"bad_segments", "bands", "detrend", "name", "normalisation", "option_set", "overlap", "periodic", "scaling", "segment_samples", "window", "window_samples"}));
    CHECK(!kb.contains("resolve") && kx.contains("resolve") && kx["resolve"] == "bins");
    CHECK(sourcepsd_compute(ds, sub, ox).hit && sourcepsd_compute(ds, sub, ob).hit);
    SourcePsdOptions swap = ob;
    swap.name = "pb";
    CHECK(code_of([&] { sourcepsd_compute(ds, sub, swap); }) != "");
    swap.replace = true;
    CHECK(!sourcepsd_compute(ds, sub, swap).hit);               // replaced by a band run: the bins statistics are moved aside with it
  }
  // storage precision follows the input: a float64 recording gives float64 bins
  {
    SourcePsdOptions o64 = ox;
    o64.recording = "rest64"; o64.name = "p64";
    const SourcePsdResult z = sourcepsd_compute(ds, sub, o64);
    const StoredStatistic s64 = read_statistic(sub, "p64_alpha_bins");
    CHECK(ArrayReader::open(sub.store(), text(s64.row, "path")).dtype() == nxr::io::DType::Float64);
    double worst = 0;
    for (int i = 0; i < N; ++i) for (int k = 0; k < 17; ++k) worst = std::max(worst, std::abs(s64.values[static_cast<std::size_t>(i * 17 + k)] / z.bins[0](i, k) - 1));
    CHECK(worst == 0.0);
  }
  // refusals: no kernel named; a normalisation; a bad resolution; sourcetensor takes no resolution; the span is always the caller's
  {
    SourcePsdOptions bad = ox;
    bad.name = "x1"; bad.kernel = std::nullopt;
    CHECK(code_of([&] { sourcepsd_compute(ds, sub, bad); }) == code::invalid_argument);
    bad = ox; bad.name = "x2"; bad.normalisation = "relative";
    CHECK(code_of([&] { sourcepsd_compute(ds, sub, bad); }) == code::invalid_argument);
    bad = ox; bad.name = "x3"; bad.resolve = "fine";
    CHECK(code_of([&] { sourcepsd_compute(ds, sub, bad); }) == code::invalid_argument);
    bad = ox; bad.name = "x4"; bad.bands = {{"nowhere", 49.9, 49.95}};
    CHECK(code_of([&] { sourcepsd_compute(ds, sub, bad); }) == code::invalid_argument);
    bad = ox; bad.name = "x5"; bad.start = 0.0; bad.stop = 2.0; bad.chunks = std::nullopt;   // shorter than one segment
    CHECK(code_of([&] { sourcepsd_compute(ds, sub, bad); }) == code::invalid_argument);
    bad = ox; bad.name = "x6";
    CHECK(code_of([&] { sourcetensor_compute(ds, sub, bad); }) == code::invalid_argument);
  }
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_source_psd_bins");
}
