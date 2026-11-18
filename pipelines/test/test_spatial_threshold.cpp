// THE SPATIAL EVENTS' THRESHOLD by its false-alarm rate (spec docs/specs/2026-11-14-spatial-events.md, decision 2, corrected 2026-11-14), on a synthetic sphere (never a real dataset): an identity kernel, WHITE NOISE
// only (no plant), a window of "bursts". `run spatial_baseline`, `run spatial_threshold` (the relative power's distribution OUTSIDE the window), then `run spatial_events`:
//   - the stored histogram equals `nxr::stats::histogram` of the DIRECT outside-frame relative power (taps, Kaiser-sinc at the frames, the dense exact hat, ÷ the baseline), counts to 0.1 % of the total
//   - `threshold: percentile` (the default, 99): on pure noise the share of vertex-frames above the threshold inside the window (the home map's counts) is about 1 % (tolerance 0.5-1.6 %), and the derived threshold is in the provenance
//   - `threshold: ratio` 2: the same share is about 13.5 % (a single frame's relative power is about exponential; tolerance 9-18 %)
//   - the distribution and the events do not depend on the storage chunk (4096 against 1000) or the block of frames (100 against 256); a rerun is a hit
//   - refusals: a distribution of another reference level, a percentile outside (0, 100), percentile with no stored distribution
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <map>
#include <set>

#include "check.h"
#include "measurement_helpers.h"
#include "store_at.h"
#include "mesh_helpers.h"
#include "brain_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/yaml.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/stats/histogram.h"
#include "nxr/signal/filters_fir.h"
#include "nxr/pipelines/eigen.h"
#include "run_definition.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"

// the frames of the 8-12 Hz band: the definition `select_time_grid` rule frames gives 36 frames to a cycle of the band's centre (the geometric mean of its ends), exactly (not a power-of-two tile)
constexpr double kFramesPerCycle = 36.0;

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
std::vector<float> read_floats(const fs::path& p) {
  ArrayReader r = nxrtest::open_array(p);
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_spatial_threshold_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const double R = 0.1, sfreq = 250.0, w0 = 5.0, wm = 8.5, w1 = 12.0, fps = kFramesPerCycle * std::sqrt(8.0 * 12.0);   // the window 5-12 s is the "bursts"; 5 s + 4.4 s of noise outside it
  const int T = 4096;
  // a two-hemisphere "brain" (the cortex has a medial wall, so its hemispheres are glued by anatomy: the events live on the glued cortex)
  const Brain brn = brain(2, false);
  const std::vector<double>& U = brn.P;
  const std::vector<std::int32_t>& F0 = brn.F;
  const int N = static_cast<int>(U.size() / 3), M = N;
  const int vA = 0;                                                            // the narrow blob here; the wide one at the vertex farthest from it that is not its antipode's neighbour
  int vB = 0;
  { double best = 1e9;
    for (int v = 0; v < N; ++v) { const double c = (U[3 * v] * U[0] + U[3 * v + 1] * U[1] + U[3 * v + 2] * U[2]) / (R * R); if (std::abs(c - 0.0) < best) { best = std::abs(c); vB = v; } } }   // 90 degrees from A
  const double sA = 0.02, sB = 0.05, amp = 0.0;   // NO PLANT: white noise only, so the bursts hold nothing the outside lacks
  auto ang = [&](int a, int b) { const double c = (U[3 * a] * U[3 * b] + U[3 * a + 1] * U[3 * b + 1] + U[3 * a + 2] * U[3 * b + 2]) / (R * R); return R * std::acos(std::max(-1.0, std::min(1.0, c))); };
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  std::vector<float> x(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < N; ++v) {
    const double aA = amp * std::exp(-std::pow(ang(v, vA), 2) / (2 * sA * sA)), aB = amp * std::exp(-std::pow(ang(v, vB), 2) / (2 * sB * sB));
    for (int t = 0; t < T; ++t) {
      const double tt = t / sfreq;
      x[static_cast<std::size_t>(v) * T + t] = static_cast<float>(nd(rng) + (tt >= w0 && tt < wm ? aA * std::cos(2 * M_PI * 10.0 * tt) : 0.0) + (tt >= wm && tt < w1 ? aB * std::cos(2 * M_PI * 10.0 * tt) : 0.0));
    }
  }
  std::vector<float> ident(static_cast<std::size_t>(N) * M, 0.0f);
  for (int i = 0; i < N; ++i) ident[static_cast<std::size_t>(i) * M + i] = 1.0f;
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, first_rid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = brn.Sph;
      std::vector<std::int32_t> dk(static_cast<std::size_t>(brn.n), 1);
      for (int i = 0; i < brn.n; ++i) { const double y = U[static_cast<std::size_t>(3 * i + 1)]; if ((i < brn.nHalf && y < 0.7) || (i >= brn.nHalf && y > -0.7)) dk[static_cast<std::size_t>(i)] = 0; }
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      surf = add_surface(sub, so, U, F0).id;
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(4096)), std::make_pair(std::string("rest_b"), std::int64_t(1000))}) {
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(x, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "A");
        const std::string rid = sub.field(std::move(rec));
        if (first_rid.empty()) first_rid = rid;
      }
      nxr::bst::TimeSpansOptions wo; wo.line_id = tid; wo.of_field_id = first_rid; wo.description = "the bursts";
      nxr::bst::time_spans(sub, "time/window", {{w0, w1, 0}}, {"burst"}, wo);
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      OperatorSpec op;
      op.name = "ident"; op.path = "ident"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = N; op.n_cols = M;
      op.data = view(ident, {N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "scalar");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  // the line's one time grid is the frames of the band below (a definition finds `<recording> time grid`): no constant-Q grid beside it
  nxrtest::make_tiles(ds, sub);
  { EigenSolveOptions eo; eo.modes = 100; eo.surface = "cortex"; eigen_solve(ds, sub, eo, {}); }          // stores the glued surface the band is evaluated on
  for (const char* rec : {"rest", "rest_b"}) { nxrtest::make_frames(ds, sub, rec, "8-12 Hz"); }
  g::RunOptions all; all.chunks = "all"; all.replace = true;
  const Json lv = Json::array({1, 2, 3, 4});
  const int refl = 4;
  const Json scales = Json::array({"index:1", "index:2", "index:3", "index:4"});   // the hats as the diagnostic index chart (the definitions take quantities)
  const Json bset = Json{{"spatial_scales", scales}, {"transform", "exact"}, {"bursts", "window"}, {"glued_surface", "cortex_glued"}};
  const Json tset = Json{{"spatial_scales", scales}, {"reference_scale", "index:4"}, {"transform", "exact"}, {"bursts", "window"}, {"glued_surface", "cortex_glued"}};
  Json eset = tset;
  const g::Definition dbase = g::load("spatial_baseline"), dthr = g::load("spatial_threshold"), dev = g::load("spatial_events");
  g::run(ds, sub, dbase, bset, all);
  const auto rt = g::run(ds, sub, dthr, tset, all);
  CHECK(rt.size() == 2);
  const StoredStatistic Hs = read_statistic(sub, "spatial_threshold.histogram.rest"), Hb = read_statistic(sub, "spatial_threshold.histogram.rest_b");
  CHECK(text(Hs.row, "method") == "histogram" && text(Hs.row, "measure") == "count" && Hs.values.size() == 4096);

  // ── THE DIRECT COMPUTATION of the relative power outside the window at the reference level ──
  const SurfaceInput S = read_surface(sub, std::string("cortex"), true);
  SpatialBandOptions sbo; sbo.transform = "exact"; sbo.spatial_shape = "mexhat"; sbo.glued = true;
  const SpatialBand band = make_spatial_band(ds, sub, S, read_surface_level(sub, S, refl), sbo);
  const std::int64_t Vg = band.glued_vertices;
  const std::int64_t count = static_cast<std::int64_t>(std::floor(static_cast<double>(T - 1) * fps / sfreq)) + 1;
  std::vector<double> pos(static_cast<std::size_t>(count));
  for (std::int64_t j = 0; j < count; ++j) pos[static_cast<std::size_t>(j)] = static_cast<double>(j) * sfreq / fps;
  const Eigen::MatrixXcd Zc = nxrtest::analytic_at_frames(x, M, T, sfreq, 8.0, 12.0, pos);
  const Eigen::MatrixXd Zre = Zc.real(), Zim = Zc.imag();   // [M x count]
  Eigen::MatrixXd I = Eigen::MatrixXd::Zero(N, M);
  for (int i = 0; i < N; ++i) I(i, i) = 1.0;
  const Eigen::MatrixXd Mc = band.apply_glued(I);
  const Eigen::MatrixXd A = Mc * Zre, B = Mc * Zim;
  const StoredStatistic Base = read_statistic(sub, "spatial_baseline.baseline.rest");
  const std::int64_t L = static_cast<std::int64_t>(lv.size());
  std::vector<std::int64_t> outside;
  for (std::int64_t j = 0; j < count; ++j) { const double sec = std::floor(pos[static_cast<std::size_t>(j)]) / sfreq; if (!(sec >= w0 && sec < w1)) outside.push_back(j); }
  std::vector<float> rel(static_cast<std::size_t>(Vg) * outside.size());           // [Vg x outside frames], row-major
  for (std::int64_t v = 0; v < Vg; ++v)
    for (std::size_t k = 0; k < outside.size(); ++k)
      rel[static_cast<std::size_t>(v) * outside.size() + k] = static_cast<float>((A(v, outside[k]) * A(v, outside[k]) + B(v, outside[k]) * B(v, outside[k])) / Base.values[static_cast<std::size_t>(v * L + (refl - 1))]);
  
  CHECK(outside.size() > 3000);
  const std::size_t range = 3000;
  double mx = 0;
  for (std::int64_t v = 0; v < Vg; ++v) for (std::size_t k = 0; k < range; ++k) { const double r = static_cast<double>(rel[static_cast<std::size_t>(v) * outside.size() + k]); if (r > mx) mx = r; }

  const nxr::stats::histogram::LogBins lb = nxr::stats::histogram::logBinsFor(mx, 4096);
  // the stored range is fixed from the run's first 3000 good frames; the frames read in blocks (with a halo) differ from the whole-record direct value by ~1e-3 of the peak (test_frames_wt), most at the record's first
  // frames where the range's maximum lies: the direct histogram is binned over the STORED range, and the two ranges are compared to 3 %
  const nxr::stats::histogram::LogBins lbs{Hs.options["lo"].get<double>(), Hs.options["hi"].get<double>(), 4096};
  const nxr::stats::histogram::Counts dc = histogramTiles(rel.data(), static_cast<std::size_t>(Vg), outside.size(), outside.size(), 1, lbs);
  double l1 = 0, total = 0;
  std::vector<std::int64_t> dcounts(4096);
  for (int k = 0; k < 4096; ++k) { dcounts[static_cast<std::size_t>(k)] = dc.counts[static_cast<std::size_t>(k)]; l1 += std::abs(Hs.values[static_cast<std::size_t>(k)] - dcounts[static_cast<std::size_t>(k)]); total += dcounts[static_cast<std::size_t>(k)]; }
  const double p99 = nxr::stats::histogram::binnedPercentile(dcounts, lbs, 99.0);
  // the bins are 0.34 % wide and the blocks' values differ from the direct by ~1e-3: counts move between neighbouring bins, so the distributions are compared by their CUMULATIVE share (the largest difference), not bin by bin
  double cdf_gap = 0, cs = 0, cd = 0;
  for (int k = 0; k < 4096; ++k) { cs += Hs.values[static_cast<std::size_t>(k)] / total; cd += static_cast<double>(dcounts[static_cast<std::size_t>(k)]) / total; cdf_gap = std::max(cdf_gap, std::abs(cs - cd)); }
  std::printf("  %lld glued vertices, %zu outside frames; the stored histogram vs the direct one: largest cumulative difference %.3g (L1 %.3g of %.0f counts; frames %lld), direct 99th percentile %.4f\n", static_cast<long long>(Vg), outside.size(), cdf_gap, l1, total,
              static_cast<long long>(Hs.options["frames"].get<std::int64_t>()), p99);
  // a frame belongs to the window when its tile is in it: the two frames at the window's edges the direct computation (the sample the instant falls in) puts the other way
  CHECK(std::abs(Hs.options["frames"].get<std::int64_t>() - static_cast<std::int64_t>(outside.size())) <= 2 && cdf_gap < 5e-3 && std::abs(Hs.options["hi"].get<double>() / lb.hi - 1) < 3e-2);
  CHECK(Hs.options["chain"]["outside"] == "window" && Hs.options["chain"]["levels"] == lv && Hs.options["chain"]["relative_to"] == "spatial_baseline.baseline.rest");
  // chunk size and block size: the same distribution
  { auto p99_of = [&](const StoredStatistic& H) {
      std::vector<std::int64_t> c(H.values.begin(), H.values.end());
      return nxr::stats::histogram::binnedPercentile(c, nxr::stats::histogram::LogBins{H.options["lo"].get<double>(), H.options["hi"].get<double>(), 4096}, 99.0); };
    const double d = std::abs(p99_of(Hs) / p99_of(Hb) - 1);
    std::printf("  chunks of 4096 vs 1000: the 99th percentiles differ by %.3g\n", d);
    CHECK(d < 1e-2 && Hs.options["frames"] == Hb.options["frames"] && Hs.options["first_frame"] == Hb.options["first_frame"]); }
  { g::RunOptions small = all; small.block_frames = 100;
    g::run(ds, sub, dthr, tset, small);
    const StoredStatistic H2 = read_statistic(sub, "spatial_threshold.histogram.rest");
    double d = 0, tot = 0;
    for (std::size_t i = 0; i < Hs.values.size(); ++i) { d += std::abs(Hs.values[i] - H2.values[i]); tot += Hs.values[i]; }
    std::printf("  blocks of 100 frames vs 256: L1 difference %.3g\n", d);
    CHECK(d < 1e-3 * tot);
    CHECK(g::run(ds, sub, dthr, tset, small)[0]["hit"] == true);
    g::run(ds, sub, dthr, tset, all); }
  { g::RunOptions again; again.chunks = "all"; const auto r2 = g::run(ds, sub, dthr, tset, again); CHECK(r2.size() == 2 && r2[0]["hit"] == true); }

  // ── THE FALSE-ALARM RATE: the share of vertex-frames above the threshold inside the window, on pure noise ──
  auto share = [&](const Json& set, double& threshold) {
    g::run(ds, sub, dev, set, all);
    const StoredStatistic H = read_statistic(sub, "spatial_events.home.rest");
    double above = 0;
    for (std::int64_t v = 0; v < Vg; ++v) above += H.values[static_cast<std::size_t>(2 * v)];
    const auto row = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "spatial_events.events.rest", .any_status = true});
    const Json pj = row->params;
    threshold = pj["detection"]["record"]["threshold"].get<double>();
    return above / (static_cast<double>(Vg) * H.values[1]);
  };
  double thr_p = 0, thr_r = 0;
  const double fa_p = share(eset, thr_p);                                       // the DEFAULT: percentile 99
  { const auto row = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "spatial_events.events.rest", .any_status = true});
    const Json rec = row->params["detection"]["record"];
    CHECK(rec["threshold_rule"].get<std::string>().rfind("percentile", 0) == 0 && rec["percentile"] == 99.0 && rec["threshold_statistic"]["name"] == "spatial_threshold.histogram.rest"); }
  Json rset = eset; rset["threshold"] = "ratio"; rset["ratio"] = 2.0;
  const double fa_r = share(rset, thr_r);
  std::printf("  noise only, in the window: above the threshold: percentile 99 (threshold %.3f x baseline, the direct %.3f): %.2f %% of vertex-frames; ratio 2: %.2f %%\n", thr_p, p99, 100 * fa_p, 100 * fa_r);
  CHECK(std::abs(thr_p / p99 - 1) < 0.01);                                      // the derived threshold is the direct distribution's 99th percentile (to the range's rounding)
  // the same share computed DIRECTLY on the window's frames (the pipeline's own count against the direct one), and the rate against its 1 % target: the noise's spatial correlation leaves ~10 independent regions
  // (the reference level's hat is wide on this small sphere), so one realization's share scatters about 1 % by tens of percent: the tolerance is 0.5-2.2 %
  { std::int64_t n_in = 0, n_above = 0;
    for (std::int64_t j = 0; j < count; ++j) {
      const double tj = pos[static_cast<std::size_t>(j)] / sfreq;
      if (!(tj >= w0 && tj < w1)) continue;
      ++n_in;
      for (std::int64_t v = 0; v < Vg; ++v) if ((A(v, j) * A(v, j) + B(v, j) * B(v, j)) / Base.values[static_cast<std::size_t>(v * L + (refl - 1))] >= thr_p) ++n_above;
    }
    const double direct = static_cast<double>(n_above) / (static_cast<double>(Vg) * static_cast<double>(n_in));
    std::printf("  the same share counted directly on the window's %lld frames: %.2f %%\n", static_cast<long long>(n_in), 100 * direct);
    CHECK(std::abs(direct - fa_p) < 1e-3); }
  CHECK(fa_p > 0.005 && fa_p < 0.022);                                          // about 1 %
  CHECK(fa_r > 0.09 && fa_r < 0.18 && thr_r == 2.0);                            // about 13.5 %: exp(-2)
  // the events and the home map at the default do not depend on the chunk or the block
  { g::run(ds, sub, dev, eset, all);
    g::RunOptions small = all; small.block_frames = 100;
    const StoredStatistic H1 = read_statistic(sub, "spatial_events.home.rest");
    g::run(ds, sub, dev, eset, small);
    const StoredStatistic H2 = read_statistic(sub, "spatial_events.home.rest");
    CHECK(H1.values == H2.values);
    CHECK(g::run(ds, sub, dev, eset, small)[0]["hit"] == true);
    g::run(ds, sub, dev, eset, all); }
  // THE PLAN
  { Json s = eset; s["reference_scale"] = "index:3"; CHECK(what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); }).find("another spatial_band") != std::string::npos); }
  { Json s = eset; s["percentile"] = 100.0; CHECK(what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); }).find("percentile") != std::string::npos); }
  { Json s = eset; s["threshold_statistic"] = "no_such_statistic"; const std::string why = what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); });   // a named distribution that is not stored is refused by the input finder
    CHECK(why.find("inputs.threshold") != std::string::npos && why.find("nothing in the subject matches") != std::string::npos); }
  { Json s = tset; s["bursts"] = "window"; s["reference_scale"] = "index:3"; Json e2 = eset; e2["reference_scale"] = "index:3";
    g::run(ds, sub, dthr, s, all);                                               // a distribution of level 3 is then stored under the same name: events at level 4 refuse it
    CHECK(what_of([&] { g::run(ds, sub, dev, eset, g::RunOptions{}); }).find("another spatial_band") != std::string::npos); }
  return nxrtest::finish("test_spatial_threshold") || g_failures ? 1 : 0;
}
