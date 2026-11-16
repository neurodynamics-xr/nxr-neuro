// run global_bursts (spec docs/specs/2026-11-13-cortical-events.md, "Global bursts: WHEN, from the cortex") on a synthetic source space (never a real dataset): one planted episode of a 12 Hz
// oscillation in each hemisphere, each the level's own pattern of one tile (F_l δ at its centre vertex) under a double-sigmoid envelope with known onset, offset and rise and fall times.
// Checks: the trace (the max over the spatial tiles of the joint cell amplitude) and the bursts found on it, whole brain (two global bursts, no channel) and per hemisphere (one burst on each
// row, the row recorded); each burst's edges refined on SAMPLES by Rheome's risefall fit to within the planted sigmoids; the per-burst roll-up (amplitude, peak, duration) and fit measures and
// their statistics stored; `min_r2` sets a fit NaN; a rerun a hit; the plan's refusals (the posterior model not built).
#include <unistd.h>

#include <cmath>
#include <map>
#include <set>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/measures.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/joint.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::vector<float> read_floats(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
Json out_of(const Json& r, const std::string& o) { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_global_bursts_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 8192;
  const double sfreq = 250.0;
  const int L = 4;
  const std::int64_t W = 256;
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, cid_s;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      cid_s = cid;
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
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      std::vector<float> kid(static_cast<std::size_t>(N) * M, 0.0f);
      for (int v = 0; v < N; ++v) kid[static_cast<std::size_t>(v) * M + v] = 1.0f;
      OperatorSpec op;
      op.name = "fixed"; op.path = "fixed"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = N; op.n_cols = M;
      op.data = view(kid, {N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "scalar");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
    });
    sub.compose([&] { nxr::io::constantq::write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  tiles_compute(ds, sub, TilesComputeOptions{});
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  const SurfaceLevel SL = read_surface_level(sub, S, L);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S, SL, sbo);
  int timeLevel = -1;
  for (const Row& m : sub.db().all("SELECT member_ordinal, params_json FROM selection WHERE subject_id = ? AND member_of_id IS NOT NULL AND params_json LIKE '%resolves_hz%'", {V(sub.id())})) {
    const Json pj = Json::parse(text(m, "params_json"));
    if (pj["resolves_hz"]["lo"] == 8.0 && pj["resolves_hz"]["hi"] == 16.0 && pj["window_samples"] == W) timeLevel = static_cast<int>(integer(m, "member_ordinal"));
  }
  CHECK(timeLevel >= 0);
  if (timeLevel < 0) return 1;

  // THE PLANTED EPISODES: the level's own pattern of one tile in each hemisphere (left: tile 2; right: tile 16 + 5), 12 Hz under a double sigmoid a = S((t - on)/tr) S(-(t - off)/tf)
  const std::vector<std::int32_t> centres = [&] {
    ArrayReader r = ArrayReader::open(sub.store(), SL.record["node"].get<std::string>() + "/lc/L" + std::to_string(L) + "/centre");
    const auto raw = r.read();
    std::vector<std::int32_t> c(raw.size() / 4);
    std::memcpy(c.data(), raw.data(), raw.size());
    return c;
  }();
  struct Episode { int tile; double on, off, tr, tf; };
  const std::vector<Episode> eps = {{2, 8.0, 14.0, 0.30, 0.45}, {16 + 5, 20.0, 25.0, 0.25, 0.35}};
  auto pattern = [&](int tile) { Eigen::MatrixXd e = Eigen::MatrixXd::Zero(N, 1); e(centres[static_cast<std::size_t>(tile)], 0) = 1.0; Eigen::MatrixXd f = sband.apply(e); return Eigen::VectorXd(f.col(0) / f.col(0).cwiseAbs().maxCoeff()); };
  const Eigen::VectorXd p0 = pattern(eps[0].tile), p1 = pattern(eps[1].tile);
  auto S_ = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double s = t / sfreq, osc = std::cos(2 * M_PI * 12.0 * s);
      double x = 0.005 * nd(rng);
      x += p0(v) * S_((s - eps[0].on) / eps[0].tr) * S_(-(s - eps[0].off) / eps[0].tf) * osc;
      x += p1(v) * S_((s - eps[1].on) / eps[1].tr) * S_(-(s - eps[1].off) / eps[1].tf) * osc;
      xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>(x);
    }
  {
    SubjectWriter w = SubjectWriter::open(ds, "S1");
    w.compose([&] {
      FieldSpec rec;
      const std::string prod = w.product({cid_s, tid});
      rec.name = "rest"; rec.path = node_path(w.db(), prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, std::int64_t(4096)}; rec.array.compress = false; rec.cols.set("unit", "A");
      w.field(std::move(rec));
    });
  }
  JointComputeOptions jo; jo.recording = "rest"; jo.space_level = L; jo.time_level = timeLevel;
  joint_compute(ds, sub, jo);

  const g::Definition d = g::load("global_bursts");
  g::RunOptions all; all.chunks = "all";
  Json set1 = Json{{"space_level", L}, {"transform", "exact"}, {"spatial_shape", "meyer"}};   // whole brain, combine mean: global events (no row)
  auto spans_of = [&](const Json& r) {
    const Json bj = out_of(r, "bursts");
    std::vector<std::tuple<double, double, int>> s;
    const auto sel = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(bj["selection"].get<std::string>())});
    for (const Row& x : sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*sel, "id"))})) s.emplace_back(std::stod(text(x, "start")), std::stod(text(x, "stop")), 0);
    return std::make_pair(s, Json::parse(text(*sub.db().one("SELECT params_json FROM selection WHERE id = ?", {V(text(*sel, "id"))}), "params_json")));
  };
  auto measure_of = [&](const Json& r, const std::string& o, std::int64_t& rows, std::int64_t& events) {
    std::vector<std::int64_t> shape;
    const auto a = nxrtest::read_packed(sub.folder(), out_of(r, o), &shape);
    rows = shape[0]; events = shape[1];
    return a;
  };

  // ── WHOLE BRAIN: one trace, two global bursts, each refined on samples ──
  const auto r1 = g::run(ds, sub, d, set1, all);
  CHECK_EQ(static_cast<int>(r1.size()), 1);
  const auto [s1, pj1] = spans_of(r1[0]);
  std::printf("  whole brain: %zu burst(s)", s1.size());
  for (const auto& [a, z, _] : s1) std::printf(" [%.2f, %.2f)", a, z);
  std::printf("\n");
  CHECK_EQ(static_cast<int>(s1.size()), 2);
  CHECK(!pj1.contains("channels"));                                           // global: no row recorded
  std::int64_t R = 0, E = 0;
  const auto ton = measure_of(r1[0], "t_on", R, E), toff = measure_of(r1[0], "t_off", R, E), taur = measure_of(r1[0], "tau_rise", R, E), tauf = measure_of(r1[0], "tau_fall", R, E), r2 = measure_of(r1[0], "fit_r2", R, E);
  CHECK(R == 1 && E == 2);
  for (int e = 0; e < 2 && e < static_cast<int>(s1.size()); ++e) {
    const auto& ep = eps[static_cast<std::size_t>(e)];
    std::printf("  burst %d: tile edges [%.2f, %.2f) s; risefall tOn %.3f (planted %.2f), tOff %.3f (%.2f), tauR %.3f (%.2f), tauF %.3f (%.2f), r2 %.4f\n", e, std::get<0>(s1[static_cast<std::size_t>(e)]), std::get<1>(s1[static_cast<std::size_t>(e)]),
                ton[static_cast<std::size_t>(e * 2)], ep.on, toff[static_cast<std::size_t>(e * 2)], ep.off, taur[static_cast<std::size_t>(e * 2)], ep.tr, tauf[static_cast<std::size_t>(e * 2)], ep.tf, r2[static_cast<std::size_t>(e * 2)]);
    CHECK(std::get<0>(s1[static_cast<std::size_t>(e)]) < ep.on + 2 * ep.tr && std::get<1>(s1[static_cast<std::size_t>(e)]) > ep.off - 2 * ep.tf);   // the burst holds the plateau
    CHECK(std::abs(ton[static_cast<std::size_t>(e * 2)] - ep.on) < 0.25 && std::abs(toff[static_cast<std::size_t>(e * 2)] - ep.off) < 0.3);        // the fit's half-rise and half-fall: the planted sigmoids'
    CHECK(r2[static_cast<std::size_t>(e * 2)] > 0.9 && taur[static_cast<std::size_t>(e * 2)] > 0 && tauf[static_cast<std::size_t>(e * 2)] > 0);
  }
  // the per-burst roll-up of the trace's tiles, the duration, and the statistics across the bursts
  {
    std::int64_t r = 0, e = 0;
    const auto amp = measure_of(r1[0], "burst_amplitude", r, e), pk = measure_of(r1[0], "burst_peak", r, e), du = measure_of(r1[0], "burst_duration", r, e);
    CHECK(amp[0] > 0 && pk[0] >= amp[0] && du[0] > 0.4 && std::isfinite(amp[2]));
    for (const char* s : {"burst_amplitude_stats", "tau_rise_stats", "fit_r2_stats", "baseline"}) CHECK(!out_of(r1[0], s).is_null());
    const auto st = sub.db().one("SELECT n_members FROM statistic WHERE subject_id = ? AND name LIKE '%burst_amplitude_stats%'", {V(sub.id())});
    CHECK(st.has_value());
  }
  // PACKED: the run's measurements over one Selection share ONE array each (the file count); each measure's rows point at its component, and the lookup reads by component
  {
    std::set<std::string> arrays;
    std::int64_t n_measures = 0;
    for (const auto& x : r1[0]["stored"]) if (x.contains("array") && x.contains("component")) { arrays.insert(x["array"].get<std::string>()); ++n_measures; }
    std::printf("  packed: %lld measurement outputs in %zu array(s)\n", static_cast<long long>(n_measures), arrays.size());
    CHECK(n_measures >= 20 && arrays.size() < static_cast<std::size_t>(n_measures) / 2);
    // measures of one array read by their own component: the lookup's values are the stored outputs'
    for (const char* o : {"t_on", "tau_rise", "fit_r2"}) {
      const Json x = out_of(r1[0], o);
      CHECK(x["packed"].get<std::int64_t>() >= 1);
      const auto own = nxrtest::read_packed(sub.folder(), x);
      const auto row = sub.db().one("SELECT selection_id, of_field_id, measure, component, array_path FROM selection_measurement WHERE array_path = ? AND component = ? AND measure <> 'count'", {V(x["array"].get<std::string>()), V(x["component"].get<std::int64_t>())});
      CHECK(row.has_value());
      const MeasurementValues mv = measurement_values(sub, text(*row, "selection_id"), text(*row, "of_field_id"), std::nullopt, std::nullopt, text(*row, "measure"));
      CHECK(mv.values.size() * (x["components"].get<std::int64_t>() + 1) == own.size());
      for (std::size_t i = 0; i < mv.values.size(); ++i) CHECK(std::isnan(mv.values[i]) ? std::isnan(own[i * 2]) : std::abs(mv.values[i] - own[i * 2]) < 1e-6);
    }
  }
  // the recipe records what preparation resolved, as the cells' does
  CHECK(r1[0]["prepared"].contains("apply_kernel:sources"));
  // a rerun is a hit
  { const auto r = g::run(ds, sub, d, set1, all); CHECK(r.size() == 1 && r[0]["hit"] == true); }

  // ── TIME SUB-BLOCKS: the pointwise tail (after the band's taps) run on sub-blocks of another width, or on the whole chunk, gives the same values ──
  auto values = [&](const Json& r) {
    std::map<std::string, std::vector<float>> v;
    for (const char* o : {"t_on", "burst_amplitude", "burst_peak", "trace_amplitude", "cell_amplitude", "extent_area_by_tile"}) { std::int64_t a = 0, b = 0; if (!out_of(r, o).is_null() && out_of(r, o).contains("array")) v[o] = measure_of(r, o, a, b); }
    return v;
  };
  auto same = [&](const std::map<std::string, std::vector<float>>& a, const std::map<std::string, std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (const auto& [k, x] : a) {
      const auto& y = b.at(k);
      if (x.size() != y.size()) return false;
      for (std::size_t i = 0; i < x.size(); ++i)
        if (!(std::isnan(x[i]) && std::isnan(y[i])) && std::abs(x[i] - y[i]) > 1e-5f * std::max(1.0f, std::abs(x[i]))) { std::printf("  %s[%zu]: %g vs %g\n", k.c_str(), i, x[i], y[i]); return false; }
    }
    return true;
  };
  const auto v1 = values(r1[0]);
  CHECK(v1.size() >= 4);
  for (const std::int64_t bw : {std::int64_t(0), std::int64_t(333)}) {
    g::RunOptions o = all; o.replace = true; o.block_samples = bw;
    const auto r = g::run(ds, sub, d, set1, o);
    CHECK(r[0]["hit"] == false);                                              // another width is another key (method rule 3) ...
    CHECK(spans_of(r[0]).first == s1);
    CHECK(same(v1, values(r[0])));                                            // ... with the same values
  }
  // ── --replace OF A RUN OVER ANOTHER SPAN: its outputs that share a Selection (the extent's area and centre per time tile) are replaced together ──
  {
    g::RunOptions part; part.start = 0.0; part.stop = 20.0; part.replace = true;
    g::run(ds, sub, d, set1, part);                                           // shorter span: other counts
    g::RunOptions rep = all; rep.replace = true;
    const auto r = g::run(ds, sub, d, set1, rep);                            // refused before (the centre's old rows were "others" of the area's count row)
    CHECK(spans_of(r[0]).first == s1 && same(v1, values(r[0])));
  }

  // ── THE BURSTS' OPTIONS: the fit gate (`bursts_kept`), the stages (onset, within, offset) from the stored fits, durations in cycles ──
  auto spans_named = [&](const Json& r, const std::string& o) {
    std::vector<std::pair<double, double>> v;
    const Json bj = out_of(r, o);
    if (bj.is_null()) return v;
    const auto sel = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(bj["selection"].get<std::string>())});
    for (const Row& x : sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*sel, "id"))})) v.emplace_back(std::stod(text(x, "start")), std::stod(text(x, "stop")));
    return v;
  };
  {
    // every planted burst fits well (r2 > 0.9, the times far from their bound): all kept; the unfiltered spans stay
    const auto kept = spans_named(r1[0], "bursts_kept");
    CHECK(kept.size() == 2 && spans_named(r1[0], "bursts").size() == 2 && kept[0].first == std::get<0>(s1[0]) && kept[1].second == std::get<1>(s1[1]));
    // the stages: the stored fit's own numbers (float32 measures) give onset t_on ± 2 tau_rise, within [t_on + 2 tau_rise, t_off − 2 tau_fall], offset t_off ± 2 tau_fall
    const auto on = spans_named(r1[0], "burst_onset"), wi = spans_named(r1[0], "burst_within"), off = spans_named(r1[0], "burst_offset");
    CHECK(on.size() == 2 && wi.size() == 2 && off.size() == 2);
    double worst = 0;
    for (std::size_t e = 0; e < 2 && on.size() == 2 && wi.size() == 2 && off.size() == 2; ++e) {
      const double a = ton[e * 2], b = toff[e * 2], tr_ = taur[e * 2], tf_ = tauf[e * 2];
      worst = std::max({worst, std::abs(on[e].first - (a - 2 * tr_)), std::abs(on[e].second - (a + 2 * tr_)), std::abs(wi[e].first - (a + 2 * tr_)), std::abs(wi[e].second - (b - 2 * tf_)), std::abs(off[e].first - (b - 2 * tf_)), std::abs(off[e].second - (b + 2 * tf_))});
      std::printf("  burst %zu stages: onset [%.3f, %.3f), within [%.3f, %.3f), offset [%.3f, %.3f)\n", e, on[e].first, on[e].second, wi[e].first, wi[e].second, off[e].first, off[e].second);
    }
    CHECK(worst < 2e-3);                                                       // the measures are stored as float32, the spans as seconds
    // the gate: r2 between the two bursts' keeps one; the stages of the burst it refused are gone
    const double lo = std::min(r2[0], r2[2]), hi = std::max(r2[0], r2[2]);
    CHECK(hi > lo);
    Json setg = set1; setg["gate_min_r2"] = (lo + hi) / 2;
    g::RunOptions rep = all; rep.replace = true;
    const auto rg = g::run(ds, sub, d, setg, rep);
    const auto kg = spans_named(rg[0], "bursts_kept"), og = spans_named(rg[0], "burst_onset");
    std::printf("  fit gate at r2 %.4f (fits %.4f, %.4f): %zu of %zu kept; %zu onset stage(s)\n", (lo + hi) / 2, r2[0], r2[2], kg.size(), spans_named(rg[0], "bursts").size(), og.size());
    CHECK(kg.size() == 1 && spans_named(rg[0], "bursts").size() == 2 && og.size() == 1);
    const auto sg = sub.db().one("SELECT params_json FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(out_of(rg[0], "bursts_kept")["selection"].get<std::string>())});
    const Json gj = Json::parse(text(*sg, "params_json"))["detection"]["gate"];
    CHECK(gj["kept"] == 1 && gj["events"] == 2 && gj["min_r2"].get<double>() == (lo + hi) / 2 && gj["refused"]["r2_below"] == 1 && gj["ordinals"].size() == 1);
    // another gate is another key: the rerun above replaced; r2 = 0 keeps both again
    Json set0 = set1; set0["gate_min_r2"] = 0.0;
    const auto r0 = g::run(ds, sub, d, set0, rep);
    CHECK(spans_named(r0[0], "bursts_kept").size() == 2);
    // DURATIONS IN CYCLES: of the band's centre (12 Hz): 3 cycles kept = 0.25 s, merged within 1 cycle = 0.0833 s; the same two bursts; recorded with the resolved seconds
    Json setc = set1; setc["durations"] = "cycles";
    const auto rc = g::run(ds, sub, d, setc, rep);
    const auto [sc, pjc] = spans_of(rc[0]);
    const Json dj = pjc["detection"]["durations"];
    std::printf("  durations in cycles: %zu burst(s); %s\n", sc.size(), dj.dump().c_str());
    CHECK(sc.size() == 2 && dj["mode"] == "cycles" && std::abs(dj["centre_hz"].get<double>() - 12.0) < 1e-12 && std::abs(dj["min_s"].get<double>() - 3.0 / 12.0) < 1e-12 && std::abs(dj["merge_s"].get<double>() - 1.0 / 12.0) < 1e-12 && dj["max_s"].is_null());
    CHECK(pjc["parameters"]["durations"] == "cycles" && pj1["parameters"]["durations"] == "seconds" && pj1["detection"]["durations"]["mode"] == "seconds");
    // risefall in cycles (nxr burst_stages': margin 10, smoothing 1, tau <= 10 cycles) is on record in the fits' provenance
    { std::int64_t a = 0, b = 0; measure_of(rc[0], "t_on", a, b);
      const auto rm = sub.db().one("SELECT params_json FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(out_of(rc[0], "bursts")["selection"].get<std::string>())});
      CHECK(rm.has_value() && Json::parse(text(*rm, "params_json"))["detection"]["durations"]["mode"] == "cycles"); }
    // a minimum longer than any burst (100 cycles = 8.3 s): no burst is found, and the run says so with empty sets (the kept bursts and the stages too)
    Json setl = set1; setl["durations"] = "cycles"; setl["min_cycles"] = 100.0;
    const auto rl = g::run(ds, sub, d, setl, rep);
    std::printf("  min 100 cycles: %zu burst(s), %zu kept, %zu onset stage(s)\n", spans_named(rl[0], "bursts").size(), spans_named(rl[0], "bursts_kept").size(), spans_named(rl[0], "burst_onset").size());
    CHECK(spans_named(rl[0], "bursts").empty() && spans_named(rl[0], "bursts_kept").empty() && spans_named(rl[0], "burst_onset").empty());
    // a durations option it does not know is refused with the valid ones
    Json setb = set1; setb["durations"] = "beats";
    CHECK(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), setb); }) == std::string(code::invalid_argument));
    g::run(ds, sub, d, set1, rep);                                             // back to seconds for what follows
  }

  // ── PER HEMISPHERE: two traces, one burst on each row, the row recorded ──
  {
    Json set2 = set1; set2["groups"] = "per_hemisphere"; set2["combine"] = "each";
    g::RunOptions rep = all; rep.replace = true;
    const auto r = g::run(ds, sub, d, set2, rep);
    const auto [s2, pj2] = spans_of(r[0]);
    std::printf("  per hemisphere: %zu burst(s); channels %s\n", s2.size(), pj2.value("channels", Json()).dump().c_str());
    CHECK_EQ(static_cast<int>(s2.size()), 2);
    CHECK(pj2.contains("channels") && pj2["channels"]["0"] == Json::array({0}) && pj2["channels"]["1"] == Json::array({1}));   // the left episode first, on row 0; the right on row 1
    std::int64_t R2 = 0, E2 = 0;
    const auto to = measure_of(r[0], "t_on", R2, E2);
    CHECK(R2 == 2 && E2 == 2 && std::abs(to[0] - eps[0].on) < 0.25 && std::isnan(to[2]) && std::isnan(to[4]) && std::abs(to[6] - eps[1].on) < 0.25);   // [row 0: event 0, NaN] [row 1: NaN, event 1], two components each
    // min_r2: a fit below it is set NaN, the burst kept
    Json set3 = set2; set3["min_r2"] = 0.99999999;
    const auto rm = g::run(ds, sub, d, set3, rep);
    std::int64_t R3 = 0, E3 = 0;
    const auto t3 = measure_of(rm[0], "t_on", R3, E3);
    CHECK(spans_of(rm[0]).first.size() == 2 && std::isnan(t3[0]) && std::isnan(t3[6]));
  }
  // ── REFUSALS ──
  {
    std::string why;
    Json bad = set1; bad["model"] = "posterior";
    try { g::plan_runs(d, sub.db(), sub.id(), bad); } catch (const Error& e) { why = e.what(); }
    CHECK(why.find("not built") != std::string::npos);
    bad = set1; bad["groups"] = "per_hemisphere";                           // the mean over the hemispheres is not a per-hemisphere detection
    why.clear();
    try { g::plan_runs(d, sub.db(), sub.id(), bad); } catch (const Error& e) { why = e.what(); }
    CHECK(why.find("per_hemisphere") != std::string::npos && why.find("each") != std::string::npos);
    bad = set1; bad["groups"] = "per_lobe";
    CHECK(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), bad); }) == std::string(code::invalid_argument));
  }
  return nxrtest::finish("test_global_bursts");
}
