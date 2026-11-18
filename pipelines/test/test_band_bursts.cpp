// band_bursts (spec docs/specs/2026-11-13-time-events.md) on a synthetic recording (never a real dataset): 3 channels at 1200 Hz, a 10 Hz rhythm at 0.1 whose
// amplitude drifts ±10 % (so the robust spread is not 0 and the background never reaches median + 1.5 SD), three planted bursts at amplitude 1 (two on
// channel 0, one on channel 1, none on channel 2), a constant-Q ladder on the time line. Checks: the amplitude stored as measurement rows on the alpha level
// (8-16 Hz, D84), equal to the direct mean of |z| per tile; the planted bursts found (each span holds its burst, each edge within the tile of the true
// edge or the next one out: the band's response), no others, each on its channel, in the importer's
// form (a spans Selection, code = the type `burst` through its dictionary, the channel in params_json.channels, the thresholds recorded); the same with
// another chunk width; `combine mean`; a bad span's tiles NaN; a rerun a hit; another job's outputs refused unless replace; a band that is no level's refused.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <random>
#include <set>

#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/io.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/run_shipped.h"
#include "nxr/stats/summary.h"
#include "nxr/signal/filters_fir.h"
#include "make_grids.h"

using namespace nxr::io;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

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
struct Planted { int channel; double start, stop; };
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_band_bursts_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};   // removed even when a check throws
  const int M = 3, T = 61440;                                                         // 51.2 s: 60 tiles of 1024 samples
  const double sfreq = 1200.0, bad0 = 40.0, bad1 = 40.5;
  const std::vector<Planted> planted = {{0, 12.0, 14.5}, {0, 30.0, 31.2}, {1, 20.3, 23.0}};
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (int c = 0; c < M; ++c)
    for (int t = 0; t < T; ++t) {
      const double s = t / sfreq;
      double a = 0.1 * (1 + 0.1 * std::sin(2 * M_PI * s / 7.3 + c));
      for (const auto& p : planted) if (p.channel == c && s >= p.start && s < p.stop) a = 1.0;
      xf[static_cast<std::size_t>(c * T + t)] = static_cast<float>(a * std::cos(2 * M_PI * 10.0 * s + c) + 0.002 * nd(rng));
    }
  DatasetWriter ds = create_dataset(root, "demo");
  std::string tid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
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
      // the same samples in two chunk widths, and with a bad span
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(32768)), std::make_pair(std::string("rest_b"), std::int64_t(3000)),
                                        std::make_pair(std::string("rest_c"), std::int64_t(32768))}) {
        FieldSpec rec;
        rec.name = name; rec.path = tk::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "T");
        const std::string rid = sub.field(std::move(rec));
        if (name == "rest_c") {
          nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
          nxr::bst::time_spans(sub, "time/rest_c_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
        }
      }
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_time_grids(ds, sub, "band", Json{{"band", "8-16 Hz"}});
  const g::Definition d = g::load("band_bursts");
  g::RunOptions all;
  all.chunks = "all";

  // THE PLAN: the alpha level found by its band (D84); a band that is no level's refused
  const auto plans = g::plan_runs(d, sub.db(), sub.id(), Json::object());
  CHECK_EQ(static_cast<int>(plans.size()), 3);
  CHECK_EQ(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), Json{{"band", Json::array({8.0, 12.0})}}); }), std::string(code::invalid_argument));

  // 1. THE RUN on every chunk of each recording
  const auto r = g::run(ds, sub, d, Json::object(), all);
  CHECK_EQ(static_cast<int>(r.size()), 3);
  auto result = [&](const std::string& rec) -> Json { for (const auto& x : r) if (x.value("anchor", std::string()) == rec) return x; return Json(); };
  const Json ra = result("rest");
  CHECK(!ra.is_null() && ra["hit"] == false);
  // the amplitude rows: on the alpha level, array-backed [3 x tiles x 2], equal to the direct mean of |z| per tile
  auto out_of = [](const Json& r, const std::string& o) -> Json { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); };
  const Json am = out_of(ra, "tiles");
  CHECK(am["kind"] == "measurement" && am["rows"] == 2);                        // amplitude and envelope_max: the two measures of one reduction
  const auto tsub = tk::Subject::of(sub);
  const nxr::io::testing::Probe probe(sub);
  const auto lvl = tsub.by_id(tk::Cls::selection, am["over"].get<std::string>());
  CHECK(lvl.has_value() && lvl->name == "rest time grid band 8-16 Hz");        // the grid is found by the name of its rule
  const Json lp = lvl->params;
  CHECK(lp["band"][0] == 8.0 && lp["band"][1] == 16.0 && lp["grid"]["width"] == 512);       // the band rule: the first 2^k samples holding 4 cycles of the band's centre (11.3 Hz: 0.35 s -> 512)
  const std::int64_t TT = lp["grid"]["count"].get<std::int64_t>(), W = lp["grid"]["width"].get<std::int64_t>();
  const auto rest = tsub.find({.cls = tk::Cls::field, .name = "rest", .any_status = true});
  CHECK(rest.has_value());
  // the values of one measure over a selection, [rows x members]
  auto values_of = [&](const std::string& selection_id, const std::string& field_id, const char* measure) { return measurement_values(sub, selection_id, field_id, std::nullopt, std::nullopt, measure); };
  const MeasurementValues amp_mv = values_of(lvl->id, rest->id, "amplitude"), peak_mv = values_of(lvl->id, rest->id, "envelope_max");
  CHECK(amp_mv.rows == M && amp_mv.members == TT && peak_mv.members == TT);
  const std::vector<double>& arr = amp_mv.values;
  {
    const auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, 8.0, 16.0); };
    const auto h = nxr::signal::gainTaps(gain, sfreq, false), q = nxr::signal::gainTaps(gain, sfreq, true);
    const auto re = nxr::signal::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::signal::convolveRows(xf.data(), M, T, q, 0, T, true);
    double worst = 0, scale = 0;
    for (int c = 0; c < M; ++c)
      for (std::int64_t t = 0; t < T / W; ++t) {
        double s = 0;
        for (std::int64_t k = t * W; k < (t + 1) * W; ++k) { const double a = re[static_cast<std::size_t>(c * T + k)], b = im[static_cast<std::size_t>(c * T + k)]; s += static_cast<float>(std::sqrt(a * a + b * b)); }
        const double want = s / W, got = arr[static_cast<std::size_t>(c * TT + t)];
        worst = std::max(worst, std::abs(got - want));
        scale = std::max(scale, want);
      }
    std::printf("  amplitude per tile vs the direct |z| mean: %.2e (of %.2e)\n", worst, scale);
    CHECK(worst < 1e-5 * scale);
  }
  // the bursts: a spans Selection in the importer's form, each planted burst found to within a tile on its channel, no others
  const Json bs = out_of(ra, "bursts");
  // THE BASELINE, a statistic (summarize: median and mad per channel over the level's good tiles), the thresholds' ingredients the detection read
  {
    const Json bl = out_of(ra, "baseline");
    CHECK(!bl.is_null());
    if (!bl.is_null()) {
      const StoredStatistic st = read_statistic(sub, bl["name"].get<std::string>());
      CHECK(text(st.row, "method") == "summary" && text(st.row, "measure") == "amplitude" && st.shape == (std::vector<std::int64_t>{M, 2}));
      CHECK(st.axes[1]["values"][0]["name"] == "median" && st.axes[1]["values"][1]["name"] == "mad");
      // the baseline is the median and robust SD of the stored amplitude tiles (channel 2: no burst)
      std::vector<double> c2;
      for (std::int64_t t = 0; t < TT; ++t) { const double v = arr[static_cast<std::size_t>(2 * TT + t)]; if (std::isfinite(v)) c2.push_back(v); }
      CHECK(std::abs(st.values[2 * 2] - nxr::stats::median(c2)) < 1e-12 && std::abs(st.values[2 * 2 + 1] - nxr::stats::mad(c2)) < 1e-12);
      // the same from the STORED amplitude by the verb (summarize a stored measurement): the run's baseline
      const auto vr = nxr::pipelines::run_shipped(ds, sub, "summarize", Json{{"over", lvl->name}, {"of", "rest"}, {"measure", "amplitude"}, {"summaries", Json::array({"median", "mad"})}, {"name", "rest_alpha_baseline"}}, false);
      const StoredStatistic vs = read_statistic(sub, "rest_alpha_baseline");
      CHECK(!vr.empty() && vs.values == st.values);
    }
  }
  const auto sel_rows = probe.rows("selection", {{"id", V(bs["id"].get<std::string>())}});
  const std::optional<Row> sel = sel_rows.empty() ? std::nullopt : std::optional<Row>(sel_rows.front());
  CHECK(sel && text(*sel, "type") == "spans" && text(*sel, "manifold_id") == tid && text(*sel, "cell") == "1" && !is_null(*sel, "dictionary_id"));
  const auto entries = probe.rows("dictionary_entry", {{"dictionary_id", V(text(*sel, "dictionary_id"))}});
  CHECK(!entries.empty() && text(entries.front(), "name") == "burst");
  const Json sp = Json::parse(text(*sel, "params_json"));
  CHECK(sp.contains("cache") && sp["detection"]["thresholds"].size() == 3 && sp["detection"]["thresholds"][2]["spans"] == 0);
  const auto found = probe.rows("selection_span", {{"selection_id", V(text(*sel, "id"))}}, "ordinal");
  CHECK_EQ(found.size(), planted.size());
  const double w = W / sfreq;
  for (std::size_t k = 0; k < found.size() && k < planted.size(); ++k) {
    const double a = std::stod(text(found[k], "start")), b = std::stod(text(found[k], "stop"));
    const int ch = sp["channels"][std::to_string(k)][0].get<int>();
    bool matched = false;
    // the span holds the planted burst, and each edge lies in the tile holding the true edge or the next one out (the band's response carries an edge
    // into the next tile: a 10-fold burst's decay tail stays above the exit threshold there)
    for (const auto& p : planted) matched = matched || (p.channel == ch && a <= p.start && a > p.start - 2 * w && b >= p.stop && b < p.stop + 2 * w);
    std::printf("  burst %zu on channel %d: [%.3f, %.3f) s\n", k, ch, a, b);
    CHECK(matched);
  }
  // PER-BURST MEASUREMENTS (docs/specs/2026-11-13-burst-statistics.md): each burst measured from the tiles it covers (the tiles' ingredients merged), stored over the run's own bursts Selection,
  // [rows x bursts] a measure, on the row of the channel that found it; equal to the direct values over the burst's samples
  std::vector<double> dur_all;
  {
    const std::int64_t TW = W;
    const auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, 8.0, 16.0); };
    const auto h = nxr::signal::gainTaps(gain, sfreq, false), q = nxr::signal::gainTaps(gain, sfreq, true);
    const auto re = nxr::signal::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::signal::convolveRows(xf.data(), M, T, q, 0, T, true);
    auto env = [&](int c, std::int64_t k) { const double a = re[static_cast<std::size_t>(c * T + k)], b = im[static_cast<std::size_t>(c * T + k)]; return static_cast<double>(static_cast<float>(std::sqrt(a * a + b * b))); };
    const auto evs = probe.rows("selection_span", {{"selection_id", V(bs["id"].get<std::string>())}}, "ordinal");
    const Json params = tsub.require_id(tk::Cls::selection, bs["id"].get<std::string>()).params;
    const std::int64_t E = static_cast<std::int64_t>(evs.size());
    CHECK_EQ(E, std::int64_t(3));
    const double wt = static_cast<double>(TW) / sfreq;
    const Json bv = out_of(ra, "burst_values");
    CHECK(bv["kind"] == "measurement" && bv["rows"] == 4 && bv["over"] == bs["id"]);   // amplitude, envelope_max, duration, interval_next
    const std::string sid_ev = bs["id"].get<std::string>();
    const MeasurementValues A = values_of(sid_ev, rest->id, "amplitude"), P = values_of(sid_ev, rest->id, "envelope_max"), D = values_of(sid_ev, rest->id, "duration"), I = values_of(sid_ev, rest->id, "interval_next");
    CHECK(A.rows == M && A.members == E && A.array_backed && D.members == E && I.members == E);
    std::vector<int> ch;
    std::vector<double> dur, gap, amps, peaks;
    std::vector<std::vector<double>> dur_by(M), amp_by(M), peak_by(M), gap_by(M);
    double worst = 0;
    for (std::int64_t k = 0; k < E; ++k) {
      const std::int64_t f = std::llround(std::stod(text(evs[static_cast<std::size_t>(k)], "start")) / wt), e = std::llround(std::stod(text(evs[static_cast<std::size_t>(k)], "stop")) / wt);
      const int c = params["channels"][std::to_string(k)][0].get<int>();
      double sa = 0, pk = 0;
      for (std::int64_t t = f * TW; t < e * TW; ++t) { sa += env(c, t); pk = std::max(pk, env(c, t)); }
      const double want_a = sa / static_cast<double>((e - f) * TW);
      for (int r = 0; r < M; ++r) {
        const auto at = [&](const MeasurementValues& v) { return v.values[static_cast<std::size_t>(r * E + k)]; };
        if (r == c) {
          worst = std::max({worst, std::abs(at(A) - want_a), std::abs(at(P) - pk)});
          CHECK(std::abs(at(D) - static_cast<double>(e - f) * wt) < 1e-5);
        } else {
          CHECK(std::isnan(at(A)) && std::isnan(at(P)) && std::isnan(at(D)) && std::isnan(at(I)));
        }
      }
      amp_by[c].push_back(want_a); peak_by[c].push_back(pk); dur_by[c].push_back(static_cast<double>(e - f) * wt);
      ch.push_back(c); dur.push_back(static_cast<double>(e - f) * wt);
      gap.push_back(static_cast<double>(f));
    }
    std::printf("  per-burst amplitude and peak vs the direct values over the burst's samples: %.2e\n", worst);
    CHECK(worst < 1e-5);
    // the interval to the next burst on the same channel: channel 0's two bursts; the last on a channel is NaN
    {
      std::int64_t first0 = -1, second0 = -1;
      for (std::int64_t k = 0; k < E; ++k) if (ch[static_cast<std::size_t>(k)] == 0) (first0 < 0 ? first0 : second0) = k;
      const double s1 = std::stod(text(evs[static_cast<std::size_t>(second0)], "start")), e0 = std::stod(text(evs[static_cast<std::size_t>(first0)], "stop"));
      CHECK(std::abs(I.values[static_cast<std::size_t>(0 * E + first0)] - (s1 - e0)) < 1e-4 && std::isnan(I.values[static_cast<std::size_t>(0 * E + second0)]));
      for (std::int64_t k = 0; k < E; ++k) if (ch[static_cast<std::size_t>(k)] == 1) CHECK(std::isnan(I.values[static_cast<std::size_t>(1 * E + k)]));
      gap_by[0].push_back(s1 - e0);
    }
    CHECK(std::abs(A.values[static_cast<std::size_t>(ch[0] * E)] - amp_by[static_cast<std::size_t>(ch[0])][0]) < 1e-5);
    // THE STATISTICS ACROSS THE BURSTS, per channel and measure: count, sum, median, mean, sd, percentiles 25, 75, 90 (the default); channel 2 has none: count and sum 0.
    // One statistic: axes row x measure (amplitude, envelope_max, duration, interval_next) x summary.
    {
      const StoredStatistic ss = read_statistic(sub, out_of(ra, "burst_stats")["name"].get<std::string>());
      const std::int64_t K = 4, S = 8;
      CHECK(ss.shape == (std::vector<std::int64_t>{M, K, S}));
      CHECK(ss.axes[1]["values"][0]["name"] == "amplitude" && ss.axes[1]["values"][2]["name"] == "duration" && ss.axes[2]["values"][0]["name"] == "count" && ss.axes[2]["values"][1]["name"] == "sum" && ss.axes[2]["values"][7]["name"] == "percentile:90");
      auto at = [&](int c, int m, int sm) { return ss.values[static_cast<std::size_t>((c * K + m) * S + sm)]; };
      for (int c = 0; c < M; ++c) {
        const auto& d = dur_by[static_cast<std::size_t>(c)];
        const double n = static_cast<double>(d.size());
        CHECK(at(c, 2, 0) == n && at(c, 0, 0) == n);
        double tot = 0; for (double v : d) tot += v;
        CHECK(std::abs(at(c, 2, 1) - tot) < 1e-5);
        if (d.empty()) { CHECK(std::isnan(at(c, 2, 2))); continue; }
        CHECK(std::abs(at(c, 2, 2) - nxr::stats::median(d)) < 1e-5 && std::abs(at(c, 2, 3) - nxr::stats::mean(d)) < 1e-5);
        CHECK(std::abs(at(c, 0, 2) - nxr::stats::median(amp_by[static_cast<std::size_t>(c)])) < 1e-5);
        CHECK(std::abs(at(c, 1, 3) - nxr::stats::mean(peak_by[static_cast<std::size_t>(c)])) < 1e-5);
        CHECK(std::abs(at(c, 0, 7) - nxr::stats::percentile(amp_by[static_cast<std::size_t>(c)], 90)) < 1e-5);
      }
      CHECK(at(0, 3, 0) == 1 && std::abs(at(0, 3, 1) - gap_by[0][0]) < 1e-4 && at(1, 3, 0) == 0 && at(2, 3, 0) == 0);   // one interval on channel 0; none on 1 and 2
      CHECK(text(ss.row, "partition_selection_id") == bs["id"].get<std::string>());
    }
  }
  // 2. ANOTHER CHUNK WIDTH: the same tiles and bursts
  {
    const Json rb = result("rest_b");
    const auto rest_b = tsub.find({.cls = tk::Cls::field, .name = "rest_b", .any_status = true});
    CHECK(rest_b.has_value());
    const MeasurementValues mb = values_of(out_of(rb, "tiles")["over"].get<std::string>(), rest_b->id, "amplitude");
    double worst = 0;
    CHECK(mb.values.size() == arr.size());
    for (std::size_t i = 0; i < arr.size(); ++i) if (std::isfinite(arr[i])) worst = std::max(worst, std::abs(mb.values[i] - arr[i]) / std::max(1e-6, arr[i]));
    std::printf("  chunk width 32768 vs 3000: %.2e\n", worst);
    CHECK(worst < 1e-4);
    CHECK(out_of(rb, "bursts")["spans"] == bs["spans"]);
  }
  // 3. A BAD SPAN [40, 40.5) s: tiles 93 and 94 NaN
  {
    const auto rest_c = tsub.find({.cls = tk::Cls::field, .name = "rest_c", .any_status = true});
    CHECK(rest_c.has_value());
    const MeasurementValues mc = values_of(out_of(result("rest_c"), "tiles")["over"].get<std::string>(), rest_c->id, "amplitude");
    CHECK(std::isnan(mc.values[93]) && std::isnan(mc.values[94]) && std::isfinite(mc.values[92]) && std::isfinite(mc.values[95]));
  }
  // 4. A RERUN is a hit; other parameters refused unless replace; COMBINE MEAN: the three bursts on one trace, no channel
  {
    const auto again = g::run(ds, sub, d, Json::object(), all);
    bool hits = true;
    for (const auto& x : again) hits = hits && x["hit"] == true;
    CHECK(hits);
    CHECK_EQ(code_of([&] { g::run(ds, sub, d, Json{{"combine", "mean"}}, all); }), std::string(code::already_exists));
    g::RunOptions rep = all;
    rep.replace = true;
    const auto mean = g::run(ds, sub, d, Json{{"combine", "mean"}}, rep);
    Json rm;
    for (const auto& x : mean) if (x.value("anchor", std::string()) == "rest") rm = x;
    CHECK(rm["hit"] == false && out_of(rm, "bursts")["spans"] == 3);
    const auto s2 = tsub.by_id(tk::Cls::selection, out_of(rm, "bursts")["id"].get<std::string>());
    CHECK(s2 && !s2->params.contains("channels"));
    // the mean trace's bursts are measured on every row, each on its own tiles, and their statistics are one row
    const MeasurementValues Am = values_of(out_of(rm, "burst_values")["over"].get<std::string>(), rest->id, "amplitude");
    CHECK(Am.rows == M && Am.members == 3);
    for (int r = 0; r < M; ++r) for (int k = 0; k < 3; ++k) CHECK(std::isfinite(Am.values[static_cast<std::size_t>(r * 3 + k)]));   // each row on its own tiles
    const StoredStatistic cs = read_statistic(sub, out_of(rm, "burst_stats")["name"].get<std::string>());
    CHECK(cs.shape == (std::vector<std::int64_t>{1, 4, 8}) && cs.values[(2) * 8] == 3);   // one row; the duration's count
  }
  // 5. THE MEDIAN of the band's envelope per tile (exact; numpy's), held per tile until it closes: equal to the direct median of |z|, whatever the chunk width
  {
    const std::string yaml =
        "pipeline: band_median\nversion: 1\ndescription: test\nscope: subject\n"
        "inputs:\n  recording: {field: recording, each: true}\n  grid: {selection: any, manifold: recording.time, where: {name: \"{recording} time grid band 8-16 Hz\"}}\n"
        "  bad: {selection: spans, manifold: recording.time, where: {name: \"{recording}_bad_segments\"}, required: false}\n"
        "parameters:\n  band: {default: \"8-16 Hz\", meaning: test, unit: Hz}\nsteps:\n  chunks:\n    over: recording\n    do:\n"
        "      - {node: signal.envelope, in: {x: recording}, with: {band: $band}, out: amp}\n"
        "      - {node: measurement.reduce, in: {x: amp, over: grid, exclude: bad}, with: {measures: [median]}, out: [med, med_ingredients]}\n"
        "outputs:\n  med: {kind: measurement, over: grid, of: recording}\n";
    const g::Definition dm = g::parse(parse_yaml(yaml, "test.yaml"), "test.yaml", yaml);
    const auto rm = g::run(ds, sub, dm, Json::object(), all);
    const auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, 8.0, 16.0); };
    const auto h = nxr::signal::gainTaps(gain, sfreq, false), q = nxr::signal::gainTaps(gain, sfreq, true);
    const auto re = nxr::signal::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::signal::convolveRows(xf.data(), M, T, q, 0, T, true);
    auto env = [&](int c, std::int64_t k) { const double a = re[static_cast<std::size_t>(c * T + k)], b = im[static_cast<std::size_t>(c * T + k)]; return static_cast<double>(static_cast<float>(std::sqrt(a * a + b * b))); };
    std::vector<double> first;
    for (const char* rec : {"rest", "rest_b"}) {
      Json one;
      for (const auto& x : rm) if (x.value("anchor", std::string()) == rec) one = x;
      CHECK(!one.is_null());
      const Json md = out_of(one, "med");
      CHECK(!md.is_null());
      const auto fld = tsub.find({.cls = tk::Cls::field, .name = rec, .any_status = true});
      const MeasurementValues a = values_of(md["over"].get<std::string>(), fld->id, "median");
      double worst = 0, scale = 0;
      for (int c = 0; c < M; ++c)
        for (std::int64_t t = 0; t < T / W; ++t) {
          std::vector<double> v;
          for (std::int64_t k = t * W; k < (t + 1) * W; ++k) v.push_back(env(c, k));
          const double want = nxr::stats::median(v), got = a.values[static_cast<std::size_t>(c * a.members + t)];
          worst = std::max(worst, std::abs(got - want)); scale = std::max(scale, want);
        }
      std::printf("  median per tile (%s) vs the direct median of |z|: %.2e (of %.2e)\n", rec, worst, scale);
      CHECK(worst < 1e-5 * scale);
      if (first.empty()) first = a.values; else CHECK(a.values == first);            // the chunk width changes nothing
    }
  }
  return nxrtest::finish("test_band_bursts");
}
