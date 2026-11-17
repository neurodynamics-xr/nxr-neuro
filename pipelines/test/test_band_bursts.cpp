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
#include "nxr/filterbank.h"
#include "nxr/io.h"
#include "nxr/pipelines/constant_q.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/statistics.h"
#include "nxr/stats.h"
#include "nxr/taps.h"

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
    sub.compose([&] { nxr::pipelines::constant_q_write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
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
  const Json am = out_of(ra, "amplitude");
  CHECK(am["measurement"] == "amplitude");
  const auto tsub = tk::Subject::of(sub);
  const nxr::io::testing::Probe probe(sub);
  const auto lvl = tsub.find({.cls = tk::Cls::selection, .name = am["over"].get<std::string>(), .any_status = true});
  CHECK(lvl.has_value());
  const Json lp = lvl->params;
  CHECK(lp["resolves_hz"]["lo"] == 8.0 && lp["resolves_hz"]["hi"] == 16.0 && lp["window_samples"] == 1024);
  const std::int64_t TT = lp["tiling"]["count"].get<std::int64_t>(), W = 1024;
  std::vector<std::int64_t> shape;
  const auto arr = nxrtest::read_packed(sub.folder(), am, &shape);
  CHECK(shape == (std::vector<std::int64_t>{M, TT, 2}));
  const auto rest = tsub.find({.cls = tk::Cls::field, .name = "rest", .any_status = true});
  // the rows of one (selection, field, measure) in the measurement table
  auto measured = [&](const std::string& selection_id, const std::string& measure, const std::string& order_by = "") {
    return probe.rows("selection_measurement", {{"selection_id", V(selection_id)}, {"of_field_id", V(rest->id)}, {"measure", V(measure)}}, order_by);
  };
  std::vector<Row> rows;
  for (const char* m : {"amplitude", "count"}) for (const Row& r : measured(lvl->id, m)) rows.push_back(r);
  std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return integer(a, "component") < integer(b, "component"); });
  CHECK(rows.size() == 2 && text(rows[0], "measure") == "amplitude" && text(rows[1], "measure") == "count" && opt_text(rows[0], "computed_by").value_or("").rfind("pipeline band_bursts@1", 0) == 0);
  // the band's peak per tile is stored beside it on the same level and shares the count row
  CHECK_EQ(measured(lvl->id, "envelope_max").size(), std::size_t(1));
  CHECK_EQ(measured(lvl->id, "count").size(), std::size_t(1));
  {
    const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 16.0); };
    const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
    const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
    double worst = 0, scale = 0;
    for (int c = 0; c < M; ++c)
      for (std::int64_t t = 0; t < T / W; ++t) {
        double s = 0;
        for (std::int64_t k = t * W; k < (t + 1) * W; ++k) { const double a = re[static_cast<std::size_t>(c * T + k)], b = im[static_cast<std::size_t>(c * T + k)]; s += static_cast<float>(std::sqrt(a * a + b * b)); }
        const double want = s / W, got = arr[static_cast<std::size_t>((c * TT + t) * 2)];
        worst = std::max(worst, std::abs(got - want));
        scale = std::max(scale, want);
        CHECK_EQ(static_cast<std::int64_t>(arr[static_cast<std::size_t>((c * TT + t) * 2 + 1)]), W);
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
      const StoredStatistic st = read_statistic(sub, bl["statistic"].get<std::string>());
      CHECK(text(st.row, "method") == "summary" && text(st.row, "measure") == "amplitude" && st.shape == (std::vector<std::int64_t>{M, 2}));
      CHECK(st.axes[1]["values"][0]["name"] == "median" && st.axes[1]["values"][1]["name"] == "mad" && text(st.row, "partition_selection_id") == lvl->id);
      // the baseline is the median and robust SD of the stored amplitude tiles (channel 2: no burst)
      std::vector<double> c2;
      for (std::int64_t t = 0; t < TT; ++t) { const float v = arr[static_cast<std::size_t>((2 * TT + t) * 2)]; if (std::isfinite(v)) c2.push_back(v); }
      CHECK(std::abs(st.values[2 * 2] - nxr::stats::median(c2)) < 1e-12 && std::abs(st.values[2 * 2 + 1] - nxr::stats::mad(c2)) < 1e-12);
      // the same from the STORED amplitude by the verb (statistic compute, array-backed rows): the run's baseline
      nxr::pipelines::StatisticComputeOptions so;
      so.over = lvl->name; so.of = "rest"; so.measure = "amplitude"; so.summaries = {"median", "mad"}; so.name = "rest_alpha_baseline";
      const auto vr = nxr::pipelines::statistic_compute(ds, sub, so);
      const StoredStatistic vs = read_statistic(sub, "rest_alpha_baseline");
      CHECK(vr.info["rows"] == M && vs.values == st.values && vs.axes[0].contains("manifold_id"));
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
  // PER-BURST MEASUREMENTS (docs/specs/2026-11-13-burst-statistics.md): each burst measured from the tiles it covers, stored over the run's own bursts Selection,
  // [rows x bursts x 2], on the row of the channel that found it; equal to the direct values over the burst's samples
  {
    const std::int64_t TW = 1024;
    const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 16.0); };
    const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
    const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
    auto env = [&](int c, std::int64_t k) { const double a = re[static_cast<std::size_t>(c * T + k)], b = im[static_cast<std::size_t>(c * T + k)]; return static_cast<double>(static_cast<float>(std::sqrt(a * a + b * b))); };
    const auto evs = probe.rows("selection_span", {{"selection_id", V(bs["id"].get<std::string>())}}, "ordinal");
    const Json params = tsub.require_id(tk::Cls::selection, bs["id"].get<std::string>()).params;
    const std::int64_t E = static_cast<std::int64_t>(evs.size());
    CHECK_EQ(E, std::int64_t(3));
    const double wt = static_cast<double>(TW) / sfreq;
    auto matrix = [&](const char* output, std::vector<std::int64_t>* shape = nullptr) {
      const Json m = out_of(ra, output);
      CHECK(!m.is_null() && m["rows"] == M);
      return nxrtest::read_packed(sub.folder(), m, shape);
    };
    std::vector<std::int64_t> sh;
    const auto A = matrix("burst_amplitude", &sh);
    CHECK(sh == (std::vector<std::int64_t>{M, E, 2}));
    const auto P = matrix("burst_peak"), D = matrix("burst_duration"), I = matrix("burst_interval");
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
        const auto at = [&](const std::vector<float>& v) { return v[static_cast<std::size_t>((r * E + k) * 2)]; };
        if (r == c) {
          worst = std::max({worst, std::abs(at(A) - want_a), std::abs(at(P) - pk)});
          CHECK(std::abs(at(D) - static_cast<double>(e - f) * wt) < 1e-5);
          CHECK_EQ(static_cast<std::int64_t>(A[static_cast<std::size_t>((r * E + k) * 2 + 1)]), (e - f) * TW);
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
      CHECK(std::abs(I[static_cast<std::size_t>((0 * E + first0) * 2)] - (s1 - e0)) < 1e-4 && std::isnan(I[static_cast<std::size_t>((0 * E + second0) * 2)]));
      for (std::int64_t k = 0; k < E; ++k) if (ch[static_cast<std::size_t>(k)] == 1) CHECK(std::isnan(I[static_cast<std::size_t>((1 * E + k) * 2)]));
      gap_by[0].push_back(s1 - e0);
    }
    // read back by the lookup as [rows x bursts]
    {
      const auto sid_ev = bs["id"].get<std::string>();
      const MeasurementValues mv = measurement_values(sub, sid_ev, rest->id, std::nullopt, std::nullopt, "amplitude");
      CHECK(mv.rows == M && mv.members == E && mv.array_backed);
      CHECK(std::abs(mv.values[static_cast<std::size_t>(ch[0] * E)] - amp_by[static_cast<std::size_t>(ch[0])][0]) < 1e-5);
    }
    // THE STATISTICS ACROSS THE BURSTS, per channel: count, sum, median, mean, sd, percentiles 25, 75, 90 (the default); channel 2 has none: count and sum 0
    {
      auto stat = [&](const char* output) { return read_statistic(sub, out_of(ra, output)["statistic"].get<std::string>()); };
      const StoredStatistic sd_ = stat("burst_duration_stats"), sa_ = stat("burst_amplitude_stats"), sp_ = stat("burst_peak_stats"), si_ = stat("burst_interval_stats");
      CHECK(sd_.shape == (std::vector<std::int64_t>{M, 8}));
      CHECK(sd_.axes[1]["values"][0]["name"] == "count" && sd_.axes[1]["values"][1]["name"] == "sum" && sd_.axes[1]["values"][7]["p"] == 90.0);
      for (int c = 0; c < M; ++c) {
        const auto& d = dur_by[static_cast<std::size_t>(c)];
        const double n = static_cast<double>(d.size());
        CHECK(sd_.values[static_cast<std::size_t>(c * 8)] == n && sa_.values[static_cast<std::size_t>(c * 8)] == n);
        double tot = 0; for (double v : d) tot += v;
        CHECK(std::abs(sd_.values[static_cast<std::size_t>(c * 8 + 1)] - tot) < 1e-5);
        if (d.empty()) { CHECK(std::isnan(sd_.values[static_cast<std::size_t>(c * 8 + 2)])); continue; }
        CHECK(std::abs(sd_.values[static_cast<std::size_t>(c * 8 + 2)] - nxr::stats::median(d)) < 1e-5 && std::abs(sd_.values[static_cast<std::size_t>(c * 8 + 3)] - nxr::stats::mean(d)) < 1e-5);
        CHECK(std::abs(sa_.values[static_cast<std::size_t>(c * 8 + 2)] - nxr::stats::median(amp_by[static_cast<std::size_t>(c)])) < 1e-5);
        CHECK(std::abs(sp_.values[static_cast<std::size_t>(c * 8 + 3)] - nxr::stats::mean(peak_by[static_cast<std::size_t>(c)])) < 1e-5);
        CHECK(std::abs(sa_.values[static_cast<std::size_t>(c * 8 + 7)] - nxr::stats::percentile(amp_by[static_cast<std::size_t>(c)], 90)) < 1e-5);
      }
      CHECK(si_.values[0] == 1 && std::abs(si_.values[1] - gap_by[0][0]) < 1e-4 && si_.values[8] == 0 && si_.values[9] == 0);   // one interval on channel 0; none on 1 and 2
      // the span and its good seconds are recorded: rate, occupancy and CV are derived at read
      const Json opt = Json::parse(text(sd_.row, "options_json"));
      CHECK(std::abs(opt["good_s"].get<double>() - T / sfreq) < 1e-9 && opt["span"]["stop_sample"] == T);
      CHECK(text(sd_.row, "partition_selection_id") == bs["id"].get<std::string>());
    }
  }
  // 2. ANOTHER CHUNK WIDTH: the same tiles and bursts
  {
    const Json rb = result("rest_b");
    const auto arr_b = nxrtest::read_packed(sub.folder(), out_of(rb, "amplitude"));
    double worst = 0;
    for (std::size_t i = 0; i < arr.size(); i += 2) if (std::isfinite(arr[i])) worst = std::max(worst, static_cast<double>(std::abs(arr_b[i] - arr[i]) / std::max(1e-6f, arr[i])));
    std::printf("  chunk width 32768 vs 3000: %.2e\n", worst);
    CHECK(worst < 1e-4);
    CHECK(out_of(rb, "bursts")["spans"] == bs["spans"]);
  }
  // 3. A BAD SPAN [40, 40.5) s: tiles 46 and 47 NaN, their counts short
  {
    const auto arr_c = nxrtest::read_packed(sub.folder(), out_of(result("rest_c"), "amplitude"));
    CHECK(std::isnan(arr_c[46 * 2]) && std::isnan(arr_c[47 * 2]) && std::isfinite(arr_c[45 * 2]) && std::isfinite(arr_c[48 * 2]));
    CHECK(arr_c[46 * 2 + 1] < W && arr_c[47 * 2 + 1] < W);
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
    const Json ba = out_of(rm, "burst_amplitude");
    std::vector<std::int64_t> shm;
    const auto Am = nxrtest::read_packed(sub.folder(), ba, &shm);
    CHECK(shm == (std::vector<std::int64_t>{M, 3, 2}));
    for (int r = 0; r < M; ++r) for (int k = 0; k < 3; ++k) CHECK(std::isfinite(Am[static_cast<std::size_t>((r * 3 + k) * 2)]));   // each row on its own tiles
    const StoredStatistic cs = read_statistic(sub, out_of(rm, "burst_duration_stats")["statistic"].get<std::string>());
    CHECK(cs.shape == (std::vector<std::int64_t>{1, 8}) && cs.values[0] == 3);
  }
  // 5. THE MEDIAN of the band's envelope per tile (exact; numpy's), held per tile until it closes: equal to the direct median of |z|, whatever the chunk width;
  //    and over the run's whole span (no `over`), per row
  {
    const std::string yaml =
        "pipeline: band_median\nversion: 1\ndescription: test\nscope: subject\n"
        "inputs:\n  recording: {field: recording, each: true}\n  ladder: {selection: ladder, manifold: recording.time}\n"
        "  bad: {selection: spans, manifold: recording.time, where: {name: \"{recording}_bad_segments\"}, required: false}\n"
        "parameters:\n  band: [8.0, 16.0]\nsteps:\n  chunks:\n    over: recording\n    do:\n"
        "      - {step: envelope, in: {x: recording}, with: {band: $band}, out: amp}\n"
        "      - {step: measure, in: {x: amp, over: ladder, exclude: bad}, with: {measure: median}, out: med}\n"
        "      - {step: measure, in: {x: amp, exclude: bad}, with: {measure: median}, out: whole}\n"
        "outputs:\n  med: {kind: measurement}\n  whole: {kind: median envelope}\n";
    const g::Definition dm = g::parse(parse_yaml(yaml, "test.yaml"), "test.yaml", yaml);
    const auto rm = g::run(ds, sub, dm, Json::object(), all);
    const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 16.0); };
    const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
    const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
    auto env = [&](int c, std::int64_t k) { const double a = re[static_cast<std::size_t>(c * T + k)], b = im[static_cast<std::size_t>(c * T + k)]; return static_cast<double>(static_cast<float>(std::sqrt(a * a + b * b))); };
    std::vector<float> first;
    for (const char* rec : {"rest", "rest_b"}) {
      Json one;
      for (const auto& x : rm) if (x.value("anchor", std::string()) == rec) one = x;
      CHECK(!one.is_null());
      Json tilesarr; std::string wholearr;
      for (const auto& x : one["stored"]) { if (x["output"] == "med") tilesarr = x; if (x["output"] == "whole") wholearr = x["path"].get<std::string>(); }
      CHECK(!tilesarr.is_null() && !wholearr.empty());
      std::vector<std::int64_t> sh;
      const auto a = nxrtest::read_packed(sub.folder(), tilesarr, &sh);
      const std::int64_t TT = sh[1], W = 1024;
      double worst = 0, scale = 0;
      for (int c = 0; c < M; ++c)
        for (std::int64_t t = 0; t < T / W; ++t) {
          std::vector<double> v;
          for (std::int64_t k = t * W; k < (t + 1) * W; ++k) v.push_back(env(c, k));
          const double want = nxr::stats::median(v), got = a[static_cast<std::size_t>((c * TT + t) * 2)];
          worst = std::max(worst, std::abs(got - want)); scale = std::max(scale, want);
        }
      std::printf("  median per tile (%s) vs the direct median of |z|: %.2e (of %.2e)\n", rec, worst, scale);
      CHECK(worst < 1e-5 * scale);
      if (first.empty()) first = a; else CHECK(a == first);            // the chunk width changes nothing
      const auto wh = read_floats((sub.folder() / (wholearr)), &sh);
      for (int c = 0; c < M; ++c) {
        std::vector<double> v;
        for (std::int64_t k = 0; k < T; ++k) v.push_back(env(c, k));
        CHECK(std::abs(wh[static_cast<std::size_t>(c)] - nxr::stats::median(v)) < 1e-5 * scale);
      }
    }
  }
  // 6. THE COUNT ROW is one per (level, recording): the amplitude run and the median run (same span) share it; a measure over another span is refused, --replace rewrites it
  {
    auto counts = [&] { std::size_t n = 0; for (const Row& r : measured(lvl->id, "count")) n += !is_null(r, "array_path"); return n; };
    auto nmeasure = [&](const char* m) { return measured(lvl->id, m).size(); };
    CHECK(nmeasure("amplitude") == 1 && nmeasure("median") == 1 && counts() == 1);
    const std::string yaml =
        "pipeline: band_power_tiles\nversion: 1\ndescription: test\nscope: subject\n"
        "inputs:\n  recording: {field: recording, each: true}\n  ladder: {selection: ladder, manifold: recording.time}\n"
        "  bad: {selection: spans, manifold: recording.time, where: {name: \"{recording}_bad_segments\"}, required: false}\n"
        "parameters:\n  band: [8.0, 16.0]\nsteps:\n  chunks:\n    over: recording\n    do:\n"
        "      - {step: analytic, in: {x: recording}, with: {band: $band}, out: z}\n"
        "      - {step: power, in: {x: z}, out: pw}\n"
        "      - {step: measure, in: {x: pw, over: ladder, exclude: bad}, with: {measure: power}, out: pwr}\n"
        "outputs:\n  pwr: {kind: measurement}\n";
    const g::Definition dp = g::parse(parse_yaml(yaml, "test.yaml"), "test.yaml", yaml);
    g::RunOptions one;
    one.chunks = "0";
    const std::string msg = [&] { try { g::run(ds, sub, dp, Json::object(), one); } catch (const Error& e) { CHECK_EQ(e.code(), std::string(code::already_exists)); return std::string(e.what()); } return std::string(); }();
    CHECK(msg.find("another span") != std::string::npos);
    CHECK(nmeasure("power") == 0 && counts() == 1);
    one.replace = true;
    // amplitude and median share the count row: --replace over another span is refused, naming them; nothing is deleted
    const std::string msg2 = [&] { try { g::run(ds, sub, dp, Json::object(), one); } catch (const Error& e) { CHECK_EQ(e.code(), std::string(code::already_exists)); return std::string(e.what()); } return std::string(); }();
    CHECK(msg2.find("amplitude") != std::string::npos && msg2.find("median") != std::string::npos && msg2.find("envelope_max") != std::string::npos && msg2.find("another span") != std::string::npos);
    CHECK(nmeasure("power") == 0 && nmeasure("amplitude") == 1 && nmeasure("median") == 1 && counts() == 1);
    // with those removed (by the caller), the power measure is the sole one: the count row is rewritten to its span
    std::set<std::string> measured_selections;
    for (const char* m : {"amplitude", "median", "envelope_max"}) for (const Row& r : probe.rows("selection_measurement", {{"measure", V(std::string(m))}})) measured_selections.insert(text(r, "selection_id"));
    for (const std::string& sid : measured_selections)
      for (const char* m : {"amplitude", "median", "envelope_max"}) tsub.remove_measurements(sid, std::nullopt, std::string(m));
    const auto rp = g::run(ds, sub, dp, Json::object(), one);
    bool cr = false;
    for (const auto& x : rp) for (const auto& st : x["stored"]) cr = cr || st.value("count_replaced", false);
    CHECK(cr && nmeasure("power") == 1 && counts() == 1);
  }
  return nxrtest::finish("test_band_bursts");
}
