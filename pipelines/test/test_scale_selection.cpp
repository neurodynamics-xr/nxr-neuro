// scale selection (spec docs/specs/2026-11-13-cortical-events.md, "Phase 2e") on a synthetic source space (never a real dataset): `run cortical_event_pairs` over the product of two spatial levels (3, 4) and two time bands
// (4-8 Hz, 8-16 Hz), then `scaleselection compute`. Two episodes, in different hemispheres and at different times, each with its OWN size and rhythm: A is the level-3 pattern of one left tile at 6 Hz, B the level-4 pattern of
// one right tile at 12 Hz. Checks: the stored cells, baselines and labellings of every pair (the per-pair extents are `cell_extent`'s, unchanged, each against its pair's own bursts: the bursts' name is the pair's own);
// the cross-pair events are the episodes (every pair's events at each episode's time are ONE cross-pair event, linked by the ladders' nesting and time overlap); A's best pair is (level 3, 4-8 Hz) and B's (level 4, 8-16 Hz);
// the footprint, the response and the events' facts against an independent computation from the stored labellings and cells (z against the tile's own median and robust SD); the tile area, the band, the pair count and
// the edge flags; the span of each event; scale_event over each pair's labelling; a rerun a hit; another option refused without --replace and made with it; min_link large splits the events; the mean response; the refusals
// (a missing pair, a missing run, a labelling of another pipeline, a repeated level, unknown recording and response).
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/joint.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/scaleselection.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class F> std::string why_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
template <class T> std::vector<T> read_vec(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}
Json out_of(const Json& r, const std::string& o) { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_scale_selection_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 12288;
  const double sfreq = 250.0;
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
  const SurfaceInput S0 = read_surface(sub, std::nullopt, true);
  // the time levels of the two bands: the constant-Q levels that resolve 4-8 Hz and 8-16 Hz
  int tl_slow = -1, tl_fast = -1;
  std::int64_t w_slow = 0, w_fast = 0;
  const task::Subject tsub = task::Subject::of(sub);
  for (const task::Node& m : tsub.find_all({.cls = task::Cls::selection, .params_present = {"resolves_hz"}})) {
    if (!m.member_of_id) continue;
    const Json& pj = m.params;
    if (pj["resolves_hz"]["lo"] == 4.0 && pj["resolves_hz"]["hi"] == 8.0) { tl_slow = static_cast<int>(m.member_ordinal.value_or(0)); w_slow = pj["window_samples"].get<std::int64_t>(); }
    if (pj["resolves_hz"]["lo"] == 8.0 && pj["resolves_hz"]["hi"] == 16.0) { tl_fast = static_cast<int>(m.member_ordinal.value_or(0)); w_fast = pj["window_samples"].get<std::int64_t>(); }
  }
  CHECK(tl_slow >= 0 && tl_fast >= 0);
  if (tl_slow < 0 || tl_fast < 0) return 1;
  std::printf("  time levels: 4-8 Hz is level %d (%lld samples), 8-16 Hz level %d (%lld samples)\n", tl_slow, static_cast<long long>(w_slow), tl_fast, static_cast<long long>(w_fast));
  const std::vector<int> levels = {3, 4};
  const std::vector<int> tlevels = {tl_fast, tl_slow};                         // the declared order: the 8-16 Hz level first
  const int SP = 2, TP = 2;                                                    // spatial and time level counts: the pairs in the declared order (space level major)

  // THE PLANTED EPISODES. A (left hemisphere): the level-3 pattern of tile 2 at 6 Hz, 8 s to 17 s. B (right hemisphere): the level-4 pattern of tile 16 + 5 at 12 Hz, 30 s to 35 s.
  auto centres_of = [&](int level) {
    const SurfaceLevel sl = read_surface_level(sub, S0, level);
    ArrayReader r = ArrayReader::open(sub.store(), sl.record["node"].get<std::string>() + "/lc/L" + std::to_string(level) + "/centre");
    const auto raw = r.read();
    std::vector<std::int32_t> c(raw.size() / 4);
    std::memcpy(c.data(), raw.data(), raw.size());
    return c;
  };
  auto pattern = [&](int level, int tile) {
    const SurfaceLevel sl = read_surface_level(sub, S0, level);
    SpatialBandOptions o; o.transform = "exact";
    const SpatialBand sb = make_spatial_band(S0, sl, o);
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(N, 1);
    e(centres_of(level)[static_cast<std::size_t>(tile)], 0) = 1.0;
    Eigen::MatrixXd f = sb.apply(e);
    return Eigen::VectorXd(f.col(0) / f.col(0).cwiseAbs().maxCoeff());
  };
  const Eigen::VectorXd pA = pattern(3, 2), pB = pattern(4, 16 + 5);
  auto S_ = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double s = t / sfreq;
      double x = 0.005 * nd(rng);
      x += pA(v) * S_((s - 8.0) / 0.3) * S_(-(s - 17.0) / 0.4) * std::cos(2 * M_PI * 6.0 * s);
      x += pB(v) * S_((s - 30.0) / 0.25) * S_(-(s - 35.0) / 0.35) * std::cos(2 * M_PI * 12.0 * s);
      xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>(x);
    }
  {
    SubjectWriter w = SubjectWriter::open(ds, "S1");
    w.compose([&] {
      FieldSpec rec;
      const std::string prod = w.product({cid_s, tid});
      rec.name = "rest"; rec.path = task::Subject::of(w).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, std::int64_t(4096)}; rec.array.compress = false; rec.cols.set("unit", "A");
      w.field(std::move(rec));
    });
  }
  for (const int l : levels)
    for (const int m : tlevels) { JointComputeOptions jo; jo.recording = "rest"; jo.space_level = l; jo.time_level = m; joint_compute(ds, sub, jo); }

  // ── THE RUN: every pair's cells, global bursts and extents, one pass ──
  const g::Definition d = g::load("cortical_event_pairs");
  g::RunOptions all; all.chunks = "all";
  Json set1 = Json{{"space_levels", Json::array({3, 4})}, {"bands", Json::array({Json::array({8.0, 16.0}), Json::array({4.0, 8.0})})}, {"transform", "exact"}, {"extent_enter_sd", 30.0}, {"extent_exit_sd", 15.0}, {"min_cells", 2}};
  const auto r1 = g::run(ds, sub, d, set1, all);
  CHECK(r1.size() == 1 && r1[0]["hit"] == false);
  const std::vector<std::string> band_name = {"8_16", "4_8"};
  const std::int64_t windows[2] = {w_fast, w_slow};
  struct PairData { int l, m, bi; std::int64_t S, F, W; std::vector<float> amp; std::vector<std::int32_t> lab; Eigen::MatrixXd z; Json lab_out; int events; };
  std::vector<PairData> pd;
  for (int a = 0; a < SP; ++a)
    for (int bi = 0; bi < TP; ++bi) {
      PairData p; p.l = levels[static_cast<std::size_t>(a)]; p.m = tlevels[static_cast<std::size_t>(bi)]; p.bi = bi; p.W = windows[bi];
      const std::string tag = "l" + std::to_string(p.l) + "_b" + band_name[static_cast<std::size_t>(bi)];
      std::vector<std::int64_t> sh;
      p.amp = nxrtest::read_packed(sub.folder(), out_of(r1[0], "amplitude_" + tag), &sh);
      p.S = sh[0]; p.F = sh[1];
      p.lab_out = out_of(r1[0], "extents_" + tag);
      CHECK(!p.lab_out.is_null() && !out_of(r1[0], "bursts_" + tag).is_null() && !out_of(r1[0], "cell_baseline_" + tag).is_null());
      p.lab = read_vec<std::int32_t>((sub.folder() / (p.lab_out["path"].get<std::string>())), &sh);
      CHECK(sh == (std::vector<std::int64_t>{p.S, p.F}));
      p.events = p.lab_out["events"].get<int>();
      // the independent z: the tile's median and robust SD over the run's time tiles
      Eigen::MatrixXd a2(p.S, p.F);
      for (int s = 0; s < p.S; ++s) for (int f = 0; f < p.F; ++f) a2(s, f) = p.amp[static_cast<std::size_t>((s * p.F + f) * 2)];
      p.z = Eigen::MatrixXd::Constant(p.S, p.F, std::nan(""));
      auto median = [](std::vector<double> x) { std::sort(x.begin(), x.end()); const std::size_t n = x.size(); return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]); };
      for (int s = 0; s < p.S; ++s) {
        std::vector<double> x;
        for (int f = 0; f < p.F; ++f) if (std::isfinite(a2(s, f))) x.push_back(a2(s, f));
        const double med = median(x);
        std::vector<double> dv;
        for (const double v : x) dv.push_back(std::abs(v - med));
        const double sd = 1.4826 * median(dv);
        for (int f = 0; f < p.F; ++f) if (sd > 0 && std::isfinite(a2(s, f))) p.z(s, f) = (a2(s, f) - med) / sd;
      }
      std::printf("  pair (level %d, %s Hz): %lld x %lld cells, %d events\n", p.l, band_name[static_cast<std::size_t>(bi)].c_str(), static_cast<long long>(p.S), static_cast<long long>(p.F), p.events);
      pd.push_back(std::move(p));
    }
  // each labelling names the bursts of ITS pair
  for (const auto& p : pd) {
    const Json pj = tsub.require_id(task::Cls::selection, p.lab_out["id"].get<std::string>()).params;
    CHECK(pj["bursts"].is_object() && pj["bursts"]["output"].get<std::string>() == "bursts_l" + std::to_string(p.l) + "_b" + band_name[static_cast<std::size_t>(p.bi)]);
  }

  // ── SCALE SELECTION ──
  ScaleSelectionOptions so;
  so.recording = "rest"; so.space_levels = levels; so.time_levels = tlevels;
  auto run_verb = [&](const ScaleSelectionOptions& o) { return scaleselection_compute(ds, sub, o); };
  const auto v1 = run_verb(so);
  CHECK(!v1.hit);
  const std::int64_t E = v1.info["events"].get<std::int64_t>();
  const std::string sid = v1.info["id"].get<std::string>();
  const int P = SP * TP;
  CHECK_EQ(E, 2);
  auto measure = [&](const std::string& selection, const std::string& name, std::vector<std::int64_t>& shape, int& comp) {
    const std::string rest_id = tsub.require({.cls = task::Cls::field, .name = "rest"}, "the recording").id;
    const auto found = testing::Probe(sub).rows("selection_measurement", {{"selection_id", selection}, {"of_field_id", rest_id}, {"measure", name}});
    const auto row = found.empty() ? std::optional<Row>() : std::optional<Row>(found.front());
    CHECK(row.has_value());
    if (!row) return std::vector<double>();
    comp = static_cast<int>(integer(*row, "component"));
    return read_vec<double>((sub.folder() / (text(*row, "array_path"))), &shape);
  };
  // the planted episodes' time windows in samples, and each pair's events' facts, independent of nxr::scale (from the stored labellings and cells)
  const std::array<std::pair<double, double>, 2> episode = {std::pair<double, double>{8.0 * sfreq, 17.0 * sfreq}, std::pair<double, double>{30.0 * sfreq, 35.0 * sfreq}};
  struct Fact { double peak = -1e300, mean = 0; int cells = 0; std::int64_t first = 1 << 30, stop = -1; int episode = -1; };
  std::vector<std::vector<Fact>> facts(static_cast<std::size_t>(P));
  for (int a = 0; a < P; ++a) {
    const PairData& p = pd[static_cast<std::size_t>(a)];
    facts[static_cast<std::size_t>(a)].resize(static_cast<std::size_t>(p.events));
    for (int s = 0; s < p.S; ++s)
      for (int f = 0; f < p.F; ++f) {
        const int e = p.lab[static_cast<std::size_t>(s * p.F + f)];
        if (e < 0) continue;
        Fact& x = facts[static_cast<std::size_t>(a)][static_cast<std::size_t>(e)];
        x.peak = std::max(x.peak, p.z(s, f)); x.mean += p.z(s, f); ++x.cells;
        x.first = std::min<std::int64_t>(x.first, f * p.W); x.stop = std::max<std::int64_t>(x.stop, (f + 1) * p.W);
      }
    for (auto& x : facts[static_cast<std::size_t>(a)]) {
      x.mean /= x.cells;
      for (int k = 0; k < 2; ++k) if (x.first < episode[static_cast<std::size_t>(k)].second && episode[static_cast<std::size_t>(k)].first < x.stop) x.episode = k;
    }
    CHECK(p.events == 1 && facts[static_cast<std::size_t>(a)][0].episode >= 0);
  }
  // the pair of each episode: the planted one has an event at its own pair; the others, if any, are other pairs' events of the same episode
  std::printf("  pair events: ");
  for (int a = 0; a < P; ++a) std::printf("[pair %d: episode %d, peak z %.1f, cells %d] ", a, facts[static_cast<std::size_t>(a)][0].episode, facts[static_cast<std::size_t>(a)][0].peak, facts[static_cast<std::size_t>(a)][0].cells);
  std::printf("\n");
  // MATCHING: the cross-pair event of each pair's event (scale_event over the pair's labelling) is the episode's
  std::vector<int> cross_of_episode = {-1, -1};
  for (int a = 0; a < P; ++a) {
    std::vector<std::int64_t> sh; int c = 0;
    const auto cx = measure(pd[static_cast<std::size_t>(a)].lab_out["id"].get<std::string>(), "scale_event", sh, c);
    CHECK(sh == (std::vector<std::int64_t>{1, 1, 1}) && cx.size() == 1);
    const int ep = facts[static_cast<std::size_t>(a)][0].episode;
    if (cross_of_episode[static_cast<std::size_t>(ep)] < 0) cross_of_episode[static_cast<std::size_t>(ep)] = static_cast<int>(cx[0]);
    CHECK_EQ(static_cast<int>(cx[0]), cross_of_episode[static_cast<std::size_t>(ep)]);        // the same episode at every pair: ONE cross-pair event
  }
  CHECK(cross_of_episode[0] >= 0 && cross_of_episode[1] >= 0 && cross_of_episode[0] != cross_of_episode[1]);
  CHECK_EQ(cross_of_episode[0], 0);                                                           // numbered by the earliest sample: A (6 s) before B (28 s)
  CHECK_EQ(cross_of_episode[1], 1);
  // THE EVENTS' MEASURES (one array [1 x E x 8]) and the footprint [P x E x 1]
  std::vector<std::int64_t> shE, shF; int cE = 0, cF = 0;
  const auto spaceLevel = measure(sid, "scale_space_level", shE, cE);
  CHECK(shE == (std::vector<std::int64_t>{1, E, 8}) && cE == 0);
  const auto at = [&](const std::vector<double>& v, int comp, int k) { return v[static_cast<std::size_t>(k * 8 + comp)]; };
  const auto foot = measure(sid, "scale_response", shF, cF);
  CHECK(shF == (std::vector<std::int64_t>{P, E, 1}));
  std::printf("  footprint (rows: pairs (3,8-16) (3,4-8) (4,8-16) (4,4-8); columns: A B)\n");
  for (int a = 0; a < P; ++a) std::printf("    pair %d: %10.1f %10.1f\n", a, foot[static_cast<std::size_t>(a * E + 0)], foot[static_cast<std::size_t>(a * E + 1)]);
  const int bestExpected[2] = {1, 2};                                                          // A: level 3 x 4-8 Hz (declared pair 1); B: level 4 x 8-16 Hz (declared pair 2)
  for (int k = 0; k < 2; ++k) {
    int best = -1; double mx = -1e300, second = -1e300;
    for (int a = 0; a < P; ++a) {
      const double ind = facts[static_cast<std::size_t>(a)][0].episode == k ? facts[static_cast<std::size_t>(a)][0].peak : std::nan("");
      const double got = foot[static_cast<std::size_t>(a * E + k)];
      CHECK(std::isnan(ind) == std::isnan(got) && (std::isnan(ind) || std::abs(got - ind) < 1e-4 * std::abs(ind)));   // the footprint = each pair's event's peak z, NaN where none
      if (std::isfinite(ind)) { if (ind > mx) { second = mx; mx = ind; best = a; } else if (ind > second) second = ind; }
    }
    CHECK_EQ(best, bestExpected[k]);
    CHECK(mx > 2 * second || second < -1e290);                                                // the planted pair is clearly the largest
    CHECK_EQ(static_cast<int>(at(spaceLevel, 0, k)), pd[static_cast<std::size_t>(best)].l);
    CHECK_EQ(static_cast<int>(at(spaceLevel, 1, k)), pd[static_cast<std::size_t>(best)].m);
    CHECK(std::abs(at(spaceLevel, 2, k) - mx) < 1e-4 * mx);
    const SurfaceLevel sl = read_surface_level(sub, S0, pd[static_cast<std::size_t>(best)].l);
    double area = 0;
    for (const double x : sl.tile_area) area += x;
    CHECK(std::abs(at(spaceLevel, 3, k) - area / static_cast<double>(sl.nodes)) < 1e-9 * area);       // the mean tile area of the best level (m2)
    CHECK(at(spaceLevel, 4, k) == (pd[static_cast<std::size_t>(best)].bi == 0 ? 8.0 : 4.0) && at(spaceLevel, 5, k) == (pd[static_cast<std::size_t>(best)].bi == 0 ? 16.0 : 8.0));
    int n = 0;
    for (int a = 0; a < P; ++a) n += facts[static_cast<std::size_t>(a)][0].episode == k;
    CHECK_EQ(static_cast<int>(at(spaceLevel, 6, k)), n);
  }
  CHECK_EQ(static_cast<int>(at(spaceLevel, 7, 0)), 2 | 8);                                    // A: the coarsest level (2) and the longest window (8) of the set: on the set's edge
  CHECK_EQ(static_cast<int>(at(spaceLevel, 7, 1)), 1 | 4);                                    // B: the finest level (1) and the shortest window (4)
  {   // the spans Selection: the union of the members' time tiles, in seconds
    const auto rows = tsub.spans(tsub.require_id(task::Cls::selection, sid));
    CHECK_EQ(static_cast<int>(rows.size()), 2);
    for (int k = 0; k < 2; ++k) {
      std::int64_t first = 1 << 30, stop = -1;
      for (int a = 0; a < P; ++a) if (facts[static_cast<std::size_t>(a)][0].episode == k) { first = std::min(first, facts[static_cast<std::size_t>(a)][0].first); stop = std::max(stop, facts[static_cast<std::size_t>(a)][0].stop); }
      CHECK(std::abs(rows[static_cast<std::size_t>(k)].start - first / sfreq) < 1e-6 && std::abs(rows[static_cast<std::size_t>(k)].stop - stop / sfreq) < 1e-6);
    }
    const auto sel = tsub.require_id(task::Cls::selection, sid);
    const Json& pj = sel.params;
    CHECK(sel.type == "spans" && pj["scale"]["pairs"].size() == 4 && pj["scale"]["response"] == "peak" && pj["scale"]["pairs"][1]["space_level"] == 3 && pj["scale"]["pairs"][1]["time_level"] == tl_slow && pj["scale"]["warnings"].size() == 3);
  }
  // a rerun is a hit; another option is refused without --replace
  { const auto r = run_verb(so); CHECK(r.hit && r.info["events"].get<int>() == 2); }
  ScaleSelectionOptions mean = so; mean.response = "mean";
  CHECK_EQ(code_of([&] { run_verb(mean); }), std::string(code::already_exists));
  mean.replace = true;
  {   // the MEAN response: the selection is made on the mean z of the events' cells
    const auto r = run_verb(mean);
    CHECK(!r.hit && r.info["events"] == 2);
    std::vector<std::int64_t> sh; int c = 0;
    const auto m2 = measure(r.info["id"].get<std::string>(), "scale_peak", sh, c);
    for (int k = 0; k < 2; ++k) {
      double mx = -1e300;
      for (int a = 0; a < P; ++a) if (facts[static_cast<std::size_t>(a)][0].episode == k) mx = std::max(mx, facts[static_cast<std::size_t>(a)][0].mean);
      CHECK(std::abs(m2[static_cast<std::size_t>(k * 8 + 2)] - mx) < 1e-4 * mx);
    }
  }
  // min_link large: nothing links, so every pair's event stands alone (4 events); a rerun of the first options is a miss under --replace and gives back two
  {
    ScaleSelectionOptions ml = so; ml.min_link = 1000000; ml.replace = true;
    const auto r = run_verb(ml);
    CHECK_EQ(r.info["events"].get<int>(), 4);
    ScaleSelectionOptions back = so; back.replace = true;
    const auto b2 = run_verb(back);
    CHECK(!b2.hit && b2.info["events"] == 2);
    const auto b3 = run_verb(so);
    CHECK(b3.hit);
  }
  // REFUSALS
  {
    ScaleSelectionOptions o = so;
    o.space_levels = {3, 4, 5};
    std::string w = why_of([&] { run_verb(o); });
    CHECK(w.find("no joint Selection") != std::string::npos && w.find("--space-level 5 --time-level " + std::to_string(tl_fast)) != std::string::npos && w.find("--space-level 5 --time-level " + std::to_string(tl_slow)) != std::string::npos);   // EVERY missing pair, its command
    JointComputeOptions jo5; jo5.recording = "rest"; jo5.space_level = 3; jo5.time_level = 4;
    joint_compute(ds, sub, jo5);                                                              // a joint Selection with no run over it
    o = so; o.space_levels = {3}; o.time_levels = {tl_fast, 4};
    w = why_of([&] { run_verb(o); });
    CHECK(w.find("no stored cells, baseline and labelling made by cortical_event_pairs") != std::string::npos && w.find("space level 3 x time level 4") != std::string::npos);
    o = so; o.source = "global_bursts";
    CHECK(why_of([&] { run_verb(o); }).find("made by global_bursts") != std::string::npos);
    o = so; o.space_levels = {3, 3};
    CHECK(why_of([&] { run_verb(o); }).find("listed twice") != std::string::npos);
    o = so; o.recording = "nope";
    CHECK_EQ(code_of([&] { run_verb(o); }), std::string(code::no_such_row));
    o = so; o.response = "max";
    CHECK(why_of([&] { run_verb(o); }).find("peak") != std::string::npos);
    o = so; o.min_link = 0;
    CHECK(why_of([&] { run_verb(o); }).find("min-link") != std::string::npos);
    o = so; o.space_levels.clear();
    CHECK(why_of([&] { run_verb(o); }).find("--space-levels") != std::string::npos);
    o = so; o.time_levels.clear();
    CHECK(why_of([&] { run_verb(o); }).find("--time-levels") != std::string::npos);
  }
  return nxrtest::finish("test_scale_selection");
}
