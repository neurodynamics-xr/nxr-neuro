// THE SPATIAL EVENTS (spec docs/specs/2026-11-14-spatial-events.md, pieces 2-6) on a synthetic sphere (never a real dataset): an identity kernel (a source a channel), white noise everywhere, and inside a window of
// "bursts" two PLANTED BLOBS of different widths in different places (2 cm at one vertex, 5 cm at another), oscillating at 10 Hz. `run spatial_baseline` (the noise outside the window) then `run spatial_events`:
//   - the strongest event of each burst frame is at the planted blob (the narrow one in the first half of the window, the wide one in the second), inside the window; 2 513 of 2 520 frames
//   - the SIZE ORDER: the wide blob's events have the larger characteristic size (size_cm: the level's centre wavelength) and the larger diameter than the narrow blob's
//   - the HOME MAP: each planted vertex lies in a patch in about the half of the burst frames its blob is on
//   (the sphere is small: its finest hat is 18.6 cm, so a patch is wider than the blob; on the cortex, levels reach 1 cm. What the test fixes is the chain's agreement with the plant, not the instrument's resolution.)
//   - stored as point events (a spans Selection of type patch, a member per patch on its frame, the attributes as measurements over it) and a home map statistic [vertex x (frames in a patch, frames)]
//   - the result does not depend on the storage chunk (4096 against 1000) or the block of frames (100 against 256); a rerun is a hit
//   - the DEFAULT threshold (percentile 99 of the relative power outside the window, `run spatial_threshold`) still finds the blobs
//   - the plan: a baseline of another chain (spatial shape), a threshold that is not above 0, `percentile` with no stored distribution are refused
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <map>
#include <set>

#include "check.h"
#include "store_at.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/spatial_null.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"

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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_spatial_events_planted_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const double R = 0.1, sfreq = 250.0, w0 = 5.0, wm = 8.5, w1 = 12.0, fps = 360.0;
  const int T = 4096;
  std::vector<double> U;
  std::vector<std::int32_t> F0;
  icosphere(3, U, F0, R);
  const int N = static_cast<int>(U.size() / 3), M = N;
  const int vA = 0;                                                            // the narrow blob here; the wide one at the vertex farthest from it that is not its antipode's neighbour
  int vB = 0;
  { double best = 1e9;
    for (int v = 0; v < N; ++v) { const double c = (U[3 * v] * U[0] + U[3 * v + 1] * U[1] + U[3 * v + 2] * U[2]) / (R * R); if (std::abs(c - 0.0) < best) { best = std::abs(c); vB = v; } } }   // 90 degrees from A
  const double sA = 0.02, sB = 0.05, amp = 6.0;
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
      SurfaceOptions so; so.name = "cortex"; so.sphere = U;
      surf = add_surface(sub, so, U, F0).id;
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      { const std::vector<std::int64_t> zeros(static_cast<std::size_t>(M), 0);            // the channels' families (the PSD and the cross-spectrum need them): every channel a good MEG MAG
        nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", zeros, {"MEG MAG"}, {}}, {"quality", zeros, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                            [](const std::string& fam) { return std::make_pair(fam, std::string("app")); }); }
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
    sub.compose([&] { nxr::io::constantq::write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  tiles_compute(ds, sub, TilesComputeOptions{});
  for (const char* rec : {"rest", "rest_b"}) { FrameTilingOptions fo; fo.recording = rec; fo.band = std::make_pair(8.0, 12.0); frame_tiling_compute(ds, sub, fo); }
  std::printf("  %d vertices on a 10 cm sphere; blobs 2 cm at vertex %d and 5 cm at vertex %d (%.1f cm apart)\n", N, vA, vB, 100 * ang(vA, vB));

  g::RunOptions all; all.chunks = "all"; all.replace = true;
  const Json lv = Json::array({1, 2, 3, 4});
  const Json set = Json{{"levels", lv}, {"reference_level", 4}, {"threshold", "ratio"}, {"ratio", 100.0}, {"transform", "exact"}, {"bursts", "window"}};
  const Json bset = Json{{"levels", lv}, {"transform", "exact"}, {"bursts", "window"}};
  g::run(ds, sub, g::load("spatial_baseline"), bset, all);
  const g::Definition dev = g::load("spatial_events");
  const auto r = g::run(ds, sub, dev, set, all);
  CHECK(r.size() == 2);
  struct Ev { std::vector<std::int64_t> frame; std::vector<float> cv, peak, size, diam, level, area; Json params; std::vector<double> start; };
  auto read_ev = [&](const std::string& rec) {
    Ev e;
    const auto row = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "spatial_events.events." + rec, .any_status = true});
    if (!row) return e;
    const std::string id = row->id;
    e.params = row->params.is_null() ? Json::object() : row->params;
    for (const auto& s : nxr::io::task::Subject::of(sub).spans(*row)) { e.start.push_back(s.start); e.frame.push_back(std::llround(e.start.back() * fps)); }
    auto meas = [&](const char* name) {
      std::vector<float> out;
      const auto ms = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"selection_id", V(id)}, {"measure", V(name)}});
      if (ms.empty()) return out;
      const auto& m = ms.front();
      ArrayReader rd = ArrayReader::open(sub.store(), text(m, "array_path"));
      const auto sh = rd.shape();
      const auto raw = rd.read();
      std::vector<float> a(raw.size() / 4);
      std::memcpy(a.data(), raw.data(), raw.size());
      const std::int64_t C = sh[2], comp = integer(m, "component");
      for (std::int64_t t = 0; t < sh[1]; ++t) out.push_back(a[static_cast<std::size_t>(t * C + comp)]);
      return out;
    };
    e.cv = meas("centre_vertex"); e.peak = meas("relative_peak"); e.size = meas("size_cm"); e.diam = meas("diameter_cm"); e.level = meas("characteristic_level"); e.area = meas("area");
    return e;
  };
  const Ev E = read_ev("rest"), Eb = read_ev("rest_b");
  std::printf("  %zu events stored (frames %lld … %lld)\n", E.frame.size(), E.frame.empty() ? 0LL : static_cast<long long>(E.frame.front()), E.frame.empty() ? 0LL : static_cast<long long>(E.frame.back()));
  CHECK(E.frame.size() > 100 && E.cv.size() == E.frame.size() && E.size.size() == E.frame.size() && E.peak.size() == E.frame.size() && E.diam.size() == E.frame.size());
  CHECK(E.params["detection"]["record"]["threshold"] == 100.0 && E.params["detection"]["record"]["events"] == static_cast<std::int64_t>(E.frame.size()));
  // the STRONGEST event of each frame: inside the window, at the blob planted for that half of it (the narrow blob in the first half, the wide one in the second)
  std::map<std::int64_t, std::size_t> top_of;
  for (std::size_t k = 0; k < E.frame.size(); ++k) {
    const auto it = top_of.find(E.frame[k]);
    if (it == top_of.end() || E.peak[k] > E.peak[it->second]) top_of[E.frame[k]] = k;
  }
  std::vector<double> sizeA, sizeB, diamA, diamB;
  int right = 0, outside_window = 0;
  for (std::size_t k = 0; k < E.frame.size(); ++k) if (!(E.start[k] >= w0 && E.start[k] < w1)) ++outside_window;
  for (const auto& [f, k] : top_of) {
    const bool first = E.start[k] < wm;
    const int cv = static_cast<int>(E.cv[k]);
    if (first && ang(cv, vA) <= 0.03) { ++right; sizeA.push_back(E.size[k]); diamA.push_back(E.diam[k]); }
    if (!first && ang(cv, vB) <= 0.05) { ++right; sizeB.push_back(E.size[k]); diamB.push_back(E.diam[k]); }
  }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
  std::printf("  the strongest event of %zu frames: %d at the planted blob (%zu narrow: size %.1f cm, diameter %.1f cm; %zu wide: size %.1f cm, diameter %.1f cm); %d events outside the window\n", top_of.size(), right, sizeA.size(), med(sizeA), med(diamA), sizeB.size(), med(sizeB), med(diamB), outside_window);
  CHECK(outside_window == 0 && top_of.size() > 1000 && right * 10 >= static_cast<int>(top_of.size()) * 9);
  CHECK(med(sizeB) > med(sizeA) && med(diamB) > med(diamA));                    // the wide blob's events are the larger, in characteristic size and in extent
  // the home map
  const StoredStatistic H = read_statistic(sub, "spatial_events.home.rest");
  CHECK(text(H.row, "method") == "occupancy" && H.shape == std::vector<std::int64_t>({N, 2}));
  auto frac = [&](int v) { return H.values[static_cast<std::size_t>(2 * v)] / H.values[static_cast<std::size_t>(2 * v + 1)]; };
  int anti = 0;
  { double best = 1e9; for (int v = 0; v < N; ++v) { const double d = ang(v, vA) + ang(v, vB); (void)d; const double far = -(ang(v, vA) + ang(v, vB)); if (far < best) { best = far; anti = v; } } }
  std::printf("  home map: the narrow blob's vertex in a patch in %.2f of the burst frames, the wide blob's %.2f, the farthest vertex %.2f (%lld burst frames)\n", frac(vA), frac(vB), frac(anti), static_cast<long long>(H.values[1]));
  CHECK(frac(vA) > 0.4 && frac(vA) < 0.6 && frac(vB) > 0.4 && frac(vB) < 0.6 && H.values[1] > 1000);   // each blob is on for half of the burst frames
  // chunk size and block size: the same events
  { bool same = Eb.frame == E.frame && Eb.cv == E.cv && Eb.level == E.level;
    std::printf("  storage chunks of 1000 vs 4096: %zu vs %zu events, frames/vertices/levels %s\n", Eb.frame.size(), E.frame.size(), same ? "identical" : "DIFFER");
    CHECK(same); }
  { g::RunOptions small = all; small.block_frames = 100;
    g::run(ds, sub, dev, set, small);
    const Ev E2 = read_ev("rest");
    const StoredStatistic H2 = read_statistic(sub, "spatial_events.home.rest");
    bool same = E2.frame == E.frame && E2.cv == E.cv && E2.level == E.level && E2.area == E.area && H2.values == H.values;
    std::printf("  blocks of 100 frames vs 256: %zu vs %zu events, %s\n", E2.frame.size(), E.frame.size(), same ? "identical" : "DIFFER");
    CHECK(same);
    CHECK(g::run(ds, sub, dev, set, small)[0]["hit"] == true);
    g::run(ds, sub, dev, set, all); }
  { g::RunOptions again; again.chunks = "all"; const auto r2 = g::run(ds, sub, dev, set, again); CHECK(r2.size() == 2 && r2[0]["hit"] == true); }
  // THE DEFAULT THRESHOLD (percentile 99 of the relative power outside the window, `run spatial_threshold`): the blobs are still found
  { Json tset = bset; tset["reference_level"] = 4;
    g::run(ds, sub, g::load("spatial_threshold"), tset, all);
    Json d = set; d.erase("threshold"); d.erase("ratio");
    g::run(ds, sub, dev, d, all);
    const Ev Ed = read_ev("rest");
    std::map<std::int64_t, std::size_t> top;
    for (std::size_t k = 0; k < Ed.frame.size(); ++k) { const auto it = top.find(Ed.frame[k]); if (it == top.end() || Ed.peak[k] > Ed.peak[it->second]) top[Ed.frame[k]] = k; }
    int ok = 0;
    for (const auto& [f, k] : top) { const bool first = Ed.start[k] < wm; const int cv = static_cast<int>(Ed.cv[k]); if ((first && ang(cv, vA) <= 0.03) || (!first && ang(cv, vB) <= 0.05)) ++ok; }
    const auto& rec = Ed.params["detection"]["record"];
    std::printf("  default threshold (percentile 99 of the relative power outside the window = %.3f x baseline): the strongest event of %zu frames is at the planted blob in %d\n", rec["threshold"].get<double>(), top.size(), ok);
    CHECK(rec["threshold_rule"].get<std::string>().rfind("percentile", 0) == 0 && rec["percentile"] == 99.0 && rec["threshold"].get<double>() > 3.0 && rec["threshold"].get<double>() < 8.0);
    CHECK(top.size() > 1000 && ok * 10 >= static_cast<int>(top.size()) * 9);
    // THE SUMMARY of the events inside the window, and THE NOISE REFERENCE: the same counts on the frames outside it (`run spatial_events_outside`, events not stored)
    auto sum_of = [&](const StoredStatistic& S, const std::string& prefix) { double a = 0; const Json& v = S.axes[0]["values"]; for (std::size_t k = 0; k < v.size(); ++k) if (v[k]["name"].get<std::string>().rfind(prefix, 0) == 0) a += S.values[k]; return a; };
    auto get = [&](const StoredStatistic& S, const std::string& name) { const Json& v = S.axes[0]["values"]; for (std::size_t k = 0; k < v.size(); ++k) if (v[k]["name"] == name) return S.values[k]; return -1.0; };
    const StoredStatistic Si = read_statistic(sub, "spatial_events.summary.rest");
    std::printf("  the summary inside the window: %.0f patches in %.0f frames (the stored events: %zu); by area %.0f, by level %.0f, by per-frame count %.0f frames\n", get(Si, "patches"), get(Si, "frames"), Ed.frame.size(), sum_of(Si, "area_bin_"), sum_of(Si, "level_"), sum_of(Si, "per_frame_"));
    CHECK(text(Si.row, "measure") == "patch_summary" && get(Si, "patches") == static_cast<double>(Ed.frame.size()) && sum_of(Si, "area_bin_") == get(Si, "patches") && sum_of(Si, "level_") == get(Si, "patches") && sum_of(Si, "per_frame_") == get(Si, "frames") && get(Si, "frames") == H.values[1]);
    g::run(ds, sub, g::load("spatial_events_outside"), d, all);
    const StoredStatistic So = read_statistic(sub, "spatial_events_outside.summary.rest"), Th = read_statistic(sub, "spatial_threshold.histogram.rest");
    std::printf("  the noise reference outside the window: %.0f patches in %.0f frames (%.2f a frame; inside %.2f a frame); the events outside are not stored: %s\n", get(So, "patches"), get(So, "frames"), get(So, "patches") / get(So, "frames"), get(Si, "patches") / get(Si, "frames"),
                nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "spatial_events_outside.events.rest", .any_status = true}) ? "STORED" : "yes");
    CHECK(get(So, "frames") == static_cast<double>(integer(Th.row, "n_members")) && get(So, "patches") > 0 && sum_of(So, "area_bin_") == get(So, "patches") && sum_of(So, "level_") == get(So, "patches") && sum_of(So, "per_frame_") == get(So, "frames")
          && !nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "spatial_events_outside.events.rest", .any_status = true}) && So.options["events_stored"] == false && So.options["threshold"] == Si.options["threshold"]);
    // THE LAG-FREE NULL (piece 10, spatial events): the ingredients, then draws in batches against the data's home map and size distribution
    {
      PsdOptions po; po.recording = "rest"; po.chunks = "all"; po.segment_level = 8; psd_compute(ds, sub, po);
      CrossSpectrumOptions co; co.recording = "rest"; co.channels_of = "ident"; co.chunks = "all"; co.segment_level = 8; co.bands = {{"alpha", 8.0, 12.0}}; crossspectrum_compute(ds, sub, co);
      g::run(ds, sub, g::load("lagfree_envelope"), Json{{"recording", "rest"}, {"cross_spectrum", "rest_crossspectrum_alpha"}, {"level", 5}}, all);
      Json nset = d; nset["cross_spectrum"] = "rest_crossspectrum_alpha";
      // one draw: stored as small statistics, a function of its seed
      Json one = nset; one["seed"] = 0; one["recording"] = "rest";
      g::run(ds, sub, g::load("spatial_events_null"), one, all);
      const StoredStatistic D0 = read_statistic(sub, "spatial_events_null.draw0.home.rest"), D0s = read_statistic(sub, "spatial_events_null.draw0.summary.rest");
      CHECK(D0.shape == H.shape && text(D0s.row, "measure") == "patch_summary" && !nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "spatial_events_null.draw0.events.rest", .any_status = true}));
      SpatialNullOptions sn;
      sn.chunks = "all";                 // the whole recording, stated
      sn.recording = "rest"; sn.batch = 2; sn.min_draws = 2; sn.max_draws = 4; sn.h = 10; sn.replace = true;
      sn.set = nset;
      const auto rn = spatial_null_compute(ds, sub, sn);
      const std::int64_t nd2 = rn.info["draws"].get<std::int64_t>();
      const StoredStatistic hp = read_statistic(sub, "spatial_events.null.rest.home_p"), hn = read_statistic(sub, "spatial_events.null.rest.home_null"), sz = read_statistic(sub, "spatial_events.null.rest.size_null");
      // direct: the draws read back, the data's fraction per vertex (the stored home map of the percentile run just made)
      const StoredStatistic Hn = read_statistic(sub, "spatial_events.home.rest");
      auto fracn = [&](int v) { return Hn.values[static_cast<std::size_t>(2 * v)] / Hn.values[static_cast<std::size_t>(2 * v + 1)]; };
      Eigen::VectorXd exd = Eigen::VectorXd::Zero(N);
      for (std::int64_t i = 0; i < nd2; ++i) {
        const StoredStatistic Hd = read_statistic(sub, "spatial_events_null.draw" + std::to_string(i) + ".home.rest");
        for (int v = 0; v < N; ++v) { const double fr = Hd.values[static_cast<std::size_t>(2 * v + 1)], dd = fr > 0 ? Hd.values[static_cast<std::size_t>(2 * v)] / fr : 0.0; if (Hn.values[static_cast<std::size_t>(2 * v)] > 0 && dd >= fracn(v)) exd[v] += 1; }
      }
      bool same = hp.shape == std::vector<std::int64_t>({N, 3});
      for (int v = 0; same && v < N; ++v) same = hp.values[static_cast<std::size_t>(3 * v + 2)] == exd[v];
      const double pA = hp.values[static_cast<std::size_t>(3 * vA)], pFar = hp.values[static_cast<std::size_t>(3 * anti)];
      const double nullA = hn.values[static_cast<std::size_t>(vA)];
      std::printf("  spatialnull: %lld draws (%s); the exceedances equal the direct count: %s; the narrow blob's vertex: data %.2f in a patch, null %.3f, p %.3f; the farthest vertex p %.3f (data %.2f)\n", static_cast<long long>(nd2), rn.info["stopped"].get<std::string>().c_str(), same ? "yes" : "NO",
                  fracn(vA), nullA, pA, pFar, fracn(anti));
      CHECK(same && nd2 == 4 && text(hp.row, "measure") == "null_p" && text(hn.row, "measure") == "null_fraction" && text(sz.row, "measure") == "null_size");
      // WHAT THE NULL SAYS: it keeps the data's zero-lag covariance, and the planted blobs ARE zero-lag spatial structure (the band's covariance holds both of them), so the null occupies the narrow blob's vertex in nearly every burst frame (0.97), as the data does
      // (0.99): the covariance explains a STATIONARY home map. What the data has beyond it is the small excess over the null at that vertex (none of the 4 draws reaches it: p = 1 / 5, the smallest 4 draws can give).
      CHECK(nullA > 0.8 && pA <= 0.2 + 1e-9);
      CHECK(sz.shape[1] == 6 && sz.shape[0] >= 5);
      { SpatialNullOptions again = sn; again.replace = false; CHECK(spatial_null_compute(ds, sub, again).hit); }
      CHECK(what_of([&] { SpatialNullOptions bad = sn; bad.recording = "no_such"; spatial_null_compute(ds, sub, bad); }).find("no_such") != std::string::npos);
    }
    g::run(ds, sub, dev, set, all); }
  // THE PLAN
  { Json s = set; s["spatial_shape"] = "meyer"; CHECK(what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); }).find("another spatial_band") != std::string::npos); }
  { Json s = set; s["ratio"] = 0.0; CHECK(what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); }).find("ratio") != std::string::npos); }
  { Json s = set; s["threshold"] = "percentile"; s["threshold_statistic"] = "no_such_statistic"; CHECK(what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); }).find("none is stored") != std::string::npos); }
  { Json s = set; s["reference_level"] = 7; CHECK(what_of([&] { g::run(ds, sub, dev, s, g::RunOptions{}); }).find("not one of the field's levels") != std::string::npos); }
  return nxrtest::finish("test_spatial_events_planted") || g_failures ? 1 : 0;
}
