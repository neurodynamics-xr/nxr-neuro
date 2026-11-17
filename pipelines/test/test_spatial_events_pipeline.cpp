// THE SPATIAL BASELINE and `relative` (spec docs/specs/2026-11-14-spatial-events.md, pieces 1 and 2), run on a synthetic two-hemisphere source space (never a real dataset): a free kernel of random currents, a recording of a 10 Hz
// oscillation plus noise, the frame tiling of the 8-12 Hz band (36 frames a cycle), a window of "bursts", bad spans one inside it and one outside.
//   - `run spatial_baseline`: the band power per glued vertex and hat level (levels 2 and 3, the free current through each band, power summed over the components) over the frames OUTSIDE the window and the bad spans, stored as a
//     STATISTIC (measure power, method mean, [vertex x level], the vertex axis on the glued surface); equal to the DIRECT computation (taps, Kaiser-sinc at the frames, the dense exact hat of each level)
//   - the result does not depend on the chunk size (storage chunks 4096 and 1000) or the block of frames (100 against 256); a rerun is a hit
//   - `relative`: power divided by that baseline read by name: the mean over the burst frames of power / baseline equals the direct mean power over the baseline, at each level; a baseline of another chain (band) is refused,
//     a field that is no power is refused, a baseline over a run with no good frame is refused
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/filterbank.h"
#include "nxr/frame_tiling.h"
#include "nxr/tower.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/eigen.h"
#include "run_definition.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/taps.h"

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
Json set_of(std::initializer_list<std::pair<const char*, Json>> kv) { Json j = Json::object(); for (const auto& [k, v] : kv) j[k] = v; return j; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_spatial_baseline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = 12, T = 4096, level = 3;
  const double sfreq = 250.0, w0 = 4.0, w1 = 9.0, bad0 = 6.0, bad1 = 6.5, bad2 = 12.0, bad3 = 12.5;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T), kfree(static_cast<std::size_t>(3 * N) * M);
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c) * T + t] = static_cast<float>(std::cos(2 * M_PI * 10.0 * t / sfreq + 0.7 * c) + 0.3 * nd(rng));
  for (auto& k : kfree) k = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, first_rid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      std::vector<std::int32_t> dk(static_cast<std::size_t>(b.n), 1);
      for (int i = 0; i < b.n; ++i) { const double y = b.P[static_cast<std::size_t>(3 * i + 1)]; if ((i < b.nHalf && y < 0.7) || (i >= b.nHalf && y > -0.7)) dk[static_cast<std::size_t>(i)] = 0; }
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      surf = add_surface(sub, so, b.P, b.F).id;
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
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "A");
        const std::string rid = sub.field(std::move(rec));
        if (first_rid.empty()) first_rid = rid;
        nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad0, bad1, 0}, {bad2, bad3, 0}}, {"BAD"}, bo);
      }
      nxr::bst::TimeSpansOptions wo; wo.line_id = tid; wo.of_field_id = first_rid; wo.description = "the bursts";
      nxr::bst::time_spans(sub, "time/window", {{w0, w1, 0}}, {"burst"}, wo);
      nxr::bst::time_spans(sub, "time/whole_run", {{0.0, 100.0, 0}}, {"burst"}, wo);
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      OperatorSpec op;
      op.name = "free"; op.path = "free"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
      op.data = view(kfree, {3 * N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
    });
    sub.compose([&] { nxr::io::constantq::write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  tiles_compute(ds, sub, TilesComputeOptions{});
  { EigenSolveOptions eo; eo.modes = 100; eo.surface = "cortex"; eigen_solve(ds, sub, eo, {}); }          // stores the glued surface the curl lives on
  for (const char* rec : {"rest", "rest_b"}) { nxrtest::run_definition(ds, sub, "frametiling", Json{{"recording", rec}, {"band", Json::array({8.0, 12.0})}}); }
  const double fps = 1.0 / nxr::tower::periodS(-25);

  // ── THE DIRECT COMPUTATION: the band's analytic signal on the sensors (Morlet taps), sampled at the frames by the Kaiser sinc, each level's free current through its dense exact hat on the glued surface ──
  const std::vector<int> lvs = {2, 3};
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  SpatialBandOptions sbo; sbo.transform = "exact"; sbo.spatial_shape = "mexhat"; sbo.glued = true;
  std::vector<SpatialBand> bands;
  for (const int l : lvs) bands.push_back(make_spatial_band(ds, sub, S, read_surface_level(sub, S, l), sbo));
  const std::int64_t Vg = bands[0].glued_vertices, L = static_cast<std::int64_t>(lvs.size());
  const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 12.0); };
  const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
  const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
  const std::int64_t count = static_cast<std::int64_t>(std::floor(static_cast<double>(T - 1) * fps / sfreq)) + 1;
  std::vector<double> pos(static_cast<std::size_t>(count));
  for (std::int64_t j = 0; j < count; ++j) pos[static_cast<std::size_t>(j)] = static_cast<double>(j) * sfreq / fps;
  const std::vector<float> Zr = nxr::frametiling::sincInterpolate(re.data(), M, T, pos, 32), Zi = nxr::frametiling::sincInterpolate(im.data(), M, T, pos, 32);   // [M x count]
  Eigen::MatrixXd Zre(M, count), Zim(M, count);
  for (int c = 0; c < M; ++c) for (std::int64_t j = 0; j < count; ++j) { Zre(c, j) = Zr[static_cast<std::size_t>(c * count + j)]; Zim(c, j) = Zi[static_cast<std::size_t>(c * count + j)]; }
  Eigen::MatrixXd Kfree(3 * N, M);
  for (int r = 0; r < 3 * N; ++r) for (int c = 0; c < M; ++c) Kfree(r, c) = kfree[static_cast<std::size_t>(r) * M + c];
  // the power of each glued vertex and level at every frame: [Vg*L x count], row = vertex * L + level
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Vg * L, count);
  for (std::size_t li = 0; li < lvs.size(); ++li)
    for (int comp = 0; comp < 3; ++comp) {
      Eigen::MatrixXd Kc(N, M);
      for (int v = 0; v < N; ++v) Kc.row(v) = Kfree.row(3 * v + comp);
      const Eigen::MatrixXd Mc = bands[li].apply_glued(Kc);
      const Eigen::MatrixXd A = Mc * Zre, B = Mc * Zim;
      for (std::int64_t v = 0; v < Vg; ++v) P.row(v * L + static_cast<std::int64_t>(li)) += (A.row(v).array().square() + B.row(v).array().square()).matrix();
    }
  auto in_span = [](double t, double a, double b) { return t >= a && t < b; };
  std::vector<std::int64_t> outside, inside;                         // the baseline's frames (outside the window and the bad spans, by the sample at or before the frame) and the burst frames
  for (std::int64_t j = 0; j < count; ++j) {
    const double sec = std::floor(pos[static_cast<std::size_t>(j)]) / sfreq, tj = pos[static_cast<std::size_t>(j)] / sfreq;
    const bool bad = in_span(sec, bad0, bad1) || in_span(sec, bad2, bad3);
    if (!bad && !in_span(sec, w0, w1)) outside.push_back(j);
    if (!bad && in_span(tj, w0, w1)) inside.push_back(j);
  }
  std::printf("  %lld glued vertices, %lld levels, %lld frames: %zu baseline frames, %zu burst frames\n", static_cast<long long>(Vg), static_cast<long long>(L), static_cast<long long>(count), outside.size(), inside.size());
  CHECK(Vg > N && outside.size() > 500 && inside.size() > 500);
  auto mean_over = [&](const std::vector<std::int64_t>& cols) {
    Eigen::VectorXd m = Eigen::VectorXd::Zero(P.rows());
    for (const auto j : cols) m += P.col(j);
    return Eigen::VectorXd(m / static_cast<double>(cols.size()));
  };
  const Eigen::VectorXd base_direct = mean_over(outside);

  // ── THE BASELINE: run spatial_baseline ──
  g::RunOptions all; all.chunks = "all"; all.replace = true;
  const Json set = set_of({{"levels", Json::array({2, 3})}, {"transform", "exact"}, {"bursts", "window"}});
  const g::Definition dbase = g::load("spatial_baseline");
  const auto rb = g::run(ds, sub, dbase, set, all);
  CHECK(rb.size() == 2);                                                       // one run for each recording
  const StoredStatistic B = read_statistic(sub, "spatial_baseline.baseline.rest"), Bb = read_statistic(sub, "spatial_baseline.baseline.rest_b");
  CHECK(text(B.row, "method") == "mean" && text(B.row, "measure") == "power" && integer(B.row, "n_members") == static_cast<std::int64_t>(outside.size()) && B.shape == std::vector<std::int64_t>({Vg, L}));
  CHECK(B.axes[0]["manifold_id"] == Json(nxr::io::task::Subject::of(sub).require({.cls = nxr::io::task::Cls::manifold, .name = "cortex_glued", .any_status = true}, "the glued cortex").id) && B.axes[1]["values"][1]["name"] == "level3");
  double e = 0, top = 0;
  for (std::int64_t i = 0; i < Vg * L; ++i) { top = std::max(top, base_direct(i)); e = std::max(e, std::abs(B.values[static_cast<std::size_t>(i)] - base_direct(i))); }
  std::printf("  the stored baseline vs the direct one (taps, sinc, dense exact hats): max difference %.2e of the largest value\n", e / top);
  CHECK(e < 5e-4 * top && top > 0);
  { double lvl = 0;                                                            // the levels are distinct (not one operator repeated)
    for (std::int64_t v = 0; v < Vg; ++v) lvl = std::max(lvl, std::abs(B.values[static_cast<std::size_t>(v * L)] - B.values[static_cast<std::size_t>(v * L + 1)]));
    CHECK(lvl > 1e-3 * top); }
  // chunk size: the same baseline from storage chunks of 1000
  { double d = 0;
    for (std::size_t i = 0; i < B.values.size(); ++i) d = std::max(d, std::abs(B.values[i] - Bb.values[i]));
    std::printf("  chunks of 4096 vs 1000: max difference %.2e\n", d / top);
    CHECK(d < 1e-5 * top && B.options["columns"] == Bb.options["columns"]); }
  // block size: 100 frames a block against 256, then back; the block size is in the key: the rerun at 100 is a hit
  { g::RunOptions small = all; small.block_frames = 100;
    g::run(ds, sub, dbase, set, small);
    const StoredStatistic B2 = read_statistic(sub, "spatial_baseline.baseline.rest");
    double d = 0;
    for (std::size_t i = 0; i < B.values.size(); ++i) d = std::max(d, std::abs(B.values[i] - B2.values[i]));
    std::printf("  blocks of 100 frames vs 256: max difference %.2e\n", d / top);
    CHECK(d < 1e-6 * top);
    CHECK(g::run(ds, sub, dbase, set, small)[0]["hit"] == true);
    g::run(ds, sub, dbase, set, all); }
  { g::RunOptions again; again.chunks = "all"; const auto r2 = g::run(ds, sub, dbase, set, again); CHECK(r2.size() == 2 && r2[0]["hit"] == true); }
  // no good frame: bursts that cover the run leave no baseline
  { Json s = set; s["bursts"] = "whole_run"; CHECK(what_of([&] { g::run(ds, sub, dbase, s, all); }).find("no good frame") != std::string::npos); }

  // ── RELATIVE: power over the stored baseline, read by name ──
  auto def = [&](const std::string& x_step, const std::string& extra) {
    return "pipeline: relative_check\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0}\n  tiles: {selection: ladder, manifold: kernel.to}\n"
           "  tiling: {selection: spans, manifold: recording.time, where: {name: \"{recording} frames 0.00257492 s\"}}\n  bursts: {selection: spans, manifold: recording.time, where: {name: window}}\n"
           "  bad: {selection: spans, manifold: recording.time, where: {name: \"{recording}_bad_segments\"}, required: false}\n  baseline: {statistic: mean, where: {name: spatial_baseline.baseline.rest}}\n"
           "steps:\n  chunks:\n    over: recording\n    do:\n      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}\n      - {step: frames, in: {x: z, tiling: tiling, within: bursts}, out: zf}\n"
           "      - {step: apply_kernel, in: {x: zf, kernel: kernel, tiles: tiles}, with: {levels: [2, 3], glued: true, transform: exact" + extra + "}, out: j}\n      - {step: power, in: {x: j}, out: p}\n" + x_step +
           "      - {step: at_level, in: {x: r}, with: {level: 2}, out: r2}\n      - {step: at_level, in: {x: r}, with: {level: 3}, out: r3}\n"
           "      - {step: measure, in: {x: r2, exclude: bad}, with: {bad_segments: exclude}, out: rel2}\n      - {step: measure, in: {x: r3, exclude: bad}, with: {bad_segments: exclude}, out: rel3}\noutputs:\n  rel2: {kind: relative power}\n  rel3: {kind: relative power}\n";
  };
  const std::string rel_step = "      - {step: relative, in: {x: p, baseline: baseline}, out: r}\n";
  {
    const std::string t = def(rel_step, "");
    const auto r = g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), all);
    CHECK(r.size() == 1 && r[0]["stored"].size() == 2);
    const Eigen::VectorXd pm = mean_over(inside);
    for (std::size_t li = 0; li < lvs.size(); ++li) {
      const std::vector<float> got = read_floats((sub.folder() / (r[0]["stored"][li]["path"].get<std::string>())));
      double w = 0, tp = 0;
      for (std::int64_t v = 0; v < Vg; ++v) { const double want = pm(v * L + static_cast<std::int64_t>(li)) / base_direct(v * L + static_cast<std::int64_t>(li)); tp = std::max(tp, want); w = std::max(w, std::abs(got[static_cast<std::size_t>(v)] - want)); }
      std::printf("  level %d: the burst frames' mean of power / baseline vs the direct one: %.2e (largest ratio %.3g)\n", lvs[li], w / tp, tp);
      CHECK(got.size() == static_cast<std::size_t>(Vg) && w < 1e-3 * tp);
    }
  }
  // a baseline of another chain is no baseline of this one; a field that is no power is refused
  { const std::string t = def(rel_step, ", spatial_shape: meyer");
    CHECK(what_of([&] { g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), g::RunOptions{}); }).find("another spatial_band") != std::string::npos); }
  { const std::string t = def("      - {step: relative, in: {x: j, baseline: baseline}, out: r}\n", "");
    CHECK(what_of([&] { g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), g::RunOptions{}); }).find("band's power") != std::string::npos); }
  return nxrtest::finish("test_spatial_events_pipeline") || g_failures ? 1 : 0;
}
