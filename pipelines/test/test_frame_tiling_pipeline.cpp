// THE FRAME TILING, stored and used, on a synthetic recording (never a real dataset):
//   - `frametiling compute` stores a closed-form spans Selection like <recording>_chunks: count = round(samples × fps / sfreq), the frame rate, centre and frames a cycle in the record; a rerun keeps it;
//     the band's middle is the centre; refusals (no centre, both, a rate the line cannot hold)
//   - the `frames` step samples a 0.37 Hz recording at the frames inside each chunk; the run's mean per channel equals the direct sum of cos(2π f t_j + φ) over the frames; with `within` only the frames inside the
//     named span count, and a bad span's frames are left out
//   - the result does not depend on the chunk size: the same samples stored in chunks of 4096 and of 1000 give the same numbers
//   - the plan: a step that reads the samples' own time (band_filter) refuses a field on the frames; a tiling of another rate is refused
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/frame_tiling.h"
#include "nxr/tower.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/pipelines/runner.h"

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
std::string def(const std::string& recording_filter, bool within, const std::string& extra_step = "") {
  return "pipeline: frames_run\ninputs:\n  recording: {field: recording, each: true, where: {name: " + recording_filter + "}}\n"
         "  tiling: {selection: spans, manifold: recording.time, where: {name: \"{recording} frames 0.00257492 s\"}}\n"
         "  bad: {selection: spans, manifold: recording.time, where: {name: \"{recording}_bad_segments\"}, required: false}\n" +
         (within ? "  within: {selection: spans, manifold: recording.time, where: {name: window}}\n" : "") +
         "steps:\n  chunks:\n    over: recording\n    do:\n      - {step: frames, in: {x: recording, tiling: tiling" + std::string(within ? ", within: within" : "") + "}, out: f}\n" + extra_step +
         "      - {step: measure, in: {x: f, exclude: bad}, with: {measure: mean, bad_segments: exclude}, out: mean}\noutputs:\n  mean: {kind: band power}\n";
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_frame_tiling_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int M = 5, T = 8192;
  const double sfreq = 250.0, bad0 = 20.0, bad1 = 21.0;
  const double w0 = 6.3, w1 = 25.2, f1 = 0.37;           // the window: frames in [6.3, 25.2) s count; the recording a 0.37 Hz cosine (a mean that depends on which frames are counted)
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c) * T + t] = static_cast<float>(std::cos(2 * M_PI * f1 * t / sfreq + 0.5 * c));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string tid, first_rid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      add_surface(sub, so, b.P, b.F);
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
        rec.name = name; rec.path = node_path(sub.db(), prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "A");
        const std::string rid = sub.field(std::move(rec));
        if (first_rid.empty()) first_rid = rid;
        nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
      }
      nxr::bst::TimeSpansOptions wo; wo.line_id = tid; wo.of_field_id = first_rid; wo.description = "the window";
      nxr::bst::time_spans(sub, "time/window", {{w0, w1, 0}}, {"burst"}, wo);
    });
    sub.compose([&] { nxr::io::constantq::write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  // ── THE STORED TILING ──
  FrameTilingOptions fo;
  fo.recording = "rest"; fo.band = std::make_pair(8.0, 12.0);
  const auto t1 = frame_tiling_compute(ds, sub, fo);
  const double fps = 1.0 / nxr::tower::periodS(-25);
  const std::int64_t count = static_cast<std::int64_t>(std::floor(static_cast<double>(T - 1) * fps / sfreq)) + 1;
  const auto trow = sub.db().one("SELECT * FROM selection WHERE id = ?", {V(t1.info["id"].get<std::string>())});
  CHECK(trow && text(*trow, "name") == "rest frames 0.00257492 s" && text(*trow, "type") == "spans" && integer(*trow, "n_members") == count && !t1.existed && t1.info["fps"] == fps && t1.info["frame_level"] == -25);
  const Json tp = Json::parse(text(*trow, "params_json"));
  CHECK(tp["tiling"]["count"] == count && tp["tiling"]["width"] == 1.0 / fps && tp["atlas"]["frame_level"] == -25 && tp["atlas"]["frame_code0"] == 0 && tp["atlas"]["anchor"] == "recording" && tp["atlas"]["frame_s"] == 1.0 / fps && tp["tiling"]["origin"] == 0.0 && tp["frame_tiling"]["centre_hz"] == 10.0 && tp["frame_tiling"]["frames_per_cycle"] == 36 && tp["frame_tiling"]["band"] == Json::array({8.0, 12.0}));
  CHECK(frame_tiling_compute(ds, sub, fo).existed);
  FrameTilingOptions fb = fo; fb.recording = "rest_b";
  frame_tiling_compute(ds, sub, fb);
  CHECK(what_of([&] { FrameTilingOptions e = fo; e.band.reset(); frame_tiling_compute(ds, sub, e); }).find("--frame-level") != std::string::npos);
  CHECK(what_of([&] { FrameTilingOptions e = fo; e.centre_hz = 10.0; frame_tiling_compute(ds, sub, e); }).find("not both") != std::string::npos);
  CHECK(what_of([&] { FrameTilingOptions e = fo; e.band.reset(); e.centre_hz = 10.0; e.frames_per_cycle = 0; frame_tiling_compute(ds, sub, e); }).find("at least 1") != std::string::npos);
  // a named level; a band derives its level (12 Hz x 36 -> -26, 10 Hz x 36 -> -25); both is refused; another level is another tiling
  { FrameTilingOptions c; c.recording = "rest"; c.frame_level = -20; const auto r = frame_tiling_compute(ds, sub, c); CHECK(r.info["selection"] == "rest frames 0.0823975 s" && r.info["frame_level"] == -20 && !r.existed); }
  { FrameTilingOptions c; c.recording = "rest"; c.centre_hz = 12.0; CHECK(frame_tiling_compute(ds, sub, c).info["frame_level"] == -26); }
  CHECK(what_of([&] { FrameTilingOptions e; e.recording = "rest"; e.frame_level = -20; e.centre_hz = 10.0; frame_tiling_compute(ds, sub, e); }).find("not both") != std::string::npos);
  CHECK(what_of([&] { FrameTilingOptions e; e.recording = "rest"; e.frame_level = 1; frame_tiling_compute(ds, sub, e); }).find("finer") != std::string::npos);
  { FrameTilingOptions c; c.recording = "rest"; c.frame_level = -20; c.start_s = 43200.5; c.replace = true; frame_tiling_compute(ds, sub, c);   // a day anchor: code0 and the origin before the first sample
    const auto row = sub.db().one("SELECT params_json FROM selection WHERE name = ?", {V(std::string("rest frames 0.0823975 s"))});
    const Json p = Json::parse(text(*row, "params_json")); const auto tt = nxr::tower::makeFrameTiling(sfreq, T, -20, 43200.5);
    CHECK(p["atlas"]["anchor"] == "day" && p["atlas"]["frame_code0"] == tt.code0 && p["atlas"]["start_s"] == 43200.5 && p["tiling"]["origin"] == tt.origin_s() && p["tiling"]["count"] == tt.count); }

  // ── THE DIRECT MEAN over the frames in the window (and good): cos(2π f t_j + 0.5 c), t_j = j / fps ──
  auto direct = [&](bool within) {
    std::vector<double> sum(M, 0.0);
    std::int64_t n = 0;
    for (std::int64_t j = 0; j < count; ++j) {
      const double pos = j * sfreq / fps, tj = j / fps;
      const double sec = std::floor(pos) / sfreq;
      if (sec >= bad0 && sec < bad1) continue;
      if (within && !(tj >= w0 && tj < w1)) continue;
      ++n;
      for (int c = 0; c < M; ++c) sum[static_cast<std::size_t>(c)] += std::cos(2 * M_PI * f1 * tj + 0.5 * c);
    }
    for (auto& s : sum) s /= static_cast<double>(n);
    return sum;
  };
  g::RunOptions all; all.chunks = "all"; all.replace = true;
  auto run_mean = [&](const std::string& rec, bool within) {
    const std::string t = def(rec, within);
    const g::Definition d = g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t);
    const auto r = g::run(ds, sub, d, Json::object(), all);
    CHECK(r.size() == 1 && r[0]["stored"].size() == 1);
    return r.size() == 1 ? read_floats((sub.folder() / (r[0]["stored"][0]["path"].get<std::string>()))) : std::vector<float>();
  };
  for (bool within : {true, false}) {
    const auto want = direct(within);
    const auto a = run_mean("rest", within), bb = run_mean("rest_b", within);
    double ea = 0, eb = 0, ab = 0;
    for (int c = 0; c < M && a.size() == static_cast<std::size_t>(M) && bb.size() == static_cast<std::size_t>(M); ++c) {
      ea = std::max(ea, std::abs(a[static_cast<std::size_t>(c)] - want[static_cast<std::size_t>(c)]));
      eb = std::max(eb, std::abs(bb[static_cast<std::size_t>(c)] - want[static_cast<std::size_t>(c)]));
      ab = std::max(ab, static_cast<double>(std::abs(a[static_cast<std::size_t>(c)] - bb[static_cast<std::size_t>(c)])));
    }
    std::printf("  %s: mean over the frames vs the direct sum: chunks of 4096 %.2e, of 1000 %.2e; between them %.2e\n", within ? "within [6.3, 25.2) s, bad 1 s left out" : "whole run, bad 1 s left out", ea, eb, ab);
    CHECK(a.size() == static_cast<std::size_t>(M) && bb.size() == static_cast<std::size_t>(M));
    if (within) CHECK(ea < 2e-4 && eb < 2e-4);
    else CHECK(ea < 5e-3 && eb < 5e-3);                 // the first and last frames read a window cut by the recording's ends (0 beyond)
    CHECK(ab < 1e-6);                                    // the chunk size does not matter
  }
  // ── THE PLAN ──
  CHECK(what_of([&] { const std::string t = def("rest", false, "      - {step: band_filter, in: {x: f}, with: {band: [8.0, 12.0]}, out: g}\n"); g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), all); }).find("frame tiling") != std::string::npos);
  { FrameTilingOptions c; c.recording = "rest"; c.frame_level = -22; frame_tiling_compute(ds, sub, c); }
  { std::string t = def("rest", false); t.replace(t.find("0.00257492"), 10, "0.0205994");
    CHECK(what_of([&] { g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), all); })== std::string()); }   // a tiling of another level is another tiling: it runs
  return nxrtest::finish("test_frame_tiling_pipeline") || g_failures ? 1 : 0;
}
