// run cortical_cell_pairs (spec docs/specs/2026-11-13-cortical-events.md, phase 2c) on a synthetic source space (never a real dataset): the cells of 2 spatial levels x 2 time bands in ONE pass equal four
// single-pair `cortical_cells` runs cell for cell; the same cells with another chunk width; a rerun is a hit; the written-out definition (`for:`) is flat and checked; the refusals: a missing joint partition (every
// missing pair named with its command), a band that is no level, an empty list, another job's rows without --replace.

#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/constant_q.h"
#include "nxr/io/yaml.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/joint.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/taps.h"

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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cortical_cell_pairs_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 8192;
  const double sfreq = 250.0, bad0 = 10.0, bad1 = 10.5;
    // the planted source: 12 Hz on a bump around one vertex of the left hemisphere, during time tiles 14 … 17 of the 256-sample level (3.58 s … 4.6 s ... see below), noise everywhere
  auto P = [&](int i) { return Eigen::Vector3d(b.P[3 * i], b.P[3 * i + 1], b.P[3 * i + 2]); };
  int centre = 0;
  for (int i = 0; i < b.nHalf; ++i) if (P(i).x() > P(centre).x()) centre = i;
  const std::int64_t W = 256, t0 = 14 * W, t1 = 18 * W;
  std::mt19937 rng(3);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v) {
    const double amp = std::exp(-(P(v) - P(centre)).squaredNorm() / (2 * 0.8 * 0.8));
    for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>((t >= t0 && t < t1 ? amp * std::cos(2 * M_PI * 12.0 * t / sfreq + 0.1 * v) : 0.0) + 0.02 * nd(rng));
  }
  std::vector<float> kid(static_cast<std::size_t>(N) * M, 0.0f), kfree(static_cast<std::size_t>(3 * N) * M, 0.0f);
  const double alpha[3] = {1.0, 2.0, 2.0};                    // a free kernel: three components a vertex, their norm 3 |x|
  for (int v = 0; v < N; ++v) { kid[static_cast<std::size_t>(v) * M + v] = 1.0f; for (int c = 0; c < 3; ++c) kfree[static_cast<std::size_t>(3 * v + c) * M + v] = static_cast<float>(alpha[c]); }
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
      const std::string prod = sub.product({cid, tid});
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(4096)), std::make_pair(std::string("rest_b"), std::int64_t(1000))}) {   // the same samples, two chunk widths
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "A");
        const std::string rid = sub.field(std::move(rec));
        nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
      }
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to, const char* method) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        op.cols.json("params_json", Json{{"inverse", Json{{"measure", method}}}});
        sub.operator_(std::move(op));
      };
      kernel("fixed", kid, N, "scalar", "dSPM");
      kernel("free", kfree, 3 * N, "vector3", "MNE");
    });
    sub.compose([&] { nxr::pipelines::constant_q_write(sub, tid); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  tiles_compute(ds, sub, TilesComputeOptions{});
  // ── THE PAIRS: spatial levels 3 and 4 x the time bands 8-16 Hz and 16-32 Hz; each band is a constant-Q level of the recording's time line; the joint partition of every pair, both recordings ──
  const std::int64_t Wb[2] = {256, 128};
  const double blo[2] = {8.0, 16.0}, bhi[2] = {16.0, 32.0};
  const int levels[2] = {3, 4};
  std::int64_t frames[2] = {0, 0};
  int timeLevel[2] = {-1, -1};
  for (const auto& m : nxr::io::task::Subject::of(sub).find_all({.cls = nxr::io::task::Cls::selection, .params_present = {"resolves_hz"}, .any_status = true})) {
    if (!m.member_of_id) continue;
    const Json pj = m.params;
    for (int bi = 0; bi < 2; ++bi)
      if (pj["resolves_hz"]["lo"] == blo[bi] && pj["resolves_hz"]["hi"] == bhi[bi] && pj["window_samples"] == Wb[bi]) { timeLevel[bi] = static_cast<int>(*m.member_ordinal); frames[bi] = pj["tiling"]["count"].get<std::int64_t>(); }
  }
  CHECK(timeLevel[0] >= 0 && timeLevel[1] >= 0);
  if (timeLevel[0] < 0 || timeLevel[1] < 0) return 1;
  auto make_joint = [&](const std::string& rec, int li, int bi) { JointComputeOptions jo; jo.recording = rec; jo.space_level = li; jo.time_level = timeLevel[bi]; joint_compute(ds, sub, jo); };
  for (const char* rec : {"rest", "rest_b"}) for (int li : levels) for (int bi = 0; bi < 2; ++bi) make_joint(rec, li, bi);

  const g::Definition d = g::load("cortical_cell_pairs"), one = g::load("cortical_cells");
  g::RunOptions all; all.chunks = "all";
  const Json bands = Json::array({Json::array({blo[0], bhi[0]}), Json::array({blo[1], bhi[1]})});
  const Json set = Json{{"space_levels", Json::array({levels[0], levels[1]})}, {"bands", bands}, {"method", "dSPM"}, {"transform", "exact"}};

  // ── 0. THE WRITTEN-OUT DEFINITION: flat, one analytic a band, one apply_kernel a pair; the repeated inputs and outputs named by their pair ──
  {
    CHECK(!d.templ.is_null());
    const g::Definition e = g::expand(d, g::parameters_of(d, set));
    CHECK(e.templ.is_object() && e.do_.size() == 2 + 4 * 4);
    CHECK(e.inputs.contains("cells_l3_b8_16") && e.inputs.contains("cells_l4_b16_32") && e.outputs.contains("amplitude_l4_b8_16") && e.outputs.contains("peak_l3_b16_32") && e.outputs.size() == 8);
    CHECK(e.inputs["cells_l4_b16_32"]["where"]["params.joint.space_level"] == 4 && e.inputs["cells_l4_b16_32"]["where"]["params.joint.band"] == Json::array({16.0, 32.0}));
    int analytics = 0, kernels = 0;
    for (const auto& u : e.do_) { analytics += u.step == "analytic"; kernels += u.step == "apply_kernel"; }
    CHECK(analytics == 2 && kernels == 4);
    const auto plans = g::plan_runs(d, sub.db(), sub.id(), set);
    CHECK_EQ(static_cast<int>(plans.size()), 2);
    CHECK(plans[0].steps.size() == 18);
  }
  // ── 1. THE REFUSALS, before any data: a missing joint partition names EVERY missing pair with its command; a band that is no level; an empty list ──
  {
    std::string why;
    try { g::plan_runs(d, sub.db(), sub.id(), Json{{"space_levels", Json::array({3, 5})}, {"bands", bands}, {"method", "dSPM"}}); } catch (const Error& e) { why = e.what(); }
    CHECK(why.find("cells_l5_b8_16") != std::string::npos && why.find("cells_l5_b16_32") != std::string::npos && why.find("run joint") != std::string::npos && why.find("--space-level 5") != std::string::npos);
    CHECK(why.find("cells_l3_") == std::string::npos);                                          // the pairs that exist are not named
    why.clear();
    try { g::plan_runs(d, sub.db(), sub.id(), Json{{"bands", Json::array({Json::array({9.0, 17.0})})}, {"method", "dSPM"}}); } catch (const Error& e) { why = e.what(); }
    CHECK(why.find("cells_l5_b9_17") != std::string::npos && why.find("no joint partition") != std::string::npos);
    CHECK(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), Json{{"bands", Json::array()}}); }) == std::string(code::invalid_argument));
    CHECK(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), Json{{"space_levels", 4}}); }) == std::string(code::invalid_argument));       // not a list
  }
  // ── 2. THE RUN: every pair of both recordings in one pass ──
  const auto r = g::run(ds, sub, d, set, all);
  CHECK_EQ(static_cast<int>(r.size()), 2);
  struct Cells { std::vector<float> a, p; std::vector<std::int64_t> shape; };
  std::map<std::string, Cells> pairs;                   // "<recording> l<level> b<band>" -> cells
  for (const auto& run : r) {
    CHECK(run["hit"] == false && run["stored"].size() == 8);
    const std::string rec = run["anchor"].get<std::string>();
    for (int li : levels)
      for (int bi = 0; bi < 2; ++bi) {
        const std::string tag = "l" + std::to_string(li) + "_b" + (bi == 0 ? "8_16" : "16_32");
        const Json am = out_of(run, "amplitude_" + tag), pk = out_of(run, "peak_" + tag);
        CHECK(!am.is_null() && !pk.is_null());
        Cells c;
        c.a = nxrtest::read_packed(sub.folder(), am, &c.shape);
        c.p = nxrtest::read_packed(sub.folder(), pk);
        CHECK(c.shape == (std::vector<std::int64_t>{2 * (1 << li), frames[bi], 2}));
        // the rows are of the recording on THIS pair's joint Selection
        const auto js = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = am["over"].get<std::string>(), .any_status = true});
        const Json jp = js->params["joint"];
        CHECK(jp["space_level"] == li && jp["time_level"] == timeLevel[bi] && jp["recording"] == rec);
        pairs[rec + " " + tag] = std::move(c);
      }
    // each level's spatial band is recorded once a step, the resolved transform in the key
    for (int li : levels) for (int bi = 0; bi < 2; ++bi) {
      const std::string key = std::string("apply_kernel:sources_l") + std::to_string(li) + "_b" + (bi == 0 ? "8_16" : "16_32");
      CHECK(run["prepared"].contains(key) && run["prepared"][key]["spatial_band"]["level"] == li && run["prepared"][key]["spatial_band"]["transform"] == "exact");
    }
  }
  // ── 3. THE CHUNK WIDTH: the same samples stored in chunks of 4096 and of 1000 give the same cells, every pair ──
  {
    double diff = 0;
    for (int li : levels) for (int bi = 0; bi < 2; ++bi) {
      const std::string tag = "l" + std::to_string(li) + "_b" + (bi == 0 ? "8_16" : "16_32");
      const Cells &x = pairs["rest " + tag], &y = pairs["rest_b " + tag];
      for (std::size_t i = 0; i < x.a.size(); ++i) if (!std::isnan(x.a[i]) || !std::isnan(y.a[i])) diff = std::max(diff, static_cast<double>(std::abs(x.a[i] - y.a[i])));
      for (std::size_t i = 0; i < x.p.size(); ++i) if (!std::isnan(x.p[i]) || !std::isnan(y.p[i])) diff = std::max(diff, static_cast<double>(std::abs(x.p[i] - y.p[i])));
    }
    std::printf("  chunk widths 4096 and 1000, every pair: max difference %.2e\n", diff);
    CHECK(diff < 1e-5);
  }
  // ── 4. A RERUN IS A HIT; another pair set over rows that exist is another job's (refused unless --replace) ──
  {
    const auto h = g::run(ds, sub, d, set, all);
    CHECK(h.size() == 2 && h[0]["hit"] == true && h[1]["hit"] == true && h[0]["stored"].size() == 8);
    CHECK(code_of([&] { g::run(ds, sub, d, Json{{"space_levels", Json::array({3})}, {"bands", Json::array({bands[0]})}, {"method", "dSPM"}, {"transform", "exact"}}, all); }) == std::string(code::already_exists));
  }
  // ── 5. EQUAL TO FOUR SINGLE-PAIR RUNS cell for cell (the rows are replaced: the same measures on the same Selections) ──
  {
    g::RunOptions rep = all; rep.replace = true;
    double worst = 0, top = 0;
    std::int64_t compared = 0;
    for (int li : levels)
      for (int bi = 0; bi < 2; ++bi) {
        const std::string tag = "l" + std::to_string(li) + "_b" + (bi == 0 ? "8_16" : "16_32");
        const auto rs = g::run(ds, sub, one, Json{{"space_level", li}, {"band", Json::array({blo[bi], bhi[bi]})}, {"method", "dSPM"}, {"transform", "exact"}, {"spatial_shape", "meyer"}, {"shape", "meyer"}}, rep);
        CHECK(rs.size() == 2);
        for (const auto& run : rs) {
          const Cells& c = pairs[run["anchor"].get<std::string>() + " " + tag];
          const std::vector<float> a = nxrtest::read_packed(sub.folder(), out_of(run, "amplitude")), p = nxrtest::read_packed(sub.folder(), out_of(run, "peak"));
          CHECK(a.size() == c.a.size() && p.size() == c.p.size());
          for (std::size_t i = 0; i < a.size() && i < c.a.size(); ++i) {
            CHECK(std::isnan(a[i]) == std::isnan(c.a[i]));
            if (std::isnan(a[i])) continue;
            worst = std::max(worst, static_cast<double>(std::abs(a[i] - c.a[i]))); top = std::max(top, static_cast<double>(std::abs(a[i])));
            ++compared;
          }
          for (std::size_t i = 0; i < p.size() && i < c.p.size(); ++i) if (!std::isnan(p[i])) worst = std::max(worst, static_cast<double>(std::abs(p[i] - c.p[i])));
        }
      }
    std::printf("  one pass over 4 pairs vs four single-pair runs: %lld cells compared, worst difference %.2e of %.3g\n", static_cast<long long>(compared), worst, top);
    CHECK(compared > 1000 && worst <= 1e-6 * top);
  }
  return nxrtest::finish("test_cortical_cell_pairs");
}
