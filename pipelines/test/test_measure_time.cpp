// measure_over_time on a small synthetic recording (no real data): 2 channels, 40000 samples at 1200 Hz (one whole chunk and a partial one), a constant-Q ladder written
// on its line and two closed-form spans tilings. Checks: each window's mean, mean_square, max, min, abs_max and count equal a direct computation on the float32 samples;
// the default span is chunk 0 (NaN beyond it); `--chunks all` measures the partial last window; a span given in seconds snaps to the windows (tiles), so none is cut by it; windows that cross the chunk edge give the
// same values as a direct computation (the result does not depend on the chunk size); the rows (D96: one array, one row a measure); refusal and replace; the refusals of
// a bad request; and the command.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>

#include "check.h"
#include "store_at.h"
#include "nxr/pipelines/constant_q.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/pipelines/measure.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::vector<float> read_floats(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
bool near(double a, double b) { return std::abs(a - b) <= 1e-5 * std::max(1.0, std::abs(b)); }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_measure_time_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  const std::int64_t M = 2, T = 40000;
  const double sfreq = 1200.0;
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (std::int64_t t = 0; t < T; ++t) {
    xf[static_cast<std::size_t>(t)] = static_cast<float>(3.0 * std::sin(2 * M_PI * 10.0 * t / sfreq) + 0.5 * nd(rng) + 1.0);
    xf[static_cast<std::size_t>(T + t)] = static_cast<float>(1e-12 * nd(rng));
  }
  DatasetWriter ds = create_dataset(root, "demo");
  std::string tid;
  ConstantQWritten cqw;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M * 3), 0.0f);
      pos[3] = 0.1f;
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors";
      m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = T; d.spacing = 1.0 / sfreq; d.origin = 0.0; d.unit = "s";
      tg.dimensions.push_back(d);
      tm.geometry = tg;
      tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = "rest"; rec.path = task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T});
      rec.array.chunks = Shape{M, 32768};
      rec.array.compress = false;
      rec.cols.set("unit", "T");
      sub.field(std::move(rec));
      // two closed-form tilings of the line: windows of 1000 samples (window 32 crosses the chunk edge at 32768), and windows not starting at the line's origin
      for (const auto& [name, origin] : {std::pair<std::string, double>{"rest_seconds", 0.0}, {"rest_offset", 0.5}}) {
        SelectionSpec s;
        s.name = name; s.type = "spans"; s.manifold_id = tid;
        s.cols.set("cell", "1").set("n_elements", T).set("n_members", static_cast<std::int64_t>(40));
        s.cols.json("params_json", Json{{"tiling", Json{{"origin", origin}, {"width", 1000.0 / sfreq}, {"hop", 1000.0 / sfreq}, {"count", 40}}}, {"end", static_cast<double>(T) / sfreq}});
        sub.selection(std::move(s));
      }
      cqw = constant_q_write(sub, tid, {}, std::string("test"));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  int lev = -1;   // the ordinal of the ladder level whose windows are 256 samples
  for (std::size_t i = 0; i < cqw.levels.size(); ++i) if (cqw.levels[i].window_samples == 256) lev = static_cast<int>(i);
  CHECK(lev >= 0);
  if (lev < 0) return 1;

  // the direct computation of window [a, b) of row r: {mean, mean_square, max, min, abs_max, count}
  auto direct = [&](std::int64_t r, std::int64_t a, std::int64_t b) {
    double s = 0, q = 0, mx = -1e300, mn = 1e300;
    for (std::int64_t t = a; t < b; ++t) { const double v = xf[static_cast<std::size_t>(r * T + t)]; s += v; q += v * v; mx = std::max(mx, v); mn = std::min(mn, v); }
    const double n = static_cast<double>(b - a);
    return std::vector<double>{s / n, q / n, mx, mn, std::max(std::abs(mx), std::abs(mn)), n};
  };
  // the stored array against the direct computation on windows [w0, w1]; NaN on every other window
  auto check_array = [&](const std::string& path, std::int64_t width, std::int64_t w0, std::int64_t w1, const char* what) {
    std::vector<std::int64_t> sh;
    const std::vector<float> a = read_floats((sub.folder() / (path)), &sh);
    const std::int64_t W = sh[1], C = sh[2];
    CHECK(sh[0] == M && C == 6);
    int bad = 0;
    for (std::int64_t r = 0; r < M; ++r)
      for (std::int64_t k = 0; k < W; ++k) {
        const float* v = a.data() + (r * W + k) * C;
        if (k < w0 || k > w1) { if (!std::isnan(v[0]) || !std::isnan(v[5])) ++bad; continue; }
        const auto d = direct(r, k * width, std::min(T, (k + 1) * width));
        for (std::int64_t c = 0; c < C; ++c) if (!near(v[c], d[static_cast<std::size_t>(c)])) ++bad;
      }
    std::printf("  %s: %lld windows x %lld components, %d mismatches\n", what, static_cast<long long>(W), static_cast<long long>(C), bad);
    CHECK_EQ(bad, 0);
  };

  // 1. a ladder level, the default span (chunk 0): windows 0..127 measured, the rest NaN
  MeasureOverTimeOptions o;
  o.over = "rest_time_constantQ";
  o.chunks = "0";                      // the first chunk's samples, stated
  o.level = lev;
  o.fields = "rest";
  o.measures = {"mean", "mean_square", "max", "min", "abs_max"};
  const auto r1 = measure_over_time(ds, sub, o);
  const Json& m1 = r1.info["measured"][0];
  CHECK(m1["windows"] == Json::array({0, 127}) && m1["span"] == "chunk 0");
  const std::string path = m1["array"].get<std::string>();
  check_array(path, 256, 0, 127, "ladder level, chunk 0");
  const auto rows = testing::Probe(sub).rows("selection_measurement", {{"array_path", path}}, "component");
  CHECK_EQ(static_cast<int>(rows.size()), 6);
  if (rows.size() == 6) {
    CHECK(text(rows[0], "measure") == "mean" && text(rows[5], "measure") == "count" && integer(rows[4], "component") == 4);
    CHECK(opt_text(rows[1], "unit") == std::optional<std::string>("(T)^2") && opt_text(rows[0], "unit") == std::optional<std::string>("T"));
  }
  // refused unless replace; replace keeps one set of rows
  CHECK_EQ(code_of([&] { measure_over_time(ds, sub, o); }), std::string(code::invalid_argument));
  o.replace = true;
  o.chunks = "all";
  const auto r2 = measure_over_time(ds, sub, o);
  CHECK(r2.info["measured"][0]["replaced"] == true && r2.info["measured"][0]["windows"] == Json::array({0, 156}));   // 40000 = 156 * 256 + 64: the partial last window too
  check_array(path, 256, 0, 156, "ladder level, all chunks");
  CHECK_EQ(static_cast<int>(testing::Probe(sub).count("selection_measurement", {{"array_path", path}})), 6);

  // 2. a span in seconds snaps to the tiling's windows (tiles of 256 samples;), so no window is cut by the span's edge: --start at sample
  //    120 snaps to 0 (windows 0..7), at 130 to 256 (windows 1..7); the snapped extent is what was measured
  o.chunks.reset();
  o.start = 120.0 / sfreq;
  o.stop = 2048.0 / sfreq;
  const auto r3 = measure_over_time(ds, sub, o);
  CHECK(r3.info["measured"][0]["windows"] == Json::array({0, 7}));
  check_array(path, 256, 0, 7, "ladder level, [120, 2048) snapped to [0, 2048)");
  o.start = 130.0 / sfreq;
  const auto r3b = measure_over_time(ds, sub, o);
  CHECK(r3b.info["measured"][0]["windows"] == Json::array({1, 7}));
  check_array(path, 256, 1, 7, "ladder level, [130, 2048) snapped to [256, 2048)");
  o.start.reset();
  o.stop.reset();

  // 3. windows crossing the chunk edge: the result does not depend on where the chunks fall
  MeasureOverTimeOptions s;
  s.over = "rest_seconds";
  s.fields = "rest";
  s.chunks = "all";
  s.measures = o.measures;
  const auto r4 = measure_over_time(ds, sub, s);
  CHECK(r4.info["measured"][0]["windows"] == Json::array({0, 39}));
  check_array(r4.info["measured"][0]["array"].get<std::string>(), 1000, 0, 39, "1000-sample windows, all chunks (window 32 across the chunk edge)");

  // 4. the refusals: a ladder without its level, a level on a non-ladder, an unknown measure, no Field, a tiling not from the line's start, a tiling not stored
  auto refused = [&](MeasureOverTimeOptions e) { return code_of([&] { measure_over_time(ds, sub, e); }); };
  { MeasureOverTimeOptions e = o; e.level.reset(); CHECK_EQ(refused(e), std::string(code::invalid_argument)); }
  { MeasureOverTimeOptions e = s; e.level = 0; CHECK_EQ(refused(e), std::string(code::invalid_argument)); }
  { MeasureOverTimeOptions e = s; e.replace = true; e.measures = {"median"}; CHECK_EQ(refused(e), std::string(code::invalid_argument)); }
  { MeasureOverTimeOptions e = s; e.replace = true; e.fields = "nope"; CHECK_EQ(refused(e), std::string(code::invalid_argument)); }
  { MeasureOverTimeOptions e = s; e.over = "rest_offset"; CHECK_EQ(refused(e), std::string(code::invalid_argument)); }
  { MeasureOverTimeOptions e = s; e.over = "nope"; CHECK_EQ(refused(e), std::string(code::no_such_row)); }

  ds.close();
#ifdef NXR_TEST_CLI
  {
    const std::string cli = NXR_TEST_CLI, d = (root / "demo").string();
    const std::string common = " run measure --dataset " + d + " --subject S1 --fields rest --chunks 0";
    CHECK_EQ(std::system((cli + common + " --over rest_time_constantQ --level " + std::to_string(lev) + " --replace > " + (root / "cli.log").string() + " 2>&1").c_str()), 0);
    CHECK(std::system((cli + common + " --over rest_time_constantQ --level " + std::to_string(lev) + " --weight none > /dev/null 2>&1").c_str()) != 0);
    DatasetWriter d2 = open_writer(root / "demo");
    SubjectWriter s2 = SubjectWriter::open(d2, "S1");
    CHECK_EQ(static_cast<int>(testing::Probe(s2).count("selection_measurement", {{"array_path", path}})), 5);   // the default measures and count
  }
#endif
  fs::remove_all(root);
  return nxrtest::finish("test_measure_time");
}
