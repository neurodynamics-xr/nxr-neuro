// THE INSTRUMENT'S BLUR (`pointspread compute`, pipelines/include/nxr/pipelines/pointspread.h; spec docs/specs/2026-11-14-spatial-events.md, "Statistics and nulls") on a synthetic sphere (never a real dataset). The head model is a SMOOTHING gain:
// sensor m (one at each vertex) reads a source at vertex u with weight exp(-d(m, u)^2 / (2 sigma^2)), the same for the three orientations; the kernel hands each sensor back to its own vertex (all three components). So the resolution matrix's column of a
// unit current at u is a Gaussian blob of KNOWN width sigma, centred on u. Shown:
//   - the peak is at the source (distance 0) for every sampled vertex and orientation
//   - a wider blur makes a larger patch: the diameter and the size of the chain grow with sigma (2 cm, then 6 cm); a sharper head model (sigma 0: the identity) makes the smallest
//   - the three orientations give the same blur here (the model has no orientation preference): no orientation is assumed, each is measured
//   - stored as ONE statistic of the recording [vertices x 3 x 6]; a rerun is a hit; the refusals (a kernel that is not the head model's, a reference level that is not a level, no recording)
//   (the sphere is small, 10 cm in radius: its finest hat is 18.6 cm wide, so a patch is wider than the blob it reads; the test fixes the ORDER and the localization, not the absolute width)
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "check.h"
#include "mesh_helpers.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/constant_q.h"
#include "nxr/pipelines/pointspread.h"
#include "nxr/pipelines/tiles.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_pointspread_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const double R = 0.1;
  const int T = 128;
  std::vector<double> U;
  std::vector<std::int32_t> F0;
  icosphere(3, U, F0, R);
  const int N = static_cast<int>(U.size() / 3), M = N;
  auto ang = [&](int a, int b) { const double c = (U[3 * a] * U[3 * b] + U[3 * a + 1] * U[3 * b + 1] + U[3 * a + 2] * U[3 * b + 2]) / (R * R); return R * std::acos(std::max(-1.0, std::min(1.0, c))); };
  // the head models: a Gaussian smoothing gain of width sigma (sigma <= 0: the identity); the kernel returns each sensor to its vertex, all three components
  auto gain = [&](double sigma) {
    std::vector<float> g(static_cast<std::size_t>(M) * 3 * N, 0.0f);
    for (int m = 0; m < M; ++m) for (int v = 0; v < N; ++v) { const double w = sigma > 0 ? std::exp(-std::pow(ang(m, v), 2) / (2 * sigma * sigma)) : (m == v ? 1.0 : 0.0); for (int c = 0; c < 3; ++c) g[static_cast<std::size_t>(m) * 3 * N + 3 * v + c] = static_cast<float>(w); }
    return g;
  };
  std::vector<float> kern(static_cast<std::size_t>(3) * N * M, 0.0f);
  for (int v = 0; v < N; ++v) for (int c = 0; c < 3; ++c) kern[static_cast<std::size_t>(3 * v + c) * M + v] = 1.0f;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::vector<float> x(static_cast<std::size_t>(M) * T);
  for (auto& e : x) e = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf;
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
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / 250.0; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(x, {M, T}); rec.array.chunks = Shape{M, T}; rec.array.compress = false; rec.cols.set("unit", "A");
      sub.field(std::move(rec));
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto add_forward = [&](const std::string& name, double sigma) {
        std::vector<float> g = gain(sigma);
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "forward"; op.from_manifold_id = surf; op.layout = "dense"; op.n_rows = M; op.n_cols = 3 * N;
        op.data = view(g, {M, 3 * N}); op.array.compress = false;
        op.cols.set("to_manifold_id", cid).set("from_value_type", "vector3").set("to_value_type", "scalar");
        sub.operator_(std::move(op));
      };
      add_forward("hm_sharp", 0.0); add_forward("hm_2cm", 0.02); add_forward("hm_6cm", 0.06);
      OperatorSpec op;
      op.name = "back"; op.path = "back"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
      op.data = view(kern, {3 * N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  tiles_compute(ds, sub, TilesComputeOptions{});
  std::printf("  %d vertices on a 10 cm sphere; head models: the identity, a 2 cm and a 6 cm Gaussian blur\n", N);
  struct Res { std::vector<double> err, diam, size, level; };
  auto run = [&](const std::string& hm) {
    PointSpreadOptions po;
    po.recording = "rest"; po.forward = hm; po.kernel = "back"; po.levels = {1, 2, 3, 4}; po.reference_level = 4; po.transform = "exact"; po.vertices = 6; po.name = "pointspread." + hm;
    const auto r = pointspread_compute(ds, sub, po);
    const StoredStatistic S = read_statistic(sub, "pointspread." + hm);
    CHECK(S.shape == std::vector<std::int64_t>({6, 3, 6}) && text(S.row, "measure") == "point_spread" && text(S.row, "method") == "point_source" && !r.hit);
    Res o;
    for (std::size_t i = 0; i < S.values.size(); i += 6) { o.err.push_back(S.values[i]); o.diam.push_back(S.values[i + 1]); o.level.push_back(S.values[i + 3]); o.size.push_back(S.values[i + 4]); }
    return std::make_pair(o, S);
  };
  auto mean = [](const std::vector<double>& v) { double s = 0; for (double e : v) s += e; return s / static_cast<double>(v.size()); };
  const auto [A, SA] = run("hm_sharp");
  const auto [B, SB] = run("hm_2cm");
  const auto [C, SC] = run("hm_6cm");
  double emax = 0;
  for (const Res* r : {&A, &B, &C}) for (const double e : r->err) emax = std::max(emax, std::isfinite(e) ? e : 1e9);
  std::printf("  the peak is at the source: the largest distance %.2e cm over 54 sources\n", emax);
  CHECK(emax < 1e-6);
  std::printf("  patch diameter (cm): sharp %.1f, 2 cm blur %.1f, 6 cm blur %.1f; size of the chain (cm): %.1f, %.1f, %.1f; characteristic level %.1f, %.1f, %.1f\n", mean(A.diam), mean(B.diam), mean(C.diam), mean(A.size), mean(B.size), mean(C.size), mean(A.level), mean(B.level), mean(C.level));
  CHECK(mean(A.diam) <= mean(B.diam) + 1e-9 && mean(B.diam) < mean(C.diam) && mean(C.diam) > 1.15 * mean(A.diam));
  CHECK(mean(A.size) <= mean(B.size) + 1e-9 && mean(B.size) <= mean(C.size) + 1e-9);
  // no orientation preference: the three orientations of one source give the same blur
  { double d = 0; for (std::size_t s = 0; s < 6; ++s) d = std::max(d, std::max(std::abs(C.diam[s * 3] - C.diam[s * 3 + 1]), std::abs(C.diam[s * 3] - C.diam[s * 3 + 2])));
    CHECK(d < 1e-9); }
  // a rerun is a hit; the refusals
  { PointSpreadOptions po; po.recording = "rest"; po.forward = "hm_2cm"; po.kernel = "back"; po.levels = {1, 2, 3, 4}; po.reference_level = 4; po.transform = "exact"; po.vertices = 6; po.name = "pointspread.hm_2cm";
    CHECK(pointspread_compute(ds, sub, po).hit);
    PointSpreadOptions b = po; b.reference_level = 7; CHECK(what_of([&] { pointspread_compute(ds, sub, b); }).find("reference level") != std::string::npos);
    b = po; b.recording = "no_such"; CHECK(what_of([&] { pointspread_compute(ds, sub, b); }).find("no_such") != std::string::npos);
    b = po; b.forward = "no_such"; CHECK(what_of([&] { pointspread_compute(ds, sub, b); }).find("no_such") != std::string::npos);
    b = po; b.list = {N + 3}; b.name = "pointspread.bad"; CHECK(what_of([&] { pointspread_compute(ds, sub, b); }).find("not one of") != std::string::npos); }
  return nxrtest::finish("test_pointspread") ? 1 : 0;
}
