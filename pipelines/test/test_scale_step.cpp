// THE SCALE STEP (spec docs/specs/2026-11-14-spatial-events.md, piece 4; decision 4: L2-normalised responses, Lindeberg 1998) on a synthetic sphere (never a real dataset): PLANTED GAUSSIAN BLOBS of known widths.
// A recording of an identity kernel (a source a channel) carries a Gaussian blob of width sigma at one vertex, oscillating at 10 Hz; `apply_kernel` with several hat levels, `power`, then `scale` give each vertex its
// characteristic level.
//   - the stored characteristic level of EVERY vertex equals the direct computation: the dense exact hat of each level applied to the blob, squared, over the squared L2 norm of the operator's row (its atom), the level of the maximum
//   - the level selected at the blob's centre TRACKS THE PLANTED WIDTH: the wider the blob, the coarser the level (the level's size 2π/√λ_c grows); the widths span several levels
//   - the plan: a field that is no power, a single-level field, a field not on the glued surface are refused
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>

#include "check.h"
#include "store_at.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/pipelines/runner.h"
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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_scale_step_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const double R = 0.1, sfreq = 250.0;                                         // a sphere of 10 cm radius: the cortex's scale
  const int T = 1024;
  std::vector<double> U;
  std::vector<std::int32_t> F0;
  icosphere(3, U, F0, R);
  const int N = static_cast<int>(U.size() / 3), M = N;
  const std::vector<double> sigmas = {0.012, 0.02, 0.035, 0.06};                // the planted widths (m)
  const int v0 = 0;
  auto blob = [&](double sigma) {
    std::vector<float> x(static_cast<std::size_t>(M) * T);
    for (int v = 0; v < N; ++v) {
      const double c = (U[3 * v] * U[0] + U[3 * v + 1] * U[1] + U[3 * v + 2] * U[2]) / (R * R), d = R * std::acos(std::max(-1.0, std::min(1.0, c)));
      const double a = std::exp(-d * d / (2 * sigma * sigma));
      for (int t = 0; t < T; ++t) x[static_cast<std::size_t>(v) * T + t] = static_cast<float>(a * std::cos(2 * M_PI * 10.0 * t / sfreq));
    }
    return x;
  };
  std::vector<float> ident(static_cast<std::size_t>(N) * M, 0.0f);
  for (int i = 0; i < N; ++i) ident[static_cast<std::size_t>(i) * M + i] = 1.0f;
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid;
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
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      for (std::size_t k = 0; k < sigmas.size(); ++k) {
        const std::vector<float> x = blob(sigmas[k]);
        FieldSpec rec;
        rec.name = "blob" + std::to_string(k); rec.path = node_path(sub.db(), prod, rec.name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(x, {M, T}); rec.array.chunks = Shape{M, T}; rec.array.compress = false; rec.cols.set("unit", "A");
        sub.field(std::move(rec));
      }
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
  for (std::size_t k = 0; k < sigmas.size(); ++k) { FrameTilingOptions fo; fo.recording = "blob" + std::to_string(k); fo.band = std::make_pair(8.0, 12.0); frame_tiling_compute(ds, sub, fo); }

  // the levels: those of the ladder the hats are made for (several, so the planted widths can spread over them)
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  const auto ladder = sub.db().one("SELECT n_members FROM selection WHERE subject_id = ? AND type = 'ladder'", {V(sub.id())});
  const int top = static_cast<int>(integer(*ladder, "n_members")) - 1;
  std::vector<int> lvs;
  for (int l = std::max(1, top - 4); l <= top; ++l) lvs.push_back(l);
  std::printf("  %d vertices on a %.0f cm sphere; ladder levels 0 … %d; hats at levels", N, R * 100, top);
  for (const int l : lvs) std::printf(" %d", l);
  std::printf("\n");
  CHECK(lvs.size() >= 4);
  const int NL = static_cast<int>(lvs.size());
  // the DIRECT operators: each level's dense exact hat on the sphere (the one closed piece is its own glued domain), its atoms' squared norms (the operator's rows), the hat's centre wavelength
  SpatialBandOptions sbo; sbo.transform = "exact"; sbo.spatial_shape = "mexhat"; sbo.glued = true;
  std::vector<Eigen::MatrixXd> H;
  std::vector<Eigen::VectorXd> n2;
  std::vector<double> size_cm;
  for (const int l : lvs) {
    const SpatialBand b = make_spatial_band(ds, sub, S, read_surface_level(sub, S, l), sbo);
    CHECK(b.glued_vertices == N);
    H.push_back(b.apply_glued(Eigen::MatrixXd::Identity(N, N)));
    n2.push_back(H.back().rowwise().squaredNorm());
    size_cm.push_back(100.0 * 2 * M_PI / std::sqrt(b.info["hat_centre_lambda"].get<double>()));
  }
  std::printf("  the levels' sizes (2π/√λ_c, cm):");
  for (const double s : size_cm) std::printf(" %.1f", s);
  std::printf("\n");
  for (std::size_t k = 1; k < size_cm.size(); ++k) CHECK(size_cm[k] < size_cm[k - 1]);           // a higher level is finer
  const std::string def_text = std::string("pipeline: scale_check\ninputs:\n  recording: {field: recording, each: true}\n  kernel: {operator: inverse kernel, from: recording.manifold.0}\n  tiles: {selection: ladder, manifold: kernel.to}\n"
                                           "  tiling: {selection: spans, manifold: recording.time, where: {name: \"{recording} frames 0.00257492 s\"}}\nsteps:\n  chunks:\n    over: recording\n    do:\n"
                                           "      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}\n      - {step: frames, in: {x: z, tiling: tiling}, out: zf}\n"
                                           "      - {step: apply_kernel, in: {x: zf, kernel: kernel, tiles: tiles}, with: {levels: [") + [&] { std::string s; for (const int l : lvs) s += (s.empty() ? "" : ", ") + std::to_string(l); return s; }() +
                             "], glued: true, transform: exact}, out: j}\n      - {step: power, in: {x: j}, out: p}\n      - {step: scale, in: {x: p}, with: {transform: exact}, out: lev}\n"
                             "      - {step: measure, in: {x: lev}, with: {bad_segments: ignore}, out: level}\noutputs:\n  level: {kind: characteristic level}\n";
  const g::Definition d = g::parse(parse_yaml(def_text, "f.yaml"), "f.yaml", def_text);
  g::RunOptions all; all.chunks = "all"; all.replace = true;
  std::vector<int> at_centre;
  const auto rall = g::run(ds, sub, d, Json::object(), all);
  CHECK(rall.size() == sigmas.size());
  for (std::size_t k = 0; k < sigmas.size(); ++k) {
    Json mine;
    for (const auto& r : rall) if (r["anchor"] == "blob" + std::to_string(k)) mine = r;
    CHECK(!mine.is_null() && mine["stored"].size() == 1);
    if (mine.is_null()) continue;
    const std::vector<float> got = read_floats((sub.folder() / (mine["stored"][0]["path"].get<std::string>())));
    // the direct: the blob's pattern through each level, squared, over the atom's squared norm, the level of the maximum
    std::vector<double> f(static_cast<std::size_t>(N));
    for (int v = 0; v < N; ++v) {
      const double c = (U[3 * v] * U[0] + U[3 * v + 1] * U[1] + U[3 * v + 2] * U[2]) / (R * R), dd = R * std::acos(std::max(-1.0, std::min(1.0, c)));
      f[static_cast<std::size_t>(v)] = std::exp(-dd * dd / (2 * sigmas[k] * sigmas[k]));
    }
    int wrong = 0;
    for (int v = 0; v < N; ++v) {
      double best = -1;
      int at = 0;
      for (int li = 0; li < NL; ++li) {
        double rr = 0;
        for (int u = 0; u < N; ++u) rr += H[static_cast<std::size_t>(li)](v, u) * f[static_cast<std::size_t>(u)];
        const double nr = rr * rr / n2[static_cast<std::size_t>(li)](v);
        if (nr > best) { best = nr; at = li; }
      }
      if (std::abs(got[static_cast<std::size_t>(v)] - lvs[static_cast<std::size_t>(at)]) > 1e-3) ++wrong;
    }
    at_centre.push_back(static_cast<int>(std::lround(got[static_cast<std::size_t>(v0)])));
    std::printf("  blob %.1f cm wide: level %d at its centre (size %.1f cm); %d of %d vertices differ from the direct level\n", sigmas[k] * 100, at_centre.back(), size_cm[static_cast<std::size_t>(std::find(lvs.begin(), lvs.end(), at_centre.back()) - lvs.begin())], wrong, N);
    CHECK(wrong <= N / 100);
  }
  // THE SELECTED SIZE TRACKS THE PLANTED WIDTH: wider blob, coarser level (lower index), and the widths span several levels
  bool monotone = true;
  for (std::size_t k = 1; k < at_centre.size(); ++k) monotone = monotone && at_centre[k] <= at_centre[k - 1];
  std::set<int> distinct(at_centre.begin(), at_centre.end());
  std::printf("  levels at the centre for widths %.1f … %.1f cm: ", sigmas.front() * 100, sigmas.back() * 100);
  for (const int l : at_centre) std::printf("%d ", l);
  std::printf("(monotone %s, %zu distinct)\n", monotone ? "yes" : "NO", distinct.size());
  CHECK(monotone && distinct.size() >= 3 && at_centre.front() > at_centre.back());
  // THE PLAN
  auto refused = [&](const std::string& steps) {
    const std::string t = std::string("pipeline: x\ninputs:\n  recording: {field: recording, each: true, where: {name: blob0}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0}\n  tiles: {selection: ladder, manifold: kernel.to}\n"
                                      "  tiling: {selection: spans, manifold: recording.time, where: {name: \"{recording} frames 0.00257492 s\"}}\nsteps:\n  chunks:\n    over: recording\n    do:\n"
                                      "      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}\n      - {step: frames, in: {x: z, tiling: tiling}, out: zf}\n") + steps +
                          "      - {step: measure, in: {x: lev}, with: {bad_segments: ignore}, out: level}\noutputs:\n  level: {kind: characteristic level}\n";
    return what_of([&] { g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), g::RunOptions{}); });
  };
  CHECK(refused("      - {step: apply_kernel, in: {x: zf, kernel: kernel, tiles: tiles}, with: {levels: [3, 4], glued: true, transform: exact}, out: j}\n      - {step: scale, in: {x: j}, out: lev}\n").find("band's power") != std::string::npos);
  CHECK(refused("      - {step: apply_kernel, in: {x: zf, kernel: kernel, tiles: tiles}, with: {level: 3, glued: true, transform: exact}, out: j}\n      - {step: power, in: {x: j}, out: p}\n      - {step: scale, in: {x: p}, out: lev}\n").find("multi-level") != std::string::npos);
  CHECK(refused("      - {step: apply_kernel, in: {x: zf, kernel: kernel, tiles: tiles}, with: {levels: [3, 4], transform: exact}, out: j}\n      - {step: power, in: {x: j}, out: p}\n      - {step: scale, in: {x: p}, out: lev}\n").find("glued") != std::string::npos);
  return nxrtest::finish("test_scale_step") || g_failures ? 1 : 0;
}
