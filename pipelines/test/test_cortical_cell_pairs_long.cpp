// run cortical_cell_pairs on a synthetic recording of SEVERAL LOOP CHUNKS (40960 samples; the chunk service's loop chunk is the stored chunk, 4096 samples here): CONTRIBUTING.md method rule 6, a result must not depend on the
// chunk size. Two bands with different halos (8-16 Hz, 16-32 Hz) x two spatial levels, `--chunks all` (ten loop chunks, a 12 Hz bump planted across the first edge): every cell, the time tiles on both sides of the edges included, equals the
// DIRECT computation on the whole recording at once (the band's analytic signal over all samples, the level's exact spatial band, |z|, the area-weighted mean per cell).
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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cortical_cell_pairs_long_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 40960;       // ten loop chunks of 4096 (the stored chunk): the halo crosses nine edges
  const double sfreq = 250.0, bad0 = 10.0, bad1 = 10.5;
    // the planted source: 12 Hz on a bump around one vertex of the left hemisphere, during time tiles 14 … 17 of the 256-sample level (3.58 s … 4.6 s ... see below), noise everywhere
  auto P = [&](int i) { return Eigen::Vector3d(b.P[3 * i], b.P[3 * i + 1], b.P[3 * i + 2]); };
  int centre = 0;
  for (int i = 0; i < b.nHalf; ++i) if (P(i).x() > P(centre).x()) centre = i;
  const std::int64_t W = 256, t0 = 13 * W, t1 = 19 * W;       // a 12 Hz bump planted ACROSS a loop-chunk edge (the loop chunk is the stored chunk, 4096 samples: time tile 16 starts at the edge)
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
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(4096))}) {   // the same samples, two chunk widths
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
  // ── the pairs and their joint partitions ──
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
  for (int li : levels) for (int bi = 0; bi < 2; ++bi) { JointComputeOptions jo; jo.recording = "rest"; jo.space_level = li; jo.time_level = timeLevel[bi]; joint_compute(ds, sub, jo); }
  const g::Definition d = g::load("cortical_cell_pairs");
  g::RunOptions all; all.chunks = "all";
  const Json set = Json{{"space_levels", Json::array({levels[0], levels[1]})}, {"bands", Json::array({Json::array({blo[0], bhi[0]}), Json::array({blo[1], bhi[1]})})}, {"method", "dSPM"}, {"transform", "exact"}};
  const auto r = g::run(ds, sub, d, set, all);
  CHECK(r.size() == 1 && r[0]["hit"] == false);
  CHECK(r[0]["run"]["chunks_read"].size() == 10 && r[0]["run"]["blocks"] == 10 && r[0]["run"]["halo"].get<int>() > 0);   // ten loop chunks: the halo crossed nine edges
  // ── the direct computation, whole recording at once ──
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  nxr::ladder::Mesh lm; lm.positions = S.vertices; lm.faces = S.faces; lm.n = N;
  const std::vector<double> va = nxr::ladder::vertexAreas(lm);
  for (int bi = 0; bi < 2; ++bi) {
    const auto gain = [&](double f) { return nxr::filterbank::meyerBand(f, blo[bi], bhi[bi], 1.0 / 3.0); };
    const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
    const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false);
    const auto im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
    Eigen::MatrixXd Re(N, T), Im(N, T);
    for (int v = 0; v < N; ++v) for (int t = 0; t < T; ++t) { Re(v, t) = re[static_cast<std::size_t>(v) * T + t]; Im(v, t) = im[static_cast<std::size_t>(v) * T + t]; }
    for (int li : levels) {
      const SurfaceLevel SL = read_surface_level(sub, S, li);
      SpatialBandOptions sbo; sbo.transform = "exact";
      const SpatialBand sband = make_spatial_band(S, SL, sbo);
      const std::int64_t tiles = 2 * (1 << li), F = frames[bi], W2 = Wb[bi];
      std::vector<double> tarea(static_cast<std::size_t>(tiles), 0.0);
      for (int v = 0; v < N; ++v) tarea[static_cast<std::size_t>(SL.tile[static_cast<std::size_t>(v)])] += va[static_cast<std::size_t>(v)];
      const Eigen::MatrixXd FRe = sband.apply(Re), FIm = sband.apply(Im);
      std::vector<double> amp(static_cast<std::size_t>(tiles * F), std::nan("")), mx(amp.size(), std::nan(""));
      for (std::int64_t k = 0; k < F; ++k) {
        bool whole = true;
        for (std::int64_t t = k * W2; t < (k + 1) * W2; ++t) { const double sec = t / sfreq; if (sec >= bad0 && sec < bad1) whole = false; }
        if (!whole) continue;
        std::vector<double> sum(static_cast<std::size_t>(tiles), 0.0), big(static_cast<std::size_t>(tiles), 0.0);
        for (int v = 0; v < N; ++v) {
          const int s = SL.tile[static_cast<std::size_t>(v)];
          double sv = 0;
          for (std::int64_t t = k * W2; t < (k + 1) * W2; ++t) {
            const double zr = static_cast<float>(FRe(v, t)), zi = static_cast<float>(FIm(v, t)), z = std::sqrt(zr * zr + zi * zi);
            sv += z; big[static_cast<std::size_t>(s)] = std::max(big[static_cast<std::size_t>(s)], z);
          }
          sum[static_cast<std::size_t>(s)] += va[static_cast<std::size_t>(v)] * sv;
        }
        for (std::int64_t s = 0; s < tiles; ++s) { amp[static_cast<std::size_t>(s * F + k)] = sum[static_cast<std::size_t>(s)] / (static_cast<double>(W2) * tarea[static_cast<std::size_t>(s)]); mx[static_cast<std::size_t>(s * F + k)] = big[static_cast<std::size_t>(s)]; }
      }
      const std::string tag = "l" + std::to_string(li) + "_b" + (bi == 0 ? "8_16" : "16_32");
      const std::vector<float> ga = nxrtest::read_packed(sub.folder(), out_of(r[0], "amplitude_" + tag)), gp = nxrtest::read_packed(sub.folder(), out_of(r[0], "peak_" + tag));
      auto worst = [&](const std::vector<float>& arr, const std::vector<double>& want, std::int64_t k0, std::int64_t k1) {   // time tiles [k0, k1) only
        double w = 0, top = 0;
        for (std::int64_t s = 0; s < tiles; ++s) for (std::int64_t k = k0; k < k1; ++k) {
          const std::size_t i = static_cast<std::size_t>(s * F + k);
          if (std::isnan(want[i])) { if (!std::isnan(arr[i * 2])) return 1e9; continue; }
          if (std::isnan(arr[i * 2])) return 1e9;
          top = std::max(top, std::abs(want[i])); w = std::max(w, std::abs(arr[i * 2] - want[i]));
        }
        return top > 0 ? w / top : w;
      };
      const std::int64_t edge = 4096 / W2;                                   // the first time tile of the second loop chunk (the planted bump straddles it)
      const double all_a = worst(ga, amp, 0, F), all_p = worst(gp, mx, 0, F), at_edge = worst(ga, amp, edge - 3, edge + 3);
      std::printf("  level %d, band %g-%g Hz: every cell vs direct %.2e (envelope_max %.2e); time tiles %lld..%lld round the loop-chunk edge %.2e\n", li, blo[bi], bhi[bi], all_a, all_p, static_cast<long long>(edge - 3), static_cast<long long>(edge + 2), at_edge);
      CHECK(all_a < 2e-4 && all_p < 2e-4 && at_edge < 2e-4);
      // the bump straddles the edge: its cells are the largest of the run, so the comparison there is not of noise
      double edge_top = 0, rest_top = 0;
      for (std::int64_t s = 0; s < tiles; ++s) for (std::int64_t k = 0; k < F; ++k) { const double a = amp[static_cast<std::size_t>(s * F + k)]; if (std::isnan(a)) continue; (k >= edge - 3 && k < edge + 3 ? edge_top : rest_top) = std::max(k >= edge - 3 && k < edge + 3 ? edge_top : rest_top, a); }
      if (bi == 0) CHECK(edge_top > 5 * rest_top);
    }
  }
  return nxrtest::finish("test_cortical_cell_pairs_long");
}
