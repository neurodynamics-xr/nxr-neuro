// cyclepattern compute (spec docs/specs/2026-11-13-cycle-pattern.md, stage 2) against GROUND TRUTH, on the two glued unit spheres of test_phasegradient_band
// (one channel a vertex; a free kernel giving each source its channel along the outward normal, so the current is normal and equals the channel). Three
// 16 s recordings at 10 Hz through `crossspectrum compute --estimator analytic` (4 s windows, 125 Hz):
//   STANDING  cos(ωt) f(v), f = cos(3z): one pattern, real: its normal map is f (the band "modes"), its share ≈ 1, its standing share ≈ 0;
//   ROTATING  cos(ωt − 2λ) round each sphere's axis: the pattern e^{−2iλ}, as standing as it is travelling: standing share ≈ 1;
//   SWITCHING cos(ωt) f for 8 s, then cos(ωt) cos(2λ): the windows of the second half are unlike the first window, those of the first half alike.
// Also: the bridge NaN; Weyl members; the coefficient fields on the eigen axis; a re-run is a cache hit; refusals (a Welch cross-spectrum, a fixed kernel).
#include <unistd.h>

#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/task/task.h"
#include "nxr/bst/entities.h"
#include "nxr/io/statistic.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
using Cd = std::complex<double>;

// the decomposition's STATISTIC `<stem>` [vertex, pattern, quantity (stream, potential, normal), re/im]: one pattern's quantity as [vertex x 2] floats (the bridge NaN)
static std::vector<float> pat(SubjectWriter& sub, const std::string& stem, int pattern, int quantity) {
  const StoredStatistic st = read_statistic(sub, stem);
  const std::int64_t V = st.shape[0], q = st.shape[1];
  std::vector<float> out(static_cast<std::size_t>(2 * V));
  for (std::int64_t x = 0; x < V; ++x) for (int c = 0; c < 2; ++c) out[static_cast<std::size_t>(2 * x + c)] = static_cast<float>(st.values[static_cast<std::size_t>(((x * q + pattern) * 3 + quantity) * 2 + c)]);
  return out;
}
// the phase-free match |⟨x, y⟩| / (‖x‖ ‖y‖) of a stored complex field on the glued surface's cortex (glued vertex i is input vertex support[i]) with a
// complex reference on the input's vertices
static double match(const std::vector<float>& f, const std::vector<Cd>& ref, const std::vector<std::int64_t>& support) {
  Cd dot = 0; double a = 0, b = 0;
  for (std::size_t i = 0; i < support.size(); ++i) {
    const Cd x(f[2 * i], f[2 * i + 1]), y = ref[static_cast<size_t>(support[i])];
    dot += std::conj(x) * y; a += std::norm(x); b += std::norm(y);
  }
  return std::abs(dot) / std::sqrt(a * b);
}

static int run(const fs::path& root);
int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cyclepattern_" + std::to_string(::getpid()));
  fs::remove_all(root);
  int rc = 1;
  try { rc = run(root); } catch (const std::exception& e) { std::printf("  uncaught: %s\n", e.what()); }
  fs::remove_all(root);                                       // the test's dataset (hundreds of MB) is never left behind
  return rc;
}

static int run(const fs::path& root) {
  const Brain b = brain(3, false, 0.0, Vector3d(1, 1, 1));
  const int N = b.n, M = N, T = 4096;   // 4 whole windows of level 10 (1024 samples: 4 s at 250 Hz snaps to it); the switch at a window edge
  const double sfreq = 250.0, omega = 2 * M_PI * 10.0;
  std::vector<double> lam(static_cast<size_t>(N)), z(static_cast<size_t>(N));
  std::vector<std::int32_t> dk(static_cast<size_t>(N), 1);
  for (int v = 0; v < N; ++v) {
    const double cy = v < b.nHalf ? 1.3 : -1.3;
    const double x = b.P[static_cast<size_t>(3 * v)], y = b.P[static_cast<size_t>(3 * v + 1)] - cy;
    lam[static_cast<size_t>(v)] = std::atan2(y, x);
    z[static_cast<size_t>(v)] = b.P[static_cast<size_t>(3 * v + 2)];
    if ((v < b.nHalf && y < -0.75) || (v >= b.nHalf && y > 0.75)) dk[static_cast<size_t>(v)] = 0;   // the caps facing each other: the medial wall
  }
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  std::vector<float> stand(static_cast<size_t>(M * T)), rotate(static_cast<size_t>(M * T)), swtch(static_cast<size_t>(M * T)), plane(static_cast<size_t>(M * T));
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double w = omega * t / sfreq, f = std::cos(3 * z[static_cast<size_t>(v)]), g = std::cos(2 * lam[static_cast<size_t>(v)]);
      stand[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w) * f + 0.02 * nd(rng));
      rotate[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w - 2 * lam[static_cast<size_t>(v)]) + 0.02 * nd(rng));
      swtch[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w) * (t < T / 2 ? f : g) + 0.02 * nd(rng));
      plane[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w - 3.0 * z[static_cast<size_t>(v)]) + 0.02 * nd(rng));   // travelling along +z
    }
  std::vector<float> kf(static_cast<size_t>(3 * N * M), 0.0f);
  for (int v = 0; v < N; ++v) for (int c = 0; c < 3; ++c) kf[static_cast<size_t>((3 * v + c) * M + v)] = static_cast<float>(b.N(v, c));
  DatasetWriter ds = create_dataset(root, "demo");
  {   // the dataset's default subject: the spatial bands (the Eigenvalue line) are made in it
    SubjectWriter def = SubjectWriter::create(ds, "default_template", [] { SubjectOptions o; o.kind = "template"; return o; }());
    def.compose([&] {});
  }
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      const std::string surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<size_t>(3 * M));
      for (int i = 0; i < 3 * M; ++i) pos[static_cast<size_t>(i)] = static_cast<float>(1.2 * b.P[static_cast<size_t>(i)]);
      ManifoldSpec mm;
      mm.id = cid; mm.name = "sensors"; mm.path = "sensor/sensors"; mm.type = "points"; mm.n_vertices = M;
      mm.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; mm.geometry = g;
      mm.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(mm));
      std::vector<std::int64_t> zeros(static_cast<size_t>(M), 0);
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", zeros, {"MEG MAG"}, {}}, {"quality", zeros, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = lattice_counts({T});
      if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      for (const auto& [name, data] : {std::make_pair(std::string("stand"), &stand), std::make_pair(std::string("rotate"), &rotate), std::make_pair(std::string("switch"), &swtch), std::make_pair(std::string("plane"), &plane)}) {
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(*data, {M, T}); rec.array.chunks = Shape{M, 1700}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
      }
      std::vector<std::int32_t> chans(static_cast<size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      OperatorSpec op;
      op.name = "free"; op.path = "free"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
      op.data = view(kf, {3 * N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
      sub.operator_(std::move(op));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  EigenSolveOptions eo;
  eo.modes = 300;
  CHECK(eigen_solve(ds, sub, eo).info["glued"].get<bool>());
  for (const char* r : {"stand", "rotate", "switch", "plane"}) {
    CrossSpectrumOptions o;
    o.recording = r; o.chunks = "all"; o.channels_of = "free"; o.bands = {{"alpha", 8, 12}}; o.estimator = "analytic";
    CHECK_EQ(crossspectrum_compute(ds, sub, o).windows[0].sum.size(), std::size_t(16));   // stored at alpha's constant-Q window, level 8 (256 samples)
  }
  // the cortex on the glued surface: its first support.size() vertices, glued vertex i being input vertex support[i] (the package's support Selection)
  std::vector<std::int64_t> support;
  {
    const auto r = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "hodgeDirac_support", .any_status = true});
    CHECK(r.has_value());
    ArrayReader a = ArrayReader::open(sub.store(), *r->array_path);
    const auto raw = a.read();
    const std::size_t sz = dtype_size(a.dtype());
    for (std::size_t i = 0; i < raw.size() / sz; ++i) {
      if (sz == 4) { std::int32_t x; std::memcpy(&x, raw.data() + 4 * i, 4); support.push_back(x); }
      else { std::int64_t x; std::memcpy(&x, raw.data() + 8 * i, 8); support.push_back(x); }
    }
  }
  const int kept = static_cast<int>(support.size());
  // STANDING: one real pattern, the normal map f
  CyclePatternOptions co;
  co.cross_spectrum = "stand_crossspectrum_alpha_windows"; co.spatial_band = "modes"; co.patterns = 2;
  const CyclePatternResult cs = cyclepattern_compute(ds, sub, co);
  const auto& bs = cs.info["spatial_bands"][0];
  std::vector<Cd> f(static_cast<size_t>(N)), e2(static_cast<size_t>(N));
  for (int v = 0; v < N; ++v) { f[static_cast<size_t>(v)] = std::cos(3 * z[static_cast<size_t>(v)]); e2[static_cast<size_t>(v)] = std::polar(1.0, -2 * lam[static_cast<size_t>(v)]); }
  const auto rho = pat(sub, "stand_crossspectrum_alpha_cycle_modes", 0, 2);
  const double mStand = match(rho, f, support), share = bs["span_shares"][0].get<double>();
  const StoredStatistic sst = read_statistic(sub, "stand_crossspectrum_alpha_cycle_modes");
  const Json pj = sst.producer;
  CHECK(sst.row.at("measure") == V(std::string("cycle_pattern")) && sst.row.at("method") == V(std::string("eigendecomposition")) && sst.shape == std::vector<std::int64_t>({sst.shape[0], 2, 3, 2}));
  CHECK(!text(sst.row, "of_statistic_id").empty());             // the statistic it decomposes: the windowed cross-spectrum
  {
    bool per_pattern = false;
    for (const auto& fld : nxr::io::task::Subject::of(sub).find_all({.cls = nxr::io::task::Cls::field, .any_status = true})) per_pattern = per_pattern || fld.name.rfind("stand_crossspectrum_alpha_cycle", 0) == 0;
    CHECK(!per_pattern);
  }   // no per-pattern Field
  const double standing = pj["standing_share"][0].get<double>();
  std::printf("  standing: the normal map vs f %.4f; share %.3f; standing share %.4f\n", mStand, share, standing);
  CHECK(mStand > 0.99 && share > 0.95 && standing < 0.05);
  // THE WINDOW READ (the time-tile amendment): 4 s by default, snapped to level 10 (1024 samples), the stored level-8 tiles summed by fours; --level 8 reads
  // the stored tiles themselves; a level finer than the stored one is refused
  CHECK(pj["window_level"] == 10 && pj["window_level_stored"] == 8 && pj["windows"]["index"].size() == 4);
  {
    CyclePatternOptions c8 = co; c8.level = 8; c8.name = "stand_l8";
    CHECK_EQ(cyclepattern_compute(ds, sub, c8).info["windows"].get<std::int64_t>(), std::int64_t(16));
    CyclePatternOptions c7 = co; c7.level = 7; c7.name = "stand_l7";
    bool finer = false;
    try { cyclepattern_compute(ds, sub, c7); } catch (const Error&) { finer = true; }
    CHECK(finer);
  }
  // its imaginary part (after the rotation to the frame of largest variance) is ~0: a standing pattern is real
  double re = 0, im = 0;
  for (int v = 0; v < kept; ++v) { re += std::pow(rho[static_cast<size_t>(2 * v)], 2); im += std::pow(rho[static_cast<size_t>(2 * v + 1)], 2); }
  CHECK(im < 0.05 * re);
  // the bridge is NaN; the coefficient fields are on the eigen axis
  CHECK(rho.size() > static_cast<size_t>(2 * kept) && std::isnan(rho.back()));
  CHECK(read_statistic(sub, "stand_crossspectrum_alpha_cycle_modes_modes").shape.size() == 4 && !pat(sub, "stand_crossspectrum_alpha_cycle_modes", 1, 0).empty());
  // ROTATING: the pattern e^{−2iλ}, standing share ≈ 1
  co.cross_spectrum = "rotate_crossspectrum_alpha_windows";
  const CyclePatternResult cr = cyclepattern_compute(ds, sub, co);
  const auto rr = pat(sub, "rotate_crossspectrum_alpha_cycle_modes", 0, 2);
  const auto rpj = read_statistic(sub, "rotate_crossspectrum_alpha_cycle_modes").producer;
  const double mRot = match(rr, e2, support), stRot = rpj["standing_share"][0].get<double>();
  std::printf("  rotating: the normal map vs e^{-2i lambda} %.4f; share %.3f; standing share %.3f\n", mRot, cr.info["spatial_bands"][0]["span_shares"][0].get<double>(), stRot);
  // not exactly 1: the medial-wall caps (the bridge, not cortex) cut part of each circle, so Σ m² over the cortex is not 0
  CHECK(mRot > 0.97 && stRot > 0.8);
  // SWITCHING: the second half's windows are unlike the first window, the first half's alike
  co.cross_spectrum = "switch_crossspectrum_alpha_windows";
  const auto sw = cyclepattern_compute(ds, sub, co).info["spatial_bands"][0]["window_similarity_to_first"];
  double firstHalf = 1, secondHalf = 0;
  for (int w = 0; w < 4; ++w) { const double x = sw[static_cast<size_t>(w)].get<double>(); if (w < 2) firstHalf = std::min(firstHalf, x); else secondHalf = std::max(secondHalf, x); }
  std::printf("  switching: similarity to window 1, first half >= %.3f, second half <= %.3f\n", firstHalf, secondHalf);
  CHECK(firstHalf > 0.99 && secondHalf < 0.2);
  // THE LAGS OF A PATTERN: a plane wave cos(ωt − 3z) gives one pattern whose normal field's timing map is −3z/ω (slope 1, r ≈ 1), its lags almost all
  // timing; the rotating pattern's lags are mostly rotation (they circulate round each sphere's axis)
  {
    CyclePatternOptions cp; cp.cross_spectrum = "plane_crossspectrum_alpha_windows"; cp.spatial_band = "modes"; cp.patterns = 1; cp.timing_of = {"normal"};
    const CyclePatternResult pr = cyclepattern_compute(ds, sub, cp);
    const auto& lg = pr.info["spatial_bands"][0]["lags"][0];
    const StoredStatistic lagst = read_statistic(sub, "plane_crossspectrum_alpha_cycle_modes_lags");   // [vertex, pattern 1, of normal, (timing, speed, inconsistency)]
    CHECK(lagst.shape == std::vector<std::int64_t>({lagst.shape[0], 1, 1, 3}));
    std::vector<double> tm(support.size());
    for (std::size_t i = 0; i < support.size(); ++i) tm[i] = lagst.values[3 * i];
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0; const double n = static_cast<double>(support.size());
    for (std::size_t i = 0; i < support.size(); ++i) {
      const double x = -3.0 * z[static_cast<size_t>(support[i])] / omega, y = tm[i];
      sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    const double cov = sxy - sx * sy / n, r = cov / std::sqrt((sxx - sx * sx / n) * (syy - sy * sy / n)), slope = cov / (sxx - sx * sx / n);
    std::printf("  plane wave: timing map vs -kz/omega r %.4f slope %.3f; shares timing %.3f rotation %.3f\n", r, slope, lg["timing"].get<double>(), lg["rotation"].get<double>());
    CHECK(r > 0.99 && std::abs(slope - 1) < 0.05 && lg["timing"].get<double>() > 0.9);
    CyclePatternOptions cq = cp; cq.cross_spectrum = "rotate_crossspectrum_alpha_windows"; cq.name = "rotate_lags";
    const CyclePatternResult rq = cyclepattern_compute(ds, sub, cq);
    const auto& lr = rq.info["spatial_bands"][0]["lags"][0];
    std::printf("  rotating: shares timing %.3f rotation %.3f\n", lr["timing"].get<double>(), lr["rotation"].get<double>());
    CHECK(lr["rotation"].get<double>() > lr["timing"].get<double>());
    CyclePatternOptions bad = cp; bad.timing_of = {"phase"};
    bool refused = false; try { cyclepattern_compute(ds, sub, bad); } catch (const Error&) { refused = true; }
    CHECK(refused);
  }
  // Weyl members: every band stored; a re-run is a cache hit
  CyclePatternOptions cw;
  cw.cross_spectrum = "stand_crossspectrum_alpha_windows"; cw.spatial_band = "weyl:all"; cw.patterns = 1;
  const CyclePatternResult rw = cyclepattern_compute(ds, sub, cw);
  CHECK(!rw.hit && rw.info["spatial_bands"].size() >= 2);
  CHECK(cyclepattern_compute(ds, sub, cw).hit);
  // refusals: a Welch cross-spectrum is not a series; zero patterns
  CrossSpectrumOptions wo; wo.recording = "stand"; wo.chunks = "all"; wo.channels_of = "free"; wo.bands = {{"alpha", 8, 12}}; wo.name = "stand_welch";
  crossspectrum_compute(ds, sub, wo);
  auto refused = [&](CyclePatternOptions o) { try { cyclepattern_compute(ds, sub, o); } catch (const Error&) { return true; } return false; };
  { CyclePatternOptions o = co; o.cross_spectrum = "stand_welch_alpha"; CHECK(refused(o)); }
  { CyclePatternOptions o = co; o.patterns = 0; CHECK(refused(o)); }
  return nxrtest::finish("test_cyclepattern");
}
