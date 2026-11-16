// run cell_phase_current (spec docs/specs/2026-11-13-burst-phase-current.md) on a synthetic source space (never a real dataset): a two-hemisphere surface, a FREE kernel that is the identity on three
// components (the recording's channel 3v + c is component c of vertex v), a 12 Hz PLANE WAVE of known wave vector k (z_c = u_c exp(-i k.x) after the analytic signal: the phase falls along k, the wave
// travels along k, phase_current = |z|^2 grad(phase) = -|u|^2 k_tan), a STANDING wave of the same pattern (no phase gradient), and the plane wave with component signs flipped.
// Checks: the stored cell phase current (a vector measurement [spatial tiles x time tiles x 4]: components 0-2 the vector, 3 the count; the rows phase_current_x/y/z and count) and power equal the DIRECT
// computation (the band's analytic signal, the level's spatial band on each component, nxr::phase::phaseCurrent, the area-weighted mean per cell); ONE count row a Selection and recording; the roll-up over
// the stored bursts is the count-weighted mean of the cells over each burst's tiles (computed directly from the stored cells), a global burst on every row; the derived phase_gradient is the ratio of the
// rolled-up means and points along -k for the plane wave, ~0 for the standing wave; component signs flipped leave the result unchanged (a global and a per-component sign: a per-vertex flip would make a
// discontinuity, and the gradient there would change); the chunk width (4096 and 1000) and the sub-block width do not matter; the plan's refusals (a field not the phase_current step's, a joint partition
// that is not the band's level, stored spans of another recording, spans off the tile edges).
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/forms.h"
#include "nxr/io.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/measures.h"
#include "nxr/io/yaml.h"
#include "nxr/ladder.h"
#include "nxr/phase.h"
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
template <class F> std::string why_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cell_phase_current_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = 3 * N, T = 4096;
  const double sfreq = 250.0, f0 = 12.0;
  const int L = 4;
  const std::int64_t W = 256, frames = T / W;
  const Eigen::Vector3d k = 2.7 * Eigen::Vector3d(0.9, 0.4, -0.3).normalized(), u(1.0, 0.5, -0.7);   // the wave vector (rad/length: |k| 2.7 is inside the level-4 band, whose pass peaks near 2.5-3 on this surface) and the polarisation of the plane wave
  auto P = [&](int i) { return Eigen::Vector3d(b.P[3 * i], b.P[3 * i + 1], b.P[3 * i + 2]); };
  // recordings: channel 3v + c = component c of vertex v
  auto make = [&](double (*phase)(double, double), const Eigen::Vector3d& sgn) {
    std::vector<float> x(static_cast<std::size_t>(M) * T);
    for (int v = 0; v < N; ++v) {
      const double th = k.dot(P(v));
      for (int c = 0; c < 3; ++c)
        for (int t = 0; t < T; ++t) x[static_cast<std::size_t>(3 * v + c) * T + t] = static_cast<float>(sgn(c) * u(c) * phase(th, 2 * M_PI * f0 * t / sfreq));
    }
    return x;
  };
  const std::vector<float> xplane = make([](double th, double wt) { return std::cos(th - wt); }, Eigen::Vector3d(1, 1, 1));       // cos(k.x - wt): travels along +k
  const std::vector<float> xstand = make([](double th, double wt) { return std::cos(th) * std::cos(wt); }, Eigen::Vector3d(1, 1, 1));   // the same pattern standing
  const std::vector<float> xflip = make([](double th, double wt) { return std::cos(th - wt); }, Eigen::Vector3d(-1, 1, -1));        // component signs flipped
  std::vector<float> kfree(static_cast<std::size_t>(M) * M, 0.0f);
  for (int i = 0; i < M; ++i) kfree[static_cast<std::size_t>(i) * M + i] = 1.0f;
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, cid_s;
  std::map<std::string, std::string> rid;
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
      struct Rec { const char* name; const std::vector<float>* x; std::int64_t chunk; };
      for (const Rec& r : {Rec{"rest", &xplane, 4096}, Rec{"rest_b", &xplane, 1000}, Rec{"stand", &xstand, 4096}, Rec{"flip", &xflip, 4096}}) {   // the plane wave in two chunk widths
        FieldSpec rec;
        rec.name = r.name; rec.path = node_path(sub.db(), prod, r.name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(*r.x, {M, T}); rec.array.chunks = Shape{M, r.chunk}; rec.array.compress = false; rec.cols.set("unit", "A");
        rid[r.name] = sub.field(std::move(rec));
      }
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
  int timeLevel = -1;
  for (const Row& m : sub.db().all("SELECT member_ordinal, params_json FROM selection WHERE subject_id = ? AND member_of_id IS NOT NULL AND params_json LIKE '%resolves_hz%'", {V(sub.id())})) {
    const Json pj = Json::parse(text(m, "params_json"));
    if (pj["resolves_hz"]["lo"] == 8.0 && pj["resolves_hz"]["hi"] == 16.0 && pj["window_samples"] == W) timeLevel = static_cast<int>(integer(m, "member_ordinal"));
  }
  CHECK(timeLevel >= 0);
  if (timeLevel < 0) return 1;
  for (const char* r : {"rest", "rest_b", "stand", "flip"}) { JointComputeOptions jo; jo.recording = r; jo.space_level = L; jo.time_level = timeLevel; joint_compute(ds, sub, jo); }
  { JointComputeOptions jo; jo.recording = "rest"; jo.space_level = 3; jo.time_level = timeLevel; joint_compute(ds, sub, jo); }   // a joint partition of ANOTHER spatial level (a refusal below)

  // THE STORED BURSTS (a spans Selection of each recording on its time line, as `global_bursts` stores them: global, no channel): tiles [3, 7) and [10, 14) of the 1.024 s level
  const double tw = W / sfreq;
  auto store_spans = [&](const std::string& name, const std::string& rec, const std::vector<nxr::bst::TimeSpan>& spans, const std::string& pipeline) {
    SubjectWriter w = SubjectWriter::open(ds, "S1");
    w.compose([&] {
      nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid.at(rec); bo.description = name;
      bo.params = Json{{"pipeline", Json{{"name", pipeline}}}, {"inputs", Json{{"recording", Json{{"name", rec}}}}}};
      nxr::bst::time_spans(w, "time/" + name, spans, {"burst"}, bo);
    });
  };
  const std::vector<nxr::bst::TimeSpan> bursts = {{3 * tw, 7 * tw, 0}, {10 * tw, 14 * tw, 0}};
  for (const char* r : {"rest", "rest_b", "stand", "flip"}) store_spans(std::string("global_bursts.bursts.") + r, r, bursts, "global_bursts");

  // ── THE DIRECT COMPUTATION: the band's analytic signal on every channel, the level's spatial band on each component, nxr::phase::phaseCurrent, the area-weighted mean per cell ──
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  const SurfaceLevel SL = read_surface_level(sub, S, L);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S, SL, sbo);
  const std::int64_t tiles = SL.nodes;
  nxr::ladder::Mesh lm; lm.positions = S.vertices; lm.faces = S.faces; lm.n = N;
  const std::vector<double> va = nxr::ladder::vertexAreas(lm);
  std::vector<double> tarea(static_cast<std::size_t>(tiles), 0.0);
  for (int v = 0; v < N; ++v) tarea[static_cast<std::size_t>(SL.tile[static_cast<std::size_t>(v)])] += va[static_cast<std::size_t>(v)];
  std::vector<float> Pf(S.vertices.begin(), S.vertices.end());
  const auto fop = nxr::forms::assembleFaceGradient(Pf.data(), S.faces.data(), N, static_cast<int>(S.n_faces()));
  Eigen::MatrixXd Pm(N, 3), Nm(N, 3);
  Eigen::Matrix<int32_t, Eigen::Dynamic, 3, Eigen::RowMajor> Fm(fop.nF, 3);
  for (int v = 0; v < N; ++v) for (int d = 0; d < 3; ++d) { Pm(v, d) = fop.position[3 * v + d]; Nm(v, d) = fop.vertexNormals[3 * v + d]; }
  for (int f = 0; f < fop.nF; ++f) for (int d = 0; d < 3; ++d) Fm(f, d) = fop.faces[3 * f + d];
  const nxr::phase::Gradient G = nxr::phase::vertexGradient(Pm, Fm);
  const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 16.0); };
  const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
  struct Direct { std::vector<double> pc, pw; };   // [tiles x frames x 3], [tiles x frames]
  auto direct = [&](const std::vector<float>& x) {
    const auto re = nxr::taps::convolveRows(x.data(), M, T, h, 0, T, false);
    const auto im = nxr::taps::convolveRows(x.data(), M, T, q, 0, T, true);
    Eigen::MatrixXcd Z(M, T);
    for (int c = 0; c < 3; ++c) {
      Eigen::MatrixXd Re(N, T), Im(N, T);
      for (int v = 0; v < N; ++v) for (int t = 0; t < T; ++t) { Re(v, t) = re[static_cast<std::size_t>(3 * v + c) * T + t]; Im(v, t) = im[static_cast<std::size_t>(3 * v + c) * T + t]; }
      const Eigen::MatrixXd FRe = sband.apply(Re), FIm = sband.apply(Im);
      for (int v = 0; v < N; ++v) for (int t = 0; t < T; ++t) Z(3 * v + c, t) = {static_cast<double>(static_cast<float>(FRe(v, t))), static_cast<double>(static_cast<float>(FIm(v, t)))};
    }
    const Eigen::MatrixXd J = nxr::phase::phaseCurrent(Z, 3, G, Nm);
    Direct out;
    out.pc.assign(static_cast<std::size_t>(tiles * frames * 3), 0.0);
    out.pw.assign(static_cast<std::size_t>(tiles * frames), 0.0);
    for (std::int64_t kf = 0; kf < frames; ++kf)
      for (int v = 0; v < N; ++v) {
        const std::int64_t s = SL.tile[static_cast<std::size_t>(v)];
        double pw = 0, pc[3] = {0, 0, 0};
        for (std::int64_t t = kf * W; t < (kf + 1) * W; ++t) {
          for (int c = 0; c < 3; ++c) { pw += std::norm(Z(3 * v + c, t)); pc[c] += J(3 * v + c, t); }
        }
        // J rows are [3v + d]: the vector of vertex v (the phase_current step's three rows a vertex)
        out.pw[static_cast<std::size_t>(s * frames + kf)] += va[static_cast<std::size_t>(v)] * pw / (static_cast<double>(W) * tarea[static_cast<std::size_t>(s)]);
        for (int c = 0; c < 3; ++c) out.pc[static_cast<std::size_t>((s * frames + kf) * 3 + c)] += va[static_cast<std::size_t>(v)] * pc[c] / (static_cast<double>(W) * tarea[static_cast<std::size_t>(s)]);
      }
    return out;
  };
  // relative worst error of a stored array's component `comp` (of `C` per element) against the direct values, over the cells whose time tile is not at the recording's edges (the filter's end effects are the same in both)
  auto worst = [&](const std::vector<float>& arr, int C, int comp, const std::vector<double>& want, int wc, int wn) {
    double w = 0, top = 0;
    for (std::int64_t i = 0; i < tiles * frames; ++i) {
      const double got = arr[static_cast<std::size_t>(i * C + comp)], exp = want[static_cast<std::size_t>(i * wn + wc)];
      if (std::isnan(got)) return 1e9;
      top = std::max(top, std::abs(exp));
      w = std::max(w, std::abs(got - exp));
    }
    return w / top;
  };

  // ── 1. THE RUN: every recording, exact spatial band, the free kernel ──
  const g::Definition d = g::load("cell_phase_current");
  g::RunOptions all; all.chunks = "all";
  const Json set1 = Json{{"space_level", L}, {"transform", "exact"}, {"spatial_shape", "meyer"}};
  const auto plans = g::plan_runs(d, sub.db(), sub.id(), set1);
  CHECK_EQ(static_cast<int>(plans.size()), 4);
  const auto r1 = g::run(ds, sub, d, set1, all);
  CHECK_EQ(static_cast<int>(r1.size()), 4);
  std::map<std::string, Json> run_of;
  for (const auto& r : r1) run_of[r["anchor"].get<std::string>()] = r;
  const Direct dplane = direct(xplane), dstand = direct(xstand);
  std::map<std::string, std::vector<float>> cellpc;
  for (const char* rec : {"rest", "rest_b", "stand", "flip"}) {
    const Json& r = run_of.at(rec);
    CHECK(r["hit"] == false);
    const Json pc = out_of(r, "cell_phase_current"), pw = out_of(r, "cell_power");
    CHECK(!pc.is_null() && !pw.is_null());
    std::vector<std::int64_t> shape;
    cellpc[rec] = nxrtest::read_packed(sub.folder(), pc, &shape);
    CHECK(shape == (std::vector<std::int64_t>{tiles, frames, 4}));        // components 0-2 the vector, 3 the count
    const std::vector<float> cpw = nxrtest::read_packed(sub.folder(), pw, &shape);
    CHECK(shape == (std::vector<std::int64_t>{tiles, frames, 2}));
    if (std::string(rec) == "rest" || std::string(rec) == "stand" ) {
      const Direct& D = std::string(rec) == "rest" ? dplane : dstand;
      double ex = 0;
      for (int c = 0; c < 3; ++c) ex = std::max(ex, worst(cellpc[rec], 4, c, D.pc, c, 3));
      const double ep = worst(cpw, 2, 0, D.pw, 0, 1);
      std::printf("  %s: cell phase_current vs direct: worst relative error %.2e; power %.2e\n", rec, ex, ep);
      if (std::string(rec) == "rest") CHECK(ex < 1e-3);
      else {   // the standing wave: the direct current is ~0 itself (float rounding only): compare to the plane wave's scale
        double top = 0, w = 0;
        for (double v : dplane.pc) top = std::max(top, std::abs(v));
        for (std::size_t i = 0; i < D.pc.size(); ++i) w = std::max(w, std::abs(static_cast<double>(cellpc[rec][(i / 3) * 4 + i % 3]) - D.pc[i]));
        CHECK(w < 1e-3 * top);
      }
      CHECK(ep < 1e-3);
    }
    // the count row: ONE a Selection and recording, the samples of a whole tile, at component 3 of the vector measurement's array
    const std::string jid = [&] { const auto row = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(pc["over"].get<std::string>())}); return text(*row, "id"); }();
    const auto crow = sub.db().all("SELECT array_path, component FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'count'", {V(jid), V(rid.at(rec))});
    CHECK(crow.size() == 1);
    const auto kinds = sub.db().all("SELECT measure, component FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? ORDER BY measure", {V(jid), V(rid.at(rec))});
    CHECK(kinds.size() == 5 && text(kinds[0], "measure") == "count" && text(kinds[1], "measure") == "phase_current_x" && text(kinds[2], "measure") == "phase_current_y" && text(kinds[3], "measure") == "phase_current_z" && text(kinds[4], "measure") == "power");
    CHECK(integer(kinds[1], "component") == 0 && integer(kinds[2], "component") == 1 && integer(kinds[3], "component") == 2);
    const MeasurementValues cv = measurement_values(sub, jid, rid.at(rec), std::nullopt, std::nullopt, "count");
    CHECK(cv.rows == tiles && cv.members == frames && cv.values[static_cast<std::size_t>(5)] == static_cast<double>(W));
    const MeasurementValues px = measurement_values(sub, jid, rid.at(rec), std::nullopt, std::nullopt, "phase_current_x");
    CHECK(px.rows == tiles && px.members == frames);
    for (std::int64_t i = 0; i < tiles * frames; ++i) CHECK(px.values[static_cast<std::size_t>(i)] == static_cast<double>(cellpc[rec][static_cast<std::size_t>(i * 4)]));   // measurement_values reads each component by its own row
  }
  // the chunk width: the plane wave stored in chunks of 4096 and of 1000 gives the same cells
  {
    double diff = 0;
    for (std::size_t i = 0; i < cellpc["rest"].size(); ++i) diff = std::max(diff, static_cast<double>(std::abs(cellpc["rest"][i] - cellpc["rest_b"][i])));
    std::printf("  chunk widths 4096 and 1000: max cell difference %.2e\n", diff);
    CHECK(diff < 1e-5);
  }
  // ORIENTATION-FREE: component signs flipped (a per-component sign and a global one) leave the cells unchanged
  {
    double diff = 0, top = 0;
    for (std::size_t i = 0; i < cellpc["rest"].size(); ++i) if (i % 4 != 3) { diff = std::max(diff, static_cast<double>(std::abs(cellpc["rest"][i] - cellpc["flip"][i]))); top = std::max(top, static_cast<double>(std::abs(cellpc["rest"][i]))); }
    std::printf("  component signs flipped (-,+,-): max difference %.2e of %.3g\n", diff, top);
    CHECK(diff < 1e-5 * top);
  }

  // ── THE ROLL-UP over each stored burst: the count-weighted mean of the cells over its tiles, computed directly from the stored cells; a global burst holds on every spatial tile ──
  struct Burst { std::vector<double> pc, pw; std::vector<double> grad; std::string sel; };   // pc [tiles x 2 x 3], pw [tiles x 2]
  std::map<std::string, Burst> rolled;
  for (const char* rec : {"rest", "rest_b", "stand", "flip"}) {
    const Json& r = run_of.at(rec);
    const Json bpc = out_of(r, "burst_phase_current"), bpw = out_of(r, "burst_power");
    CHECK(!bpc.is_null() && !bpw.is_null() && bpc["over"] == std::string("global_bursts.bursts.") + rec);
    std::vector<std::int64_t> shape;
    const std::vector<float> a = nxrtest::read_packed(sub.folder(), bpc, &shape);
    CHECK(shape == (std::vector<std::int64_t>{tiles, 2, 4}));
    const std::vector<float> w = nxrtest::read_packed(sub.folder(), bpw, &shape);
    CHECK(shape == (std::vector<std::int64_t>{tiles, 2, 2}));
    // directly: the mean of the stored cells' values over the burst's tiles (every tile whole: equal counts)
    const std::vector<float>& cp = cellpc[rec];
    const std::vector<float> cw = nxrtest::read_packed(sub.folder(), out_of(r, "cell_power"));
    double worst_pc = 0, worst_pw = 0, top = 0;
    const std::int64_t firsts[2] = {3, 10}, stops[2] = {7, 14};
    for (std::int64_t s = 0; s < tiles; ++s)
      for (int e = 0; e < 2; ++e) {
        double mw = 0;
        for (std::int64_t t = firsts[e]; t < stops[e]; ++t) mw += cw[static_cast<std::size_t>((s * frames + t) * 2)];
        mw /= static_cast<double>(stops[e] - firsts[e]);
        worst_pw = std::max(worst_pw, std::abs(w[static_cast<std::size_t>((s * 2 + e) * 2)] - mw) / std::max(mw, 1e-12));
        for (int c = 0; c < 3; ++c) {
          double m = 0;
          for (std::int64_t t = firsts[e]; t < stops[e]; ++t) m += cp[static_cast<std::size_t>((s * frames + t) * 4 + c)];
          m /= static_cast<double>(stops[e] - firsts[e]);
          top = std::max(top, std::abs(m));
          worst_pc = std::max(worst_pc, std::abs(a[static_cast<std::size_t>((s * 2 + e) * 4 + c)] - m));
        }
        CHECK(a[static_cast<std::size_t>((s * 2 + e) * 4 + 3)] == static_cast<float>(W * (stops[e] - firsts[e])));    // the count: the samples over the burst's tiles
      }
    std::printf("  %s: burst roll-up vs the cells' mean: phase_current %.2e (of %.3g), power %.2e (relative)\n", rec, worst_pc, top, worst_pw);
    CHECK(worst_pc < 1e-5 * std::max(top, 1e-12) + 1e-9 && worst_pw < 1e-5);
    // the burst's count row: one a Selection and recording, on the bursts' Selection
    const auto brow = sub.db().one("SELECT id FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(bpc["over"].get<std::string>())});
    CHECK(brow.has_value());
    CHECK(sub.db().all("SELECT 1 FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = 'count'", {V(text(*brow, "id")), V(rid.at(rec))}).size() == 1);
    // the DERIVED phase gradient is the ratio of the rolled-up means, per component: rad/m
    Burst B;
    B.sel = text(*brow, "id");
    B.grad.assign(static_cast<std::size_t>(tiles * 2 * 3), 0.0);
    const char* gn[3] = {"phase_gradient_x", "phase_gradient_y", "phase_gradient_z"};
    for (int c = 0; c < 3; ++c) {
      const MeasurementValues gv = derived_measurement_values(sub, B.sel, rid.at(rec), std::nullopt, std::nullopt, gn[c]);
      CHECK(gv.rows == tiles && gv.members == 2);
      for (std::int64_t s = 0; s < tiles; ++s)
        for (int e = 0; e < 2; ++e) {
          const double want = static_cast<double>(a[static_cast<std::size_t>((s * 2 + e) * 4 + c)]) / static_cast<double>(w[static_cast<std::size_t>((s * 2 + e) * 2)]);
          const double got = gv.values[static_cast<std::size_t>(s * 2 + e)];
          B.grad[static_cast<std::size_t>((s * 2 + e) * 3 + c)] = got;
          CHECK(std::abs(got - want) <= 1e-6 * std::abs(want) + 1e-9);
        }
    }
    rolled[rec] = B;
  }
  // THE DIRECTION: the whole-brain phase gradient of a burst (the area-weighted sum of the tiles' means over the tiles' power: a ratio of sums) of the plane wave points along -k (phase falls along the
  // travelling direction); the standing wave has none. Per tile the expected is the tangent projection of -k, averaged over the tile's vertices.
  {
    auto whole = [&](const Burst& B, const std::vector<float>& cells, const std::vector<float>& cpw, int e, double out[3]) {
      (void)cells; (void)cpw; (void)B; (void)e; (void)out;
    };
    (void)whole;
    auto brain_gradient = [&](const std::string& rec, int e, Eigen::Vector3d& gvec, double& power) {
      const Json bpc = out_of(run_of.at(rec), "burst_phase_current"), bpw = out_of(run_of.at(rec), "burst_power");
      const std::vector<float> a = nxrtest::read_packed(sub.folder(), bpc), w = nxrtest::read_packed(sub.folder(), bpw);
      gvec.setZero(); power = 0;
      for (std::int64_t s = 0; s < tiles; ++s) {
        for (int c = 0; c < 3; ++c) gvec(c) += tarea[static_cast<std::size_t>(s)] * a[static_cast<std::size_t>((s * 2 + e) * 4 + c)];
        power += tarea[static_cast<std::size_t>(s)] * w[static_cast<std::size_t>((s * 2 + e) * 2)];
      }
      gvec /= power;
    };
    // the mean tangent projection of -k over the vertices (area-weighted): what the unfiltered plane wave gives
    Eigen::Vector3d ktan = Eigen::Vector3d::Zero();
    double atot = 0;
    for (int v = 0; v < N; ++v) { const Eigen::Vector3d n(Nm(v, 0), Nm(v, 1), Nm(v, 2)); ktan += va[static_cast<std::size_t>(v)] * (k - k.dot(n) * n); atot += va[static_cast<std::size_t>(v)]; }
    ktan /= atot;
    for (int e = 0; e < 2; ++e) {
      Eigen::Vector3d gp, gs;
      double pp, ps;
      brain_gradient("rest", e, gp, pp);
      brain_gradient("stand", e, gs, ps);
      const double cosk = gp.dot(-ktan) / (gp.norm() * ktan.norm());
      std::printf("  burst %d: whole-brain phase gradient (%.3f %.3f %.3f) rad/m, cos with -k_tan %.4f, |g|/|k_tan| %.3f; standing wave |g| %.2e\n", e, gp(0), gp(1), gp(2), cosk, gp.norm() / ktan.norm(), gs.norm());
      CHECK(cosk > 0.95);                       // the phase falls along k: the gradient points along -k, the wave travels along +k
      // the length is SHRUNK, not |k|: the level's band is a band-pass on a curved, coarsely meshed surface, so what is measured is the gradient of the band's own field, not of the unfiltered wave (`docs/reference.md`
      // limit: the gradient is of the band at the level). It is bracketed, never claimed equal to |k|.
      CHECK(gp.norm() > 0.25 * ktan.norm() && gp.norm() < 1.25 * ktan.norm());
      CHECK(gs.norm() < 0.02 * gp.norm());      // a standing wave has no phase gradient
    }
  }

  // ── THE SUB-BLOCK WIDTH: the pointwise tail on blocks of 700 samples gives the same cells and bursts (another job's key: --replace) ──
  {
    g::RunOptions blk = all; blk.block_samples = 700; blk.replace = true;
    const auto rb = g::run(ds, sub, d, set1, blk);
    double diff = 0;
    for (const auto& r : rb) {
      const std::string rec = r["anchor"].get<std::string>();
      const std::vector<float> a = nxrtest::read_packed(sub.folder(), out_of(r, "cell_phase_current"));
      for (std::size_t i = 0; i < a.size(); ++i) diff = std::max(diff, static_cast<double>(std::abs(a[i] - cellpc[rec][i])));
    }
    std::printf("  sub-block 700: max cell difference %.2e\n", diff);
    CHECK(diff < 1e-5);
  }
  CHECK(g::run(ds, sub, d, set1, [&] { g::RunOptions o = all; o.block_samples = 700; return o; }())[0]["hit"] == true);   // a rerun is a hit

  // ── THE PLAN'S REFUSALS ──
  {
    const std::string head = R"(pipeline: )";
    auto defn = [&](const std::string& name, const std::string& inputs_extra, const std::string& steps, const std::string& level = "4") {
      const std::string y = head + name + R"(
inputs:
  recording: {field: recording, each: true, where: {name: rest}}
  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: free}}
  tiles: {selection: ladder, manifold: kernel.to}
  cells: {selection: set, where: {params.joint.recording: "{recording}", params.joint.space_level: )" + level + R"(}}
)" + inputs_extra + R"(
parameters: {band: [8.0, 16.0]}
steps:
  chunks:
    over: recording
    do:
      - {step: analytic, in: {x: recording}, with: {band: $band}, out: z}
      - {step: apply_kernel, in: {x: z, kernel: kernel, tiles: tiles}, with: {level: 4, transform: exact}, out: sources}
      - {step: phase_current, in: {x: sources}, out: current}
      - {step: power, in: {x: sources}, out: pw}
      - {step: magnitude, in: {x: sources}, out: mag}
)" + steps + R"(
outputs:
  cell_pc: {kind: measurement}
)";
      return g::parse(parse_yaml(y, name + ".yaml"), name + ".yaml", y);
    };
    // phase_current over cells of a field that is not the phase_current step's: the power, |z|, and the vector the wrong way round
    for (const char* x : {"pw", "mag", "sources"}) {
      const g::Definition bd = defn("pc_wrong_field", "", std::string("      - {step: measure, in: {x: ") + x + ", over: cells}, with: {measure: phase_current}, out: cell_pc}");
      const std::string why = why_of([&] { g::plan_runs(bd, sub.db(), sub.id(), Json::object()); });
      std::printf("  refused (%s): %s\n", x, why.c_str());
      CHECK(why.find("measure") != std::string::npos && (std::string(x) == "sources" || why.find("phase_current") != std::string::npos));
    }
    {   // power over cells of the vector
      const g::Definition bd = defn("pw_of_current", "", "      - {step: measure, in: {x: current, over: cells}, with: {measure: power}, out: cell_pc}");
      CHECK(why_of([&] { g::plan_runs(bd, sub.db(), sub.id(), Json::object()); }).find("power") != std::string::npos);
    }
    // a joint partition whose spatial level is not the band's (the cells of level 3, the field filtered to level 4)
    {
      const g::Definition bd = defn("pc_level", "", "      - {step: measure, in: {x: current, over: cells}, with: {measure: phase_current}, out: cell_pc}", "3");
      const std::string why = why_of([&] { g::plan_runs(bd, sub.db(), sub.id(), Json::object()); });
      std::printf("  refused (level): %s\n", why.c_str());
      CHECK(why.find("spatial level") != std::string::npos);
    }
    // stored spans of ANOTHER RECORDING (rest_b's bursts on a run of rest), and spans OFF the tile edges
    store_spans("other_recording_bursts", "rest_b", bursts, "global_bursts");
    store_spans("offgrid_bursts", "rest", {{1.0, 3.0, 0}}, "global_bursts");
    const std::string rolls = "      - {step: measure, in: {x: current, over: cells}, with: {measure: phase_current}, out: cell_pc}\n  after:\n    - {step: rollup_events, in: {x: cell_pc, events: bursts}, with: {measure: phase_current}, out: burst_pc}";
    for (const char* name : {"other_recording_bursts", "offgrid_bursts"}) {
      const std::string out_extra = std::string("  bursts: {selection: spans, where: {name: ") + name + "}}";
      std::string y = head + "rollup_" + name + R"(
inputs:
  recording: {field: recording, each: true, where: {name: rest}}
  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: free}}
  tiles: {selection: ladder, manifold: kernel.to}
  cells: {selection: set, where: {params.joint.recording: "{recording}", params.joint.space_level: 4}}
)" + out_extra + R"(
parameters: {band: [8.0, 16.0]}
steps:
  chunks:
    over: recording
    do:
      - {step: analytic, in: {x: recording}, with: {band: $band}, out: z}
      - {step: apply_kernel, in: {x: z, kernel: kernel, tiles: tiles}, with: {level: 4, transform: exact}, out: sources}
      - {step: phase_current, in: {x: sources}, out: current}
)" + rolls + R"(
outputs:
  cell_pc: {kind: measurement}
  burst_pc: {kind: measurement}
)";
      const g::Definition bd = g::parse(parse_yaml(y, std::string(name) + ".yaml"), std::string(name) + ".yaml", y);
      const std::string why = why_of([&] { g::plan_runs(bd, sub.db(), sub.id(), Json::object()); });
      std::printf("  refused (%s): %s\n", name, why.c_str());
      if (std::string(name) == "other_recording_bursts") CHECK(why.find("rest_b") != std::string::npos && why.find("recording") != std::string::npos);
      else CHECK(why.find("does not snap") != std::string::npos && why.find("span 0") != std::string::npos);
    }
  }
  return nxrtest::finish("test_cell_phase_current");
}
