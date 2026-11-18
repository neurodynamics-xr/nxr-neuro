// cellquadrature compute (spec docs/specs/2026-11-13-cortical-events.md, phase 2d) on a synthetic source space (never a real dataset): a two-hemisphere surface, one channel a vertex (the kernel maps each
// channel to its vertex: fixed, or free with three components), a 12 Hz oscillation on every vertex that TRAVELS along z in time tiles 4 … 7 (cos(wt - k z)) and STANDS in tiles 12 … 15 (cos(wt) cos(k z)),
// noise everywhere, a bad span (the cross-spectrum drops the time tiles it touches). Through `crossspectrum compute --estimator analytic`, `joint compute` and `cellquadrature compute`.
// Checks: the ingredients equal the DENSE T = B C B^T's squared norms (the trace formula) to round-off, for both kernels; travelling tiles read a high quadrature fraction and standing tiles ~0; the per-burst
// quantity equals the dense form on the SUM of the burst's tile cross-spectra, and is not the sum of the cells' ingredients; a burst with a dropped tile reads NaN; the fraction is derived at read; a rerun is
// a hit; another job's rows are refused unless --replace; the refusals (tile level mismatch, no joint, not a cross-spectrum, not bursts).
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/task/task.h"
#include "nxr/io/measures.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/cellquadrature.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class F> std::string message_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cellquadrature_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 8192;
  const double sfreq = 250.0, bad0 = 10.0, bad1 = 10.5, omega = 2 * M_PI * 12.0, kz = 4.0;
  const int spaceLevel = 4;
  const std::int64_t W = 256;
  const double tile_s = W / sfreq;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v) {
    const double z = b.P[static_cast<std::size_t>(3 * v + 2)];
    for (int t = 0; t < T; ++t) {
      const std::int64_t k = t / W;
      const double w = omega * t / sfreq;
      double s = 0;
      if (k >= 4 && k <= 7) s = std::cos(w - kz * z);                      // TRAVELLING along z
      if (k >= 12 && k <= 15) s = std::cos(w) * std::cos(kz * z);          // STANDING
      xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>(s + 0.01 * nd(rng));
    }
  }
  std::vector<float> kid(static_cast<std::size_t>(N) * M, 0.0f), kfree(static_cast<std::size_t>(3 * N) * M, 0.0f);
  const double alpha[3] = {1.0, 2.0, -1.5};
  for (int v = 0; v < N; ++v) { kid[static_cast<std::size_t>(v) * M + v] = 1.0f; for (int c = 0; c < 3; ++c) kfree[static_cast<std::size_t>(3 * v + c) * M + v] = static_cast<float>(alpha[c]); }
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, rid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      std::vector<std::int64_t> zeros(static_cast<std::size_t>(M), 0);
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", zeros, {"MEG MAG"}, {}}, {"quality", zeros, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      FieldSpec rec;
      rec.name = "rest"; rec.path = task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 4096}; rec.array.compress = false; rec.cols.set("unit", "A");
      rid = sub.field(std::move(rec));
      nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
      nxr::bst::time_spans(sub, "time/rest_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
      // the bursts: tiles 4-7 (travelling), 12-15 (standing), 6-8 (travelling, then a noise tile), 8-10 (two tiles dropped)
      nxr::bst::TimeSpansOptions bb; bb.line_id = tid; bb.of_field_id = rid; bb.description = "test bursts";
      nxr::bst::time_spans(sub, "time/test_bursts", {{4 * tile_s, 8 * tile_s, 0}, {12 * tile_s, 16 * tile_s, 0}, {6 * tile_s, 9 * tile_s, 0}, {8 * tile_s, 11 * tile_s, 0}}, {"burst"}, bb);
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        sub.operator_(std::move(op));
      };
      kernel("fixed", kid, N, "scalar");
      kernel("free", kfree, 3 * N, "vector3");
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_time_grids(ds, sub);
  nxrtest::make_tiles(ds, sub);
  int timeLevel = -1;
  for (const task::Node& m : task::Subject::of(sub).find_all({.cls = task::Cls::selection, .params_present = {"resolves_hz"}})) {
    if (!m.member_of_id) continue;
    const Json& pj = m.params;
    if (pj["resolves_hz"]["lo"] == 8.0 && pj["resolves_hz"]["hi"] == 16.0 && pj["window_samples"] == W) timeLevel = static_cast<int>(m.member_ordinal.value_or(0));
  }
  CHECK(timeLevel >= 0);
  if (timeLevel < 0) return 1;
  JointComputeOptions jo;
  jo.recording = "rest"; jo.space_level = spaceLevel; jo.time_level = timeLevel;
  const auto j1 = joint_compute(ds, sub, jo);
  const std::string jid = j1.info["id"].get<std::string>();
  const std::int64_t frames = T / W;
  // THE STORED CROSS-SPECTRUM: Σ z zᴴ per level-8 tile of the 8-16 Hz band, on the kernel's channels
  {
    CrossSpectrumOptions o;
    o.recording = "rest"; o.chunks = "all"; o.channels_of = "fixed"; o.bands = {{"alpha", 8, 16}}; o.estimator = "analytic"; o.level = 8;
    const auto cs = crossspectrum_compute(ds, sub, o);
    CHECK_EQ(static_cast<std::int64_t>(cs.windows[0].sum.size()), frames - 2);        // the two tiles the bad span touches are dropped
  }
  const std::string csname = "rest_crossspectrum_alpha_windows";

  // ── THE DENSE REFERENCE: the level's band (exact modes) on the kernel's rows, T = B C Bᵀ formed, its squared norms ──
  const SurfaceInput S = read_surface(sub, std::string("cortex"), true);
  const SurfaceLevel SL = read_surface_level(sub, S, spaceLevel);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S, SL, sbo);
  const std::int64_t tiles = SL.nodes;
  std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
  for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
  const CrossSpectrumWindows win = crossspectrum_windows_read(sub, csname, chans, "stored");
  auto matrix_of = [&](const std::vector<float>& k, int comps) {
    Eigen::MatrixXd A(comps * N, M);
    for (int c = 0; c < comps; ++c) {
      Eigen::MatrixXd Kc(N, M);
      for (int v = 0; v < N; ++v) for (int j = 0; j < M; ++j) Kc(v, j) = k[static_cast<std::size_t>((v * comps + c)) * M + j];
      const Eigen::MatrixXd Mc = sband.apply(Kc);
      for (int v = 0; v < N; ++v) A.row(v * comps + c) = Mc.row(v);
    }
    return A;
  };
  // the dense (‖T‖², ‖Im T‖²) of tile s for a cross-spectrum Cm: T formed (n_s·comps)²
  auto dense = [&](const Eigen::MatrixXd& A, int comps, std::int64_t s, const Eigen::MatrixXcd& Cm) {
    std::vector<Eigen::Index> rows;
    for (int v = 0; v < N; ++v) if (SL.tile[static_cast<std::size_t>(v)] == s) for (int c = 0; c < comps; ++c) rows.push_back(v * comps + c);
    Eigen::MatrixXd B(static_cast<Eigen::Index>(rows.size()), M);
    for (std::size_t i = 0; i < rows.size(); ++i) B.row(static_cast<Eigen::Index>(i)) = A.row(rows[i]);
    const Eigen::MatrixXcd Tm = B * Cm * B.transpose();
    return std::make_pair(Tm.squaredNorm(), Tm.imag().squaredNorm());
  };
  auto read = [&](const std::string& selection, const char* measure) { return measurement_values(sub, selection, rid, std::nullopt, std::nullopt, measure); };
  auto worst = [&](const MeasurementValues& tot, const MeasurementValues& qd, const Eigen::MatrixXd& A, int comps, const std::vector<std::int64_t>& members, const std::vector<Eigen::MatrixXcd>& Cs) {
    double w = 0;
    for (std::size_t i = 0; i < members.size(); ++i)
      for (std::int64_t s = 0; s < tiles; ++s) {
        const auto d = dense(A, comps, s, Cs[i]);
        const double gt = tot.values[static_cast<std::size_t>(s * tot.members + members[i])], gq = qd.values[static_cast<std::size_t>(s * qd.members + members[i])];
        w = std::max({w, std::abs(gt - d.first) / std::max(d.first, 1e-300), std::abs(gq - d.second) / std::max(d.second, 1e-300)});
      }
    return w;
  };
  // the planted travelling and standing tiles' fraction: the quadrature fraction of the spatial tiles that carry the pattern (≥ 20 % of the largest ‖T‖² of the member)
  auto planted = [&](const MeasurementValues& tot, const MeasurementValues& qd, std::int64_t member) {
    double top = 0;
    for (std::int64_t s = 0; s < tiles; ++s) top = std::max(top, tot.values[static_cast<std::size_t>(s * tot.members + member)]);
    double a = 0, wsum = 0;
    int n = 0;
    for (std::int64_t s = 0; s < tiles; ++s) {
      const double t = tot.values[static_cast<std::size_t>(s * tot.members + member)], q = qd.values[static_cast<std::size_t>(s * qd.members + member)];
      if (!(t >= 0.2 * top)) continue;
      a += t * std::sqrt(q / t); wsum += t; ++n;
    }
    return std::make_pair(a / wsum, n);
  };

  // ── 1. THE CELLS, the fixed kernel, exact spatial band ──
  CellQuadratureOptions co;
  co.cross_spectrum = csname; co.space_level = spaceLevel; co.kernel = "fixed"; co.transform = "exact"; co.weights = "none";      // the unweighted behaviour first (the dense forms below have no weights)
  const auto r1 = cellquadrature_compute(ds, sub, co);
  CHECK(!r1.cells_hit && r1.info["cells"]["rows"] == tiles && r1.info["cells"]["members"] == frames);
  const MeasurementValues tot1 = read(jid, "cross_spectrum_norm2"), qd1 = read(jid, "cross_spectrum_quadrature_norm2");
  CHECK(tot1.rows == tiles && tot1.members == frames && tot1.array_backed);
  const Eigen::MatrixXd A1 = matrix_of(kid, 1);
  std::vector<std::int64_t> members;
  std::vector<Eigen::MatrixXcd> Cs;
  for (std::size_t i = 0; i < win.index.size(); ++i) { members.push_back(win.index[i]); Cs.push_back(win.sum[i]); }
  const double w1 = worst(tot1, qd1, A1, 1, members, Cs);
  std::printf("  cells, fixed kernel: trace formula vs the dense T: worst relative error %.2e over %zu time tiles x %lld spatial tiles\n", w1, members.size(), static_cast<long long>(tiles));
  CHECK(w1 < 1e-8);
  // a time tile that is not stored (the bad span's) is NaN in every spatial tile; the others are finite
  int nans = 0;
  for (std::int64_t f = 0; f < frames; ++f) { const bool dropped = f == 9 || f == 10; for (std::int64_t s = 0; s < tiles; ++s) { const double v = tot1.values[static_cast<std::size_t>(s * frames + f)]; if (std::isnan(v) != dropped) ++nans; } }
  CHECK_EQ(nans, 0);
  // travelling vs standing
  const auto ptr = planted(tot1, qd1, 5), pst = planted(tot1, qd1, 13), pqt = planted(tot1, qd1, 20);
  std::printf("  quadrature fraction of the planted tiles: travelling (time tile 5) %.3f over %d tiles, standing (13) %.4f over %d, noise only (20) %.3f over %d\n", ptr.first, ptr.second, pst.first, pst.second, pqt.first, pqt.second);
  CHECK(ptr.second >= 4 && ptr.first > 0.3 && pst.second >= 4 && pst.first < 0.05);
  // the fraction derived at read = √(‖Im T‖² / ‖T‖²)
  const MeasurementValues frac1 = derived_measurement_values(sub, jid, rid, std::nullopt, std::nullopt, "quadrature_fraction");
  double dv = 0;
  for (std::size_t i = 0; i < frac1.values.size(); ++i) { const double want = std::sqrt(qd1.values[i] / tot1.values[i]); if (std::isnan(want)) { if (!std::isnan(frac1.values[i])) dv = 1; } else dv = std::max(dv, std::abs(frac1.values[i] - want)); }
  CHECK(dv < 1e-12);
  CHECK(code_of([&] { measurement_values(sub, jid, rid, std::nullopt, std::nullopt, "quadrature_fraction"); }) == std::string(code::invalid_argument));   // derived: never stored
  CHECK(r1.info["warnings"].empty() || r1.info["warnings"].is_array());

  // ── 2. A RERUN IS A HIT; another job's rows are refused unless --replace ──
  CHECK(cellquadrature_compute(ds, sub, co).cells_hit);
  CellQuadratureOptions cf = co; cf.kernel = "free";
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, cf); }) == std::string(code::already_exists));
  cf.replace = true;
  const auto rf = cellquadrature_compute(ds, sub, cf);
  CHECK(!rf.cells_hit && rf.info["cells_replaced"] == true);
  const MeasurementValues totf = read(jid, "cross_spectrum_norm2"), qdf = read(jid, "cross_spectrum_quadrature_norm2");
  const Eigen::MatrixXd Af = matrix_of(kfree, 3);
  const double wf = worst(totf, qdf, Af, 3, members, Cs);
  std::printf("  cells, free kernel (3 components): trace formula vs the dense T: worst relative error %.2e\n", wf);
  CHECK(wf < 1e-8);
  CHECK(cellquadrature_compute(ds, sub, cf).cells_hit);
  // an orientation prior is no part of it: the free kernel's tiles read their own quadrature, the standing ones still ~0
  const auto fpt = planted(totf, qdf, 5), fps = planted(totf, qdf, 13);
  CHECK(fpt.first > 0.3 && fps.first < 0.05);

  // ── 3. THE BURSTS: C summed over each burst's tiles first, then the ingredients ──
  CellQuadratureOptions cb = cf; cb.replace = false; cb.bursts = "test_bursts";
  const auto rb = cellquadrature_compute(ds, sub, cb);
  CHECK(rb.cells_hit && !rb.bursts_hit);
  const auto bsel = task::Subject::of(sub).find({.cls = task::Cls::selection, .name = "test_bursts"});
  CHECK(bsel.has_value());
  const std::string bid = bsel->id;
  const MeasurementValues totb = read(bid, "cross_spectrum_norm2"), qdb = read(bid, "cross_spectrum_quadrature_norm2");
  CHECK(totb.rows == tiles && totb.members == 4);
  auto sum_tiles = [&](std::int64_t a, std::int64_t z) {
    Eigen::MatrixXcd Cb = Eigen::MatrixXcd::Zero(M, M);
    for (std::size_t i = 0; i < win.index.size(); ++i) if (win.index[i] >= a && win.index[i] <= z) Cb += win.sum[i];
    return Cb;
  };
  const std::vector<std::pair<std::int64_t, std::int64_t>> burst_tiles = {{4, 7}, {12, 15}, {6, 8}};
  double wb = 0;
  for (std::size_t k = 0; k < burst_tiles.size(); ++k) {
    const Eigen::MatrixXcd Cb = sum_tiles(burst_tiles[k].first, burst_tiles[k].second);
    for (std::int64_t s = 0; s < tiles; ++s) {
      const auto d = dense(Af, 3, s, Cb);
      wb = std::max({wb, std::abs(totb.values[static_cast<std::size_t>(s * 4 + static_cast<std::int64_t>(k))] - d.first) / d.first, std::abs(qdb.values[static_cast<std::size_t>(s * 4 + static_cast<std::int64_t>(k))] - d.second) / d.second});
    }
  }
  std::printf("  bursts: ingredients of B (sum of the burst's tile C) B^T vs the dense form: worst relative error %.2e\n", wb);
  CHECK(wb < 1e-8);
  // the burst of the dropped tiles reads NaN in every spatial tile; the others are finite
  int nb = 0;
  for (std::int64_t s = 0; s < tiles; ++s) { if (!std::isnan(totb.values[static_cast<std::size_t>(s * 4 + 3)])) ++nb; for (int k = 0; k < 3; ++k) if (std::isnan(totb.values[static_cast<std::size_t>(s * 4 + k)])) ++nb; }
  CHECK_EQ(nb, 0);
  CHECK(rb.info["bursts"]["events_read"][3]["complete"] == false && rb.info["bursts"]["events_read"][0]["complete"] == true && rb.info["bursts"]["events_read"][0]["tiles"] == 4);
  // the per-burst quantity is NOT the cells' ingredients summed over the burst's time tiles: the tiles' cross terms are in the first only
  double top_s = 0, sum_cells = 0, burst = 0;
  for (std::int64_t s = 0; s < tiles; ++s) {
    double sc = 0;
    for (int f = 4; f <= 7; ++f) sc += totf.values[static_cast<std::size_t>(s * frames + f)];
    if (sc > top_s) { top_s = sc; sum_cells = sc; burst = totb.values[static_cast<std::size_t>(s * 4)]; }
  }
  std::printf("  the travelling burst's largest tile: ||T of the summed C||^2 %.4g against the cells' ingredients summed over its tiles %.4g (ratio %.2f)\n", burst, sum_cells, burst / sum_cells);
  CHECK(burst > 1.5 * sum_cells);
  const auto bt = planted(totb, qdb, 0), bs = planted(totb, qdb, 1), bm = planted(totb, qdb, 2);
  std::printf("  burst quadrature fraction: travelling %.3f, standing %.4f, travelling then nothing %.3f\n", bt.first, bs.first, bm.first);
  CHECK(bt.first > 0.3 && bs.first < 0.05);
  CHECK(cellquadrature_compute(ds, sub, cb).bursts_hit);


  // ── 3b. AREA WEIGHTS (the default) and the TILE PAIRS ──
  // `weights: none` is the old behaviour: its values (the free-kernel cells above) are what a rerun with the option gives again, after the weighted job has replaced them
  const std::vector<double> before_tot = totf.values, before_quad = qdf.values;
  nxr::select::Mesh whole; whole.n = N; whole.positions = S.vertices; whole.faces = S.faces;
  const std::vector<double> va = nxr::select::vertexAreas(whole);
  CellQuadratureOptions cw = cf; cw.replace = false; cw.weights = "area"; cw.bursts = "test_bursts";
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, cw); }) == std::string(code::already_exists));              // the unweighted rows are another job's
  cw.replace = true;
  const auto rw = cellquadrature_compute(ds, sub, cw);
  CHECK(!rw.cells_hit && !rw.bursts_hit && !rw.cell_pairs_hit && !rw.burst_pairs_hit && rw.info["weights"] == "area");
  const MeasurementValues totw = read(jid, "cross_spectrum_norm2"), qdw = read(jid, "cross_spectrum_quadrature_norm2");
  // the dense weighted form for the free kernel: rows of a vertex's three components weigh its area
  auto dense_pair = [&](const Eigen::MatrixXd& A, int comps, std::int64_t s0, std::int64_t s1, const Eigen::MatrixXcd& Cm, bool weighted) {
    std::vector<Eigen::Index> ra, rb;
    for (int v = 0; v < N; ++v) for (int c = 0; c < comps; ++c) { if (SL.tile[static_cast<std::size_t>(v)] == s0) ra.push_back(v * comps + c); if (SL.tile[static_cast<std::size_t>(v)] == s1) rb.push_back(v * comps + c); }
    Eigen::MatrixXd Ba(static_cast<Eigen::Index>(ra.size()), M), Bb(static_cast<Eigen::Index>(rb.size()), M);
    for (std::size_t i = 0; i < ra.size(); ++i) Ba.row(static_cast<Eigen::Index>(i)) = A.row(ra[i]);
    for (std::size_t i = 0; i < rb.size(); ++i) Bb.row(static_cast<Eigen::Index>(i)) = A.row(rb[i]);
    const Eigen::MatrixXcd Tm = Ba * Cm * Bb.transpose();
    double tot = 0, qd = 0;
    for (std::size_t i = 0; i < ra.size(); ++i) for (std::size_t j = 0; j < rb.size(); ++j) {
      const double w = weighted ? va[static_cast<std::size_t>(ra[i] / comps)] * va[static_cast<std::size_t>(rb[j] / comps)] : 1.0;
      tot += w * std::norm(Tm(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)));
      qd += w * Tm(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)).imag() * Tm(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)).imag();
    }
    return std::make_pair(tot, qd);
  };
  double ww = 0, wdiff = 0;
  for (std::size_t i = 0; i < members.size(); ++i)
    for (std::int64_t s0 = 0; s0 < tiles; ++s0) {
      const auto d = dense_pair(Af, 3, s0, s0, Cs[i], true);
      const std::size_t at = static_cast<std::size_t>(s0 * frames + members[i]);
      ww = std::max({ww, std::abs(totw.values[at] - d.first) / d.first, std::abs(qdw.values[at] - d.second) / d.second});
      wdiff = std::max(wdiff, std::abs(totw.values[at] / totf.values[at] - 1));
    }
  std::printf("  cells, area weights, free kernel: trace formula vs the dense weighted T: worst relative error %.2e; the weights change the value by up to %.0f %%\n", ww, 100 * wdiff);
  CHECK(ww < 1e-8 && wdiff > 1e-3);
  CHECK(cellquadrature_compute(ds, sub, cw).cells_hit);
  // THE PAIRS: the rows are the level's touching tiles (tileAdjacency), listed beside
  const std::int64_t P = static_cast<std::int64_t>(SL.pairs.size());
  CHECK(P > 8 && rw.info["cell_pairs"]["rows"] == P && rw.info["cell_pairs"]["members"] == frames);
  {
    ArrayReader pr = ArrayReader::open(sub.store(), rw.info["cell_pairs"]["pair_array"].get<std::string>());
    const std::vector<std::uint8_t> raw = pr.read();
    CHECK_EQ(static_cast<std::int64_t>(raw.size() / sizeof(std::int32_t)), 2 * P);
    const std::int32_t* pl = reinterpret_cast<const std::int32_t*>(raw.data());
    bool same = true;
    for (std::int64_t k = 0; k < P; ++k) same = same && pl[2 * k] == SL.pairs[static_cast<std::size_t>(k)].first && pl[2 * k + 1] == SL.pairs[static_cast<std::size_t>(k)].second;
    CHECK(same);
  }
  const MeasurementValues ptot = read(jid, "tile_pair_cross_spectrum_norm2"), pqd = read(jid, "tile_pair_cross_spectrum_quadrature_norm2");
  CHECK(ptot.rows == P && ptot.members == frames && ptot.array_backed);
  double wp = 0;
  for (std::size_t i = 0; i < members.size(); ++i)
    for (std::int64_t k = 0; k < P; ++k) {
      const auto d = dense_pair(Af, 3, SL.pairs[static_cast<std::size_t>(k)].first, SL.pairs[static_cast<std::size_t>(k)].second, Cs[i], true);
      const std::size_t at = static_cast<std::size_t>(k * frames + members[i]);
      wp = std::max({wp, std::abs(ptot.values[at] - d.first) / d.first, std::abs(pqd.values[at] - d.second) / std::max(d.second, 1e-300)});
    }
  std::printf("  tile pairs (%lld touching pairs), area weights, free kernel: trace formula vs the dense cross block: worst relative error %.2e\n", static_cast<long long>(P), wp);
  CHECK(wp < 1e-8);
  // a time tile that is not stored is NaN in every pair
  int pnan = 0;
  for (std::int64_t f = 0; f < frames; ++f) for (std::int64_t k = 0; k < P; ++k) if (std::isnan(ptot.values[static_cast<std::size_t>(k * frames + f)]) != (f == 9 || f == 10)) ++pnan;
  CHECK_EQ(pnan, 0);
  // the pair fraction is derived at read, and a pair of a travelling wave reads high where the tiles' own blocks read low
  const MeasurementValues pfrac = derived_measurement_values(sub, jid, rid, std::nullopt, std::nullopt, "tile_pair_quadrature_fraction");
  double dvp = 0;
  for (std::size_t i = 0; i < pfrac.values.size(); ++i) { const double want = std::sqrt(pqd.values[i] / ptot.values[i]); if (std::isnan(want)) { if (!std::isnan(pfrac.values[i])) dvp = 1; } else dvp = std::max(dvp, std::abs(pfrac.values[i] - want)); }
  CHECK(dvp < 1e-12);
  auto pair_power_fraction = [&](const MeasurementValues& tt, const MeasurementValues& qq, std::int64_t member) {   // the power-weighted mean of √(q/t) over the rows
    double a = 0, wsum = 0;
    for (std::int64_t k = 0; k < tt.rows; ++k) { const double t = tt.values[static_cast<std::size_t>(k * tt.members + member)], q = qq.values[static_cast<std::size_t>(k * qq.members + member)]; a += t * std::sqrt(q / t); wsum += t; }
    return a / wsum;
  };
  const double within_trav = pair_power_fraction(totw, qdw, 5), pair_trav = pair_power_fraction(ptot, pqd, 5), within_stand = pair_power_fraction(totw, qdw, 13), pair_stand = pair_power_fraction(ptot, pqd, 13);
  std::printf("  quadrature fraction (power-weighted over the level): travelling tile 5: within the tiles %.3f, between touching tiles %.3f; standing tile 13: within %.4f, between %.4f\n", within_trav, pair_trav, within_stand, pair_stand);
  CHECK(pair_trav > within_trav + 0.1);        // a wave crossing a tile boundary is in the cross blocks
  CHECK(pair_stand < 0.05 && within_stand < 0.05);   // a pattern standing across two tiles has none
  // the bursts: C summed first, then the pair
  const MeasurementValues pbt = read(bid, "tile_pair_cross_spectrum_norm2"), pbq = read(bid, "tile_pair_cross_spectrum_quadrature_norm2");
  CHECK(pbt.rows == P && pbt.members == 4);
  double wpb = 0, sum_pairs_cells = 0, pair_burst = 0;
  for (std::size_t k = 0; k < burst_tiles.size(); ++k) {
    const Eigen::MatrixXcd Cb = sum_tiles(burst_tiles[k].first, burst_tiles[k].second);
    for (std::int64_t q = 0; q < P; ++q) {
      const auto d = dense_pair(Af, 3, SL.pairs[static_cast<std::size_t>(q)].first, SL.pairs[static_cast<std::size_t>(q)].second, Cb, true);
      wpb = std::max({wpb, std::abs(pbt.values[static_cast<std::size_t>(q * 4 + static_cast<std::int64_t>(k))] - d.first) / d.first, std::abs(pbq.values[static_cast<std::size_t>(q * 4 + static_cast<std::int64_t>(k))] - d.second) / std::max(d.second, 1e-300)});
    }
  }
  std::printf("  bursts, tile pairs: ingredients of B_s (sum of the burst's tile C) B_s'^T vs the dense form: worst relative error %.2e\n", wpb);
  CHECK(wpb < 1e-8);
  for (std::int64_t q = 0; q < P; ++q) { double sc = 0; for (int f = 4; f <= 7; ++f) sc += ptot.values[static_cast<std::size_t>(q * frames + f)]; sum_pairs_cells += sc; pair_burst += pbt.values[static_cast<std::size_t>(q * 4)]; }
  CHECK(pair_burst > 1.5 * sum_pairs_cells);   // not the cells' ingredients summed over time
  int pbnan = 0;
  for (std::int64_t q = 0; q < P; ++q) if (!std::isnan(pbt.values[static_cast<std::size_t>(q * 4 + 3)]) || std::isnan(pbt.values[static_cast<std::size_t>(q * 4)])) ++pbnan;
  CHECK_EQ(pbnan, 0);
  CHECK(cellquadrature_compute(ds, sub, cw).bursts_hit);
  CellQuadratureOptions cr = cw; const auto rr = cellquadrature_compute(ds, sub, cr);
  CHECK(rr.cells_hit && rr.bursts_hit && rr.cell_pairs_hit && rr.burst_pairs_hit);                                  // a rerun is a hit, pairs too
  // the pairs off: the same cells, no pair job needed
  CellQuadratureOptions cn = cw; cn.tile_pairs = false;
  CHECK(cellquadrature_compute(ds, sub, cn).cells_hit);
  // `weights: none` again: the old rows, to the digit
  CellQuadratureOptions c0 = cf; c0.replace = true; c0.weights = "none";
  const auto r0 = cellquadrature_compute(ds, sub, c0);
  CHECK(!r0.cells_hit && r0.info["weights"] == "none");
  const MeasurementValues tot0 = read(jid, "cross_spectrum_norm2"), qd0 = read(jid, "cross_spectrum_quadrature_norm2");
  auto same_values = [](const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) if (!(a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i])))) return false;
    return true;
  };
  CHECK(same_values(tot0.values, before_tot) && same_values(qd0.values, before_quad));
  CellQuadratureOptions bad = cw; bad.weights = "mass";
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, bad); }) == std::string(code::invalid_argument));

  // ── 4. THE REFUSALS ──
  {
    CrossSpectrumOptions o;
    o.recording = "rest"; o.chunks = "all"; o.channels_of = "fixed"; o.bands = {{"alpha", 8, 16}}; o.estimator = "analytic"; o.level = 9; o.name = "rest_l9";
    crossspectrum_compute(ds, sub, o);
  }
  CellQuadratureOptions mm = cf; mm.cross_spectrum = "rest_l9_alpha_windows"; mm.replace = false;
  const std::string msg = message_of([&] { cellquadrature_compute(ds, sub, mm); });
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, mm); }) == std::string(code::invalid_argument) && msg.find("level 9") != std::string::npos && msg.find("time level") != std::string::npos);   // the tile level of the cross-spectrum is not the pair's
  CellQuadratureOptions nj = cf; nj.space_level = 3; nj.replace = false;
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, nj); }) == std::string(code::invalid_argument));                 // no joint partition at that spatial level
  CellQuadratureOptions nc = cf; nc.cross_spectrum = "nonexistent"; nc.replace = false;
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, nc); }) == std::string(code::no_such_row));
  CellQuadratureOptions nb2 = cf; nb2.bursts = "kern_channels"; nb2.replace = false;
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, nb2); }) == std::string(code::invalid_argument));                // not events
  CellQuadratureOptions e = cf; e.space_level = -1;
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, e); }) == std::string(code::invalid_argument));
  CellQuadratureOptions e2 = cf; e2.transform = "bogus";
  CHECK(code_of([&] { cellquadrature_compute(ds, sub, e2); }) == std::string(code::invalid_argument));
  return 0;
}
