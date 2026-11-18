// the `cellquadrature` definition run through the runner on a small synthetic subject (never a real dataset): a two-hemisphere surface, 128 channels each on one vertex (the kernel maps each channel to its vertex: fixed, or
// free with three components), a 12 Hz oscillation on every vertex that TRAVELS along z in time tiles 6 to 13 (cos(wt - k z)) and STANDS in tiles 18 to 25 (cos(wt) cos(k z)), noise everywhere, a bad span in tiles 14 and 15.
// The band's analytic cross-spectrum is `run measure --set analytic_cross_spectrum_tiles`; the cells are `run select_product` at level 4 x the band's tiles; one band, one chunk (32 tiles of 128 samples).
// Checks: the ingredients equal the dense T = B C B^T's squared norms (the trace formula) to round-off, for the fixed kernel and for the free one with area weights; the tile pairs and the bursts likewise (a burst's
// quantity is on the SUM of its tiles' cross-spectra, not the sum of the cells' ingredients); travelling tiles read a high quadrature fraction and standing tiles ~0; a tile without a cross-spectrum reads NaN; the
// fraction is derived at read; a rerun is a hit; the refusals (a grid that holds no cross-spectrum, bursts that are not events, a bad weight).
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/task/task.h"
#include "nxr/io/measures.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
template <class F> std::string why_of(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cellquadrature_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = 128, T = 4096;                       // 128 channels, each on one vertex (every 2.5th); the kernel maps a channel to its vertex, the other vertices read none
  const double sfreq = 250.0, bad0 = 7.3, bad1 = 7.8, omega = 2 * M_PI * 12.0, kz = 4.0;
  const std::int64_t W = 128, frames = T / W;
  const double tile_s = W / sfreq;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  std::vector<int> vertex_of(static_cast<std::size_t>(M));
  for (int c = 0; c < M; ++c) vertex_of[static_cast<std::size_t>(c)] = c * N / M;
  for (int v = 0; v < M; ++v) {
    const double z = b.P[static_cast<std::size_t>(3 * vertex_of[static_cast<std::size_t>(v)] + 2)];
    for (int t = 0; t < T; ++t) {
      const std::int64_t k = t / W;
      const double w = omega * t / sfreq;
      double s = 0;
      if (k >= 6 && k <= 13) s = std::cos(w - kz * z);                      // TRAVELLING along z
      if (k >= 18 && k <= 25) s = std::cos(w) * std::cos(kz * z);           // STANDING
      xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>(s + 0.01 * nd(rng));
    }
  }
  std::vector<float> kid(static_cast<std::size_t>(N) * M, 0.0f), kfree(static_cast<std::size_t>(3 * N) * M, 0.0f);
  const double alpha[3] = {1.0, 2.0, -1.5};
  for (int j = 0; j < M; ++j) { const int v = vertex_of[static_cast<std::size_t>(j)]; kid[static_cast<std::size_t>(v) * M + j] = 1.0f; for (int c = 0; c < 3; ++c) kfree[static_cast<std::size_t>(3 * v + c) * M + j] = static_cast<float>(alpha[c]); }
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
      rec.name = "rest"; rec.path = tk::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 4096}; rec.array.compress = false; rec.cols.set("unit", "A");
      rid = sub.field(std::move(rec));
      nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
      nxr::bst::time_spans(sub, "time/rest_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
      // the bursts: tiles 6-13 (travelling), 18-25 (standing), 12-17 (dropped tiles in it)
      nxr::bst::TimeSpansOptions bb; bb.line_id = tid; bb.of_field_id = rid; bb.description = "test bursts";
      nxr::bst::time_spans(sub, "time/test_bursts", {{6 * tile_s, 14 * tile_s, 0}, {18 * tile_s, 26 * tile_s, 0}, {12 * tile_s, 18 * tile_s, 0}}, {"burst"}, bb);
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
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_tiles(ds, sub);
  const std::string grid = "rest time grid band 8-16 Hz", cells = "rest cortex product";
  nxrtest::make_time_grids(ds, sub, "band", Json{{"band", "8-16 Hz"}, {"recording", "rest"}});
  nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"surface", "cortex"}, {"time_grid", grid}, {"couple", Json::array({Json{{"factor", 0}, {"scale", "parts 16"}}})}});
  { g::RunOptions po; po.chunks = "all"; po.preset = "analytic_cross_spectrum_tiles";
    const auto rm = g::run(ds, sub, g::load("measure"), Json{{"recording", "rest"}, {"band", "8-16 Hz"}}, po);
    CHECK(rm.size() == 1 && rm[0]["hit"] == false); }
  const std::string jid = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = cells, .any_status = true})->id;
  { const auto pr = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = cells, .any_status = true}); std::printf("PRODUCT %s\n", pr->params["product"].dump().c_str()); const auto gr = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = grid, .any_status = true}); std::printf("GRID %s\n", gr->params.dump().substr(0,700).c_str()); }
  const std::string bid = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = "test_bursts", .any_status = true})->id;

  // ── THE DENSE REFERENCE: the level's band (exact modes) on the kernel's rows, T = B C B^T formed, its squared norms ──
  const SurfaceInput S = read_surface(sub, std::string("cortex"), true);
  const int spaceLevel = 4;
  const SurfaceLevel SL = read_surface_level(sub, S, spaceLevel);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S, SL, sbo);
  const std::int64_t tiles = SL.nodes;
  std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
  for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
  const CrossSpectrumWindows win = crossspectrum_windows_read(sub, grid, chans);
  std::set<std::int64_t> stored(win.index.begin(), win.index.end());
  { std::printf("  windows level %d:", win.level); for (auto k : stored) std::printf(" %lld", (long long)k); std::printf("\n"); }
  CHECK(win.level == 7 && stored.count(14) == 0 && stored.count(15) == 0);                                 // the tile the bad span touches is no window
  for (std::int64_t k = 6; k <= 13; ++k) CHECK(stored.count(k) == 1 && stored.count(k + 12) == 1);
  std::vector<std::int64_t> members;                                             // the dense reference is formed on five of the stored tiles (travelling, standing, noise): it is the costly part
  std::vector<Eigen::MatrixXcd> Cs;
  for (std::size_t i = 0; i < win.index.size(); ++i) if (win.index[i] == 2 || win.index[i] == 9 || win.index[i] == 13 || win.index[i] == 21 || win.index[i] == 28) { members.push_back(win.index[i]); Cs.push_back(win.sum[i]); }
  CHECK(members.size() == 5);
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
  nxr::select::Mesh whole; whole.n = N; whole.positions = S.vertices; whole.faces = S.faces;
  const std::vector<double> va = nxr::select::vertexAreas(whole);
  // the dense (sum |T_ij|^2, sum (Im T_ij)^2) of the block of tiles s0 x s1 (the vertex areas weigh the rows when `weighted`)
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
  auto read = [&](const std::string& selection, const char* measure) { return measurement_values(sub, selection, rid, std::nullopt, std::nullopt, measure); };
  auto worst = [&](const MeasurementValues& tot, const MeasurementValues& qd, const Eigen::MatrixXd& A, int comps, bool weighted) {
    double w = 0;
    for (std::size_t i = 0; i < members.size(); ++i)
      for (std::int64_t s = 0; s < tiles; ++s) {
        const auto d = dense_pair(A, comps, s, s, Cs[i], weighted);
        const double gt = tot.values[static_cast<std::size_t>(s * tot.members + members[i])], gq = qd.values[static_cast<std::size_t>(s * qd.members + members[i])];
        w = std::max({w, std::abs(gt - d.first) / std::max(d.first, 1e-300), std::abs(gq - d.second) / std::max(d.second, 1e-300)});
      }
    return w;
  };
  // the planted tiles' quadrature fraction: the power-weighted mean of sqrt(q / t) over the spatial tiles that carry the pattern (at least 20 % of the member's largest total)
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
  auto run = [&](const std::string& kernel, const std::string& weights, bool pairs, bool bursts, bool replace) {
    Json set = Json{{"cross_spectrum", grid}, {"recording", "rest"}, {"kernel", kernel}, {"joint", cells}, {"spatial_scale", "parts 16"}, {"weights", weights}, {"tile_pairs", pairs}, {"transform", "exact"}};
    if (bursts) set["bursts"] = "test_bursts";
    return nxrtest::run_definition(ds, sub, "cellquadrature", set, replace);
  };

  // ── 1. THE CELLS, the fixed kernel, unweighted, no pairs ──
  const auto r1 = run("fixed", "none", false, false, false);
  CHECK(r1.size() == 1 && r1[0]["hit"] == false);
  const MeasurementValues tot1 = read(jid, "cross_spectrum_norm2"), qd1 = read(jid, "cross_spectrum_quadrature_norm2");
  CHECK(tot1.rows == tiles && tot1.members == frames && tot1.array_backed);
  const double w1 = worst(tot1, qd1, matrix_of(kid, 1), 1, false);
  std::printf("  cells, fixed kernel: trace formula vs the dense T: worst relative error %.2e over %zu time tiles x %lld spatial tiles\n", w1, members.size(), static_cast<long long>(tiles));
  CHECK(w1 < 1e-8);
  int nans = 0;                                                                   // a tile that is not a window is NaN in every spatial tile; the others are finite
  for (std::int64_t f = 0; f < frames; ++f) for (std::int64_t s = 0; s < tiles; ++s) if (std::isnan(tot1.values[static_cast<std::size_t>(s * frames + f)]) != (stored.count(f) == 0)) ++nans;
  CHECK_EQ(nans, 0);
  const auto ptr = planted(tot1, qd1, 9), pst = planted(tot1, qd1, 21);
  std::printf("  quadrature fraction of the planted tiles: travelling (time tile 9) %.3f over %d tiles, standing (21) %.4f over %d\n", ptr.first, ptr.second, pst.first, pst.second);
  CHECK(ptr.second >= 4 && ptr.first > 0.3 && pst.second >= 4 && pst.first < 0.05);
  const MeasurementValues frac1 = derived_measurement_values(sub, jid, rid, std::nullopt, std::nullopt, "quadrature_fraction");   // derived at read: sqrt(Im T^2 / T^2), never stored
  double dv = 0;
  for (std::size_t i = 0; i < frac1.values.size(); ++i) { const double want = std::sqrt(qd1.values[i] / tot1.values[i]); if (std::isnan(want)) { if (!std::isnan(frac1.values[i])) dv = 1; } else dv = std::max(dv, std::abs(frac1.values[i] - want)); }
  CHECK(dv < 1e-12);
  CHECK(!why_of([&] { measurement_values(sub, jid, rid, std::nullopt, std::nullopt, "quadrature_fraction"); }).empty());
  CHECK(run("fixed", "none", false, false, false)[0]["hit"] == true);              // a rerun is a hit

  // ── 2. THE FREE KERNEL with area weights, the tile pairs and the bursts (replacing the unweighted rows) ──
  const auto r2 = run("free", "area", true, true, true);
  CHECK(r2.size() == 1 && r2[0]["hit"] == false);
  const Eigen::MatrixXd Af = matrix_of(kfree, 3);
  const MeasurementValues totw = read(jid, "cross_spectrum_norm2"), qdw = read(jid, "cross_spectrum_quadrature_norm2");
  const double ww = worst(totw, qdw, Af, 3, true);
  std::printf("  cells, free kernel, area weights: trace formula vs the dense weighted T: worst relative error %.2e\n", ww);
  CHECK(ww < 1e-8);
  const auto fpt = planted(totw, qdw, 9), fps = planted(totw, qdw, 21);            // no orientation prior: the free kernel's tiles read their own quadrature
  CHECK(fpt.first > 0.3 && fps.first < 0.05);
  // the pairs: the rows are the level's touching tiles (tileAdjacency)
  const std::int64_t P = static_cast<std::int64_t>(SL.pairs.size());
  const MeasurementValues ptot = read(jid, "tile_pair_cross_spectrum_norm2"), pqd = read(jid, "tile_pair_cross_spectrum_quadrature_norm2");
  CHECK(P > 8 && ptot.rows == P && ptot.members == frames);
  double wp = 0;
  for (std::size_t i = 0; i < members.size(); ++i)
    for (std::int64_t k = 0; k < P; ++k) {
      const auto d = dense_pair(Af, 3, SL.pairs[static_cast<std::size_t>(k)].first, SL.pairs[static_cast<std::size_t>(k)].second, Cs[i], true);
      const std::size_t at = static_cast<std::size_t>(k * frames + members[i]);
      wp = std::max({wp, std::abs(ptot.values[at] - d.first) / d.first, std::abs(pqd.values[at] - d.second) / d.first});   // the quadrature part against the block's total: where it is nearly nil its own relative error is round-off
    }
  std::printf("  tile pairs (%lld touching pairs): trace formula vs the dense cross block: worst relative error %.2e\n", static_cast<long long>(P), wp);
  CHECK(wp < 1e-7);
  auto pair_power_fraction = [&](const MeasurementValues& tt, const MeasurementValues& qq, std::int64_t member) {
    double a = 0, wsum = 0;
    for (std::int64_t k = 0; k < tt.rows; ++k) { const double t = tt.values[static_cast<std::size_t>(k * tt.members + member)], q = qq.values[static_cast<std::size_t>(k * qq.members + member)]; a += t * std::sqrt(q / t); wsum += t; }
    return a / wsum;
  };
  const double within_trav = pair_power_fraction(totw, qdw, 9), pair_trav = pair_power_fraction(ptot, pqd, 9), pair_stand = pair_power_fraction(ptot, pqd, 21);
  std::printf("  travelling tile 9: quadrature within the tiles %.3f, between touching tiles %.3f; standing tile 21 between %.4f\n", within_trav, pair_trav, pair_stand);
  CHECK(pair_trav > within_trav + 0.1 && pair_stand < 0.05);                       // a wave crossing a tile boundary is in the cross blocks; a standing pattern has none
  // the bursts: C summed over the burst's tiles first, then the ingredients; a burst holding a dropped tile reads NaN
  const MeasurementValues totb = read(bid, "cross_spectrum_norm2"), qdb = read(bid, "cross_spectrum_quadrature_norm2");
  CHECK(totb.rows == tiles && totb.members == 3);
  auto sum_tiles = [&](std::int64_t a, std::int64_t z) {
    Eigen::MatrixXcd Cb = Eigen::MatrixXcd::Zero(M, M);
    for (std::size_t i = 0; i < win.index.size(); ++i) if (win.index[i] >= a && win.index[i] <= z) Cb += win.sum[i];
    return Cb;
  };
  const std::vector<std::pair<std::int64_t, std::int64_t>> burst_tiles = {{6, 13}, {18, 25}};
  double wb = 0;
  for (std::size_t k = 0; k < burst_tiles.size(); ++k) {
    const Eigen::MatrixXcd Cb = sum_tiles(burst_tiles[k].first, burst_tiles[k].second);
    for (std::int64_t s = 0; s < tiles; ++s) {
      const auto d = dense_pair(Af, 3, s, s, Cb, true);
      wb = std::max({wb, std::abs(totb.values[static_cast<std::size_t>(s * 3 + static_cast<std::int64_t>(k))] - d.first) / d.first, std::abs(qdb.values[static_cast<std::size_t>(s * 3 + static_cast<std::int64_t>(k))] - d.second) / d.second});
    }
  }
  std::printf("  bursts: ingredients of B (sum of the burst's tile C) B^T vs the dense form: worst relative error %.2e\n", wb);
  CHECK(wb < 1e-8);
  int nb = 0;
  for (std::int64_t s = 0; s < tiles; ++s) { if (!std::isnan(totb.values[static_cast<std::size_t>(s * 3 + 2)])) ++nb; for (int k = 0; k < 2; ++k) if (std::isnan(totb.values[static_cast<std::size_t>(s * 3 + k)])) ++nb; }
  CHECK_EQ(nb, 0);
  double top_s = 0, sum_cells = 0, burst = 0;                                      // the burst's quantity is NOT the cells' ingredients summed over its tiles: the tiles' cross terms are in the first only
  for (std::int64_t s = 0; s < tiles; ++s) {
    double sc = 0;
    for (int f = 6; f <= 13; ++f) sc += totw.values[static_cast<std::size_t>(s * frames + f)];
    if (sc > top_s) { top_s = sc; sum_cells = sc; burst = totb.values[static_cast<std::size_t>(s * 3)]; }
  }
  CHECK(burst > 1.5 * sum_cells);
  const auto bt = planted(totb, qdb, 0), bs = planted(totb, qdb, 1);
  std::printf("  burst quadrature fraction: travelling %.3f, standing %.4f\n", bt.first, bs.first);
  CHECK(bt.first > 0.3 && bs.first < 0.05);
  CHECK(run("free", "area", true, true, false)[0]["hit"] == true);

  // ── 3. THE REFUSALS ──
  CHECK(!why_of([&] { run("free", "mass", true, false, true); }).empty());                                                       // a weighting that is neither area nor none
  CHECK(!why_of([&] { Json set = Json{{"cross_spectrum", "kern_channels"}, {"recording", "rest"}, {"kernel", "free"}, {"joint", cells}, {"spatial_scale", "parts 16"}}; nxrtest::run_definition(ds, sub, "cellquadrature", set, true); }).empty());   // a selection that is no grid of cross-spectrum tiles
  CHECK(!why_of([&] { Json set = Json{{"cross_spectrum", grid}, {"recording", "rest"}, {"kernel", "free"}, {"joint", cells}, {"spatial_scale", "parts 16"}, {"bursts", "kern_channels"}}; nxrtest::run_definition(ds, sub, "cellquadrature", set, true); }).empty());   // not events
  return 0;
}
