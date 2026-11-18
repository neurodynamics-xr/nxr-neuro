// run cortical_cells / global_bursts with `envelope: spatiotemporal` (spec docs/specs/2026-11-13-cortical-events.md, "Spatial analytic signal (monogenic)") on a synthetic source space (never a real dataset):
// the phase 2b HAND-OVER, the level's own pattern of tile A fading into a neighbour tile B's inside one episode (left hemisphere). The patterns are signed, so |z| of their sum dips where they have opposite
// sign; the monogenic amplitude sqrt(|z|^2 + |R z|^2) does not. Checks: the cell amplitudes of each envelope against an independent direct computation (the band's analytic signal, the level's band and
// its Riesz part by the dense exact modes, the norm, the area-weighted tile mean); `temporal` is the default and reproduces the old values (it is the same run as an unnamed envelope: a hit); the dip of the
// hand-over in the strongest cell's trace under `temporal` is removed under `spatiotemporal`; the same samples in another chunk width give the same cells; a rerun is a hit; the other envelope's rows are
// refused unless --replace and the recipe names the envelope; a bad envelope and a free kernel are checked.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <iostream>
#include <random>
#include "brain_helpers.h"
#include "check.h"
#include "portable_random.h"
#include "store_at.h"
#include "nxr/io.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/io/yaml.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/signal/filters_fir.h"
#include "nxr/signal/graphfilterbank.h"
#include "make_grids.h"
#include "run_definition.h"

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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_burst_extent_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 12288;
  const double sfreq = 250.0;
  const int L = 4;
  std::int64_t W = 0;                                     // the band's time tile (samples), read from its grid
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
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      std::vector<float> kid(static_cast<std::size_t>(N) * M, 0.0f);
      for (int v = 0; v < N; ++v) kid[static_cast<std::size_t>(v) * M + v] = 1.0f;
      OperatorSpec op;
      op.name = "fixed"; op.path = "fixed"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = N; op.n_cols = M;
      op.data = view(kid, {N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "scalar");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
      std::vector<float> kfree(static_cast<std::size_t>(3 * N) * M, 0.0f);          // a free kernel: three components a vertex (1, 2, 2), their norm 3 |x|
      const double alpha[3] = {1.0, 2.0, 2.0};
      for (int v = 0; v < N; ++v) for (int c = 0; c < 3; ++c) kfree[static_cast<std::size_t>(3 * v + c) * M + v] = static_cast<float>(alpha[c]);
      OperatorSpec of;
      of.name = "free"; of.path = "free"; of.kind = "inverse kernel"; of.from_manifold_id = cid; of.layout = "dense"; of.n_rows = 3 * N; of.n_cols = M;
      of.data = view(kfree, {3 * N, M}); of.array.compress = false;
      of.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
      of.cols.json("params_json", Json{{"inverse", Json{{"measure", "dSPM"}}}});
      sub.operator_(std::move(of));
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_tiles(ds, sub);
  const SurfaceInput S0 = read_surface(sub, std::string("cortex"), true);
  const SurfaceLevel SL = read_surface_level(sub, S0, L);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S0, SL, sbo);
  // THE PLANTED EPISODES: the level's own pattern of one tile in each hemisphere (left: tile 2; right: tile 16 + 5), 12 Hz under a double sigmoid a = S((t - on)/tr) S(-(t - off)/tf)
  const std::vector<std::int32_t> centres = [&] {
    ArrayReader r = ArrayReader::open(sub.store(), SL.record["node"].get<std::string>() + "/lc/L" + std::to_string(L) + "/centre");
    const auto raw = r.read();
    std::vector<std::int32_t> c(raw.size() / 4);
    std::memcpy(c.data(), raw.data(), raw.size());
    return c;
  }();
  // EPISODE 1 (moving, left hemisphere): the level's own pattern of tile A = 2 hands over to the pattern of a NEIGHBOUR tile B around 12.5 s (both on between 11.5 s and 13.5 s), 8 s to 17 s, one global burst.
  // EPISODE 2 (static, right hemisphere): tile 16 + 5 alone, 30 s to 35 s: a second burst.
  const int tileA = 2;
  int tileB = -1;
  for (const auto& [a, b2] : SL.pairs) { if (a == tileA && b2 < 16) { tileB = b2; break; } if (b2 == tileA && a < 16) { tileB = a; break; } }
  const int tileC = 16 + 5;
  CHECK(tileB >= 0);
  auto pattern = [&](int tile) { Eigen::MatrixXd e = Eigen::MatrixXd::Zero(N, 1); e(centres[static_cast<std::size_t>(tile)], 0) = 1.0; Eigen::MatrixXd f = sband.apply(e); return Eigen::VectorXd(f.col(0) / f.col(0).cwiseAbs().maxCoeff()); };
  const Eigen::VectorXd pA = pattern(tileA), pB = pattern(tileB), pC = pattern(tileC);
  auto S_ = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  std::mt19937 rng(21);
  nxr::testing::PortableNormal nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double s = t / sfreq, osc = std::cos(2 * M_PI * 12.0 * s);
      double x = 0.005 * nd(rng);
      const double a1 = S_((s - 8.0) / 0.3) * S_(-(s - 17.0) / 0.4), wA = S_(-(s - 13.5) / 0.5), wB = S_((s - 11.5) / 0.5);   // A fades out by 13.5 s, B fades in from 11.5 s: both on at 12.5 s
      x += a1 * osc * (wA * pA(v) + wB * pB(v));
      x += pC(v) * S_((s - 30.0) / 0.25) * S_(-(s - 35.0) / 0.35) * osc;
      xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>(x);
    }
  {
    SubjectWriter w = SubjectWriter::open(ds, "S1");
    w.compose([&] {
      const std::string prod = w.product({cid_s, tid});
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(4096)), std::make_pair(std::string("rest_b"), std::int64_t(1000))}) {   // the same samples, two chunk widths
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(w).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "A");
        w.field(std::move(rec));
      }
    });
  }
  // THE CELLS: each recording's band 8-16 Hz grid and its product with the level's tiles (`select_product`, the level coupled to the scale `parts 16`), measured by the preset `cortical_band_cells`;
  // the per-sample product `global_bursts` fits the edges on
  nxrtest::make_time_grids(ds, sub, "band", Json{{"band", "8-16 Hz"}});
  for (const char* rn : {"rest", "rest_b"}) {
    nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", rn}, {"surface", "cortex"}, {"time_grid", std::string(rn) + " time grid band 8-16 Hz"}, {"couple", Json::array({Json{{"factor", 0}, {"scale", "parts 16"}}})}});
    nxrtest::make_sample_product(ds, sub, rn, "cortex", sfreq, "parts 16");
  }
  const auto grid_node = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "rest time grid band 8-16 Hz", .any_status = true});
  CHECK(grid_node.has_value());
  if (!grid_node) return 1;
  W = grid_node->params["grid"]["width"].get<std::int64_t>();
  CHECK(W > 0);
  const std::int64_t tiles = 2 * (1 << L), frames = T / W;
  const double width_s = static_cast<double>(W) / sfreq;

  // ── THE DIRECT COMPUTATION: the band's analytic signal on every channel; the level's band and Riesz part by the dense exact modes; |z| (temporal) or sqrt(|z|^2 + |R z|^2) (spatiotemporal);
  //    the area-weighted mean per cell ──
  SpatialBandOptions rbo = sbo; rbo.riesz = true;
  const SpatialBand rband = make_spatial_band(S0, SL, rbo);
  nxr::select::Mesh lm; lm.positions = S0.vertices; lm.faces = S0.faces; lm.n = N;
  const std::vector<double> va = nxr::select::vertexAreas(lm);
  std::vector<double> tarea(static_cast<std::size_t>(tiles), 0.0);
  for (int v = 0; v < N; ++v) tarea[static_cast<std::size_t>(SL.tile[static_cast<std::size_t>(v)])] += va[static_cast<std::size_t>(v)];
  const auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, 8.0, 16.0); };
  const auto hh = nxr::signal::gainTaps(gain, sfreq, false), qq = nxr::signal::gainTaps(gain, sfreq, true);
  const auto re = nxr::signal::convolveRows(xf.data(), M, T, hh, 0, T, false);
  const auto im = nxr::signal::convolveRows(xf.data(), M, T, qq, 0, T, true);
  Eigen::MatrixXd Re(N, T), Im(N, T);
  for (int v = 0; v < N; ++v) for (int t = 0; t < T; ++t) { Re(v, t) = re[static_cast<std::size_t>(v) * T + t]; Im(v, t) = im[static_cast<std::size_t>(v) * T + t]; }
  const Eigen::MatrixXd FRe = rband.apply(Re), FIm = rband.apply(Im), RRe = rband.apply_riesz(Re), RIm = rband.apply_riesz(Im);
  auto direct = [&](bool mono) {
    std::vector<double> amp(static_cast<std::size_t>(tiles * frames), 0.0);
    for (std::int64_t k = 0; k < frames; ++k) {
      std::vector<double> sum(static_cast<std::size_t>(tiles), 0.0);
      for (int v = 0; v < N; ++v) {
        double sv = 0;
        for (std::int64_t t = k * W; t < (k + 1) * W; ++t) {
          double a = static_cast<float>(FRe(v, t)) * static_cast<double>(static_cast<float>(FRe(v, t))) + static_cast<double>(static_cast<float>(FIm(v, t))) * static_cast<float>(FIm(v, t));
          if (mono) for (int d = 0; d < 3; ++d) { const double r1 = static_cast<float>(RRe(3 * v + d, t)), r2 = static_cast<float>(RIm(3 * v + d, t)); a += r1 * r1 + r2 * r2; }
          sv += std::sqrt(a);
        }
        sum[static_cast<std::size_t>(SL.tile[static_cast<std::size_t>(v)])] += va[static_cast<std::size_t>(v)] * sv;
      }
      for (std::int64_t s = 0; s < tiles; ++s) amp[static_cast<std::size_t>(s * frames + k)] = sum[static_cast<std::size_t>(s)] / (static_cast<double>(W) * tarea[static_cast<std::size_t>(s)]);
    }
    return amp;
  };
  auto worst = [&](const std::vector<float>& arr, const std::vector<double>& want) {
    double w = 0, top = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      const double got = arr[i * 2];
      if (std::isnan(got)) return 1e9;
      top = std::max(top, std::abs(want[i]));
      w = std::max(w, std::abs(got - want[i]));
    }
    return w / top;
  };

  // ── THE RUNS: temporal (named and unnamed: the same run), then spatiotemporal (another recipe: refused unless --replace) ──
  g::RunOptions all; all.chunks = "all";
  g::RunOptions rep = all;
  rep.replace = true;
  g::RunOptions preset = all;
  preset.preset = "cortical_band_cells";
  const g::Definition dm = g::load("measure");
  // the preset's measurement of one recording: its band on the cortex (the chain's own parameters in `chain_parameters`) over the recording's product of coupled tiles
  auto cells_run = [&](const std::string& recording, const Json& chain, const g::RunOptions& o) {
    Json cp = Json{{"recording", recording}, {"band", "8-16 Hz"}, {"spatial_scale", "parts 16"}, {"transform", "exact"}, {"spatial_shape", "meyer"}, {"method", "MNE"}};
    for (const auto& [k, v] : chain.items()) cp[k] = v;
    return g::run(ds, sub, dm, Json{{"recording", recording}, {"band", "8-16 Hz"}, {"spatial_scale", "parts 16"}, {"chain_parameters", cp}}, o);
  };
  const Json no_extra = Json::object(), mono = Json{{"envelope", "spatiotemporal"}}, temporal = Json{{"envelope", "temporal"}};
  // the stored cells' amplitude of a recording, as (value, count) pairs [tile x frame]
  auto cells_of = [&](const std::string& recording) {
    const auto rec = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = recording, .any_status = true});
    const auto prod = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = recording + " cells 8-16 Hz parts 16", .any_status = true});     // the product the preset makes (its time grid and the level coupled to the scale)
    CHECK(rec.has_value() && prod.has_value());
    const MeasurementValues mv = measurement_values(sub, prod->id, rec->id, std::nullopt, std::nullopt, "amplitude");
    CHECK(mv.members == tiles * frames);
    std::vector<float> out(static_cast<std::size_t>(tiles * frames * 2), std::nanf(""));
    for (std::int64_t k = 0; k < frames; ++k) for (std::int64_t s2 = 0; s2 < tiles; ++s2) { out[static_cast<std::size_t>((s2 * frames + k) * 2)] = static_cast<float>(mv.values[static_cast<std::size_t>(s2 + tiles * k)]); out[static_cast<std::size_t>((s2 * frames + k) * 2 + 1)] = 1.0f; }
    return out;
  };
  const auto rt = cells_run("rest", no_extra, preset);
  CHECK(rt.size() == 1 && rt[0]["hit"] == false);
  const auto rtb = cells_run("rest_b", no_extra, preset);
  const std::vector<float> ct = cells_of("rest"), ctb = cells_of("rest_b");
  const std::vector<double> want_t = direct(false), want_s = direct(true);
  const double et = worst(ct, want_t);
  std::printf("  temporal cells vs direct: worst relative error %.2e\n", et);
  CHECK(et < 2e-4);
  // a rerun of the `measure` preset with the same recipe is a cache hit (the method rule); another envelope's rows are another recipe's: refused without --replace, never silently replaced
  { const auto r = cells_run("rest", no_extra, preset); CHECK(r.size() == 1 && r[0]["hit"] == true); }
  CHECK(code_of([&] { cells_run("rest", temporal, preset); }) == std::string(code::already_exists));
  CHECK(code_of([&] { cells_run("rest", mono, preset); }) == std::string(code::already_exists));   // another envelope's rows are another recipe's: refused, never silently replaced
  g::RunOptions preset_rep = preset;
  preset_rep.replace = true;
  const auto rs = cells_run("rest", mono, preset_rep);
  CHECK(rs.size() == 1 && rs[0]["hit"] == false);
  cells_run("rest_b", mono, preset_rep);
  const std::vector<float> cs = cells_of("rest"), csb = cells_of("rest_b");
  const double es = worst(cs, want_s);
  std::printf("  spatiotemporal cells vs direct: worst relative error %.2e\n", es);
  CHECK(es < 5e-4);
  {
    // the recipe names the envelope; the measurement's rows record how they were computed
    const auto rec = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = "rest", .any_status = true});
    const auto cells_product = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = "rest cells 8-16 Hz parts 16", .any_status = true});
    CHECK(cells_product.has_value());
    const auto rows = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"selection_id", V(cells_product->id)}, {"of_field_id", V(rec->id)}, {"measure", V("amplitude")}});
    CHECK(!rows.empty() && !text(rows.front(), "computed_by").empty());
    const std::string dumped = rs[0].dump();
    CHECK(dumped.find("spatiotemporal") != std::string::npos);
  }
  // a rerun is a hit; the other chunk width gives the same cells; the temporal rerun is refused again (the rows are the spatiotemporal ones now) unless --replace
  { const auto r = cells_run("rest", mono, preset); CHECK(r.size() == 1 && r[0]["hit"] == true); }
  {
    double diff = 0;
    for (std::size_t i = 0; i < cs.size(); ++i) if (!std::isnan(cs[i]) || !std::isnan(csb[i])) diff = std::max(diff, static_cast<double>(std::abs(cs[i] - csb[i])));
    std::printf("  chunk widths 4096 and 1000, spatiotemporal: max difference %.2e\n", diff);
    CHECK(diff < 1e-6);
  }
  CHECK(code_of([&] { cells_run("rest", no_extra, preset); }) == std::string(code::already_exists));
  CHECK(code_of([&] { cells_run("rest", Json{{"envelope", "spatial"}}, preset); }) != std::string());

  // ── THE HAND-OVER: the strongest cell's trace (the trace the bursts are detected on, here the max over the tiles) inside the episode (8 s … 17 s) ──
  auto trace = [&](const std::vector<float>& c) {
    std::vector<double> tr(static_cast<std::size_t>(frames), 0.0);
    for (std::int64_t k = 0; k < frames; ++k) for (std::int64_t s = 0; s < tiles; ++s) tr[static_cast<std::size_t>(k)] = std::max(tr[static_cast<std::size_t>(k)], static_cast<double>(c[static_cast<std::size_t>((s * frames + k) * 2)]));
    return tr;
  };
  const std::vector<double> trt = trace(ct), trs = trace(cs);
  const int f_lo = static_cast<int>(std::ceil(9.0 / width_s)), f_hi = static_cast<int>(std::floor(16.0 / width_s));      // the episode's plateau, away from its sigmoid edges
  auto dip = [&](const std::vector<double>& tr) {
    double mn = 1e300, mx = 0;
    for (int k = f_lo; k < f_hi; ++k) { mn = std::min(mn, tr[static_cast<std::size_t>(k)]); mx = std::max(mx, tr[static_cast<std::size_t>(k)]); }
    return mn / mx;
  };
  std::printf("  the episode's trace (frames %d .. %d): min / max = %.3f under temporal, %.3f under spatiotemporal\n", f_lo, f_hi - 1, dip(trt), dip(trs));
  CHECK(dip(trs) > dip(trt));

  // a FREE kernel (three components a vertex): the amplitude is the norm over the components of z and of R z, so alpha = (1, 2, 2) gives 3 times the fixed kernel's cells
  {
    cells_run("rest", Json{{"envelope", "spatiotemporal"}, {"method", "dSPM"}}, preset_rep);
    const std::vector<float> cf = cells_of("rest");
    double w = 0, top = 0;
    for (std::size_t i = 0; i < cs.size(); i += 2) { if (std::isnan(cs[i])) continue; top = std::max(top, 3.0 * std::abs(static_cast<double>(cs[i]))); w = std::max(w, std::abs(cf[i] - 3.0 * cs[i])); }
    std::printf("  free kernel (1, 2, 2), spatiotemporal: cells vs 3 x the fixed kernel's: worst relative difference %.2e\n", w / top);
    CHECK(w / top < 2e-4);
  }

  // ── THE EVENT: the extent's z = (cell - the tile's median) / the tile's robust SD (as global_bursts does); the exit level between the two envelopes' troughs, found from the stored cells ──
  auto median = [](std::vector<double> x) { std::sort(x.begin(), x.end()); const std::size_t n = x.size(); return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]); };
  auto zs = [&](const std::vector<float>& c) {
    std::vector<double> z(static_cast<std::size_t>(tiles * frames), 0.0);
    for (std::int64_t s2 = 0; s2 < tiles; ++s2) {
      std::vector<double> x, dv;
      for (std::int64_t k = 0; k < frames; ++k) x.push_back(c[static_cast<std::size_t>((s2 * frames + k) * 2)]);
      const double md = median(x);
      for (const double v : x) dv.push_back(std::abs(v - md));
      const double sdv = 1.4826 * median(dv);
      for (std::int64_t k = 0; k < frames; ++k) z[static_cast<std::size_t>(s2 * frames + k)] = sdv > 0 ? (x[static_cast<std::size_t>(k)] - md) / sdv : 0.0;
    }
    return z;
  };
  const std::vector<double> zt = zs(ct), zsp = zs(cs);
  auto ztrace = [&](const std::vector<double>& z, int k) { double m2 = -1e300; for (int s2 = 0; s2 < 16; ++s2) m2 = std::max(m2, z[static_cast<std::size_t>(s2 * frames + k)]); return m2; };
  double trough_t = 1e300, trough_s = 1e300, peak_t = 0, peak_s = 0;
  for (int k = f_lo; k < f_hi; ++k) { trough_t = std::min(trough_t, ztrace(zt, k)); trough_s = std::min(trough_s, ztrace(zsp, k)); peak_t = std::max(peak_t, ztrace(zt, k)); peak_s = std::max(peak_s, ztrace(zsp, k)); }
  std::printf("  extent z over the left episode: temporal trough %.0f peak %.0f; spatiotemporal trough %.0f peak %.0f\n", trough_t, peak_t, trough_s, peak_s);
  CHECK(trough_s > trough_t);
  {
    const g::Definition dg = g::load("global_bursts");
    const double ex = 0.5 * (trough_t + trough_s);
    int cnt[2] = {0, 0};
    for (int e = 0; e < 2; ++e) {
      const double enter_z = 1.1 * ex,                       // each half of the hand-over, under the temporal envelope, must reach it on its own
                    area_two = 1.5 * *std::max_element(tarea.begin(), tarea.end());   // an event of at least two cells
      Json st = Json{{"recording", "rest"}, {"spatial_scale", "parts 16"}, {"product", "rest cortex product"}, {"sample_product", "rest cortex samples product"}, {"transform", "exact"}, {"spatial_shape", "meyer"}, {"method", "MNE"},
                     {"envelope", e ? "spatiotemporal" : "temporal"}, {"extent_enter_sd", enter_z}, {"extent_exit_sd", ex}, {"min_area", area_two}};
      const auto r = g::run(ds, sub, dg, st, rep);
      const Json x = out_of(r[0], "extents");
      ArrayReader rd = ArrayReader::open(sub.store(), x["path"].get<std::string>());
      const auto raw = rd.read();
      std::vector<std::int32_t> lab(raw.size() / 4);                                // the cells are numbered s + tiles k
      std::memcpy(lab.data(), raw.data(), raw.size());
      std::set<int> ev;
      for (int s2 = 0; s2 < 16; ++s2) for (int k = static_cast<int>(8.0 / width_s); k < static_cast<int>(std::ceil(17.0 / width_s)); ++k) if (lab[static_cast<std::size_t>(s2 + tiles * k)] >= 0) ev.insert(lab[static_cast<std::size_t>(s2 + tiles * k)]);
      cnt[e] = static_cast<int>(ev.size());
    }
    std::printf("  events of the left episode at exit z %.0f: temporal %d, spatiotemporal %d\n", ex, cnt[0], cnt[1]);
    CHECK(cnt[1] == 1 && cnt[0] >= 2);          // the hand-over: split by the temporal envelope's dip, one event under the spatiotemporal
  }
  return nxrtest::finish("test_spatial_analytic");
}
