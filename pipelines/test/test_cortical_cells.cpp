// run cortical_cells (spec docs/specs/2026-11-13-cortical-events.md, phase 2a) on a synthetic source space (never a real dataset): a two-hemisphere surface, a recording with one channel a vertex (the
// kernel is the identity: the sources are the channels), a 12 Hz oscillation planted on a spatial bump during known time tiles, a bad span. Checks: the joint partition (the atlas's code layout,
// closed form, found by its pair); the stored cell amplitude and envelope_max equal the DIRECT computation (the band's analytic signal, the level's spatial band applied to the field, |z|, the area-weighted
// mean per cell, over the tiles whole and good); the planted cells stand out; a free kernel's norm over its components; the same with another chunk width; a rerun a hit; the Chebyshev transform close to
// the exact one; another job's rows refused unless --replace; the plan's refusals (no spatial band, a band that is no joint's).
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/io.h"
#include "nxr/pipelines/constant_q.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
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
namespace tk = nxr::io::task;

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
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_cortical_cells_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 8192;
  const double sfreq = 250.0, bad0 = 10.0, bad1 = 10.5;
  const int spaceLevel = 4;       // 32 tiles of ~5 vertices: at level 4 EVERY tile's own pattern localizes (level 3: 12 of 16 do, its band is a few global modes: the issues log)
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
        rec.name = name; rec.path = tk::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
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
  // the alpha level of the time ladder (8-16 Hz)
  int timeLevel = -1;
  std::int64_t frames = 0;
  for (const tk::Node& m : tk::Subject::of(sub).find_all({.cls = tk::Cls::selection, .params_present = {"resolves_hz"}, .any_status = true})) {
    if (!m.member_of_id) continue;
    const Json& pj = m.params;
    if (pj["resolves_hz"]["lo"] == 8.0 && pj["resolves_hz"]["hi"] == 16.0 && pj["window_samples"] == W) { timeLevel = static_cast<int>(m.member_ordinal.value_or(0)); frames = pj["tiling"]["count"].get<std::int64_t>(); }
  }
  CHECK(timeLevel >= 0 && frames == T / W);
  if (timeLevel < 0) return 1;

  // ── THE JOINT PARTITION: the atlas's layout, closed form, one per pair and recording ──
  JointComputeOptions jo;
  jo.recording = "rest"; jo.space_level = spaceLevel; jo.time_level = timeLevel;
  const auto j1 = joint_compute(ds, sub, jo);
  const auto jrow = tk::Subject::of(sub).by_id(tk::Cls::selection, j1.info["id"].get<std::string>());
  CHECK(jrow.has_value() && jrow->type.value_or("") == "set" && !jrow->array_path.has_value());
  const Json jp = jrow->params["joint"];
  const std::int64_t tiles = jp["tiles"].get<std::int64_t>();
  CHECK(jp["bits"] == 5 && jp["frames"] == frames && tiles == 2 * (1 << spaceLevel) && jp["space_level"] == spaceLevel && jp["time_level"] == timeLevel && jrow->n_members.value_or(0) == tiles * frames);
  CHECK(joint_bits(1) == 1 && joint_bits(2) == 1 && joint_bits(3) == 2 && joint_bits(32) == 5 && joint_bits(33) == 6);   // the atlas's bitsFor
  CHECK(joint_compute(ds, sub, jo).existed);
  JointComputeOptions jb = jo; jb.recording = "rest_b";
  joint_compute(ds, sub, jb);
  CHECK(code_of([&] { JointComputeOptions e = jo; e.space_level = 99; joint_compute(ds, sub, e); }) == std::string(code::invalid_argument));
  CHECK(code_of([&] { JointComputeOptions e = jo; e.time_level = 99; joint_compute(ds, sub, e); }) == std::string(code::invalid_argument));

  // ── THE DIRECT COMPUTATION: the band's analytic signal on every channel, the level's spatial band (exact modes) on the field, |z|, the area-weighted mean per cell ──
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  const SurfaceLevel SL = read_surface_level(sub, S, spaceLevel);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S, SL, sbo);
  nxr::ladder::Mesh lm; lm.positions = S.vertices; lm.faces = S.faces; lm.n = N;
  const std::vector<double> va = nxr::ladder::vertexAreas(lm);
  std::vector<double> tarea(static_cast<std::size_t>(tiles), 0.0);
  for (int v = 0; v < N; ++v) tarea[static_cast<std::size_t>(SL.tile[static_cast<std::size_t>(v)])] += va[static_cast<std::size_t>(v)];
  const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 16.0); };
  const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
  const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false);
  const auto im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
  Eigen::MatrixXd Re(N, T), Im(N, T);
  for (int v = 0; v < N; ++v) for (int t = 0; t < T; ++t) { Re(v, t) = re[static_cast<std::size_t>(v) * T + t]; Im(v, t) = im[static_cast<std::size_t>(v) * T + t]; }
  const Eigen::MatrixXd FRe = sband.apply(Re), FIm = sband.apply(Im);
  // per cell: amplitude, envelope_max; a time tile whole and free of bad samples, else NaN
  auto direct = [&](double scale, std::vector<double>& amp, std::vector<double>& mx) {
    amp.assign(static_cast<std::size_t>(tiles * frames), std::nan(""));
    mx.assign(amp.size(), std::nan(""));
    for (std::int64_t k = 0; k < frames; ++k) {
      bool whole = true;
      for (std::int64_t t = k * W; t < (k + 1) * W; ++t) { const double sec = t / sfreq; if (sec >= bad0 && sec < bad1) whole = false; }
      if (!whole) continue;
      std::vector<double> sum(static_cast<std::size_t>(tiles), 0.0), big(static_cast<std::size_t>(tiles), 0.0);
      for (int v = 0; v < N; ++v) {
        const int s = SL.tile[static_cast<std::size_t>(v)];
        double sv = 0;
        for (std::int64_t t = k * W; t < (k + 1) * W; ++t) {
          const double zr = static_cast<float>(FRe(v, t)), zi = static_cast<float>(FIm(v, t));
          const double z = scale * std::sqrt(zr * zr + zi * zi);
          sv += z; big[static_cast<std::size_t>(s)] = std::max(big[static_cast<std::size_t>(s)], z);
        }
        sum[static_cast<std::size_t>(s)] += va[static_cast<std::size_t>(v)] * sv;
      }
      for (std::int64_t s = 0; s < tiles; ++s) { amp[static_cast<std::size_t>(s * frames + k)] = sum[static_cast<std::size_t>(s)] / (static_cast<double>(W) * tarea[static_cast<std::size_t>(s)]); mx[static_cast<std::size_t>(s * frames + k)] = big[static_cast<std::size_t>(s)]; }
    }
  };
  auto worst = [&](const std::vector<float>& arr, int comp, const std::vector<double>& want) {
    double w = 0, top = 0;
    std::int64_t nan_ok = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
      const double got = arr[i * 2 + static_cast<std::size_t>(comp)];
      if (std::isnan(want[i])) { nan_ok += std::isnan(got); continue; }
      if (std::isnan(got)) return 1e9;
      top = std::max(top, std::abs(want[i]));
      w = std::max(w, std::abs(got - want[i]));
    }
    (void)nan_ok;
    return w / top;
  };

  // ── 1. THE RUN: exact spatial band, the fixed (scalar) kernel, every chunk ──
  const g::Definition d = g::load("cortical_cells");
  g::RunOptions all; all.chunks = "all";
  const Json set1 = Json{{"space_level", spaceLevel}, {"method", "dSPM"}, {"transform", "exact"}, {"spatial_shape", "meyer"}};
  const auto plans = g::plan_runs(d, sub.db(), sub.id(), set1);
  CHECK_EQ(static_cast<int>(plans.size()), 2);
  const auto r1 = g::run(ds, sub, d, set1, all);
  CHECK_EQ(static_cast<int>(r1.size()), 2);
  std::vector<double> amp, mx;
  direct(1.0, amp, mx);
  std::vector<float> got_amp[2];
  for (const auto& r : r1) {
    const int i = r.value("anchor", std::string()) == "rest" ? 0 : 1;
    const Json am = out_of(r, "amplitude"), pk = out_of(r, "peak");
    CHECK(!am.is_null() && !pk.is_null() && r["hit"] == false);
    std::vector<std::int64_t> shape;
    got_amp[i] = nxrtest::read_packed(sub.folder(), am, &shape);
    CHECK(shape == (std::vector<std::int64_t>{tiles, frames, 2}));
    const std::vector<float> gp = nxrtest::read_packed(sub.folder(), pk);
    const double ea = worst(got_amp[i], 0, amp), ep = worst(gp, 0, mx);
    std::printf("  %s: cell amplitude vs direct: worst relative error %.2e; envelope_max %.2e\n", r["anchor"].get<std::string>().c_str(), ea, ep);
    CHECK(ea < 2e-4 && ep < 2e-4);
    CHECK(am["over"].get<std::string>().find("x") != std::string::npos);
    // the rows are of the recording, on the joint Selection (a count row beside them); the count of a good tile is its window
    const auto ts = tk::Subject::of(sub);
    const auto recid = ts.find({.cls = tk::Cls::field, .name = r["anchor"].get<std::string>(), .any_status = true});
    const auto jsel = ts.find({.cls = tk::Cls::selection, .name = am["over"].get<std::string>(), .any_status = true});
    const auto rows = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"selection_id", V(jsel->id)}, {"of_field_id", V(recid->id)}}, "measure");
    CHECK(rows.size() == 3 && text(rows[0], "measure") == "amplitude" && text(rows[1], "measure") == "count" && text(rows[2], "measure") == "envelope_max");
    CHECK(gp.size() == got_amp[i].size());
    CHECK(got_amp[i][static_cast<std::size_t>((0 * frames + 12) * 2 + 1)] == static_cast<float>(W));     // the count component: the samples of a whole tile
  }
  // THE DEFAULT SPATIAL SHAPE: the Mexican hat (the step's and the definition's default), another operator than Meyer's: the run is refused unless replaced, its
  // provenance names the shape, and its cells differ from the Meyer band's (the hat's own numerics are tested in test_spatial_hat)
  {
    Json hat = set1; hat.erase("spatial_shape");
    CHECK(d.parameters["spatial_shape"] == "mexhat");
    g::RunOptions rep = all; rep.replace = true;
    const auto rh = g::run(ds, sub, d, hat, rep);
    CHECK_EQ(static_cast<int>(rh.size()), 2);
    for (const auto& r : rh) {
      const Json am = out_of(r, "amplitude");
      CHECK(!am.is_null());
      const std::vector<float> a = nxrtest::read_packed(sub.folder(), am);
      const int i = r.value("anchor", std::string()) == "rest" ? 0 : 1;
      double dsum = 0, top = 0;
      for (std::size_t k = 0; k < a.size() && k < got_amp[i].size(); k += 2) { if (std::isnan(a[k]) || std::isnan(got_amp[i][k])) continue; dsum = std::max(dsum, static_cast<double>(std::abs(a[k] - got_amp[i][k]))); top = std::max(top, static_cast<double>(std::abs(got_amp[i][k]))); }
      std::printf("  %s: Mexican hat vs Meyer cells: max difference %.2e of %.3g\n", r["anchor"].get<std::string>().c_str(), dsum, top);
      CHECK(top > 0 && dsum > 1e-3 * top);
    }
    g::run(ds, sub, d, set1, rep);                                            // back to the Meyer rows for what follows
  }
  // the chunk width: the same samples stored in chunks of 4096 and of 1000 give the same cells
  {
    double diff = 0;
    for (std::size_t i = 0; i < got_amp[0].size(); ++i) if (!std::isnan(got_amp[0][i]) || !std::isnan(got_amp[1][i])) diff = std::max(diff, static_cast<double>(std::abs(got_amp[0][i] - got_amp[1][i])));
    std::printf("  chunk widths 4096 and 1000: max difference %.2e\n", diff);
    CHECK(diff < 1e-6);
  }
  // the planted cells stand out: the largest cell is in a planted time tile (14 … 17); a quiet time tile of the same spatial tile is far below
  {
    std::size_t best = 0;
    for (std::size_t i = 0; i < amp.size(); ++i) if (!std::isnan(amp[i]) && (std::isnan(amp[best]) || amp[i] > amp[best])) best = i;
    const std::int64_t s = static_cast<std::int64_t>(best) / frames, k = static_cast<std::int64_t>(best) % frames;
    std::printf("  the largest cell: spatial tile %lld (the bump's centre is in tile %d: a bump wider than a tile is not one tile's), time tile %lld; %.4g against %.4g in the same tile at time tile 4\n", static_cast<long long>(s), SL.tile[static_cast<std::size_t>(centre)], static_cast<long long>(k), amp[best], amp[static_cast<std::size_t>(s * frames + 4)]);
    CHECK(k >= 14 && k <= 17 && amp[best] > 5 * amp[static_cast<std::size_t>(s * frames + 4)]);   // at level 3 the band's modes are global (a hemisphere holds 8 tiles): the time tile is the planted one, the spatial tile is the band's
    // the bad span's time tiles are NaN in every cell
    bool nan_bad = true;
    for (std::int64_t sx = 0; sx < tiles; ++sx) nan_bad = nan_bad && std::isnan(got_amp[0][static_cast<std::size_t>((sx * frames + 9) * 2)]) && std::isnan(got_amp[0][static_cast<std::size_t>((sx * frames + 10) * 2)]);
    CHECK(nan_bad);
  }
  // a rerun is a hit; the same outputs of another transform are another job's, refused unless --replace; the Chebyshev band is close to the exact one
  {
    const auto r = g::run(ds, sub, d, set1, all);
    CHECK(r.size() == 2 && r[0]["hit"] == true && r[1]["hit"] == true);
    Json set2 = set1; set2["transform"] = "chebyshev";
    CHECK(code_of([&] { g::run(ds, sub, d, set2, all); }) == std::string(code::already_exists));
    g::RunOptions rep = all; rep.replace = true;
    const auto rc = g::run(ds, sub, d, set2, rep);
    CHECK(rc.size() == 2);
    // what preparation RESOLVED (the series degree chosen from the tolerance) is reported by the step and kept with the rows: in the report and in the recipe the rows name
    {
      const Json sbj = rc[0]["prepared"]["apply_kernel:sources"]["spatial_band"];
      std::printf("  resolved Chebyshev order %d (series error %.2g)\n", sbj["order"].get<int>(), sbj["series_error"].get<double>());
      CHECK(sbj["order"].get<int>() >= 40 && sbj["series_error"].get<double>() <= 0.01 && sbj["transform"] == "chebyshev" && sbj["level"] == spaceLevel);
      const nxr::io::testing::Probe probe(sub);
      std::optional<Row> rec;     // the newest recipe entry of this pipeline that names the Chebyshev transform
      for (const Row& dd : probe.rows("dictionary", {{"name", V(std::string("recipes"))}}))
        for (const Row& e : probe.rows("dictionary_entry", {{"dictionary_id", V(text(dd, "id"))}}))
          if (text(e, "name").rfind("pipeline cortical_cells", 0) == 0 && text(e, "params_json").find("chebyshev") != std::string::npos) rec = e;
      CHECK(rec.has_value() && Json::parse(text(*rec, "params_json"))["prepared"]["apply_kernel:sources"]["spatial_band"]["order"] == sbj["order"]);
      std::vector<Row> by;
      for (const Row& r : probe.rows("selection_measurement", {{"measure", V(std::string("amplitude"))}})) if (text(r, "computed_by").rfind("pipeline cortical_cells", 0) == 0) by.push_back(r);
      bool named = false;
      for (const Row& r : by) named = named || (rec && text(r, "computed_by").rfind(text(*rec, "name"), 0) == 0);   // the rows name the recipe that holds the record
      CHECK(named);
    }
    // THE DEFAULT: this surface has no stored eigenmodes and cannot be glued (no medial-wall atlas), so `auto` falls back to the series, SAID: a warning in the
    // report and the provenance, never silent; the same result as the forced series
    {
      Json set4 = set1; set4.erase("transform");
      const auto ra = g::run(ds, sub, d, set4, rep);
      CHECK(ra.size() == 2 && ra[0]["warnings"].is_array() && ra[0]["warnings"].size() == 1);
      CHECK(ra[0]["warnings"][0].get<std::string>().find("CHEBYSHEV fallback") != std::string::npos && ra[0]["warnings"][0].get<std::string>().find("level " + std::to_string(spaceLevel)) != std::string::npos);
      const Json sba = ra[0]["prepared"]["apply_kernel:sources"]["spatial_band"];
      CHECK(sba["requested"] == "auto" && sba["transform"] == "chebyshev" && sba["warnings"].size() == 1);
    }
    const std::vector<float> cheb = nxrtest::read_packed(sub.folder(), out_of(rc[0], "amplitude"));
    std::vector<double> ca(cheb.size() / 2);
    for (std::size_t i = 0; i < ca.size(); ++i) ca[i] = cheb[i * 2];
    double w = 0, top = 0;
    for (std::size_t i = 0; i < ca.size(); ++i) if (!std::isnan(amp[i])) { w = std::max(w, std::abs(ca[i] - amp[i])); top = std::max(top, amp[i]); }
    std::printf("  Chebyshev vs exact spatial band: max cell difference %.2e of %.3g\n", w, top);
    CHECK(w < 0.03 * top);
  }
  // a FREE kernel: three components a vertex with weights 1, 2, 2: the norm over the components is 3 |x|
  {
    Json set3 = set1; set3["method"] = "MNE";
    g::RunOptions rep = all; rep.replace = true;
    const auto rf = g::run(ds, sub, d, set3, rep);
    CHECK(rf.size() == 2);
    std::vector<double> amp3, mx3;
    direct(3.0, amp3, mx3);
    const double e = worst(nxrtest::read_packed(sub.folder(), out_of(rf[0], "amplitude")), 0, amp3);
    std::printf("  free kernel (norm over 3 components = 3 |x|): worst relative error %.2e\n", e);
    CHECK(e < 2e-4);
  }
  // ── WHERE: a pattern of the level's OWN scale (the level's band filtered delta at a tile's centre vertex, F_l δ_c), oscillating at 12 Hz in known time tiles, in two tiles at different times: the
  //    largest cell of each planted time tile is its planted spatial tile (the cell grid localizes what the band's scale can hold) ──
  {
    const std::vector<std::int32_t> centres = [&] {
      ArrayReader r = ArrayReader::open(sub.store(), SL.record["node"].get<std::string>() + "/lc/L" + std::to_string(spaceLevel) + "/centre");
      const auto raw = r.read();
      std::vector<std::int32_t> c(raw.size() / 4);
      std::memcpy(c.data(), raw.data(), raw.size());
      return c;
    }();
    const int tA = 2, tB = 5;                                      // two planted spatial tiles
    auto pattern = [&](int tile) { Eigen::MatrixXd e = Eigen::MatrixXd::Zero(N, 1); e(centres[static_cast<std::size_t>(tile)], 0) = 1.0; Eigen::MatrixXd f = sband.apply(e); return Eigen::VectorXd(f.col(0) / f.col(0).cwiseAbs().maxCoeff()); };
    const Eigen::VectorXd pa = pattern(tA), pb = pattern(tB);
    std::vector<float> xl(static_cast<std::size_t>(M) * T);
    std::mt19937 r2(9); std::normal_distribution<double> n2;
    for (int v = 0; v < M; ++v)
      for (int t = 0; t < T; ++t) {
        const std::int64_t k = t / W;
        const double osc = std::cos(2 * M_PI * 12.0 * t / sfreq);
        xl[static_cast<std::size_t>(v) * T + t] = static_cast<float>((k >= 12 && k <= 14 ? pa(v) * osc : 0.0) + (k >= 20 && k <= 22 ? pb(v) * osc : 0.0) + 0.01 * n2(r2));
      }
    {
      SubjectWriter w = SubjectWriter::open(ds, "S1");
      w.compose([&] {
        FieldSpec rec;
        const std::string prod = w.product({cid_s, tid});
        rec.name = "loc"; rec.path = tk::Subject::of(w).node_path(prod, "loc"); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xl, {M, T}); rec.array.chunks = Shape{M, std::int64_t(4096)}; rec.array.compress = false; rec.cols.set("unit", "A");
        w.field(std::move(rec));
      });
    }
    JointComputeOptions jl = jo; jl.recording = "loc";
    joint_compute(ds, sub, jl);
    Json set3 = set1; set3["method"] = "MNE";
    const auto rl = g::run(ds, sub, d, set3, all);
    Json am;
    for (const auto& r : rl) if (r.value("anchor", std::string()) == "loc") am = out_of(r, "amplitude");
    CHECK(!am.is_null());
    const std::vector<float> a = nxrtest::read_packed(sub.folder(), am);
    auto at = [&](std::int64_t s, std::int64_t k) { return a[static_cast<std::size_t>((s * frames + k) * 2)]; };
    auto top = [&](std::int64_t k) { std::int64_t b2 = 0; for (std::int64_t s = 1; s < tiles; ++s) if (at(s, k) > at(b2, k)) b2 = s; return b2; };
    std::printf("  localization (level %d): planted tile %d in time tiles 12-14 -> top tiles %lld %lld %lld; planted tile %d in 20-22 -> %lld %lld %lld\n", spaceLevel, tA, static_cast<long long>(top(12)), static_cast<long long>(top(13)), static_cast<long long>(top(14)),
                tB, static_cast<long long>(top(20)), static_cast<long long>(top(21)), static_cast<long long>(top(22)));
    for (int k = 12; k <= 14; ++k) CHECK_EQ(top(k), tA);
    for (int k = 20; k <= 22; ++k) CHECK_EQ(top(k), tB);
    // the largest cell of the whole recording is a planted cell, and a quiet time tile is far below it
    std::int64_t bs = 0, bk = 0;
    for (std::int64_t s = 0; s < tiles; ++s) for (std::int64_t k = 0; k < frames; ++k) if (!std::isnan(at(s, k)) && at(s, k) > at(bs, bk)) { bs = s; bk = k; }
    CHECK((bs == tA && bk >= 12 && bk <= 14) || (bs == tB && bk >= 20 && bk <= 22));
    CHECK(at(tA, 13) > 10 * at(tA, 4) && at(tB, 21) > 10 * at(tB, 4));
  }
  // ── THE PLAN'S REFUSALS ──
  {
    CHECK(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), Json{{"band", Json::array({16.0, 32.0})}}); }) == std::string(code::invalid_argument));       // no joint partition of that band
    CHECK(code_of([&] { g::plan_runs(d, sub.db(), sub.id(), Json{{"space_level", 6}}); }) == std::string(code::invalid_argument));                          // none of that spatial level
    const std::string bad = R"(pipeline: cells_no_spatial_band
inputs:
  recording: {field: recording, each: true, where: {name: rest}}
  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: fixed}}
  cells: {selection: set, where: {params.joint.recording: "{recording}"}}
parameters: {band: [8.0, 16.0]}
steps:
  chunks:
    over: recording
    do:
      - {step: analytic, in: {x: recording}, with: {band: $band}, out: z}
      - {step: apply_kernel, in: {x: z, kernel: kernel}, out: j}
      - {step: magnitude, in: {x: j}, out: m}
      - {step: measure, in: {x: m, over: cells}, with: {measure: amplitude}, out: amplitude}
outputs:
  amplitude: {kind: measurement}
)";
    const g::Definition bd = g::parse(parse_yaml(bad, "bad.yaml"), "bad.yaml", bad);
    std::string why;
    try { g::plan_runs(bd, sub.db(), sub.id(), Json::object()); } catch (const Error& e) { why = e.what(); }
    CHECK(why.find("spatial band") != std::string::npos);
    const std::string env = bad.substr(0, bad.find("      - {step: magnitude")) + "      - {step: envelope, in: {x: recording}, with: {band: $band}, out: m}\n" + bad.substr(bad.find("      - {step: measure"));
    // measure over a joint cell is taken of |z| per vertex: an envelope of sensors is refused (and so is a field not on a surface)
    const std::string env2 = R"(pipeline: cells_envelope
inputs:
  recording: {field: recording, each: true, where: {name: rest}}
  cells: {selection: set, where: {params.joint.recording: "{recording}"}}
parameters: {band: [8.0, 16.0]}
steps:
  chunks:
    over: recording
    do:
      - {step: envelope, in: {x: recording}, with: {band: $band}, out: m}
      - {step: measure, in: {x: m, over: cells}, with: {measure: amplitude}, out: amplitude}
outputs:
  amplitude: {kind: measurement}
)";
    (void)env;
    const g::Definition ed = g::parse(parse_yaml(env2, "env.yaml"), "env.yaml", env2);
    why.clear();
    try { g::plan_runs(ed, sub.db(), sub.id(), Json::object()); } catch (const Error& e) { why = e.what(); }
    CHECK(why.find("magnitude") != std::string::npos);
  }
  return nxrtest::finish("test_cortical_cells");
}
