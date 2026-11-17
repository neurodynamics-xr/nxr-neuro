// phasegradient compute AT A SPATIAL SCALE against GROUND TRUTH: two unit spheres (one channel a vertex, a free kernel giving each source its channel
// along the outward normal), each with a cap facing the other as its 'unassigned' Desikan-Killiany wall, glued and solved (`eigen solve`). Three recordings
// at 10 Hz: a PLANE wave travelling along +z, cos(ωt − k z) (one timing map explains it: timing = −k z / ω, the inconsistency ≈ 0); a wave ROTATING round
// each sphere's z axis, cos(ωt − m λ) (no timing map explains it: the inconsistency high); a STANDING pattern cos(ωt) cos(k z) (travelling index ≈ 0).
// Through `crossspectrum compute`, the stored cross-spectrum, the fused kernel, the band, the synthesis and the reading in the modes. Also: Weyl members; the bridge
// NaN; the refusals (signal normal, field current); a re-run is a cache hit.
// STORED (decision 1 of the measurement pyramid): the timing map, propagation, speed, stream function and inconsistency are ONE statistic of vertex x quantity (the modes' coefficients another, on the eigen
// axis), no per-vertex Fields; the band's power, the propagation's flux and the phase field's circulation are array-backed measurements on the leaf partition: the flux against the plane wave's divergence
// (-2 k z on the unit sphere), the circulation against the rotating wave's winding (2 pi round a pole's leaf; 0 for the plane wave).
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <tuple>

#include "brain_helpers.h"
#include "nxr/io/spatial_bands.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/bst/frequency.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

static std::vector<double> read_field_at(SubjectWriter& sub, const std::string& array_path) {
  ArrayReader a = ArrayReader::open(sub.store(), array_path);
  const auto raw = a.read();
  std::vector<double> out(raw.size() / sizeof(double));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(double));
  return out;
}
// one quantity (a column) of a vertex x quantity statistic, per vertex
static std::vector<double> quantity(SubjectWriter& sub, const std::string& name, int q) {
  const StoredStatistic st = read_statistic(sub, name);
  std::vector<double> out;
  const std::int64_t nq = st.shape[1];
  for (std::int64_t v = 0; v < st.shape[0]; ++v) out.push_back(st.values[static_cast<std::size_t>(v * nq + q)]);
  return out;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? NAN : v[v.size() / 2]; }
// the correlation and the regression slope of y on x
static std::pair<double, double> fit(const std::vector<double>& x, const std::vector<double>& y) {
  double mx = 0, my = 0;
  for (std::size_t i = 0; i < x.size(); ++i) { mx += x[i]; my += y[i]; }
  mx /= x.size(); my /= y.size();
  double sxy = 0, sxx = 0, syy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); syy += (y[i] - my) * (y[i] - my); }
  return {sxy / std::sqrt(sxx * syy), sxy / sxx};
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_phasegradient_band_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(3, false, 0.0, Vector3d(1, 1, 1));
  const int N = b.n, M = N, T = 2000, m = 2;
  const double sfreq = 250.0, omega = 2 * M_PI * 10.0, k = 3.0;
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
  std::vector<float> plane(static_cast<size_t>(M * T)), rotate(static_cast<size_t>(M * T)), stand(static_cast<size_t>(M * T)), patchy(static_cast<size_t>(M * T));
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double w = omega * t / sfreq, kz = k * z[static_cast<size_t>(v)], l = m * lam[static_cast<size_t>(v)];
      plane[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w - kz) + 0.02 * nd(rng));
      rotate[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w - l) + 0.02 * nd(rng));
      stand[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w) * std::cos(kz) + 0.02 * nd(rng));
      // the plane wave with an amplitude that falls to near zero along lines (real data's power varies widely): H/P is spiky there
      const double amp = 0.02 + std::abs(std::cos(2.5 * b.P[static_cast<size_t>(3 * v)]));
      patchy[static_cast<size_t>(v * T + t)] = static_cast<float>(amp * std::cos(w - kz) + 0.02 * nd(rng));
    }
  std::vector<float> kf(static_cast<size_t>(3 * N * M), 0.0f);
  for (int v = 0; v < N; ++v) for (int c = 0; c < 3; ++c) kf[static_cast<size_t>((3 * v + c) * M + v)] = static_cast<float>(b.N(v, c));
  DatasetWriter ds = create_dataset(root, "demo");
  {   // the dataset's default subject: the spatial bands (the Eigenvalue line) are made in it
    SubjectWriter def = SubjectWriter::create(ds, "default_template", [] { SubjectOptions o; o.kind = "template"; return o; }());
    def.compose([&] { nxr::bst::ensure_frequency(def); });   // with the Frequency manifold: its 'alpha' band makes the joint (frequency x spatial) key
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
      for (const auto& [name, data] : {std::make_pair(std::string("plane"), &plane), std::make_pair(std::string("rotate"), &rotate), std::make_pair(std::string("stand"), &stand), std::make_pair(std::string("patchy"), &patchy)}) {
        FieldSpec rec;
        rec.name = name; rec.path = tk::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(*data, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
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
  const EigenSolveResult er = eigen_solve(ds, sub, eo);
  CHECK(er.info["glued"].get<bool>());
  for (const char* r : {"plane", "rotate", "stand", "patchy"}) {
    CrossSpectrumOptions o;
    o.recording = r; o.chunks = "all"; o.channels_of = "free"; o.bands = {{"alpha", 8, 12}};
    crossspectrum_compute(ds, sub, o);
  }
  PhaseGradientOptions po;
  po.cross_spectrum = "plane_crossspectrum_alpha"; po.signal = "ambient"; po.spatial_band = "modes";
  // refusals: the normal signal, the weighted fit (not built), an unknown band
  auto refused = [&](PhaseGradientOptions o) { try { phasegradient_compute(ds, sub, o); } catch (const std::invalid_argument&) { return true; } catch (const Error&) { return true; } return false; };
  { PhaseGradientOptions o = po; o.signal = "normal"; CHECK(refused(o)); }
  { PhaseGradientOptions o = po; o.field = "current"; CHECK(refused(o)); }
  { PhaseGradientOptions o = po; o.edge_lag = "normal"; CHECK(refused(o)); }
  { PhaseGradientOptions o = po; o.fit = "lsq"; CHECK(refused(o)); }
  { PhaseGradientOptions o = po; o.fit = "mesh"; o.weight = "amplitude"; CHECK(refused(o)); }
  { PhaseGradientOptions o = po; o.spatial_band = "octave:3"; CHECK(refused(o)); }
  const PhaseGradientResult pr = phasegradient_compute(ds, sub, po);
  CHECK(!pr.hit);
  const double incPlane = pr.info["spatial_bands"][0]["inconsistency"].get<double>();
  PhaseGradientOptions pq = po; pq.cross_spectrum = "rotate_crossspectrum_alpha";
  const double incRot = phasegradient_compute(ds, sub, pq).info["spatial_bands"][0]["inconsistency"].get<double>();
  PhaseGradientOptions ps = po; ps.cross_spectrum = "stand_crossspectrum_alpha";
  phasegradient_compute(ds, sub, ps);
  // the timing map of the plane wave against the truth −k z / ω, on the cortex (the glued surface's first vertices are the kept ones, in support order)
  const auto support = [&] {
    const auto r = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = "hodgeDirac_support", .any_status = true});
    ArrayReader a = ArrayReader::open(sub.store(), r->array_path.value_or(""));
    const auto raw = a.read();
    std::vector<std::int32_t> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
  }();
  const auto timing = quantity(sub, "plane_free_phasegradient_modes_map_alpha", 0);
  CHECK(timing.size() > support.size());
  std::vector<double> truth, got;
  for (std::size_t i = 0; i < support.size(); ++i) {
    truth.push_back(-k * z[static_cast<size_t>(support[i])] / omega);
    got.push_back(timing[i]);
  }
  bool bridgeNan = true;
  for (std::size_t i = support.size(); i < timing.size(); ++i) bridgeNan = bridgeNan && std::isnan(timing[i]);
  const auto [r, slope] = fit(truth, got);
  std::printf("  plane wave (field gradient): timing map against −k z / ω: r %.4f, slope %.3f; inconsistency %.4f\n", r, slope, incPlane);
  std::printf("  rotating wave: inconsistency %.3f | bridge NaN %d\n", incRot, bridgeNan);
  CHECK(r > 0.99);
  CHECK(std::abs(slope - 1) < 0.08);
  CHECK(incPlane < 0.02);
  CHECK(incRot > 0.5);
  // NO PER-VERTEX FIELDS: the fits are one statistic (and the modes' coefficients another), the leaves a byte-less Field with array-backed measurements
  {
    for (const char* old : {"_power_", "_timing_", "_propagation_", "_speed_", "_stream_", "_divergence_", "_curl_", "_inconsistency_", "_modes_"})
      CHECK(!tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = std::string("plane_free_phasegradient_modes") + old + "alpha", .any_status = true}).has_value());
    const StoredStatistic st = read_statistic(sub, "plane_free_phasegradient_modes_map_alpha");
    CHECK(st.shape.size() == 2 && st.shape[1] == 7 && st.shape[0] > static_cast<std::int64_t>(support.size()) && st.axes[1]["values"][0] == "timing" && st.axes[1]["values"][4] == "speed");
    CHECK(st.row.count("of_statistic_id") && !text(st.row, "of_statistic_id").empty());       // of the cross-spectrum statistic
    CHECK(read_statistic(sub, "plane_free_phasegradient_modes_modes_alpha").shape == std::vector<std::int64_t>{read_statistic(sub, "plane_free_phasegradient_modes_modes_alpha").shape[0]});
    const auto fr = tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = "plane_free_phasegradient_modes_alpha", .any_status = true});
    CHECK(fr.has_value() && !fr->path.has_value());
    int rows = 0;
    for (const char* m : {"phase_power", "phase_flux", "phase_circulation"})
      for (const Row& r : nxr::io::testing::Probe(sub).rows("selection_measurement", {{"of_field_id", V(fr->id)}, {"measure", V(std::string(m))}})) rows += !is_null(r, "array_path");
    CHECK_EQ(rows, 3);
  }
  // THE TILES' EXACT INTEGRALS against the ground truth. The plane wave's propagation is k grad z: its divergence on the unit sphere is -2 k z, so a leaf's flux over its area is -2 k times the mean z of its vertices;
  // the rotating wave's phase field grad(-m lambda) turns round each pole: the circulation round a leaf holding a pole is one winding, 2 pi (the edges' lags wrap at pi: the double winding m = 2 is not resolved by a few edges), and ~ 0 for a gradient
  {
    const Json lj = pr.info["spatial_bands"][0];
    const auto part = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = lj["leaves"].get<std::string>(), .any_status = true});
    CHECK(part.has_value());
    ArrayReader ca = ArrayReader::open(sub.store(), part->array_path.value_or(""));
    const auto craw = ca.read();
    std::vector<std::int32_t> leaf(craw.size() / 4);
    std::memcpy(leaf.data(), craw.data(), craw.size());
    const int L = [&] { std::int32_t mx = 0; for (auto c : leaf) mx = std::max(mx, c); return mx + 1; }();
    auto leaves = [&](const std::string& cs, const char* measure) {
      const auto f = tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = cs, .any_status = true});
      const auto m = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"of_field_id", V(f->id)}, {"measure", V(std::string(measure))}});
      return read_field_at(sub, text(m.front(), "array_path"));
    };
    const std::vector<double> flux = leaves("plane_free_phasegradient_modes_alpha", "phase_flux");
    CHECK_EQ(static_cast<int>(flux.size()), L);
    std::vector<double> zbar(static_cast<size_t>(L), 0.0), cnt(static_cast<size_t>(L), 0.0), truth, got;
    for (int v = 0; v < N; ++v) if (leaf[static_cast<size_t>(v)] >= 0) { zbar[static_cast<size_t>(leaf[static_cast<size_t>(v)])] += z[static_cast<size_t>(v)]; cnt[static_cast<size_t>(leaf[static_cast<size_t>(v)])] += 1; }
    // a leaf's area (the lumped areas of its vertices) from the stored `w`
    std::vector<double> area_leaf = flux;                                  // placeholder sizes
    for (int l = 0; l < L; ++l) area_leaf[static_cast<size_t>(l)] = 0.0;
    {
      const SurfaceInput Sc = read_surface(sub, std::string("cortex"), true);
      std::vector<double> va(static_cast<size_t>(N), 0.0);
      for (std::size_t f = 0; f < Sc.faces.size(); f += 3) {
        const Eigen::Vector3d a0(&Sc.vertices[3 * static_cast<size_t>(Sc.faces[f])]), a1(&Sc.vertices[3 * static_cast<size_t>(Sc.faces[f + 1])]), a2(&Sc.vertices[3 * static_cast<size_t>(Sc.faces[f + 2])]);
        const double A = 0.5 * (a1 - a0).cross(a2 - a0).norm();
        for (int c = 0; c < 3; ++c) va[static_cast<size_t>(Sc.faces[f + c])] += A / 3;
      }
      for (int v = 0; v < N; ++v) if (leaf[static_cast<size_t>(v)] >= 0) area_leaf[static_cast<size_t>(leaf[static_cast<size_t>(v)])] += va[static_cast<size_t>(v)];
    }
    for (int l = 0; l < L; ++l) if (cnt[static_cast<size_t>(l)] > 3 && area_leaf[static_cast<size_t>(l)] > 0) { truth.push_back(-2 * k * zbar[static_cast<size_t>(l)] / cnt[static_cast<size_t>(l)]); got.push_back(flux[static_cast<size_t>(l)] / area_leaf[static_cast<size_t>(l)]); }
    const auto [rf, sf] = fit(truth, got);
    std::printf("  plane wave: leaf flux / area against -2 k z (%zu leaves): r %.4f, slope %.3f\n", truth.size(), rf, sf);
    CHECK(rf > 0.95);
    CHECK(std::abs(sf - 1) < 0.25);
    const std::vector<double> gp = leaves("plane_free_phasegradient_modes_alpha", "phase_circulation");
    const std::vector<double> gr = leaves("rotate_free_phasegradient_modes_alpha", "phase_circulation");
    double mp = 0, mr = 0, second = 0;
    for (double x : gp) mp = std::max(mp, std::abs(x));
    std::vector<double> ar;
    for (double x : gr) ar.push_back(std::abs(x));
    std::sort(ar.begin(), ar.end());
    mr = ar.back(); second = ar[ar.size() - 2];
    std::printf("  circulation: plane wave max |Gamma| %.3f; rotating wave: the 6 largest", mp);
    for (std::size_t i = 0; i < 6 && i < ar.size(); ++i) std::printf(" %.3f", ar[ar.size() - 1 - i]);
    std::printf(" (2 pi = %.3f; leaves %d)\n", 2 * M_PI, L);
    CHECK(mp < 0.1 * 2 * M_PI * m);                                   // a gradient field turns nowhere
    CHECK(std::abs(mr - 2 * M_PI) < 0.05 * 2 * M_PI);                 // the leaf holding a pole: one winding (an edge's lag wraps at pi, so a double winding that a few edges cannot resolve counts once)
    (void)second;
  }
  // the WEIGHTED fits (each edge's lag counted by its cross-power or coherence), on the whole complex and within the band's exact modes
  for (const char* fitk : {"mesh", "modes"})
    for (const char* wk : {"power", "coherence"}) {
      PhaseGradientOptions pw2 = po; pw2.fit = fitk; pw2.weight = wk; pw2.name = std::string("plane_") + fitk + "_" + wk;
      const double inc = phasegradient_compute(ds, sub, pw2).info["spatial_bands"][0]["inconsistency"].get<double>();
      PhaseGradientOptions rw2 = pq; rw2.fit = fitk; rw2.weight = wk; rw2.name = std::string("rotate_") + fitk + "_" + wk;
      const double incR = phasegradient_compute(ds, sub, rw2).info["spatial_bands"][0]["inconsistency"].get<double>();
      const auto tw = quantity(sub, *pw2.name + "_modes_map_alpha", 0);
      std::vector<double> g2;
      for (std::size_t i = 0; i < support.size(); ++i) g2.push_back(tw[i]);
      const auto [rw, sw] = fit(truth, g2);
      std::printf("  weighted fit %s / %s: plane wave r %.4f, slope %.3f, inconsistency %.4f | rotating wave inconsistency %.3f\n", fitk, wk, rw, sw, inc, incR);
      CHECK(rw > 0.995);
      CHECK(std::abs(sw - 1) < 0.05);
      CHECK(inc < 0.02);
      CHECK(incR > 0.5);
    }
  // another choice under the same name is refused, then replaced (one byte-less Field, one set of leaf measurements, the statistics moved aside)
  {
    PhaseGradientOptions a = po; a.name = std::string("plane_replace");
    phasegradient_compute(ds, sub, a);
    PhaseGradientOptions b2 = a; b2.fit = "mesh";
    CHECK(refused(b2));
    b2.replace = true;
    const PhaseGradientResult rr = phasegradient_compute(ds, sub, b2);
    CHECK(!rr.hit);
    const auto ts = tk::Subject::of(sub);
    CHECK_EQ(static_cast<int>(ts.find_all({.cls = tk::Cls::field, .name = "plane_replace_modes_alpha", .stored = false, .any_status = true}).size()), 1);
    std::int64_t replaced_rows = 0;
    for (const tk::Node& f : ts.find_all({.cls = tk::Cls::field, .name = "plane_replace_modes_alpha", .any_status = true})) replaced_rows += nxr::io::testing::Probe(sub).count("selection_measurement", {{"of_field_id", V(f.id)}});
    CHECK_EQ(static_cast<int>(replaced_rows), 3);
    CHECK(read_statistic(sub, "plane_replace_modes_map_alpha").producer["fit"] == "mesh");
  }
  // power falling to near zero along lines: the timing map is still the plane wave's
  {
    PhaseGradientOptions px = po; px.cross_spectrum = "patchy_crossspectrum_alpha"; px.name = std::string("patchy");
    const double ii = phasegradient_compute(ds, sub, px).info["spatial_bands"][0]["inconsistency"].get<double>();
    const auto tx = quantity(sub, "patchy_modes_map_alpha", 0);
    std::vector<double> g2;
    for (std::size_t i = 0; i < support.size(); ++i) g2.push_back(tx[i]);
    const auto [rp, sp] = fit(truth, g2);
    std::printf("  patchy plane wave: r %.4f, slope %.3f, inconsistency %.4f\n", rp, sp, ii);
    CHECK(rp > 0.99);
    CHECK(std::abs(sp - 1) < 0.08);
  }
  // Weyl's members: a TIGHT frame (Σ_b g_b² = 1 on every mode), so the bands' powers integrated over the surface add up to the unfiltered (all modes) power
  PhaseGradientOptions pw = po; pw.spatial_band = "weyl:all";
  const PhaseGradientResult wr = phasegradient_compute(ds, sub, pw);
  CHECK(wr.info["spatial_bands"].size() >= 3);
  // the band's power: the array-backed phase_power rows of the band's byte-less Field, summed over the leaves
  auto integral = [&](const std::string& field) {
    const auto f = tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = field, .any_status = true});
    CHECK(f.has_value());
    const auto ms = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"of_field_id", V(f->id)}, {"measure", V(std::string("phase_power"))}});
    CHECK(!ms.empty());
    double s2 = 0;
    for (double x : read_field_at(sub, text(ms.front(), "array_path"))) s2 += x;
    return s2;
  };
  double sumBands = 0;
  for (const auto& bnd : wr.info["spatial_bands"]) sumBands += integral(bnd["name"].get<std::string>() + "_alpha");
  const double whole = integral("plane_free_phasegradient_modes_alpha");
  std::printf("  Weyl members: %zu; Σ of their integrated power / the unfiltered: %.4f\n", wr.info["spatial_bands"].size(), sumBands / whole);
  CHECK(std::abs(sumBands / whole - 1) < 0.05);
  // the spatial band is the measurements' WITHIN key, and the bank is THE ONE `pyramid spatial-bands` stores (one per surface; the stored spans are Weyl's law on the ladder's mean piece area)
  {
    const auto ts = tk::Subject::of(sub);
    const auto banks = ts.find_all({.cls = tk::Cls::selection, .type = "spans", .name_like = "% spatial bands", .any_status = true});
    CHECK(!banks.empty());
    CHECK(banks.size() == 1);
    const auto bank = banks.empty() ? std::nullopt : std::optional<tk::Node>(banks.front());
    const std::int64_t n_bank = static_cast<std::int64_t>(ts.spans(*bank).size());
    int joint_rows = 0;
    for (const auto& bnd : wr.info["spatial_bands"]) {
      const std::int64_t level = std::stoll(bnd["spatial_band"].get<std::string>().substr(4));
      const auto f = ts.find({.cls = tk::Cls::field, .name = bnd["name"].get<std::string>() + "_alpha", .any_status = true});
      CHECK(f.has_value());
      for (const char* measure : {"phase_power", "phase_flux", "phase_circulation"}) {
        const auto mrows = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"of_field_id", V(f->id)}, {"measure", V(std::string(measure))}});
        const std::optional<Row> m = mrows.empty() ? std::nullopt : std::optional<Row>(mrows.front());
        CHECK(m.has_value() && !is_null(*m, "within_selection_id"));
        const WithinKind wk = within_kind(sub.db(), text(*m, "within_selection_id"));
        const std::int64_t code = integer(*m, "within_code");
        if (wk.kind == WithinKind::SpatialBands) CHECK(text(*m, "within_selection_id") == bank->id && code == level);
        else { ++joint_rows; CHECK(wk.kind == WithinKind::BandKey && wk.n_spatial == n_bank); std::int64_t fb2 = 0, sb2 = 0; band_key_split(code, wk.n_spatial, fb2, sb2); CHECK(sb2 == level); }
      }
    }
  }
  CHECK(phasegradient_compute(ds, sub, po).hit);
  fs::remove_all(root);
  return nxrtest::finish("test_phasegradient_band");
}
