// phasegradient compute against GROUND TRUTH, stored on the measurement pyramid (spec docs/specs/2026-11-14-measurement-pyramid.md, step 5): two spherical hemispheres (unit radius), one channel a vertex and a free
// kernel that gives each source its channel along the outward normal (so the normal current at v IS channel v). Two recordings at 10 Hz: a wave TRAVELLING round each sphere's z axis, cos(omega t - m lambda)
// (m = 2), and a STANDING pattern cos(omega t) cos(m lambda). Through the whole chain (crossspectrum compute, the stored cross-spectrum read on the kernel's channels, the P1 gradient) the run stores NO per-vertex
// Field: a byte-less Field and, per leaf of the ladder's finest tiles, ARRAY-BACKED sums (phase power, gradient power, the phase current in the canonical frames, its magnitude); the gradient, speed and travelling index
// are DERIVED AT READ by the lookup (a ratio of the sums). Checked: the leaf sums equal the vertex-area-weighted sums of the per-vertex ingredients computed directly (1e-9), the derived gradient is the
// power-weighted mean of the vertices' gradients, a roll-up to a coarser level equals a run on those merged leaves, the speed and travelling index against the ground truth (travelling ~1, standing ~0), the ambient
// signal, a re-run is a cache hit, the old Fields are not written, a missing cross-spectrum and a surface with no canonical frames are refused.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/bundle.h"
#include "nxr/io/measurement_query.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/bst/entities.h"
#include "nxr/operators/hodge_synthesis_weights.h"
#include "nxr/io/measures.h"
#include "nxr/select/dyadic.h"
#include "nxr/signal/phase_gradient.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "run_definition.h"
#include "make_grids.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

template <class T> static std::vector<T> read_field_at(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader a = ArrayReader::open(sub.store(), path);
  if (shape) *shape = a.shape();
  const auto raw = a.read();
  std::vector<T> out(raw.size() / sizeof(T));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? NAN : v[v.size() / 2]; }

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_phasegradient_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(3, false, 0.0, Vector3d(1, 1, 1));
  const int N = b.n, M = N, T = 2000, m = 2;
  const double sfreq = 250.0, omega = 2 * M_PI * 10.0;
  std::vector<double> lam(static_cast<size_t>(N)), rho(static_cast<size_t>(N)), cy(static_cast<size_t>(N));
  for (int v = 0; v < N; ++v) {
    cy[static_cast<size_t>(v)] = v < b.nHalf ? 1.3 : -1.3;
    const double x = b.P[static_cast<size_t>(3 * v)], y = b.P[static_cast<size_t>(3 * v + 1)] - cy[static_cast<size_t>(v)];
    lam[static_cast<size_t>(v)] = std::atan2(y, x);
    rho[static_cast<size_t>(v)] = std::hypot(x, y);
  }
  std::mt19937 rng(7);
  std::normal_distribution<double> nd;
  std::vector<float> travel(static_cast<size_t>(M * T)), stand(static_cast<size_t>(M * T));
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double w = omega * t / sfreq, l = m * lam[static_cast<size_t>(v)];
      travel[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w - l) + 0.02 * nd(rng));
      stand[static_cast<size_t>(v * T + t)] = static_cast<float>(std::cos(w) * std::cos(l) + 0.02 * nd(rng));
    }
  std::vector<float> kf(static_cast<size_t>(3 * N * M), 0.0f);
  for (int v = 0; v < N; ++v) for (int c = 0; c < 3; ++c) kf[static_cast<size_t>((3 * v + c) * M + v)] = static_cast<float>(b.N(v, c));
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
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
      for (const auto& [name, data] : {std::make_pair(std::string("travel"), &travel), std::make_pair(std::string("stand"), &stand)}) {
        FieldSpec rec;
        rec.name = name; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
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
  // `phasegradient` is the shipped definition (the hand-wired verb is gone): its parameters as a small struct, the run through the runner
  struct PhaseGradientOptions { std::string cross_spectrum, signal = "normal", name; std::optional<int> depth; };   // depth: the grid level, stated to the definition as `finest: parts 2^depth`
  struct PhaseGradientResult { bool hit = false; };
  auto phasegradient_compute = [&](DatasetWriter& d, SubjectWriter& s, const PhaseGradientOptions& o) {
    // the cross-spectrum is one statistic, band x channel x channel: its old per-band name `<stem>_<band>` is the statistic `<stem>` and the band
    std::string cs = o.cross_spectrum;
    Json set = Json{{"signal", o.signal}, {"kernel", "free"}};
    if (cs.size() > 6 && cs.compare(cs.size() - 6, 6, "_alpha") == 0) { cs.resize(cs.size() - 6); set["band"] = "alpha"; }
    set["cross_spectrum"] = cs;
    set["name"] = !o.name.empty() ? o.name + "_alpha" : cs.substr(0, cs.size() - std::string("_crossspectrum").size()) + "_free_phasegradient_alpha";
    if (o.depth) set["finest"] = "parts " + std::to_string(1 << *o.depth);
    const auto runs = nxrtest::run_definition(d, s, "phasegradient", set);
    PhaseGradientResult r;
    r.hit = !runs.empty() && runs.front().value("hit", false);
    return r;
  };
  PhaseGradientOptions po;
  po.cross_spectrum = "travel_crossspectrum_alpha";
  auto code_of = [&](auto f) { try { f(); } catch (const Error& e) { return e.code(); } return std::string(); };
  CHECK(!code_of([&] { phasegradient_compute(ds, sub, po); }).empty());            // no such cross-spectrum
  for (const char* r : {"travel", "stand"}) {
    nxrtest::run_definition(ds, sub, "crossspectrum", Json{{"recording", r}, {"channels_of", "free"}, {"bands", Json::array({Json{{"name", "alpha"}, {"lo", 8}, {"hi", 12}}})}});
  }
  CHECK(code_of([&] { phasegradient_compute(ds, sub, po); }) == "invalid_argument");   // no canonical frames yet
  nxrtest::run_definition(ds, sub, "surface_frames", Json{{"surface", "cortex"}});
  nxrtest::make_tiles(ds, sub, "", std::string("cortex"));
  const PhaseGradientResult pr = phasegradient_compute(ds, sub, po);
  CHECK(!pr.hit);
  PhaseGradientOptions ps = po; ps.cross_spectrum = "stand_crossspectrum_alpha";
  phasegradient_compute(ds, sub, ps);
  PhaseGradientOptions pa = po; pa.signal = "ambient"; pa.name = "travel_free_ambient";
  phasegradient_compute(ds, sub, pa);
  CHECK(phasegradient_compute(ds, sub, po).hit);
  // nothing per vertex is stored: the old Fields are gone, one byte-less Field of kind `phase gradient` carries the rows
  {
    bool any = false;
    for (const char* gone : {"travel_free_phasegradient_velocity_alpha", "travel_free_phasegradient_speed_alpha", "travel_free_phasegradient_travelling_alpha", "travel_free_phasegradient_power_alpha"})
      any = any || nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = gone, .any_status = true}).has_value();
    CHECK(!any);
  }
  const auto fld = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = "travel_free_phasegradient_alpha", .kind = "phase gradient", .any_status = true});
  {   // no array of its own (its values are its measurements, stored beside it under its path)
    bool own_array = false;
    if (fld && fld->path) { try { ArrayReader::open(sub.store(), *fld->path); own_array = true; } catch (const std::exception&) {} }
    CHECK(fld && !own_array);
  }
  const std::string fid = fld->id;
  const Json fpj = fld->producer;
  CHECK_EQ(fpj["studio"]["type"].get<std::string>(), std::string("complex"));   // a tangent vector a vertex: the Studio reads the vector3 as complex
  CHECK(fpj["leaves"]["count"].get<int>() > 8);   // (the producer no longer records a time axis "none": a band has no frames in it)
  const nxr::io::testing::Probe probe(sub);
  const auto first_measured = probe.rows("selection_measurement", {{"of_field_id", V(fid)}});
  const auto part = first_measured.empty() ? std::optional<nxr::io::task::Node>() : nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::selection, text(first_measured.front(), "selection_id"));
  // the measurements are over the grid (the ladder `cortex tiles`) at ONE level, stored as an array [rows x members]: its members are that level's tiles (slots << level)
  CHECK(part && part->type == "ladder");
  const std::int64_t slots = part->params["grid"].value("slots", std::int64_t(1)), depth = part->params["grid"]["depth"].get<std::int64_t>();
  const nxr::io::MeasurementValues stored_ph = nxr::io::query::rolled_values(sub, part->id, fid, std::nullopt, std::nullopt, "phase_power", nxr::io::query::RollUp{});
  const int L = static_cast<int>(stored_ph.members);
  int D = 0;
  while ((slots << D) < L) ++D;
  CHECK((slots << D) == L);
  const auto kinds = probe.rows("selection_measurement", {{"of_field_id", V(fid)}}, "measure");
  CHECK(kinds.size() == 5);

  // THE GROUND TRUTH AND THE SUMS, per vertex, computed directly (the chain's own pure functions on the stored cross-spectrum)
  const SurfaceInput S = read_surface(sub, std::string("cortex"), true);
  const nxr::operators::RowMat3 Pm = Eigen::Map<const nxr::operators::RowMat3>(S.vertices.data(), N, 3);
  const nxr::operators::FaceMat Fm = Eigen::Map<const nxr::operators::FaceMat>(S.faces.data(), S.n_faces(), 3);
  const nxr::operators::FaceOperators ops(Pm, Fm, false);
  const auto kern = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::operator_, .name = "free", .any_status = true});
  std::vector<std::int64_t> kshape;
  const std::vector<float> kdata = read_field_at<float>(sub, *kern->path, &kshape);
  const Eigen::MatrixXd K = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(kdata.data(), kshape[0], kshape[1]).cast<double>();
  std::vector<std::int32_t> all_chans(static_cast<size_t>(M));
  for (int i = 0; i < M; ++i) all_chans[static_cast<size_t>(i)] = i;
  auto vertex_sums = [&](const char* cs, bool ambient) {
    const StoredCrossSpectrum C = crossspectrum_read(sub, cs, all_chans);
    nxr::signal::PhaseSums sums;
    auto add = [&](const Eigen::MatrixXd& A) {
      Eigen::MatrixXd G[3];
      ops.meanFaceGradient(A, G);
      for (int v = 0; v < N; ++v) { const Eigen::RowVectorXd dot = G[0].row(v) * ops.normals(v, 0) + G[1].row(v) * ops.normals(v, 1) + G[2].row(v) * ops.normals(v, 2); for (int d = 0; d < 3; ++d) G[d].row(v) -= dot * ops.normals(v, d); }
      nxr::signal::addPhaseSums(sums, nxr::signal::phaseSums(A, G, C.S, C.Sq));
    };
    if (!ambient) { Eigen::MatrixXd A(N, M); for (int v = 0; v < N; ++v) A.row(v) = ops.normals(v, 0) * K.row(3 * v) + ops.normals(v, 1) * K.row(3 * v + 1) + ops.normals(v, 2) * K.row(3 * v + 2); add(A); }
    else for (int c = 0; c < 3; ++c) { Eigen::MatrixXd A(N, M); for (int v = 0; v < N; ++v) A.row(v) = K.row(3 * v + c); add(A); }
    return sums;
  };
  const nxr::signal::PhaseSums sv = vertex_sums("travel_crossspectrum_alpha", false);
  const StoredFrames sf = read_stored_frames(sub, "cortex", N);
  const auto jf = nxr::signal::phaseCurrentInFrames(sv.phase, sf.e1, sf.e2);
  nxr::select::Mesh mesh; mesh.n = N; mesh.positions = S.vertices; mesh.faces = S.faces;
  const std::vector<double> area = nxr::select::vertexAreas(mesh);
  // the tile of every vertex at level D: the finest level's member of the vertex, shifted
  std::vector<std::int32_t> codes;
  {
    const auto levels = nxr::io::task::Subject::of(sub).members(*part);
    CHECK(static_cast<std::int64_t>(levels.size()) == depth + 1);
    for (const std::int32_t c : nxr::io::task::Subject::of(sub).element_members(levels.back().id).member_of) codes.push_back(c < 0 ? c : static_cast<std::int32_t>(c >> (depth - D)));
  }
  auto ask = [&](const char* measure, int level, const std::string& field) {
    nxr::io::query::RollUp roll;                                           // coarser by D - level (the closed-form hierarchy); the whole for a negative level
    if (level < 0) roll.whole = true; else roll.shift = D - level;
    return nxr::io::query::rolled_values(sub, part->id, field, std::nullopt, std::nullopt, measure, roll).values;
  };
  // (1) the leaf sums are the vertex sums, weighted by vertex area, grouped by leaf
  double worst = 0;
  {
    std::vector<double> w_p(static_cast<size_t>(L)), w_q(static_cast<size_t>(L)), w_1(static_cast<size_t>(L)), w_2(static_cast<size_t>(L)), w_m(static_cast<size_t>(L));
    for (int v = 0; v < N; ++v) {
      if (codes[static_cast<size_t>(v)] < 0 || !std::isfinite(jf.j1(v))) continue;
      const size_t l = static_cast<size_t>(codes[static_cast<size_t>(v)]);
      w_p[l] += area[static_cast<size_t>(v)] * sv.power(v); w_q[l] += area[static_cast<size_t>(v)] * sv.gradientPower(v); w_1[l] += area[static_cast<size_t>(v)] * jf.j1(v); w_2[l] += area[static_cast<size_t>(v)] * jf.j2(v); w_m[l] += area[static_cast<size_t>(v)] * jf.magnitude(v);
    }
    const auto g_p = ask("phase_power", D, fid), g_q = ask("phase_gradient_power", D, fid), g_1 = ask("phase_current_1", D, fid), g_2 = ask("phase_current_2", D, fid), g_m = ask("phase_current_magnitude", D, fid);
    for (int l = 0; l < L; ++l) {
      auto rel = [&](double a, double b) { return std::abs(a - b) / (1e-300 + std::abs(b) + 1e-12); };
      if (!std::isfinite(g_p[static_cast<size_t>(l)])) continue;
      worst = std::max({worst, rel(g_p[static_cast<size_t>(l)], w_p[static_cast<size_t>(l)]), rel(g_q[static_cast<size_t>(l)], w_q[static_cast<size_t>(l)] / 1.0), rel(g_1[static_cast<size_t>(l)], w_1[static_cast<size_t>(l)]), rel(g_2[static_cast<size_t>(l)], w_2[static_cast<size_t>(l)]), rel(g_m[static_cast<size_t>(l)], w_m[static_cast<size_t>(l)])});
    }
    std::printf("  leaf sums vs the vertex sums, %d leaves: worst relative difference %.2e\n", L, worst);
    CHECK(worst < 1e-9);
  }
  // (2) the derived gradient at leaf level: |sum j| / P is the power-weighted mean of the vertices' phase gradients (weights a P); it equals the area-weighted mean of H/P only where P is uniform
  {
    const auto gm = ask("phase_gradient_magnitude", D, fid);
    std::vector<double> sj1(static_cast<size_t>(L)), sj2(static_cast<size_t>(L)), sp(static_cast<size_t>(L)), mean_g1(static_cast<size_t>(L)), mean_g2(static_cast<size_t>(L)), sa(static_cast<size_t>(L));
    for (int v = 0; v < N; ++v) {
      if (codes[static_cast<size_t>(v)] < 0 || !std::isfinite(jf.j1(v)) || !(sv.power(v) > 0)) continue;
      const size_t l = static_cast<size_t>(codes[static_cast<size_t>(v)]);
      const double a = area[static_cast<size_t>(v)], P = sv.power(v);
      sj1[l] += a * jf.j1(v); sj2[l] += a * jf.j2(v); sp[l] += a * P;
      mean_g1[l] += a * jf.j1(v) / P; mean_g2[l] += a * jf.j2(v) / P; sa[l] += a;
    }
    double d_pw = 0, d_area = 0;
    int n = 0;
    std::vector<double> rel_area;
    for (int l = 0; l < L; ++l) {
      if (!std::isfinite(gm[static_cast<size_t>(l)]) || sp[static_cast<size_t>(l)] <= 0) continue;
      const double direct_pw = std::hypot(sj1[static_cast<size_t>(l)], sj2[static_cast<size_t>(l)]) / sp[static_cast<size_t>(l)];
      d_pw = std::max(d_pw, std::abs(gm[static_cast<size_t>(l)] - direct_pw) / direct_pw);
      const double direct_area = std::hypot(mean_g1[static_cast<size_t>(l)], mean_g2[static_cast<size_t>(l)]) / sa[static_cast<size_t>(l)];
      rel_area.push_back(std::abs(gm[static_cast<size_t>(l)] - direct_area) / direct_area);
      ++n;
    }
    d_area = median(rel_area);
    std::printf("  derived gradient, %d leaves: vs the power-weighted mean of the vertex gradients max %.2e; vs the area-weighted mean of H/P (P not uniform) median %.3f\n", n, d_pw, d_area);
    CHECK(n > 8 && d_pw < 1e-9);
    CHECK(d_area < 0.1);
  }
  // (3) a roll-up to a coarser level equals a run on those merged leaves (the sums add exactly; ratios from the rolled-up sums)
  {
    const int k = D - 1;
    PhaseGradientOptions pk = po; pk.depth = k; pk.name = "travel_free_k";
    phasegradient_compute(ds, sub, pk);   // (finest "parts 2^k": a run at the coarser level)
    const auto fk = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = "travel_free_k_alpha", .any_status = true});
    const auto partk = part;   // the same grid: the coarser run's members are that level's tiles
    const nxr::io::MeasurementValues stored_k = nxr::io::query::rolled_values(sub, partk->id, fk->id, std::nullopt, std::nullopt, "phase_power", nxr::io::query::RollUp{});
    CHECK(stored_k.members == (L >> (D - k)));
    double wr = 0, wd = 0;
    for (const char* m : {"phase_power", "phase_gradient_power", "phase_current_1", "phase_current_2", "phase_current_magnitude", "phase_gradient_magnitude", "travelling_index", "phase_coherence"}) {
      const auto rolled = ask(m, k, fid);
      const std::vector<double> direct = nxr::io::query::rolled_values(sub, partk->id, fk->id, std::nullopt, std::nullopt, m, nxr::io::query::RollUp{}).values;
      for (size_t i = 0; i < rolled.size(); ++i) {
        if (!std::isfinite(rolled[i]) && !std::isfinite(direct[i])) continue;
        wr = std::max(wr, std::abs(rolled[i] - direct[i]) / (std::abs(direct[i]) + 1e-12));
      }
    }
    wd = wr;
    std::printf("  roll-up D -> %d vs a run on the merged leaves: worst relative difference %.2e\n", k, wd);
    CHECK(wd < 1e-9);
  }
  // (4) the ground truth: through the whole chain the leaf speed and travelling index; the standing pattern's index ~ 0; the ambient signal agrees
  {
    const auto trav = ask("travelling_index", D, fid), gm = ask("phase_gradient_magnitude", D, fid);
    const auto fs = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = "stand_free_phasegradient_alpha", .any_status = true});
    const auto fa = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = "travel_free_ambient_alpha", .any_status = true});
    const auto trs = ask("travelling_index", D, fs->id), gma = ask("phase_gradient_magnitude", D, fa->id);
    std::vector<double> err, tr, st, amb;
    for (int l = 0; l < L; ++l) {
      double rl = 0, aa = 0;
      for (int v = 0; v < N; ++v) if (codes[static_cast<size_t>(v)] == l) { rl += area[static_cast<size_t>(v)] * rho[static_cast<size_t>(v)]; aa += area[static_cast<size_t>(v)]; }
      if (!(aa > 0) || rl / aa < 0.5 || !std::isfinite(gm[static_cast<size_t>(l)])) continue;
      const double expect = omega * (rl / aa) / m, speed = omega / gm[static_cast<size_t>(l)];
      err.push_back(std::abs(speed - expect) / expect);
      tr.push_back(trav[static_cast<size_t>(l)]);
      if (std::isfinite(trs[static_cast<size_t>(l)])) st.push_back(trs[static_cast<size_t>(l)]);
      if (std::isfinite(gma[static_cast<size_t>(l)])) amb.push_back(std::abs(gma[static_cast<size_t>(l)] - gm[static_cast<size_t>(l)]) / gm[static_cast<size_t>(l)]);
    }
    std::printf("  travelling wave, %zu leaves off the poles: speed error median %.3f, max %.3f; travelling index median %.3f; standing pattern index median %.3f; ambient vs normal: median %.3f\n", err.size(), median(err),
                *std::max_element(err.begin(), err.end()), median(tr), median(st), median(amb));
    CHECK(err.size() > 8 && median(err) < 0.05 && median(tr) > 0.95 && median(st) < 0.1 && median(amb) < 0.05);
    // the vertices' own direction (the stored sums are these, grouped): the velocity along e_lambda
    nxr::signal::PhaseReadout ro = nxr::signal::phaseReadout(sv, omega);
    std::vector<double> cosang;
    for (int v = 0; v < N; ++v) {
      if (rho[static_cast<size_t>(v)] < 0.5 || !std::isfinite(ro.velocity(v, 0))) continue;
      const double x = b.P[static_cast<size_t>(3 * v)], y = b.P[static_cast<size_t>(3 * v + 1)] - cy[static_cast<size_t>(v)], r = rho[static_cast<size_t>(v)];
      cosang.push_back(Eigen::Vector3d(ro.velocity.row(v)).normalized().dot(Eigen::Vector3d(-y / r, x / r, 0)));
    }
    CHECK(!cosang.empty() && *std::min_element(cosang.begin(), cosang.end()) > 0.98);
  }
  fs::remove_all(root);
  return nxrtest::finish("test_phasegradient");
}
