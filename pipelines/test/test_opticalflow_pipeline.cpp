// Hodge-Dirac eigenmode power of source maps, the pipeline on a small synthetic dataset (an icosphere, its solved package, a free-orientation kernel on 20 channels):
// the stored fields equal diag(B S B^T) from the cross-spectrum route computed here from the same package, a cache hit, a fixed-orientation kernel refused.
#include <unistd.h>

#include <cmath>
#include <set>
#include <filesystem>
#include <iostream>
#include <random>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/intrinsic_surface.h"
#include "nxr/hodge_weights.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/pyramid.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/welch.h"
#include "nxr/timebank.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;
using Eigen::MatrixXd;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_opticalflow_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;
  std::vector<double> Vx;
  std::vector<std::int32_t> Fx;
  icosphere(1, Vx, Fx, 0.08);                                  // 42 vertices
  const std::int64_t M = 20, N = static_cast<std::int64_t>(Vx.size() / 3), T = 1600;
  const double sfreq = 100.0;
  MatrixXd X(M, T), K(3 * N, M), Kfix(N, M);
  for (auto& v : X.reshaped()) v = static_cast<float>(nd(rng));
  for (int t = 0; t < T; ++t) for (int i = 0; i < M; ++i) X(i, t) += (3.0 + std::sin(2 * M_PI * 0.5 * t / sfreq)) * std::sin(2 * M_PI * 10.0 * t / sfreq + 0.3 * i);
  X = X.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  for (auto& v : K.reshaped()) v = static_cast<float>(nd(rng) * 1e-3);
  K = K.unaryExpr([](double v) { return static_cast<double>(static_cast<float>(v)); });
  for (auto& v : Kfix.reshaped()) v = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = Vx; add_surface(sub, so, Vx, Fx); });   // the icosphere is its own registered sphere
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  EigenSolveOptions eo;
  eo.modes = 16;
  const EigenSolveResult pk = eigen_solve(ds, sub, eo);
  const SurfaceInput surf = read_surface(sub, std::string("cortex"));
  {
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M * 3));
      for (auto& p : pos) p = static_cast<float>(nd(rng) * 0.1);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", std::vector<std::int64_t>(M, 0), {"MEG MAG"}, {}}, {"quality", std::vector<std::int64_t>(M, 0), {"good"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      const auto lc = lattice_counts({T});
      if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = T; d.spacing = 1.0 / sfreq; d.origin = 0.0; d.unit = "s";
      tg.dimensions.push_back(d);
      tm.geometry = tg;
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      std::vector<float> xf(static_cast<std::size_t>(M * T));
      for (int i = 0; i < M; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t));
      FieldSpec rec;
      rec.name = "rest"; rec.path = tk::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
      sub.field(std::move(rec));
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto add_kernel = [&](const std::string& name, const MatrixXd& Km, const char* value_type) {
        std::vector<float> kf(static_cast<std::size_t>(Km.rows() * M));
        for (int i = 0; i < Km.rows(); ++i) for (int j = 0; j < M; ++j) kf[static_cast<std::size_t>(i * M + j)] = static_cast<float>(Km(i, j));
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = Km.rows(); op.n_cols = M;
        op.data = view(kf, {Km.rows(), M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf.id).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", value_type);
        sub.operator_(std::move(op));
      };
      add_kernel("kern", K, "vector3");
      add_kernel("fixed", Kfix, "scalar");
    });
  }
  OpticalflowComputeOptions o;
  o.recording = "rest";
  o.kernel = "kern";
  o.start = 2.0;
  o.stop = 6.0;
  o.family = nxr::timebank::Family::Morlet;            // the 10 Hz wavelet member (cycles 6), gain 2: |c| is the amplitude
  o.gain = 2.0;
  o.freq = 10.0;
  o.centres = {10.0};
  CHECK_EQ(OpticalflowComputeOptions{}.method, std::string("pcg"));      // pcg is the default method
  o.method = "direct";                                  // the reference: Rheome's exact solve
  const OpticalflowComputeResult a = opticalflow_compute(ds, sub, o);
  CHECK(!a.hit);
  const std::int64_t Kn = pk.modes.n_modes();
  CHECK_EQ(a.coefficients.rows(), Kn);
  const std::int64_t steps = a.coefficients.cols();
  CHECK(std::abs(a.info["centre_hz"].get<double>() - 10.0) < 1e-9);
  CHECK(std::abs(a.info["rate"].get<double>() * 4.0 - double(steps + 1)) < 2.0);     // the member's own rate over the 4 s window
  CHECK_EQ(a.info["velocity_frames"].get<std::int64_t>(), steps);
  CHECK(a.coefficients.allFinite());
  CHECK_EQ(a.divergence.rows(), N);
  CHECK(a.speed.minCoeff() >= 0);
  const double ef = a.info["summary"]["exact_fraction"].get<double>();
  std::printf("  exact fraction of the flow's power %.3f, speed mean %.3e m/s\n", ef, a.speed.mean());
  CHECK(ef >= 0 && ef <= 1);
  // stored: the coefficients on eigen axis x the window's time line, three maps on the sources; no per-frame velocity
  const auto ts = tk::Subject::of(sub);
  const auto f = ts.find({.cls = tk::Cls::field, .name = "rest_kern_opticalflow_flow_modes"});
  CHECK(f.has_value());
  if (f) {
    CHECK_EQ(f->kind.value_or(""), std::string("optical flow"));
    ArrayReader r = ArrayReader::open(sub.store(), f->path.value_or(""));
    CHECK_EQ(r.shape()[0], Kn);
    CHECK_EQ(r.shape()[1], steps);
  }
  // no per-vertex time-mean Field is written (step 5): a byte-less Field `<stem>_flow` and the leaf x frame sums on a joint Selection
  for (const char* n : {"rest_kern_opticalflow_divergence", "rest_kern_opticalflow_curl", "rest_kern_opticalflow_speed"})
    CHECK(!ts.find({.cls = tk::Cls::field, .name = n, .any_status = true}).has_value());
  for (const tk::Node& fv : ts.find_all({.cls = tk::Cls::field, .any_status = true})) CHECK(fv.name.find("velocity") == std::string::npos);
  const auto flowf = ts.find({.cls = tk::Cls::field, .name = "rest_kern_opticalflow_flow", .any_status = true});
  CHECK(flowf.has_value() && !flowf->path.has_value() && flowf->kind.value_or("") == "optical flow");
  CHECK_EQ(flowf->producer["studio"]["type"].get<std::string>(), std::string("complex"));   // a tangent velocity: the Studio's complex
  const auto leafparts = ts.find_all({.cls = tk::Cls::selection, .type = "set", .name_like = "cortex tiles L%", .any_status = true});   // by name: the last is the finest
  const std::optional<tk::Node> leafpart = leafparts.empty() ? std::nullopt : std::optional<tk::Node>(leafparts.back());
  CHECK(leafpart.has_value());
  const auto joint = ts.find({.cls = tk::Cls::selection, .type = "set", .name_like = "rest \xc3\x97 rest_kern_opticalflow%", .any_status = true});
  CHECK(joint.has_value());
  // THE LEAF x FRAME SUMS ARE THE VERTEX TIME SUMS: summed over the frames, a leaf's sum is steps x the area-weighted sum of its vertices' time means; the counts add to the steps
  if (flowf && leafpart && joint) {
    const std::int64_t L = leafpart->n_members.value_or(0);
    nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(N); mesh.positions = surf.vertices; mesh.faces = surf.faces;
    const std::vector<double> area = nxr::ladder::vertexAreas(mesh);
    std::vector<std::int32_t> codes;
    { ArrayReader r = ArrayReader::open(sub.store(), leafpart->array_path.value_or("")); const auto raw = r.read(); codes.resize(raw.size() / 4); std::memcpy(codes.data(), raw.data(), raw.size()); }
    const int D = leafpart->params["atlas"]["level"].get<int>();
    double worst = 0;
    const Eigen::MatrixXd* maps[3] = {&a.divergence, &a.curl, &a.speed};
    const char* names[3] = {"flow_divergence", "flow_curl", "flow_speed"};
    for (int m = 0; m < 3; ++m) {
      std::vector<double> want(static_cast<size_t>(L), 0.0), got(static_cast<size_t>(L), 0.0);
      for (std::int64_t v = 0; v < N; ++v) if (codes[static_cast<size_t>(v)] >= 0) want[static_cast<size_t>(codes[static_cast<size_t>(v)])] += area[static_cast<size_t>(v)] * (*maps[m])(v, 0) * double(steps);
      for (const Json& r : query_measurements(sub.db(), sub.store(), Json{{"selectionId", joint->id}, {"measure", names[m]}, {"ofFieldId", flowf->id}, {"spaceLevel", D}})) got[r["code"].get<size_t>()] += r["value"].get<double>();
      for (std::int64_t l = 0; l < L; ++l) worst = std::max(worst, std::abs(got[static_cast<size_t>(l)] - want[static_cast<size_t>(l)]) / (std::abs(want[static_cast<size_t>(l)]) + 1e-9 * (1 + std::abs(want[static_cast<size_t>(l)]))));
    }
    std::printf("  leaf x frame sums over the frames vs steps x the vertex time means (div, curl, speed), %lld leaves: worst relative difference %.2e\n", static_cast<long long>(L), worst);
    CHECK(worst < 1e-5);                                     // the vertex means are float64 here; the sums add the same numbers in another order
    // the density is the time mean per unit area: a leaf's speed density times its area is its sum over the steps divided by the steps
    double tot_steps = 0;
    for (const Json& r : query_measurements(sub.db(), sub.store(), Json{{"selectionId", joint->id}, {"measure", "flow_speed_density"}, {"ofFieldId", flowf->id}, {"spaceLevel", -1}, {"timeLevel", 40}})) tot_steps += r["weight"].get<double>() > 0 ? 1.0 : 0.0;
    CHECK(tot_steps >= 1);
    // the rolled-up mean speed (the whole cortex, the whole window) is the area-weighted mean of the vertex mean speeds
    double a_sum = 0, s_sum = 0;
    for (std::int64_t v = 0; v < N; ++v) if (codes[static_cast<size_t>(v)] >= 0) { a_sum += area[static_cast<size_t>(v)]; s_sum += area[static_cast<size_t>(v)] * a.speed(v, 0); }
    const auto whole = query_measurements(sub.db(), sub.store(), Json{{"selectionId", joint->id}, {"measure", "flow_speed_density"}, {"ofFieldId", flowf->id}, {"spaceLevel", -1}, {"timeLevel", 40}});
    CHECK(whole.size() >= 1);
    if (whole.size() >= 1) {
      double num = 0, den = 0;
      for (const Json& r : whole) { num += r["value"].get<double>() * r["weight"].get<double>(); den += r["weight"].get<double>(); }
      std::printf("  whole-cortex mean speed from the sums %.6e, from the vertex means %.6e\n", num / den, s_sum / a_sum);
      CHECK(std::abs(num / den - s_sum / a_sum) < 1e-4 * std::abs(s_sum / a_sum));
    }
  }
  CHECK(opticalflow_compute(ds, sub, o).hit);
  // a fixed-orientation kernel works too (the amplitude is |K x|)
  OpticalflowComputeOptions fx = o;
  fx.kernel = "fixed";
  CHECK(!opticalflow_compute(ds, sub, fx).hit);
  {   // --tiles: the maps' tile summaries per level: the stored scalars, and (computed in this run) the time-mean velocity in the Levi-Civita form, in the signpost gauge as this surface has no canonical frames
    OpticalflowComputeOptions tf = o;
    tf.name = "tiled"; tf.tiles = true; tf.tile_max_level = 2;
    const OpticalflowComputeResult t = opticalflow_compute(ds, sub, tf);
    CHECK(!t.hit && t.info.contains("tiles") && t.info["tiles"][0]["levels"].size() == 2);
    CHECK(t.info["tiles"][0]["gauge"].get<std::string>() == "signpost gauge" && t.info["tiles"][0]["scalars"].size() == 0 && t.info["tiles"][0]["vectors"].size() == 1);
    CHECK(tk::Subject::of(sub).find({.cls = tk::Cls::selection, .type = "ladder", .any_status = true}).has_value());
    tf.replace = true;                                    // a cache hit: no time-mean map is stored, so none is summarised
    const OpticalflowComputeResult h = opticalflow_compute(ds, sub, tf);
    CHECK(h.hit && h.info["tiles"][0]["scalars"].size() == 0 && h.info["tiles"][0]["vectors"].size() == 0);
  }
  // the alternative methods (nxr/opticalflow_methods.h): same fields, timing in the result; pcg equals direct to its tolerance
  {
    OpticalflowComputeOptions pc = o;
    pc.method = "pcg";
    pc.name = "rest_pcg";
    const OpticalflowComputeResult b = opticalflow_compute(ds, sub, pc);
    CHECK(!b.hit);
    CHECK_EQ(b.coefficients.cols(), steps);
    const double d = (b.coefficients - a.coefficients).norm() / a.coefficients.norm();
    std::printf("  pcg vs direct: relative difference of the flow weights %.2e; %.4f s/frame (direct %.4f)\n", d, b.seconds_per_frame, a.seconds_per_frame);
    CHECK(d < 1e-5);
    CHECK(b.info["summary"].contains("iterations_per_frame_mean"));
    CHECK(opticalflow_compute(ds, sub, pc).hit);
    OpticalflowComputeOptions ps = o;
    ps.method = "perscale";
    ps.name = "rest_scales";
    ps.scale_family = "logitersine";
    ps.scale_voices = 1;
    const OpticalflowComputeResult e = opticalflow_compute(ds, sub, ps);
    CHECK(e.series_coefficients.size() >= 1);
    CHECK_EQ(e.series_coefficients.size(), e.info["summary"]["series"].size());
    CHECK(tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = "rest_scales_m00_flow", .any_status = true}).has_value());
    std::printf("  perscale: %zu members\n", e.series_coefficients.size());
    // each member's fields carry that member's own summary (name, centre wavenumber, powers, mean speed), not member 0's
    {
      const Json& ser = e.info["summary"]["series"];
      CHECK(!e.info["summary"].contains("exact_fraction"));          // the run's summary has no member's numbers of its own
      std::set<double> centres, speeds;
      for (std::size_t si = 0; si < ser.size(); ++si) {
        char nm[40];
        std::snprintf(nm, sizeof nm, "rest_scales_m%02zu_flow", si);
        const auto fr = tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = nm, .any_status = true});
        CHECK(fr.has_value());
        if (!fr) continue;
        const Json own = fr->producer["summary"];
        char tag[8];
        std::snprintf(tag, sizeof tag, "m%02zu", si);
        CHECK_EQ(own["name"].get<std::string>(), std::string(tag));
        CHECK(own["centre_wavenumber"].get<double>() == ser[si]["centre_wavenumber"].get<double>());
        CHECK(own["mean_speed"].get<double>() == ser[si]["mean_speed"].get<double>());
        CHECK(own["exact_fraction"].get<double>() == ser[si]["exact_fraction"].get<double>());
        CHECK_EQ(own["series"].size(), ser.size());
        centres.insert(own["centre_wavenumber"].get<double>());
        speeds.insert(own["mean_speed"].get<double>());
      }
      std::printf("  perscale: %zu distinct centre wavenumbers, %zu distinct mean speeds in the members' own summaries\n", centres.size(), speeds.size());
      CHECK_EQ(centres.size(), ser.size());
      CHECK(speeds.size() > 1 || ser.size() == 1);
    }
    OpticalflowComputeOptions bad = o;
    bad.method = "perscale";
    bool threw = false;
    try { opticalflow_compute(ds, sub, bad); } catch (const std::exception&) { threw = true; }
    CHECK(threw);                                          // the family is required
  }
  // with the surface's frames stored (frames compute), the time-mean velocity is also reported along anterior and the second anatomical axis
  {
    FramesComputeOptions fo;
    fo.side = "left";                                      // one piece
    fo.min_projection = 0.2;
    frames_compute(ds, sub, fo);
    OpticalflowComputeOptions an = o;
    an.method = "pcg";
    an.name = "rest_anat";
    const OpticalflowComputeResult r = opticalflow_compute(ds, sub, an);
    CHECK(!r.hit);
    CHECK_EQ(r.anterior.rows(), N);
    CHECK_EQ(r.second.rows(), N);
    int finite = 0, bad = 0;
    for (std::int64_t v = 0; v < N; ++v) {
      if (!std::isfinite(r.anterior(v, 0))) continue;
      ++finite;
      // the mean velocity is no longer than the mean speed
      if (std::hypot(r.anterior(v, 0), r.second(v, 0)) > r.speed(v, 0) * (1 + 1e-9) + 1e-18) ++bad;
    }
    std::printf("  anatomical components: %d of %lld vertices reliable, %d longer than the mean speed\n", finite, (long long)N, bad);
    CHECK(finite > N / 2 && bad == 0);
    for (const char* n : {"rest_anat_velocity_anterior", "rest_anat_velocity_second"}) CHECK(!tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = n, .any_status = true}).has_value());
    {   // the canonical-frame and anatomical velocity sums are stored as leaf x frame measures (the frames exist now)
      const auto jf = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .type = "set", .name_like = "rest \xc3\x97 rest_anat%", .any_status = true});
      CHECK(jf.has_value());
      if (jf) for (const char* m : {"flow_velocity_1", "flow_velocity_2", "flow_velocity_anterior", "flow_velocity_second"})
        CHECK([&] { for (const Row& r : nxr::io::testing::Probe(sub).rows("selection_measurement", {{"selection_id", V(jf->id)}, {"measure", V(std::string(m))}})) if (!is_null(r, "array_path")) return true; return false; }());
    }
    CHECK(opticalflow_compute(ds, sub, an).hit);
  }
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_opticalflow_pipeline");
}
