// apply_kernel with SEVERAL hat levels (`levels`,, spatial-events piece 3): one operator a level, all bound at preparation, the output [vertex × level] (row = (vertex · levels + level) · components +
// component), on a synthetic two-hemisphere source space (never a real dataset): a free kernel of random currents, a recording of a 10 Hz oscillation plus noise.
//   - with a FORM (curl_tan): the band power of each level read back with `at_level` equals the DIRECT dense exact Mexican hat of that level (the same operator the single-level step and the nxr golden of
//     test_glued_form compare with: nxr `vortex_cores.hat_operator` on the glued mesh, levels 2 and 3), for every level of the list
//   - with NO form and `glued: true` (a free current, no orientation assumed): each component through the band on every glued vertex, the power summed over the components, equal to the direct computation
//   - the levels are distinct operators (the two levels' power differs); one level of `levels` equals the same level asked alone (`level`)
//   - the plan: `level` and `levels` together, a repeated level, a level the ladder lacks, `at_level` of a level not in the field, `at_level` of a single-level field, `glued` without a level, levels with the
//     spatiotemporal envelope: all refused with the reason
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
#include "nxr/signal/forms.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/filters_fir.h"
#include "make_grids.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;

namespace {
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
std::vector<float> read_floats(const fs::path& p) {
  ArrayReader r = nxrtest::open_array(p);
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
std::string def(const std::string& form, const std::string& levels, bool glued, const std::vector<int>& read) {
  std::string t = "pipeline: levels_run\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: free}}\n"
                  "  tiles: {selection: ladder, manifold: kernel.to}\nsteps:\n  chunks:\n    over: recording\n    do:\n      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0], shape: morlet}, out: z}\n"
                  "      - {step: apply_kernel, in: {x: z, kernel: kernel, tiles: tiles}, with: {" + levels + ", transform: exact, spatial_shape: mexhat, form: " + form + ", glued: " + (glued ? "true" : "false") + "}, out: j}\n"
                  "      - {step: power, in: {x: j}, out: p}\n";
  std::string outs = "outputs:\n";
  for (const int l : read) {
    t += "      - {step: at_level, in: {x: p}, with: {level: " + std::to_string(l) + "}, out: p" + std::to_string(l) + "}\n      - {step: measure, in: {x: p" + std::to_string(l) + "}, with: {bad_segments: ignore}, out: power" + std::to_string(l) + "}\n";
    outs += "  power" + std::to_string(l) + ": {kind: band power}\n";
  }
  return t + outs;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_apply_kernel_levels_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = 12, T = 4096;
  const double sfreq = 250.0;
  const int level = 3;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T), kfree(static_cast<std::size_t>(3 * N) * M), kfix(static_cast<std::size_t>(N) * M);
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c) * T + t] = static_cast<float>(std::cos(2 * M_PI * 10.0 * t / sfreq + 0.7 * c) + 0.3 * nd(rng));
  for (auto& k : kfree) k = static_cast<float>(nd(rng));
  for (auto& k : kfix) k = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      // the Desikan-Killiany atlas with a medial wall on each hemisphere's inner side: the hemispheres are glued by anatomy (the bridge grown between the rims)
      std::vector<std::int32_t> dk(static_cast<std::size_t>(b.n), 1);
      for (int i = 0; i < b.n; ++i) { const double y = b.P[static_cast<std::size_t>(3 * i + 1)]; if ((i < b.nHalf && y < 0.7) || (i >= b.nHalf && y > -0.7)) dk[static_cast<std::size_t>(i)] = 0; }
      so.atlases.push_back(AtlasMember{"Desikan-Killiany", dk, {"unassigned", "cortex"}, {{{0, 0, 0}}, {{200, 200, 200}}}});
      surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
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
      FieldSpec rec;
      rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 4096}; rec.array.compress = false; rec.cols.set("unit", "A");
      sub.field(std::move(rec));
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
        sub.operator_(std::move(op));
      };
      kernel("fixed", kfix, N, "scalar");
      kernel("free", kfree, 3 * N, "vector3");
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_time_grids(ds, sub);
  nxrtest::make_tiles(ds, sub);
  { EigenSolveOptions eo; eo.modes = 100; eo.surface = "cortex"; eigen_solve(ds, sub, eo, {}); }          // stores the glued surface (cortex_glued) and its modes: the form's output lives on it

  // the DIRECT computation: Z the band's analytic signal on the channels (Morlet taps), M = F_l form(K) with the dense exact hat, |M Z|² averaged over the run
  const SurfaceInput S = read_surface(sub, std::string("cortex"), true);
  const SurfaceLevel SL = read_surface_level(sub, S, level);
  SpatialBandOptions sbo; sbo.transform = "exact"; sbo.spatial_shape = "mexhat"; sbo.glued = true;
  const SpatialBand sband = make_spatial_band(ds, sub, S, SL, sbo);
  const std::int64_t Vg = sband.glued_vertices;
  CHECK(Vg > N && sband.glued_surface == "cortex_glued");
  const auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, 8.0, 12.0); };
  const auto h = nxr::signal::gainTaps(gain, sfreq, false), q = nxr::signal::gainTaps(gain, sfreq, true);
  const auto re = nxr::signal::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::signal::convolveRows(xf.data(), M, T, q, 0, T, true);
  Eigen::MatrixXd Re(M, T), Im(M, T);
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) { Re(c, t) = re[static_cast<std::size_t>(c) * T + t]; Im(c, t) = im[static_cast<std::size_t>(c) * T + t]; }
  const SurfaceInput S32 = read_surface(sub, std::string("cortex"));
  std::vector<float> P32(S32.vertices.begin(), S32.vertices.end());
  const auto op = nxr::signal::assembleFaceGradient(P32.data(), S32.faces.data(), static_cast<int>(S32.n_vertices()), static_cast<int>(S32.n_faces()));
  Eigen::MatrixXd Kfree(3 * N, M);
  for (int r = 0; r < 3 * N; ++r) for (int c = 0; c < M; ++c) Kfree(r, c) = kfree[static_cast<std::size_t>(r) * M + c];
  auto direct = [&](const Eigen::MatrixXd& Mop) {
    const Eigen::MatrixXd A = Mop * Re, B = Mop * Im;
    std::vector<double> p(static_cast<std::size_t>(Mop.rows()));
    for (Eigen::Index v = 0; v < Mop.rows(); ++v) { double s = 0; for (int t = 0; t < T; ++t) s += A(v, t) * A(v, t) + B(v, t) * B(v, t); p[static_cast<std::size_t>(v)] = s / T; }
    return p;
  };
  auto worst = [&](const std::vector<float>& got, const std::vector<double>& want) {
    double w = 0, top = 0;
    for (std::size_t i = 0; i < want.size(); ++i) { top = std::max(top, want[i]); w = std::max(w, std::abs(got[i] - want[i])); }
    return got.size() == want.size() ? w / top : 1e9;
  };
  using nxr::signal::Form;
  const SurfaceLevel SL2 = read_surface_level(sub, S, 2);
  const SpatialBand sband2 = make_spatial_band(ds, sub, S, SL2, sbo);
  CHECK(sband2.glued_vertices == Vg);
  const std::vector<SpatialBand*> bands = {const_cast<SpatialBand*>(&sband2), const_cast<SpatialBand*>(&sband)};     // levels 2, 3
  const std::vector<int> lvs = {2, 3};
  g::RunOptions replace; replace.replace = true; replace.chunks = "all";
  auto run_def = [&](const std::string& t) { const g::Definition d = g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t); return g::run(ds, sub, d, Json::object(), replace); };
  auto stored_power = [&](const Json& r, int l) {
    for (const auto& o : r["stored"]) if (o["output"] == "power" + std::to_string(l)) return read_floats((sub.folder() / (o["path"].get<std::string>())));
    return std::vector<float>();
  };
  // WITH A FORM: curl_tan, the levels 2 and 3 in one operator
  std::vector<std::vector<double>> wants;
  {
    const auto r = run_def(def("curl_tan", "levels: [2, 3]", false, lvs));
    CHECK(r.size() == 1 && r[0]["stored"].size() == 2);
    for (std::size_t k = 0; k < lvs.size(); ++k) {
      const Eigen::MatrixXd Mop = bands[k]->apply_glued(nxr::signal::formOfFreeColumns(Form::curl_tan, Kfree, op));
      wants.push_back(direct(Mop));
      const std::vector<float> got = stored_power(r[0], lvs[k]);
      const double e = worst(got, wants.back());
      std::printf("  curl_tan, levels [2, 3]: level %d band power per vertex vs the direct dense exact hat: %.2e\n", lvs[k], e);
      CHECK(e < 5e-4 && got.size() == static_cast<std::size_t>(Vg));
    }
    double dd = 0, top = 0;
    for (std::size_t i = 0; i < wants[0].size(); ++i) { dd = std::max(dd, std::abs(wants[0][i] - wants[1][i])); top = std::max(top, wants[1][i]); }
    CHECK(dd > 1e-3 * top);                                                      // the levels are distinct operators
    const auto& prov = r[0].contains("prepared") ? r[0]["prepared"] : Json::object();
    CHECK(prov.dump().find("\"levels\":[2,3]") != std::string::npos);            // the provenance names them
    // one level of the list equals the same level asked alone
    std::string one = def("curl_tan", "level: 3", false, {});
    one.replace(one.find("      - {step: power"), std::string("      - {step: power, in: {x: j}, out: p}\n").size(), "      - {step: power, in: {x: j}, out: p}\n      - {step: measure, in: {x: p}, with: {bad_segments: ignore}, out: power3}\n");
    one.replace(one.find("outputs:\n"), 9, "outputs:\n  power3: {kind: band power}\n");
    const auto r1 = run_def(one);
    const std::vector<float> a = stored_power(r1[0], 3), bb = stored_power(r[0], 3);
    double dmax = 0;
    for (std::size_t i = 0; i < a.size() && a.size() == bb.size(); ++i) dmax = std::max(dmax, static_cast<double>(std::abs(a[i] - bb[i])));
    std::printf("  level 3 of [2, 3] vs level 3 alone: max difference %.2e\n", dmax);
    CHECK(a.size() == bb.size() && a.size() == static_cast<std::size_t>(Vg) && dmax < 1e-4 * top);
  }
  // NO FORM, glued: the free current's three components each through the band, the power summed over them
  {
    const auto r = run_def(def("none", "levels: [2, 3]", true, lvs));
    CHECK(r.size() == 1 && r[0]["stored"].size() == 2);
    for (std::size_t k = 0; k < lvs.size(); ++k) {
      std::vector<double> want(static_cast<std::size_t>(Vg), 0.0);
      for (int comp = 0; comp < 3; ++comp) {
        Eigen::MatrixXd Kc(N, M);
        for (int v = 0; v < N; ++v) Kc.row(v) = Kfree.row(3 * v + comp);
        const std::vector<double> pc = direct(bands[k]->apply_glued(Kc));
        for (std::size_t v = 0; v < want.size(); ++v) want[v] += pc[v];
      }
      const std::vector<float> got = stored_power(r[0], lvs[k]);
      const double e = worst(got, want);
      std::printf("  free current (no form), glued, level %d: power per glued vertex vs the direct components: %.2e\n", lvs[k], e);
      CHECK(e < 5e-4 && got.size() == static_cast<std::size_t>(Vg));
    }
  }
  // THE PLAN
  auto refused = [&](const std::string& t) { return what_of([&] { const g::Definition d = g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t); g::run(ds, sub, d, Json::object(), g::RunOptions{}); }); };
  CHECK(refused(def("curl_tan", "level: 3, levels: [2, 3]", false, {3})).find("not both") != std::string::npos);
  CHECK(refused(def("curl_tan", "levels: [3, 3]", false, {3})).find("repeats") != std::string::npos);
  CHECK(refused(def("curl_tan", "levels: [2, 40]", false, {2})).find("no level 40") != std::string::npos);
  CHECK(refused(def("curl_tan", "levels: [2, 3]", false, {5})).find("not one of the field's levels") != std::string::npos);
  CHECK(refused(def("curl_tan", "level: 3", false, {3})).find("not a multi-level field") != std::string::npos);
  CHECK(refused(def("none", "levels: [2, 3], envelope: spatiotemporal", false, {3})).find("temporal envelope") != std::string::npos);
  { std::string t = def("none", "level: 3", true, {});
    t.replace(t.find("transform: exact"), 16, "transform: exact");
    CHECK(refused("pipeline: x\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: free}}\nsteps:\n  chunks:\n    over: recording\n    do:\n"
                  "      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}\n      - {step: apply_kernel, in: {x: z, kernel: kernel}, with: {glued: true}, out: j}\n      - {step: power, in: {x: j}, out: p}\n"
                  "      - {step: measure, in: {x: p}, with: {bad_segments: ignore}, out: power}\noutputs:\n  power: {kind: band power}\n").find("needs `tiles`") != std::string::npos); }
  return nxrtest::finish("test_apply_kernel_levels") || g_failures ? 1 : 0;
}
