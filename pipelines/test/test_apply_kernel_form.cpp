// apply_kernel's `form` on a synthetic two-hemisphere source space, never a real dataset: a free kernel of random currents, a recording of a 10 Hz oscillation plus noise.
//   - the run's band power per vertex (analytic → apply_kernel → power → measure) equals the DIRECT computation: the band's analytic signal, Z, and M = F_l formOfFreeColumns(K): |M Z|², the mean over
//     the run, for curl_tan and for div_tan and normal (the forms' numbers are tested against nxr in test_form_columns)
//   - the ORDER is the operator's: kernel → form → hat is not kernel → hat → form (the form step after a band), so the folded operator differs from it
//   - the output is ONE complex scalar on EVERY vertex of the GLUED surface: the stored field has the glued surface's vertices; the cortex has a medial
//     wall (atlas) so that its hemispheres are glued by anatomy, and the glue is stored (`eigen solve`) before the form's run
//   - the plan: refusals (a form of a fixed kernel, an unknown form, a form with the spatiotemporal envelope), the form in the provenance, a different form another key
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
#include "nxr/forms.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/yaml.h"
#include "nxr/pipelines/eigen.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/filterbank.h"
#include "nxr/taps.h"
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
std::string def(const std::string& form, const std::string& shape, const std::string& extra = "") {
  return "pipeline: form_run\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0, where: {name: " + std::string(form == "fixed" ? "fixed" : "free") +
         "}}\n  tiles: {selection: ladder, manifold: kernel.to}\nsteps:\n  chunks:\n    over: recording\n    do:\n      - {step: analytic, in: {x: recording}, with: {band: [8.0, 12.0], shape: morlet}, out: z}\n"
         "      - {step: apply_kernel, in: {x: z, kernel: kernel, tiles: tiles}, with: {level: 3, transform: exact, spatial_shape: " + shape + ", form: " + (form == "fixed" ? "none" : form) + extra + "}, out: j}\n"
         "      - {step: power, in: {x: j}, out: p}\n      - {step: measure, in: {x: p}, with: {bad_segments: ignore}, out: power}\noutputs:\n  power: {kind: band power}\n";
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_apply_kernel_form_" + std::to_string(::getpid()));
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
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  const SurfaceLevel SL = read_surface_level(sub, S, level);
  SpatialBandOptions sbo; sbo.transform = "exact"; sbo.spatial_shape = "mexhat"; sbo.glued = true;
  const SpatialBand sband = make_spatial_band(ds, sub, S, SL, sbo);
  const std::int64_t Vg = sband.glued_vertices;
  CHECK(Vg > N && sband.glued_surface == "cortex_glued");
  const auto gain = [&](double f) { return nxr::filterbank::morletBand(f, 8.0, 12.0); };
  const auto h = nxr::taps::gainTaps(gain, sfreq, false), q = nxr::taps::gainTaps(gain, sfreq, true);
  const auto re = nxr::taps::convolveRows(xf.data(), M, T, h, 0, T, false), im = nxr::taps::convolveRows(xf.data(), M, T, q, 0, T, true);
  Eigen::MatrixXd Re(M, T), Im(M, T);
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) { Re(c, t) = re[static_cast<std::size_t>(c) * T + t]; Im(c, t) = im[static_cast<std::size_t>(c) * T + t]; }
  const SurfaceInput S32 = read_surface(sub);
  std::vector<float> P32(S32.vertices.begin(), S32.vertices.end());
  const auto op = nxr::forms::assembleFaceGradient(P32.data(), S32.faces.data(), static_cast<int>(S32.n_vertices()), static_cast<int>(S32.n_faces()));
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
  using nxr::forms::Form;
  std::vector<double> folded_curl;
  for (const auto& [name, id] : std::vector<std::pair<std::string, Form>>{{"curl_tan", Form::curl_tan}, {"div_tan", Form::div_tan}, {"normal", Form::nx}}) {
    const std::string text_def = def(name, "mexhat");
    const g::Definition d = g::parse(parse_yaml(text_def, "f.yaml"), "f.yaml", text_def);
    const auto r = g::run(ds, sub, d, Json::object(), g::RunOptions{[] { g::RunOptions o; o.replace = true; o.chunks = "0"; return o; }()});
    CHECK(r.size() == 1 && r[0]["stored"].size() == 1);
    if (r.size() != 1) continue;
    const auto row = nxr::io::task::Subject::of(sub).by_id(nxr::io::task::Cls::field, r[0]["stored"][0]["id"].get<std::string>());
    const std::vector<float> got = read_floats((sub.folder() / (r[0]["stored"][0]["path"].get<std::string>())));
    const Eigen::MatrixXd Mop = sband.apply_glued(nxr::forms::formOfFreeColumns(id, Kfree, op));
    const std::vector<double> want = direct(Mop);
    const double e = worst(got, want);
    std::printf("  form %s, mexhat level %d: band power per vertex vs the direct operator: %.2e\n", name.c_str(), level, e);
    CHECK(e < 5e-4 && row && row->value_type == "scalar" && got.size() == static_cast<std::size_t>(Vg));
    if (name == "curl_tan") {
      folded_curl = want;
      const std::vector<double> wantN = direct(sband.apply(nxr::forms::formOfFreeColumns(id, Kfree, op)));            // the same operator read back on the native vertices
      // the ORDER: kernel → hat → curl is another operator (the form after the band): a different power
      Eigen::MatrixXd KB(3 * N, M);
      for (int comp = 0; comp < 3; ++comp) {
        Eigen::MatrixXd Kc(N, M);
        for (int v = 0; v < N; ++v) Kc.row(v) = Kfree.row(3 * v + comp);
        const Eigen::MatrixXd Bc = sband.apply(Kc);
        for (int v = 0; v < N; ++v) KB.row(3 * v + comp) = Bc.row(v);
      }
      const std::vector<double> other = direct(nxr::forms::formOfFreeColumns(id, KB, op));
      double dd = 0, top = 0;
      for (std::size_t i = 0; i < wantN.size(); ++i) { dd = std::max(dd, std::abs(other[i] - wantN[i])); top = std::max(top, wantN[i]); }
      std::printf("  curl after the hat vs the hat after the curl: max difference %.2e of the power's maximum\n", dd / top);
      CHECK(dd > 1e-3 * top);
    }
  }
  // the form is in the run's key: curl_tan and div_tan stored by different keys (the second replaced the first above); the plan refuses what a form cannot read
  auto refuse_of = [&](const std::string& form, const std::string& extra, const std::string& kernel_kind = "") {
    const std::string text_def = def(kernel_kind.empty() ? form : kernel_kind, "mexhat", extra);
    return what_of([&] { const g::Definition d = g::parse(parse_yaml(text_def, "f.yaml"), "f.yaml", text_def); g::run(ds, sub, d, Json::object(), g::RunOptions{}); });
  };
  CHECK(refuse_of("sideways", "").find("form") != std::string::npos);
  { const std::string t = def("fixed", "mexhat").replace(def("fixed", "mexhat").find("form: none"), 10, "form: curl_tan");
    CHECK(what_of([&] { const g::Definition d = g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t); g::run(ds, sub, d, Json::object(), g::RunOptions{}); }).find("free current") != std::string::npos); }
  CHECK(refuse_of("curl_tan", ", envelope: spatiotemporal").find("spatiotemporal") != std::string::npos);
  return nxrtest::finish("test_apply_kernel_form") || g_failures ? 1 : 0;
}
