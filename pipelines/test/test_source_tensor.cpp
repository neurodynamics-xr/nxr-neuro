// sourcetensor compute on a synthetic two-hemisphere brain: three sensors carrying a current that TURNS from +x towards +y at 10 Hz (cos, sin), a
// kernel that gives every source the sensors' three channels as its x y z. Through the whole chain (the recording read in chunks, Welch's
// cross-spectrum with its quadrature part, the tensor, the canonical frames): the tensor is stored as hermitian3 and its trace equals sourcepsd's band
// power of the same options; at every vertex the turn is seen from its outward normal, so S3 has the sign of n·z (counter-clockwise about +z);
// S0 + the normal power is the trace; a re-run is a cache hit; refused: a surface without canonical frames, a fixed-orientation kernel.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "run_definition.h"
#include "nxr/io/task/task.h"
#include "nxr/bst/entities.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/meeg.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

template <class T> static std::vector<T> read_field(SubjectWriter& sub, const std::string& name) {
  const auto r = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = name, .any_status = true});
  ArrayReader a = ArrayReader::open(sub.store(), *r->path);
  const auto raw = a.read();
  std::vector<T> out(raw.size() / sizeof(T));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

// a band of a map statistic as its values [n x n_components]: the statistic's values [n x entries x n_components] at the entry (the slice's index)
static std::vector<double> slice_values(nxr::io::SubjectWriter& sub, const std::string& statistic, const nxr::io::task::MapSlice& m) {
  const nxr::io::StoredStatistic st = nxr::io::read_statistic(sub, statistic);
  const std::int64_t K = st.shape.at(1), C = m.n_components;
  std::vector<double> v(static_cast<std::size_t>(m.n * C));
  for (std::int64_t i = 0; i < m.n; ++i)
    for (std::int64_t c = 0; c < C; ++c) v[static_cast<std::size_t>(i * C + c)] = st.values[static_cast<std::size_t>((i * K + m.index) * C + c)];
  return v;
}

// a map of a statistic by its name (`<statistic>_<band>`), as float [n x components] (since 2026-11-13 the source maps are statistics)
static std::vector<float> map_named(nxr::io::SubjectWriter& sub, const std::string& statistic, const std::string& name) {
  const std::string surf = nxr::io::read_statistic(sub, statistic).axes[0]["manifold_id"].get<std::string>();
  for (const auto& m : nxr::io::task::Subject::of(sub).map_slices(surf))
    if (m.name == name) { const auto d = slice_values(sub, statistic, m); return std::vector<float>(d.begin(), d.end()); }
  return {};
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_source_tensor_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(2, false);
  const int N = b.n, M = 3, T = 6000;
  const double sfreq = 250.0;
  std::mt19937 rng(11);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (int t = 0; t < T; ++t) {
    const double w = 2 * M_PI * 10.0 * t / sfreq;
    xf[static_cast<size_t>(t)] = static_cast<float>(std::cos(w) + 0.05 * nd(rng));
    xf[static_cast<size_t>(T + t)] = static_cast<float>(std::sin(w) + 0.05 * nd(rng));
    xf[static_cast<size_t>(2 * T + t)] = static_cast<float>(0.05 * nd(rng));
  }
  std::vector<float> kf(static_cast<std::size_t>(3 * N * M), 0.0f), kfix(static_cast<std::size_t>(N * M), 0.0f);
  for (int v = 0; v < N; ++v) for (int a = 0; a < 3; ++a) kf[static_cast<size_t>((3 * v + a) * M + a)] = 1.0f;
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      const std::string surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 0}, {"MEG MAG"}, {}}, {"quality", {0, 0, 0}, {"good"}, {}}}, cid, "the montage families", std::nullopt,
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
      FieldSpec rec;
      rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 700}; rec.array.compress = false; rec.cols.set("unit", "T");
      sub.field(std::move(rec));
      std::vector<std::int32_t> chans = {0, 1, 2};
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        sub.operator_(std::move(op));
      };
      kernel("free", kf, 3 * N, "vector3");
      kernel("fixed", kfix, N, "scalar");
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  // the definitions through the runner: sourcetensor, sourcepsd and surface_frames (the canonical frames the tensor needs)
  const Json bands = Json::array({Json{{"name", "alpha"}, {"lo", 8}, {"hi", 12}}});
  auto base = [&](const char* kernel) { return Json{{"recording", "rest"}, {"kernel", kernel}, {"bands", bands}, {"segment_seconds", 4.0}}; };
  auto run = [&](const char* def, const Json& set) { return nxrtest::run_definition(ds, sub, def, set, false); };
  // refused without the canonical frames
  bool no_frames = false;
  try { run("sourcetensor", base("free")); } catch (const std::exception&) { no_frames = true; }
  CHECK(no_frames);
  nxrtest::run_definition(ds, sub, "surface_frames", Json{{"surface", "cortex"}}, false);
  const auto t = run("sourcetensor", base("free"));
  CHECK(!t.empty() && !t[0].value("hit", false));
  run("sourcepsd", base("free"));
  const auto herm = map_named(sub, "rest_free_tensor", "rest_free_tensor_alpha"), psd = map_named(sub, "rest_free_psd", "rest_free_psd_alpha");
  const auto stokes = map_named(sub, "rest_free_tensor_stokes", "rest_free_tensor_stokes_alpha"), npow = map_named(sub, "rest_free_tensor_normalpower", "rest_free_tensor_normalpower_alpha");
  CHECK_EQ(static_cast<int>(herm.size()), N * 18);
  // the tensor a statistic whose values are hermitian3 (as its Field's were), the Stokes parameters quaternion, both on the band axis
  const StoredStatistic ts = read_statistic(sub, "rest_free_tensor"), qs = read_statistic(sub, "rest_free_tensor_stokes");
  CHECK(text(ts.row, "value_type") == "hermitian3" && integer(ts.row, "n_components") == 18 && ts.shape == (std::vector<std::int64_t>{N, 1, 3, 3, 2}) && text(ts.row, "measure") == "cross_spectral_density");
  CHECK(text(qs.row, "value_type") == "quaternion" && qs.shape == (std::vector<std::int64_t>{N, 1, 4}) && text(read_statistic(sub, "rest_free_tensor_normalshare").row, "measure") == "share");
  // the trace is the band power; S0 + normal power is the trace; S3 has the sign of n·z
  double worst = 0, split = 0;
  int agree = 0, decided = 0;
  // the outward normals: the mesh's, area-weighted
  std::vector<double> nz(static_cast<size_t>(N), 0.0);
  for (int f = 0; f < b.nF; ++f) {
    const int a = b.F[3 * f], c1 = b.F[3 * f + 1], c2 = b.F[3 * f + 2];
    const double ux = b.P[3 * c1] - b.P[3 * a], uy = b.P[3 * c1 + 1] - b.P[3 * a + 1], vx = b.P[3 * c2] - b.P[3 * a], vy = b.P[3 * c2 + 1] - b.P[3 * a + 1];
    for (int v : {a, c1, c2}) nz[static_cast<size_t>(v)] += ux * vy - uy * vx;
  }
  for (int v = 0; v < N; ++v) {
    const double tr = herm[static_cast<size_t>(18 * v)] + herm[static_cast<size_t>(18 * v + 8)] + herm[static_cast<size_t>(18 * v + 16)];
    worst = std::max(worst, std::abs(tr - psd[static_cast<size_t>(v)]) / psd[static_cast<size_t>(v)]);
    const double s0 = stokes[static_cast<size_t>(4 * v)], s3 = stokes[static_cast<size_t>(4 * v + 3)];
    if (!std::isfinite(s0)) continue;
    split = std::max(split, std::abs(s0 + npow[static_cast<size_t>(v)] - tr) / tr);
    // the normal's z share: |n_z| / |n| > 0.5 decides the turn's sense
    if (std::abs(s3) > 0.2 * s0) { ++decided; agree += (s3 > 0) == (nz[static_cast<size_t>(v)] > 0); }
  }
  std::printf("  trace vs sourcepsd: %.2e; S0 + normal vs trace: %.2e; S3's sign is n·z's at %d of %d vertices that turn\n", worst, split, agree, decided);
  CHECK(worst < 1e-6);
  CHECK(split < 1e-5);
  CHECK(decided > N / 3 && agree == decided);
  // a re-run is a cache hit; a fixed-orientation kernel is refused
  { const auto again = run("sourcetensor", base("free")); CHECK(!again.empty() && again[0].value("hit", false)); }
  bool fixed = false;
  try { run("sourcetensor", base("fixed")); } catch (const std::exception&) { fixed = true; }
  CHECK(fixed);
  fs::remove_all(root);
  return nxrtest::finish("test_source_tensor");
}
