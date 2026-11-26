// `run sensor_surface` (pipelines/definitions/sensor_surface.yaml) on a synthetic subject: 150 axial gradiometers on a helmet-shaped cap (eight integration points each, the first
// four round the pickup coil, the other four 5 cm further out: the reference coil) and 6 electrodes, stored as the importer stores channels (the point = the first coil; the coils
// manifold and the csr coil weights). The pick is the gradiometers in a shuffled order. Checks: the surface is a manifold derived from the channels, its vertices the picked
// channels in the pick's order at Brainstorm's positions (the pickup coil's centre), its map the pick, its faces nxr::geometry::pointset::pointSetMesh of those positions (a disc,
// outward); position_rule stored gives the first coils; the refusals (no pick, a pick of another manifold's size); then `run eigenmodes` on the surface (an open disc) with the
// *1 floor named: the mode axis, both kinds of modes, the 0-, 1- and 2-form parts. `run sensor_mesh`: the good gradiometers (one flagged bad) chosen from the type and
// quality labellings, the surface, its map and its stored channels, a cache hit on a second run, and the refusals (no sensor type; three good electrodes). `run sensor_flow` end to end
// on the helmet, then on A SPAN (issue #291): a one-chunk canary (`--chunks 1`) on the same stored time grid equals the whole run on its tiles (NaN outside), is never a hit on the whole
// run's measurements, records its span, and is a hit when asked again. THE FIBRE NETWORK: a bundle stored by `add fibres`, `tile_network` on the cortex's tiles and `connectome_flow` on the
// canary's chunk spectra, their outputs stored.
#include "nxr/io/subject_api.h"
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/geometry/point_set_mesh.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/pipelines/input.h"
#include "make_grids.h"
#include "run_definition.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {

std::string refusal(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.what(); } catch (const std::exception& e) { return e.what(); } return ""; }

template <class T>
std::vector<T> read_as(SubjectWriter& sub, const std::string& path) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  const auto raw = r.read();
  if (r.dtype() == DType::Float32) { std::vector<float> f(raw.size() / 4); std::memcpy(f.data(), raw.data(), raw.size()); return std::vector<T>(f.begin(), f.end()); }
  if (r.dtype() == DType::Float64) { std::vector<double> f(raw.size() / 8); std::memcpy(f.data(), raw.data(), raw.size()); return std::vector<T>(f.begin(), f.end()); }
  if (r.dtype() == DType::Int8 || r.dtype() == DType::UInt8 || r.dtype() == DType::Bool) { std::vector<std::int8_t> f(raw.size()); std::memcpy(f.data(), raw.data(), raw.size()); return std::vector<T>(f.begin(), f.end()); }
  if (r.dtype() == DType::Int16) { std::vector<std::int16_t> f(raw.size() / 2); std::memcpy(f.data(), raw.data(), raw.size()); return std::vector<T>(f.begin(), f.end()); }
  std::vector<std::int32_t> f(raw.size() / 4); std::memcpy(f.data(), raw.data(), raw.size()); return std::vector<T>(f.begin(), f.end());
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);                                            // unbuffered: a crash keeps every line printed before it
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_sensor_surface_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const int G = 150, E = 6, M = G + E;
  // the pickup coils' centres on an ellipsoidal cap (a jittered spiral, no four points cocircular), the electrodes lower down
  std::vector<double> centre, normal;
  std::uint64_t s = 99;
  auto rnd = [&] { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<double>(s >> 11) / 9007199254740992.0; };
  const double golden = M_PI * (3.0 - std::sqrt(5.0)), z0 = std::cos(1.6);
  for (int i = 0; i < M; ++i) {
    const double z = i < G ? 1.0 - (1.0 - z0) * (i + 0.5) / G + 0.002 * (rnd() - 0.5) : -0.2 + 0.05 * i / M;
    const double rho = std::sqrt(std::max(0.0, 1.0 - z * z)), phi = golden * i + 0.05 * (rnd() - 0.5);
    const double u[3] = {rho * std::cos(phi), rho * std::sin(phi), z};
    for (int k = 0; k < 3; ++k) { normal.push_back(u[k]); centre.push_back((k == 0 ? 0.09 : k == 1 ? 0.1 : 0.11) * u[k] + (k == 2 ? 0.04 : 0.0)); }
  }
  // the coils: a gradiometer's eight points (four 4 mm off its centre in its tangent plane, then the same 5 cm outward); an electrode has none
  std::vector<float> cp, cw, stored;
  std::vector<std::int32_t> indptr(1, 0), indices;
  for (int i = 0; i < M; ++i) {
    const double* n = &normal[static_cast<std::size_t>(3 * i)];
    const double* c = &centre[static_cast<std::size_t>(3 * i)];
    double t1[3] = {n[1], -n[0], 0};
    double l = std::hypot(t1[0], t1[1]);
    if (l < 1e-9) { t1[0] = 1; t1[1] = 0; l = 1; }
    for (double& x : t1) x /= l;
    const double t2[3] = {n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]};
    if (i < G) {
      for (int coil = 0; coil < 2; ++coil)
        for (int k = 0; k < 4; ++k) {
          const double a = (k & 1 ? 1 : -1) * 0.004, b = (k & 2 ? 1 : -1) * 0.004;
          for (int d = 0; d < 3; ++d) cp.push_back(static_cast<float>(c[d] + a * t1[d] + b * t2[d] + coil * 0.05 * n[d]));
          cw.push_back(coil ? -0.25f : 0.25f);
          indices.push_back(static_cast<std::int32_t>(indices.size()));
        }
      for (int d = 0; d < 3; ++d) stored.push_back(cp[cp.size() - 24 + static_cast<std::size_t>(d)]);   // the importer's point: the first coil
    } else {
      for (int d = 0; d < 3; ++d) stored.push_back(static_cast<float>(c[d]));
    }
    indptr.push_back(static_cast<std::int32_t>(indices.size()));
  }
  // the pick: the gradiometers, shuffled
  std::vector<std::int32_t> pick;
  for (int i = 0; i < G; ++i) pick.push_back((i * 37) % G);

  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7(), coil_id = uuid7();
      auto points = [&](const std::string& id, const std::string& name, const std::vector<float>& a, std::int64_t n) {
        ManifoldSpec m;
        m.id = id; m.name = name; m.path = "sensor/" + name; m.type = "points"; m.n_vertices = n;
        m.topology = {"none", 0, 1, std::nullopt, {}};
        GeometrySpec g;
        g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/" + name;
        m.geometry = g;
        m.populate = [&sub, a, n, id, name](nxr::io::Store&, const std::string&) { sub.write_array("sensor/" + name, view(a, {n, 3}), {}, std::nullopt, "manifold", id); };
        sub.manifold(std::move(m));
      };
      points(cid, "sensors", stored, M);
      std::vector<std::int64_t> tcode, qcode(M, 0);
      for (int i = 0; i < M; ++i) tcode.push_back(i < G ? 0 : 1);
      qcode[17] = 1;                                         // one gradiometer bad (sensor_mesh leaves it out; sensor_surface takes its pick as named)
      qcode[G + 3] = qcode[G + 4] = qcode[G + 5] = 1;        // three electrodes bad: three good EEG channels, fewer than sensor_mesh's four
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", tcode, {"MEG GRAD", "EEG"}, {}}, {"quality", qcode, {"good", "bad"}, {}}}, cid, "the montage families", std::nullopt,
                          [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
      const std::int64_t P = static_cast<std::int64_t>(cw.size());
      points(coil_id, "sensors_coils", cp, P);
      OperatorSpec op;
      op.name = "sensors_coil_weights"; op.path = op.name; op.kind = "coil weights"; op.from_manifold_id = coil_id; op.layout = "csr"; op.n_rows = M; op.n_cols = P;
      op.cols.set("to_manifold_id", cid).set("nnz", P);
      op.sparse = SparseArrays{{{"indptr", view(indptr, {M + 1})}, {"indices", view(indices, {P})}, {"data", view(cw, {P})}}};
      sub.operator_(std::move(op));
      SelectionSpec ss; ss.name = "sensors_meg"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(pick, {G}); ss.picked = true;
      sub.selection(std::move(ss));
      std::vector<std::int32_t> few = {0, 1, 2};
      SelectionSpec sf; sf.name = "three"; sf.path = sf.name; sf.type = "indices"; sf.manifold_id = coil_id; sf.data = view(few, {3}); sf.picked = true;
      sub.selection(std::move(sf));
      // a recording on the channels for the flow section: a plane wave at 10 Hz travelling along x across the cap (phase k x), small noise; 8 s at 256 Hz, chunks of 256
      const std::int64_t T = 2048;
      const double sfreq = 256.0;
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
      std::vector<float> wave(static_cast<std::size_t>(M * T));
      for (int ch = 0; ch < M; ++ch)
        for (std::int64_t t = 0; t < T; ++t)
          wave[static_cast<std::size_t>(ch * T + t)] = static_cast<float>(std::cos(2 * M_PI * 10.0 * static_cast<double>(t) / sfreq - 40.0 * stored[static_cast<std::size_t>(3 * ch)]) + 0.05 * (rnd() - 0.5));
      FieldSpec rec;
      rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(wave, {M, T}); rec.array.chunks = Shape{M, 256}; rec.array.compress = false; rec.cols.set("unit", "T");
      sub.field(std::move(rec));
    });
  }

  // THE CORTEX AND ITS FIBRES for the connectome flow at the end: a synthetic two-hemisphere brain (about 10 cm across, SCS metres) and 400 straight fibres between its vertices,
  // stored by `add fibres` (task::add_fibres, the command's library function) as the importers store a bundle. The command opens the dataset itself, so the writer is let go first.
  const Brain brainSurface = [] { Brain b = brain(3, false); for (double& x : b.P) x *= 0.04; return b; }();
  {
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    sub.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = brainSurface.Sph; so.primary = false; add_surface(sub, so, brainSurface.P, brainSurface.F); });
  }
  ds.close();
  {
    std::uint64_t f = 7;
    auto pick = [&] { f = f * 6364136223846793005ULL + 1442695040888963407ULL; return static_cast<int>((f >> 33) % static_cast<std::uint64_t>(brainSurface.n)); };
    std::vector<double> fibre;
    for (int k = 0; k < 400; ++k) {
      const int a = pick(), c = pick();
      for (int t = 0; t < 3; ++t) for (int d = 0; d < 3; ++d) fibre.push_back(brainSurface.P[static_cast<std::size_t>(3 * a + d)] + 0.5 * t * (brainSurface.P[static_cast<std::size_t>(3 * c + d)] - brainSurface.P[static_cast<std::size_t>(3 * a + d)]));
    }
    std::vector<std::uint8_t> bytes(fibre.size() * sizeof(double));
    std::memcpy(bytes.data(), fibre.data(), bytes.size());
    nxr::io::task::AddFibresRequest af;
    af.dataset = (root / "demo").string(); af.subject = "S1"; af.name = "bundle"; af.created_by = "test_sensor_surface";
    af.points = nxr::io::task::ArrayInput::memory("bundle.npy", "float64", {400, 3, 3}, bytes);
    af.origin = Json{{"source", Json{{"name", "synthetic straight fibres"}}}}.dump();
    const Json added = nxr::io::task::add_fibres(af).result;
    CHECK(added.value("n_fibres", 0) == 400 && added.value("precision", std::string()) == "float64" && added.value("type", std::string()) == "curve");
  }
  ds = open_writer(root / "demo");
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  const auto rep = nxrtest::run_definition(ds, sub, "sensor_surface", Json{{"channels", "sensors"}, {"pick", "sensors_meg"}, {"name", "helmet"}});
  CHECK(!rep.empty());
  const nxr::pipelines::SurfaceInput S = nxr::pipelines::read_surface(sub, "helmet", true);
  CHECK_EQ(S.n_vertices(), static_cast<std::int64_t>(G));
  const auto ch = nxr::io::find_manifold(sub, "sensors");
  CHECK(ch.has_value());
  CHECK(opt_text(S.manifold, "derived_from_id") == std::optional<std::string>(text(*ch, "id")));
  CHECK_EQ(text(S.manifold, "type"), std::string("surface"));
  // the vertices: the pickup coils' centres (the mean of the first four points), in the pick's order (float32 as stored)
  double err = 0;
  for (int v = 0; v < G; ++v) for (int d = 0; d < 3; ++d) err = std::max(err, std::abs(S.vertices[static_cast<std::size_t>(3 * v + d)] - centre[static_cast<std::size_t>(3 * pick[static_cast<std::size_t>(v)] + d)]));
  std::printf("  helmet: %lld vertices, %lld triangles; vertices vs the pickup coils' centres: max |difference| %.1e m\n", (long long)S.n_vertices(), (long long)S.n_faces(), err);
  CHECK(err < 1e-6);
  // the coil shapes (Brainstorm's PlotCoils): one quadrilateral (two triangles) per picked gradiometer, its centre the channel position, each vertex mapped to its channel
  {
    nxrtest::run_definition(ds, sub, "sensor_coils", Json{{"channels", "sensors"}, {"pick", "sensors_meg"}, {"name", "coils_surface"}});
    const nxr::pipelines::SurfaceInput K = nxr::pipelines::read_surface(sub, "coils_surface", true);
    CHECK_EQ(K.n_vertices(), static_cast<std::int64_t>(4 * G));
    CHECK_EQ(K.n_faces(), static_cast<std::int64_t>(2 * G));
    const auto map = read_as<int>(sub, text(*sub.find("field", nxr::io::task::Subject::of(sub).node_path(K.id, "coils_surface_map")), "path"));
    double cerr = 0;
    for (int v = 0; v < 4 * G; ++v) CHECK_EQ(map[static_cast<std::size_t>(v)], pick[static_cast<std::size_t>(v / 4)]);
    for (int g = 0; g < G; ++g)
      for (int d = 0; d < 3; ++d) {
        double m = 0;
        for (int q = 0; q < 4; ++q) m += K.vertices[static_cast<std::size_t>(3 * (4 * g + q) + d)] / 4;
        cerr = std::max(cerr, std::abs(m - centre[static_cast<std::size_t>(3 * pick[static_cast<std::size_t>(g)] + d)]));
      }
    std::printf("  coils: %lld vertices, %lld triangles; shape centres vs channel positions: max |difference| %.1e m\n", (long long)K.n_vertices(), (long long)K.n_faces(), cerr);
    CHECK(cerr < 1e-6);
  }
  // the faces: the mesher's on those positions (the same triangles, the same turn); a disc
  {
    std::vector<double> pos(S.vertices.begin(), S.vertices.end());
    const auto m = nxr::geometry::pointset::pointSetMesh(pos);
    CHECK(m.faces == S.faces);
    CHECK_EQ(m.boundary_loops, 1);
    CHECK_EQ(m.euler_characteristic, std::int64_t(1));
    const Json prod = Json::parse(opt_text(S.manifold, "producer_json").value_or("{}"));
    CHECK(prod.contains("mesh") && prod["mesh"].value("boundary_loops", 0) == 1);
    CHECK(prod.contains("sensor_surface") && prod["sensor_surface"]["pick"].value("name", std::string()) == "sensors_meg");
  }
  // the map: vertex -> channel, the pick
  {
    const auto f = sub.find("field", nxr::io::task::Subject::of(sub).node_path(S.id, "helmet_map"));
    CHECK(f.has_value());
    if (f) CHECK(read_as<std::int32_t>(sub, text(*f, "path")) == pick);
  }
  // position_rule stored: the first coils
  CHECK(!nxrtest::run_definition(ds, sub, "sensor_surface", Json{{"channels", "sensors"}, {"pick", "sensors_meg"}, {"name", "helmet_first_coil"}, {"position_rule", "stored"}}).empty());
  {
    const nxr::pipelines::SurfaceInput F = nxr::pipelines::read_surface(sub, "helmet_first_coil", true);
    double e2 = 0;
    for (int v = 0; v < G; ++v) for (int d = 0; d < 3; ++d) e2 = std::max(e2, std::abs(F.vertices[static_cast<std::size_t>(3 * v + d)] - stored[static_cast<std::size_t>(3 * pick[static_cast<std::size_t>(v)] + d)]));
    CHECK(e2 < 1e-7);
  }
  // refusals: no pick (the caller's choice, never defaulted); a pick of the coils manifold, not of the channels
  {
    const std::string a = refusal([&] { nxrtest::run_definition(ds, sub, "sensor_surface", Json{{"channels", "sensors"}, {"name", "x1"}}); });
    std::printf("  no pick: %s\n", a.substr(0, 140).c_str());
    CHECK(!a.empty());
    const std::string b = refusal([&] { nxrtest::run_definition(ds, sub, "sensor_surface", Json{{"channels", "sensors"}, {"pick", "three"}, {"name", "x2"}}); });
    std::printf("  a pick of another manifold: %s\n", b.substr(0, 140).c_str());
    CHECK(!b.empty());
  }
  // sensor_mesh: the good channels of one sensor type chosen from the channels' type and quality labellings (meeg.good_channels_of_type), meshed as sensor_surface meshes
  {
    CHECK(!nxrtest::run_definition(ds, sub, "sensor_mesh", Json{{"channels", "sensors"}, {"sensor_type", "MEG"}}).empty());
    const nxr::pipelines::SurfaceInput H = nxr::pipelines::read_surface(sub, "sensors_MEG_surface", true);
    CHECK_EQ(H.n_vertices(), static_cast<std::int64_t>(G - 1));                     // the 150 gradiometers less the bad one
    std::vector<std::int32_t> good_gradiometers;
    for (int i = 0; i < G; ++i) if (i != 17) good_gradiometers.push_back(i);
    const auto map = sub.find("field", nxr::io::task::Subject::of(sub).node_path(H.id, "sensors_MEG_surface_map"));
    CHECK(map.has_value());
    if (map) CHECK(read_as<std::int32_t>(sub, text(*map, "path")) == good_gradiometers);   // ascending, no reference, no electrode, no bad channel
    const auto chosen = sub.find("selection", nxr::io::task::Subject::of(sub).node_path(text(*ch, "id"), "sensors_MEG_surface_channels"));
    CHECK(chosen.has_value());
    if (chosen) CHECK(read_as<std::int32_t>(sub, text(*chosen, "array_path")) == good_gradiometers);
    std::vector<double> positions(H.vertices.begin(), H.vertices.end());
    CHECK(nxr::geometry::pointset::pointSetMesh(positions).faces == H.faces);
    // run again: the stored surface is found (a cache hit), no second surface
    const auto again = nxrtest::run_definition(ds, sub, "sensor_mesh", Json{{"channels", "sensors"}, {"sensor_type", "MEG"}});
    CHECK(!again.empty() && again.front().value("hit", false));
    // refusals: the sensor type left required; three good EEG channels (fewer than four)
    const std::string unnamed = refusal([&] { nxrtest::run_definition(ds, sub, "sensor_mesh", Json{{"channels", "sensors"}}); });
    std::printf("  sensor_mesh, no sensor type: %s\n", unnamed.substr(0, 140).c_str());
    CHECK(unnamed.find("sensor_type") != std::string::npos);
    const std::string few_eeg = refusal([&] { nxrtest::run_definition(ds, sub, "sensor_mesh", Json{{"channels", "sensors"}, {"sensor_type", "EEG"}}); });
    std::printf("  sensor_mesh, three good EEG channels: %s\n", few_eeg.substr(0, 140).c_str());
    CHECK(few_eeg.find("fewer than the four") != std::string::npos);
  }
  // the Hodge-Dirac modes of the helmet (an open disc) through `eigenmodes`, the *1 floor named
  {
    const auto er = nxrtest::run_definition(ds, sub, "eigenmodes", Json{{"surface", "helmet"}, {"modes", 40}, {"star1_floor", 1e-6}});
    CHECK(!er.empty());
    const nxr::pipelines::StoredEigensolve e = nxr::pipelines::stored_eigensolve(sub, "hodgeDirac__helmet");   // the line, the kinds labelling, the one eigenvectors operator to the triangulation
    int exact = 0, coexact = 0;
    for (const auto k : e.kind) { exact += k == 0; coexact += k == 2; }
    std::printf("  helmet modes: %zu (exact %d, coexact %d)\n", e.kind.size(), exact, coexact);
    CHECK(exact > 0 && coexact > 0 && exact + coexact == static_cast<int>(e.kind.size()));
    CHECK(text(e.surface, "id") == S.id && text(e.triangulation, "name") == "helmet_intrinsic");
    CHECK_EQ(integer(e.eigenvectors, "n_rows"), e.V + e.E + e.F);
    const std::string nofloor = refusal([&] { nxrtest::run_definition(ds, sub, "eigenmodes", Json{{"surface", "helmet"}, {"modes", 40}}, true); });
    std::printf("  without a *1 floor: %s\n", nofloor.empty() ? "solved (no negative rim weight)" : nofloor.substr(0, 160).c_str());
  }
  // ── THE SENSOR FLOW on that solve, end to end: its per-member results stored as MEASUREMENTS over the kernel sets' members (the flow consolidation) ──
  // order: the whole-recording flow first (no atoms: its stored chunk spectra make the frequency line, <time line>_freq_L<log2 transform>), then the atoms on that line and the bands on the
  // eigenvalue line (select_kernels), their products (select_kernel_product), then sensor_flow
  {
    const Json serve = Json{{"recording", "rest"}, {"package", "hodgeDirac__helmet"}, {"channel_map", "helmet_map"}, {"overlap_samples", 128}, {"transform_samples", 512},
                            {"frequency_span", Json::array({0.0, 60.0})}};
    std::printf("  [flow] run sensor_flow_bins\n");
    CHECK(!nxrtest::run_definition(ds, sub, "sensor_flow_bins", serve).empty());
    const nxr::pipelines::StoredEigensolve e = nxr::pipelines::stored_eigensolve(sub, "hodgeDirac__helmet");
    std::vector<double> mu;
    for (std::size_t k = 0; k < e.mu.size(); ++k) if (e.kind[k] == 0 && e.mu[k] > 0) mu.push_back(e.mu[k]);
    std::sort(mu.begin(), mu.end());
    const Json bandCentres = Json::array({mu[mu.size() / 4], mu[mu.size() / 2], mu[3 * mu.size() / 4]});
    std::printf("  [flow] run select_kernels\n");
    CHECK(!nxrtest::run_definition(ds, sub, "select_kernels", Json{{"line", "rest_time_freq_L9"}, {"quantity", "frequency"}, {"profile", "gaussian"}, {"chart", "linear"},
                                                                   {"width_rule", "proportional"}, {"width", 0.4}, {"gain", 1.0}, {"centres", Json::array({8.0, 10.0, 12.0})}, {"name", "atoms"}}).empty());
    std::printf("  [flow] run select_kernels\n");
    CHECK(!nxrtest::run_definition(ds, sub, "select_kernels", Json{{"line", text(e.line, "name")}, {"quantity", "wavenumber"}, {"profile", "cos2_partition"}, {"chart", "log2"}, {"centres", bandCentres},
                                                                   {"name", "bands"}}).empty());
    std::printf("  [flow] run select_kernel_product\n");
    CHECK(!nxrtest::run_definition(ds, sub, "select_kernel_product", Json{{"factors", 3}, {"first", "atoms"}, {"second", "bands"}, {"third", "bands"}, {"name", "atom_band_pairs"}}).empty());
    std::printf("  [flow] run select_kernel_product\n");
    CHECK(!nxrtest::run_definition(ds, sub, "select_kernel_product", Json{{"factors", 2}, {"first", "atoms"}, {"second", "bands"}, {"name", "atom_bands"}}).empty());
    // the time cells' stored grid (G7): tiles of 0.125 s (32 samples at 256 Hz: no coarser than the 12 Hz atom's own window) and its product with the atoms and bands
    std::printf("  [flow] run select_time_grid\n");
    CHECK(!nxrtest::run_definition(ds, sub, "select_time_grid", Json{{"recording", "rest"}, {"rule", "tiles"}, {"duration", "0.125 s"}, {"name", "rest tiles"}}).empty());
    std::printf("  [flow] run select_product\n");
    CHECK(!nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"time_grid", "rest tiles"}, {"kernels", "atoms_scales"}, {"atoms", "atoms"}, {"scales", "bands"}, {"name", "rest cells"}}).empty());
    Json flow = serve;
    flow["atoms"] = "atoms"; flow["bands"] = "bands"; flow["pairs"] = "atom_band_pairs"; flow["atom_bands"] = "atom_bands"; flow["patterns"] = 50; flow["name"] = "flow";
    flow["time_grid"] = "rest tiles"; flow["time_atom_bands"] = "rest cells";
    flow["aperiodic"] = true; flow["aperiodic_knee"] = true; flow["aperiodic_range"] = Json::array({4.0, 40.0});   // both aperiodic models in one run
    std::printf("  [flow] run sensor_flow\n");
    CHECK(!nxrtest::run_definition(ds, sub, "sensor_flow", flow).empty());
    std::printf("  [flow] reading the stored measurements\n");
    const nxr::io::testing::Probe probe(sub);
    auto selection_id = [&](const std::string& name) { const auto r = probe.rows("selection", {{"name", V(name)}}); return r.empty() ? std::string() : text(r.front(), "id"); };
    auto measured = [&](const std::string& over, const std::string& measure) { return probe.rows("selection_measurement", {{"selection_id", V(selection_id(over))}, {"measure", V(measure)}}).size(); };
    CHECK(measured("atoms", "phase_current") == 1 && measured("atoms", "power") == 1 && measured("atoms", "flow_coefficient") == 1);         // the whole field, per atom
    // the measures the registry does not know are DECLARED by the nodes that write them: a coefficient sums, a consistency and a p value never do
    auto declared = [&](const std::string& measure) { const auto r = probe.rows("measure_declaration", {{"name", V(measure)}}); return r.empty() ? std::string() : text(r.front(), "merge"); };
    CHECK(declared("flow_coefficient") == "sum" && declared("consistency") == "none" && declared("familywise_p") == "none" && declared("cells") == "sum");
    CHECK(declared("phase_current").empty());                                                                                                // registered: its entry is the registry's
    // both aperiodic models from the one stored power, each its own statistic
    CHECK(probe.rows("statistic", {{"name", V("flow_aperiodic")}}).size() == 1 && probe.rows("statistic", {{"name", V("flow_aperiodic_knee")}}).size() == 1);
    CHECK(measured("atom_band_pairs", "phase_current") == 1 && measured("atom_band_pairs", "power") == 1);                                   // every band pair, per atom
    CHECK(measured("atom_bands", "consistency") == 1 && measured("atom_bands", "threshold") == 1);                                         // the sign-flip test, per atom and band
    // the time cells STORED (G7): sums over the grid's tiles, over grid x atoms x bands, declared sum (a coarser level is a roll-up at read)
    CHECK(measured("rest cells", "phase_current") == 1 && measured("rest cells", "power_sum") == 1 && measured("rest cells", "vertex_phase_current_x") == 1);
    CHECK(declared("vertex_phase_current_x") == "sum");
    // the member axis of each stored measurement is its selection's members, in the product's code order (the first factor fastest: atom + A (band p + Q band q))
    auto members_of = [&](const std::string& over, const std::string& measure) {
      const auto rows = probe.rows("selection_measurement", {{"selection_id", V(selection_id(over))}, {"measure", V(measure)}});
      if (rows.empty()) return std::int64_t(-1);
      return ArrayReader::open(sub.store(), text(rows.front(), "array_path")).shape().at(1);
    };
    auto n_members = [&](const std::string& name) { return integer(probe.rows("selection", {{"name", V(name)}}).front(), "n_members"); };
    CHECK(n_members("atom_band_pairs") == n_members("atoms") * n_members("bands") * n_members("bands"));
    CHECK(members_of("atoms", "phase_current") == n_members("atoms") && members_of("atom_band_pairs", "phase_current") == n_members("atom_band_pairs") &&
          members_of("atom_bands", "consistency") == n_members("atom_bands") && members_of("rest cells", "phase_current") == n_members("rest cells"));
    CHECK(n_members("rest cells") == n_members("rest tiles") * n_members("atom_bands"));                                                   // the grid's tiles x atoms x bands
    // a later run reads the stored flow measurements: the speed and direction across frequencies
    std::printf("  [flow] run sensor_flow_across_frequencies\n");
    CHECK(!nxrtest::run_definition(ds, sub, "sensor_flow_across_frequencies", Json{{"recording", "rest"}, {"package", "hodgeDirac__helmet"}, {"atoms", "atoms"}}).empty());
    CHECK(!probe.rows("statistic", {{"name", V("rest_hodgeDirac__helmet_eigenvectors_across_frequencies_atom_pairs")}}).empty());
    CHECK(probe.rows("statistic", {{"name", V("flow_time_cells")}}).empty() && probe.rows("statistic", {{"name", V("flow_flow_edges")}}).empty());   // no stored cells; no flow statistics

    // ── A ONE-CHUNK CANARY (issue #291): sensor_flow on chunk 1 alone (samples [256, 512) of 2048) on the same stored grid, restricted to its tiles (codes 8..15 of 64, the same codes) ──
    auto cells_of = [&](const std::string& measure) {                         // the stored cells of a measure over "rest cells" [rows x members], and the row's producer
      const auto rows = probe.rows("selection_measurement", {{"selection_id", V(selection_id("rest cells"))}, {"measure", V(measure)}});
      CHECK(rows.size() == 1);
      return std::make_pair(read_as<double>(sub, text(rows.front(), "array_path")), Json::parse(text(rows.front(), "producer_json")));
    };
    const auto [wholeCurrent, wholeProducer] = cells_of("phase_current");
    const auto [wholePower, wholePowerProducer] = cells_of("power_sum");
    CHECK(wholeProducer["run"]["span"].is_object());
    Json canary = flow;
    canary["name"] = "flow_chunk1";
    canary["aperiodic"] = false;
    // the whole run's measurements are there: a span run is another result (another cache key), never a hit on them, and is refused without --replace
    const std::string notReplaced = refusal([&] { nxrtest::run_definition(ds, sub, "sensor_flow", canary, false, "1"); });
    std::printf("  [span] chunk 1 without --replace: %s\n", notReplaced.substr(0, 140).c_str());
    CHECK(notReplaced.find("already exist") != std::string::npos);
    std::printf("  [span] run sensor_flow --chunks 1\n");
    const auto spanRun = nxrtest::run_definition(ds, sub, "sensor_flow", canary, true, "1");
    CHECK(spanRun.size() == 1 && !spanRun[0].value("hit", false));
    const auto [spanCurrent, spanProducer] = cells_of("phase_current");
    const auto [spanPower, spanPowerProducer] = cells_of("power_sum");
    CHECK(spanCurrent.size() == wholeCurrent.size() && spanPower.size() == wholePower.size());   // the same members: the grid's codes, not a new grid
    const std::int64_t tiles = n_members("rest tiles"), groups = n_members("atom_bands");
    CHECK(tiles == 64);
    double worst = 0, scale = 0;
    bool outsideNaN = true, insideFinite = true;
    auto compare = [&](const std::vector<double>& part, const std::vector<double>& whole) {
      const std::int64_t members = tiles * groups, rows = static_cast<std::int64_t>(whole.size()) / members;
      for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t q = 0; q < groups; ++q)
          for (std::int64_t n = 0; n < tiles; ++n) {
            const std::size_t at = static_cast<std::size_t>(r * members + n + tiles * q);
            if (n >= 8 && n < 16) { insideFinite = insideFinite && std::isfinite(part[at]); worst = std::max(worst, std::abs(part[at] - whole[at])); scale = std::max(scale, std::abs(whole[at])); }
            else outsideNaN = outsideNaN && std::isnan(part[at]);
          }
    };
    compare(spanCurrent, wholeCurrent);
    compare(spanPower, wholePower);
    std::printf("  [span] chunk 1 against the whole run on its tiles: %.2e of %.3g\n", worst, scale);
    CHECK(insideFinite && outsideNaN && worst <= 1e-5 * scale);
    // the span in the producer: the run's (the chunk read) and the cells' grid (the tiles measured)
    const Json runSpan = spanProducer["run"]["span"];
    std::printf("  [span] run.span %s\n", runSpan.dump().substr(0, 200).c_str());
    CHECK(runSpan != wholeProducer["run"]["span"]);
    CHECK(spanProducer.dump().find("\"first_tile\":8") != std::string::npos && spanProducer.dump().find("\"whole_line\":false") != std::string::npos);
    // the sign-flip test ran on the span's cells alone and stored its measurements
    CHECK(measured("atom_bands", "consistency") == 1);
    // the same span again: a cache hit
    const auto spanAgain = nxrtest::run_definition(ds, sub, "sensor_flow", canary, false, "1");
    CHECK(spanAgain.size() == 1 && spanAgain[0].value("hit", false));

    // ── THE FIBRE NETWORK FROM `add fibres`, END TO END: tile_network (the graph manifold) on the cortex's tiles, then connectome_flow on the one-chunk span's stored chunk spectra
    // through a free-orientation kernel from the gradiometers to the cortex; its outputs stored ──
    std::printf("  [fibres] the cortex's tiles, a kernel, tile_network, connectome_flow\n");
    CHECK(!nxrtest::make_tiles(ds, sub, "parts 4", std::string("cortex")).empty());
    {
      const nxr::io::task::Subject ts = nxr::io::task::Subject::of(sub);
      const auto cortex = ts.find({.cls = nxr::io::task::Cls::manifold, .name = "cortex", .type = "surface"});
      const auto meg = probe.rows("selection", {{"name", V("sensors_meg")}});
      CHECK(cortex.has_value() && meg.size() == 1);
      const std::int64_t nv = brainSurface.n;
      std::vector<float> K(static_cast<std::size_t>(3 * nv * G));
      std::uint64_t k = 11;
      for (auto& x : K) { k = k * 6364136223846793005ULL + 1442695040888963407ULL; x = static_cast<float>(static_cast<double>(k >> 11) / 9007199254740992.0 - 0.5); }
      sub.compose([&] {
        OperatorSpec op;
        op.name = "kernel"; op.path = "kernel"; op.kind = "inverse kernel"; op.from_manifold_id = text(probe.rows("manifold", {{"name", V("sensors")}}).front(), "id");
        op.layout = "dense"; op.n_rows = 3 * nv; op.n_cols = G; op.data = view(K, {3 * nv, static_cast<std::int64_t>(G)});
        op.cols.set("to_manifold_id", cortex->id).set("from_selection_id", text(meg.front(), "id")).set("from_value_type", "scalar").set("to_value_type", "vector3");
        sub.operator_(std::move(op));
      });
    }
    const Json net = Json{{"fibers", "bundle"}, {"end_surface", "cortex"}, {"tiles", "cortex tiles"}, {"scale", "parts 4"}, {"min_streamlines", 3}, {"name", "net"}};
    CHECK(!nxrtest::run_definition(ds, sub, "tile_network", net).empty());
    CHECK(probe.rows("manifold", {{"name", V("net")}, {"type", V("graph")}}).size() == 1 && probe.rows("field", {{"name", V("net_conductance")}}).size() == 1);
    CHECK(!nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"time_grid", "rest tiles"}, {"kernels", "atoms"}, {"atoms", "atoms"}, {"name", "rest atom cells"}}).empty());
    const Json cf = Json{{"spectra", "flow_chunk1_chunk_spectra"}, {"package", "hodgeDirac__helmet"}, {"channel_map", "helmet_map"}, {"atoms", "atoms"}, {"kernel", "kernel"}, {"tiles", "cortex tiles"},
                         {"network", "net"}, {"scale", "parts 4"}, {"min_streamlines", 3}, {"time_grid", "rest tiles"}, {"time_atoms", "rest atom cells"}, {"patterns", 50}};
    const auto flowRun = nxrtest::run_definition(ds, sub, "connectome_flow", cf);
    CHECK(flowRun.size() == 1);
    std::printf("  [fibres] the network: %lld tiles, %lld edges; connectome_flow stored %zu outputs\n", static_cast<long long>(integer(probe.rows("manifold", {{"name", V("net")}}).front(), "n_vertices")),
                static_cast<long long>(integer(probe.rows("manifold", {{"name", V("net")}}).front(), "n_edges")), flowRun.empty() ? std::size_t(0) : flowRun[0].value("stored", Json::array()).size());
    // its outputs stored: the time cells over grid x atoms (the span's tiles, the others NaN), the per-atom flow, the sign-flip test and the network's readings, all measurements
    CHECK(measured("rest atom cells", "cross_spectrum_real") == 1 && measured("rest atom cells", "power_sum") == 1);
    CHECK(measured("atoms", "mean_edge_flow") == 1 && measured("atoms", "familywise_p") == 1 && measured("atoms", "flow_coefficient") >= 2);   // the test; the sensor's and the network's coefficients
    CHECK(probe.rows("operator", {{"name", V("hodgeDirac__net_eigenvectors")}}).size() == 1);   // the network's modes, made first as their own run
    {
      const auto rows = probe.rows("selection_measurement", {{"selection_id", V(selection_id("rest atom cells"))}, {"measure", V("power_sum")}});
      const std::vector<double> cells = read_as<double>(sub, text(rows.front(), "array_path"));
      const std::int64_t members = n_members("rest atom cells"), T = static_cast<std::int64_t>(cells.size()) / members;
      bool spanOnly = true;
      for (std::int64_t r = 0; r < T; ++r)
        for (std::int64_t m = 0; m < members; ++m) { const std::int64_t n = m % 64; const double v = cells[static_cast<std::size_t>(r * members + m)]; spanOnly = spanOnly && ((n >= 8 && n < 16) ? std::isfinite(v) && v >= 0 : std::isnan(v)); }
      CHECK(T == 8 && spanOnly);                                                                // eight tiles of the network; chunk 1's grid tiles measured
    }
  }
  fs::remove_all(root);
  return nxrtest::finish("test_sensor_surface");
}
