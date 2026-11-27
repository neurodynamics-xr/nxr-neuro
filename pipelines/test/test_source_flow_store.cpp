// `run source_flow` STORES its results (issue #292), end to end on a tiny synthetic subject: two 42-vertex spheres stored as one cortex with two hemispheres, each
// hemisphere's Hodge-Dirac modes solved by `run eigenmodes`, a random free-orientation kernel on 20 channels, a 10 Hz recording of 2048 samples in chunks of 256.
//   1. option B per hemisphere, reading the recording, with time cells: the run's own chunk spectra are stored as a statistic whose axes name what they run along (the chunks on
//      the time line, the bins on the overlapped transform's frequency line, the coordinates on the channels), and cortical_flow's single-precision outputs are stored as
//      measurements (they were refused as "gives 0 values"): per atom and per band pair, on each hemisphere (the rows' manifold in the key), the time cells and the sign-flip test;
//   2. the same from the STORED chunk spectra (`--spectra`): the recording is not read, the measurements are stored again (--replace), no spectra stored twice;
//   3. ANOTHER CHANNEL SET (the empty room): a recording on its own channel set (the kernel's channels by name in another order, plus one extra) reads the rest's kernel matched by
//      name (`match: by_name`): its measurements equal the rest's on the same samples, the mapping is in the provenance and the cache key; a set missing a kernel channel is refused.
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <random>
#include <set>

#include "check.h"
#include "mesh_helpers.h"
#include "nxr/bst/entities.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/statistic.h"
#include "nxr/io/subject_api.h"
#include "nxr/io/task/commands.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/pipelines/input.h"
#include "run_definition.h"

using namespace nxr::io;
namespace fs = std::filesystem;

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_source_flow_store_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;

  // ── the cortex: two spheres, the left and right hemispheres of one surface ──
  std::vector<double> U;
  std::vector<std::int32_t> F0;
  icosphere(1, U, F0, 0.03);                                                // 42 vertices, 30 mm
  const std::int64_t nv = static_cast<std::int64_t>(U.size() / 3);
  std::vector<double> pos;
  for (std::int64_t i = 0; i < nv; ++i) { pos.push_back(U[3 * i] - 0.04); pos.push_back(U[3 * i + 1]); pos.push_back(U[3 * i + 2]); }
  for (std::int64_t i = 0; i < nv; ++i) { pos.push_back(U[3 * i] + 0.04); pos.push_back(U[3 * i + 1]); pos.push_back(U[3 * i + 2]); }
  std::vector<std::int32_t> faces = F0;
  for (std::int32_t f : F0) faces.push_back(f + static_cast<std::int32_t>(nv));
  const std::int64_t N = 2 * nv, M = 20, T = 2048;
  const double sfreq = 256.0;

  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    SurfaceOptions so{.name = "cortex"};
    so.hemispheres = {Hemisphere{"left", 0, nv}, Hemisphere{"right", nv, nv}};
    sub.compose([&] { add_surface(sub, so, pos, faces); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  const std::string cortex_id = text(*find_manifold(sub, "cortex"), "id");
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  std::vector<std::string> names;
  for (std::int64_t i = 0; i < M; ++i) names.push_back("C" + std::to_string(i));
  sub.compose([&] {
    // the channels, the time line, the recording (a 10 Hz rhythm in noise), the kernel's pick and the kernel [3 V x channels] on the cortex
    const std::string cid = uuid7();
    std::vector<float> cpos(static_cast<std::size_t>(M * 3));
    for (auto& p : cpos) p = static_cast<float>(nd(rng) * 0.1);
    ManifoldSpec m;
    m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(cpos, {M, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", std::vector<std::int64_t>(M, 0), {"MEG MAG"}, {}}, {"quality", std::vector<std::int64_t>(M, 0), {"good"}, {}}}, cid, "the montage families", std::nullopt,
                        [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
    {
      std::vector<std::int64_t> number(static_cast<std::size_t>(M));
      for (std::int64_t i = 0; i < M; ++i) number[static_cast<std::size_t>(i)] = i;
      nxr::bst::LabellingOptions lo;
      lo.manifold_id = cid; lo.dictionary = "sensors_names"; lo.description = "the channel names";
      nxr::bst::labelling(sub, "sensor/sensors_names", number, names, lo);
    }
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
    for (std::int64_t i = 0; i < M; ++i)
      for (std::int64_t t = 0; t < T; ++t)
        xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(std::cos(2 * M_PI * 10.0 * static_cast<double>(t) / sfreq - 0.3 * static_cast<double>(i)) + 0.1 * nd(rng));
    FieldSpec rec;
    rec.name = "rest"; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
    rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, 256}; rec.array.compress = false; rec.cols.set("unit", "T");
    sub.field(std::move(rec));
    std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
    for (std::int64_t i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
    SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
    const std::string sel = sub.selection(std::move(ss));
    std::vector<float> kf(static_cast<std::size_t>(3 * N * M));
    for (auto& v : kf) v = static_cast<float>(nd(rng) * 1e-3);
    OperatorSpec op;
    op.name = "kern"; op.path = "kern"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
    op.data = view(kf, {3 * N, M}); op.array.compress = false;
    op.cols.set("to_manifold_id", cortex_id).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
    sub.operator_(std::move(op));
  });

  // ── each hemisphere's modes, the frequency line of the overlapped transform (512 samples), the atoms, the bands, their products, the time grid ──
  for (const char* h : {"cortex_1", "cortex_2"}) CHECK(!nxrtest::run_definition(ds, sub, "eigenmodes", Json{{"surface", h}, {"modes", 24}}).empty());
  sub.compose([&] {   // the line the run's chunk spectra lie on (found by its name: the 257 bins of a 512-sample transform at 256 Hz), for the atoms made before the run
    ManifoldSpec fl;
    fl.name = "rest_freq_overlapped"; fl.path = "spectral/rest_freq_overlapped"; fl.type = "line"; fl.n_vertices = 257;
    fl.topology = {"lattice", 1, 1, std::nullopt, {}};
    const auto flc = lattice_counts({257});
    if (flc.n_edges) fl.cols.set("n_edges", *flc.n_edges);
    GeometrySpec fg; fg.form = "closed_form"; fg.cs_key = "hz";
    DimensionSpec fd; fd.name = "frequency"; fd.n_vertices = 257; fd.spacing = sfreq / 512.0; fd.origin = 0.0; fd.unit = "Hz";
    fg.dimensions.push_back(fd);
    fl.geometry = fg;
    sub.manifold(std::move(fl));
  });
  const nxr::pipelines::StoredEigensolve e = nxr::pipelines::stored_eigensolve(sub, "hodgeDirac__cortex_1");
  std::vector<double> mu;
  for (std::size_t k = 0; k < e.mu.size(); ++k) if (e.kind[k] == 0 && e.mu[k] > 0) mu.push_back(e.mu[k]);
  std::sort(mu.begin(), mu.end());
  CHECK(mu.size() >= 4);
  const Json bandCentres = Json::array({mu[mu.size() / 4], mu[mu.size() / 2], mu[3 * mu.size() / 4]});
  CHECK(!nxrtest::run_definition(ds, sub, "select_kernels", Json{{"line", "rest_freq_overlapped"}, {"quantity", "frequency"}, {"profile", "gaussian"}, {"chart", "linear"},
                                                                 {"width_rule", "proportional"}, {"width", 0.4}, {"gain", 1.0}, {"centres", Json::array({8.0, 10.0, 12.0})}, {"name", "atoms"}}).empty());
  CHECK(!nxrtest::run_definition(ds, sub, "select_kernels", Json{{"line", text(e.line, "name")}, {"quantity", "wavenumber"}, {"profile", "cos2_partition"}, {"chart", "log2"}, {"centres", bandCentres},
                                                                 {"name", "bands"}}).empty());
  CHECK(!nxrtest::run_definition(ds, sub, "select_kernel_product", Json{{"factors", 3}, {"first", "atoms"}, {"second", "bands"}, {"third", "bands"}, {"name", "atom_band_pairs"}}).empty());
  CHECK(!nxrtest::run_definition(ds, sub, "select_kernel_product", Json{{"factors", 2}, {"first", "atoms"}, {"second", "bands"}, {"name", "atom_bands"}}).empty());
  CHECK(!nxrtest::run_definition(ds, sub, "select_time_grid", Json{{"recording", "rest"}, {"rule", "tiles"}, {"duration", "0.125 s"}, {"name", "rest tiles"}}).empty());
  CHECK(!nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"time_grid", "rest tiles"}, {"kernels", "atoms"}, {"atoms", "atoms"}, {"name", "rest whole cells"}}).empty());
  CHECK(!nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"time_grid", "rest tiles"}, {"kernels", "atoms_scales"}, {"atoms", "atoms"}, {"scales", "bands"}, {"name", "rest cells"}}).empty());

  const nxr::io::testing::Probe probe(sub);
  auto id_of = [&](const char* cls, const std::string& name) { const auto r = probe.rows(cls, {{"name", V(name)}}); return r.empty() ? std::string() : text(r.front(), "id"); };
  auto measured = [&](const std::string& over, const std::string& measure) { return probe.rows("selection_measurement", {{"selection_id", V(id_of("selection", over))}, {"measure", V(measure)}}); };

  // ── 1. option B per hemisphere, reading the recording, with the time cells ──
  const Json run = Json{{"recording", "rest"}, {"kernel", "kern"}, {"cortex", "per_hemisphere"}, {"left", "hodgeDirac__cortex_1"}, {"right", "hodgeDirac__cortex_2"},
                        {"columns", "channels"}, {"atoms", "atoms"}, {"bands", "bands"}, {"pairs", "atom_band_pairs"}, {"atom_bands", "atom_bands"},
                        {"overlap_samples", 128}, {"transform_samples", 512}, {"frequency_span", Json::array({0.0, 60.0})}, {"frequency_normalisation", "none"},
                        {"time_cells", true}, {"time_grid", "rest tiles"}, {"time_atoms", "rest whole cells"}, {"time_atom_bands", "rest cells"}, {"patterns", 50}, {"name", "sf"}};
  std::string why;
  try { CHECK(!nxrtest::run_definition(ds, sub, "source_flow", run).empty()); } catch (const std::exception& x) { why = x.what(); }
  std::printf("  source_flow, reading the recording: %s\n", why.empty() ? "stored" : why.substr(0, 300).c_str());
  CHECK(why.empty());
  {   // the chunk spectra: a statistic whose three axes name what they run along
    const StoredStatistic st = read_statistic(sub, "sf_chunk_spectra");
    CHECK(st.axes.size() == 3);
    if (st.axes.size() == 3) {
      CHECK(st.axes[0]["name"] == "chunk" && st.axes[0]["values"].size() == 8 && st.axes[0]["values"][0]["manifold_id"] == id_of("manifold", "rest_time"));   // each chunk a span of the time line
      CHECK(st.axes[1]["name"] == "frequency" && st.axes[1].contains("manifold_id"));
      CHECK(st.axes[2]["name"] == "coordinate" && st.axes[2].contains("manifold_id") && st.axes[2]["manifold_id"] == id_of("manifold", "sensors"));
      CHECK(st.axes[2]["members"].size() == static_cast<std::size_t>(M) && st.axes[2]["members"][3] == 3);
    }
  }
  // cortical_flow's float32 measurements: one per hemisphere (the rows' manifold in the key), per atom and per band pair
  for (const char* measure : {"power", "phase_current", "flow_coefficient"}) {
    CHECK(measured("atoms", measure).size() == 2);
    CHECK(measured("atom_band_pairs", measure).size() == 2);
  }
  {
    const auto rows = measured("atoms", "power");
    std::set<std::string> on;
    for (const auto& r : rows) if (!is_null(r, "rows_manifold_id")) on.insert(text(r, "rows_manifold_id"));
    CHECK(on.size() == 2);
    for (const auto& r : rows) {   // the values are stored and finite: [rows x members x components]
      const ArrayReader ar = ArrayReader::open(sub.store(), text(r, "array_path"));
      CHECK(ar.shape().size() == 3 && ar.shape()[1] == 3);
      const auto raw = ar.read();
      CHECK(!raw.empty());
      bool finite = true, nonzero = false;
      std::vector<double> vals;
      if (ar.dtype() == DType::Float32) { std::vector<float> f(raw.size() / 4); std::memcpy(f.data(), raw.data(), raw.size()); vals.assign(f.begin(), f.end()); }
      else { vals.resize(raw.size() / 8); std::memcpy(vals.data(), raw.data(), raw.size()); }
      for (double x : vals) { finite = finite && std::isfinite(x); nonzero = nonzero || x != 0.0; }
      CHECK(finite && nonzero);
    }
  }
  CHECK(measured("rest cells", "phase_current").size() == 2 && measured("rest whole cells", "phase_current").size() == 2);   // the time cells, each hemisphere
  CHECK(measured("atom_bands", "consistency").size() == 2);                                                                   // the sign-flip test per atom and band

  // ── 2. the same from the stored chunk spectra: the recording is not read, nothing stored twice ──
  Json again = run;
  again["spectra"] = "sf_chunk_spectra";
  again["time_cells"] = false;
  why.clear();
  try { CHECK(!nxrtest::run_definition(ds, sub, "source_flow", again, true).empty()); } catch (const std::exception& x) { why = x.what(); }
  std::printf("  source_flow --spectra: %s\n", why.empty() ? "stored" : why.substr(0, 300).c_str());
  CHECK(why.empty());
  CHECK(probe.rows("statistic", {{"name", V("sf_chunk_spectra")}}).size() == 1);
  CHECK(measured("atoms", "power").size() == 2 && measured("atom_band_pairs", "phase_current").size() == 2);

  // ── 3. another channel set: the empty room on its own channels, the rest's kernel matched by name ──
  // the empty room's set: the rest's names reversed, with an extra channel X1 in the middle; its samples the rest's rows by name (X1 noise), so its flow is the rest's; the short set lacks C3
  std::vector<std::string> er_names(names.rbegin(), names.rend());
  er_names.insert(er_names.begin() + 7, "X1");
  std::vector<std::string> short_names;
  for (const auto& n : names) if (n != "C3") short_names.push_back(n);
  auto channel_set = [&](const std::string& set, const std::vector<std::string>& set_names, const std::string& recording) {
    const std::int64_t n = static_cast<std::int64_t>(set_names.size());
    const std::string cid = uuid7();
    std::vector<float> cpos(static_cast<std::size_t>(n * 3));
    for (auto& p : cpos) p = static_cast<float>(nd(rng) * 0.1);
    ManifoldSpec m;
    m.id = cid; m.name = set; m.path = "sensor/" + set; m.type = "points"; m.n_vertices = n;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/" + set; m.geometry = g;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/" + set, view(cpos, {n, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    nxr::bst::label_set(sub, "sensor/" + set + "_labels", {{"type", std::vector<std::int64_t>(n, 0), {"MEG MAG"}, {}}, {"quality", std::vector<std::int64_t>(n, 0), {"good"}, {}}}, cid, "the montage families",
                        std::nullopt, [](const std::string& fam) { return std::make_pair(fam, std::string("app")); });
    std::vector<std::int64_t> number(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) number[static_cast<std::size_t>(i)] = i;
    nxr::bst::LabellingOptions lo;
    lo.manifold_id = cid; lo.dictionary = set + "_names"; lo.description = "the channel names";
    nxr::bst::labelling(sub, "sensor/" + set + "_names", number, set_names, lo);
    ManifoldSpec tm;
    tm.name = recording + "_time"; tm.path = "time/" + recording + "_time"; tm.type = "line"; tm.n_vertices = T;
    tm.topology = {"lattice", 1, 1, std::nullopt, {}};
    const auto lc = lattice_counts({T});
    if (lc.n_edges) tm.cols.set("n_edges", *lc.n_edges);
    GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
    DimensionSpec d; d.name = "time"; d.n_vertices = T; d.spacing = 1.0 / sfreq; d.origin = 0.0; d.unit = "s";
    tg.dimensions.push_back(d);
    tm.geometry = tg;
    const std::string tid = sub.manifold(std::move(tm));
    const std::string prod = sub.product({cid, tid});
    std::vector<float> rf(static_cast<std::size_t>(n * T));
    for (std::int64_t i = 0; i < n; ++i) {
      const auto it = std::find(names.begin(), names.end(), set_names[static_cast<std::size_t>(i)]);
      for (std::int64_t t = 0; t < T; ++t)
        rf[static_cast<std::size_t>(i * T + t)] = it == names.end() ? static_cast<float>(nd(rng)) : xf[static_cast<std::size_t>((it - names.begin()) * T + t)];
    }
    FieldSpec rec;
    rec.name = recording; rec.path = nxr::io::task::Subject::of(sub).node_path(prod, recording); rec.kind = "recording"; rec.manifold_id = prod;
    rec.data = view(rf, {n, T}); rec.array.chunks = Shape{n, 256}; rec.array.compress = false; rec.cols.set("unit", "T");
    sub.field(std::move(rec));
    ManifoldSpec fl;   // the line its chunk spectra lie on, as the rest's
    fl.name = recording + "_freq_overlapped"; fl.path = "spectral/" + recording + "_freq_overlapped"; fl.type = "line"; fl.n_vertices = 257;
    fl.topology = {"lattice", 1, 1, std::nullopt, {}};
    const auto flc = lattice_counts({257});
    if (flc.n_edges) fl.cols.set("n_edges", *flc.n_edges);
    GeometrySpec fg; fg.form = "closed_form"; fg.cs_key = "hz";
    DimensionSpec fd; fd.name = "frequency"; fd.n_vertices = 257; fd.spacing = sfreq / 512.0; fd.origin = 0.0; fd.unit = "Hz";
    fg.dimensions.push_back(fd);
    fl.geometry = fg;
    sub.manifold(std::move(fl));
  };
  sub.compose([&] { channel_set("emptyroom_sensors", er_names, "emptyroom"); channel_set("short_sensors", short_names, "short"); });
  Json er = run;
  er["recording"] = "emptyroom";
  er["time_cells"] = false;
  er["name"] = "er";
  std::vector<Json> er_report;
  why.clear();
  try { er_report = nxrtest::run_definition(ds, sub, "source_flow", er); } catch (const std::exception& x) { why = x.what(); }
  std::printf("  source_flow on the empty room, the rest's kernel by name: %s\n", why.empty() ? "stored" : why.substr(0, 400).c_str());
  CHECK(why.empty());
  {   // the mode operator is the kernel's (stored by the rest's run): not stored again by the empty room's, and said so
    const std::string report = er_report.empty() ? std::string() : er_report.front().dump();
    CHECK(report.find("\"output\":\"left_operator\",\"stored\":false") != std::string::npos);
    CHECK(probe.rows("operator", {{"name", V("kern__in__hodgeDirac__cortex_1_eigenvectors")}}).size() <= 1);
  }
  {
    const std::string rest_id = id_of("field", "rest"), er_id = id_of("field", "emptyroom");
    auto of = [&](const std::string& measure, const std::string& field) {
      std::vector<Row> out;
      for (const auto& r : measured("atoms", measure)) if (opt_text(r, "of_field_id").value_or("") == field) out.push_back(r);
      std::sort(out.begin(), out.end(), [](const Row& a, const Row& b) { return text(a, "rows_manifold_id") < text(b, "rows_manifold_id"); });
      return out;
    };
    auto values = [&](const Row& r) {
      const ArrayReader ar = ArrayReader::open(sub.store(), text(r, "array_path"));
      const auto raw = ar.read();
      std::vector<double> v;
      if (ar.dtype() == DType::Float32) { std::vector<float> f(raw.size() / 4); std::memcpy(f.data(), raw.data(), raw.size()); v.assign(f.begin(), f.end()); }
      else { v.resize(raw.size() / 8); std::memcpy(v.data(), raw.data(), raw.size()); }
      return v;
    };
    for (const char* measure : {"power", "phase_current", "flow_coefficient"}) {
      const auto a = of(measure, rest_id), b = of(measure, er_id);
      CHECK(a.size() == 2 && b.size() == 2);
      if (a.size() != 2 || b.size() != 2) continue;
      double worst = 0;
      for (std::size_t h = 0; h < 2; ++h) {
        CHECK(text(a[h], "rows_manifold_id") == text(b[h], "rows_manifold_id"));
        const auto x = values(a[h]), y = values(b[h]);
        CHECK(x.size() == y.size() && !x.empty());
        double scale = 0, diff = 0;
        for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) { scale = std::max(scale, std::abs(x[i])); diff = std::max(diff, std::abs(x[i] - y[i])); }
        worst = std::max(worst, diff / std::max(scale, 1e-300));
      }
      std::printf("  %-16s empty room vs rest (same samples by name): max relative difference %.2e\n", measure, worst);
      CHECK(worst < 1e-5);                                                  // float32 measurements of the same samples, the channels in the same order
    }
    // the mapping, in the provenance of what the run stored
    const auto b = of("power", er_id);
    CHECK(!b.empty());
    if (!b.empty()) {
      const Json pj = Json::parse(opt_text(b.front(), "producer_json").value_or("{}"), nullptr, false);
      std::string where = pj.dump();
      const bool named = where.find("\"matched_to\":\"emptyroom_sensors\"") != std::string::npos && where.find("\"unused\":[\"X1\"]") != std::string::npos &&
                         where.find("\"channels\":\"sensors\"") != std::string::npos;
      std::printf("  the provenance names the match: %s\n", named ? "yes" : where.substr(0, 300).c_str());
      CHECK(named);
    }
    // a rerun is a hit (both channel sets in the key)
    std::string again_why;
    std::vector<Json> rerun;
    try { rerun = nxrtest::run_definition(ds, sub, "source_flow", er); } catch (const std::exception& x) { again_why = x.what(); }
    CHECK(again_why.empty());
    bool hit = !rerun.empty();
    for (const auto& r : rerun) hit = hit && r.value("hit", false);
    std::printf("  rerun: %s\n", hit ? "a hit" : (again_why.empty() ? (rerun.empty() ? std::string("nothing") : rerun.front().dump().substr(0, 300)) : again_why.substr(0, 300)).c_str());
    CHECK(hit);
  }
  {   // a kernel channel the recording does not hold: refused, naming it
    Json sh = er;
    sh["recording"] = "short";
    sh["name"] = "short";
    std::string refused;
    try { nxrtest::run_definition(ds, sub, "source_flow", sh); } catch (const std::exception& x) { refused = x.what(); }
    std::printf("  a set missing C3: %s\n", refused.empty() ? "NOT refused" : refused.substr(0, 300).c_str());
    CHECK(refused.find("C3") != std::string::npos && refused.find("short_sensors") != std::string::npos);
  }
  return nxrtest::finish("test_source_flow_store");
}
