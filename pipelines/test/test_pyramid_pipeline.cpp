// `pyramid build` on a synthetic two-hemisphere brain (never a real dataset): six sensors carrying a 10 Hz current and noise, a constrained-orientation kernel, a bad span, 40 s at 250 Hz.
//   - the rows nxr stores: the tree partition's `atlas` block, the `atlas bands` spans on the Frequency manifold, the tower frame tiling with its `count` row (of the source estimate), the byte-less
//     source estimate (by_operator the kernel, derived_from the recording), the joint Selection (code (tile << B) | frame, layout [band, tile, frame]) and its 3 x bands array-backed measurement rows
//     (power_sum, envelope_sum, envelope_max; within = the bands, code = the band's index); the arrays [band, leaf, frame] and the per-frame good-sample count
//   - the arrays are nxr::pyramid's chain over the whole record in one block (power_partition shape, a 4 s halo: interior frames agree to the filter's halo tolerance)
//   - the result does not depend on the chunk width (the same samples in chunks of 2048 and of 1000, Morlet, interior frames)
//   - one chunk by default: frames outside the span are 0 (maxima -inf) with a count of 0
//   - the free-orientation tensor (a free kernel, the stored canonical frames): rows, arrays tensor/{t11..total}, against the chain over the whole record, a cache hit; refused without canonical frames
//   - refusals: a free-orientation kernel without canonical frames, a rebuild without --replace, an unknown shape, a depth beyond the ladder
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/bst/entities.h"
#include "nxr/io/measures.h"
#include "nxr/io/rollup.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/bst/frequency.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frame_tiling.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/pyramid.h"
#include "nxr/pyramid.h"
#include "nxr/tower.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;
namespace py = nxr::pyramid;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
template <class T> std::vector<T> read_at(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), raw.size());
  return v;
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  // NXR_TEST_KEEP=<folder>: the dataset is made there and kept, for the parity test that has nxr's Python read it (io/test/parity/test_pyramid_parity.py)
  const char* keep = std::getenv("NXR_TEST_KEEP");
  const fs::path root = keep ? fs::path(keep) : fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_pyramid_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; bool keep; ~Cleanup() { if (!keep) fs::remove_all(p); } } cleanup{root, keep != nullptr};
  const Brain b = brain(2, false);
  const int N = b.n, M = 6, T = 10000;
  const double sfreq = 250.0, bad0 = 20.0, bad1 = 20.5;
  std::mt19937 rng(5);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T), kfix(static_cast<std::size_t>(N) * M), kfree(static_cast<std::size_t>(3 * N) * M, 0.0f);
  for (int c = 0; c < M; ++c) { const double amp = 1 + 0.3 * c; for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c) * T + t] = static_cast<float>(amp * std::sin(2 * M_PI * 10.0 * t / sfreq + c) + 0.5 * nd(rng)); }
  for (auto& k : kfix) k = static_cast<float>(1e-3 * nd(rng));
  for (auto& k : kfree) k = static_cast<float>(1e-3 * nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter def = SubjectWriter::create(ds, "default_template", [] { SubjectOptions o; o.kind = "template"; return o; }());
    def.compose([&] {});   // the default subject has NO Frequency manifold: the first build makes it (additive)
  }
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      const std::string surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      std::vector<float> pos(static_cast<std::size_t>(M) * 3, 0.1f);
      ManifoldSpec m;
      m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = M;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "stored"; g.cs_key = "scs"; g.positions_path = "sensor/sensors"; m.geometry = g;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {M, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", {0, 0, 0, 0, 0, 0}, {"MEG MAG"}, {}}, {"quality", {0, 0, 0, 0, 0, 0}, {"good"}, {}}}, cid, "the montage families", std::nullopt,
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
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(2048)), std::make_pair(std::string("rest_b"), std::int64_t(1000))}) {
        FieldSpec rec;
        rec.name = name; rec.path = task::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "T");
        const std::string rid = sub.field(std::move(rec));
        nxr::bst::TimeSpansOptions so2; so2.line_id = tid; so2.of_field_id = rid; so2.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, so2);
      }
      std::vector<std::int32_t> chans = {0, 1, 2, 3, 4, 5};
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      auto kernel = [&](const std::string& name, const std::vector<float>& k, int rows, const char* to) {
        OperatorSpec op;
        op.name = name; op.path = name; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = rows; op.n_cols = M;
        op.data = view(k, {rows, M}); op.array.compress = false;
        op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", to);
        sub.operator_(std::move(op));
      };
      kernel("fixed", kfix, N, "scalar");
      kernel("free", kfree, 3 * N, "vector3");
      kernel("MN_MEG_KERNEL_261005_2149", kfix, N, "scalar");          // a Brainstorm-made kernel's name carries nxr's method and stamp
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  // refusals first (nothing is stored by them)
  // a dataset with no default subject to hold the Frequency manifold is refused BEFORE any chunk is computed (the run never starts), and nothing is stored
  {
    testing::Probe(ds).tamper("UPDATE subject SET kind = 'group' WHERE kind = 'template'");
    bool ran = false;
    PyramidBuildOptions o; o.recording = "rest"; o.kernel = "fixed"; o.chunks = "0";
    CHECK(code_of([&] { pyramid_build(ds, sub, o, [&](const Json& j) { if (j.value("stage", "") == "run") ran = true; }); }) == "no_such_row");
    CHECK(!ran);
    CHECK(testing::Probe(sub).count("selection", {{"name", "atlas bands"}}) == 0);
    testing::Probe(ds).tamper("UPDATE subject SET kind = 'template' WHERE name = 'default_template'");
  }
  CHECK(testing::Probe(ds).count("manifold", {{"name", "Frequency"}}) == 0);
  { PyramidBuildOptions o; o.recording = "rest"; o.kernel = "free"; CHECK(code_of([&] { pyramid_build(ds, sub, o); }) == "invalid_argument"); }
  { PyramidBuildOptions o; o.recording = "rest"; o.kernel = "fixed"; o.shape = "meyer"; CHECK(code_of([&] { pyramid_build(ds, sub, o); }) == "invalid_argument"); }
  { PyramidBuildOptions o; o.recording = "rest"; o.kernel = "fixed"; o.depth = 40; CHECK(code_of([&] { pyramid_build(ds, sub, o); }) == "invalid_argument"); }
  { PyramidBuildOptions o; o.recording = "rest"; o.kernel = "fixed"; o.bands = {{1, 2}, {3, 4}}; o.shape = "power_partition"; CHECK(code_of([&] { pyramid_build(ds, sub, o); }) == "invalid_argument"); }

  // the full build on the power-partition bank, every chunk, a 4 s halo
  PyramidBuildOptions po;
  po.recording = "rest"; po.kernel = "fixed"; po.shape = "power_partition"; po.chunks = "all"; po.halo_seconds = 4.0;
  const PyramidBuildResult r = pyramid_build(ds, sub, po);
  // the first build made the default subject's Frequency manifold and bands; a second run reuses them (no new manifold)
  CHECK(testing::Probe(ds).count("manifold", {{"name", "Frequency"}}) == 1 && testing::Probe(ds).count("selection", {{"name", nxr::bst::kBandsName}}) >= 1);
  const int L = r.info["leaves"].get<int>(), NB = r.info["bands"].get<int>(), F = r.info["frames"].get<int>(), depth = r.info["depth"].get<int>();
  std::printf("  %d leaves (depth %d), %d bands, %d frames\n", L, depth, NB, F);
  CHECK(NB == 6);
  // the rows
  const task::Subject tsub = task::Subject::of(sub);
  const auto est = tsub.find({.cls = task::Cls::field, .kind = "source estimate"});
  CHECK(est && !est->path && est->fact("by_operator_id") == tsub.require({.cls = task::Cls::operator_, .name = "fixed"}, "the fixed operator").id);
  CHECK(est->fact("derived_from_id") == tsub.require({.cls = task::Cls::field, .name = "rest", .kind = "recording"}, "the recording").id);
  const auto part = tsub.find({.cls = task::Cls::selection, .name_like = "cortex tiles L%", .params_present = {"atlas.node"}});
  CHECK(part);
  const Json pp = part->params;
  CHECK(pp["atlas"]["level"] == depth && pp["atlas"]["tree"] == "subject" && pp["atlas"]["node"] == pp["node"] && pp["atlas"]["areas"] == pp["areas"] && pp["atlas"].contains("finest") && pp["atlas"].contains("gauge"));
  const auto bands = tsub.find({.cls = task::Cls::selection, .name = "atlas bands", .type = "spans"});
  CHECK(bands && bands->n_members == 6);
  const auto spans = tsub.spans(*bands);
  CHECK(spans.size() == 6 && spans[3].start == 8.0 && spans[3].stop == 16.0 && spans[3].code == 3);
  const auto frames = tsub.find({.cls = task::Cls::selection, .name = frame_tiling_name("rest", -18)});
  CHECK(frames && frames->type == "spans" && frames->n_members == F);
  const auto cnt_rows = testing::Probe(sub).rows("selection_measurement", {{"selection_id", frames->id}, {"measure", "count"}});
  const auto cnt = cnt_rows.empty() ? std::optional<Row>() : std::optional<Row>(cnt_rows.front());
  CHECK(cnt && text(*cnt, "of_field_id") == est->id);
  const auto joint = tsub.find({.cls = task::Cls::selection, .params_present = {"joint"}});
  CHECK(joint && joint->fact("of_field_id") == est->id && joint->type == "set");
  const Json jp = joint->params;
  CHECK(jp["joint"]["layout"] == Json::array({"band", "tile", "frame"}) && jp["joint"]["tiles"] == L && jp["joint"]["frames"] == F && jp["joint"]["space"] == part->id && jp["joint"]["time"] == frames->id);
  CHECK(joint->n_members == static_cast<std::int64_t>(L) * F && (1 << jp["joint"]["bits"].get<int>()) >= F);
  auto rows = testing::Probe(sub).rows("selection_measurement", {{"selection_id", joint->id}}, "within_code");
  std::stable_sort(rows.begin(), rows.end(), [](const Row& x, const Row& y) { return text(x, "measure") < text(y, "measure"); });
  CHECK(rows.size() == 18);
  for (const Row& m : rows) CHECK(text(m, "within_selection_id") == bands->id && text(m, "of_field_id") == est->id && !is_null(m, "array_path"));
  CHECK(text(rows[0], "measure") == "envelope_max" && text(rows[6], "measure") == "envelope_sum" && text(rows[12], "measure") == "power_sum");
  // the arrays
  std::vector<std::int64_t> shp;
  const std::string tg = jp["atlas"]["node"].get<std::string>();
  const std::vector<double> power = read_at<double>(sub, tg + "/power", &shp), env = read_at<double>(sub, tg + "/env"), envmax = read_at<double>(sub, tg + "/envmax");
  CHECK(shp == (std::vector<std::int64_t>{NB, L, F}));
  const std::vector<std::int64_t> samples = read_at<std::int64_t>(sub, text(*cnt, "array_path"));
  CHECK(static_cast<int>(samples.size()) == F);
  std::int64_t good_total = 0;
  for (auto s : samples) good_total += s;
  CHECK(good_total == T - static_cast<std::int64_t>((bad1 - bad0) * sfreq));            // the bad span carries no count
  // the reference: the chain over the whole record in one block, the leaves and areas as stored
  const std::vector<std::int32_t> codes = read_at<std::int32_t>(sub, part->array_path.value());
  nxr::ladder::Mesh mesh; mesh.n = N; mesh.positions = b.P; mesh.faces = b.F;
  const std::vector<double> area = nxr::ladder::vertexAreas(mesh);
  Eigen::MatrixXd K(N, M), X(T, M);
  for (int v = 0; v < N; ++v) for (int c = 0; c < M; ++c) K(v, c) = kfix[static_cast<std::size_t>(v) * M + c];
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) X(t, c) = xf[static_cast<std::size_t>(c) * T + t];
  std::vector<std::uint8_t> good(T, 1);
  for (int t = 0; t < T; ++t) if (t / sfreq >= bad0 && t / sfreq < bad1) good[t] = 0;
  const nxr::tower::FrameTiling ft = nxr::tower::makeFrameTiling(sfreq, T, -18);
  CHECK(ft.count == F);
  const std::vector<std::int32_t> tile = py::frameOfSamples(ft, 0, T, 0);
  const py::Spectrum spec = py::spectrumOf(X, sfreq);
  const Eigen::MatrixXd gs = py::bandGains(spec.freq, py::octaveBands(), py::Shape::PowerPartition);
  double worst = 0, scale = 0;
  for (int bnd = 0; bnd < NB; ++bnd) {
    nxr::partition::AnalyticSums acc(L, F);
    py::bandSums(acc, K, py::analyticBand(spec, gs.row(bnd).transpose()), good.data(), codes.data(), area.data(), tile.data());
    for (int l = 0; l < L; ++l) for (int f = 16; f < F - 16; ++f) {      // interior frames: 16 frames = 5.3 s from either end, beyond the 4 s halo
      const double ref = acc.power[static_cast<std::size_t>(f) * L + l], got = power[(static_cast<std::size_t>(bnd) * L + l) * F + f];
      worst = std::max(worst, std::abs(ref - got)); scale = std::max(scale, std::abs(ref));
    }
  }
  std::printf("  power: stored vs the chain over the whole record, interior frames: max |diff| / max = %.2e\n", worst / scale);
  CHECK(worst / scale < 5e-3);
  CHECK(!std::isfinite(envmax[0]) || envmax[0] >= 0);
  // THE CACHE (method rule 3): the same options again is a HIT (nothing computed, nothing rewritten); another option is another result, refused unless --replace (the stored name collides)
  CHECK(r.info["cache"] == "miss");
  const std::string est_id0 = est->id, joint_id0 = joint->id;
  const auto ck = est->producer["cache"];
  CHECK(ck["pipeline"] == "pyramid_build" && ck["params"]["shape"] == "power_partition" && ck["params"]["halo_seconds"] == 4.0 && ck["params"]["depth"] == depth && ck["params"].contains("span") && ck["params"].contains("block_samples")
        && ck["params"].contains("bands_hz") && ck["params"].contains("frame_level") && ck["input"].contains("kernel_data") && ck["input"].contains("bad_segments"));
  CHECK(est->producer.contains("halo_seconds"));        // every option is in the producer record too
  const std::vector<double> power_before = power;
  {
    PyramidBuildResult again = pyramid_build(ds, sub, po);
    CHECK(again.info["cache"] == "hit" && again.info["estimate_id"] == est_id0 && again.info["joint_id"] == joint_id0);
    const auto ests2 = tsub.find_all({.cls = task::Cls::field, .kind = "source estimate"});
    CHECK(ests2.size() == 1 && ests2[0].id == est_id0);
    CHECK(read_at<double>(sub, tg + "/power") == power_before);                        // byte for byte
  }
  for (int which = 0; which < 5; ++which) {                                           // each option changes the key: refused without --replace
    PyramidBuildOptions o2 = po;
    if (which == 0) o2.halo_seconds = 3.0;
    if (which == 1) o2.shape = "morlet";
    if (which == 2) o2.exclude_bad_segments = false;
    if (which == 3) o2.block_samples = 256;
    if (which == 4) o2.chunks = "0";
    CHECK(code_of([&] { pyramid_build(ds, sub, o2); }) == "invalid_argument");
  }
  po.replace = true;
  pyramid_build(ds, sub, po);
  CHECK(testing::Probe(ds).count("manifold", {{"name", "Frequency"}}) == 1 && testing::Probe(ds).count("selection", {{"name", "atlas bands"}}) == 1);   // the later runs reuse them
  CHECK(tsub.find_all({.cls = task::Cls::field, .kind = "source estimate"}).size() == 1);
  CHECK(tsub.find_all({.cls = task::Cls::selection, .params_present = {"joint"}}).size() == 1);

  // the chunk width does not matter (Morlet, interior frames): the same samples in chunks of 2048 and 1000
  PyramidBuildOptions a, c2;
  a.recording = "rest"; a.kernel = "fixed"; a.chunks = "all"; a.halo_seconds = 4.0; a.replace = true;
  c2 = a; c2.recording = "rest_b";
  pyramid_build(ds, sub, a);
  pyramid_build(ds, sub, c2);
  auto arrays_of = [&](const std::string& rec) {
    std::optional<task::Node> e;
    for (const task::Node& n : tsub.find_all({.cls = task::Cls::field, .kind = "source estimate"})) if (n.name.rfind(rec + " ", 0) == 0) e = n;
    const auto j = tsub.find({.cls = task::Cls::selection, .of_field_id = e->id, .params_present = {"joint"}});
    return read_at<double>(sub, j->params["atlas"]["node"].get<std::string>() + "/power");
  };
  const std::vector<double> pa = arrays_of("rest"), pb = arrays_of("rest_b");
  double dw = 0, sc2 = 0;
  for (int bnd = 0; bnd < NB; ++bnd) for (int l = 0; l < L; ++l) for (int f = 16; f < F - 16; ++f) {
    const std::size_t i = (static_cast<std::size_t>(bnd) * L + l) * F + f;
    dw = std::max(dw, std::abs(pa[i] - pb[i])); sc2 = std::max(sc2, std::abs(pa[i]));
  }
  std::printf("  Morlet, chunks of 2048 vs 1000 samples (4 s halo), interior frames: %.2e\n", dw / sc2);
  CHECK(dw / sc2 < 1e-3);

  // one chunk by default: the frames outside the span carry nothing
  PyramidBuildOptions d;
  d.recording = "rest"; d.kernel = "fixed"; d.chunks = "0"; d.replace = true; d.halo_seconds = 2.0;
  pyramid_build(ds, sub, d);
  const std::vector<double> pd = arrays_of("rest");
  std::int64_t first_nonzero = -1, last_nonzero = -1;
  for (int f = 0; f < F; ++f) { double s = 0; for (int l = 0; l < L; ++l) s += pd[(static_cast<std::size_t>(0) * L + l) * F + f]; if (s > 0) { if (first_nonzero < 0) first_nonzero = f; last_nonzero = f; } }
  const std::int64_t span_frames = static_cast<std::int64_t>(std::floor(2048 / sfreq / nxr::tower::periodS(-18))) + 2;
  std::printf("  chunk 0 only: frames %lld .. %lld carry power of %d\n", static_cast<long long>(first_nonzero), static_cast<long long>(last_nonzero), F);
  CHECK(first_nonzero == 0 && last_nonzero >= 0 && last_nonzero <= span_frames);
  // the node key as nxr builds it: `<recording>__<method>_<stamp>` from a Brainstorm kernel's name (the stored dtypes: float64 sums, int64 counts; the estimate's data_type float32, as nxr)
  {
    PyramidBuildOptions k;
    k.recording = "rest"; k.chunks = "0"; k.kernel = "MN_MEG_KERNEL_261005_2149"; k.halo_seconds = 2.0;
    pyramid_build(ds, sub, k);
    const auto e = tsub.find({.cls = task::Cls::field, .name = "rest \xc3\x97 MN_MEG_KERNEL_261005_2149"});
    CHECK(e && e->fact("data_type") == "float32" && e->fact("label") == "MN_MEG (constrained)");
    const Json pj = e->producer;
    CHECK(pj["method"] == "MN_MEG" && pj["stamp"] == "261005_2149");
    const auto j = tsub.find({.cls = task::Cls::selection, .of_field_id = e->id, .params_present = {"joint"}});
    const std::string node = j->params["atlas"]["node"].get<std::string>();
    CHECK(node.find("rest__MN_MEG_261005_2149_atlas/subject") != std::string::npos);
    CHECK(ArrayReader::open(sub.store(), node + "/power").dtype() == DType::Float64 && ArrayReader::open(sub.store(), node + "/envmax").dtype() == DType::Float64);
    const auto cnt2_rows = testing::Probe(sub).rows("selection_measurement", {{"of_field_id", e->id}, {"measure", "count"}});
    const auto cnt2 = cnt2_rows.empty() ? std::optional<Row>() : std::optional<Row>(cnt2_rows.front());
    CHECK(cnt2 && text(*cnt2, "array_path").find("rest__MN_MEG_261005_2149_samples") != std::string::npos && ArrayReader::open(sub.store(), text(*cnt2, "array_path")).dtype() == DType::Int64);
  }
  // THE FREE-ORIENTATION TENSOR: refused without the surface's canonical frames; with them, nxr's tensor per band x leaf x frame in the node's tensor/ arrays
  {
    FramesComputeOptions fo;
    frames_compute(ds, sub, fo);
    PyramidBuildOptions f;
    f.recording = "rest"; f.kernel = "free"; f.shape = "power_partition"; f.chunks = "all"; f.halo_seconds = 4.0;
    const PyramidBuildResult fr = pyramid_build(ds, sub, f);
    CHECK(fr.info["orientation"] == "free" && fr.info["arrays"].size() == 10 && fr.info["cache"] == "miss");
    const auto e = tsub.find({.cls = task::Cls::field, .name = "rest \xc3\x97 free"});
    CHECK(e && e->value_type == "vector3" && !e->path && e->fact("label") == "free (free)");
    const auto j = tsub.find({.cls = task::Cls::selection, .of_field_id = e->id, .params_present = {"joint"}});
    CHECK(j);
    const std::string node = j->params["atlas"]["node"].get<std::string>();
    const auto rows2 = testing::Probe(sub).rows("selection_measurement", {{"selection_id", j->id}}, "within_code");
    CHECK(rows2.size() == 60);
    std::set<std::string> measures;
    for (const Row& m : rows2) measures.insert(text(m, "measure"));
    CHECK(measures == (std::set<std::string>{"tensor_11", "tensor_22", "tensor_nn", "tensor_12_re", "tensor_12_im", "tensor_1n_re", "tensor_1n_im", "tensor_2n_re", "tensor_2n_im", "tensor_total"}));
    // the reference: the frames as stored, the kernel projected on them, the chain over the whole record in one block
    const StoredFrames sf = read_stored_frames(sub, "cortex", N);
    CHECK(sf.canonical);
    std::vector<char> sing(N, 0);
    std::vector<double> wt(area.begin(), area.end()), wtot(area.begin(), area.end());
    int n_sing = 0;
    for (int v = 0; v < N; ++v) { for (int k = 0; k < 3; ++k) if (!std::isfinite(sf.e1(v, k))) sing[v] = 1; if (sing[v]) { wt[v] = 0; ++n_sing; } }
    std::printf("  free kernel: %d of %d vertices have no canonical frame\n", n_sing, N);
    Eigen::MatrixXd K3(3 * N, M);
    for (int r = 0; r < 3 * N; ++r) for (int c = 0; c < M; ++c) K3(r, c) = kfree[static_cast<std::size_t>(r) * M + c];
    const py::FrameKernel fk = py::frameKernel(K3, sf.e1, sf.e2, sf.n, sing);
    double worst2 = 0, scale2 = 0, trace_gap = 0;
    std::vector<nxr::partition::TensorSums> ref(NB, nxr::partition::TensorSums(L, F));
    for (int bnd = 0; bnd < NB; ++bnd) py::tensorBandSums(ref[bnd], fk, py::analyticBand(spec, gs.row(bnd).transpose()), good.data(), codes.data(), wt.data(), wtot.data(), tile.data());
    for (int k = 0; k < 10; ++k) {
      const std::string key = nxr::partition::TensorSums::keys()[k];
      const std::vector<double> got = read_at<double>(sub, node + "/tensor/" + key);
      for (int bnd = 0; bnd < NB; ++bnd) for (int l = 0; l < L; ++l) for (int fr2 = 16; fr2 < F - 16; ++fr2) {
        const double rv = (*ref[bnd].arrays()[k])[static_cast<std::size_t>(fr2) * L + l], gv = got[(static_cast<std::size_t>(bnd) * L + l) * F + fr2];
        worst2 = std::max(worst2, std::abs(rv - gv)); scale2 = std::max(scale2, std::abs(rv));
      }
    }
    std::printf("  tensor: stored vs the chain over the whole record, interior frames: max |diff| / max = %.2e\n", worst2 / scale2);
    CHECK(worst2 / scale2 < 5e-3);
    const std::vector<double> t11 = read_at<double>(sub, node + "/tensor/t11"), t22 = read_at<double>(sub, node + "/tensor/t22"), tnn = read_at<double>(sub, node + "/tensor/tnn"), tot = read_at<double>(sub, node + "/tensor/total");
    double s_tr = 0, s_tot = 0;
    for (std::size_t i = 0; i < tot.size(); ++i) { s_tr += t11[i] + t22[i] + tnn[i]; s_tot += tot[i]; }
    trace_gap = (s_tot - s_tr) / s_tot;
    CHECK(trace_gap >= -1e-12 && (n_sing > 0 ? trace_gap > 0 : trace_gap < 1e-9));        // the vertices with no frame are in the total only
    CHECK(code_of([&] { std::vector<std::int64_t> sh; ArrayReader::open(sub.store(), node + "/power"); }) != "");    // a free estimate has no power array
    CHECK(pyramid_build(ds, sub, f).info["cache"] == "hit");
  }
  // THE READ-TIME ROLL-UP in the lookup (step 3): `query_measurements` with the store rolls the array-backed rows up in space (reshape) and time (the tower), by the measure's merge; densities from rolled-up sums
  {
    const auto e = tsub.find({.cls = task::Cls::field, .name = "rest \xc3\x97 fixed"});
    const auto j = tsub.find({.cls = task::Cls::selection, .of_field_id = e->id, .params_present = {"joint"}});
    const Json jp2 = j->params;
    const std::string node = jp2["atlas"]["node"].get<std::string>(), jid = j->id, eid = e->id;
    const auto part2 = tsub.by_id(task::Cls::selection, jp2["joint"]["space"].get<std::string>());
    const auto fr2 = tsub.by_id(task::Cls::selection, jp2["joint"]["time"].get<std::string>());
    const int D = part2->params["atlas"]["level"].get<int>(), Lq = static_cast<int>(part2->n_members.value_or(0));
    const int Fq = static_cast<int>(jp2["joint"]["frames"].get<std::int64_t>());
    const std::int64_t code0 = fr2->params["atlas"]["frame_code0"].get<std::int64_t>();
    const std::vector<double> pw = read_at<double>(sub, node + "/power"), em = read_at<double>(sub, node + "/envmax");
    const std::vector<double> area_leaf = read_at<double>(sub, part2->params["areas"].get<std::string>());
    const auto cnt3 = testing::Probe(sub).rows("selection_measurement", {{"selection_id", fr2->id}, {"measure", "count"}});
    const std::vector<std::int64_t> smp = read_at<std::int64_t>(sub, text(cnt3.front(), "array_path"));
    auto ask = [&](const char* measure, int band, int sl, int tl) {
      std::map<std::pair<std::int64_t, std::int64_t>, double> out;
      for (const Json& r : query_measurements(sub.db(), sub.store(), Json{{"selectionId", jid}, {"measure", measure}, {"ofFieldId", eid}, {"band", band}, {"spaceLevel", sl}, {"timeLevel", tl}}))
        out[{r["code"].get<std::int64_t>(), r["tile"].get<std::int64_t>()}] = r["value"].get<double>();
      return out;
    };
    const int b = 2;
    // brute force: the leaf's node at level lv is l >> (D - lv) (-1: node 0), the frame's tile at level m is (code0 + f) >> m
    auto brute = [&](const std::vector<double>& arr, int lv, int m, bool take_max) {
      std::map<std::pair<std::int64_t, std::int64_t>, double> out;
      for (int l = 0; l < Lq; ++l) for (int f = 0; f < Fq; ++f) {
        const std::pair<std::int64_t, std::int64_t> k{lv < 0 ? 0 : l >> (D - lv), (code0 + f) >> m};
        const double v = arr[(static_cast<std::size_t>(b) * Lq + l) * Fq + f];
        auto it = out.find(k);
        if (it == out.end()) out[k] = v; else it->second = take_max ? std::max(it->second, v) : it->second + v;
      }
      return out;
    };
    for (const auto& [lv, m] : std::vector<std::pair<int, int>>{{D, 0}, {-1, 3}, {2, 4}, {0, 1}, {D - 1, 8}}) {
      const auto got = ask("power_sum", b, lv, m), want = brute(pw, lv, m, false);
      bool ok = got.size() == want.size();
      for (const auto& [k, v] : want) { const auto it = got.find(k); ok = ok && it != got.end() && std::abs(it->second - v) <= 1e-12 * (1 + std::abs(v)); }
      CHECK(ok);
      const auto gm = ask("envelope_max", b, lv, m), wm = brute(em, lv, m, true);          // max merges by max (name ends in _max)
      bool okm = true;
      for (const auto& [k, v] : wm) { if (!std::isfinite(v)) continue; const auto it = gm.find(k); okm = okm && it != gm.end() && std::abs(it->second - v) <= 1e-12 * (1 + std::abs(v)); }
      CHECK(okm);
    }
    // the density: rolled-up power over (rolled-up area x rolled-up samples)
    {
      const int lv = 1, m = 3;
      const auto got = ask("power_sum_density", b, lv, m), sums = brute(pw, lv, m, false);
      std::map<std::int64_t, double> ar;
      for (int l = 0; l < Lq; ++l) ar[l >> (D - lv)] += area_leaf[l];
      std::map<std::int64_t, double> sm;
      for (int f = 0; f < Fq; ++f) sm[(code0 + f) >> m] += static_cast<double>(smp[f]);
      bool ok = true;
      std::size_t finite = 0;
      for (const auto& [k, v] : sums) { const double d = ar[k.first] * sm[k.second]; if (!(d > 0)) { ok = ok && !got.count(k); continue; } ++finite; const auto it = got.find(k); ok = ok && it != got.end() && std::abs(it->second - v / d) <= 1e-12 * (1 + std::abs(v / d)); }
      CHECK(ok && got.size() == finite);
    }
    // io's roll-up is the same arithmetic as compute's (the table forbids sharing one: io never compute): the same arrays, every merge, several anchors
    {
      std::mt19937 r2(9);
      std::normal_distribution<double> g2;
      std::vector<double> x(4 * 150);
      for (auto& v : x) v = g2(r2);
      for (const std::int64_t c0 : {0, 5, 77, 1023})
        for (int m = 0; m < 9; ++m)
          for (int mi = 0; mi < 3; ++mi) {
            std::int64_t t1 = 0, t2 = 0;
            const auto a = nxr::io::rollup::time(x, 4, 150, m, mi == 0 ? nxr::io::rollup::Merge::Sum : mi == 1 ? nxr::io::rollup::Merge::Max : nxr::io::rollup::Merge::Min, c0, &t1);
            const auto c = nxr::tower::rollTime(x, 4, 150, m, mi == 0 ? nxr::tower::Merge::Sum : mi == 1 ? nxr::tower::Merge::Max : nxr::tower::Merge::Min, c0, &t2);
            CHECK(t1 == t2 && a == c);
          }
    }
    // a joint of several bands needs one named; a level finer than the depth is refused
    CHECK(code_of([&] { query_measurements(sub.db(), sub.store(), Json{{"selectionId", jid}, {"measure", "power_sum"}, {"ofFieldId", eid}, {"spaceLevel", 0}}); }) == "invalid_argument");
    CHECK(code_of([&] { query_measurements(sub.db(), sub.store(), Json{{"selectionId", jid}, {"measure", "power_sum"}, {"ofFieldId", eid}, {"band", 1}, {"spaceLevel", D + 1}}); }) == "invalid_argument");
    // the query without a scale is the old one (scalar rows)
    CHECK(query_measurements(sub.db(), sub.store(), Json{{"selectionId", jid}, {"measure", "power_sum"}, {"ofFieldId", eid}}).empty());
  }
  // deleting what has no store path of its own, by NAME: a source estimate and its joint Selection leave no array behind (pyramid arrays, tensor/*, the per-frame counts)
  {
    SubjectWriter s2 = SubjectWriter::open(ds, "S1");
    auto count_named = [&](const char* pat) {
      std::size_t n = 0;
      const fs::path folder = root / "demo" / "S1.nxr.zarr";
      if (!fs::exists(folder)) return n;
      for (const auto& e : fs::recursive_directory_iterator(folder)) if (e.path().string().find(pat) != std::string::npos) ++n;
      return n;
    };
    CHECK(count_named("tensor") > 0 || count_named("_samples") > 0);
    std::vector<std::pair<std::string, std::string>> doomed;   // table, name
    const task::Subject ts2 = task::Subject::of(s2);
    for (const task::Node& n : ts2.find_all({.cls = task::Cls::selection, .name_like = "% \xc3\x97 %", .any_status = true}))
      if ((!n.path || n.path->empty()) && n.n_members.value_or(0) > 0) doomed.push_back({"selection", n.name});
    for (const task::Node& n : ts2.find_all({.cls = task::Cls::field, .kind = "source estimate", .any_status = true})) doomed.push_back({"field", n.name});
    CHECK(!doomed.empty());
    for (const auto& [t, name] : doomed) {
      if (const auto r = s2.by_name(t, name)) s2.remove(t, text(*r, "id"));
    }
    CHECK(ts2.find_all({.cls = task::Cls::field, .kind = "source estimate", .any_status = true}).empty());
    {
      bool tensor_rows = false;
      for (const Row& m : testing::Probe(s2).rows("selection_measurement")) {
        const auto owner = ts2.by_id(task::Cls::selection, text(m, "selection_id"));
        if (owner && owner->subject_id == s2.id() && opt_text(m, "array_path").value_or("").find("tensor") != std::string::npos) tensor_rows = true;
      }
      CHECK(!tensor_rows);
    }
    CHECK(count_named("tensor") == 0);
    CHECK(count_named("_samples") == 0);
  }
  std::printf("test_pyramid_pipeline: ok\n");
  return 0;
}
