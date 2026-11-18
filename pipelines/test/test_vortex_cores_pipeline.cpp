// THE VORTEX CORES' TWO PASSES (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 5; docs/specs/2026-11-14-spatial-events.md, decisions 1 and 5), run on a synthetic two-hemisphere source space (never a real dataset):
// a free kernel of random currents, a recording of a 10 Hz oscillation plus noise, the frame tiling of the 8-12 Hz band (36 frames a cycle), a window of "bursts" with a bad span inside it.
//   - `run vortex_core_threshold`: the distribution of |hatted curl| over the burst frames (inside the window, outside the bad span) stored as a STATISTIC (measure count, method histogram): its range fixed from the
//     first 3000 good frames (the window holds more), its counts equal the DIRECT histogram of the direct field (nxr::stats::histogram on M Re(Z) sampled at the frames)
//   - `run vortex_cores`: reads that statistic BY NAME as an input, derives the 98th percentile, and stores the cores: a spans Selection with ONE MEMBER PER CORE on its frame (several members share a frame), and the
//     attributes (frame, sense, centre_vertex, centre, area, peak) as measurements over it, all equal to nxr::select::corePatches called directly on the direct field with the same threshold
//   - the result does not depend on the chunk size: the same samples stored in chunks of 4096 and of 1000 give the same distribution and the same cores
//   - the percentile is derived at read: another percentile is another threshold from the same stored counts (a lower one finds more cores); a rerun is a hit
//   - the TRACKS (piece 6): `vortex_cores` links the cores frame to frame (`tracks`, after the chunk loop) and stores a spans Selection `vortex_cores.tracks.<recording>` (one member per track of at least `min_frames`
//     frames, its span first to last core's frame) with first_frame, last_frame, n_frames, path_length, displacement, sense as measurements, and the labelling `track` (a measurement over the cores); all equal to
//     nxr::select::link called directly on the stored cores; the same tracks from chunks of 1000 and 4096; a longer link distance joins more; min_frames 1 stores the lone cores too; a rerun is a hit
//   - the CROSSINGS (piece 7): `tile_crossings` stores ONE measurement over the level's tiles (rows stage x sense, members the tiles then `bridge`, components the vector sum then the count), equal to nxr::select::track_crossings::crossings
//     called directly on the stored tracks' steps with the frames and the transport made here; R and the unbiased rho2 are derived at read from them; a rerun is a hit; a stage named without its Selection is refused; `repeat compute --stage NAME` reads the stage names the rows carry
//   - the RUN-TO-RUN REPEAT (piece 8): `repeat compute` over the two recordings' stored crossings (here the same samples cut in other chunks: the same motion twice) stores the per-tile dot products and the count-weighted mean
//     cosine as two statistics, equal to nxr::select::track_crossings::repeat on the stored arrays, the cosine 1; with one run's vectors reversed it is -1; a rerun is a hit; the same recording twice, a missing run and a stage out of range
//     are refused
//   - the plan: a threshold made from another chain (the spatial shape) is refused, a field not on the frame tiling is refused, a percentile out of range is refused, a threshold that is no histogram is refused
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "psd_run.h"
#include "check.h"
#include "portable_random.h"
#include "store_at.h"
#include "nxr/bst/entities.h"
#include "nxr/select/cores.h"
#include "nxr/signal/graphfilterbank.h"
#include "nxr/signal/forms.h"
#include "nxr/stats/histogram.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/io/yaml.h"
#include "nxr/select/dyadic.h"
#include "nxr/pipelines/eigen.h"
#include "run_definition.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/signal/filters_fir.h"
#include "nxr/select/tracks.h"
#include "nxr/select/track_crossings.h"
#include "nxr/geometry/charts.h"
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/frames.h"
#include "nxr/io/measures.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/run_repeat.h"
#include "measurement_helpers.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/surrogate_compare.h"
#include "nxr/stats/nulltest.h"
#include "make_grids.h"

// the frames of the 8-12 Hz band: the definition `select_time_grid` rule frames gives 36 frames to a cycle of the band's centre (the geometric mean of its ends), exactly (not a power-of-two tile)
constexpr double kFramesPerCycle = 36.0;

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace tk = nxr::io::task;
namespace fs = std::filesystem;

namespace {
// the run-to-run repeat is the definition `run_repeat`: a thin local caller keeps the checks' vocabulary (a run pair, a stage by index or by name, the tiles needed)
struct RunRepeatOptions { std::string recording_a, recording_b, name; int min_crossings = 10, stage = 0; std::optional<std::string> stage_name; };
struct RunRepeatResult { bool hit = false; Json info; };
template <class F> std::string what_of(F f) { try { f(); } catch (const Error& e) { return e.what(); } return ""; }
std::vector<float> read_floats(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
// a stored measurement over the cores: its component `component` for every member
std::vector<float> measure_of(SubjectWriter& sub, const std::string& selection_id, const std::string& measure) {
  const auto rr = nxr::io::testing::Probe(sub).rows("selection_measurement", {{"selection_id", V(selection_id)}, {"measure", V(measure)}});
  if (rr.empty()) return {};
  const auto r = std::optional<Row>(rr.front());
  std::vector<std::int64_t> shape;
  const std::vector<float> a = read_floats((sub.folder() / (text(*r, "array_path"))), &shape);
  const std::int64_t comp = integer(*r, "component"), T = shape[1], C = shape[2];
  std::vector<float> out(static_cast<std::size_t>(T));
  for (std::int64_t t = 0; t < T; ++t) out[static_cast<std::size_t>(t)] = a[static_cast<std::size_t>(t * C + comp)];
  return out;
}
struct Stored {
  std::string id;
  std::vector<std::int64_t> frame;
  std::vector<double> start, stop;
  std::vector<float> f, sense, cv, cx, cy, cz, area, peak;
  Json params;
};
Stored read_cores(SubjectWriter& sub, const std::string& recording, double fps, double origin) {
  Stored s;
  const auto ts = tk::Subject::of(sub);
  const auto row = ts.find({.cls = tk::Cls::selection, .name = "vortex_cores.cores." + recording});
  if (!row) return s;
  s.id = row->id;
  s.params = row->params.is_null() ? Json::object() : row->params;
  for (const auto& r : ts.spans(*row)) {
    s.start.push_back(r.start); s.stop.push_back(r.stop);
    s.frame.push_back(static_cast<std::int64_t>(std::llround((s.start.back() - origin) * fps)));
  }
  s.f = measure_of(sub, s.id, "frame"); s.sense = measure_of(sub, s.id, "sense"); s.cv = measure_of(sub, s.id, "centre_vertex");
  s.cx = measure_of(sub, s.id, "centre_x"); s.cy = measure_of(sub, s.id, "centre_y"); s.cz = measure_of(sub, s.id, "centre_z");
  s.area = measure_of(sub, s.id, "area"); s.peak = measure_of(sub, s.id, "peak");
  return s;
}
struct StoredTracks {
  std::string id;
  std::vector<double> start, stop;
  std::vector<float> first, last, n, path, disp, sense;
  Json params;
};
StoredTracks read_tracks(SubjectWriter& sub, const std::string& recording) {
  StoredTracks t;
  const auto ts = tk::Subject::of(sub);
  const auto row = ts.find({.cls = tk::Cls::selection, .name = "vortex_cores.tracks." + recording});
  if (!row) return t;
  t.id = row->id;
  t.params = row->params.is_null() ? Json::object() : row->params;
  for (const auto& r : ts.spans(*row)) { t.start.push_back(r.start); t.stop.push_back(r.stop); }
  t.first = measure_of(sub, t.id, "first_frame"); t.last = measure_of(sub, t.id, "last_frame"); t.n = measure_of(sub, t.id, "n_frames");
  t.path = measure_of(sub, t.id, "path_length"); t.disp = measure_of(sub, t.id, "displacement"); t.sense = measure_of(sub, t.id, "sense");
  return t;
}
Json set_of(std::initializer_list<std::pair<const char*, Json>> kv) { Json j = Json::object(); for (const auto& [k, v] : kv) j[k] = v; return j; }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_vortex_cores_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = 12, T = 4096, level = 3;
  const double sfreq = 250.0, w0 = 2.0, w1 = 14.0, bad0 = 5.0, bad1 = 5.5;
  std::mt19937 rng(11);
  nxr::testing::PortableNormal nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T), kfree(static_cast<std::size_t>(3 * N) * M);
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(c) * T + t] = static_cast<float>(std::cos(2 * M_PI * 10.0 * t / sfreq + 0.7 * c) + 0.3 * nd(rng));
  for (auto& k : kfree) k = static_cast<float>(nd(rng));
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, first_rid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
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
      { const std::vector<std::int64_t> zeros(static_cast<std::size_t>(M), 0);            // the channels' families (the PSD and the cross-spectrum need them): every channel a good MEG MAG
        nxr::bst::label_set(sub, "sensor/sensors_labels", {{"type", zeros, {"MEG MAG"}, {}}, {"quality", zeros, {"good"}, {}}}, cid, "the montage families", std::nullopt,
                            [](const std::string& fam) { return std::make_pair(fam, std::string("app")); }); }
      ManifoldSpec tm;
      tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
      tm.topology = {"lattice", 1, 1, std::nullopt, {}};
      tm.cols.set("n_edges", T - 1);
      GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
      DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / sfreq; dm.origin = 0.0; dm.unit = "s";
      tg.dimensions.push_back(dm); tm.geometry = tg;
      tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(4096)), std::make_pair(std::string("rest_b"), std::int64_t(1000))}) {
        FieldSpec rec;
        rec.name = name; rec.path = tk::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "A");
        const std::string rid = sub.field(std::move(rec));
        if (first_rid.empty()) first_rid = rid;
        nxr::bst::TimeSpansOptions bo; bo.line_id = tid; bo.of_field_id = rid; bo.description = "bad segments";
        nxr::bst::time_spans(sub, "time/" + name + "_bad_segments", {{bad0, bad1, 0}}, {"BAD"}, bo);
      }
      nxr::bst::TimeSpansOptions wo; wo.line_id = tid; wo.of_field_id = first_rid; wo.description = "the bursts";
      nxr::bst::time_spans(sub, "time/window", {{w0, w1, 0}}, {"burst"}, wo);
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      OperatorSpec op;
      op.name = "free"; op.path = "free"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 3 * N; op.n_cols = M;
      op.data = view(kfree, {3 * N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "vector3");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  // the line's one time grid is the frames of the band below (a definition finds `<recording> time grid`): no constant-Q grid beside it
  nxrtest::make_tiles(ds, sub);
  { EigenSolveOptions eo; eo.modes = 100; eo.surface = "cortex"; eigen_solve(ds, sub, eo, {}); }          // stores the glued surface the curl lives on
  for (const char* rec : {"rest", "rest_b"}) { nxrtest::make_frames(ds, sub, rec, "8-12 Hz"); }
  const double fps = kFramesPerCycle * std::sqrt(8.0 * 12.0);

  // ── THE DIRECT FIELD: the band's real signal (Morlet taps), the folded operator M = F_l form(K) on the glued surface (dense exact hat), M Re of the band's analytic signal read at the frames by wt ──
  const SurfaceInput S = read_surface(sub, std::string("cortex"), true);
  const SurfaceLevel SL = read_surface_level(sub, S, level);
  SpatialBandOptions sbo; sbo.transform = "exact"; sbo.spatial_shape = "mexhat"; sbo.glued = true;
  const SpatialBand sband = make_spatial_band(ds, sub, S, SL, sbo);
  const std::int64_t Vg = sband.glued_vertices;
  const auto gain = [&](double f) { return nxr::signal::graphfilterbank::morletBand(f, 8.0, 12.0); };
  const auto h = nxr::signal::gainTaps(gain, sfreq, false);
  const auto re = nxr::signal::convolveRows(xf.data(), M, T, h, 0, T, false);
  Eigen::MatrixXd Re(M, T);                                                      // the band-passed real signal (taps), for the power check below
  for (int c = 0; c < M; ++c) for (int t = 0; t < T; ++t) Re(c, t) = re[static_cast<std::size_t>(c) * T + t];
  const SurfaceInput S32 = read_surface(sub, std::string("cortex"));
  std::vector<float> P32(S32.vertices.begin(), S32.vertices.end());
  const auto fop = nxr::signal::assembleFaceGradient(P32.data(), S32.faces.data(), static_cast<int>(S32.n_vertices()), static_cast<int>(S32.n_faces()));
  Eigen::MatrixXd Kfree(3 * N, M);
  for (int r = 0; r < 3 * N; ++r) for (int c = 0; c < M; ++c) Kfree(r, c) = kfree[static_cast<std::size_t>(r) * M + c];
  const Eigen::MatrixXd Mop = sband.apply_glued(nxr::signal::formOfFreeColumns(nxr::signal::Form::curl_tan, Kfree, fop));
  const std::int64_t count = static_cast<std::int64_t>(std::floor(static_cast<double>(T - 1) * fps / sfreq)) + 1;
  std::vector<double> pos(static_cast<std::size_t>(count));
  for (std::int64_t j = 0; j < count; ++j) pos[static_cast<std::size_t>(j)] = static_cast<double>(j) * sfreq / fps;
  const Eigen::MatrixXd Rf = nxrtest::analytic_at_frames(xf, M, T, sfreq, 8.0, 12.0, pos).real();      // the band's analytic signal's real part at the frames [M x count]
  const Eigen::MatrixXd X = Mop * Rf;                                                                  // M Re at the frames [Vg x count]
  std::vector<float> Cf(static_cast<std::size_t>(Vg * count));
  for (std::int64_t v = 0; v < Vg; ++v) for (std::int64_t j = 0; j < count; ++j) Cf[static_cast<std::size_t>(v * count + j)] = static_cast<float>(X(v, j));   // [Vg x count]
  std::vector<std::int64_t> good;                                              // the burst frames: inside the window, outside the bad span
  for (std::int64_t j = 0; j < count; ++j) {
    const double tj = static_cast<double>(j) / fps, sec = std::floor(pos[static_cast<std::size_t>(j)]) / sfreq;
    if (tj >= w0 && tj < w1 && !(sec >= bad0 && sec < bad1)) good.push_back(j);
  }
  std::printf("  %lld glued vertices, %lld frames, %zu burst frames (the range holds the first 3000)\n", static_cast<long long>(Vg), static_cast<long long>(count), good.size());
  CHECK(Vg > N && good.size() > 3000 && good.size() < static_cast<std::size_t>(count));

  // ── PASS ONE: the distribution, stored as a statistic ──
  g::RunOptions all; all.chunks = "all"; all.replace = true;
  auto run_repeat_compute = [&](nxr::io::DatasetWriter& d, SubjectWriter& sb, const RunRepeatOptions& o) {
    const auto tsr = tk::Subject::of(sb);                                       // `over` is the LEVEL of the ladder the crossings were counted in (a Selection of its own), found as the runs' measurements are
    const auto ladr = tsr.find({.cls = tk::Cls::selection, .type = "ladder", .manifold_id = surf});
    const auto lvr = ladr ? tsr.member(*ladr, level) : std::nullopt;
    Json set = Json{{"recording_a", o.recording_a}, {"recording_b", o.recording_b}, {"over", lvr ? lvr->name : std::string("cortex tiles")}, {"spatial_scale", "parts " + std::to_string(1 << level)}, {"min_crossings", o.min_crossings}, {"stage", o.stage}};
    if (o.stage_name) set["stage_name"] = *o.stage_name;
    if (!o.name.empty()) set["name"] = o.name;
    const auto rep = nxrtest::run_definition(d, sb, "run_repeat", set);
    RunRepeatResult r;
    r.hit = !rep.empty() && rep[0].value("hit", false);
    const std::string stem = o.name.empty() ? "vortex_cores.run_repeat." + o.recording_a + "." + o.recording_b : o.name;
    std::int64_t used = 0;
    for (const double v : read_statistic(sb, stem + ".tiles").values) used += std::isfinite(v);
    r.info = Json{{"tiles_used", used}};
    return r;
  };
  const Json base = set_of({{"spatial_scale", "index:" + std::to_string(level)}, {"transform", "exact"}, {"bursts", "window"}, {"glued_surface", "cortex_glued"}, {"onset", Json()}, {"within", Json()}, {"offset", Json()}});
  const g::Definition dthr = g::load("vortex_core_threshold"), dcor = g::load("vortex_cores");
  Json tbase = base; for (const char* k : {"onset", "within", "offset"}) tbase.erase(k);               // (the stages' selections are the cores' inputs, not the threshold's)
  const auto rt = g::run(ds, sub, dthr, tbase, all);
  CHECK(rt.size() == 2);                                                       // one run for each recording
  std::map<std::string, StoredStatistic> hist;
  for (const char* rec : {"rest", "rest_b"}) hist.emplace(rec, read_statistic(sub, std::string("vortex_core_threshold.histogram.") + rec));
  const StoredStatistic& H = hist.at("rest");
  CHECK(text(H.row, "method") == "histogram" && text(H.row, "measure") == "count" && integer(H.row, "n_members") == static_cast<std::int64_t>(good.size()) && H.values.size() == 4096);
  // the direct histogram: the range from the first 3000 good frames' maximum, every good frame counted
  double mx = 0;
  for (std::size_t i = 0; i < 3000; ++i) for (std::int64_t v = 0; v < Vg; ++v) mx = std::max(mx, static_cast<double>(std::abs(Cf[static_cast<std::size_t>(v * count + good[i])])));
  const nxr::stats::histogram::LogBins lb = nxr::stats::histogram::logBinsFor(mx);
  std::vector<float> A(static_cast<std::size_t>(Vg) * good.size());
  for (std::int64_t v = 0; v < Vg; ++v) for (std::size_t i = 0; i < good.size(); ++i) A[static_cast<std::size_t>(v) * good.size() + i] = std::abs(Cf[static_cast<std::size_t>(v * count + good[i])]);
  const nxr::stats::histogram::Counts direct = histogramTiles(A.data(), static_cast<std::size_t>(Vg), good.size(), good.size(), 1, lb);
  double l1 = 0, total = 0;
  for (int k = 0; k < 4096; ++k) { l1 += std::abs(H.values[static_cast<std::size_t>(k)] - static_cast<double>(direct.counts[static_cast<std::size_t>(k)])); total += H.values[static_cast<std::size_t>(k)]; }
  std::printf("  the stored histogram vs the direct one: hi %.9g vs %.9g, counts differ by %.0f of %.0f\n", H.options["hi"].get<double>(), lb.hi, l1, total);
  CHECK(total == static_cast<double>(Vg) * static_cast<double>(good.size()) && std::abs(H.options["hi"].get<double>() - lb.hi) < 1e-5 * lb.hi && H.options["lo"].get<double>() > 0 && l1 <= 2e-4 * total);
  const double dthr98 = nxr::stats::histogram::binnedPercentile(direct.counts, lb, 98);
  // chunk size independence: the same distribution from chunks of 4096 and of 1000
  { const StoredStatistic& Hb = hist.at("rest_b");
    double lb1 = 0;
    for (int k = 0; k < 4096; ++k) lb1 += std::abs(Hb.values[static_cast<std::size_t>(k)] - H.values[static_cast<std::size_t>(k)]);
    std::printf("  chunks of 4096 vs 1000: hi %.9g vs %.9g, counts differ by %.0f\n", H.options["hi"].get<double>(), Hb.options["hi"].get<double>(), lb1);
    CHECK(std::abs(Hb.options["hi"].get<double>() - H.options["hi"].get<double>()) < 1e-6 * H.options["hi"].get<double>() && lb1 <= 2e-3 * total && Hb.options["frames"] == H.options["frames"]); }
  CHECK(H.options["chain"]["form"] == "curl_tan" && H.options["chain"]["glued"] == true && H.options["range_frames"] == 3000);
  // a rerun is a hit
  { const auto again = g::run(ds, sub, dthr, tbase, g::RunOptions{[] { g::RunOptions o; o.chunks = "all"; return o; }()});
    CHECK(again.size() == 2 && again[0]["hit"] == true); }

  // ── PASS TWO: the cores, the threshold read by name ──
  const auto rc = g::run(ds, sub, dcor, base, all);
  CHECK(rc.size() == 2);
  const Stored C = read_cores(sub, "rest", fps, 0.0), Cb = read_cores(sub, "rest_b", fps, 0.0);
  const double thr = nxr::stats::histogram::binnedPercentile([&] { std::vector<std::int64_t> c(4096); for (int k = 0; k < 4096; ++k) c[static_cast<std::size_t>(k)] = std::llround(H.values[static_cast<std::size_t>(k)]); return c; }(),
                                                  nxr::stats::histogram::LogBins{H.options["lo"].get<double>(), H.options["hi"].get<double>(), 4096}, 98.0);
  // DIRECT: corePatches on the burst frames with the same threshold, the glued mesh's edges, vertex areas and positions
  nxr::select::Mesh mesh; mesh.n = static_cast<int>(sband.glued_vertices); const SurfaceInput GS = read_surface(sub, "cortex_glued", true);
  mesh.positions = GS.vertices; mesh.faces = GS.faces;
  const std::vector<double> va = nxr::select::vertexAreas(mesh);
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  for (std::size_t f = 0; f + 2 < GS.faces.size(); f += 3) for (int k = 0; k < 3; ++k) pairs.emplace_back(GS.faces[f + static_cast<std::size_t>(k)], GS.faces[f + static_cast<std::size_t>((k + 1) % 3)]);
  Eigen::MatrixXd cm(Vg, static_cast<Eigen::Index>(good.size()));
  for (std::int64_t v = 0; v < Vg; ++v) for (std::size_t i = 0; i < good.size(); ++i) cm(v, static_cast<Eigen::Index>(i)) = Cf[static_cast<std::size_t>(v * count + good[i])];
  Eigen::MatrixXd gp(Vg, 3);
  for (std::int64_t v = 0; v < Vg; ++v) for (int d = 0; d < 3; ++d) gp(v, d) = GS.vertices[static_cast<std::size_t>(3 * v + d)];
  const nxr::select::Cores want = nxr::select::corePatches(cm, pairs, Eigen::Map<const Eigen::VectorXd>(va.data(), Vg), gp, thr, 0);
  std::printf("  threshold %.6g (direct histogram's %.6g); %lld cores stored, %lld direct\n", thr, dthr98, static_cast<long long>(C.frame.size()), static_cast<long long>(want.size()));
  CHECK(std::abs(thr - dthr98) <= 0.01 * dthr98);                              // the stored and the direct distributions are within a bin
  CHECK(want.size() > 100 && std::abs(static_cast<double>(C.frame.size()) - static_cast<double>(want.size())) <= 0.02 * static_cast<double>(want.size()));
  // member by member: the frame, the sense, the vertex exactly (same threshold: the direct run's bin edge may differ by one bin, so compare only when the counts are equal)
  bool same_n = C.frame.size() == static_cast<std::size_t>(want.size());
  if (same_n) {
    double dc = 0, da = 0, dp = 0;
    bool exact = true;
    for (std::int64_t i = 0; i < want.size(); ++i) {
      const std::size_t k = static_cast<std::size_t>(i);
      exact = exact && C.frame[k] == good[static_cast<std::size_t>(want.frame[k])] && C.sense[k] == want.sense[k] && C.cv[k] == want.centre_vertex[k] && C.f[k] == static_cast<float>(C.frame[k]);
      dc = std::max({dc, std::abs(C.cx[k] - want.centre(i, 0)), std::abs(C.cy[k] - want.centre(i, 1)), std::abs(C.cz[k] - want.centre(i, 2))});
      da = std::max(da, std::abs(C.area[k] - want.area[k]) / want.area[k]);
      dp = std::max(dp, std::abs(C.peak[k] - want.peak[k]) / want.peak[k]);
    }
    std::printf("  cores vs corePatches on the direct field: frame, sense, vertex %s; centre %.1e, area %.1e, peak %.1e (relative to the value)\n", exact ? "identical" : "DIFFER", dc, da, dp);
    CHECK(exact && dc < 1e-5 && da < 1e-5 && dp < 1e-4);
  } else {
    std::printf("  (the direct run counted %lld cores against %zu: its threshold fell on another bin edge; frames compared by set)\n", static_cast<long long>(want.size()), C.frame.size());
    CHECK(false);
  }
  // the spans: a member per core on its frame (several share one), inside the bursts, outside the bad span, each [t_j, t_j + 1/fps)
  std::set<std::int64_t> distinct(C.frame.begin(), C.frame.end());
  bool inside = true, width_ok = true;
  for (std::size_t k = 0; k < C.frame.size(); ++k) {
    const double tj = static_cast<double>(C.frame[k]) / fps;
    inside = inside && tj >= w0 && tj < w1 && !(tj >= bad0 && tj < bad1);
    width_ok = width_ok && std::abs(C.start[k] - tj) < 1e-9 && std::abs(C.stop[k] - C.start[k] - 1.0 / fps) < 1e-9;
  }
  std::printf("  %zu cores on %zu distinct frames; the members' spans are their frames: %s\n", C.frame.size(), distinct.size(), width_ok ? "yes" : "NO");
  CHECK(distinct.size() < C.frame.size() && inside && width_ok);                // members share a frame; the storage allowed it
  CHECK(C.params["detection"]["record"]["cores"] == static_cast<std::int64_t>(C.frame.size()) && std::abs(C.params["detection"]["record"]["threshold"].get<double>() - thr) < 1e-12 * thr);
  CHECK(C.params["detection"]["record"]["percentile"] == 98.0 && C.params["detection"]["record"]["threshold_statistic"]["name"] == "vortex_core_threshold.histogram.rest");
  // chunk size: the same cores from chunks of 1000
  { bool same = Cb.frame == C.frame && Cb.sense == C.sense && Cb.cv == C.cv;
    double dc = 0;
    for (std::size_t k = 0; same && k < C.frame.size(); ++k) dc = std::max({dc, static_cast<double>(std::abs(Cb.cx[k] - C.cx[k])), static_cast<double>(std::abs(Cb.area[k] - C.area[k]) / C.area[k])});
    std::printf("  cores from chunks of 1000 vs 4096: %zu vs %zu, frames/senses/vertices %s, centre and area differ by %.1e\n", Cb.frame.size(), C.frame.size(), same ? "identical" : "DIFFER", dc);
    CHECK(same && dc < 1e-4); }   // the frames carry the tail the member's taps leave out, so a chunk's edge frames differ at that level
  // block size: the same cores and the same distribution when the tail runs on blocks of 100 frames (the default is 256, `--block-frames`): only the memory changes
  { g::RunOptions small = all; small.block_frames = 100;
    g::run(ds, sub, dthr, tbase, small);
    const StoredStatistic H2 = read_statistic(sub, "vortex_core_threshold.histogram.rest");
    double l2 = 0;
    for (int k = 0; k < 4096; ++k) l2 += std::abs(H2.values[static_cast<std::size_t>(k)] - H.values[static_cast<std::size_t>(k)]);
    g::run(ds, sub, dcor, base, small);
    const Stored Cs = read_cores(sub, "rest", fps, 0.0);
    bool same = Cs.frame == C.frame && Cs.sense == C.sense && Cs.cv == C.cv;
    double dc = 0;
    for (std::size_t k = 0; same && k < C.frame.size(); ++k) dc = std::max({dc, static_cast<double>(std::abs(Cs.cx[k] - C.cx[k])), static_cast<double>(std::abs(Cs.area[k] - C.area[k]) / C.area[k])});
    std::printf("  blocks of 100 frames vs 256: histogram counts differ by %.0f, %zu vs %zu cores, frames/senses/vertices %s, centre and area differ by %.1e\n", l2, Cs.frame.size(), C.frame.size(), same ? "identical" : "DIFFER", dc);
    CHECK(l2 <= 2e-4 * total && same && dc < 1e-6);
    CHECK(g::run(ds, sub, dcor, base, small)[0]["hit"] == true);                 // the block size is in the key: the rerun at 100 is a hit, the one at 256 would not be
    g::run(ds, sub, dthr, tbase, all); g::run(ds, sub, dcor, base, all);          // back to the default for the rest
  }
  // a rerun is a hit; another percentile is another threshold from the same stored counts (a lower one: more cores)
  { const auto again = g::run(ds, sub, dcor, base, g::RunOptions{[] { g::RunOptions o; o.chunks = "all"; return o; }()});
    CHECK(again.size() == 2 && again[0]["hit"] == true); }
  { Json s90 = base; s90["percentile"] = 90.0;
    g::run(ds, sub, dcor, s90, all);
    const Stored C90 = read_cores(sub, "rest", fps, 0.0);
    std::printf("  percentile 90: threshold %.6g, %zu cores (98: %.6g, %zu)\n", C90.params["detection"]["record"]["threshold"].get<double>(), C90.frame.size(), thr, C.frame.size());
    CHECK(C90.params["detection"]["record"]["threshold"].get<double>() < thr && C90.frame.size() > C.frame.size()); }

  // ── THE TRACKS (piece 6): the stored cores linked directly, against the stored tracks ──
  {
    g::run(ds, sub, dcor, base, all);                                            // the default percentile again (the run above stored the 90th's)
    const Stored C = read_cores(sub, "rest", fps, 0.0);                          // re-read: the run above replaced the Selection
    const StoredTracks T = read_tracks(sub, "rest"), Tb = read_tracks(sub, "rest_b");
    const double link_m = 15.0e-3;
    // the cores as stored (float32 centres): linked directly
    std::vector<std::int64_t> cf(C.frame.begin(), C.frame.end());
    std::vector<std::int8_t> cs(C.frame.size());
    Eigen::MatrixXd cc(static_cast<Eigen::Index>(C.frame.size()), 3);
    for (std::size_t k = 0; k < C.frame.size(); ++k) { cs[k] = C.sense[k] > 0 ? 1 : -1; cc(static_cast<Eigen::Index>(k), 0) = C.cx[k]; cc(static_cast<Eigen::Index>(k), 1) = C.cy[k]; cc(static_cast<Eigen::Index>(k), 2) = C.cz[k]; }
    const nxr::select::Linked L = nxr::select::link(cf, cs, cc, link_m);
    const nxr::select::TrackStats S = nxr::select::trackStats(cf, cs, cc, L.track, L.previous);
    std::vector<std::int64_t> keep;                                              // the tracks of >= 2 frames, in the order they start
    for (std::int64_t t = 0; t < S.size(); ++t) if (S.n_frames[static_cast<std::size_t>(t)] >= 2) keep.push_back(t);
    std::printf("  tracks: %zu stored (%lld found, %zu of 2+ frames linking the stored cores directly); the longest %.0f frames\n", T.first.size(), static_cast<long long>(L.tracks), keep.size(),
                T.n.empty() ? 0.0 : static_cast<double>(*std::max_element(T.n.begin(), T.n.end())));
    CHECK(!keep.empty() && T.first.size() == keep.size() && T.start.size() == keep.size());
    bool ok = T.first.size() == keep.size();
    double dp = 0;
    for (std::size_t k = 0; ok && k < keep.size(); ++k) {
      const std::size_t t = static_cast<std::size_t>(keep[k]);
      ok = ok && T.first[k] == static_cast<float>(S.first_frame[t]) && T.last[k] == static_cast<float>(S.last_frame[t]) && T.n[k] == static_cast<float>(S.n_frames[t]) && T.sense[k] == static_cast<float>(S.sense[t]);
      dp = std::max({dp, std::abs(T.path[k] - S.path[t]), std::abs(T.disp[k] - S.displacement[t])});   // float32 storage of the centres
      // the span: the first core's frame start to the last core's frame end
      ok = ok && std::abs(T.start[k] - static_cast<double>(S.first_frame[t]) / fps) < 1e-9 && std::abs(T.stop[k] - static_cast<double>(S.last_frame[t] + 1) / fps) < 1e-9;
    }
    std::printf("  stored tracks vs nxr::select on the stored cores: first, last, n, sense, spans %s; path and displacement differ by %.1e (absolute, the unit of the surface)\n", ok ? "identical" : "DIFFER", dp);
    CHECK(ok && dp < 1e-5);
    // the labelling: each core's track = the member ordinal among the stored tracks (-1 for a core in a track of one frame)
    const std::vector<float> lab = measure_of(sub, C.id, "track");
    std::vector<std::int64_t> ordinal(static_cast<std::size_t>(S.size()), -1);
    for (std::size_t k = 0; k < keep.size(); ++k) ordinal[static_cast<std::size_t>(keep[k])] = static_cast<std::int64_t>(k);
    bool labok = lab.size() == C.frame.size();
    std::size_t in_tracks = 0;
    for (std::size_t k = 0; labok && k < lab.size(); ++k) { labok = lab[k] == static_cast<float>(ordinal[static_cast<std::size_t>(L.track[k])]); in_tracks += lab[k] >= 0; }
    std::printf("  the labelling core -> track: %zu of %zu cores are in a stored track: %s\n", in_tracks, lab.size(), labok ? "as linked directly" : "DIFFERS");
    CHECK(labok && in_tracks > 0 && in_tracks < lab.size() + 1);
    CHECK(T.params["detection"]["record"]["tracks"] == static_cast<std::int64_t>(keep.size()) && T.params["detection"]["record"]["link_mm"] == 15.0 && T.params["detection"]["record"]["warnings"].is_array());
    // chunk size: the same tracks from chunks of 1000
    { bool pathsame = Tb.path.size() == T.path.size();     // the frames are float32 and chunk-independent to 1e-7 of the peak: the paths agree to that, not bit for bit
      for (std::size_t i = 0; pathsame && i < T.path.size(); ++i) pathsame = std::abs(Tb.path[i] - T.path[i]) <= 1e-5 * std::max(1.0, std::abs(static_cast<double>(T.path[i])));
      CHECK(Tb.first == T.first && Tb.last == T.last && Tb.n == T.n && Tb.sense == T.sense && pathsame); }
    // a longer link distance joins more (fewer tracks, each longer); min_frames 1 also stores the lone cores
    { Json s = base; s["link_mm"] = 500.0;
      g::run(ds, sub, dcor, s, all);
      const StoredTracks W = read_tracks(sub, "rest");
      std::printf("  link 500 mm: %zu tracks (15 mm: %zu)\n", W.first.size(), T.first.size());
      CHECK(W.first.size() < T.first.size() && W.first.size() > 0 && *std::max_element(W.n.begin(), W.n.end()) >= *std::max_element(T.n.begin(), T.n.end())); }
    { Json s = base; s["min_frames"] = 1;
      g::run(ds, sub, dcor, s, all);
      const StoredTracks W = read_tracks(sub, "rest");
      std::printf("  min_frames 1: %zu tracks (2: %zu; %lld found)\n", W.first.size(), T.first.size(), static_cast<long long>(L.tracks));
      CHECK(W.first.size() == static_cast<std::size_t>(L.tracks) && W.first.size() > T.first.size()); }
    g::run(ds, sub, dcor, base, all);                                            // back to the defaults
  }

  // ── THE CROSSINGS (piece 7): the stored tile_crossings against nxr::select::track_crossings on the stored tracks' steps ──
  {
    const Stored C = read_cores(sub, "rest", fps, 0.0);
    const std::vector<float> lab = measure_of(sub, C.id, "track");
    const auto ts = tk::Subject::of(sub);
    const auto fld = ts.find({.cls = tk::Cls::field, .name = "rest"});
    const auto lad = ts.find({.cls = tk::Cls::selection, .type = "ladder", .manifold_id = surf});
    const auto lvl = lad ? ts.member(*lad, level) : std::nullopt;
    CHECK(fld && lad && lvl && lab.size() == C.frame.size());
    const std::string sel = lvl->id, fid = fld->id;
    const std::int64_t tiles = lvl->n_members.value_or(0);
    auto stored = [&](const char* m) { return measurement_values(sub, sel, fid, std::nullopt, std::nullopt, m); };
    const MeasurementValues sx = stored("crossing_sum_x"), sy = stored("crossing_sum_y"), sz = stored("crossing_sum_z"), nn = stored("crossings");
    std::printf("  tile_crossings: [%lld rows x %lld members] (the %lld tiles and the bridge)\n", static_cast<long long>(nn.rows), static_cast<long long>(nn.members), static_cast<long long>(tiles));
    CHECK(nn.array_backed && nn.rows == 2 && nn.members == tiles + 1);
    // DIRECT: the steps of the stored tracks (each core after its track's first, from the previous core's centre), the surface's frames, the transport of each tile centre's frame, the glued domain
    nxr::select::track_crossings::Steps st;
    std::map<std::int64_t, std::size_t> last;
    std::vector<std::array<double, 6>> sv;
    for (std::size_t k = 0; k < lab.size(); ++k) {
      if (lab[k] < 0) continue;
      const auto it = last.find(static_cast<std::int64_t>(lab[k]));
      if (it != last.end()) {
        const std::size_t q = it->second;
        sv.push_back({C.cx[q], C.cy[q], C.cz[q], C.cx[k] - C.cx[q], C.cy[k] - C.cy[q], C.cz[k] - C.cz[q]});
        st.sense.push_back(C.sense[k] > 0 ? 1 : -1); st.track.push_back(static_cast<std::int32_t>(lab[k]));
      }
      last[static_cast<std::int64_t>(lab[k])] = k;
    }
    st.from.resize(static_cast<Eigen::Index>(sv.size()), 3); st.disp.resize(static_cast<Eigen::Index>(sv.size()), 3);
    for (std::size_t i = 0; i < sv.size(); ++i) for (int k = 0; k < 3; ++k) { st.from(static_cast<Eigen::Index>(i), k) = sv[i][static_cast<std::size_t>(k)]; st.disp(static_cast<Eigen::Index>(i), k) = sv[i][static_cast<std::size_t>(3 + k)]; }
    const SurfaceInput NS = read_surface(sub, "cortex", true);
    const int nV = static_cast<int>(NS.n_vertices());
    const SurfaceLevel NL = read_surface_level(sub, NS, level);
    nxr::geometry::ChartKit kit(NS.vertices.data(), nV, NS.faces.data(), static_cast<int>(NS.n_faces()));
    const nxr::geometry::Basis B = kit.basis();
    nxr::geometry::Mat3 e1(nV, 3), e2(nV, 3);
    for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) { e1(i, k) = B.e1[static_cast<std::size_t>(3 * i + k)]; e2(i, k) = B.e2[static_cast<std::size_t>(3 * i + k)]; }
    const Eigen::VectorXd gauge = nxr::geometry::gaugeAngle(kit, e1);
    nxr::select::Mesh nm; nm.n = nV; nm.positions = NS.vertices; nm.faces = NS.faces;
    const std::vector<double> narea = nxr::select::vertexAreas(nm);
    const std::vector<int> cen = nxr::select::tileCentres(NS.vertices.data(), NL.tile, static_cast<int>(NL.nodes), narea);
    const nxr::select::LeviCivita lc = nxr::select::leviCivita(kit, e1, e2, gauge, NL.tile, static_cast<int>(NL.nodes), cen);
    const nxr::geometry::Mat3 nrm = nxr::geometry::vertexNormals(NS.vertices.data(), nV, NS.faces.data(), static_cast<int>(NS.n_faces()));
    const internal::GluedDomain dom = internal::glued_domain(sub, NS);
    CHECK(dom.found);
    const std::int64_t Vg = dom.surface.n_vertices(), kept = static_cast<std::int64_t>(dom.support.size());
    nxr::select::track_crossings::Surface G;
    G.positions = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(dom.surface.vertices.data(), Vg, 3);
    G.normals = nxr::geometry::vertexNormals(dom.surface.vertices.data(), static_cast<int>(Vg), dom.surface.faces.data(), static_cast<int>(dom.surface.n_faces()));
    G.e1 = Eigen::MatrixXd::Constant(Vg, 3, std::nan("")); G.e2 = G.e1; G.rho = Eigen::VectorXd::Constant(Vg, std::nan(""));
    G.node.assign(static_cast<std::size_t>(Vg), nxr::select::track_crossings::kBridge);
    for (std::int64_t i = 0; i < kept; ++i) { const int v = dom.support[static_cast<std::size_t>(i)]; G.normals.row(i) = nrm.row(v); G.e1.row(i) = e1.row(v); G.e2.row(i) = e2.row(v); G.rho[i] = lc.rho[v]; G.node[static_cast<std::size_t>(i)] = NL.tile[static_cast<std::size_t>(v)]; }
    const nxr::select::track_crossings::Result want = nxr::select::track_crossings::crossings(st, {}, G, static_cast<int>(NL.nodes));
    long total = 0, diff = 0, bridge_total = 0;
    double dsum = 0, smax = 0;
    for (int sg = 0; sg < 2; ++sg) {
      const Eigen::MatrixXd v = nxr::select::track_crossings::embedAtCentres(want.sum.col(sg), cen, e1, e2);
      for (std::int64_t t = 0; t < tiles; ++t) {
        const std::size_t at = static_cast<std::size_t>(sg * nn.members + t);
        total += want.n(static_cast<Eigen::Index>(t), sg);
        diff += std::abs(static_cast<long>(nn.values[at]) - static_cast<long>(want.n(static_cast<Eigen::Index>(t), sg)));
        dsum = std::max({dsum, std::abs(sx.values[at] - v(static_cast<Eigen::Index>(t), 0)), std::abs(sy.values[at] - v(static_cast<Eigen::Index>(t), 1)), std::abs(sz.values[at] - v(static_cast<Eigen::Index>(t), 2))});
        smax = std::max(smax, std::abs(sx.values[at]));
      }
      const std::size_t bat = static_cast<std::size_t>(sg * nn.members + tiles);
      bridge_total += want.bridge_n[sg];
      diff += std::abs(static_cast<long>(nn.values[bat]) - static_cast<long>(want.bridge_n[sg]));
    }
    std::printf("  %zu steps of the stored tracks make %ld crossings in the tiles (%ld on the bridge); the stored counts differ from nxr::select::track_crossings on the stored cores by %ld, the sums by at most %.1e (largest component %.2f)\n",
                st.sense.size(), total, bridge_total, diff, dsum, smax);
    CHECK(total > 50 && diff <= std::max<long>(2, total / 100) && dsum < 0.05);    // the step links and places doubles; the test the stored float32 centres: a step on a tile border may fall in the next
    // R and the unbiased rho2 derived at read from the stored sum and count
    const MeasurementValues R = derived_measurement_values(sub, sel, fid, std::nullopt, std::nullopt, "crossing_consistency"), RHO = derived_measurement_values(sub, sel, fid, std::nullopt, std::nullopt, "crossing_rho2");
    bool derived = R.values.size() == nn.values.size() && RHO.values.size() == nn.values.size();
    int checked = 0;
    for (std::size_t i = 0; derived && i < nn.values.size(); ++i) {
      const double n = nn.values[i], s2 = double(sx.values[i]) * sx.values[i] + double(sy.values[i]) * sy.values[i] + double(sz.values[i]) * sz.values[i];
      if (n >= 2) { derived = derived && std::abs(R.values[i] - std::sqrt(s2) / n) < 1e-9 && std::abs(RHO.values[i] - (s2 - n) / (n * (n - 1))) < 1e-9; ++checked; }
      else if (n == 1) derived = derived && std::abs(R.values[i] - 1) < 1e-4 && std::isnan(RHO.values[i]);
    }
    std::printf("  R = |S|/n and the unbiased rho2 derived at read for %d tile-senses with 2+ crossings: %s\n", checked, derived ? "as the formulas" : "DIFFER");
    CHECK(derived && checked > 5);
    // a STAGE: the steps whose frame starts inside a named spans Selection (here the window's first part, 2-8 s): its rows are the direct run with that mask, and the burst rows are unchanged
    { sub.compose([&] { nxr::bst::TimeSpansOptions eo; eo.line_id = tid; eo.of_field_id = first_rid; eo.description = "an early stage"; nxr::bst::time_spans(sub, "time/early", {{2.0, 8.0, 0}}, {"early"}, eo); });
      Json s3 = base; s3["stages"] = Json::array({"burst", "within"}); s3["within"] = "early";
      g::run(ds, sub, dcor, s3, all);
      const MeasurementValues n3 = measurement_values(sub, sel, fid, std::nullopt, std::nullopt, "crossings");
      std::vector<std::uint8_t> mask;
      std::size_t k2 = 0;
      std::map<std::int64_t, std::size_t> last2;
      for (std::size_t k = 0; k < lab.size(); ++k) {
        if (lab[k] < 0) continue;
        const auto it = last2.find(static_cast<std::int64_t>(lab[k]));
        if (it != last2.end()) { const double tj = static_cast<double>(C.frame[k]) / fps; mask.push_back(tj >= 2.0 && tj < 8.0); ++k2; }
        last2[static_cast<std::int64_t>(lab[k])] = k;
      }
      const nxr::select::track_crossings::Result part = nxr::select::track_crossings::crossings(st, mask, G, static_cast<int>(NL.nodes));
      long before = 0, in_stage = 0, direct = 0, same = 1;
      for (std::int64_t t = 0; t <= tiles; ++t) for (int sg = 0; sg < 2; ++sg) {
        before += static_cast<long>(nn.values[static_cast<std::size_t>(sg * nn.members + t)]);
        same = same && n3.values[static_cast<std::size_t>(sg * n3.members + t)] == nn.values[static_cast<std::size_t>(sg * nn.members + t)];
        in_stage += static_cast<long>(n3.values[static_cast<std::size_t>((2 + sg) * n3.members + t)]);
        direct += t < tiles ? part.n(static_cast<Eigen::Index>(t), sg) : part.bridge_n[sg];
      }
      std::printf("  a stage (2-8 s): %ld crossings in the stage (direct %ld) of %ld in the burst rows (%zu steps in the stage)\n", in_stage, direct, before, static_cast<std::size_t>(std::count(mask.begin(), mask.end(), 1)));
      CHECK(n3.rows == 4 && same && in_stage == direct && in_stage > 0 && in_stage < before);
      // the stages' names are on the rows: `repeat compute` selects a stage by NAME (same as its index), and a stage the runs do not have is refused with the names they do
      { RunRepeatOptions rn; rn.recording_a = "rest"; rn.recording_b = "rest_b"; rn.min_crossings = 1;
        RunRepeatOptions ri = rn;
        rn.stage_name = "within"; rn.name = "by_name"; ri.stage = 1; ri.name = "by_index";
        run_repeat_compute(ds, sub, rn); run_repeat_compute(ds, sub, ri);
        const StoredStatistic tn = read_statistic(sub, "by_name.tiles"), ti = read_statistic(sub, "by_index.tiles");
        bool eq = tn.values.size() == ti.values.size() && !tn.values.empty();
        for (std::size_t t = 0; eq && t < tn.values.size(); ++t) eq = (std::isnan(tn.values[t]) && std::isnan(ti.values[t])) || tn.values[t] == ti.values[t];
        RunRepeatOptions rb = rn; rb.stage_name = "burst"; rb.name = "burst_named";
        run_repeat_compute(ds, sub, rb);
        const StoredStatistic tb = read_statistic(sub, "burst_named.tiles");
        bool differs = false;
        for (std::size_t t = 0; t < tb.values.size(); ++t) if (!(std::isnan(tb.values[t]) && std::isnan(tn.values[t])) && tb.values[t] != tn.values[t]) differs = true;
        const std::string why = what_of([&] { RunRepeatOptions x = rn; x.stage_name = "offset"; x.name = "off"; run_repeat_compute(ds, sub, x); });
        std::printf("  repeat --stage within = --stage 1: %s; burst differs from within: %s; 'offset' refused: %s\n", eq ? "same" : "DIFFER", differs ? "yes" : "NO", why.find("burst, within") != std::string::npos ? "with the names the runs have" : why.c_str());
        CHECK(eq && differs && why.find("no stage 'offset'") != std::string::npos && why.find("burst, within") != std::string::npos); }
      g::run(ds, sub, dcor, base, all); }
    // a rerun is a hit; a stage named without its Selection is refused
    CHECK(g::run(ds, sub, dcor, base, g::RunOptions{[] { g::RunOptions o; o.chunks = "all"; return o; }()})[0]["hit"] == true);
    { Json s2 = base; s2["stages"] = Json::array({"burst", "onset"});
      CHECK(what_of([&] { g::run(ds, sub, dcor, s2, g::RunOptions{}); }).find("onset") != std::string::npos); }
  }

  // ── THE RUN-TO-RUN REPEAT (piece 8) ──
  {
    const auto ts = tk::Subject::of(sub);
    const auto fa = ts.find({.cls = tk::Cls::field, .name = "rest"}), fb = ts.find({.cls = tk::Cls::field, .name = "rest_b"});
    const auto lad = ts.find({.cls = tk::Cls::selection, .type = "ladder", .manifold_id = surf});
    const auto lvl = ts.member(*lad, level);
    const std::string sel = lvl->id;
    RunRepeatOptions ro;
    ro.recording_a = "rest"; ro.recording_b = "rest_b"; ro.min_crossings = 10;
    const auto r1 = run_repeat_compute(ds, sub, ro);
    // DIRECT: nxr::select::track_crossings::repeat on the stored arrays (rows +1 and -1 pooled, the bridge member left out)
    auto pooled = [&](const std::string& field) {
      auto get = [&](const char* m) { return measurement_values(sub, sel, field, std::nullopt, std::nullopt, m); };
      const MeasurementValues x = get("crossing_sum_x"), y = get("crossing_sum_y"), z = get("crossing_sum_z"), n = get("crossings");
      const std::int64_t T = n.members - 1;
      Eigen::MatrixXd p(T, 3), m(T, 3); Eigen::VectorXd np(T), nm(T);
      for (std::int64_t t = 0; t < T; ++t) { p.row(t) << x.values[static_cast<std::size_t>(t)], y.values[static_cast<std::size_t>(t)], z.values[static_cast<std::size_t>(t)]; np[t] = n.values[static_cast<std::size_t>(t)];
        const std::size_t b = static_cast<std::size_t>(n.members + t); m.row(t) << x.values[b], y.values[b], z.values[b]; nm[t] = n.values[b]; }
      return nxr::select::track_crossings::poolSenses(p, np, m, nm);
    };
    const nxr::select::track_crossings::Repeat want = nxr::select::track_crossings::repeat(pooled(fa->id), pooled(fb->id), 10);
    const StoredStatistic st = read_statistic(sub, "vortex_cores.run_repeat.rest.rest_b.tiles"), sw = read_statistic(sub, "vortex_cores.run_repeat.rest.rest_b.whole");
    bool dots = st.values.size() == static_cast<std::size_t>(want.dot.size());
    for (std::size_t t = 0; dots && t < st.values.size(); ++t) dots = (std::isnan(st.values[t]) && std::isnan(want.dot[static_cast<Eigen::Index>(t)])) || std::abs(st.values[t] - want.dot[static_cast<Eigen::Index>(t)]) < 1e-9;
    std::printf("  repeat rest vs rest_b (the same motion twice): cosine %.6f over %lld of %zu tiles (direct %.6f)\n", sw.values[0], static_cast<long long>(r1.info["tiles_used"].get<std::int64_t>()), st.values.size(), want.cosine);
    CHECK(dots && std::abs(sw.values[0] - want.cosine) < 1e-9 && sw.values[0] > 0.99 && r1.info["tiles_used"].get<std::int64_t>() >= 3 && text(st.row, "measure") == "repeat_dot" && text(sw.row, "method") == "run_repeat");
    CHECK(run_repeat_compute(ds, sub, ro).hit);                                    // a rerun is a hit
    // the other run's vectors REVERSED (the motion backwards): the same arrays with the vector components negated
    { std::vector<Row> rows;
      for (const Row& r : nxr::io::testing::Probe(sub).rows("selection_measurement", {{"selection_id", V(sel)}, {"of_field_id", V(fb->id)}})) if (text(r, "measure").rfind("crossing_sum_", 0) == 0) rows.push_back(r);
      CHECK(rows.size() == 3);
      const std::string path = text(rows[0], "array_path");
      std::vector<std::int64_t> shape;
      std::vector<float> a = read_floats(sub.folder() / path, &shape);
      for (std::size_t i = 0; i < a.size(); i += 4) for (int k = 0; k < 3; ++k) a[i + static_cast<std::size_t>(k)] = -a[i + static_cast<std::size_t>(k)];
      sub.compose([&] { sub.store().erase_prefix(path); sub.write_array(path, view(a, shape)); });
      RunRepeatOptions rr = ro; rr.name = "reversed";
      const auto r2 = run_repeat_compute(ds, sub, rr);
      const StoredStatistic w2 = read_statistic(sub, "reversed.whole"), t2 = read_statistic(sub, "reversed.tiles");
      bool neg = true;
      for (const double d : t2.values) if (!std::isnan(d)) neg = neg && d <= 0;
      std::printf("  repeat with one run reversed: cosine %.6f, dots all <= 0: %s\n", w2.values[0], neg ? "yes" : "NO");
      CHECK(std::abs(w2.values[0] + 1) < 1e-6 && neg && r2.info["tiles_used"] == r1.info["tiles_used"]);
      // and put back as it was (a rerun of the run would find its rows and call it a hit: the arrays are restored by hand)
      for (std::size_t i = 0; i < a.size(); i += 4) for (int k = 0; k < 3; ++k) a[i + static_cast<std::size_t>(k)] = -a[i + static_cast<std::size_t>(k)];
      sub.compose([&] { sub.store().erase_prefix(path); sub.write_array(path, view(a, shape)); }); }
    // too few crossings: no tile qualifies (NaN, stored)
    { RunRepeatOptions rf = ro; rf.min_crossings = 100000; rf.name = "none";
      run_repeat_compute(ds, sub, rf);
      const StoredStatistic wn = read_statistic(sub, "none.whole");
      CHECK(std::isnan(wn.values[0])); }
    CHECK(what_of([&] { RunRepeatOptions x = ro; x.recording_b = "rest"; run_repeat_compute(ds, sub, x); }).find("same recording") != std::string::npos);
    CHECK(what_of([&] { RunRepeatOptions x = ro; x.recording_b = "no_such"; run_repeat_compute(ds, sub, x); }).find("recording_b") != std::string::npos);   // the input the name did not find
    CHECK(what_of([&] { RunRepeatOptions x = ro; x.stage = 3; x.name = "stage3"; run_repeat_compute(ds, sub, x); }).find("stage 3") != std::string::npos);
    g::run(ds, sub, dcor, base, all);                                              // the crossings as the run made them (the reversed copy replaced)
  }

  // ── THE LAG-FREE NULL (piece 10): the ingredients, one draw, its node ──
  {
    nxrtest::run_psd(ds, sub, "rest", Json{{"segment_samples", 256}});
    nxrtest::run_definition(ds, sub, "crossspectrum", Json{{"recording", "rest"}, {"channels_of", "free"}, {"segment_samples", 256}, {"bands", Json::array({Json{{"name", "alpha"}, {"lo", 8.0}, {"hi", 12.0}}})}});
    const g::Definition dnull = g::load("surrogate_lagfree_vortex");
    // THE INGREDIENTS the draws are made from, two stored MEASUREMENTS (the cross-spectrum's channels, tiles of 32 samples): the PSD's mean over those channels (`measure` of a stored statistic, preset psd_channel_mean) and the band's
    // power per tile (preset surrogate_lagfree_envelope)
    const auto pm = nxrtest::run_preset(ds, sub, "measure", "psd_channel_mean", set_of({{"statistic", "rest_psd_welch"}, {"channels", "rest_crossspectrum"}}));
    CHECK(pm.size() == 1 && pm[0]["hit"] == false);
    {
      const auto ts0 = tk::Subject::of(sub);
      const auto pst = ts0.find({.cls = tk::Cls::statistic, .name = "rest_psd_welch"});
      const auto pset = ts0.find({.cls = tk::Cls::selection, .name = "rest_psd_welch channels rest_crossspectrum one member"});
      CHECK(pst && pset);
      const MeasurementValues pmv = measurement_values(sub, pset->id, std::nullopt, pst->id, std::nullopt, "mean");
      const StoredStatistic P = read_statistic(sub, "rest_psd_welch");
      // the same mean made directly: over the channels the cross-spectrum holds, each frequency
      const StoredStatistic CS = read_statistic(sub, "rest_crossspectrum");
      std::printf("  the PSD's channel mean: [%lld frequencies x %lld member] (the PSD has %lld channels)\n", static_cast<long long>(pmv.rows), static_cast<long long>(pmv.members), static_cast<long long>(P.shape[0]));
      CHECK(pmv.rows == P.shape[1] && pmv.members == 1 && pmv.array_backed && CS.shape.size() >= 2);
      // the cross-spectrum's channels as stored (its channel axis is a pick of the PSD's channels)
      std::vector<std::int32_t> in_pick = crossspectrum_read(sub, "rest_crossspectrum").channels;
      const std::int64_t nf = P.shape[1], nch = P.shape[0];
      if (in_pick.empty()) for (std::int64_t c = 0; c < nch; ++c) in_pick.push_back(static_cast<std::int32_t>(c));
      double worst = 0;
      for (std::int64_t f = 0; f < nf; ++f) { double m = 0; for (const std::int32_t c : in_pick) m += P.values[static_cast<std::size_t>(c * nf + f)]; m /= static_cast<double>(in_pick.size()); worst = std::max(worst, std::abs(m - pmv.values[static_cast<std::size_t>(f)]) / std::max(1e-30, std::abs(m))); }
      CHECK(worst < 1e-9);
    }
    const auto em = nxrtest::run_preset(ds, sub, "measure", "surrogate_lagfree_envelope", set_of({{"recording", "rest"}, {"band", "8-12 Hz"}, {"channels", "rest_crossspectrum"}, {"duration", "0.128 s"}}));
    CHECK(em.size() == 1 && em[0]["hit"] == false);
    const auto tsE = tk::Subject::of(sub);
    const auto recE = tsE.find({.cls = tk::Cls::field, .name = "rest"});
    const auto prodE = tsE.find({.cls = tk::Cls::selection, .name = "rest envelope tiles 0.128 s"});
    CHECK(recE && prodE);
    const MeasurementValues E = measurement_values(sub, prodE->id, recE->id, std::nullopt, std::nullopt, "power");
    double sum = 0, cnt = 0;
    for (const double v : E.values) if (std::isfinite(v)) { sum += v; cnt += 1; }
    // the direct mean |z|^2: 2 Re^2 for a band signal (the analytic signal's power is twice the band-passed one's), over the good samples
    double direct = 0; std::int64_t nd2 = 0;
    for (int t = 0; t < T; ++t) { const double sec = static_cast<double>(t) / sfreq; if (sec >= bad0 && sec < bad1) continue; direct += Re.col(t).squaredNorm(); nd2 += M; }
    direct = 2 * direct / static_cast<double>(nd2);
    std::printf("  the envelope measurement: %lld tiles of 32 samples, mean |z|^2 %.4f (direct %.4f)\n", static_cast<long long>(E.members), sum / cnt, direct);
    CHECK(E.rows == 1 && E.members == T / 32 && std::abs(sum / cnt / direct - 1) < 0.1);
    // bad samples are left out: a tile the bad span (5-5.5 s) covers holds no value (a short mean is never a tile's mean)
    const std::int64_t tbad = static_cast<std::int64_t>(5.2 * sfreq) / 32;
    CHECK(std::isnan(E.values[static_cast<std::size_t>(tbad)]) && std::isfinite(E.values[static_cast<std::size_t>(tbad - 8)]));

    const auto ts2 = tk::Subject::of(sub);
    const auto lad2 = ts2.find({.cls = tk::Cls::selection, .type = "ladder", .manifold_id = surf});
    const auto lvl2 = ts2.member(*lad2, level);
    const std::string sel_of_level = lvl2->id;
    const std::int64_t NL_tiles_plus_bridge = lvl2->n_members.value_or(0) + 1;
    Json nb = base; nb["recording"] = "rest"; nb["cross_spectrum"] = "rest_crossspectrum"; nb["seed"] = 1;
    g::run(ds, sub, dnull, nb, all);
    const StoredStatistic D1 = read_statistic(sub, "surrogate_lagfree_vortex.draw1.null_crossings.rest");
    std::printf("  one draw: %s [%lld x %lld x %lld]\n", text(D1.row, "measure").c_str(), static_cast<long long>(D1.shape[0]), static_cast<long long>(D1.shape[1]), static_cast<long long>(D1.shape[2]));
    CHECK(text(D1.row, "measure") == "crossing_sum" && text(D1.row, "method") == "lagfree_null" && D1.shape.size() == 3 && D1.shape[0] == 2 && D1.shape[2] == 4 && D1.shape[1] == NL_tiles_plus_bridge);
    // a rerun is a hit; another seed is another draw and its own node; the data's measurement is untouched
    CHECK(g::run(ds, sub, dnull, nb, g::RunOptions{[] { g::RunOptions o; o.chunks = "all"; return o; }()})[0]["hit"] == true);
    Json nb2 = nb; nb2["seed"] = 2;
    g::run(ds, sub, dnull, nb2, all);
    const StoredStatistic D2 = read_statistic(sub, "surrogate_lagfree_vortex.draw2.null_crossings.rest");
    double dd = 0, tot1 = 0, tot2 = 0;
    for (std::size_t i = 0; i < D1.values.size(); ++i) { dd += std::abs(D1.values[i] - D2.values[i]); if (i % 4 == 3) { tot1 += D1.values[i]; tot2 += D2.values[i]; } }
    std::printf("  draw 1 and 2 differ by %.1f over their sums; crossings %.0f and %.0f\n", dd, tot1, tot2);
    CHECK(dd > 0 && tot1 > 0 && tot2 > 0);
    // the same seed again (replaced) is the same draw, to the last bit: the draw is a function of the seed
    g::RunOptions rep = all; rep.replace = true;
    g::run(ds, sub, dnull, nb, rep);
    const StoredStatistic D1b = read_statistic(sub, "surrogate_lagfree_vortex.draw1.null_crossings.rest");
    CHECK(D1b.values == D1.values);
    // a data run still holds its own crossings
    CHECK(measurement_values(sub, sel_of_level, tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = "rest"})->id, std::nullopt, std::nullopt, "crossings").rows == 2);
    // THE TESTS: the draws in batches, the per-tile p of the crossing consistency and of the repeat, against the same numbers made directly from the stored draws
    {
      nxrtest::run_psd(ds, sub, "rest_b", Json{{"segment_samples", 256}});
      nxrtest::run_definition(ds, sub, "crossspectrum", Json{{"recording", "rest_b"}, {"channels_of", "free"}, {"segment_samples", 256}, {"bands", Json::array({Json{{"name", "alpha"}, {"lo", 8.0}, {"hi", 12.0}}})}});
      nxrtest::run_preset(ds, sub, "measure", "psd_channel_mean", set_of({{"statistic", "rest_b_psd_welch"}, {"channels", "rest_b_crossspectrum"}}));
      nxrtest::run_preset(ds, sub, "measure", "surrogate_lagfree_envelope", set_of({{"recording", "rest_b"}, {"band", "8-12 Hz"}, {"channels", "rest_b_crossspectrum"}, {"duration", "0.128 s"}}));
      SurrogateVortexCompareOptions no;
      no.chunks = "all";                 // the whole recording, stated
      no.recording_a = "rest"; no.recording_b = "rest_b"; no.batch = 3; no.min_draws = 3; no.max_draws = 6; no.h = 10; no.min_crossings = 5; no.replace = true;   // draws 1 and 2 above were made with other parameters
      no.set = set_of({{"spatial_scale", "index:" + std::to_string(level)}, {"transform", "exact"}, {"bursts", "window"}, {"glued_surface", "cortex_glued"}, {"onset", Json()}, {"within", Json()}, {"offset", Json()}, {"cross_spectrum", "{recording}_crossspectrum"}});
      const auto nr = surrogate_lagfree_vortex_compare(ds, sub, no);
      const std::int64_t nd = nr.info["draws"].get<std::int64_t>();
      std::printf("  surrogate_lagfree_vortex_compare: %lld draws (%s), %lld tiles; consistency: %s; repeat cosine %s p %s\n", static_cast<long long>(nd), nr.info["stopped"].get<std::string>().c_str(), static_cast<long long>(nr.info["tiles"].get<std::int64_t>()),
                  nr.info["consistency"].dump().c_str(), nr.info["repeat"]["cosine"].dump().c_str(), nr.info["repeat"]["p"].dump().c_str());
      CHECK(nd == 6 && nr.info["stopped"] == "max_draws");                           // the data's tiles are not decided by 3 draws: the second batch runs; the maximum stops it
      // the same, made directly: the draws read back, the senses pooled, the unbiased rho2 per tile
      auto pooled_draw = [&](const std::string& rec, std::int64_t i) {
        const StoredStatistic D = read_statistic(sub, "surrogate_lagfree_vortex.draw" + std::to_string(i) + ".null_crossings." + rec);
        const std::int64_t Mm = D.shape[1], T2 = Mm - 1;
        Eigen::MatrixXd pl(T2, 3), mi(T2, 3); Eigen::VectorXd np(T2), nmm(T2);
        for (std::int64_t t = 0; t < T2; ++t) { const std::size_t a = static_cast<std::size_t>(t * 4), b = static_cast<std::size_t>((Mm + t) * 4);
          pl.row(t) << D.values[a], D.values[a + 1], D.values[a + 2]; np[t] = D.values[a + 3]; mi.row(t) << D.values[b], D.values[b + 1], D.values[b + 2]; nmm[t] = D.values[b + 3]; }
        return nxr::select::track_crossings::poolSenses(pl, np, mi, nmm);
      };
      auto data_pooled = [&](const std::string& rec) {
        const auto f2 = tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = rec});
        auto get = [&](const char* m) { return measurement_values(sub, sel_of_level, f2->id, std::nullopt, std::nullopt, m); };
        const MeasurementValues x = get("crossing_sum_x"), y = get("crossing_sum_y"), z = get("crossing_sum_z"), n = get("crossings");
        const std::int64_t T2 = n.members - 1;
        Eigen::MatrixXd pl(T2, 3), mi(T2, 3); Eigen::VectorXd np(T2), nmm(T2);
        for (std::int64_t t = 0; t < T2; ++t) { const std::size_t a = static_cast<std::size_t>(t), b = static_cast<std::size_t>(n.members + t);
          pl.row(t) << x.values[a], y.values[a], z.values[a]; np[t] = n.values[a]; mi.row(t) << x.values[b], y.values[b], z.values[b]; nmm[t] = n.values[b]; }
        return nxr::select::track_crossings::poolSenses(pl, np, mi, nmm);
      };
      auto rho = [](const nxr::select::track_crossings::Pooled& p, int minn) { Eigen::VectorXd r(p.n.size()); for (Eigen::Index t = 0; t < p.n.size(); ++t) { const double n = p.n[t]; r[t] = n >= std::max(minn, 2) ? (p.sum.row(t).squaredNorm() - n) / (n * (n - 1)) : std::nan(""); } return r; };
      const nxr::select::track_crossings::Pooled pa = data_pooled("rest"), pbb = data_pooled("rest_b");
      const Eigen::VectorXd ra = rho(pa, 5);
      Eigen::VectorXd ex = Eigen::VectorXd::Zero(ra.size()), dex = Eigen::VectorXd::Zero(ra.size());
      const auto rd = nxr::select::track_crossings::repeat(pa, pbb, 5);
      double nsum = 0, dsum = 0, cex = 0;
      for (std::int64_t i = 0; i < nd; ++i) {
        const auto da = pooled_draw("rest", i), db = pooled_draw("rest_b", i + 100000);      // run B's seeds are offset (independent draws)
        const Eigen::VectorXd rn = rho(da, 2);
        for (Eigen::Index t = 0; t < ra.size(); ++t) if (std::isfinite(ra[t]) && std::isfinite(rn[t]) && rn[t] >= ra[t]) ex[t] += 1;
        const auto rp = nxr::select::track_crossings::repeat(da, db, 5);
        for (Eigen::Index t = 0; t < ra.size(); ++t) if (std::isfinite(rd.dot[t]) && std::isfinite(rp.dot[t]) && rp.dot[t] >= rd.dot[t]) dex[t] += 1;
        if (std::isfinite(rd.cosine) && std::isfinite(rp.cosine) && rp.cosine >= rd.cosine) cex += 1;
        nsum += da.n.sum(); dsum += pa.n.sum();
      }
      const StoredStatistic sp = read_statistic(sub, "surrogate_lagfree_vortex_compare.rest.consistency_p"), rp2 = read_statistic(sub, "vortex_cores.run_repeat.rest.rest_b.p"), wp = read_statistic(sub, "vortex_cores.run_repeat.rest.rest_b.whole_p");
      bool same = sp.values.size() == static_cast<std::size_t>(3 * ra.size()) && rp2.values.size() == sp.values.size();
      for (Eigen::Index t = 0; same && t < ra.size(); ++t) {
        const double want = std::isfinite(ra[t]) ? nxr::stats::sequentialP(static_cast<std::int64_t>(ex[t]), nd, 10) : std::nan(""), wantr = std::isfinite(rd.dot[t]) ? nxr::stats::sequentialP(static_cast<std::int64_t>(dex[t]), nd, 10) : std::nan("");
        const double got = sp.values[static_cast<std::size_t>(3 * t)], gotr = rp2.values[static_cast<std::size_t>(3 * t)];
        same = same && ((std::isnan(want) && std::isnan(got)) || std::abs(want - got) < 1e-12) && ((std::isnan(wantr) && std::isnan(gotr)) || std::abs(wantr - gotr) < 1e-12);
        if (std::isfinite(ra[t])) same = same && sp.values[static_cast<std::size_t>(3 * t + 2)] == ex[t];
      }
      std::int64_t tested = 0; for (Eigen::Index t = 0; t < ra.size(); ++t) tested += std::isfinite(ra[t]);
      std::printf("  the stored p and exceedances equal the direct count over the stored draws (%lld tiles tested, repeat tiles %lld): %s; whole p %.3f (direct %.3f)\n", static_cast<long long>(tested), static_cast<long long>(rd.tilesUsed), same ? "yes" : "NO", wp.values[0],
                  std::isfinite(rd.cosine) ? nxr::stats::sequentialP(static_cast<std::int64_t>(cex), nd, 10) : std::nan(""));
      CHECK(same && tested > 0 && text(sp.row, "measure") == "null_p" && text(sp.row, "method") == "besag_clifford");
      CHECK(!std::isfinite(rd.cosine) || std::abs(wp.values[0] - nxr::stats::sequentialP(static_cast<std::int64_t>(cex), nd, 10)) < 1e-12);
      const StoredStatistic cr = read_statistic(sub, "surrogate_lagfree_vortex_compare.rest.crossing_ratio"), nrs = read_statistic(sub, "surrogate_lagfree_vortex_compare.rest.resultant");
      std::printf("  data-to-null crossings %.2f (data %.0f, null mean %.1f); the null's mean resultant stored [%lld x %lld]\n", dsum / nsum * static_cast<double>(nd) / static_cast<double>(nd), dsum / static_cast<double>(nd), nsum / static_cast<double>(nd), static_cast<long long>(nrs.shape[0]), static_cast<long long>(nrs.shape[1]));
      CHECK(cr.values.size() == static_cast<std::size_t>(ra.size()) && nrs.shape.size() == 2 && nrs.shape[1] == 4);
      // a rerun is a hit (no draw is made again); one with every tile out of reach stops at min_draws ("decided": no tile to test)
      { SurrogateVortexCompareOptions again = no; again.replace = false; CHECK(surrogate_lagfree_vortex_compare(ds, sub, again).hit); }
      { SurrogateVortexCompareOptions n2 = no; n2.recording_b.reset(); n2.min_crossings = 100000; n2.replace = true; n2.min_draws = 3; n2.max_draws = 12;
        const auto r2 = surrogate_lagfree_vortex_compare(ds, sub, n2);
        std::printf("  with no tile to test the draws stop at the minimum: %lld (%s)\n", static_cast<long long>(r2.info["draws"].get<std::int64_t>()), r2.info["stopped"].get<std::string>().c_str());
        CHECK(r2.info["draws"] == 3 && r2.info["stopped"] == "decided"); }
      CHECK(what_of([&] { SurrogateVortexCompareOptions n3 = no; n3.recording_b = "rest"; surrogate_lagfree_vortex_compare(ds, sub, n3); }).find("same recording") != std::string::npos);
      CHECK(what_of([&] { SurrogateVortexCompareOptions n3 = no; n3.stage_name = "offset"; n3.replace = true; surrogate_lagfree_vortex_compare(ds, sub, n3); }).find("no stage 'offset'") != std::string::npos);
    }
    // refusals: a seed that is not a whole number; a power series of another recording's name that does not exist
    { Json bad = nb; bad["seed"] = -1; CHECK(what_of([&] { g::run(ds, sub, dnull, bad, all); }).find("seed") != std::string::npos); }
    { Json bad = nb; bad["power_series_statistic"] = "no.such"; CHECK(!what_of([&] { g::run(ds, sub, dnull, bad, all); }).empty()); }
  }

  // ── THE PLAN ──
  auto refused = [&](const g::Definition& d, const Json& s) { return what_of([&] { g::run(ds, sub, d, s, g::RunOptions{}); }); };
  { Json s = base; s["spatial_shape"] = "meyer";                                 // a threshold of another shape is no threshold of this chain
    CHECK(refused(dcor, s).find("another spatial_band") != std::string::npos); }
  { Json s = base; s["percentile"] = 0.0; CHECK(refused(dcor, s).find("percentile") != std::string::npos); }
  { Json s = base; s["threshold_statistic"] = "no.such.statistic"; CHECK(refused(dcor, s).find("threshold") != std::string::npos); }
  { std::string t = "pipeline: no_frames\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0}\n  tiles: {selection: ladder, manifold: kernel.to}\n"
                    "  threshold: {statistic: histogram, where: {name: vortex_core_threshold.histogram.rest}}\nsteps:\n  chunks:\n    over: recording\n    do:\n"
                    "      - {step: signal.analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}\n      - {step: meeg.apply_kernel, in: {x: z, kernel: kernel, tiles: tiles}, with: {spatial_scale: \"index:3\", transform: exact, form: curl_tan}, out: curl}\n"
                    "      - {step: select.events.cores, in: {x: curl, threshold: threshold}, out: [cores, frame, sense, centre_vertex, centre, area, peak]}\noutputs:\n  cores: {kind: spans, types: [core]}\n  frame: {kind: measurement}\n  sense: {kind: measurement}\n"
                    "  centre_vertex: {kind: measurement}\n  centre: {kind: measurement}\n  area: {kind: measurement}\n  peak: {kind: measurement}\n";
    { const std::string why = what_of([&] { g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), g::RunOptions{}); }); CHECK(why.find("frame tiling") != std::string::npos); } }
  { std::string t = "pipeline: wrong_threshold\ninputs:\n  recording: {field: recording, each: true, where: {name: rest}}\n  kernel: {operator: inverse kernel, from: recording.manifold.0}\n  tiles: {selection: ladder, manifold: kernel.to}\n"
                    "  tiling: {selection: spans, manifold: recording.time, where: {name: \"{recording} time grid\"}}\n  welch: {statistic: welch}\nsteps:\n  chunks:\n    over: recording\n    do:\n"
                    "      - {step: signal.analytic, in: {x: recording}, with: {band: [8.0, 12.0]}, out: z}\n      - {step: meeg.apply_kernel, in: {x: z, kernel: kernel, tiles: tiles}, with: {spatial_scale: \"index:3\", transform: exact, form: curl_tan}, out: curl}\n"
                    "      - {step: signal.analytic_frames, in: {x: curl, tiling: tiling}, out: cf}\n      - {step: select.events.cores, in: {x: cf, threshold: welch}, out: [cores, frame, sense, centre_vertex, centre, area, peak]}\noutputs:\n  cores: {kind: spans, types: [core]}\n  frame: {kind: measurement}\n"
                    "  sense: {kind: measurement}\n  centre_vertex: {kind: measurement}\n  centre: {kind: measurement}\n  area: {kind: measurement}\n  peak: {kind: measurement}\n";
    const std::string w = what_of([&] { g::run(ds, sub, g::parse(parse_yaml(t, "f.yaml"), "f.yaml", t), Json::object(), g::RunOptions{}); });
    CHECK(!w.empty()); }                                                         // no welch statistic stored: refused (nothing matches), never run
  return nxrtest::finish("test_vortex_cores_pipeline") || g_failures ? 1 : 0;
}
