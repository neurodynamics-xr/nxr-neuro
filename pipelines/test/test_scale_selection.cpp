// the `scaleselection` definition run through the runner on a small synthetic subject (never a real dataset): two episodes of a 12 Hz oscillation, each the pattern of ONE tile at its OWN spatial scale: A in the left hemisphere
// the level-3 pattern of a tile (8 parts a hemisphere), B in the right hemisphere the level-4 pattern (16 parts). The stored pairs are made as a user makes them: `select_product` (the spatial scale x the 8-16 Hz band's
// time grid, and the one-sample grid) and `global_bursts` at each of the two spatial scales; then `scaleselection` over the pairs, from the stored cells, baselines and labellings only. One band, one chunk.
// Checks: the cross-pair events are the episodes (the same episode seen at both pairs is ONE event); each one's best pair is the pair it was planted at; its footprint, peak, the pair count and the tile area against an
// independent computation from the stored cells and labellings (z against the tile's own median and robust SD); the span of each event; a rerun a hit; the mean response; `min_link` large splits the events; the
// refusals (a pair with no run, a repeated scale, an unknown response, a min_link below 1, a recording that is absent).
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/io.h"
#include "nxr/io/measures.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/statistic.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
template <class F> std::string why_of(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
Json out_of(const Json& r, const std::string& o) { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_scale_selection_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 4096;
  const double sfreq = 250.0;
  DatasetWriter ds = create_dataset(root, "demo");
  std::string surf, tid, cid_s;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      surf = add_surface(sub, so, b.P, b.F).id;
      const std::string cid = uuid7();
      cid_s = cid;
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
      std::vector<std::int32_t> chans(static_cast<std::size_t>(M));
      for (int i = 0; i < M; ++i) chans[static_cast<std::size_t>(i)] = i;
      SelectionSpec ss; ss.name = "kern_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {M}); ss.picked = true;
      const std::string sel = sub.selection(std::move(ss));
      std::vector<float> kid(static_cast<std::size_t>(N) * M, 0.0f);
      for (int v = 0; v < N; ++v) kid[static_cast<std::size_t>(v) * M + v] = 1.0f;
      OperatorSpec op;
      op.name = "fixed"; op.path = "fixed"; op.kind = "inverse kernel"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = N; op.n_cols = M;
      op.data = view(kid, {N, M}); op.array.compress = false;
      op.cols.set("to_manifold_id", surf).set("from_selection_id", sel).set("from_value_type", "scalar").set("to_value_type", "scalar");
      op.cols.json("params_json", Json{{"inverse", Json{{"measure", "MNE"}}}});
      sub.operator_(std::move(op));
    });
    /* the constant-Q time grid is made by `select_time_grid` (rule constant_q) after the subject is reopened */
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_tiles(ds, sub);
  const SurfaceInput S0 = read_surface(sub, std::string("cortex"), true);
  auto centres_of = [&](int level) {
    const SurfaceLevel sl = read_surface_level(sub, S0, level);
    ArrayReader r = ArrayReader::open(sub.store(), sl.record["node"].get<std::string>() + "/lc/L" + std::to_string(level) + "/centre");
    const auto raw = r.read();
    std::vector<std::int32_t> c(raw.size() / 4);
    std::memcpy(c.data(), raw.data(), raw.size());
    return c;
  };
  auto pattern = [&](int level, int tile) {
    const SurfaceLevel sl = read_surface_level(sub, S0, level);
    SpatialBandOptions o; o.transform = "exact";
    const SpatialBand sb = make_spatial_band(S0, sl, o);
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(N, 1);
    e(centres_of(level)[static_cast<std::size_t>(tile)], 0) = 1.0;
    Eigen::MatrixXd f = sb.apply(e);
    return Eigen::VectorXd(f.col(0) / f.col(0).cwiseAbs().maxCoeff());
  };
  // THE PLANTED EPISODES, the level-4 pattern of one tile each: A in the left hemisphere (tile 2), 4 s to 7 s; B in the right hemisphere (tile 16 + 5), 11 s to 13.5 s. Both at 12 Hz.
  const Eigen::VectorXd pA = pattern(4, 2), pB = pattern(4, 16 + 5);
  auto S_ = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double s = t / sfreq, osc = std::cos(2 * M_PI * 12.0 * s);
      double x = 0.005 * nd(rng);
      x += pA(v) * S_((s - 4.0) / 0.3) * S_(-(s - 7.0) / 0.4) * osc;
      x += pB(v) * S_((s - 11.0) / 0.25) * S_(-(s - 13.5) / 0.35) * osc;
      xf[static_cast<std::size_t>(v) * T + t] = static_cast<float>(x);
    }
  {
    SubjectWriter w = SubjectWriter::open(ds, "S1");
    w.compose([&] {
      FieldSpec rec;
      const std::string prod = w.product({cid_s, tid});
      rec.name = "rest"; rec.path = tk::Subject::of(w).node_path(prod, "rest"); rec.kind = "recording"; rec.manifold_id = prod;
      rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, std::int64_t(4096)}; rec.array.compress = false; rec.cols.set("unit", "A");
      w.field(std::move(rec));
    });
  }
  // THE STORED PAIR, made as a user makes it: the band's time grid and the one-sample grid, their products with the surface's level at "parts 16", and `global_bursts` over them
  const std::string grid = "rest time grid band 8-16 Hz";
  nxrtest::make_time_grids(ds, sub, "band", Json{{"band", "8-16 Hz"}, {"recording", "rest"}});
  std::string sgrid;
  for (const auto& r : nxrtest::make_time_grids(ds, sub, "tiles", Json{{"duration", "0.004 s"}, {"recording", "rest"}})) for (const auto& x : r["stored"]) if (x["output"] == "grid") sgrid = x["name"].get<std::string>();
  const std::string scale = "parts 16";
  nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"surface", "cortex"}, {"time_grid", grid}, {"couple", Json::array({Json{{"factor", 0}, {"scale", scale}}})}, {"name", "rest cells"}});
  nxrtest::run_definition(ds, sub, "select_product", Json{{"recording", "rest"}, {"surface", "cortex"}, {"time_grid", sgrid}, {"couple", Json::array({Json{{"factor", 0}, {"scale", scale}}})}, {"name", "rest samples"}});
  const auto gb = nxrtest::run_definition(ds, sub, "global_bursts", Json{{"recording", "rest"}, {"spatial_scale", scale}, {"product", "rest cells"}, {"sample_product", "rest samples"}, {"transform", "exact"}, {"spatial_shape", "meyer"},
                                                                        {"extent_enter_sd", 30.0}, {"extent_exit_sd", 15.0}});
  CHECK(gb.size() == 1 && gb[0]["hit"] == false);
  const Json sel = Json{{"recording", "rest"}, {"surface", "cortex"}, {"spatial_scales", Json::array({scale})}, {"bands", Json::array({"8-16 Hz"})}};
  auto run_sel = [&](Json set, bool replace = false) { return nxrtest::run_definition(ds, sub, "scaleselection", set, replace); };
  const auto r1 = run_sel(sel);
  CHECK(r1.size() == 1 && r1[0]["hit"] == false);
  const std::string rid = tk::Subject::of(sub).find({.cls = tk::Cls::field, .name = "rest", .any_status = true})->id;
  const std::string cells_id = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = "rest cells", .any_status = true})->id;
  const auto events = tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = "rest_scale_selection", .any_status = true});
  CHECK(events.has_value());
  auto measure = [&](const char* m) { return measurement_values(sub, tk::Subject::of(sub).find({.cls = tk::Cls::selection, .name = "rest_scale_selection", .any_status = true})->id, rid, std::nullopt, std::nullopt, m).values; };

  // ── THE INDEPENDENT FACTS, from the stored cells and the stored labelling: z = (amplitude - the tile's median) / its robust SD (1.4826 mad) over the run's time tiles ──
  const SurfaceLevel SL = read_surface_level(sub, S0, 4);
  const std::int64_t tiles = SL.nodes, frames = T / 128, W = 128;
  const MeasurementValues amp = measurement_values(sub, cells_id, rid, std::nullopt, std::nullopt, "amplitude");
  CHECK(amp.members == tiles * frames);                                               // the product's order: the space tile fastest, code = tile + tiles x time tile
  auto median = [](std::vector<double> x) { std::sort(x.begin(), x.end()); const std::size_t n = x.size(); return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]); };
  Eigen::MatrixXd z(tiles, frames);
  for (std::int64_t s = 0; s < tiles; ++s) {
    std::vector<double> x, dev;
    for (std::int64_t t = 0; t < frames; ++t) x.push_back(amp.values[static_cast<std::size_t>(s + tiles * t)]);
    const double med = median(x);
    for (const double v : x) dev.push_back(std::abs(v - med));
    const double sd = 1.4826 * median(dev);
    for (std::int64_t t = 0; t < frames; ++t) z(s, t) = (x[static_cast<std::size_t>(t)] - med) / sd;
  }
  nxr::io::task::Want ext_want{.cls = nxr::io::task::Cls::selection};
  ext_want.name = "global_bursts.extents.rest";
  const nxr::io::task::Node ext = nxr::io::task::Subject::of(sub).require(ext_want, "the burst extents");
  std::vector<std::int32_t> cell_member;
  { ArrayReader lr = ArrayReader::open(sub.store(), *ext.array_path); const auto raw = lr.read(); cell_member.resize(raw.size() / 4); std::memcpy(cell_member.data(), raw.data(), raw.size()); }
  CHECK(static_cast<std::int64_t>(cell_member.size()) == tiles * frames);
  const std::vector<std::int64_t> event_of_member = nxr::io::selection::member_labels(sub, ext.id);
  struct Fact { double peak = -1e300, mean = 0; int cells = 0; std::int64_t first = 1 << 30, last = -1; };
  std::map<std::int64_t, Fact> by_event;
  for (std::int64_t s = 0; s < tiles; ++s)
    for (std::int64_t t = 0; t < frames; ++t) {
      const std::int32_t m = cell_member[static_cast<std::size_t>(s + tiles * t)];
      if (m < 0 || static_cast<std::size_t>(m) >= event_of_member.size()) continue;
      Fact& f = by_event[event_of_member[static_cast<std::size_t>(m)]];
      f.peak = std::max(f.peak, z(s, t)); f.mean += z(s, t); ++f.cells; f.first = std::min(f.first, t); f.last = std::max(f.last, t);
    }
  std::vector<Fact> facts;                                                            // the cross-pair events are numbered by their earliest sample
  for (auto& [e, f] : by_event) { f.mean /= f.cells; facts.push_back(f); }
  std::sort(facts.begin(), facts.end(), [](const Fact& a, const Fact& b) { return a.first < b.first; });
  CHECK(facts.size() == 2);                                                           // the two planted episodes, each seen at the one pair
  if (facts.size() != 2) return 1;
  std::printf("  pair events: A tiles %lld-%lld peak z %.1f (%d cells); B tiles %lld-%lld peak z %.1f (%d cells)\n", static_cast<long long>(facts[0].first), static_cast<long long>(facts[0].last), facts[0].peak, facts[0].cells,
              static_cast<long long>(facts[1].first), static_cast<long long>(facts[1].last), facts[1].peak, facts[1].cells);
  CHECK((facts[0].last + 1) * W / sfreq > 4.0 && facts[0].first * W / sfreq < 7.0 && (facts[1].last + 1) * W / sfreq > 11.0 && facts[1].first * W / sfreq < 13.5);   // each lies on its planted episode

  // ── THE CROSS-PAIR EVENTS: the facts, the pair and the span ──
  const auto peak = measure("scale_peak"), level = measure("scale_space_level"), tlevel = measure("scale_time_level"), pairs = measure("scale_pairs"), edge = measure("scale_edge"), lo = measure("scale_band_lo"), hi = measure("scale_band_hi"),
             area = measure("scale_tile_area"), foot = measure("scale_response_s4_t7");
  CHECK(peak.size() == 2 && foot.size() == 2);
  double mean_area = 0;
  for (const double x : SL.tile_area) mean_area += x;
  mean_area /= static_cast<double>(SL.nodes);
  for (int k = 0; k < 2; ++k) {
    CHECK(std::abs(peak[static_cast<std::size_t>(k)] - facts[static_cast<std::size_t>(k)].peak) < 1e-6 * facts[static_cast<std::size_t>(k)].peak);           // the response is the event's peak z
    CHECK(foot[static_cast<std::size_t>(k)] == peak[static_cast<std::size_t>(k)]);                                                                         // its footprint at the one pair is the same
    CHECK(level[static_cast<std::size_t>(k)] == 4 && tlevel[static_cast<std::size_t>(k)] == 7 && pairs[static_cast<std::size_t>(k)] == 1 && edge[static_cast<std::size_t>(k)] == 0);
    CHECK(lo[static_cast<std::size_t>(k)] == 8.0 && hi[static_cast<std::size_t>(k)] == 16.0 && std::abs(area[static_cast<std::size_t>(k)] - mean_area) < 1e-9 * mean_area);
  }
  {   // the spans Selection: the event's time tiles, in seconds
    const auto spans = tk::Subject::of(sub).spans(*events);
    CHECK(spans.size() == 2);
    for (std::size_t k = 0; k < spans.size() && k < 2; ++k) CHECK(std::abs(spans[k].start - facts[k].first * W / sfreq) < 1e-6 && std::abs(spans[k].stop - (facts[k].last + 1) * W / sfreq) < 1e-6);
  }
  {   // the links: the cross-pair event of each of the pair's events
    const StoredStatistic lk = read_statistic(sub, "rest_scale_selection.links");
    CHECK(lk.shape == (std::vector<std::int64_t>{1, 2}));
    std::set<int> cross;
    for (const double v : lk.values) cross.insert(static_cast<int>(v));
    CHECK(cross == (std::set<int>{0, 1}));
  }
  // a rerun is a hit; the mean response is another option, made with --replace, and the selection is then on the events' mean z
  CHECK(run_sel(sel)[0]["hit"] == true);
  {
    Json mean = sel; mean["response"] = "mean";
    const auto r = run_sel(mean, true);
    CHECK(r[0]["hit"] == false);
    const auto m2 = measure("scale_peak");
    for (int k = 0; k < 2; ++k) CHECK(std::abs(m2[static_cast<std::size_t>(k)] - facts[static_cast<std::size_t>(k)].mean) < 1e-6 * facts[static_cast<std::size_t>(k)].mean);
  }
  // THE REFUSALS, each with its reason
  {
    Json o = sel; o["spatial_scales"] = Json::array({"parts 8"});
    const std::string w = why_of([&] { run_sel(o, true); });   // (replace: the refusal of an existing result of other options comes first)
    CHECK(w.find("space level 3 x time level 7") != std::string::npos && w.find("select_product") != std::string::npos);   // a pair without stored cells: named
    o = sel; o["bands"] = Json::array({"4-8 Hz"});
    CHECK(why_of([&] { run_sel(o, true); }).find("space level 4 x time level 8") != std::string::npos);                       // no time grid of that band: the command that makes it
    o = sel; o["spatial_scales"] = Json::array({scale, scale});
    CHECK(why_of([&] { run_sel(o); }).find("listed twice") != std::string::npos);
    o = sel; o["response"] = "max";
    CHECK(why_of([&] { run_sel(o); }).find("peak") != std::string::npos);
    o = sel; o["min_link"] = 0;
    CHECK(why_of([&] { run_sel(o); }).find("min_link") != std::string::npos);
    o = sel; o["recording"] = "nope";
    CHECK(!why_of([&] { run_sel(o); }).empty());
    o = sel; o["source"] = "cortical_band";
    CHECK(why_of([&] { run_sel(o, true); }).find("cortical_band") != std::string::npos);                          // labellings made by another pipeline are not used
  }
  return 0;
}
