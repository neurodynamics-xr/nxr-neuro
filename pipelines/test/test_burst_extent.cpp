// run global_bursts: WHERE per burst (spec docs/specs/2026-11-13-cortical-events.md, "Phase 2b: where per global burst") on a synthetic source space (never a real dataset): one episode moving from the level's
// own pattern of one tile to a neighbour's inside ONE global burst (left hemisphere), a static one elsewhere (right hemisphere, a second burst). Checks: the labelling of the joint cells against an
// independent flood fill of the stored cells (the per-tile baseline, the hysteresis, the graph of touching tiles and the next time tile, confined to the burst); the planted tiles in the events and the
// event's strongest tile moving from the first to the second; every measure of each event against a direct computation from the labelling (cells, tiles, the union area, the weighted amplitude, the peak,
// first and last time tile, duration, the burst it belongs to, the area and centre per time tile, the path length); the statistics; the Selection's record; a rerun a hit, another threshold refused
// without --replace; no event with a high cell minimum; per hemisphere the events confined to their hemisphere; the plan's refusals.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "store_at.h"
#include "nxr/io.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/io/yaml.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/runner.h"
#include "nxr/pipelines/spatial_band.h"
#include "make_grids.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::vector<float> read_floats(const fs::path& p, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = nxrtest::open_array(p);
  if (shape) *shape = r.shape();
  const auto raw = r.read();
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}
Json out_of(const Json& r, const std::string& o) { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); }
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_burst_extent_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(2, false);
  const int N = b.n, M = N, T = 12288;
  const double sfreq = 250.0;
  const int L = 4;
  const std::int64_t W = 256;
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
  nxrtest::make_time_grids(ds, sub);
  nxrtest::make_tiles(ds, sub);
  const SurfaceInput S0 = read_surface(sub, std::nullopt, true);
  const SurfaceLevel SL = read_surface_level(sub, S0, L);
  SpatialBandOptions sbo; sbo.transform = "exact";
  const SpatialBand sband = make_spatial_band(S0, SL, sbo);
  int timeLevel = -1;
  for (const tk::Node& m : tk::Subject::of(sub).find_all({.cls = tk::Cls::selection, .params_present = {"resolves_hz"}, .any_status = true})) {
    if (!m.member_of_id) continue;
    const Json& pj = m.params;
    if (pj["resolves_hz"]["lo"] == 8.0 && pj["resolves_hz"]["hi"] == 16.0 && pj["window_samples"] == W) timeLevel = static_cast<int>(m.member_ordinal.value_or(0));
  }
  CHECK(timeLevel >= 0);
  if (timeLevel < 0) return 1;

  // THE PLANTED EPISODES: the level's own pattern of one tile in each hemisphere (left: tile 2; right: tile 16 + 5), 12 Hz under a double sigmoid a = S((t - on)/tr) S(-(t - off)/tf)
  const std::vector<std::int32_t> centres = [&] {
    ArrayReader r = ArrayReader::open(sub.store(), SL.record["node"].get<std::string>() + "/lc/L" + std::to_string(L) + "/centre");
    const auto raw = r.read();
    std::vector<std::int32_t> c(raw.size() / 4);
    std::memcpy(c.data(), raw.data(), raw.size());
    return c;
  }();
  // EPISODE 1 (moving, left hemisphere): the level's own pattern of tile A = 2 hands over to the pattern of a NEIGHBOUR tile B around 12.5 s (both on between 11.5 s and 13.5 s), 8 s to 17 s, one global burst.
  // EPISODE 2 (static, right hemisphere): tile 16 + 5 alone, 30 s to 35 s: a second burst.
  const int tileA = 2;
  int tileB = -1;
  for (const auto& [a, b2] : SL.pairs) { if (a == tileA && b2 < 16) { tileB = b2; break; } if (b2 == tileA && a < 16) { tileB = a; break; } }
  const int tileC = 16 + 5;
  CHECK(tileB >= 0);
  auto pattern = [&](int tile) { Eigen::MatrixXd e = Eigen::MatrixXd::Zero(N, 1); e(centres[static_cast<std::size_t>(tile)], 0) = 1.0; Eigen::MatrixXd f = sband.apply(e); return Eigen::VectorXd(f.col(0) / f.col(0).cwiseAbs().maxCoeff()); };
  const Eigen::VectorXd pA = pattern(tileA), pB = pattern(tileB), pC = pattern(tileC);
  auto S_ = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  std::mt19937 rng(21);
  std::normal_distribution<double> nd;
  std::vector<float> xf(static_cast<std::size_t>(M) * T);
  for (int v = 0; v < M; ++v)
    for (int t = 0; t < T; ++t) {
      const double s = t / sfreq, osc = std::cos(2 * M_PI * 12.0 * s);
      double x = 0.005 * nd(rng);
      const double a1 = S_((s - 8.0) / 0.3) * S_(-(s - 17.0) / 0.4), wA = S_(-(s - 13.5) / 0.5), wB = S_((s - 11.5) / 0.5);   // A fades out by 13.5 s, B fades in from 11.5 s: both on at 12.5 s
      x += a1 * osc * (wA * pA(v) + wB * pB(v));
      x += pC(v) * S_((s - 30.0) / 0.25) * S_(-(s - 35.0) / 0.35) * osc;
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
  JointComputeOptions jo; jo.recording = "rest"; jo.space_level = L; jo.time_level = timeLevel;
  joint_compute(ds, sub, jo);

  // ── THE CELLS (run cortical_cells, the same chain: the reference for the independent detection) ──
  const g::Definition dc = g::load("cortical_cells");
  g::RunOptions all; all.chunks = "all";
  Json setc = Json{{"space_level", L}, {"transform", "exact"}, {"spatial_shape", "meyer"}, {"shape", "meyer"}};
  const auto rc = g::run(ds, sub, dc, setc, all);
  std::vector<std::int64_t> cshape;
  const auto cells = nxrtest::read_packed(sub.folder(), out_of(rc[0], "amplitude"), &cshape);
  const int S = static_cast<int>(cshape[0]), F = static_cast<int>(cshape[1]);
  CHECK(S == 32 && F == 48);
  auto cell = [&](int s, int f) { return static_cast<double>(cells[static_cast<std::size_t>((s * F + f) * 2)]); };
  const double width_s = static_cast<double>(W) / sfreq;
  // the tiles' areas and vertex areas, from the mesh directly
  nxr::ladder::Mesh mesh; mesh.n = N; mesh.positions = S0.vertices; mesh.faces = S0.faces;
  const std::vector<double> varea = nxr::ladder::vertexAreas(mesh);
  std::vector<double> tarea(static_cast<std::size_t>(S), 0.0);
  for (int v = 0; v < N; ++v) if (SL.tile[static_cast<std::size_t>(v)] >= 0) tarea[static_cast<std::size_t>(SL.tile[static_cast<std::size_t>(v)])] += varea[static_cast<std::size_t>(v)];
  std::vector<std::vector<int>> nbr(static_cast<std::size_t>(S));
  for (const auto& [a, b2] : SL.pairs) { nbr[static_cast<std::size_t>(a)].push_back(b2); nbr[static_cast<std::size_t>(b2)].push_back(a); }
  auto median = [](std::vector<double> x) { std::sort(x.begin(), x.end()); const std::size_t n = x.size(); return n % 2 ? x[n / 2] : 0.5 * (x[n / 2 - 1] + x[n / 2]); };
  std::vector<double> med(static_cast<std::size_t>(S)), sd(static_cast<std::size_t>(S));
  for (int s = 0; s < S; ++s) {
    std::vector<double> x;
    for (int f = 0; f < F; ++f) if (std::isfinite(cell(s, f))) x.push_back(cell(s, f));
    med[static_cast<std::size_t>(s)] = median(x);
    std::vector<double> dv;
    for (const double v : x) dv.push_back(std::abs(v - med[static_cast<std::size_t>(s)]));
    sd[static_cast<std::size_t>(s)] = 1.4826 * median(dv);
  }
  auto z = [&](int s, int f) { return (cell(s, f) - med[static_cast<std::size_t>(s)]) / sd[static_cast<std::size_t>(s)]; };

  // the labelling read back: the Selection's array [S x F] int32
  auto read_labels = [&](const Json& r, std::vector<std::int32_t>& lab) {
    const Json x = out_of(r, "extents");
    ArrayReader rd = ArrayReader::open(sub.store(), x["path"].get<std::string>());
    CHECK(rd.shape() == (std::vector<std::int64_t>{S, F}));
    const auto raw = rd.read();
    lab.resize(raw.size() / 4);
    std::memcpy(lab.data(), raw.data(), raw.size());
    return x;
  };
  auto measure_of = [&](const Json& r, const std::string& o, std::int64_t& rows, std::int64_t& cols) {
    const Json x = out_of(r, o);
    CHECK(!x.is_null());
    std::vector<std::int64_t> shape;
    const auto a = nxrtest::read_packed(sub.folder(), x, &shape);
    rows = shape[0]; cols = shape[1];
    return a;
  };
  // the bursts found, in time tiles [first, stop) with their row
  auto bursts_of = [&](const Json& r) {
    const Json bj = out_of(r, "bursts");
    const auto ts = tk::Subject::of(sub);
    const auto sel = ts.find({.cls = tk::Cls::selection, .name = bj["selection"].get<std::string>(), .any_status = true});
    std::vector<std::array<int, 3>> out;
    const Json pj = sel->params.is_null() ? Json::object() : sel->params;
    int k = 0;
    for (const tk::Span& x : ts.spans(*sel)) {
      const int row = pj.contains("channels") && pj["channels"].contains(std::to_string(k)) ? pj["channels"][std::to_string(k)][0].get<int>() : -1;
      out.push_back({static_cast<int>(std::llround(x.start / width_s)), static_cast<int>(std::llround(x.stop / width_s)), row});
      ++k;
    }
    return out;
  };
  // THE INDEPENDENT DETECTION: a flood fill, per burst window (and row's hemisphere when per_hemisphere), the per-tile baseline above, hysteresis enter / exit on z
  auto flood = [&](const std::vector<std::array<int, 3>>& bursts, double enter, double exit, int min_cells, bool per_hemisphere) {
    std::vector<std::int32_t> lab(static_cast<std::size_t>(S) * static_cast<std::size_t>(F), -1);
    int next = 0;
    for (const auto& bu : bursts) {
      std::vector<std::int32_t> comp(static_cast<std::size_t>(S) * static_cast<std::size_t>(F), -1);
      auto ok = [&](int s, int f) { return f >= bu[0] && f < bu[1] && std::isfinite(sd[static_cast<std::size_t>(s)]) && sd[static_cast<std::size_t>(s)] > 0 && std::isfinite(cell(s, f)) && z(s, f) >= exit && (!per_hemisphere || bu[2] < 0 || (s >> L) == bu[2]); };
      std::vector<std::vector<int>> comps;
      for (int f = bu[0]; f < bu[1]; ++f)
        for (int s = 0; s < S; ++s) {
          if (!ok(s, f) || comp[static_cast<std::size_t>(f * S + s)] >= 0) continue;
          std::vector<int> stack{f * S + s}, members;
          comp[static_cast<std::size_t>(f * S + s)] = static_cast<std::int32_t>(comps.size());
          while (!stack.empty()) {
            const int n = stack.back(); stack.pop_back(); members.push_back(n);
            const int ns = n % S, nf = n / S;
            std::vector<int> adj;
            for (const int t : nbr[static_cast<std::size_t>(ns)]) adj.push_back(nf * S + t);
            if (nf > 0) adj.push_back((nf - 1) * S + ns);
            if (nf + 1 < F) adj.push_back((nf + 1) * S + ns);
            for (const int m : adj) if (ok(m % S, m / S) && comp[static_cast<std::size_t>(m)] < 0) { comp[static_cast<std::size_t>(m)] = comp[static_cast<std::size_t>(n)]; stack.push_back(m); }
          }
          std::sort(members.begin(), members.end());
          comps.push_back(members);
        }
      std::sort(comps.begin(), comps.end());           // by the lowest cell (frame, then tile)
      for (const auto& c : comps) {
        bool hi = false;
        for (const int n : c) hi = hi || z(n % S, n / S) >= enter;
        if (!hi || static_cast<int>(c.size()) < min_cells) continue;
        for (const int n : c) lab[static_cast<std::size_t>(n % S) * static_cast<std::size_t>(F) + static_cast<std::size_t>(n / S)] = next;
        ++next;
      }
    }
    return std::make_pair(lab, next);
  };

  // ── WHOLE BRAIN: two global bursts; the moving episode one event, the static one another ──
  const g::Definition d = g::load("global_bursts");
  Json set1 = Json{{"space_level", L}, {"transform", "exact"}, {"spatial_shape", "meyer"}, {"extent_enter_sd", 7000.0}, {"extent_exit_sd", 4000.0}, {"min_cells", 2}, {"shape", "meyer"}};
  const auto r1 = g::run(ds, sub, d, set1, all);
  const auto bursts = bursts_of(r1[0]);
  std::printf("  bursts (time tiles):");
  for (const auto& bu : bursts) std::printf(" [%d, %d)", bu[0], bu[1]);
  std::printf("\n");
  CHECK_EQ(static_cast<int>(bursts.size()), 2);
  std::vector<std::int32_t> lab;
  const Json ext = read_labels(r1[0], lab);
  const auto [expected, nexp] = flood(bursts, 7000.0, 4000.0, 2, false);
  int mismatches = 0, labelled = 0;
  for (std::size_t i = 0; i < lab.size(); ++i) { mismatches += lab[i] != expected[i]; labelled += lab[i] >= 0; }
  std::printf("  events %d (independent flood fill: %d); labelled cells %d; mismatching cells %d\n", static_cast<int>(ext["events"].get<int>()), nexp, labelled, mismatches);
  CHECK_EQ(mismatches, 0);
  CHECK_EQ(ext["events"].get<int>(), nexp);
  CHECK_EQ(nexp, 2);
  auto L_ = [&](int s, int f) { return lab[static_cast<std::size_t>(s) * static_cast<std::size_t>(F) + static_cast<std::size_t>(f)]; };
  // the planted tiles are in their events at time tiles inside the episode; the strongest labelled tile of event 0 moves from A to B
  auto strongest = [&](int ev, int f) { int best = -1; for (int s = 0; s < S; ++s) if (L_(s, f) == ev && (best < 0 || cell(s, f) > cell(best, f))) best = s; return best; };
  const int fa = static_cast<int>(10.0 / width_s), fb = static_cast<int>(15.5 / width_s);   // well before and well after the 12.5 s hand-over
  std::printf("  event 0: strongest tile at time tile %d: %d (planted A %d); at time tile %d: %d (planted B %d)\n", fa, strongest(0, fa), tileA, fb, strongest(0, fb), tileB);
  CHECK_EQ(strongest(0, fa), tileA);
  CHECK_EQ(strongest(0, fb), tileB);
  CHECK(L_(tileA, fa) == 0 && L_(tileB, fb) == 0);
  const int fc = static_cast<int>(32.5 / width_s);
  CHECK_EQ(strongest(1, fc), tileC);
  for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) if (L_(s, f) >= 0) CHECK((L_(s, f) == 0) == (f >= bursts[0][0] && f < bursts[0][1]));   // an event lies inside its burst's time tiles
  // the Selection's record
  {
    const auto row = tk::Subject::of(sub).by_id(tk::Cls::selection, ext["id"].get<std::string>());
    CHECK(row.has_value());
    const Json pj = row->params.is_null() ? Json::object() : row->params;
    CHECK(row->type.value_or("") == "set" && row->n_members.value_or(0) == 2 && row->n_elements.value_or(0) == S * F);
    CHECK(pj["extent"]["thresholds"]["enter_sd"] == 7000.0 && pj["extent"]["space_level"] == L && pj["cells_of"]["space_level"] == L);
    CHECK(pj.contains("bursts") && pj["bursts"]["selection"].is_string() && pj.contains("cache") && pj["extent"]["baseline"]["summaries"].size() == 2);
    CHECK(pj["extent"]["bursts"].size() == 2 && pj["extent"]["event_list"][0]["burst"] == 0 && pj["extent"]["event_list"][1]["burst"] == 1);
  }
  // THE MEASURES, each against a direct computation from the labelling
  std::int64_t R = 0, E = 0;
  const auto ncells = measure_of(r1[0], "extent_n_cells", R, E), ntiles = measure_of(r1[0], "extent_n_tiles", R, E), area = measure_of(r1[0], "extent_area", R, E), amp = measure_of(r1[0], "extent_amplitude", R, E);
  const auto first = measure_of(r1[0], "extent_first_tile", R, E), last = measure_of(r1[0], "extent_last_tile", R, E), dur = measure_of(r1[0], "extent_duration", R, E), bor = measure_of(r1[0], "extent_burst", R, E);
  const auto pathl = measure_of(r1[0], "extent_path_length", R, E);
  CHECK(R == 1 && E == 2);
  const double kSamples = static_cast<double>(W);
  std::vector<double> path_direct(2, 0.0);
  for (int e = 0; e < 2; ++e) {
    int nc = 0, f0 = F, f1 = -1;
    std::set<int> tiles;
    double wsum = 0, asum = 0;
    for (int f = 0; f < F; ++f)
      for (int s = 0; s < S; ++s)
        if (L_(s, f) == e) { ++nc; tiles.insert(s); f0 = std::min(f0, f); f1 = std::max(f1, f); wsum += tarea[static_cast<std::size_t>(s)] * kSamples; asum += tarea[static_cast<std::size_t>(s)] * kSamples * cell(s, f); }
    double ua = 0;
    for (const int s : tiles) ua += tarea[static_cast<std::size_t>(s)];
    const std::size_t i = static_cast<std::size_t>(e * 2);
    std::printf("  event %d: %d cells on %zu tiles, area %.4f, time tiles %d … %d, amplitude %.4f, path %.4f\n", e, nc, tiles.size(), area[i], f0, f1, amp[i], pathl[i]);
    CHECK(static_cast<int>(ncells[i]) == nc && static_cast<std::size_t>(ntiles[i]) == tiles.size());
    CHECK(std::abs(area[i] - ua) < 1e-5 * ua);                                                    // float32 storage
    CHECK(std::abs(amp[i] - asum / wsum) < 1e-5 * (asum / wsum));
    CHECK(static_cast<int>(first[i]) == f0 && static_cast<int>(last[i]) == f1 && std::abs(dur[i] - (f1 - f0 + 1) * width_s) < 1e-5 && static_cast<int>(bor[i]) == e);
    // the centre per time tile: the vertex nearest the area-weighted centroid of the vertices of the event's tiles, and the chord path
    std::vector<int> centre_of(static_cast<std::size_t>(F), -1);
    for (int f = f0; f <= f1; ++f) {
      double w = 0; double c3[3] = {0, 0, 0};
      for (int v = 0; v < N; ++v) { const int s = SL.tile[static_cast<std::size_t>(v)]; if (s >= 0 && L_(s, f) == e) { w += varea[static_cast<std::size_t>(v)]; for (int k = 0; k < 3; ++k) c3[k] += varea[static_cast<std::size_t>(v)] * S0.vertices[static_cast<std::size_t>(v) * 3 + k]; } }
      double best = 1e300;
      for (int v = 0; v < N; ++v) {
        const int s = SL.tile[static_cast<std::size_t>(v)];
        if (s < 0 || L_(s, f) != e) continue;
        double d2 = 0;
        for (int k = 0; k < 3; ++k) { const double dd = S0.vertices[static_cast<std::size_t>(v) * 3 + k] - c3[k] / w; d2 += dd * dd; }
        if (d2 < best) { best = d2; centre_of[static_cast<std::size_t>(f)] = v; }
      }
    }
    for (int f = f0; f < f1; ++f) {
      double d2 = 0;
      for (int k = 0; k < 3; ++k) { const double dd = S0.vertices[static_cast<std::size_t>(centre_of[static_cast<std::size_t>(f)]) * 3 + k] - S0.vertices[static_cast<std::size_t>(centre_of[static_cast<std::size_t>(f + 1)]) * 3 + k]; d2 += dd * dd; }
      path_direct[static_cast<std::size_t>(e)] += std::sqrt(d2);
    }
    CHECK(std::abs(pathl[i] - path_direct[static_cast<std::size_t>(e)]) < 1e-5 * std::max(1.0, path_direct[static_cast<std::size_t>(e)]));
    // the per-time-tile measures (rows = the events, members = the time level's tiles)
    std::int64_t Rr = 0, Tt = 0;
    const auto apt = measure_of(r1[0], "extent_area_by_tile", Rr, Tt), cpt = measure_of(r1[0], "extent_centre_by_tile", Rr, Tt);
    CHECK(Rr == 2 && Tt == F);
    for (int f = 0; f < F; ++f) {
      double a = 0; bool any = false;
      for (int s = 0; s < S; ++s) if (L_(s, f) == e) { a += tarea[static_cast<std::size_t>(s)]; any = true; }
      const std::size_t k = static_cast<std::size_t>((e * F + f) * 2);
      if (any) CHECK(std::abs(apt[k] - a) < 1e-5 * a && static_cast<int>(cpt[k]) == centre_of[static_cast<std::size_t>(f)]);
      else CHECK(std::isnan(apt[k]) && std::isnan(cpt[k]));
    }
  }
  CHECK(path_direct[0] > 0 && path_direct[1] < 0.5 * path_direct[0]);          // the moving episode's centre travels, the static one's hardly
  std::printf("  path length: moving %.4f, static %.4f\n", path_direct[0], path_direct[1]);
  // the peak: the largest cell peak of the cells (envelope_max of the same cells, from run cortical_cells)
  {
    std::vector<std::int64_t> sh;
    const auto cp = nxrtest::read_packed(sub.folder(), out_of(rc[0], "peak"), &sh);
    std::int64_t r = 0, e = 0;
    const auto pk = measure_of(r1[0], "extent_peak", r, e);
    for (int ev = 0; ev < 2; ++ev) {
      double mx = -1;
      for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) if (L_(s, f) == ev) mx = std::max(mx, static_cast<double>(cp[static_cast<std::size_t>((s * F + f) * 2)]));
      CHECK(std::abs(pk[static_cast<std::size_t>(ev * 2)] - mx) < 1e-6 * mx);
    }
  }
  // the baseline and the statistics across the events are stored
  for (const char* o : {"cell_baseline", "extent_n_cells_stats", "extent_area_stats", "extent_amplitude_stats", "extent_peak_stats", "extent_duration_stats", "extent_path_length_stats"}) CHECK(!out_of(r1[0], o).is_null());
  {
    std::vector<tk::Node> st;
    for (const tk::Node& q : tk::Subject::of(sub).find_all({.cls = tk::Cls::statistic, .any_status = true})) if (q.name.find("extent_area_stats") != std::string::npos) st.push_back(q);
    CHECK(!st.empty() && st.front().n_members.value_or(0) == 2);
  }
  // a rerun is a hit; another threshold is refused without --replace
  { const auto r = g::run(ds, sub, d, set1, all); CHECK(r.size() == 1 && r[0]["hit"] == true); }
  {
    Json other = set1; other["extent_enter_sd"] = 6500.0;
    CHECK(code_of([&] { g::run(ds, sub, d, other, all); }) == std::string(code::already_exists));
  }
  g::RunOptions rep = all; rep.replace = true;
  // ── a high minimum of cells: no event; the labelling is stored (all -1) and nothing breaks ──
  {
    Json none = set1; none["min_cells"] = 100000;
    const auto r = g::run(ds, sub, d, none, rep);
    std::vector<std::int32_t> l0;
    const Json x = read_labels(r[0], l0);
    int any = 0;
    for (const auto v : l0) any += v >= 0;
    const auto row = tk::Subject::of(sub).by_id(tk::Cls::selection, x["id"].get<std::string>());
    CHECK(any == 0 && x["events"] == 0 && row->n_members.value_or(-1) == 0);
    const auto rr = g::run(ds, sub, d, none, all);
    CHECK(rr[0]["hit"] == true);                                              // an all -1 labelling is a hit too
  }
  // ── PER HEMISPHERE: the burst's row is its hemisphere; its events lie in it ──
  {
    Json set2 = set1; set2["groups"] = "per_hemisphere"; set2["combine"] = "each";
    const auto r = g::run(ds, sub, d, set2, rep);
    const auto b2 = bursts_of(r[0]);
    std::vector<std::int32_t> l2;
    const Json x = read_labels(r[0], l2);
    const auto [exp2, n2] = flood(b2, 7000.0, 4000.0, 2, true);
    int mm = 0;
    for (std::size_t i = 0; i < l2.size(); ++i) mm += l2[i] != exp2[i];
    std::printf("  per hemisphere: %zu bursts, %d events, mismatches %d\n", b2.size(), x["events"].get<int>(), mm);
    CHECK_EQ(mm, 0);
    CHECK_EQ(x["events"].get<int>(), 2);
    bool left = true, right = true;
    for (int f = 0; f < F; ++f) for (int s = 0; s < S; ++s) { const auto v = l2[static_cast<std::size_t>(s) * static_cast<std::size_t>(F) + static_cast<std::size_t>(f)]; if (v == 0 && s >= 16) left = false; if (v == 1 && s < 16) right = false; }
    CHECK(left && right);
  }
  // ── REFUSALS (the plan's, before any data) ──
  {
    auto why = [&](const Json& set) { std::string w; try { g::plan_runs(d, sub.db(), sub.id(), set); } catch (const Error& e) { w = e.what(); } return w; };
    Json bad = set1; bad["extent_enter_sd"] = 1.0; bad["extent_exit_sd"] = 2.0;
    CHECK(why(bad).find("enter_sd is at least exit_sd") != std::string::npos);
    bad = set1; bad["min_cells"] = 0;
    CHECK(why(bad).find("min_cells") != std::string::npos);
    bad = set1; bad["min_area"] = -1.0;
    CHECK(why(bad).find("min_area") != std::string::npos);
    // a definition that asks for a measure the step does not have, and one that gives cell_extent the wrong events
    const fs::path ydir = root / "defs";
    fs::create_directories(ydir);
    std::string yaml;
    yaml = d.text;
    auto write = [&](const std::string& name, std::string text) { std::ofstream(ydir / name) << text; return (ydir / name).string(); };
    auto replace1 = [](std::string t, const std::string& a, const std::string& b2) { const auto i = t.find(a); if (i != std::string::npos) t.replace(i, a.size(), b2); return t; };
    const g::Definition dm = g::load(write("measure.yaml", replace1(yaml, "with: {measure: n_cells}", "with: {measure: nonsense}")));
    std::string w;
    try { g::plan_runs(dm, sub.db(), sub.id(), set1); } catch (const Error& e) { w = e.what(); }
    CHECK(w.find("extent_measure: measure") != std::string::npos);
    const g::Definition dp = g::load(write("peak.yaml", replace1(yaml, "{step: extent_measure, in: {labels: extents, x: cell_peak},      with: {measure: envelope_max}", "{step: extent_measure, in: {labels: extents, x: cell_amplitude}, with: {measure: envelope_max}")));
    w.clear();
    try { g::plan_runs(dp, sub.db(), sub.id(), set1); } catch (const Error& e) { w = e.what(); }
    CHECK(w.find("extent_measure: x must be the `envelope_max`") != std::string::npos);
    const g::Definition dn = g::load(write("amp.yaml", replace1(yaml, "in: {labels: extents, x: cell_amplitude}, with: {measure: amplitude}", "in: {labels: extents}, with: {measure: amplitude}")));
    w.clear();
    try { g::plan_runs(dn, sub.db(), sub.id(), set1); } catch (const Error& e) { w = e.what(); }
    CHECK(w.find("give x") != std::string::npos);
  }
  return nxrtest::finish("test_burst_extent");
}
