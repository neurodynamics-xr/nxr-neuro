// gabor_bursts (spec docs/specs/2026-11-20-gabor-bursts.md) on a synthetic recording (never a real dataset): 3 channels at 250 Hz, 40 s of white noise (SD 0.02) with planted Gaussian bursts
// A e^(−(t−t0)²/2s²) cos(2πf(t−t0) + φ) of known (t0, f, s, A, φ); a narrower grid than the default (10-30 Hz, σ 50-400 ms, 2 widths an octave, 16 ms tiles) so the test stays small.
// Checks: each planted burst recovered once, on its channel, within its resolution (t0 within a tile, f within σ_f/4, σ within 5 %, A within 5 %, φ within 0.2 rad); every burst passes the gates
// (a peak along width, at least 2 cycles, R² at least 0.7) and a burst narrower than the ladder is a candidate but no burst; a stricter R² keeps fewer; tangent edges against half-maximum edges;
// the same bursts to the last bit with another chunk width; the global windows by time-frequency overlap (and by time alone).
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <random>
#include <tuple>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/measures.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "nxr/pipelines/runner.h"
#include "make_grids.h"

using namespace nxr::io;
namespace g = nxr::pipelines::graph;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
constexpr double kPi = 3.14159265358979323846;
struct Planted { int channel; double t0, f, s, A, phi; };
struct Burst { int channel; double start, stop, t0, f, s, A, phi, cycles, r2, width_peak, onset, offset; };
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_gabor_bursts_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const int M = 3;
  const double sfreq = 250.0;
  const int T = 10000;                                                                // 40 s
  const std::vector<Planted> planted = {{0, 15.0, 12.0, 0.15, 1.0, 0.7}, {0, 30.0, 25.0, 0.08, 1.0, -1.0}, {1, 22.0, 18.0, 0.30, 0.8, 2.0},
                                        {2, 15.05, 12.5, 0.12, 0.9, 0.0}, {2, 30.05, 12.0, 0.15, 0.9, 1.5}};
  const Planted narrow{1, 9.0, 20.0, 0.02, 1.0, 0.0};                                // below the ladder (σ 50 ms): a candidate, never a burst
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0, 0.02);
  std::vector<float> xf(static_cast<std::size_t>(M * T));
  for (int c = 0; c < M; ++c)
    for (int i = 0; i < T; ++i) {
      const double t = i / sfreq;
      double v = nd(rng);
      std::vector<Planted> all = planted;
      all.push_back(narrow);
      for (const auto& p : all)
        if (p.channel == c) v += p.A * std::exp(-(t - p.t0) * (t - p.t0) / (2 * p.s * p.s)) * std::cos(2 * kPi * p.f * (t - p.t0) + p.phi);
      xf[static_cast<std::size_t>(c * T + i)] = static_cast<float>(v);
    }
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] {
      const std::string cid = uuid7();
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
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
      const std::string tid = sub.manifold(std::move(tm));
      const std::string prod = sub.product({cid, tid});
      // the same samples in two chunk widths: one chunk, and chunks of 3000 samples
      for (const auto& [name, width] : {std::make_pair(std::string("rest"), std::int64_t(32768)), std::make_pair(std::string("rest_b"), std::int64_t(3000))}) {
        FieldSpec rec;
        rec.name = name; rec.path = tk::Subject::of(sub).node_path(prod, name); rec.kind = "recording"; rec.manifold_id = prod;
        rec.data = view(xf, {M, T}); rec.array.chunks = Shape{M, width}; rec.array.compress = false; rec.cols.set("unit", "T");
        sub.field(std::move(rec));
      }
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  nxrtest::make_time_grids(ds, sub, "tiles", Json{{"duration", "0.016 s"}});
  const g::Definition d = g::load("gabor_bursts");
  const Json small = Json{{"band", "10-30 Hz"}, {"sigma_min", 0.05}, {"sigma_max", 0.4}, {"per_octave", 2}, {"tile", "0.016 s"}};
  auto run = [&](Json set, bool replace) {
    for (auto& [k, v] : small.items()) if (!set.contains(k)) set[k] = v;
    g::RunOptions o;
    o.chunks = "all";
    o.replace = replace;
    return g::run(ds, sub, d, set, o);
  };
  const auto tsub = tk::Subject::of(sub);
  const nxr::io::testing::Probe probe(sub);
  auto out_of = [](const Json& r, const std::string& o) -> Json { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); };
  auto result = [](const std::vector<Json>& r, const std::string& rec) -> Json { for (const auto& x : r) if (x.value("anchor", std::string()) == rec) return x; return Json(); };
  auto field_id = [&](const std::string& rec) { return tsub.find({.cls = tk::Cls::field, .name = rec, .any_status = true})->id; };
  // a spans output's members: [start, stop) s and the channel; with the measurement over them read measure by measure
  auto spans_of = [&](const Json& stored) {
    std::vector<std::tuple<double, double, int>> out;
    const auto rows = probe.rows("selection", {{"id", V(stored["id"].get<std::string>())}});
    const Json params = Json::parse(text(rows.front(), "params_json"));
    const auto sp = probe.rows("selection_span", {{"selection_id", V(stored["id"].get<std::string>())}}, "ordinal");
    for (std::size_t k = 0; k < sp.size(); ++k)
      out.emplace_back(std::stod(text(sp[k], "start")), std::stod(text(sp[k], "stop")), params.contains("channels") && params["channels"].contains(std::to_string(k)) ? params["channels"][std::to_string(k)][0].get<int>() : -1);
    return std::make_pair(out, params);
  };
  auto bursts_of = [&](const Json& r, const std::string& rec) {
    std::vector<Burst> out;
    const Json bs = out_of(r, "bursts");
    if (bs.is_null()) return out;
    const auto [spans, params] = spans_of(bs);
    if (spans.empty()) return out;
    const std::string sid = bs["id"].get<std::string>(), fid = field_id(rec);
    std::map<std::string, MeasurementValues> mv;
    for (const char* m : {"atom_time", "atom_frequency", "atom_width", "atom_amplitude", "atom_phase", "atom_cycles", "atom_r2", "atom_width_peak", "onset", "offset"}) mv[m] = measurement_values(sub, sid, fid, std::nullopt, std::nullopt, m);
    const std::int64_t E = static_cast<std::int64_t>(spans.size());
    for (std::int64_t k = 0; k < E; ++k) {
      const int c = std::get<2>(spans[static_cast<std::size_t>(k)]);
      auto at = [&](const char* m) { return mv[m].values[static_cast<std::size_t>(c * E + k)]; };
      out.push_back({c, std::get<0>(spans[static_cast<std::size_t>(k)]), std::get<1>(spans[static_cast<std::size_t>(k)]), at("atom_time"), at("atom_frequency"), at("atom_width"), at("atom_amplitude"), at("atom_phase"),
                     at("atom_cycles"), at("atom_r2"), at("atom_width_peak"), at("onset"), at("offset")});
    }
    return out;
  };
  const double tile = 4 / sfreq;

  // 1. THE RUN on every recording (the two chunk widths), tangent edges
  const auto r1 = run(Json::object(), false);
  CHECK_EQ(static_cast<int>(r1.size()), 2);
  const Json ra = result(r1, "rest"), rb = result(r1, "rest_b");
  CHECK(!ra.is_null() && !rb.is_null());
  const std::vector<Burst> A = bursts_of(ra, "rest"), B = bursts_of(rb, "rest_b");
  std::printf("  %zu bursts found\n", A.size());
  // RECOVERY: each planted burst found once, on its channel, within its resolution
  std::vector<int> used(A.size(), 0);
  for (const auto& p : planted) {
    int hits = 0;
    for (std::size_t k = 0; k < A.size(); ++k) {
      const Burst& b = A[k];
      if (b.channel != p.channel || std::abs(b.t0 - p.t0) > tile || std::abs(b.f - p.f) > 1 / (2 * kPi * p.s)) continue;
      ++hits; used[k] = 1;
      const double dphi = std::arg(std::polar(1.0, b.phi - (p.phi + 2 * kPi * p.f * (b.t0 - p.t0))));
      std::printf("  planted ch %d t0 %.2f f %.1f s %.3f A %.2f: found t0 %.3f f %.2f s %.3f A %.3f dphi %.3f cycles %.2f R2 %.3f edges [%.3f, %.3f]\n", p.channel, p.t0, p.f, p.s, p.A, b.t0, b.f, b.s, b.A, dphi, b.cycles, b.r2, b.onset, b.offset);
      CHECK(std::abs(b.f - p.f) < 1 / (2 * kPi * p.s) / 4 && std::abs(b.s / p.s - 1) < 0.05 && std::abs(b.A / p.A - 1) < 0.05 && std::abs(dphi) < 0.2);
      CHECK(b.onset < b.t0 && b.t0 < b.offset && std::abs(b.start - b.onset) < 1e-6 && std::abs(b.stop - b.offset) < 1e-6);
    }
    CHECK_EQ(hits, 1);
  }
  // THE GATES: every burst a peak along width, at least 2 cycles, R² at least 0.7; any burst not planted is small; the narrow burst a candidate but no burst
  for (std::size_t k = 0; k < A.size(); ++k) {
    CHECK(A[k].width_peak == 1.0 && A[k].cycles >= 2.0 && A[k].r2 >= 0.7);
    if (!used[k]) { std::printf("  unplanted burst ch %d t0 %.3f f %.2f A %.3f\n", A[k].channel, A[k].t0, A[k].f, A[k].A); CHECK(A[k].A < 0.2); }
    CHECK(!(A[k].channel == narrow.channel && std::abs(A[k].t0 - narrow.t0) < 0.1));
  }
  {
    const auto [cands, cparams] = spans_of(out_of(ra, "candidates"));
    int near = 0;
    for (const auto& [a, b, c] : cands) near += c == narrow.channel && std::abs(0.5 * (a + b) - narrow.t0) < 0.1;
    CHECK(near >= 1);
    CHECK(cands.size() > A.size());
  }
  // CHUNK INVARIANCE: the same bursts, to the last bit, with chunks of 3000 samples
  CHECK_EQ(A.size(), B.size());
  for (std::size_t k = 0; k < std::min(A.size(), B.size()); ++k)
    CHECK(A[k].channel == B[k].channel && A[k].t0 == B[k].t0 && A[k].f == B[k].f && A[k].s == B[k].s && A[k].A == B[k].A && A[k].onset == B[k].onset && A[k].offset == B[k].offset);
  // THE GLOBAL WINDOWS: by time-frequency overlap, the 15 s bursts of channels 0 and 2 (12 and 12.5 Hz) share one window; at 30 s, 25 Hz (channel 0) and 12 Hz (channel 2) do not
  auto windows_near = [&](const Json& r, double t) {
    const auto [w, wp] = spans_of(out_of(r, "windows"));
    int n = 0;
    for (const auto& [a, b, c] : w) n += a <= t && t <= b && c == -1;
    return n;
  };
  CHECK_EQ(windows_near(ra, 15.02), 1);
  CHECK_EQ(windows_near(ra, 30.02), 2);
  {
    const auto [w, wp] = spans_of(out_of(ra, "windows"));
    bool two = false;
    for (std::size_t k = 0; k < w.size(); ++k) if (std::get<0>(w[k]) <= 15.02 && 15.02 <= std::get<1>(w[k])) two = wp["detection"]["union"]["members"][k].size() == 2;
    CHECK(two);
  }

  // 2. HALF-MAXIMUM EDGES: t0 ∓ FWHM/2 = t0 ∓ 1.1775 σ; the same atoms as the tangent run
  const auto r2 = run(Json{{"recording", "rest"}, {"edges", "half_max"}}, true);
  const std::vector<Burst> H = bursts_of(r2.front(), "rest");
  CHECK_EQ(H.size(), A.size());
  for (std::size_t k = 0; k < std::min(H.size(), A.size()); ++k) {
    CHECK(std::abs(H[k].onset - (H[k].t0 - 1.1775 * H[k].s)) < 1e-9 && std::abs(H[k].offset - (H[k].t0 + 1.1775 * H[k].s)) < 1e-9);
    CHECK(H[k].t0 == A[k].t0 && H[k].f == A[k].f);
    CHECK(H[k].onset > A[k].onset);                                                 // the tangent reaches the trough: wider than the atom's half maximum
  }
  // 3. THE WINDOWS BY TIME ALONE: the two 30 s bursts now share a window
  const auto r3 = run(Json{{"recording", "rest"}, {"overlap", "time"}}, true);
  CHECK_EQ(windows_near(r3.front(), 30.02), 1);
  // 4. A STRICTER FIT GATE keeps fewer bursts, each at least it
  const auto r4 = run(Json{{"recording", "rest"}, {"min_r2", 0.995}}, true);
  const std::vector<Burst> S = bursts_of(r4.front(), "rest");
  CHECK(S.size() < A.size());
  for (const auto& b : S) CHECK(b.r2 >= 0.995);
  // refusals: an unknown edges rule, a band past Nyquist
  bool refused = false;
  try { run(Json{{"recording", "rest"}, {"edges", "steepest"}}, true); } catch (const Error&) { refused = true; }
  CHECK(refused);
  refused = false;
  try { run(Json{{"recording", "rest"}, {"band", "10-124 Hz"}}, true); } catch (const Error&) { refused = true; }
  CHECK(refused);
  return nxrtest::finish("test_gabor_bursts");
}
