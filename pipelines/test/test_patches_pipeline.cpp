// the `patches` definition run through the runner on a synthetic two-hemisphere brain (never a real dataset): a map with planted smooth bumps and a dip of known size and height on a smooth background plus a
// little noise. At the scale whose Weyl band matches the bumps' size (`parts 32` a hemisphere) each bump is ONE patch holding the bump's centre and no other patch is found; a very much larger bump is not
// found at that scale; the dip is found with sign negative (and the bumps are not); the patches' area, mean, peak and centre equal an independent computation from the stored band field and the mesh; the plain
// tile means of the map (`measure`) and the band's are separate rows; a rerun is a hit; the refusals (a scale the ladder lacks, a map that is not a surface scalar, a map with holes, a constant map, an unknown
// sign). One map, one scale, one run each.
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/task/task.h"
#include "nxr/io/testing/probe.h"
#include "portable_random.h"
#include "nxr/select/events.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/select/dyadic.h"
#include "make_grids.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

template <class T> static std::vector<T> read_array(SubjectWriter& sub, const std::string& path) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}
static bool refused(const std::function<void()>& f, std::string* what = nullptr) {
  try { f(); } catch (const std::exception& e) { if (what) *what = e.what(); return true; }
  return false;
}
static Json out_of(const Json& r, const std::string& o) { for (const auto& x : r["stored"]) if (x["output"] == o) return x; return Json(); }

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_patches_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(4, false);
  const int nV = b.n, half = b.nHalf;
  auto P = [&](int i) { return Eigen::Vector3d(b.P[3 * i], b.P[3 * i + 1], b.P[3 * i + 2]); };
  // the planted vertices: by extremes of each hemisphere (far apart on the ellipsoid)
  auto extreme = [&](int h, int axis, int sgn) { int best = h * half; for (int i = h * half; i < (h + 1) * half; ++i) if (sgn * P(i)[axis] > sgn * P(best)[axis]) best = i; return best; };
  const int cA = extreme(0, 0, +1), cD = extreme(0, 0, -1), cB = extreme(1, 0, +1), cE = extreme(1, 2, -1);
  // THE MARGIN of the fixture (the patch count must not flip with a small change): the entry threshold is 3 SD of the tile means (median and MAD over the 64 tiles).
  // The bumps (height 2.5) put their tiles' band means at 0.34 and above, 1.8 times the threshold (0.186); the strongest tile that is not a patch (a texture tile or
  // a side lobe) is at 0.86 of it. The dip (height -1.75) is at 1.4 times its threshold (0.127) and the strongest other tile at 0.88 of it. The test prints both and
  // checks them (bumps: above 1.5 / below 0.9; dip: above 1.3 / below 0.9). The side lobes grow with the bump, so a taller bump alone does not widen the margin:
  // the texture (`tex`, the background spread that sets the threshold) is what puts the threshold between a lobe and a bump tile.
  const double sigma = 0.35;   // a bump about half a wavelength across at level 5 (the band's wavelength 1.65, tile 0.9 m²)
  struct Bump { int c; double h, s; };
  // the maps: smooth background + noise + bumps; `pet` has two bumps of the level's size and one very much larger, `pet_dip` one dip (hemisphere-masked: a bump is on its own hemisphere)
  const std::vector<Bump> bumps = {{cA, 2.5, sigma}, {cB, 2.5, sigma}, {cE, 2.0, 1.2}}, dips = {{cD, -1.75, sigma}};
  // the texture every real map has: 60 random plane waves of wavenumber 2 .. 5 /m (the band's own range), so the tiles' baseline has a spread of its own
  const double tex = 0.07;
  std::vector<std::array<double, 5>> waves;
  { std::mt19937 wr(5); nxr::testing::PortableUniform u(0, 1), ph(0, 2 * M_PI); nxr::testing::PortableNormal g(0, 1);
    for (int j = 0; j < 60; ++j) { Eigen::Vector3d d(g(wr), g(wr), g(wr)); d.normalize(); const double k = 2 + 3 * u(wr); waves.push_back({k * d.x(), k * d.y(), k * d.z(), ph(wr), tex / std::sqrt(30.0)}); } }
  auto plant = [&](const std::vector<Bump>& list, unsigned seed) {
    std::mt19937 rng(seed); nxr::testing::PortableNormal nd(0, 1);
    std::vector<float> v(static_cast<size_t>(nV));
    for (int i = 0; i < nV; ++i) {
      double x = 0.5 + (0.2 * b.Sph[3 * i] + 0.1 * b.Sph[3 * i + 2]) + 0.01 * nd(rng);   // a smooth background (the registered sphere's coordinates), a little noise
      for (const auto& w : waves) x += w[4] * std::cos(w[0] * P(i).x() + w[1] * P(i).y() + w[2] * P(i).z() + w[3]);
      for (const auto& bp : list) if ((i < half) == (bp.c < half)) x += bp.h * std::exp(-(P(i) - P(bp.c)).squaredNorm() / (2 * bp.s * bp.s));
      v[i] = static_cast<float>(x);
    }
    return v;
  };
  const std::vector<float> pet = plant(bumps, 11), pet_dip = plant(dips, 12);
  DatasetWriter ds = create_dataset(root, "demo");
  std::string mid;
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S");
    sub.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      mid = add_surface(sub, so, b.P, b.F).id;
      auto field = [&](const std::string& name, const std::vector<float>& v, const std::string& vt = "scalar", std::int64_t comps = 1) {
        FieldSpec f; f.name = name; f.path = "source/" + name; f.kind = "map"; f.manifold_id = mid; f.value_type = vt;
        f.data = comps == 1 ? view(v, {static_cast<std::int64_t>(nV)}) : view(v, {static_cast<std::int64_t>(nV), comps});
        f.cols.set("unit", "SUVR");
        return sub.field(std::move(f));
      };
      field("pet", pet);
      field("pet_dip", pet_dip);
      std::vector<float> holes = pet; holes[5] = std::nanf("");
      field("pet_holes", holes);
      field("flat", std::vector<float>(static_cast<size_t>(nV), 1.0f));
      field("vec", std::vector<float>(static_cast<size_t>(nV) * 3, 1.0f), "vector3", 3);
      // a Field on a manifold that is not a surface
      ManifoldSpec m;
      m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = 3;
      m.topology = {"none", 0, 1, std::nullopt, {}};
      std::vector<float> pos = {0.1f, 0, 0, 0, 0.1f, 0, 0, 0, 0.1f};
      GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
      const std::string cid = uuid7(); m.id = cid;
      m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {3, 3}), {}, std::nullopt, "manifold", cid); };
      sub.manifold(std::move(m));
      std::vector<float> sv = {1, 2, 3};
      FieldSpec f; f.name = "sensor_map"; f.path = "source/sensor_map"; f.kind = "map"; f.manifold_id = cid; f.value_type = "scalar"; f.data = view(sv, {3});
      sub.field(std::move(f));
    });
    // a second subject with no ladder
    SubjectWriter s2 = SubjectWriter::create(ds, "S2");
    s2.compose([&] {
      SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph;
      const std::string m2 = add_surface(s2, so, b.P, b.F).id;
      FieldSpec f; f.name = "pet"; f.path = "source/pet"; f.kind = "map"; f.manifold_id = m2; f.value_type = "scalar"; f.data = view(pet, {static_cast<std::int64_t>(nV)});
      s2.field(std::move(f));
    });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S");
  nxrtest::make_tiles(ds, sub);
  nxr::select::Mesh lm; lm.positions = b.P; lm.faces = b.F; lm.n = nV;
  const std::vector<double> va = nxr::select::vertexAreas(lm);
  auto field_id = [&](const std::string& n) { return nxr::io::task::Subject::of(sub).require({.cls = nxr::io::task::Cls::field, .name = n, .any_status = true}, n).id; };
  auto sel_row = [&](const std::string& n) { return nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::selection, .name = n, .any_status = true}); };
  const int level = 5;                                   // 32 tiles a hemisphere, ~0.9 m² each; the band's wavelength 1.65 m (half of it 0.8: about a bump)
  const std::string scale = "parts 32";
  auto patches = [&](const std::string& map, const std::string& sign, double enter, double exit, bool replace = false, const std::string& sc = "parts 32") {
    return nxrtest::run_definition(ds, sub, "patches", Json{{"map", map}, {"spatial_scale", sc}, {"sign", sign}, {"enter_sd", enter}, {"exit_sd", exit}}, replace);   // a band-pass has side lobes of the opposite sign around a bump: 3 SD keeps to the bumps
  };
  auto codes_of = [&](const Json& r) { return read_array<std::int32_t>(sub, *sel_row(out_of(r, "patches")["name"].get<std::string>())->array_path); };
  auto patch_values = [&](const Json& r, const std::string& output, const std::optional<std::string>& of_field, const std::string& m) {
    const Json x = out_of(r, output);
    return measurement_values(sub, x["over"].get<std::string>(), of_field, std::nullopt, std::nullopt, m).values;
  };
  const std::vector<std::int32_t> tile5 = read_array<std::int32_t>(sub, *sel_row("cortex tiles L5")->array_path);
  // the independent computation: the stored band field, the tiles' means, the baseline, regions by a breadth-first search over the tiles' mesh adjacency
  auto direct = [&](const Json& r, const std::string& sgn, double enter_sd, double exit_sd) {
    const auto bf = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = out_of(r, "band")["name"].get<std::string>(), .any_status = true});
    const std::vector<double> x = read_array<double>(sub, *bf->path);
    const int T = 2 << level;
    std::vector<double> sw(T, 0.0), sx(T, 0.0);
    for (int i = 0; i < nV; ++i) { sw[tile5[i]] += va[i]; sx[tile5[i]] += va[i] * x[i]; }
    Eigen::VectorXd m(T);
    for (int t = 0; t < T; ++t) m(t) = (sgn == "negative" ? -1 : 1) * sx[t] / sw[t];
    if (sgn == "magnitude") m = m.cwiseAbs();
    const double med = nxr::select::median(m), sd = nxr::select::mad(m);
    std::vector<std::set<int>> nb(T);
    for (int f = 0; f < b.nF; ++f) for (int k = 0; k < 3; ++k) { const int a = tile5[b.F[3 * f + k]], c = tile5[b.F[3 * f + (k + 1) % 3]]; if (a != c) { nb[a].insert(c); nb[c].insert(a); } }
    std::vector<int> lab(T, -1), comp(T, -1);
    int nc = 0;
    for (int t = 0; t < T; ++t) {
      if (comp[t] >= 0 || !(m(t) >= med + exit_sd * sd)) continue;
      std::queue<int> q; q.push(t); comp[t] = nc;
      bool entered = false;
      std::vector<int> mem;
      while (!q.empty()) { const int u = q.front(); q.pop(); mem.push_back(u); if (m(u) >= med + enter_sd * sd) entered = true; for (int v : nb[u]) if (comp[v] < 0 && m(v) >= med + exit_sd * sd) { comp[v] = nc; q.push(v); } }
      ++nc;
      if (entered) { const int ord = *std::max_element(lab.begin(), lab.end()) + 1; for (int u : mem) lab[u] = ord; }
    }
    struct Out { std::vector<int> vertex_code; std::vector<double> area, mean, peak; int n = 0; double enter = 0, weakest_in = 0, strongest_out = 0; };
    Out o; o.n = *std::max_element(lab.begin(), lab.end()) + 1; if (o.n < 0) o.n = 0;
    o.enter = med + enter_sd * sd;
    o.weakest_in = 1e300; o.strongest_out = -1e300;   // the tile means of the patches' entry tiles against the rest: the margin of the threshold, in both directions
    for (int t = 0; t < T; ++t) { if (lab[t] >= 0 && m(t) >= o.enter) o.weakest_in = std::min(o.weakest_in, m(t)); if (!(lab[t] >= 0 && m(t) >= o.enter)) o.strongest_out = std::max(o.strongest_out, m(t)); }
    o.vertex_code.assign(nV, -1);
    o.area.assign(o.n, 0.0); o.mean.assign(o.n, 0.0); o.peak.assign(o.n, -1e300);
    for (int i = 0; i < nV; ++i) {
      const int k = lab[tile5[i]];
      o.vertex_code[i] = k;
      if (k < 0) continue;
      o.area[k] += va[i]; o.mean[k] += va[i] * x[i];
      o.peak[k] = std::max(o.peak[k], sgn == "negative" ? -x[i] : sgn == "magnitude" ? std::abs(x[i]) : x[i]);
    }
    for (int k = 0; k < o.n; ++k) o.mean[k] /= o.area[k];
    return o;
  };

  // ── THE BUMPS: positive sign finds the two bumps, each one patch of its centre's tile, and nothing else ──
  const auto r1 = patches("pet", "positive", 3.0, 1.5);
  CHECK(r1.size() == 1 && r1[0]["hit"] == false);
  const std::vector<std::int32_t> codes1 = codes_of(r1[0]);
  int n1 = 0;
  for (const std::int32_t c : codes1) n1 = std::max(n1, c + 1);
  CHECK_EQ(n1, 2);                                                                               // the two bumps and nothing else
  CHECK(codes1[cA] >= 0 && codes1[cB] >= 0 && codes1[cA] != codes1[cB]);                         // each bump's centre is in a patch, not the same one
  CHECK(codes1[cE] == -1);                                                                       // the bump of another size is not found at this scale
  {
    std::set<int> tilesA, tilesB;
    for (int v = 0; v < nV; ++v) { if (codes1[v] == codes1[cA]) tilesA.insert(tile5[v]); if (codes1[v] == codes1[cB]) tilesB.insert(tile5[v]); }
    CHECK(tilesA == std::set<int>({tile5[cA]}) && tilesB == std::set<int>({tile5[cB]}));         // the expected tiles: the bump centre's own
    for (int v = 0; v < nV; ++v) if (codes1[v] >= 0) CHECK((v < half) == (codes1[v] == codes1[cA]));   // a patch never crosses the hemispheres
  }
  // the band's value at a bump's centre: the planted height times the band's response (about 0.66 for the height 2.5 at sigma 0.35, level 5)
  const std::string band_name = out_of(r1[0], "band")["name"].get<std::string>();
  const std::vector<double> band5 = read_array<double>(sub, *nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = band_name, .any_status = true})->path);
  std::printf("  the band at the bumps' centres: A %.3f, B %.3f, the large bump %.3f\n", band5[cA], band5[cB], band5[cE]);
  CHECK(band5[cA] > 0.45 && band5[cA] < 0.85 && band5[cB] > 0.45 && band5[cB] < 0.85 && std::abs(band5[cE]) < 0.12);
  // area, mean and peak against the direct computation, from the stored band field and the mesh; the label of every vertex
  {
    const auto d = direct(r1[0], "positive", 3.0, 1.5);
    CHECK_EQ(d.n, 2);
    std::printf("  threshold margin: entry tiles' weakest mean %.4f, entry threshold %.4f (ratio %.2f), strongest tile mean below it %.4f (ratio %.2f)\n", d.weakest_in, d.enter, d.weakest_in / d.enter, d.strongest_out, d.strongest_out / d.enter);
    CHECK(d.weakest_in > 1.5 * d.enter && d.strongest_out < 0.9 * d.enter);   // THE MARGIN (see the fixture's comment): a bump's tile at least 1.5 times the entry threshold, every other tile below 0.9 of it
    bool same = true;
    for (int v = 0; v < nV; ++v) same = same && d.vertex_code[v] == codes1[v];
    CHECK(same);
    const auto area = patch_values(r1[0], "patch_own", field_id(band_name), "area"), ntiles = patch_values(r1[0], "patch_own", field_id(band_name), "n_tiles"), centre = patch_values(r1[0], "patch_own", field_id(band_name), "centre_vertex");
    const auto mean = patch_values(r1[0], "patch_measures", field_id(band_name), "mean"), mx = patch_values(r1[0], "patch_measures", field_id(band_name), "max");
    CHECK(area.size() == 2 && mean.size() == 2 && mx.size() == 2);
    double worst = 0;
    for (int k = 0; k < d.n; ++k) worst = std::max({worst, std::abs(area[k] - d.area[k]) / d.area[k], std::abs(mean[k] - d.mean[k]) / std::abs(d.mean[k]), std::abs(mx[k] - d.peak[k]) / d.peak[k]});
    std::printf("  patches vs direct computation: worst relative error %.2e (areas %.4g, %.4g m²; peaks %.4g, %.4g)\n", worst, area[0], area[1], mx[0], mx[1]);
    CHECK(worst < 1e-9);
    CHECK_EQ(static_cast<int>(ntiles[0]), 1);
    for (int k = 0; k < d.n; ++k) {      // the centre: the patch's vertex nearest its area-weighted centroid
      Eigen::Vector3d c = Eigen::Vector3d::Zero(); double w = 0;
      for (int v = 0; v < nV; ++v) if (codes1[v] == k) { c += va[v] * P(v); w += va[v]; }
      c /= w;
      int best = -1; double bd = 1e300;
      for (int v = 0; v < nV; ++v) if (codes1[v] == k && (P(v) - c).squaredNorm() < bd) { bd = (P(v) - c).squaredNorm(); best = v; }
      CHECK_EQ(static_cast<int>(centre[k]), best);
    }
    // the statistics across the patches: count and sum
    const StoredStatistic sa = read_statistic(sub, out_of(r1[0], "area")["name"].get<std::string>());
    CHECK(sa.values.size() == 8 && sa.values[0] == 2.0 && std::abs(sa.values[1] - (d.area[0] + d.area[1])) < 1e-9 * (d.area[0] + d.area[1]));
    const StoredStatistic sp = read_statistic(sub, out_of(r1[0], "peak")["name"].get<std::string>());
    CHECK(sp.values.size() == 8 && sp.values[0] == 2.0);
    CHECK(read_statistic(sub, out_of(r1[0], "baseline")["name"].get<std::string>()).values.size() == 2);
  }
  // THE SEPARATE ROWS: a plain `measure` of the map over the same tiling is its own (of the map, not of its band), and neither overwrites the other
  {
    const std::string tiling = "cortex tiles L5";
    const auto tile_means = [&](const std::string& fld) { return measurement_values(sub, sel_row(tiling)->id, field_id(fld), std::nullopt, std::nullopt, "mean").values; };
    const auto before = tile_means(band_name);
    nxrtest::run_definition(ds, sub, "measure", Json{{"field", "pet"}, {"over", tiling}, {"weight", "barycentric"}, {"measures", Json::array({"mean"})}});
    const auto plain = tile_means("pet"), after = tile_means(band_name);
    CHECK(before == after && plain.size() == after.size());
    double diff = 0; for (std::size_t t = 0; t < plain.size(); ++t) diff = std::max(diff, std::abs(plain[t] - after[t]));
    CHECK(diff > 0.3);                                                                                // the map's own tile means (~0.5 + the bumps) are not the band's
  }
  CHECK(patches("pet", "positive", 3.0, 1.5)[0]["hit"] == true);                                      // a rerun is a hit
  {
    std::string why;
    CHECK(refused([&] { patches("pet", "positive", 8.0, 1.5); }, &why) && why.find("--replace") != std::string::npos);   // another choice is refused without --replace
  }

  // ── THE DIP: sign negative finds it, as one patch of its centre's tile; the bumps themselves are never marked by the negative sign; magnitude finds the bumps too (each sign of one map is its own result, made with --replace) ──
  {
    const auto r = patches("pet_dip", "negative", 3.0, 1.5);
    const std::vector<std::int32_t> c = codes_of(r[0]);
    int n = 0;
    for (const std::int32_t v : c) n = std::max(n, v + 1);
    CHECK_EQ(n, 1);
    CHECK(c[cD] >= 0);
    std::set<int> t; for (int v = 0; v < nV; ++v) if (c[v] == c[cD]) t.insert(tile5[v]);
    CHECK(t.count(tile5[cD]) == 1);                                                                   // the patch holds the dip centre's tile (a dip may straddle two)
    const auto d = direct(r[0], "negative", 3.0, 1.5);
    std::printf("  dip margin: entry tile's mean %.4f, threshold %.4f (ratio %.2f), strongest other tile %.4f (ratio %.2f)\n", d.weakest_in, d.enter, d.weakest_in / d.enter, d.strongest_out, d.strongest_out / d.enter);
    CHECK(d.weakest_in > 1.3 * d.enter && d.strongest_out < 0.9 * d.enter);   // the dip's margin: its tile at least 1.3 times the threshold, every other tile below 0.9 of it
    bool same = d.n == 1;
    for (int v = 0; v < nV; ++v) same = same && d.vertex_code[v] == c[v];
    CHECK(same);
    const auto mn = patch_values(r[0], "patch_measures", field_id(out_of(r[0], "band")["name"].get<std::string>()), "min");
    CHECK(mn.size() == 1 && std::abs(-mn[0] - d.peak[0]) < 1e-9 * d.peak[0] && mn[0] < -0.2);              // the peak of a dip is its minimum
    // the bumps' map, sign negative: their side lobes are negative, never the bumps themselves
    const std::vector<std::int32_t> cn = codes_of(patches("pet", "negative", 3.0, 1.5, true)[0]);
    CHECK(cn[cA] == -1 && cn[cB] == -1 && cn[cE] == -1);
    const std::vector<std::int32_t> cm = codes_of(patches("pet", "magnitude", 3.0, 1.5, true)[0]);
    CHECK(cm[cA] >= 0 && cm[cB] >= 0 && cm[cE] == -1);
    CHECK(codes_of(patches("pet", "positive", 3.0, 1.5, true)[0]) == codes1);                            // the first choice again (another sign is another result: --replace): the same two patches
  }

  // ── A BUMP OF ANOTHER SIZE: the large one (sigma 1.2, 3.4 times the bumps') is not found at this scale, and is at a coarser one ──
  {
    std::vector<std::string> found;
    for (const char* coarser : {"parts 8", "parts 4"}) {
      const auto r = patches("pet", "positive", 1.5, 1.0, false, coarser);
      if (codes_of(r[0])[cE] >= 0) found.push_back(coarser);
    }
    std::printf("  the large bump (sigma 1.2) is found at %zu coarser scale(s); not at parts 32\n", found.size());
    CHECK(!found.empty());
  }

  // ── REFUSALS, each with its reason ──
  {
    std::string why;
    CHECK(refused([&] { patches("vec", "positive", 3.0, 1.5); }, &why) && why.find("scalar") != std::string::npos);                    // not a scalar
    CHECK(refused([&] { patches("sensor_map", "positive", 3.0, 1.5); }, &why));                                                         // not on a surface
    CHECK(refused([&] { patches("pet_holes", "positive", 3.0, 1.5); }, &why) && why.find("no value") != std::string::npos);            // a hole
    CHECK(refused([&] { patches("flat", "positive", 3.0, 1.5); }, &why) && why.find("round-off") != std::string::npos);                // nothing varies
    CHECK(refused([&] { patches("nope", "positive", 3.0, 1.5); }, &why));                                                               // no such map
    CHECK(refused([&] { patches("pet", "up", 3.0, 1.5); }, &why) && why.find("positive") != std::string::npos);
    CHECK(refused([&] { patches("pet", "positive", 3.0, 5.0); }, &why));                                                                // exit above entry
    CHECK(refused([&] { patches("pet", "positive", 3.0, 1.5, false, "parts 4096"); }, &why));                                           // a scale the grid lacks
  }
  return 0;
}
