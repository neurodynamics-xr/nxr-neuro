// patches compute on a synthetic two-hemisphere brain: a map with planted smooth bumps and a dip of known size and height on a smooth background plus a little noise.
// At the level whose Weyl band matches the bumps' size each bump is ONE patch holding the bump's centre and no other patch is found; a very much larger bump is
// not found at that level (and is at a coarser one); the dip is found with sign negative (and the bumps are not); the patches' area, mean, peak and centre equal
// an independent computation from the stored band Field and the mesh; the plain tile means of the map (measure compute) and the band's are separate rows; a re-run
// is a cache hit at every stage, another choice is refused unless --replace; the refusal paths (no ladder, a level the ladder lacks, a Field that is not a surface
// scalar, a map with holes, a constant map). The exact transform and the Chebyshev series give the same band.
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
#include "portable_random.h"
#include "nxr/intervals.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/measure.h"
#include "nxr/pipelines/patches.h"
#include "nxr/pipelines/tiles.h"

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
  try { f(); } catch (const Error& e) { if (what) *what = e.what(); return true; } catch (const std::invalid_argument& e) { if (what) *what = e.what(); return true; }
  return false;
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_patches_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain b = brain(4, false);
  const int nV = b.n, half = b.nHalf;
  auto P = [&](int i) { return Eigen::Vector3d(b.P[3 * i], b.P[3 * i + 1], b.P[3 * i + 2]); };
  // the planted vertices: by extremes of each hemisphere (far apart on the ellipsoid)
  auto extreme = [&](int h, int axis, int sgn) { int best = h * half; for (int i = h * half; i < (h + 1) * half; ++i) if (sgn * P(i)[axis] > sgn * P(best)[axis]) best = i; return best; };
  const int cA = extreme(0, 0, +1), cD = extreme(0, 0, -1), cB = extreme(1, 0, +1), cE = extreme(1, 2, -1);
  const double sigma = 0.35;   // a bump about half a wavelength across at level 5 (the band's wavelength 1.65, tile 0.9 m²)
  struct Bump { int c; double h, s; };
  // the maps: smooth background + noise + bumps; `pet` has two bumps of the level's size and one very much larger, `pet_dip` one dip (hemisphere-masked: a bump is on its own hemisphere)
  const std::vector<Bump> bumps = {{cA, 1.0, sigma}, {cB, 1.0, sigma}, {cE, 2.0, 1.2}}, dips = {{cD, -1.5, sigma}};
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
  const TilesComputeResult tr = tiles_compute(ds, sub, TilesComputeOptions{});
  const int depth = tr.info["depth"].get<int>();
  std::printf("  ladder: %d levels, %d tiles at depth %d\n", tr.info["levels"].get<int>(), tr.info["tiles_at_depth"].get<int>(), depth);

  nxr::ladder::Mesh lm; lm.positions = b.P; lm.faces = b.F; lm.n = nV;
  const std::vector<double> va = nxr::ladder::vertexAreas(lm);
  auto field_id = [&](const std::string& n) { return text(*sub.db().one("SELECT id FROM field WHERE subject_id = ? AND name = ?", {V(sub.id()), V(n)}), "id"); };
  auto sel_row = [&](const std::string& n) { return sub.db().one("SELECT * FROM selection WHERE subject_id = ? AND name = ?", {V(sub.id()), V(n)}); };
  auto patch_codes = [&](const std::string& name) { return read_array<std::int32_t>(sub, text(*sel_row(name), "array_path")); };
  // the independent computation: the stored band Field, the tiles' means, the baseline, regions by a breadth-first search over the tiles' mesh adjacency
  auto direct = [&](int level, const std::string& map, const std::string& sgn, double enter_sd, double exit_sd) {
    const std::string band = map + "_band_L" + std::to_string(level);
    const auto bf = sub.db().one("SELECT path FROM field WHERE subject_id = ? AND name = ?", {V(sub.id()), V(band)});
    const std::vector<double> x = read_array<double>(sub, text(*bf, "path"));
    const std::vector<std::int32_t> tile = read_array<std::int32_t>(sub, text(*sel_row("cortex tiles L" + std::to_string(level)), "array_path"));
    const int T = 2 << level;
    std::vector<double> sw(T, 0.0), sx(T, 0.0);
    for (int i = 0; i < nV; ++i) { sw[tile[i]] += va[i]; sx[tile[i]] += va[i] * x[i]; }
    Eigen::VectorXd m(T);
    for (int t = 0; t < T; ++t) m(t) = (sgn == "negative" ? -1 : 1) * sx[t] / sw[t];
    if (sgn == "magnitude") m = m.cwiseAbs();
    const double med = nxr::intervals::median(m), sd = nxr::intervals::mad(m);
    std::vector<std::set<int>> nb(T);
    for (int f = 0; f < b.nF; ++f) for (int k = 0; k < 3; ++k) { const int a = tile[b.F[3 * f + k]], c = tile[b.F[3 * f + (k + 1) % 3]]; if (a != c) { nb[a].insert(c); nb[c].insert(a); } }
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
    // lab ordinals follow discovery order = ascending lowest tile
    struct Out { std::vector<int> vertex_code; std::vector<double> area, mean, peak; int n = 0; };
    Out o; o.n = *std::max_element(lab.begin(), lab.end()) + 1; if (o.n < 0) o.n = 0;
    o.vertex_code.assign(nV, -1);
    o.area.assign(o.n, 0.0); o.mean.assign(o.n, 0.0); o.peak.assign(o.n, -1e300);
    for (int i = 0; i < nV; ++i) {
      const int k = lab[tile[i]];
      o.vertex_code[i] = k;
      if (k < 0) continue;
      o.area[k] += va[i]; o.mean[k] += va[i] * x[i];
      o.peak[k] = std::max(o.peak[k], sgn == "negative" ? -x[i] : sgn == "magnitude" ? std::abs(x[i]) : x[i]);
    }
    for (int k = 0; k < o.n; ++k) o.mean[k] /= o.area[k];
    return o;
  };
  const int level = 5;                                   // 32 tiles a hemisphere, ~0.9 m² each; the band's wavelength 1.65 m (half of it 0.8: about a bump)
  PatchesComputeOptions po;
  po.map = "pet"; po.level = level; po.enter_sd = 3.0; po.exit_sd = 1.5;   // a band-pass has side lobes of the opposite sign around a bump (see README): 3 SD keeps to the bumps
  const std::string tiling = "cortex tiles L" + std::to_string(level);
  const std::vector<std::int32_t> tile5 = read_array<std::int32_t>(sub, text(*sel_row(tiling), "array_path"));
  const auto rows_of = [&](const std::string& sel, const std::optional<std::string>& fid, const std::string& m) {
    const MeasurementValues v = measurement_values(sub, text(*sel_row(sel), "id"), fid, std::nullopt, std::nullopt, m);
    return v.values;
  };

  // ── THE BUMPS: positive sign finds the two bumps, each one patch of its centre's tile, and nothing else ──
  const auto r1 = patches_compute(ds, sub, po);
  const std::string pname = "pet patches L5 positive";
  const std::vector<std::int32_t> codes1 = patch_codes(pname);
  std::printf("  level %d: band member %d (lambda %.3g .. %.3g), %s transform, order %d (series error %.2g); baseline %.3g ± %.3g; %lld patches\n", level, r1.info["band"]["member"].get<int>(), r1.info["band"]["lambda_min"].get<double>(),
              r1.info["band"]["lambda_max"].get<double>(), r1.info["band"]["transform"].get<std::string>().c_str(), r1.info["band"]["order"].get<int>(), r1.info["band"]["series_error"].get<double>(),
              r1.info["baseline"]["median_used"].get<double>(), r1.info["baseline"]["sd_used"].get<double>(), r1.info["patches"].get<long long>());
  CHECK(!r1.hit && r1.info["cache"] == "miss");
  // the default transform is `auto`; this surface has no stored modes and no medial-wall atlas to glue it, so the series is used, ANNOUNCED
  CHECK(r1.info["band"]["requested"] == "auto" && r1.info["band"]["transform"] == "chebyshev" && r1.info["warnings"].size() == 1);
  CHECK(r1.info["warnings"][0].get<std::string>().find("CHEBYSHEV fallback") != std::string::npos && r1.info["warnings"][0].get<std::string>().find("level 5") != std::string::npos);
  CHECK_EQ(r1.info["patches"].get<int>(), 2);                                                   // the two bumps and nothing else
  CHECK(codes1[cA] >= 0 && codes1[cB] >= 0 && codes1[cA] != codes1[cB]);                         // each bump's centre is in a patch, not the same one
  CHECK(codes1[cE] == -1);                                                                       // the bump of another size is not found at this level
  {
    std::set<int> tilesA, tilesB;
    for (int v = 0; v < nV; ++v) { if (codes1[v] == codes1[cA]) tilesA.insert(tile5[v]); if (codes1[v] == codes1[cB]) tilesB.insert(tile5[v]); }
    CHECK(tilesA == std::set<int>({tile5[cA]}) && tilesB == std::set<int>({tile5[cB]}));         // the expected tiles: the bump centre's own
    for (int v = 0; v < nV; ++v) if (codes1[v] >= 0) CHECK((v < half) == (codes1[v] == codes1[cA]));   // a patch never crosses the hemispheres
  }
  // the band's value at a bump's centre: the planted height times the band's response (σ² ∫ g e^(−k²σ²/2) k dk ≈ 0.27 for σ 0.35 at level 5)
  const std::vector<double> band5 = read_array<double>(sub, text(*sub.db().one("SELECT path FROM field WHERE subject_id = ? AND name = 'pet_band_L5'", {V(sub.id())}), "path"));
  std::printf("  the band at the bumps' centres: A %.3f, B %.3f, the large bump %.3f\n", band5[cA], band5[cB], band5[cE]);
  CHECK(band5[cA] > 0.18 && band5[cA] < 0.35 && band5[cB] > 0.18 && band5[cB] < 0.35 && std::abs(band5[cE]) < 0.05);

  // area, mean and peak against the direct computation, from the stored band Field and the mesh; the label of every vertex
  {
    const auto d = direct(level, "pet", "positive", po.enter_sd, po.exit_sd);
    CHECK_EQ(d.n, 2);
    bool same = true;
    for (int v = 0; v < nV; ++v) same = same && d.vertex_code[v] == codes1[v];
    CHECK(same);
    const auto area = rows_of(pname, std::nullopt, "area"), ntiles = rows_of(pname, std::nullopt, "n_tiles"), centre = rows_of(pname, std::nullopt, "centre_vertex");
    const auto mean = rows_of(pname, field_id("pet_band_L5"), "mean"), mx = rows_of(pname, field_id("pet_band_L5"), "max");
    double worst = 0;
    for (int k = 0; k < d.n; ++k) worst = std::max({worst, std::abs(area[k] - d.area[k]) / d.area[k], std::abs(mean[k] - d.mean[k]) / std::abs(d.mean[k]), std::abs(mx[k] - d.peak[k]) / d.peak[k]});
    std::printf("  patches vs direct computation: worst relative error %.2e (areas %.4g, %.4g m²; peaks %.4g, %.4g)\n", worst, area[0], area[1], mx[0], mx[1]);
    CHECK(worst < 1e-9);
    CHECK_EQ(static_cast<int>(ntiles[0]), 1);
    // the centre: the patch's vertex nearest its area-weighted centroid
    for (int k = 0; k < d.n; ++k) {
      Eigen::Vector3d c = Eigen::Vector3d::Zero(); double w = 0;
      for (int v = 0; v < nV; ++v) if (codes1[v] == k) { c += va[v] * P(v); w += va[v]; }
      c /= w;
      int best = -1; double bd = 1e300;
      for (int v = 0; v < nV; ++v) if (codes1[v] == k && (P(v) - c).squaredNorm() < bd) { bd = (P(v) - c).squaredNorm(); best = v; }
      CHECK_EQ(static_cast<int>(centre[k]), best);
    }
    // the stored record and the statistics across the patches
    const Json pj = Json::parse(text(*sel_row(pname), "params_json"));
    CHECK(pj["pipeline"] == "patches" && pj["level"] == level && pj["sign"] == "positive" && pj["band"]["member"] == level && pj["baseline"]["statistic"] == "pet_band_L5_baseline" && pj.contains("cache") && pj["merge"] == "none");
    CHECK(text(*sel_row(pname), "type") == "set" && integer(*sel_row(pname), "n_members") == 2);
    const StoredStatistic sa = read_statistic(sub, "pet_patches_L5_positive_area");
    CHECK(sa.values.size() == 8 && sa.values[0] == 2.0 && std::abs(sa.values[1] - (d.area[0] + d.area[1])) < 1e-9 * (d.area[0] + d.area[1]));   // count, sum
    const StoredStatistic sp = read_statistic(sub, "pet_patches_L5_positive_peak");
    CHECK(sp.values.size() == 8 && sp.values[0] == 2.0);
    const StoredStatistic base = read_statistic(sub, "pet_band_L5_baseline");
    CHECK(base.values.size() == 2 && std::abs(base.values[0] - r1.info["baseline"]["median"].get<double>()) == 0);
  }
  // THE SEPARATE ROWS: a plain `measure compute` of the map over the same tiling is its own (of the map, not of its band), and neither overwrites the other
  {
    const auto before = rows_of(tiling, field_id("pet_band_L5"), "mean");
    MeasureComputeOptions mo; mo.labelling = tiling; mo.fields = "^pet$"; mo.weight = "barycentric"; mo.measures = {"mean"};
    measure_compute(ds, sub, mo);
    const auto plain = rows_of(tiling, field_id("pet"), "mean"), after = rows_of(tiling, field_id("pet_band_L5"), "mean");
    CHECK(before == after && plain.size() == after.size());
    double diff = 0; for (std::size_t t = 0; t < plain.size(); ++t) diff = std::max(diff, std::abs(plain[t] - after[t]));
    CHECK(diff > 0.3);                                                                                // the map's own tile means (~0.5 + the bumps) are not the band's
    const auto cnt = sub.db().one("SELECT COUNT(DISTINCT of_field_id) AS n FROM selection_measurement WHERE selection_id = ? AND measure = 'mean'", {V(text(*sel_row(tiling), "id"))});
    CHECK(cnt && integer(*cnt, "n") == 2);
  }
  // a RE-RUN is a hit at every stage; another choice is refused unless --replace
  {
    const auto nrows = [&] { return integer(*sub.db().one("SELECT COUNT(*) AS n FROM selection_measurement"), "n"); };
    const std::int64_t n0 = nrows();
    const auto r2 = patches_compute(ds, sub, po);
    CHECK(r2.hit && r2.info["cache"] == "hit" && r2.info["patches"].get<int>() == 2 && nrows() == n0);
    for (const auto& st : r2.info["statistics"]) CHECK(st["cache"] == "hit");
    PatchesComputeOptions other = po; other.enter_sd = 6.0;
    std::string why;
    CHECK(refused([&] { patches_compute(ds, sub, other); }, &why) && why.find("--replace") != std::string::npos);
    other.replace = true;
    const auto r3 = patches_compute(ds, sub, other);
    CHECK(!r3.hit && r3.info["patches"].get<int>() == 0);          // at 6 SD nothing reaches the entry level; the replaced patches' statistics are gone with them
    CHECK(!sub.db().one("SELECT id FROM statistic WHERE subject_id = ? AND name = 'pet_patches_L5_positive_area'", {V(sub.id())}).has_value());
    std::printf("  enter at 6 SD, --replace: %d patch(es)\n", r3.info["patches"].get<int>());
    PatchesComputeOptions back = po; back.replace = true;
    const auto r4 = patches_compute(ds, sub, back);
    CHECK(!r4.hit && r4.info["patches"].get<int>() == 2);                                              // back to the first choice: the same two patches
    CHECK(patch_codes(pname) == codes1);
  }

  // ── THE DIP: sign negative finds it, as one patch of its centre's tile; the bumps' negative side lobes are below the threshold; magnitude finds the bumps too ──
  {
    PatchesComputeOptions n = po; n.map = "pet_dip"; n.sign = "negative";
    const auto r = patches_compute(ds, sub, n);
    const std::vector<std::int32_t> c = patch_codes("pet_dip patches L5 negative");
    std::printf("  dip, sign negative: %d patch(es); baseline %.3g ± %.3g, enter at %.3g\n", r.info["patches"].get<int>(), r.info["baseline"]["median_used"].get<double>(), r.info["baseline"]["sd_used"].get<double>(), r.info["thresholds"]["enter"].get<double>());
    CHECK_EQ(r.info["patches"].get<int>(), 1);
    CHECK(c[cD] >= 0);
    {
      std::set<int> t; for (int v = 0; v < nV; ++v) if (c[v] == c[cD]) t.insert(tile5[v]);
      CHECK(t.count(tile5[cD]) == 1);                                                                   // the patch holds the dip centre's tile (a dip may straddle two)
    }
    const auto d = direct(level, "pet_dip", "negative", n.enter_sd, n.exit_sd);
    bool same = d.n == 1;
    for (int v = 0; v < nV; ++v) same = same && d.vertex_code[v] == c[v];
    CHECK(same);
    const auto mn = rows_of("pet_dip patches L5 negative", field_id("pet_dip_band_L5"), "min");
    CHECK(mn.size() == 1 && std::abs(-mn[0] - d.peak[0]) < 1e-9 * d.peak[0] && mn[0] < -0.2);              // the peak of a dip is its minimum
    const StoredStatistic sp = read_statistic(sub, "pet_dip_patches_L5_negative_peak");
    CHECK(sp.values[0] == 1.0 && std::abs(sp.values[2] - mn[0]) < 1e-12);                                  // count 1, the median of one value
    // the bumps' map, sign negative: their side lobes are negative but weaker than the threshold: no patch
    PatchesComputeOptions nb = po; nb.sign = "negative";
    CHECK_EQ(patches_compute(ds, sub, nb).info["patches"].get<int>(), 0);
    PatchesComputeOptions m = po; m.sign = "magnitude";
    const auto rm = patches_compute(ds, sub, m);
    const std::vector<std::int32_t> cm = patch_codes("pet patches L5 magnitude");
    std::printf("  bumps, sign magnitude: %d patch(es)\n", rm.info["patches"].get<int>());
    CHECK(cm[cA] >= 0 && cm[cB] >= 0 && cm[cE] == -1);
    CHECK(!rm.info["baseline"]["note"].is_null());
    // the positive patches are still there, untouched by the other signs
    CHECK(patch_codes(pname) == codes1);
  }

  // ── A BUMP OF ANOTHER SIZE: the large one (σ 1.2, 3.4 times the bumps') is not found at level 5, and is at a coarser level ──
  {
    std::vector<int> found;
    for (int lv = 1; lv <= 4; ++lv) {
      PatchesComputeOptions c = po; c.level = lv; c.enter_sd = 1.5; c.exit_sd = 1.0;
      const auto r = patches_compute(ds, sub, c);
      if (patch_codes("pet patches L" + std::to_string(lv) + " positive")[cE] >= 0) found.push_back(lv);
    }
    std::printf("  the large bump (sigma 1.2) is found at level(s)");
    for (int lv : found) std::printf(" %d", lv);
    std::printf("; not at level 5\n");
    CHECK(!found.empty());
  }

  // ── the transforms: the series follows the exact modes (a smaller mesh, the dense solve is capped), a series too short is refused, not used ──
  {
    std::string why;
    PatchesComputeOptions e = po; e.transform = "exact";
    CHECK(refused([&] { patches_compute(ds, sub, e); }, &why) && why.find("4096") != std::string::npos);   // 5124 vertices
    PatchesComputeOptions lo = po; lo.order = 20;
    CHECK(refused([&] { patches_compute(ds, sub, lo); }, &why) && why.find("--order") != std::string::npos);
    const Brain s = brain(3, false);
    const fs::path root2 = root.string() + "_small";
    fs::remove_all(root2);
    DatasetWriter d2 = create_dataset(root2, "small");
    std::vector<float> m(static_cast<size_t>(s.n));
    for (int i = 0; i < s.n; ++i) m[i] = static_cast<float>(std::sin(3 * s.Sph[3 * i]) * std::cos(2 * s.Sph[3 * i + 2]) + 0.3 * s.Sph[3 * i + 1]);
    { SubjectWriter w = SubjectWriter::create(d2, "S");
      w.compose([&] {
        SurfaceOptions so; so.name = "cortex"; so.sphere = s.Sph;
        const std::string id = add_surface(w, so, s.P, s.F).id;
        FieldSpec f; f.name = "m"; f.path = "source/m"; f.kind = "map"; f.manifold_id = id; f.value_type = "scalar"; f.data = view(m, {static_cast<std::int64_t>(s.n)});
        w.field(std::move(f));
      }); }
    SubjectWriter w = SubjectWriter::open(d2, "S");
    tiles_compute(d2, w, TilesComputeOptions{});
    PatchesComputeOptions a; a.map = "m"; a.level = 4; a.transform = "exact"; a.enter_sd = 1.0; a.exit_sd = 0.5;
    patches_compute(d2, w, a);
    auto band = [&] { const auto r = w.db().one("SELECT path FROM field WHERE name = 'm_band_L4'"); return read_array<double>(w, text(*r, "path")); };
    const std::vector<double> ex = band();
    a.transform = "chebyshev"; a.replace = true;
    const auto rc = patches_compute(d2, w, a);
    const std::vector<double> ch = band();
    double dmax = 0, vmax = 0;
    for (std::size_t i = 0; i < ex.size(); ++i) { dmax = std::max(dmax, std::abs(ex[i] - ch[i])); vmax = std::max(vmax, std::abs(ex[i])); }
    std::printf("  exact vs Chebyshev (order %d, series error %.2g): max |difference| %.2e of a band of size %.3f\n", rc.info["band"]["order"].get<int>(), rc.info["band"]["series_error"].get<double>(), dmax, vmax);
    CHECK(vmax > 0.05 && dmax < 0.03 * vmax);
    fs::remove_all(root2);
  }

  // ── REFUSALS, each with its reason ──
  {
    std::string why;
    PatchesComputeOptions q = po;
    SubjectWriter s2 = SubjectWriter::open(ds, "S2");
    CHECK(refused([&] { patches_compute(ds, s2, q); }, &why) && why.find("tiles compute") != std::string::npos);          // no ladder
    q.level = 99;
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("no level 99") != std::string::npos);            // a level the ladder lacks
    q = po; q.map = "vec";
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("scalar") != std::string::npos);                // not a scalar
    q.map = "sensor_map";
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("not a surface") != std::string::npos);         // not on a surface
    q.map = "pet_holes";
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("no value") != std::string::npos);              // a hole
    q.map = "flat";
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("round-off") != std::string::npos);             // nothing varies
    q.map = "nope";
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("no stored Field") != std::string::npos);
    q = po; q.sign = "up";
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why) && why.find("positive") != std::string::npos);
    q = po; q.exit_sd = 5.0;
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why));
    q = po; q.level = -1;
    CHECK(refused([&] { patches_compute(ds, sub, q); }, &why));
  }
  fs::remove_all(root);
  return nxrtest::finish("test_patches_pipeline");
}
