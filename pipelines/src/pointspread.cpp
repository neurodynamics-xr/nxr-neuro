// nxr-pipelines — the instrument's blur (pointspread.h).
#include "nxr/pipelines/pointspread.h"

#include <algorithm>
#include <cmath>

#include "nxr/cores.h"
#include "nxr/io/statistic.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/scalespace.h"

namespace nxr::pipelines {

using nxr::io::Error;
using nxr::io::Json;
using nxr::io::Row;
using nxr::io::text;
using nxr::io::V;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(nxr::io::code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(nxr::io::code::no_such_row, m); }
template <class T> std::vector<T> read_all(nxr::io::Store& store, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  nxr::io::ArrayReader ar = nxr::io::ArrayReader::open(store, path);
  if (shape) *shape = ar.shape();
  const auto raw = ar.read();
  std::vector<T> v(raw.size() / sizeof(T));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
  return v;
}
}  // namespace

PointSpreadResult pointspread_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const PointSpreadOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (opt.recording.empty()) invalid("pointspread compute: name the recording (--recording): the statistic is OF it");
  if (opt.levels.empty() || std::find(opt.levels.begin(), opt.levels.end(), opt.reference_level) == opt.levels.end()) invalid("pointspread compute: the reference level must be one of the levels");
  if (!(opt.fraction > 0 && opt.fraction < 1)) invalid("pointspread compute: --fraction is in (0, 1)");
  const auto rec = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(opt.recording)});
  if (!rec) missing(sub.name() + ": no recording (Field) named '" + opt.recording + "'");
  // the head model and the kernel
  Row fwd, ker;
  if (opt.forward) { const auto r = sub.db().one("SELECT * FROM operator WHERE subject_id = ? AND kind = 'forward' AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.forward)}); if (!r) missing(sub.name() + ": no head model named '" + *opt.forward + "'"); fwd = *r; }
  else { const auto rs = sub.db().all("SELECT * FROM operator WHERE subject_id = ? AND kind = 'forward' AND status = 'complete' AND path IS NOT NULL", {V(sub.id())}); if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " head models; name one with --forward"); fwd = rs[0]; }
  if (opt.kernel) { const auto r = sub.db().one("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.kernel)}); if (!r) missing(sub.name() + ": no inverse kernel named '" + *opt.kernel + "'"); ker = *r; }
  else { const auto rs = sub.db().all("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND status = 'complete' AND to_manifold_id = ?", {V(sub.id()), V(text(fwd, "from_manifold_id"))}); if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " inverse kernels of " + text(fwd, "name") + "; name one with --kernel"); ker = rs[0]; }
  if (text(ker, "to_manifold_id") != text(fwd, "from_manifold_id")) invalid("pointspread compute: the kernel " + text(ker, "name") + " reads other sources than the head model " + text(fwd, "name"));
  const std::string kname = text(ker, "name"), name = opt.name ? *opt.name : "pointspread." + kname;
  // the gain on the kernel's channels, and the kernel
  const auto csel = sub.db().read("selection", text(ker, "from_selection_id"));
  if (!csel || nxr::io::is_null(*csel, "array_path")) invalid(kname + ": its channels (from_selection) are not stored");
  const std::vector<std::int32_t> chans = read_all<std::int32_t>(sub.store(), text(*csel, "array_path"));
  const std::int64_t M = static_cast<std::int64_t>(chans.size());
  std::vector<std::int64_t> gshape, kshape;
  const std::vector<float> gf = read_all<float>(sub.store(), text(fwd, "path"), &gshape), kf = read_all<float>(sub.store(), text(ker, "path"), &kshape);
  if (gshape.size() != 2 || gshape[1] % 3 != 0) invalid(text(fwd, "name") + ": the gain is not [channels x 3N]");
  const std::int64_t Mall = gshape[0], N = gshape[1] / 3;
  if (kshape.size() != 2 || kshape[0] != 3 * N || kshape[1] != M) invalid(kname + ": its array is [" + std::to_string(kshape.empty() ? 0 : kshape[0]) + " x " + std::to_string(kshape.size() > 1 ? kshape[1] : 0) + "], the head model has 3N = " + std::to_string(3 * N) + " source rows and the kernel " + std::to_string(M) + " channels");
  Eigen::MatrixXd G(M, 3 * N), K(3 * N, M);
  for (std::int64_t i = 0; i < M; ++i) {
    if (chans[static_cast<std::size_t>(i)] < 0 || chans[static_cast<std::size_t>(i)] >= Mall) invalid(kname + ": a channel index is outside the gain's channels");
    for (std::int64_t j = 0; j < 3 * N; ++j) G(i, j) = gf[static_cast<std::size_t>(chans[static_cast<std::size_t>(i)] * 3 * N + j)];
  }
  for (std::int64_t r = 0; r < 3 * N; ++r) for (std::int64_t c = 0; c < M; ++c) K(r, c) = kf[static_cast<std::size_t>(r * M + c)];
  // the surface, its glued domain, the mesh the patch is cut on
  const auto man = sub.db().read("manifold", text(fwd, "from_manifold_id"));
  const SurfaceInput S = read_surface(sub, text(*man, "name"), true);
  if (S.n_vertices() != N) invalid("pointspread compute: the head model's " + std::to_string(N) + " sources are not the " + std::to_string(S.n_vertices()) + " vertices of " + S.name);
  const internal::GluedDomain dom = internal::glued_domain(sub, S);
  if (!dom.found) invalid("pointspread compute: " + dom.why);
  const std::int64_t Vg = dom.surface.n_vertices();
  std::vector<std::pair<std::int32_t, std::int32_t>> pairs;
  for (std::int64_t f = 0; f < dom.surface.n_faces(); ++f)
    for (int k = 0; k < 3; ++k) { std::int32_t a = dom.surface.faces[static_cast<std::size_t>(3 * f + k)], b = dom.surface.faces[static_cast<std::size_t>(3 * f + (k + 1) % 3)]; if (a > b) std::swap(a, b); pairs.emplace_back(a, b); }
  std::sort(pairs.begin(), pairs.end()); pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  nxr::ladder::Mesh mesh; mesh.n = static_cast<int>(Vg); mesh.positions = dom.surface.vertices; mesh.faces = dom.surface.faces;
  const std::vector<double> va = nxr::ladder::vertexAreas(mesh);
  const Eigen::VectorXd area = Eigen::Map<const Eigen::VectorXd>(va.data(), static_cast<Eigen::Index>(va.size()));
  const Eigen::MatrixXd pos = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(dom.surface.vertices.data(), Vg, 3);
  // the sample
  std::vector<int> sample = opt.list;
  if (sample.empty()) { const int n = std::max(1, std::min<int>(opt.vertices, static_cast<int>(N))); for (int i = 0; i < n; ++i) sample.push_back(static_cast<int>(static_cast<std::int64_t>(i) * N / n)); }
  for (const int v : sample) if (v < 0 || v >= N) invalid("pointspread compute: vertex " + std::to_string(v) + " is not one of the " + std::to_string(N) + " vertices");
  // the levels' operators, their sizes and the atoms' squared norms (as `scale`)
  const std::int64_t NL = static_cast<std::int64_t>(opt.levels.size());
  std::vector<SpatialBand> bands;
  std::vector<double> size_cm;
  Eigen::VectorXd norm2(Vg * NL);
  for (std::int64_t li = 0; li < NL; ++li) {
    const SurfaceLevel L = read_surface_level(sub, S, opt.levels[static_cast<std::size_t>(li)]);
    SpatialBandOptions bo; bo.transform = opt.transform; bo.modes = opt.modes; bo.order = opt.order; bo.tolerance = opt.tolerance; bo.spatial_shape = opt.spatial_shape; bo.glued = true;
    bands.push_back(make_spatial_band(ds, sub, S, L, bo, [&](const Json& j) { say(j); }));
    const auto& m = L.bank.members[static_cast<std::size_t>(L.l)];
    const double lc = m.lo > 0 ? std::sqrt(m.lo * m.hi) : m.hi / std::sqrt(2.0);
    size_cm.push_back(100.0 * 2.0 * M_PI / std::sqrt(lc));
    std::vector<double> n2(static_cast<std::size_t>(Vg), 0.0);
    for (std::int64_t k0 = 0; k0 < N; k0 += 256) {
      const std::int64_t w = std::min<std::int64_t>(256, N - k0);
      Eigen::MatrixXd I = Eigen::MatrixXd::Zero(N, w);
      for (std::int64_t i = 0; i < w; ++i) I(k0 + i, i) = 1.0;
      const Eigen::MatrixXd A = bands.back().apply_glued(I);
      for (std::int64_t v = 0; v < Vg; ++v) n2[static_cast<std::size_t>(v)] += A.row(v).squaredNorm();
    }
    for (std::int64_t v = 0; v < Vg; ++v) { if (!(n2[static_cast<std::size_t>(v)] > 0)) invalid("pointspread compute: the atom of glued vertex " + std::to_string(v) + " has norm 0 at level " + std::to_string(opt.levels[static_cast<std::size_t>(li)])); norm2(v * NL + li) = n2[static_cast<std::size_t>(v)]; }
  }
  const std::int64_t ref = std::find(opt.levels.begin(), opt.levels.end(), opt.reference_level) - opt.levels.begin();
  // THE CACHE
  CacheKey key;
  key.pipeline = "pointspread compute";
  key.params = Json{{"levels", opt.levels}, {"reference_level", opt.reference_level}, {"spatial_shape", opt.spatial_shape}, {"fraction", opt.fraction}, {"sample", sample}, {"name", name}};
  key.input = Json{{"recording", text(*rec, "id")}, {"forward", text(fwd, "id")}, {"forward_modified", nxr::io::opt_text(fwd, "modified_utc").value_or("")}, {"kernel", text(ker, "id")}, {"kernel_modified", nxr::io::opt_text(ker, "modified_utc").value_or("")}};
  key.compute = NXR_COMPUTE_VERSION;
  const std::string path = "statistics/" + nxr::io::safe_name(name);
  PointSpreadResult res;
  if (!opt.replace) if (const auto hit = find_cached(sub.db(), sub.id(), "statistic", key, path)) { res.hit = true; res.info = Json{{"name", name}, {"cache", "hit"}}; return res; }
  // THE SOURCES: x, y, z at each sampled vertex, through the gain and the kernel, then the chain
  const int NS = static_cast<int>(sample.size());
  std::vector<double> out(static_cast<std::size_t>(NS) * 3 * 6, std::nan(""));
  for (int si = 0; si < NS; ++si) {
    for (int o = 0; o < 3; ++o) {
      const std::int64_t col = 3 * static_cast<std::int64_t>(sample[static_cast<std::size_t>(si)]) + o;
      const Eigen::VectorXd shat = K * G.col(col);                                       // the resolution matrix's column of the unit current
      Eigen::MatrixXd P = Eigen::MatrixXd::Zero(Vg * NL, 1);
      for (std::int64_t li = 0; li < NL; ++li)
        for (int c = 0; c < 3; ++c) {
          Eigen::MatrixXd comp(N, 1);
          for (std::int64_t v = 0; v < N; ++v) comp(v, 0) = shat[3 * v + c];
          const Eigen::MatrixXd h = bands[static_cast<std::size_t>(li)].apply_glued(comp);
          for (std::int64_t v = 0; v < Vg; ++v) P(v * NL + li, 0) += h(v, 0) * h(v, 0);
        }
      // the peak of the reference level's power, the patch around it, its centre's characteristic level
      Eigen::MatrixXd Pr(Vg, 1);
      std::int64_t peak = 0;
      for (std::int64_t v = 0; v < Vg; ++v) { Pr(v, 0) = P(v * NL + ref, 0); if (Pr(v, 0) > Pr(peak, 0)) peak = v; }
      const double mx = Pr(peak, 0);
      if (!(mx > 0)) continue;
      const nxr::cores::Cores C = nxr::cores::corePatches(Pr, pairs, area, pos, opt.fraction * mx, 0);
      std::int64_t pi = -1;
      for (std::int64_t i = 0; i < C.size(); ++i) { const auto cv = C.centre_vertex[static_cast<std::size_t>(i)]; if (pi < 0 || C.peak[static_cast<std::size_t>(i)] > C.peak[static_cast<std::size_t>(pi)]) { (void)cv; pi = i; } }
      if (pi < 0) continue;
      const std::int32_t cv = C.centre_vertex[static_cast<std::size_t>(pi)];
      Eigen::MatrixXd Pc(NL, 1);
      Eigen::VectorXd n2c(NL);
      for (std::int64_t li = 0; li < NL; ++li) { Pc(li, 0) = P(static_cast<std::int64_t>(cv) * NL + li, 0); n2c[li] = norm2(static_cast<std::int64_t>(cv) * NL + li); }
      const nxr::scalespace::LevelChoice ch = nxr::scalespace::selectLevel(Pc, n2c, 1);
      // the source's own place: the glued vertex that is the source's (the kept vertices lead, in `support` order)
      std::int64_t src = -1;
      for (std::size_t k = 0; k < dom.support.size(); ++k) if (dom.support[k] == sample[static_cast<std::size_t>(si)]) { src = static_cast<std::int64_t>(k); break; }
      double* r = &out[static_cast<std::size_t>((si * 3 + o) * 6)];
      r[0] = src >= 0 ? 100.0 * (pos.row(peak) - pos.row(src)).norm() : std::nan("");     // the peak's distance from the source, cm
      r[1] = 100.0 * 2.0 * std::sqrt(C.area[static_cast<std::size_t>(pi)] / M_PI);       // the patch's equal-area diameter, cm
      r[2] = 1e4 * C.area[static_cast<std::size_t>(pi)];                                 // its area, cm2
      r[3] = opt.levels[static_cast<std::size_t>(ch.index(0, 0))];
      r[4] = size_cm[static_cast<std::size_t>(ch.index(0, 0))];
      r[5] = mx;
    }
    say(Json{{"stage", "pointspread"}, {"vertex", sample[static_cast<std::size_t>(si)]}, {"done", si + 1}, {"of", NS}});
  }
  bool replaced = false;
  clear_stored(ds, sub, {{"statistic", path}}, opt.replace, name, replaced);
  Json vals = Json::array(), ors = Json::array({Json{{"name", "x"}}, Json{{"name", "y"}}, Json{{"name", "z"}}});
  for (const int v : sample) vals.push_back(Json{{"name", "vertex" + std::to_string(v)}, {"vertex", v}});
  nxr::io::StatisticSpec st;
  st.name = name; st.path = path; st.measure = "point_spread"; st.method = "point_source";
  st.of_field_id = text(*rec, "id"); st.n_members = NS;
  st.axes = Json::array({Json{{"name", "source_vertex"}, {"values", vals}, {"size", NS}}, Json{{"name", "orientation"}, {"values", ors}, {"size", 3}}, Json{{"name", "measure"}, {"size", 6}}});
  st.components = Json::array({Json{{"name", "peak_error_cm"}}, Json{{"name", "diameter_cm"}}, Json{{"name", "area_cm2"}}, Json{{"name", "characteristic_level"}}, Json{{"name", "size_cm"}}, Json{{"name", "peak_power"}}});
  st.data = nxr::io::view(out, {NS, 3, 6});
  st.description = "the instrument's blur: a unit current at each sampled vertex (x, y and z separately) through " + text(fwd, "name") + " and " + kname + ", read by the spatial events' chain (levels " + Json(opt.levels).dump() + ", the patch where the reference level's power is at least " + std::to_string(opt.fraction) + " of its peak)";
  st.options = Json{{"forward", text(fwd, "name")}, {"kernel", kname}, {"levels", opt.levels}, {"reference_level", opt.reference_level}, {"fraction", opt.fraction}, {"spatial_shape", opt.spatial_shape}, {"sample", sample}, {"surface", S.name}, {"level_sizes_cm", size_cm}};
  Json pj = stamp(key);
  pj["by"] = "nxr-neuro pointspread compute";
  pj["warnings"] = Json::array({"the blur of THIS head model and kernel (their noise model and regularisation) for a source ALONE: many sources blur together differently",
                                "sizes are octave-spaced levels; the patch is cut at a fraction of the peak (a point source has no noise floor), not at the events' false-alarm threshold",
                                "a source is planted along each Cartesian axis, no orientation is assumed; a vertex the sample misses is not measured"});
  st.producer = pj;
  std::string id;
  sub.compose([&] { id = sub.statistic(std::move(st)); });
  res.info = Json{{"name", name}, {"id", id}, {"sources", NS}, {"replaced", replaced}, {"cache", "miss"}};
  return res;
}

}  // namespace nxr::pipelines
