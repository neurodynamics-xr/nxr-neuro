// nxr-pipelines — quadrature per cell and per burst (cellquadrature.h). Preparation reads the stored nodes and type-checks them; the nodes are the pure functions of compute/
// (quadrature::groupedCongruenceNorms, partition::sumOverMembers); the results go back to io as measurements.
#include "nxr/pipelines/cellquadrature.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>

#include "nxr/io/array_reader.h"
#include "nxr/io/finder.h"
#include "nxr/io/hash.h"
#include "nxr/io/layout.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/ladder.h"
#include "nxr/partition.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/quadrature.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }

const char* const kTotal = "cross_spectrum_norm2";
const char* const kQuad = "cross_spectrum_quadrature_norm2";
const char* const kPairTotal = "tile_pair_cross_spectrum_norm2";
const char* const kPairQuad = "tile_pair_cross_spectrum_quadrature_norm2";

template <class T> std::vector<T> read_array(SubjectWriter& sub, const std::string& path, std::vector<std::int64_t>* shape = nullptr) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

// how the rows were made, as `run` names it: the pipeline and its key's hash (the recipe it names is an entry of the dataset's `recipes` dictionary)
std::string made_by(const CacheKey& key) { return "pipeline " + key.pipeline + " \xc2\xb7 cache " + sha256_hex(key.json().dump()).substr(0, 16) + " \xc2\xb7 nxr-neuro " + NXR_NEURO_VERSION; }

std::vector<std::string> computed_by_of(const Database& db, const std::string& selection_id, const std::string& field_id, const std::string& measure) {
  std::vector<std::string> out;
  for (const auto& r : db.all("SELECT DISTINCT computed_by FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = ?", {V(selection_id), V(field_id), V(measure)}))
    out.push_back(opt_text(r, "computed_by").value_or(""));
  return out;
}

// the two ingredient arrays of one product, [S × members] each, written as ONE array [S × members × 2] beside the recording and two rows
Json store_ingredients(SubjectWriter& sub, const Row& rec, const std::string& selection_id, const std::string& stem, const Eigen::MatrixXd& total, const Eigen::MatrixXd& quad, const std::string& by,
                       const char* mtotal = kTotal, const char* mquad = kQuad, const std::vector<std::pair<std::int32_t, std::int32_t>>* pair_list = nullptr) {
  const std::int64_t S = total.rows(), N = total.cols();
  std::vector<double> arr(static_cast<std::size_t>(S * N * 2));
  for (std::int64_t s = 0; s < S; ++s)
    for (std::int64_t n = 0; n < N; ++n) {
      arr[static_cast<std::size_t>((s * N + n) * 2)] = total(s, n);
      arr[static_cast<std::size_t>((s * N + n) * 2 + 1)] = quad(s, n);
    }
  const std::string path = text(rec, "path") + "_" + safe_name(stem);
  sub.write_array(path, view(arr, {S, N, 2}));
  Json rows = Json::array({Json{{"selection_id", selection_id}, {"of_field_id", text(rec, "id")}, {"measure", mtotal}, {"array_path", path}, {"component", 0}, {"computed_by", by}},
                           Json{{"selection_id", selection_id}, {"of_field_id", text(rec, "id")}, {"measure", mquad}, {"array_path", path}, {"component", 1}, {"computed_by", by}}});
  write_measurements(sub.db(), rows, by);
  Json out{{"array", path}, {"rows", S}, {"members", N}};
  if (pair_list) {   // the rows ARE the tile pairs: which pair a row is, as int32 [pairs × 2], beside the values
    std::vector<std::int32_t> pl;
    for (const auto& p : *pair_list) { pl.push_back(p.first); pl.push_back(p.second); }
    const std::string ppath = path + "_pairs";
    sub.write_array(ppath, view(pl, {static_cast<std::int64_t>(pair_list->size()), 2}));
    out["pair_array"] = ppath;
  }
  return out;
}
}  // namespace

CellQuadratureResult cellquadrature_compute(DatasetWriter& ds, SubjectWriter& sub, const CellQuadratureOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  // ── the request, checked before anything is read ──
  if (opt.cross_spectrum.empty()) invalid("cellquadrature: --cross-spectrum names the stored analytic cross-spectrum (`<stem>_<band>_windows`, `crossspectrum compute --estimator analytic`)");
  if (opt.space_level < 0) invalid("cellquadrature: --space-level is the surface ladder's level (0, 1, 2, \xe2\x80\xa6)");
  if (opt.transform != "auto" && opt.transform != "eigen" && opt.transform != "chebyshev" && opt.transform != "exact") invalid("cellquadrature: --transform is auto (eigenmodes first, chebyshev an announced fallback), eigen, chebyshev or exact");
  if (opt.weights != "area" && opt.weights != "none") invalid("cellquadrature: --weights is area (the vertices count by their area, the default) or none (every vertex counts equally)");
  if (opt.order < 0) invalid("cellquadrature: --order is at least 0 (0: chosen from the tolerance)");
  if (!(opt.tolerance > 0)) invalid("cellquadrature: --tolerance is above 0");
  const int l = opt.space_level;

  // ── PREPARATION (host): the rows it reads, type-checked; nothing is computed ──
  // the cross-spectrum: its row (identity), its channels' manifold, its provenance
  std::optional<Row> crow;
  std::string cs_manifold;
  for (const Row& r : statistics_of(sub.db(), sub.id())) {
    if (text(r, "name") != opt.cross_spectrum || text(r, "method") != "analytic" || text(r, "measure") != "cross_spectrum_sum") continue;
    const Json axes = Json::parse(text(r, "axes_json"));
    const auto sel = sub.db().read("selection", axes[1].value("selection_id", std::string()));
    if (!sel) invalid(opt.cross_spectrum + ": its channels (the channel axis's Selection) are not stored");
    crow = r;
    cs_manifold = text(*sel, "manifold_id");
    break;
  }
  if (!crow) missing(sub.name() + ": no analytic cross-spectrum statistic named '" + opt.cross_spectrum + "' (`nxr-neuro crossspectrum compute --estimator analytic` first; one stored as an Operator before 2026-11-13 is not read: --replace it)");
  const Json cpj = Json::parse(opt_text(*crow, "producer_json").value_or("{}"), nullptr, false);
  if (!cpj.is_object() || !cpj.contains("recording") || !cpj.contains("window_level") || !cpj.contains("window_samples")) invalid(opt.cross_spectrum + ": its provenance names no recording and tile level (made before the time tiles: --replace it)");
  const std::string recording = cpj["recording"].get<std::string>();
  const int window_level = cpj["window_level"].get<int>();
  const std::int64_t window_samples = cpj["window_samples"].get<std::int64_t>();
  if (window_level < 0 || window_samples != (std::int64_t(1) << window_level)) invalid(opt.cross_spectrum + ": not stored on a dyadic tile level (level " + std::to_string(window_level) + ", " + std::to_string(window_samples) + " samples): --replace it");
  const auto rec = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ? AND kind = 'recording' AND status = 'complete'", {V(sub.id()), V(recording)});
  if (!rec) missing(sub.name() + ": the cross-spectrum's recording '" + recording + "' is not stored");
  const finder::Value rv = finder::value_of(sub.db(), "field", *rec);
  if (!rv.time) invalid(recording + ": it has no time line");
  const std::string tid = *rv.time;
  // the joint partition of the pair (space level l, the time level whose window IS the cross-spectrum's tile level)
  std::optional<Row> jrow;
  std::string offered;
  for (const Row& r : sub.db().all("SELECT * FROM selection WHERE subject_id = ? AND type = 'set' AND status = 'complete' AND params_json LIKE '%\"joint\"%' ORDER BY name", {V(sub.id())})) {
    const Json jp = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
    if (!jp.is_object() || !jp.contains("joint")) continue;
    const Json& j = jp["joint"];
    if (j.value("recording", std::string()) != recording || j.value("space_level", -1) != l) continue;
    const std::int64_t ws = j.value("window_samples", std::int64_t(0));
    offered += (offered.empty() ? "" : ", ") + std::string("time level ") + std::to_string(j.value("time_level", -1)) + " (tiles of " + std::to_string(ws) + " samples)";
    if (opt.joint ? text(r, "name") == *opt.joint : ws == window_samples) { if (!jrow) jrow = r; }
  }
  if (!jrow) {
    if (opt.joint) invalid("cellquadrature: no joint Selection named '" + *opt.joint + "' for " + recording + " at spatial level " + std::to_string(l) + " (`joint compute`)");
    invalid("cellquadrature: the cross-spectrum '" + opt.cross_spectrum + "' is on tiles of " + std::to_string(window_samples) + " samples (level " + std::to_string(window_level) + "), and " + recording + " has no joint partition at spatial level " + std::to_string(l) +
            " on that time level" + (offered.empty() ? " (none at this spatial level: `nxr-neuro joint compute`)" : " (it has " + offered + "): the tile level of the cross-spectrum must equal the pair's time level; recompute the cross-spectrum with --level " + std::to_string(window_level) + " for the pair's window, or `joint compute` the pair"));
  }
  const Json jj = Json::parse(text(*jrow, "params_json"))["joint"];
  if (jj.value("window_samples", std::int64_t(0)) != window_samples)
    invalid("cellquadrature: the joint Selection '" + text(*jrow, "name") + "' has time tiles of " + std::to_string(jj.value("window_samples", std::int64_t(0))) + " samples, the cross-spectrum '" + opt.cross_spectrum + "' of " + std::to_string(window_samples) +
            " (level " + std::to_string(window_level) + "): the tile level of the cross-spectrum must equal the pair's time level");
  const std::int64_t frames = jj["frames"].get<std::int64_t>();
  // the kernel
  Row kern;
  if (opt.kernel) {
    const auto r = sub.db().one("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.kernel)});
    if (!r) missing(sub.name() + ": no inverse kernel named '" + *opt.kernel + "'");
    kern = *r;
  } else {
    const auto rs = sub.db().all("SELECT * FROM operator WHERE subject_id = ? AND kind = 'inverse kernel' AND status = 'complete' AND path IS NOT NULL", {V(sub.id())});
    if (rs.size() != 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " inverse kernels; name one with --kernel");
    kern = rs[0];
  }
  const std::string kname = text(kern, "name");
  if (text(kern, "from_manifold_id") != cs_manifold) invalid(kname + " is on other channels than " + opt.cross_spectrum + " (the cross-spectrum's channels are not the kernel's sensors)");
  const SurfaceInput S = read_surface(sub, jj["surface"].get<std::string>(), true);
  if (text(kern, "to_manifold_id") != S.id) invalid(kname + " maps to another surface than the joint Selection's (" + S.name + ")");
  const std::int64_t nV = S.n_vertices();
  const SurfaceLevel SL = read_surface_level(sub, S, l);
  if (SL.level.id != jj["space"].get<std::string>()) invalid("cellquadrature: the joint Selection '" + text(*jrow, "name") + "' was made on another tiling of level " + std::to_string(l) + " than the surface's now (`joint compute --replace`)");
  const std::int64_t tiles = SL.nodes;
  if (is_null(kern, "from_selection_id")) invalid(kname + ": its channels (from_selection) are not stored");
  const auto ksel = sub.db().read("selection", text(kern, "from_selection_id"));
  if (!ksel || is_null(*ksel, "array_path")) invalid(kname + ": its channels (from_selection) are not stored");
  const std::vector<std::int32_t> chans = read_array<std::int32_t>(sub, text(*ksel, "array_path"));
  const std::int64_t m = static_cast<std::int64_t>(chans.size());
  // the bursts (optional): a spans Selection on the recording's time line
  std::optional<Row> brow;
  std::vector<std::pair<double, double>> spans;
  if (opt.bursts) {
    const auto rs = sub.db().all("SELECT * FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'", {V(sub.id()), V(*opt.bursts)});
    if (rs.empty()) missing(sub.name() + ": no Selection named '" + *opt.bursts + "' (the bursts: `run global_bursts`'s `bursts` output)");
    if (rs.size() > 1) invalid(sub.name() + ": " + std::to_string(rs.size()) + " Selections are named '" + *opt.bursts + "'");
    brow = rs[0];
    if (text(*brow, "type") != "spans") invalid(*opt.bursts + " is a " + text(*brow, "type") + " Selection, not spans (events with a start and a stop)");
    if (text(*brow, "manifold_id") != tid) invalid(*opt.bursts + " is on another time line than " + recording + "'s (the bursts must be events of the cross-spectrum's recording)");
    for (const auto& r : sub.db().all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(text(*brow, "id"))})) spans.emplace_back(std::stod(text(r, "start")), std::stod(text(r, "stop")));
  }
  // the time line: the tile k covers [origin + k W dt, origin + (k + 1) W dt)
  const auto ld = sub.db().one("SELECT d.spacing AS spacing, d.origin AS origin FROM geometry_dimension d JOIN manifold mm ON mm.geometry_id = d.geometry_id WHERE mm.id = ? AND d.ordinal = 0", {V(tid)});
  if (!ld || is_null(*ld, "spacing")) invalid(recording + ": its time line is not a closed-form line (no sampling step)");
  const double dt = std::stod(text(*ld, "spacing")), t0 = is_null(*ld, "origin") ? 0.0 : std::stod(text(*ld, "origin"));
  // the spatial band's choice, decided from the request and what is stored BEFORE the key (a plan reads, writes nothing)
  SpatialBandOptions bo; bo.transform = opt.transform; bo.order = opt.order; bo.tolerance = opt.tolerance; bo.modes = opt.modes;
  if (opt.transform == "exact" && nV > 4096) invalid("cellquadrature: --transform exact is the dense eigen-decomposition: at most 4096 vertices, this surface has " + std::to_string(nV) + "; use --transform chebyshev");
  const SpatialBandPlan bplan = plan_spatial_band(sub, S, SL, bo);

  // ── THE CACHE KEYS: the cells', and the bursts' (which add the bursts' identity) ──
  auto identity = [&](const Row& r) { return Json{{"id", text(r, "id")}, {"modified", opt_text(r, "modified_utc").value_or("")}}; };
  CacheKey ckey;
  ckey.pipeline = "cellquadrature compute";
  ckey.params = Json{{"space_level", l}, {"window_level", window_level}, {"spatial_transform", bplan.key}, {"weight", opt.weights}, {"ingredients", Json::array({kTotal, kQuad})}};
  if (opt.weights == "area") ckey.params["vertex_area"] = "barycentric";           // `none` keeps the key (and the rows) it had before the option
  ckey.input = Json{{"cross_spectrum", identity(*crow)}, {"kernel", identity(kern)}, {"joint", Json{{"id", text(*jrow, "id")}, {"layout", sha256_hex(text(*jrow, "params_json")).substr(0, 16)}}}, {"mesh", mesh_key(S.faces)}, {"positions", positions_key(S.vertices)}, {"channels", m}};
  ckey.compute = NXR_COMPUTE_VERSION;
  CacheKey pkey = ckey;       // the tile pairs' key: the same inputs and options, the pair ingredients, the level's tile graph
  pkey.params["ingredients"] = Json::array({kPairTotal, kPairQuad});
  pkey.params["tile_pairs"] = "nxr::ladder::tileAdjacency";
  CacheKey bkey = ckey;
  if (brow) {   // the Selection's modified time moves when measurements are written over it: its identity is its id and its events
    std::string ev;
    for (const auto& sp : spans) ev += std::to_string(sp.first) + "," + std::to_string(sp.second) + ";";
    bkey.input["bursts"] = Json{{"id", text(*brow, "id")}, {"events", sha256_hex(ev).substr(0, 16)}, {"n", spans.size()}};
  }
  CacheKey pbkey = pkey;
  pbkey.input["bursts"] = bkey.input.value("bursts", Json(nullptr));
  const std::string jid = text(*jrow, "id"), fid = text(*rec, "id");
  const std::string cby = made_by(ckey), bby = made_by(bkey), pby = made_by(pkey), pbby = made_by(pbkey);
  auto status = [&](const std::string& sel, const std::string& by, bool& hit, const char* mt = kTotal, const char* mq = kQuad) {
    const auto a = computed_by_of(sub.db(), sel, fid, mt), b = computed_by_of(sub.db(), sel, fid, mq);
    hit = false;
    if (a.empty() && b.empty()) return false;                                         // nothing stored
    if (a.size() == 1 && b.size() == 1 && a[0] == by && b[0] == by) { hit = true; return true; }
    return true;                                                                      // rows of another job
  };
  CellQuadratureResult res;
  const bool cells_have = status(jid, cby, res.cells_hit);
  const bool bursts_have = brow && status(text(*brow, "id"), bby, res.bursts_hit);
  const bool cpairs_have = opt.tile_pairs && status(jid, pby, res.cell_pairs_hit, kPairTotal, kPairQuad);
  const bool bpairs_have = opt.tile_pairs && brow && status(text(*brow, "id"), pbby, res.burst_pairs_hit, kPairTotal, kPairQuad);
  if (cpairs_have && !res.cell_pairs_hit && !opt.replace) throw Error(code::already_exists, sub.name() + ": " + kPairTotal + " and " + kPairQuad + " of " + text(*jrow, "name") + " exist (made by another job); pass --replace");
  if (bpairs_have && !res.burst_pairs_hit && !opt.replace) throw Error(code::already_exists, sub.name() + ": " + kPairTotal + " and " + kPairQuad + " of " + *opt.bursts + " exist (made by another job); pass --replace");
  if (cells_have && !res.cells_hit && !opt.replace) throw Error(code::already_exists, sub.name() + ": " + kTotal + " and " + kQuad + " of " + text(*jrow, "name") + " exist (made by another job); pass --replace");
  if (bursts_have && !res.bursts_hit && !opt.replace) throw Error(code::already_exists, sub.name() + ": " + kTotal + " and " + kQuad + " of " + *opt.bursts + " exist (made by another job); pass --replace");
  say(Json{{"stage", "prepared"}, {"cross_spectrum", opt.cross_spectrum}, {"recording", recording}, {"joint", text(*jrow, "name")}, {"tile_level", window_level}, {"tiles", tiles}, {"time_tiles", frames}, {"channels", m},
           {"bursts", brow ? Json(static_cast<std::int64_t>(spans.size())) : Json(nullptr)}, {"cells_cache", res.cells_hit ? "hit" : "miss"}, {"bursts_cache", brow ? Json(res.bursts_hit ? "hit" : "miss") : Json(nullptr)},
           {"weights", opt.weights}, {"tile_pairs", opt.tile_pairs ? Json(static_cast<std::int64_t>(SL.pairs.size())) : Json(nullptr)}, {"pairs_cache", opt.tile_pairs ? Json(res.cell_pairs_hit ? "hit" : "miss") : Json(nullptr)}});
  res.info = Json{{"cross_spectrum", opt.cross_spectrum}, {"recording", recording}, {"joint", text(*jrow, "name")}, {"surface", S.name}, {"space_level", l}, {"tile_level", window_level}, {"tiles", tiles}, {"time_tiles", frames},
                  {"kernel", kname}, {"channels", m}, {"band", Json{{"name", cpj.value("band_name", std::string()) }, {"lo_hz", cpj.value("band_lo_hz", std::nan(""))}, {"hi_hz", cpj.value("band_hi_hz", std::nan(""))}}},
                  {"joint_band_hz", jj["band"]}, {"cells_cache", res.cells_hit ? "hit" : "miss"}, {"warnings", Json::array()}, {"measures", opt.tile_pairs ? Json::array({kTotal, kQuad, kPairTotal, kPairQuad}) : Json::array({kTotal, kQuad})},
                  {"weights", opt.weights}, {"tile_pairs", opt.tile_pairs ? Json(static_cast<std::int64_t>(SL.pairs.size())) : Json(nullptr)}};
  if (brow) res.info["bursts"] = Json{{"selection", *opt.bursts}, {"events", spans.size()}, {"cache", res.bursts_hit ? "hit" : "miss"}};
  // a band that is not the joint's is not a refusal (the level is what must match), but it is said
  if (cpj.contains("band_lo_hz") && jj.contains("band") && jj["band"].is_array() && jj["band"].size() == 2) {
    const double lo = cpj["band_lo_hz"].get<double>(), hi = cpj["band_hi_hz"].get<double>();
    if (std::abs(lo - jj["band"][0].get<double>()) > 1e-6 || std::abs(hi - jj["band"][1].get<double>()) > 1e-6)
      res.info["warnings"].push_back("the cross-spectrum's band (" + std::to_string(lo) + " \xe2\x80\xa6 " + std::to_string(hi) + " Hz) is not the band the joint's time level resolves (" + std::to_string(jj["band"][0].get<double>()) + " \xe2\x80\xa6 " + std::to_string(jj["band"][1].get<double>()) + " Hz): the tiles match, the bands are the caller's");
  }
  if (res.cells_hit && (!brow || res.bursts_hit) && (!opt.tile_pairs || (res.cell_pairs_hit && (!brow || res.burst_pairs_hit)))) return res;                           // everything asked is stored by this job

  // ── LOADING: the tiles of the cross-spectrum on the kernel's channels, in its order (refused for a channel it does not cover) ──
  const CrossSpectrumWindows win = crossspectrum_windows_read(sub, opt.cross_spectrum, chans, "stored");
  if (win.level != window_level) invalid(opt.cross_spectrum + ": read at level " + std::to_string(win.level) + ", recorded at " + std::to_string(window_level));
  if (win.sum.empty()) invalid(opt.cross_spectrum + ": it stores no tile");
  for (const std::int64_t k : win.index) if (k < 0 || k >= frames) invalid(opt.cross_spectrum + ": a stored tile (" + std::to_string(k) + ") is outside the joint's " + std::to_string(frames) + " time tiles");

  // ── M_l = (F_l ⊗ I) K on the kernel's channels, bound ONCE (the level's band applied to every component of the kernel's rows; a linear map before any non-linear step, D106) ──
  std::vector<std::int64_t> kshape;
  const std::vector<float> kf = read_array<float>(sub, text(kern, "path"), &kshape);
  if (kshape.size() != 2 || kshape[1] != m) invalid(kname + ": its array is not [rows x " + std::to_string(m) + " channels]");
  if (kshape[0] % nV != 0) invalid(kname + ": its " + std::to_string(kshape[0]) + " rows are not a whole number of components for the " + std::to_string(nV) + " vertices of " + S.name);
  const std::int64_t comps = kshape[0] / nV;
  const SpatialBand band = make_spatial_band(ds, sub, S, SL, bo, progress);
  Eigen::MatrixXd A(kshape[0], m);
  for (std::int64_t comp = 0; comp < comps; ++comp) {
    Eigen::MatrixXd Kc(nV, m);
    for (std::int64_t v = 0; v < nV; ++v) for (std::int64_t j = 0; j < m; ++j) Kc(v, j) = kf[static_cast<std::size_t>((v * comps + comp) * m + j)];
    const Eigen::MatrixXd Mc = band.apply(Kc);
    for (std::int64_t v = 0; v < nV; ++v) A.row(v * comps + comp) = Mc.row(v);
  }
  std::vector<int> row_tile(static_cast<std::size_t>(kshape[0]));
  for (std::int64_t v = 0; v < nV; ++v) for (std::int64_t c = 0; c < comps; ++c) row_tile[static_cast<std::size_t>(v * comps + c)] = SL.tile[static_cast<std::size_t>(v)];
  // the area weights: a row (vertex, component) weighs its vertex's barycentric area (the tiles' areas are sums of these); `none`: no weights
  Eigen::VectorXd row_w;
  if (opt.weights == "area") {
    namespace lad = nxr::ladder;
    lad::Mesh whole; whole.n = static_cast<int>(nV); whole.positions = S.vertices; whole.faces = S.faces;
    const std::vector<double> va = lad::vertexAreas(whole);
    row_w.resize(kshape[0]);
    for (std::int64_t v = 0; v < nV; ++v) for (std::int64_t c = 0; c < comps; ++c) row_w[v * comps + c] = va[static_cast<std::size_t>(v)];
  }
  res.info["spatial_band"] = band.info;
  for (const auto& w : band.warnings) res.info["warnings"].push_back(w);
  say(Json{{"stage", "spatial_band"}, {"transform", band.info.value("transform", std::string())}, {"source", band.info.value("source", std::string())}, {"rows", kshape[0]}, {"components", comps}});

  // the recipe: every option, the inputs, what the spatial band resolved (the rows' `computed_by` names it)
  const std::string producer = internal::producer_string("cellquadrature compute");
  auto recipe = [&](const CacheKey& key, const std::string& by_name) {
    DictionaryEntry e;
    e.name = by_name.substr(0, by_name.find(" \xc2\xb7 nxr-neuro"));
    e.cols.set("function", std::string("pipeline"));
    e.cols.set("params_json", jdump(Json{{"pipeline", "cellquadrature compute"}, {"producer", producer}, {"parameters", key.params}, {"inputs", key.input}, {"cross_spectrum", opt.cross_spectrum}, {"joint", text(*jrow, "name")},
                                         {"kernel", kname}, {"spatial_band", band.info}, {"warnings", res.info["warnings"]}, {"cache", key.json()}, {"nxr_neuro", NXR_NEURO_VERSION},
                                         {"note", "T = B C B^T per spatial tile of the level; B the tile's rows of the kernel with the level's Weyl band, all components; stored: sum |T|^2 and sum |Im T|^2; the quadrature fraction is derived at read; weights: " + opt.weights + " (area: sums of a_i a_j |T_ij|^2, a the barycentric vertex areas); tile pairs: T_ss' = B_s C B_s'^T of the level's touching tiles"}}));
    DictionarySpec spec;
    spec.name = "recipes"; spec.scope = "dataset"; spec.description = "the recipes of this dataset: one entry a version, the recipe in params_json"; spec.entries = std::vector<DictionaryEntry>{e};
    sub.dictionary(spec);
  };

  // the graph's node for a list of cross-spectra (the cells' time tiles, or the bursts' summed ones): the tiles' own blocks and, with tile pairs, the blocks between touching tiles
  const int nt = static_cast<int>(tiles);
  struct Norms { Eigen::MatrixXd total, quad, ptotal, pquad; };
  auto norms_of = [&](const std::vector<Eigen::MatrixXcd>& Cs) {
    Norms n;
    if (opt.tile_pairs) {
      const auto r = quadrature::pairCongruenceNorms(A, row_tile, nt, SL.pairs, Cs, row_w);
      n.total = r.tiles.total; n.quad = r.tiles.quadrature; n.ptotal = r.pairTotal; n.pquad = r.pairQuadrature;
    } else {
      const auto r = quadrature::groupedCongruenceNorms(A, row_tile, nt, Cs, row_w);
      n.total = r.total; n.quad = r.quadrature;
    }
    return n;
  };
  // [rows × frames] with the stored tiles at their ordinals, NaN elsewhere
  auto place = [&](const Eigen::MatrixXd& m) {
    Eigen::MatrixXd o = Eigen::MatrixXd::Constant(m.rows(), frames, std::nan(""));
    for (std::size_t w = 0; w < win.sum.size(); ++w) o.col(win.index[w]) = m.col(static_cast<Eigen::Index>(w));
    return o;
  };
  const std::string cstem = "cellquadrature." + opt.cross_spectrum + "." + text(*jrow, "name");

  // ── GRAPH + READBACK, the cells: per spatial tile (and tile pair) and time tile ──
  if (!res.cells_hit || (opt.tile_pairs && !res.cell_pairs_hit)) {
    const Norms n = norms_of(win.sum);
    if (!res.cells_hit) {
      if (cells_have) { delete_measurements(sub.db(), Json{{"selectionId", jid}, {"ofFieldId", fid}, {"measure", kTotal}}); delete_measurements(sub.db(), Json{{"selectionId", jid}, {"ofFieldId", fid}, {"measure", kQuad}}); res.info["cells_replaced"] = true; }
      Json stored;
      sub.compose([&] { recipe(ckey, cby); stored = store_ingredients(sub, *rec, jid, cstem, place(n.total), place(n.quad), cby); });
      res.info["cells"] = stored;
      res.info["cells"]["stored_tiles"] = win.sum.size();
      say(Json{{"stage", "cells"}, {"stored", stored}, {"cache", "miss"}});
    }
    if (opt.tile_pairs && !res.cell_pairs_hit) {
      if (cpairs_have) { delete_measurements(sub.db(), Json{{"selectionId", jid}, {"ofFieldId", fid}, {"measure", kPairTotal}}); delete_measurements(sub.db(), Json{{"selectionId", jid}, {"ofFieldId", fid}, {"measure", kPairQuad}}); res.info["cell_pairs_replaced"] = true; }
      Json stored;
      sub.compose([&] { recipe(pkey, pby); stored = store_ingredients(sub, *rec, jid, "tilepairs." + cstem.substr(std::string("cellquadrature.").size()), place(n.ptotal), place(n.pquad), pby, kPairTotal, kPairQuad, &SL.pairs); });
      res.info["cell_pairs"] = stored;
      say(Json{{"stage", "cell_pairs"}, {"stored", stored}, {"cache", "miss"}});
    }
  }

  // ── GRAPH + READBACK, the bursts: C summed over each burst's time tiles FIRST (products add: exact), then the ingredients per spatial tile (and pair) ──
  if (brow && (!res.bursts_hit || (opt.tile_pairs && !res.burst_pairs_hit))) {
    const std::int64_t B = static_cast<std::int64_t>(spans.size());
    const double eps = 0.5 * dt, width = static_cast<double>(window_samples) * dt;
    std::map<std::int64_t, std::size_t> at;                                    // tile ordinal -> position in win
    for (std::size_t w = 0; w < win.index.size(); ++w) at[win.index[w]] = w;
    std::vector<Eigen::MatrixXcd> sums(static_cast<std::size_t>(B));
    Json report = Json::array();
    for (std::int64_t b = 0; b < B; ++b) {
      const double a = spans[static_cast<std::size_t>(b)].first, z = spans[static_cast<std::size_t>(b)].second;
      // the tiles WHOLLY inside the event [start, stop] (the bursts are tile-aligned, as detect_spans makes them)
      std::int64_t k0 = static_cast<std::int64_t>(std::ceil((a - t0 - eps) / width)), k1 = static_cast<std::int64_t>(std::floor((z - t0 + eps) / width)) - 1;
      k0 = std::max<std::int64_t>(k0, 0);
      std::vector<Eigen::MatrixXcd> parts;
      std::int64_t absent = 0;
      for (std::int64_t k = k0; k <= k1; ++k) { const auto it = at.find(k); if (it == at.end()) ++absent; else parts.push_back(win.sum[it->second]); }
      const bool whole = k1 >= k0 && absent == 0;
      if (whole) sums[static_cast<std::size_t>(b)] = partition::sumOverMembers(parts, std::vector<int>(parts.size(), 0), 1).sum[0];     // an empty matrix: NaN
      report.push_back(Json{{"burst", b}, {"start", a}, {"stop", z}, {"tiles", k1 >= k0 ? k1 - k0 + 1 : 0}, {"first_tile", k0}, {"missing_tiles", absent}, {"complete", whole}});
    }
    const Norms n = norms_of(sums);
    const std::string bid = text(*brow, "id");
    res.info["bursts"]["events_read"] = report;
    if (!res.bursts_hit) {
      if (bursts_have) { delete_measurements(sub.db(), Json{{"selectionId", bid}, {"ofFieldId", fid}, {"measure", kTotal}}); delete_measurements(sub.db(), Json{{"selectionId", bid}, {"ofFieldId", fid}, {"measure", kQuad}}); res.info["bursts"]["replaced"] = true; }
      Json stored;
      sub.compose([&] { recipe(bkey, bby); stored = store_ingredients(sub, *rec, bid, "cellquadrature." + opt.cross_spectrum + "." + *opt.bursts, n.total, n.quad, bby); });
      res.info["bursts"]["stored"] = stored;
      say(Json{{"stage", "bursts"}, {"stored", stored}, {"cache", "miss"}});
    }
    if (opt.tile_pairs && !res.burst_pairs_hit) {
      if (bpairs_have) { delete_measurements(sub.db(), Json{{"selectionId", bid}, {"ofFieldId", fid}, {"measure", kPairTotal}}); delete_measurements(sub.db(), Json{{"selectionId", bid}, {"ofFieldId", fid}, {"measure", kPairQuad}}); res.info["bursts"]["pairs_replaced"] = true; }
      Json stored;
      sub.compose([&] { recipe(pbkey, pbby); stored = store_ingredients(sub, *rec, bid, "tilepairs." + opt.cross_spectrum + "." + *opt.bursts, n.ptotal, n.pquad, pbby, kPairTotal, kPairQuad, &SL.pairs); });
      res.info["bursts"]["pairs_stored"] = stored;
      say(Json{{"stage", "burst_pairs"}, {"stored", stored}, {"cache", "miss"}});
    }
  }
  return res;
}

}  // namespace nxr::pipelines
