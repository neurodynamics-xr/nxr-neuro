// nxr-pipelines — patches on a static cortical map (patches.h). Preparation reads rows and binds the operators; the nodes are the pure functions of
// compute/ (filterbank band, partition measurements, intervals median/mad, regions); the results are stored by io (Field, Selection, measurement rows, statistics).
#include "nxr/pipelines/patches.h"

#include "nxr/pipelines/task.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>

#include <Eigen/Eigenvalues>

#include "nxr/intrinsic_surface.h"
#include "nxr/filterbank.h"
#include "nxr/intervals.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/cache.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/measure.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/statistics.h"
#include "nxr/pipelines/spatial_band.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/regions.h"

#ifndef NXR_COMPUTE_VERSION
#define NXR_COMPUTE_VERSION "unknown"
#endif

namespace nxr::pipelines {

using namespace nxr::io;
namespace fb = nxr::filterbank;
namespace lad = nxr::ladder;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }

template <class T> std::vector<T> read_array(SubjectWriter& sub, const std::string& path) {
  ArrayReader r = ArrayReader::open(sub.store(), path);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<T> out(raw.size() / sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}
std::string regex_escape(const std::string& s) {
  std::string o;
  for (char c : s) { if (std::strchr("\\^$.|?*+()[]{}", c)) o += '\\'; o += c; }
  return o;
}
// the whole name of one Field, as measure compute's `fields` pattern
std::string exactly(const std::string& name) { return "^" + regex_escape(name) + "$"; }
}  // namespace

PatchesComputeResult patches_compute(DatasetWriter& ds, SubjectWriter& sub, const PatchesComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };

  // ── the request, checked before anything is read ──
  if (opt.map.empty()) invalid("patches: --map names the stored Field to find patches in");
  if (opt.level < 0) invalid("patches: --level is the ladder's level (0, 1, 2, …): the level whose Weyl band the map is filtered to");
  nxr::regions::Sign sign;
  try { sign = nxr::regions::parseSign(opt.sign); } catch (const std::invalid_argument& e) { invalid(std::string("patches: --") + e.what()); }
  if (!(opt.exit_sd <= opt.enter_sd)) invalid("patches: --exit-sd (" + std::to_string(opt.exit_sd) + ") is at most --enter-sd (" + std::to_string(opt.enter_sd) + ")");
  if (opt.min_area && *opt.min_area < 0) invalid("patches: --min-area is at least 0 (m²)");
  if (opt.max_area && *opt.max_area < opt.min_area.value_or(0)) invalid("patches: --max-area is at least --min-area (m²)");
  if (opt.transform != "auto" && opt.transform != "eigen" && opt.transform != "chebyshev" && opt.transform != "exact") invalid("patches: --transform is auto (eigenmodes first, chebyshev an announced fallback), eigen, chebyshev or exact");
  if (opt.order < 0) invalid("patches: --order is at least 0 (0: chosen from the tolerance)");
  if (!(opt.tolerance > 0)) invalid("patches: --tolerance is above 0");
  const int l = opt.level;

  // ── PREPARATION (host): the rows it reads, type-checked; nothing is computed ──
  const task::Subject subj = task::Subject::of(sub);
  const auto frow = subj.find({.cls = task::Cls::field, .name = opt.map, .stored = true});
  if (!frow) missing(sub.name() + ": no stored Field '" + opt.map + "' (patches are found in a Field; a band of a map statistic is not read here)");
  if (frow->value_type.value_or(std::string()) != "scalar") invalid(opt.map + ": a " + frow->value_type.value_or(std::string()) + " Field; patches are found in a scalar map (one value per vertex)");
  const auto man = subj.by_id(task::Cls::manifold, frow->manifold_id.value_or(std::string()));
  if (!man || man->type.value_or(std::string()) != "surface") invalid(opt.map + ": its manifold is not a surface (a recording, a sensor array or a frequency axis has no tiles)");
  const SurfaceInput S = read_surface(sub, man->name, true);
  const std::int64_t nV = S.n_vertices();
  if (opt.transform == "exact" && nV > 4096) invalid("patches: --transform exact is the dense eigen-decomposition: at most 4096 vertices, this surface has " + std::to_string(nV) + "; use --transform chebyshev");
  const std::string surface = S.name, mid = S.id;
  const auto stored = stored_map(sub, opt.map, nV);
  if (!stored) invalid(opt.map + ": not one value per vertex of " + surface + " (" + std::to_string(nV) + ")");
  if (stored->cols() != 1) invalid(opt.map + ": " + std::to_string(stored->cols()) + " components per vertex; patches are found in a scalar map");
  std::int64_t nonfinite = 0;
  for (Eigen::Index i = 0; i < stored->rows(); ++i) if (!std::isfinite((*stored)(i, 0))) ++nonfinite;
  if (nonfinite) invalid(opt.map + ": " + std::to_string(nonfinite) + " vertices have no value (NaN or infinite); a band-pass of a map with holes is not defined: fill or mask them first");
  const Eigen::VectorXd x = stored->col(0);

  const SurfaceLevel SL = read_surface_level(sub, S, l);          // the ladder, the level, its tiles and graph, the Weyl pairing checked
  const std::string tiling = SL.tiling;
  const auto& lvl = SL.level;
  const std::int64_t nodes = SL.nodes;
  const std::vector<std::int32_t>& tile = SL.tile;
  const std::vector<double>& tile_area = SL.tile_area;
  const auto& pairs = SL.pairs;
  const Json& lp = SL.record;
  const double piece_area = SL.piece_area;
  const fb::Member& member = SL.bank.members[static_cast<std::size_t>(l)];
  nxr::ladder::Mesh whole; whole.n = static_cast<int>(nV); whole.positions = S.vertices; whole.faces = S.faces;
  const std::vector<double> vertex_area = lad::vertexAreas(whole);

  const std::string band_name = safe_name(opt.map + "_band_L" + std::to_string(l));
  const std::string band_path = subj.node_path(mid, band_name);
  const std::string patches_name = opt.map + " patches L" + std::to_string(l) + " " + opt.sign;
  const std::string patches_path = subj.node_path(mid, safe_name(patches_name));
  const std::string by = std::string("nxr-neuro patches compute \xc2\xb7 nxr-neuro ") + NXR_NEURO_VERSION;
  const std::string producer = internal::producer_string("patches compute");
  const std::string mkey = mesh_key(S.faces), pkey = positions_key(S.vertices);
  say(Json{{"stage", "prepared"}, {"map", opt.map}, {"surface", surface}, {"level", l}, {"tiles", nodes}, {"tile_pairs", pairs.size()}, {"band_lambda", Json::array({member.lo, member.hi})}});
  Json warnings = Json::array();

  // ── NODE spatial_band → MATERIALIZED: the band-passed map `<map>_band_L<l>` ──
  // the transform is decided from the request and what is stored BEFORE the key (a plan reads, writes nothing): eigenmodes first, chebyshev an announced fallback
  SpatialBandOptions bo; bo.transform = opt.transform; bo.order = opt.order; bo.tolerance = opt.tolerance; bo.modes = opt.modes;
  const SpatialBandPlan bplan = plan_spatial_band(sub, S, SL, bo);
  CacheKey bkey;
  bkey.pipeline = "patches spatial_band";
  bkey.params = Json{{"level", l}, {"bank", "weyl"}, {"member", l}, {"half_width_octaves", 1.0 / 3.0}, {"weyl_area_m2", piece_area}, {"spatial_transform", bplan.key}};
  bkey.input = Json{{"field", frow->id}, {"tiling", lvl.id}, {"mesh", mkey}, {"positions", pkey}};
  bkey.compute = NXR_COMPUTE_VERSION;
  PatchesComputeResult res;
  Json band_info;
  std::string band_id;
  bool band_hit = false;
  if (const auto hit = subj.cached(task::Cls::field, bkey.json(), band_path)) {
    band_hit = true;
    band_id = hit->id;
    band_info = (hit->producer.is_object() ? hit->producer.value("spatial_band", Json::object()) : Json::object());
    say(Json{{"stage", "spatial_band"}, {"field", band_name}, {"cache", "hit"}});
  } else {
    const SpatialBand sb = make_spatial_band(ds, sub, S, SL, bo, progress);
    band_info = sb.info;
    const Eigen::MatrixXd W = sb.apply(Eigen::MatrixXd(x));
    const Eigen::VectorXd band = W.col(0);
    std::vector<double> vals(band.data(), band.data() + band.size());
    Json pj = Json{{"by", "nxr-neuro patches compute"}, {"producer", producer}, {"spatial_band", band_info}, {"warnings", band_info.value("warnings", Json::array())},
                   {"source", Json{{"field", opt.map}, {"id", frow->id}}},
                   {"note", "the map filtered to the level's Weyl band (graph wavelet, Meyer edges ±1/3 octave, on the cotangent Laplacian pencil of the intrinsic Delaunay mesh): signed, local to the band's wavelength; a tile mean of it is a height relative to the surroundings at that scale, not the map's own value"}};
    const Json st = stamp(bkey);
    pj["cache"] = st["cache"];
    bool replaced = false;      // after the maths, so a refusal above leaves what is stored as it was
    clear_stored(ds, sub, {{"field", band_path}}, opt.replace, band_name + " (made with other choices)", replaced);
    sub.compose([&] {
      FieldSpec f;
      f.name = band_name; f.path = band_path; f.kind = "map"; f.manifold_id = mid; f.value_type = "scalar";
      f.data = view(vals, {nV});
      f.cols.set("function", "spatial_band").set("description", "'" + opt.map + "' filtered to the Weyl band of level " + std::to_string(l) + " of " + surface + "'s tile ladder");
      if (const auto u = frow->unit) f.cols.set("unit", *u);
      f.cols.json("producer_json", pj);
      band_id = sub.field(std::move(f));
    });
    say(Json{{"stage", "spatial_band"}, {"field", band_name}, {"cache", "miss"}, {"transform", band_info.value("transform", std::string())}, {"source", band_info.value("source", std::string())}, {"order", band_info.value("order", 0)}, {"series_error", band_info.value("series_error", 0.0)}});
  }

  if (band_info.contains("warnings") && band_info["warnings"].is_array()) for (const auto& w : band_info["warnings"]) warnings.push_back(w);
  if (band_hit) for (const auto& w : bplan.warnings) say(Json{{"stage", "spatial_band"}, {"warning", w}});
  // ── NODE measure (reduction over the level's partition): the band's mean, max, min, abs_max per tile ──
  MeasureComputeOptions mo;
  mo.labelling = tiling; mo.fields = exactly(band_name); mo.weight = "barycentric"; mo.measures = {"mean", "max", "min", "abs_max"}; mo.recipe = "patches-barycentric"; mo.replace = opt.replace;
  const MeasureComputeResult tiles_measured = measure_compute(ds, sub, mo);
  const MeasurementValues tile_mean_v = measurement_values(sub, lvl.id, band_id, std::nullopt, std::nullopt, "mean");
  Eigen::VectorXd mean = Eigen::VectorXd::Constant(nodes, std::nan(""));
  for (std::int64_t t = 0; t < std::min<std::int64_t>(nodes, tile_mean_v.members); ++t) mean(t) = tile_mean_v.values[static_cast<std::size_t>(t)];

  // ── NODE summarize → the baseline across the level's tiles (median, robust SD) ──
  StatisticComputeOptions so;
  so.over = tiling; so.of = band_name; so.measure = "mean"; so.summaries = {"median", "mad"}; so.combine = "each"; so.name = band_name + "_baseline"; so.replace = opt.replace;
  const StatisticComputeResult base = statistic_compute(ds, sub, so);
  const StoredStatistic bstat = read_statistic(sub, so.name.value());
  if (bstat.values.size() != 2) invalid(*so.name + ": expected a median and a robust SD, found " + std::to_string(bstat.values.size()) + " values");
  double med = bstat.values[0], sd = bstat.values[1];
  Json baseline = Json{{"statistic", *so.name}, {"id", base.info.value("id", std::string())}, {"median", med}, {"sd", sd}, {"over", "the level's tiles, of the band's tile means"}};
  // ── sign: the caller's, before the thresholds (a baseline of the SIGNED values) ──
  const Eigen::VectorXd signed_v = nxr::regions::applySign(mean, sign);
  if (sign == nxr::regions::Sign::Negative) med = -med;
  if (sign == nxr::regions::Sign::Magnitude) {
    std::vector<double> a;
    for (Eigen::Index t = 0; t < signed_v.size(); ++t) if (std::isfinite(signed_v(t))) a.push_back(signed_v(t));
    const Eigen::Map<const Eigen::VectorXd> av(a.data(), static_cast<Eigen::Index>(a.size()));
    med = nxr::intervals::median(av); sd = nxr::intervals::mad(av);
    baseline["note"] = "magnitude: the median and robust SD are of the absolute tile means, computed with the same functions (the stored statistic is of the signed means)";
  }
  baseline["median_used"] = med; baseline["sd_used"] = sd;
  const double scale = x.cwiseAbs().maxCoeff();
  if (!std::isfinite(sd) || !(sd > 1e-9 * scale)) invalid(opt.map + ": the robust SD of the level " + std::to_string(l) + " band's tile means is " + std::to_string(sd) + ", nothing beyond round-off (a constant map, or a level the map has nothing at): thresholds relative to it are not defined");
  const double enter = med + opt.enter_sd * sd, exit = med + opt.exit_sd * sd;

  // ── NODE detect_regions → the patches Selection (cached by its key) ──
  CacheKey pk;
  pk.pipeline = "patches compute";
  pk.params = Json{{"level", l}, {"sign", opt.sign}, {"enter_sd", opt.enter_sd}, {"exit_sd", opt.exit_sd}, {"min_area_m2", opt.min_area ? Json(*opt.min_area) : Json(nullptr)},
                   {"max_area_m2", opt.max_area ? Json(*opt.max_area) : Json(nullptr)}, {"weight", "barycentric"}, {"merge", "none"},
                   {"baseline", "median and robust SD (1.4826 MAD) across the level's tiles"}};
  pk.input = Json{{"band", band_id}, {"tiling", lvl.id}, {"map", frow->id}};
  pk.compute = NXR_COMPUTE_VERSION;
  std::int64_t n_patches = 0;
  std::string psel_id;
  Json patches_json = Json::array();
  std::vector<std::int32_t> pcode(static_cast<std::size_t>(nV), -1);
  if (const auto hit = subj.cached(task::Cls::selection, pk.json(), patches_path)) {
    res.hit = true;
    psel_id = hit->id;
    n_patches = subj.entry_count(hit->dictionary_id.value_or(std::string()));
    say(Json{{"stage", "detect_regions"}, {"selection", patches_name}, {"cache", "hit"}, {"patches", n_patches}});
  } else {
    Eigen::VectorXd w = Eigen::Map<const Eigen::VectorXd>(tile_area.data(), static_cast<Eigen::Index>(tile_area.size()));
    nxr::regions::Regions R;
    try {
      R = nxr::regions::detectRegions(signed_v, pairs, w, enter, exit, opt.min_area.value_or(0.0), opt.max_area.value_or(std::numeric_limits<double>::infinity()));
    } catch (const std::invalid_argument& e) { invalid(std::string("patches: ") + e.what()); }
    n_patches = static_cast<std::int64_t>(R.regions.size());
    for (std::int64_t v = 0; v < nV; ++v) { const std::int32_t t = tile[static_cast<std::size_t>(v)]; if (t >= 0) pcode[static_cast<std::size_t>(v)] = R.label[static_cast<std::size_t>(t)]; }
    const std::vector<int> centres = lad::tileCentres(S.vertices.data(), pcode, static_cast<int>(n_patches), vertex_area);
    Json pj = Json{{"by", "nxr-neuro patches compute"}, {"producer", producer}, {"pipeline", "patches"}, {"map", opt.map}, {"level", l}, {"tiling", tiling}, {"band", band_info}, {"band_field", band_name}, {"sign", opt.sign},
                   {"enter_sd", opt.enter_sd}, {"exit_sd", opt.exit_sd}, {"thresholds", Json{{"enter", enter}, {"exit", exit}, {"unit", "of the signed band-passed tile mean"}}},
                   {"min_area_m2", opt.min_area ? Json(*opt.min_area) : Json(nullptr)}, {"max_area_m2", opt.max_area ? Json(*opt.max_area) : Json(nullptr)},
                   {"baseline", baseline}, {"merge", "none"}, {"weight", "barycentric"}, {"graph", "tiles of the level joined by a mesh edge (stored adjacency), hemispheres never touch"}};
    pj["cache"] = stamp(pk)["cache"];
    bool replaced = false;
    clear_stored(ds, sub, {{"selection", patches_path}}, opt.replace, "the patches '" + patches_name + "' (found with other choices)", replaced);
    sub.compose([&] {
      DictionarySpec dsp;
      dsp.name = patches_name; dsp.scope = "subject";
      dsp.description = "the patches of '" + opt.map + "' at level " + std::to_string(l) + " (code = the patch's ordinal, by its lowest tile)";
      dsp.entries = std::vector<DictionaryEntry>();
      for (std::int64_t k = 0; k < n_patches; ++k) {
        const auto& g = R.regions[static_cast<std::size_t>(k)];
        DictionaryEntry e;
        e.code = k; e.ordinal = k; e.name = "patch " + std::to_string(k);
        e.cols.set("measurement", g.weight).set("measurement_unit", "m2")
            .json("attributes_json", Json{{"level", l}, {"tiles", g.nodes}, {"n_tiles", g.nodes.size()}, {"centre_vertex", centres[static_cast<std::size_t>(k)]}, {"peak_tile", g.peak}});
        dsp.entries->push_back(e);
      }
      const DictionaryResult dr = sub.dictionary(dsp);
      SelectionSpec ss;
      ss.name = patches_name; ss.path = patches_path; ss.type = "set"; ss.manifold_id = mid; ss.data = view(pcode, {nV});
      ss.cols.set("dictionary_id", dr.id).set("label", std::to_string(n_patches) + " patches").set("description", "the patches of '" + opt.map + "' (" + opt.sign + ") at level " + std::to_string(l) + " of " + surface + "'s tile ladder")
          .json("params_json", pj);
      psel_id = sub.selection(std::move(ss));
    });
    say(Json{{"stage", "detect_regions"}, {"selection", patches_name}, {"cache", "miss"}, {"patches", n_patches}, {"enter", enter}, {"exit", exit}});
    // the patches' own rows: tiles and centre (a count and a place; the mean, extremes and areas are measured below, by the partition)
    Json rows = Json::array();
    for (std::int64_t k = 0; k < n_patches; ++k) {
      const auto& g = R.regions[static_cast<std::size_t>(k)];
      rows.push_back(Json{{"selection_id", psel_id}, {"code", k}, {"of_field_id", nullptr}, {"measure", "n_tiles"}, {"value", static_cast<double>(g.nodes.size())}, {"weight", nullptr}, {"unit", nullptr}, {"computed_utc", now_utc()}, {"computed_by", by}});
      rows.push_back(Json{{"selection_id", psel_id}, {"code", k}, {"of_field_id", nullptr}, {"measure", "centre_vertex"}, {"value", static_cast<double>(centres[static_cast<std::size_t>(k)])}, {"weight", nullptr}, {"unit", nullptr}, {"computed_utc", now_utc()}, {"computed_by", by}});
    }
    if (!rows.empty()) subj.write_measurements(rows, by);
  }

  // ── NODE measure over the patches (the roll-up of the tiles is the measurement of their union) and summarize across the patches ──
  Json stats = Json::array();
  if (n_patches > 0) {
    MeasureComputeOptions po = mo;
    po.labelling = patches_name; po.replace = false;
    measure_compute(ds, sub, po);
    const std::string peak_measure = sign == nxr::regions::Sign::Positive ? "max" : sign == nxr::regions::Sign::Negative ? "min" : "abs_max";
    struct Want { std::string what, of, measure; };
    for (const Want& wnt : {Want{"area", "", "area"}, Want{"n_tiles", "", "n_tiles"}, Want{"mean", band_name, "mean"}, Want{"peak", band_name, peak_measure}}) {
      StatisticComputeOptions s;
      s.replace = opt.replace; s.over = patches_name; s.of = band_name; s.own = wnt.of.empty(); s.measure = wnt.measure; s.summaries = opt.summaries; s.combine = "each"; s.name = safe_name(opt.map + "_patches_L" + std::to_string(l) + "_" + opt.sign + "_" + wnt.what);
      const StatisticComputeResult r = statistic_compute(ds, sub, s);
      stats.push_back(Json{{"name", *s.name}, {"of", wnt.what == "area" || wnt.what == "n_tiles" ? "the patches' own measure" : band_name}, {"measure", wnt.measure}, {"cache", r.hit ? "hit" : "miss"}});
    }
    // READBACK: each patch as it is stored
    const auto own = [&](const std::string& m, const std::optional<std::string>& fid) { return measurement_values(sub, psel_id, fid, std::nullopt, std::nullopt, m).values; };
    const auto area = own("area", std::nullopt), ntl = own("n_tiles", std::nullopt), ctr = own("centre_vertex", std::nullopt), pm = own("mean", band_id), pp = own(peak_measure, band_id);
    for (std::int64_t k = 0; k < n_patches; ++k)
      patches_json.push_back(Json{{"patch", k}, {"area_m2", area[static_cast<std::size_t>(k)]}, {"n_tiles", static_cast<std::int64_t>(ntl[static_cast<std::size_t>(k)])}, {"centre_vertex", static_cast<std::int64_t>(ctr[static_cast<std::size_t>(k)])},
                                  {"mean", pm[static_cast<std::size_t>(k)]}, {"peak", pp[static_cast<std::size_t>(k)]}});
  }
  if (n_patches == 0 && opt.replace) {   // nothing found now: the statistics of the patches it replaced describe patches that are gone
    for (const char* what : {"area", "n_tiles", "mean", "peak"}) {
      const std::string sname = safe_name(opt.map + "_patches_L" + std::to_string(l) + "_" + opt.sign + "_" + what);
      if (const auto old = subj.find({.cls = task::Cls::statistic, .name = sname, .any_status = true})) {
        bool replaced = false;
        clear_stored(ds, sub, {{"statistic", old->path.value_or(std::string())}}, true, sname, replaced);
      }
    }
  }
  res.info = Json{{"map", opt.map}, {"surface", surface}, {"level", l}, {"tiling", tiling}, {"band_field", band_name}, {"band", band_info}, {"sign", opt.sign}, {"baseline", baseline},
                  {"thresholds", Json{{"enter", enter}, {"exit", exit}}}, {"selection", patches_name}, {"patches", n_patches}, {"patch_list", patches_json}, {"statistics", stats},
                  {"cache", res.hit ? "hit" : "miss"}, {"warnings", warnings}};
  (void)tiles_measured;
  return res;
}

}  // namespace nxr::pipelines
