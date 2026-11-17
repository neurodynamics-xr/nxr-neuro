#include "nxr/io/selection_store.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/pipelines/tiles.h"
#include "nxr/io/statistic.h"
#include "nxr/io/measures.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <set>

#include "nxr/chart.h"
#include "nxr/ladder.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "nxr/pipelines/meeg_internal.h"
#include "nxr/pipelines/run_shipped.h"

namespace nxr::pipelines {

using namespace nxr::io;
namespace fs = std::filesystem;
namespace lad = nxr::ladder;
using nxr::frames::Mat3;
using Eigen::VectorXd;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

std::string level_name(const std::string& surface, int l) { return surface + " tiles L" + std::to_string(l); }
std::string ladder_name(const std::string& surface) { return surface + " tiles"; }

std::vector<std::int32_t> read_i32(const nxr::io::Store& store, const std::string& key) {
  ArrayReader r = ArrayReader::open(store, key);
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<std::int32_t> out(raw.size() / 4);
  if (!out.empty()) std::memcpy(out.data(), raw.data(), out.size() * 4);
  return out;
}

struct Stored { Row ladder; std::vector<Row> levels; };
std::optional<Stored> stored_ladder(SubjectWriter& sub, const std::string& mid, const std::string& surface) {
  const auto l = nxr::io::selection::ladder_row(sub.db(), sub.id(), ladder_name(surface), mid);
  if (!l) return std::nullopt;
  Stored s;
  s.ladder = *l;
  s.levels = nxr::io::selection::member_rows(sub.db(), text(*l, "id"));
  return s;
}

void put_array(SubjectWriter& sub, const std::string& path, const ArrayView& v) { sub.write_array(path, v); }

}  // namespace

// THE TILE LADDER of a surface is a definition (`run tiles`, pipelines/definitions/tiles.yaml: the registered node `tile_grid` makes the dyadic grid with compute's builder, the runner stores it
// through io's save_grid); this is the library's entry to it for the callers that need the ladder first (the pyramid, the tile summaries, `dataset create`).
namespace {
Json params_of(const Row& r) {
  const Json p = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
  return p.is_object() ? p : Json::object();
}
// what a stored ladder says of itself, the shape `tiles_compute` has always answered
Json ladder_info(SubjectWriter& sub, const std::string& mid, const std::string& sname) {
  const std::optional<Stored> st = stored_ladder(sub, mid, sname);
  if (!st) invalid(sname + ": no ladder `" + ladder_name(sname) + "` is stored");
  const Json p = params_of(st->ladder);
  const int depth = p.value("depth", 0);
  const std::int64_t slots = p.contains("grid") && p["grid"].is_object() ? p["grid"].value("slots", std::int64_t(2)) : 2;
  return Json{{"surface", sname}, {"ladder", ladder_name(sname)}, {"levels", st->levels.size()}, {"depth", depth}, {"finest", p.value("finest", depth)}, {"pieces", p.value("pieces", 0)},
              {"tiles_at_depth", slots << depth}, {"area_m2", p.value("area_m2", 0.0)}, {"selections", Json::array({ladder_name(sname), level_name(sname, 0) + " ... " + level_name(sname, depth)})}, {"kept", false}};
}
}  // namespace

TilesComputeResult tiles_compute(DatasetWriter& ds, SubjectWriter& sub, const TilesComputeOptions& opt, const std::function<void(const Json&)>& progress) {
  if (opt.tree != "subject" && opt.tree != "group") invalid("tiles: tree must be subject or group, not " + opt.tree);
  if (opt.tree == "group") invalid("tiles: the group tree is made by `run group_tree` on the default subject (group_tree_compute)");
  const SurfaceInput S = read_surface(sub, opt.surface, true);   // float64 as stored: the Studio's stored ladder and nxr read it so
  TilesComputeResult res;
  if (const auto have = stored_ladder(sub, S.id, S.name); have && !opt.replace) {
    res.existed = true;
    res.info = Json{{"surface", S.name}, {"ladder", ladder_name(S.name)}, {"levels", have->levels.size()}, {"kept", true}};
    return res;
  }
  Json set = Json{{"surface", S.name}};
  if (opt.depth >= 0) set["depth"] = opt.depth;
  run_shipped(ds, sub, "tiles", set, opt.replace, progress);
  res.info = ladder_info(sub, S.id, S.name);
  return res;
}

Json group_tree_definitions(const GroupTreeOptions& o) {
  Json bands = Json::array();
  for (int b = 0; b < 6; ++b) bands.push_back(Json::array({1.0 * (1 << b), 2.0 * (1 << b)}));   // nxr's DEFAULT_BANDS: octaves 1-2 ... 32-64 Hz
  return Json{{"default", true}, {"tree", "group"}, {"depth", o.atlas_depth}, {"resolution", o.resolution}, {"frames", nullptr},
              {"frame_s", 86400.0 * std::pow(2.0, o.frame_level)}, {"frame_level", o.frame_level},
              {"time_levels", "frames are cycles of the day-anchored tower at frame_level (86 400·2^frame_level s); a level-m tile is tower level frame_level + m — a longer timescale is a coarser level"},
              {"bands_hz", bands}, {"lc_levels", o.lc_levels}, {"trajectory", Json{{"space_level", 4}, {"time_level", 2}}}};
}

TilesComputeResult group_tree_compute(DatasetWriter& ds, SubjectWriter& sub, const GroupTreeOptions& o, const std::function<void(const Json&)>& progress) {
  if (text(sub.row(), "kind") != "template") invalid(sub.name() + ": the group tree is the default subject's (a template), not a " + text(sub.row(), "kind") + "'s");
  const SurfaceInput S = read_surface(sub, std::nullopt, true);
  if (stored_ladder(sub, S.id, S.name)) invalid(sub.name() + ": its primary cortex already has a ladder");
  Json lc = Json::array();
  for (const int l : o.lc_levels) lc.push_back(l);
  run_shipped(ds, sub, "group_tree", Json{{"surface", S.name}, {"resolution", o.resolution}, {"atlas_depth", o.atlas_depth}, {"frame_level", o.frame_level}, {"lc_levels", lc}}, false, progress);
  TilesComputeResult r;
  r.info = ladder_info(sub, S.id, S.name);
  const std::optional<Stored> st = stored_ladder(sub, S.id, S.name);
  const Json p = params_of(st->ladder);
  r.info["frames"] = p.contains("atlas") && p["atlas"].is_object() ? p["atlas"].value("frames", Json()) : Json();
  r.info["levi_civita"] = p.contains("levi_civita") && p["levi_civita"].is_object() ? p["levi_civita"].value("levels", Json()) : Json();
  return r;
}

std::optional<Eigen::MatrixXd> stored_map(SubjectWriter& sub, const std::string& field_name, std::int64_t n, TileMapSource* source) {
  const auto f = find_field(sub.db(), sub.id(), FieldQuery{field_name, std::nullopt, std::nullopt, true});
  if (!f) {
    // a MAP OF A STATISTIC of that name (2026-11-13: the source maps are statistics whose band k reads as the Field it replaced): [n x components]
    for (const Row& st : nxr::io::statistics_of(sub.db(), sub.id())) {
      const Json axes = Json::parse(text(st, "axes_json"), nullptr, false);
      if (!axes.is_array() || axes.empty() || !axes[0].contains("manifold_id")) continue;
      for (const auto& ms : nxr::io::map_slices(sub.db(), sub.id(), axes[0]["manifold_id"].get<std::string>())) {
        if (ms.name != field_name || text(ms.statistic, "id") != text(st, "id") || ms.n != n) continue;
        if (source) { source->statistic_id = text(st, "id"); source->index = ms.index; source->name = ms.name; source->unit = ms.unit; }
        const std::vector<double> d = nxr::io::read_map_slice(sub, ms);
        Eigen::MatrixXd m(n, static_cast<Eigen::Index>(ms.n_components));
        for (std::int64_t i = 0; i < n; ++i) for (std::int64_t c = 0; c < ms.n_components; ++c) m(i, c) = d[static_cast<std::size_t>(i * ms.n_components + c)];
        return m;
      }
    }
    return std::nullopt;
  }
  if (source) { source->field_id = text(*f, "id"); source->name = text(*f, "name"); source->unit = opt_text(*f, "unit"); }
  ArrayReader r = ArrayReader::open(sub.store(), text(*f, "path"));
  const std::vector<std::uint8_t> raw = r.read();
  const std::size_t total = raw.size() / r.itemsize();
  if (n <= 0 || total % static_cast<std::size_t>(n) != 0) return std::nullopt;
  Eigen::MatrixXd m(n, static_cast<Eigen::Index>(total / static_cast<std::size_t>(n)));
  for (std::size_t i = 0; i < total; ++i) {
    double v;
    if (r.dtype() == DType::Float32) { float x; std::memcpy(&x, raw.data() + i * 4, 4); v = x; }
    else if (r.dtype() == DType::Float64) std::memcpy(&v, raw.data() + i * 8, 8);
    else if (r.dtype() == DType::Int32) { std::int32_t x; std::memcpy(&x, raw.data() + i * 4, 4); v = x; }
    else return std::nullopt;
    m(static_cast<Eigen::Index>(i / static_cast<std::size_t>(m.cols())), static_cast<Eigen::Index>(i % static_cast<std::size_t>(m.cols()))) = v;
  }
  return m;
}

Json tile_summaries_write(DatasetWriter& ds, SubjectWriter& sub, const TileMaps& maps, const TileSummaryOptions& opt, const std::function<void(const Json&)>& progress) {
  auto say = [&](const Json& j) { if (progress) progress(j); };
  if (maps.stem.empty()) invalid("tiles: the maps need a stem");
  const SurfaceInput S = read_surface(sub, opt.surface, true);   // float64 as stored: the Studio's stored ladder and nxr read it so
  const int nV = static_cast<int>(S.n_vertices()), nF = static_cast<int>(S.n_faces());
  const std::string sname = S.name, mid = S.id;
  std::optional<Stored> st = stored_ladder(sub, mid, sname);
  if (!st) {
    TilesComputeOptions to; to.surface = sname;
    tiles_compute(ds, sub, to, progress);
    st = stored_ladder(sub, mid, sname);
  }
  const int depth = static_cast<int>(st->levels.size()) - 1;
  const Json top = Json::parse(opt.surface ? "{}" : "{}");
  (void)top;
  const std::vector<std::int32_t> deepest = read_i32(sub.store(), text(st->levels.back(), "array_path"));
  if (static_cast<int>(deepest.size()) != nV) invalid(sname + ": the stored tiles label " + std::to_string(deepest.size()) + " vertices, the surface has " + std::to_string(nV));
  std::vector<int> levels = opt.levels;
  if (levels.empty()) for (int l = 1; l <= std::min(depth, opt.max_level); ++l) levels.push_back(l);
  for (int l : levels) if (l < 0 || l > depth) invalid("tiles: level " + std::to_string(l) + " is outside the stored ladder 0.." + std::to_string(depth));

  lad::Mesh whole; whole.n = nV; whole.positions = S.vertices; whole.faces = S.faces;
  const std::vector<double> area = lad::vertexAreas(whole);
  const Mat3 normals = nxr::frames::vertexNormals(S.vertices.data(), nV, S.faces.data(), nF);
  StoredFrames sf = read_stored_frames(sub, sname, nV);
  const bool have_vectors = !maps.vectors.empty();
  std::optional<nxr::chart::ChartKit> kit;
  Mat3 e1, e2;
  std::string frame = "canonical";                                         // the frame the tiles' vectors are reported in (G6: "frame", never "gauge")
  Json gauge_rec;                                                          // the gauge record of the Levi-Civita form (G6)
  VectorXd gauge_angle;
  std::vector<char> singular;
  if (have_vectors) {
    kit.emplace(S.vertices.data(), nV, S.faces.data(), nF);
    if (sf.canonical) { e1 = sf.e1; e2 = sf.e2; }
    else {
      const nxr::chart::Basis b = kit->basis();
      e1.resize(nV, 3); e2.resize(nV, 3);
      for (int i = 0; i < nV; ++i) for (int k = 0; k < 3; ++k) { e1(i, k) = b.e1[static_cast<size_t>(3 * i + k)]; e2(i, k) = b.e2[static_cast<size_t>(3 * i + k)]; }
      frame = "gauge";
    }
    gauge_rec = gauge_block(*kit, nV, S.id);
    singular.assign(static_cast<size_t>(nV), 0);
    for (int i = 0; i < nV; ++i) singular[static_cast<size_t>(i)] = !(std::isfinite(e1(i, 0)) && std::isfinite(e1(i, 1)) && std::isfinite(e1(i, 2)));
    gauge_angle = nxr::frames::gaugeAngle(*kit, e1);
  }
  for (const auto& m : maps.scalars) if (m.second.size() != nV) invalid("tiles: map " + m.first + " has " + std::to_string(m.second.size()) + " values for " + std::to_string(nV) + " vertices");
  for (const auto& m : maps.vectors) if (m.second.rows() != nV) invalid("tiles: map " + m.first + " has " + std::to_string(m.second.rows()) + " rows for " + std::to_string(nV) + " vertices");

  Json written = Json::array();
  const Json ladder_params = Json::parse(opt_text(st->ladder, "params_json").value_or("{}"), nullptr, false).is_object() ? Json::parse(opt_text(st->ladder, "params_json").value_or("{}"), nullptr, false) : Json::object();
  sub.compose([&] {
    for (int l : levels) {
      const Row lrow = st->levels[static_cast<size_t>(l)];
      Json p = Json::parse(opt_text(lrow, "params_json").value_or("{}"), nullptr, false);
      const std::string dir = p.value("node", std::string());
      if (dir.empty()) invalid(level_name(sname, l) + ": no atlas node recorded");
      std::vector<std::int32_t> node(deepest.size());
      for (size_t i = 0; i < deepest.size(); ++i) node[i] = deepest[i] >= 0 ? (deepest[i] >> (depth - l)) : -1;
      const int nodes = static_cast<int>(st->levels.size()) > 0 ? p.value("tiles", 1 << l) : 1;
      const std::string base = dir + "/summary/" + maps.stem;
      if (opt.replace) sub.store().erase_prefix(base);
      else if (sub.store().exists_prefix(base)) invalid(level_name(sname, l) + ": summaries of '" + maps.stem + "' are stored (use --replace)");
      Json records = p.contains("summaries") && p["summaries"].is_array() ? p["summaries"] : Json::array();
      if (opt.replace) { Json keep = Json::array(); for (const Json& r : records) if (r.value("stem", std::string()) != maps.stem) keep.push_back(r); records = keep; }
      auto vec = [&](const std::string& path, const VectorXd& v) { put_array(sub, path, view(v.data(), {static_cast<std::int64_t>(v.size())})); };
      // the level's `atlas` block (nxr's: what the lookup reads to roll the rows up the ladder)
      if (!p.contains("atlas") || !p["atlas"].is_object())
        p["atlas"] = Json{{"tree", ladder_params.value("tree", std::string("subject"))}, {"level", l}, {"finest", ladder_params.value("finest", depth)}, {"frame", nullptr}, {"areas", p.value("areas", std::string())}, {"node", dir}};
      const std::string selection_id = text(lrow, "id");
      for (const auto& [mname, x] : maps.scalars) {
        const auto src = maps.sources.find(mname);
        if (src == maps.sources.end() || src->second.field_id.empty() == src->second.statistic_id.empty())
          invalid("tiles: the scalar map " + mname + " needs its source (TileMapSource): the Field, or the statistic slice, it is a map of");
        // A MAP OF A STATISTIC's slice has no Field of its own: it gets a BYTE-LESS Field (`<slice> tile map`, the pattern of the source estimate and phasegradient: no array, rows only), so its rows are
        // array-backed rows of a Field (the schema: array_path needs of_field_id) and are read like any map's; the rows also name the statistic and the slice (the Field's producer_json)
        TileMapSource so = src->second;
        if (so.field_id.empty()) {
          const std::string bname = (so.name.empty() ? maps.stem + "_" + mname : so.name) + " tile map";
          const auto old = find_field(sub.db(), sub.id(), FieldQuery{bname, std::nullopt, std::nullopt, false, false});
          if (old) so.field_id = text(*old, "id");
          else {
            FieldSpec fs;
            fs.name = bname; fs.kind = "tile map"; fs.manifold_id = text(lrow, "manifold_id"); fs.value_type = "scalar";
            fs.cols.set("function", "tile map").set("description", "byte-less: the map of slice " + std::to_string(so.index) + " of a statistic summarised over tiles; its summaries are the array-backed measurement rows of this Field");
            fs.cols.json("producer_json", Json{{"of_statistic_id", so.statistic_id}, {"of_index", so.index}, {"map", so.name}});
            if (so.unit) fs.cols.set("unit", *so.unit);
            so.field_id = sub.field(std::move(fs));
          }
        }
        const std::string mapname = so.name.empty() ? maps.stem + "_" + mname : so.name;
        const lad::ScalarSummary s = lad::scalarSummary(node, nodes, area, x.data());
        const std::string b = dir + "/maps/" + nxr::io::safe_name(mapname);
        // the rows of this map on this level: the measures nxr's atlas stores (ATLAS_MEASURES): w weight, s1 sum, s2 sum_square, n count, min, max
        const std::vector<std::pair<const char*, const char*>> parts = {{"w", "weight"}, {"s1", "sum"}, {"s2", "sum_square"}, {"n", "count"}, {"min", "min"}, {"max", "max"}};
        Json where = Json{{"selectionId", selection_id}};
        if (!so.field_id.empty()) where["ofFieldId"] = so.field_id; else { where["ofStatisticId"] = so.statistic_id; if (so.index >= 0) where["ofIndex"] = so.index; }
        nxr::io::selection::MeasurementFilter weight_rows;
        weight_rows.measure = "weight";
        if (so.field_id.empty()) { weight_rows.of_statistic_id = so.statistic_id; if (so.index >= 0) weight_rows.of_index = so.index; } else weight_rows.of_field_id = so.field_id;
        const bool have = nxr::io::selection::has_measurement(sub.db(), selection_id, weight_rows);
        if (have) {
          if (!opt.replace) invalid(level_name(sname, l) + ": summaries of the map '" + mapname + "' are stored (use --replace)");
          for (const auto& pt : parts) { Json w2 = where; w2["measure"] = pt.second; nxr::io::delete_measurements(sub.db(), w2); }
          sub.store().erase_prefix(b);
        }
        vec(b + "/w", s.w); vec(b + "/s1", s.s1); vec(b + "/s2", s.s2); vec(b + "/n", s.n); vec(b + "/min", s.min); vec(b + "/max", s.max);
        Json rows = Json::array();
        for (const auto& pt : parts) {
          Json r{{"selection_id", selection_id}, {"measure", pt.second}, {"array_path", b + "/" + pt.first}, {"computed_by", "nxr-neuro tile summaries"}};
          if (!so.field_id.empty()) r["of_field_id"] = so.field_id; else { r["of_statistic_id"] = so.statistic_id; if (so.index >= 0) r["of_index"] = so.index; }
          if (so.unit && std::string(pt.second) != "weight" && std::string(pt.second) != "count") r["unit"] = *so.unit;
          rows.push_back(r);
        }
        nxr::io::write_measurements(sub.db(), rows, "nxr-neuro tile summaries");
        records.push_back(Json{{"stem", maps.stem}, {"map", mname}, {"kind", "scalar"}, {"reduction", "area-weighted"}, {"path", b}, {"arrays", Json::array({"w", "s1", "s2", "n", "min", "max"})},
                               {"measures", Json::array({"weight", "sum", "sum_square", "count", "min", "max"})}, {"of", so.statistic_id.empty() ? Json{{"field", so.field_id}} : Json{{"field", so.field_id}, {"statistic", so.statistic_id}, {"index", so.index}}},
                               {"derived_at_read", "mean = sum / weight, mean_square, variance (measure query)"}});
      }
      if (!maps.vectors.empty()) {
        const std::vector<int> cen = lad::tileCentres(S.vertices.data(), node, nodes, area, singular);
        lad::LeviCivita lc;
        if (opt.levi_civita) {
          lc = lad::leviCivita(*kit, e1, e2, gauge_angle, node, nodes, cen);
          const std::string lb = dir + "/lc/L" + std::to_string(l) + "/";
          if (opt.replace) { for (const char* k : {"rho", "omega", "centre_vector"}) sub.store().erase_prefix(lb + k); }
          if (!sub.store().exists_prefix(lb + "rho")) {
            vec(lb + "rho", lc.rho);
            Eigen::MatrixXd om = lad::antisymmetric(lc.omega);     // antisymmetrised as nxr's atlas stores `lc/L*/omega`; row-major [nodes x nodes]
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> omr = om;
            put_array(sub, lb + "omega", view(omr.data(), {static_cast<std::int64_t>(nodes), static_cast<std::int64_t>(nodes)}));
          }
        }
        for (const auto& [mname, v] : maps.vectors) {
          const std::string b = base + "/" + mname;
          Json rec = Json{{"stem", maps.stem}, {"map", mname}, {"kind", "vector"}, {"frame", frame}, {"centres", dir + "/lc/L" + std::to_string(l) + "/centre"}};
          Json forms = Json::array();
          if (frame == "canonical") {
            const lad::FrameSummary f = lad::frameSummary(node, nodes, area, e1, e2, normals, v);
            const std::string fb = b + "/frame";
            vec(fb + "/w", f.w); vec(fb + "/c1", f.c1); vec(fb + "/c2", f.c2); vec(fb + "/cn", f.cn); vec(fb + "/c11", f.c11); vec(fb + "/c22", f.c22); vec(fb + "/c12", f.c12); vec(fb + "/mag", f.mag);
            forms.push_back(Json{{"reduction", "frame"}, {"frame", "canonical"}, {"path", fb}, {"arrays", Json::array({"w", "c1", "c2", "cn", "c11", "c22", "c12", "mag"})}, {"merge", "sum"}});
          }
          if (opt.levi_civita) {
            VectorXd w, mg;
            const Eigen::VectorXcd z = lad::lcResultant(node, nodes, area, e1, e2, lc.rho, v, &w, &mg);
            VectorXd re = z.real(), im = z.imag(), len(nodes);
            for (int k = 0; k < nodes; ++k) len[k] = mg[k] > 0 ? std::abs(z[k]) / mg[k] : std::numeric_limits<double>::quiet_NaN();
            const std::string zb = b + "/levi_civita";
            vec(zb + "/re", re); vec(zb + "/im", im); vec(zb + "/w", w); vec(zb + "/length", len);
            forms.push_back(Json{{"reduction", "levi-civita"}, {"frame", frame}, {"gauge", gauge_rec}, {"spin", 1}, {"path", zb}, {"arrays", Json::array({"re", "im", "w", "length"})}, {"transport", "ChartKit::transported (vector heat) of each tile centre's e1, calibrated to the identity at its source"}, {"merge", "per level, never rolled up"}});
          }
          rec["forms"] = forms;
          records.push_back(rec);
        }
      }
      p["summaries"] = records;
      nxr::io::selection::set_params(sub.db(), text(lrow, "id"), p);
      Json one = Json{{"level", l}, {"tiles", nodes}, {"selection", level_name(sname, l)}, {"maps", maps.scalars.size() + maps.vectors.size()}};
      written.push_back(one);
      say(Json{{"stage", "tiles"}, {"stem", maps.stem}, {"level", l}, {"tiles", nodes}});
    }
  });
  return Json{{"surface", sname}, {"stem", maps.stem}, {"levels", written}, {"frame", maps.vectors.empty() ? Json() : Json(frame)}, {"gauge", maps.vectors.empty() ? Json() : gauge_rec},
              {"scalars", [&] { Json a = Json::array(); for (const auto& m : maps.scalars) a.push_back(m.first); return a; }()}, {"vectors", [&] { Json a = Json::array(); for (const auto& m : maps.vectors) a.push_back(m.first); return a; }()},
              {"levi_civita", opt.levi_civita}};
}

}  // namespace nxr::pipelines
