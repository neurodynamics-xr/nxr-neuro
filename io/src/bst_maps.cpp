// nxr::bst — nxr.convert.results_map: static source maps and region maps.
#include "nxr/bst/maps.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "bst_pyvalue.h"
#include "nxr/io/layout.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/surface.h"

namespace nxr::bst {

namespace code = nxr::io::code;
namespace fs = std::filesystem;
using nxr::io::Row;
using nxr::io::SubjectWriter;
using nxr::io::view;
using namespace detail;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

std::string path_str(const fs::path& p) { return py_path(p.string()); }

std::string removeprefix(const std::string& s, const std::string& p) { return starts_with(s, p) ? s.substr(p.size()) : s; }

// "SUVR" when the map's name or Comment says so
std::optional<std::string> suvr_unit(const std::string& stem, const std::string& comment) {
  return lower(stem + " " + comment).find("suvr") != std::string::npos ? std::optional<std::string>("SUVR") : std::nullopt;
}

// What made a map, for its producer_json: the file's `Provenance` (JSON text, written by the pipeline that made the map), overlaid by `extra` (`map --provenance FILE`),
// and Brainstorm's own `History` rows ("date | type | text").
Json provenance_of(const std::map<std::string, Value>& mat, const std::optional<Json>& extra) {
  Json out = Json::object();
  const std::string txt = text_field(get(mat, "Provenance"));
  if (!txt.empty()) {
    Json j = Json::parse(txt, nullptr, false);
    out["provenance"] = j.is_discarded() ? Json{{"text", txt}} : j;
  }
  if (extra) {
    Json merged = out.contains("provenance") && out["provenance"].is_object() ? out["provenance"] : Json::object();
    for (auto it = extra->begin(); it != extra->end(); ++it) merged[it.key()] = it.value();
    out["provenance"] = merged;
  }
  const Value* h = get(mat, "History");
  if (h && h->numel()) {
    Json rows = Json::array();
    if (h->is_cell()) {
      const std::size_t n = h->rows(), c = h->cols();
      for (std::size_t r = 0; r < n; ++r) {
        std::string row;
        for (std::size_t j = 0; j < c; ++j) row += (j ? " | " : "") + text_field(&h->cell(r + j * n));
        rows.push_back(row);
      }
    } else {
      rows.push_back(text_field(h));
    }
    out["history"] = rows;
  }
  return out;
}

// One value per row: one column, or Brainstorm's two identical ones (`Time = [0 1]`); anything else refused.
std::vector<double> static_column(const Value* values, const std::string& name) {
  const Mat a = as_mat(values);
  std::size_t cols = a.ndim >= 2 ? a.cols : 1;
  bool same = cols == 2;
  if (same) for (std::size_t r = 0; r < a.rows; ++r) {
    const double x = a.at(r, 0), y = a.at(r, 1);
    if (!(x == y || (std::isnan(x) && std::isnan(y)))) { same = false; break; }
  }
  if (same) cols = 1;
  if (cols != 1) value_error(name + ": a map over " + std::to_string(a.cols) + " time points needs a time Line \xe2\x80\x94 only static maps (one value per row) are converted");
  std::vector<double> out(a.rows);
  for (std::size_t r = 0; r < a.rows; ++r) out[r] = a.ndim >= 2 ? a.at(r, 0) : a.d[r];
  return out;
}

// the regular sampling of a Time vector: (t0, step)
std::pair<double, double> regular(const std::vector<double>& t, const std::string& what) {
  bool ok = t.size() >= 2 && t[1] - t[0] > 0;
  if (ok) {
    const double d0 = t[1] - t[0];
    for (std::size_t i = 1; i < t.size(); ++i) if (std::fabs((t[i] - t[i - 1]) - d0) > 1e-6 * std::fabs(d0)) { ok = false; break; }
  }
  if (!ok) value_error(what + ": its Time is not a regular sampling \xe2\x80\x94 a time Line needs one");
  return {t[0], t[1] - t[0]};
}

struct SurfaceRef { Row row; std::string spath; fs::path file; };
// The subject's manifold for a Brainstorm `SurfaceFile` (exported beside the primary if absent)
SurfaceRef surface_of(SubjectWriter& sub, const std::string& surface_rel, const fs::path& bst_root, const std::string& what) {
  if (surface_rel.empty()) value_error(what + ": no SurfaceFile \xe2\x80\x94 a map with no surface has nowhere to live");
  const fs::path surface_path = bst_root / "anat" / surface_rel;
  const std::string sname = surface_node_name(surface_path.filename());
  const std::string spath = nxr::io::folder_for("surface", sname) + "/" + sname;
  auto surf = sub.find("manifold", spath);
  if (!surf) {
    std::error_code ec;
    if (!fs::is_regular_file(surface_path, ec)) throw nxr::io::Error(code::not_found, what + ": its surface " + surface_rel + " is not in the protocol");
    export_surface(sub, surface_path, path_str(bst_root), false);
    surf = sub.find("manifold", spath);
    if (!surf) throw nxr::io::Error(code::no_such_row, "'" + spath + "'");
  }
  return {*surf, spath, surface_path};
}

// the time Line of the acquisition's series whose name ends with `share` at the same sampling, or nullopt
std::optional<std::string> shared_line(SubjectWriter& sub, const std::vector<double>& t, const std::optional<std::string>& acquisition, const std::string& share,
                                       const std::string& what) {
  const auto [t0, step] = regular(t, what);
  if (share.empty()) return std::nullopt;
  std::string pat;
  for (char c : share) { if (c == '\\' || c == '%' || c == '_') pat += '\\'; pat += c; }
  const auto row = sub.db().one(
      "SELECT m.id FROM manifold m JOIN geometry_dimension d ON d.geometry_id = m.geometry_id WHERE m.subject_id = ? AND m.type = 'line' "
      "AND m.acquisition_id IS (SELECT a.id FROM acquisition a WHERE a.subject_id = m.subject_id AND a.label = ?) AND d.n_vertices = ? AND abs(d.spacing - ?) <= 1e-9 * ? AND abs(coalesce(d.origin, 0) - ?) <= 1e-9 * ? "
      "AND m.path LIKE ? AND m.path LIKE ? ESCAPE '\\'",
      {sub.id(), nxr::io::V(acquisition), static_cast<std::int64_t>(t.size()), step, step, t0, step, nxr::io::folder_for("line", "", std::string("time")) + "/%", "%" + pat + "\\_time"});
  if (!row) return std::nullopt;
  return nxr::io::text(*row, "id");
}


}  // namespace

bool is_source_map(const std::map<std::string, Value>& mat) {
  const std::string dt = text_field(get(mat, "DataType"));
  const Value* amp = get(mat, "ImageGridAmp");
  const Value* ik = get(mat, "ImagingKernel");
  return (dt.empty() || dt == "results") && amp && amp->numel() > 0 && !(ik && ik->numel() > 0);
}

std::vector<fs::path> find_source_maps(const fs::path& cond_dir) {
  std::vector<fs::path> out;
  std::error_code ec;
  if (!fs::is_directory(cond_dir, ec)) return out;
  for (const std::string& n : glob_names(cond_dir, "results_", ".mat"))
    if (n.find("KERNEL") == std::string::npos && !starts_with(n, "._")) out.push_back(cond_dir / n);
  return out;
}

Json export_source_map(SubjectWriter& sub, const fs::path& p, const fs::path& bst_root, const std::optional<std::string>& acquisition, bool replace,
                       const std::optional<Json>& provenance, const MapHooks* hooks) {
  const std::string fname = p.filename().string();
  const std::map<std::string, Value> m = read_all_variables(p);      // a time-resolved map is loaded whole
  if (!is_source_map(m)) value_error(fname + ": not a source map (needs a non-empty ImageGridAmp and no ImagingKernel)");
  const Value* tv = get(m, "Time");
  const std::vector<double> t = tv && tv->is_numeric() ? tv->doubles() : std::vector<double>();
  const Mat amp = as_mat(get(m, "ImageGridAmp"));
  const bool timed = t.size() > 2;
  std::vector<double> column;
  if (!timed) column = static_column(get(m, "ImageGridAmp"), fname);
  const std::string surface_rel = text_field(get(m, "SurfaceFile"));
  const SurfaceRef sr = surface_of(sub, surface_rel, bst_root, fname);
  const std::string sname = sr.spath.substr(sr.spath.rfind('/') + 1);
  const std::int64_t nv = nxr::io::integer(sr.row, "n_vertices");
  if (nv != static_cast<std::int64_t>(amp.rows)) value_error(fname + ": " + std::to_string(amp.rows) + " values for a " + std::to_string(nv) + "-vertex surface (" + sname + ")");
  const std::string comment = text_field(get(m, "Comment")).empty() ? p.stem().string() : text_field(get(m, "Comment"));
  const std::string stem = nxr::io::sanitize_node_name(removeprefix(p.stem().string(), "results_"));
  if (const auto have = sub.db().one("SELECT * FROM field WHERE subject_id = ? AND name = ?", {sub.id(), stem})) {
    if (!replace) value_error(fname + ": the subject already holds " + nxr::io::text(*have, "path") + " (--replace)");
    sub.remove("field", nxr::io::text(*have, "id"));
  }
  const auto unit = suvr_unit(stem, comment);
  Json producer = Json{{"surface", surface_rel}};
  if (!text_field(get(m, "DisplayUnits")).empty()) producer["display_units"] = text_field(get(m, "DisplayUnits"));
  const Json prov = provenance_of(m, provenance);
  for (auto it = prov.begin(); it != prov.end(); ++it) producer[it.key()] = it.value();
  if (timed) {      // a BOLD series: on the surface x its own time Line, stored as a recording is
    const std::size_t ncols = amp.ndim >= 2 ? amp.cols : 1;
    if (ncols != t.size()) value_error(fname + ": " + std::to_string(ncols) + " columns for " + std::to_string(t.size()) + " time points");
    if (!hooks || !hooks->series)
      throw nxr::io::Error(code::unsupported, fname + ": a map over time is written by the recording writer of the session port (MapHooks::series), which is not ported yet");
    const auto [t0, step] = regular(t, fname);
    SeriesRequest rq;
    rq.stem = stem;
    rq.x.assign(amp.d.begin(), amp.d.end());
    rq.rows = static_cast<std::int64_t>(amp.rows);
    rq.samples = static_cast<std::int64_t>(t.size());
    rq.t0 = t0;
    rq.step = step;
    rq.rows_id = nxr::io::text(sr.row, "id");
    rq.acquisition = acquisition;
    rq.unit = unit;
    rq.comment = comment;
    rq.source = fname;
    rq.description = comment + " \xe2\x80\x94 one series per vertex of " + sname;
    rq.producer = producer;
    const std::string id = hooks->series(sub, rq);
    return Json{{"name", stem}, {"id", id}, {"surface", sname}, {"n_vertices", static_cast<std::int64_t>(amp.rows)}, {"kind", comment},
                {"n_samples", static_cast<std::int64_t>(t.size())}, {"spacing_s", step}};
  }
  std::vector<float> data(column.begin(), column.end());
  nxr::io::FieldSpec f;
  f.name = stem;
  f.path = nxr::io::node_path(sub.db(), nxr::io::text(sr.row, "id"), stem);
  f.kind = comment;
  f.manifold_id = nxr::io::text(sr.row, "id");
  f.data = view(data, {static_cast<std::int64_t>(data.size())});
  f.acquisition = acquisition;
  f.cols.set("description", comment + " \xe2\x80\x94 one value per vertex of " + sname).set("comment", comment).set("source", fname).set("unit", unit).json("producer_json", producer);
  const std::string fid = sub.field(std::move(f));
  return Json{{"name", stem}, {"id", fid}, {"surface", sname}, {"n_vertices", static_cast<std::int64_t>(amp.rows)}, {"kind", comment}};
}

bool is_matrix_series(const fs::path& path) {
  const auto v = read_variables(path, {"Time"});
  const Value* t = get(v, "Time");
  return t && t->numel() > 2 && !is_region_map(path);
}

Json export_matrix_series(SubjectWriter& sub, const fs::path& p, const std::optional<std::string>& acquisition, const MapHooks* hooks) {
  const std::string fname = p.filename().string();
  const std::map<std::string, Value> m = read_all_variables(p);
  const Mat v = as_mat(get(m, "Value"));
  const Value* tv = get(m, "Time");
  const std::vector<double> t = tv && tv->is_numeric() ? tv->doubles() : std::vector<double>();
  std::vector<std::string> names;
  for (const std::string& d : strings_of(get(m, "Description"))) names.push_back(strip(d));
  if (v.ndim != 2 || v.rows != names.size() || v.cols != t.size()) {
    std::string shape = v.ndim == 2 ? "(" + std::to_string(v.rows) + ", " + std::to_string(v.cols) + ")" : v.ndim == 1 ? "(" + std::to_string(v.rows) + ",)" : "()";
    value_error(fname + ": Value " + shape + " for " + std::to_string(names.size()) + " names \xc3\x97 " + std::to_string(t.size()) + " time points");
  }
  const std::string comment = text_field(get(m, "Comment")).empty() ? p.stem().string() : text_field(get(m, "Comment"));
  const std::string stem = nxr::io::sanitize_node_name(removeprefix(p.stem().string(), "matrix_"));
  std::vector<std::string> fields;
  for (const auto& nm : names) fields.push_back(nxr::io::sanitize_node_name(stem + "_" + nm));
  {
    std::set<std::string> uniq(fields.begin(), fields.end());
    if (uniq.size() != fields.size()) {
      std::string list = "[";
      for (std::size_t i = 0; i < names.size(); ++i) list += (i ? ", '" : "'") + names[i] + "'";
      value_error(fname + ": row names " + list + "] do not make distinct field names");
    }
  }
  std::vector<std::string> taken;
  for (const auto& f : fields) if (sub.db().one("SELECT 1 FROM field WHERE subject_id = ? AND name = ?", {sub.id(), f})) taken.push_back(f);
  if (!taken.empty()) {
    std::string list;
    for (std::size_t i = 0; i < taken.size() && i < 3; ++i) list += (i ? ", " : "") + taken[i];
    value_error(fname + ": the subject already holds " + list);
  }
  const auto [t0, step] = regular(t, fname);
  const auto aid = sub.acquisition(acquisition);
  const auto us = stem.find('_');
  std::optional<std::string> line = shared_line(sub, t, acquisition, us == std::string::npos ? stem : stem.substr(us + 1), fname);
  if (!line) {
    if (!hooks || !hooks->time_line)
      throw nxr::io::Error(code::unsupported, fname + ": a matrix over time needs the time Line writer of the session port (MapHooks::time_line), which is not ported yet");
    line = hooks->time_line(sub, TimeLineRequest{nxr::io::folder_for("line", "", std::string("time")) + "/" + stem + "_time", aid, static_cast<std::int64_t>(t.size()), 1.0 / step, t0});
  }
  for (std::size_t k = 0; k < names.size(); ++k) {
    std::vector<float> row(t.size());
    for (std::size_t j = 0; j < t.size(); ++j) row[j] = static_cast<float>(v.at(k, j));
    nxr::io::FieldSpec f;
    f.name = fields[k];
    f.path = nxr::io::node_path(sub.db(), *line, fields[k]);
    f.kind = comment;
    f.manifold_id = *line;
    f.data = view(row, {static_cast<std::int64_t>(row.size())});
    f.acquisition = acquisition;
    f.cols.set("source", fname).set("comment", comment).set("description", comment + " \xe2\x80\x94 " + names[k] + " over time")
        .json("producer_json", Json{{"confound", names[k]}, {"row", static_cast<std::int64_t>(k)}, {"matrix", fname}});
    sub.field(std::move(f));
  }
  return Json{{"name", stem}, {"id", nullptr}, {"fields", fields}, {"rows", static_cast<std::int64_t>(names.size())}, {"kind", comment},
              {"n_samples", static_cast<std::int64_t>(t.size())}, {"spacing_s", step}};
}

std::optional<std::tuple<std::vector<std::string>, std::vector<std::int64_t>, std::string>> atlas_regions(SubjectWriter& sub, const std::string& cortex_path, const fs::path& cortex_file,
                                                                                                          const std::string& atlas) {
  const std::string fam = nxr::io::sanitize_node_name(atlas);
  const auto atlases = sub.find("selection", cortex_path + "_atlases");
  std::optional<nxr::io::records::SelectionRow> member;
  if (atlases) member = nxr::io::selection::find_member(sub.db(), nxr::io::text(*atlases, "id"), fam);
  const auto vars = read_variables(cortex_file, {"Atlas"});
  const Value* at = get(vars, "Atlas");
  const Value* entry = nullptr;
  std::size_t entry_i = 0;
  if (at && at->is_struct() && at->has_field("Name"))
    for (std::size_t e = 0; e < at->numel() && !entry; ++e)
      if (nxr::io::sanitize_node_name(py_str(at->field("Name", e))) == fam) { entry = at; entry_i = e; }
  if (!member || !entry) return std::nullopt;
  std::vector<std::string> levels = {"unassigned"};
  const Value* scouts = entry->field("Scouts", entry_i);
  if (scouts && scouts->is_struct() && scouts->has_field("Label")) for (std::size_t s = 0; s < scouts->numel(); ++s) levels.push_back(py_str(scouts->field("Label", s)));
  std::map<std::string, std::int64_t> code_of_name;
  if (member->dictionary_id)
    for (const auto& e : nxr::io::selection::load_lookup(sub.db(), *member->dictionary_id)) code_of_name[e.name] = e.code;
  std::vector<std::int64_t> code_of;
  for (const auto& nm : levels) {
    const auto it = code_of_name.find(nm);
    if (it == code_of_name.end()) throw nxr::io::Error(code::no_such_row, "'" + nm + "'");
    code_of.push_back(it->second);
  }
  return std::make_tuple(levels, code_of, member->id);
}

int region_measurements(SubjectWriter& sub, const std::string& selection_id, const std::vector<RegionValue>& rows, const std::string& measure, const std::optional<std::string>& unit,
                        const std::string& computed_by) {
  // the rows name regions by their atlas label; a member is kept by its number, so the label is looked up in the members' labels (decoding)
  std::map<std::int64_t, std::int64_t> number;
  {
    const std::vector<std::int64_t> labels = nxr::io::selection::member_labels(sub.db(), sub.store(), selection_id);
    for (std::size_t m = 0; m < labels.size(); ++m) number.emplace(labels[m], static_cast<std::int64_t>(m));
  }
  const auto member_of = [&](std::int64_t label) {
    const auto it = number.find(label);
    if (it == number.end()) throw nxr::io::Error(code::no_such_row, "region label " + std::to_string(label) + " is no member of the atlas");
    return it->second;
  };
  std::vector<nxr::io::selection::MemberValue> values;
  for (const RegionValue& r : rows) values.push_back({member_of(r.code), r.value, r.within_code ? std::optional<std::int64_t>(member_of(*r.within_code)) : std::nullopt});
  // a region's value within a region's code: the band key is the atlas itself (within_selection_id = the selection)
  return static_cast<int>(nxr::io::selection::save_member_values(sub, selection_id, measure, values, unit, computed_by, selection_id));
}

bool is_region_map(const fs::path& path) {
  const auto v = read_variables(path, {"Atlas", "SurfaceFile"});
  const Value* at = get(v, "Atlas");
  return !text_field(get(v, "SurfaceFile")).empty() && at && at->is_struct() && at->has_field("Name") && at->numel() > 0;
}

Json export_region_map(SubjectWriter& sub, const fs::path& p, const fs::path& bst_root, const std::optional<std::string>& /*acquisition: a region map's measurements carry none*/) {
  const std::string fname = p.filename().string();
  const std::map<std::string, Value> m = read_all_variables(p);
  const Value* at = get(m, "Atlas");
  if (!at || !at->is_struct() || !at->has_field("Name") || at->numel() == 0) value_error(fname + ": no Atlas \xe2\x80\x94 a region map names the atlas its rows are regions of");
  const std::vector<double> values = static_column(get(m, "Value"), fname);
  std::vector<std::string> labels;
  for (const std::string& d : strings_of(get(m, "Description"))) {
    const auto cut = d.find(" @ ");
    labels.push_back(strip(cut == std::string::npos ? d : d.substr(0, cut)));
  }
  if (labels.size() != values.size()) value_error(fname + ": " + std::to_string(values.size()) + " values for " + std::to_string(labels.size()) + " region labels");
  const SurfaceRef sr = surface_of(sub, text_field(get(m, "SurfaceFile")), bst_root, fname);
  const std::string aname = py_str(at->field("Name", 0));
  const auto regions = atlas_regions(sub, sr.spath, sr.file, aname);
  if (!regions) value_error(fname + ": no atlas '" + aname + "' on " + sr.spath);
  const auto& [levels, code_of, sid] = *regions;
  std::vector<std::string> missing;
  for (const auto& lab : labels) if (std::find(levels.begin(), levels.end(), lab) == levels.end()) missing.push_back(lab);
  if (!missing.empty()) {
    std::string list = "[";
    for (std::size_t i = 0; i < missing.size(); ++i) list += (i ? ", '" : "'") + missing[i] + "'";
    value_error(fname + ": regions not in the atlas '" + aname + "': " + list + "]");
  }
  const std::string comment = text_field(get(m, "Comment")).empty() ? p.stem().string() : text_field(get(m, "Comment"));
  const std::string stem = nxr::io::sanitize_node_name(removeprefix(p.stem().string(), "matrix_"));
  if (nxr::io::selection::has_measurement(sub.db(), sid, [&] { nxr::io::selection::MeasurementFilter f; f.measure = comment; f.of_field_none = true; f.whole_only = true; return f; }()))
    value_error(fname + ": the " + aname + " atlas already holds a measure '" + comment + "'");
  const std::string fam = nxr::io::sanitize_node_name(aname);
  const auto unit = suvr_unit(stem, comment);
  std::vector<RegionValue> rows;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (!std::isfinite(values[i])) continue;
    const auto k = static_cast<std::size_t>(std::find(levels.begin(), levels.end(), labels[i]) - levels.begin());
    rows.push_back({code_of[k], values[i], std::nullopt});
  }
  const int n = region_measurements(sub, sid, rows, comment, unit);
  return Json{{"name", stem}, {"id", sid}, {"surface", sr.spath.substr(sr.spath.rfind('/') + 1)}, {"atlas", fam}, {"regions", static_cast<std::int64_t>(labels.size())},
              {"measurements", n}, {"kind", comment}};
}

Json export_map(SubjectWriter& sub, const fs::path& path, const fs::path& bst_root, const std::optional<std::string>& acquisition, bool replace, const std::optional<Json>& provenance,
                const MapHooks* hooks) {
  const bool matrix = starts_with(path.filename().string(), "matrix_");
  if (matrix && is_matrix_series(path)) return export_matrix_series(sub, path, acquisition, hooks);
  if (matrix) return export_region_map(sub, path, bst_root, acquisition);
  return export_source_map(sub, path, bst_root, acquisition, replace, provenance, hooks);
}

std::vector<fs::path> template_map_files(const fs::path& root) {
  const fs::path data = root / "data";
  std::vector<fs::path> homes = {data / "@default_study", data / "@inter"};
  std::error_code ec;
  if (fs::is_directory(data / "Group_analysis", ec)) {
    std::vector<fs::path> groups;
    for (const std::string& n : glob_names(data / "Group_analysis", "", "")) if (fs::is_directory(data / "Group_analysis" / n, ec)) groups.push_back(data / "Group_analysis" / n);
    homes.insert(homes.end(), groups.begin(), groups.end());
  }
  std::vector<fs::path> out;
  for (const fs::path& h : homes) {
    std::vector<fs::path> cands = find_source_maps(h);
    if (fs::is_directory(h, ec)) for (const std::string& n : glob_names(h, "matrix_", ".mat")) if (!starts_with(n, "._")) cands.push_back(h / n);
    for (const fs::path& q : cands) {
      const auto v = read_variables(q, {"SurfaceFile"});
      std::string sf = text_field(get(v, "SurfaceFile"));
      std::replace(sf.begin(), sf.end(), '\\', '/');
      if (starts_with(sf, "@default_subject/") && (starts_with(q.filename().string(), "results_") || is_region_map(q))) out.push_back(q);
    }
  }
  return out;
}

}  // namespace nxr::bst
