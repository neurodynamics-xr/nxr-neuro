// nxr-io — reads of the selection module as stored rows (selection_reads.h). The one place the pipelines' and the command line's selection questions are SQL.
#include "nxr/io/selection_reads.h"

#include <algorithm>

#include "nxr/io/statistic.h"

namespace nxr::io::selection {

std::optional<Row> named_row(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& stored_type, const std::optional<std::string>& manifold_id) {
  std::string sql = "SELECT * FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'";
  std::vector<Value> a{V(subject_id), V(name)};
  if (stored_type) { sql += " AND type = ?"; a.push_back(V(*stored_type)); }
  if (manifold_id) { sql += " AND manifold_id = ?"; a.push_back(V(*manifold_id)); }
  return db.one(sql, a);
}

std::optional<Row> ladder_row(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& manifold_id) {
  std::string sql = "SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'ladder' AND member_of_id IS NULL AND status = 'complete'";
  std::vector<Value> a{V(subject_id), V(name)};
  if (manifold_id) { sql += " AND manifold_id = ?"; a.push_back(V(*manifold_id)); }
  return db.one(sql, a);
}

std::optional<Row> member_row(const Database& db, const std::string& parent_id, const std::string& name) {
  return db.one("SELECT * FROM selection WHERE member_of_id = ? AND name = ? AND status = 'complete'", {V(parent_id), V(name)});
}

std::vector<Row> member_rows(const Database& db, const std::string& parent_id, bool complete_only) {
  return db.all(std::string("SELECT * FROM selection WHERE member_of_id = ?") + (complete_only ? " AND status = 'complete'" : "") + " ORDER BY member_ordinal", {V(parent_id)});
}

std::optional<Row> row_over_field(const Database& db, const std::string& subject_id, const std::string& of_field_id, const std::string& name) {
  return db.one("SELECT * FROM selection WHERE subject_id = ? AND of_field_id = ? AND name = ? AND status = 'complete'", {V(subject_id), V(of_field_id), V(name)});
}

std::vector<Row> rows_over_field(const Database& db, const std::string& subject_id, const std::string& of_field_id) {
  return db.all("SELECT * FROM selection WHERE subject_id = ? AND of_field_id = ? AND status = 'complete'", {V(subject_id), V(of_field_id)});
}

std::optional<Row> labelling_row(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& manifold_id) {
  std::string sql = "SELECT * FROM selection WHERE subject_id = ? AND name = ? AND type = 'set' AND array_path IS NOT NULL AND status = 'complete'";
  std::vector<Value> a{V(subject_id), V(name)};
  if (manifold_id) { sql += " AND manifold_id = ?"; a.push_back(V(*manifold_id)); }
  return db.one(sql + " ORDER BY path LIMIT 1", a);
}

std::vector<Row> set_rows_with_param(const Database& db, const std::string& subject_id, const std::string& param_key) {
  return db.all("SELECT * FROM selection WHERE subject_id = ? AND type = 'set' AND status = 'complete' AND params_json LIKE ? ORDER BY name", {V(subject_id), V("%\"" + param_key + "\"%")});
}

std::vector<Row> joint_rows_on_tree(const Database& db, const std::string& subject_id, const std::string& tree) {
  return db.all("SELECT x.*, f.producer_json AS fp, f.id AS fid, f.name AS fname, r.name AS rec FROM selection x JOIN field f ON f.id = x.of_field_id JOIN field r ON r.id = f.derived_from_id "
                "WHERE x.subject_id = ? AND json_extract(x.params_json, '$.joint') IS NOT NULL AND json_extract(x.params_json, '$.atlas.tree') = ? AND x.status = 'complete'", {V(subject_id), V(tree)});
}

std::optional<Row> tree_level_row(const Database& db, const std::string& subject_id, const std::string& tree, int level) {
  return db.one("SELECT * FROM selection WHERE subject_id = ? AND json_extract(params_json, '$.atlas.tree') = ? AND json_extract(params_json, '$.joint') IS NULL AND name LIKE '% group tiles L' || ? AND status = 'complete'",
                {V(subject_id), V(tree), V(std::to_string(level))});
}

std::optional<Row> default_atlas_ladder_row(const Database& db, const std::string& subject_id) {
  return db.one("SELECT * FROM selection WHERE subject_id = ? AND type = 'ladder' AND json_extract(params_json, '$.atlas.default') = 1", {V(subject_id)});
}

std::vector<std::pair<double, double>> span_bounds(const Database& db, const std::string& selection_id) {
  std::vector<std::pair<double, double>> out;
  for (const Row& r : db.all("SELECT start, stop FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(selection_id)})) out.emplace_back(std::stod(text(r, "start")), std::stod(text(r, "stop")));
  return out;
}

std::map<std::int64_t, std::string> lookup_names(const Database& db, const std::string& dictionary_id) {
  std::map<std::int64_t, std::string> out;
  for (const Row& e : db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", {V(dictionary_id)})) out[integer(e, "code")] = text(e, "name");
  return out;
}

std::optional<std::int64_t> lookup_code_of_any(const Database& db, const std::string& dictionary_id, const std::vector<std::string>& names) {
  std::string in;
  std::vector<Value> a{V(dictionary_id)};
  for (const std::string& n : names) { in += in.empty() ? "?" : ", ?"; a.push_back(V(n)); }
  if (names.empty()) return std::nullopt;
  const auto r = db.one("SELECT code FROM dictionary_entry WHERE dictionary_id = ? AND name IN (" + in + ") ORDER BY name", a);
  if (!r) return std::nullopt;
  return integer(*r, "code");
}

std::optional<Row> lookup_entry_named(const Database& db, const std::string& dictionary_name, const std::string& entry_name) {
  return db.one("SELECT * FROM dictionary_entry WHERE name = ? AND dictionary_id IN (SELECT id FROM dictionary WHERE name = ?)", {V(entry_name), V(dictionary_name)});
}

std::optional<MeasurementArray> measurement_array(const Database& db, const std::string& selection_id, const std::string& of_field_id, const std::string& measure) {
  const auto r = db.one("SELECT array_path, component FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND measure = ?", {V(selection_id), V(of_field_id), V(measure)});
  if (!r) return std::nullopt;
  return MeasurementArray{text(*r, "array_path"), integer(*r, "component")};
}

std::optional<MeasurementArray> measurement_array_over_field(const Database& db, const std::string& of_field_id, const std::string& measure) {
  const auto r = db.one("SELECT array_path, component FROM selection_measurement WHERE of_field_id = ? AND measure = ? AND array_path IS NOT NULL", {V(of_field_id), V(measure)});
  if (!r) return std::nullopt;
  return MeasurementArray{text(*r, "array_path"), integer(*r, "component")};
}

std::map<std::string, std::string> measurement_arrays(const Database& db, const std::string& selection_id) {
  std::map<std::string, std::string> out;
  for (const Row& r : db.all("SELECT DISTINCT measure, array_path FROM selection_measurement WHERE selection_id = ? AND array_path IS NOT NULL", {V(selection_id)})) out[text(r, "measure")] = text(r, "array_path");
  return out;
}

std::vector<std::string> measures_over_field(const Database& db, const std::string& selection_id, const std::string& of_field_id, const std::optional<std::string>& except) {
  std::string sql = "SELECT DISTINCT measure FROM selection_measurement WHERE selection_id = ? AND of_field_id = ?";
  std::vector<Value> a{V(selection_id), V(of_field_id)};
  if (except) { sql += " AND measure <> ?"; a.push_back(V(*except)); }
  std::vector<std::string> out;
  for (const Row& r : db.all(sql + " ORDER BY measure", a)) out.push_back(text(r, "measure"));
  return out;
}

std::vector<std::string> computed_by_over_field(const Database& db, const std::string& selection_id, const std::string& of_field_id, const std::optional<std::string>& measure) {
  std::string sql = "SELECT DISTINCT computed_by FROM selection_measurement WHERE selection_id = ? AND of_field_id = ?";
  std::vector<Value> a{V(selection_id), V(of_field_id)};
  if (measure) { sql += " AND measure = ?"; a.push_back(V(*measure)); }
  std::vector<std::string> out;
  for (const Row& r : db.all(sql, a)) out.push_back(opt_text(r, "computed_by").value_or(""));
  return out;
}

std::vector<std::string> computed_by_over_statistic(const Database& db, const std::string& selection_id, const std::string& of_statistic_id, const std::optional<std::string>& measure) {
  if (!has_column(db, "selection_measurement", "of_statistic_id")) return {};   // a dataset with no measurements of statistics
  std::string sql = "SELECT DISTINCT computed_by FROM selection_measurement WHERE selection_id = ? AND of_statistic_id = ?";
  std::vector<Value> a{V(selection_id), V(of_statistic_id)};
  if (measure) { sql += " AND measure = ?"; a.push_back(V(*measure)); }
  std::vector<std::string> out;
  for (const Row& r : db.all(sql, a)) out.push_back(opt_text(r, "computed_by").value_or(""));
  return out;
}

std::optional<std::string> within_selection_of(const Database& db, const std::string& selection_id) {
  const auto r = db.one("SELECT DISTINCT within_selection_id FROM selection_measurement WHERE selection_id = ? AND within_selection_id IS NOT NULL", {V(selection_id)});
  if (!r) return std::nullopt;
  return text(*r, "within_selection_id");
}

std::vector<Row> array_rows_with_field(const Database& db, const std::string& selection_id) {
  return db.all("SELECT m.*, f.kind AS fkind, f.unit AS funit FROM selection_measurement m JOIN field f ON f.id = m.of_field_id WHERE m.selection_id = ? AND m.array_path IS NOT NULL", {V(selection_id)});
}

std::vector<Row> computed_by_and_unit(const Database& db, const std::string& selection_id, const std::string& measure, const MeasuredOf& of) {
  std::string w = "selection_id = ? AND measure = ? AND ";
  std::vector<Value> a{V(selection_id), V(measure)};
  if (of.statistic_id) {
    w += "of_statistic_id = ? AND of_index = ?";
    a.push_back(V(*of.statistic_id));
    a.push_back(of.index ? V(*of.index) : V(nullptr));
  } else {
    if (of.field_id) { w += "of_field_id = ?"; a.push_back(V(*of.field_id)); } else w += "of_field_id IS NULL";
    if (has_column(db, "selection_measurement", "of_statistic_id")) w += " AND of_statistic_id IS NULL";
  }
  return db.all("SELECT DISTINCT computed_by, unit FROM selection_measurement WHERE " + w, a);
}

}  // namespace nxr::io::selection
