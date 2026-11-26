// nxr-io — what is measured on a selection (selection_measurement.h): the row-level writes, reads and deletes of `selection_measurement`; the batch calls forward to measures.cpp.
#include "nxr/io/selection_measurement.h"

#include "nxr/io/array_writer.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"
#include "nxr/io/records/records.h"

namespace nxr::io::selection {

void save_measurement(SubjectWriter& sub, const std::string& selection_id, const Cols& row) {
  sub.update([&] {
    Row mr = {{"selection_id", selection_id}};
    for (const auto& [k, v] : row) mr[k] = v;
    sub.db().insert("selection_measurement", mr);
  });
}

void save_measurement_array(SubjectWriter& sub, const std::string& selection_id, const std::string& array_path, const ArrayView& values, const Cols& row) {
  sub.write_array(array_path, values);
  Cols c = row;
  c.set("array_path", array_path);
  save_measurement(sub, selection_id, c);
}

std::int64_t save_member_values(SubjectWriter& sub, const std::string& selection_id, const std::string& measure, const std::vector<MemberValue>& values,
                                const std::optional<std::string>& unit, const std::string& computed_by, const std::optional<std::string>& within_selection_id) {
  const std::string stamp = now_utc();
  sub.db().tx([&] {
    for (const MemberValue& r : values) {
      Row row = {{"selection_id", selection_id}, {"code", V(r.code)}, {"of_field_id", V(nullptr)}, {"measure", measure}, {"value", r.value}, {"unit", V(unit)},
                 {"within_selection_id", r.within_code && within_selection_id ? V(*within_selection_id) : V(nullptr)},
                 {"within_code", r.within_code ? V(*r.within_code) : V(nullptr)}, {"computed_utc", stamp}, {"computed_by", computed_by}};
      sub.db().insert("selection_measurement", row);
    }
  });
  return static_cast<std::int64_t>(values.size());
}

std::int64_t save_measurements(Database& db, const Json& rows, const std::optional<std::string>& by) { return write_measurements(db, rows, by); }

std::vector<records::SelectionMeasurementRow> measurements_of(const Database& db, const std::string& selection_id, bool own) { return records::Selections(db).measurements(selection_id, own); }

Json measurement_kinds_of(const Database& db, const std::string& selection_id) { return measurement_kinds(db, selection_id); }

void update_own_measurements(SubjectWriter& sub, const std::string& selection_id, const std::vector<Cols>& rows) {
  sub.update([&] {
    sub.db().execute("DELETE FROM selection_measurement WHERE selection_id = ? AND code IS NULL", {V(selection_id)});
    for (const auto& c : rows) {
      Row mr = {{"selection_id", selection_id}};
      for (const auto& [k, v] : c) mr[k] = v;
      sub.db().insert("selection_measurement", mr);
    }
  });
}

std::int64_t remove_measurements(Database& db, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure,
                                 const std::optional<std::string>& of_statistic_id) {
  Json where{{"selectionId", selection_id}};
  if (of_field_id) where["ofFieldId"] = *of_field_id;
  if (of_statistic_id) where["ofStatisticId"] = *of_statistic_id;
  if (measure) where["measure"] = *measure;
  return delete_measurements(db, where);
}

bool has_measurement(const Database& db, const std::string& selection_id, const MeasurementFilter& f) {
  std::string sql = "SELECT 1 AS x FROM selection_measurement WHERE selection_id = ?";
  std::vector<Value> a{V(selection_id)};
  if (f.measure) { sql += " AND measure = ?"; a.push_back(V(*f.measure)); }
  if (f.of_field_id) { sql += " AND of_field_id = ?"; a.push_back(V(*f.of_field_id)); }
  if (f.of_field_none) sql += " AND of_field_id IS NULL";
  if (f.of_statistic_id) { sql += " AND of_statistic_id = ?"; a.push_back(V(*f.of_statistic_id)); }
  if (f.of_index) { sql += " AND of_index = ?"; a.push_back(V(*f.of_index)); }
  if (f.whole_only) sql += " AND within_code IS NULL";
  if (f.rows_manifold_id) {
    if (!has_column(db, "selection_measurement", "rows_manifold_id")) return false;
    sql += " AND rows_manifold_id = ?"; a.push_back(V(*f.rows_manifold_id));
  }
  return db.one(sql + " LIMIT 1", a).has_value();
}

std::vector<std::string> selections_measured(const Database& db, const std::string& of_field_id, const std::string& measure) {
  std::vector<std::string> out;
  for (const Row& r : db.all("SELECT DISTINCT selection_id FROM selection_measurement WHERE of_field_id = ? AND measure = ? ORDER BY selection_id", {V(of_field_id), V(measure)}))
    out.push_back(text(r, "selection_id"));
  return out;
}

std::vector<std::string> measurement_array_paths_of_field(const Database& db, const std::string& of_field_id) {
  std::vector<std::string> out;
  for (const Row& r : db.all("SELECT array_path FROM selection_measurement WHERE of_field_id = ? AND array_path IS NOT NULL", {V(of_field_id)})) out.push_back(text(r, "array_path"));
  return out;
}

std::vector<std::pair<std::string, std::string>> measurement_array_paths_of_subject(const Database& db, const std::string& subject_id) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const Row& r : db.all("SELECT m.selection_id AS id, m.array_path AS p FROM selection_measurement m JOIN selection s ON s.id = m.selection_id WHERE s.subject_id = ?", {V(subject_id)}))
    if (!is_null(r, "p")) out.emplace_back(text(r, "id"), text(r, "p"));
  return out;
}

std::vector<SubjectArray> measurement_array_paths_in_dataset(const Query& all) {
  std::vector<SubjectArray> out;
  for (const Row& r : all("SELECT s.path AS sp, s.name AS sn, r.array_path AS p FROM selection_measurement r JOIN selection x ON x.id = r.selection_id JOIN subject s ON s.id = x.subject_id WHERE r.array_path IS NOT NULL"))
    out.push_back({text(r, "sp"), text(r, "sn"), text(r, "p")});
  return out;
}

std::vector<Row> subject_own_measurement_rows(const Database& db, const std::string& subject_id) {
  return db.all("SELECT m.* FROM selection_measurement m JOIN selection s ON s.id = m.selection_id WHERE s.subject_id = ? AND m.code IS NULL ORDER BY m.selection_id, m.measure", {V(subject_id)});
}

}  // namespace nxr::io::selection
