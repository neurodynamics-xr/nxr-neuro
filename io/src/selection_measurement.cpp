// nxr-io — what is measured on a selection (selection_measurement.h): the row-level writes, reads and deletes of `selection_measurement`; the batch calls forward to measures.cpp.
#include "nxr/io/selection_measurement.h"

#include "nxr/io/array_writer.h"
#include "nxr/io/measures.h"
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

std::int64_t remove_measurements(Database& db, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) {
  Json where{{"selectionId", selection_id}};
  if (of_field_id) where["ofFieldId"] = *of_field_id;
  if (measure) where["measure"] = *measure;
  return delete_measurements(db, where);
}

}  // namespace nxr::io::selection
