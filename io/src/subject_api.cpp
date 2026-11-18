// nxr-io — the writers' reads (subject_api.h): each the function of the same name on the writer's database, subject and store.
#include "nxr/io/subject_api.h"

namespace nxr::io {

std::optional<std::string> coordinate_system_name_of_geometry(SubjectWriter& sub, const std::string& geometry_id) { return coordinate_system_name_of_geometry(sub.db(), geometry_id); }
std::vector<Row> complete_subjects(DatasetWriter& ds) { return complete_subjects(ds.db()); }
std::optional<Row> find_manifold(SubjectWriter& sub, const std::string& name, const std::optional<std::string>& type) { return find_manifold(sub.db(), sub.id(), name, type); }
std::optional<Row> manifold_with_geometry(SubjectWriter& sub, const std::string& manifold_id) { return manifold_with_geometry(sub.db(), manifold_id); }
std::string manifold_coordinate_system(SubjectWriter& sub, const std::string& manifold_id) { return manifold_coordinate_system(sub.db(), manifold_id); }
std::string manifold_coordinate_system_name(SubjectWriter& sub, const std::string& manifold_id) { return manifold_coordinate_system_name(sub.db(), manifold_id); }
std::optional<Row> find_field(SubjectWriter& sub, const FieldQuery& q) { return find_field(sub.db(), sub.id(), q); }
std::vector<std::string> surface_names(SubjectWriter& sub) {
  std::vector<std::string> out;
  for (const Row& r : sub.db().all("SELECT m.name AS name FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.status = 'complete' AND m.type = 'surface' AND (m.producer_json IS NULL OR json_extract(m.producer_json, '$.intrinsic_of') IS NULL) ORDER BY m.name", {V(sub.id())}))
    out.push_back(text(r, "name"));
  return out;
}
std::vector<Row> statistics_of(SubjectWriter& sub, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) { return statistics_of(sub.db(), sub.id(), of_field_id, measure); }
std::optional<OriginRecord> origin_of(SubjectWriter& sub, const std::string& table, const std::string& id) { return origin_of(sub.db(), table, id); }
void set_origin(DatasetWriter& ds, const std::string& table, const std::string& id, const OriginRecord& origin) { set_origin(ds.db(), table, id, origin); }
std::optional<Row> manifold_row_named(SubjectWriter& sub, const std::string& name) {
  return sub.db().one("SELECT m.id AS id, m.producer_json AS pj, g.positions_path AS pp FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.name = ?", {V(sub.id()), V(name)});
}
std::vector<std::string> selection_names_on(SubjectWriter& sub, const std::string& manifold_id) {
  std::vector<std::string> out;
  for (const Row& r : sub.db().all("SELECT name FROM selection WHERE manifold_id = ?", {V(manifold_id)})) out.push_back(text(r, "name"));
  return out;
}

namespace selection {
std::optional<Row> named_row(SubjectWriter& sub, const std::string& name, const std::optional<std::string>& stored_type, const std::optional<std::string>& manifold_id) { return named_row(sub.db(), sub.id(), name, stored_type, manifold_id); }
std::optional<Row> labelling_row(SubjectWriter& sub, const std::string& name, const std::optional<std::string>& manifold_id) { return labelling_row(sub.db(), sub.id(), name, manifold_id); }
std::map<std::int64_t, std::string> lookup_names(SubjectWriter& sub, const std::string& dictionary_id) { return lookup_names(sub.db(), dictionary_id); }
Loaded load(SubjectWriter& sub, const std::string& selection_id) { return load(sub.db(), sub.store(), selection_id); }
std::vector<Row> set_rows_with_param(SubjectWriter& sub, const std::string& param_key) { return set_rows_with_param(sub.db(), sub.id(), param_key); }
std::optional<MeasurementArray> measurement_array(SubjectWriter& sub, const std::string& selection_id, const std::string& of_field_id, const std::string& measure) {
  return measurement_array(sub.db(), selection_id, of_field_id, measure);
}
std::vector<std::int64_t> member_labels(SubjectWriter& sub, const std::string& selection_id) { return member_labels(sub.db(), sub.store(), selection_id); }
ElementMembers read_element_members(SubjectWriter& sub, const std::string& selection_id) { return read_element_members(sub.db(), sub.store(), selection_id); }
}  // namespace selection

}  // namespace nxr::io
