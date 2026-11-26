// nxr-io — THE SUBJECT AND DATASET WRITERS' READS: the same typed questions as subject_reads.h, selection_reads.h, selection_store.h, statistic.h and subject_package.h, asked of the writer
// that holds the subject (or the dataset), so a pipeline or a test never names the database handle (the records layering guard: only io touches the database). Each is the
// function of the same name on `writer.db()` (and `writer.id()`, `writer.store()`); nothing else.
#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/statistic.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/subject_reads.h"
#include "nxr/io/writer.h"

namespace nxr::io {

std::optional<std::string> coordinate_system_name_of_geometry(SubjectWriter& sub, const std::string& geometry_id);
std::vector<Row> complete_subjects(DatasetWriter& ds);
std::optional<Row> find_manifold(SubjectWriter& sub, const std::string& name, const std::optional<std::string>& type = std::nullopt);
std::optional<Row> manifold_with_geometry(SubjectWriter& sub, const std::string& manifold_id);
std::string manifold_coordinate_system(SubjectWriter& sub, const std::string& manifold_id);
std::string manifold_coordinate_system_name(SubjectWriter& sub, const std::string& manifold_id);
std::optional<Row> find_field(SubjectWriter& sub, const FieldQuery& q);
/** the names of the subject's complete surfaces (with geometry), by name */
std::vector<std::string> surface_names(SubjectWriter& sub);
std::vector<Row> statistics_of(SubjectWriter& sub, const std::optional<std::string>& of_field_id = std::nullopt, const std::optional<std::string>& measure = std::nullopt);
std::optional<OriginRecord> origin_of(SubjectWriter& sub, const std::string& table, const std::string& id);
void set_origin(DatasetWriter& ds, const std::string& table, const std::string& id, const OriginRecord& origin);
/** the subject's package: give an item its origin record; and its items (complete nodes with a path) that have none */
void set_origin(SubjectWriter& sub, const std::string& table, const std::string& id, const OriginRecord& origin);
std::vector<PackageItem> items_without_origin(SubjectWriter& sub);
/** the manifolds of the subject by name, with the producer record and the geometry's positions path (`pj`, `pp`), and the names of the selections that lie on each */
std::optional<Row> manifold_row_named(SubjectWriter& sub, const std::string& name);
std::vector<std::string> selection_names_on(SubjectWriter& sub, const std::string& manifold_id);

namespace selection {
std::optional<Row> named_row(SubjectWriter& sub, const std::string& name, const std::optional<std::string>& stored_type = std::nullopt, const std::optional<std::string>& manifold_id = std::nullopt);
std::optional<Row> labelling_row(SubjectWriter& sub, const std::string& name, const std::optional<std::string>& manifold_id = std::nullopt);
std::map<std::int64_t, std::string> lookup_names(SubjectWriter& sub, const std::string& dictionary_id);
Loaded load(SubjectWriter& sub, const std::string& selection_id);
std::vector<Row> set_rows_with_param(SubjectWriter& sub, const std::string& param_key);
std::optional<MeasurementArray> measurement_array(SubjectWriter& sub, const std::string& selection_id, const std::string& of_field_id, const std::string& measure);
std::vector<std::int64_t> member_labels(SubjectWriter& sub, const std::string& selection_id);
ElementMembers read_element_members(SubjectWriter& sub, const std::string& selection_id);
}  // namespace selection

}  // namespace nxr::io
