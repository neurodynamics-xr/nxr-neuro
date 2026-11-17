// nxr-io — reads of a subject's other rows as stored rows (subject_reads.h).
#include "nxr/io/subject_reads.h"

#include "nxr/io/statistic.h"

namespace nxr::io {

std::optional<Row> find_field(const Database& db, const std::string& subject_id, const FieldQuery& q) {
  std::string sql = "SELECT * FROM field WHERE subject_id = ? AND name = ?";
  std::vector<Value> a{V(subject_id), V(q.name)};
  if (q.complete_only) sql += " AND status = 'complete'";
  if (q.kind) { sql += " AND kind = ?"; a.push_back(V(*q.kind)); }
  if (q.manifold_id) { sql += " AND manifold_id = ?"; a.push_back(V(*q.manifold_id)); }
  if (q.stored) sql += *q.stored ? " AND path IS NOT NULL" : " AND path IS NULL";
  return db.one(sql, a);
}

std::vector<Row> fields_on_manifold(const Database& db, const std::string& subject_id, const std::string& manifold_id) {
  return db.all("SELECT * FROM field WHERE subject_id = ? AND manifold_id = ? AND status = 'complete' ORDER BY rowid", {V(subject_id), V(manifold_id)});
}

std::vector<Row> group_atlas_fields(const Database& db, const std::string& subject_id) {
  return db.all("SELECT * FROM field WHERE subject_id = ? AND path IS NULL AND json_extract(producer_json, '$.atlas') IN ('spatial', 'time')", {V(subject_id)});
}

std::optional<Row> find_operator(const Database& db, const std::string& subject_id, const std::string& kind, const std::string& name) {
  return db.one("SELECT * FROM operator WHERE subject_id = ? AND kind = ? AND name = ? AND status = 'complete'", {V(subject_id), V(kind), V(name)});
}

std::vector<Row> stored_operators(const Database& db, const std::string& subject_id, const std::string& kind) {
  return db.all("SELECT * FROM operator WHERE subject_id = ? AND kind = ? AND status = 'complete' AND path IS NOT NULL", {V(subject_id), V(kind)});
}

std::optional<Row> find_manifold(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& type) {
  std::string sql = "SELECT * FROM manifold WHERE subject_id = ? AND name = ? AND status = 'complete'";
  std::vector<Value> a{V(subject_id), V(name)};
  if (type) { sql += " AND type = ?"; a.push_back(V(*type)); }
  return db.one(sql, a);
}

std::optional<Row> manifold_with_geometry(const Database& db, const std::string& manifold_id) {
  return db.one("SELECT m.*, g.positions_path AS positions_path, g.coordinate_system_id AS cs_id FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.id = ?", {V(manifold_id)});
}

std::optional<Row> manifold_with_geometry_named(const Database& db, const std::string& subject_id, const std::string& name) {
  return db.one("SELECT m.*, g.positions_path AS positions_path, g.coordinate_system_id AS cs_id FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.name = ? AND m.status = 'complete'",
                {V(subject_id), V(name)});
}

std::optional<Row> surface_with_geometry(const Database& db, const std::string& subject_id, const std::optional<std::string>& name) {
  const std::string q = "SELECT m.*, g.positions_path AS positions_path FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.subject_id = ? AND m.status = 'complete' AND " +
                        std::string(name ? "m.type = 'surface' AND m.name = ?" : "m.is_primary = 1");
  std::vector<Value> a{V(subject_id)};
  if (name) a.push_back(V(*name));
  return db.one(q, a);
}

std::optional<std::string> topology_faces_path(const Database& db, const std::string& topology_id) {
  const auto r = db.one("SELECT path FROM topology_cell WHERE topology_id = ? AND rank = 2", {V(topology_id)});
  if (!r) return std::nullopt;
  return text(*r, "path");
}

std::string manifold_coordinate_unit(const Database& db, const std::string& manifold_id) {
  const auto r = db.one("SELECT cs.unit AS unit FROM manifold m JOIN geometry g ON g.id = m.geometry_id JOIN coordinate_system cs ON cs.id = g.coordinate_system_id WHERE m.id = ?", {V(manifold_id)});
  return r ? opt_text(*r, "unit").value_or("") : std::string();
}

std::string manifold_coordinate_system(const Database& db, const std::string& manifold_id) {
  const auto r = db.one("SELECT g.coordinate_system_id AS cs FROM manifold m JOIN geometry g ON g.id = m.geometry_id WHERE m.id = ?", {V(manifold_id)});
  return r ? opt_text(*r, "cs").value_or("") : std::string();
}

std::optional<std::string> coordinate_system_name_of_geometry(const Database& db, const std::string& geometry_id) {
  const auto c = db.one("SELECT c.name FROM geometry g JOIN coordinate_system c ON c.id = g.coordinate_system_id WHERE g.id = ?", {V(geometry_id)});
  if (!c) return std::nullopt;
  return text(*c, "name");
}

std::vector<std::string> factor_manifolds(const Database& db, const std::string& manifold_id) {
  std::vector<std::string> out;
  for (const Row& r : db.all("SELECT d.manifold_id AS mid FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? ORDER BY d.ordinal", {V(manifold_id)})) out.push_back(text(r, "mid"));
  return out;
}

std::optional<LineFactor> line_factor(const Database& db, const std::string& manifold_id) {
  const auto r = db.one("SELECT d.spacing AS spacing, d.origin AS origin, d.n_vertices AS n FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? AND d.ordinal = 0", {V(manifold_id)});
  if (!r) return std::nullopt;
  LineFactor f;
  f.spacing = is_null(*r, "spacing") ? 0.0 : std::stod(text(*r, "spacing"));
  if (!is_null(*r, "origin")) f.origin = std::stod(text(*r, "origin"));
  f.count = integer(*r, "n");
  return f;
}

std::string acquisition_label_of(const Database& db, const std::string& acquisition_id) {
  const auto a = db.one("SELECT label FROM acquisition WHERE id = ?", {V(acquisition_id)});
  return a ? text(*a, "label") : std::string();
}

std::vector<Row> complete_subjects(const Database& db) { return db.all("SELECT id, name FROM subject WHERE kind = 'subject' AND status = 'complete' ORDER BY name"); }

std::optional<Row> find_subject(const Database& db, const std::string& name, const std::optional<std::string>& dataset_id, const std::optional<std::string>& kind) {
  std::string sql = "SELECT * FROM subject WHERE name = ? AND status = 'complete'";
  std::vector<Value> a{V(name)};
  if (dataset_id) { sql += " AND dataset_id = ?"; a.push_back(V(*dataset_id)); }
  if (kind) { sql += " AND kind = ?"; a.push_back(V(*kind)); }
  return db.one(sql, a);
}

std::optional<Row> dataset_row(const Database& db) { return db.one("SELECT id, name FROM dataset"); }

std::string producer_column(const std::string& table) { return table == "selection" ? "params_json" : "producer_json"; }

std::vector<Row> producer_rows(const Database& db, const std::string& subject_id, const std::string& table, const std::optional<std::string>& path) {
  if (table == "statistic" && !has_table(db, "statistic")) return {};   // a dataset made before the statistic table has none
  std::string sql = "SELECT * FROM " + table + " WHERE subject_id = ? AND status = 'complete' AND " + producer_column(table) + " IS NOT NULL";
  std::vector<Value> a{V(subject_id)};
  if (path) { sql += " AND path = ?"; a.push_back(V(*path)); }
  return db.all(sql, a);
}

}  // namespace nxr::io
