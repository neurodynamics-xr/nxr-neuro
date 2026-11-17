// nxr-io — a subject's bundle (see bundle.h): the Studio's `bundleOf`, query for query.
#include "nxr/io/selection_measurement.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/bundle.h"

#include <set>
#include <vector>

namespace nxr::io {

Json row_json(const Row& r) {
  Json o = Json::object();
  for (const auto& [k, v] : r) {
    if (std::holds_alternative<std::monostate>(v)) o[k] = nullptr;
    else if (const auto* i = std::get_if<std::int64_t>(&v)) o[k] = *i;
    else if (const auto* d = std::get_if<double>(&v)) o[k] = *d;
    else o[k] = std::get<std::string>(v);
  }
  return o;
}

namespace {
using Ids = std::vector<std::string>;
Json rows_json(const std::vector<Row>& rows) {
  Json a = Json::array();
  for (const Row& r : rows) a.push_back(row_json(r));
  return a;
}
std::string marks(std::size_t n) {
  std::string s;
  for (std::size_t i = 0; i < n; ++i) s += i ? ", ?" : "?";
  return s;
}
std::vector<Value> vals(const Ids& ids) { return std::vector<Value>(ids.begin(), ids.end()); }
// distinct, non-empty, first-seen order (a JS `new Set` of the same list)
void add(Ids& out, std::set<std::string>& seen, const Row& r, const char* col) {
  const auto it = r.find(col);
  if (it == r.end()) return;
  if (const auto* s = std::get_if<std::string>(&it->second); s && !s->empty() && seen.insert(*s).second) out.push_back(*s);
}
std::optional<std::string> str(const Row& r, const char* col) {
  const auto it = r.find(col);
  if (it == r.end()) return std::nullopt;
  if (const auto* s = std::get_if<std::string>(&it->second)) return *s;
  return std::nullopt;
}
}  // namespace

std::optional<Json> subject_bundle(const Database& db, const std::string& sid) {
  const auto subject = db.one("SELECT * FROM subject WHERE id = ?", {Value(sid)});
  if (!subject) return std::nullopt;
  auto q = [&](const std::string& sql, std::vector<Value> args = {}) { return db.all(sql, args); };
  const Value S(sid);
  const auto manifolds = q("SELECT * FROM manifold WHERE subject_id = ? AND status = 'complete' ORDER BY path, name", {S});
  const auto selections = selection::subject_rows(db, sid);
  const std::string dataset_id = str(*subject, "dataset_id").value_or("");
  const auto vocab = selection::vocabulary_rows(db, {"frequency bands", "recipes"}, dataset_id, sid);
  Ids dict_ids; std::set<std::string> dseen;
  for (const Row& s : selections) add(dict_ids, dseen, s, "dictionary_id");
  for (const Row& d : vocab) add(dict_ids, dseen, d, "id");
  const auto dataset = db.one("SELECT * FROM dataset WHERE id = ?", {Value(dataset_id)});
  const auto fields = q("SELECT * FROM field WHERE subject_id = ? AND status = 'complete' ORDER BY path, name", {S});
  const auto operators = q("SELECT * FROM operator WHERE subject_id = ? AND status = 'complete' ORDER BY path, name", {S});
  const auto dimensions = q("SELECT d.* FROM geometry_dimension d JOIN geometry g ON g.id = d.geometry_id WHERE g.subject_id = ? ORDER BY d.geometry_id, d.ordinal", {S});
  std::set<std::string> own, own_ops, own_fields;
  for (const Row& m : manifolds) own.insert(str(m, "id").value_or(""));
  const auto contributions = q("SELECT c.* FROM field_contribution c JOIN field f ON f.id = c.field_id WHERE f.subject_id = ? ORDER BY c.field_id, c.ordinal", {S});
  for (const Row& o : operators) own_ops.insert(str(o, "id").value_or(""));
  for (const Row& f : fields) own_fields.insert(str(f, "id").value_or(""));
  Ids f_op_ids, f_field_ids; std::set<std::string> s1, s2;
  for (const Row& f : fields) if (auto x = str(f, "by_operator_id"); x && !x->empty() && !own_ops.count(*x) && s1.insert(*x).second) f_op_ids.push_back(*x);
  for (const Row& f : fields) if (auto x = str(f, "derived_from_id"); x && !x->empty() && !own_fields.count(*x) && s2.insert(*x).second) f_field_ids.push_back(*x);
  for (const Row& c : contributions) if (auto x = str(c, "source_field_id"); x && !x->empty() && !own_fields.count(*x) && s2.insert(*x).second) f_field_ids.push_back(*x);
  const auto foreign_ops = f_op_ids.empty() ? std::vector<Row>{} : q("SELECT * FROM operator WHERE id IN (" + marks(f_op_ids.size()) + ") AND status = 'complete' ORDER BY path, name", vals(f_op_ids));
  const auto foreign_fields = f_field_ids.empty() ? std::vector<Row>{}
      : q("SELECT f.* FROM field f JOIN manifold m ON m.id = f.manifold_id WHERE f.id IN (" + marks(f_field_ids.size()) + ") AND f.status = 'complete' AND m.type <> 'product' ORDER BY f.path, f.name", vals(f_field_ids));
  // the manifolds the rows name that are not the subject's own
  Ids foreign_ids; std::set<std::string> fseen;
  auto named = [&](const Row& r, const char* col) { if (auto x = str(r, col); x && !x->empty() && !own.count(*x) && fseen.insert(*x).second) foreign_ids.push_back(*x); };
  for (const Row& o : operators) { named(o, "from_manifold_id"); named(o, "to_manifold_id"); }
  for (const Row& f : fields) named(f, "manifold_id");
  for (const Row& s : selections) named(s, "manifold_id");
  for (const Row& m : manifolds) named(m, "derived_from_id");
  for (const Row& d : dimensions) named(d, "manifold_id");
  for (const Row& o : foreign_ops) { named(o, "from_manifold_id"); named(o, "to_manifold_id"); }
  for (const Row& f : foreign_fields) named(f, "manifold_id");
  std::vector<Row> foreign;
  if (!foreign_ids.empty()) {
    std::vector<Value> a = vals(foreign_ids), b = vals(foreign_ids);
    a.insert(a.end(), b.begin(), b.end());
    foreign = q("SELECT * FROM manifold WHERE id IN (" + marks(foreign_ids.size()) + ") OR component_of_id IN (" + marks(foreign_ids.size()) + ") ORDER BY path, name", a);
  }
  auto ids_of = [](const std::vector<Row>& rows, const char* col) { Ids out; std::set<std::string> seen; for (const Row& r : rows) add(out, seen, r, col); return out; };
  auto by_ids = [&](const std::string& sql, const Ids& list) {
    if (list.empty()) return std::vector<Row>{};
    std::string s = sql;
    s.replace(s.find('$'), 1, marks(list.size()));
    return q(s, vals(list));
  };
  const auto f_topologies = by_ids("SELECT * FROM topology WHERE id IN ($)", ids_of(foreign, "topology_id"));
  const auto f_geometries = by_ids("SELECT * FROM geometry WHERE id IN ($)", ids_of(foreign, "geometry_id"));
  const auto f_systems = by_ids("SELECT * FROM coordinate_system WHERE id IN ($)", ids_of(f_geometries, "coordinate_system_id"));
  Ids related_ids; std::set<std::string> rseen;
  for (const std::vector<Row>* list : std::initializer_list<const std::vector<Row>*>{&foreign, &foreign_ops, &foreign_fields, &contributions})
    for (const Row& r : *list) if (auto x = str(r, "subject_id"); x && !x->empty() && *x != sid && rseen.insert(*x).second) related_ids.push_back(*x);
  Json related = Json::array();
  for (const std::string& id : related_ids) if (auto r = db.one("SELECT * FROM subject WHERE id = ?", {Value(id)})) related.push_back(row_json(*r));
  auto cat = [](std::vector<Row> a, const std::vector<Row>& b) { a.insert(a.end(), b.begin(), b.end()); return rows_json(a); };
  Ids topo_ids, geom_ids;
  for (const Row& t : f_topologies) topo_ids.push_back(str(t, "id").value_or(""));
  for (const Row& g : f_geometries) geom_ids.push_back(str(g, "id").value_or(""));
  const bool has_acq = !q("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'acquisition'").empty();
  const bool has_statistic = !q("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'statistic'").empty();
  Json b = Json::object();
  b["related_subjects"] = related;
  std::optional<Row> datastore;
  if (dataset) if (auto ds_id = str(*dataset, "datastore_id")) datastore = db.one("SELECT * FROM datastore WHERE id = ?", {Value(*ds_id)});
  b["datastore"] = datastore ? row_json(*datastore) : Json(nullptr);
  b["dataset"] = dataset ? row_json(*dataset) : Json(nullptr);
  b["subject"] = row_json(*subject);
  b["acquisitions"] = has_acq ? rows_json(q("SELECT * FROM acquisition WHERE subject_id = ? ORDER BY label", {S})) : Json::array();
  b["manifolds"] = cat(manifolds, foreign);
  b["spaces"] = cat(q("SELECT * FROM space WHERE subject_id = ?", {S}), by_ids("SELECT * FROM space WHERE id IN ($)", ids_of(f_systems, "space_id")));
  b["measurements"] = rows_json(selection::subject_own_measurement_rows(db, sid));
  b["contributions"] = rows_json(contributions);
  b["jobs"] = rows_json(q("SELECT * FROM job WHERE subject_id = ? ORDER BY created_utc DESC", {S}));
  b["topologies"] = cat(q("SELECT * FROM topology WHERE subject_id = ?", {S}), f_topologies);
  b["topology_cells"] = cat(q("SELECT c.* FROM topology_cell c JOIN topology t ON t.id = c.topology_id WHERE t.subject_id = ? ORDER BY c.topology_id, c.rank", {S}),
                            by_ids("SELECT * FROM topology_cell WHERE topology_id IN ($) ORDER BY topology_id, rank", topo_ids));
  b["geometries"] = cat(q("SELECT * FROM geometry WHERE subject_id = ?", {S}), f_geometries);
  b["dimensions"] = cat(dimensions, by_ids("SELECT * FROM geometry_dimension WHERE geometry_id IN ($) ORDER BY geometry_id, ordinal", geom_ids));
  b["coordinate_systems"] = cat(q("SELECT c.* FROM coordinate_system c JOIN space s ON s.id = c.space_id WHERE s.subject_id = ?", {S}), f_systems);
  b["selections"] = rows_json(selections);
  b["extents"] = rows_json(selection::subject_extent_rows(db, sid));
  b["spans"] = rows_json(selection::subject_span_rows(db, sid));
  b["fields"] = rows_json(fields);
  b["operators"] = rows_json(operators);
  // the subject's STATISTICS (a database from before the statistic table has none)
  b["statistics"] = has_statistic ? rows_json(q("SELECT * FROM statistic WHERE subject_id = ? AND status = 'complete' ORDER BY path, name", {S})) : Json::array();
  b["lineage_fields"] = rows_json(foreign_fields);
  b["lineage_operators"] = rows_json(foreign_ops);
  b["operator_factors"] = rows_json(q("SELECT f.* FROM operator_factor f JOIN operator o ON o.id = f.operator_id WHERE o.subject_id = ? ORDER BY f.operator_id, f.ordinal", {S}));
  b["dictionaries"] = rows_json(selection::dictionary_rows(db, dict_ids));
  b["entries"] = rows_json(selection::dictionary_entry_rows(db, dict_ids));
  return b;
}

std::optional<Json> bundle_by_path(const Database& db, const std::string& store_path) {
  const auto r = db.one("SELECT s.id FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE ds.path || '/' || s.path = ?", {Value(store_path)});
  if (!r) return std::nullopt;
  return subject_bundle(db, str(*r, "id").value_or(""));
}

}  // namespace nxr::io
