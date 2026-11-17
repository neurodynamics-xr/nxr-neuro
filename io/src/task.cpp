// nxr-io — the task-level API (io/include/nxr/io/task/task.h) on the repositories and io's helpers. The only place orchestration's questions become SQL.
#include "nxr/io/task/task.h"

#include <algorithm>

#include "nxr/io/database.h"
#include "nxr/io/layout.h"
#include "nxr/io/measures.h"
#include "nxr/io/records/records.h"
#include "nxr/io/statistic.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::io::task {

const char* to_string(Cls c) {
  switch (c) {
    case Cls::manifold: return "manifold";
    case Cls::field: return "field";
    case Cls::operator_: return "operator";
    case Cls::selection: return "selection";
    case Cls::statistic: return "statistic";
  }
  return "";
}

namespace {
namespace rec = nxr::io::records;

std::optional<std::string> str(const Json& j, const char* k) {
  if (!j.contains(k) || j[k].is_null()) return std::nullopt;
  return j[k].get<std::string>();
}
std::optional<std::int64_t> num(const Json& j, const char* k) {
  if (!j.contains(k) || j[k].is_null()) return std::nullopt;
  return j[k].get<std::int64_t>();
}

Node node_of(Cls cls, Json row) {
  Node n;
  n.cls = cls;
  n.id = row["id"].get<std::string>();
  n.name = row.contains("name") && row["name"].is_string() ? row["name"].get<std::string>() : std::string();
  n.subject_id = row.contains("subject_id") && row["subject_id"].is_string() ? row["subject_id"].get<std::string>() : std::string();
  n.path = str(row, "path");
  n.kind = str(row, "kind");
  n.type = str(row, "type");
  n.manifold_id = str(row, "manifold_id");
  n.from_manifold_id = str(row, "from_manifold_id"); n.to_manifold_id = str(row, "to_manifold_id");
  n.from_selection_id = str(row, "from_selection_id"); n.to_selection_id = str(row, "to_selection_id");
  n.array_path = str(row, "array_path");
  n.dictionary_id = str(row, "dictionary_id"); n.member_of_id = str(row, "member_of_id");
  n.member_ordinal = num(row, "member_ordinal"); n.code = num(row, "code"); n.n_elements = num(row, "n_elements");
  n.n_members = num(row, "n_members"); n.n_vertices = num(row, "n_vertices");
  n.value_type = str(row, "value_type"); n.to_value_type = str(row, "to_value_type"); n.unit = str(row, "unit");
  n.method = str(row, "method"); n.acquisition_id = str(row, "acquisition_id"); n.modified_utc = str(row, "modified_utc");
  n.params = row.contains("params_json") ? row["params_json"] : Json();
  n.producer = row.contains("producer_json") ? row["producer_json"] : Json();
  n.row = std::move(row);
  return n;
}

template <class R>
Node node_of(Cls cls, const R& r) { return node_of(cls, r.to_json()); }

std::vector<Node> list_nodes(const rec::Records& rc, const std::string& subject_id, const Want& w) {
  std::optional<rec::Status> st = w.any_status ? std::nullopt : std::optional<rec::Status>(rec::Status::complete);
  std::vector<Node> out;
  auto bad = [&](const char* member) {
    throw Error(code::invalid_argument, std::string("a Want of class ") + to_string(w.cls) + " does not take `" + member + "`");
  };
  switch (w.cls) {
    case Cls::manifold: {
      if (w.kind || w.from_manifold_id || w.to_manifold_id || w.of_field_id || w.member_of_id || w.dictionary_id || w.code || w.name_like || !w.params.empty()) bad("this filter");
      rec::ManifoldWhere q{.subject_id = subject_id, .type = w.type, .name = w.name, .stored = w.stored, .status = st};
      for (const auto& r : rc.manifolds().list(q)) out.push_back(node_of(w.cls, r));
      break;
    }
    case Cls::field: {
      if (w.type || w.from_manifold_id || w.to_manifold_id || w.member_of_id || w.dictionary_id || w.code || w.name_like || !w.params.empty()) bad("this filter");
      rec::FieldWhere q{.subject_id = subject_id, .kind = w.kind, .manifold_id = w.manifold_id, .of_field_id = w.of_field_id, .name = w.name, .stored = w.stored, .status = st};
      for (const auto& r : rc.fields().list(q)) out.push_back(node_of(w.cls, r));
      break;
    }
    case Cls::operator_: {
      if (w.type || w.manifold_id || w.of_field_id || w.member_of_id || w.dictionary_id || w.code || w.name_like || !w.params.empty()) bad("this filter");
      rec::OperatorWhere q{.subject_id = subject_id, .kind = w.kind, .name = w.name, .from_manifold_id = w.from_manifold_id, .to_manifold_id = w.to_manifold_id, .stored = w.stored, .status = st};
      for (const auto& r : rc.operators().list(q)) out.push_back(node_of(w.cls, r));
      break;
    }
    case Cls::selection: {
      if (w.kind || w.from_manifold_id || w.to_manifold_id) bad("this filter");
      rec::SelectionWhere q{.subject_id = subject_id, .manifold_id = w.manifold_id, .of_field_id = w.of_field_id, .name = w.name, .member_of_id = w.member_of_id,
                            .dictionary_id = w.dictionary_id, .type = w.type, .code = w.code, .name_like = w.name_like, .stored = w.stored};
      for (const auto& p : w.params) q.params.push_back({p.path, p.value});
      q.params_present = w.params_present; q.params_absent = w.params_absent; q.status = st;
      for (const auto& r : rc.selections().list(q)) out.push_back(node_of(w.cls, r));
      break;
    }
    case Cls::statistic: break;   // listed by Subject::find_all
  }
  return out;
}
}  // namespace

struct Subject::Impl {
  Database* db = nullptr;            // writable when made from a writer
  SubjectWriter* writer = nullptr;
  std::string id;
  rec::Records records() const { return rec::Records::of(*db); }
};

Subject Subject::of(SubjectWriter& sub) {
  Subject s;
  s.impl_ = std::make_shared<Impl>();
  s.impl_->db = &sub.db();
  s.impl_->writer = &sub;
  s.impl_->id = sub.id();
  return s;
}

Subject Subject::of(const SubjectPackage& package) {
  Subject s;
  s.impl_ = std::make_shared<Impl>();
  s.impl_->db = const_cast<Database*>(&package.db());   // read-only use: no write call is made on a package-made Subject (commit needs a writer)
  s.impl_->id = text(package.subject_row(), "id");
  return s;
}

Subject Subject::of(const DatasetWriter& ds, const std::string& subject_id) {
  Subject s;
  s.impl_ = std::make_shared<Impl>();
  s.impl_->db = const_cast<Database*>(&ds.db());
  s.impl_->id = subject_id;
  return s;
}

Subject Subject::of(const Database& db, const std::string& subject_id) {
  Subject s;
  s.impl_ = std::make_shared<Impl>();
  s.impl_->db = const_cast<Database*>(&db);   // read-only use, as for a package
  s.impl_->id = subject_id;
  return s;
}

const std::string& Subject::id() const { return impl_->id; }

std::vector<Node> Subject::find_all(const Want& w) const {
  if (w.cls == Cls::statistic) {
    std::vector<Node> out;
    if (!has_table(*impl_->db, "statistic")) return out;   // a dataset made before the statistic table has none
    std::string sql = "SELECT * FROM statistic WHERE subject_id = ?";
    std::vector<Value> args{V(id())};
    if (!w.any_status) sql += " AND status = 'complete'";
    if (w.name) { sql += " AND name = ?"; args.push_back(V(*w.name)); }
    if (w.of_field_id) { sql += " AND of_field_id = ?"; args.push_back(V(*w.of_field_id)); }
    if (w.stored) sql += *w.stored ? " AND path IS NOT NULL" : " AND path IS NULL";
    for (const Row& r : impl_->db->all(sql + " ORDER BY name, id", args)) out.push_back(node_of(w.cls, rec::StatisticRow::from_row(r)));
    return out;
  }
  return list_nodes(impl_->records(), id(), w);
}

std::optional<Node> Subject::find(const Want& w) const {
  auto all = find_all(w);
  if (all.empty()) return std::nullopt;
  if (all.size() > 1) {
    std::string ids;
    for (const Node& n : all) ids += (ids.empty() ? "" : ", ") + n.id;
    throw Error(code::ambiguous, std::string("more than one ") + to_string(w.cls) + (w.name ? " named '" + *w.name + "'" : std::string()) + " matches (" + ids + "); say which");
  }
  return std::move(all.front());
}

Node Subject::require(const Want& w, const std::string& what) const {
  auto n = find(w);
  if (!n) throw Error(code::no_such_row, "no " + std::string(to_string(w.cls)) + " found for " + what + (w.name ? " ('" + *w.name + "')" : std::string()));
  return std::move(*n);
}

std::optional<Node> Subject::by_id(Cls cls, const std::string& row_id) const {
  const auto rc = impl_->records();
  switch (cls) {
    case Cls::manifold: if (auto r = rc.manifolds().row(row_id)) return node_of(cls, *r); break;
    case Cls::field: if (auto r = rc.fields().read(row_id)) return node_of(cls, *r); break;
    case Cls::operator_: if (auto r = rc.operators().read(row_id)) return node_of(cls, *r); break;
    case Cls::selection: if (auto r = rc.selections().read(row_id)) return node_of(cls, *r); break;
    case Cls::statistic: if (auto r = impl_->db->read("statistic", row_id)) return node_of(cls, rec::StatisticRow::from_row(*r)); break;
  }
  return std::nullopt;
}

std::optional<Node> Subject::at_path(Cls cls, const std::string& path) const {
  const std::string table = to_string(cls);
  const auto r = impl_->db->one("SELECT * FROM " + table + " WHERE subject_id = ? AND path = ?", {V(id()), V(path)});
  if (!r) return std::nullopt;
  switch (cls) {
    case Cls::manifold: return node_of(cls, rec::ManifoldRow::from_row(*r));
    case Cls::field: return node_of(cls, rec::FieldRow::from_row(*r));
    case Cls::operator_: return node_of(cls, rec::OperatorRow::from_row(*r));
    case Cls::selection: return node_of(cls, rec::SelectionRow::from_row(*r));
    case Cls::statistic: return node_of(cls, rec::StatisticRow::from_row(*r));
  }
  return std::nullopt;
}

Node Subject::require_id(Cls cls, const std::string& row_id) const {
  auto n = by_id(cls, row_id);
  if (!n) throw Error(code::no_such_row, std::string("no ") + to_string(cls) + " with id " + row_id);
  return std::move(*n);
}

std::vector<Node> Subject::members(const Node& partition) const {
  std::vector<Node> out;
  for (const auto& r : impl_->records().selections().members(partition.id)) out.push_back(node_of(Cls::selection, r));
  return out;
}

std::optional<Node> Subject::member(const Node& partition, std::int64_t ordinal) const {
  if (auto r = impl_->records().selections().member(partition.id, ordinal)) return node_of(Cls::selection, *r);
  return std::nullopt;
}

std::vector<Node> Subject::fields_on_axis(const std::string& factor_manifold_id, std::int64_t ordinal) const {
  std::vector<Node> out;
  for (const Row& r : impl_->db->all("SELECT f.* FROM field f JOIN manifold m ON m.id = f.manifold_id JOIN geometry_dimension d ON d.geometry_id = m.geometry_id AND d.ordinal = ? "
                                     "WHERE f.subject_id = ? AND f.status = 'complete' AND d.manifold_id = ? AND f.path IS NOT NULL ORDER BY f.name",
                                     {V(static_cast<long long>(ordinal)), V(id()), V(factor_manifold_id)}))
    out.push_back(node_of(Cls::field, rec::FieldRow::from_row(r)));
  return out;
}

std::int64_t Subject::entry_count(const std::string& dictionary_id) const {
  return integer(*impl_->db->one("SELECT COUNT(*) AS n FROM dictionary_entry WHERE dictionary_id = ?", {V(dictionary_id)}), "n");
}

std::vector<Span> Subject::spans(const Node& selection) const {
  std::vector<Span> out;
  for (const auto& r : impl_->records().selections().spans(selection.id)) out.push_back({r.ordinal, r.start, r.stop, r.code});
  return out;
}

Geometry Subject::geometry(const std::string& manifold_id) const {
  Geometry g;
  const auto m = impl_->records().manifolds().read(manifold_id);
  if (!m) throw Error(code::no_such_row, "no manifold with id " + manifold_id);
  for (const auto& d : m->dimensions) g.axes.push_back({d.ordinal, d.n_vertices, d.name, d.unit, d.manifold_id, d.spacing, d.origin});
  if (m->coordinate_system) g.coordinate_unit = m->coordinate_system->unit;
  if (m->geometry) g.positions_path = m->geometry->positions_path;
  for (const auto& c : m->cells) g.cells.emplace_back(c.rank, c.path);
  return g;
}

finder::Value Subject::describe(const Node& n) const {
  const auto r = impl_->db->read(to_string(n.cls), n.id);
  if (!r) throw Error(code::no_such_row, std::string("no ") + to_string(n.cls) + " with id " + n.id);
  return finder::value_of(*impl_->db, to_string(n.cls), *r);
}

std::optional<Node> Subject::manifold_of(const Node& n) const {
  if (!n.manifold_id) return std::nullopt;
  return by_id(Cls::manifold, *n.manifold_id);
}

std::vector<std::string> Subject::factors(const std::string& manifold_id) const { return finder::factors(*impl_->db, manifold_id); }

std::vector<MapSlice> Subject::map_slices(const std::string& manifold_id) const {
  std::vector<MapSlice> out;
  for (const auto& m : nxr::io::map_slices(*impl_->db, id(), manifold_id))
    out.push_back({text(m.statistic, "id"), m.name, m.manifold_id, m.index, m.n, m.n_components, m.unit, m.value_type});
  return out;
}

std::optional<MapSlice> Subject::map_slice(const std::string& name) const {
  std::optional<MapSlice> found;
  for (const Row& st : statistics_of(*impl_->db, id())) {
    const Json axes = Json::parse(text(st, "axes_json"), nullptr, false);
    if (!axes.is_array() || axes.empty() || !axes[0].contains("manifold_id")) continue;
    for (const MapSlice& ms : map_slices(axes[0]["manifold_id"].get<std::string>()))
      if (ms.name == name && ms.statistic_id == text(st, "id")) found = ms;
  }
  return found;
}

std::string Subject::node_path(const std::string& manifold_id, const std::string& name) const { return nxr::io::node_path(*impl_->db, manifold_id, name); }
std::string Subject::atlas_node(const std::string& manifold_id, const std::string& name) const { return nxr::io::atlas_node(*impl_->db, manifold_id, name); }

std::optional<Node> Subject::cached(Cls cls, const Json& key, const std::optional<std::string>& path) const {
  if (cls == Cls::statistic && !has_table(*impl_->db, "statistic")) return std::nullopt;   // a dataset made before the statistic table has none
  const std::string table = to_string(cls);
  // a Selection has no producer_json: its producer records how it was made in params_json (the Studio's searchAtlas: params_json.search), the key under "cache"
  const std::string col = cls == Cls::selection ? "params_json" : "producer_json";
  std::string sql = "SELECT * FROM " + table + " WHERE subject_id = ? AND status = 'complete' AND " + col + " IS NOT NULL";
  std::vector<Value> args{V(id())};
  if (path) {
    sql += " AND path = ?";
    args.push_back(V(*path));
  }
  for (const Row& r : impl_->db->all(sql, args)) {
    const Json p = Json::parse(text(r, col), nullptr, false);
    if (!(p.is_object() && p.contains("cache") && p["cache"] == key)) continue;
    switch (cls) {
      case Cls::manifold: return node_of(cls, rec::ManifoldRow::from_row(r));
      case Cls::field: return node_of(cls, rec::FieldRow::from_row(r));
      case Cls::operator_: return node_of(cls, rec::OperatorRow::from_row(r));
      case Cls::selection: return node_of(cls, rec::SelectionRow::from_row(r));
      case Cls::statistic: return node_of(cls, rec::StatisticRow::from_row(r));
    }
  }
  return std::nullopt;
}

std::vector<Json> Subject::measured_kinds(const std::string& selection_id) const {
  std::vector<Json> out;
  for (const Json& k : measurement_kinds(*impl_->db, selection_id)) out.push_back(k);
  return out;
}

std::vector<std::string> Subject::measured_selections(const std::string& of_field_id, const std::string& measure) const {
  std::vector<std::string> out;
  for (const Row& r : impl_->db->all("SELECT DISTINCT selection_id FROM selection_measurement WHERE of_field_id = ? AND measure = ? ORDER BY selection_id", {V(of_field_id), V(measure)}))
    out.push_back(text(r, "selection_id"));
  return out;
}

bool Subject::has_measures(const std::string& selection_id, const std::string& of_field_id) const {
  return impl_->db->one("SELECT 1 AS x FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? LIMIT 1", {V(selection_id), V(of_field_id)}).has_value();
}

bool Subject::has_measure(const std::string& selection_id, const std::string& measure, const std::optional<std::string>& of_field_id) const {
  std::string sql = "SELECT 1 AS x FROM selection_measurement WHERE selection_id = ? AND measure = ?";
  std::vector<Value> args{V(selection_id), V(measure)};
  if (of_field_id) { sql += " AND of_field_id = ?"; args.push_back(V(*of_field_id)); }
  return impl_->db->one(sql + " LIMIT 1", args).has_value();
}

std::optional<Node> Subject::by_id_or_name(Cls cls, const std::string& key) const {
  const std::string t = to_string(cls);
  const auto r = impl_->db->one("SELECT id FROM " + t + " WHERE subject_id = ? AND (id = ? OR name = ?) AND status = 'complete' ORDER BY (id = ?) DESC",
                                {V(impl_->id), V(key), V(key), V(key)});
  if (!r) return std::nullopt;
  return by_id(cls, text(*r, "id"));
}

std::optional<std::string> Subject::line_named(const std::string& name) const {
  const auto r = impl_->db->one("SELECT id FROM manifold WHERE subject_id = ? AND name = ? AND type = 'line'", {V(impl_->id), V(name)});
  if (!r) return std::nullopt;
  return text(*r, "id");
}

namespace {
// the time lines (a product's line factors) of the complete recordings of a subject
const char* kRecordingLines =
    "SELECT gd.manifold_id AS line FROM field f JOIN manifold m ON m.id = f.manifold_id JOIN geometry_dimension gd ON gd.geometry_id = m.geometry_id "
    "JOIN manifold t ON t.id = gd.manifold_id WHERE f.subject_id = ? AND f.kind = 'recording' AND f.status = 'complete' AND t.type = 'line'";
}  // namespace

std::vector<std::string> Subject::recording_time_lines() const {
  std::vector<std::string> out;
  for (const Row& r : impl_->db->all(kRecordingLines, {V(impl_->id)}))
    if (std::find(out.begin(), out.end(), text(r, "line")) == out.end()) out.push_back(text(r, "line"));
  return out;
}

std::optional<std::string> Subject::recording_time_line(const std::string& recording) const {
  const auto r = impl_->db->one(std::string(kRecordingLines) + " AND f.name = ?", {V(impl_->id), V(recording)});
  if (!r) return std::nullopt;
  return text(*r, "line");
}

NodeCounts Subject::node_counts() const {
  auto count = [&](const char* t) { return integer(*impl_->db->one(std::string("SELECT count(*) AS n FROM ") + t + " WHERE subject_id = ?", {V(impl_->id)}), "n"); };
  return NodeCounts{count("manifold"), count("field"), count("operator"), count("selection")};
}

Json Subject::read_measurements(const Json& q) const {
  if (impl_->writer) return query_measurements(*impl_->db, impl_->writer->store(), q);
  if (q.contains("spaceLevel") || q.contains("timeLevel") || q.contains("band") || q.contains("spatialBand"))
    throw Error(code::invalid_argument, "read_measurements: a pyramid's levels are read through the subject's store (a Subject made from a writer)");
  return query_measurements(*impl_->db, q);
}

Unit Subject::unit(const By& by) const {
  if (!impl_->writer) throw Error(code::not_writable, "unit: this Subject was opened for reading");
  return Unit(impl_, by);
}

void Subject::commit(const std::function<void()>& body, const By& by) const {
  Unit u = unit(by);
  body();   // an exception leaves `u` to roll back
  u.commit();
}

// One database transaction around the composition: the writer's own abort undoes the nodes it made; the transaction undoes every other row written inside
// (measurements on an existing selection, parameters), so a failed unit stores nothing.
Unit::Unit(std::shared_ptr<Subject::Impl> impl, By by) : impl_(std::move(impl)), by_(std::move(by)) {
  impl_->db->tx_begin();
  open_ = true;
}
Unit::Unit(Unit&& o) noexcept : impl_(std::move(o.impl_)), by_(std::move(o.by_)), open_(o.open_) { o.open_ = false; }
Unit::~Unit() { rollback(); }
const By& Unit::by() const { return by_; }

void Unit::commit() {
  if (!open_) throw Error(code::not_writable, "unit: already committed or rolled back");
  try {
    impl_->writer->complete();
  } catch (...) {
    rollback();
    throw;
  }
  open_ = false;
  try {
    impl_->db->tx_commit();
  } catch (...) {
    try { impl_->writer->abort(); } catch (...) {}
    throw;
  }
}

void Unit::rollback() noexcept {
  if (!open_) return;
  open_ = false;
  try { impl_->writer->abort(); } catch (...) {}
  impl_->db->tx_rollback();
}

std::int64_t Unit::write_measurements(const Json& rows) {
  if (!open_) throw Error(code::not_writable, "unit: closed");
  return nxr::io::write_measurements(*impl_->db, rows, by_.tool);
}
std::int64_t Unit::remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) {
  if (!open_) throw Error(code::not_writable, "unit: closed");
  Json where{{"selectionId", selection_id}};
  if (of_field_id) where["ofFieldId"] = *of_field_id;
  if (measure) where["measure"] = *measure;
  return nxr::io::delete_measurements(*impl_->db, where);
}
void Unit::set_params(const std::string& selection_id, const Json& params) {
  if (!open_) throw Error(code::not_writable, "unit: closed");
  impl_->db->execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(params)), V(selection_id)});
}


std::int64_t Subject::write_measurements(const Json& rows, const std::optional<std::string>& by) const {
  std::int64_t n = 0;
  impl_->db->tx([&] { n = nxr::io::write_measurements(*impl_->db, rows, by); });
  return n;
}

std::int64_t Subject::remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) const {
  Json where{{"selectionId", selection_id}};
  if (of_field_id) where["ofFieldId"] = *of_field_id;
  if (measure) where["measure"] = *measure;
  return nxr::io::delete_measurements(*impl_->db, where);
}

void Subject::set_params(const std::string& selection_id, const Json& params) const {
  impl_->db->execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(params)), V(selection_id)});
}

}  // namespace nxr::io::task
