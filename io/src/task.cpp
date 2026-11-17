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
#include "nxr/bst/frequency.h"
#include "nxr/io/store.h"

#include <cmath>

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

std::vector<DictionaryLevel> Subject::dictionary_entries(const std::string& dictionary_id) const {
  std::vector<DictionaryLevel> out;
  for (const auto& r : impl_->db->all("SELECT code, name, color_json FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {V(dictionary_id)})) {
    DictionaryLevel e;
    e.code = integer(r, "code");
    e.name = text(r, "name");
    if (const auto c = opt_text(r, "color_json")) {
      const Json j = Json::parse(*c, nullptr, false);
      if (j.is_array() && j.size() >= 3 && j[0].is_number() && j[1].is_number() && j[2].is_number()) e.color = std::array<int, 3>{j[0].get<int>(), j[1].get<int>(), j[2].get<int>()};
    }
    out.push_back(std::move(e));
  }
  return out;
}

std::optional<Origin> Subject::origin(const Node& n) const {
  if (!has_table(*impl_->db, "origin")) return std::nullopt;
  const auto o = origin_of(*impl_->db, to_string(n.cls), n.id);
  if (!o) return std::nullopt;
  return Origin{o->source, o->licence, o->attribution, o->redistributable};
}

std::vector<KeyDigest> Subject::key_digests(const std::string& prefix) const {
  std::vector<KeyDigest> out;
  if (!has_table(*impl_->db, "key_digest")) return out;
  for (const auto& r : impl_->db->all("SELECT key, size_bytes, sha256 FROM key_digest WHERE key = ? OR (key >= ? AND key < ?) ORDER BY key", {V(prefix), V(prefix + "/"), V(prefix + "0")}))
    out.push_back({text(r, "key"), integer(r, "size_bytes"), text(r, "sha256")});
  return out;
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
  // a row that names how it was made keeps that (a result's rows may carry more than the unit's tool); the others are recorded as computed by the unit's tool
  Json named = rows;
  if (named.is_array()) for (auto& r : named) if (r.is_object() && (!r.contains("computed_by") || r["computed_by"].is_null())) r["computed_by"] = by_.tool;
  return nxr::io::write_measurements(*impl_->db, named, std::nullopt);
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


// ── THE TYPED WRITES (spec docs/specs/2026-11-17-definition-outputs.md, section 4): each result kind written through the subject's writer inside the unit's
// transaction, with the unit's producer record (and so its cache key), the output's name and the result's own producer facts.
namespace {
ArrayView view_of(const Values& v, const std::string& what) {
  std::int64_t n = 1;
  for (const auto d : v.shape) n *= d;
  if (v.f32.empty() == v.f64.empty()) throw Error(code::invalid_argument, what + ": values in float32 or float64 (one of the two)");
  if (n != v.size()) throw Error(code::invalid_argument, what + ": " + std::to_string(v.size()) + " values for the shape of " + std::to_string(n));
  return v.f32.empty() ? view(v.f64, v.shape) : view(v.f32, v.shape);
}
Json producer_of(const By& by, const std::string& output, const Json& own) {
  Json p = by.producer.is_object() ? by.producer : Json::object();
  if (!output.empty()) p["output"] = output;
  if (own.is_object()) for (const auto& [k, v] : own.items()) p[k] = v;
  return p;
}
void need_open(bool open) { if (!open) throw Error(code::not_writable, "unit: closed"); }
}  // namespace

std::string Unit::path_of(const FieldOut& f) const { return f.path ? *f.path : node_path(*impl_->db, f.manifold_id, f.name); }
std::string Unit::path_of(const OperatorOut& o) const { return o.path ? *o.path : safe_name(o.name); }
std::string Unit::path_of(const SelectionOut& s) const { return s.path ? *s.path : node_path(*impl_->db, s.manifold_id, s.name); }
std::string Unit::path_of(const StatisticOut& s) const {
  if (s.path) return *s.path;
  if (s.of_field_id)
    if (const auto r = impl_->db->one("SELECT path FROM field WHERE id = ?", {V(*s.of_field_id)})) return text(*r, "path") + "_statistics/" + safe_name(s.name);
  return "statistics/" + safe_name(s.name);
}

std::string Unit::replace(Cls cls, const std::string& path, bool replace, const std::string& what) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  const auto r = sub.find(to_string(cls), path);
  if (!r) return {};
  if (!replace) throw Error(code::already_exists, sub.name() + ": " + what + " (" + path + ") already exists (made by another job); pass --replace");
  std::string moved;
  const std::string loc = cls == Cls::selection ? opt_text(*r, "array_path").value_or("") : opt_text(*r, "path").value_or("");
  if (!loc.empty() && sub.store().exists_prefix(loc)) {
    // a folder of its own for each replacement of the same node the same day (`_replaced-<date>`, then `-2`, `-3`, …): never over what an earlier one moved
    DatasetWriter& ds = sub.dataset();
    const std::string date = now_utc(false).substr(0, 10);
    auto aside = [&](int n) -> std::pair<std::string, std::string> {
      const std::string stem = "_replaced-" + date + (n > 1 ? "-" + std::to_string(n) : std::string());
      if (ds.is_package()) return {join_key(sub.prefix(), stem), (ds.root() / stem).string()};
      const std::string base = stem + "/" + text(sub.row(), "path");
      return {base, (ds.root() / base).string()};
    };
    int n = 1;
    auto a = aside(n);
    while (ds.node_store().exists_prefix(join_key(a.first, loc))) a = aside(++n);
    sub.datastore().rename_prefix(join_key(sub.prefix(), loc), join_key(a.first, loc));
    moved = a.second;
  }
  sub.remove(to_string(cls), text(*r, "id"));
  return moved;
}

std::string Unit::field(const FieldOut& f) {
  need_open(open_);
  FieldSpec fs;
  fs.name = f.name; fs.path = path_of(f); fs.kind = f.kind; fs.manifold_id = f.manifold_id; fs.value_type = f.value_type;
  if (!f.data.shape.empty()) fs.data = view_of(f.data, "field " + f.name);   // no data: a field with no array (its measurements carry the values)
  fs.array.compress = false;
  if (f.acquisition) fs.acquisition = *f.acquisition;
  if (f.unit) fs.cols.set("unit", *f.unit);
  if (f.description) fs.cols.set("description", *f.description);
  if (f.derived_from_id) fs.cols.set("derived_from_id", *f.derived_from_id);
  if (f.of_operator_id) fs.cols.set("of_operator_id", *f.of_operator_id);
  fs.cols.set("source", by_.tool);
  fs.cols.json("producer_json", producer_of(by_, f.output, f.producer));
  return impl_->writer->field(std::move(fs));
}

std::string Unit::statistic(const StatisticOut& s) {
  need_open(open_);
  StatisticSpec st;
  st.name = s.name; st.path = path_of(s); st.measure = s.measure; st.method = s.method; st.value_type = s.value_type;
  st.of_field_id = s.of_field_id; st.of_statistic_id = s.of_statistic_id; st.partition_selection_id = s.partition_selection_id;
  st.axes = s.axes; st.components = s.components; st.span = s.span; st.scaling = s.scaling; st.options = s.options;
  st.n_members = s.n_members; st.members_skipped = s.members_skipped; st.dof = s.dof; st.detached = s.detached;
  st.unit = s.unit; st.description = s.description; st.label = s.label; st.acquisition = s.acquisition;
  st.data = view_of(s.data, "statistic " + s.name);
  st.producer = producer_of(by_, s.output, s.producer);
  return impl_->writer->statistic(std::move(st));
}

std::string Unit::operator_(const OperatorOut& o) {
  need_open(open_);
  if (o.data.shape.size() != 2) throw Error(code::invalid_argument, "operator " + o.name + ": a dense [rows x cols] array");
  OperatorSpec op;
  op.name = o.name; op.path = path_of(o); op.kind = o.kind; op.from_manifold_id = o.from_manifold_id; op.layout = "dense";
  op.n_rows = o.data.shape[0]; op.n_cols = o.data.shape[1];
  op.data = view_of(o.data, "operator " + o.name);
  op.array.compress = false;
  if (o.acquisition) op.acquisition = *o.acquisition;
  op.created_utc = now_utc(false);
  op.cols.set("to_manifold_id", o.to_manifold_id).set("from_value_type", o.from_value_type).set("to_value_type", o.to_value_type);
  if (o.from_selection_id) op.cols.set("from_selection_id", *o.from_selection_id);
  if (o.to_selection_id) op.cols.set("to_selection_id", *o.to_selection_id);
  if (o.derived_from_id) op.cols.set("derived_from_id", *o.derived_from_id);
  if (o.of_statistic_id) op.cols.set("of_statistic_id", *o.of_statistic_id);
  if (o.inverts_id) op.cols.set("inverts_id", *o.inverts_id).set("inverse_kind", o.inverse_kind.value_or("regularized"));
  if (o.description) op.cols.set("description", *o.description);
  if (!o.params.is_null()) op.cols.json("params_json", o.params);
  op.cols.json("producer_json", producer_of(by_, o.output, o.producer));
  return impl_->writer->operator_(std::move(op));
}

std::int64_t Unit::measurements(const MeasurementOut& m) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  std::optional<Json> band;
  if (m.band.is_object()) band = nxr::bst::band_ref(*impl_->db, m.band.value("name", std::string()), m.band.value("lo_hz", 0.0), m.band.value("hi_hz", 0.0));
  std::int64_t n = 0;
  for (const auto& x : m.measures) {
    const std::string path = m.path + "/" + x.measure;
    sub.write_array(path, view_of(x.values, "measurement " + x.measure));
    Cols c;
    c.set("measure", x.measure).set("array_path", path).set("unit", x.unit).set("computed_by", by_.tool);
    if (!m.of_field_id.empty()) c.set("of_field_id", m.of_field_id);        // none: the members' own (a size)
    if (band && !(*band)["selection_id"].is_null()) c.set("within_selection_id", (*band)["selection_id"].get<std::string>()).set("within_code", (*band)["code"].get<std::int64_t>());
    sub.measurement(m.selection_id, c);
    ++n;
  }
  return n;
}

std::string Unit::selection(const SelectionOut& s) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  SelectionSpec ss;
  ss.name = s.name; ss.path = path_of(s); ss.manifold_id = s.manifold_id;
  if (s.acquisition) ss.acquisition = *s.acquisition;
  Json params = s.params.is_null() ? Json::object() : s.params;
  std::vector<std::int32_t> codes;
  if (s.type == "partition" || s.type == "set") {
    if (s.codes.empty()) throw Error(code::invalid_argument, "selection " + s.name + ": a " + s.type + " has a code per element");
    codes.assign(s.codes.begin(), s.codes.end());
    ss.type = "set";                                 // a partition is a set Selection: a code per element
    ss.data = view(codes, {static_cast<std::int64_t>(codes.size())});
  } else if (s.type == "indices") {                  // a pick of elements, in order (the channels a covariance covers)
    codes.assign(s.codes.begin(), s.codes.end());
    ss.type = "indices";
    ss.data = view(codes, {static_cast<std::int64_t>(codes.size())});
    ss.picked = true;
  } else if (s.type == "cells") {                    // a closed-form set (the joint cells of space x time): its rule in params, no element stored
    if (!s.n_members) throw Error(code::invalid_argument, "selection " + s.name + ": cells name their number (n_members)");
    ss.type = "set";
    ss.no_elements = true;
    ss.cols.set("cell", "0,1").set("n_members", *s.n_members);
    if (s.label) ss.cols.set("label", *s.label);
  } else if (s.type == "spans") {
    ss.type = "spans";
    std::vector<SpanSpec> sp;
    for (const auto& x : s.spans) sp.push_back(SpanSpec{x.start, x.stop, x.code});
    ss.spans = sp;
    ss.cols.set("cell", "1");
  } else if (s.type == "tiling") {
    if (!s.tiling) throw Error(code::invalid_argument, "selection " + s.name + ": a tiling needs its origin, width, hop and count");
    const Tiling& t = *s.tiling;
    ss.type = "spans";
    ss.cols.set("cell", "1").set("unit", t.unit).set("n_members", t.count + t.skipped);
    params["tiling"] = Json{{"origin", t.origin}, {"width", t.width}, {"hop", t.hop}, {"count", t.count + t.skipped}};
    params["end"] = t.end;
    params["segments_skipped"] = t.skipped;
  } else {
    throw Error(code::invalid_argument, "selection " + s.name + ": type " + s.type + " (partition, set, indices, spans, tiling or cells)");
  }
  if (!s.entries.empty()) {
    DictionarySpec d;
    d.name = s.name;
    d.names = s.entries;
    ss.cols.set("dictionary_id", sub.dictionary(d).id);
  }
  if (s.of_field_id) ss.cols.set("of_field_id", *s.of_field_id);
  if (s.description) ss.cols.set("description", *s.description);
  if (s.label) ss.cols.set("label", *s.label);
  // a Selection has no producer column: its producer record is in its parameters, the cache key under "cache" (the Studio's searchAtlas: params_json.search)
  const Json pj = producer_of(by_, s.output, s.producer);
  if (pj.contains("cache")) params["cache"] = pj["cache"];
  params["producer"] = pj;
  ss.cols.json("params_json", params);
  return sub.selection(std::move(ss));
}

std::string Unit::line(const LineOut& l) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  Database& db = *impl_->db;
  auto named = [&](const std::string& n) { return db.one("SELECT m.id AS id FROM manifold m WHERE m.subject_id = ? AND m.name = ? AND m.type = 'line'", {V(impl_->id), V(n)}); };
  auto fits = [&](const std::string& lid) {
    const auto d = db.one("SELECT d.spacing AS spacing, d.n_vertices AS n, d.origin AS origin FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? AND d.ordinal = 0", {V(lid)});
    return d && integer(*d, "n") == l.n && std::abs(std::stod(text(*d, "spacing")) - l.spacing) <= 1e-12 * std::max(1.0, std::abs(l.spacing)) &&
           std::abs(std::stod(text(*d, "origin")) - l.origin) <= 1e-12 * std::max(1.0, std::abs(l.origin));
  };
  std::string name = l.name;
  if (const auto r = named(name)) {
    if (fits(text(*r, "id"))) return text(*r, "id");
    if (l.alt.empty()) throw Error(code::already_exists, sub.name() + ": the line " + name + " exists with another grid; give another name");
    name = l.alt;
    if (const auto r2 = named(name)) {
      if (fits(text(*r2, "id"))) return text(*r2, "id");
      throw Error(code::already_exists, sub.name() + ": the lines " + l.name + " and " + name + " exist with other grids");
    }
  }
  ManifoldSpec ms;
  ms.name = name;
  ms.path = folder_for("line", "", l.folder) + "/" + name;
  ms.type = "line";
  ms.n_vertices = l.n;
  ms.topology = {"lattice", 1, 1, std::nullopt, {}};
  const auto lc = lattice_counts({l.n});
  if (l.acquisition_id) ms.cols.set("acquisition_id", *l.acquisition_id); else ms.cols.null("acquisition_id");
  if (lc.n_edges) ms.cols.set("n_edges", *lc.n_edges); else ms.cols.null("n_edges");
  if (lc.n_faces) ms.cols.set("n_faces", *lc.n_faces); else ms.cols.null("n_faces");
  if (lc.n_volumes) ms.cols.set("n_volumes", *lc.n_volumes); else ms.cols.null("n_volumes");
  if (l.derived_from_id) ms.cols.set("derived_from_id", *l.derived_from_id);
  else if (l.dimension == "frequency") if (const auto fr = nxr::bst::find_frequency(db)) ms.cols.set("derived_from_id", text(fr->manifold, "id"));
  GeometrySpec g;
  g.form = "closed_form";
  g.cs_key = l.cs_key;
  DimensionSpec d;
  d.name = l.dimension; d.n_vertices = l.n; d.spacing = l.spacing; d.origin = l.origin; d.unit = l.unit;
  g.dimensions.push_back(d);
  ms.geometry = g;
  return sub.manifold(std::move(ms));
}


std::string Unit::surface(const SurfaceOut& o) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  if (o.vertices.shape.size() != 2 || o.vertices.shape[1] != 3 || o.vertices.f64.empty()) throw Error(code::invalid_argument, "surface " + o.name + ": vertices [n x 3] in float64");
  if (o.vertices.size() != o.vertices.shape[0] * 3) throw Error(code::invalid_argument, "surface " + o.name + ": vertices hold " + std::to_string(o.vertices.size()) + " values, not n x 3");
  if (o.faces.empty() || o.faces.size() % 3) throw Error(code::invalid_argument, "surface " + o.name + ": faces [m x 3]");
  const std::int64_t n = o.vertices.shape[0], m = static_cast<std::int64_t>(o.faces.size() / 3);
  for (const auto f : o.faces) if (f < 0 || f >= n) throw Error(code::invalid_argument, "surface " + o.name + ": a face names vertex " + std::to_string(f) + " of " + std::to_string(n));
  const std::string path = o.path ? *o.path : "source/" + safe_name(o.name);
  if (const auto r = sub.find("manifold", path)) {   // found again: the same arrays are the same surface
    const Node have = node_of(Cls::manifold, rec::ManifoldRow::from_row(*r));
    if (have.n_vertices && *have.n_vertices == n) return have.id;
    throw Error(code::already_exists, sub.name() + ": the surface " + o.name + " (" + path + ") exists with another mesh; give another name");
  }
  ManifoldSpec ms;
  ms.name = o.name;
  ms.path = path;
  ms.type = "surface";
  ms.n_vertices = n;
  ms.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{2, path + "/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = o.cs_key;
  g.positions_path = path + "/vertices";
  ms.geometry = g;
  if (o.acquisition_id) ms.cols.set("acquisition_id", *o.acquisition_id);
  if (o.derived_from_id) ms.cols.set("derived_from_id", *o.derived_from_id);
  if (o.description) ms.cols.set("comment", *o.description);
  ms.cols.json("producer_json", producer_of(by_, o.output, o.producer));
  const Values& v = o.vertices;
  const std::vector<std::int32_t>& f = o.faces;
  ms.populate = [&sub, &v, &f, path, n, m](Store&, const std::string&) {
    sub.write_array(path + "/vertices", view(v.f64, {n, 3}));
    sub.write_array(path + "/faces", view(f, {m, 3}));
  };
  return sub.manifold(std::move(ms));
}

std::string Unit::product(const ProductOut& p) {
  need_open(open_);
  if (p.factors.size() < 2) throw Error(code::invalid_argument, "product: two factors or more");
  return impl_->writer->product(p.factors, p.acquisition_id);
}

}  // namespace nxr::io::task
