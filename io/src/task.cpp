// nxr-io — the task-level API (io/include/nxr/io/task/task.h) on the repositories and io's helpers. The only place orchestration's questions become SQL.
#include "nxr/io/task/task.h"

#include <algorithm>

#include "nxr/io/array_reader.h"
#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
#include "nxr/io/layout.h"
#include "nxr/io/measures.h"
#include "nxr/io/records/records.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/io/selection_reads.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/statistic.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"
#include "nxr/bst/frequency.h"
#include "nxr/io/store.h"

#include <cmath>
#include <limits>
#include <set>

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
  return selection::lookup_size(*impl_->db, dictionary_id);
}

std::vector<DictionaryLevel> Subject::dictionary_entries(const std::string& dictionary_id) const {
  std::vector<DictionaryLevel> out;
  for (const selection::Entry& x : selection::load_lookup(*impl_->db, dictionary_id)) {
    DictionaryLevel e;
    e.code = x.code;
    e.name = x.name;
    e.color = x.color;
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
  return selection::selections_measured(*impl_->db, of_field_id, measure);
}

bool Subject::has_measures(const std::string& selection_id, const std::string& of_field_id) const {
  selection::MeasurementFilter f;
  f.of_field_id = of_field_id;
  return selection::has_measurement(*impl_->db, selection_id, f);
}

bool Subject::has_measure(const std::string& selection_id, const std::string& measure, const std::optional<std::string>& of_field_id) const {
  selection::MeasurementFilter f;
  f.measure = measure;
  f.of_field_id = of_field_id;
  return selection::has_measurement(*impl_->db, selection_id, f);
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

std::optional<std::string> Subject::manifold_named(const std::string& name) const {
  const auto r = impl_->db->one("SELECT id FROM manifold WHERE subject_id = ? AND name = ?", {V(impl_->id), V(name)});
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
  return selection::save_measurements(*impl_->db, named, std::nullopt);
}
std::int64_t Unit::remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) {
  if (!open_) throw Error(code::not_writable, "unit: closed");
  return selection::remove_measurements(*impl_->db, selection_id, of_field_id, measure);
}
void Unit::set_params(const std::string& selection_id, const Json& params) {
  if (!open_) throw Error(code::not_writable, "unit: closed");
  selection::set_params(*impl_->db, selection_id, params);
}


std::int64_t Subject::write_measurements(const Json& rows, const std::optional<std::string>& by) const {
  std::int64_t n = 0;
  impl_->db->tx([&] { n = selection::save_measurements(*impl_->db, rows, by); });
  return n;
}

std::int64_t Subject::remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) const {
  return selection::remove_measurements(*impl_->db, selection_id, of_field_id, measure);
}

void Subject::set_params(const std::string& selection_id, const Json& params) const { selection::set_params(*impl_->db, selection_id, params); }

std::optional<Node> Subject::stored_operator(const Json& key) const { return cached(Cls::operator_, key); }

std::vector<Node> Subject::fields_of_operator(const std::string& operator_id) const {
  std::vector<Node> out;
  for (const Row& r : impl_->db->all("SELECT * FROM field WHERE subject_id = ? AND of_operator_id = ? AND status = 'complete' ORDER BY name, id", {V(id()), V(operator_id)}))
    out.push_back(node_of(Cls::field, rec::FieldRow::from_row(r)));
  return out;
}

std::vector<Chart> Subject::charts(const std::string& manifold_id) const {
  std::vector<Chart> out;
  for (const Row& r : impl_->db->all("SELECT c2.* FROM manifold m JOIN geometry g ON g.id = m.geometry_id JOIN coordinate_system c1 ON c1.id = g.coordinate_system_id "
                                     "JOIN coordinate_system c2 ON c2.space_id = c1.space_id WHERE m.id = ? ORDER BY c2.is_reference DESC, c2.name", {V(manifold_id)})) {
    Chart c;
    c.id = text(r, "id"); c.name = text(r, "name"); c.space_id = text(r, "space_id");
    c.unit = opt_text(r, "unit"); c.transition = opt_text(r, "transition"); c.description = opt_text(r, "description");
    c.is_reference = integer(r, "is_reference") != 0;
    if (const auto tj = opt_text(r, "transition_json")) c.transition_json = Json::parse(*tj, nullptr, false);
    out.push_back(std::move(c));
  }
  return out;
}


// ── THE TYPED WRITES (spec docs/specs/2026-11-17-definition-outputs.md, section 4): each result kind written through the subject's writer inside the unit's
// transaction, with the unit's producer record (and so its cache key), the output's name and the result's own producer facts.
namespace {
ArrayView view_of(const Values& v, const std::string& what) {
  std::int64_t n = 1;
  for (const auto d : v.shape) n *= d;
  const int filled = !v.f32.empty() + !v.f64.empty() + !v.i32.empty() + !v.u8.empty();
  if (filled != 1) throw Error(code::invalid_argument, what + ": values in one of float32, float64, int32 or uint8");
  if (n != v.size()) throw Error(code::invalid_argument, what + ": " + std::to_string(v.size()) + " values for the shape of " + std::to_string(n));
  if (!v.f32.empty()) return view(v.f32, v.shape);
  if (!v.i32.empty()) return view(v.i32, v.shape);
  if (!v.u8.empty()) return view(v.u8, v.shape);
  return view(v.f64, v.shape);
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
std::string Unit::path_of(const PointsOut& p) const {
  return p.path ? *p.path : folder_for("points", "", std::nullopt, p.operator_name.empty() ? std::optional<std::string>() : std::optional<std::string>(p.operator_name)) + "/" + safe_name(p.name);
}
std::string Unit::path_of(const ComplexOut& c) const { return c.path ? *c.path : "source/" + safe_name(c.name); }
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
  if (cls == Cls::manifold) {
    // what lies on a manifold goes with it, each moved aside with its own arrays first (a row removed alone would leave its arrays where the new ones are written)
    const std::string mid = text(*r, "id");
    std::vector<std::pair<Cls, std::string>> deps;
    for (const Row& d : sub.db().all("SELECT path FROM field WHERE manifold_id = ? AND path IS NOT NULL", {V(mid)})) deps.emplace_back(Cls::field, text(d, "path"));
    for (const Row& d : sub.db().all("SELECT path FROM operator WHERE (from_manifold_id = ? OR to_manifold_id = ?) AND path IS NOT NULL", {V(mid), V(mid)})) deps.emplace_back(Cls::operator_, text(d, "path"));
    for (const Row& d : sub.db().all("SELECT array_path AS path FROM selection WHERE manifold_id = ? AND array_path IS NOT NULL", {V(mid)})) deps.emplace_back(Cls::selection, text(d, "path"));
    for (const auto& [c, p] : deps) this->replace(c, p, true, what);
  }
  std::string moved;
  // a node's folder moved aside, never deleted: a folder of its own for each replacement of the same node the same day (`_replaced-<date>`, then `-2`, `-3`, …): never over what an earlier one moved
  const auto move_aside = [&](const std::string& loc) {
    if (loc.empty() || !sub.store().exists_prefix(loc)) return;
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
  };
  move_aside(cls == Cls::selection ? opt_text(*r, "array_path").value_or("") : opt_text(*r, "path").value_or(""));
  if (cls == Cls::selection && text(*r, "type") == "ladder") {
    // a grid's ladder: its level rows go with it, and their arrays and nodes (tile codes, areas, centres, adjacency), as they were written together, are MOVED ASIDE with it (--replace
    // never deletes; a constant-Q ladder's levels hold no node)
    std::set<std::string> nodes;
    for (const Row& m : selection::member_rows(sub.db(), text(*r, "id"), false)) {
      const Json p = Json::parse(opt_text(m, "params_json").value_or("{}"), nullptr, false);
      if (p.is_object() && p.contains("node") && p["node"].is_string()) nodes.insert(p["node"].get<std::string>());
      move_aside(opt_text(m, "array_path").value_or(""));
      sub.remove("selection", text(m, "id"));
    }
    for (const std::string& n : nodes) move_aside(n);
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
  if (f.cell) fs.cols.set("cell", *f.cell);
  if (f.description) fs.cols.set("description", *f.description);
  if (f.derived_from_id) fs.cols.set("derived_from_id", *f.derived_from_id);
  if (f.of_operator_id) fs.cols.set("of_operator_id", *f.of_operator_id);
  if (f.by_operator_id) fs.cols.set("by_operator_id", *f.by_operator_id);
  fs.cols.set("source", by_.tool);
  fs.cols.json("producer_json", producer_of(by_, f.output, f.producer));
  return impl_->writer->field(std::move(fs));
}

std::string Unit::statistic(const StatisticOut& s) {
  need_open(open_);
  // a value per MEMBER of a selection is a MEASUREMENT (one table for values over members, spec 6.1): a statistic holds values on axes that are not a selection's members (a PSD on channels x frequency,
  // a covariance, a histogram's bins, the summaries across members). Its partition is PROVENANCE, never an axis.
  if (s.axes.is_array())
    for (const auto& ax : s.axes)
      if (ax.is_object() && ax.contains("selection_id") && !ax["selection_id"].is_null())
        throw Error(code::invalid_argument, "statistic " + s.name + ": an axis lies on the members of a selection (" + ax["selection_id"].dump() + "): values over a selection's members are a measurement (Unit::measurements), not a statistic");
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
  OperatorSpec op;
  op.name = o.name; op.path = path_of(o); op.kind = o.kind; op.from_manifold_id = o.from_manifold_id;
  if (o.sparse) {
    const OperatorOut::Sparse& sp = *o.sparse;
    if (sp.layout != "csr" && sp.layout != "csc") throw Error(code::invalid_argument, "operator " + o.name + ": a sparse operator is csr or csc");
    const std::int64_t major = sp.layout == "csr" ? sp.n_rows : sp.n_cols, nnz = o.data.size();
    if (static_cast<std::int64_t>(sp.indptr.size()) != major + 1 || static_cast<std::int64_t>(sp.indices.size()) != nnz || sp.indptr.back() != nnz)
      throw Error(code::invalid_argument, "operator " + o.name + ": indptr, indices and values do not describe " + std::to_string(nnz) + " values in " + std::to_string(major) + " " + (sp.layout == "csr" ? "rows" : "columns"));
    op.layout = sp.layout; op.n_rows = sp.n_rows; op.n_cols = sp.n_cols;
    op.sparse = SparseArrays{{{"indptr", view(sp.indptr, {major + 1})}, {"indices", view(sp.indices, {nnz})}, {"data", view_of(o.data, "operator " + o.name)}}};
    op.cols.set("nnz", nnz).set("data_type", o.data.f32.empty() ? "float64" : "float32");
  } else {
    if (o.data.shape.size() != 2) throw Error(code::invalid_argument, "operator " + o.name + ": a dense [rows x cols] array");
    op.layout = "dense";
    op.n_rows = o.data.shape[0]; op.n_cols = o.data.shape[1];
    op.data = view_of(o.data, "operator " + o.name);
    if (!o.chunk_shape.empty()) op.array.chunks = o.chunk_shape;
    if (!o.shard_shape.empty()) op.array.shards = o.shard_shape;
  }
  op.array.compress = false;
  if (o.from_cell) op.cols.set("from_cell", *o.from_cell);
  if (o.to_cell) op.cols.set("to_cell", *o.to_cell);
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
  Json op_producer = producer_of(by_, o.output, o.producer);
  if (o.cache_key) op_producer["cache"] = *o.cache_key;   // its own provenance: found by Subject::stored_operator
  op.cols.json("producer_json", op_producer);
  return impl_->writer->operator_(std::move(op));
}

std::int64_t Unit::measurements(const MeasurementOut& m) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  std::optional<Json> band;
  if (m.band.is_object()) band = nxr::bst::band_ref(*impl_->db, m.band.value("name", std::string()), m.band.value("lo_hz", 0.0), m.band.value("hi_hz", 0.0));
  std::int64_t n = 0;
  if (m.of_statistic_id) ensure_statistics(*impl_->db);       // the measurement of a statistic's slice: its columns added in place first
  Json attrs = m.attributes.is_object() ? m.attributes : Json::object();
  if (m.value_type != "scalar") attrs["value_type"] = m.value_type;
  if (m.components.is_array() && !m.components.empty()) attrs["components"] = m.components;
  if (m.level) attrs["level"] = *m.level;
  for (const auto& x : m.measures) {
    const std::string path = m.path + "/" + x.measure;
    const ArrayView v = view_of(x.values, "measurement " + x.measure);
    const ArrayMeta laid = sub.write_array(path, v);
    if (!attrs.empty()) {   // the array describes its own axes: laid out again with them, then filled (the same bytes)
      layout_array(sub.store(), path, laid, jdump(attrs));
      populate_array(sub.store(), path, v);
    }
    Cols c;
    c.set("measure", x.measure).set("array_path", path).set("unit", x.unit).set("computed_by", by_.tool);
    if (!m.of_field_id.empty()) c.set("of_field_id", m.of_field_id);        // none: the members' own (a size)
    if (m.of_statistic_id) { c.set("of_statistic_id", *m.of_statistic_id); c.set("of_index", m.of_index.value_or(0)); }
    if (m.within_selection_id) { c.set("within_selection_id", *m.within_selection_id); if (m.within_code) c.set("within_code", *m.within_code); }
    else if (band && !(*band)["selection_id"].is_null()) c.set("within_selection_id", (*band)["selection_id"].get<std::string>()).set("within_code", (*band)["code"].get<std::int64_t>());
    selection::save_measurement(sub, m.selection_id, c);
    ++n;
  }
  if (!m.pair_measure.empty()) {                    // two codes a row: the member and the other
    std::vector<selection::MemberValue> rows;
    rows.reserve(m.pairs.size());
    for (const auto& p : m.pairs) rows.push_back({p.code, p.value, p.within_code});
    n += selection::save_member_values(sub, m.selection_id, m.pair_measure, rows, m.pair_unit.empty() ? std::nullopt : std::optional<std::string>(m.pair_unit), by_.tool, m.selection_id);
  }
  return n;
}

std::string Unit::selection(const SelectionOut& s) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  // the result as the selection module's description, and where it goes; the module maps it onto the rows (selection_store.h)
  selection::Description d;
  selection::Placement at;
  at.name = s.name; at.path = path_of(s); at.manifold_id = s.manifold_id;
  if (s.acquisition) at.acquisition = *s.acquisition;
  Json params = s.params.is_null() ? Json::object() : s.params;
  if (s.type == "grid" && s.grid && s.grid->time) {   // a line's TIME GRID: one ladder row, its rule closed form (schema 53); found and reused when the line has it
    const TimeGridOut& t = *s.grid->time;
    return selection::save_time_grid(sub, s.manifold_id, selection::TimeGridRule{t.origin, t.rate, t.samples}, std::nullopt).id;
  }
  if (s.type == "product") {   // a product of selections (a grid when every factor is): the factors in order, each a set, or a level of a grid (selection_factor.level), mixed radix, first factor fastest
    if (s.factor_ids.size() < 2) throw Error(code::invalid_argument, "selection " + s.name + ": a product has two or more factors (factor_ids)");
    d.type = selection::Type::set;
    d.radices = s.radices;
    d.factor_levels = s.factor_levels;
    d.factors.resize(s.factor_ids.size());          // the factors are named by id (Placement::factor_ids); their descriptions are what a load fills
    at.factor_ids = s.factor_ids;
    at.factor_levels = s.factor_levels;
    if (s.description) at.description = *s.description;
    if (s.label) at.label = *s.label;
    Json pparams = s.params.is_null() ? Json::object() : s.params;
    const Json ppj = producer_of(by_, s.output, s.producer);
    if (ppj.contains("cache")) pparams["cache"] = ppj["cache"];
    pparams["producer"] = ppj;
    at.params = pparams;
    return selection::save(sub, d, at).id;
  }
  if (s.type == "grid") {     // a dyadic grid with levels (made by compute's `dyadic` or `transfer`): the ladder row and one set row per level, through io's save_grid
    if (!s.grid || s.grid->levels.empty()) throw Error(code::invalid_argument, "selection " + s.name + ": a grid has its levels (grid)");
    const GridOut& g = *s.grid;
    const Json pj = producer_of(by_, s.output, s.producer);
    Json lp = g.params.is_object() ? g.params : Json::object();
    if (pj.contains("cache")) lp["cache"] = pj["cache"];
    lp["producer"] = pj;
    // a dictionary as the node described it ({name, description, entries}), or the one already stored
    auto dictionary_of = [&](const Json& dj, const std::optional<std::string>& stored, const std::string& surface_unit) -> std::optional<std::string> {
      if (stored) return stored;
      if (!dj.is_object()) return std::nullopt;
      DictionarySpec spec;
      spec.name = dj.value("name", std::string());
      spec.scope = "subject";
      spec.description = dj.value("description", std::string());
      spec.entries = std::vector<DictionaryEntry>();
      for (const auto& e : dj.value("entries", Json::array())) {
        DictionaryEntry x;
        x.code = e["code"].get<std::int64_t>();
        x.ordinal = x.code;
        x.name = e.value("name", std::string());
        // a dyadic level's size is its mean TILE DIAMETER in mm (`diameter_mm` in its attributes, or `diameter` in the surface's unit): what a query by physical size reads (measurement_query.h), whatever else
        // the maker put in `measurement` (a level's wavelength is an attribute, not its size)
        const Json attrs = e.value("attributes", Json::object());
        auto number = [&](const char* k) { return attrs.is_object() && attrs.contains(k) && attrs[k].is_number() ? std::optional<double>(attrs[k].get<double>()) : std::nullopt; };
        if (const auto mm = number("diameter_mm")) x.cols.set("measurement", *mm).set("measurement_unit", "mm");
        else if (const auto d = number("diameter"); d && (surface_unit == "m" || surface_unit == "cm" || surface_unit == "mm"))
          x.cols.set("measurement", *d * (surface_unit == "m" ? 1e3 : surface_unit == "cm" ? 1e1 : 1.0)).set("measurement_unit", "mm");
        else if (e.contains("measurement") && e["measurement"].is_number()) x.cols.set("measurement", e["measurement"].get<double>()).set("measurement_unit", e.value("measurement_unit", std::string()));
        if (e.contains("attributes")) x.cols.json("attributes_json", e["attributes"]);
        spec.entries->push_back(std::move(x));
      }
      return sub.dictionary(spec).id;
    };
    selection::Placement top;
    top.name = s.name; top.path = at.path; top.manifold_id = s.manifold_id;
    if (s.acquisition) top.acquisition = *s.acquisition;
    top.dictionary_id = dictionary_of(g.dictionary, g.dictionary_id, g.unit);
    top.cols.set("unit", g.unit).set("label", g.label).set("description", g.description);
    top.params = lp;
    std::vector<selection::GridLevel> levels;
    for (const GridLevelOut& L : g.levels) {
      const std::string dir = atlas_node(*impl_->db, s.manifold_id, L.name);       // the level's node: its arrays beside its tile codes
      bool areas = false;
      for (const GridArrayOut& a : L.arrays) {
        if (a.suffix == "w") areas = true;
        const std::string key = dir + "/" + a.suffix;
        if (!a.i32.empty() || a.f64.empty()) sub.write_array(key, view(a.i32, a.shape));
        else sub.write_array(key, view(a.f64, a.shape));
      }
      selection::GridLevel gl;
      gl.at.name = L.name; gl.at.manifold_id = s.manifold_id;
      gl.at.dictionary_id = dictionary_of(L.dictionary, L.dictionary_id, g.unit);
      Json rp = L.params.is_object() ? L.params : Json::object();
      rp["node"] = dir;
      if (areas) rp["areas"] = dir + "/w";
      gl.at.params = rp;
      gl.at.cols.set("unit", g.unit).set("label", L.label).set("description", L.description);
      gl.array_path = dir + "/codes";
      gl.tile_codes = L.codes;
      gl.tiles = L.tiles;
      levels.push_back(std::move(gl));
    }
    const selection::Saved saved = selection::save_grid(sub, top, levels);
    for (std::size_t l = 0; l < g.levels.size(); ++l) {
      const GridLevelOut& L = g.levels[l];
      const std::string& sid = saved.member_ids.at(l);
      if (L.centres.is_object()) {                  // the tiles' centre vertices, a pick on the surface (-1: an empty tile)
        selection::Description picks;
        picks.type = selection::Type::indices;
        for (const auto& i : L.centres["indices"]) picks.indices.push_back(i.get<std::int32_t>());
        selection::Placement cs;
        cs.name = L.centres.value("name", std::string()); cs.path = cs.name; cs.manifold_id = s.manifold_id;
        cs.description = L.centres.value("description", std::string());
        selection::save(sub, picks, cs);
      }
      if (L.pairs.is_object()) {                    // a value over pairs of the level's tiles: a MEASUREMENT over a selection of pairs (indices: pair number a * tiles + b), one value a pair
        const std::int64_t tiles = L.pairs.value("tiles", std::int64_t(0));
        if (tiles < 1 || tiles * tiles > std::numeric_limits<std::int32_t>::max()) throw Error(code::invalid_argument, "grid " + s.name + ": level " + L.name + " has too many tiles for a selection of pairs (" + std::to_string(tiles) + ")");
        selection::Description pd;
        pd.type = selection::Type::indices;
        for (const auto& i : L.pairs["indices"]) pd.indices.push_back(i.get<std::int32_t>());
        selection::Placement ps;
        ps.name = L.pairs.value("name", std::string()); ps.path = ps.name; ps.manifold_id = s.manifold_id;
        ps.description = L.pairs.value("description", std::string());
        ps.params = Json{{"pairs_of", sid}, {"tiles", tiles}, {"pair_number", "a * tiles + b (a, b the level's member numbers)"}};
        const selection::Saved pairs = selection::save(sub, pd, ps);
        std::vector<double> values;
        for (const auto& v : L.pairs["values"]) values.push_back(v.get<double>());
        if (values.size() != pd.indices.size()) throw Error(code::invalid_argument, "grid " + s.name + ": " + std::to_string(values.size()) + " values for " + std::to_string(pd.indices.size()) + " pairs");
        const std::string measure = L.pairs.value("measure", std::string());
        const std::string vpath = "measurements/" + safe_name(ps.name) + "/" + safe_name(measure);
        Cols mc;
        mc.set("measure", measure).set("array_path", vpath).set("unit", L.pairs.value("unit", std::string())).set("computed_by", by_.tool);
        selection::save_measurement_array(sub, pairs.id, vpath, view(values, {static_cast<std::int64_t>(values.size())}), mc);
      }
    }
    return saved.id;
  }
  if (s.type == "partition" || s.type == "set") {     // a partition is a set per element: a member number per element, a label per member
    if (s.member_of.empty()) throw Error(code::invalid_argument, "selection " + s.name + ": a " + s.type + " has a member number per element (member_of)");
    d.type = selection::Type::set;
    d.member_of = s.member_of;
    d.member_codes = s.member_codes;
  } else if (s.type == "indices") {                  // a pick of elements, in order (the channels a covariance covers)
    d.type = selection::Type::indices;
    d.indices.assign(s.codes.begin(), s.codes.end());
  } else if (s.type == "cells") {                    // a closed-form set (the joint cells of space x time): its rule in params, no element stored
    if (!s.n_members) throw Error(code::invalid_argument, "selection " + s.name + ": cells name their number (n_members)");
    d.type = selection::Type::set;
    at.n_members = *s.n_members;
    at.cell = "0,1";
  } else if (s.type == "spans") {
    d.type = selection::Type::set;
    for (const auto& x : s.spans) {
      selection::Description m;
      m.type = selection::Type::span;
      m.code = x.code;
      m.extents.push_back(selection::Extent{0, x.start, x.stop});
      d.members.push_back(std::move(m));
    }
  } else if (s.type == "tiling") {
    if (!s.tiling) throw Error(code::invalid_argument, "selection " + s.name + ": a tiling needs its origin, width, hop and count");
    const Tiling& t = *s.tiling;
    d.type = selection::Type::set;
    d.unit = t.unit;
    d.tiling = selection::Tiling{t.origin, t.width, t.hop, t.count + t.skipped};
    params["end"] = t.end;
    params["segments_skipped"] = t.skipped;
  } else {
    throw Error(code::invalid_argument, "selection " + s.name + ": type " + s.type + " (partition, set, indices, spans, tiling, cells, grid or product)");
  }
  for (std::size_t k = 0; k < s.entries.size(); ++k) d.lookup.push_back(selection::Entry{static_cast<std::int64_t>(k), s.entries[k], std::nullopt});   // the codes' names
  if (s.of_field_id) at.of_field_id = *s.of_field_id;
  if (s.description) at.description = *s.description;
  if (s.label) at.label = *s.label;
  // a Selection has no producer column: its producer record is in its parameters, the cache key under "cache" (the Studio's searchAtlas: params_json.search)
  const Json pj = producer_of(by_, s.output, s.producer);
  if (pj.contains("cache")) params["cache"] = pj["cache"];
  params["producer"] = pj;
  at.params = params;
  return selection::save(sub, d, at).id;
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

std::string Unit::points(const PointsOut& p) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  if (p.positions.shape.size() != 1 || p.positions.f64.empty() || p.positions.size() != p.positions.shape[0]) throw Error(code::invalid_argument, "points " + p.name + ": coordinates [n] in float64");
  const std::int64_t n = p.positions.shape[0];
  const std::string path = path_of(p);
  if (const auto r = sub.find("manifold", path)) {   // the same coordinates are the same axis; another one's is the caller's to replace first
    const Node have = node_of(Cls::manifold, rec::ManifoldRow::from_row(*r));
    if (have.n_vertices && *have.n_vertices == n) {
      ArrayReader old = ArrayReader::open(sub.store(), path);
      const std::vector<std::uint8_t> raw = old.read();
      if (raw.size() == static_cast<std::size_t>(n) * sizeof(double) && std::equal(raw.begin(), raw.end(), reinterpret_cast<const std::uint8_t*>(p.positions.f64.data()))) return have.id;
    }
    throw Error(code::already_exists, sub.name() + ": the points " + p.name + " (" + path + ") exist with other coordinates; replace them first");
  }
  ManifoldSpec ms;
  ms.name = p.name;
  ms.path = path;
  ms.type = "points";
  ms.n_vertices = n;
  ms.topology = {"none", 0, 1, std::nullopt, {}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = p.cs_key;
  g.positions_path = path;
  ms.geometry = g;
  if (p.derived_from_id) ms.cols.set("derived_from_id", *p.derived_from_id);
  if (!p.operator_name.empty()) ms.cols.set("operator", p.operator_name);
  if (p.description) ms.cols.set("comment", *p.description);
  ms.cols.json("producer_json", producer_of(by_, p.output, p.producer));
  const Values& v = p.positions;
  ms.populate = [&sub, &v, path, n](Store&, const std::string&) { sub.write_array(path, view(v.f64, {n})); };
  return sub.manifold(std::move(ms));
}

std::string Unit::cell_complex(const ComplexOut& o) {
  need_open(open_);
  SubjectWriter& sub = *impl_->writer;
  if (o.vertices.shape.size() != 2 || o.vertices.shape[1] != 3 || o.vertices.f64.empty() || o.vertices.size() != o.vertices.shape[0] * 3) throw Error(code::invalid_argument, "complex " + o.name + ": vertices [n x 3] in float64");
  if (o.edges.empty() || o.edges.size() % 2 || o.faces.empty() || o.faces.size() % 3) throw Error(code::invalid_argument, "complex " + o.name + ": edges [nE x 2] and faces [nF x 3]");
  const std::int64_t n = o.vertices.shape[0], nE = static_cast<std::int64_t>(o.edges.size() / 2), nF = static_cast<std::int64_t>(o.faces.size() / 3);
  for (const auto e : o.edges) if (e < 0 || e >= n) throw Error(code::invalid_argument, "complex " + o.name + ": an edge names vertex " + std::to_string(e) + " of " + std::to_string(n));
  for (const auto f : o.faces) if (f < 0 || f >= n) throw Error(code::invalid_argument, "complex " + o.name + ": a face names vertex " + std::to_string(f) + " of " + std::to_string(n));
  const std::string path = path_of(o);
  if (const auto r = sub.find("manifold", path)) {   // found again when it has the same cells; another one's is the caller's to replace first
    const Node have = node_of(Cls::manifold, rec::ManifoldRow::from_row(*r));
    if (have.n_vertices && *have.n_vertices == n && integer(*r, "n_edges") == nE && integer(*r, "n_faces") == nF) return have.id;
    throw Error(code::already_exists, sub.name() + ": the complex " + o.name + " (" + path + ") exists with other cells; replace it first");
  }
  ManifoldSpec ms;
  ms.name = o.name;
  ms.path = path;
  ms.type = "surface";
  ms.n_vertices = n;
  ms.topology = {"simplicial", 2, 1, std::string("ccw-outward"), {{1, path + "/edges"}, {2, path + "/faces"}}};
  GeometrySpec g;
  g.form = "stored";
  g.cs_key = o.cs_key;
  g.positions_path = path + "/vertices";
  ms.geometry = g;
  if (o.acquisition_id) ms.cols.set("acquisition_id", *o.acquisition_id);
  if (o.derived_from_id) ms.cols.set("derived_from_id", *o.derived_from_id);
  if (o.description) ms.cols.set("comment", *o.description);
  ms.cols.set("n_edges", nE).set("n_faces", nF);
  ms.cols.json("producer_json", producer_of(by_, o.output, o.producer));
  const Values& v = o.vertices;
  const std::vector<std::int32_t>& e = o.edges;
  const std::vector<std::int32_t>& f = o.faces;
  ms.populate = [&sub, &v, &e, &f, path, n, nE, nF](Store&, const std::string&) {
    sub.write_array(path + "/vertices", view(v.f64, {n, 3}));
    sub.write_array(path + "/edges", view(e, {nE, 2}));
    sub.write_array(path + "/faces", view(f, {nF, 3}));
  };
  return sub.manifold(std::move(ms));
}

std::string Unit::product(const ProductOut& p) {
  need_open(open_);
  if (p.factors.size() < 2) throw Error(code::invalid_argument, "product: two factors or more");
  return impl_->writer->product(p.factors, p.acquisition_id);
}

}  // namespace nxr::io::task
