// nxr-io — the record API's read side (io/include/nxr/io/records/records.h). The only place that writes the node tables' SELECTs for orchestration.
#include "nxr/io/records/records.h"

#include "nxr/io/database.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::io::records {

const char* to_string(Status s) {
  switch (s) {
    case Status::pending: return "pending";
    case Status::complete: return "complete";
    case Status::deleting: return "deleting";
  }
  return "complete";
}

Status status_from(const std::string& s) {
  if (s == "pending") return Status::pending;
  if (s == "deleting") return Status::deleting;
  return Status::complete;
}

const char* table_of(Kind k) {
  switch (k) {
    case Kind::subject: return "subject";
    case Kind::manifold: return "manifold";
    case Kind::selection: return "selection";
    case Kind::field: return "field";
    case Kind::operator_: return "operator";
    case Kind::statistic: return "statistic";
    case Kind::dictionary: return "dictionary";
    case Kind::acquisition: return "acquisition";
  }
  return "";
}

namespace {

// `SELECT * FROM <table> WHERE ...`, conditions added as the Where gives them.
struct Query {
  std::string table;
  std::vector<std::string> conds;
  std::vector<Value> args;
  std::string order = " ORDER BY name, id";

  void eq(const char* column, const std::optional<std::string>& v) {
    if (v) { conds.push_back(std::string(column) + " = ?"); args.push_back(V(*v)); }
  }
  void eq(const char* column, const std::optional<std::int64_t>& v) {
    if (v) { conds.push_back(std::string(column) + " = ?"); args.push_back(V(static_cast<long long>(*v))); }
  }
  void stored(const char* column, const std::optional<bool>& v) {
    if (v) conds.push_back(std::string(column) + (*v ? " IS NOT NULL" : " IS NULL"));
  }
  void flag(const char* column, const std::optional<bool>& v) {
    if (v) { conds.push_back(std::string(column) + " = ?"); args.push_back(V(*v)); }
  }
  void status(const std::optional<Status>& s) {
    if (s) { conds.push_back("status = ?"); args.push_back(V(to_string(*s))); }
  }
  std::string sql(const char* limit = "") const {
    std::string s = "SELECT * FROM " + table;
    for (std::size_t i = 0; i < conds.size(); ++i) s += (i ? " AND " : " WHERE ") + conds[i];
    return s + order + limit;
  }
};

std::string json_path(const std::string& p) { return p.rfind("$", 0) == 0 ? p : "$." + p; }

Value json_value(const Json& v) {
  if (v.is_string()) return V(v.get<std::string>());
  if (v.is_boolean()) return V(v.get<bool>());
  if (v.is_number_integer()) return V(static_cast<long long>(v.get<std::int64_t>()));
  if (v.is_number()) return V(v.get<double>());
  throw Error(code::invalid_argument, "a params filter value must be a string, a number or a boolean");
}

template <class R>
std::vector<R> run(const Database& db, const Query& q) {
  std::vector<R> out;
  for (const Row& r : db.all(q.sql(), q.args)) out.push_back(R::from_row(r));
  return out;
}

// One row or none; several is an error that names them.
template <class R>
std::optional<R> run_one(const Database& db, const Query& q, const char* what) {
  const auto rows = db.all(q.sql(" LIMIT 3"), q.args);
  if (rows.empty()) return std::nullopt;
  if (rows.size() > 1) {
    std::string ids;
    for (const Row& r : rows) ids += (ids.empty() ? "" : ", ") + text(r, "id");
    throw Error(code::ambiguous, std::string("more than one ") + what + " matches (" + ids + "); say which in the Where");
  }
  return R::from_row(rows.front());
}

template <class R>
std::optional<R> by_id(const Database& db, const char* table, std::string_view id) {
  const auto r = db.read(table, std::string(id));
  if (!r) return std::nullopt;
  return R::from_row(*r);
}

}  // namespace

// ── manifolds ────────────────────────────────────────────────────────────────────────────────────────────────────────

namespace {
Query manifold_query(const ManifoldWhere& w) {
  Query q{"manifold", {}, {}};
  q.eq("subject_id", w.subject_id); q.eq("type", w.type); q.eq("name", w.name); q.eq("component_of_id", w.component_of_id);
  q.eq("derived_from_id", w.derived_from_id); q.eq("acquisition_id", w.acquisition_id);
  q.stored("path", w.stored); q.flag("is_primary", w.is_primary); q.status(w.status);
  return q;
}
}  // namespace

std::optional<ManifoldRow> Manifolds::row(std::string_view id) const { return by_id<ManifoldRow>(*db_, "manifold", id); }

std::optional<ManifoldRecord> Manifolds::read(std::string_view id) const {
  auto m = row(id);
  if (!m) return std::nullopt;
  ManifoldRecord rec;
  rec.row = std::move(*m);
  rec.topology = by_id<TopologyRow>(*db_, "topology", rec.row.topology_id);
  for (const Row& r : db_->all("SELECT * FROM topology_cell WHERE topology_id = ? ORDER BY rank", {V(rec.row.topology_id)}))
    rec.cells.push_back(TopologyCellRow::from_row(r));
  if (rec.row.geometry_id) {
    rec.geometry = by_id<GeometryRow>(*db_, "geometry", *rec.row.geometry_id);
    rec.dimensions = factors(id);
    if (rec.geometry && rec.geometry->coordinate_system_id)
      rec.coordinate_system = by_id<CoordinateSystemRow>(*db_, "coordinate_system", *rec.geometry->coordinate_system_id);
  }
  return rec;
}

std::vector<ManifoldRow> Manifolds::list(const ManifoldWhere& w) const { return run<ManifoldRow>(*db_, manifold_query(w)); }
std::optional<ManifoldRow> Manifolds::find(const ManifoldWhere& w) const { return run_one<ManifoldRow>(*db_, manifold_query(w), "manifold"); }

std::vector<GeometryDimensionRow> Manifolds::factors(std::string_view id) const {
  std::vector<GeometryDimensionRow> out;
  for (const Row& r : db_->all("SELECT d.* FROM geometry_dimension d JOIN manifold m ON m.geometry_id = d.geometry_id WHERE m.id = ? ORDER BY d.ordinal", {V(std::string(id))}))
    out.push_back(GeometryDimensionRow::from_row(r));
  return out;
}

// ── fields ───────────────────────────────────────────────────────────────────────────────────────────────────────────

namespace {
Query field_query(const FieldWhere& w) {
  Query q{"field", {}, {}};
  q.eq("subject_id", w.subject_id); q.eq("kind", w.kind); q.eq("manifold_id", w.manifold_id); q.eq("of_field_id", w.of_field_id);
  q.eq("of_operator_id", w.of_operator_id); q.eq("acquisition_id", w.acquisition_id); q.eq("name", w.name);
  q.stored("path", w.stored); q.status(w.status);
  return q;
}
}  // namespace

std::optional<FieldRow> Fields::read(std::string_view id) const { return by_id<FieldRow>(*db_, "field", id); }
std::vector<FieldRow> Fields::list(const FieldWhere& w) const { return run<FieldRow>(*db_, field_query(w)); }
std::optional<FieldRow> Fields::find(const FieldWhere& w) const { return run_one<FieldRow>(*db_, field_query(w), "field"); }

// ── operators ────────────────────────────────────────────────────────────────────────────────────────────────────────

namespace {
Query operator_query(const OperatorWhere& w) {
  Query q{"operator", {}, {}};
  q.eq("subject_id", w.subject_id); q.eq("kind", w.kind); q.eq("name", w.name); q.eq("from_manifold_id", w.from_manifold_id);
  q.eq("to_manifold_id", w.to_manifold_id); q.eq("derived_from_id", w.derived_from_id); q.eq("acquisition_id", w.acquisition_id);
  q.stored("path", w.stored); q.status(w.status);
  return q;
}
}  // namespace

std::optional<OperatorRow> Operators::read(std::string_view id) const { return by_id<OperatorRow>(*db_, "operator", id); }
std::vector<OperatorRow> Operators::list(const OperatorWhere& w) const { return run<OperatorRow>(*db_, operator_query(w)); }
std::optional<OperatorRow> Operators::find(const OperatorWhere& w) const { return run_one<OperatorRow>(*db_, operator_query(w), "operator"); }

// ── selections ───────────────────────────────────────────────────────────────────────────────────────────────────────

namespace {
Query selection_query(const SelectionWhere& w) {
  Query q{"selection", {}, {}};
  q.eq("subject_id", w.subject_id); q.eq("manifold_id", w.manifold_id); q.eq("of_field_id", w.of_field_id); q.eq("name", w.name);
  q.eq("member_of_id", w.member_of_id); q.eq("dictionary_id", w.dictionary_id); q.eq("type", w.type); q.eq("code", w.code);
  if (w.name_like) { q.conds.push_back("name LIKE ?"); q.args.push_back(V(*w.name_like)); }
  q.stored("array_path", w.stored);
  for (const ParamEquals& p : w.params) {
    q.conds.push_back("json_extract(params_json, ?) = ?");
    q.args.push_back(V(json_path(p.path)));
    q.args.push_back(json_value(p.value));
  }
  for (const std::string& p : w.params_present) { q.conds.push_back("json_extract(params_json, ?) IS NOT NULL"); q.args.push_back(V(json_path(p))); }
  for (const std::string& p : w.params_absent) { q.conds.push_back("json_extract(params_json, ?) IS NULL"); q.args.push_back(V(json_path(p))); }
  q.status(w.status);
  return q;
}
}  // namespace

std::optional<SelectionRow> Selections::read(std::string_view id) const { return by_id<SelectionRow>(*db_, "selection", id); }
std::vector<SelectionRow> Selections::list(const SelectionWhere& w) const { return run<SelectionRow>(*db_, selection_query(w)); }
std::optional<SelectionRow> Selections::find(const SelectionWhere& w) const { return run_one<SelectionRow>(*db_, selection_query(w), "selection"); }

std::vector<SelectionRow> Selections::members(std::string_view partition_id) const {
  Query q{"selection", {}, {}, " ORDER BY member_ordinal"};
  q.eq("member_of_id", std::string(partition_id)); q.status(Status::complete);
  return run<SelectionRow>(*db_, q);
}

std::optional<SelectionRow> Selections::member(std::string_view partition_id, std::int64_t ordinal) const {
  Query q{"selection", {}, {}, " ORDER BY member_ordinal"};
  q.eq("member_of_id", std::string(partition_id)); q.eq("member_ordinal", ordinal); q.status(Status::complete);
  return run_one<SelectionRow>(*db_, q, "partition member");
}

std::vector<SelectionExtentRow> Selections::extents(std::string_view selection_id) const {
  std::vector<SelectionExtentRow> out;
  for (const Row& r : db_->all("SELECT * FROM selection_extent WHERE selection_id = ? ORDER BY axis", {V(std::string(selection_id))})) out.push_back(SelectionExtentRow::from_row(r));
  return out;
}

std::vector<SelectionMeasurementRow> Selections::measurements(std::string_view selection_id, bool own) const {
  std::vector<SelectionMeasurementRow> out;
  for (const Row& r : db_->all(std::string("SELECT * FROM selection_measurement WHERE selection_id = ?") + (own ? " AND code IS NULL" : "") + " ORDER BY coalesce(code, -1), measure",
                               {V(std::string(selection_id))}))
    out.push_back(SelectionMeasurementRow::from_row(r));
  return out;
}

std::vector<SelectionSpanRow> Selections::spans(std::string_view selection_id) const {
  std::vector<SelectionSpanRow> out;
  for (const Row& r : db_->all("SELECT * FROM selection_span WHERE selection_id = ? ORDER BY ordinal", {V(std::string(selection_id))}))
    out.push_back(SelectionSpanRow::from_row(r));
  return out;
}

// ── the generic read, and where a Records comes from ─────────────────────────────────────────────────────────────────

std::optional<Json> Records::read(const Ref& ref) const {
  const auto r = db_->read(table_of(ref.kind), ref.id);
  if (!r) return std::nullopt;
  switch (ref.kind) {
    case Kind::subject: return SubjectRow::from_row(*r).to_json();
    case Kind::manifold: return ManifoldRow::from_row(*r).to_json();
    case Kind::selection: return SelectionRow::from_row(*r).to_json();
    case Kind::field: return FieldRow::from_row(*r).to_json();
    case Kind::operator_: return OperatorRow::from_row(*r).to_json();
    case Kind::statistic: return StatisticRow::from_row(*r).to_json();
    case Kind::dictionary: return DictionaryRow::from_row(*r).to_json();
    case Kind::acquisition: return AcquisitionRow::from_row(*r).to_json();
  }
  return std::nullopt;
}

Records Records::of(const SubjectPackage& package) { return Records(package.db()); }
Records Records::of(const DatasetWriter& writer) { return Records(writer.db()); }

}  // namespace nxr::io::records
