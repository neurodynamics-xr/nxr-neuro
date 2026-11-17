// nxr-io — the RECORD API: every row of a package or dataset database, read through typed repositories. Orchestration (pipelines, the command line, the
// bindings) asks io typed questions (`fields.find`, `selections.members`, ...) and never writes SQL or holds a database handle (ARCHITECTURE.md; the
// design is docs/specs/2026-11-17-io-record-api.md, ported from the Cortical Flow Studio's repositories).
//
//   auto rec = nxr::io::records::Records::of(package);          // one database: a subject package, a group package, a dataset
//   auto f = rec.fields().find({.subject_id = id, .name = "sources"});   // one row, none, or Error(ambiguous)
//
// Conventions: rows are the generated structs of rows.h; a `Where` member left unset does not filter; `list` and `find` return `complete` rows unless the
// Where's `status` says otherwise; `find` with several matches throws Error(ambiguous) (code::ambiguous); a missing row is std::nullopt, never an exception.
// `list` is ordered by name, then id (members by ordinal, spans by ordinal), so a result never depends on the engine's plan.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nxr/io/records/rows.h"

namespace nxr::io {
class Database;
class SubjectPackage;
class DatasetWriter;
}  // namespace nxr::io

namespace nxr::io::records {

// A filter on a selection's `params_json`: `path` is a JSON path below `$` (`atlas.tree` or `$.atlas.tree`).
struct ParamEquals {
  std::string path;
  Json value;   // string, number or boolean; compared as SQLite's json_extract gives it
};

struct ManifoldWhere {
  std::optional<std::string> subject_id, type, name, component_of_id, derived_from_id, acquisition_id;
  std::optional<bool> stored;        // path IS NOT NULL (false: a manifold no node stores, a product)
  std::optional<bool> is_primary;
  std::optional<Status> status = Status::complete;
};

struct FieldWhere {
  std::optional<std::string> subject_id, kind, manifold_id, of_field_id, of_operator_id, acquisition_id, name;
  std::optional<bool> stored;        // path IS NOT NULL / IS NULL (virtual, or not yet written)
  std::optional<Status> status = Status::complete;
};

struct OperatorWhere {
  std::optional<std::string> subject_id, kind, name, from_manifold_id, to_manifold_id, derived_from_id, acquisition_id;
  std::optional<bool> stored;        // path IS NOT NULL (false: an analytic operator)
  std::optional<Status> status = Status::complete;
};

struct SelectionWhere {
  std::optional<std::string> subject_id, manifold_id, of_field_id, name, member_of_id, dictionary_id;
  std::optional<std::string> type;                 // 'set', 'spans', 'ladder', ...
  std::optional<std::int64_t> code;
  std::optional<std::string> name_like;            // SQL LIKE pattern on the name
  std::optional<bool> stored;                      // array_path IS NOT NULL
  std::vector<ParamEquals> params;                 // params_json equalities
  std::vector<std::string> params_present;         // params_json has this key (not JSON null)
  std::vector<std::string> params_absent;          // params_json lacks this key
  std::optional<Status> status = Status::complete;
};

// manifolds.read: the Studio's row and its topology, geometry, dimensions.
struct ManifoldRecord {
  ManifoldRow row;
  std::optional<TopologyRow> topology;
  std::vector<TopologyCellRow> cells;
  std::optional<GeometryRow> geometry;
  std::vector<GeometryDimensionRow> dimensions;                // by ordinal
  std::optional<CoordinateSystemRow> coordinate_system;        // the geometry's
};

class Manifolds {
 public:
  explicit Manifolds(const Database& db) : db_(&db) {}
  std::optional<ManifoldRow> row(std::string_view id) const;                // the row alone (any status)
  std::optional<ManifoldRecord> read(std::string_view id) const;            // the row with its children (any status)
  std::vector<ManifoldRow> list(const ManifoldWhere&) const;
  std::optional<ManifoldRow> find(const ManifoldWhere&) const;
  std::vector<GeometryDimensionRow> factors(std::string_view id) const;     // a manifold's geometry dimensions, by ordinal (a product's factor lines)
 private:
  const Database* db_;
};

class Fields {
 public:
  explicit Fields(const Database& db) : db_(&db) {}
  std::optional<FieldRow> read(std::string_view id) const;                  // any status
  std::vector<FieldRow> list(const FieldWhere&) const;
  std::optional<FieldRow> find(const FieldWhere&) const;
 private:
  const Database* db_;
};

class Operators {
 public:
  explicit Operators(const Database& db) : db_(&db) {}
  std::optional<OperatorRow> read(std::string_view id) const;               // any status
  std::vector<OperatorRow> list(const OperatorWhere&) const;
  std::optional<OperatorRow> find(const OperatorWhere&) const;
 private:
  const Database* db_;
};

class Selections {
 public:
  explicit Selections(const Database& db) : db_(&db) {}
  std::optional<SelectionRow> read(std::string_view id) const;              // any status
  std::vector<SelectionRow> list(const SelectionWhere&) const;
  std::optional<SelectionRow> find(const SelectionWhere&) const;
  // The members of a partition (member_of_id), complete ones, by member_ordinal; `ordinal` picks one.
  std::vector<SelectionRow> members(std::string_view partition_id) const;
  std::optional<SelectionRow> member(std::string_view partition_id, std::int64_t ordinal) const;
  std::vector<SelectionSpanRow> spans(std::string_view selection_id) const;  // by ordinal
 private:
  const Database* db_;
};

// A table's kind, for the generic row read.
enum class Kind { subject, manifold, selection, field, operator_, statistic, dictionary, acquisition };
const char* table_of(Kind k);
struct Ref {
  Kind kind;
  std::string id;
};

// One database: a subject package, a group package, a dataset. Cheap to copy (a handle); the database must outlive it.
class Records {
 public:
  static Records of(const Database& db) { return Records(db); }
  static Records of(const SubjectPackage& package);
  static Records of(const DatasetWriter& writer);

  const Manifolds& manifolds() const { return manifolds_; }
  const Fields& fields() const { return fields_; }
  const Operators& operators() const { return operators_; }
  const Selections& selections() const { return selections_; }

  // A row of any node table by Ref as its JSON (keys are the column names; `*_json` columns parsed), or nullopt.
  std::optional<Json> read(const Ref&) const;

 private:
  explicit Records(const Database& db) : manifolds_(db), fields_(db), operators_(db), selections_(db), db_(&db) {}
  Manifolds manifolds_;
  Fields fields_;
  Operators operators_;
  Selections selections_;
  const Database* db_;
};

}  // namespace nxr::io::records
