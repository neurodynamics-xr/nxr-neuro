// nxr-io — LEVEL 2 of the record API: the TASK-LEVEL surface pipelines and the command line use (docs/specs/2026-11-17-io-record-api.md, section 0).
// Pipelines never see a row struct, a table name or a database handle. They find their inputs, ask what is measured, look up a cached result and commit a result as
// one unit; io answers from its repositories (nxr/io/records/, level 1, inside io).
//
//   task::Subject s = task::Subject::of(sub);                                   // io's handle to one subject (of a package or a dataset)
//   Node f = s.require({.cls = Cls::field, .name = "sources", .kind = "recording"}, "the recording");   // one, or a clear Error
//   for (const Node& m : s.members(partition)) ...                              // a partition's members, by ordinal
//
// Conventions: `find` returns one node or none and throws Error(ambiguous) when several match (the message names them); `require` throws Error(no_such_row) naming
// what was asked for; only complete nodes are found unless `Want::status` says otherwise; `find_all` is ordered by name, then id.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/json_text.h"

namespace nxr::io {
class SubjectWriter;
class DatasetWriter;
class SubjectPackage;
}  // namespace nxr::io

namespace nxr::io::task {

enum class Cls { manifold, field, operator_, selection, statistic };
const char* to_string(Cls c);

// One stored thing a pipeline found: what it is, where its arrays are, and what it records about itself. Unset members were NULL (or do not apply to the class).
struct Node {
  Cls cls = Cls::field;
  std::string id, name, subject_id;
  std::optional<std::string> path;                 // the store key of its arrays (none: virtual, an analytic operator, a product manifold)
  std::optional<std::string> kind;                 // field, operator: what it is; statistic: its measure's kind
  std::optional<std::string> type;                 // manifold: surface, line, ...; selection: set, spans, ladder, ...
  std::optional<std::string> manifold_id;          // field, selection: the manifold it lies on
  std::optional<std::string> from_manifold_id, to_manifold_id;   // operator
  std::optional<std::string> from_selection_id, to_selection_id; // operator
  std::optional<std::string> array_path;           // selection: its stored per-element array
  std::optional<std::string> dictionary_id, member_of_id;        // selection
  std::optional<std::int64_t> member_ordinal, code, n_elements, n_members, n_vertices;
  std::optional<std::string> value_type, to_value_type, unit, method, acquisition_id, modified_utc;
  Json params;                                     // selection, analytic operator: its parameters (null when none)
  Json producer;                                   // how it was made, including its cache key (null when none)
  Json row;                                        // every column as the Studio's row object (for a fact no member above names)
  Json fact(const std::string& column) const { return row.contains(column) ? row[column] : Json(); }
};

struct ParamIs {
  std::string path;                                // a key path in `params`, `atlas.tree`
  Json value;                                      // string, number or boolean
};

// What to find. Unset members do not filter.
struct Want {
  Cls cls = Cls::field;
  std::optional<std::string> name, kind, type, manifold_id, from_manifold_id, to_manifold_id, of_field_id, member_of_id, dictionary_id;
  std::optional<std::int64_t> code;
  std::optional<std::string> name_like;            // SQL LIKE pattern
  std::optional<bool> stored;                      // has arrays (path, or array_path for a selection)
  std::vector<ParamIs> params;                     // selection `params` equalities
  std::vector<std::string> params_present, params_absent;
  bool any_status = false;                         // include pending and deleting nodes
};

struct Span {
  std::int64_t ordinal = 0;
  double start = 0, stop = 0;
  std::optional<std::int64_t> code;
};

// A manifold's grid and place: its axes (a line has one), the unit of its coordinate system, its positions and cells.
struct Axis {
  std::int64_t ordinal = 0, n_vertices = 0;
  std::optional<std::string> name, unit, manifold_id;   // manifold_id: the factor line of a product
  std::optional<double> spacing, origin;
};
struct Geometry {
  std::vector<Axis> axes;
  std::optional<std::string> coordinate_unit, positions_path;
  std::vector<std::pair<std::int64_t, std::string>> cells;   // (rank, path of the cell array)
};

class Measures;

// io's handle to ONE subject in one database. Cheap to copy; the writer or package it was made from must outlive it.
class Subject {
 public:
  static Subject of(SubjectWriter& sub);
  static Subject of(const SubjectPackage& package);
  static Subject of(const DatasetWriter& ds, const std::string& subject_id);

  const std::string& id() const;

  // ---- find inputs
  std::optional<Node> find(const Want&) const;
  Node require(const Want&, const std::string& what) const;
  std::vector<Node> find_all(const Want&) const;
  std::optional<Node> by_id(Cls, const std::string& id) const;        // any status
  Node require_id(Cls, const std::string& id) const;
  std::vector<Node> members(const Node& partition) const;            // by member ordinal
  std::optional<Node> member(const Node& partition, std::int64_t ordinal) const;
  std::vector<Span> spans(const Node& selection) const;
  Geometry geometry(const std::string& manifold_id) const;
  std::optional<Node> manifold_of(const Node& n) const;              // the manifold a field or selection lies on
  // The naming rules for a node's store key (io's layout).
  std::string node_path(const std::string& manifold_id, const std::string& name) const;
  std::string atlas_node(const std::string& manifold_id, const std::string& name) const;

  // ---- cache lookup: the complete node whose producer record holds exactly `key` (a selection's: its parameters); with `path`, only that node
  std::optional<Node> cached(Cls, const Json& key, const std::optional<std::string>& path = std::nullopt) const;

  // ---- what is measured on a selection
  std::vector<Json> measured_kinds(const std::string& selection_id) const;            // io's measurement_kinds: {of_field_id, measure, members, min, max, computed_by, unit}
  bool has_measure(const std::string& selection_id, const std::string& measure, const std::optional<std::string>& of_field_id = std::nullopt) const;

  // ---- commit a result as ONE unit: everything written by `body` (nodes, arrays, measurements, parameters) is stored together or not at all
  void commit(const std::function<void()>& body) const;
  std::int64_t write_measurements(const Json& rows, const std::optional<std::string>& by) const;
  std::int64_t remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id = std::nullopt,
                                   const std::optional<std::string>& measure = std::nullopt) const;
  void set_params(const std::string& selection_id, const Json& params) const;              // a selection's parameters, replaced

 private:
  struct Impl;
  Subject() = default;
  std::shared_ptr<Impl> impl_;
};

}  // namespace nxr::io::task
