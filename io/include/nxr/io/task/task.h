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

#include "nxr/io/finder.h"
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

// A band of a map statistic read as a map (a statistic whose first axis lies on a manifold and whose second lists named entries): slice `index`, named
// `<statistic>_<entry name>`.
struct MapSlice {
  std::string statistic_id, name, manifold_id;
  std::int64_t index = 0, n = 0, n_components = 1;
  std::optional<std::string> unit;
  std::string value_type = "scalar";
};

// Who made a result, for its provenance: the program (what a measurement row records as `computed_by`) and the producer JSON a node keeps (how it was made, with its cache key).
struct By {
  std::string tool;
  Json producer;                                   // null: none
};

class Unit;

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
  std::optional<Node> at_path(Cls, const std::string& path) const;   // the node whose arrays are at store path `path` (any status)
  Node require_id(Cls, const std::string& id) const;
  std::vector<Node> members(const Node& partition) const;            // by member ordinal
  std::optional<Node> member(const Node& partition, std::int64_t ordinal) const;
  // the stored fields whose manifold is a product with `factor_manifold_id` as its axis `ordinal` (the fields over one time line, by name)
  std::vector<Node> fields_on_axis(const std::string& factor_manifold_id, std::int64_t ordinal) const;
  std::int64_t entry_count(const std::string& dictionary_id) const;   // the entries of a dictionary (a labelling's levels)
  std::vector<Span> spans(const Node& selection) const;
  Geometry geometry(const std::string& manifold_id) const;
  // The model's description of a found node (its manifolds, value type, time line and rate, unit): io's finder::Value.
  finder::Value describe(const Node& n) const;
  std::optional<Node> manifold_of(const Node& n) const;              // the manifold a field or selection lies on
  std::vector<std::string> factors(const std::string& manifold_id) const;   // a manifold's factors in axis order (itself when it is not a product)
  std::vector<MapSlice> map_slices(const std::string& manifold_id) const;
  std::optional<MapSlice> map_slice(const std::string& name) const;         // the slice of a map statistic of this subject by its name
  // The naming rules for a node's store key (io's layout).
  std::string node_path(const std::string& manifold_id, const std::string& name) const;
  std::string atlas_node(const std::string& manifold_id, const std::string& name) const;

  // ---- cache lookup: the complete node whose producer record holds exactly `key` (a selection's: its parameters); with `path`, only that node
  std::optional<Node> cached(Cls, const Json& key, const std::optional<std::string>& path = std::nullopt) const;

  // ---- what is measured on a selection
  std::vector<Json> measured_kinds(const std::string& selection_id) const;            // io's measurement_kinds: {of_field_id, measure, members, min, max, computed_by, unit}
  // the selections that carry `measure` measured on `of_field_id` (ids, ascending)
  std::vector<std::string> measured_selections(const std::string& of_field_id, const std::string& measure) const;
  bool has_measures(const std::string& selection_id, const std::string& of_field_id) const;   // any measure of that field on the selection
  bool has_measure(const std::string& selection_id, const std::string& measure, const std::optional<std::string>& of_field_id = std::nullopt) const;

  // ---- commit a result as ONE unit: everything written inside it (nodes and arrays through the subject's writer, measurements, parameters) is stored together or not at all.
  //   auto u = subject.unit(By{"nxr-neuro tiles"});   // opens it
  //   ... writer calls, u.write_measurements(rows), u.set_params(id, p) ...
  //   u.commit();                                      // every node it made complete, one drain, the transaction committed; a Unit destroyed without it rolls back
  Unit unit(const By& by) const;
  // the same as a scope: `body` runs in a Unit that commits when it returns and rolls back when it throws
  void commit(const std::function<void()>& body, const By& by = By{"nxr-neuro", {}}) const;
  // measurement rows and parameters outside a unit (each is its own transaction); inside one use the Unit's
  std::int64_t write_measurements(const Json& rows, const std::optional<std::string>& by) const;
  std::int64_t remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id = std::nullopt,
                                   const std::optional<std::string>& measure = std::nullopt) const;
  void set_params(const std::string& selection_id, const Json& params) const;              // a selection's parameters, replaced

 private:
  friend class Unit;
  struct Impl;
  Subject() = default;
  std::shared_ptr<Impl> impl_;
};

// ONE atomic unit of writing in a subject (opened by Subject::unit). Move-only. `commit()` stores everything; destroyed (or `rollback()`) without it, nothing is stored.
class Unit {
 public:
  Unit(Unit&&) noexcept;
  Unit& operator=(Unit&&) = delete;
  ~Unit();
  const By& by() const;
  // the measurement rows of the unit, recorded as computed by `by().tool`; returns the number written
  std::int64_t write_measurements(const Json& rows);
  std::int64_t remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id = std::nullopt,
                                   const std::optional<std::string>& measure = std::nullopt);
  void set_params(const std::string& selection_id, const Json& params);
  void commit();
  void rollback() noexcept;
  bool open() const { return open_; }

 private:
  friend class Subject;
  Unit(std::shared_ptr<Subject::Impl> impl, By by);
  std::shared_ptr<Subject::Impl> impl_;
  By by_;
  bool open_ = false;
};

}  // namespace nxr::io::task
