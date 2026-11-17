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
#include <array>
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
class Database;
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

// ---- WHAT A UNIT WRITES (spec docs/specs/2026-11-17-definition-outputs.md, section 4): one struct per kind of result, naming found nodes by id and holding its
// values; never a row or a column. Each node written records the unit's `By` (the producer record, with the cache key), the output's name and its own `producer`.

// an array handed to io: C order, in float32 or float64 (one of the two filled: the stored precision follows the input)
struct Values {
  std::vector<std::int64_t> shape;
  std::vector<float> f32;
  std::vector<double> f64;
  std::int64_t size() const { return static_cast<std::int64_t>(f32.empty() ? f64.size() : f32.size()); }
};
// a Field on a manifold
struct FieldOut {
  std::string name, output, kind, manifold_id;
  std::optional<std::string> path;                 // default: io's node path for the name on the manifold
  std::string value_type = "scalar";               // scalar · complex · vector3 · quaternion · hermitian3 …
  std::optional<std::string> unit, description, derived_from_id, acquisition;
  std::optional<std::string> of_operator_id;       // a field of an operator (a head model's source positions)
  Json producer;                                   // added to the unit's producer record (null: nothing added)
  Values data;                                     // [n_vertices] or [n_vertices x components]; empty: a field with no array (what measurements are of)
};
// a Statistic: what it is (measure, method), of what, from which partition, along which axes
struct StatisticOut {
  std::string name, output, measure, method;
  std::optional<std::string> path;                 // default: beside the field it is of (`<field path>_statistics/<name>`), else `statistics/<name>`
  std::string value_type = "scalar";
  std::optional<std::string> of_field_id, of_statistic_id, partition_selection_id;
  Json axes = Json::array(), components, span, scaling, options;
  std::optional<std::int64_t> n_members, members_skipped;
  std::optional<double> dof;                       // degrees of freedom, where the method defines them (a covariance's divisor)
  bool detached = false;                           // computed elsewhere (an imported covariance): of no field in the package, its provenance in `options`
  std::optional<std::string> unit, description, label, acquisition;
  Json producer;
  Values data;
};
// an Operator from one manifold to another
struct OperatorOut {
  std::string name, output, kind, from_manifold_id, to_manifold_id;
  std::optional<std::string> path;                 // default: the name
  std::string from_value_type = "scalar", to_value_type = "scalar";
  std::optional<std::string> inverts_id, inverse_kind;   // the operator it is the (regularized) inverse of: a kernel of its head model
  std::optional<std::string> from_selection_id, to_selection_id, derived_from_id, of_statistic_id, description, acquisition;   // of_statistic_id: the statistic it is derived from
  Json params;                                     // the operator's parameters (null: none)
  Json producer;
  Values data;                                     // dense [rows x cols]
};
// MEASUREMENTS OF A FIELD OVER A PARTITION, array-backed: one value per member for each measure (the pyramid's leaves: additive ingredients, ratios derived at read).
// Each measure's values are stored at `<path>/<measure>` and recorded as a measurement row of `selection_id`; `band` (name, lo_hz, hi_hz) records the band it is within.
struct MeasurementOut {
  std::string output, selection_id, of_field_id, path;
  struct Measure { std::string measure, unit; Values values; };
  std::vector<Measure> measures;
  Json band;                                       // {name, lo_hz, hi_hz} or null
};
// a closed-form tiling of a time line: `count` tiles of `width` a new one every `hop` from `origin` (in `unit`), `skipped` of them left out
struct Tiling { double origin = 0, width = 0, hop = 0, end = 0; std::int64_t count = 0, skipped = 0; std::string unit = "s"; };
// a Selection on a manifold: a partition or set (a code per element), spans (on a time line), or a tiling (a closed-form spans Selection)
struct SelectionOut {
  std::string name, output, type, manifold_id;     // type: partition · set · indices · spans · tiling · cells (a closed-form set: codes by a rule, no element stored)
  std::optional<std::string> path;                 // default: io's node path for the name on the manifold
  std::vector<std::int64_t> codes;                 // indices: the elements picked, in order
  std::vector<std::int32_t> member_of;             // partition, set: the member NUMBER of each element (dense 0 .. n - 1, -1: none), made from the labels by compute's `from_labels` (pipelines)
  std::vector<std::int64_t> member_codes;          // partition, set: each member's LABEL, one per member (empty: member i is labelled i); `entries` name the labels
  std::vector<Span> spans;                         // spans: [start, stop) in the manifold's unit, each with its code
  std::optional<Tiling> tiling;                    // tiling
  std::optional<std::int64_t> n_members;           // cells: the number of cells
  std::vector<std::string> entries;                // the codes' names (a dictionary of the selection's own), when any
  std::optional<std::string> of_field_id, description, label, acquisition;
  Json params;                                     // the selection's parameters (null: none)
  Json producer;
};
// a closed-form line (a frequency axis, a bin axis): n points of `spacing` from `origin`, in `unit`; found by name and reused when it fits
struct LineOut {
  std::string name, alt;                           // `alt`: the name used when a line of `name` exists and does not fit
  std::int64_t n = 0;
  double spacing = 1, origin = 0;
  std::string unit = "Hz", dimension = "frequency", folder = "spectral", cs_key = "hz";
  std::optional<std::string> acquisition_id, derived_from_id;
};

// a surface: vertices [n x 3] (in its coordinate system) and triangles [m x 3] from a node's arrays; found by name and reused when its arrays are the same
struct SurfaceOut {
  std::string name, output;
  std::optional<std::string> path;                 // default: `source/<name>`
  std::string cs_key = "scs";                      // its coordinate system
  std::optional<std::string> derived_from_id, acquisition_id, description;
  Values vertices;                                 // [n x 3], float64
  std::vector<std::int32_t> faces;                 // [m x 3], counter-clockwise seen from outside
  Json producer;
};
// a product of manifolds (a sensor × time line, a surface × frequency): its factors in axis order, existing or made in the same unit
struct ProductOut {
  std::vector<std::string> factors;
  std::optional<std::string> acquisition_id;
};
// where a stored node came from and under what terms it may go on (the package's `origin` record); what an export carries with every file
struct Origin {
  std::string source, licence, attribution;
  bool redistributable = false;
};
// one level of a dictionary (a labelling's names, with the colour when the level has one: [r, g, b], 0 to 255)
struct DictionaryLevel {
  std::int64_t code = 0;
  std::string name;
  std::optional<std::array<int, 3>> color;
};
// the size and SHA-256 of one stored key (a package records them as it writes)
struct KeyDigest {
  std::string key;
  std::int64_t size_bytes = 0;
  std::string sha256;
};
// How many nodes a subject holds, of each class, any status.
struct NodeCounts {
  std::int64_t manifolds = 0, fields = 0, operators = 0, selections = 0;
};

class Unit;

// io's handle to ONE subject in one database. Cheap to copy; the writer or package it was made from must outlive it.
class Subject {
 public:
  static Subject of(SubjectWriter& sub);
  static Subject of(const SubjectPackage& package);
  static Subject of(const DatasetWriter& ds, const std::string& subject_id);
  static Subject of(const Database& db, const std::string& subject_id);   // reading only (a plan, before any run): no write call is made on it

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
  std::vector<DictionaryLevel> dictionary_entries(const std::string& dictionary_id) const;   // by display order, then code
  // the origin record of a node with arrays (none: the package holds no record for it, or the database has no origin table)
  std::optional<Origin> origin(const Node& n) const;
  // the recorded digests of the keys under `prefix` (a package records them; a database that does not gives none), by key
  std::vector<KeyDigest> key_digests(const std::string& prefix) const;
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

  // ---- what the command line looks up by name
  std::optional<Node> by_id_or_name(Cls, const std::string& key) const;   // the complete node with that id or that name (an id match first)
  std::optional<std::string> line_named(const std::string& name) const;   // the id of the manifold of type line with that name (any status)
  // the time lines a complete recording lies on (the line factors of its product manifold): every recording's, first seen first; or the named recording's first
  std::vector<std::string> recording_time_lines() const;
  std::optional<std::string> recording_time_line(const std::string& recording) const;
  NodeCounts node_counts() const;

  // ---- cache lookup: the complete node whose producer record holds exactly `key` (a selection's: its parameters); with `path`, only that node
  std::optional<Node> cached(Cls, const Json& key, const std::optional<std::string>& path = std::nullopt) const;

  // ---- what is measured on a selection
  std::vector<Json> measured_kinds(const std::string& selection_id) const;            // io's measurement_kinds: {of_field_id, measure, members, min, max, computed_by, unit}
  // the selections that carry `measure` measured on `of_field_id` (ids, ascending)
  std::vector<std::string> measured_selections(const std::string& of_field_id, const std::string& measure) const;
  bool has_measures(const std::string& selection_id, const std::string& of_field_id) const;   // any measure of that field on the selection
  bool has_measure(const std::string& selection_id, const std::string& measure, const std::optional<std::string>& of_field_id = std::nullopt) const;

  // ---- read measurements: io's read-time query (nxr/io/measures.h, query_measurements): a stored measure as stored, a derived one from its ingredients, a pyramid's
  // array-backed measurement rolled up at read to the levels `q` names (that needs the subject's store: a Subject made from a writer)
  Json read_measurements(const Json& q) const;

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
  // the measurement rows of the unit, recorded as computed by `by().tool` unless a row names its own `computed_by`; returns the number written
  std::int64_t write_measurements(const Json& rows);
  std::int64_t remove_measurements(const std::string& selection_id, const std::optional<std::string>& of_field_id = std::nullopt,
                                   const std::optional<std::string>& measure = std::nullopt);
  void set_params(const std::string& selection_id, const Json& params);
  // ---- typed writes, each returning the new node's id (call `replace` on its `path_of` first: a node already there is refused or moved aside)
  std::string field(const FieldOut&);
  std::string statistic(const StatisticOut&);
  std::string operator_(const OperatorOut&);
  std::string selection(const SelectionOut&);
  std::int64_t measurements(const MeasurementOut&);   // the rows written (one a measure); a measure already there for the field is replaced
  // MANIFOLDS (the sixth kind): made first, so the other results of the unit can lie on them; each found and reused when identical, never replaced
  std::string line(const LineOut&);
  std::string surface(const SurfaceOut&);
  std::string product(const ProductOut&);
  // the place a result would take (what `replace` and the cache look at)
  std::string path_of(const FieldOut&) const;
  std::string path_of(const StatisticOut&) const;
  std::string path_of(const OperatorOut&) const;
  std::string path_of(const SelectionOut&) const;
  // a node of class `cls` already at `path`: refused (code already_exists, naming `what`) unless `replace`, then moved aside (its arrays to a dated
  // `_replaced-*` folder, never deleted) and its row removed; returns where its arrays went (empty: nothing was there, or it had none)
  std::string replace(Cls cls, const std::string& path, bool replace, const std::string& what);
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
