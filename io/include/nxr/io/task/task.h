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
#include <map>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "nxr/io/chunk_loop.h"
#include "nxr/io/finder.h"
#include "nxr/io/json_text.h"
#include "nxr/io/measurement_query.h"
#include "nxr/io/selection_store.h"

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
  std::optional<std::string> coordinate_system, space;       // the coordinate system its positions are in (its name: scs, clock, ...) and the space it is a chart of
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
  std::vector<std::int32_t> i32;                   // integer values (an index, a count): exactly one of the four is filled
  std::vector<std::uint8_t> u8;                    // small codes (a form degree, a flag)
  std::int64_t size() const { return static_cast<std::int64_t>(f32.size() + f64.size() + i32.size() + u8.size()); }
};
// a Field on a manifold
struct FieldOut {
  std::string name, output, kind, manifold_id;
  std::optional<std::string> path;                 // default: io's node path for the name on the manifold
  std::string value_type = "scalar";               // scalar · complex · vector3 · quaternion · hermitian3 …
  std::optional<std::string> unit, description, derived_from_id, acquisition;
  std::optional<std::string> cell;                 // WHICH cells of its manifold the values sit on (the schema's cell key, D92): '0' vertices (the default), '1' edges, '2' faces, '1,0' on a product
  std::optional<std::string> of_operator_id;       // a field of an operator (a head model's source positions)
  std::optional<std::string> by_operator_id;       // a field made BY an operator (a source estimate: the inverse kernel applied to a recording; derived_from_id is the recording)
  Json producer;                                   // added to the unit's producer record (null: nothing added)
  // the fields a mapped field is made from (the schema's field_contribution): the subject each lies in (its id), the field when it is the subject's own, and its weight
  struct Contribution { std::string subject_id; std::optional<std::string> source_field_id; double weight = 1; };
  std::vector<Contribution> contributions;
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
  std::optional<Json> cache_key;                   // the operator's OWN provenance key (the fused operator of a chain, a kernel's inputs): stored in its producer record, found by `Subject::stored_operator(key)`
                                                   // instead of being rebuilt or held in a process-wide cache (overrides the unit's key for this operator)
  std::optional<std::string> from_cell, to_cell;   // the cell keys of its two ends (D92): a differential operator maps within ONE manifold and changes the cell (d0: '0' -> '1')
  std::vector<std::int64_t> chunk_shape, shard_shape;   // a dense array's stored layout (empty: io's policy)
  // a SPARSE operator (csr or csc, the schema's D160): `data` holds the nnz values (float32 or float64), these the structure; rows x cols is its shape
  struct Sparse {
    std::string layout = "csr";
    std::int64_t n_rows = 0, n_cols = 0;
    std::vector<std::int64_t> indptr;              // [n_rows + 1] for csr, [n_cols + 1] for csc
    std::vector<std::int32_t> indices;             // [nnz]: the column of each value in a csr row, the row in a csc column
  };
  std::optional<Sparse> sparse;
  // an ANALYTIC operator (layout analytic): no array, only the method that evaluates it and its parameters (`params`); rows x cols is its shape
  struct Analytic { std::string method; std::int64_t n_rows = 0, n_cols = 0; };
  std::optional<Analytic> analytic;
  Values data;                                     // dense [rows x cols], or the sparse operator's nnz values
};
// MEASUREMENTS OF A FIELD (or of a statistic's slice) OVER A SELECTION'S MEMBERS, array-backed: one value per member for each measure (the pyramid's leaves: additive ingredients, ratios derived at read).
// ONE table holds every value over a selection's members (spec 6.1): a per-member value that is not a scalar (a window's z zH, a tile's counts) is an array [rows x members x components].
// Each measure's values are stored at `<path>/<measure>` and recorded as a measurement row of `selection_id`; `band` (name, lo_hz, hi_hz) records the band it is within.
struct MeasurementOut {
  std::string output, selection_id, of_field_id, path;
  struct Measure { std::string measure, unit; Values values; };
  std::vector<Measure> measures;
  Json attributes;                                 // what each measure's array says of its own axes (the zarr attributes: rows = ..., columns = ...); null: none
  Json band;                                       // {name, lo_hz, hi_hz} or null: the band of the dataset's standard bank
  std::optional<std::string> within_selection_id;  // the Selection a measure is WITHIN (a bank of bands whose code is the band's index: the pyramid's `atlas bands`) and the member of it: wins over `band`
  std::optional<std::int64_t> within_code;
  std::string value_type = "scalar";               // each member's value, as a field's (scalar, complex, vector3, hermitian3, ...): its shape trails [rows x members] as `components`
  Json components;                                 // the value's parts when there are several, in order: ["re", "im"], matrix entries; null: one
  std::optional<std::string> of_statistic_id;      // the measurement is of a slice of a statistic (a band of a map statistic) rather than of a field: with `of_index`
  std::optional<std::int64_t> of_index;
  std::optional<int> level;                        // the LEVEL of the selection's grid the members are at (the product's factor level, a dyadic level, a time level k): recorded in the array's attributes, so a reader
                                                   // of the array alone knows the scale; the selection (or its product) is the authority
  // TWO CODES A ROW (a measurement of a non-chunk array: the fiber counts between the regions of a partition): one value per ordered pair of the selection's members, a row (code, within_code, value)
  // on `selection_id` (the member and the other); the rows are of the members themselves (no field)
  struct Pair { std::int64_t code = 0, within_code = 0; double value = 0; };
  std::string pair_measure, pair_unit;
  std::vector<Pair> pairs;
};
// a closed-form tiling of a time line: `count` tiles of `width` a new one every `hop` from `origin` (in `unit`), `skipped` of them left out
struct Tiling { double origin = 0, width = 0, hop = 0, end = 0; std::int64_t count = 0, skipped = 0; std::string unit = "s"; };
// one array beside a grid level's tile codes, in the level's node (the tiles' areas `w`, the tile centres, the tiles' adjacency): float64 or int32, its shape
struct GridArrayOut {
  std::string suffix;                              // the array's key under the level's node, e.g. `w`, `lc/L3/centre`
  std::vector<std::int64_t> shape;
  std::vector<double> f64;                         // float64 values, or ...
  std::vector<std::int32_t> i32;                   // ... int32 values (the one that is not empty)
};
// one level of a dyadic GRID as a node made it: the tile of every element as the tile CODE h 2^l + t (which is the member number; -1 none), the level's row and its arrays
struct GridLevelOut {
  std::string name, label, description;            // the level's selection name (`<surface> tiles L3`), its label and description
  Json params;                                     // the level row's parameters (the store adds `node` and, when an array `w` is given, `areas`)
  std::int64_t tiles = 0;                          // the level's member count (empty tiles included)
  std::vector<std::int32_t> codes;                 // the tile code of every element at this level
  Json dictionary;                                 // {name, description, entries: [{code, name, measurement, measurement_unit, attributes}]} to make, or null
  std::optional<std::string> dictionary_id;        // or the id of a dictionary already stored (the group tree's levels reuse the template's)
  std::vector<GridArrayOut> arrays;
  Json centres;                                    // {name, description, indices}: a pick of the tiles' centre vertices made beside the level (null: none)
  Json pairs;                                      // {name, description, measure, unit, tiles, indices, values}: a value over PAIRS of the level's members, as a measurement over a selection of pairs (an `indices` selection made beside the level, pair number a * tiles + b; null: none)
};
// a line's TIME GRID (schema 53): the one `ladder` row `<line> time grid`, its rule closed form: sample i at origin + i / rate, level k tiles 2^k samples from the line's FIRST sample; nothing stored per level or tile
// (io's save_time_grid). Found and reused when the line has it; another rule on the same line is refused.
struct TimeGridOut {
  double origin = 0, rate = 0;                     // seconds, Hz
  std::int64_t samples = 0;                        // the line's samples
  std::string rule_name;                           // the grid named by its rule ("band 8-16 Hz"): `<line> time grid <rule_name>`; empty: `<line> time grid`
};
// a dyadic GRID with its levels: a `ladder` row (its levels' dictionary) and one `set` row per level; the hierarchy is the closed form, no parent arrays (io's save_grid); or, with `time`, a line's time grid
struct GridOut {
  std::optional<TimeGridOut> time;                 // a time grid: the other members are not used
  std::string label, description, unit;
  Json params;                                     // the ladder row's parameters (including the rule `grid` {kind, depth, slots, pieces})
  Json dictionary;                                 // the levels' dictionary to make: {name, description, entries: [...]}, or null
  std::optional<std::string> dictionary_id;        // or one already stored
  std::vector<GridLevelOut> levels;
};
// a Selection on a manifold: a partition or set (a code per element), spans (on a time line), or a tiling (a closed-form spans Selection)
// one KERNEL member of a kernel set: its anchor, a span [start, stop) on the manifold's coordinate (an instant when start == stop: the profile's centre), and its profile
struct KernelOut {
  double start = 0, stop = 0;
  std::int64_t code = 0;
  std::string family, chart;                       // gaussian | box ...; log2 | linear | log10
  double gain = 1;
  std::map<std::string, double> params;            // the family's parameters (gaussian: fwhm, in the chart's units)
};
struct SelectionOut {
  std::string name, output, type, manifold_id;     // type: partition · set · indices · spans · tiling · cells (a closed-form set: codes by a rule, no element stored) · grid (a dyadic grid with levels, or a line's time grid: `grid`) · product (`factor_ids`)
  std::optional<GridOut> grid;                     // grid
  std::optional<std::string> path;                 // default: io's node path for the name on the manifold
  std::vector<std::int64_t> codes;                 // indices: the elements picked, in order
  std::vector<std::int32_t> member_of;             // partition, set: the member NUMBER of each element (dense 0 .. n - 1, -1: none), made from the labels by compute's `from_labels` (pipelines)
  std::vector<std::int64_t> member_codes;          // partition, set: each member's LABEL, one per member (empty: member i is labelled i); `entries` name the labels
  std::vector<Span> spans;                         // spans: [start, stop) in the manifold's unit, each with its code
  std::optional<Tiling> tiling;                    // tiling
  std::optional<std::int64_t> n_members;           // cells: the number of cells
  std::vector<std::string> factor_ids;             // product (the Studio's D100, mixed radix, first factor fastest): the factors' stored selections, in order (>= 2)
  std::vector<int> factor_levels;                  // product (schema 53): the LEVEL of each factor that is a grid (`ladder`), -1 for a factor as it is; empty: all -1. A coupled level (a band's) is named here, never made a row of its own
  std::vector<std::int64_t> radices;               // product: each factor's radix; required for a factor that is a level of a grid (its member count there); empty: the factors' member counts
  std::vector<std::string> entries;                // the codes' names (a dictionary of the selection's own), when any
  std::optional<std::string> of_field_id, description, label, acquisition;
  Json params;                                     // the selection's parameters (null: none)
  Json producer;
  Values weights;                                  // one weight a member, float32 or float64 (empty: none): stored beside the selection and named in its parameters (`weights`)
  std::vector<KernelOut> kernels;                  // kernels: one member a kernel (type `kernels`): stored as a set whose members are kernel selections
};
// a closed-form line (a frequency axis, a bin axis): n points of `spacing` from `origin`, in `unit`; found by name and reused when it fits
struct LineOut {
  std::string name, alt;                           // `alt`: the name used when a line of `name` exists and does not fit
  std::int64_t n = 0;
  double spacing = 1, origin = 0;
  std::string unit = "Hz", dimension = "frequency", folder = "spectral", cs_key = "hz";
  std::optional<std::string> acquisition_id, derived_from_id;
  // a line whose points are NOT evenly spaced (an eigenvalue axis): its `coordinates` [n], strictly increasing, in `unit` along `dimension`, stored (topology path); `kinds`, one integer code a point (optional).
  // n, spacing and origin are not used then.
  std::vector<double> coordinates;
  std::vector<std::int32_t> kinds;
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
// a manifold of POINTS with ONE coordinate each (an operator's mode axis: the eigenvalues): found by path and REPLACED when asked, its coordinates [n] in float64
struct PointsOut {
  std::string name, output;
  std::optional<std::string> path;                 // default: io's folder for the operator's points (`eigenvalue/<name>`)
  std::string cs_key = "eigenvalue";               // its coordinate system
  std::string operator_name;                       // the operator whose by-product this axis is (`hodgeDirac`); recorded on the manifold
  std::optional<std::string> derived_from_id, description;
  Values positions;                                // [n], float64
  Json producer;
};
// a CELL COMPLEX on a set of vertices: a surface manifold whose vertices, edges (1-cells, [nE x 2] tail, tip) and faces (2-cells, [nF x 3]) are STORED (two edges may join one
// pair of vertices, so the incidence is read from the operators d0 and d1 beside it, never from vertex pairs); the cells' fields and operators lie on it
struct ComplexOut {
  std::string name, output;
  std::optional<std::string> path;                 // default: `source/<name>`
  std::string cs_key = "scs";
  std::optional<std::string> derived_from_id, description, acquisition_id;
  Values vertices;                                 // [n x 3], float64
  std::vector<std::int32_t> edges, faces;          // [nE x 2], [nF x 3]
  Json producer;
};
// a product of MANIFOLDS (a sensor × time line, a surface × frequency): its factors in axis order, existing or made in the same unit. (A product of SELECTIONS, with factor levels, is `SelectionOut` type "product".)
struct ProductOut {
  std::vector<std::string> factors;
  std::optional<std::string> acquisition_id;
  std::optional<std::string> name;                 // default: the factors' names joined by " × "
};
// a DICTIONARY of its own (scope dataset or subject): codes with their names and what goes with them; one that exists is refused
struct DictionaryOut {
  struct Entry {
    std::int64_t code = 0;
    std::string name;
    std::optional<std::string> label, description;
    std::optional<std::array<int, 3>> colour;      // [r, g, b], 0 to 255
    Json params;                                   // null: none
  };
  std::string name, scope = "subject";             // subject | dataset
  std::optional<std::string> description;
  std::vector<Entry> entries;
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
// A chart of a space: one coordinate system the space's manifolds are read in (the Frequency line: `linear` in Hz, the reference, and `log10`; the Eigenvalue line: the eigenvalue itself and the wavelength).
struct Chart {
  std::string id, name, space_id;
  std::optional<std::string> unit, transition, description;   // transition: identity (the reference) · affine · log · power · operator
  bool is_reference = false;
  Json transition_json;                            // the transition's parameters (log base, power scale and exponent), null: none
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
  // the spans of a spans selection, in member order: its stored rows, or a tiling's members by its rule (start, stop, code = the member number; a writer's subject)
  std::vector<Span> spans(const Node& selection) const;
  // a stored set per element as it is read: each element's member number, each member's label and the lookup (a writer's subject only; Error otherwise)
  selection::ElementMembers element_members(const std::string& selection_id) const;
  // the measurement QUERY (measurement_query.h: filters, roll-ups through a grid's hierarchy, to the whole, across subjects; each measure merged by its registry rule, in SQL): the rows, one a member
  Json query_members(const query::Query& q) const;
  // A GRID's LEVEL DICTIONARY (the Studio's surface-ladder / ConstantQLevelInfo attributes), what a roll-up to a physical size reads: for a ladder (or a level of one, or a time grid) the list
  // [{level, measurement?, unit?, attributes}] one entry a level, ascending. A dyadic grid: `measurement` the mean tile diameter in mm, its attributes (area_m2, diameter_mm, weyl_band_per_m2, ...) as
  // the maker wrote them; a constant-Q ladder: the level's window in s with its band; a line's TIME GRID (closed form, nothing stored per level): level k tiles 2^k samples, `measurement` its
  // seconds. Empty when the selection is of no grid.
  Json grid_levels(const std::string& selection_id) const;
  // scalar measurement rows written as they are ({selection_id, code, measure, value, weight?}: io's write_measurements), for a caller that holds values already (a writer's subject only)
  std::int64_t write_member_rows(const Json& rows, const std::string& computed_by) const;
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
  std::optional<std::string> manifold_named(const std::string& name) const;   // the id of the manifold with that name (the subject's, any status)
  // the one-sided frequency line of a span of n samples at fs on a time line, found by its key (its analytic `fft` operator row); none when it was never made (Unit::frequency_line makes it)
  std::optional<std::string> frequency_line_of(const std::string& time_line_id, std::int64_t n, double fs) const;
  // the time lines a complete recording lies on (the line factors of its product manifold): every recording's, first seen first; or the named recording's first
  std::vector<std::string> recording_time_lines() const;
  std::optional<std::string> recording_time_line(const std::string& recording) const;
  NodeCounts node_counts() const;

  // ---- cache lookup: the complete node whose producer record holds exactly `key` (a selection's: its parameters); with `path`, only that node
  std::optional<Node> cached(Cls, const Json& key, const std::optional<std::string>& path = std::nullopt) const;
  // the complete operator stored with `key` as its own provenance (`OperatorOut::cache_key`): its values are read by the caller through io's readers; none when it was never stored
  std::optional<Node> stored_operator(const Json& key) const;
  // the fields OF an operator (`of_operator_id`): an operator's eigenvalues are a field of it on its mode axis (the schema's `of_operator_id`), besides the axis's coordinates
  std::vector<Node> fields_of_operator(const std::string& operator_id) const;
  // the CHARTS of a manifold's space: its coordinate systems, the reference first (the Frequency line's `linear` and `log10`, the Eigenvalue line's eigenvalue and wavelength). Empty: no geometry
  std::vector<Chart> charts(const std::string& manifold_id) const;

  // ---- what is measured on a selection
  std::vector<Json> measured_kinds(const std::string& selection_id) const;            // io's measurement_kinds: {of_field_id, measure, members, min, max, computed_by, unit}
  // the selections that carry `measure` measured on `of_field_id` (ids, ascending)
  std::vector<std::string> measured_selections(const std::string& of_field_id, const std::string& measure) const;
  bool has_measures(const std::string& selection_id, const std::string& of_field_id) const;   // any measure of that field on the selection
  bool has_measure(const std::string& selection_id, const std::string& measure, const std::optional<std::string>& of_field_id = std::nullopt, const std::optional<std::string>& of_statistic_id = std::nullopt) const;

  // ---- read measurements: io's read-time query (nxr/io/measures.h, query_measurements): a stored measure as stored, a derived one from its ingredients, a pyramid's
  // array-backed measurement rolled up at read to the levels `q` names (that needs the subject's store: a Subject made from a writer)
  Json read_measurements(const Json& q) const;
  // the same measurements as a TABLE with filters and roll-ups, in SQL (nxr/io/measurement_query.h `query::members`, where the filter and roll-up structs are): through a grid's hierarchy and across subjects.
  // The query API is io's; the face (pipelines) applies nothing of its own.

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
                                   const std::optional<std::string>& measure = std::nullopt, const std::optional<std::string>& of_statistic_id = std::nullopt) const;
  void set_params(const std::string& selection_id, const Json& params) const;              // a selection's parameters, replaced
  // the projection a reading of recording `field` applies under the `projectors` option, with its provenance record (`load_projection`); the node form gives none when the field has no stored row
  ProjectionOf projection_of(const Row& field, Projectors projectors) const;
  std::optional<ProjectionOf> projection_of(const Node& field, Projectors projectors) const;

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
                                   const std::optional<std::string>& measure = std::nullopt, const std::optional<std::string>& of_statistic_id = std::nullopt);
  void set_params(const std::string& selection_id, const Json& params);
  // ---- typed writes, each returning the new node's id (call `replace` on its `path_of` first: a node already there is refused or moved aside)
  std::string field(const FieldOut&);
  std::string statistic(const StatisticOut&);
  std::string operator_(const OperatorOut&);
  std::string selection(const SelectionOut&);
  std::int64_t measurements(const MeasurementOut&);   // the rows written (one a measure); a measure already there for the field is replaced
  // MANIFOLDS (the sixth kind): made first, so the other results of the unit can lie on them; each found and reused when identical, never replaced
  std::string line(const LineOut&);
  // the one-sided frequency line of a span of n samples at fs on a time line of this subject (origin 0, spacing fs / n, n / 2 + 1 bins): found by (time line, n, fs) through its analytic `fft` operator
  // row, else a line of the same grid under `legacy_name` (the old per-recording name), else made with that operator row. Named `<time line>_freq_L<k>` for n = 2^k, else `_freq_n<n>`.
  std::string frequency_line(const std::string& time_line_id, std::int64_t n, double fs, const std::string& legacy_name = "");
  std::string surface(const SurfaceOut&);
  std::string product(const ProductOut&);
  std::string points(const PointsOut&);            // found by path and kept when it holds the same coordinates; another one's is replaced by `replace` first
  std::string cell_complex(const ComplexOut&);     // found by path and kept when it holds the same cells; another one's is replaced by `replace` first
  std::string dictionary(const DictionaryOut&);    // refused (already_exists) when a dictionary of that name and scope exists
  // EDITS of what is stored (each inside the unit, a node's description only, never its arrays): a manifold's label, description and primary flag (making a surface primary clears the one before it,
  // in the same unit); a dictionary's entry or its whole entry list
  void edit_manifold(const std::string& manifold_id, const std::optional<bool>& primary, const std::optional<std::string>& label, const std::optional<std::string>& description);
  struct EntryEdit { std::optional<std::string> name, label, description; std::optional<std::array<int, 3>> colour; Json params; };
  void edit_dictionary_entry(const std::string& dictionary_id, std::int64_t code, const EntryEdit& edit);
  void replace_dictionary_entries(const std::string& dictionary_id, const std::vector<DictionaryOut::Entry>& entries);
  // a PROJECTOR operator's status (0 inactive, 1 active; 2, already applied to the samples, is the importer's and is not set here): its parameters' `status` (and the store's attributes, synced from
  // the row), with an edit entry appended to its provenance (`producer_json.edits`: who, when, the note, the status before and after); nothing is removed. Another kind of operator is refused.
  // Returns the entry appended (null when the status was already the one asked: nothing changes).
  Json edit_projector_status(const std::string& operator_id, int status, const std::optional<std::string>& note, const std::optional<std::string>& by);
  // a node with no arrays (an analytic operator) removed so that a new one of its name takes its place
  void remove_unstored(Cls cls, const std::string& id);
  // the place a result would take (what `replace` and the cache look at)
  std::string path_of(const FieldOut&) const;
  std::string path_of(const StatisticOut&) const;
  std::string path_of(const OperatorOut&) const;
  std::string path_of(const SelectionOut&) const;
  std::string path_of(const PointsOut&) const;
  std::string path_of(const ComplexOut&) const;
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
