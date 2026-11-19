// nxr-io — the WRITER: a dataset and its subjects, written by the procedure. The twin of nxr.io.writer, with its nouns and its names.
//
//   auto ds  = create_dataset(datastore, "demo", {.source_tool = "x"});      // or open_writer("<datastore>/demo")
//   auto sub = SubjectWriter::create(ds, "S1");
//   sub.compose([&] {                                                       // complete on return, abort on an exception
//     auto m = sub.manifold({.name = "cortex", .path = "source/cortex", .type = "surface", .n_vertices = 12, ...});
//     sub.field({.name = "f", .path = "source/cortex/f", .kind = "map", .manifold_id = m, .data = view(values, {12})});
//   });
//
// THE PROCEDURE (D141/D142): CREATE = the rows, one transaction, `pending` -> LAY OUT (the node's zarr.json from the schema's view) ->
// POPULATE (the bytes) -> COMPLETE (`complete`, zarr.json refreshed). BYTES FIRST, THE ROW LAST. UPDATE = the row, then zarr.json. DELETE =
// `deleting` -> the arrays only this row named removed (D164) -> the row removed. RECOVER at open: `pending` rolled back, `deleting` finished.
// A composition that fails rolls back what it created (`abort`, `rollback_to`).
//
// ONE WRITER A DATASET (`writer_lock.h`): a writing open takes `<dataset>/.writer.lock` BEFORE it recovers or writes; closing releases it.
#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/io/array_writer.h"
#include "nxr/io/database.h"
#include "nxr/io/small_file_store.h"
#include "nxr/io/store.h"
#include "nxr/io/writer_lock.h"

namespace nxr::io {

using Populate = std::function<void(Store&, const std::string&)>;   // writes a node's bytes: given the datastore's store and the node's key prefix in it

// ── the dataset and the catalog (D140) ────────────────────────────────────────────────────────────────────────────

// RETIRED (docs/specs/2026-11-16-subject-centred-datasets.md section 3): the datastore and its `catalog.json` are replaced by datasets addressed by their own path
// (catalogue.h). Nothing of format 53 reads or writes it; these calls stay only to read and migrate a dataset of formats 50 to 52 (`split_dataset` leaves a catalog.json as it was).
// `<datastore>/catalog.json`, the index of the datastore's datasets; nullopt when absent or unreadable.
std::optional<Json> read_catalog(const std::filesystem::path& datastore);
// Rewritten whole and atomically from the datasets' databases. Its `id` is the datastore's.
Json write_catalog(const std::filesystem::path& datastore, const std::optional<std::string>& id = std::nullopt);

struct CreateOptions {
  std::optional<std::string> source_tool, source_protocol, source_path, created_by;
  int store_format = kStoreFormatNewest;   // 52: each subject's small keys are rows of its small_files.sqlite; 51: the folder tree only (what `dataset compact` moves from)
  std::optional<std::string> built_by;   // the database's meta `built_by` (default: created_by, else "nxr-convert"); the program, not the person
};
struct OpenOptions {
  bool lock = true;                      // false: a READER, no lock and no recovery
  std::optional<bool> recover;           // default: whether THIS process took the lock (a borrowed lock recovers nothing)
  // A READER opens SQLite read-only (SQLITE_OPEN_READONLY): it never migrates an older schema (50 to 53 is read as it is), never makes a table or a view and never writes `meta`.
  // Default: true for a reader (lock = false), false for a writer. Only a writable open, under the writer lock, migrates.
  std::optional<bool> read_only;
};

// An open dataset: its database, the datastore root its paths are under, and its row. Closing (or destroying) releases the writer lock.
class DatasetWriter {
 public:
  DatasetWriter(DatasetWriter&&) noexcept;
  DatasetWriter& operator=(DatasetWriter&&) noexcept;
  ~DatasetWriter();

  Database& db() { return *db_; }
  const Database& db() const { return *db_; }
  const std::filesystem::path& root() const { return root_; }       // the datastore folder
  // A dataset's store; throws in package mode (a subject package has no datastore: use node_store()).
  DatastoreStore& store();                                           // the datastore's keys: `<dataset>/<subject>/…` (format 52: a subject's small keys are rows of its small_files.sqlite)
  // The store the nodes' keys are in: keys `<dataset.path>/<subject.path>/…`. For a dataset it is store(); for a subject package it is the package's own keys under
  // that prefix (subject_package.h), so the writer, the procedure and the pipelines work on a package exactly as on a dataset.
  Store& node_store() { return package_store_ ? *package_store_ : static_cast<Store&>(*store_); }
  bool is_package() const { return static_cast<bool>(package_store_); }
  const Row& row() const { return row_; }                           // the dataset row
  std::string id() const { return text(row_, "id"); }
  std::filesystem::path folder() const { return root_ / text(row_, "path"); }
  std::optional<Row> subject(const std::string& name) const;
  std::optional<Row> template_() const;                             // the dataset's default subject
  bool locked() const { return static_cast<bool>(lock_); }
  // Close the writer: its `before_close` hook (a package's record and content hash) runs first and may throw (the lock is released all the same).
  void close();
  // Package mode (used by subject_package.cpp): the writer over a subject package's database. `make_store` is given the database in its final place.
  static DatasetWriter over_package(Database db, std::filesystem::path package_folder, Row home_dataset, WriterLock lock,
                                    const std::function<std::unique_ptr<Store>(Database&)>& make_store, std::function<void(DatasetWriter&)> before_close);

 private:
  friend DatasetWriter open_writer(const std::filesystem::path&, const OpenOptions&);
  friend DatasetWriter create_dataset(const std::filesystem::path&, const std::string&, const CreateOptions&);
  DatasetWriter(Database db, std::filesystem::path root, Row row, WriterLock lock, bool package = false);
  WriterLock lock_;   // declared before the database: released after it is closed
  std::unique_ptr<Database> db_;   // by pointer: the stores that share its connection keep their view of it across a move of this writer
  std::filesystem::path root_;
  std::unique_ptr<DatastoreStore> store_;   // by pointer: the subjects' views of it survive a move of this writer
  std::unique_ptr<Store> package_store_;    // package mode (declared after db_: destroyed first, it uses the database's connection)
  std::function<void(DatasetWriter&)> before_close_;
  Row row_;
};

// CREATE a dataset: its database from the schema, its row, its folder (laid out from the row), the catalog. The writer lock is taken.
DatasetWriter create_dataset(const std::filesystem::path& datastore, const std::string& name, const CreateOptions& opt = {});

// The app-wide `frequency bands` dictionary (D109), made if the dataset lacks it (each dataset carries its own copy, D140): returns its id. As nxr.io.writer's.
std::string ensure_band_dictionary(Database& db);

// OPEN a dataset by its folder (`<datastore>/<dataset>`): its database at this schema version, recovered.
DatasetWriter open_writer(const std::filesystem::path& folder, const OpenOptions& opt = {});
// DELETE a dataset: its folder (its database with it), then the catalog — refused while another writer holds it.
void delete_dataset(const std::filesystem::path& datastore, const std::string& name);
// The datasets of a datastore, from its catalog (throws `not_found` for a folder that is not a datastore).
Json list_datasets(const std::filesystem::path& datastore);

// ── the procedure ─────────────────────────────────────────────────────────────────────────────────────────────────
void set_status(const Database& db, const std::string& table, const std::string& id, const std::string& status);
// CREATE a node by the procedure: `rows` writes its row(s) inside the transaction; the node is laid out from them; `populate` writes its
// bytes at the node's location; then it is complete. A failure in `populate` rolls the node back.
void create_node(Database& db, Store& store, const std::string& table, const std::string& id,
                 const std::function<void()>& rows, const Populate& populate = nullptr);
void update_node(Database& db, Store& store, const std::function<void()>& rows);
// DELETE: marked 'deleting' (so a crash finishes it), the arrays only this row named removed (D164), then the row.
void remove_node(Database& db, Store& store, const std::string& table, const std::string& id);
// DELETE a row of `table` by id. A MANIFOLD (and the components that cascade with it) takes the geometries and topologies no manifold of its subject stands on
// any more (their dimensions and cells cascade): a surface deleted leaves no row behind naming its positions or faces. Any other table: its row.
void delete_row(const Database& db, const std::string& table, const std::string& id);
// THE STUDIO'S STORE NODE REMOVAL (service/model.ts removeNode): the node `rel` of the subject's store `store`, every segment found by its exact name and a real
// group (never a link), the target inside the store; false when it is not there. Throws invalid_argument when refused.
bool remove_store_node(Store& store, const std::string& rel);
// A ROW'S DELETE TAKES ITS ARRAYS (D164; the Studio's deletingRows): `del` runs; then every path the subject's rows named before it and no remaining row names
// (equal, or one inside the other) is removed from its store with remove_store_node. A refused removal is skipped (the row is gone). Returns the paths removed.
std::vector<std::string> deleting_rows(const Database& db, Store& store, const std::string& subject_id, const std::function<void()>& del);
// The row of the subject that names `rel` or overlaps it ("<table> <name>"), or nullopt.
std::optional<std::string> named_by(const Database& db, const std::string& subject_id, const std::string& rel);
struct Recovered { int rolled_back = 0; int finished = 0; };
// RECOVER: every 'pending' row rolled back, every 'deleting' row finished.
Recovered recover(Database& db, Store& store);

// ── coordinate systems and the f-vector ───────────────────────────────────────────────────────────────────────────
// The id of the coordinate system `key` names for a subject — its space and systems created on demand (scs · ncs · mni · freesurfer-tkr ·
// voxel:<name> · freesurfer-sphere · frequency · octave · clock · eigenvalue:<operator>).
std::string coordinate_system(Database& db, const std::string& subject_id, const std::string& key);
struct FVector { std::optional<std::int64_t> n_vertices, n_edges, n_faces, n_volumes; int rank = 0; };
struct Factor { std::optional<std::int64_t> n_vertices, n_edges, n_faces, n_volumes; int rank = 0; };
FVector product_counts(const std::vector<Factor>& factors);
// edges, faces and top cells of a regular grid whose vertices are the sample sites
FVector lattice_counts(const std::vector<std::int64_t>& ns);
// the BIDS entities an acquisition's label names, each nullopt when absent (`ses-02_task-rest_run-02` -> ses 02, task rest, run 02)
std::map<std::string, std::optional<std::string>> bids_entities(const std::string& label);

// ── a subject: the entities ───────────────────────────────────────────────────────────────────────────────────────

struct CellSpec { int rank = 2; std::string path; };
struct TopologySpec {
  std::string kind;                    // none · lattice · path · circle · simplicial · graph · product
  int rank = 0;
  int n_components = 1;
  std::optional<std::string> winding;
  std::vector<CellSpec> cells;
};
struct DimensionSpec {
  std::optional<std::string> name;
  std::int64_t n_vertices = 0;
  std::optional<double> spacing, origin;
  std::optional<std::string> unit;
  std::optional<std::string> manifold_id;   // a product's factor
};
struct GeometrySpec {
  std::string form;                          // closed_form · stored · product
  std::optional<std::string> cs_key;
  std::optional<std::string> positions_path;
  std::vector<DimensionSpec> dimensions;
  std::optional<Json> direction;
};

struct ManifoldSpec {
  std::string name;
  std::optional<std::string> path;          // nullopt: a manifold no node stores (a product)
  std::string type;                         // points · line · image · volume · curve · surface · tetrahedral · graph · product
  std::int64_t n_vertices = 0;
  TopologySpec topology;                    // its own topology, or
  std::string topology_id;                  // the id of one it shares (wins when given)
  std::optional<GeometrySpec> geometry;
  Populate populate;                        // writes its arrays (vertices · faces …)
  std::optional<std::string> acquisition;   // an acquisition's label (made on first use)
  std::optional<std::string> id;
  std::optional<std::string> created_utc;
  Cols cols;                                // any other column: n_edges · n_faces · n_volumes · is_primary · label · comment · source · producer_json · …
};

struct ContributionSpec { double weight = 1; std::string subject_id; std::optional<std::string> source_field_id; };
struct SparseArrays {                       // CSC / CSR (D160): a group at the node's path
  std::vector<std::pair<std::string, ArrayView>> parts;   // indptr · indices · data (SPEC §5)
};

struct FieldSpec {
  std::string name;
  std::optional<std::string> path;
  std::string kind;
  std::string manifold_id;
  std::string value_type = "scalar";        // scalar · integer · complex · vector2 · vector3 · matrix3 · quaternion
  std::optional<ArrayView> data;            // laid out from its meta and written at once, or
  std::optional<ArrayMeta> meta;            // an explicit layout (then `populate` writes the bytes)
  ArrayOptions array;                       // chunks · compress · shards · dense
  Populate populate;
  std::vector<ContributionSpec> contributions;
  std::optional<SparseArrays> sparse;
  std::optional<std::string> acquisition;
  std::optional<std::string> id;
  std::optional<std::string> created_utc;
  Cols cols;
};

// A STATISTIC (docs/specs/2026-11-13-statistics.md): computed from a partition's members over a span of a field (a Welch PSD, a cross-spectrum, a
// distribution). The values `data` are shaped by `axes` (each {name, size, and manifold_id | selection_id | edges | values_path, unit?}); the measure
// and the method are names of the statistics registry (statistics.json). `SubjectWriter::statistic` writes the row and the array in one composition and
// adds the table to a dataset made before it, in place (ensure_statistics).
struct StatisticSpec {
  std::string name;
  std::optional<std::string> path;          // store-relative; default `statistics/<name>`
  std::string measure, method;
  std::string value_type = "scalar";         // each value, as a field's: scalar · integer · complex · vector2 · vector3 · matrix3 · quaternion · hermitian3; its
                                             // shape trails the axes in `data` (complex [2], vector3 [3], quaternion [4], hermitian3 [3, 3, 2])
  std::optional<std::string> of_field_id, of_statistic_id, partition_selection_id;
  Json span, axes = Json::array(), components, scaling, options, producer;
  std::optional<std::int64_t> n_members, members_skipped;
  std::optional<double> dof;
  bool detached = false;                    // computed elsewhere (an imported covariance): no field in the package; its provenance is in `options`
  std::optional<std::string> unit, description, label, acquisition, id, created_utc;
  ArrayView data;
  ArrayOptions array;
};

struct OperatorSpec {
  std::string name;
  std::optional<std::string> path;
  std::string kind;
  std::string from_manifold_id;
  std::string layout = "dense";             // dense · csc · csr · analytic …
  std::int64_t n_rows = 0, n_cols = 0;
  std::optional<ArrayView> data;
  std::optional<ArrayMeta> meta;
  ArrayOptions array;
  Populate populate;
  std::optional<SparseArrays> sparse;
  std::optional<std::string> acquisition;
  std::optional<std::string> id;
  std::optional<std::string> created_utc;
  Cols cols;
};

struct SpanSpec { double start = 0, stop = 0; std::optional<std::int64_t> code; };
struct ExtentSpec { std::int64_t axis = 0; double start = 0, stop = 0; };

struct SelectionSpec {
  std::string name;
  std::optional<std::string> path;
  std::string type;                         // point · span · distance · spans · indices · kernel · set · ladder
  std::optional<std::string> manifold_id;
  std::optional<ArrayView> data;            // its per-element array (codes of a partition, indices of a subset) at its path
  std::optional<ArrayMeta> meta;
  std::optional<ArrayView> elements;        // what the row counts (default: `data`)
  bool no_elements = false;                 // the row counts nothing
  bool picked = false;                      // a subset: n_elements only (a partition also counts its distinct codes)
  std::optional<std::vector<SpanSpec>> spans;          // member rows (D87)
  std::vector<ExtentSpec> extents;
  std::vector<Cols> measurements;           // its own (array-backed or valued) rows
  Populate populate;
  bool dense = false;
  std::optional<std::string> acquisition;
  std::optional<std::string> id;
  std::optional<std::string> created_utc;
  Cols cols;                                // dictionary_id · code · cell · params_json · …
};

struct DictionaryEntry {
  std::int64_t code = 0;
  std::string name;
  std::optional<std::int64_t> ordinal;
  std::optional<std::array<int, 3>> color;
  Cols cols;                                // measurement · measurement_unit · function · params_json · attributes_json …
};
struct DictionarySpec {
  std::string name;
  std::string scope = "subject";            // app · dataset · subject
  std::vector<std::string> names;           // the names to code 0…n-1 (or `entries`)
  std::vector<std::optional<std::array<int, 3>>> colors;
  std::optional<std::vector<DictionaryEntry>> entries;
  std::optional<std::string> description;
};
struct DictionaryResult { std::string id; std::vector<std::int64_t> codes; };

struct SubjectOptions {
  std::string kind = "subject";             // subject · template · group
  std::optional<std::string> folder;
  std::optional<std::string> source_format, source_path, created_by;
  bool default_template = true;             // a `subject` points at the dataset's default subject (D127) …
  std::optional<std::string> template_id;   // … or at this one
  bool immediate = false;                   // the whole procedure per node
};
struct OpenSubjectOptions { bool immediate = false; std::optional<std::string> created_by; };

class SubjectWriter {
 public:
  // CREATE a subject (pending) — its store folder follows its row.
  static SubjectWriter create(DatasetWriter& ds, const std::string& name, const SubjectOptions& opt = {});
  // An existing (complete) subject, by name, path or id, to ADD entities to.
  static SubjectWriter open(DatasetWriter& ds, const std::string& name, const OpenSubjectOptions& opt = {});

  const std::string& id() const { return id_; }
  const std::string& name() const { return name_; }
  const Row& row() const { return row_; }
  // The subject's keys (`source/m/vertices/zarr.json`: what its rows' `path` columns name), a view of the dataset's store.
  Store& store() { return *kv_; }
  const Store& store() const { return *kv_; }
  // The subject's folder on disk (for messages and JSON output; every read and write goes through `store()`).
  const std::filesystem::path& folder() const { return store_; }
  // The dataset's store (keys `<dataset>/<subject>/…`) and this subject's place in it.
  DatasetWriter& dataset() const { return *ds_; }          // the dataset (or subject package) it writes into
  Store& datastore() const { return ds_->node_store(); }   // the store the nodes' keys are in (a dataset's, or a subject package's own)
  const std::string& prefix() const { return rel_; }
  Database& db() { return *db_; }

  // COMPLETE the composition: every node it made -> 'complete', then ONE drain.
  Drained complete();
  // Remove everything this composition made — the subject, when it made it — and the dictionaries and acquisitions nothing reads any more.
  void abort();
  // `body` as a composition: complete on return, abort on any exception (which propagates).
  void compose(const std::function<void()>& body);
  // A point to roll the composition back to; a part that fails is removed, the rest kept.
  std::size_t mark() const { return created_.size(); }
  std::size_t rollback_to(std::size_t mark);

  // ── the entities ──
  // A MANIFOLD node: its topology, its geometry, its row; `populate` writes its arrays.
  std::string manifold(ManifoldSpec spec);
  // The PRODUCT of manifolds, in array-dimension order (D37) — no node, a row its subject's root carries.
  std::string product(const std::vector<std::string>& factors, const std::optional<std::string>& acquisition_id = std::nullopt,
                      bool inherit_acquisition = true, const std::optional<std::string>& name = std::nullopt);
  // A FIELD node: its row (the array's layout from its meta), laid out, then populated.
  std::string field(FieldSpec spec);
  // An OPERATOR node: dense (an array), sparse (a group of indptr · indices · data), analytic (no bytes) or declared without its bytes.
  std::string operator_(OperatorSpec spec);
  std::string statistic(StatisticSpec spec);
  // A SELECTION node (and its spans, extents and measurements).
  std::string selection(SelectionSpec spec);
  // A selection's measurement row (added to a selection of this subject). The selection CRUD (save, load, update, delete, lookup, measurements) is gathered in io's
  // selection module (selection_store.h, selection_measurement.h); these members are its entry points for a composition and delegate to it.
  void measurement(const std::string& selection_id, const Cols& row);
  // UPDATES OF A STORED SELECTION (the Studio's selections.setSpans / setExtents / setMeasurements): each replaces what it names, in one transaction
  void set_spans(const std::string& selection_id, const std::vector<SpanSpec>& spans);            // a spans set's member rows (n_members follows)
  void set_extents(const std::string& selection_id, const std::vector<ExtentSpec>& extents);      // a span's corners
  void set_own_measurements(const std::string& selection_id, const std::vector<Cols>& rows);      // its own values (code null); its members' rows are the measurement path's
  // A DICTIONARY by name in its scope, and the CODE each requested name has in it.
  DictionaryResult dictionary(const DictionarySpec& spec);
  // THE ACQUISITION a node was made from or for (D161): the subject's row of that label, made on first use with its BIDS entities.
  std::optional<std::string> acquisition(const std::optional<std::string>& label);
  std::string topology(const TopologySpec& spec);
  std::string geometry(const GeometrySpec& spec);
  std::string cs(const std::string& key);   // coordinate_system for this subject
  // A coordinate system on a space of the subject (`head`, the subject's head space, or a space of its own), found by name or made. A system that is not
  // its space's first is defined by a transition OPERATOR (kind `transition`), which the caller writes in the same compose: never an `affine` with no parameters.
  struct FrameSpec {
    std::string space, name;                 // the space (made when new) and the system on it
    std::optional<std::string> unit, description;
    std::string space_description;
  };
  std::string frame(const FrameSpec& f);
  // An array at a store-relative path, laid out (canonical zarr.json) and populated; the groups above it plain groups. With `table`/`id` the
  // node's attributes ride in its document from the start.
  ArrayMeta write_array(const std::string& path, const ArrayView& data, const ArrayOptions& opt = {}, const std::optional<ArrayMeta>& meta = std::nullopt,
                        const std::string& table = "", const std::string& id = "");
  // A sparse group: indptr · indices · data at `path/<part>`.
  void write_sparse(const std::string& path, const SparseArrays& parts);
  // DELETE one of this subject's nodes (a row of manifold · field · operator · selection · subject).
  void remove(const std::string& table, const std::string& id);
  // CHANGE the rows of a stored node of this subject: `body` runs as an update (the whole procedure per call when the subject is immediate, else one transaction).
  // The selection module's updates (selection_store.h) run through this; the SQL is theirs.
  void update(const std::function<void()>& body);

  // ── reads ──
  std::optional<Row> find(const std::string& table, const std::string& path) const;
  std::optional<Row> by_name(const std::string& table, const std::string& name) const;

  SubjectWriter(SubjectWriter&&) = default;
  SubjectWriter& operator=(SubjectWriter&&) = default;

 private:
  SubjectWriter(DatasetWriter& ds, Row row, bool immediate, bool owns_subject, std::optional<std::string> created_by);
  std::string node(const std::string& table, const std::string& id, const std::function<void()>& rows, const Populate& populate);
  Cols acquired(Cols cols, const std::optional<std::string>& acquisition);
  Row prov(const std::optional<std::string>& created_utc) const;
  Populate array_populate(const std::string& table, const std::string& id, const std::optional<ArrayMeta>& meta, const std::optional<ArrayView>& data,
                          const Populate& populate, bool dense);
  void drop_unused(const std::vector<std::string>& dictionary_ids, const std::vector<std::string>& acquisition_ids);
  std::optional<std::string> find_product(const std::vector<std::string>& factors) const;

  DatasetWriter* ds_;
  Database* db_;
  std::filesystem::path root_;
  Row row_;
  std::string id_, name_, rel_;
  std::filesystem::path store_;
  std::unique_ptr<PrefixStore> kv_;
  bool immediate_, owns_subject_;
  std::vector<std::pair<std::string, std::string>> created_;
  std::optional<std::string> created_by_;
  std::map<std::vector<std::string>, std::string> products_;
};

}  // namespace nxr::io
