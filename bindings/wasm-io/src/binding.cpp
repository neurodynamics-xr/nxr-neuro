// nxr-io in WebAssembly — the module behind `wasmClient` (js/index.mjs; spec docs/specs/2026-11-11-wasm-io.md).
//
// A `Datastore` opened on a datastore folder (as the host's file system shows it: the real disk in Node). Every method returns a JS value; a failure
// returns `{ error, code }` (code: nxr::io::code), which the JS layer throws as an Error. Rows travel as JSON text, array bytes as Uint8Array copies.
//   status()                                   { root, file, catalog }  (the datastore's catalog.json, D140)
//   bundle(storePath)                          a subject's bundle (nxr/io/bundle.h: the Studio's bundleOf), or null
//   view(name, whereJson, limit)               the Studio's `model.list`: SELECT * FROM <view|table> WHERE <col> = ? … over EVERY dataset's database (the
//                                              union, the datastore row once, a dataset row with its `writer_lock` holder or null), the allowed names only
//   row(table, id)                             the Studio's `model.read`: the first dataset's row with that id, or null
//   read(storePath, arrayPath, sliceJson)      { dtype, shape, bytes }  ([[start, stop], …] per axis, or null for the whole array)
//   create(storePath, arrayPath, metaJson)     lay out an array: { shape, dtype, chunks?, fill_value? } → { path, fresh }
//   put(storePath, arrayPath, bytes, dtype, regionJson)   write any region [lo, hi) per axis (io's write_region: read-modify-write of the chunks it touches)
//   remove(storePath, arrayPath)               delete an array's folder (no row may name it: the caller's check)
//   updateSubject(id, patchJson, by)           the Studio's `updateSubject`: name · kind · template_id · source_format · source_path only; path and dataset kept
//   removeSubject(id)                          the Studio's `removeSubject`: its selections, then its row (the rest cascades); the catalog rewritten; the folder is the caller's
//   createField(rowJson, by, childrenJson)     the Studio's `createField` (service/model.ts) + `fieldRepository.create`: a VIRTUAL field (no bytes: its input
//                                              and operator named) or a STORED one whose array already arrived (D59) — its layout read off the store, its
//                                              value count checked against the manifold's cells; contributions and child selections in one transaction;
//                                              then the sync queue drained (zarr.json attributes). A field made BY an operator FROM an input must meet
//                                              its step (the Studio's stepProblems, D106).
//   selection(op, rowJson, childrenJson, by)   the Studio's selectionRepository: op create · update · delete (rowJson the id); children {extents, spans,
//                                              measurements (its own, code null)}; the cell checked; a position in its space's system; then the drain
//   measurements(op, json, by)                 the Studio's measurementRepository (nxr/io/measures.h): op post (rows) · kinds (selection id) · query (q) · delete (where)
//   operator(op, rowJson, childrenJson, by)    the Studio's createOperator + operatorRepository: create (analytic: no bytes; stored: checked against its
//                                              array(s)) · update · delete; cells, the inverse pairing and the composition's chain checked; factors written
//   fieldUpdate(rowJson, by) · fieldDelete(id)  the Studio's fieldRepository.update / delete
//   manifold(op, rowJson, childrenJson, by)    the Studio's createManifold + manifoldRepository: create — a PRODUCT from its factors (productManifoldFrom), or a
//                                              row with its topology (+ cells) and geometry (+ dimensions; a `coordinate_system_key` resolved to the subject's
//                                              system), one transaction, then the drain · update · delete (its geometry and arrays with it)
//   dictionary(op, rowJson, childrenJson, by)  the Studio's dictionaryRepository: create · update (entries given replace; the selections reading it touched) ·
//                                              delete. An APP-scope dictionary is written to every dataset's database, as the service routes it
//   job(op, json, by)                          the Studio's jobRepository: get (an id, or a where {subjectId, datasetId, status, kind}) · create · update
//                                              (running stamps started_utc; done · failed · cancelled stamp finished_utc)
//   layout(storePath, qJson)                   the Studio's `layout` (D157/D164): where a node named `name` goes — with a manifold of the subject (its id), or one
//                                              about to be written (`type` with `role` / `operator` / `manifoldName`; a product's `factors`: ids or `<type>:<role>`)
//   datasetCreate(name, sourceJson)            the Studio's ensureDataset: idempotent by name; a new one's database from the schema, its rows, its folder, the band
//                                              dictionary, the catalog (io's create_dataset); held for writing
//   datasetDelete(name)                        the Studio's removeDataset (D167): the whole folder (database, subject stores, lock) and its catalog entry; a dataset
//                                              that is not there is not an error; refused while another writer holds it
//   catalog()                                  the datastore's catalog, rewritten from the databases (D140)
//   cli(argvJson)                              `nxr-neuro <argv>` IN PROCESS — the command line's own code: {code, lines (its JSON lines, parsed), stderr}
//   importSubjects(reqJson)                    the Studio's importSubjects + importBrainstorm (db/import.ts), with `nxr-neuro convert …` (above) where the Studio runs
//                                              its converter: the checks, a replace's `subject delete` and the riders re-linked, the counts of what was added
//   close()                                    release every dataset held for writing
// A store path is `<dataset path>/<subject path>`; its dataset's database is `<root>/<dataset path>/dataset.sqlite`.
// A dataset WRITTEN is opened once and held (its writer lock with it) until close(), as the Studio's data service holds the datasets it serves: other
// writers are kept out, and pending sync jobs merge as they do there (a reopen would run the recovery and settle them early).
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <fstream>
#include <map>
#include <memory>
#include <set>

#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/bundle.h"
#include "nxr/io/cells.h"
#include "nxr/io/measures.h"
#include "nxr/io/errors.h"
#include "nxr/io/layout.h"
#include "nxr/io/store.h"
#include "nxr/io/writer.h"
#include "nxr/io/writer_lock.h"

#include <iostream>
#include <sstream>

int nxr_neuro_cli_main(int argc, char** argv);   // cli/main.cpp's main, compiled into this module

using emscripten::val;
using namespace nxr::io;
namespace fs = std::filesystem;

namespace {

val failure(const std::string& message, const std::string& c) {
  val o = val::object();
  o.set("error", message);
  o.set("code", c);
  return o;
}
template <class F>
val guarded(F&& f) {
  try {
    return f();
  } catch (const Error& e) {
    return failure(e.what(), e.code());
  } catch (const std::exception& e) {
    return failure(e.what(), "internal");
  }
}
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }

// a relative path inside the datastore: no absolute path, no `..`
fs::path safe_rel(const std::string& p, const char* what) {
  const fs::path r(p);
  if (p.empty() || r.is_absolute()) invalid(std::string(what) + ": '" + p + "' is not a relative path");
  for (const auto& part : r) if (part == "..") invalid(std::string(what) + ": '" + p + "' leaves the datastore");
  return r;
}
std::vector<std::int64_t> ints(const Json& j) {
  std::vector<std::int64_t> v;
  for (const auto& x : j) v.push_back(x.get<std::int64_t>());
  return v;
}
val bytes_val(const std::uint8_t* p, std::size_t n) { return val::global("Uint8Array").new_(emscripten::typed_memory_view(n, p)); }   // a copy
val json_val(const Json& j) { return val(j.dump()); }

// the Studio's column lists (backend/src/db/database.ts): an upsert writes every one, NULL for one not given
const std::vector<std::string> kFieldCols = {"id", "subject_id", "name", "path", "kind", "label", "acquisition_id", "description", "manifold_id", "cell", "value_type", "n_components", "unit",
    "data_type", "layout", "nnz", "chunk_shape_json", "codecs_json", "fill_value_json", "of_field_id", "of_operator_id", "by_operator_id", "function", "derived_from_id", "comment",
    "source", "producer_json", "created_utc", "created_by", "modified_utc", "modified_by"};
const std::vector<std::string> kOperatorCols = {"id", "subject_id", "name", "path", "kind", "label", "acquisition_id", "description", "from_manifold_id", "to_manifold_id", "from_cell", "to_cell",
    "from_selection_id", "to_selection_id", "from_value_type", "to_value_type", "from_coordinate_system_id", "to_coordinate_system_id", "layout", "method", "params_json", "n_rows", "n_cols",
    "nnz", "data_type", "chunk_shape_json", "codecs_json", "fill_value_json", "derived_from_id", "inverts_id", "inverse_kind", "comment", "source", "producer_json", "created_utc", "created_by",
    "modified_utc", "modified_by"};
const std::vector<std::string> kManifoldCols = {"id", "subject_id", "name", "path", "type", "label", "acquisition_id", "is_primary", "n_vertices", "n_edges", "n_faces", "n_volumes",
    "topology_id", "geometry_id", "component_of_id", "component_ordinal", "from_index", "derived_from_id", "operator", "comment", "source", "producer_json", "created_utc", "created_by",
    "modified_utc", "modified_by"};
const std::vector<std::string> kTopologyCols = {"id", "subject_id", "kind", "rank", "n_components", "winding", "closed", "orientable"};
const std::vector<std::string> kGeometryCols = {"id", "subject_id", "form", "coordinate_system_id", "positions_path", "direction_json"};
const std::vector<std::string> kSelectionCols = {"id", "subject_id", "name", "path", "type", "label", "acquisition_id", "description", "manifold_id", "space_id", "cell", "of_field_id",
    "member_of_id", "member_ordinal", "code", "dictionary_id", "n_elements", "n_members", "array_path", "center", "position", "coordinate_system_id", "radius", "unit",
    "weight_fn", "params_json", "created_utc", "created_by", "modified_utc", "modified_by"};
Value value_of(const Json& x) {
  if (x.is_null()) return Value(std::monostate{});
  if (x.is_boolean()) return Value(static_cast<std::int64_t>(x.get<bool>()));
  if (x.is_number_integer()) return Value(x.get<std::int64_t>());
  if (x.is_number()) return Value(x.get<double>());
  if (x.is_string()) return Value(x.get<std::string>());
  return Value(x.dump());
}
void upsert(const Database& db, const std::string& table, const std::vector<std::string>& cols, const Json& r) {
  std::string names, marks, set;
  std::vector<Value> args;
  for (const std::string& c : cols) {
    names += (names.empty() ? "" : ", ") + c;
    marks += marks.empty() ? "?" : ", ?";
    if (c != "id") set += (set.empty() ? "" : ", ") + c + " = excluded." + c;
    args.push_back(r.contains(c) ? value_of(r[c]) : Value(std::monostate{}));
  }
  db.execute("INSERT INTO " + table + " (" + names + ") VALUES (" + marks + ") ON CONFLICT (id) DO UPDATE SET " + set, args);
}
std::string str_or(const Json& r, const char* k, const std::string& d = "") { return r.contains(k) && r[k].is_string() ? r[k].get<std::string>() : d; }
bool given(const Json& r, const char* k) { return r.contains(k) && !r[k].is_null() && !(r[k].is_string() && r[k].get<std::string>().empty()); }
// a selection's cell on the way in (the Studio's selectionCell, D94): a span is edges; a ladder on a line is edge chains
std::string selection_cell(const Database& db, const Json& r) {
  const std::string c = r.contains("cell") && !r["cell"].is_null() ? r["cell"].get<std::string>() : "0";
  if (c != "0" || !given(r, "manifold_id")) return c;
  const auto t = db.one("SELECT t.rank AS rank, m.type AS type FROM manifold m JOIN topology t ON t.id = m.topology_id WHERE m.id = ?", {Value(r["manifold_id"].get<std::string>())});
  const std::string type = str_or(r, "type");
  if (type == "span" || type == "spans") {
    std::int64_t rank = 1;
    if (t) if (const auto* x = std::get_if<std::int64_t>(&t->at("rank"))) rank = *x;
    return std::to_string(std::max<std::int64_t>(1, rank));
  }
  if (type == "ladder" && t && std::get<std::string>(t->at("type")) == "line") return "1";
  return c;
}
// A FIELD'S STEP (D106; the Studio's db/pairs.ts stepProblems, its refusals): the operator reads the input's manifold (or a factor of it) and writes the
// field's; a registration carries a vertex's value by kind (scalar → scalar, vector → vector3) on the vertices, either way along the correspondence; the value
// types agree unless the field is a pointwise nonlinear function of the output.
std::vector<std::string> step_refusals(const Database& db, const Json& field, const Row& op_row, const Json& input) {
  const Json op = row_json(op_row);
  auto S = [](const Json& j, const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
  auto factors_of = [&](const std::string& mid) {
    std::vector<std::string> out;
    const auto m = db.one("SELECT geometry_id FROM manifold WHERE id = ?", {Value(mid)});
    if (!m) return out;
    const auto g = m->find("geometry_id");
    if (g == m->end() || !std::holds_alternative<std::string>(g->second)) return out;
    for (const Row& d : db.all("SELECT manifold_id FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal", {g->second}))
      if (const auto* x = std::get_if<std::string>(&d.at("manifold_id"))) out.push_back(*x);
    return out;
  };
  auto on = [&](const std::string& m, const std::string& where) {
    if (m == where) return true;
    for (const auto& f : factors_of(where)) if (f == m) return true;
    return false;
  };
  const std::string fname = S(field, "name"), oname = S(op, "name"), iname = S(input, "name");
  const std::string from = S(op, "from_manifold_id"), to = S(op, "to_manifold_id"), in_m = S(input, "manifold_id"), f_m = S(field, "manifold_id");
  const bool registration = S(op, "kind") == "registration";
  std::vector<std::string> refused;
  const bool reversed = registration && !to.empty() && !on(from, in_m) && on(to, in_m) && on(from, f_m);
  if (!reversed) {
    if (!on(from, in_m)) refused.push_back(fname + ": " + oname + " reads a manifold its input " + iname + " is not on");
    if (!to.empty() && !on(to, f_m)) refused.push_back(fname + ": " + oname + " writes a manifold the field is not on");
  }
  const std::string fvt = S(field, "value_type"), ivt = S(input, "value_type");
  auto cell = [&](const Json& j, const char* k) { const std::string c = S(j, k); return c.empty() ? std::string("0") : c; };
  if (registration) {
    const bool vec = ivt == "vector3" || ivt == "quaternion" || ivt == "complex";
    if (!((ivt == "scalar" && fvt == "scalar") || (vec && fvt == "vector3")))
      refused.push_back(fname + ": " + oname + " carries a scalar to a scalar or a vector to a vector3 — not " + ivt + " to " + fvt);
    if (cell(field, "cell") != "0" || cell(input, "cell") != "0" || cell(op, "from_cell") != "0" || cell(op, "to_cell") != "0")
      refused.push_back(fname + ": " + oname + " maps vertices — the fields must be on the vertices (cell '0')");
  }
  const std::string ofvt = S(op, "from_value_type"), otvt = S(op, "to_value_type"), fn = S(field, "function");
  if (!registration && !ofvt.empty() && !ivt.empty() && ofvt != ivt && !(ofvt == "vector3" && ivt == "quaternion"))
    refused.push_back(fname + ": " + oname + " reads " + ofvt + ", its input " + iname + " is " + ivt);
  const bool pointwise = fn == "magnitude" || fn == "power" || fn == "abs" || fn == "log";
  if (!registration && !otvt.empty() && !fvt.empty() && otvt != fvt && !pointwise && fn.empty())
    refused.push_back(fname + ": " + oname + " writes " + otvt + ", the field is " + fvt);
  return refused;
}

// what the store holds at `location` (the Studio's storedArray): shape, data type, the CHUNK (the inner one of a shard), codecs, fill value — as JSON text
struct Stored { std::vector<std::int64_t> shape; std::string data_type; Json chunk_shape_json, codecs_json, fill_value_json; };
Stored stored_array(const Store& store, const std::string& location) {
  const auto raw = store.get_text(join_key(location, "zarr.json"));
  if (!raw) throw Error(code::not_found, location + ": no array in the store — POST it through nxr:// first");
  const Json doc = Json::parse(*raw);
  if (doc.value("node_type", std::string()) != "array" || !doc.contains("shape") || !doc.contains("data_type")) throw Error(code::invalid_argument, location + ": not an array");
  Stored st;
  for (const auto& x : doc["shape"]) st.shape.push_back(x.get<std::int64_t>());
  st.data_type = doc["data_type"].get<std::string>();
  // the chunk: a sharded array's inner chunk (sharding_indexed's configuration.chunk_shape), else the grid's
  Json chunk;
  if (doc.contains("codecs")) for (const auto& c : doc["codecs"]) if (c.value("name", std::string()) == "sharding_indexed") chunk = c["configuration"]["chunk_shape"];
  if (chunk.is_null() && doc.contains("chunk_grid")) chunk = doc["chunk_grid"]["configuration"]["chunk_shape"];
  st.chunk_shape_json = chunk.is_null() ? Json(nullptr) : Json(chunk.dump());
  st.codecs_json = doc.contains("codecs") ? Json(doc["codecs"].dump()) : Json(nullptr);
  st.fill_value_json = doc.contains("fill_value") ? Json(doc["fill_value"].dump()) : Json(nullptr);
  return st;
}

// what the renderer may read, by name (the Studio's service/model.ts VIEWS and TABLES; keep in step)
const std::set<std::string> kViews = {"manifold_view", "recording_view", "operator_view", "selection_view", "field_view", "acquisition_view", "subject_status", "atlas_view", "dashboard_counts"};
const std::set<std::string> kTables = {"datastore", "dataset", "subject", "acquisition", "manifold", "selection", "field", "operator", "dictionary", "coordinate_system", "job"};
bool column_name(const std::string& s) {   // /^[a-z_][a-z0-9_]*$/
  if (s.empty() || !(std::islower(static_cast<unsigned char>(s[0])) || s[0] == '_')) return false;
  for (char c : s) if (!(std::islower(static_cast<unsigned char>(c)) || std::isdigit(static_cast<unsigned char>(c)) || c == '_')) return false;
  return true;
}

class Datastore {
 public:
  // `program`: the program the datasets this client makes are stamped as built by (the database's meta `built_by`), as the Studio's service names itself
  Datastore(const std::string& root, const std::string& program) : root_(root), store_(root_), program_(program.empty() ? "nxr-neuro" : program) {}

  val status() const {
    return guarded([&] {
      const fs::path f = root_ / kCatalogFile;
      if (!fs::is_regular_file(f)) throw Error(code::not_found, f.string() + ": no catalog (not a datastore)");
      std::ifstream in(f);
      const Json cat = Json::parse(in);
      return json_val(Json{{"root", root_.string()}, {"file", f.string()}, {"catalog", cat}});
    });
  }

  val bundle(const std::string& store_path) const {
    return guarded([&] {
      DatasetWriter& ds = reader(store_path);
      const auto b = bundle_by_path(ds.db(), store_path);
      return json_val(b ? *b : Json(nullptr));
    });
  }

  val view(const std::string& name, const std::string& where_json, int limit) const {
    return guarded([&] {
      if (!kViews.count(name) && !kTables.count(name)) invalid(name + ": not a view the renderer may read");
      const Json where = where_json.empty() ? Json::object() : Json::parse(where_json);
      std::string sql = "SELECT * FROM " + name;
      std::vector<Value> args;
      std::string sep = " WHERE ";
      for (const auto& [k, v] : where.items()) {
        if (!column_name(k)) invalid(k + ": not a column name");
        if (v.is_null()) sql += sep + k + " IS NULL";
        else { sql += sep + k + " = ?"; args.push_back(v.is_string() ? Value(v.get<std::string>()) : v.is_number_integer() ? Value(v.get<std::int64_t>()) : Value(v.get<double>())); }
        sep = " AND ";
      }
      const int lim = limit > 0 ? limit : 5000;
      sql += " LIMIT " + std::to_string(std::max(1, std::min(lim, 100000)));
      Json rows = Json::array();
      std::set<std::string> datastores;
      for (const std::string& d : datasets()) {
        DatasetWriter& ds = reader(d);
        for (const Row& r : ds.db().all(sql, args)) {
          Json j = row_json(r);
          if (name == "datastore" && !datastores.insert(j.value("id", std::string())).second) continue;
          if (name == "dataset") { const auto h = read_writer_lock(root_ / d); j["writer_lock"] = h ? *h : Json(nullptr); }
          rows.push_back(j);
        }
      }
      Json out = Json::array();
      for (std::size_t i = 0; i < rows.size() && static_cast<int>(i) < lim; ++i) out.push_back(rows[i]);
      return json_val(out);
    });
  }

  val row(const std::string& table, const std::string& id) const {
    return guarded([&] {
      if (!kTables.count(table)) invalid(table + ": not a table the renderer may read");
      for (const std::string& d : datasets()) {
        DatasetWriter& ds = reader(d);
        if (const auto r = ds.db().one("SELECT * FROM " + table + " WHERE id = ?", {Value(id)})) return json_val(row_json(*r));
      }
      return json_val(Json(nullptr));
    });
  }

  val read(const std::string& store_path, const std::string& array_path, const std::string& slice_json) const {
    return guarded([&] {
      const ArrayReader r = ArrayReader::open(store_, array_key(store_path, array_path));
      std::vector<std::uint8_t> bytes;
      std::vector<std::int64_t> shape = r.shape();
      const Json slice = slice_json.empty() ? Json(nullptr) : Json::parse(slice_json);
      if (slice.is_null()) {
        bytes = r.read();
      } else {
        if (!slice.is_array() || slice.size() != shape.size()) invalid("read: the slice has " + std::to_string(slice.size()) + " axes, the array " + std::to_string(shape.size()));
        std::vector<std::int64_t> start, stop;
        for (std::size_t a = 0; a < shape.size(); ++a) {
          start.push_back(slice[a][0].get<std::int64_t>());
          stop.push_back(slice[a][1].get<std::int64_t>());
          shape[a] = stop[a] - start[a];
        }
        bytes = r.read(start, stop);
      }
      val o = val::object();
      o.set("dtype", std::string(dtype_to_string(r.dtype())));
      val sh = val::array();
      for (auto s : shape) sh.call<void>("push", static_cast<double>(s));
      o.set("shape", sh);
      o.set("bytes", bytes_val(bytes.data(), bytes.size()));
      return o;
    });
  }

  val create(const std::string& store_path, const std::string& array_path, const std::string& meta_json) const {
    return guarded([&] {
      const std::string at = array_key(store_path, array_path);
      if (store_.exists(join_key(at, "zarr.json"))) throw Error(code::already_exists, array_path + ": an array is already there");
      const bool fresh = !store_.exists_prefix(at);
      const Json m = Json::parse(meta_json);
      ArrayOptions ao;
      ao.in_pieces = true;   // recorded in zarr.json (so `dataset compact` keeps it unsharded); `put` writes any region: a region of a sharded array is refused
      if (m.contains("chunks") && !m["chunks"].is_null()) ao.chunks = ints(m["chunks"]);
      ArrayMeta meta = array_meta(dtype_from_string(m.at("dtype").get<std::string>()), ints(m.at("shape")), ao);
      if (m.contains("fill_value") && !m["fill_value"].is_null()) meta.fill_value = m["fill_value"];
      layout_array(store_, at, meta);
      return json_val(Json{{"path", array_path}, {"fresh", fresh}});
    });
  }

  val put(const std::string& store_path, const std::string& array_path, val data, const std::string& dtype, const std::string& region_json) const {
    return guarded([&] {
      const std::string at = array_key(store_path, array_path);
      const std::size_t n = data["length"].as<std::size_t>();
      std::vector<std::uint8_t> buf(n);
      val(emscripten::typed_memory_view(n, buf.data())).call<void>("set", data);
      const ArrayReader r = ArrayReader::open(store_, at);   // its shape and type
      const DType dt = dtype_from_string(dtype);
      if (dt != r.dtype()) invalid("put: " + dtype + " data for a " + dtype_to_string(r.dtype()) + " array");
      const Json region = Json::parse(region_json);
      const auto& shape = r.shape();
      if (!region.is_array() || region.size() != shape.size()) invalid("put: the region has " + std::to_string(region.size()) + " axes, the array " + std::to_string(shape.size()));
      Shape start, slab;
      for (std::size_t a = 0; a < shape.size(); ++a) {
        const auto s0 = region[a][0].get<std::int64_t>(), s1 = region[a][1].get<std::int64_t>();
        if (s0 < 0 || s1 < s0 || s1 > shape[a]) invalid("put: region [" + std::to_string(s0) + ", " + std::to_string(s1) + ") exceeds axis " + std::to_string(a) + " of " + std::to_string(shape[a]));
        start.push_back(s0);
        slab.push_back(s1 - s0);
      }
      const ArrayView v{buf.data(), dt, slab};
      if (v.nbytes() != n) invalid("put: " + std::to_string(n) + " bytes for a region of " + std::to_string(v.nbytes()));
      write_region(store_, at, start, v);   // any region: the chunks it touches read, patched and written again (io's write_region)
      return val(true);
    });
  }

  val updateSubject(const std::string& id, const std::string& patch_json, const std::string& by) const {
    return guarded([&] {
      const Json patch = Json::parse(patch_json.empty() ? "{}" : patch_json);
      for (const std::string& d : datasets()) {
        DatasetWriter& ds = writer(d);
        const auto row = ds.db().one("SELECT * FROM subject WHERE id = ?", {Value(id)});
        if (!row) continue;
        Json next = row_json(*row);
        for (const char* k : {"name", "kind"}) if (patch.contains(k) && !patch[k].is_null()) next[k] = patch[k];
        for (const char* k : {"template_id", "source_format", "source_path"}) if (patch.contains(k)) next[k] = patch[k];
        next["modified_utc"] = now_utc(false);
        if (!by.empty()) next["modified_by"] = by;
        auto v = [&](const char* k) { const Json& x = next[k]; return x.is_null() ? Value(std::monostate{}) : Value(x.get<std::string>()); };
        ds.db().tx([&] {
          ds.db().execute("UPDATE subject SET name = ?, kind = ?, template_id = ?, source_format = ?, source_path = ?, modified_utc = ?, modified_by = ? WHERE id = ?",
                          {v("name"), v("kind"), v("template_id"), v("source_format"), v("source_path"), v("modified_utc"), v("modified_by"), Value(id)});
        });
        return json_val(row_json(*ds.db().one("SELECT * FROM subject WHERE id = ?", {Value(id)})));
      }
      throw Error(code::no_such_row, "no subject " + id);
    });
  }

  val removeSubject(const std::string& id) const {
    return guarded([&] {
      for (const std::string& d : datasets()) {
        DatasetWriter& ds = writer(d);
        if (!ds.db().one("SELECT 1 FROM subject WHERE id = ?", {Value(id)})) continue;
        ds.db().tx([&] {
          ds.db().execute("DELETE FROM selection WHERE subject_id = ?", {Value(id)});   // selections first, as the Studio's
          ds.db().execute("DELETE FROM subject WHERE id = ?", {Value(id)});
        });
        write_catalog(root_);
        return json_val(Json{{"id", id}});
      }
      throw Error(code::no_such_row, "no subject " + id);
    });
  }

  val createField(const std::string& row_json_text, const std::string& by, const std::string& children_json) const {
    return guarded([&] {
      Json row = Json::parse(row_json_text);
      const Json ch = children_json.empty() ? Json::object() : Json::parse(children_json);
      const std::string subject_id = str_or(row, "subject_id"), name = str_or(row, "name");
      const auto [d, ds_path] = dataset_of_subject(subject_id);
      DatasetWriter& ds = writer(d);
      const Database& db = ds.db();
      const std::string store = ds_path + "/" + subject_path(db, subject_id);
      auto read1 = [&](const char* table, const std::string& id) { return db.one(std::string("SELECT * FROM ") + table + " WHERE id = ?", {Value(id)}); };
      // THE STEP (D106): made BY an operator FROM an input
      if (given(row, "by_operator_id")) {
        const auto op = read1("operator", row["by_operator_id"].get<std::string>());
        std::vector<std::string> inputs;
        if (op) {
          if (given(row, "derived_from_id")) inputs.push_back(row["derived_from_id"].get<std::string>());
          else if (std::get<std::string>(op->at("kind")) == "registration")
            for (const auto& c : ch.value("contributions", Json::array())) if (given(c, "source_field_id")) inputs.push_back(c["source_field_id"].get<std::string>());
          if (std::get<std::string>(op->at("kind")) == "registration" && inputs.empty())
            invalid("field " + name + ": made by the registration " + std::get<std::string>(op->at("name")) + ", it names no source — a contribution's source_field_id, or derived_from_id");
          for (const auto& in : inputs) {
            const auto input = read1("field", in);
            if (!input) continue;
            const auto refused = step_refusals(db, row, *op, row_json(*input));
            if (!refused.empty()) {
              std::string m;
              for (const auto& r : refused) m += (m.empty() ? "" : "; ") + r;
              invalid(m);
            }
          }
        }
      }
      if (str_or(row, "data_type") == "virtual") {
        if (given(row, "path")) invalid("field " + name + ": a virtual field stores no bytes, so it has no path");
        if (!given(row, "derived_from_id") || !given(row, "by_operator_id")) invalid("field " + name + ": a virtual field names its input (derived_from_id) and its operator (by_operator_id)");
        if (!read1("field", row["derived_from_id"].get<std::string>())) invalid("field " + name + ": its input " + row["derived_from_id"].get<std::string>() + " has no row");
        if (!read1("operator", row["by_operator_id"].get<std::string>())) invalid("field " + name + ": its operator " + row["by_operator_id"].get<std::string>() + " has no row");
        if (!given(row, "manifold_id") || !read1("manifold", row["manifold_id"].get<std::string>())) invalid("field " + name + ": manifold " + str_or(row, "manifold_id") + " has no row — POST it first");
        row["layout"] = "dense"; row["nnz"] = nullptr; row["data_type"] = "virtual";
        row["chunk_shape_json"] = nullptr; row["codecs_json"] = nullptr; row["fill_value_json"] = nullptr;
        return json_val(create_field(ds, row, by, ch));
      }
      if (!given(row, "path")) invalid("field " + name + ": a stored field needs a path — POST its array through nxr:// first");
      for (const auto& x : ch.value("selections", Json::array())) if (given(x, "array_path")) stored_array(store_, store + "/" + x["array_path"].get<std::string>());
      const Stored meta = stored_array(store_, store + "/" + row["path"].get<std::string>());
      if (!given(row, "manifold_id") || !read1("manifold", row["manifold_id"].get<std::string>())) invalid("field " + name + ": manifold " + str_or(row, "manifold_id") + " has no row — POST it first");
      const std::string mid = row["manifold_id"].get<std::string>();
      const auto on = read1("manifold", mid);
      std::int64_t elements = 1;
      for (auto x : meta.shape) elements *= x;
      const std::string cell = given(row, "cell") ? row["cell"].get<std::string>() : "0";
      const auto cells = cell_count_of(db, mid, cell);
      const std::string on_name = std::get<std::string>(on->at("name"));
      if (!cells) invalid("field " + name + ": '" + on_name + "' does not know how many '" + cell + "' cells it has — count them on the manifold first");
      const std::int64_t comps = std::max<std::int64_t>(1, row.value("n_components", static_cast<std::int64_t>(1)));
      if (elements != *cells * comps) {
        std::string shape;
        for (auto x : meta.shape) shape += (shape.empty() ? "" : "×") + std::to_string(x);
        invalid("field " + name + ": the stored array holds " + std::to_string(elements) + " values (" + shape + ") but '" + on_name + "' cells '" + cell + "' × " +
                std::to_string(row.value("n_components", static_cast<std::int64_t>(1))) + " component(s) needs " + std::to_string(*cells * comps));
      }
      const std::string layout = given(row, "layout") ? row["layout"].get<std::string>() : "dense";
      row["layout"] = layout;
      row["nnz"] = layout == "csc" && row.contains("nnz") ? row["nnz"] : Json(nullptr);
      row["data_type"] = meta.data_type;
      row["chunk_shape_json"] = meta.chunk_shape_json;
      row["codecs_json"] = meta.codecs_json;
      row["fill_value_json"] = meta.fill_value_json;
      return json_val(create_field(ds, row, by, ch));
    });
  }

  val selection(const std::string& op, const std::string& row_text, const std::string& children_json, const std::string& by) const {
    return guarded([&] {
      if (op == "delete") {
        const std::string id = row_text;
        DatasetWriter& ds = dataset_of_row("selection", id);
        Database& db = ds.db();
        delete_row(ds, "selection", id);
        drain(db, store_);
        return json_val(Json{{"id", id}});
      }
      Json row = Json::parse(row_text);
      const Json ch = children_json.empty() ? Json(nullptr) : Json::parse(children_json);
      const std::string t = now_utc(true);
      DatasetWriter& ds = op == "create" ? writer(dataset_of_subject(str_or(row, "subject_id")).first) : dataset_of_row("selection", str_or(row, "id"));
      Database& db = ds.db();
      if (op == "create") {
        if (!given(row, "id")) row["id"] = uuid7();
        row["created_utc"] = t;
        row["created_by"] = !by.empty() ? Json(by) : row.value("created_by", Json(nullptr));
        row["modified_utc"] = t;
        row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
        if (given(row, "manifold_id")) check_cell(db, row["manifold_id"].get<std::string>(), given(row, "cell") ? row["cell"].get<std::string>() : "0", "selection " + str_or(row, "name"));
      } else if (op == "update") {
        if (!db.one("SELECT 1 FROM selection WHERE id = ?", {Value(str_or(row, "id"))})) throw Error(code::no_such_row, "selection " + str_or(row, "id") + ": not found — create it");
        row["modified_utc"] = t;
        row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
      } else {
        invalid("selection: op is create, update or delete");
      }
      // ON A SPACE (D95): its position is written in one of that space's systems
      if (given(row, "space_id") && given(row, "coordinate_system_id")) {
        const auto cs = db.one("SELECT space_id FROM coordinate_system WHERE id = ?", {Value(row["coordinate_system_id"].get<std::string>())});
        if (cs && std::get<std::string>(cs->at("space_id")) != row["space_id"].get<std::string>()) invalid("selection " + str_or(row, "name") + ": its coordinate system is not of its space");
      }
      const std::string id = row["id"];
      db.tx([&] {
        row["cell"] = selection_cell(db, row);
        upsert(db, "selection", kSelectionCols, row);
        if (ch.is_object()) {
          if (ch.contains("extents")) {
            db.execute("DELETE FROM selection_extent WHERE selection_id = ?", {Value(id)});
            for (const auto& e : ch["extents"]) db.execute("INSERT INTO selection_extent (selection_id, axis, start, stop) VALUES (?, ?, ?, ?)", {Value(id), value_of(e["axis"]), value_of(e["start"]), value_of(e["stop"])});
          }
          if (ch.contains("spans")) {
            db.execute("DELETE FROM selection_span WHERE selection_id = ?", {Value(id)});
            std::int64_t k = 0;
            for (const auto& sp : ch["spans"])
              db.execute("INSERT INTO selection_span (selection_id, ordinal, start, stop, code) VALUES (?, ?, ?, ?, ?)", {Value(id), Value(k++), value_of(sp["start"]), value_of(sp["stop"]), value_of(sp.value("code", Json(nullptr)))});
            db.execute("UPDATE selection SET n_members = ? WHERE id = ?", {Value(static_cast<std::int64_t>(ch["spans"].size())), Value(id)});
          }
          if (ch.contains("measurements")) {
            db.execute("DELETE FROM selection_measurement WHERE selection_id = ? AND code IS NULL", {Value(id)});
            for (const auto& m : ch["measurements"]) {
              auto o = [&](const char* k) { return value_of(m.value(k, Json(nullptr))); };
              db.execute("INSERT INTO selection_measurement (selection_id, code, of_field_id, measure, value, weight, within_selection_id, within_code, unit, computed_utc, computed_by, array_path, component) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         {Value(id), o("code"), o("of_field_id"), o("measure"), o("value"), o("weight"), o("within_selection_id"), o("within_code"), o("unit"), o("computed_utc"), o("computed_by"), o("array_path"), o("component")});
            }
          }
        }
      });
      drain(db, store_);
      return json_val(row_json(*db.one("SELECT * FROM selection WHERE id = ?", {Value(id)})));
    });
  }

  val measurements(const std::string& op, const std::string& j, const std::string& by) const {
    return guarded([&] {
      const Json x = Json::parse(j);
      const std::optional<std::string> who = by.empty() ? std::nullopt : std::optional<std::string>(by);
      if (op == "post") {
        if (!x.is_array() || x.empty()) return json_val(Json(0));
        DatasetWriter& ds = dataset_of_row("selection", x[0].value("selection_id", std::string()));
        const auto n = write_measurements(ds.db(), x, who);
        drain(ds.db(), store_);   // always: the insert trigger queues the selection's upsert
        return json_val(Json(n));
      }
      if (op == "kinds") return json_val(measurement_kinds(dataset_of_row("selection", x.get<std::string>(), false).db(), x.get<std::string>()));
      if (op == "query") return json_val(query_measurements(dataset_of_row("selection", x.value("selectionId", std::string()), false).db(), x));
      if (op == "delete") {
        DatasetWriter& ds = dataset_of_row("selection", x.value("selectionId", std::string()));
        std::int64_t n = 0;
        const std::string sid = std::get<std::string>(ds.db().one("SELECT subject_id FROM selection WHERE id = ?", {Value(x.value("selectionId", std::string()))})->at("subject_id"));
        deleting(ds, sid, [&] { n = delete_measurements(ds.db(), x); });
        drain(ds.db(), store_);
        return json_val(Json(n));
      }
      invalid("measurements: op is post, kinds, query or delete");
    });
  }

  val operatorOp(const std::string& op, const std::string& row_text, const std::string& children_json, const std::string& by) const {
    return guarded([&] {
      if (op == "delete") {
        DatasetWriter& ds = dataset_of_row("operator", row_text);
        delete_row(ds, "operator", row_text);
        drain(ds.db(), store_);
        return json_val(Json{{"id", row_text}});
      }
      Json row = Json::parse(row_text);
      const Json ch = children_json.empty() ? Json(nullptr) : Json::parse(children_json);
      const std::string name = str_or(row, "name"), t = now_utc(true);
      if (op == "create") {
        const auto [d, ds_path] = dataset_of_subject(str_or(row, "subject_id"));
        DatasetWriter& ds = writer(d);
        const std::string layout = str_or(row, "layout");
        if (layout == "analytic") {
          if (given(row, "path")) invalid("operator " + name + ": an analytic operator stores no bytes, so it has no path");
          row["data_type"] = nullptr; row["chunk_shape_json"] = nullptr; row["codecs_json"] = nullptr; row["fill_value_json"] = nullptr;
        } else {
          if (!given(row, "path")) invalid("operator " + name + ": a stored operator needs a path — POST its array(s) through nxr:// first");
          const std::string location = ds_path + "/" + subject_path(ds.db(), str_or(row, "subject_id")) + "/" + row["path"].get<std::string>();
          const Stored meta = stored_array(store_, layout == "dense" ? location : location + "/data");
          if (layout == "dense") {
            const auto nr = row.value("n_rows", Json(nullptr)), nc = row.value("n_cols", Json(nullptr));
            if (meta.shape.size() != 2 || !nr.is_number() || !nc.is_number() || meta.shape[0] != nr.get<std::int64_t>() || meta.shape[1] != nc.get<std::int64_t>()) {
              std::string shape;
              for (auto x : meta.shape) shape += (shape.empty() ? "" : "×") + std::to_string(x);
              invalid("operator " + name + ": the stored array is " + shape + " but the row says " + nr.dump() + "×" + nc.dump());
            }
          } else {
            for (const char* part : {"indices", "indptr"})
              if (!store_.exists(join_key(join_key(location, part), "zarr.json"))) invalid("operator " + name + ": a " + layout + " operator needs " + location + "/" + part);
          }
          row["data_type"] = meta.data_type; row["chunk_shape_json"] = meta.chunk_shape_json; row["codecs_json"] = meta.codecs_json; row["fill_value_json"] = meta.fill_value_json;
        }
        if (!given(row, "id")) row["id"] = uuid7();
        row["created_utc"] = t; row["created_by"] = !by.empty() ? Json(by) : row.value("created_by", Json(nullptr));
        row["modified_utc"] = t; row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
        check_cell(ds.db(), str_or(row, "from_manifold_id"), given(row, "from_cell") ? row["from_cell"].get<std::string>() : "0", "operator " + name + " (from)");
        if (given(row, "to_manifold_id")) check_cell(ds.db(), row["to_manifold_id"].get<std::string>(), given(row, "to_cell") ? row["to_cell"].get<std::string>() : "0", "operator " + name + " (to)");
        return json_val(write_operator(ds, row, ch));
      }
      if (op == "update") {
        DatasetWriter& ds = dataset_of_row("operator", str_or(row, "id"));
        row["modified_utc"] = t; row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
        return json_val(write_operator(ds, row, ch));
      }
      invalid("operator: op is create, update or delete");
    });
  }

  val fieldUpdate(const std::string& row_text, const std::string& by) const {
    return guarded([&] {
      Json row = Json::parse(row_text);
      const std::string id = str_or(row, "id");
      DatasetWriter& ds = dataset_of_row("field", id);
      row["modified_utc"] = now_utc(true);
      row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
      if (!given(row, "cell")) row["cell"] = "0";
      ds.db().tx([&] { upsert(ds.db(), "field", kFieldCols, row); });
      drain(ds.db(), store_);
      return json_val(row_json(*ds.db().one("SELECT * FROM field WHERE id = ?", {Value(id)})));
    });
  }
  val fieldDelete(const std::string& id) const {
    return guarded([&] {
      DatasetWriter& ds = dataset_of_row("field", id);
      delete_row(ds, "field", id);
      drain(ds.db(), store_);
      return json_val(Json{{"id", id}});
    });
  }

  val manifold(const std::string& op, const std::string& row_text, const std::string& children_json, const std::string& by) const {
    return guarded([&] {
      if (op == "delete") {
        DatasetWriter& ds = dataset_of_row("manifold", row_text);
        const auto m = ds.db().one("SELECT subject_id, geometry_id FROM manifold WHERE id = ?", {Value(row_text)});
        const std::string sid = std::get<std::string>(m->at("subject_id"));
        deleting(ds, sid, [&] {
          ds.db().tx([&] { nxr::io::delete_row(ds.db(), "manifold", row_text); });   // its geometry and topology with it, when no other manifold uses them
        });
        drain(ds.db(), store_);
        return json_val(Json{{"id", row_text}});
      }
      Json row = Json::parse(row_text);
      const std::string t = now_utc(true);
      if (op == "update") {
        DatasetWriter& ds = dataset_of_row("manifold", str_or(row, "id"));
        row["modified_utc"] = t;
        row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
        ds.db().tx([&] { upsert(ds.db(), "manifold", kManifoldCols, row); });
        drain(ds.db(), store_);
        return json_val(row_json(*ds.db().one("SELECT * FROM manifold WHERE id = ?", {Value(row["id"].get<std::string>())})));
      }
      if (op != "create") invalid("manifold: op is create, update or delete");
      Json ch = children_json.empty() ? Json(nullptr) : Json::parse(children_json);
      const std::string subject_id = str_or(row, "subject_id"), name = str_or(row, "name");
      DatasetWriter& ds = writer(dataset_of_subject(subject_id).first);
      Database& db = ds.db();
      if (str_or(row, "type") == "product") {
        const Json ids = ch.is_object() && ch.contains("factors") ? ch["factors"] : Json::array();
        if (ids.size() < 2) invalid("manifold " + name + ": a product needs at least two factors (manifold ids)");
        std::vector<Json> fm, ft;
        for (const auto& id : ids) {
          const auto m = db.one("SELECT * FROM manifold WHERE id = ?", {value_of(id)});
          if (!m) invalid("manifold " + name + ": factor " + (id.is_string() ? id.get<std::string>() : id.dump()) + " has no row");
          fm.push_back(row_json(*m));
          ft.push_back(row_json(*db.one("SELECT * FROM topology WHERE id = ?", {Value(std::get<std::string>(m->at("topology_id")))})));
        }
        Json built = product_from(subject_id, fm, ft, row);
        built["row"]["created_by"] = by.empty() ? Json(nullptr) : Json(by);
        built["row"]["modified_by"] = by.empty() ? Json(nullptr) : Json(by);
        return json_val(create_manifold(ds, built["row"], Json{{"topology", built["topology"]}, {"geometry", built["geometry"]}}, by));
      }
      if (ch.is_object()) {
        ch.erase("factors");
        if (ch.contains("geometry") && ch["geometry"].is_object() && ch["geometry"].contains("coordinate_system_key")) {
          const Json key = ch["geometry"]["coordinate_system_key"];
          ch["geometry"].erase("coordinate_system_key");
          if (key.is_string() && !key.get<std::string>().empty()) ch["geometry"]["coordinate_system_id"] = coordinate_system(db, subject_id, key.get<std::string>());
        }
      }
      return json_val(create_manifold(ds, row, ch, by));
    });
  }

  val dictionary(const std::string& op, const std::string& row_text, const std::string& children_json, const std::string& by) const {
    return guarded([&] {
      if (op == "delete") {
        DatasetWriter& ds = dataset_of_row("dictionary", row_text);
        ds.db().tx([&] { ds.db().execute("DELETE FROM dictionary WHERE id = ?", {Value(row_text)}); });
        return json_val(Json{{"id", row_text}});
      }
      Json row = Json::parse(row_text);
      const Json ch = children_json.empty() ? Json(nullptr) : Json::parse(children_json);
      const std::string t = now_utc(true);
      if (op == "create") {
        if (!given(row, "id")) row["id"] = uuid7();
        row["created_utc"] = t; row["created_by"] = !by.empty() ? Json(by) : row.value("created_by", Json(nullptr));
        row["modified_utc"] = t; row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
      } else if (op == "update") {
        row["modified_utc"] = t; row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
      } else {
        invalid("dictionary: op is create, update or delete");
      }
      const std::string id = row["id"];
      Json out;
      for (DatasetWriter* ds : route_rows("dictionary", row)) {
        Database& db = ds->db();
        db.tx([&] {
          upsert(db, "dictionary", {"id", "name", "scope", "dataset_id", "subject_id", "description", "created_utc", "created_by", "modified_utc", "modified_by"}, row);
          Json entries;
          if (ch.is_object() && ch.contains("entries")) entries = ch["entries"];
          else if (op == "update") { entries = Json::array(); for (const Row& e : db.all("SELECT * FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", {Value(id)})) entries.push_back(row_json(e)); }
          else entries = Json::array();
          db.execute("DELETE FROM dictionary_entry WHERE dictionary_id = ?", {Value(id)});
          for (const auto& e : entries)
            db.execute("INSERT INTO dictionary_entry (dictionary_id, code, name, color_json, measurement, measurement_unit, function, params_json, attributes_json, ordinal) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       {Value(id), value_of(e.value("code", Json(nullptr))), value_of(e.value("name", Json(nullptr))), value_of(e.value("color_json", Json(nullptr))),
                        value_of(e.value("measurement", Json(nullptr))), value_of(e.value("measurement_unit", Json(nullptr))), value_of(e.value("function", Json(nullptr))),
                        value_of(e.value("params_json", Json(nullptr))), value_of(e.value("attributes_json", Json(nullptr))), value_of(e.value("ordinal", Json(nullptr)))});
          // the selections reading its codes carry it in their store copy: touched, so the drain rewrites them
          if (op == "update") db.execute("UPDATE selection SET modified_utc = ? WHERE dictionary_id = ?", {Value(t), Value(id)});
        });
        drain(db, store_);
        out = row_json(*db.one("SELECT * FROM dictionary WHERE id = ?", {Value(id)}));
      }
      return json_val(out);
    });
  }

  val job(const std::string& op, const std::string& j, const std::string& by) const {
    return guarded([&] {
      const Json x = j.empty() ? Json(nullptr) : Json::parse(j);
      const std::string t = now_utc(true);
      const std::vector<std::string> cols = {"id", "kind", "host", "status", "dataset_id", "subject_id", "inputs_json", "params_json", "result_json", "progress", "error", "requested_by",
                                             "created_utc", "started_utc", "finished_utc"};
      if (op == "get") {
        if (x.is_string()) {
          for (const std::string& d : datasets())
            if (const auto r = reader(d).db().one("SELECT * FROM job WHERE id = ?", {Value(x.get<std::string>())})) return json_val(row_json(*r));
          return json_val(Json(nullptr));
        }
        const Json w = x.is_object() ? x : Json::object();
        std::string sql = "SELECT * FROM job";
        std::vector<Value> args;
        std::string sep = " WHERE ";
        for (const auto& [k, c] : std::vector<std::pair<const char*, const char*>>{{"subjectId", "subject_id"}, {"datasetId", "dataset_id"}, {"status", "status"}, {"kind", "kind"}})
          if (w.contains(k) && w[k].is_string() && !w[k].get<std::string>().empty()) { sql += sep + c + " = ?"; args.push_back(Value(w[k].get<std::string>())); sep = " AND "; }
        sql += " ORDER BY created_utc DESC";
        Json out = Json::array();
        for (DatasetWriter* ds : route_rows("job", w, false)) for (const Row& r : ds->db().all(sql, args)) out.push_back(row_json(r));
        return json_val(out);
      }
      if (op == "create") {
        Json row = {{"progress", nullptr}, {"error", nullptr}, {"result_json", nullptr}, {"started_utc", nullptr}, {"finished_utc", nullptr}};
        for (const auto& [k, v] : x.items()) row[k] = v;
        if (!given(row, "id")) row["id"] = uuid7();
        if (!given(row, "status")) row["status"] = "queued";
        row["created_utc"] = t;
        row["requested_by"] = !by.empty() ? Json(by) : row.value("requested_by", Json(nullptr));
        std::vector<DatasetWriter*> where = route_rows("job", row);
        for (DatasetWriter* ds : where) ds->db().tx([&] { upsert(ds->db(), "job", cols, row); });
        return json_val(row_json(*where.back()->db().one("SELECT * FROM job WHERE id = ?", {Value(row["id"].get<std::string>())})));
      }
      if (op == "update") {
        const std::string id = str_or(x, "id");
        DatasetWriter& ds = dataset_of_row("job", id);
        Json row = row_json(*ds.db().one("SELECT * FROM job WHERE id = ?", {Value(id)}));
        for (const auto& [k, v] : x.items()) row[k] = v;
        const std::string st = x.value("status", std::string());
        if (st == "running" && !given(row, "started_utc")) row["started_utc"] = t;
        if ((st == "done" || st == "failed" || st == "cancelled") && !given(row, "finished_utc")) row["finished_utc"] = t;
        ds.db().tx([&] { upsert(ds.db(), "job", cols, row); });
        return json_val(row_json(*ds.db().one("SELECT * FROM job WHERE id = ?", {Value(id)})));
      }
      invalid("job: op is get, create or update");
    });
  }

  // folder_for with the Studio's refusals (db/layout.ts folderFor: JSON-quoted names) — io's own word them as nxr's Python does
  static std::string studio_folder_for(const std::string& type, const std::string& name, const std::optional<std::string>& role, const std::optional<std::string>& op) {
    if (type == "line" && !role) invalid("line " + Json(name).dump() + ": give its role (time \xc2\xb7 spectral \xc2\xb7 eigen)");
    if (type != "volume" && type != "surface" && type != "line" && type != "points" && type != "curve")
      invalid("manifold " + Json(name).dump() + " of type " + Json(type).dump() + ": no folder in nxr.subject@6.0");
    return folder_for(type, name, role, op);
  }

  val layout(const std::string& store_path, const std::string& q_json) const {
    return guarded([&] {
      const Json q = Json::parse(q_json);
      const std::string name = q.value("name", std::string());
      bool ok = !name.empty();
      for (char c : name) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' || c == '-')) ok = false;
      if (!ok) invalid("layout: " + Json(name).dump() + " is not a node name (letters, digits, _ . -)");
      const fs::path rel = safe_rel(store_path, "store path");
      std::vector<std::string> sp;
      for (const auto& x : rel) sp.push_back(x.string());
      if (sp.size() != 2) invalid(store_path + ": not a subject of this datastore");
      DatasetWriter& ds = reader(sp[0]);
      const Database& db = ds.db();
      const auto sub = db.one("SELECT id FROM subject WHERE path = ?", {Value(sp[1])});
      if (!sub) invalid(store_path + ": not a subject of this datastore");
      const std::string subject = std::get<std::string>(sub->at("id"));
      auto opt = [&](const char* k) { return q.contains(k) && q[k].is_string() && !q[k].get<std::string>().empty() ? std::optional<std::string>(q[k].get<std::string>()) : std::nullopt; };
      std::string folder;
      if (const auto m = opt("manifold")) {
        const auto r = db.one("SELECT subject_id FROM manifold WHERE id = ?", {Value(*m)});
        if (!r) invalid("layout: manifold " + *m + " has no row");
        if (std::get<std::string>(r->at("subject_id")) != subject) invalid("layout: manifold " + *m + " belongs to another subject than " + store_path);
        folder = folder_of(db, *m);
      } else if (opt("type") == std::optional<std::string>("product")) {
        if (!q.contains("factors") || !q["factors"].is_array() || q["factors"].empty()) invalid("layout: a product names its factors");
        std::string joined;
        for (const auto& f : q["factors"]) {
          const std::string fid = f.get<std::string>();
          std::string part;
          if (db.one("SELECT 1 FROM manifold WHERE id = ?", {Value(fid)})) part = folder_of(db, fid);
          else {
            const auto colon = fid.find(':');
            const std::string type = fid.substr(0, colon), role = colon == std::string::npos ? std::string() : fid.substr(colon + 1);
            part = studio_folder_for(type, "", type == "line" && !role.empty() ? std::optional<std::string>(role) : std::nullopt,
                              type == "points" && role == "operator" ? std::optional<std::string>("operator") : std::nullopt);
          }
          joined += (joined.empty() ? "" : "-") + part;
        }
        folder = joined;
      } else if (const auto t = opt("type")) {
        folder = studio_folder_for(*t, opt("manifoldName").value_or(""), opt("role"), opt("operator"));
      } else {
        invalid("layout: give the manifold (its id) or the type of the one about to be written");
      }
      return json_val(Json{{"path", folder + "/" + name}, {"folder", folder}});
    });
  }

  val datasetCreate(const std::string& name, const std::string& source_json) const {
    return guarded([&] {
      if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos || name.find("..") != std::string::npos)
        invalid("invalid dataset name: " + name);
      if (fs::is_regular_file(root_ / name / kDatasetFile)) {
        DatasetWriter& ds = writer(name);
        return json_val(row_json(ds.row()));
      }
      const Json src = source_json.empty() ? Json(nullptr) : Json::parse(source_json);
      CreateOptions co;
      co.built_by = program_;   // the program, in the database's meta; the dataset row's creator stays empty, as the Studio's ensureDataset leaves it
      if (src.is_object()) {
        if (src.contains("tool") && src["tool"].is_string()) co.source_tool = src["tool"].get<std::string>();
        if (src.contains("protocol") && src["protocol"].is_string()) co.source_protocol = src["protocol"].get<std::string>();
        if (src.contains("path") && src["path"].is_string()) co.source_path = src["path"].get<std::string>();
      }
      readers_.erase(name);
      auto made = std::make_unique<DatasetWriter>(create_dataset(root_, name, co));
      const Json row = row_json(made->row());
      held_[name] = std::move(made);
      return json_val(row);
    });
  }

  val datasetDelete(const std::string& name) const {
    return guarded([&] {
      if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos || name.find("..") != std::string::npos)
        invalid("invalid dataset name: " + Json(name).dump() + " \xe2\x80\x94 refused");
      const fs::path target = root_ / name;
      std::error_code ec;
      if (!fs::exists(fs::symlink_status(target, ec))) return json_val(Json{{"deleted", nullptr}});
      if (fs::is_symlink(fs::symlink_status(target, ec)) || fs::weakly_canonical(target) != fs::weakly_canonical(root_) / name)
        invalid(name + ": the dataset folder is (or is reached through) a symbolic link \xe2\x80\x94 refused");
      held_.erase(name);      // our own hold (its lock) first: the delete takes the lock itself
      readers_.erase(name);
      delete_dataset(root_, name);
      return json_val(Json{{"deleted", name}});
    });
  }

  val catalog() const {
    return guarded([&] { return json_val(write_catalog(root_)); });
  }

  // `nxr-neuro <argv>` in process: stdout and stderr captured; every dataset this client holds released first (the command takes its own lock)
  Json run_cli(const std::vector<std::string>& argv) const {
    held_.clear();
    readers_.clear();
    std::vector<std::string> args = {"nxr-neuro"};
    args.insert(args.end(), argv.begin(), argv.end());
    std::vector<char*> cargv;
    for (auto& a : args) cargv.push_back(a.data());
    cargv.push_back(nullptr);
    std::ostringstream out, err;
    auto* o = std::cout.rdbuf(out.rdbuf());
    auto* e = std::cerr.rdbuf(err.rdbuf());
    int code = 0;
    try { code = nxr_neuro_cli_main(static_cast<int>(args.size()), cargv.data()); } catch (...) { std::cout.rdbuf(o); std::cerr.rdbuf(e); throw; }
    std::cout.rdbuf(o);
    std::cerr.rdbuf(e);
    Json lines = Json::array();
    std::istringstream in(out.str());
    for (std::string l; std::getline(in, l);) {
      if (l.empty()) continue;
      Json j = Json::parse(l, nullptr, false);
      lines.push_back(j.is_discarded() ? Json(l) : j);
    }
    return Json{{"code", code}, {"lines", lines}, {"stderr", err.str()}};
  }
  val cli(const std::string& argv_json) const {
    return guarded([&] {
      std::vector<std::string> argv;
      for (const auto& a : Json::parse(argv_json)) argv.push_back(a.get<std::string>());
      return json_val(run_cli(argv));
    });
  }

  val importSubjects(const std::string& req_json) const {
    return guarded([&] {
      const Json req = Json::parse(req_json);
      const std::string dname = req.value("dataset", std::string());
      if (!fs::is_regular_file(root_ / dname / kDatasetFile)) invalid(dname + ": not a dataset of this datastore \xe2\x80\x94 create it first");
      const Json src = req.value("source", Json::object());
      const std::string kind = src.value("kind", std::string());
      if (kind == "store") throw Error(code::unsupported, "imports: a store source (copying a subject store in) is not in nxr-neuro's client yet");
      if (kind != "brainstorm" && kind != "brainstorm-template" && kind != "brainstorm-group") invalid("imports: source.kind is brainstorm, brainstorm-template or brainstorm-group");
      const bool replace = req.value("replace", false), all_sessions = req.value("replaceAllSessions", false);
      const std::string stem = kind == "brainstorm" ? src.value("subject", std::string()) : kind == "brainstorm-template" ? "template" : "group";
      const std::string folder = stem + ".nxr.zarr";
      if (replace && kind == "brainstorm" && !all_sessions)
        invalid(stem + ": replace removes the WHOLE subject (every session) before converting one condition, and is not atomic \xe2\x80\x94 pass replaceAllSessions: true to mean that");
      Json dataset = row_json(reader(dname).row());
      const std::string dataset_id = dataset["id"], dataset_name = dataset["name"];
      const std::string dataset_dir = (root_ / dataset["path"].get<std::string>()).string();
      auto db = [&]() -> const Database& { return reader(dname).db(); };
      auto complete_subject = [&]() -> std::optional<Row> {
        return db().one("SELECT * FROM subject WHERE dataset_id = ? AND path = ? AND status = 'complete'", {Value(dataset_id), Value(folder)});
      };
      auto counts = [&](const std::optional<std::string>& sid) {
        Json c = {{"spaces", 0}, {"topologies", 0}, {"manifolds", 0}, {"dictionaries", 0}, {"selections", 0}, {"elements", 0}, {"fields", 0}, {"operators", 0}, {"skipped", 0}};
        if (!sid) return c;
        auto n = [&](const std::string& sql) { return std::get<std::int64_t>(db().one(sql, {Value(*sid)})->at("n")); };
        c["spaces"] = n("SELECT count(*) AS n FROM space WHERE subject_id = ?");
        c["topologies"] = n("SELECT count(*) AS n FROM topology WHERE subject_id = ?");
        c["manifolds"] = n("SELECT count(*) AS n FROM manifold WHERE subject_id = ?");
        c["dictionaries"] = n("SELECT count(*) AS n FROM dictionary WHERE subject_id = ?");
        c["selections"] = n("SELECT count(*) AS n FROM selection WHERE subject_id = ?");
        c["elements"] = n("SELECT coalesce(sum(n_elements), 0) AS n FROM selection WHERE subject_id = ? AND array_path IS NOT NULL AND type IN ('set', 'indices')");
        c["fields"] = n("SELECT count(*) AS n FROM field WHERE subject_id = ?");
        c["operators"] = n("SELECT count(*) AS n FROM operator WHERE subject_id = ?");
        return c;
      };
      auto run = [&](const std::vector<std::string>& argv) {
        const Json r = run_cli(argv);
        if (r["code"].get<int>() != 0) {
          std::string m;
          for (const auto& l : r["lines"]) if (l.is_object() && l.value("stage", std::string()) == "error") m = l.value("message", std::string());
          if (m.empty()) m = r["stderr"].get<std::string>();
          throw Error(code::invalid_argument, "nxr-neuro " + argv[0] + " " + (argv.size() > 1 ? argv[1] : "") + ": " + m);
        }
        return r;
      };
      const auto any_row = db().one("SELECT id FROM subject WHERE dataset_id = ? AND (path = ? OR name = ?) AND status = 'complete'", {Value(dataset_id), Value(folder), Value(stem)});
      std::vector<std::string> riding;
      if (any_row && replace) {
        const std::string old = std::get<std::string>(any_row->at("id"));
        for (const Row& r : db().all("SELECT id FROM subject WHERE template_id = ?", {Value(old)})) riding.push_back(std::get<std::string>(r.at("id")));
        run({"delete", "subject", "--dataset", dataset_dir, "--subject", stem});
        if (db().one("SELECT 1 FROM subject WHERE id = ?", {Value(old)})) invalid(stem + ": still a subject of " + dataset_name + " after the converter's remove");
      } else if (any_row && kind != "brainstorm") {
        invalid(folder + ": already a subject of " + dataset_name + " \xe2\x80\x94 pass replace to overwrite");
      }
      std::optional<std::string> before_id;
      if (!replace) if (const auto c = complete_subject()) before_id = std::get<std::string>(c->at("id"));
      const Json before = counts(before_id);
      std::vector<std::string> argv;
      const std::string root = src.value("root", std::string());
      if (kind == "brainstorm") {
        argv = {"import", "condition", root, "--subject", stem, "--condition", src.value("condition", std::string()), "--dataset", dataset_dir};
        for (const auto& a : src.value("args", Json::array())) argv.push_back(a.get<std::string>());
      } else if (kind == "brainstorm-template") {
        argv = {"import", "template", root, "--dataset", dataset_dir};
        for (const auto& sf : src.value("surfaces", Json::array())) { argv.push_back("--surface"); argv.push_back(sf.get<std::string>()); }
      } else {
        argv = {"import", "group", root, "--dataset", dataset_dir};
      }
      try { run(argv); }
      catch (const Error& e) {
        if (any_row && replace) throw Error(e.code(), std::string(e.what()) + " \xe2\x80\x94 the old " + stem + " was REMOVED before this conversion (replace is not atomic); convert it again");
        throw;
      }
      const auto subject = complete_subject();
      if (!subject) invalid(folder + ": the converter exited 0 but wrote no complete subject row in " + dataset_name);
      const std::string sid = std::get<std::string>(subject->at("id"));
      DatasetWriter& ds = writer(dname);
      if (std::get<std::string>(subject->at("kind")) == "template" && !riding.empty())
        ds.db().tx([&] { for (const auto& r : riding) ds.db().execute("UPDATE subject SET template_id = ? WHERE id = ?", {Value(sid), Value(r)}); });
      const Json after = counts(sid);
      Json added = Json::object();
      for (const auto& [k, v] : after.items()) added[k] = std::max<std::int64_t>(0, v.get<std::int64_t>() - before[k].get<std::int64_t>());
      ensure_band_dictionary(ds.db());   // the Studio's seed after an import (the band labelling; kernels and channel colours: see the spec)
      drain(ds.db(), store_);
      write_catalog(root_);
      return json_val(Json::array({Json{{"subject", row_json(*ds.db().one("SELECT * FROM subject WHERE id = ?", {Value(sid)}))}, {"counts", added}}}));
    });
  }

  // an environment variable of the module (its own environment: the host's is not passed in), e.g. NXR_IO_DETERMINISTIC for the parity tests, read by the command line
  void setEnv(const std::string& name, const std::string& value) const { ::setenv(name.c_str(), value.c_str(), 1); }

  // let go of ONE dataset (its writer lock, its reader): before a caller changes its folder outside the rows (the Studio's claimDataset)
  void releaseDataset(const std::string& name) const { held_.erase(name); readers_.erase(name); }

  void close() { held_.clear(); readers_.clear(); }

  val remove(const std::string& store_path, const std::string& array_path) const {
    return guarded([&] {
      // the Studio's removeArray: a node of a subject's store (<dataset>/<subject>.nxr.zarr/<folder>/…); refused while a row names it (D164: a row's arrays
      // go with the row); then removed through the store-node checks (no symbolic link, inside the store)
      const fs::path rel = safe_rel(store_path, "store path");
      std::vector<std::string> sp;
      for (const auto& x : rel) sp.push_back(x.string());
      if (sp.size() != 2 || sp[0].empty() || sp[0][0] == '.' || sp[1].size() <= 9 || sp[1].substr(sp[1].size() - 9) != ".nxr.zarr")
        invalid(store_path + "/" + array_path + ": not a node inside a subject's store (<dataset>/<subject>.nxr.zarr/<folder>/…) — refused");
      safe_rel(array_path, "array path");
      DatasetWriter& ds = writer(sp[0]);
      const auto sub = ds.db().one("SELECT id FROM subject WHERE path = ?", {Value(sp[1])});
      if (!sub) throw Error(code::no_such_row, store_path + ": not a subject of this datastore");
      if (const auto by = named_by(ds.db(), std::get<std::string>(sub->at("id")), array_path))
        throw Error(code::already_exists, store_path + "/" + array_path + ": named by " + *by + " — a row's arrays are deleted with the row, never here");
      PrefixStore st(store_, rel.generic_string());
      return val(remove_store_node(st, array_path));
    });
  }

 private:
  fs::path root_;
  mutable DatastoreStore store_;   // the datastore's keys: `<dataset>/<subject>/<array>/…` (a format 52 subject's small keys are rows of its small_files.sqlite)
  std::string program_;
  mutable std::map<std::string, std::unique_ptr<DatasetWriter>> held_;   // the datasets held for writing

  DatasetWriter& writer(const std::string& d) const {
    auto it = held_.find(d);
    if (it == held_.end()) it = held_.emplace(d, std::make_unique<DatasetWriter>(open(d, true))).first;
    return *it->second;
  }

  // the dataset (folder) a subject id belongs to
  std::pair<std::string, std::string> dataset_of_subject(const std::string& subject_id) const {
    for (const std::string& d : datasets()) {
      DatasetWriter& ds = reader(d);
      if (const auto r = ds.db().one("SELECT ds.path AS p FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = ?", {Value(subject_id)}))
        return {d, std::get<std::string>(r->at("p"))};
    }
    throw Error(code::no_such_row, "no subject " + subject_id);
  }
  // the Studio's checkOperatorLinks + the upsert with its factors + the drain
  Json write_operator(DatasetWriter& ds, Json row, const Json& ch) const {
    Database& db = ds.db();
    const std::string name = str_or(row, "name");
    auto read1 = [&](const char* table, const std::string& id) { return db.one(std::string("SELECT * FROM ") + table + " WHERE id = ?", {Value(id)}); };
    auto S = [](const Json& j, const char* k) { return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : std::string(); };
    auto C = [&](const Json& j, const char* k) { const std::string c = S(j, k); return c.empty() ? std::string("0") : c; };
    if (given(row, "inverts_id")) {
      if (!given(row, "inverse_kind")) invalid("operator " + name + ": inverts " + row["inverts_id"].get<std::string>() + " but says not how (inverse_kind)");
      const auto f = read1("operator", row["inverts_id"].get<std::string>());
      std::vector<std::string> p;
      if (!f) p.push_back(name + ": inverts an operator that has no row");
      else {
        const Json fw = row_json(*f);
        const std::string fname = S(fw, "name");
        if (S(row, "from_manifold_id") != S(fw, "to_manifold_id") || S(row, "to_manifold_id") != S(fw, "from_manifold_id")) p.push_back(name + " does not invert " + fname + ": its ends are not " + fname + "'s swapped");
        if (C(row, "from_cell") != C(fw, "to_cell") || C(row, "to_cell") != C(fw, "from_cell")) p.push_back(name + " does not invert " + fname + ": the cells are not swapped");
        auto vt = [](const std::string& a, const std::string& b) { return a.empty() || b.empty() || a == b; };
        if (!vt(S(row, "from_value_type"), S(fw, "to_value_type")) || !vt(S(row, "to_value_type"), S(fw, "from_value_type"))) p.push_back(name + " does not invert " + fname + ": the value types are not swapped");
      }
      if (!p.empty()) { std::string m; for (auto& x : p) m += (m.empty() ? "" : "; ") + x; invalid(m); }
    }
    const Json factors = ch.is_object() && ch.contains("factors") ? ch["factors"] : Json(nullptr);
    if (factors.is_array() && !factors.empty()) {
      std::vector<Json> ops;
      for (const auto& f : factors) {
        if (given(f, "factor_operator_id")) {
          const auto o = read1("operator", f["factor_operator_id"].get<std::string>());
          if (!o) invalid("operator " + name + ": factor " + f["factor_operator_id"].get<std::string>() + " has no row");
          ops.push_back(row_json(*o));
        } else {
          if (!given(f, "factor_selection_id") || !read1("selection", f["factor_selection_id"].get<std::string>())) invalid("operator " + name + ": gain " + S(f, "factor_selection_id") + " has no row");
        }
      }
      std::vector<std::string> p;
      if (ops.empty()) p.push_back(name + ": a composition of gains alone maps nothing");
      else {
        if (S(ops.front(), "from_manifold_id") != S(row, "from_manifold_id")) p.push_back(name + ": its first factor " + S(ops.front(), "name") + " does not start where it does");
        if (S(ops.back(), "to_manifold_id") != S(row, "to_manifold_id")) p.push_back(name + ": its last factor " + S(ops.back(), "name") + " does not end where it does");
        for (std::size_t i = 1; i < ops.size(); ++i) {
          const Json &a = ops[i - 1], &b = ops[i];
          if (S(a, "to_manifold_id") != S(b, "from_manifold_id")) p.push_back(name + ": " + S(b, "name") + " does not read what " + S(a, "name") + " writes");
          else if (C(a, "to_cell") != C(b, "from_cell")) p.push_back(name + ": " + S(b, "name") + " reads cells " + S(b, "from_cell") + ", " + S(a, "name") + " writes " + S(a, "to_cell"));
          else if (!S(a, "to_value_type").empty() && !S(b, "from_value_type").empty() && S(a, "to_value_type") != S(b, "from_value_type"))
            p.push_back(name + ": " + S(b, "name") + " reads " + S(b, "from_value_type") + ", " + S(a, "name") + " writes " + S(a, "to_value_type"));
        }
      }
      if (!p.empty()) { std::string m; for (auto& x : p) m += (m.empty() ? "" : "; ") + x; invalid(m); }
    }
    for (const char* c : {"from_cell", "to_cell"}) if (!given(row, c)) row[c] = "0";
    const std::string id = row["id"];
    db.tx([&] {
      upsert(db, "operator", kOperatorCols, row);
      if (factors.is_array()) {
        db.execute("DELETE FROM operator_factor WHERE operator_id = ?", {Value(id)});
        std::int64_t k = 0;
        for (const auto& f : factors)
          db.execute("INSERT INTO operator_factor (operator_id, ordinal, factor_operator_id, factor_selection_id, transposed) VALUES (?, ?, ?, ?, ?)",
                     {Value(id), Value(k++), value_of(f.value("factor_operator_id", Json(nullptr))), value_of(f.value("factor_selection_id", Json(nullptr))), value_of(f.value("transposed", Json(0)))});
      }
    });
    drain(db, store_);
    return row_json(*db.one("SELECT * FROM operator WHERE id = ?", {Value(id)}));
  }

  // the Studio's productManifoldFrom (+ productCounts): the product's row, topology, geometry and dimensions from its factors
  static Json product_from(const std::string& subject_id, const std::vector<Json>& fm, const std::vector<Json>& ft, const Json& given) {
    auto num = [](const Json& j, const char* k) { return j.contains(k) && j[k].is_number() ? std::optional<std::int64_t>(j[k].get<std::int64_t>()) : std::nullopt; };
    std::vector<std::optional<std::int64_t>> poly = {1};
    std::int64_t rank = 0;
    for (std::size_t f = 0; f < fm.size(); ++f) {
      const std::int64_t r = num(ft[f], "rank").value_or(0);
      const std::vector<std::optional<std::int64_t>> g = {num(fm[f], "n_vertices"), r >= 1 ? num(fm[f], "n_edges") : std::optional<std::int64_t>(0),
                                                          r >= 2 ? num(fm[f], "n_faces") : std::optional<std::int64_t>(0), r >= 3 ? num(fm[f], "n_volumes") : std::optional<std::int64_t>(0)};
      std::vector<std::optional<std::int64_t>> out(4, std::optional<std::int64_t>(0));
      for (std::size_t i = 0; i < poly.size(); ++i)
        for (std::size_t j = 0; j < 4 && i + j < 4; ++j) {
          const auto a = poly[i], b = g[j], cur = out[i + j];
          if (!cur || !a || !b) out[i + j] = ((a && *a == 0) || (b && *b == 0)) ? cur : std::nullopt;
          else out[i + j] = *cur + *a * *b;
        }
      poly = out;
      rank += r;
    }
    auto at = [&](std::int64_t k) -> Json { if (k > rank || static_cast<std::size_t>(k) >= poly.size() || !poly[static_cast<std::size_t>(k)]) return nullptr; return *poly[static_cast<std::size_t>(k)]; };
    bool reduces = true;
    std::int64_t ncomp = 1;
    for (std::size_t f = 0; f < fm.size(); ++f) {
      const std::string kind = ft[f].value("kind", std::string());
      if (kind != "none" && kind != "lattice" && kind != "path") reduces = false;
      ncomp *= kind == "none" ? num(fm[f], "n_vertices").value_or(0) : num(ft[f], "n_components").value_or(0);
    }
    const std::string id = given.contains("id") && given["id"].is_string() ? given["id"].get<std::string>() : uuid7();
    const std::string tid = given.contains("topology_id") && given["topology_id"].is_string() && !given["topology_id"].get<std::string>().empty() ? given["topology_id"].get<std::string>() : uuid7();
    const std::string gid = given.contains("geometry_id") && given["geometry_id"].is_string() && !given["geometry_id"].get<std::string>().empty() ? given["geometry_id"].get<std::string>() : uuid7();
    Json topology = {{"id", tid}, {"kind", rank == 0 ? "none" : reduces ? "lattice" : "product"}, {"rank", rank}, {"n_components", ncomp}, {"winding", nullptr}, {"closed", nullptr}, {"orientable", nullptr}};
    Json dims = Json::array();
    std::string joined;
    for (std::size_t f = 0; f < fm.size(); ++f) {
      dims.push_back(Json{{"ordinal", f}, {"name", fm[f]["name"]}, {"n_vertices", fm[f]["n_vertices"]}, {"spacing", nullptr}, {"origin", nullptr}, {"unit", nullptr}, {"manifold_id", fm[f]["id"]}});
      joined += (joined.empty() ? "" : " \xc3\x97 ") + fm[f]["name"].get<std::string>();
    }
    Json geometry = {{"id", gid}, {"form", "product"}, {"coordinate_system_id", nullptr}, {"positions_path", nullptr}, {"direction_json", nullptr}, {"dimensions", dims}};
    // createManifold passes `acquisition_id: row.acquisition_id ?? null`, so the factors' acquisition is never taken (the option is given, as null)
    const Json acq = given.contains("acquisition_id") ? given["acquisition_id"] : Json(nullptr);
    Json row = {{"id", id}, {"subject_id", subject_id}, {"name", given.contains("name") && given["name"].is_string() && !given["name"].get<std::string>().empty() ? given["name"] : Json(joined)},
                {"path", nullptr}, {"type", "product"}, {"label", nullptr}, {"acquisition_id", acq}, {"is_primary", 0}, {"n_vertices", at(0)}, {"n_edges", at(1)}, {"n_faces", at(2)},
                {"n_volumes", at(3)}, {"topology_id", tid}, {"geometry_id", gid}, {"component_of_id", nullptr}, {"component_ordinal", nullptr}, {"from_index", nullptr},
                {"derived_from_id", nullptr}, {"operator", nullptr}, {"comment", nullptr}, {"source", nullptr}, {"producer_json", nullptr}, {"created_utc", nullptr}, {"created_by", nullptr},
                {"modified_utc", nullptr}, {"modified_by", nullptr}};
    return Json{{"row", row}, {"topology", topology}, {"geometry", geometry}};
  }
  // the Studio's manifoldRepository.create
  Json create_manifold(DatasetWriter& ds, Json row, const Json& c, const std::string& by) const {
    Database& db = ds.db();
    const bool has_top = c.is_object() && c.contains("topology") && c["topology"].is_object();
    const bool has_geo = c.is_object() && c.contains("geometry") && c["geometry"].is_object();
    if (!has_top && !given(row, "topology_id")) invalid("manifold " + str_or(row, "name") + ": needs a topology — its own, or an existing one's id");
    const std::string t = now_utc(true);
    const std::string tid = has_top ? (given(c["topology"], "id") ? c["topology"]["id"].get<std::string>() : uuid7()) : row["topology_id"].get<std::string>();
    const Json gid = has_geo ? Json(given(c["geometry"], "id") ? c["geometry"]["id"].get<std::string>() : uuid7()) : row.value("geometry_id", Json(nullptr));
    if (!given(row, "id")) row["id"] = uuid7();
    row["topology_id"] = tid;
    row["geometry_id"] = gid;
    row["created_utc"] = t; row["created_by"] = !by.empty() ? Json(by) : row.value("created_by", Json(nullptr));
    row["modified_utc"] = t; row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
    const std::string sid = str_or(row, "subject_id");
    db.tx([&] {
      if (has_top) {
        Json top = c["topology"];
        const Json cells = top.value("cells", Json::array());
        top.erase("cells");
        top["id"] = tid; top["subject_id"] = sid;
        upsert(db, "topology", kTopologyCols, top);
        db.execute("DELETE FROM topology_cell WHERE topology_id = ?", {Value(tid)});
        for (const auto& x : cells) db.execute("INSERT INTO topology_cell (topology_id, rank, path) VALUES (?, ?, ?)", {Value(tid), value_of(x["rank"]), value_of(x["path"])});
      }
      if (has_geo) {
        Json g = c["geometry"];
        const Json dims = g.value("dimensions", Json::array());
        g.erase("dimensions");
        g["id"] = gid; g["subject_id"] = sid;
        upsert(db, "geometry", kGeometryCols, g);
        db.execute("DELETE FROM geometry_dimension WHERE geometry_id = ?", {value_of(gid)});
        for (const auto& x : dims)
          db.execute("INSERT INTO geometry_dimension (geometry_id, ordinal, name, n_vertices, spacing, origin, unit, manifold_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     {value_of(gid), value_of(x.value("ordinal", Json(nullptr))), value_of(x.value("name", Json(nullptr))), value_of(x.value("n_vertices", Json(nullptr))),
                      value_of(x.value("spacing", Json(nullptr))), value_of(x.value("origin", Json(nullptr))), value_of(x.value("unit", Json(nullptr))), value_of(x.value("manifold_id", Json(nullptr)))});
      }
      upsert(db, "manifold", kManifoldCols, row);
    });
    drain(db, store_);
    return row_json(*db.one("SELECT * FROM manifold WHERE id = ?", {Value(row["id"].get<std::string>())}));
  }
  // THE SERVICE'S ROUTING (routedRepositories): the datasets a row (or a where) belongs to — every one when the datastore has one, or an app-scope dictionary;
  // else by its subject, selection, dataset or id. Refused when it names none of several.
  std::vector<DatasetWriter*> route_rows(const char* table, const Json& o, bool write = true) const {
    const auto all = datasets();
    auto pick = [&](const std::string& d) { return write ? &writer(d) : &reader(d); };
    std::vector<DatasetWriter*> out;
    if (all.size() == 1 || (std::string(table) == "dictionary" && o.is_object() && o.value("scope", std::string()) == "app")) { for (const auto& d : all) out.push_back(pick(d)); return out; }
    if (o.is_object()) {
      for (const char* k : {"subject_id", "subjectId"}) if (given(o, k)) { out.push_back(pick(dataset_of_subject(o[k].get<std::string>()).first)); return out; }
      for (const char* k : {"selection_id", "selectionId"}) if (given(o, k)) { out.push_back(&dataset_of_row("selection", o[k].get<std::string>(), write)); return out; }
      for (const char* k : {"dataset_id", "datasetId"})
        if (given(o, k)) {
          for (const auto& d : all) if (reader(d).db().one("SELECT 1 FROM dataset WHERE id = ?", {Value(o[k].get<std::string>())})) { out.push_back(pick(d)); return out; }
          throw Error(code::no_such_row, "no dataset " + o[k].get<std::string>());
        }
      if (given(o, "id")) { out.push_back(&dataset_of_row(table, o["id"].get<std::string>(), write)); return out; }
    }
    invalid(std::string(table) + ": " + std::to_string(all.size()) + " datasets are open and this call names none of them (a subject, a dataset, a selection or an id)");
  }
  // a subject's keys in the datastore's store
  PrefixStore store_of(DatasetWriter& ds, const std::string& subject_id) const {
    return PrefixStore(store_, text(ds.row(), "path") + "/" + subject_path(ds.db(), subject_id));
  }
  // `del` as a delete that takes its arrays (D164): the rows' arrays no remaining row names are removed from the subject's store
  void deleting(DatasetWriter& ds, const std::string& subject_id, const std::function<void()>& del) const {
    PrefixStore st = store_of(ds, subject_id);
    deleting_rows(ds.db(), st, subject_id, del);
  }
  // DELETE a row that may own arrays (D164, the service's routed delete): its arrays no remaining row names go with it
  void delete_row(DatasetWriter& ds, const char* table, const std::string& id) const {
    const auto r = ds.db().one(std::string("SELECT subject_id FROM ") + table + " WHERE id = ?", {Value(id)});
    auto del = [&] { ds.db().tx([&] { ds.db().execute(std::string("DELETE FROM ") + table + " WHERE id = ?", {Value(id)}); }); };
    if (!r) { del(); return; }
    const std::string sid = std::get<std::string>(r->at("subject_id"));
    deleting(ds, sid, del);
  }
  // the dataset holding a row of `table` (held for writing, or read)
  DatasetWriter& dataset_of_row(const char* table, const std::string& id, bool write = true) const {
    for (const std::string& d : datasets()) {
      if (reader(d).db().one(std::string("SELECT 1 FROM ") + table + " WHERE id = ?", {Value(id)})) return write ? writer(d) : reader(d);
    }
    throw Error(code::no_such_row, std::string(table) + " " + id + ": not found");
  }
  static std::string subject_path(const Database& db, const std::string& subject_id) {
    return std::get<std::string>(db.one("SELECT path FROM subject WHERE id = ?", {Value(subject_id)})->at("path"));
  }
  // the Studio's fieldRepository.create: stamps, the cell checked, contributions and child selections checked, one transaction, then the drain
  Json create_field(DatasetWriter& ds, Json row, const std::string& by, const Json& ch) const {
    Database& db = ds.db();
    const std::string t = now_utc(true);
    if (!given(row, "id")) row["id"] = uuid7();
    row["created_utc"] = t;
    row["created_by"] = !by.empty() ? Json(by) : row.value("created_by", Json(nullptr));
    row["modified_utc"] = t;
    row["modified_by"] = !by.empty() ? Json(by) : row.value("modified_by", Json(nullptr));
    if (!given(row, "cell")) row["cell"] = "0";
    const std::string name = str_or(row, "name");
    check_cell(db, str_or(row, "manifold_id"), row["cell"].get<std::string>(), "field " + name);
    const Json contributions = ch.value("contributions", Json::array());
    for (const auto& x : contributions) {
      if (!db.one("SELECT 1 FROM subject WHERE id = ?", {Value(str_or(x, "subject_id"))})) invalid("field " + name + ": a contribution names subject " + str_or(x, "subject_id") + ", which has no row");
      if (given(x, "source_field_id") && !db.one("SELECT 1 FROM field WHERE id = ?", {Value(x["source_field_id"].get<std::string>())}))
        invalid("field " + name + ": a contribution names field " + x["source_field_id"].get<std::string>() + ", which has no row");
    }
    std::vector<Json> sels;
    for (Json x : ch.value("selections", Json::array())) {
      if (!given(x, "id")) x["id"] = uuid7();
      x["subject_id"] = row["subject_id"]; x["of_field_id"] = row["id"];
      x["created_utc"] = t; x["created_by"] = !by.empty() ? Json(by) : x.value("created_by", Json(nullptr));
      x["modified_utc"] = t; x["modified_by"] = !by.empty() ? Json(by) : x.value("modified_by", Json(nullptr));
      if (given(x, "manifold_id")) check_cell(db, x["manifold_id"].get<std::string>(), given(x, "cell") ? x["cell"].get<std::string>() : "0", "selection " + str_or(x, "name"));
      sels.push_back(x);
    }
    db.tx([&] {
      upsert(db, "field", kFieldCols, row);
      if (!contributions.empty()) {
        db.execute("DELETE FROM field_contribution WHERE field_id = ?", {Value(row["id"].get<std::string>())});
        std::int64_t k = 0;
        for (const auto& c : contributions)
          db.execute("INSERT INTO field_contribution (field_id, subject_id, source_field_id, weight, ordinal) VALUES (?, ?, ?, ?, ?)",
                     {Value(row["id"].get<std::string>()), Value(str_or(c, "subject_id")), value_of(c.value("source_field_id", Json(nullptr))), Value(c.value("weight", 1.0)), Value(k++)});
      }
      for (Json x : sels) { x["cell"] = selection_cell(db, x); upsert(db, "selection", kSelectionCols, x); }
    });
    drain(db, store_);
    return row_json(*db.one("SELECT * FROM field WHERE id = ?", {Value(row["id"].get<std::string>())}));
  }

  // a dataset to read: the held writer when there is one, else a reader (no lock) kept for the next read
  mutable std::map<std::string, std::unique_ptr<DatasetWriter>> readers_;
  DatasetWriter& reader(const std::string& path) const {
    const std::string d = safe_rel(path, "store path").begin()->string();
    if (auto it = held_.find(d); it != held_.end()) return *it->second;
    auto it = readers_.find(d);
    if (it == readers_.end()) it = readers_.emplace(d, std::make_unique<DatasetWriter>(open(d, false))).first;
    return *it->second;
  }
  // the datastore's datasets, in its catalog's order (their folders)
  std::vector<std::string> datasets() const {
    std::ifstream in(root_ / kCatalogFile);
    if (!in) throw Error(code::not_found, (root_ / kCatalogFile).string() + ": no catalog (not a datastore)");
    const Json cat = Json::parse(in);
    std::vector<std::string> out;
    for (const auto& d : cat.value("datasets", Json::array())) out.push_back(d.value("path", d.value("name", std::string())));
    return out;
  }
  // the dataset a store path (or a dataset path) belongs to, opened to read (no lock)
  DatasetWriter open(const std::string& path, bool write) const {
    const fs::path rel = safe_rel(path, "store path");
    OpenOptions o;
    o.lock = write;
    return open_writer(root_ / *rel.begin(), o);
  }
  // the array's key in the datastore's store (both parts validated)
  std::string array_key(const std::string& store_path, const std::string& array_path) const {
    return join_key(safe_rel(store_path, "store path").generic_string(), safe_rel(array_path, "array path").generic_string());
  }
};

}  // namespace

EMSCRIPTEN_BINDINGS(nxr_io) {
  emscripten::class_<Datastore>("Datastore")
      .constructor<std::string, std::string>()
      .function("status", &Datastore::status)
      .function("bundle", &Datastore::bundle)
      .function("view", &Datastore::view)
      .function("row", &Datastore::row)
      .function("read", &Datastore::read)
      .function("create", &Datastore::create)
      .function("put", &Datastore::put)
      .function("remove", &Datastore::remove)
      .function("updateSubject", &Datastore::updateSubject)
      .function("removeSubject", &Datastore::removeSubject)
      .function("createField", &Datastore::createField)
      .function("selection", &Datastore::selection)
      .function("measurements", &Datastore::measurements)
      .function("operator", &Datastore::operatorOp)
      .function("fieldUpdate", &Datastore::fieldUpdate)
      .function("fieldDelete", &Datastore::fieldDelete)
      .function("manifold", &Datastore::manifold)
      .function("dictionary", &Datastore::dictionary)
      .function("job", &Datastore::job)
      .function("layout", &Datastore::layout)
      .function("datasetCreate", &Datastore::datasetCreate)
      .function("datasetDelete", &Datastore::datasetDelete)
      .function("catalog", &Datastore::catalog)
      .function("cli", &Datastore::cli)
      .function("importSubjects", &Datastore::importSubjects)
      .function("setEnv", &Datastore::setEnv)
      .function("releaseDataset", &Datastore::releaseDataset)
      .function("close", &Datastore::close);
}
