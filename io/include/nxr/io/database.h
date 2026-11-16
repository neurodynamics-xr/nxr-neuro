// nxr-io — a dataset's database, read and write: `<datastore>/<dataset>/dataset.sqlite`, created from `schema/model.sql` (a byte
// copy of the Cortical Flow Studio's schema, embedded at build) at schema 50. nxr.io.database.
//
// Nothing here knows a column by heart: the column sets are read from the schema (`PRAGMA table_info`).
#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "nxr/io/errors.h"
#include "nxr/io/json_text.h"
#include "nxr/io/store.h"

namespace nxr::io {

// A SQLite value: NULL (monostate), INTEGER, REAL, TEXT (a BLOB comes back as its bytes).
using Value = std::variant<std::monostate, std::int64_t, double, std::string>;
// One row: column name -> value. No per-table code.
using Row = std::map<std::string, Value>;

constexpr const char* kDatasetFile = "dataset.sqlite";
constexpr const char* kCatalogFile = "catalog.json";

// The schema (`model.sql`), embedded; its `PRAGMA user_version`.
const char* model_sql();
int schema_version();
// Value constructors that never pick the wrong alternative (an `int` is an INTEGER, a `const char*` is TEXT).
inline Value V(std::nullptr_t) { return std::monostate{}; }
inline Value V(int x) { return static_cast<std::int64_t>(x); }
inline Value V(long x) { return static_cast<std::int64_t>(x); }
inline Value V(long long x) { return static_cast<std::int64_t>(x); }
inline Value V(unsigned long x) { return static_cast<std::int64_t>(x); }
inline Value V(double x) { return x; }
inline Value V(const char* s) { return std::string(s); }
inline Value V(const std::string& s) { return s; }
inline Value V(const std::optional<std::string>& s) { return s ? Value(*s) : Value(std::monostate{}); }
inline Value V(unsigned int x) { return static_cast<std::int64_t>(x); }
inline Value V(unsigned long long x) { return static_cast<std::int64_t>(x); }
inline Value V(bool x) { return static_cast<std::int64_t>(x ? 1 : 0); }
inline Value V(const Value& v) { return v; }

// Extra columns of a row, by name: the options structs carry the common ones as fields, and any other column of the table goes here
// (`Cols().set("comment", "x").set("n_faces", 80).json("producer_json", Json{{"k", 1}})`). Table-driven: a new column needs no new code.
struct Cols : std::map<std::string, Value> {
  template <class T>
  Cols& set(const std::string& k, T&& v) { (*this)[k] = V(std::forward<T>(v)); return *this; }
  // a `*_json` column, written as JavaScript's JSON.stringify writes it
  Cols& json(const std::string& k, const Json& v) { (*this)[k] = jdump(v); return *this; }
  Cols& null(const std::string& k) { (*this)[k] = std::monostate{}; return *this; }
};

bool is_null(const Value& v);
// A row's text / integer / real, with a default when NULL or absent.
std::string text(const Row& r, const std::string& column);
std::int64_t integer(const Row& r, const std::string& column, std::int64_t dflt = 0);
bool is_null(const Row& r, const std::string& column);
std::optional<std::string> opt_text(const Row& r, const std::string& column);

// A version-7 UUID, the primary key of every row (D16), and ISO 8601 UTC as JavaScript's toISOString writes it.
std::string uuid7();
std::string now_utc(bool ms = true);

namespace testing {
// Deterministic ids (00000000-0000-7000-8000-<counter>) and a fixed clock, so two writers' output can be compared byte for byte.
// Used only by the conformance tests.
void deterministic(bool on, std::uint64_t start = 0);   // `start`: the counter's first value (ids continue after it)
}  // namespace testing

class Database {
 public:
  // Open a dataset's database. `create`: an empty file gets the schema (tables, version and meta in ONE transaction).
  // A database at another `user_version` is refused (D139). `read_only`: SQLITE_OPEN_READONLY.
  static Database open(const std::filesystem::path& file, bool create = false, const std::string& built_by = "nxr-convert",
                       bool read_only = false);
  // Open (or create) a SQLite file that is NOT a model database (no schema, no version check): the participants table's file. Foreign keys are on.
  static Database open_plain(const std::filesystem::path& file, bool create = false, bool read_only = false);
  Database(Database&&) noexcept;
  Database& operator=(Database&&) noexcept;
  ~Database();

  const std::filesystem::path& path() const { return path_; }
  // The open sqlite3 connection, for the one other class that works in the same transactions (SmallFileStore over a subject package's subject.sqlite).
  void* handle() const { return db_; }
  bool read_only() const { return read_only_; }

  // Run `body` in a transaction (a nested one is a savepoint); any exception rolls back and propagates.
  void tx(const std::function<void()>& body);

  std::vector<Row> all(const std::string& sql, const std::vector<Value>& params = {}) const;
  std::optional<Row> one(const std::string& sql, const std::vector<Value>& params = {}) const;
  Value value(const std::string& sql, const std::vector<Value>& params = {}) const;
  void execute(const std::string& sql, const std::vector<Value>& params = {}) const;
  // several statements with no parameters (a block of the schema: ensure_statistics)
  void exec_script(const std::string& sql) const;
  std::optional<Row> read(const std::string& table, const std::string& id) const;

  const std::vector<std::string>& columns(const std::string& table) const;
  const std::vector<std::string>& primary_key(const std::string& table) const;
  void insert(const std::string& table, const Row& row) const;
  // INSERT, or on the primary key UPDATE the given columns (never a REPLACE: a delete would cascade).
  void upsert(const std::string& table, const Row& row) const;
  void close();
  // Run `body` with the stored schema writable (SQLite's writable_schema, its defensive mode lifted for the while and restored after) — ONLY for a
  // constraint change that leaves every stored row valid (allow_value_type); the schema_version is bumped by the body.
  void with_writable_schema(const std::function<void()>& body);

 private:
  Database() = default;
  std::filesystem::path path_;
  bool read_only_ = false;
  void* db_ = nullptr;  // sqlite3*
  int depth_ = 0;
  mutable std::map<std::string, std::vector<std::string>> cols_, pk_;
  void load_columns(const std::string& table) const;
};

// ── the canonical zarr.json: the drain (D141's dirty-tracker) ────────────────────────────────────────────────────────

// meta `store_format` of a dataset whose arrays are written as format 51 (additive: user_version unchanged; every array's own zarr.json says which separator it has)
constexpr const char* kStoreFormat51 = "51: chunk key separator \".\", arrays over 4 MiB sharded (~64 MiB); both separators are read";
// Format 52 (/15): each subject's small keys (every zarr.json, every value under 1 MiB) are rows of the subject's small_files.sqlite
// (io/include/nxr/io/small_file_store.h); the shards over the threshold stay files. Formats 51 and 50 are read and written as before.
constexpr const char* kStoreFormat52 = "52: small keys (every zarr.json, values under 1 MiB) in the subject's small_files.sqlite, shards as 51; the layouts of 51 and 50 are read";
constexpr int kStoreFormatNewest = 52;
// Format 53: a subject PACKAGE (`<subject>.nxr.zarr/subject.sqlite`: the subject's rows, its small keys and a digest of every key); see subject_package.h.
constexpr const char* kStoreFormat53 = "53: subject package: the subject's rows and small keys in its own subject.sqlite, a size and SHA-256 for every key, shards as 51";
constexpr int kStoreFormatPackage = 53;
// The number a dataset's meta `store_format` starts with (50 when the dataset has none: made before format 51).
int store_format_number(const Database& db);
constexpr const char* kNodeTables[] = {"dataset", "subject", "manifold", "selection", "field", "operator", "statistic"};   // statistic: absent from a dataset made before it (has_table)
constexpr const char* kStatusTables[] = {"statistic", "manifold", "selection", "field", "operator", "subject"};  // recovery's order (a statistic before what it names)

// A VALUE TYPE added to the schema ADDITIVELY (every row valid before stays valid; the `user_version` unchanged — `hermitian3`, 2026-11-12): a database
// made before it carries the older CHECK lists in its stored `field` and `operator` tables. They are relaxed IN PLACE, the stored rows untouched
// (SQLite's procedure for a constraint change that leaves the data valid: writable_schema, the schema_version bumped, an integrity check), and noted
// in `meta` (`value_type:<type>`). Called by the writer before it writes a row of that type. Returns true when it relaxed something.
bool allow_value_type(Database& db, const std::string& value_type);

// A FOLDER removed, tolerating entries that vanish while it goes (what FolderStore's erase_prefix does for a key prefix; for the folders that are not store keys).
void remove_folder(const std::filesystem::path& p);
// Every MISSING group on the way to key `location` (and it, with self_too) as a plain group.
void ensure_groups(Store& store, const std::string& location, bool self_too = true);
// The node's attributes as the SCHEMA writes them: the text of its view (D145).
std::optional<std::string> attributes_text(const Database& db, const std::string& table, const std::string& id);
// Where a node is (datastore-relative); nullopt for a row with no node of its own (it rides in its subject's root).
std::optional<std::string> location_of(const Database& db, const std::string& table, const std::string& id);
std::string location_of_row(const Database& db, const std::string& table, const Row& row);
std::string store_path_of(const Database& db, const std::string& subject_id);
// Whether a datastore-relative path is the dataset's own folder or inside it (no `..`).
bool owns(const std::optional<std::string>& dataset_path, const std::string& path);

struct Drained {
  std::vector<std::string> written, removed, skipped;
};
// Drain `sync_job` against the datastore's store (keys `<dataset>/<subject>/…`): pending jobs in id order; an upsert writes the node's canonical zarr.json from its
// view's text; a delete removes the folder at the job's path unless a remaining row names it (D164).
Drained drain(const Database& db, Store& store);
int resync_all(const Database& db);

struct Parity {
  int nodes = 0;
  int mismatched = 0;
  Json first = Json::array();
};
// EVERY node row's zarr.json against its view, as TEXT (D145).
Parity text_parity(const Database& db, const Store& store);

}  // namespace nxr::io
