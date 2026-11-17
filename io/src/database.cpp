#include "nxr/io/database.h"
#include "nxr/io/statistic.h"

#include <sqlite3.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>

#include "named_paths.h"

namespace fs = std::filesystem;

namespace nxr::io {

// ── schema ─────────────────────────────────────────────────────────────────────────────────────────
const char* model_sql() {
  static const char* sql =
#include "gen/model_sql.inc"
      ;
  return sql;
}

int schema_version() {
  static const int v = [] {
    const std::string s = model_sql();
    std::smatch m;
    static const std::regex re(R"(\nPRAGMA user_version = (\d+);)");
    if (!std::regex_search(s, m, re)) throw Error(code::not_a_database, "model.sql: no `PRAGMA user_version = N;` statement");
    return std::stoi(m[1]);
  }();
  return v;
}

// ── values ─────────────────────────────────────────────────────────────────────────────────────────
bool is_null(const Value& v) { return std::holds_alternative<std::monostate>(v); }
bool is_null(const Row& r, const std::string& c) {
  auto it = r.find(c);
  return it == r.end() || is_null(it->second);
}
std::string text(const Row& r, const std::string& column) {
  auto it = r.find(column);
  if (it == r.end()) return "";
  if (auto* s = std::get_if<std::string>(&it->second)) return *s;
  if (auto* n = std::get_if<std::int64_t>(&it->second)) return std::to_string(*n);
  if (auto* x = std::get_if<double>(&it->second)) return js_number(*x);
  return "";
}
std::optional<std::string> opt_text(const Row& r, const std::string& column) {
  if (is_null(r, column)) return std::nullopt;
  return text(r, column);
}
std::int64_t integer(const Row& r, const std::string& column, std::int64_t dflt) {
  auto it = r.find(column);
  if (it == r.end()) return dflt;
  if (auto* n = std::get_if<std::int64_t>(&it->second)) return *n;
  if (auto* x = std::get_if<double>(&it->second)) return static_cast<std::int64_t>(*x);
  return dflt;
}

// ── ids and the clock ──────────────────────────────────────────────────────────────────────────────
namespace {
std::atomic<bool> g_deterministic{false};
std::atomic<std::uint64_t> g_counter{0};
}  // namespace

namespace testing {
void deterministic(bool on, std::uint64_t start) {
  g_deterministic = on;
  g_counter = start;
}
}  // namespace testing

std::string uuid7() {
  char buf[40];
  if (g_deterministic) {
    std::snprintf(buf, sizeof buf, "00000000-0000-7000-8000-%012llx", static_cast<unsigned long long>(++g_counter));
    return buf;
  }
  const std::uint64_t ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count()) & ((1ULL << 48) - 1);
  static std::random_device rd;
  std::uint8_t b[16];
  for (int i = 0; i < 16; i += 4) {
    const std::uint32_t r = rd();
    std::memcpy(b + i, &r, 4);
  }
  for (int i = 0; i < 6; ++i) b[i] = static_cast<std::uint8_t>(ms >> (8 * (5 - i)));
  b[6] = static_cast<std::uint8_t>((b[6] & 0x0F) | 0x70);
  b[8] = static_cast<std::uint8_t>((b[8] & 0x3F) | 0x80);
  std::snprintf(buf, sizeof buf, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3], b[4],
                b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return buf;
}

std::string now_utc(bool ms) {
  if (g_deterministic) return ms ? "2026-01-01T00:00:00.000Z" : "2026-01-01T00:00:00Z";
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[40];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%S", &tm);
  std::string s = buf;
  if (ms) {
    const auto frac = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    char f[8];
    std::snprintf(f, sizeof f, ".%03lldZ", static_cast<long long>(frac));
    return s + f;
  }
  return s + "Z";
}

// ── the database ───────────────────────────────────────────────────────────────────────────────────
namespace {
sqlite3* H(void* p) { return static_cast<sqlite3*>(p); }

[[noreturn]] void fail(sqlite3* db, const std::string& what = "") {
  throw Error(code::sqlite, std::string(sqlite3_errmsg(db)) + (what.empty() ? "" : " in: " + what));
}

void exec(sqlite3* db, const std::string& sql) {
  char* err = nullptr;
  if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
    const std::string m = err ? err : sqlite3_errmsg(db);
    sqlite3_free(err);
    throw Error(code::sqlite, m);
  }
}

void bind(sqlite3_stmt* st, const std::vector<Value>& params) {
  for (std::size_t i = 0; i < params.size(); ++i) {
    const int k = static_cast<int>(i) + 1;
    const Value& v = params[i];
    if (std::holds_alternative<std::monostate>(v)) sqlite3_bind_null(st, k);
    else if (auto* n = std::get_if<std::int64_t>(&v)) sqlite3_bind_int64(st, k, *n);
    else if (auto* x = std::get_if<double>(&v)) sqlite3_bind_double(st, k, *x);
    else {
      const std::string& s = std::get<std::string>(v);
      sqlite3_bind_text(st, k, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
    }
  }
}
}  // namespace

Database::Database(Database&& o) noexcept
    : path_(std::move(o.path_)), read_only_(o.read_only_), db_(o.db_), depth_(o.depth_), cols_(std::move(o.cols_)), pk_(std::move(o.pk_)) {
  o.db_ = nullptr;
}
Database& Database::operator=(Database&& o) noexcept {
  if (this != &o) {
    close();
    path_ = std::move(o.path_);
    read_only_ = o.read_only_;
    db_ = o.db_;
    depth_ = o.depth_;
    cols_ = std::move(o.cols_);
    pk_ = std::move(o.pk_);
    o.db_ = nullptr;
  }
  return *this;
}
Database::~Database() { close(); }
void Database::close() {
  if (db_) sqlite3_close(H(db_));
  db_ = nullptr;
}

namespace {
// SCHEMA 50 -> 51, in place (spec section 8: "the version step"): the subject kind `registration` allowed, the tables key_digest, subject_entry and template_reference
// added, the two attribute views that name the schema version made again, `user_version` 51. Additive: every row valid before stays valid; the arrays are untouched.
// What the migration is, is what model.sql says: the block between its "schema 51" markers and the two views are read from it.
void migrate_50_to_51(Database& d) {
  const std::string model = model_sql();
  const auto cut = [&](const std::string& from, const std::string& to) {
    const auto a = model.find(from);
    const auto b = a == std::string::npos ? a : model.find(to, a);
    if (b == std::string::npos) throw Error(code::not_a_database, "model.sql: no `" + from + "` in the schema, so a database at 50 cannot be migrated");
    return model.substr(a, b - a);
  };
  std::string tables = cut("CREATE TABLE key_digest", "-- ── schema 51 (end)");
  for (std::string::size_type at = 0; (at = tables.find("CREATE TABLE ", at)) != std::string::npos; at += 28) tables.replace(at, 13, "CREATE TABLE IF NOT EXISTS ");
  const std::string subject_view = cut("CREATE VIEW subject_attributes AS", ";\n") + ";\n";
  const std::string dataset_view = cut("CREATE VIEW dataset_attributes AS", ";\n") + ";\n";
  static const std::string kBefore = "kind IN ('subject', 'template', 'group')", kAfter = "kind IN ('subject', 'template', 'group', 'registration')";
  const auto sub = d.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'subject'");
  if (!sub || (text(*sub, "sql").find(kBefore) == std::string::npos && text(*sub, "sql").find("'registration'") == std::string::npos))
    throw Error(code::schema_version, "table subject: its kind list is not the schema 50's; the database cannot be migrated to 51 in place");
  const std::int64_t sv = integer(*d.one("PRAGMA schema_version"), "schema_version");
  d.with_writable_schema([&] {
    d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'subject' AND instr(sql, ?) = 0", {V(kBefore), V(kAfter), V("'registration'")});
    d.execute("PRAGMA schema_version = " + std::to_string(sv + 1));
  });
  d.exec_script("BEGIN;\n" + tables + "DROP VIEW IF EXISTS subject_attributes;\n" + subject_view + "DROP VIEW IF EXISTS dataset_attributes;\n" + dataset_view +
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_migrated:50->51', '" + now_utc() + "');\nPRAGMA user_version = 51;\nCOMMIT;");
  const auto ok = d.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after it was migrated from schema 50 to 51");
}
}  // namespace

Database Database::open(const fs::path& file, bool create, const std::string& built_by, bool read_only) {
  const std::string p = file.string();
  if (!create && !fs::is_regular_file(file)) {
    throw Error(code::not_found, p + ": no dataset database here — create the dataset first (nxr dataset create)");
  }
  Database d;
  d.path_ = file;
  d.read_only_ = read_only;
  auto connect = [&]() {
    sqlite3* h = nullptr;
    const int flags = read_only ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    if (sqlite3_open_v2(p.c_str(), &h, flags, nullptr) != SQLITE_OK) {
      const std::string m = h ? sqlite3_errmsg(h) : "cannot open";
      if (h) sqlite3_close(h);
      throw Error(code::sqlite, p + ": " + m);
    }
    sqlite3_busy_timeout(h, 5000);
    exec(h, "PRAGMA foreign_keys = ON");
    return h;
  };
  d.db_ = connect();
  try {
    const auto meta = d.one("SELECT count(*) AS n FROM sqlite_master WHERE type = 'table' AND name = 'meta'");
    if (integer(*meta, "n") == 0) {
      if (!create) throw Error(code::not_a_database, p + ": not a dataset database (no schema)");
      exec(H(d.db_), "BEGIN;\n" + std::string(model_sql()) + "\nINSERT INTO meta (key, value) VALUES ('built_utc', '" + now_utc() +
                         "');\nINSERT INTO meta (key, value) VALUES ('built_by', '" + built_by + "');\n"
                         // the array store format of a dataset made now (additive: user_version unchanged): chunk key separator ".", large arrays sharded; every array's own zarr.json says which it has
                         "INSERT INTO meta (key, value) VALUES ('store_format', '" + std::string(kStoreFormat51) + "');\nCOMMIT;");
      d.close();            // on exFAT the connection that grew the file reads it as read-only: reconnect
      d.db_ = connect();
    }
    const std::int64_t v = integer(*d.one("PRAGMA user_version"), "user_version");
    const int want = schema_version();
    if (v == 50 && want == 51) {
      // a database at 50 is a database at 51 without the additions: opened to write, it is migrated in place (additive); opened to read, it is read as it is
      if (!read_only) migrate_50_to_51(d);
    } else if (v != want) {
      const std::string how =
          v == want - 2 ? "run `nxr migrate --dataset <dataset folder>` (49 to 50; this build then migrates 50 to 51 when it opens the dataset to write; it keeps the old database and the arrays; add `--dry-run` to see what it will do first)"
                        : "re-convert it from its source (D139/D144)";
      throw Error(code::schema_version, p + ": user_version " + std::to_string(v) + ", this build reads " + std::to_string(want) + " — " + how);
    }
  } catch (...) {
    d.close();
    throw;
  }
  return d;
}

Database Database::open_plain(const fs::path& file, bool create, bool read_only) {
  const std::string p = file.string();
  if (!create && !fs::is_regular_file(file)) throw Error(code::not_found, p + ": no such database");
  Database d;
  d.path_ = file;
  d.read_only_ = read_only;
  sqlite3* h = nullptr;
  const int flags = read_only ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | (create ? SQLITE_OPEN_CREATE : 0));
  if (sqlite3_open_v2(p.c_str(), &h, flags, nullptr) != SQLITE_OK) {
    const std::string m = h ? sqlite3_errmsg(h) : "cannot open";
    if (h) sqlite3_close(h);
    throw Error(code::sqlite, p + ": " + m);
  }
  sqlite3_busy_timeout(h, 5000);
  exec(h, "PRAGMA foreign_keys = ON");
  d.db_ = h;
  return d;
}

void Database::tx(const std::function<void()>& body) {
  sqlite3* db = H(db_);
  if (depth_ == 0) exec(db, "BEGIN");
  else exec(db, "SAVEPOINT sp" + std::to_string(depth_));
  ++depth_;
  try {
    body();
  } catch (...) {
    --depth_;
    try {
      if (depth_ == 0) exec(db, "ROLLBACK");
      else { exec(db, "ROLLBACK TO sp" + std::to_string(depth_)); exec(db, "RELEASE sp" + std::to_string(depth_)); }
    } catch (...) {}
    throw;
  }
  --depth_;
  if (depth_ == 0) {
    try {
      exec(db, "COMMIT");
    } catch (const Error& e) {
      // a DEFERRED key fails at COMMIT with no row named — name it before rolling back
      std::string bad;
      try {
        int n = 0;
        for (const auto& r : all("PRAGMA foreign_key_check")) {
          if (n++ >= 5) break;
          bad += (bad.empty() ? "" : "; ") + text(r, "table") + " rowid " + text(r, "rowid") + " → " + text(r, "parent");
        }
      } catch (...) {}
      try { exec(db, "ROLLBACK"); } catch (...) {}
      throw Error(code::sqlite, std::string(e.what()) + (bad.empty() ? "" : ": " + bad));
    }
  } else {
    exec(db, "RELEASE sp" + std::to_string(depth_));
  }
}

std::vector<Row> Database::all(const std::string& sql, const std::vector<Value>& params) const {
  sqlite3* db = H(db_);
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) fail(db, sql);
  bind(st, params);
  std::vector<Row> rows;
  int rc;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {
    Row r;
    for (int c = 0, n = sqlite3_column_count(st); c < n; ++c) {
      const std::string name = sqlite3_column_name(st, c);
      switch (sqlite3_column_type(st, c)) {
        case SQLITE_INTEGER: r[name] = static_cast<std::int64_t>(sqlite3_column_int64(st, c)); break;
        case SQLITE_FLOAT: r[name] = sqlite3_column_double(st, c); break;
        case SQLITE_NULL: r[name] = std::monostate{}; break;
        default: {
          const auto* b = static_cast<const char*>(sqlite3_column_blob(st, c));
          r[name] = std::string(b ? b : "", static_cast<std::size_t>(sqlite3_column_bytes(st, c)));
        }
      }
    }
    rows.push_back(std::move(r));
  }
  std::string err;
  if (rc != SQLITE_DONE) err = sqlite3_errmsg(db);
  sqlite3_finalize(st);
  if (!err.empty()) throw Error(code::sqlite, err);
  return rows;
}

std::optional<Row> Database::one(const std::string& sql, const std::vector<Value>& params) const {
  auto rows = all(sql, params);
  if (rows.empty()) return std::nullopt;
  return std::move(rows.front());
}

Value Database::value(const std::string& sql, const std::vector<Value>& params) const {
  sqlite3* db = H(db_);
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) fail(db, sql);
  bind(st, params);
  Value out;
  const int rc = sqlite3_step(st);
  if (rc == SQLITE_ROW) {
    switch (sqlite3_column_type(st, 0)) {
      case SQLITE_INTEGER: out = static_cast<std::int64_t>(sqlite3_column_int64(st, 0)); break;
      case SQLITE_FLOAT: out = sqlite3_column_double(st, 0); break;
      case SQLITE_NULL: break;
      default: {
        const auto* b = static_cast<const char*>(sqlite3_column_blob(st, 0));
        out = std::string(b ? b : "", static_cast<std::size_t>(sqlite3_column_bytes(st, 0)));
      }
    }
  }
  std::string err;
  if (rc != SQLITE_ROW && rc != SQLITE_DONE) err = sqlite3_errmsg(db);
  sqlite3_finalize(st);
  if (!err.empty()) throw Error(code::sqlite, err);
  return out;
}

void Database::with_writable_schema(const std::function<void()>& body) {
  sqlite3* db = H(db_);
  int defensive = 0;
  sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, -1, &defensive);
  sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, 0, nullptr);
  execute("PRAGMA writable_schema = ON");
  try { body(); } catch (...) { execute("PRAGMA writable_schema = OFF"); sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, defensive, nullptr); throw; }
  execute("PRAGMA writable_schema = OFF");
  sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, defensive, nullptr);
  cols_.clear(); pk_.clear();
}

void Database::exec_script(const std::string& sql) const { exec(H(db_), sql); }

void Database::execute(const std::string& sql, const std::vector<Value>& params) const {
  sqlite3* db = H(db_);
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) fail(db, sql);
  bind(st, params);
  int rc;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {}
  std::string err;
  if (rc != SQLITE_DONE) err = sqlite3_errmsg(db);
  sqlite3_finalize(st);
  if (!err.empty()) throw Error(code::sqlite, err);
}

std::optional<Row> Database::read(const std::string& table, const std::string& id) const {
  return one("SELECT * FROM " + table + " WHERE id = ?", {id});
}

void Database::load_columns(const std::string& table) const {
  if (cols_.count(table)) return;
  auto info = all("PRAGMA table_info(" + table + ")");
  if (info.empty()) throw Error(code::no_such_row, "no table '" + table + "' in the schema");
  std::vector<std::string> cols;
  std::vector<std::pair<std::int64_t, std::string>> pk;
  for (const auto& r : info) {
    cols.push_back(text(r, "name"));
    if (const auto k = integer(r, "pk")) pk.push_back({k, text(r, "name")});
  }
  std::sort(pk.begin(), pk.end());
  cols_[table] = cols;
  auto& p = pk_[table];
  for (auto& x : pk) p.push_back(x.second);
}
const std::vector<std::string>& Database::columns(const std::string& table) const {
  load_columns(table);
  return cols_.at(table);
}
const std::vector<std::string>& Database::primary_key(const std::string& table) const {
  load_columns(table);
  return pk_.at(table);
}

namespace {
void check_known(const Database& db, const std::string& table, const Row& row) {
  const auto& cols = db.columns(table);
  std::string unknown;
  for (const auto& [k, _] : row)
    if (std::find(cols.begin(), cols.end(), k) == cols.end()) unknown += (unknown.empty() ? "" : ", ") + std::string("'") + k + "'";
  if (!unknown.empty()) throw Error(code::no_such_row, table + ": no column(s) [" + unknown + "]");
}
}  // namespace

void Database::insert(const std::string& table, const Row& row) const {
  check_known(*this, table, row);
  std::string names, marks;
  std::vector<Value> args;
  for (const auto& c : columns(table)) {
    auto it = row.find(c);
    if (it == row.end()) continue;
    names += (names.empty() ? "" : ", ") + c;
    marks += (marks.empty() ? "?" : ", ?");
    args.push_back(it->second);
  }
  execute("INSERT INTO " + table + " (" + names + ") VALUES (" + marks + ")", args);
}

void Database::upsert(const std::string& table, const Row& row) const {
  check_known(*this, table, row);
  const auto& pk = primary_key(table);
  std::string names, marks, conflict, sets;
  std::vector<Value> args;
  for (const auto& c : columns(table)) {
    auto it = row.find(c);
    if (it == row.end()) continue;
    names += (names.empty() ? "" : ", ") + c;
    marks += (marks.empty() ? "?" : ", ?");
    args.push_back(it->second);
    if (std::find(pk.begin(), pk.end(), c) == pk.end()) sets += (sets.empty() ? "" : ", ") + c + " = excluded." + c;
  }
  for (const auto& k : pk) conflict += (conflict.empty() ? "" : ", ") + k;
  execute("INSERT INTO " + table + " (" + names + ") VALUES (" + marks + ") ON CONFLICT (" + conflict + ") DO " +
              (sets.empty() ? std::string("NOTHING") : "UPDATE SET " + sets),
          args);
}

int store_format_number(const Database& db) {
  const auto r = db.one("SELECT value FROM meta WHERE key = 'store_format'");
  if (!r) return 50;
  const std::string v = text(*r, "value");
  int n = 0, digits = 0;
  for (char c : v) {
    if (c < '0' || c > '9') break;
    n = n * 10 + (c - '0');
    ++digits;
  }
  return digits ? n : 50;
}

// ── files ──────────────────────────────────────────────────────────────────────────────────────────
void remove_folder(const fs::path& p) {
  std::error_code ec;
  if (!fs::exists(fs::symlink_status(p, ec))) return;
  for (int attempt = 0; attempt < 2; ++attempt) {
    ec.clear();
    fs::remove_all(p, ec);  // entries that vanish while it goes are fine
    if (!fs::exists(fs::symlink_status(p, ec))) return;
  }
  throw Error(code::io, p.string() + ": cannot be removed");
}

static std::vector<std::string> split_path(const std::string& s) {
  std::vector<std::string> parts;
  std::stringstream ss(s);
  std::string p;
  while (std::getline(ss, p, '/')) if (!p.empty()) parts.push_back(p);
  return parts;
}

void ensure_groups(Store& store, const std::string& location, bool self_too) {
  const auto parts = split_path(location);
  const std::size_t upto = self_too ? parts.size() : (parts.empty() ? 0 : parts.size() - 1);
  std::string d;
  for (std::size_t i = 0; i < upto; ++i) {
    d = join_key(d, parts[i]);
    const std::string f = join_key(d, "zarr.json");
    if (!store.exists(f)) store.set(f, std::string(kEmptyGroup));
  }
}

std::optional<std::string> attributes_text(const Database& db, const std::string& table, const std::string& id) {
  const Value v = db.value("SELECT attributes FROM " + table + "_attributes WHERE id = ?", {id});
  if (is_null(v)) return std::nullopt;
  return std::get<std::string>(v);
}

std::string store_path_of(const Database& db, const std::string& subject_id) {
  auto r = db.one("SELECT ds.path AS d, s.path AS s FROM subject s JOIN dataset ds ON ds.id = s.dataset_id WHERE s.id = ?", {subject_id});
  if (!r) throw Error(code::no_such_row, "subject " + subject_id + ": no such subject");
  return text(*r, "d") + "/" + text(*r, "s");
}

std::string location_of_row(const Database& db, const std::string& table, const Row& attrs) {
  const std::string path = text(attrs, "path");
  if (table == "dataset") return path;
  if (table == "subject") {
    auto ds = db.read("dataset", text(attrs, "dataset_id"));
    if (!ds) throw Error(code::no_such_row, "subject " + text(attrs, "id") + ": no dataset");
    return text(*ds, "path") + "/" + path;
  }
  return store_path_of(db, text(attrs, "subject_id")) + "/" + path;
}

std::optional<std::string> location_of(const Database& db, const std::string& table, const std::string& id) {
  auto r = db.read(table, id);
  if (!r || text(*r, "path").empty()) return std::nullopt;
  if (table == "subject") return store_path_of(db, id);
  if (table == "dataset") return text(*r, "path");
  return store_path_of(db, text(*r, "subject_id")) + "/" + text(*r, "path");
}

bool owns(const std::optional<std::string>& dataset_path, const std::string& path) {
  if (!dataset_path || dataset_path->empty()) return false;
  if (!path.empty() && path[0] == '/') return false;
  const auto parts = split_path(path), own = split_path(*dataset_path);
  if (std::find(parts.begin(), parts.end(), "..") != parts.end()) return false;
  if (parts.size() < own.size()) return false;
  return std::equal(own.begin(), own.end(), parts.begin());
}

// ── the drain ──────────────────────────────────────────────────────────────────────────────────────
static void finish(const Database& db, std::int64_t id, const std::optional<std::string>& error) {
  db.execute("UPDATE sync_job SET done_utc = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), error = ? WHERE id = ?",
             {error ? Value(*error) : Value(std::monostate{}), Value(id)});
}

static void write_zarr_json(Store& store, const std::string& key, const std::string& attributes) {
  Json doc = Json::object();
  if (const auto raw = store.get_text(key)) doc = Json::parse(*raw);
  else { doc["zarr_format"] = 3; doc["node_type"] = "group"; }
  store.set(key, zarr_json_text(doc, attributes));
}

Drained drain(const Database& db_const, Store& store) {
  Database& db = const_cast<Database&>(db_const);
  Drained out;
  const auto ownrow = db.one("SELECT path FROM dataset LIMIT 1");
  const std::optional<std::string> own = ownrow ? opt_text(*ownrow, "path") : std::nullopt;

  auto sync = [&](const Row& job) {
    const std::int64_t jid = integer(job, "id");
    if (text(job, "op") == "delete") {
      const std::string jp = text(job, "path");
      // ONLY INSIDE THIS DATASET'S FOLDER (2026-10-26)
      if (!jp.empty() && !owns(own, jp)) {
        out.skipped.push_back(jp);
        return finish(db, jid, "refused: " + jp + " is outside this dataset's folder (" + own.value_or("") + ")");
      }
      // D164: a node's folder stays while a row that remains names it (equal, inside or containing)
      if (!jp.empty() && named_by_remaining(db, own, jp)) {
        out.skipped.push_back(jp);
        return finish(db, jid, std::nullopt);
      }
      if (!jp.empty() && store.exists_prefix(jp)) store.erase_prefix(jp);
      if (!jp.empty()) out.removed.push_back(jp);
      return finish(db, jid, std::nullopt);
    }
    const auto txt = attributes_text(db, text(job, "table_name"), text(job, "row_id"));
    if (!txt) {
      out.skipped.push_back(text(job, "row_id"));
      return finish(db, jid, "row gone before sync");
    }
    const Json attrs = Json::parse(*txt);
    const auto path = attrs.contains("path") && attrs["path"].is_string() ? attrs["path"].get<std::string>() : std::string();
    if (path.empty()) {
      if (attrs.contains("subject_id") && attrs["subject_id"].is_string()) {
        const std::string sid = attrs["subject_id"].get<std::string>();
        db.execute("INSERT INTO sync_job (table_name, row_id, op) SELECT 'subject', ?, 'upsert' WHERE NOT EXISTS "
                   "(SELECT 1 FROM sync_job j WHERE j.table_name = 'subject' AND j.row_id = ? AND j.op = 'upsert' AND j.done_utc IS NULL)",
                   {sid, sid});
      }
      out.skipped.push_back(text(job, "row_id"));
      return finish(db, jid, std::nullopt);
    }
    Row r;
    r["path"] = path;
    for (const char* k : {"dataset_id", "subject_id", "id"})
      if (attrs.contains(k) && attrs[k].is_string()) r[k] = attrs[k].get<std::string>();
    const std::string location = location_of_row(db, text(job, "table_name"), r);
    const std::string file = join_key(location, "zarr.json");
    if (!store.exists(file)) ensure_groups(store, location);
    write_zarr_json(store, file, *txt);  // WHOLESALE: the attributes ARE the row
    out.written.push_back(location);
    finish(db, jid, std::nullopt);
  };

  while (true) {
    const auto jobs = db.all("SELECT * FROM sync_job WHERE done_utc IS NULL ORDER BY id");
    if (jobs.empty()) break;
    std::optional<Error> failed;
    std::optional<std::string> failed_what;
    db.tx([&] {
      for (const auto& job : jobs) {
        try {
          sync(job);
        } catch (const std::exception& e) {
          finish(db, integer(job, "id"), std::string(e.what()));
          failed_what = e.what();
          if (auto* er = dynamic_cast<const Error*>(&e)) failed = *er;
          else failed = Error(code::io, e.what());
          break;
        }
      }
    });
    if (failed) throw *failed;
  }
  return out;
}

int resync_all(const Database& db) {
  int n = 0;
  for (const char* table : kNodeTables) {
    if (!has_table(db, table)) continue;                     // a dataset made before the statistic table
    for (const auto& r : db.all(std::string("SELECT id FROM ") + table + " WHERE path IS NOT NULL")) {
      db.execute("INSERT OR IGNORE INTO sync_job (table_name, row_id, op) VALUES (?, ?, 'upsert')", {std::string(table), text(r, "id")});
      ++n;
    }
  }
  return n;
}

Parity text_parity(const Database& db, const Store& store) {
  Parity out;
  for (const char* table : kNodeTables) {
    if (!has_table(db, table)) continue;                     // a dataset made before the statistic table
    for (const auto& r : db.all(std::string("SELECT id FROM ") + table + " WHERE path IS NOT NULL")) {
      const Row row = *db.read(table, text(r, "id"));
      const std::string file = join_key(location_of_row(db, table, row), "zarr.json");
      ++out.nodes;
      auto bad = [&](const char* why) {
        ++out.mismatched;
        if (out.first.size() < 5) out.first.push_back(Json{{"table", table}, {"path", text(row, "path")}, {"reason", why}});
      };
      const auto have = store.get_text(file);
      if (!have) { bad("missing"); continue; }
      const std::string& raw = *have;
      const std::string txt = attributes_text(db, table, text(r, "id")).value_or("null");
      const std::string want = zarr_json_text(Json::parse(raw), txt);
      if (raw != want) bad(raw.find(txt) == std::string::npos ? "not the view's text" : "not canonical");
    }
  }
  return out;
}

bool allow_value_type(Database& db, const std::string& value_type) {
  static const std::string kBefore = "'matrix3', 'quaternion')";
  const std::string quoted = "'" + value_type + "'", after = "'matrix3', 'quaternion', " + quoted + ")";
  bool need = false;
  for (const Row& r : db.all("SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN ('field', 'operator')"))
    if (text(r, "sql").find(quoted) == std::string::npos) {
      if (text(r, "sql").find(kBefore) == std::string::npos) throw Error(code::schema_version, "table " + text(r, "name") + ": its value-type list is not the schema's; " + value_type + " cannot be allowed in place");
      need = true;
    }
  if (!need) return false;
  const std::int64_t v = integer(*db.one("PRAGMA schema_version"), "schema_version");
  db.with_writable_schema([&] {
    db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name IN ('field', 'operator') AND instr(sql, ?) = 0", {V(kBefore), V(after), V(quoted)});
    db.execute("PRAGMA schema_version = " + std::to_string(v + 1));
  });
  const auto ok = db.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after " + value_type + " was allowed");
  db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V("value_type:" + value_type), V(now_utc() + " allowed in place (additive; user_version unchanged)")});
  return true;
}

}  // namespace nxr::io
