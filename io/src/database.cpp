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
#include "reader_vfs.h"

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
    : path_(std::move(o.path_)), read_only_(o.read_only_), db_(o.db_), depth_(o.depth_), stmts_(std::move(o.stmts_)), cols_(std::move(o.cols_)), pk_(std::move(o.pk_)) {
  o.db_ = nullptr;
  o.stmts_.clear();
}
Database& Database::operator=(Database&& o) noexcept {
  if (this != &o) {
    close();
    path_ = std::move(o.path_);
    read_only_ = o.read_only_;
    db_ = o.db_;
    depth_ = o.depth_;
    stmts_ = std::move(o.stmts_);
    o.stmts_.clear();
    cols_ = std::move(o.cols_);
    pk_ = std::move(o.pk_);
    o.db_ = nullptr;
  }
  return *this;
}
Database::~Database() { close(); }
void Database::finalize_prepared() noexcept {
  for (auto& kv : stmts_) sqlite3_finalize(static_cast<sqlite3_stmt*>(kv.second));
  stmts_.clear();
}
void Database::close() {
  finalize_prepared();
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

// SCHEMA 51 -> 52, in place: the table selection_factor (a product selection's factors, in order) and its index, the two attribute views made again (they name the schema
// version), `user_version` 52. Additive: every row valid before stays valid and the arrays are untouched. The DATA of the selections made before (label arrays, the JSON
// form of a set per element) is converted by `migrate_selection_storage` (migrate.h), an explicit step that sets meta `selection_storage`; a database migrated here but not
// there is read by this build's selection store only after that step (it says so).
void migrate_51_to_52(Database& d) {
  const std::string model = model_sql();
  const auto cut = [&](const std::string& from, const std::string& to) {
    const auto a = model.find(from);
    const auto b = a == std::string::npos ? a : model.find(to, a);
    if (b == std::string::npos) throw Error(code::not_a_database, "model.sql: no `" + from + "` in the schema, so a database at 51 cannot be migrated");
    return model.substr(a, b - a);
  };
  std::string tables = cut("CREATE TABLE selection_factor", "-- ── schema 52 (end)");
  for (std::string::size_type at = 0; (at = tables.find("CREATE TABLE ", at)) != std::string::npos; at += 28) tables.replace(at, 13, "CREATE TABLE IF NOT EXISTS ");
  for (std::string::size_type at = 0; (at = tables.find("CREATE INDEX ", at)) != std::string::npos; at += 28) tables.replace(at, 13, "CREATE INDEX IF NOT EXISTS ");
  const std::string subject_view = cut("CREATE VIEW subject_attributes AS", ";\n") + ";\n";
  const std::string dataset_view = cut("CREATE VIEW dataset_attributes AS", ";\n") + ";\n";
  d.exec_script("BEGIN;\n" + tables + "DROP VIEW IF EXISTS subject_attributes;\n" + subject_view + "DROP VIEW IF EXISTS dataset_attributes;\n" + dataset_view +
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_migrated:51->52', '" + now_utc() + "');\nPRAGMA user_version = 52;\nCOMMIT;");
  const auto ok = d.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after it was migrated from schema 51 to 52");
}
// SCHEMA 52 -> 53, in place (docs/specs/2026-11-17-consolidation.md, section 6.2): `selection_factor.level` (a product's factor may be a LEVEL of a stored grid), the two views the
// measurement query reads (member_value, product_factor_level), the two attribute views made again (they name the schema version), `user_version` 53. Additive: every row valid
// before stays valid and the arrays are untouched (a null level is the factor as a whole). The data step (statistic rows with a member axis into selection_measurement, products
// made for a coupling rewritten as products of (grid, level)) is `migrate_measurement_storage` (migrate.h), explicit and not run here. Idempotent: a database that has the column
// (one made from this model.sql, or a 51 that passed through migrate_51_to_52) is not altered again.
void migrate_52_to_53(Database& d) {
  const std::string model = model_sql();
  const auto cut = [&](const std::string& from, const std::string& to) {
    const auto a = model.find(from);
    const auto b = a == std::string::npos ? a : model.find(to, a);
    if (b == std::string::npos) throw Error(code::not_a_database, "model.sql: no `" + from + "` in the schema, so a database at 52 cannot be migrated");
    return model.substr(a, b - a);
  };
  ensure_statistics(d);   // the views read selection_measurement.of_statistic_id and of_index: a database made before the statistics has them added first (in place, user_version unchanged)
  const bool has_level = !d.all("SELECT 1 FROM pragma_table_info('selection_factor') WHERE name = 'level'").empty();
  const std::string views = cut("CREATE VIEW IF NOT EXISTS member_value AS", "-- ── schema 53 (end)");
  const std::string subject_view = cut("CREATE VIEW subject_attributes AS", ";\n") + ";\n";
  const std::string dataset_view = cut("CREATE VIEW dataset_attributes AS", ";\n") + ";\n";
  d.exec_script("BEGIN;\n" + std::string(has_level ? "" : "ALTER TABLE selection_factor ADD COLUMN level INTEGER;\n") + views + "DROP VIEW IF EXISTS subject_attributes;\n" + subject_view +
                "DROP VIEW IF EXISTS dataset_attributes;\n" + dataset_view +
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_migrated:52->53', '" + now_utc() + "');\nPRAGMA user_version = 53;\nCOMMIT;");
  const auto ok = d.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after it was migrated from schema 52 to 53");
}
// SCHEMA 53 -> 54, in place: the CHECK of `selection_measurement` that an array-backed measurement is per member of a FIELD becomes "of a field or of a statistic" (`measure` takes a stored statistic
// as its input; the measurement of a PSD over its channels is [frequencies x members x components] with `of_statistic_id` and no field). SQLite cannot alter a CHECK: its text in
// sqlite_master is replaced under writable_schema, `schema_version` bumped, the database checked (SQLite's procedure for a change that leaves every stored row valid: the new CHECK admits all the
// old one did). The two attribute views that name the schema version are made again, `user_version` 54. Idempotent: a database whose CHECK is already the new one is not rewritten.
// Written and tested on synthetic databases (test_schema54); never run on a real dataset by the agents that wrote it.
void migrate_53_to_54(Database& d) {
  const std::string model = model_sql();
  const auto cut = [&](const std::string& from, const std::string& to) {
    const auto a = model.find(from);
    const auto b = a == std::string::npos ? a : model.find(to, a);
    if (b == std::string::npos) throw Error(code::not_a_database, "model.sql: no `" + from + "` in the schema, so a database at 53 cannot be migrated");
    return model.substr(a, b - a);
  };
  ensure_statistics(d);   // the new CHECK reads of_statistic_id: a database made before the statistics has the column added first (in place, user_version unchanged)
  static const std::string kBefore = "CHECK (array_path IS NULL OR (code IS NULL AND of_field_id IS NOT NULL))", kAfter = "CHECK (array_path IS NULL OR (code IS NULL AND (of_field_id IS NOT NULL OR of_statistic_id IS NOT NULL)))";
  const auto tbl = d.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'selection_measurement'");
  if (!tbl || (text(*tbl, "sql").find(kBefore) == std::string::npos && text(*tbl, "sql").find(kAfter) == std::string::npos))
    throw Error(code::schema_version, "table selection_measurement: its array CHECK is not the schema 53's; the database cannot be migrated to 54 in place");
  const std::string subject_view = cut("CREATE VIEW subject_attributes AS", ";\n") + ";\n";
  const std::string dataset_view = cut("CREATE VIEW dataset_attributes AS", ";\n") + ";\n";
  if (text(*tbl, "sql").find(kAfter) == std::string::npos) {
    const std::int64_t sv = integer(*d.one("PRAGMA schema_version"), "schema_version");
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'selection_measurement'", {V(kBefore), V(kAfter)});
      d.execute("PRAGMA schema_version = " + std::to_string(sv + 1));
    });
  }
  d.exec_script("BEGIN;\nDROP VIEW IF EXISTS subject_attributes;\n" + subject_view + "DROP VIEW IF EXISTS dataset_attributes;\n" + dataset_view +
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_migrated:53->54', '" + now_utc() + "');\nPRAGMA user_version = 54;\nCOMMIT;");
  const auto ok = d.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after it was migrated from schema 53 to 54");
}
// SCHEMA 54 -> 55, in place: the operator's two cell CHECKs admit '+' (a key of ranks joined by '+', '0+1+2': the whole complex of one manifold, its rows stacked rank by rank; D92 amended
// for the Hodge–Dirac eigenvectors, ONE operator from the eigenvalue line to the intrinsic triangulation's cells 0, 1 and 2). SQLite cannot alter a CHECK: its text in sqlite_master is replaced
// under writable_schema, `schema_version` bumped, the database checked (the new CHECK admits every key the old one did). The two attribute views that name the schema version are made
// again, `user_version` 55. Idempotent: a database whose CHECKs are already the new ones is not rewritten. Written and tested on synthetic databases (test_schema55).
void migrate_54_to_55(Database& d) {
  const std::string model = model_sql();
  const auto cut = [&](const std::string& from, const std::string& to) {
    const auto a = model.find(from);
    const auto b = a == std::string::npos ? a : model.find(to, a);
    if (b == std::string::npos) throw Error(code::not_a_database, "model.sql: no `" + from + "` in the schema, so a database at 54 cannot be migrated");
    return model.substr(a, b - a);
  };
  static const std::string kFromBefore = "from_cell NOT GLOB '*[^0-3,*]*'", kFromAfter = "from_cell NOT GLOB '*[^0-3,*+]*'";
  static const std::string kToBefore = "to_cell NOT GLOB '*[^0-3,*]*'", kToAfter = "to_cell NOT GLOB '*[^0-3,*+]*'";
  const auto tbl = d.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'operator'");
  const std::string sql = tbl ? text(*tbl, "sql") : std::string();
  const bool before = sql.find(kFromBefore) != std::string::npos && sql.find(kToBefore) != std::string::npos;
  const bool after = sql.find(kFromAfter) != std::string::npos && sql.find(kToAfter) != std::string::npos;
  if (!before && !after) throw Error(code::schema_version, "table operator: its cell CHECKs are not the schema 54's; the database cannot be migrated to 55 in place");
  const std::string subject_view = cut("CREATE VIEW subject_attributes AS", ";\n") + ";\n";
  const std::string dataset_view = cut("CREATE VIEW dataset_attributes AS", ";\n") + ";\n";
  if (!after) {
    const std::int64_t sv = integer(*d.one("PRAGMA schema_version"), "schema_version");
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(replace(sql, ?, ?), ?, ?) WHERE type = 'table' AND name = 'operator'", {V(kFromBefore), V(kFromAfter), V(kToBefore), V(kToAfter)});
      d.execute("PRAGMA schema_version = " + std::to_string(sv + 1));
    });
  }
  d.exec_script("BEGIN;\nDROP VIEW IF EXISTS subject_attributes;\n" + subject_view + "DROP VIEW IF EXISTS dataset_attributes;\n" + dataset_view +
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_migrated:54->55', '" + now_utc() + "');\nPRAGMA user_version = 55;\nCOMMIT;");
  const auto ok = d.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after it was migrated from schema 54 to 55");
}
// SCHEMA 55 -> 56, in place: THE TEMPLATE HOLDS THE COHORT (spec 2026-11-24-template-holds-the-cohort). The subject's CHECKs become `kind IN ('subject', 'template',
// 'registration')` and `template_id IS NULL` (the column stays: SQLite cannot drop a column a foreign key and views name); the catalogue's `template_reference` table is
// dropped; the attribute views are made again, `user_version` 56. Only the schema text changes: a database whose rows still name a template (`template_id`) or hold a
// subject of kind `group` is REFUSED here, since moving those rows changes the package and its catalogue entry: `nxr-neuro migrate` does that (and then this step).
// Idempotent. Tested on synthetic databases (test_schema56).
void migrate_55_to_56(Database& d) {
  const std::string model = model_sql();
  const auto cut = [&](const std::string& from, const std::string& to) {
    const auto a = model.find(from);
    const auto b = a == std::string::npos ? a : model.find(to, a);
    if (b == std::string::npos) throw Error(code::not_a_database, "model.sql: no `" + from + "` in the schema, so a database at 55 cannot be migrated");
    return model.substr(a, b - a);
  };
  if (const auto n = d.one("SELECT count(*) AS n FROM subject WHERE template_id IS NOT NULL OR kind = 'group'"); n && integer(*n, "n") > 0)
    throw Error(code::schema_version, "this database names a template from a subject or holds a group package (schema 55): bring its dataset to 56 with `nxr-neuro migrate --dataset <folder>`, "
                                      "which moves cohort results into the dataset's template and drops each subject's link to it");
  static const std::string kKindBefore = "kind IN ('subject', 'template', 'group', 'registration')", kKindAfter = "kind IN ('subject', 'template', 'registration')";
  static const std::string kTemplateBefore = "CHECK (template_id IS NULL OR kind = 'subject')", kTemplateAfter = "CHECK (template_id IS NULL)";
  const auto tbl = d.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'subject'");
  const std::string sql = tbl ? text(*tbl, "sql") : std::string();
  const bool before = sql.find(kKindBefore) != std::string::npos && sql.find(kTemplateBefore) != std::string::npos;
  const bool after = sql.find(kKindAfter) != std::string::npos && sql.find(kTemplateAfter) != std::string::npos;
  if (!before && !after) throw Error(code::schema_version, "table subject: its CHECKs are not the schema 55's; the database cannot be migrated to 56 in place");
  const std::string subject_view = cut("CREATE VIEW subject_attributes AS", ";\n") + ";\n";
  const std::string dataset_view = cut("CREATE VIEW dataset_attributes AS", ";\n") + ";\n";
  if (!after) {
    const std::int64_t sv = integer(*d.one("PRAGMA schema_version"), "schema_version");
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(replace(sql, ?, ?), ?, ?) WHERE type = 'table' AND name = 'subject'", {V(kKindBefore), V(kKindAfter), V(kTemplateBefore), V(kTemplateAfter)});
      d.execute("PRAGMA schema_version = " + std::to_string(sv + 1));
    });
  }
  d.exec_script("BEGIN;\nDROP TABLE IF EXISTS template_reference;\nDROP VIEW IF EXISTS subject_attributes;\n" + subject_view + "DROP VIEW IF EXISTS dataset_attributes;\n" + dataset_view +
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_migrated:55->56', '" + now_utc() + "');\nPRAGMA user_version = 56;\nCOMMIT;");
  const auto ok = d.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after it was migrated from schema 55 to 56");
}
}  // namespace

namespace detail {
#ifdef __EMSCRIPTEN__
namespace {
// THE READER VFS (reader_vfs.h): the default VFS, its main-database files wrapped so that the shared lock waits while the rollback journal beside them is hot, and the reserved lock reads as
// held while one exists. Everything else is the default VFS's.
sqlite3_vfs* base_vfs() { return sqlite3_vfs_find(nullptr); }
struct ReaderFile {
  sqlite3_file file;                 // pMethods: the reader's methods
  sqlite3_file* real;                // the default VFS's file, in the memory after this struct
  char journal[1024];                // `<database>-journal` (empty: not a main database)
};
sqlite3_file* real_of(sqlite3_file* f) { return reinterpret_cast<ReaderFile*>(f)->real; }
// the journal exists, and is hot when its first byte is not zero (SQLite's own test: a header is written by a writer that committed or spilled, never by one that has only begun)
bool journal_exists(const ReaderFile* f) { if (!f->journal[0]) return false; FILE* j = std::fopen(f->journal, "rb"); if (!j) return false; std::fclose(j); return true; }
bool journal_hot(const ReaderFile* f) {
  if (!f->journal[0]) return false;
  FILE* j = std::fopen(f->journal, "rb");
  if (!j) return false;
  const int c = std::fgetc(j);
  std::fclose(j);
  return c != EOF && c != 0;
}
int x_close(sqlite3_file* f) { sqlite3_file* r = real_of(f); const int rc = r->pMethods ? r->pMethods->xClose(r) : SQLITE_OK; f->pMethods = nullptr; return rc; }
int x_read(sqlite3_file* f, void* b, int n, sqlite3_int64 o) { sqlite3_file* r = real_of(f); return r->pMethods->xRead(r, b, n, o); }
int x_write(sqlite3_file* f, const void* b, int n, sqlite3_int64 o) { sqlite3_file* r = real_of(f); return r->pMethods->xWrite(r, b, n, o); }
int x_truncate(sqlite3_file* f, sqlite3_int64 n) { sqlite3_file* r = real_of(f); return r->pMethods->xTruncate(r, n); }
int x_sync(sqlite3_file* f, int flags) { sqlite3_file* r = real_of(f); return r->pMethods->xSync(r, flags); }
int x_size(sqlite3_file* f, sqlite3_int64* n) { sqlite3_file* r = real_of(f); return r->pMethods->xFileSize(r, n); }
int x_lock(sqlite3_file* f, int level) {
  if (level == SQLITE_LOCK_SHARED && journal_hot(reinterpret_cast<ReaderFile*>(f))) return SQLITE_BUSY;   // a writer is committing: wait (the busy timeout), as a native reader does
  sqlite3_file* r = real_of(f);
  return r->pMethods->xLock(r, level);
}
int x_unlock(sqlite3_file* f, int level) { sqlite3_file* r = real_of(f); return r->pMethods->xUnlock(r, level); }
int x_reserved(sqlite3_file* f, int* out) {
  sqlite3_file* r = real_of(f);
  const int rc = r->pMethods->xCheckReservedLock(r, out);
  if (rc == SQLITE_OK && !*out && journal_exists(reinterpret_cast<ReaderFile*>(f))) *out = 1;   // the journal is a writer's: never rolled back by a reader
  return rc;
}
int x_control(sqlite3_file* f, int op, void* arg) { sqlite3_file* r = real_of(f); return r->pMethods->xFileControl(r, op, arg); }
int x_sector(sqlite3_file* f) { sqlite3_file* r = real_of(f); return r->pMethods->xSectorSize(r); }
int x_device(sqlite3_file* f) { sqlite3_file* r = real_of(f); return r->pMethods->xDeviceCharacteristics(r); }
const sqlite3_io_methods kReaderMethods = {1, x_close, x_read, x_write, x_truncate, x_sync, x_size, x_lock, x_unlock, x_reserved, x_control, x_sector, x_device,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
int v_open(sqlite3_vfs*, sqlite3_filename name, sqlite3_file* f, int flags, int* out_flags) {
  ReaderFile* rf = reinterpret_cast<ReaderFile*>(f);
  rf->file.pMethods = nullptr;
  rf->real = reinterpret_cast<sqlite3_file*>(rf + 1);
  rf->journal[0] = 0;
  const int rc = base_vfs()->xOpen(base_vfs(), name, rf->real, flags, out_flags);
  if (rc != SQLITE_OK) return rc;
  if ((flags & SQLITE_OPEN_MAIN_DB) && name) std::snprintf(rf->journal, sizeof rf->journal, "%s-journal", name);
  rf->file.pMethods = &kReaderMethods;
  return SQLITE_OK;
}
int v_delete(sqlite3_vfs*, const char* n, int sync) { return base_vfs()->xDelete(base_vfs(), n, sync); }
int v_access(sqlite3_vfs*, const char* n, int flags, int* out) { return base_vfs()->xAccess(base_vfs(), n, flags, out); }
int v_full(sqlite3_vfs*, const char* n, int size, char* out) { return base_vfs()->xFullPathname(base_vfs(), n, size, out); }
void* v_dlopen(sqlite3_vfs*, const char* n) { return base_vfs()->xDlOpen ? base_vfs()->xDlOpen(base_vfs(), n) : nullptr; }
void v_dlerror(sqlite3_vfs*, int n, char* m) { if (base_vfs()->xDlError) base_vfs()->xDlError(base_vfs(), n, m); }
void (*v_dlsym(sqlite3_vfs*, void* h, const char* s))(void) { return base_vfs()->xDlSym ? base_vfs()->xDlSym(base_vfs(), h, s) : nullptr; }
void v_dlclose(sqlite3_vfs*, void* h) { if (base_vfs()->xDlClose) base_vfs()->xDlClose(base_vfs(), h); }
int v_random(sqlite3_vfs*, int n, char* out) { return base_vfs()->xRandomness(base_vfs(), n, out); }
int v_sleep(sqlite3_vfs*, int us) { return base_vfs()->xSleep(base_vfs(), us); }
int v_time(sqlite3_vfs*, double* t) { return base_vfs()->xCurrentTime(base_vfs(), t); }
int v_error(sqlite3_vfs*, int n, char* m) { return base_vfs()->xGetLastError ? base_vfs()->xGetLastError(base_vfs(), n, m) : 0; }
int v_time64(sqlite3_vfs*, sqlite3_int64* t) {
  if (base_vfs()->iVersion >= 2 && base_vfs()->xCurrentTimeInt64) return base_vfs()->xCurrentTimeInt64(base_vfs(), t);
  double d = 0;
  const int rc = base_vfs()->xCurrentTime(base_vfs(), &d);
  *t = static_cast<sqlite3_int64>(d * 86400000.0);
  return rc;
}
}  // namespace
const char* reader_vfs() {
  static sqlite3_vfs vfs = [] {
    sqlite3_vfs* b = base_vfs();
    sqlite3_vfs v{};
    v.iVersion = 2;
    v.szOsFile = static_cast<int>(sizeof(ReaderFile)) + b->szOsFile;
    v.mxPathname = b->mxPathname;
    v.zName = "nxr-reader";
    v.xOpen = v_open; v.xDelete = v_delete; v.xAccess = v_access; v.xFullPathname = v_full;
    v.xDlOpen = v_dlopen; v.xDlError = v_dlerror; v.xDlSym = v_dlsym; v.xDlClose = v_dlclose;
    v.xRandomness = v_random; v.xSleep = v_sleep; v.xCurrentTime = v_time; v.xGetLastError = v_error; v.xCurrentTimeInt64 = v_time64;
    return v;
  }();
  static const bool registered = sqlite3_vfs_register(&vfs, 0) == SQLITE_OK;   // SQLite keeps the pointer: the static, once
  return registered ? "nxr-reader" : nullptr;
}
#else
const char* reader_vfs() { return nullptr; }
#endif
}  // namespace detail

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
    if (sqlite3_open_v2(p.c_str(), &h, flags, read_only ? detail::reader_vfs() : nullptr) != SQLITE_OK) {   // a read-only open writes nothing (reader_vfs.h)
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
                         "INSERT INTO meta (key, value) VALUES ('store_format', '" + std::string(kStoreFormat51) + "');\n"
                         // a dataset made now stores its selections in the member form (selection_store.h); one made before has the key only after `migrate_selection_storage`
                         "INSERT INTO meta (key, value) VALUES ('selection_storage', '" + std::string(kSelectionStorage) + "');\n"
                         // and its measurements in the one table for values over members (schema 53); one made before has the key only after `migrate_measurement_storage`
                         "INSERT INTO meta (key, value) VALUES ('measurement_storage', '" + std::string(kMeasurementStorage) + "');\nCOMMIT;");
      d.close();            // on exFAT the connection that grew the file reads it as read-only: reconnect
      d.db_ = connect();
    }
    const std::int64_t v = integer(*d.one("PRAGMA user_version"), "user_version");
    const int want = schema_version();
    if (v >= 50 && v <= 55 && want == 56) {
      // a database at 50 to 55 is a database at 56 without the additions: opened to write, it is migrated in place (step by step; 55 -> 56 refuses rows that name a template,
      // which `nxr-neuro migrate` moves first); opened to read, it is read as it is (the query reads the columns it needs by asking the database whether it has them)
      if (!read_only) {
        if (v == 50) migrate_50_to_51(d);
        if (v <= 51) migrate_51_to_52(d);
        if (v <= 52) migrate_52_to_53(d);
        if (v <= 53) migrate_53_to_54(d);
        if (v <= 54) migrate_54_to_55(d);
        migrate_55_to_56(d);
      }
    } else if (v != want) {
      const std::string how =
          v == 49 ? "run `nxr migrate --dataset <dataset folder>` (49 to 50; this build then migrates 50 to 56 when it opens the dataset to write; it keeps the old database and the arrays; add `--dry-run` to see what it will do first)"
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
  if (sqlite3_open_v2(p.c_str(), &h, flags, read_only ? detail::reader_vfs() : nullptr) != SQLITE_OK) {   // a read-only open writes nothing (reader_vfs.h)
    const std::string m = h ? sqlite3_errmsg(h) : "cannot open";
    if (h) sqlite3_close(h);
    throw Error(code::sqlite, p + ": " + m);
  }
  sqlite3_busy_timeout(h, 5000);
  exec(h, "PRAGMA foreign_keys = ON");
  d.db_ = h;
  return d;
}

void Database::tx_begin() {
  sqlite3* db = H(db_);
  if (depth_ == 0) exec(db, "BEGIN");
  else exec(db, "SAVEPOINT sp" + std::to_string(depth_));
  ++depth_;
}

void Database::tx_rollback() noexcept {
  sqlite3* db = H(db_);
  --depth_;
  try {
    if (depth_ == 0) exec(db, "ROLLBACK");
    else { exec(db, "ROLLBACK TO sp" + std::to_string(depth_)); exec(db, "RELEASE sp" + std::to_string(depth_)); }
  } catch (...) {}
}

void Database::tx_commit() {
  sqlite3* db = H(db_);
  --depth_;
  if (depth_ == 0) {
    try {
      exec(db, "COMMIT");
    } catch (const Error& e) {
      // a DEFERRED key fails at COMMIT with no row named: name it before rolling back
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

void Database::tx(const std::function<void()>& body) {
  tx_begin();
  try {
    body();
  } catch (...) {
    tx_rollback();
    throw;
  }
  tx_commit();
}

namespace {
// A cached statement in use: reset (and its bindings cleared) when the call ends, however it ends, so it holds no read lock and the next use starts clean.
struct Use {
  sqlite3_stmt* st;
  explicit Use(void* s) : st(static_cast<sqlite3_stmt*>(s)) {}
  ~Use() { sqlite3_reset(st); sqlite3_clear_bindings(st); }
};
}  // namespace

// The prepared statement for `sql`: prepared on first use and kept. SQLite re-prepares it itself when the schema changes. The cache is bounded (statements built with a
// variable number of placeholders would otherwise grow it without end): past 512 entries it starts again.
void* Database::prepared(const std::string& sql) const {
  auto it = stmts_.find(sql);
  if (it != stmts_.end()) return it->second;
  sqlite3* db = H(db_);
  if (stmts_.size() >= 512) { for (auto& kv : stmts_) sqlite3_finalize(static_cast<sqlite3_stmt*>(kv.second)); stmts_.clear(); }
  sqlite3_stmt* st = nullptr;
  if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) fail(db, sql);
  if (!st) throw Error(code::sqlite, "empty statement");
  stmts_.emplace(sql, st);
  return st;
}

std::vector<Row> Database::all(const std::string& sql, const std::vector<Value>& params) const {
  sqlite3* db = H(db_);
  Use use(prepared(sql));
  sqlite3_stmt* st = use.st;
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
  Use use(prepared(sql));
  sqlite3_stmt* st = use.st;
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
  Use use(prepared(sql));
  sqlite3_stmt* st = use.st;
  bind(st, params);
  int rc;
  while ((rc = sqlite3_step(st)) == SQLITE_ROW) {}
  std::string err;
  if (rc != SQLITE_DONE) err = sqlite3_errmsg(db);
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
