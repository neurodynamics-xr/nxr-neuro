#include "nxr/io/testing/probe.h"

#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::io::testing {

Probe::Probe(const SubjectPackage& package) : db_(&package.db()) {}
Probe::Probe(const DatasetWriter& writer) : db_(&writer.db()) {}
Probe::Probe(SubjectWriter& subject) : db_(&subject.db()) {}

static void where_clause(const Row& where, std::string& sql, std::vector<Value>& args) {
  bool first = true;
  for (const auto& [col, v] : where) {
    sql += first ? " WHERE " : " AND ";
    first = false;
    if (is_null(v)) sql += col + " IS NULL";
    else { sql += col + " = ?"; args.push_back(v); }
  }
}

std::int64_t Probe::count(const std::string& table, const Row& where) const {
  std::string sql = "SELECT COUNT(*) AS n FROM " + table;
  std::vector<Value> args;
  where_clause(where, sql, args);
  return integer(*db_->one(sql, args), "n");
}

std::vector<Row> Probe::rows(const std::string& table, const Row& where, const std::string& order_by) const {
  std::string sql = "SELECT * FROM " + table;
  std::vector<Value> args;
  where_clause(where, sql, args);
  return db_->all(sql + " ORDER BY " + (order_by.empty() ? "" : order_by + ", ") + "rowid", args);
}

std::vector<std::string> Probe::integrity() const {
  std::vector<std::string> out;
  for (const Row& r : db_->all("PRAGMA foreign_key_check"))
    out.push_back("foreign key: " + text(r, "table") + " row " + std::to_string(integer(r, "rowid")) + " -> " + text(r, "parent"));
  for (const Row& r : db_->all("PRAGMA integrity_check"))
    if (text(r, "integrity_check") != "ok") out.push_back("integrity: " + text(r, "integrity_check"));
  return out;
}

void Probe::tamper(const std::string& sql) const { db_->exec_script(sql); }

bool Probe::read_only() const { return db_->read_only(); }

void rewrite_as_schema_50(const std::filesystem::path& file) {
  Database d = Database::open(file);
  d.execute("PRAGMA foreign_keys = OFF");
  d.exec_script("DROP TABLE IF EXISTS key_digest; DROP TABLE IF EXISTS subject_entry; DROP TABLE IF EXISTS template_reference; DROP TABLE IF EXISTS selection_factor; DROP VIEW IF EXISTS statistic_attributes; DROP TABLE IF EXISTS statistic;");
  d.with_writable_schema([&] {
    d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE name = 'subject'", {Value(std::string("'group', 'registration')")), Value(std::string("'group')"))});
    d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'view'", {Value(std::string("'schema_version', 54")), Value(std::string("'schema_version', 50"))});
    d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
  });
  d.execute("DELETE FROM meta WHERE key LIKE 'schema_migrated:%'");
  d.execute("PRAGMA user_version = 50");
  d.close();
}

int user_version_of(const std::filesystem::path& file) {
  Database r = Database::open(file, false, "", true);
  return static_cast<int>(integer(*r.one("PRAGMA user_version"), "user_version"));
}

}  // namespace nxr::io::testing
