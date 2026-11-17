#include "nxr/io/testing/probe.h"

#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::io::testing {

Probe::Probe(const SubjectPackage& package) : db_(&package.db()) {}
Probe::Probe(const DatasetWriter& writer) : db_(&writer.db()) {}

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

std::vector<Row> Probe::rows(const std::string& table, const Row& where) const {
  std::string sql = "SELECT * FROM " + table;
  std::vector<Value> args;
  where_clause(where, sql, args);
  return db_->all(sql + " ORDER BY rowid", args);
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

}  // namespace nxr::io::testing
