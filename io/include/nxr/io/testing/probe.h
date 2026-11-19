// nxr-io — the TEST-ONLY probe (target nxr_io_testing; linked by test executables only: the layering check refuses this header elsewhere).
// It offers what the record API deliberately does not: read-only counts and rows of ANY table for assertions about what was written, the integrity
// checks, and `tamper`, named so that a test that corrupts a database on purpose says so. Tests otherwise use the record API, as pipelines do.
#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "nxr/io/database.h"

namespace nxr::io {
class SubjectPackage;
class DatasetWriter;
class SubjectWriter;
}  // namespace nxr::io

namespace nxr::io::testing {

class Probe {
 public:
  explicit Probe(const Database& db) : db_(&db) {}
  explicit Probe(const SubjectPackage& package);
  explicit Probe(const DatasetWriter& writer);
  explicit Probe(SubjectWriter& subject);

  // Rows of `table` whose columns equal `where` (a monostate value: IS NULL); no filter when empty.
  std::int64_t count(const std::string& table, const Row& where = {}) const;
  // `order_by` (a column, e.g. "ordinal"; default: insertion order) sorts the rows; ties keep insertion order.
  std::vector<Row> rows(const std::string& table, const Row& where = {}, const std::string& order_by = "") const;
  // PRAGMA foreign_key_check and integrity_check: empty when the database is sound, else one line per problem.
  std::vector<std::string> integrity() const;
  // Whether the database was opened read-only (a reader).
  bool read_only() const;
  // Run `sql` against the database to corrupt it on purpose (a test of what a reader or a check does with a damaged database).
  void tamper(const std::string& sql) const;

 private:
  const Database* db_;
};

// Rewrite a schema-54 database file AS SCHEMA 50 on purpose (a fixture for "an older dataset read as it is"): the tables and column of 51 to 54 dropped,
// the subject kind list and the views as the build before had them, the migration records removed, user_version 50.
void rewrite_as_schema_50(const std::filesystem::path& database_file);
// The file's PRAGMA user_version, opened read-only.
int user_version_of(const std::filesystem::path& database_file);

}  // namespace nxr::io::testing
