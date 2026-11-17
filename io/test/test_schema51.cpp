// Schema 51 (docs/specs/2026-11-16-subject-centred-datasets.md section 8, "the version step"): the subject kind `registration` and the tables key_digest, subject_entry
// and template_reference. A database at 50 (made here by putting schema 50 back: the older kind list, no new tables, the older views, user_version 50) is
//   - READ as it is when opened to read (it stays at 50: nothing is written),
//   - MIGRATED IN PLACE when opened to write: user_version 51, the three tables, the kind allowed, the two attribute views made again, a note in `meta`,
//     every older row untouched, the database passing its check, and a second open changing nothing;
// and a database made now is the same database, table for table and view for view, as the migrated one.
#include <unistd.h>

#include <filesystem>
#include <iostream>

#include "check.h"
#include "nxr/io.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
std::string code_of(const std::function<void()>& f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::map<std::string, std::string> schema_of(const Database& db) {
  std::map<std::string, std::string> out;
  for (const auto& r : db.all("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'")) out[text(r, "name")] = text(r, "sql");
  return out;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_schema51_" + std::to_string(::getpid()));
  fs::remove_all(root);
  CHECK_EQ(schema_version(), 53);
  fs::path file;
  std::map<std::string, std::string> fresh;
  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S");
    sub.compose([] {});
    fresh = schema_of(ds.db());
    CHECK(fresh.count("key_digest") && fresh.count("subject_entry") && fresh.count("template_reference"));
    CHECK(fresh.at("subject").find("'registration'") != std::string::npos);
    CHECK(fresh.at("subject_attributes").find("'schema_version', 53") != std::string::npos);
    file = ds.folder() / "dataset.sqlite";
    ds.close();
  }
  // schema 50, as the previous build made it
  {
    Database d = Database::open(file);
    d.execute("PRAGMA foreign_keys = OFF");
    d.exec_script("DROP TABLE key_digest; DROP TABLE subject_entry; DROP TABLE template_reference;");
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE name = 'subject'", {V("'group', 'registration')"), V("'group')")});
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'view'", {V("'schema_version', 53"), V("'schema_version', 50")});
      d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
    });
    d.execute("DELETE FROM meta WHERE key = 'schema_migrated:50->51'");
    d.execute("PRAGMA user_version = 50");
    d.close();
  }
  // opened to read: read as it is, nothing written
  {
    Database r = Database::open(file, false, "", true);
    CHECK_EQ(integer(*r.one("PRAGMA user_version"), "user_version"), 50);
    CHECK(r.one("SELECT name FROM subject WHERE name = 'S'").has_value());
    CHECK_EQ(integer(*Database::open(file, false, "", true).one("PRAGMA user_version"), "user_version"), 50);
    const Dataset old = Dataset::open(file.parent_path());
    CHECK_EQ(integer(old.query("SELECT count(*) AS n FROM subject").at(0), "n"), 1);
  }
  // opened to write: migrated in place
  {
    Database d = Database::open(file);
    CHECK_EQ(integer(*d.one("PRAGMA user_version"), "user_version"), 53);
    CHECK(d.one("SELECT 1 FROM subject WHERE name = 'S'").has_value());
    const auto now = schema_of(d);
    for (const auto& [name, sql] : fresh) {
      CHECK(now.count(name));
      if (now.count(name) && now.at(name) != sql) { std::cout << "  differs from a database made now: " << name << "\n"; CHECK(false); }
    }
    for (const auto& [name, sql] : now) if (!fresh.count(name)) { std::cout << "  only in the migrated one: " << name << "\n"; CHECK(false); }
    CHECK(d.one("SELECT 1 FROM meta WHERE key = 'schema_migrated:50->51'").has_value());
    CHECK_EQ(text(*d.one("PRAGMA quick_check"), "quick_check"), std::string("ok"));
    // the kind is allowed: a registration subject row can be written
    const auto ds = d.one("SELECT id FROM dataset");
    d.insert("subject", {{"id", uuid7()}, {"dataset_id", text(*ds, "id")}, {"name", "R"}, {"path", "R.nxr.zarr"}, {"kind", "registration"}});
    CHECK(d.one("SELECT 1 FROM subject WHERE kind = 'registration'").has_value());
    d.execute("DELETE FROM subject WHERE kind = 'registration'");
  }
  {
    const std::string before = text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:50->51'"), "value");
    Database d = Database::open(file);                                    // a second open changes nothing
    CHECK_EQ(text(*d.one("SELECT value FROM meta WHERE key = 'schema_migrated:50->51'"), "value"), before);
  }
  // a dataset below 50 is refused as before, naming the version
  {
    Database d = Database::open(file);
    d.execute("PRAGMA user_version = 49");
    d.close();
    CHECK_EQ(code_of([&] { Database::open(file); }), std::string(code::schema_version));
  }
  fs::remove_all(root);
  return nxrtest::finish("test_schema51");
}
