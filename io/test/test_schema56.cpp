// Schema 56 (spec docs/specs/2026-11-24-template-holds-the-cohort): THE TEMPLATE HOLDS THE COHORT. A subject names no template: `subject.template_id` is retired
// (CHECK template_id IS NULL), the kind `group` is retired, the catalogue's `template_reference` table is dropped. A database at 55 whose rows name no template is
// migrated in place when opened to write (`migrate_55_to_56`: CHECK text replaced under writable_schema, the table dropped, the views made again, idempotent) and read
// as it is when opened to read; one whose rows still name a template, or hold a group subject, is REFUSED there (`nxr-neuro migrate` moves those rows). Synthetic only.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
bool inserts(Database& d, const std::string& dataset_id, const std::string& name, const std::string& kind, const Value& template_id) {
  try {
    d.execute("INSERT INTO subject (id, dataset_id, name, path, kind, template_id) VALUES (?, ?, ?, ?, ?, ?)", {V(name), V(dataset_id), V(name), V(name + ".nxr.zarr"), V(kind), template_id});
    d.execute("DELETE FROM subject WHERE id = ?", {V(name)});
    return true;
  } catch (const Error&) { return false; }
}
void put_back_55(const fs::path& file) {
  Database d = Database::open(file);
  d.with_writable_schema([&] {
    d.execute("UPDATE sqlite_master SET sql = replace(replace(sql, ?, ?), ?, ?) WHERE type = 'table' AND name = 'subject'",
              {V("kind IN ('subject', 'template', 'registration')"), V("kind IN ('subject', 'template', 'group', 'registration')"), V("CHECK (template_id IS NULL)"),
               V("CHECK (template_id IS NULL OR kind = 'subject')")});
    d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'view'", {V("'schema_version', 56"), V("'schema_version', 55")});
    d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
  });
  d.exec_script("CREATE TABLE template_reference (name TEXT NOT NULL, edition TEXT NOT NULL DEFAULT 'full', version TEXT NOT NULL, subject_id TEXT NOT NULL, content_hash TEXT NOT NULL, "
                "location_hint TEXT, PRIMARY KEY (name, edition, version)) STRICT;");
  d.execute("DELETE FROM meta WHERE key = 'schema_migrated:55->56'");
  d.execute("PRAGMA user_version = 55");
  d.close();
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_schema56_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  CHECK_EQ(schema_version(), 56);
  CHECK_EQ(kSchemaVersion, 56);

  fs::path file;
  std::string dataset_id, template_id;
  {
    DatasetWriter ds = create_dataset(root, "demo");
    dataset_id = ds.id();
    SubjectOptions to; to.kind = "template";
    SubjectWriter t = SubjectWriter::create(ds, "T", to);
    t.compose([] {});
    template_id = t.id();
    SubjectWriter s = SubjectWriter::create(ds, "S1");
    s.compose([] {});
    Database& db = ds.db();
    CHECK_EQ(integer(*db.one("PRAGMA user_version"), "user_version"), 56);
    CHECK(!db.one("SELECT 1 FROM subject WHERE template_id IS NOT NULL").has_value());              // a subject made now names no template
    CHECK(!db.one("SELECT 1 FROM sqlite_master WHERE name = 'template_reference'").has_value());
    CHECK(inserts(db, dataset_id, "S2", "subject", Value()));
    CHECK(!inserts(db, dataset_id, "S3", "subject", V(template_id)));                              // the retired column refuses a value
    CHECK(!inserts(db, dataset_id, "G", "group", Value()));                                         // the retired kind is refused
    CHECK(inserts(db, dataset_id, "R", "registration", Value()));
    file = ds.folder() / "dataset.sqlite";
    ds.close();
  }

  // a database at 55 whose rows name no template
  put_back_55(file);
  {   // opened to read: read as it is
    Database r = Database::open(file, false, "", true);
    CHECK_EQ(integer(*r.one("PRAGMA user_version"), "user_version"), 55);
    CHECK(r.one("SELECT 1 FROM sqlite_master WHERE name = 'template_reference'").has_value());
  }
  {   // opened to write: migrated in place, rows kept, the new CHECKs in force
    Database d = Database::open(file);
    CHECK_EQ(integer(*d.one("PRAGMA user_version"), "user_version"), 56);
    CHECK(d.one("SELECT 1 FROM meta WHERE key = 'schema_migrated:55->56'").has_value());
    CHECK(!d.one("SELECT 1 FROM sqlite_master WHERE name = 'template_reference'").has_value());
    CHECK_EQ(text(*d.one("PRAGMA quick_check"), "quick_check"), std::string("ok"));
    CHECK_EQ(integer(*d.one("SELECT count(*) AS n FROM subject"), "n"), 2);
    CHECK(!inserts(d, dataset_id, "G", "group", Value()));
    CHECK(!inserts(d, dataset_id, "S3", "subject", V(template_id)));
    CHECK(text(*d.one("SELECT attributes FROM subject_attributes LIMIT 1"), "attributes").find("\"schema_version\":56") != std::string::npos);
  }
  {   // a second open changes nothing
    const std::string before = text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:55->56'"), "value");
    CHECK_EQ(text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:55->56'"), "value"), before);
  }

  // a database at 55 whose subject still names its template: refused when opened to write, read as it is
  put_back_55(file);
  {
    Database d = Database::open(file, false, "", true);
    (void)d;
  }
  {
    Database d = Database::open_plain(file, false, false);
    d.execute("UPDATE subject SET template_id = ? WHERE name = 'S1'", {V(template_id)});
  }
  bool refused = false;
  try { Database::open(file); } catch (const Error& e) { refused = std::string(e.what()).find("nxr-neuro migrate") != std::string::npos; }
  CHECK(refused);
  {
    Database r = Database::open(file, false, "", true);
    CHECK_EQ(integer(*r.one("PRAGMA user_version"), "user_version"), 55);
  }
  return nxrtest::finish("test_schema56");
}
