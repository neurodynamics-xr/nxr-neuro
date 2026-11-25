// `migrate template-links` (format 56: THE TEMPLATE HOLDS THE COHORT) on a synthetic catalogue dataset put back to the format-55 form: a subject package whose row names
// its template (`template_id`), which carries the template's `subject` row as an anchor and names the template in its record, and a catalogue whose copy of the subject
// row names it too. The dry run reports and changes nothing; the migration drops every link, brings each database to 56, registers the packages again (content hash
// recomputed) and rebuilds the index; a second run changes nothing. Synthetic only.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/catalogue.h"
#include "nxr/io/dataset_index.h"
#include "nxr/io/migrate.h"
#include "nxr/io/subject_package.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
// the subject table's CHECKs and the version as schema 55 had them (the rows untouched)
void to_55(Database& d) {
  d.with_writable_schema([&] {
    d.execute("UPDATE sqlite_master SET sql = replace(replace(sql, ?, ?), ?, ?) WHERE type = 'table' AND name = 'subject'",
              {V("kind IN ('subject', 'template', 'registration')"), V("kind IN ('subject', 'template', 'group', 'registration')"), V("CHECK (template_id IS NULL)"),
               V("CHECK (template_id IS NULL OR kind = 'subject')")});
    d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
  });
  d.execute("PRAGMA user_version = 55");
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_template_links_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const fs::path ds = root / "cohort";
  Catalogue cat = create_catalogue(ds, CreateCatalogueOptions{.name = "cohort"});
  std::string tid, sid;
  {
    CreatePackageOptions po = cat.package_options();
    po.subject.kind = "template";
    DatasetWriter w = create_package(ds / "T.nxr.zarr", "T", po);
    SubjectWriter t = SubjectWriter::open(w, "T");
    tid = t.id();
    t.complete();
  }
  {
    DatasetWriter w = create_package(ds / "S1.nxr.zarr", "S1", cat.package_options());
    SubjectWriter s = SubjectWriter::open(w, "S1");
    sid = s.id();
    s.complete();
  }
  RegisterOptions ro;
  ro.registered_by = "test";
  CHECK_EQ(register_packages(ds, {ds / "T.nxr.zarr", ds / "S1.nxr.zarr"}, ro).size(), std::size_t(2));

  // put back to 55: the subject names its template, carries its row as an anchor, names it in its record; the catalogue's copy names it too
  {
    Database p = Database::open_plain(ds / "S1.nxr.zarr" / kPackageDatabaseFile);
    to_55(p);
    p.execute("PRAGMA foreign_keys = OFF");
    p.execute("INSERT INTO subject (id, dataset_id, name, path, kind) SELECT ?, dataset_id, 'T', 'T.nxr.zarr', 'template' FROM subject WHERE id = ?", {V(tid), V(sid)});
    p.execute("INSERT INTO anchor_row (table_name, row_id, from_subject_id) VALUES ('subject', ?, ?)", {V(tid), V(tid)});
    p.execute("UPDATE subject SET template_id = ? WHERE id = ?", {V(tid), V(sid)});
    PackageRecord rec = read_package_record(p);
    rec.template_reference = Json{{"name", "cohort-default"}, {"edition", "full"}, {"version", "local"}, {"content_hash", "x"}, {"subject_id", tid}};
    write_package_record(p, rec);
  }
  {
    Database c = Database::open_plain(ds / kCatalogueFile);
    to_55(c);
    c.execute("UPDATE subject SET template_id = ? WHERE id = ?", {V(tid), V(sid)});
  }
  { Database t = Database::open_plain(ds / "T.nxr.zarr" / kPackageDatabaseFile); to_55(t); }

  // the dry run: reported, nothing changed
  TemplateLinksOptions dry{"test", true};
  const Json plan = migrate_template_links(ds, dry);
  bool seen = false;
  for (const auto& s : plan["subjects"]) if (s["name"] == "S1") { seen = true; CHECK(s["template_id_set"] == 1 && s["anchor_rows_from_template"] == 1); }
  CHECK(seen);
  CHECK_EQ(integer(*Database::open_plain(ds / "S1.nxr.zarr" / kPackageDatabaseFile, false, true).one("SELECT count(*) AS n FROM anchor_row"), "n"), 1);

  // the migration
  const Json r = migrate_template_links(ds, TemplateLinksOptions{"test", false});
  CHECK_EQ(r["registered"].size(), std::size_t(2));
  {
    const SubjectPackage p = SubjectPackage::open(ds / "S1.nxr.zarr");
    CHECK(p.verify().ok);
    CHECK(is_null(p.subject_row(), "template_id"));
    CHECK(!p.db().one("SELECT 1 FROM anchor_row").has_value());
    CHECK(!p.db().one("SELECT 1 FROM subject WHERE kind = 'template'").has_value());
    CHECK(p.record().template_reference.is_null());
    CHECK_EQ(integer(*p.db().one("PRAGMA user_version"), "user_version"), 56);
  }
  CHECK_EQ(integer(*Database::open(ds / kCatalogueFile, false, "", true).one("PRAGMA user_version"), "user_version"), 56);
  CHECK_EQ(Catalogue::open(ds).find("S1")->status, std::string("registered"));
  CHECK(index_is_current(ds));
  // a second run changes nothing
  const std::string hash = Catalogue::open(ds).find("S1")->content_hash;
  migrate_template_links(ds, TemplateLinksOptions{"test", false});
  CHECK_EQ(Catalogue::open(ds).find("S1")->content_hash, hash);
  return nxrtest::finish("test_template_links");
}
