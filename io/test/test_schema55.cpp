// Schema 55 (D92 amended): an operator's cell key may join ranks with '+' ('0+1+2': every cell of those ranks of ONE manifold, its rows stacked rank by rank), the form of the Hodge-Dirac
// eigenvectors (one operator from the eigenvalue line to the intrinsic triangulation's cells 0, 1 and 2). The only change from 54 is the two cell CHECKs on `operator`; a database at 54 is
// migrated in place when opened to write (`migrate_54_to_55`: the CHECK text replaced under writable_schema, every row kept, idempotent) and read as it is when opened to read. Synthetic only.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
bool inserts(Database& d, const std::string& sid, const std::string& from, const std::string& to, const std::string& to_cell, const std::string& name) {
  try {
    d.execute("INSERT INTO operator (id, subject_id, name, kind, from_manifold_id, to_manifold_id, from_cell, to_cell, layout, method, n_rows, n_cols) VALUES (?, ?, ?, 'test', ?, ?, '0', ?, 'analytic', 'test', 1, 1)",
              {V(name), V(sid), V(name), V(from), V(to), V(to_cell)});
    return true;
  } catch (const Error&) { return false; }
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_schema55_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  CHECK_EQ(schema_version(), 55);

  fs::path file;
  std::string sid, mid;
  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sid = sub.id();
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 4;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 4; d.spacing = 1.0; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      mid = sub.manifold(t);
    });
    Database& db = sub.db();
    CHECK_EQ(integer(*db.one("PRAGMA user_version"), "user_version"), 55);
    CHECK(inserts(db, sid, mid, mid, "0+1+2", "whole"));          // the whole complex, stacked
    CHECK(inserts(db, sid, mid, mid, "1,0", "product"));          // a product's key, as before
    CHECK(!inserts(db, sid, mid, mid, "0-1", "bad"));             // anything else refused
    CHECK(!inserts(db, sid, mid, mid, "", "empty"));
    db.execute("DELETE FROM operator WHERE name = 'whole'");
    file = ds.folder() / "dataset.sqlite";
    ds.close();
  }

  // a database at 54: the operator's CHECKs put back to 54's, the version and the migration record put back
  {
    Database d = Database::open(file);
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(sql, '*[^0-3,*+]*', '*[^0-3,*]*') WHERE type = 'table' AND name = 'operator'");
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'view'", {V("'schema_version', 55"), V("'schema_version', 54")});
      d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
    });
    d.execute("DELETE FROM meta WHERE key = 'schema_migrated:54->55'");
    d.execute("PRAGMA user_version = 54");
    d.close();
  }
  {   // opened to read: read as it is; the old CHECK still refuses the stacked key
    Database r = Database::open(file, false, "", true);
    CHECK_EQ(integer(*r.one("PRAGMA user_version"), "user_version"), 54);
    CHECK(text(*r.one("SELECT sql FROM sqlite_master WHERE name = 'operator'"), "sql").find("*[^0-3,*+]*") == std::string::npos);
    const Dataset old = Dataset::open(file.parent_path());
    CHECK_EQ(integer(old.query("SELECT count(*) AS n FROM subject").at(0), "n"), 1);
  }
  {   // opened to write: migrated in place, the rows kept, the stacked key now accepted
    Database d = Database::open(file);
    CHECK_EQ(integer(*d.one("PRAGMA user_version"), "user_version"), 55);
    CHECK(d.one("SELECT 1 FROM meta WHERE key = 'schema_migrated:54->55'").has_value());
    CHECK_EQ(text(*d.one("PRAGMA quick_check"), "quick_check"), std::string("ok"));
    CHECK_EQ(integer(*d.one("SELECT count(*) AS n FROM operator WHERE name = 'product'"), "n"), 1);
    CHECK(inserts(d, sid, mid, mid, "0+1+2", "whole"));
    CHECK(!inserts(d, sid, mid, mid, "0-1", "bad"));
    CHECK(text(*d.one("SELECT attributes FROM subject_attributes LIMIT 1"), "attributes").find("\"schema_version\":55") != std::string::npos);
  }
  {   // a second open changes nothing
    const std::string before = text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:54->55'"), "value");
    CHECK_EQ(text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:54->55'"), "value"), before);
  }
  return nxrtest::finish("test_schema55");
}
