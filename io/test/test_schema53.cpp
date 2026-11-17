// Schema 53 (docs/specs/2026-11-17-consolidation.md section 6.2): `selection_factor.level`, the two query views, `user_version` 53, meta `measurement_storage`; a database at 52 migrated in place when
// opened to write (additive, idempotent) and read as it is when opened to read; a line's TIME GRID (one ladder row, closed form) with its levels in physical units; a product whose factor is a
// LEVEL of a grid (stored, loaded, refused where it is not allowed); the Frequency line's charts.
#include <unistd.h>

#include <cmath>
#include <filesystem>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/measurement_query.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/task/task.h"
#include "nxr/bst/frequency.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
bool has_object(const Database& d, const char* type, const char* name) { return d.one("SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?", {V(type), V(name)}).has_value(); }
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_schema53_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  CHECK_EQ(schema_version(), 53);
  CHECK_EQ(kSchemaVersion, 53);

  fs::path file;
  std::string line_id;
  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    Database& db = sub.db();
    // a fresh database is the model's
    CHECK_EQ(integer(*db.one("PRAGMA user_version"), "user_version"), 53);
    CHECK(has_column(db, "selection_factor", "level") && has_object(db, "view", "member_value") && has_object(db, "view", "product_factor_level"));
    CHECK_EQ(text(*db.one("SELECT value FROM meta WHERE key = 'measurement_storage'"), "value"), std::string(kMeasurementStorage));
    CHECK(text(*db.one("SELECT attributes FROM subject_attributes LIMIT 1"), "attributes").find("\"schema_version\":53") != std::string::npos);

    // A LINE'S TIME GRID: one ladder row, closed form; found and reused; another rule on the line refused
    std::string grid_id, elements_id, product_id;
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 100;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 100; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      t.cols.set("n_edges", 99);
      line_id = sub.manifold(t);
      grid_id = selection::save_time_grid(sub, line_id, selection::TimeGridRule{0.0, 100.0, 100}).id;
    });
    CHECK(!grid_id.empty());
    std::string again;
    sub.compose([&] { again = selection::save_time_grid(sub, line_id, selection::TimeGridRule{0.0, 100.0, 100}).id; });
    CHECK_EQ(again, grid_id);
    // (a refusal is checked outside a composition: an aborted composition of a writer that owns its subject removes the subject)
    CHECK_EQ(code_of([&] { selection::save_time_grid(sub, line_id, selection::TimeGridRule{0.0, 200.0, 100}); }), std::string(code::invalid_argument));
    const auto found = selection::find_time_grid(db, line_id);
    CHECK(found && found->first.id == grid_id && found->second.samples == 100 && found->second.rate == 100.0);
    CHECK(is_null(*db.read("selection", grid_id), "member_of_id") && text(*db.read("selection", grid_id), "type") == "ladder" && selection::is_grid(db, grid_id));
    // its levels in PHYSICAL units: 2^k samples at 100 Hz, k = 0 .. 6
    const auto levels = selection::grid_level_sizes(db, grid_id);
    CHECK(levels.size() == 7 && levels[3].code == 3 && std::abs(levels[3].size - 0.08) < 1e-12 && levels[3].unit == "s");
    const auto near_third = selection::level_of_size(db, grid_id, 0.3, "s");
    CHECK(near_third.code == 5 && std::abs(near_third.size - 0.32) < 1e-12);
    CHECK_EQ(code_of([&] { selection::level_of_size(db, grid_id, 0.3, "mm"); }), std::string(code::invalid_argument));

    // A PRODUCT whose first factor is LEVEL 3 of the time grid (13 tiles) and the second an elements grid (4 channels): mixed radix, first factor fastest
    sub.compose([&] {
      selection::Description e;
      e.type = selection::Type::set;
      e.grid = selection::Grid{selection::Grid::Kind::elements, 4};
      selection::Placement ep;
      ep.name = "channels"; ep.manifold_id = line_id;
      elements_id = selection::save(sub, e, ep).id;
      selection::Description pd;
      pd.type = selection::Type::set;
      pd.factors.resize(2);
      pd.radices = {13, 4};
      selection::Placement pp;
      pp.name = "tiles x channels"; pp.manifold_id = line_id;
      pp.factor_ids = {grid_id, elements_id};
      pp.factor_levels = {3, -1};
      product_id = selection::save(sub, pd, pp).id;
    });
    CHECK((selection::product_factor_levels(db, product_id) == std::vector<int>{3, -1}));
    CHECK((selection::product_radices(db, product_id) == std::vector<std::int64_t>{13, 4}) && selection::is_grid(db, product_id));
    const auto loaded = selection::load(db, sub.store(), product_id);
    CHECK((loaded.description.factor_levels == std::vector<int>{3, -1}) && loaded.description.factors.size() == 2 && loaded.description.radices[0] == 13);
    // the physical size of a product's factor level, as the query reads it
    const auto sz = query::level_size(db, product_id, 0);
    CHECK(sz && std::abs(sz->size - 0.08) < 1e-12 && sz->unit == "s");
    // refusals: a grid factor without a level, a level on a factor that is no grid, a level without its radix
    auto make = [&](std::vector<std::string> ids, std::vector<int> lv, std::vector<std::int64_t> rad) {
      return code_of([&] {
        selection::Description pd; pd.type = selection::Type::set; pd.factors.resize(2); pd.radices = rad;
        selection::Placement pp; pp.name = "refused"; pp.manifold_id = line_id; pp.factor_ids = ids; pp.factor_levels = lv;
        selection::save(sub, pd, pp);
      });
    };
    CHECK_EQ(make({grid_id, elements_id}, {}, {13, 4}), std::string(code::invalid_argument));
    CHECK_EQ(make({grid_id, elements_id}, {3, 1}, {13, 4}), std::string(code::invalid_argument));
    CHECK_EQ(make({grid_id, elements_id}, {3, -1}, {}), std::string(code::invalid_argument));
    file = ds.folder() / "dataset.sqlite";
    ds.close();
  }

  // a database at 52: the column and the views taken away, the version put back
  {
    Database d = Database::open(file);
    d.execute("PRAGMA foreign_keys = OFF");
    d.exec_script("DROP VIEW member_value; DROP VIEW product_factor_level; ALTER TABLE selection_factor DROP COLUMN level;");
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'view'", {V("'schema_version', 53"), V("'schema_version', 52")});
      d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
    });
    d.execute("DELETE FROM meta WHERE key = 'schema_migrated:52->53'");
    d.execute("PRAGMA user_version = 52");
    d.close();
  }
  {   // opened to read: read as it is; Dataset reads it
    Database r = Database::open(file, false, "", true);
    CHECK_EQ(integer(*r.one("PRAGMA user_version"), "user_version"), 52);
    CHECK(!has_column(r, "selection_factor", "level"));
    const Dataset old = Dataset::open(file.parent_path());
    CHECK_EQ(integer(old.query("SELECT count(*) AS n FROM subject").at(0), "n"), 1);
  }
  {   // opened to write: migrated in place
    Database d = Database::open(file);
    CHECK_EQ(integer(*d.one("PRAGMA user_version"), "user_version"), 53);
    CHECK(has_column(d, "selection_factor", "level") && has_object(d, "view", "member_value") && has_object(d, "view", "product_factor_level"));
    CHECK(d.one("SELECT 1 FROM meta WHERE key = 'schema_migrated:52->53'").has_value());
    CHECK_EQ(text(*d.one("PRAGMA quick_check"), "quick_check"), std::string("ok"));
    CHECK(text(*d.one("SELECT attributes FROM subject_attributes LIMIT 1"), "attributes").find("\"schema_version\":53") != std::string::npos);
  }
  {   // a second open changes nothing
    const std::string before = text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:52->53'"), "value");
    CHECK_EQ(text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:52->53'"), "value"), before);
  }
  // below 50 is refused as before
  {
    Database d = Database::open(file);
    d.execute("PRAGMA user_version = 49");
    d.close();
    CHECK_EQ(code_of([&] { Database::open(file); }), std::string(code::schema_version));
  }
  return nxrtest::finish("test_schema53");
}
