// Schema 54 (docs/specs/2026-11-17-consolidation.md, section 6.1): an array-backed measurement may be of a STATISTIC (`of_statistic_id`, no field): a PSD [channels x frequencies] measured over its
// channel axis is [frequencies x members x components]. The only change from 53 is the CHECK on `selection_measurement`; a database at 53 is migrated in place when opened to write
// (`migrate_53_to_54`: the CHECK text replaced under writable_schema, every row kept, idempotent) and read as it is when opened to read. Synthetic datasets only.
#include <unistd.h>

#include <filesystem>

#include "check.h"
#include "nxr/io.h"
#include "nxr/io/statistic.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
bool inserts(Database& d, const std::string& sel, const std::optional<std::string>& field, const std::optional<std::string>& stat, const std::optional<std::int64_t>& code, const char* measure) {
  try {
    d.execute("INSERT INTO selection_measurement (selection_id, code, of_field_id, of_statistic_id, of_index, measure, array_path, component) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
              {V(sel), code ? V(*code) : Value(), field ? V(*field) : Value(), stat ? V(*stat) : Value(), stat ? V(std::int64_t(0)) : Value(), V(measure), V(std::string("measurements/") + measure)});
    return true;
  } catch (const Error&) { return false; }
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_schema54_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  CHECK_EQ(schema_version(), 56);

  fs::path file;
  std::string fid, sid, sel;
  {
    DatasetWriter ds = create_dataset(root, "demo");
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    std::vector<float> x(100, 1.0f);
    const std::vector<double> psd = {1.5, 2.5, 3.5, 4.5};
    std::string tid;
    sub.compose([&] {
      ManifoldSpec t;
      t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 100;
      t.topology = {"path", 1, 1, std::nullopt, {}};
      GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
      DimensionSpec d; d.name = "time"; d.n_vertices = 100; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
      g.dimensions.push_back(d); t.geometry = g;
      t.cols.set("n_edges", 99);
      tid = sub.manifold(t);
      FieldSpec f;
      f.name = "rec"; f.path = node_path(sub.db(), tid, "rec"); f.kind = "recording"; f.manifold_id = tid; f.data = view(x, {100});
      fid = sub.field(f);
      StatisticSpec s;
      s.name = "rec_psd"; s.measure = "power_density"; s.method = "welch"; s.of_field_id = fid;
      s.axes = Json::array({Json{{"name", "frequency"}, {"size", 4}, {"values", Json::array({0.0, 1.0, 2.0, 3.0})}, {"unit", "Hz"}}});
      s.n_members = 3; s.dof = 6.0; s.unit = "T^2/Hz";
      s.span = Json{{"start_s", 0.0}, {"stop_s", 1.0}};
      s.options = Json::object(); s.producer = Json{{"cache", Json{{"pipeline", "test"}}}};
      s.data = view(psd, {4});
      sid = sub.statistic(s);
      SelectionSpec ps;
      ps.name = "halves"; ps.path = "time/halves"; ps.type = "spans"; ps.manifold_id = tid;
      ps.cols.set("cell", "1").set("unit", "s").set("n_members", 2);
      sel = sub.selection(ps);
    });
    Database& db = sub.db();
    CHECK_EQ(integer(*db.one("PRAGMA user_version"), "user_version"), 56);
    // the model's CHECK: an array of a field, or of a statistic; never of neither, never with a code
    CHECK(inserts(db, sel, fid, std::nullopt, std::nullopt, "a_field"));
    CHECK(inserts(db, sel, std::nullopt, sid, std::nullopt, "a_statistic"));
    CHECK(!inserts(db, sel, std::nullopt, std::nullopt, std::nullopt, "of_nothing"));
    CHECK(!inserts(db, sel, std::nullopt, sid, std::int64_t(0), "with_a_code"));
    file = ds.folder() / "dataset.sqlite";
    ds.close();
  }

  // a database at 53: the CHECK put back to 53's, the statistic's array row (not valid at 53) removed, the version and the migration record put back
  {
    Database d = Database::open(file);
    d.execute("DELETE FROM selection_measurement WHERE measure = 'a_statistic'");
    static const std::string kNew = "CHECK (array_path IS NULL OR (code IS NULL AND (of_field_id IS NOT NULL OR of_statistic_id IS NOT NULL)))", kOld = "CHECK (array_path IS NULL OR (code IS NULL AND of_field_id IS NOT NULL))";
    d.with_writable_schema([&] {
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'selection_measurement'", {V(kNew), V(kOld)});
      d.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'view'", {V("'schema_version', 56"), V("'schema_version', 53")});
      d.execute("PRAGMA schema_version = " + std::to_string(integer(*d.one("PRAGMA schema_version"), "schema_version") + 1));
    });
    d.execute("DELETE FROM meta WHERE key = 'schema_migrated:53->54'");
    d.execute("DELETE FROM meta WHERE key = 'schema_migrated:54->55'");
    d.execute("PRAGMA user_version = 53");
    d.close();
  }
  {   // opened to read: read as it is; the old CHECK still refuses a statistic's array
    Database r = Database::open(file, false, "", true);
    CHECK_EQ(integer(*r.one("PRAGMA user_version"), "user_version"), 53);
    CHECK(text(*r.one("SELECT sql FROM sqlite_master WHERE name = 'selection_measurement'"), "sql").find("OR of_statistic_id IS NOT NULL") == std::string::npos);
    const Dataset old = Dataset::open(file.parent_path());
    CHECK_EQ(integer(old.query("SELECT count(*) AS n FROM subject").at(0), "n"), 1);
  }
  {   // opened to write: migrated in place (through 54 to 55), the rows kept, the statistic's array now accepted
    Database d = Database::open(file);
    CHECK_EQ(integer(*d.one("PRAGMA user_version"), "user_version"), 56);
    CHECK(d.one("SELECT 1 FROM meta WHERE key = 'schema_migrated:53->54'").has_value());
    CHECK_EQ(text(*d.one("PRAGMA quick_check"), "quick_check"), std::string("ok"));
    CHECK_EQ(integer(*d.one("SELECT count(*) AS n FROM selection_measurement WHERE measure = 'a_field'"), "n"), 1);
    CHECK(inserts(d, sel, std::nullopt, sid, std::nullopt, "a_statistic"));
    CHECK(!inserts(d, sel, std::nullopt, std::nullopt, std::nullopt, "of_nothing"));
    CHECK(!inserts(d, sel, std::nullopt, sid, std::int64_t(0), "with_a_code"));
    CHECK(text(*d.one("SELECT attributes FROM subject_attributes LIMIT 1"), "attributes").find("\"schema_version\":56") != std::string::npos);
  }
  {   // a second open changes nothing
    const std::string before = text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:53->54'"), "value");
    CHECK_EQ(text(*Database::open(file).one("SELECT value FROM meta WHERE key = 'schema_migrated:53->54'"), "value"), before);
  }
  // a database at 52 goes through 53 and 54 to 55 in one open
  {
    Database d = Database::open(file);
    d.execute("PRAGMA user_version = 52");   // (its migration to 53 is idempotent: the column and the views are there)
    d.close();
    Database m = Database::open(file);
    CHECK_EQ(integer(*m.one("PRAGMA user_version"), "user_version"), 56);
  }
  return nxrtest::finish("test_schema54");
}
