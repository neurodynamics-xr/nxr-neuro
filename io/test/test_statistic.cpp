// The statistic node (statistic.h, docs/specs/2026-11-13-statistics.md) on synthetic subjects: written and read back (row, axes, values), its zarr.json
// the row (class statistic), listed by its field; refusals (an unknown measure or method, axes that are not the values', no parent); removed. And a
// dataset made BEFORE the table (simulated: the table, its view and sync_job's widening taken out): it has no statistics, recovers, and its first
// statistic adds the table in place (recorded in meta, user_version unchanged, the database's check ok).
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "check.h"
#include "nxr/bst/timefreq.h"

extern "C" {
#include "matio.h"
}
#include "nxr/io.h"
#include "nxr/io/measures.h"
#include "nxr/io/statistic.h"

using namespace nxr::io;
namespace fs = std::filesystem;

namespace {
template <class F> std::string code_of(F f) { try { f(); } catch (const Error& e) { return e.code(); } return ""; }
std::string read_text(const fs::path& p) { std::ifstream in(p); std::stringstream s; s << in.rdbuf(); return s.str(); }
// a subject with a line and a field on it
std::string make_subject(DatasetWriter& ds, const std::string& name) {
  SubjectWriter sub = SubjectWriter::create(ds, name);
  std::string fid;
  std::vector<float> x(100, 1.0f);
  sub.compose([&] {
    ManifoldSpec t;
    t.name = "line"; t.path = "time/line"; t.type = "line"; t.n_vertices = 100;
    t.topology = {"path", 1, 1, std::nullopt, {}};
    GeometrySpec g; g.form = "closed_form"; g.cs_key = "clock";
    DimensionSpec d; d.name = "time"; d.n_vertices = 100; d.spacing = 0.01; d.origin = 0.0; d.unit = "s";
    g.dimensions.push_back(d); t.geometry = g;
    t.cols.set("n_edges", 99);
    const std::string tid = sub.manifold(t);
    FieldSpec f;
    f.name = "rec"; f.path = node_path(sub.db(), tid, "rec"); f.kind = "recording"; f.manifold_id = tid; f.data = view(x, {100});
    fid = sub.field(f);
  });
  return fid;
}
StatisticSpec spec_of(const std::string& fid, const std::vector<double>& v) {
  StatisticSpec s;
  s.name = "rec_psd"; s.measure = "power_density"; s.method = "welch"; s.of_field_id = fid;
  s.axes = Json::array({Json{{"name", "frequency"}, {"size", static_cast<std::int64_t>(v.size())}, {"values", Json::array({0.0, 25.0, 50.0})}, {"unit", "Hz"}}});
  s.n_members = 7; s.dof = 14.0; s.unit = "T^2/Hz";
  s.span = Json{{"start_s", 0.0}, {"stop_s", 1.0}};
  s.options = Json{{"window", "hamming"}, {"segment_samples", 4}};
  s.producer = Json{{"cache", Json{{"pipeline", "test"}}}};
  s.data = view(v, {static_cast<std::int64_t>(v.size())});
  return s;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_statistic_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const std::vector<double> v = {1.5, 2.5, 3.5};

  // 1. A NEW DATASET: the table is the model's
  {
    DatasetWriter ds = create_dataset(root, "new");
    const std::string fid = make_subject(ds, "S1");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    CHECK(has_table(sub.db(), "statistic"));
    CHECK(has_column(sub.db(), "selection_measurement", "of_statistic_id"));   // the measurement columns are the model's too
    CHECK(!ensure_statistics(sub.db()));
    std::string id;
    sub.compose([&] { id = sub.statistic(spec_of(fid, v)); });
    const StoredStatistic st = read_statistic(sub, "rec_psd");
    CHECK(st.values == v && st.shape == std::vector<std::int64_t>{3} && st.axes[0]["name"] == "frequency" && integer(st.row, "n_members") == 7);
    CHECK(text(st.row, "path") == "time/rec_statistics/rec_psd" && st.options["window"] == "hamming" && text(st.row, "of_field_id") == fid);
    const std::string zj = sub.store().get_text(join_key(text(st.row, "path"), "zarr.json")).value_or("");
    CHECK(zj.find("\"class\":\"statistic\"") != std::string::npos && zj.find("\"measure\":\"power_density\"") != std::string::npos);
    CHECK(statistics_of(sub.db(), sub.id(), fid, std::string("power_density")).size() == 1 && statistics_of(sub.db(), sub.id(), fid, std::string("count")).empty());
    // refusals
    auto refused = [&](auto mutate) { StatisticSpec s = spec_of(fid, v); s.name = "x"; mutate(s); return code_of([&] { sub.compose([&] { sub.statistic(s); }); }); };
    CHECK_EQ(refused([](StatisticSpec& s) { s.measure = "loudness"; }), std::string(code::invalid_argument));
    CHECK_EQ(refused([](StatisticSpec& s) { s.method = "guess"; }), std::string(code::invalid_argument));
    CHECK_EQ(refused([](StatisticSpec& s) { s.axes[0]["size"] = 4; }), std::string(code::invalid_argument));
    CHECK_EQ(refused([](StatisticSpec& s) { s.axes[0].erase("values"); }), std::string(code::invalid_argument));
    CHECK_EQ(refused([](StatisticSpec& s) { s.of_field_id.reset(); }), std::string(code::invalid_argument));
    CHECK_EQ(code_of([&] { read_statistic(sub, "nope"); }), std::string(code::no_such_row));
    // MEASUREMENTS OF A STATISTIC'S SLICE (of_statistic_id, of_index): written, read through the lookup, apart from the geometric rows, in the selection's zarr.json
    CHECK(has_column(sub.db(), "selection_measurement", "of_statistic_id"));
    std::string psel;
    sub.compose([&] {
      SelectionSpec ps;
      ps.name = "halves"; ps.path = "time/halves"; ps.type = "spans"; ps.manifold_id = text(*sub.db().one("SELECT manifold_id FROM field WHERE id = ?", {V(fid)}), "manifold_id");
      ps.cols.set("cell", "1").set("unit", "s").set("n_members", 2);
      ps.cols.json("params_json", Json{{"tiling", Json{{"origin", 0.0}, {"width", 0.5}, {"hop", 0.5}, {"count", 2}}}});
      psel = sub.selection(ps);
    });
    write_measurements(sub.db(), Json::array({Json{{"selection_id", psel}, {"code", 0}, {"of_statistic_id", id}, {"of_index", 2}, {"measure", "mean"}, {"value", 7.0}},
                                              Json{{"selection_id", psel}, {"code", 0}, {"measure", "n_elements"}, {"value", 50.0}},
                                              Json{{"selection_id", psel}, {"of_statistic_id", id}, {"of_index", 2}, {"measure", "max"}, {"value", 9.0}}}), std::string("test"));   // the last: the selection's own (zarr.json)
    const Json q = query_measurements(sub.db(), Json{{"selectionId", psel}, {"measure", "mean"}, {"ofStatisticId", id}, {"ofIndex", 2}});
    CHECK(q.size() == 1 && q[0]["value"] == 7.0);
    CHECK(query_measurements(sub.db(), Json{{"selectionId", psel}, {"measure", "mean"}, {"ofFieldId", nullptr}}).empty());   // the geometric rows only
    drain(ds.db(), ds.store());
    const std::string sj = sub.store().get_text("time/halves/zarr.json").value_or("");
    CHECK(sj.find("\"of_statistic_id\":\"" + id + "\"") != std::string::npos && sj.find("\"of_index\":2") != std::string::npos);
    // removed: the row and its array
    const std::string where = text(st.row, "path");
    sub.remove("statistic", id);
    CHECK(statistics_of(sub.db(), sub.id()).empty() && !sub.store().exists_prefix(where));
    CHECK(query_measurements(sub.db(), Json{{"selectionId", psel}, {"measure", "mean"}, {"ofStatisticId", id}}).empty());   // its measurements went with it
  }

  // 2. A DATASET MADE BEFORE THE TABLE (simulated): no statistics, recovers, and the first write adds the table in place
  {
    DatasetWriter ds = create_dataset(root, "old");
    const std::string fid = make_subject(ds, "S1");
    Database& db = ds.db();
    // ...and before the measurement columns: the (empty) table's SQL without them, the key and the selection view as they were
    CHECK(integer(*db.one("SELECT count(*) AS n FROM selection_measurement"), "n") == 0);
    const std::string view = text(*db.one("SELECT sql FROM sqlite_master WHERE name = 'selection_attributes'"), "sql");
    const std::string wide = ", 'of_statistic_id', m.of_statistic_id, 'of_index', m.of_index)";
    CHECK(view.find(wide) != std::string::npos);
    db.exec_script("DROP VIEW selection_attributes; DROP INDEX selection_measurement_key; DROP INDEX selection_measurement_statistic; DROP VIEW statistic_attributes; DROP TABLE statistic;");
    const std::int64_t sv = integer(*db.one("PRAGMA schema_version"), "schema_version");
    db.with_writable_schema([&] {
      db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'sync_job'", {V(std::string("'field', 'operator', 'statistic'))")), V(std::string("'field', 'operator'))"))});
      db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, '') WHERE type = 'table' AND name = 'selection_measurement'", {V(std::string("  of_statistic_id TEXT REFERENCES statistic (id) ON DELETE CASCADE,\n  of_index     INTEGER,\n"))});
      // (and the array CHECK as schema 53 had it: schema 54's reads of_statistic_id, a column this simulated old table does not have)
      db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'selection_measurement'",
                 {V(std::string("CHECK (array_path IS NULL OR (code IS NULL AND (of_field_id IS NOT NULL OR of_statistic_id IS NOT NULL)))")), V(std::string("CHECK (array_path IS NULL OR (code IS NULL AND of_field_id IS NOT NULL))"))});
      db.execute("PRAGMA schema_version = " + std::to_string(sv + 1));
    });
    db.exec_script("CREATE UNIQUE INDEX selection_measurement_key ON selection_measurement (selection_id, coalesce(code, -1), coalesce(of_field_id, ''), coalesce(within_selection_id, ''), coalesce(within_code, -1), measure);\n" +
                   std::string(view).replace(view.find(wide), wide.size(), ")") + ";");
    CHECK(!has_column(db, "selection_measurement", "of_statistic_id"));
    CHECK(!has_table(db, "statistic"));
    const std::int64_t uv = integer(*db.one("PRAGMA user_version"), "user_version");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");                 // opens and recovers with no statistic table
    CHECK(statistics_of(sub.db(), sub.id()).empty());
    CHECK(!sub.find("statistic", "time/rec_statistics/rec_psd") && !sub.by_name("statistic", "rec_psd"));   // looked up before the first write (a producer's replace check)
    CHECK_EQ(code_of([&] { read_statistic(sub, "rec_psd"); }), std::string(code::no_such_row));
    sub.compose([&] { sub.statistic(spec_of(fid, v)); });
    CHECK(has_table(sub.db(), "statistic") && read_statistic(sub, "rec_psd").values == v);
    CHECK(integer(*sub.db().one("PRAGMA user_version"), "user_version") == uv);
    CHECK(has_column(sub.db(), "selection_measurement", "of_statistic_id") && has_column(sub.db(), "selection_measurement", "of_index"));
    const auto m = sub.db().one("SELECT value FROM meta WHERE key = 'table:statistic'");
    CHECK(m && text(*m, "value").find("added in place") != std::string::npos);
    CHECK(text(*sub.db().one("PRAGMA quick_check"), "quick_check") == "ok");
    CHECK(sub.store().exists(join_key(text(read_statistic(sub, "rec_psd").row, "path"), "zarr.json")));
  }

  // 4. A BRAINSTORM SENSOR-LEVEL PSD (timefreq, DataType data): a detached statistic, rows x time x freq, power_density by welch, its axes' values inline
  {
    const fs::path mat = root / "timefreq_psd_x.mat";
    fs::create_directories(root);
    mat_t* mf = Mat_CreateVer(mat.string().c_str(), nullptr, MAT_FT_MAT5);
    auto txt = [](const char* name, const std::string& t) { size_t d[2] = {1, t.size()}; return Mat_VarCreate(name, MAT_C_CHAR, MAT_T_UINT8, 2, d, const_cast<char*>(t.c_str()), 0); };
    auto dbl = [](const char* name, std::vector<size_t> d, double* x) { return Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, static_cast<int>(d.size()), d.data(), x, 0); };
    double tf[6] = {1, 2, 3, 4, 5, 6};                               // [2 rows x 1 time x 3 freq], column-major: row 0 = 1 3 5, row 1 = 2 4 6
    double tm[1] = {0}, fr[3] = {0, 25, 50};
    Mat_VarWrite(mf, dbl("TF", {2, 1, 3}, tf), MAT_COMPRESSION_NONE);
    Mat_VarWrite(mf, dbl("Time", {1, 1}, tm), MAT_COMPRESSION_NONE);
    Mat_VarWrite(mf, dbl("Freqs", {1, 3}, fr), MAT_COMPRESSION_NONE);
    Mat_VarWrite(mf, txt("Measure", "power"), MAT_COMPRESSION_NONE);
    Mat_VarWrite(mf, txt("Method", "psd"), MAT_COMPRESSION_NONE);
    Mat_VarWrite(mf, txt("DataType", "data"), MAT_COMPRESSION_NONE);
    Mat_VarWrite(mf, txt("Comment", "PSD: MEG"), MAT_COMPRESSION_NONE);
    size_t d12[2] = {1, 2};
    matvar_t** cells = static_cast<matvar_t**>(std::calloc(3, sizeof(matvar_t*)));
    cells[0] = txt(nullptr, "MEG0111"); cells[1] = txt(nullptr, "MEG0112");
    Mat_VarWrite(mf, Mat_VarCreate("RowNames", MAT_C_CELL, MAT_T_CELL, 2, d12, cells, 0), MAT_COMPRESSION_NONE);
    Mat_Close(mf);
    DatasetWriter ds = create_dataset(root / "bst", "bst");
    make_subject(ds, "S1");
    SubjectWriter sub = SubjectWriter::open(ds, "S1");
    sub.compose([&] { CHECK(nxr::bst::export_timefreq_statistic(sub, mat, std::nullopt).has_value()); });
    const auto r = read_statistic(sub, "timefreq_psd_x");
    CHECK(r.values == (std::vector<double>{1, 3, 5, 2, 4, 6}));        // C order: row, time, freq
    CHECK_EQ(text(r.row, "measure"), std::string("power_density"));
    CHECK_EQ(text(r.row, "method"), std::string("welch"));
    CHECK(text(r.row, "axes_json").find("MEG0112") != std::string::npos && text(r.row, "axes_json").find("\"frequency\"") != std::string::npos);
  }
  return nxrtest::finish("test_statistic");
}
