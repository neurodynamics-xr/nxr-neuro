// nxr-io — the statistic node (statistic.h).
#include "nxr/io/statistic.h"

#include <cstring>

#include "nxr/io/array_reader.h"
#include "nxr/io/errors.h"
#include "nxr/io/layout.h"
#include "nxr/io/surface.h"
#include "nxr/io/writer.h"

namespace nxr::io {

const char* model_sql();   // database.cpp: the embedded schema

namespace {
const char* kBegin = "-- \xe2\x94\x80\xe2\x94\x80 statistic (begin";
const char* kEnd = "-- \xe2\x94\x80\xe2\x94\x80 statistic (end)";
// the block of model.sql that makes the table (one text for new and old datasets: they cannot drift)
std::string statistic_ddl() {
  const std::string s = model_sql();
  const auto a = s.find(kBegin), b = s.find(kEnd);
  if (a == std::string::npos || b == std::string::npos || b < a) throw Error(code::not_a_database, "model.sql: no statistic block between its markers");
  const auto line = s.find('\n', a);
  return s.substr(line + 1, b - line - 1);
}
}  // namespace

bool has_table(const Database& db, const std::string& name) {
  return db.one("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", {V(name)}).has_value();
}

bool has_column(const Database& db, const std::string& table, const std::string& column) {
  for (const Row& r : db.all("SELECT name FROM pragma_table_info(?)", {V(table)})) if (text(r, "name") == column) return true;
  return false;
}

// MEASUREMENTS OF A STATISTIC (2026-11-13, the maintainers: a map statistic read as a map): a measurement row names a band (an index along the statistic's other
// axis) of a statistic as it names a Field: `of_statistic_id`, `of_index`; the unique key and the selection's zarr.json view take them. Added in place,
// once, to every dataset (new ones on their first statistic): selection_measurement's CREATE TABLE stays the shared model's.
bool ensure_statistic_measurements(Database& db) {
  if (has_column(db, "selection_measurement", "of_statistic_id")) return false;
  const auto view = db.one("SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'selection_attributes'");
  static const std::string kBefore = "'component', m.component)", kAfter = "'component', m.component, 'of_statistic_id', m.of_statistic_id, 'of_index', m.of_index)";
  if (!view || text(*view, "sql").find(kBefore) == std::string::npos) throw Error(code::schema_version, "selection_attributes: its measurement list is not the schema's; measurements of statistics cannot be added in place");
  std::string vsql = text(*view, "sql");
  vsql.replace(vsql.find(kBefore), kBefore.size(), kAfter);
  db.tx([&] {
    db.exec_script("ALTER TABLE selection_measurement ADD COLUMN of_statistic_id TEXT REFERENCES statistic (id) ON DELETE CASCADE;\n"
                   "ALTER TABLE selection_measurement ADD COLUMN of_index INTEGER;\n"
                   "DROP INDEX selection_measurement_key;\n"
                   "CREATE UNIQUE INDEX selection_measurement_key ON selection_measurement (selection_id, coalesce(code, -1), coalesce(of_field_id, ''), coalesce(of_statistic_id, ''), "
                   "coalesce(of_index, -1), coalesce(within_selection_id, ''), coalesce(within_code, -1), measure);\n"
                   "CREATE INDEX selection_measurement_statistic ON selection_measurement (of_statistic_id, of_index, measure);\n"
                   "DROP VIEW selection_attributes;\n" + vsql + ";");
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V("columns:selection_measurement.of_statistic_id"), V(now_utc() + " added in place (additive; user_version unchanged)")});
  });
  return true;
}

namespace {
// a statistic table made before its value type (2026-11-13): the columns added in place and its view made again from the model's block
bool ensure_statistic_value_type(Database& db) {
  if (has_column(db, "statistic", "value_type")) return false;
  const std::string ddl = statistic_ddl();
  const auto a = ddl.find("CREATE VIEW statistic_attributes"), b = ddl.find("FROM statistic t;", a);
  if (a == std::string::npos || b == std::string::npos) throw Error(code::not_a_database, "model.sql: the statistic block has no statistic_attributes view");
  db.tx([&] {
    db.exec_script("ALTER TABLE statistic ADD COLUMN value_type TEXT NOT NULL DEFAULT 'scalar' CHECK (value_type IN ('scalar', 'integer', 'complex', 'vector2', 'vector3', 'matrix3', 'quaternion', 'hermitian3'));\n"
                   "ALTER TABLE statistic ADD COLUMN n_components INTEGER NOT NULL DEFAULT 1 CHECK (n_components >= 1);\n"
                   "DROP VIEW statistic_attributes;\n" + ddl.substr(a, b + std::string("FROM statistic t;").size() - a));
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V("columns:statistic.value_type"), V(now_utc() + " added in place (additive; user_version unchanged)")});
  });
  return true;
}
}  // namespace

bool ensure_statistics(Database& db) {
  if (has_table(db, "statistic")) { const bool v = ensure_statistic_value_type(db); return ensure_statistic_measurements(db) || v; }
  // sync_job's table list widened in place (additive: every row stays valid), as allow_value_type widens the value types
  static const std::string kBefore = "'field', 'operator'))", kAfter = "'field', 'operator', 'statistic'))";
  const auto sj = db.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sync_job'");
  if (sj && text(*sj, "sql").find("'statistic'") == std::string::npos) {
    if (text(*sj, "sql").find(kBefore) == std::string::npos) throw Error(code::schema_version, "sync_job: its table list is not the schema's; the statistic table cannot be added in place");
    const std::int64_t v = integer(*db.one("PRAGMA schema_version"), "schema_version");
    db.with_writable_schema([&] {
      db.execute("UPDATE sqlite_master SET sql = replace(sql, ?, ?) WHERE type = 'table' AND name = 'sync_job'", {V(kBefore), V(kAfter)});
      db.execute("PRAGMA schema_version = " + std::to_string(v + 1));
    });
  }
  db.tx([&] {
    db.exec_script(statistic_ddl());
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V("table:statistic"), V(now_utc() + " added in place (additive; user_version unchanged)")});
  });
  ensure_statistic_measurements(db);
  const auto ok = db.one("PRAGMA quick_check");
  if (!ok || text(*ok, "quick_check") != "ok") throw Error(code::schema_version, "the database failed its check after the statistic table was added");
  return true;
}

const Json& statistic_registry() {
  static const Json r = Json::parse(
#include "gen/statistics_json.inc"
  );
  return r;
}
namespace {
bool known(const char* list, const std::string& name) {
  for (const auto& e : statistic_registry()[list]) if (e["name"] == name) return true;
  return false;
}
}  // namespace
bool statistic_measure_known(const std::string& name) { return known("measures", name); }
bool statistic_method_known(const std::string& name) { return known("methods", name); }

std::vector<Row> statistics_of(const Database& db, const std::string& subject_id, const std::optional<std::string>& of_field_id, const std::optional<std::string>& measure) {
  if (!has_table(db, "statistic")) return {};
  std::string sql = "SELECT * FROM statistic WHERE subject_id = ? AND status = 'complete'";
  std::vector<Value> args{V(subject_id)};
  if (of_field_id) { sql += " AND of_field_id = ?"; args.push_back(V(*of_field_id)); }
  if (measure) { sql += " AND measure = ?"; args.push_back(V(*measure)); }
  return db.all(sql + " ORDER BY name", args);
}

StoredStatistic read_statistic(SubjectWriter& sub, const std::string& name) {
  std::optional<Row> r;
  for (const Row& x : statistics_of(sub.db(), sub.id())) if (text(x, "name") == name) r = x;
  if (!r) {
    std::string have;
    for (const Row& x : statistics_of(sub.db(), sub.id())) have += (have.empty() ? "" : ", ") + text(x, "name");
    throw Error(code::no_such_row, sub.name() + ": no statistic named '" + name + "' (" + (have.empty() ? std::string("it has none") : "it has: " + have) + ")");
  }
  StoredStatistic out;
  out.row = *r;
  auto js = [&](const char* col) { return is_null(*r, col) ? Json() : Json::parse(text(*r, col), nullptr, false); };
  out.axes = js("axes_json"); out.components = js("components_json"); out.span = js("span_json"); out.options = js("options_json");
  out.producer = js("producer_json"); out.scaling = js("scaling_json");
  if (is_null(*r, "path")) return out;
  const ArrayReader ar = ArrayReader::open(sub.store(), text(*r, "path"));
  out.shape = ar.shape();
  const auto raw = ar.read();
  const DType dt = ar.dtype();
  if (dt == DType::Float32) {
    std::vector<float> f(raw.size() / 4);
    std::memcpy(f.data(), raw.data(), raw.size());
    out.values.assign(f.begin(), f.end());
  } else if (dt == DType::Float64) {
    out.values.resize(raw.size() / 8);
    std::memcpy(out.values.data(), raw.data(), raw.size());
  } else {
    throw Error(code::invalid_argument, name + ": a statistic's values are float32 or float64, not " + dtype_to_string(dt));
  }
  return out;
}

std::vector<MapSlice> map_slices(const Database& db, const std::string& subject_id, const std::string& manifold_id) {
  std::vector<MapSlice> out;
  for (const Row& r : statistics_of(db, subject_id)) {
    const Json axes = Json::parse(text(r, "axes_json"), nullptr, false);
    if (!axes.is_array() || axes.size() != 2 || axes[0].value("manifold_id", std::string()) != manifold_id || !axes[1].contains("values") || !axes[1]["values"].is_array()) continue;
    const Json& entries = axes[1]["values"];
    for (std::size_t k = 0; k < entries.size(); ++k) {
      if (!entries[k].is_object() || !entries[k].contains("name")) continue;
      MapSlice m;
      m.statistic = r;
      m.index = static_cast<std::int64_t>(k);
      m.n = axes[0].value("size", std::int64_t(0));
      m.name = sanitize_node_name(text(r, "name") + "_" + entries[k]["name"].get<std::string>());
      m.manifold_id = manifold_id;
      m.unit = opt_text(r, "unit");
      m.value_type = opt_text(r, "value_type").value_or("scalar");
      m.n_components = r.count("n_components") && !is_null(r, "n_components") ? integer(r, "n_components") : 1;
      out.push_back(m);
    }
  }
  return out;
}

std::vector<double> read_map_slice(SubjectWriter& sub, const MapSlice& slice) {
  const StoredStatistic st = read_statistic(sub, text(slice.statistic, "name"));
  if (st.shape.size() < 2 || st.shape[0] != slice.n) throw Error(code::invalid_argument, slice.name + ": its statistic is not [" + std::to_string(slice.n) + " x entries x …]");
  const std::int64_t K = st.shape[1], C = slice.n_components;
  if (static_cast<std::int64_t>(st.values.size()) != slice.n * K * C) throw Error(code::invalid_argument, slice.name + ": its values are not " + std::to_string(C) + " a " + slice.value_type);
  std::vector<double> v(static_cast<std::size_t>(slice.n * C));                // [n x components], each value's components in the field's order
  for (std::int64_t i = 0; i < slice.n; ++i)
    for (std::int64_t c = 0; c < C; ++c) v[static_cast<std::size_t>(i * C + c)] = st.values[static_cast<std::size_t>((i * K + slice.index) * C + c)];
  return v;
}

}  // namespace nxr::io
