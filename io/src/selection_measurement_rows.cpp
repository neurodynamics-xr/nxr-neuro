// nxr-io — the rows of what is measured on a selection: the `selection_measurement` writes, the roll-up and pyramid reads, the kinds and the deletes. The row-level machinery behind
// selection_measurement.h (its callers use that header); the registry, the derived measures' formulas and the measure dictionary are in measures.cpp. Port of the Studio's database.ts measurements.
#include "nxr/io/measures.h"
#include <cstring>
#include "nxr/io/writer.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/statistic.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>

#include "nxr/io/bundle.h"
#include "nxr/io/errors.h"
#include "nxr/io/rollup.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/io/store.h"

namespace nxr::io {

namespace {
Value v(const Json& x) {
  if (x.is_null()) return Value(std::monostate{});
  if (x.is_number_integer()) return Value(x.get<std::int64_t>());
  if (x.is_number()) return Value(x.get<double>());
  if (x.is_boolean()) return Value(static_cast<std::int64_t>(x.get<bool>()));
  if (x.is_string()) return Value(x.get<std::string>());
  return Value(x.dump());
}
Json opt(const Json& j, const char* k) { return j.contains(k) ? j[k] : Json(nullptr); }

// the reduction a read uses: the subject's declaration, else the registry's rule (`mean` for a derived one); "undeclared" for a measure in neither (read as stored, refused when merged)
std::string read_reduction(const Database& db, const std::string& measure) {
  const ReadRule r = measure_read_rule(db, measure);
  if (!r.known) return "undeclared";
  return r.rule == MergeRule::derived ? "mean" : merge_rule_name(r.rule);
}

Json stored_query(const Database& db, const Json& q, const std::string& measure, const std::string& reduction, bool threshold) {
  const std::int64_t shift = q.value("shift", static_cast<std::int64_t>(0));
  if (shift < 0) throw Error(code::invalid_argument, "measurements: shift must be a whole number \xe2\x89\xa5 0, got " + std::to_string(shift));
  // reduction "none" (the median within a member): the member's own value; rows of members cannot be merged, so a shift that groups members is refused below
  const std::string value = reduction == "none" || reduction == "undeclared" ? "max(value)" : reduction == "sum" ? "sum(value)" : reduction == "max" ? "max(value)" : reduction == "min" ? "min(value)"
                                                : "sum(value * coalesce(weight, 1.0)) / sum(coalesce(weight, 1.0))";
  std::vector<std::string> where = {"selection_id = ?", "code IS NOT NULL", "measure = ?"};
  std::vector<Value> args = {v(q.at("selectionId")), Value(measure)};
  if (!q.contains("ofFieldId") || q["ofFieldId"].is_null()) { if (q.contains("ofFieldId")) where.push_back("of_field_id IS NULL"); }
  else { where.push_back("of_field_id = ?"); args.push_back(v(q["ofFieldId"])); }
  // a statistic's slice (ofStatisticId, ofIndex); asked for a field or for the geometric rows, a statistic's rows are not among them
  if (has_column(db, "selection_measurement", "of_statistic_id")) {
    if (q.contains("ofStatisticId") && !q["ofStatisticId"].is_null()) {
      where.push_back("of_statistic_id = ?"); args.push_back(v(q["ofStatisticId"]));
      if (q.contains("ofIndex") && !q["ofIndex"].is_null()) { where.push_back("of_index = ?"); args.push_back(v(q["ofIndex"])); }
    } else if (q.contains("ofFieldId")) {
      where.push_back("of_statistic_id IS NULL");
    }
  }
  if (!q.contains("withinSelectionId") || q["withinSelectionId"].is_null()) where.push_back("within_selection_id IS NULL");
  else {
    where.push_back("within_selection_id = ?"); args.push_back(v(q["withinSelectionId"]));
    if (!q.contains("withinCode") || q["withinCode"].is_null()) where.push_back("within_code IS NULL");
    else { where.push_back("within_code = ?"); args.push_back(v(q["withinCode"])); }
  }
  std::vector<std::string> having;
  if (threshold && q.contains("minValue") && !q["minValue"].is_null()) having.push_back(value + " >= ?");
  if (threshold && q.contains("maxValue") && !q["maxValue"].is_null()) having.push_back(value + " <= ?");
  std::string w, h;
  for (const auto& x : where) w += (w.empty() ? "" : " AND ") + x;
  for (const auto& x : having) h += (h.empty() ? "" : " AND ") + x;
  std::string sql = "SELECT (code >> " + std::to_string(shift) + ") AS code, " + value + " AS value, sum(coalesce(weight, 1.0)) AS weight, count(*) AS members FROM selection_measurement WHERE " + w +
                    " GROUP BY 1" + (h.empty() ? "" : " HAVING " + h) + " ORDER BY value " + (q.value("order", std::string()) == "asc" ? "ASC" : "DESC");
  if (threshold && q.contains("limit") && q["limit"].is_number() && q["limit"].get<double>() != 0) sql += " LIMIT " + std::to_string(static_cast<std::int64_t>(std::floor(q["limit"].get<double>())));
  if (threshold && q.contains("minValue") && !q["minValue"].is_null()) args.push_back(v(q["minValue"]));
  if (threshold && q.contains("maxValue") && !q["maxValue"].is_null()) args.push_back(v(q["maxValue"]));
  Json out = Json::array();
  for (const Row& r : db.all(sql, args)) out.push_back(row_json(r));
  if (reduction == "undeclared")      // neither declared nor registered: read as stored, never merged
    for (const auto& r : out)
      if (r["members"].get<std::int64_t>() > 1) throw Error(code::unsupported, undeclared_refusal(measure) + " (shift " + std::to_string(shift) + ")");
  if (reduction == "none")
    for (const auto& r : out)
      if (r["members"].get<std::int64_t>() > 1) throw Error(code::unsupported, "measure " + measure + " does not roll up: its value is a member's own (a median of two members together is no combination of their medians), so rows of members cannot be merged (shift " + std::to_string(shift) + ")");
  return out;
}
}  // namespace

std::int64_t write_measurements(Database& db, const Json& rows, const std::optional<std::string>& by) {
  if (!rows.is_array() || rows.empty()) return 0;
  for (const auto& r : rows) {
    const std::string m = r.value("measure", std::string());
    if (measure_is_derived(m)) {
      std::string from;
      for (const auto& x : (*measure_spec(m))["derived_from"]) from += (from.empty() ? "" : " and ") + x.get<std::string>();
      throw Error(code::invalid_argument, "measurements: '" + m + "' is DERIVED — it is computed at read from " + from + ", never stored");
    }
  }
  ensure_measure_dictionary(db, by);
  bool of_statistic = false;                                   // a row measuring a statistic's slice (of_statistic_id, of_index): the columns added in place first
  for (const auto& r : rows) of_statistic = of_statistic || (r.contains("of_statistic_id") && !r["of_statistic_id"].is_null());
  if (of_statistic) ensure_statistics(db);
  bool with_producer = false;                                  // a row that says how it was made: the column added in place first
  for (const auto& r : rows) with_producer = with_producer || (r.contains("producer_json") && r["producer_json"].is_object());
  if (with_producer) ensure_measurement_producer(db);
  const std::string t = now_utc(true);
  std::vector<std::string> own;
  for (const auto& r : rows) if (!r.contains("code") || r["code"].is_null()) { const std::string s = r.value("selection_id", std::string()); if (std::find(own.begin(), own.end(), s) == own.end()) own.push_back(s); }
  db.tx([&] {
    for (const auto& m : rows) {
      const Json cu = m.contains("computed_utc") && !m["computed_utc"].is_null() ? m["computed_utc"] : Json(t);
      const Json cb = by ? Json(*by) : opt(m, "computed_by");
      if (with_producer)
        db.execute("INSERT INTO selection_measurement (selection_id, code, of_field_id, measure, value, weight, within_selection_id, within_code, unit, computed_utc, computed_by, array_path, component" +
                       std::string(of_statistic ? ", of_statistic_id, of_index" : "") + ", producer_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?" + (of_statistic ? ", ?, ?" : "") + ", ?)",
                   [&] {
                     std::vector<Value> a = {v(opt(m, "selection_id")), v(opt(m, "code")), v(opt(m, "of_field_id")), v(opt(m, "measure")), v(opt(m, "value")), v(opt(m, "weight")), v(opt(m, "within_selection_id")),
                                             v(opt(m, "within_code")), v(opt(m, "unit")), v(cu), v(cb), v(opt(m, "array_path")), v(opt(m, "component"))};
                     if (of_statistic) { a.push_back(v(opt(m, "of_statistic_id"))); a.push_back(v(opt(m, "of_index"))); }
                     a.push_back(m.contains("producer_json") && m["producer_json"].is_object() ? Value(m["producer_json"].dump()) : Value(std::monostate{}));
                     return a;
                   }());
      else if (of_statistic)
        db.execute("INSERT INTO selection_measurement (selection_id, code, of_field_id, measure, value, weight, within_selection_id, within_code, unit, computed_utc, computed_by, array_path, component, of_statistic_id, of_index) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   {v(opt(m, "selection_id")), v(opt(m, "code")), v(opt(m, "of_field_id")), v(opt(m, "measure")), v(opt(m, "value")), v(opt(m, "weight")), v(opt(m, "within_selection_id")),
                    v(opt(m, "within_code")), v(opt(m, "unit")), v(cu), v(cb), v(opt(m, "array_path")), v(opt(m, "component")), v(opt(m, "of_statistic_id")), v(opt(m, "of_index"))});
      else
        db.execute("INSERT INTO selection_measurement (selection_id, code, of_field_id, measure, value, weight, within_selection_id, within_code, unit, computed_utc, computed_by, array_path, component) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   {v(opt(m, "selection_id")), v(opt(m, "code")), v(opt(m, "of_field_id")), v(opt(m, "measure")), v(opt(m, "value")), v(opt(m, "weight")), v(opt(m, "within_selection_id")),
                    v(opt(m, "within_code")), v(opt(m, "unit")), v(cu), v(cb), v(opt(m, "array_path")), v(opt(m, "component"))});
    }
    for (const auto& id : own) db.execute("UPDATE selection SET modified_utc = ? WHERE id = ?", {Value(t), Value(id)});
  });
  return static_cast<std::int64_t>(rows.size());
}

Json measurement_provenance(const Database& db, const std::string& selection_id, const std::optional<std::string>& of_field_id, const std::string& measure) {
  Json out = Json{{"declaration", nullptr}, {"declared", false}, {"producers", Json::array()}};
  if (const auto d = declared_measure(db, measure)) { out["declaration"] = *d; out["declared"] = true; }
  else if (const auto r = measure_spec(measure)) out["declaration"] = *r;
  if (!has_column(db, "selection_measurement", "producer_json")) return out;
  std::string sql = "SELECT DISTINCT producer_json FROM selection_measurement WHERE selection_id = ? AND measure = ? AND producer_json IS NOT NULL";
  std::vector<Value> a{Value(selection_id), Value(measure)};
  if (of_field_id) { sql += " AND of_field_id = ?"; a.push_back(Value(*of_field_id)); }
  for (const Row& r : db.all(sql, a)) {
    const Json j = Json::parse(text(r, "producer_json"), nullptr, false);
    if (!j.is_discarded()) out["producers"].push_back(j);
  }
  return out;
}

Json query_measurements(const Database& db, const Json& q) {
  const std::string measure = q.at("measure");
  const auto spec = measure_spec(measure);
  if (!spec || (*spec)["derived_from"].is_null())
    return stored_query(db, q, measure, q.contains("reduction") && q["reduction"].is_string() ? q["reduction"].get<std::string>() : read_reduction(db, measure), true);
  std::vector<std::string> from;
  for (const auto& x : (*spec)["derived_from"]) from.push_back(x);
  std::vector<std::map<std::int64_t, Json>> by_code;
  Json first;
  for (std::size_t i = 0; i < from.size(); ++i) {
    const Json part = stored_query(db, q, from[i], read_reduction(db, from[i]), false);
    if (i == 0) first = part;
    std::map<std::int64_t, Json> m;
    for (const auto& r : part) m[r["code"].get<std::int64_t>()] = r;
    by_code.push_back(std::move(m));
  }
  std::vector<Json> out;
  for (const auto& r : first) {
    std::map<std::string, double> m;
    bool ok = true;
    for (std::size_t i = 0; i < from.size(); ++i) {
      const auto it = by_code[i].find(r["code"].get<std::int64_t>());
      if (it == by_code[i].end() || !it->second["value"].is_number()) { ok = false; break; }
      m[from[i]] = it->second["value"].get<double>();
    }
    if (!ok) continue;
    const double value = measure_formula(measure, m);
    if (!std::isfinite(value)) continue;
    if (q.contains("minValue") && q["minValue"].is_number() && value < q["minValue"].get<double>()) continue;
    if (q.contains("maxValue") && q["maxValue"].is_number() && value > q["maxValue"].get<double>()) continue;
    out.push_back(Json{{"code", r["code"]}, {"value", value}, {"weight", r["weight"]}, {"members", r["members"]}});
  }
  const bool asc = q.value("order", std::string()) == "asc";
  std::stable_sort(out.begin(), out.end(), [&](const Json& a, const Json& b) { return asc ? a["value"].get<double>() < b["value"].get<double>() : a["value"].get<double>() > b["value"].get<double>(); });
  if (q.contains("limit") && q["limit"].is_number() && q["limit"].get<double>() != 0) out.resize(std::min(out.size(), static_cast<std::size_t>(std::floor(q["limit"].get<double>()))));
  return Json(out);
}

Json measurement_kinds(const Database& db, const std::string& selection_id) {
  std::vector<Json> stored;
  for (const Row& r : db.all("SELECT of_field_id, measure, count(*) AS members, min(value) AS min, max(value) AS max FROM selection_measurement WHERE selection_id = ? AND code IS NOT NULL "
                             "GROUP BY of_field_id, measure ORDER BY measure", {Value(selection_id)}))
    stored.push_back(row_json(r));
  std::vector<std::pair<std::string, std::set<std::string>>> have;   // first-seen order of of_field_id
  for (const auto& k : stored) {
    const std::string key = k["of_field_id"].is_null() ? "" : k["of_field_id"].get<std::string>();
    auto it = std::find_if(have.begin(), have.end(), [&](const auto& p) { return p.first == key; });
    if (it == have.end()) { have.push_back({key, {}}); it = have.end() - 1; }
    it->second.insert(k["measure"].get<std::string>());
  }
  std::vector<Json> extra;
  for (const auto& spec : measure_registry()["measures"]) {
    if (spec["derived_from"].is_null()) continue;
    for (const auto& [key, names] : have) {
      bool all = true;
      for (const auto& n : spec["derived_from"]) if (!names.count(n.get<std::string>())) all = false;
      if (!all) continue;
      const std::string first = spec["derived_from"][0];
      const auto src = std::find_if(stored.begin(), stored.end(), [&](const Json& k) { return (k["of_field_id"].is_null() ? "" : k["of_field_id"].get<std::string>()) == key && k["measure"] == first; });
      extra.push_back(Json{{"of_field_id", src != stored.end() ? (*src)["of_field_id"] : Json(nullptr)}, {"measure", spec["name"]}, {"members", src != stored.end() ? (*src)["members"] : Json(0)},
                           {"min", nullptr}, {"max", nullptr}});
    }
  }
  stored.insert(stored.end(), extra.begin(), extra.end());
  std::stable_sort(stored.begin(), stored.end(), [](const Json& a, const Json& b) { return a["measure"].get<std::string>() < b["measure"].get<std::string>(); });
  return Json(stored);
}

std::int64_t delete_measurements(Database& db, const Json& w) {
  std::string sql = "DELETE FROM selection_measurement WHERE selection_id = ?";
  std::vector<Value> args = {v(w.at("selectionId"))};
  if (w.contains("ofFieldId")) { if (w["ofFieldId"].is_null()) sql += " AND of_field_id IS NULL"; else { sql += " AND of_field_id = ?"; args.push_back(v(w["ofFieldId"])); } }
  if (has_column(db, "selection_measurement", "of_statistic_id")) {
    if (w.contains("ofStatisticId") && !w["ofStatisticId"].is_null()) {
      sql += " AND of_statistic_id = ?"; args.push_back(v(w["ofStatisticId"]));
      if (w.contains("ofIndex") && !w["ofIndex"].is_null()) { sql += " AND of_index = ?"; args.push_back(v(w["ofIndex"])); }
    } else if (w.contains("ofFieldId")) {
      sql += " AND of_statistic_id IS NULL";
    }
  }
  if (w.contains("measure")) { sql += " AND measure = ?"; args.push_back(v(w["measure"])); }
  if (w.contains("rowsManifoldId") && !w["rowsManifoldId"].is_null()) { sql += " AND rows_manifold_id = ?"; args.push_back(v(w["rowsManifoldId"])); }   // only the rows on that manifold (issue #264)
  // the rows the statement itself deleted (SQLite's changes(), as the Studio's `.run().changes`): not the sync jobs its trigger queues
  std::int64_t n = 0;
  db.tx([&] {
    db.execute(sql, args);
    n = std::get<std::int64_t>(db.one("SELECT changes() AS n")->at("n"));
  });
  return n;
}

MeasurementValues measurement_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id,
                                     const std::optional<std::string>& of_statistic_id, const std::optional<std::int64_t>& of_index, const std::string& measure,
                                     const std::optional<std::string>& rows_manifold_id) {
  if (measure_is_derived(measure)) throw Error(code::invalid_argument, "measurements: '" + measure + "' is derived at read, never stored; summarize its ingredients");
  const Database& db = sub.db();
  std::string w = "selection_id = ? AND measure = ? AND within_selection_id IS NULL";
  std::vector<Value> a = {Value(selection_id), Value(measure)};
  const bool stat_cols = has_column(db, "selection_measurement", "of_statistic_id");
  if (of_statistic_id) {
    if (!stat_cols) throw Error(code::no_such_row, "measurements: this dataset holds no measurements of statistics");
    w += " AND of_statistic_id = ? AND of_index = ?"; a.push_back(Value(*of_statistic_id)); a.push_back(Value(of_index.value_or(0)));
  } else {
    w += of_field_id ? " AND of_field_id = ?" : " AND of_field_id IS NULL";
    if (of_field_id) a.push_back(Value(*of_field_id));
    if (stat_cols) w += " AND of_statistic_id IS NULL";
  }
  // THE MANIFOLD THE ROWS LIE ON (issue #264): named, only its array; not named, one array of the measure, or a refusal naming the manifolds
  if (has_column(db, "selection_measurement", "rows_manifold_id")) {
    if (rows_manifold_id) { w += " AND rows_manifold_id = ?"; a.push_back(Value(*rows_manifold_id)); }
    else {
      const auto on = db.all("SELECT DISTINCT rows_manifold_id FROM selection_measurement WHERE " + w + " AND array_path IS NOT NULL AND rows_manifold_id IS NOT NULL", a);
      if (on.size() > 1) {
        std::string names;
        for (const Row& r : on) {
          const auto n = db.one("SELECT name FROM manifold WHERE id = ?", {Value(text(r, "rows_manifold_id"))});
          names += (names.empty() ? "" : ", ") + (n ? text(*n, "name") : text(r, "rows_manifold_id"));
        }
        throw Error(code::invalid_argument, "measurements: '" + measure + "' is stored on several manifolds (" + names + "): name the one its rows lie on");
      }
    }
  } else if (rows_manifold_id) {
    throw Error(code::no_such_row, "measurements: no stored '" + measure + "' on that manifold (the dataset names no manifold of a measurement's rows)");
  }
  MeasurementValues out;
  std::set<std::string> by;
  // ARRAY-BACKED: [rows x members x components], this measure's component
  if (const auto r = db.one("SELECT array_path, component, unit, computed_by FROM selection_measurement WHERE " + w + " AND array_path IS NOT NULL LIMIT 1", a)) {
    const ArrayReader ar = ArrayReader::open(sub.store(), text(*r, "array_path"));
    const auto shape = ar.shape();
    if (shape.size() != 3) throw Error(code::invalid_argument, "measurements: " + text(*r, "array_path") + " is not [rows x members x components]");
    const std::int64_t R = shape[0], T = shape[1], C = shape[2], comp = integer(*r, "component");
    const auto raw = ar.read();
    std::vector<double> all(static_cast<std::size_t>(R * T * C));
    if (ar.dtype() == DType::Float32) { std::vector<float> f(all.size()); std::memcpy(f.data(), raw.data(), f.size() * 4); for (std::size_t i = 0; i < f.size(); ++i) all[i] = f[i]; }
    else if (ar.dtype() == DType::Float64) std::memcpy(all.data(), raw.data(), all.size() * 8);
    else throw Error(code::invalid_argument, "measurements: " + text(*r, "array_path") + " is not float32 or float64");
    out.rows = R; out.members = T; out.array_backed = true; out.unit = opt_text(*r, "unit");
    out.values.resize(static_cast<std::size_t>(R * T));
    for (std::int64_t i = 0; i < R * T; ++i) out.values[static_cast<std::size_t>(i)] = all[static_cast<std::size_t>(i * C + comp)];
    out.computed_by.push_back(opt_text(*r, "computed_by").value_or(""));
    return out;
  }
  // SCALAR ROWS: one value per member
  const auto rows = db.all("SELECT code, value, unit, computed_by FROM selection_measurement WHERE " + w + " AND code IS NOT NULL ORDER BY code", a);
  if (rows.empty()) throw Error(code::no_such_row, "measurements: no stored '" + measure + "' of that on selection " + selection_id);
  std::int64_t maxc = 0;
  for (const Row& r : rows) maxc = std::max(maxc, integer(r, "code"));
  out.rows = 1; out.members = maxc + 1; out.unit = opt_text(rows[0], "unit");
  out.values.assign(static_cast<std::size_t>(out.members), std::numeric_limits<double>::quiet_NaN());
  for (const Row& r : rows) {
    if (!is_null(r, "value")) out.values[static_cast<std::size_t>(integer(r, "code"))] = row_json(r)["value"].get<double>();
    by.insert(opt_text(r, "computed_by").value_or(""));
  }
  out.computed_by.assign(by.begin(), by.end());
  return out;
}

MeasurementValues derived_measurement_values(SubjectWriter& sub, const std::string& selection_id, const std::optional<std::string>& of_field_id,
                                             const std::optional<std::string>& of_statistic_id, const std::optional<std::int64_t>& of_index, const std::string& measure) {
  const auto spec = measure_spec(measure);
  if (!spec || (*spec)["derived_from"].is_null()) throw Error(code::invalid_argument, "measurements: '" + measure + "' is not a derived measure");
  std::vector<std::string> from;
  for (const auto& x : (*spec)["derived_from"]) from.push_back(x.get<std::string>());
  std::vector<MeasurementValues> parts;
  for (const auto& f : from) parts.push_back(measurement_values(sub, selection_id, of_field_id, of_statistic_id, of_index, f));
  MeasurementValues out = parts[0];
  for (const auto& p : parts) if (p.rows != out.rows || p.members != out.members) throw Error(code::invalid_argument, "measurements: the ingredients of '" + measure + "' differ in shape");
  out.unit = std::nullopt;
  out.computed_by.clear();
  for (const auto& p : parts) for (const auto& c : p.computed_by) out.computed_by.push_back(c);
  for (std::size_t i = 0; i < out.values.size(); ++i) {
    std::map<std::string, double> m;
    bool ok = true;
    for (std::size_t k = 0; k < from.size(); ++k) { const double v = parts[k].values[i]; if (!std::isfinite(v)) ok = false; m[from[k]] = v; }
    const double v = ok ? measure_formula(measure, m) : std::numeric_limits<double>::quiet_NaN();
    out.values[i] = std::isfinite(v) ? v : std::numeric_limits<double>::quiet_NaN();
  }
  return out;
}

// ── the read-time roll-up of a pyramid's array-backed measurements (measures.h) ──
namespace {
Json parsed(const Row& r, const char* col) { const auto t = opt_text(r, col); if (!t) return Json::object(); const Json j = Json::parse(*t, nullptr, false); return j.is_object() ? j : Json::object(); }
std::vector<double> doubles_at(const Store& store, const std::string& path, std::vector<std::int64_t>& shape) {
  const ArrayReader ar = ArrayReader::open(store, path);
  shape = ar.shape();
  const auto raw = ar.read();
  std::size_t n = 1;
  for (auto d : shape) n *= static_cast<std::size_t>(d);
  std::vector<double> out(n);
  if (ar.dtype() == DType::Float64) std::memcpy(out.data(), raw.data(), n * 8);
  else if (ar.dtype() == DType::Float32) { std::vector<float> f(n); std::memcpy(f.data(), raw.data(), n * 4); for (std::size_t i = 0; i < n; ++i) out[i] = f[i]; }
  else if (ar.dtype() == DType::Int64) { std::vector<std::int64_t> f(n); std::memcpy(f.data(), raw.data(), n * 8); for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<double>(f[i]); }
  else if (ar.dtype() == DType::Int32) { std::vector<std::int32_t> f(n); std::memcpy(f.data(), raw.data(), n * 4); for (std::size_t i = 0; i < n; ++i) out[i] = f[i]; }
  else throw Error(code::invalid_argument, "measurements: " + path + " is not a float or an integer array");
  return out;
}
}  // namespace

Json query_measurements(const Database& db, const Store& store, const Json& q) {
  if (!q.contains("spaceLevel") && !q.contains("timeLevel") && !q.contains("band") && !q.contains("spatialBand")) return query_measurements(db, q);
  const std::string sid = q.at("selectionId").get<std::string>(), measure = q.at("measure").get<std::string>();
  const auto sel = db.read("selection", sid);
  if (!sel) throw Error(code::no_such_row, "measurements: no selection " + sid);
  const Json sp = parsed(*sel, "params_json");
  const bool joint = sp.contains("joint");
  if (!joint && !sp.contains("atlas")) throw Error(code::invalid_argument, "measurements: '" + text(*sel, "name") + "' is neither a pyramid's joint Selection nor a spatial partition (no params_json.joint or .atlas)");
  // the space: its stored depth and its leaf areas
  const auto space = joint ? db.read("selection", sp["joint"].at("space").get<std::string>()) : sel;
  if (!space) throw Error(code::no_such_row, "measurements: the joint's space partition is missing");
  const Json spp = parsed(*space, "params_json");
  if (!spp.contains("atlas") || !spp["atlas"].contains("level")) throw Error(code::invalid_argument, "measurements: '" + text(*space, "name") + "' has no stored depth (params_json.atlas.level)");
  const int depth = spp["atlas"]["level"].get<int>();
  const std::int64_t leaves = integer(*space, "n_members");
  const int space_level = q.contains("spaceLevel") && q["spaceLevel"].is_number() ? q["spaceLevel"].get<int>() : depth;
  const int time_level = q.contains("timeLevel") && q["timeLevel"].is_number() ? q["timeLevel"].get<int>() : 0;
  // the field and the band: the rows of this selection
  std::string field_id;
  if (q.contains("ofFieldId") && q["ofFieldId"].is_string()) field_id = q["ofFieldId"].get<std::string>();
  else {
    const auto fs = db.all("SELECT DISTINCT of_field_id FROM selection_measurement WHERE selection_id = ? AND array_path IS NOT NULL AND of_field_id IS NOT NULL", {Value(sid)});
    if (fs.size() != 1) throw Error(code::invalid_argument, "measurements: '" + text(*sel, "name") + "' is measured for " + std::to_string(fs.size()) + " fields; name one (ofFieldId)");
    field_id = text(fs[0], "of_field_id");
  }
  // THE BAND OF THE ROW (the within key): a frequency band, a spatial band (the surface's Weyl level, step 4), or both (a joint bands key: code = band * n_spatial + spatial band)
  std::optional<std::int64_t> band, spatial;
  if (q.contains("band") && q["band"].is_number()) band = q["band"].get<std::int64_t>();
  if (q.contains("spatialBand") && q["spatialBand"].is_number()) spatial = q["spatialBand"].get<std::int64_t>();
  struct Within { std::optional<std::string> sel; std::int64_t code = 0; bool has = false; };
  auto matches = [&](const Row& r) {
    const std::optional<std::string> ws = opt_text(r, "within_selection_id");
    const WithinKind wk = within_kind(db, ws);
    const bool has = !is_null(r, "within_code");
    const std::int64_t c = has ? integer(r, "within_code") : 0;
    switch (wk.kind) {
      case WithinKind::None: return !band && !spatial;
      case WithinKind::Bands: return !spatial && (!band || (has && c == *band));
      case WithinKind::SpatialBands: return !band && (!spatial || (has && c == *spatial));
      case WithinKind::BandKey: {
        if (band && spatial) return has && c == band_key_code(*band, *spatial, wk.n_spatial);
        if (band) return has && c / wk.n_spatial == *band;
        if (spatial) return has && c % wk.n_spatial == *spatial;
        return true;
      }
    }
    return false;
  };
  // the within key the query reads: the one the rows of this field have that match; a joint of several needs one named
  Within chosen;
  bool chosen_none = false;                      // a band was named and no row has it
  {
    const auto cand = db.all("SELECT DISTINCT within_selection_id, within_code FROM selection_measurement WHERE selection_id = ? AND of_field_id = ? AND array_path IS NOT NULL ORDER BY within_code", {Value(sid), Value(field_id)});
    std::vector<Row> hits;
    for (const Row& r : cand) if (matches(r)) hits.push_back(r);
    if (hits.size() > 1 && (joint || band || spatial)) throw Error(code::invalid_argument, "measurements: '" + text(*sel, "name") + "' holds " + std::to_string(hits.size()) + " bands; name one (band = the row's within_code, or spatialBand)");
    chosen_none = hits.empty() && (band || spatial);
    if (!hits.empty()) { chosen.sel = opt_text(hits[0], "within_selection_id"); chosen.has = !is_null(hits[0], "within_code"); chosen.code = chosen.has ? integer(hits[0], "within_code") : 0; }
  }
  const WithinKind chosen_kind = within_kind(db, chosen.sel);
  auto row_of = [&](const std::string& m) -> std::optional<Row> {
    if (chosen_none) return std::nullopt;
    std::string w = "selection_id = ? AND measure = ? AND of_field_id = ? AND array_path IS NOT NULL";
    std::vector<Value> a = {Value(sid), Value(m), Value(field_id)};
    if (chosen.has) { w += " AND within_code = ?"; a.push_back(Value(chosen.code)); }
    if (chosen.sel) { w += " AND within_selection_id = ?"; a.push_back(Value(*chosen.sel)); }
    return db.one("SELECT * FROM selection_measurement WHERE " + w + " ORDER BY within_code LIMIT 1", a);
  };
  std::optional<std::int64_t> out_band, out_spatial;
  if (chosen.has) {
    if (chosen_kind.kind == WithinKind::BandKey) { std::int64_t b0 = 0, s0 = 0; band_key_split(chosen.code, chosen_kind.n_spatial, b0, s0); out_band = b0; out_spatial = s0; }
    else if (chosen_kind.kind == WithinKind::SpatialBands) out_spatial = chosen.code;
    else out_band = chosen.code;
  }
  const std::int64_t frames = joint ? sp["joint"].at("frames").get<std::int64_t>() : 1;
  std::int64_t code0 = 0;
  std::optional<Row> frames_sel;
  if (joint) {
    frames_sel = db.read("selection", sp["joint"].at("time").get<std::string>());
    if (frames_sel) code0 = parsed(*frames_sel, "params_json").value("atlas", Json::object()).value("frame_code0", std::int64_t(0));
  }
  if (time_level < 0 || (!joint && q.contains("timeLevel") && time_level > 0)) throw Error(code::invalid_argument, "measurements: timeLevel is 0 or more, and a spatial partition has no time axis");
  // one row's array as [leaves x frames], rolled up in space then time by its merge
  struct Cells { std::vector<double> v; std::int64_t nodes = 0, tiles = 0; };
  auto rolled = [&](const std::string& m, rollup::Merge op) -> std::optional<Cells> {
    const auto r = row_of(m);
    if (!r) return std::nullopt;
    std::vector<std::int64_t> shape;
    std::vector<double> all = doubles_at(store, text(*r, "array_path"), shape);
    std::vector<double> x;
    if (joint) {
      if (shape.size() != 3 || shape[1] != leaves || shape[2] != frames) throw Error(code::invalid_argument, "measurements: " + text(*r, "array_path") + " is not [band x " + std::to_string(leaves) + " leaves x " + std::to_string(frames) + " frames]");
      const std::int64_t b = is_null(*r, "within_code") ? 0 : integer(*r, "within_code");     // the array's leading axis is indexed by the within code
      if (b < 0 || b >= shape[0]) throw Error(code::invalid_argument, "measurements: band " + std::to_string(b) + " is outside the array's " + std::to_string(shape[0]));
      x.assign(all.begin() + static_cast<std::ptrdiff_t>(b * leaves * frames), all.begin() + static_cast<std::ptrdiff_t>((b + 1) * leaves * frames));
    } else {
      if (shape.size() != 1 || shape[0] != leaves) throw Error(code::invalid_argument, "measurements: " + text(*r, "array_path") + " is not [" + std::to_string(leaves) + " leaves]");
      x = all;
    }
    Cells c;
    try {
      // space acts on the leaf axis of [frames x leaves]; the array is [leaves x frames], so the frames are the rows once transposed
      std::vector<double> t(x.size());
      for (std::int64_t l = 0; l < leaves; ++l) for (std::int64_t f = 0; f < frames; ++f) t[static_cast<std::size_t>(f * leaves + l)] = x[static_cast<std::size_t>(l * frames + f)];
      std::vector<double> s = rollup::space(t, frames, leaves, depth, space_level, op, &c.nodes);       // [frames x nodes]
      std::vector<double> u(s.size());
      for (std::int64_t f = 0; f < frames; ++f) for (std::int64_t n = 0; n < c.nodes; ++n) u[static_cast<std::size_t>(n * frames + f)] = s[static_cast<std::size_t>(f * c.nodes + n)];
      c.v = rollup::time(u, c.nodes, frames, time_level, op, code0, &c.tiles);                           // [nodes x tiles]
    } catch (const std::invalid_argument& e) { throw Error(code::invalid_argument, std::string("measurements: ") + e.what()); }
    return c;
  };
  // the area of each node and the good samples of each tile (the weights of a density), by sums
  auto areas = [&]() -> std::vector<double> {
    const std::string path = spp["atlas"].value("areas", spp.value("areas", std::string()));
    if (path.empty()) throw Error(code::no_such_row, "measurements: '" + text(*space, "name") + "' has no leaf areas (params_json.atlas.areas)");
    std::vector<std::int64_t> sh;
    const std::vector<double> a = doubles_at(store, path, sh);
    std::int64_t nodes = 0;
    return rollup::space(a, 1, leaves, depth, space_level, rollup::Merge::Sum, &nodes);
  };
  auto samples = [&]() -> std::vector<double> {
    if (!frames_sel) throw Error(code::no_such_row, "measurements: the joint's frame tiling is missing");
    const auto r = db.one("SELECT * FROM selection_measurement WHERE selection_id = ? AND measure = 'count' AND of_field_id = ? AND array_path IS NOT NULL", {Value(text(*frames_sel, "id")), Value(field_id)});
    if (!r) throw Error(code::no_such_row, "measurements: the frame tiling '" + text(*frames_sel, "name") + "' carries no sample counts of that field (a `count` row)");
    std::vector<std::int64_t> sh;
    const std::vector<double> c = doubles_at(store, text(*r, "array_path"), sh);
    if (static_cast<std::int64_t>(c.size()) != frames) throw Error(code::invalid_argument, "measurements: the sample counts do not fit the frames");
    return rollup::time(c, 1, frames, time_level, rollup::Merge::Sum, code0);
  };
  Cells out;
  std::vector<double> weight_node, weight_tile;
  const rollup::Merge op = measure_is_derived(measure) ? rollup::Merge::Sum : rollup::merge_of(measure_merge(db, measure), measure);   // a derived measure has no merge of its own: its ingredients are merged (below)
  if (const auto c = rolled(measure, op)) {
    out = *c;
    if (joint) { weight_node = areas(); weight_tile = samples(); }
  } else if (measure.size() > 8 && measure.compare(measure.size() - 8, 8, "_density") == 0) {
    const std::string base = measure.substr(0, measure.size() - 8);
    if (rollup::merge_of(measure_merge(db, base), base) != rollup::Merge::Sum) throw Error(code::invalid_argument, "measurements: '" + measure + "': a density is a sum over area x samples; '" + base + "' is not a sum");
    const auto c = rolled(base, rollup::Merge::Sum);
    if (!c) throw Error(code::no_such_row, "measurements: '" + measure + "' is derived from '" + base + "', which this selection holds no array-backed row of");
    // a joint (space x time) Selection: sum / (area x samples). A spatial partition alone (a leaf measurement with no frame axis: phase_power, phase_flux, phase_circulation, ...): sum / area
    out = *c; weight_node = areas();
    if (joint) weight_tile = samples();
    for (std::int64_t n = 0; n < out.nodes; ++n) for (std::int64_t t = 0; t < out.tiles; ++t) {
      const double d = weight_node[static_cast<std::size_t>(n)] * (joint ? weight_tile[static_cast<std::size_t>(t)] : 1.0);
      double& x = out.v[static_cast<std::size_t>(n * out.tiles + t)];
      x = d > 0 ? x / d : std::numeric_limits<double>::quiet_NaN();
    }
  } else if (measure == "mean" || measure == "mean_square" || measure == "variance") {
    const auto w = rolled("weight", rollup::Merge::Sum), s1 = rolled("sum", rollup::Merge::Sum), s2 = rolled("sum_square", rollup::Merge::Sum);
    if (!w || !s1 || (measure != "mean" && !s2)) throw Error(code::no_such_row, "measurements: '" + measure + "' is derived from the array-backed `weight`, `sum`" + std::string(measure == "mean" ? "" : ", `sum_square`") + " rows, which this selection lacks");
    out = *s1;
    for (std::size_t i = 0; i < out.v.size(); ++i) {
      const double wi = w->v[i], mu = wi > 0 ? s1->v[i] / wi : std::numeric_limits<double>::quiet_NaN(), ms = (s2 && wi > 0) ? s2->v[i] / wi : std::numeric_limits<double>::quiet_NaN();
      out.v[i] = measure == "mean" ? mu : measure == "mean_square" ? ms : ms - mu * mu;
    }
    weight_node = w->v;
  } else if (const auto dspec = measure_spec(measure); dspec && !(*dspec)["derived_from"].is_null()) {
    // A DERIVED measure of array-backed SUMS (a ratio of the rolled-up sums, never rolled up itself): the phase gradient, the travelling index, ... Each ingredient is read as a stored row and merged by its reduction.
    std::vector<std::string> from;
    for (const auto& x : (*dspec)["derived_from"]) from.push_back(x.get<std::string>());
    std::map<std::string, Cells> got;
    for (const std::string& f : from) {
      const rollup::Merge mf = rollup::merge_of(measure_merge(db, f), f);
      if (mf == rollup::Merge::Mean) throw Error(code::invalid_argument, "measurements: '" + measure + "' is derived from '" + f + "', which is a mean, not a sum: it is derived from stored sums only");
      const auto c = rolled(f, mf);
      if (!c) throw Error(code::no_such_row, "measurements: '" + measure + "' is derived from '" + f + "', which this selection holds no array-backed row of (for that field" + (band || spatial ? " and band" : "") + ")");
      got[f] = *c;
    }
    out = got.at(from[0]);
    for (std::size_t i = 0; i < out.v.size(); ++i) {
      std::map<std::string, double> m;
      for (const std::string& f : from) m[f] = got.at(f).v[i];
      out.v[i] = measure_formula(measure, m);
    }
    weight_node = areas();
    if (joint) weight_tile = samples();
  } else {
    throw Error(code::no_such_row, "measurements: no array-backed '" + measure + "' on '" + text(*sel, "name") + "' of that field" + (band ? " in band " + std::to_string(*band) : std::string()));
  }
  // the cells as rows; members: the leaves and frames each merged
  const std::int64_t leaves_per = space_level < 0 ? leaves : leaves / (leaves >> depth << space_level);
  Json rows = Json::array();
  for (std::int64_t n = 0; n < out.nodes; ++n)
    for (std::int64_t t = 0; t < out.tiles; ++t) {
      const double val = out.v[static_cast<std::size_t>(n * out.tiles + t)];
      if (!std::isfinite(val)) continue;
      if (q.contains("minValue") && q["minValue"].is_number() && val < q["minValue"].get<double>()) continue;
      if (q.contains("maxValue") && q["maxValue"].is_number() && val > q["maxValue"].get<double>()) continue;
      const double wn = joint ? weight_node[static_cast<std::size_t>(n)] * weight_tile[static_cast<std::size_t>(t)] : (weight_node.empty() ? 1.0 : weight_node[static_cast<std::size_t>(n * out.tiles + t)]);
      rows.push_back(Json{{"code", n}, {"tile", joint ? Json((code0 >> time_level) + t) : Json(nullptr)}, {"band", out_band ? Json(*out_band) : Json(nullptr)}, {"spatial_band", out_spatial ? Json(*out_spatial) : Json(nullptr)}, {"value", val}, {"weight", wn},
                          {"members", leaves_per * (joint ? std::min<std::int64_t>(std::int64_t(1) << time_level, frames) : 1)}});
    }
  std::vector<Json> v(rows.begin(), rows.end());
  const bool asc = q.value("order", std::string()) == "asc";
  std::stable_sort(v.begin(), v.end(), [&](const Json& a, const Json& b) { return asc ? a["value"].get<double>() < b["value"].get<double>() : a["value"].get<double>() > b["value"].get<double>(); });
  if (q.contains("limit") && q["limit"].is_number() && q["limit"].get<double>() != 0) v.resize(std::min(v.size(), static_cast<std::size_t>(std::floor(q["limit"].get<double>()))));
  return Json(v);
}

}  // namespace nxr::io

namespace nxr::io {
MeasurementValues member_measure_values(SubjectWriter& sub, const std::string& selection_id, const std::string& measure) {
  const auto spec = measure_spec(measure);
  const bool derived = spec && !(*spec)["derived_from"].is_null();
  const std::string probe = derived ? (*spec)["derived_from"][0].get<std::string>() : measure;
  const auto rows = sub.db().all("SELECT DISTINCT of_field_id, of_statistic_id FROM selection_measurement WHERE selection_id = ? AND measure = ? AND array_path IS NOT NULL", {Value(selection_id), Value(probe)});
  if (rows.empty()) throw Error(code::no_such_row, "measurements: the selection holds no '" + probe + "'" + (derived ? " (an ingredient of '" + measure + "')" : std::string()));
  if (rows.size() > 1) throw Error(code::invalid_argument, "measurements: the selection holds '" + probe + "' of several fields or statistics: name the one");
  const std::optional<std::string> of_field = is_null(rows[0], "of_field_id") ? std::nullopt : std::optional<std::string>(text(rows[0], "of_field_id"));
  const std::optional<std::string> of_stat = is_null(rows[0], "of_statistic_id") ? std::nullopt : std::optional<std::string>(text(rows[0], "of_statistic_id"));
  const std::optional<std::int64_t> of_index = of_stat ? std::optional<std::int64_t>(0) : std::nullopt;
  return derived ? derived_measurement_values(sub, selection_id, of_field, of_stat, of_index, measure) : measurement_values(sub, selection_id, of_field, of_stat, of_index, measure);
}

}  // namespace nxr::io
