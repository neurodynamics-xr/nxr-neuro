// nxr-io — MIGRATE A DATASET'S MEASUREMENT STORAGE to one table for values over members (schema 53; migrate.h `migrate_measurement_storage`; spec 2026-11-17-consolidation.md section 6.3).
// In place, under the dataset's writer lock, subject by subject, RESUMABLE (a per-statistic note in `meta`, `measurement_storage:<statistic id>`; a second run changes nothing) and never deleting
// data unless `remove_statistics` is set (the PI's explicit second step). The values of a statistic with a member axis are rewritten into [rows x members x components] at `<path>.members`, verified
// byte for byte, and made a `selection_measurement` row. WRITTEN, NOT RUN.
#include <algorithm>
#include <cstring>
#include <set>

#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/migrate.h"
#include "nxr/io/selection_measurement.h"
#include "nxr/io/statistic.h"
#include "nxr/io/writer.h"

namespace nxr::io {

namespace {

std::string flag_key(const std::string& id) { return "measurement_storage:" + id; }
bool done(const Database& db, const std::string& id) { return db.one("SELECT 1 FROM meta WHERE key = ?", {V(flag_key(id))}).has_value(); }
Json parsed(const Row& r, const char* col) {
  const auto t = opt_text(r, col);
  if (!t) return Json();
  return Json::parse(*t, nullptr, false);
}
std::string members_path(const std::string& path) { return path + ".members"; }

// the plan of one statistic: which axis is the member axis, and the shape the measurement takes
struct Plan {
  std::size_t member_axis = 0;
  std::int64_t rows = 1, members = 0, components = 1;
  std::vector<std::int64_t> shape;      // the source's
  DType dtype = DType::Float64;
  std::size_t n_axes = 0;               // the statistic's axes (the leading dimensions); the rest are its components
};

// the statistic's member axis (the axis lying on the members of its partition), or nothing: a statistic with none is a statistic and stays
std::optional<std::size_t> member_axis_of(const Row& stat) {
  const Json axes = parsed(stat, "axes_json");
  const std::string part = opt_text(stat, "partition_selection_id").value_or("");
  if (!axes.is_array() || part.empty()) return std::nullopt;
  for (std::size_t k = 0; k < axes.size(); ++k)
    if (axes[k].is_object() && axes[k].contains("selection_id") && axes[k]["selection_id"].is_string() && axes[k]["selection_id"].get<std::string>() == part) return k;
  return std::nullopt;
}

Plan plan_of(const Store& store, const Row& stat, std::size_t member_axis) {
  Plan p;
  p.member_axis = member_axis;
  const ArrayReader ar = ArrayReader::open(store, text(stat, "path"));
  p.shape = ar.shape();
  p.dtype = ar.dtype();
  const Json axes = parsed(stat, "axes_json");
  const std::size_t n_axes = axes.size();
  if (p.shape.size() < n_axes) throw Error(code::invalid_argument, "statistic " + text(stat, "name") + ": its array has fewer dimensions (" + std::to_string(p.shape.size()) + ") than its axes (" + std::to_string(n_axes) + ")");
  p.members = p.shape[member_axis];
  for (std::size_t k = 0; k < n_axes; ++k) if (k != member_axis) p.rows *= p.shape[k];
  for (std::size_t k = n_axes; k < p.shape.size(); ++k) p.components *= p.shape[k];
  p.n_axes = n_axes;
  return p;
}

// the source's bytes permuted to [rows x members x components]: the trailing component dimensions are contiguous in the source and stay together as one block
std::vector<std::uint8_t> laid_out(const std::vector<std::uint8_t>& src, const Plan& p) {
  const std::size_t block = static_cast<std::size_t>(p.components) * dtype_size(p.dtype);
  std::vector<std::int64_t> stride(p.n_axes, 1);                 // strides, in blocks, over the statistic's axes
  for (std::size_t k = p.n_axes; k-- > 1;) stride[k - 1] = stride[k] * p.shape[k];
  std::vector<std::size_t> others;
  for (std::size_t k = 0; k < p.n_axes; ++k) if (k != p.member_axis) others.push_back(k);
  std::vector<std::uint8_t> out(src.size());
  for (std::int64_t r = 0; r < p.rows; ++r) {
    std::int64_t rest = r, offset = 0;                           // the row index, row-major over the other axes in order, unpacked into their coordinates
    for (std::size_t o = others.size(); o-- > 0;) {
      const std::size_t k = others[o];
      offset += (rest % p.shape[k]) * stride[k];
      rest /= p.shape[k];
    }
    for (std::int64_t m = 0; m < p.members; ++m)
      std::memcpy(out.data() + static_cast<std::size_t>(r * p.members + m) * block, src.data() + static_cast<std::size_t>(offset + m * stride[p.member_axis]) * block, block);
  }
  return out;
}

struct Counts { int converted = 0, conflicts = 0, products = 0, tilings = 0, removed = 0; std::vector<std::string> conflict_names; };

void convert_statistic(SubjectWriter& sub, const Row& stat, std::size_t member_axis, bool dry, Counts& c) {
  Database& db = sub.db();
  const std::string stat_id = text(stat, "id"), part = text(stat, "partition_selection_id"), path = text(stat, "path");
  // a measurement of that measure already on the partition (of the same field or parent statistic) is a conflict: nothing is overwritten
  const auto have = db.one("SELECT 1 FROM selection_measurement WHERE selection_id = ? AND measure = ? AND code IS NULL AND coalesce(of_field_id, '') = coalesce(?, '') AND coalesce(of_statistic_id, '') = coalesce(?, '') "
                           "AND within_selection_id IS NULL", {V(part), V(text(stat, "measure")), V(opt_text(stat, "of_field_id")), V(opt_text(stat, "of_statistic_id"))});
  if (have) { ++c.conflicts; c.conflict_names.push_back(text(stat, "name")); return; }
  const Plan p = plan_of(sub.store(), stat, member_axis);
  ++c.converted;
  if (dry) return;
  const std::string dst = members_path(path);                    // ALWAYS a copy beside the statistic's own array: the measurement never shares a path with a statistic that may be removed
  {
    const ArrayReader ar = ArrayReader::open(sub.store(), path);
    const std::vector<std::uint8_t> src = ar.read();
    const std::vector<std::uint8_t> out = laid_out(src, p);
    Json attrs{{"value_type", text(stat, "value_type")}, {"migrated_from_statistic", stat_id}};
    for (const auto& [k, col] : std::vector<std::pair<const char*, const char*>>{{"components", "components_json"}, {"axes", "axes_json"}, {"options", "options_json"}, {"producer", "producer_json"}, {"span", "span_json"}}) {
      const Json v = parsed(stat, col);
      if (!v.is_null()) attrs[k] = v;
    }
    if (const auto u = opt_text(stat, "unit")) attrs["unit"] = *u;
    const ArrayView view{out.data(), p.dtype, {p.rows, p.members, p.components}};
    sub.store().erase_prefix(dst);                       // a half-written copy of an interrupted run
    const ArrayMeta laid = sub.write_array(dst, view);
    layout_array(sub.store(), dst, laid, jdump(attrs));
    populate_array(sub.store(), dst, view);
    // VERIFIED before anything points at it: the copy reads back as the permuted bytes
    const ArrayReader back = ArrayReader::open(sub.store(), dst);
    if (back.shape() != std::vector<std::int64_t>{p.rows, p.members, p.components} || back.dtype() != p.dtype || back.read() != out)
      throw Error(code::invalid_argument, "statistic " + text(stat, "name") + ": the rewritten array " + dst + " does not read back as its source; nothing was changed");
  }
  Cols m;
  m.set("measure", text(stat, "measure")).set("array_path", dst).set("computed_by", opt_text(stat, "created_by").value_or("migrate_measurement_storage"));
  if (const auto of = opt_text(stat, "of_field_id")) m.set("of_field_id", *of);
  if (const auto os = opt_text(stat, "of_statistic_id")) m.set("of_statistic_id", *os).set("of_index", std::int64_t(0));
  if (const auto u = opt_text(stat, "unit")) m.set("unit", *u);
  sub.compose([&] { selection::save_measurement(sub, part, m); });
  sub.update([&] {
    Json options = parsed(stat, "options_json");
    if (!options.is_object()) options = Json::object();
    options["migrated_to_measurement"] = Json{{"selection_id", part}, {"measure", text(stat, "measure")}, {"array_path", dst}};
    db.execute("UPDATE statistic SET options_json = ? WHERE id = ?", {V(jdump(options)), V(stat_id)});
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V(flag_key(stat_id)), V(now_utc())});
  });
}

// step 2: a product's factor that is a level row of a ladder names the ladder and the level (codes unchanged)
void rewrite_products(SubjectWriter& sub, bool dry, Counts& c) {
  Database& db = sub.db();
  if (!has_column(db, "selection_factor", "level")) return;      // a database not yet at 53 (opened read-only): nothing to rewrite
  for (const Row& prod : db.all("SELECT s.* FROM selection s WHERE s.subject_id = ? AND EXISTS (SELECT 1 FROM selection_factor f JOIN selection l ON l.id = f.factor_selection_id JOIN selection g ON g.id = l.member_of_id "
                                "WHERE f.selection_id = s.id AND g.type = 'ladder' AND f.level IS NULL AND l.code IS NOT NULL)", {V(sub.id())})) {
    ++c.products;
    if (dry) continue;
    sub.update([&] {
      Json p = parsed(prod, "params_json");
      if (!p.is_object()) p = Json::object();
      Json entries = p.contains("product") && p["product"].is_array() ? p["product"] : Json::array();
      for (const Row& f : db.all("SELECT f.ordinal AS ordinal, l.member_of_id AS ladder_id, l.code AS level FROM selection_factor f JOIN selection l ON l.id = f.factor_selection_id JOIN selection g ON g.id = l.member_of_id "
                                 "WHERE f.selection_id = ? AND g.type = 'ladder' AND f.level IS NULL AND l.code IS NOT NULL", {V(text(prod, "id"))})) {
        const std::int64_t k = integer(f, "ordinal");
        db.execute("UPDATE selection_factor SET factor_selection_id = ?, level = ? WHERE selection_id = ? AND ordinal = ?", {V(text(f, "ladder_id")), V(integer(f, "level")), V(text(prod, "id")), V(k)});
        if (k < static_cast<std::int64_t>(entries.size()) && entries[static_cast<std::size_t>(k)].is_object()) {
          entries[static_cast<std::size_t>(k)]["selection"] = text(f, "ladder_id");
          entries[static_cast<std::size_t>(k)]["level"] = integer(f, "level");
        }
      }
      p["product"] = entries;
      db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(text(prod, "id"))});
    });
  }
}

// step 3: the frame tilings on the day-anchored tower are kept read-only and marked
void mark_towers(SubjectWriter& sub, bool dry, Counts& c) {
  Database& db = sub.db();
  for (const Row& r : db.all("SELECT * FROM selection WHERE subject_id = ? AND params_json IS NOT NULL AND (json_extract(params_json, '$.atlas.frame_level') IS NOT NULL OR json_extract(params_json, '$.atlas.frame_code0') IS NOT NULL) "
                             "AND json_extract(params_json, '$.retired_grid') IS NULL", {V(sub.id())})) {
    ++c.tilings;
    if (dry) continue;
    Json p = parsed(r, "params_json");
    p["retired_grid"] = "tower";
    sub.update([&] { db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(text(r, "id"))}); });
  }
}

// the explicit second step: a verified, migrated statistic is removed; its child statistics point at the measurement
void remove_migrated(SubjectWriter& sub, const Row& stat, bool dry, Counts& c) {
  Database& db = sub.db();
  const Json options = parsed(stat, "options_json");
  if (!options.is_object() || !options.contains("migrated_to_measurement")) return;
  const Json& to = options["migrated_to_measurement"];
  const auto row = db.one("SELECT array_path FROM selection_measurement WHERE selection_id = ? AND measure = ? AND array_path = ?",
                          {V(to.value("selection_id", std::string())), V(to.value("measure", std::string())), V(to.value("array_path", std::string()))});
  if (!row) return;                                        // its measurement is missing: not removed
  ++c.removed;
  if (dry) return;
  sub.update([&] {
    for (const Row& child : db.all("SELECT id, options_json FROM statistic WHERE of_statistic_id = ?", {V(text(stat, "id"))})) {
      Json o = parsed(child, "options_json");
      if (!o.is_object()) o = Json::object();
      o["of_measurement"] = to;
      db.execute("UPDATE statistic SET of_statistic_id = NULL, of_field_id = coalesce(of_field_id, ?), options_json = ? WHERE id = ?", {V(opt_text(stat, "of_field_id")), V(jdump(o)), V(text(child, "id"))});
    }
    db.execute("DELETE FROM statistic WHERE id = ?", {V(text(stat, "id"))});
  });
}

}  // namespace

Json migrate_measurement_storage(const std::filesystem::path& dataset_folder, const MeasurementStorageOptions& opt) {
  DatasetWriter ds = open_writer(dataset_folder);             // opening for write brings every database to schema 53 (Database::open): DDL only
  Json report = Json::object();
  report["dataset"] = ds.folder().string();
  Counts total;
  Json subjects = Json::array();
  for (const Row& s : ds.db().all("SELECT name FROM subject ORDER BY name")) {
    SubjectWriter sub = SubjectWriter::open(ds, text(s, "name"), OpenSubjectOptions{true, std::nullopt});
    Database& db = sub.db();
    Counts c;
    if (has_table(db, "statistic")) {
      for (const Row& st : db.all("SELECT * FROM statistic WHERE subject_id = ? AND status = 'complete' AND partition_selection_id IS NOT NULL AND path IS NOT NULL ORDER BY name, id", {V(sub.id())})) {
        const auto axis = member_axis_of(st);
        if (!axis) continue;                                    // a summary statistic: on axes that are not a selection's members, it stays
        if (!done(db, text(st, "id"))) convert_statistic(sub, st, *axis, opt.dry_run, c);
        else if (opt.remove_statistics) remove_migrated(sub, st, opt.dry_run, c);
      }
      if (opt.remove_statistics && !opt.dry_run)                // statistics converted in this very run are removed too, once they are noted
        for (const Row& st : db.all("SELECT * FROM statistic WHERE subject_id = ? AND status = 'complete' AND json_extract(options_json, '$.migrated_to_measurement') IS NOT NULL", {V(sub.id())})) remove_migrated(sub, st, false, c);
      if (opt.on_stage) opt.on_stage("converted", text(s, "name"));
    }
    rewrite_products(sub, opt.dry_run, c);
    if (opt.on_stage) opt.on_stage("products", text(s, "name"));
    mark_towers(sub, opt.dry_run, c);
    if (opt.on_stage) opt.on_stage("tilings", text(s, "name"));
    total.converted += c.converted; total.conflicts += c.conflicts; total.products += c.products; total.tilings += c.tilings; total.removed += c.removed;
    subjects.push_back(Json{{"subject", text(s, "name")}, {"statistics_converted", c.converted}, {"conflicts", c.conflict_names}, {"products_rewritten", c.products}, {"tower_tilings_marked", c.tilings}, {"statistics_removed", c.removed}});
  }
  report["subjects"] = subjects;
  report["statistics_converted"] = total.converted; report["conflicts"] = total.conflicts; report["products_rewritten"] = total.products;
  report["tower_tilings_marked"] = total.tilings; report["statistics_removed"] = total.removed;
  report["dry_run"] = opt.dry_run;
  if (!opt.dry_run) ds.db().execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('measurement_storage', ?)", {V(std::string(kMeasurementStorage))});
  return report;
}

}  // namespace nxr::io
