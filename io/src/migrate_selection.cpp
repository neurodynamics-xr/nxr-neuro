// nxr-io — MIGRATE A DATASET'S SELECTIONS to the member form (schema 52; migrate.h `migrate_selection_storage`). In place, under the dataset's writer lock, subject by subject,
// RESUMABLE (a crash leaves a selection old or new, a second run changes nothing) and never deleting data the new form does not replace. It converts:
//   1. the JSON form (params_json `member_numbered` / `levels`, a dictionary entry per member), if any database holds it: the array is already member numbers; the labels become
//      the sibling array `P.labels`, the dictionary one row per label; a `levels` list (a dyadic grid) becomes the grid rule `params_json.grid` and the member number the
//      tile code (the grid's closed form: number at level l = number >> (depth - l)); no level rows or parent arrays are made (a grid's hierarchy is a rule);
//   2. a per-element LABEL array (an imported atlas, flags, channel names, a label image): member numbers (distinct labels, ascending) in the array, the labels in `P.labels`
//      (absent when the labels are 0 .. n - 1), the dictionary already one row per label; measurement rows keyed by the label (`code`, `within_code` on that selection) are
//      renumbered to member numbers. The level arrays of a spatial dyadic ladder already hold tile codes = member numbers and are left as they are; the ladder row gets its rule;
//   3. a constant-Q ladder's levels: `code` becomes the power of two of the level's window (its label; identity stays `member_ordinal`), from the level's stored `window_samples`;
//      a ladder lacking it is REFUSED (nothing is guessed); the ladder's dictionary entries take the power as their code. Measurement rows name levels by row id and are untouched.
// A new array is written beside the old (`P.migrated`), the rows are committed with a per-selection note in `meta` (`selection_storage:<id>`), then the array replaces the old one;
// a rerun finishes a replacement a crash interrupted. The report counts what was converted; `dry_run` changes nothing.
#include <algorithm>
#include <cstring>
#include <map>
#include <set>

#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/migrate.h"
#include "nxr/io/selection_store.h"
#include "nxr/io/writer.h"

namespace nxr::io {

namespace {

struct Array { Shape shape; DType dtype = DType::Int32; std::vector<std::int64_t> v; };

Array read_array(const Store& store, const std::string& path) {
  ArrayReader ar = ArrayReader::open(store, path);
  Array a;
  a.shape = ar.shape();
  a.dtype = ar.dtype();
  const auto raw = ar.read();
  const std::size_t sz = dtype_size(a.dtype), n = raw.size() / sz;
  a.v.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t* p = raw.data() + i * sz;
    switch (a.dtype) {
      case DType::Int8: a.v[i] = static_cast<std::int8_t>(*p); break;
      case DType::Int16: { std::int16_t x; std::memcpy(&x, p, 2); a.v[i] = x; break; }
      case DType::Int32: { std::int32_t x; std::memcpy(&x, p, 4); a.v[i] = x; break; }
      case DType::Int64: { std::int64_t x; std::memcpy(&x, p, 8); a.v[i] = x; break; }
      case DType::UInt8: a.v[i] = *p; break;
      case DType::UInt16: { std::uint16_t x; std::memcpy(&x, p, 2); a.v[i] = x; break; }
      case DType::UInt32: { std::uint32_t x; std::memcpy(&x, p, 4); a.v[i] = x; break; }
      case DType::Float32: { float x; std::memcpy(&x, p, 4); a.v[i] = static_cast<std::int64_t>(x); break; }
      case DType::Float64: { double x; std::memcpy(&x, p, 8); a.v[i] = static_cast<std::int64_t>(x); break; }
      default: throw Error(code::invalid_argument, path + ": not an integer or float array");
    }
  }
  return a;
}

// the new array of member numbers: the same shape; an unsigned image keeps its type, anything else is int32
void write_numbers(SubjectWriter& sub, const std::string& path, const Array& old, const std::vector<std::int64_t>& numbers) {
  if (old.dtype == DType::UInt8) {
    std::vector<std::uint8_t> o(numbers.begin(), numbers.end());
    sub.write_array(path, ArrayView{o.data(), DType::UInt8, old.shape});
  } else if (old.dtype == DType::UInt16) {
    std::vector<std::uint16_t> o(numbers.begin(), numbers.end());
    sub.write_array(path, ArrayView{o.data(), DType::UInt16, old.shape});
  } else {
    std::vector<std::int32_t> o(numbers.begin(), numbers.end());
    sub.write_array(path, ArrayView{o.data(), DType::Int32, old.shape});
  }
}

bool identity(const std::vector<std::int64_t>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) if (v[i] != static_cast<std::int64_t>(i)) return false;
  return true;
}

std::string flag_key(const std::string& id) { return "selection_storage:" + id; }
bool done(const Database& db, const std::string& id) { return db.one("SELECT 1 FROM meta WHERE key = ?", {V(flag_key(id))}).has_value(); }

// renumber the measurement rows keyed by a member of `sid` (code on its own rows, within_code where it groups): two passes through an offset, so the unique key never collides
void renumber_measurements(Database& db, const std::string& sid, const std::map<std::int64_t, std::int64_t>& to) {
  constexpr std::int64_t kOffset = std::int64_t{1} << 40;
  for (const auto& [from, n] : to) {
    db.execute("UPDATE selection_measurement SET code = ? WHERE selection_id = ? AND code = ?", {V(n + kOffset), V(sid), V(from)});
    db.execute("UPDATE selection_measurement SET within_code = ? WHERE within_selection_id = ? AND within_code = ?", {V(n + kOffset), V(sid), V(from)});
  }
  db.execute("UPDATE selection_measurement SET code = code - ? WHERE selection_id = ? AND code >= ?", {V(kOffset), V(sid), V(kOffset)});
  db.execute("UPDATE selection_measurement SET within_code = within_code - ? WHERE within_selection_id = ? AND within_code >= ?", {V(kOffset), V(sid), V(kOffset)});
}

// finish a replacement: the new array (P.migrated) takes the old one's place
void finish_array(SubjectWriter& sub, const std::string& path) {
  const std::string staged = path + ".migrated";
  if (!sub.store().exists_prefix(staged)) return;
  sub.store().erase_prefix(path);
  sub.store().rename_prefix(staged, path);
  sub.update([&] { sub.db().execute("UPDATE selection SET modified_utc = ? WHERE array_path = ? AND subject_id = ?", {V(now_utc()), V(path), V(sub.id())}); });   // the drain writes the node's attributes again
}

Json params_of(const Row& r) {
  Json p = Json::parse(opt_text(r, "params_json").value_or("{}"), nullptr, false);
  return p.is_object() ? p : Json::object();
}

// ── one selection with an element array ──────────────────────────────────────────────────────────────────────────
struct Counts { int labels = 0, json = 0, grids = 0, levels = 0, ladders = 0; };

void convert_array_set(SubjectWriter& sub, const Row& r, bool dry, Counts& c) {
  const std::string sid = text(r, "id"), path = text(r, "array_path");
  Database& db = sub.db();
  Json p = params_of(r);
  const bool json_form = p.contains("member_numbered");
  const Array old = read_array(sub.store(), path);
  std::vector<std::int64_t> numbers, labels;     // the new array (empty: the old one stays) and each member's label
  std::map<std::int64_t, std::int64_t> renumber;  // measurement codes: old -> new
  struct Entry { std::int64_t label; Row row; };
  std::vector<Entry> entries;                    // the dictionary, one row per label
  bool grid = false;
  std::int64_t depth = 0, pieces = 1;
  if (json_form) {
    ++c.json;
    std::vector<Row> per_member;
    if (!is_null(r, "dictionary_id")) per_member = db.all("SELECT * FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", {V(text(r, "dictionary_id"))});
    std::set<std::int64_t> seen;
    for (const Row& e : per_member) {
      const Json a = Json::parse(opt_text(e, "attributes_json").value_or("{}"), nullptr, false);
      const std::int64_t label = a.is_object() && a.contains("label") && a["label"].is_number() ? a["label"].get<std::int64_t>() : integer(e, "code");
      labels.push_back(label);
      if (seen.insert(label).second) entries.push_back({label, e});
    }
    if (p.contains("levels") && p["levels"].is_array() && !p["levels"].empty()) {   // a dyadic grid: its number is the tile code (the finest label); the levels are a rule
      grid = true;
      depth = static_cast<std::int64_t>(p["levels"].size()) - 1;
      pieces = static_cast<std::int64_t>(p["levels"][0].value("member_codes", Json::array()).size());
      numbers.reserve(old.v.size());
      for (const std::int64_t m : old.v) numbers.push_back(m >= 0 && m < static_cast<std::int64_t>(labels.size()) ? labels[static_cast<std::size_t>(m)] : -1);
      for (std::size_t m = 0; m < labels.size(); ++m) renumber[static_cast<std::int64_t>(m)] = labels[m];
      labels.clear();   // the labels are the numbers now
      ++c.grids;
    }
    p.erase("member_numbered");
    p.erase("levels");
  } else {
    ++c.labels;
    std::set<std::int64_t> distinct;
    for (const std::int64_t x : old.v) if (x >= 0) distinct.insert(x);
    std::map<std::int64_t, std::int64_t> number;
    for (const std::int64_t x : distinct) { number[x] = static_cast<std::int64_t>(labels.size()); labels.push_back(x); }
    numbers.reserve(old.v.size());
    for (const std::int64_t x : old.v) numbers.push_back(x >= 0 ? number[x] : -1);
    for (const auto& [lab, n] : number) if (lab != n) renumber[lab] = n;
  }
  if (dry) return;
  if (grid) p["grid"] = Json{{"kind", "dyadic"}, {"depth", depth}, {"pieces", pieces}};
  // 1. what is written beside (new names; harmless if the rest is interrupted)
  const std::string lp = selection::member_labels_path(path);
  if (!labels.empty() && !identity(labels)) sub.write_array(lp, ArrayView{labels.data(), DType::Int64, {static_cast<std::int64_t>(labels.size())}});
  if (!numbers.empty()) write_numbers(sub, path + ".migrated", old, numbers);
  // 2. the rows, with the note
  sub.update([&] {
    if (json_form) {
      db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(sid)});
      if (!is_null(r, "dictionary_id")) {
        const std::string did = text(r, "dictionary_id");
        db.execute("DELETE FROM dictionary_entry WHERE dictionary_id = ?", {V(did)});
        const bool own_names = std::all_of(entries.begin(), entries.end(), [](const Entry& e) { return is_null(e.row, "color_json") && text(e.row, "name") == std::to_string(e.label); });
        std::int64_t k = 0;
        for (const Entry& e : entries) {
          if (own_names) break;
          Row n = {{"dictionary_id", did}, {"code", V(e.label)}, {"name", text(e.row, "name")}, {"ordinal", V(k++)}};
          for (const char* col : {"color_json", "measurement", "measurement_unit", "function", "params_json"}) if (!is_null(e.row, col)) n[col] = e.row.at(col);
          Json a = Json::parse(opt_text(e.row, "attributes_json").value_or("{}"), nullptr, false);
          if (a.is_object()) { a.erase("label"); if (!a.empty()) n["attributes_json"] = jdump(a); }
          db.insert("dictionary_entry", n);
        }
      }
    }
    if (grid) db.execute("UPDATE selection SET n_members = ? WHERE id = ?", {V(pieces << depth), V(sid)});
    else db.execute("UPDATE selection SET n_members = ? WHERE id = ?", {V(static_cast<std::int64_t>(labels.size())), V(sid)});
    if (!renumber.empty()) renumber_measurements(db, sid, renumber);
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", {V(flag_key(sid)), V(now_utc())});
  });
  // 3. the new array takes the old one's place
  if (!numbers.empty()) finish_array(sub, path);
}

// ── a constant-Q ladder's levels: code = the power of two of the window ──────────────────────────────────────────
void convert_time_ladder(SubjectWriter& sub, const Row& ladder, bool dry, Counts& c) {
  Database& db = sub.db();
  const std::string lid = text(ladder, "id");
  const std::vector<Row> levels = db.all("SELECT * FROM selection WHERE member_of_id = ? AND type = 'spans' ORDER BY member_ordinal", {V(lid)});
  if (levels.empty()) return;   // a spatial ladder (its levels are sets with arrays): a grid by its rule, below
  std::vector<std::int64_t> power;
  for (const Row& lv : levels) {
    const Json p = params_of(lv);
    if (!p.contains("window_samples") || !p["window_samples"].is_number_integer())
      throw Error(code::invalid_argument, "ladder '" + text(ladder, "name") + "', level '" + text(lv, "name") + "': its window in samples is not stored, so its power of two cannot be set (nothing is guessed)");
    power.push_back(constantq::levelOfWidth(p["window_samples"].get<std::int64_t>()));
  }
  bool changed = false;
  for (std::size_t k = 0; k < levels.size(); ++k) if (is_null(levels[k], "code") || integer(levels[k], "code") != power[k]) changed = true;
  if (!changed) return;
  ++c.ladders;
  c.levels += static_cast<int>(levels.size());
  if (dry) return;
  sub.update([&] {
    for (std::size_t k = 0; k < levels.size(); ++k) db.execute("UPDATE selection SET code = ? WHERE id = ?", {V(power[k]), V(text(levels[k], "id"))});
    if (!is_null(ladder, "dictionary_id")) {   // the level dictionary: code = the power, ordinal = the position (two passes: the old codes may be the new ones)
      const std::string did = text(ladder, "dictionary_id");
      constexpr std::int64_t kOffset = std::int64_t{1} << 40;
      for (std::size_t k = 0; k < levels.size(); ++k)
        db.execute("UPDATE dictionary_entry SET code = ? WHERE dictionary_id = ? AND ordinal = ?", {V(power[k] + kOffset), V(did), V(static_cast<std::int64_t>(k))});
      db.execute("UPDATE dictionary_entry SET code = code - ? WHERE dictionary_id = ? AND code >= ?", {V(kOffset), V(did), V(kOffset)});
    }
  });
}

// a spatial dyadic ladder: the rule beside its (already member-form) level arrays
void mark_spatial_grid(SubjectWriter& sub, const Row& ladder, bool dry, Counts& c) {
  Database& db = sub.db();
  const std::vector<Row> levels = db.all("SELECT * FROM selection WHERE member_of_id = ? AND type = 'set' AND array_path IS NOT NULL ORDER BY member_ordinal", {V(text(ladder, "id"))});
  if (levels.empty()) return;
  Json p = params_of(ladder);
  if (p.contains("grid")) return;
  ++c.grids;
  if (dry) return;
  const std::int64_t depth = static_cast<std::int64_t>(levels.size()) - 1, pieces = is_null(levels[0], "n_members") ? 1 : std::max<std::int64_t>(1, integer(levels[0], "n_members"));
  p["grid"] = Json{{"kind", "dyadic"}, {"depth", depth}, {"pieces", pieces}};
  sub.update([&] { db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(text(ladder, "id"))}); });
}

}  // namespace

Json migrate_selection_storage(const std::filesystem::path& dataset_folder, const SelectionStorageOptions& opt) {
  DatasetWriter ds = open_writer(dataset_folder);
  Json report = Json::object();
  report["dataset"] = ds.folder().string();
  Counts total;
  Json subjects = Json::array();
  for (const Row& s : ds.db().all("SELECT name FROM subject ORDER BY name")) {
    SubjectWriter sub = SubjectWriter::open(ds, text(s, "name"), OpenSubjectOptions{true, std::nullopt});
    Database& db = sub.db();
    Counts c;
    // a replacement a crash interrupted is finished first
    for (const Row& r : db.all("SELECT id, array_path FROM selection WHERE subject_id = ? AND array_path IS NOT NULL", {V(sub.id())}))
      if (!opt.dry_run && done(db, text(r, "id"))) finish_array(sub, text(r, "array_path"));
    for (const Row& ladder : db.all("SELECT * FROM selection WHERE subject_id = ? AND type = 'ladder' AND status = 'complete'", {V(sub.id())})) {
      convert_time_ladder(sub, ladder, opt.dry_run, c);
      mark_spatial_grid(sub, ladder, opt.dry_run, c);
    }
    // the sets with an element array: not a spatial ladder's levels (tile codes are their members already), not done before
    for (const Row& r : db.all("SELECT s.* FROM selection s LEFT JOIN selection m ON m.id = s.member_of_id WHERE s.subject_id = ? AND s.type = 'set' AND s.array_path IS NOT NULL AND s.status = 'complete' "
                               "AND (m.id IS NULL OR m.type <> 'ladder')", {V(sub.id())})) {
      if (done(db, text(r, "id"))) continue;
      convert_array_set(sub, r, opt.dry_run, c);
    }
    total.labels += c.labels; total.json += c.json; total.grids += c.grids; total.levels += c.levels; total.ladders += c.ladders;
    subjects.push_back(Json{{"subject", text(s, "name")}, {"label_arrays", c.labels}, {"json_form", c.json}, {"grids", c.grids}, {"constant_q_ladders", c.ladders}, {"constant_q_levels", c.levels}});
    if (opt.on_stage) opt.on_stage("subject", text(s, "name"));
  }
  report["subjects"] = subjects;
  report["label_arrays"] = total.labels; report["json_form"] = total.json; report["grids"] = total.grids; report["constant_q_ladders"] = total.ladders;
  report["dry_run"] = opt.dry_run;
  if (!opt.dry_run) ds.db().execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('selection_storage', ?)", {V(std::string(kSelectionStorage))});
  return report;
}

}  // namespace nxr::io
