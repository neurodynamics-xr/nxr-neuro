// nxr-io — the selection CRUD (selection_store.h): save, load, update and delete a selection of any of the five types, its lookup, and the rows beneath it.
// io stores what it is given and designs nothing: a ladder's levels arrive as tilings with their bands (made by compute); here they are only written.
#include "nxr/io/selection_store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>

#include <cstring>

#include "nxr/io/array_reader.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/errors.h"
#include "nxr/io/json_text.h"
#include "nxr/io/records/records.h"
#include "nxr/io/selection.h"
#include "nxr/io/statistic.h"

namespace nxr::io::selection {

const char* type_name(Type t) {
  switch (t) {
    case Type::point: return "point";
    case Type::span: return "span";
    case Type::distance: return "distance";
    case Type::indices: return "indices";
    case Type::set: return "set";
  }
  return "set";
}

const char* family_name(Family f) {
  switch (f) {
    case Family::grid: return "grid";
    case Family::irregular: return "irregular";
    case Family::unmarked: return "unmarked";
  }
  return "unmarked";
}

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }
Value vopt(const std::optional<std::int64_t>& x) { return x ? Value(*x) : Value(std::monostate{}); }
std::string fixed(double x, int d) { char b[64]; std::snprintf(b, sizeof b, "%.*f", d, x); return b; }
std::string i2s(std::int64_t x) { return std::to_string(x); }

bool is_span_member(const Description& m) { return !m.profile && m.type == Type::span && m.extents.size() == 1 && m.extents[0].axis == 0; }
bool is_level(const Description& m) { return m.type == Type::set && m.tiling && m.level >= 0; }
template <class P>
bool every(const std::vector<Description>& v, P pred) { return std::all_of(v.begin(), v.end(), pred); }

Json color_json(const std::optional<std::array<int, 3>>& c) { return c ? Json::array({(*c)[0], (*c)[1], (*c)[2]}) : Json(); }

// the row's own columns, common to every type
SelectionSpec base(const Description& d, const Placement& at, const char* type, const char* default_cell) {
  SelectionSpec s;
  s.name = at.name; s.path = at.path; s.type = type; s.manifold_id = at.manifold_id; s.id = at.id; s.created_utc = at.created_utc; s.acquisition = at.acquisition;
  s.cols = at.cols;
  s.measurements = at.measurements;
  if (at.acquisition_id) s.cols.set("acquisition_id", *at.acquisition_id);
  s.cols.set("cell", at.cell.value_or(default_cell));
  if (at.of_field_id) s.cols.set("of_field_id", *at.of_field_id);
  if (at.label) s.cols.set("label", *at.label);
  if (at.description) s.cols.set("description", *at.description);
  if (at.dictionary_id) s.cols.set("dictionary_id", *at.dictionary_id);
  if (d.code) s.cols.set("code", *d.code);
  if (!d.unit.empty()) s.cols.set("unit", d.unit);
  if (at.created_by) s.cols.set("created_by", *at.created_by).set("modified_by", *at.created_by);
  return s;
}

// the lookup as the selection's dictionary (when it has one and none is named)
void dictionary_of(SubjectWriter& sub, const Description& d, const Placement& at, SelectionSpec& s, Saved& out) {
  if (at.dictionary_id) { out.dictionary_id = at.dictionary_id; return; }
  if (d.lookup.empty()) return;
  out.dictionary_id = save_lookup(sub, at.name, d.lookup);
  s.cols.set("dictionary_id", *out.dictionary_id);
}

void params_of(SelectionSpec& s, const Json& params) { if (params.is_object()) s.cols.json("params_json", params); }

// ── a set per element: member numbers (the array), member labels and level arrays (siblings of it), one dictionary row per label ──────────────────────────────────
// Each member's label, one per member (a description with none labels member i as i).
std::vector<std::int64_t> labels_of(const Description& d) {
  if (!d.member_codes.empty() || d.member_of.empty()) return d.member_codes;
  std::int64_t n = 0;
  for (const std::int32_t m : d.member_of) n = std::max<std::int64_t>(n, static_cast<std::int64_t>(m) + 1);
  std::vector<std::int64_t> out;
  for (std::int64_t i = 0; i < n; ++i) out.push_back(i);
  return out;
}
bool is_identity(const std::vector<std::int64_t>& labels) {
  for (std::size_t i = 0; i < labels.size(); ++i) if (labels[i] != static_cast<std::int64_t>(i)) return false;
  return true;
}

// A whole stored array of any integer or float type, as 64-bit integers (member numbers, labels and parents are written as int32 / int64; a legacy label image may be narrower).
std::vector<std::int64_t> read_ints(const Store& store, const std::string& path) {
  ArrayReader ar = ArrayReader::open(store, path);
  const auto raw = ar.read();
  const DType t = ar.dtype();
  const std::size_t n = raw.size() / dtype_size(t);
  std::vector<std::int64_t> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint8_t* p = raw.data() + i * dtype_size(t);
    switch (t) {
      case DType::Int32: { std::int32_t v; std::memcpy(&v, p, 4); out[i] = v; break; }
      case DType::Int64: { std::int64_t v; std::memcpy(&v, p, 8); out[i] = v; break; }
      case DType::Int8: out[i] = static_cast<std::int8_t>(*p); break;
      case DType::Int16: { std::int16_t v; std::memcpy(&v, p, 2); out[i] = v; break; }
      case DType::UInt32: { std::uint32_t v; std::memcpy(&v, p, 4); out[i] = v; break; }
      case DType::UInt8: out[i] = *p; break;
      case DType::UInt16: { std::uint16_t v; std::memcpy(&v, p, 2); out[i] = v; break; }
      case DType::Float32: { float v; std::memcpy(&v, p, 4); out[i] = static_cast<std::int64_t>(v); break; }
      case DType::Float64: { double v; std::memcpy(&v, p, 8); out[i] = static_cast<std::int64_t>(v); break; }
      default: refuse(path + ": its array's type is not an integer or float");
    }
  }
  return out;
}

// the database says whether its selections are stored in the member form (meta selection_storage, kSelectionStorage); a dataset from before is converted by migrate_selection_storage
void require_member_storage(const Database& db) {
  const auto r = db.one("SELECT value FROM meta WHERE key = 'selection_storage'");
  if (!r || text(*r, "value") != kSelectionStorage)
    throw Error(code::schema_version, "this dataset stores its selections in the form before member numbers: convert it first (migrate_selection_storage, `nxr-neuro migrate selections --dataset <folder>`)");
}

Json tiling_json(const Tiling& t) { return Json{{"origin", t.origin}, {"width", t.width}, {"hop", t.hop}, {"count", t.count}}; }

// The family a description's structure says (used when the caller left it unmarked, and for a stored row from before the mark).
Family derived_family(const Description& d) {
  if (d.type != Type::set) return Family::irregular;
  if (!d.members.empty() && every(d.members, is_level)) return Family::grid;
  if (d.tiling) return d.tiling->hop >= d.tiling->width ? Family::grid : Family::irregular;
  if (d.grid) return Family::grid;
  if (!d.factors.empty()) return every(d.factors, [](const Description& f) { return (f.family != Family::unmarked ? f.family : derived_family(f)) == Family::grid; }) ? Family::grid : Family::irregular;
  return Family::irregular;
}
Family family_to_write(const Description& d) { return d.family != Family::unmarked ? d.family : derived_family(d); }

// The family a stored row says: its mark in params_json.family, else what its type and params say (a row from before the mark).
bool row_is_grid(const Database& db, const Row& f) {
  const Json p = is_null(f, "params_json") ? Json() : Json::parse(text(f, "params_json"), nullptr, false);
  if (p.is_object() && p.contains("family") && p["family"].is_string()) return p["family"].get<std::string>() == "grid";
  const std::string t = text(f, "type");
  if (t == "ladder") return true;
  if (t == "spans") return p.is_object() && p.contains("tiling") && p["tiling"].value("hop", 0.0) >= p["tiling"].value("width", 0.0);
  if (t == "set") {
    if (p.is_object() && p.contains("grid")) return true;
    if (!is_null(f, "member_of_id")) { const auto up = db.read("selection", text(f, "member_of_id")); if (up && text(*up, "type") == "ladder") return true; }   // a level of a grid
    if (p.is_object() && p.contains("product"))
      for (const Row& x : db.all("SELECT factor_selection_id FROM selection_factor WHERE selection_id = ? ORDER BY ordinal", {V(text(f, "id"))})) {
        const auto fr = db.read("selection", text(x, "factor_selection_id"));
        if (!fr || !row_is_grid(db, *fr)) return false;
      }
    return p.is_object() && p.contains("product");
  }
  return false;
}

Saved save_ladder(SubjectWriter& sub, const Description& d, const Placement& at) {
  if (!at.manifold_id) refuse("selection " + at.name + ": a set with levels lies on a closed-form time line (its manifold)");
  const TimeLine tl = time_line(sub.db(), *at.manifold_id);
  const double sfreq = 1.0 / tl.spacing;
  const double end = tl.origin + static_cast<double>(tl.count) * tl.spacing;
  const std::size_t L = d.members.size();
  // each level's window in samples (the format's own rule: a width in seconds on the line's grid); the finest level's, for what a coarser holds
  const auto samples_of = [&](double seconds) { return static_cast<std::int64_t>(std::llround(seconds * sfreq)); };
  const std::int64_t finest = samples_of(d.members.front().tiling->width);
  const auto acquisition = at.acquisition_id ? at.acquisition_id : tl.acquisition_id;
  Saved out;
  // the LEVEL dictionary: code = the level's LABEL, its power of two; ordinal = the level's IDENTITY, its position (0 the finest); an entry carries the window in samples
  // and seconds, the tile count, the finest tiles one tile holds and the band it resolves (what the window resolves is derived and referenced by nothing)
  DictionarySpec ds;
  ds.name = at.name;
  ds.description = "the levels of the ladder '" + at.name + "': code = the level's power of two (its label), ordinal = its position, 0 the finest; an entry carries the level's "
                   "window in samples and seconds, its tile count, how many finest tiles one tile holds, and the band it resolves";
  std::vector<DictionaryEntry> entries;
  for (std::size_t k = 0; k < L; ++k) {
    const Description& m = d.members[k];
    const Tiling& t = *m.tiling;
    DictionaryEntry e;
    e.code = m.level;
    e.ordinal = static_cast<std::int64_t>(k);
    e.name = i2s(samples_of(t.width)) + " samples";
    for (const Entry& le : d.lookup) if (le.code == e.code) { if (!le.name.empty()) e.name = le.name; e.color = le.color; }
    Json info{{"level", static_cast<std::int64_t>(k)}, {"power", m.level}, {"window_samples", samples_of(t.width)}, {"window_s", t.width}, {"hop_samples", samples_of(t.hop)}, {"count", t.count},
              {"finest_tiles", samples_of(t.width) / std::max<std::int64_t>(1, finest)}, {"duration_s", t.width}, {"finest_window_samples", finest}};
    if (m.band) info["resolves_hz"] = Json{{"lo", m.band->first}, {"hi", m.band->second}};
    if (k < at.level_params.size() && at.level_params[k].is_object()) for (const auto& [key, v] : at.level_params[k].items()) info[key] = v;   // what the design settled
    e.cols.set("measurement", t.width).set("measurement_unit", "s").set("attributes_json", jdump(info));
    entries.push_back(std::move(e));
  }
  ds.entries = std::move(entries);
  const std::string dict_id = sub.dictionary(ds).id;
  // a TILE dictionary per level, EMPTY: a tile's code is its index in time order, the key a measurement, a label or a note hangs on (D83)
  std::vector<std::string> tile_dicts;
  for (std::size_t k = 0; k < L; ++k) {
    const Tiling& t = *d.members[k].tiling;
    DictionarySpec td;
    td.name = at.name + " L" + std::to_string(k) + " tiles";
    td.entries = std::vector<DictionaryEntry>{};
    td.description = "the " + i2s(t.count) + " tiles of level " + std::to_string(k) + " \xe2\x80\x94 a tile's code is its index in time order, the PRIMARY KEY a measurement, a label or a note "
                     "hangs on (D83); what a tile IS (" + i2s(samples_of(t.width)) + " samples, " + fixed(t.width, 3) + " s) is the level's entry in the ladder dictionary";
    tile_dicts.push_back(sub.dictionary(td).id);
  }
  // THE LADDER and its members
  Json params = at.params.is_object() ? at.params : Json::object();
  const Json facts{{"sfreq", sfreq}, {"samples", tl.count}, {"origin", tl.origin}, {"levels", static_cast<std::int64_t>(L)}, {"end", end}};   // the format's own facts, unless the caller named them
  for (const auto& [k, v] : facts.items())
    if (!params.contains(k)) params[k] = v;
  SelectionSpec set = base(d, at, "ladder", "1");
  set.no_elements = true;
  set.cols.set("dictionary_id", dict_id).set("n_elements", tl.count).set("n_members", static_cast<std::int64_t>(L));
  if (!set.cols.count("unit")) set.cols.set("unit", d.unit.empty() ? std::string("s") : d.unit);
  if (!set.cols.count("label")) set.cols.set("label", std::to_string(L) + " levels");
  if (acquisition) set.cols.set("acquisition_id", *acquisition);
  params_of(set, params);
  out.id = sub.selection(std::move(set));
  out.dictionary_id = dict_id;
  for (std::size_t k = 0; k < L; ++k) {
    const Description& m = d.members[k];
    const Tiling& t = *m.tiling;
    Placement mp;
    mp.name = at.name + " L" + std::to_string(k);
    mp.manifold_id = at.manifold_id;
    mp.created_by = at.created_by;
    mp.cell = "1";
    SelectionSpec s = base(m, mp, "spans", "1");
    if (acquisition) s.cols.set("acquisition_id", *acquisition);
    s.cols.set("label", i2s(samples_of(t.width)) + " samples \xc2\xb7 " + fixed(t.width, 3) + " s").set("member_of_id", out.id).set("member_ordinal", static_cast<std::int64_t>(k))
        .set("code", static_cast<std::int64_t>(m.level)).set("dictionary_id", tile_dicts[k]).set("n_elements", tl.count).set("n_members", t.count);
    if (!s.cols.count("unit")) s.cols.set("unit", d.unit.empty() ? std::string("s") : d.unit);
    Json mj{{"tiling", tiling_json(t)}, {"end", end}, {"level", static_cast<std::int64_t>(k)}, {"window_samples", samples_of(t.width)}, {"hop_samples", samples_of(t.hop)},
            {"finest_tiles", samples_of(t.width) / std::max<std::int64_t>(1, finest)}};
    if (m.band) mj["resolves_hz"] = Json{{"lo", m.band->first}, {"hi", m.band->second}};
    if (k < at.level_params.size() && at.level_params[k].is_object()) for (const auto& [key, v] : at.level_params[k].items()) mj[key] = v;   // what the design settled
    s.cols.json("params_json", mj);
    out.member_ids.push_back(sub.selection(std::move(s)));
  }
  return out;
}


// the dictionary of a set per element: the lookup's entries, ONE ROW PER LABEL (none when the lookup is empty: the labels are their own names); a dictionary row always exists
// (a partition's row needs one)
std::string dictionary_of_labels(SubjectWriter& sub, const Description& d, const Placement& at, SelectionSpec& s) {
  if (at.dictionary_id) return *at.dictionary_id;   // the caller's (an app vocabulary shared by name): its entries are not rewritten
  DictionarySpec ds;
  ds.name = at.name;
  ds.description = "the labels of '" + at.name + "': one entry per label (code = the label; the members' labels are the array beside the element array)";
  std::vector<DictionaryEntry> entries;
  std::int64_t k = 0;
  for (const Entry& e : d.lookup) {
    DictionaryEntry de;
    de.code = e.code;
    de.name = e.name;
    de.ordinal = k++;
    de.color = e.color;
    entries.push_back(std::move(de));
  }
  ds.entries = std::move(entries);
  const std::string id = sub.dictionary(ds).id;
  s.cols.set("dictionary_id", id);
  return id;
}

Saved save_per_element(SubjectWriter& sub, const Description& d, const Placement& at) {
  if (!at.path) refuse("selection " + at.name + ": a set per element is stored as an array: give its path");
  const bool dyadic = d.grid && d.grid->kind == Grid::Kind::dyadic;
  // a dyadic GRID: its member numbers are its tile codes, slots << depth of them (empty tiles too), no labels; an irregular set: dense numbers, a label per member
  const std::vector<std::int64_t> labels = dyadic ? std::vector<std::int64_t>{} : labels_of(d);
  const std::int64_t n = dyadic ? (d.grid->slots << d.grid->depth) : static_cast<std::int64_t>(labels.size());
  for (const std::int32_t m : d.member_of)
    if (m < -1 || m >= n) refuse("selection " + at.name + ": member number " + i2s(m) + " outside 0.." + i2s(n - 1));
  // a GRID's levels (dyadic tiles, a constant-Q ladder) are a closed-form rule, never member rows or parent arrays; an irregular set has none
  if (!d.members.empty())
    refuse("selection " + at.name + (dyadic ? ": a dyadic grid with its levels is saved by save_grid (one level alone is saved here)" : ": an irregular set per element has no levels"));
  // the labels, beside the element array (written first: a failure leaves an array no row names, which a later save of the same path replaces)
  if (!dyadic && !is_identity(labels)) sub.write_array(member_labels_path(*at.path), view(labels, {n}));
  std::vector<std::int32_t> numbers = d.member_of;   // kept alive until the node is written
  SelectionSpec s = base(d, at, "set", "0");
  s.data = view(numbers, {static_cast<std::int64_t>(numbers.size())});
  Saved out;
  out.dictionary_id = dictionary_of_labels(sub, d, at, s);
  if (dyadic) {
    Json params = at.params.is_object() ? at.params : Json::object();
    params["grid"] = Json{{"kind", "dyadic"}, {"depth", d.grid->depth}, {"slots", d.grid->slots}};   // the closed form: number at level l = number >> (depth - l)
    if (d.grid->finest >= 0) params["grid"]["finest"] = d.grid->finest;
    params_of(s, params);
  } else {
    params_of(s, at.params);
  }
  out.id = sub.selection(std::move(s));
  sub.update([&] { sub.db().execute("UPDATE selection SET n_members = ? WHERE id = ?", {V(n), V(out.id)}); });   // the members, empty ones too (the row counted the distinct numbers)
  return out;
}

// Why a stored selection cannot be a factor of a product (the Studio's D100: a factor is a set; a grid with levels names one level), or nothing when it can. An
// irregular set (an atlas, listed spans) may be a factor: a measurement layout; it just does not roll up.
std::optional<std::string> factor_refusal_of_row(const Row& f, int level = -1) {
  const std::string t = text(f, "type");
  if (t == "ladder") return level >= 0 ? std::nullopt : std::optional<std::string>("a grid with levels: name one level (its member row, or the grid with `factor_levels`, schema 53)");
  if (level >= 0) return std::string("a level is named only for a grid with levels (a `ladder`)");
  if (t == "set" || t == "spans") return std::nullopt;
  return std::string("a ") + t + " is one selection, not a set: a box on the product manifold is a span with one extent per factor";
}

// A product of sets (a grid when every factor is): a `set` row with nothing stored per member; its factors are named, in order, in `selection_factor`, and their radices in
// `params_json.product` [{selection, members}] (the Studio's D100); its member count the product of the radices; codes mixed radix, the FIRST factor fastest.
Saved save_product(SubjectWriter& sub, const Description& d, const Placement& at) {
  if (at.factor_ids.size() != d.factors.size() || d.factors.size() < 2) refuse("selection " + at.name + ": a product has two or more factors, each named by its stored selection (Placement::factor_ids)");
  if (!d.radices.empty() && d.radices.size() != d.factors.size()) refuse("selection " + at.name + ": one radix per factor");
  std::int64_t members = 1;
  std::string cell;
  bool all_grids = true;
  Json product = Json::array();
  const std::vector<int>& levels = !at.factor_levels.empty() ? at.factor_levels : d.factor_levels;
  if (!levels.empty() && levels.size() != at.factor_ids.size()) refuse("selection " + at.name + ": one level (or -1) per factor");
  for (std::size_t k = 0; k < at.factor_ids.size(); ++k) {
    const auto f = sub.db().read("selection", at.factor_ids[k]);
    if (!f) refuse("selection " + at.name + ": factor " + at.factor_ids[k] + " has no row");
    const int level = levels.empty() ? -1 : levels[k];
    if (const auto why = factor_refusal_of_row(*f, level)) refuse("selection " + at.name + ": factor " + i2s(static_cast<std::int64_t>(k)) + " ('" + text(*f, "name") + "') cannot be a factor: " + *why);
    all_grids = all_grids && row_is_grid(sub.db(), *f);
    // a level of a grid has no member count of its own on the row (the rule is closed form): its radix is the caller's (a level's tiles of the line, or a dyadic level's slots << level)
    if (level >= 0 && d.radices.empty()) refuse("selection " + at.name + ": factor " + i2s(static_cast<std::int64_t>(k)) + " is level " + i2s(level) + " of a grid: its radix (the level's member count) is given in `radices`");
    const std::int64_t count = level >= 0 ? 1 : (is_null(*f, "n_members") ? 1 : std::max<std::int64_t>(0, integer(*f, "n_members")));
    const std::int64_t radix = d.radices.empty() ? count : d.radices[k];
    if (radix < count) refuse("selection " + at.name + ": factor " + i2s(static_cast<std::int64_t>(k)) + " has " + i2s(count) + " members, more than its radix " + i2s(radix));
    if (radix > 0 && members > std::numeric_limits<std::int64_t>::max() / radix) refuse("selection " + at.name + ": more members than a 64-bit code holds");
    members *= radix;
    Json entry{{"selection", at.factor_ids[k]}, {"members", radix}};
    if (level >= 0) entry["level"] = level;
    product.push_back(std::move(entry));
    cell += (k ? "," : "") + text(*f, "cell");
  }
  Saved out;
  SelectionSpec s = base(d, at, "set", cell.c_str());
  s.no_elements = true;
  s.cols.set("n_members", members);
  Json params = at.params.is_object() ? at.params : Json::object();
  params["product"] = std::move(product);
  params["family"] = family_name(d.family != Family::unmarked ? d.family : (all_grids ? Family::grid : Family::irregular));
  params_of(s, params);
  out.id = sub.selection(std::move(s));
  sub.update([&] {
    for (std::size_t k = 0; k < at.factor_ids.size(); ++k) {
      Row r{{"selection_id", out.id}, {"ordinal", V(static_cast<std::int64_t>(k))}, {"factor_selection_id", at.factor_ids[k]}};
      if (!levels.empty() && levels[k] >= 0) r["level"] = V(levels[k]);
      sub.db().insert("selection_factor", r);
    }
  });
  return out;
}

}  // namespace

// ── a line's time grid (schema 53) ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
std::optional<std::pair<records::SelectionRow, TimeGridRule>> find_time_grid(const Database& db, const std::string& manifold_id, const std::string& rule_name) {
  const auto line = db.one("SELECT name FROM manifold WHERE id = ?", {V(manifold_id)});
  const std::string want = line && !rule_name.empty() ? text(*line, "name") + " time grid " + rule_name : std::string();
  for (const Row& r : db.all("SELECT id FROM selection WHERE manifold_id = ? AND type = 'ladder' AND json_extract(params_json, '$.grid.kind') = 'time' ORDER BY created_utc, id", {V(manifold_id)})) {
    const auto row = records::Selections(db).read(text(r, "id"));
    if (!row) continue;
    if (!want.empty() && row->name != want) continue;
    const Json& g = row->params_json["grid"];
    return std::make_pair(*row, TimeGridRule{g.value("origin", 0.0), g.value("rate", 0.0), g.value("samples", std::int64_t(0))});
  }
  return std::nullopt;
}

Saved save_time_grid(SubjectWriter& sub, const std::string& manifold_id, const TimeGridRule& rule, const std::optional<std::string>& created_by, const std::string& rule_name) {
  if (!(rule.rate > 0) || rule.samples < 1) refuse("a time grid needs a sample rate above 0 and at least one sample");
  if (const auto have = find_time_grid(sub.db(), manifold_id, rule_name)) {
    const TimeGridRule& h = have->second;
    if (h.samples != rule.samples || std::abs(h.rate - rule.rate) > 1e-12 * rule.rate || std::abs(h.origin - rule.origin) > 1e-12 * std::max(1.0, std::abs(rule.origin)))
      refuse("the line already has a time grid of this name with another rule (" + i2s(h.samples) + " samples at " + fixed(h.rate, 6) + " Hz from " + fixed(h.origin, 6) + " s)");
    Saved out;
    out.id = have->first.id;
    out.dictionary_id = have->first.dictionary_id;
    return out;
  }
  const auto line = sub.db().one("SELECT name FROM manifold WHERE id = ?", {V(manifold_id)});
  if (!line) refuse("no manifold " + manifold_id + " for the time grid");
  int levels = 0;
  while ((std::int64_t(1) << (levels + 1)) <= rule.samples) ++levels;   // the coarsest level whose tile (2^k samples) fits the line at least once
  DictionarySpec ds;
  const std::string grid_name = text(*line, "name") + " time grid" + (rule_name.empty() ? std::string() : " " + rule_name);
  ds.name = grid_name;
  ds.description = "the levels of the time grid of '" + text(*line, "name") + "': code = ordinal = k, the tile of 2^k samples from the line's first sample; `measurement` its duration in seconds";
  std::vector<DictionaryEntry> entries;
  for (int k = 0; k <= levels; ++k) {
    DictionaryEntry e;
    e.code = k; e.ordinal = k;
    e.name = i2s(std::int64_t(1) << k) + " samples";
    e.cols.set("measurement", static_cast<double>(std::int64_t(1) << k) / rule.rate).set("measurement_unit", "s");
    e.cols.json("attributes_json", Json{{"level", k}, {"samples", std::int64_t(1) << k}, {"tiles", (rule.samples + (std::int64_t(1) << k) - 1) >> k}});
    entries.push_back(std::move(e));
  }
  ds.entries = std::move(entries);
  const std::string dict_id = sub.dictionary(ds).id;
  Placement at;
  at.name = grid_name;
  at.manifold_id = manifold_id;
  at.cell = "1";
  at.created_by = created_by;
  Description d;
  d.unit = "s";
  SelectionSpec top = base(d, at, "ladder", "1");
  top.no_elements = true;
  top.cols.set("dictionary_id", dict_id).set("n_elements", rule.samples).set("n_members", rule.samples).set("label", i2s(levels + 1) + " levels");
  Json params{{"grid", Json{{"kind", "time"}, {"origin", rule.origin}, {"rate", rule.rate}, {"samples", rule.samples}, {"levels", levels + 1}}}, {"family", "grid"}};
  params_of(top, params);
  Saved out;
  out.id = sub.selection(std::move(top));
  out.dictionary_id = dict_id;
  return out;
}

std::vector<GridLevelSize> grid_level_sizes(const Database& db, const std::string& ladder_id) {
  const auto row = db.read("selection", ladder_id);
  if (!row) throw Error(code::no_such_row, "selection " + ladder_id + ": no such row");
  if (text(*row, "type") != "ladder" || is_null(*row, "dictionary_id")) refuse("selection " + text(*row, "name") + ": it is not a grid with levels");
  std::vector<GridLevelSize> out;
  const auto level_rows = db.all("SELECT id, member_ordinal, code FROM selection WHERE member_of_id = ? ORDER BY member_ordinal", {V(ladder_id)});
  std::map<std::int64_t, std::string> row_of_code;
  for (const Row& r : level_rows) if (!is_null(r, "code")) row_of_code[integer(r, "code")] = text(r, "id");
  for (const Row& e : db.all("SELECT code, ordinal, measurement, measurement_unit FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {V(text(*row, "dictionary_id"))})) {
    if (is_null(e, "measurement")) continue;
    GridLevelSize g;
    g.code = integer(e, "code");
    g.level = static_cast<int>(integer(e, "ordinal"));
    const Value& mv = e.at("measurement");
    g.size = std::holds_alternative<double>(mv) ? std::get<double>(mv) : static_cast<double>(std::get<std::int64_t>(mv));
    g.unit = opt_text(e, "measurement_unit").value_or("");
    if (const auto it = row_of_code.find(g.code); it != row_of_code.end()) g.selection_id = it->second;
    out.push_back(std::move(g));
  }
  return out;
}

GridLevelSize level_of_size(const Database& db, const std::string& ladder_id, double size, const std::string& unit) {
  const auto levels = grid_level_sizes(db, ladder_id);
  const GridLevelSize* best = nullptr;
  for (const GridLevelSize& g : levels) {
    if (g.unit != unit) continue;
    if (!best || std::abs(g.size - size) < std::abs(best->size - size) || (std::abs(g.size - size) == std::abs(best->size - size) && g.size > best->size)) best = &g;
  }
  if (!best) refuse("the grid has no level sized in '" + unit + "'");
  return *best;
}

std::vector<int> product_factor_levels(const Database& db, const std::string& id) {
  const auto row = records::Selections(db).read(id);
  if (!row) throw Error(code::no_such_row, "selection " + id + ": no such row");
  const bool has_level = has_column(db, "selection_factor", "level");
  const auto factors = db.all(has_level ? "SELECT level FROM selection_factor WHERE selection_id = ? ORDER BY ordinal" : "SELECT NULL AS level FROM selection_factor WHERE selection_id = ? ORDER BY ordinal", {V(id)});
  if (factors.empty()) refuse("selection " + row->name + ": it is not a product");
  std::vector<int> out;
  for (const Row& f : factors) out.push_back(is_null(f, "level") ? -1 : static_cast<int>(integer(f, "level")));
  return out;
}

// ── the arrays beside a set per element's element array ──────────────────────────────────────────────────────────
std::string member_labels_path(const std::string& array_path) { return array_path + ".labels"; }

MemberNumbering number_labels(const std::vector<std::int32_t>& element_labels) {
  std::map<std::int32_t, std::int32_t> number;
  for (const std::int32_t c : element_labels) if (c >= 0) number.emplace(c, 0);
  MemberNumbering out;
  for (auto& [label, m] : number) { m = static_cast<std::int32_t>(out.member_codes.size()); out.member_codes.push_back(label); }
  out.member_of.reserve(element_labels.size());
  for (const std::int32_t c : element_labels) out.member_of.push_back(c >= 0 ? number[c] : -1);
  return out;
}

std::vector<std::int64_t> member_labels(const Database& db, const Store& store, const std::string& selection_id) {
  require_member_storage(db);
  const auto r = records::Selections(db).read(selection_id);
  if (!r || !r->array_path) throw Error(code::no_such_row, "selection " + selection_id + ": no set per element");
  const std::string lp = member_labels_path(*r->array_path);
  if (store.exists_prefix(lp)) return read_ints(store, lp);
  std::vector<std::int64_t> out(static_cast<std::size_t>(r->n_members.value_or(0)));
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::int64_t>(i);
  return out;
}

std::vector<std::int64_t> ElementMembers::element_labels() const {
  std::vector<std::int64_t> out(member_of.size(), -1);
  for (std::size_t v = 0; v < member_of.size(); ++v) {
    const std::int32_t m = member_of[v];
    if (m >= 0) out[v] = static_cast<std::size_t>(m) < member_codes.size() ? member_codes[static_cast<std::size_t>(m)] : m;
  }
  return out;
}

std::string ElementMembers::name_of(std::int64_t label) const {
  const auto it = names.find(label);
  return it != names.end() ? it->second : i2s(label);
}

ElementMembers read_element_members(const Database& db, const Store& store, const std::string& selection_id) {
  const auto r = records::Selections(db).read(selection_id);
  if (!r) throw Error(code::no_such_row, "selection " + selection_id + ": no such row");
  if (!r->array_path) refuse("selection " + r->name + ": it stores no array of elements");
  ElementMembers out;
  out.member_of = Selection::read(db, store, selection_id).elements();
  out.member_codes = member_labels(db, store, selection_id);
  if (r->dictionary_id) for (const Entry& e : load_lookup(db, *r->dictionary_id)) out.names[e.code] = e.name;
  return out;
}

Description set_from_labels(const std::vector<std::int32_t>& element_labels, Lookup lookup) {
  MemberNumbering n = number_labels(element_labels);
  Description d;
  d.type = Type::set;
  d.member_of = std::move(n.member_of);
  d.member_codes = std::move(n.member_codes);
  d.lookup = std::move(lookup);
  return d;
}

// ── a dyadic grid ────────────────────────────────────────────────────────────────────────────────────────────────
Saved save_grid(SubjectWriter& sub, const Placement& ladder, const std::vector<GridLevel>& levels) {
  if (levels.empty()) refuse("grid " + ladder.name + ": it has no levels");
  if (!ladder.dictionary_id) refuse("grid " + ladder.name + ": a ladder needs the dictionary of its levels (Placement::dictionary_id)");
  Saved out;
  SelectionSpec top = base(Description{}, ladder, "ladder", "0");
  top.no_elements = true;
  top.cols.set("dictionary_id", *ladder.dictionary_id).set("n_members", static_cast<std::int64_t>(levels.size())).set("n_elements", static_cast<std::int64_t>(levels[0].tile_codes.size()));
  Json top_params = ladder.params.is_object() ? ladder.params : Json::object();
  top_params["family"] = "grid";
  params_of(top, top_params);
  out.id = sub.selection(std::move(top));
  out.dictionary_id = ladder.dictionary_id;
  for (std::size_t l = 0; l < levels.size(); ++l) {
    const GridLevel& g = levels[l];
    for (const std::int32_t c : g.tile_codes) if (c < -1 || c >= g.tiles) refuse("grid " + ladder.name + ": level " + std::to_string(l) + " has a tile code outside 0.." + i2s(g.tiles - 1));
    sub.write_array(g.array_path, view(g.tile_codes, {static_cast<std::int64_t>(g.tile_codes.size())}));
    SelectionSpec ls = base(Description{}, g.at, "set", "0");
    ls.no_elements = true;
    ls.cols.set("member_of_id", out.id).set("member_ordinal", static_cast<std::int64_t>(l)).set("code", static_cast<std::int64_t>(l))
        .set("n_elements", static_cast<std::int64_t>(g.tile_codes.size())).set("n_members", g.tiles).set("array_path", g.array_path);
    if (g.at.dictionary_id) ls.cols.set("dictionary_id", *g.at.dictionary_id);
    Json level_params = g.at.params.is_object() ? g.at.params : Json::object();
    level_params["family"] = "grid";
    params_of(ls, level_params);
    out.member_ids.push_back(sub.selection(std::move(ls)));
  }
  return out;
}

// ── rows ─────────────────────────────────────────────────────────────────────────────────────────────────────────
void insert_row(Database& db, const Row& row, std::optional<std::int64_t> n_elements, std::optional<std::int64_t> n_members, const std::optional<std::vector<SpanSpec>>& spans,
                const std::vector<ExtentSpec>& extents, const std::vector<Cols>& own_measurements) {
  const std::string sid = text(row, "id");
  db.insert("selection", row);
  if (n_elements) {
    if (n_members) db.execute("UPDATE selection SET n_elements = ?, n_members = ? WHERE id = ?", {V(*n_elements), V(*n_members), V(sid)});
    else db.execute("UPDATE selection SET n_elements = ? WHERE id = ?", {V(*n_elements), V(sid)});
  }
  if (spans) {
    std::int64_t k = 0;
    for (const auto& s : *spans)
      db.execute("INSERT INTO selection_span (selection_id, ordinal, start, stop, code) VALUES (?, ?, ?, ?, ?)", {V(sid), V(k++), V(s.start), V(s.stop), vopt(s.code)});
    db.execute("UPDATE selection SET n_members = ? WHERE id = ?", {V(static_cast<std::int64_t>(spans->size())), V(sid)});
  }
  for (const auto& e : extents) db.insert("selection_extent", {{"selection_id", sid}, {"axis", V(e.axis)}, {"start", e.start}, {"stop", e.stop}});
  for (const auto& m : own_measurements) {
    Row mr = {{"selection_id", sid}};
    for (const auto& [k, v] : m) mr[k] = v;
    db.insert("selection_measurement", mr);
  }
}

void replace_spans(Database& db, const std::string& selection_id, const std::vector<SpanSpec>& spans) {
  db.execute("DELETE FROM selection_span WHERE selection_id = ?", {V(selection_id)});
  for (std::size_t k = 0; k < spans.size(); ++k)
    db.execute("INSERT INTO selection_span (selection_id, ordinal, start, stop, code) VALUES (?, ?, ?, ?, ?)",
               {V(selection_id), V(static_cast<std::int64_t>(k)), V(spans[k].start), V(spans[k].stop), vopt(spans[k].code)});
  db.execute("UPDATE selection SET n_members = ? WHERE id = ?", {V(static_cast<std::int64_t>(spans.size())), V(selection_id)});
}

void replace_extents(Database& db, const std::string& selection_id, const std::vector<ExtentSpec>& extents) {
  db.execute("DELETE FROM selection_extent WHERE selection_id = ?", {V(selection_id)});
  for (const auto& e : extents) db.execute("INSERT INTO selection_extent (selection_id, axis, start, stop) VALUES (?, ?, ?, ?)", {V(selection_id), V(e.axis), V(e.start), V(e.stop)});
}

void set_params(Database& db, const std::string& selection_id, const Json& params) {
  db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(params)), V(selection_id)});
}

// ── the lookup ───────────────────────────────────────────────────────────────────────────────────────────────────
std::string save_lookup(SubjectWriter& sub, const std::string& name, const Lookup& lookup, const std::optional<std::string>& description) {
  DictionarySpec ds;
  ds.name = name;
  ds.description = description;
  std::vector<DictionaryEntry> entries;
  std::int64_t k = 0;
  for (const Entry& e : lookup) {
    DictionaryEntry de;
    de.code = e.code;
    de.name = e.name;
    de.ordinal = k++;
    de.color = e.color;
    entries.push_back(std::move(de));
  }
  ds.entries = std::move(entries);
  return sub.dictionary(ds).id;
}

Lookup load_lookup(const Database& db, const std::string& dictionary_id) {
  Lookup out;
  for (const Row& r : db.all("SELECT * FROM dictionary_entry WHERE dictionary_id = ? ORDER BY coalesce(ordinal, code), code", {V(dictionary_id)})) {
    const auto e = records::DictionaryEntryRow::from_row(r);
    Entry x;
    x.code = e.code;
    x.name = e.name;
    if (e.color_json.is_array() && e.color_json.size() >= 3) x.color = std::array<int, 3>{e.color_json[0].get<int>(), e.color_json[1].get<int>(), e.color_json[2].get<int>()};
    out.push_back(std::move(x));
  }
  return out;
}

void update_lookup(SubjectWriter& sub, const std::string& dictionary_id, const Lookup& lookup) {
  sub.update([&] {
    Database& db = sub.db();
    db.execute("DELETE FROM dictionary_entry WHERE dictionary_id = ?", {V(dictionary_id)});
    std::int64_t k = 0;
    for (const Entry& e : lookup) {
      Row row = {{"dictionary_id", dictionary_id}, {"code", V(e.code)}, {"name", e.name}, {"ordinal", V(k++)}};
      if (e.color) row["color_json"] = jdump(color_json(e.color));
      db.insert("dictionary_entry", row);
    }
  });
}

// ── kernels ──────────────────────────────────────────────────────────────────────────────────────────────────────
// A member with a profile is stored as type `kernel`: its ANCHOR as its own type stores it (a span: selection_extent rows; a point: center; a distance: center, radius, unit),
// `params_json` = {anchor: point|span|distance, profile: {family, params}, chart, gain}, `weight_fn` = the family. io stores what it is given; compute resolves it.
Saved save_kernel(SubjectWriter& sub, const Description& d, const Placement& at) {
  const Profile& pr = *d.profile;
  if (pr.family.empty()) refuse("selection " + at.name + ": a kernel states its profile family");
  if (pr.chart.empty()) refuse("selection " + at.name + ": a kernel states its chart");
  if (!std::isfinite(pr.gain)) refuse("selection " + at.name + ": a kernel states a finite gain");
  Json params = at.params.is_object() ? at.params : Json::object();
  Json pj = Json::object();
  for (const auto& [k, v] : pr.params) pj[k] = v;
  params["profile"] = Json{{"family", pr.family}, {"params", pj}};
  params["chart"] = pr.chart;
  params["gain"] = pr.gain;
  SelectionSpec s;
  Saved out;
  if (d.type == Type::span) {
    s = base(d, at, "kernel", "1");
    for (const Extent& e : d.extents) {
      if (!(e.stop >= e.start)) refuse("selection " + at.name + ": a kernel's span stop is not before its start");
      s.extents.push_back(ExtentSpec{e.axis, e.start, e.stop});
    }
    params["anchor"] = "span";
  } else if (d.type == Type::point) {
    if (!d.center) refuse("selection " + at.name + ": a kernel anchored on a point has a centre element");
    s = base(d, at, "kernel", "0");
    s.cols.set("center", *d.center);
    params["anchor"] = "point";
  } else if (d.type == Type::distance) {
    if (!d.center || !(d.radius >= 0)) refuse("selection " + at.name + ": a kernel anchored on a distance has a centre element and a radius >= 0");
    s = base(d, at, "kernel", "0");
    s.cols.set("center", *d.center).set("radius", d.radius);
    params["anchor"] = "distance";
  } else {
    refuse("selection " + at.name + ": a kernel's anchor is a point, a span or a distance, not a " + type_name(d.type));
  }
  s.cols.set("weight_fn", pr.family);
  params_of(s, params);
  out.id = sub.selection(std::move(s));
  return out;
}

// ── create ───────────────────────────────────────────────────────────────────────────────────────────────────────
Saved save(SubjectWriter& sub, const Description& d, const Placement& at_in) {
  Placement at = at_in;
  if (d.type == Type::set) {   // a set records its family (grid or irregular) in params_json; a product's is read from its stored factors (save_product)
    Json p = at.params.is_object() ? at.params : Json::object();
    p["family"] = family_name(family_to_write(d));
    at.params = std::move(p);
  }
  if (d.profile) return save_kernel(sub, d, at);
  Saved out;
  switch (d.type) {
    case Type::point: {
      if (!d.center) refuse("selection " + at.name + ": a point has a centre element");
      SelectionSpec s = base(d, at, "point", "0");
      s.cols.set("center", *d.center);
      params_of(s, at.params);
      out.id = sub.selection(std::move(s));
      return out;
    }
    case Type::span: {
      SelectionSpec s = base(d, at, "span", "1");
      for (const Extent& e : d.extents) {
        if (!(e.stop >= e.start)) refuse("selection " + at.name + ": a span's stop is not before its start");
        s.extents.push_back(ExtentSpec{e.axis, e.start, e.stop});
      }
      params_of(s, at.params);
      out.id = sub.selection(std::move(s));
      return out;
    }
    case Type::distance: {
      if (!d.center || !(d.radius >= 0)) refuse("selection " + at.name + ": a distance has a centre element and a radius >= 0");
      SelectionSpec s = base(d, at, "distance", "0");
      s.cols.set("center", *d.center).set("radius", d.radius);
      params_of(s, at.params);
      out.id = sub.selection(std::move(s));
      return out;
    }
    case Type::indices: {
      if (!at.path) refuse("selection " + at.name + ": indices are stored as an array: give its path");
      std::vector<std::int32_t> picks = d.indices;   // kept alive until the node is written; the order given is kept
      SelectionSpec s = base(d, at, "indices", "0");
      s.data = view(picks, {static_cast<std::int64_t>(picks.size())});
      s.picked = true;
      dictionary_of(sub, d, at, s, out);
      params_of(s, at.params);
      out.id = sub.selection(std::move(s));
      return out;
    }
    case Type::set: break;
  }
  // a set: with levels, by a rule, by codes, of spans, of other members
  if (!d.members.empty() && every(d.members, is_level)) return save_ladder(sub, d, at);
  if (d.tiling) {
    SelectionSpec s = base(d, at, "spans", "1");
    s.cols.set("n_members", d.tiling->count);
    Json params = at.params.is_object() ? at.params : Json::object();
    params["tiling"] = tiling_json(*d.tiling);
    params_of(s, params);
    dictionary_of(sub, d, at, s, out);
    out.id = sub.selection(std::move(s));
    return out;
  }
  if (!d.factors.empty()) return save_product(sub, d, at);
  if (d.grid && d.grid->kind == Grid::Kind::elements) {   // an elements grid: every element its own member, nothing stored
    SelectionSpec s = base(d, at, "set", "0");
    s.no_elements = true;
    s.cols.set("n_members", d.grid->size);
    Json params = at.params.is_object() ? at.params : Json::object();
    params["grid"] = Json{{"kind", "elements"}, {"size", d.grid->size}};
    params_of(s, params);
    out.id = sub.selection(std::move(s));
    return out;
  }
  if (!d.member_of.empty()) return save_per_element(sub, d, at);
  if (d.members.empty() && at.n_members) {   // a set by a rule with nothing stored
    SelectionSpec s = base(d, at, "set", "0");
    s.no_elements = true;
    s.cols.set("n_members", *at.n_members);
    dictionary_of(sub, d, at, s, out);
    params_of(s, at.params);
    out.id = sub.selection(std::move(s));
    return out;
  }
  if (every(d.members, is_span_member)) {   // spans as member rows (an empty set too)
    SelectionSpec s = base(d, at, "spans", "1");
    std::vector<SpanSpec> rows;
    for (const Description& m : d.members) rows.push_back(SpanSpec{m.extents[0].start, m.extents[0].stop, m.code});
    s.spans = rows;
    dictionary_of(sub, d, at, s, out);
    params_of(s, at.params);
    out.id = sub.selection(std::move(s));
    return out;
  }
  // members of any other kind: member selections
  SelectionSpec s = base(d, at, "set", "0");
  s.no_elements = true;
  s.cols.set("n_members", static_cast<std::int64_t>(d.members.size()));
  dictionary_of(sub, d, at, s, out);
  params_of(s, at.params);
  out.id = sub.selection(std::move(s));
  for (std::size_t k = 0; k < d.members.size(); ++k) {
    Placement mp;
    mp.name = at.name + "#" + std::to_string(k);
    mp.manifold_id = at.manifold_id;
    mp.created_by = at.created_by;
    mp.cell = at.cell;
    mp.cols.set("member_of_id", out.id).set("member_ordinal", static_cast<std::int64_t>(k));
    out.member_ids.push_back(save(sub, d.members[k], mp).id);
  }
  return out;
}

// ── read ─────────────────────────────────────────────────────────────────────────────────────────────────────────
namespace {

Description describe(const Database& db, const Store& store, const records::SelectionRow& row) {
  const Selection e = Selection::read(db, store, row.id);
  Description d;
  d.code = row.code;
  d.unit = row.unit.value_or("");
  const Json& p = row.params_json;
  const std::string& t = row.type;
  if (t == "point") {
    d.type = Type::point;
    d.center = row.center;
  } else if (t == "span") {
    d.type = Type::span;
    for (const auto& x : e.extents()) d.extents.push_back(Extent{static_cast<int>(x.axis), x.start, x.stop});
  } else if (t == "distance") {
    d.type = Type::distance;
    d.center = row.center;
    d.radius = row.radius.value_or(0.0);
  } else if (t == "indices") {
    d.type = Type::indices;
    if (row.array_path) d.indices = e.elements();
  } else if (t == "spans") {
    d.type = Type::set;
    if (const auto tl = e.tiling()) {
      d.tiling = Tiling{tl->origin, tl->width, tl->hop, tl->count};
      if (p.is_object() && p.contains("window_samples") && p["window_samples"].is_number()) d.level = constantq::levelOfWidth(p["window_samples"].get<std::int64_t>());
      if (p.is_object() && p.contains("resolves_hz") && p["resolves_hz"].is_object()) { d.band = std::make_pair(p["resolves_hz"].value("lo", 0.0), p["resolves_hz"].value("hi", 0.0)); d.band_unit = "Hz"; }
    } else {
      for (const auto& s : records::Selections(db).spans(row.id)) {
        Description m;
        m.type = Type::span;
        m.code = s.code;
        m.unit = d.unit;
        m.extents.push_back(Extent{0, s.start, s.stop});
        d.members.push_back(std::move(m));
      }
    }
  } else if (t == "set" || t == "ladder") {
    d.type = Type::set;
    const bool has_level = has_column(db, "selection_factor", "level");   // a database at 50 to 52 has none (the factors are as they are)
    const auto factors = db.all(has_level ? "SELECT factor_selection_id, level FROM selection_factor WHERE selection_id = ? ORDER BY ordinal" : "SELECT factor_selection_id FROM selection_factor WHERE selection_id = ? ORDER BY ordinal", {V(row.id)});
    const Json grid = p.is_object() && p.contains("grid") && p["grid"].is_object() ? p["grid"] : Json();
    if (!factors.empty()) {   // a product: its factors, each a stored grid, and their radices (params_json.product; else their member counts)
      require_member_storage(db);
      const Json radices = p.is_object() && p.contains("product") && p["product"].is_array() ? p["product"] : Json();
      for (std::size_t k = 0; k < factors.size(); ++k) {
        const auto fr = records::Selections(db).read(text(factors[k], "factor_selection_id"));
        if (!fr) throw Error(code::no_such_row, "selection " + row.name + ": a factor has no row");
        d.radices.push_back(radices.is_array() && k < radices.size() ? radices[k].value("members", std::int64_t(0)) : fr->n_members.value_or(1));
        d.factor_levels.push_back(has_level && !is_null(factors[k], "level") ? static_cast<int>(integer(factors[k], "level")) : -1);
        d.factors.push_back(describe(db, store, *fr));
      }
    } else if (t == "ladder" && grid.is_object() && grid.value("kind", std::string()) == "time") {   // a line's time grid: closed form, nothing stored per level
      d.grid = Grid{Grid::Kind::time, grid.value("samples", std::int64_t(0)), 0, 0, -1, std::nullopt, std::nullopt, grid.value("origin", 0.0), grid.value("rate", 0.0)};
    } else if (t == "ladder" && grid.is_object() && grid.value("kind", std::string()) == "dyadic") {
      // a dyadic grid saved by save_grid: ONE description, its finest level's codes, the closed form, the levels as descriptions (label = the level)
      const auto levels = e.members();
      if (levels.empty()) refuse("selection " + row.name + ": a dyadic grid with no levels");
      const records::SelectionRow& finest = levels.back();
      if (!finest.array_path) refuse("selection " + row.name + ": its finest level stores no tile codes");
      const int depth = grid.value("depth", static_cast<int>(levels.size()) - 1);
      d.member_of = Selection::read(db, store, finest.id).elements();
      d.grid = Grid{Grid::Kind::dyadic, 0, depth, finest.n_members.value_or(0) >> depth};   // the slots the codes reserve: the finest level's members >> depth
      if (grid.contains("finest") && grid["finest"].is_number_integer()) d.grid->finest = grid["finest"].get<int>();
      // a level's BAND (the eigenvalues its tiles add, Weyl's law) lives in the level dictionary's entry (attributes lambda_min / lambda_max, 1/unit^2): reloaded here
      std::map<std::int64_t, Json> attributes;
      if (row.dictionary_id)
        for (const auto& e : lookup_entries(db, *row.dictionary_id)) attributes[e.code] = e.attributes_json;
      for (const auto& l : levels) {
        Description level;
        level.type = Type::set;
        level.family = Family::grid;
        level.level = static_cast<int>(l.code.value_or(l.member_ordinal.value_or(0)));
        level.code = level.level;
        const auto a = attributes.find(level.level);
        if (a != attributes.end() && a->second.is_object() && a->second.contains("lambda_min") && a->second.contains("lambda_max")) {
          level.band = std::make_pair(a->second["lambda_min"].get<double>(), a->second["lambda_max"].get<double>());
          level.band_unit = a->second.value("lambda_unit", std::string("1/m^2"));
        }
        if (a != attributes.end() && a->second.is_object() && a->second.contains("diameter") && a->second["diameter"].is_number()) level.diameter = a->second["diameter"].get<double>();
        d.members.push_back(std::move(level));
      }
    } else if (row.array_path) {   // a set per element: member numbers; an irregular set's labels beside them, or a dyadic grid's rule
      require_member_storage(db);
      d.member_of = e.elements();
      if (grid.is_object() && grid.value("kind", std::string()) == "dyadic") {
        const int depth = grid.value("depth", 0);
        d.grid = Grid{Grid::Kind::dyadic, 0, depth, row.n_members.value_or(0) >> depth};
      } else {
        const std::string labels = member_labels_path(*row.array_path);
        if (store.exists_prefix(labels)) d.member_codes = read_ints(store, labels);   // absent: member i is labelled i
      }
    } else if (grid.is_object() && grid.value("kind", std::string()) == "elements") {   // an elements grid: nothing stored
      d.grid = Grid{Grid::Kind::elements, grid.value("size", row.n_members.value_or(0)), 0, 0};
    } else {
      for (const auto& m : e.members()) d.members.push_back(describe(db, store, m));
    }
  } else if (t == "kernel") {
    const std::string anchor = p.is_object() ? p.value("anchor", std::string()) : std::string();
    if (anchor == "span") {
      d.type = Type::span;
      for (const auto& x : e.extents()) d.extents.push_back(Extent{static_cast<int>(x.axis), x.start, x.stop});
    } else if (anchor == "point") {
      d.type = Type::point;
      d.center = row.center;
    } else if (anchor == "distance") {
      d.type = Type::distance;
      d.center = row.center;
      d.radius = row.radius.value_or(0.0);
    } else {
      refuse("selection " + row.name + ": a kernel's anchor is not recorded (params_json.anchor)");
    }
    if (!p.contains("profile") || !p["profile"].is_object() || !p.contains("chart") || !p.contains("gain")) refuse("selection " + row.name + ": a kernel's profile, chart and gain are not recorded");
    Profile pr;
    pr.family = p["profile"].value("family", row.weight_fn.value_or(""));
    if (p["profile"].contains("params") && p["profile"]["params"].is_object())
      for (const auto& [k, v] : p["profile"]["params"].items()) pr.params[k] = v.get<double>();
    pr.chart = p["chart"].get<std::string>();
    pr.gain = p["gain"].get<double>();
    d.profile = std::move(pr);
  } else {
    refuse("selection " + row.name + ": unknown type " + t);
  }
  if (row.dictionary_id && t != "point" && t != "span" && t != "distance") d.lookup = load_lookup(db, *row.dictionary_id);
  if (d.type == Type::set) {   // the family as stored; a row from before the mark says what its structure says
    if (p.is_object() && p.contains("family") && p["family"].is_string()) d.family = p["family"].get<std::string>() == "grid" ? Family::grid : Family::irregular;
    else d.family = t == "ladder" ? Family::grid : derived_family(d);
  }
  return d;
}

}  // namespace

Loaded load(const Database& db, const Store& store, std::string_view id) {
  const auto r = records::Selections(db).read(id);
  if (!r) throw Error(code::no_such_row, "selection " + std::string(id) + ": no such row");
  return Loaded{*r, describe(db, store, *r)};
}

bool is_grid(const Database& db, const std::string& id) {
  const auto row = db.read("selection", id);
  if (!row) throw Error(code::no_such_row, "selection " + id + ": no such row");
  return row_is_grid(db, *row);
}

std::vector<std::int64_t> product_radices(const Database& db, const std::string& id) {
  const auto row = records::Selections(db).read(id);
  if (!row) throw Error(code::no_such_row, "selection " + id + ": no such row");
  const auto factors = db.all("SELECT factor_selection_id FROM selection_factor WHERE selection_id = ? ORDER BY ordinal", {V(id)});
  if (factors.empty()) refuse("selection " + row->name + ": it is not a product");
  const Json& p = row->params_json;
  const Json radices = p.is_object() && p.contains("product") && p["product"].is_array() ? p["product"] : Json();
  std::vector<std::int64_t> out;
  for (std::size_t k = 0; k < factors.size(); ++k) {
    if (radices.is_array() && k < radices.size()) { out.push_back(radices[k].value("members", std::int64_t(0))); continue; }
    const auto fr = records::Selections(db).read(text(factors[k], "factor_selection_id"));
    out.push_back(fr ? fr->n_members.value_or(1) : 1);
  }
  return out;
}

std::int64_t coarser_product_code(const std::vector<std::int64_t>& radices, std::int64_t code, int factor, int shift) {
  if (factor < 0 || factor >= static_cast<int>(radices.size()) || shift < 0 || shift > 62 || code < 0) refuse("coarser: a factor of the product, a shift 0..62 and a code >= 0");
  std::vector<std::int64_t> at;
  std::int64_t rest = code;
  for (const std::int64_t n : radices) {
    if (n < 1) refuse("coarser: a factor with no members");
    at.push_back(rest % n);
    rest /= n;
  }
  if (rest != 0) refuse("coarser: code " + i2s(code) + " is past the product of the radices");
  at[static_cast<std::size_t>(factor)] >>= shift;
  std::vector<std::int64_t> rolled = radices;
  rolled[static_cast<std::size_t>(factor)] = (radices[static_cast<std::size_t>(factor)] + (std::int64_t(1) << shift) - 1) >> shift;
  std::int64_t out = 0, scale = 1;
  for (std::size_t i = 0; i < rolled.size(); ++i) { out += at[i] * scale; scale *= rolled[i]; }
  return out;
}

std::optional<records::SelectionRow> find_by_name(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& stored_type) {
  std::string sql = "SELECT * FROM selection WHERE subject_id = ? AND name = ? AND status = 'complete'";
  std::vector<Value> a = {V(subject_id), V(name)};
  if (stored_type) { sql += " AND type = ?"; a.push_back(V(*stored_type)); }
  const auto r = db.one(sql, a);
  return r ? std::optional<records::SelectionRow>(records::SelectionRow::from_row(*r)) : std::nullopt;
}

std::optional<records::SelectionRow> find_on_manifold(const Database& db, const std::string& manifold_id, const std::string& name, const std::optional<std::string>& stored_type) {
  std::string sql = "SELECT * FROM selection WHERE manifold_id = ? AND name = ? AND status = 'complete'";
  std::vector<Value> a = {V(manifold_id), V(name)};
  if (stored_type) { sql += " AND type = ?"; a.push_back(V(*stored_type)); }
  const auto r = db.one(sql, a);
  return r ? std::optional<records::SelectionRow>(records::SelectionRow::from_row(*r)) : std::nullopt;
}

std::optional<Saved> find_with_param(const Database& db, const std::string& subject_id, const std::string& manifold_id, const std::string& stored_type, const std::string& param_key) {
  for (const Row& s : db.all("SELECT * FROM selection WHERE subject_id = ? AND type = ? AND manifold_id = ? AND status = 'complete'", {V(subject_id), V(stored_type), V(manifold_id)})) {
    if (opt_text(s, "params_json").value_or("").find("\"" + param_key + "\"") == std::string::npos) continue;
    Saved out;
    out.id = text(s, "id");
    out.dictionary_id = opt_text(s, "dictionary_id");
    for (const Row& m : db.all("SELECT id FROM selection WHERE member_of_id = ? ORDER BY member_ordinal", {V(out.id)})) out.member_ids.push_back(text(m, "id"));
    return out;
  }
  return std::nullopt;
}

std::vector<records::SelectionSpanRow> spans_of(const Database& db, std::string_view selection_id) { return records::Selections(db).spans(selection_id); }

TimeLine time_line(const Database& db, const std::string& manifold_id) {
  const auto m = db.read("manifold", manifold_id);
  if (!m) throw Error(code::no_such_row, "time line: manifold " + manifold_id + " has no row");
  if (text(*m, "type") != "line" || is_null(*m, "geometry_id")) throw Error(code::invalid_argument, "time line: '" + text(*m, "name") + "' is not a closed-form line");
  const auto d = db.one("SELECT n_vertices, spacing, origin FROM geometry_dimension WHERE geometry_id = ? ORDER BY ordinal LIMIT 1", {V(text(*m, "geometry_id"))});
  if (!d || is_null(*d, "spacing")) throw Error(code::invalid_argument, "time line: '" + text(*m, "name") + "' has no closed-form dimension");
  TimeLine t;
  t.name = text(*m, "name");
  t.count = integer(*d, "n_vertices");
  t.spacing = std::stod(text(*d, "spacing"));
  t.origin = is_null(*d, "origin") ? 0.0 : std::stod(text(*d, "origin"));
  t.acquisition_id = opt_text(*m, "acquisition_id");
  return t;
}

// ── update ───────────────────────────────────────────────────────────────────────────────────────────────────────
void update(SubjectWriter& sub, const std::string& id, const Description& d) {
  const auto row = records::Selections(sub.db()).read(id);
  if (!row) throw Error(code::no_such_row, "selection " + id + ": no such row");
  if (row->subject_id != sub.id()) refuse("selection " + row->name + ": it belongs to another subject");
  const std::string& t = row->type;
  const bool fits = (t == "point" && d.type == Type::point) || (t == "span" && d.type == Type::span) || (t == "distance" && d.type == Type::distance) ||
                    (t == "indices" && d.type == Type::indices) || ((t == "spans" || t == "set" || t == "ladder") && d.type == Type::set);
  if (!fits) refuse("selection " + row->name + ": stored as " + t + ", it cannot be updated to a " + type_name(d.type) + " (a type is not changed)");
  if (t == "ladder" || (t == "set" && !row->array_path && row->n_members.value_or(0) > 0 && !d.members.empty()))
    refuse("selection " + row->name + ": a set with member selections is updated member by member (update each member by its id)");
  std::vector<std::int32_t> array = t == "indices" ? d.indices : d.member_of;   // kept alive for the array write
  const std::string stamp = now_utc();
  sub.update([&] {
    Database& db = sub.db();
    if (t == "point") {
      if (!d.center) refuse("selection " + row->name + ": a point has a centre element");
      db.execute("UPDATE selection SET center = ? WHERE id = ?", {V(*d.center), V(id)});
    } else if (t == "distance") {
      if (!d.center || !(d.radius >= 0)) refuse("selection " + row->name + ": a distance has a centre element and a radius >= 0");
      db.execute("UPDATE selection SET center = ?, radius = ? WHERE id = ?", {V(*d.center), V(d.radius), V(id)});
    } else if (t == "span") {
      std::vector<ExtentSpec> ex;
      for (const Extent& e : d.extents) ex.push_back(ExtentSpec{e.axis, e.start, e.stop});
      replace_extents(db, id, ex);
    } else if (t == "indices") {
      if (!row->array_path) refuse("selection " + row->name + ": no array is stored for its elements");
      sub.write_array(*row->array_path, view(array, {static_cast<std::int64_t>(array.size())}));
      db.execute("UPDATE selection SET n_elements = ? WHERE id = ?", {V(static_cast<std::int64_t>(array.size())), V(id)});
    } else if (t == "spans") {
      if (d.tiling) {
        Json p = row->params_json.is_object() ? row->params_json : Json::object();
        p["tiling"] = tiling_json(*d.tiling);
        db.execute("DELETE FROM selection_span WHERE selection_id = ?", {V(id)});
        db.execute("UPDATE selection SET params_json = ?, n_members = ? WHERE id = ?", {V(jdump(p)), V(d.tiling->count), V(id)});
      } else {
        if (!every(d.members, is_span_member)) refuse("selection " + row->name + ": a spans set's members are spans");
        std::vector<SpanSpec> rows;
        for (const Description& m : d.members) rows.push_back(SpanSpec{m.extents[0].start, m.extents[0].stop, m.code});
        replace_spans(db, id, rows);
      }
    } else if (!d.radices.empty()) {   // a product: its radices (the factors stay; a radix is at least its factor's member count), n_members the product of the radices
      const auto factors = db.all("SELECT factor_selection_id FROM selection_factor WHERE selection_id = ? ORDER BY ordinal", {V(id)});
      if (factors.empty()) refuse("selection " + row->name + ": it is not a product");
      if (d.radices.size() != factors.size()) refuse("selection " + row->name + ": one radix per factor");
      Json p = row->params_json.is_object() ? row->params_json : Json::object();
      Json product = Json::array();
      std::int64_t members = 1;
      for (std::size_t k = 0; k < factors.size(); ++k) {
        const std::string fid = text(factors[k], "factor_selection_id");
        const auto fr = db.read("selection", fid);
        const std::int64_t count = !fr || is_null(*fr, "n_members") ? 1 : std::max<std::int64_t>(0, integer(*fr, "n_members"));
        const std::int64_t radix = d.radices[k];
        if (radix < count) refuse("selection " + row->name + ": factor " + i2s(static_cast<std::int64_t>(k)) + " has " + i2s(count) + " members, more than its radix " + i2s(radix));
        if (radix > 0 && members > std::numeric_limits<std::int64_t>::max() / radix) refuse("selection " + row->name + ": more members than a 64-bit code holds");
        members *= radix;
        product.push_back(Json{{"selection", fid}, {"members", radix}});
      }
      p["product"] = std::move(product);
      db.execute("UPDATE selection SET params_json = ?, n_members = ? WHERE id = ?", {V(jdump(p)), V(members), V(id)});
    } else if (d.grid && d.grid->kind == Grid::Kind::elements && !row->array_path) {   // an elements grid: its size is the rule
      Json p = row->params_json.is_object() ? row->params_json : Json::object();
      p["grid"] = Json{{"kind", "elements"}, {"size", d.grid->size}};
      db.execute("UPDATE selection SET params_json = ?, n_members = ? WHERE id = ?", {V(jdump(p)), V(d.grid->size), V(id)});
    } else {   // set per element: the member numbers and the labels beside them (the dictionary follows, below)
      if (!row->array_path) refuse("selection " + row->name + ": a set with no array and no members has nothing to update");
      if (!d.factors.empty()) refuse("selection " + row->name + ": a product's factors are not updated (give its radices to change them, or remove it and save it again)");
      const bool dyadic = d.grid && d.grid->kind == Grid::Kind::dyadic;
      const std::vector<std::int64_t> labels = dyadic ? std::vector<std::int64_t>{} : labels_of(d);
      // a dyadic grid: its numbers are tile codes, slots << depth members (empty tiles too), no labels; an irregular set: dense numbers, a label per member
      const std::int64_t n = dyadic ? (d.grid->slots << d.grid->depth) : static_cast<std::int64_t>(labels.size());
      for (const std::int32_t m : array)
        if (m < -1 || m >= n) refuse("selection " + row->name + ": member number " + i2s(m) + " outside 0.." + i2s(n - 1));
      sub.write_array(*row->array_path, view(array, {static_cast<std::int64_t>(array.size())}));
      const std::string lp = member_labels_path(*row->array_path);
      if (dyadic || is_identity(labels)) sub.store().erase_prefix(lp);   // a grid's member is named by its number; member i labelled i: nothing beside the element array
      else sub.write_array(lp, view(labels, {n}));
      if (dyadic) {
        Json p = row->params_json.is_object() ? row->params_json : Json::object();
        p["grid"] = Json{{"kind", "dyadic"}, {"depth", d.grid->depth}, {"slots", d.grid->slots}};
        db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(id)});
      }
      db.execute("UPDATE selection SET n_elements = ?, n_members = ? WHERE id = ?", {V(static_cast<std::int64_t>(array.size())), V(n), V(id)});
    }
    if (d.type == Type::set && d.family != Family::unmarked) {   // the family the caller states
      const auto now = records::Selections(db).read(id);
      Json p = now && now->params_json.is_object() ? now->params_json : Json::object();
      p["family"] = family_name(d.family);
      db.execute("UPDATE selection SET params_json = ? WHERE id = ?", {V(jdump(p)), V(id)});
    }
    if (d.code) db.execute("UPDATE selection SET code = ? WHERE id = ?", {V(*d.code), V(id)});
    if (!d.unit.empty()) db.execute("UPDATE selection SET unit = ? WHERE id = ?", {V(d.unit), V(id)});
    db.execute("UPDATE selection SET modified_utc = ? WHERE id = ?", {V(stamp), V(id)});
  });
  if (t == "set" && !d.member_of.empty() && !(d.grid && d.grid->kind == Grid::Kind::dyadic)) {   // a set per element: its dictionary is exactly the description's lookup (one row per label; none: the labels are their own names)
    if (row->dictionary_id) update_lookup(sub, *row->dictionary_id, d.lookup);
  } else if (!d.lookup.empty()) {
    if (row->dictionary_id) update_lookup(sub, *row->dictionary_id, d.lookup);
    else {
      const std::string did = save_lookup(sub, row->name, d.lookup);
      sub.update([&] { sub.db().execute("UPDATE selection SET dictionary_id = ? WHERE id = ?", {V(did), V(id)}); });
    }
  }
}

// ── delete ───────────────────────────────────────────────────────────────────────────────────────────────────────
void remove(SubjectWriter& sub, const std::string& id) { sub.remove("selection", id); }

// ── lookups of other layers ──────────────────────────────────────────────────────────────────────────────────────
std::int64_t lookup_size(const Database& db, const std::string& dictionary_id) {
  return integer(*db.one("SELECT COUNT(*) AS n FROM dictionary_entry WHERE dictionary_id = ?", {V(dictionary_id)}), "n");
}

std::optional<std::string> find_lookup(const Database& db, const std::string& scope, const std::string& name) {
  const auto r = db.one("SELECT id FROM dictionary WHERE scope = ? AND name = ? ORDER BY id LIMIT 1", {V(scope), V(name)});
  return r ? std::optional<std::string>(text(*r, "id")) : std::nullopt;
}

std::vector<records::DictionaryEntryRow> lookup_entries(const Database& db, const std::string& dictionary_id) {
  std::vector<records::DictionaryEntryRow> out;
  for (const Row& r : db.all("SELECT * FROM dictionary_entry WHERE dictionary_id = ? ORDER BY ordinal, code", {V(dictionary_id)})) out.push_back(records::DictionaryEntryRow::from_row(r));
  return out;
}

bool lookup_in_use(const Database& db, const std::string& dictionary_id) { return db.one("SELECT 1 FROM selection WHERE dictionary_id = ? LIMIT 1", {V(dictionary_id)}).has_value(); }

std::optional<records::SelectionRow> find_member(const Database& db, const std::string& parent_id, const std::string& name) {
  const auto r = db.one("SELECT * FROM selection WHERE member_of_id = ? AND name = ?", {V(parent_id), V(name)});
  return r ? std::optional<records::SelectionRow>(records::SelectionRow::from_row(*r)) : std::nullopt;
}

std::optional<records::SelectionRow> find_on_manifold_ending(const Database& db, const std::string& manifold_id, const std::string& suffix) {
  std::string like = "%";
  for (const char c : suffix) { if (c == '_' || c == '%' || c == '\\') like += '\\'; like += c; }
  const auto r = db.one("SELECT * FROM selection WHERE manifold_id = ? AND name LIKE ? ESCAPE '\\' AND member_of_id IS NULL", {V(manifold_id), V(like)});
  return r ? std::optional<records::SelectionRow>(records::SelectionRow::from_row(*r)) : std::nullopt;
}

std::optional<records::SelectionRow> find_of_field(const Database& db, const std::string& subject_id, const std::string& of_field_id, const std::string& name) {
  const auto r = db.one("SELECT * FROM selection WHERE subject_id = ? AND of_field_id = ? AND name = ? AND status = 'complete'", {V(subject_id), V(of_field_id), V(name)});
  return r ? std::optional<records::SelectionRow>(records::SelectionRow::from_row(*r)) : std::nullopt;
}

std::vector<std::string> complete_ids(const Database& db, const std::string& subject_id) {
  std::vector<std::string> out;
  for (const Row& r : db.all("SELECT id FROM selection WHERE subject_id = ? AND status = 'complete'", {V(subject_id)})) out.push_back(text(r, "id"));
  return out;
}

void remove_all(Database& db, const std::string& subject_id) { db.execute("DELETE FROM selection WHERE subject_id = ?", {V(subject_id)}); }

// ── the arrays and paths a selection names ───────────────────────────────────────────────────────────────────────
std::vector<SubjectArray> array_paths_in_dataset(const Query& all) {
  std::vector<SubjectArray> out;
  for (const Row& r : all("SELECT s.path AS sp, s.name AS sn, r.array_path AS p FROM selection r JOIN subject s ON s.id = r.subject_id WHERE r.array_path IS NOT NULL"))
    out.push_back({text(r, "sp"), text(r, "sn"), text(r, "p")});
  return out;
}

std::vector<std::pair<std::string, std::string>> named_paths_of(const Database& db, const std::string& subject_id) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const char* col : {"path", "array_path"})
    for (const Row& r : db.all(std::string("SELECT id, ") + col + " AS p FROM selection WHERE subject_id = ?", {V(subject_id)}))
      if (!is_null(r, "p")) out.emplace_back(text(r, "id"), text(r, "p"));
  // the array beside a set per element's element array: its members' labels (named whether or not the set has labels: nothing exists at an unused name)
  for (const Row& r : db.all("SELECT id, array_path AS p FROM selection WHERE subject_id = ? AND type = 'set' AND array_path IS NOT NULL", {V(subject_id)}))
    out.emplace_back(text(r, "id"), member_labels_path(text(r, "p")));
  return out;
}

std::vector<std::pair<std::string, std::string>> params_naming_arrays(const Database& db, const std::string& subject_id) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const Row& r : db.all("SELECT id, params_json AS j FROM selection WHERE subject_id = ? AND (params_json LIKE '%\"weights\"%' OR params_json LIKE '%\"atlas\"%')", {V(subject_id)}))
    out.emplace_back(text(r, "id"), text(r, "j"));
  return out;
}

// ── the rows as they travel ──────────────────────────────────────────────────────────────────────────────────────
std::vector<Row> subject_rows(const Database& db, const std::string& subject_id) {
  return db.all("SELECT * FROM selection WHERE subject_id = ? AND status = 'complete' ORDER BY path, name", {V(subject_id)});
}

std::vector<Row> subject_extent_rows(const Database& db, const std::string& subject_id) {
  return db.all("SELECT e.* FROM selection_extent e JOIN selection s ON s.id = e.selection_id WHERE s.subject_id = ? ORDER BY e.selection_id, e.axis", {V(subject_id)});
}

std::vector<Row> subject_span_rows(const Database& db, const std::string& subject_id) {
  return db.all("SELECT p.* FROM selection_span p JOIN selection s ON s.id = p.selection_id WHERE s.subject_id = ? ORDER BY p.selection_id, p.ordinal", {V(subject_id)});
}

std::vector<Row> subject_factor_rows(const Database& db, const std::string& subject_id) {
  if (!has_table(db, "selection_factor")) return {};   // a dataset of schema 50 or 51, read as it is: no product has factors
  return db.all("SELECT f.* FROM selection_factor f JOIN selection s ON s.id = f.selection_id WHERE s.subject_id = ? ORDER BY f.selection_id, f.ordinal", {V(subject_id)});
}

// ── the app-wide vocabularies ────────────────────────────────────────────────────────────────────────────────────
std::string ensure_app_lookup(Database& db, const AppLookup& spec) {
  const auto write_entries = [&](const std::string& id) {
    db.execute("DELETE FROM dictionary_entry WHERE dictionary_id = ?", {V(id)});
    std::int64_t k = 0;
    for (const AppEntry& e : spec.entries) {
      Row r = {{"dictionary_id", id}, {"code", V(e.code)}, {"name", e.name}, {"ordinal", V(e.ordinal.value_or(k))}};
      ++k;
      if (e.color) r["color_json"] = jdump(color_json(e.color));
      if (e.measurement) r["measurement"] = *e.measurement;
      if (!e.measurement_unit.empty()) r["measurement_unit"] = e.measurement_unit;
      if (!e.function.empty()) r["function"] = e.function;
      if (!e.params.is_null()) r["params_json"] = jdump(e.params);
      if (!e.attributes.is_null()) r["attributes_json"] = jdump(e.attributes);
      db.insert("dictionary_entry", r);
    }
  };
  if (const auto have = db.one("SELECT * FROM dictionary WHERE scope = ? AND name = ?", {V(spec.scope), V(spec.name)})) {
    const std::string id = text(*have, "id");
    if (spec.refresh_when_incomplete) {
      std::set<std::string> known;
      for (const Row& e : db.all("SELECT name FROM dictionary_entry WHERE dictionary_id = ?", {V(id)})) known.insert(text(e, "name"));
      bool missing = false;
      for (const AppEntry& e : spec.entries) if (!known.count(e.name)) missing = true;
      if (missing) db.tx([&] { write_entries(id); });
    }
    return id;
  }
  const std::string id = uuid7();
  db.tx([&] {
    Row r = {{"id", id}, {"name", spec.name}, {"scope", spec.scope}, {"description", spec.description}};
    if (spec.by) r["created_by"] = r["modified_by"] = *spec.by;
    db.insert("dictionary", r);
    write_entries(id);
  });
  return id;
}

namespace {
std::string marks(std::size_t n) {
  std::string s;
  for (std::size_t i = 0; i < n; ++i) s += i ? ", ?" : "?";
  return s;
}
std::vector<Value> values_of(const std::vector<std::string>& ids) { return std::vector<Value>(ids.begin(), ids.end()); }
}  // namespace

std::vector<Row> vocabulary_rows(const Database& db, const std::vector<std::string>& names, const std::string& dataset_id, const std::string& subject_id) {
  if (names.empty()) return {};
  std::vector<Value> a = values_of(names);
  a.push_back(V(dataset_id));
  a.push_back(V(subject_id));
  return db.all("SELECT * FROM dictionary WHERE name IN (" + marks(names.size()) + ") AND (scope IN ('app', 'user') OR (scope = 'dataset' AND dataset_id = ?) OR (scope = 'subject' AND subject_id = ?))", a);
}

std::vector<Row> dictionary_rows(const Database& db, const std::vector<std::string>& ids) {
  if (ids.empty()) return {};
  return db.all("SELECT * FROM dictionary WHERE id IN (" + marks(ids.size()) + ")", values_of(ids));
}

std::vector<Row> dictionary_entry_rows(const Database& db, const std::vector<std::string>& ids) {
  if (ids.empty()) return {};
  return db.all("SELECT * FROM dictionary_entry WHERE dictionary_id IN (" + marks(ids.size()) + ") ORDER BY dictionary_id, code", values_of(ids));
}

std::vector<SharedLookup> shared_lookups(const Database& db) {
  std::vector<SharedLookup> out;
  const auto cols = db.all("SELECT name FROM pragma_table_info('dictionary_entry') WHERE name <> 'dictionary_id' ORDER BY cid");
  std::string list;
  for (const auto& c : cols) list += (list.empty() ? "" : ", ") + std::string("\"") + text(c, "name") + "\"";
  for (const auto& d : db.all("SELECT id, scope, name FROM dictionary WHERE scope IN ('app', 'dataset') ORDER BY scope, name")) {
    Json entries = Json::array();
    for (const auto& e : db.all("SELECT " + list + " FROM dictionary_entry WHERE dictionary_id = ? ORDER BY code", {V(text(d, "id"))})) {
      Json a = Json::array();
      for (const auto& c : cols) {
        const Value& v = e.at(text(c, "name"));
        if (std::holds_alternative<std::monostate>(v)) a.push_back(nullptr);
        else if (const auto* n = std::get_if<std::int64_t>(&v)) a.push_back(*n);
        else if (const auto* x = std::get_if<double>(&v)) a.push_back(*x);
        else a.push_back(std::get<std::string>(v));
      }
      entries.push_back(a);
    }
    out.push_back({text(d, "scope"), text(d, "name"), std::move(entries)});
  }
  return out;
}

std::vector<std::pair<std::string, std::string>> dictionary_keys(const Database& db) {
  std::vector<std::pair<std::string, std::string>> out;
  for (const Row& r : db.all("SELECT id, scope || '/' || name AS shared FROM dictionary WHERE scope IN ('app', 'dataset', 'user') UNION ALL SELECT id, NULL FROM dictionary WHERE scope = 'subject'"))
    out.emplace_back(text(r, "id"), opt_text(r, "shared").value_or(""));
  return out;
}

}  // namespace nxr::io::selection
