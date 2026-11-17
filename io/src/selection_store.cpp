// nxr-io — the selection CRUD (selection_store.h): save, load, update and delete a selection of any of the five types, its lookup, and the rows beneath it.
// io stores what it is given and designs nothing: a ladder's levels arrive as tilings with their bands (made by compute); here they are only written.
#include "nxr/io/selection_store.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

#include "nxr/io/array_writer.h"
#include "nxr/io/constant_q.h"
#include "nxr/io/errors.h"
#include "nxr/io/json_text.h"
#include "nxr/io/records/records.h"
#include "nxr/io/selection.h"

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

namespace {

[[noreturn]] void refuse(const std::string& m) { throw Error(code::invalid_argument, m); }
Value vopt(const std::optional<std::int64_t>& x) { return x ? Value(*x) : Value(std::monostate{}); }
std::string fixed(double x, int d) { char b[64]; std::snprintf(b, sizeof b, "%.*f", d, x); return b; }
std::string i2s(std::int64_t x) { return std::to_string(x); }

bool is_span_member(const Description& m) { return m.type == Type::span && m.extents.size() == 1 && m.extents[0].axis == 0; }
bool is_level(const Description& m) { return m.type == Type::set && m.tiling && m.level >= 0; }
template <class P>
bool every(const std::vector<Description>& v, P pred) { return std::all_of(v.begin(), v.end(), pred); }

Json color_json(const std::optional<std::array<int, 3>>& c) { return c ? Json::array({(*c)[0], (*c)[1], (*c)[2]}) : Json(); }

// the row's own columns, common to every type
SelectionSpec base(const Description& d, const Placement& at, const char* type, const char* default_cell) {
  SelectionSpec s;
  s.name = at.name; s.path = at.path; s.type = type; s.manifold_id = at.manifold_id; s.id = at.id; s.created_utc = at.created_utc; s.acquisition = at.acquisition;
  s.cols = at.cols;
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

Json tiling_json(const Tiling& t) { return Json{{"origin", t.origin}, {"width", t.width}, {"hop", t.hop}, {"count", t.count}}; }

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
  // the LEVEL dictionary: code = the level's ordinal (0 the finest); an entry carries the window in samples and seconds, the tile count, the finest tiles one tile holds
  // and the band it resolves (what the window resolves is derived and referenced by nothing)
  DictionarySpec ds;
  ds.name = at.name;
  ds.description = "the levels of the ladder '" + at.name + "': code = the level, 0 the finest; an entry carries the level's window in samples and seconds, its tile count, "
                   "how many finest tiles one tile holds, and the band it resolves";
  std::vector<DictionaryEntry> entries;
  for (std::size_t k = 0; k < L; ++k) {
    const Description& m = d.members[k];
    const Tiling& t = *m.tiling;
    DictionaryEntry e;
    e.code = static_cast<std::int64_t>(k);
    e.ordinal = e.code;
    e.name = i2s(samples_of(t.width)) + " samples";
    for (const Entry& le : d.lookup) if (le.code == e.code) { if (!le.name.empty()) e.name = le.name; e.color = le.color; }
    Json info{{"level", e.code}, {"power", m.level}, {"window_samples", samples_of(t.width)}, {"window_s", t.width}, {"hop_samples", samples_of(t.hop)}, {"count", t.count},
              {"finest_tiles", samples_of(t.width) / std::max<std::int64_t>(1, finest)}, {"duration_s", t.width}, {"finest_window_samples", finest}};
    if (m.band) info["resolves_hz"] = Json{{"lo", m.band->first}, {"hi", m.band->second}};
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
        .set("code", static_cast<std::int64_t>(k)).set("dictionary_id", tile_dicts[k]).set("n_elements", tl.count).set("n_members", t.count);
    if (!s.cols.count("unit")) s.cols.set("unit", d.unit.empty() ? std::string("s") : d.unit);
    Json mj{{"tiling", tiling_json(t)}, {"end", end}, {"level", static_cast<std::int64_t>(k)}, {"window_samples", samples_of(t.width)}, {"hop_samples", samples_of(t.hop)},
            {"finest_tiles", samples_of(t.width) / std::max<std::int64_t>(1, finest)}};
    if (m.band) mj["resolves_hz"] = Json{{"lo", m.band->first}, {"hi", m.band->second}};
    s.cols.json("params_json", mj);
    out.member_ids.push_back(sub.selection(std::move(s)));
  }
  return out;
}

}  // namespace

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

// ── create ───────────────────────────────────────────────────────────────────────────────────────────────────────
Saved save(SubjectWriter& sub, const Description& d, const Placement& at) {
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
  if (!d.codes.empty()) {
    if (!at.path) refuse("selection " + at.name + ": a set by codes is stored as an array: give its path");
    std::vector<std::int32_t> codes = d.codes;
    SelectionSpec s = base(d, at, "set", "0");
    s.data = view(codes, {static_cast<std::int64_t>(codes.size())});
    Description named = d;
    if (named.lookup.empty()) {   // the codes are their own names
      std::set<std::int32_t> seen;
      for (const auto c : codes) if (c >= 0) seen.insert(c);
      for (const auto c : seen) named.lookup.push_back(Entry{c, std::to_string(c), std::nullopt});
    }
    dictionary_of(sub, named, at, s, out);
    Json params = at.params.is_object() ? at.params : Json::object();
    if (d.depth > 0) params["depth"] = d.depth;
    params_of(s, params);
    out.id = sub.selection(std::move(s));
    return out;
  }
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
    if (row.array_path) {
      d.codes = e.elements();
      if (p.is_object() && p.contains("depth") && p["depth"].is_number()) d.depth = p["depth"].get<int>();
    } else {
      for (const auto& m : e.members()) d.members.push_back(describe(db, store, m));
    }
  } else if (t == "kernel") {
    refuse("selection " + row.name + ": a kernel is not a selection (it is the weights a measurement uses)");
  } else {
    refuse("selection " + row.name + ": unknown type " + t);
  }
  if (row.dictionary_id && t != "point" && t != "span" && t != "distance") d.lookup = load_lookup(db, *row.dictionary_id);
  return d;
}

}  // namespace

Loaded load(const Database& db, const Store& store, std::string_view id) {
  const auto r = records::Selections(db).read(id);
  if (!r) throw Error(code::no_such_row, "selection " + std::string(id) + ": no such row");
  return Loaded{*r, describe(db, store, *r)};
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
  std::vector<std::int32_t> array = t == "indices" ? d.indices : d.codes;   // kept alive for the array write
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
    } else {   // set: by codes
      if (!row->array_path) refuse("selection " + row->name + ": a set with no array and no members has nothing to update");
      sub.write_array(*row->array_path, view(array, {static_cast<std::int64_t>(array.size())}));
      std::set<std::int32_t> distinct(array.begin(), array.end());
      db.execute("UPDATE selection SET n_elements = ?, n_members = ? WHERE id = ?", {V(static_cast<std::int64_t>(array.size())), V(static_cast<std::int64_t>(distinct.size())), V(id)});
    }
    if (d.code) db.execute("UPDATE selection SET code = ? WHERE id = ?", {V(*d.code), V(id)});
    if (!d.unit.empty()) db.execute("UPDATE selection SET unit = ? WHERE id = ?", {V(d.unit), V(id)});
    db.execute("UPDATE selection SET modified_utc = ? WHERE id = ?", {V(stamp), V(id)});
  });
  if (!d.lookup.empty()) {
    if (row->dictionary_id) update_lookup(sub, *row->dictionary_id, d.lookup);
    else {
      const std::string did = save_lookup(sub, row->name, d.lookup);
      sub.update([&] { sub.db().execute("UPDATE selection SET dictionary_id = ? WHERE id = ?", {V(did), V(id)}); });
    }
  }
}

// ── delete ───────────────────────────────────────────────────────────────────────────────────────────────────────
void remove(SubjectWriter& sub, const std::string& id) { sub.remove("selection", id); }

}  // namespace nxr::io::selection
