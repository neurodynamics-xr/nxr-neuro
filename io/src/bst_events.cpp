// nxr::bst — a recording's Brainstorm EVENTS as SELECTIONS (D143): nxr.convert.events. A table of occurrences (onset, duration, type, per-occurrence channels), written as
// spans Selections on the recording's time Line (`<rec>_events`, and `<rec>_bad_segments` for the types labelled bad*), `badsegments` and `events` onto existing recordings.
#include <algorithm>
#include <cmath>
#include <set>

#include "nxr/bst/entities.h"
#include "nxr/bst/pyrepr.h"
#include "nxr/bst/session.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/layout.h"

namespace nxr::bst {

namespace fs = std::filesystem;
namespace code = nxr::io::code;
using nxr::io::SubjectWriter;

namespace {

[[noreturn]] void value_error(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }
std::string lower(std::string s) { std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); }); return s; }

// the strings of a char / a cell of strings (nested cells flattened); non-strings and empty strings are dropped
void collect(const Value& v, std::vector<std::string>& out) {
  if (v.is_char()) { for (const auto& t : v.text) if (!t.empty()) out.push_back(t); }
  else if (v.is_cell()) for (const auto& e : v.elems) collect(e, out);
}

// rejoin_cellstr: pymatreader splits a one-element cell into characters; "P11" read as ['P','1','1'] is joined back
std::vector<std::string> rejoin(std::vector<std::string> items) {
  if (items.size() > 1 && std::all_of(items.begin(), items.end(), [](const std::string& s) { return s.size() == 1; })) {
    std::string j;
    for (const auto& s : items) j += s;
    return {j};
  }
  return items;
}

// np.squeeze, then the array row after row: the non-singleton dimensions of a (>2-D) array are its shape
std::vector<double> squeezed_flat(const Value* v, std::vector<std::size_t>* shape = nullptr) {
  std::vector<std::size_t> d;
  for (auto x : v->dims) if (x != 1) d.push_back(x);
  if (shape) *shape = d;
  const std::vector<double> a = v->doubles();
  if (d.size() == 2) {
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < d[0]; ++i) for (std::size_t j = 0; j < d[1]; ++j) out[i * d[1] + j] = a[i + j * d[0]];
    return out;
  }
  if (d.size() > 2) value_error("an array of more than two dimensions");
  return a;
}

std::vector<double> color_of(const Value* c) {
  std::vector<double> v = c && c->is_numeric() ? squeezed_flat(c) : std::vector<double>{};
  if (v.size() < 3) return {0.5, 0.5, 0.5};
  return {v[0], v[1], v[2]};
}

// pymatreader squeezes a one-cell array to its content (a 1x1 struct whose fields are 1x1 cells reads like a struct of values)
const Value* sq(const Value* v) {
  while (v && v->is_cell() && v->numel() == 1) v = &v->elems[0];
  return v;
}

struct Times { std::size_t rows = 1; std::vector<double> flat; };   // a (rows x n) matrix, row after row

// _times_matrix: the times as a (1 or 2) x nOccurrences matrix, `epochs` the disambiguator
Times times_matrix(const Value* times, const Value* epochs, const std::string& label) {
  std::vector<std::size_t> shape;
  std::vector<double> raw = times && times->is_numeric() ? squeezed_flat(times, &shape) : std::vector<double>{};
  if (epochs) {
    const std::size_t n_occ = epochs->numel();
    if (n_occ > 0 && raw.size() % n_occ == 0) {
      const std::size_t n_rows = raw.size() / n_occ;
      if (n_rows > 2)
        value_error("event " + str_repr(label) + ": times has " + std::to_string(raw.size()) + " elements over " + std::to_string(n_occ) + " occurrence(s) (epochs), implying " +
                    std::to_string(n_rows) + " rows \xe2\x80\x94 Brainstorm's times matrix is always 1 (simple) or 2 (extended) rows, so this shape cannot be reshaped safely; "
                    "refusing rather than silently dropping rows 2+");
      if (n_rows == 1 || n_rows == 2) return {n_rows, raw};
    }
  }
  // np.atleast_2d of the squeezed array: a matrix keeps its shape, anything else becomes a row
  Times t;
  t.flat = raw;
  if (shape.size() == 2) t.rows = shape[0];
  return t;
}

}  // namespace

EventTable parse_events(const Value* events, const std::vector<std::string>& channel_names) {
  std::map<std::string, std::int32_t> index;
  for (std::size_t i = 0; i < channel_names.size(); ++i) index[channel_names[i]] = static_cast<std::int32_t>(i);
  EventTable out;
  struct Row { double onset, duration; std::int32_t type; std::vector<std::int32_t> idx; };
  std::vector<Row> rows;
  if (events && events->is_struct()) {
    for (std::size_t e = 0; e < events->numel(); ++e) {
      const Value* lab = sq(events->field("label", e));
      const std::string label = lab && lab->is_char() ? lab->string() : "";
      const Times tm = times_matrix(sq(events->field("times", e)), sq(events->field("epochs", e)), label);
      if (tm.flat.empty()) continue;
      const std::int32_t ti = static_cast<std::int32_t>(out.type_names.size());
      out.type_names.push_back(label);
      out.type_colors.push_back(color_of(sq(events->field("color", e))));
      const std::size_t n = tm.flat.size() / tm.rows;
      const double* on = tm.flat.data();
      const double* off = tm.rows >= 2 ? tm.flat.data() + n : on;
      // per-occurrence channel scoping: a cell of one entry per occurrence, else the same for every occurrence
      const Value* ch = sq(events->field("channels", e));
      std::vector<std::vector<std::string>> per(n);
      if (ch && !(ch->numel() == 0)) {
        if (ch->is_cell() && ch->numel() == n) {
          for (std::size_t k = 0; k < n; ++k) { std::vector<std::string> items; collect(ch->elems[k], items); per[k] = rejoin(items); }
        } else {
          std::vector<std::string> items;
          collect(*ch, items);
          items = rejoin(items);
          for (std::size_t k = 0; k < n; ++k) per[k] = items;
        }
      }
      for (std::size_t k = 0; k < n; ++k) {
        std::vector<std::int32_t> idx;
        for (const auto& nm : per[k]) {
          const auto it = index.find(nm);
          if (it == index.end())
            value_error("event " + str_repr(label) + " references channel " + str_repr(nm) + ", which is not in the recording's channel list \xe2\x80\x94 refusing rather than dropping "
                        "the reference (known channels: " + std::to_string(index.size()) + ")");
          idx.push_back(it->second);
        }
        const double duration = off[k] - on[k];
        if (duration < 0)
          value_error("event " + str_repr(label) + " occurrence " + std::to_string(k) + " has offset " + float_repr(off[k]) + " before onset " + float_repr(on[k]) + " (duration " +
                      float_repr(duration) + " < 0) \xe2\x80\x94 refusing rather than producing a negative duration");
        rows.push_back({on[k], duration, ti, idx});
      }
    }
  }
  std::stable_sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.onset < b.onset; });
  out.channel_offsets.assign(1, 0);
  for (const auto& r : rows) {
    out.onset.push_back(r.onset);
    out.duration.push_back(r.duration);
    out.type.push_back(r.type);
    for (auto i : r.idx) out.channel_index.push_back(i);
    out.channel_offsets.push_back(static_cast<std::int32_t>(out.channel_index.size()));
  }
  return out;
}

Json export_events(SubjectWriter& sub, const std::string& rec_path, const EventTable& table, const std::string& line_id, const std::string& of_id,
                   const std::optional<std::string>& acquisition, const std::optional<std::string>& acquisition_id) {
  const std::size_t n = table.n_events();
  if (n == 0) return Json{{"n_events", 0}, {"n_types", 0}, {"bad_segments", 0}};
  std::vector<TimeSpan> spans;
  Json scoped = Json::object();
  for (std::size_t k = 0; k < n; ++k) {
    spans.push_back({table.onset[k], table.onset[k] + table.duration[k], table.type[k]});
    if (table.channel_offsets[k + 1] > table.channel_offsets[k]) {
      Json l = Json::array();
      for (auto i = table.channel_offsets[k]; i < table.channel_offsets[k + 1]; ++i) l.push_back(table.channel_index[static_cast<std::size_t>(i)]);
      scoped[std::to_string(k)] = l;
    }
  }
  TimeSpansOptions o;
  o.line_id = line_id;
  o.of_field_id = of_id;
  o.acquisition = acquisition;
  o.acquisition_id = acquisition_id;
  o.colors = table.type_colors;
  o.description = "the recording's " + std::to_string(n) + " Brainstorm events \xe2\x80\x94 one span per occurrence, [onset, onset + duration) in seconds; code = the event type";
  if (!scoped.empty()) o.params = Json{{"channels", scoped}};
  time_spans(sub, rec_path + "_events", spans, table.type_names, o);
  std::vector<std::size_t> bad_types;
  for (std::size_t i = 0; i < table.type_names.size(); ++i) if (lower(table.type_names[i]).find("bad") != std::string::npos) bad_types.push_back(i);
  std::vector<TimeSpan> bad;
  for (const auto& s : spans) {
    const auto it = std::find(bad_types.begin(), bad_types.end(), static_cast<std::size_t>(s.type));
    if (it != bad_types.end() && s.stop > s.start) bad.push_back({s.start, s.stop, it - bad_types.begin()});
  }
  if (!bad.empty()) {
    TimeSpansOptions b;
    b.line_id = line_id;
    b.of_field_id = of_id;
    b.acquisition = acquisition;
    b.acquisition_id = acquisition_id;
    std::vector<std::string> names;
    for (auto i : bad_types) { names.push_back(table.type_names[i]); b.colors.push_back(table.type_colors[i]); }
    b.description = "the recording's bad segments (Brainstorm events labelled bad*), one span per segment; code = the event type";
    time_spans(sub, rec_path + "_bad_segments", bad, names, b);
  }
  return Json{{"n_events", static_cast<std::int64_t>(n)}, {"n_types", static_cast<std::int64_t>(table.type_names.size())}, {"bad_segments", static_cast<std::int64_t>(bad.size())}};
}

namespace {

std::vector<std::string> factors(SubjectWriter& sub, const std::string& manifold_id) {
  std::vector<std::string> out;
  for (const auto& r : sub.db().all("SELECT d.manifold_id FROM manifold p JOIN geometry_dimension d ON d.geometry_id = p.geometry_id WHERE p.id = ? ORDER BY d.ordinal", {manifold_id}))
    out.push_back(nxr::io::text(r, "manifold_id"));
  return out;
}

// the recording and every recording DERIVED from it (a `resample` copy shares its seconds)
std::vector<nxr::io::Row> with_derived(SubjectWriter& sub, const nxr::io::Row& rec) {
  std::vector<nxr::io::Row> out = {rec};
  const std::string id = nxr::io::text(rec, "id");
  for (auto& r : sub.db().all("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND (derived_from_id = ? OR json_extract(producer_json, '$.resampled_from.field') = ?)",
                              {sub.id(), id, id}))
    out.push_back(r);
  return out;
}

std::vector<std::string> member_names(SubjectWriter& sub, const std::string& channels_id) {
  auto& db = sub.db();
  const auto sel = db.one("SELECT * FROM selection WHERE manifold_id = ? AND name LIKE '%\\_names' ESCAPE '\\' AND member_of_id IS NULL", {channels_id});
  if (!sel) value_error(channels_id + ": no member names");
  const std::string rel = nxr::io::opt_text(*sel, "array_path") ? *nxr::io::opt_text(*sel, "array_path") : nxr::io::text(*sel, "path");
  const auto rd = nxr::io::ArrayReader::open(sub.store(), rel);
  const auto bytes = rd.read();
  std::map<std::int64_t, std::string> by;
  for (const auto& e : db.all("SELECT code, name FROM dictionary_entry WHERE dictionary_id = ?", {nxr::io::text(*sel, "dictionary_id")})) by[nxr::io::integer(e, "code")] = nxr::io::text(e, "name");
  std::vector<std::string> out;
  for (std::size_t i = 0; i < bytes.size() / 4; ++i) {
    std::int32_t c;
    std::memcpy(&c, bytes.data() + 4 * i, 4);
    out.push_back(by.count(c) ? by[c] : std::to_string(c));
  }
  return out;
}

double py_round3(double x) { char b[64]; std::snprintf(b, sizeof b, "%.3f", x); return std::strtod(b, nullptr); }

}  // namespace

Json import_bad_segments(SubjectWriter& sub, const Json& src, bool replace) {
  auto& db = sub.db();
  const std::string sname = nxr::io::text(sub.row(), "name");
  Json out = Json::array();
  for (const Json& r : src.at("recordings")) {
    const std::string rname = r.at("name").get<std::string>();
    const auto rec = db.one("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND name = ?", {sub.id(), rname});
    if (!rec) throw nxr::io::Error(code::no_such_row, sname + ": no recording " + str_repr(rname));
    std::vector<Json> segs;
    if (r.contains("segments")) for (const Json& s : r["segments"]) if (lower(s.at("label").get<std::string>()).find("bad") != std::string::npos) segs.push_back(s);
    std::vector<TimeSpan> spans;
    for (std::size_t k = 0; k < segs.size(); ++k) {
      const auto& a = segs[k].at("start_s");
      const auto& b = segs[k].at("stop_s");
      for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        if (b[i].get<double>() > a[i].get<double>()) spans.push_back({a[i].get<double>(), b[i].get<double>(), static_cast<std::int64_t>(k)});
    }
    for (const auto& f : with_derived(sub, *rec)) {
      const std::string time_id = factors(sub, nxr::io::text(f, "manifold_id")).at(1);
      const std::string path = nxr::io::node_path(db, time_id, nxr::io::text(f, "name") + "_bad_segments");
      const auto have = sub.find("selection", path);
      if (have && !replace) throw nxr::io::Error(code::already_exists, sname + ": " + path + " exists (--replace)");
      if (have) sub.remove("selection", nxr::io::text(*have, "id"));
      if (!spans.empty()) {
        TimeSpansOptions o;
        o.line_id = time_id;
        o.of_field_id = nxr::io::text(f, "id");
        if (!nxr::io::is_null(f, "acquisition_id")) o.acquisition_id = nxr::io::text(f, "acquisition_id");
        o.description = "the recording's bad segments (Brainstorm events labelled bad*, detected after the conversion), one span per segment; code = the event type";
        std::vector<std::string> labels;
        for (const auto& s : segs) labels.push_back(s.at("label").get<std::string>());
        time_spans(sub, path, spans, labels, o);
      }
      Json seconds;
      if (spans.empty()) seconds = 0;
      else { double sum = 0; for (const auto& s : spans) sum += s.stop - s.start; seconds = py_round3(sum); }
      out.push_back(Json{{"recording", nxr::io::text(f, "name")}, {"bad_segments", static_cast<std::int64_t>(spans.size())}, {"seconds", seconds}});
    }
  }
  return Json{{"subject", sname}, {"recordings", out}};
}

Json import_events(SubjectWriter& sub, const fs::path& bst_data, bool replace) {
  auto& db = sub.db();
  const std::string sname = nxr::io::text(sub.row(), "name");
  const fs::path root = bst_data / sname;
  Json out = Json::array();
  for (const auto& rec : db.all("SELECT * FROM field WHERE subject_id = ? AND kind = 'recording' AND json_extract(producer_json, '$.resampled_from') IS NULL", {sub.id()})) {
    const std::string rname = nxr::io::text(rec, "name");
    const fs::path link = root / ("@raw" + rname) / ("data_0raw_" + rname + ".mat");
    std::error_code ec;
    if (!fs::is_regular_file(link, ec)) {
      out.push_back(Json{{"recording", rname}, {"events", nullptr}, {"skipped", "no raw link in the protocol (" + py_path(link.string()) + ")"}});
      continue;
    }
    const auto F = read_variable(link, "F");
    const Value* events = F && F->is_struct() ? F->field("events") : nullptr;
    const EventTable table = parse_events(events, member_names(sub, factors(sub, nxr::io::text(rec, "manifold_id")).at(0)));
    for (const auto& f : with_derived(sub, rec)) {
      const std::string time_id = factors(sub, nxr::io::text(f, "manifold_id")).at(1);
      const std::string base = nxr::io::node_path(db, time_id, nxr::io::text(f, "name"));
      for (const char* suffix : {"_events", "_bad_segments"}) {
        const auto have = sub.find("selection", base + suffix);
        if (have && !replace) throw nxr::io::Error(code::already_exists, sname + ": " + base + suffix + " exists (--replace)");
        if (have) sub.remove("selection", nxr::io::text(*have, "id"));
      }
      std::optional<std::string> aid;
      if (!nxr::io::is_null(f, "acquisition_id")) aid = nxr::io::text(f, "acquisition_id");
      const Json ev = export_events(sub, base, table, time_id, nxr::io::text(f, "id"), std::nullopt, aid);
      out.push_back(Json{{"recording", nxr::io::text(f, "name")}, {"events", ev["n_events"]}, {"event_types", ev["n_types"]}, {"bad_segments", ev["bad_segments"]}});
    }
  }
  return Json{{"subject", sname}, {"recordings", out}};
}

}  // namespace nxr::bst
