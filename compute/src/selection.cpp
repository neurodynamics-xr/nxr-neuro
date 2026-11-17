// nxr::selection (compute) — a selection in memory (selection.h) and lookups (lookup.h).
#include "nxr/selection/selection.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

#include "nxr/chart.h"

namespace nxr::selection {

// ── lookups ──────────────────────────────────────────────────────────────────────────────
const Entry& entry(const Lookup& lookup, std::int64_t code) {
  for (const Entry& e : lookup) if (e.code == code) return e;
  throw std::invalid_argument("lookup: no entry for code " + std::to_string(code));
}

void check(const Lookup& lookup) {
  std::set<std::int64_t> codes;
  for (const Entry& e : lookup) {
    if (e.name.empty()) throw std::invalid_argument("lookup: code " + std::to_string(e.code) + " has no name");
    if (!codes.insert(e.code).second) throw std::invalid_argument("lookup: code " + std::to_string(e.code) + " appears twice");
  }
}

// ── making ───────────────────────────────────────────────────────────────────────────────
const char* type_name(Type t) {
  switch (t) {
    case Type::point: return "point";
    case Type::span: return "span";
    case Type::distance: return "distance";
    case Type::spans: return "spans";
    case Type::indices: return "indices";
    case Type::set: return "set";
  }
  return "?";
}

Selection point(std::int64_t center, std::optional<std::int64_t> code) {
  if (center < 0) throw std::invalid_argument("point: element " + std::to_string(center));
  Selection s; s.type = Type::point; s.center = center; s.code = code;
  return s;
}

Selection span(std::vector<Extent> extents, std::optional<std::int64_t> code) {
  if (extents.empty()) throw std::invalid_argument("span: no axis bounded");
  std::set<int> axes;
  for (const Extent& e : extents) {
    if (!std::isfinite(e.start) || !std::isfinite(e.stop) || !(e.stop > e.start)) throw std::invalid_argument("span: axis " + std::to_string(e.axis) + " is not [start, stop)");
    if (!axes.insert(e.axis).second) throw std::invalid_argument("span: axis " + std::to_string(e.axis) + " bounded twice");
  }
  Selection s; s.type = Type::span; s.extents = std::move(extents); s.code = code;
  return s;
}

Selection distance(std::int64_t center, double radius, std::string unit, std::optional<std::int64_t> code) {
  if (center < 0) throw std::invalid_argument("distance: centre element " + std::to_string(center));
  if (!(radius >= 0) || !std::isfinite(radius)) throw std::invalid_argument("distance: the radius is >= 0");
  if (unit.empty()) throw std::invalid_argument("distance: the radius' unit is needed");
  Selection s; s.type = Type::distance; s.center = center; s.radius = radius; s.unit = std::move(unit); s.code = code;
  return s;
}

Selection spans(std::vector<SpanRow> rows, std::string unit, Lookup lookup) {
  check(lookup);
  for (const SpanRow& r : rows) {
    if (!std::isfinite(r.start) || !std::isfinite(r.stop) || !(r.stop >= r.start))
      throw std::invalid_argument("spans: [" + std::to_string(r.start) + ", " + std::to_string(r.stop) + ") is not an interval or an instant");
    if (r.code && !lookup.empty()) entry(lookup, *r.code);           // throws for a label the lookup lacks
  }
  std::stable_sort(rows.begin(), rows.end(), [](const SpanRow& a, const SpanRow& b) { return a.start < b.start || (a.start == b.start && a.stop < b.stop); });
  Selection s; s.type = Type::spans; s.rows = std::move(rows); s.unit = std::move(unit); s.lookup = std::move(lookup);
  return s;
}

Selection spans(Tiling t, std::string unit) {
  if (!(t.width > 0) || !(t.hop > 0) || t.count < 0 || !std::isfinite(t.origin)) throw std::invalid_argument("spans: a closed form needs width > 0, hop > 0, count >= 0");
  Selection s; s.type = Type::spans; s.tiling = t; s.unit = std::move(unit);
  return s;
}

Selection indices(std::vector<std::int32_t> elements, std::optional<std::int64_t> code) {
  std::sort(elements.begin(), elements.end());
  elements.erase(std::unique(elements.begin(), elements.end()), elements.end());
  if (!elements.empty() && elements.front() < 0) throw std::invalid_argument("indices: a negative element");
  Selection s; s.type = Type::indices; s.indices = std::move(elements); s.code = code;
  return s;
}

Selection set(std::vector<Selection> members, Lookup lookup) {
  check(lookup);
  if (!lookup.empty())
    for (const Selection& m : members) if (m.code) entry(lookup, *m.code);
  Selection s; s.type = Type::set; s.members = std::move(members); s.lookup = std::move(lookup);
  return s;
}

Selection labels(std::vector<std::int32_t> codes, Lookup lookup) {
  check(lookup);
  if (!lookup.empty()) {
    std::set<std::int32_t> seen;
    for (std::int32_t c : codes) if (c >= 0 && seen.insert(c).second) entry(lookup, c);
  }
  Selection s; s.type = Type::set; s.elements = static_cast<std::int64_t>(codes.size()); s.codes = std::move(codes); s.lookup = std::move(lookup);
  return s;
}

// ── reading ──────────────────────────────────────────────────────────────────────────────
namespace {
void need(const Selection& s, Type t, const char* what) {
  if (s.type != t) throw std::invalid_argument(std::string(what) + ": a " + type_name(t) + " selection, not " + type_name(s.type));
}
}  // namespace

std::int64_t count(const Selection& s) {
  need(s, Type::spans, "count");
  return s.tiling ? s.tiling->count : static_cast<std::int64_t>(s.rows.size());
}

SpanRow row(const Selection& s, std::int64_t k) {
  need(s, Type::spans, "row");
  if (k < 0 || k >= count(s)) throw std::invalid_argument("row: " + std::to_string(k) + " outside 0.." + std::to_string(count(s) - 1));
  if (s.tiling) {
    const double start = s.tiling->origin + static_cast<double>(k) * s.tiling->hop;
    return SpanRow{start, start + s.tiling->width, std::nullopt};
  }
  return s.rows[static_cast<std::size_t>(k)];
}

std::vector<std::int64_t> rows_at(const Selection& s, double value) {
  need(s, Type::spans, "rows_at");
  std::vector<std::int64_t> out;
  if (s.tiling) {
    const Tiling& t = *s.tiling;
    if (!(value >= t.origin)) return out;
    const auto last = std::min<std::int64_t>(t.count - 1, static_cast<std::int64_t>(std::floor((value - t.origin) / t.hop)));
    for (std::int64_t k = last; k >= 0 && value < t.origin + static_cast<double>(k) * t.hop + t.width; --k) out.push_back(k);
    std::reverse(out.begin(), out.end());
    return out;
  }
  for (std::size_t i = 0; i < s.rows.size() && s.rows[i].start <= value; ++i)
    if (value < s.rows[i].stop || (s.rows[i].stop == s.rows[i].start && value == s.rows[i].start)) out.push_back(static_cast<std::int64_t>(i));   // an instant holds its own time
  return out;
}

std::int64_t code_at(const Selection& s, std::int64_t element, int level) {
  need(s, Type::set, "code_at");
  if (s.codes.empty()) throw std::invalid_argument("code_at: the set has no codes (its members are selections)");
  if (element < 0 || element >= static_cast<std::int64_t>(s.codes.size())) throw std::invalid_argument("code_at: element " + std::to_string(element) + " outside the set");
  const std::int32_t c = s.codes[static_cast<std::size_t>(element)];
  if (c < 0) return -1;
  if (level < 0) return c;
  if (level > s.depth) throw std::invalid_argument("code_at: level " + std::to_string(level) + " outside 0.." + std::to_string(s.depth));
  return c >> (s.depth - level);
}

Selection member_indices(const Selection& s, std::int64_t code, int level) {
  std::vector<std::int32_t> out;
  for (std::int64_t v = 0; v < static_cast<std::int64_t>(s.codes.size()); ++v)
    if (code_at(s, v, level) == code) out.push_back(static_cast<std::int32_t>(v));
  if (out.empty()) throw std::invalid_argument("member_indices: code " + std::to_string(code) + " holds no element");
  return indices(std::move(out), code);
}

Selection member_span(const Selection& s, std::int64_t k) {
  const SpanRow r = row(s, k);
  Selection out;
  out.type = Type::span; out.extents = {Extent{0, r.start, r.stop}}; out.code = r.code; out.unit = s.unit;
  return out;
}

bool disjoint(const Selection& s) {
  if (s.type == Type::set && !s.codes.empty()) return true;
  if (s.type == Type::spans) {
    if (s.tiling) return s.tiling->hop >= s.tiling->width;
    for (std::size_t i = 1; i < s.rows.size(); ++i) if (s.rows[i].start < s.rows[i - 1].stop) return false;
    return true;
  }
  if (s.type == Type::set) {
    std::set<std::int32_t> seen;
    for (const Selection& m : s.members) {
      if (m.type != Type::indices) return false;
      for (std::int32_t v : m.indices) if (!seen.insert(v).second) return false;
    }
    return true;
  }
  return true;                                                         // a single member is disjoint
}

Selection resolve_distance(const Selection& s, chart::ChartKit& kit) {
  need(s, Type::distance, "resolve_distance");
  if (*s.center >= kit.n()) throw std::invalid_argument("resolve_distance: centre " + std::to_string(*s.center) + " outside the surface");
  std::vector<std::int32_t> out;
  for (int v : kit.disc(static_cast<int>(*s.center), s.radius)) out.push_back(static_cast<std::int32_t>(v));
  return indices(std::move(out), s.code);
}

}  // namespace nxr::selection
