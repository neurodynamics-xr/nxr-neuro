// nxr::selection (compute) — a selection in memory (selection.h) and lookups (lookup.h).
#include "nxr/selection/selection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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

Selection span(std::vector<Extent> extents, std::optional<std::int64_t> code, std::string unit) {
  if (extents.empty()) throw std::invalid_argument("span: no axis bounded");
  std::set<int> axes;
  for (const Extent& e : extents) {
    if (!std::isfinite(e.start) || !std::isfinite(e.stop) || !(e.stop >= e.start))
      throw std::invalid_argument("span: axis " + std::to_string(e.axis) + " is not [start, stop) or an instant");
    if (!axes.insert(e.axis).second) throw std::invalid_argument("span: axis " + std::to_string(e.axis) + " bounded twice");
  }
  Selection s; s.type = Type::span; s.extents = std::move(extents); s.code = code; s.unit = std::move(unit);
  return s;
}

Selection distance(std::int64_t center, double radius, std::string unit, std::optional<std::int64_t> code) {
  if (center < 0) throw std::invalid_argument("distance: centre element " + std::to_string(center));
  if (!(radius >= 0) || !std::isfinite(radius)) throw std::invalid_argument("distance: the radius is >= 0");
  if (unit.empty()) throw std::invalid_argument("distance: the radius' unit is needed");
  Selection s; s.type = Type::distance; s.center = center; s.radius = radius; s.unit = std::move(unit); s.code = code;
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

Selection set_of_spans(std::vector<SpanRow> rows, std::string unit, Lookup lookup) {
  std::stable_sort(rows.begin(), rows.end(), [](const SpanRow& a, const SpanRow& b) { return a.start < b.start || (a.start == b.start && a.stop < b.stop); });
  std::vector<Selection> members;
  members.reserve(rows.size());
  for (const SpanRow& r : rows) members.push_back(span({Extent{0, r.start, r.stop}}, r.code, unit));
  Selection s = set(std::move(members), std::move(lookup));
  s.unit = std::move(unit);
  return s;
}

Selection tiled(Tiling t, std::string unit) {
  if (!(t.width > 0) || !(t.hop > 0) || t.count < 0 || !std::isfinite(t.origin)) throw std::invalid_argument("tiled: width > 0, hop > 0 and count >= 0");
  Selection s; s.type = Type::set; s.tiling = t; s.unit = std::move(unit);
  return s;
}

Selection set_of_members(std::vector<std::int32_t> member_of, std::vector<std::int64_t> member_codes, Lookup lookup) {
  check(lookup);
  std::int64_t n = static_cast<std::int64_t>(member_codes.size());
  if (member_codes.empty()) {                                          // unlabelled: member i is labelled i
    for (std::int32_t m : member_of) n = std::max<std::int64_t>(n, static_cast<std::int64_t>(m) + 1);
    for (std::int64_t i = 0; i < n; ++i) member_codes.push_back(i);
  }
  for (std::int32_t m : member_of)
    if (m < -1 || m >= n) throw std::invalid_argument("set_of_members: member number " + std::to_string(m) + " outside 0.." + std::to_string(n - 1));
  if (!lookup.empty()) for (std::int64_t c : member_codes) entry(lookup, c);
  Selection s; s.type = Type::set; s.member_of = std::move(member_of); s.member_codes = std::move(member_codes); s.lookup = std::move(lookup);
  return s;
}

Selection from_labels(const std::vector<std::int32_t>& element_labels, Lookup lookup) {
  std::set<std::int32_t> distinct;
  for (std::int32_t c : element_labels) if (c >= 0) distinct.insert(c);
  std::map<std::int32_t, std::int32_t> number;                         // label -> member number, ascending labels
  std::vector<std::int64_t> codes;
  for (std::int32_t c : distinct) { number[c] = static_cast<std::int32_t>(codes.size()); codes.push_back(c); }
  std::vector<std::int32_t> member_of(element_labels.size(), -1);
  for (std::size_t v = 0; v < element_labels.size(); ++v) if (element_labels[v] >= 0) member_of[v] = number[element_labels[v]];
  return set_of_members(std::move(member_of), std::move(codes), std::move(lookup));
}

// ── reading ──────────────────────────────────────────────────────────────────────────────
namespace {
void need_set(const Selection& s, const char* what) {
  if (s.type != Type::set) throw std::invalid_argument(std::string(what) + ": a set, not a " + type_name(s.type));
}
bool line_span(const Selection& m) { return m.type == Type::span && m.extents.size() == 1 && m.extents[0].axis == 0; }
void need_product(const Selection& s, const char* what) {
  if (s.type != Type::set || s.factors.empty()) throw std::invalid_argument(std::string(what) + ": a product (product(factors))");
}
// a factor's member number i, as one selection: a selection that is not a set is its own one member; a set per element gives the elements of member i
Selection factor_member(const Selection& f, std::int64_t i) {
  if (f.type != Type::set) {
    if (i != 0) throw std::invalid_argument("member: a factor that is not a set has one member (0), not " + std::to_string(i));
    return f;
  }
  if (!f.member_of.empty()) return member_indices(f, i);              // an empty member when no element is in it: its number still counts
  return member(f, i);
}
}  // namespace

std::int64_t count(const Selection& s) {
  if (s.type != Type::set) return 1;
  if (!s.factors.empty()) {
    std::int64_t n = 1;
    for (const Selection& f : s.factors) {
      const std::int64_t c = count(f);                                 // a factor that is not a set is one member
      if (c > 0 && n > std::numeric_limits<std::int64_t>::max() / c) throw std::invalid_argument("product: more members than a 64-bit number holds");
      n *= c;
    }
    return n;
  }
  if (s.tiling) return s.tiling->count;
  if (!s.member_codes.empty() || !s.member_of.empty()) return static_cast<std::int64_t>(s.member_codes.size());
  return static_cast<std::int64_t>(s.members.size());
}

Selection member(const Selection& s, std::int64_t k) {
  need_set(s, "member");
  if (!s.factors.empty()) {                                           // the tuple of the factors' members, each one member: a product of one member
    const std::vector<std::int64_t> at = factor_ordinals(s, k);
    Selection out; out.type = Type::set;
    for (std::size_t f = 0; f < s.factors.size(); ++f) out.factors.push_back(factor_member(s.factors[f], at[f]));
    return out;
  }
  if (s.tiling) {
    if (k < 0 || k >= s.tiling->count) throw std::invalid_argument("member: " + std::to_string(k) + " outside 0.." + std::to_string(s.tiling->count - 1));
    const double start = s.tiling->origin + static_cast<double>(k) * s.tiling->hop;
    return span({Extent{0, start, start + s.tiling->width}}, std::nullopt, s.unit);
  }
  if (!s.member_of.empty()) throw std::invalid_argument("member: a set per element has members by number (member_indices)");
  if (k < 0 || k >= static_cast<std::int64_t>(s.members.size())) throw std::invalid_argument("member: " + std::to_string(k) + " outside the set");
  return s.members[static_cast<std::size_t>(k)];
}

std::vector<std::int64_t> members_at(const Selection& s, double value) {
  need_set(s, "members_at");
  if (!s.factors.empty()) throw std::invalid_argument("members_at: a product lies on several lines; ask each factor (factors[f]) and combine with product_ordinal");
  std::vector<std::int64_t> out;
  if (s.tiling) {
    const Tiling& t = *s.tiling;
    if (!(value >= t.origin)) return out;
    const auto last = std::min<std::int64_t>(t.count - 1, static_cast<std::int64_t>(std::floor((value - t.origin) / t.hop)));
    for (std::int64_t k = last; k >= 0 && value < t.origin + static_cast<double>(k) * t.hop + t.width; --k) out.push_back(k);
    std::reverse(out.begin(), out.end());
    return out;
  }
  for (std::size_t i = 0; i < s.members.size(); ++i) {
    const Selection& m = s.members[i];
    if (!line_span(m)) throw std::invalid_argument("members_at: member " + std::to_string(i) + " is not a span on a line");
    const Extent& e = m.extents[0];
    if ((value >= e.start && value < e.stop) || (e.start == e.stop && value == e.start)) out.push_back(static_cast<std::int64_t>(i));   // an instant holds its own value
  }
  return out;
}

std::int64_t member_at(const Selection& s, std::int64_t element, int level) {
  need_set(s, "member_at");
  if (s.member_of.empty()) throw std::invalid_argument("member_at: the set has no member per element");
  if (element < 0 || element >= static_cast<std::int64_t>(s.member_of.size())) throw std::invalid_argument("member_at: element " + std::to_string(element) + " outside the set");
  std::int64_t m = s.member_of[static_cast<std::size_t>(element)];
  if (m < 0) return -1;
  const int finest = static_cast<int>(s.members.size()) - 1;           // the set's own member numbers are the last level's
  if (level < 0 || level == finest || !has_levels(s)) return m;
  if (level > finest) throw std::invalid_argument("member_at: level " + std::to_string(level) + " outside 0.." + std::to_string(finest));
  for (int l = finest; l > level; --l) {                               // up the parent maps, one level at a time
    const std::vector<std::int32_t>& up = s.members[static_cast<std::size_t>(l)].parent;
    if (m >= static_cast<std::int64_t>(up.size())) throw std::invalid_argument("member_at: level " + std::to_string(l) + " has no parent for member " + std::to_string(m));
    m = up[static_cast<std::size_t>(m)];
  }
  return m;
}

std::optional<std::int64_t> member_label(const Selection& s, std::int64_t member, int level) {
  need_set(s, "member_label");
  if (s.tiling) return std::nullopt;
  const std::vector<std::int64_t>* codes = &s.member_codes;
  if (level >= 0 && has_levels(s) && level != static_cast<int>(s.members.size()) - 1) {
    if (level >= static_cast<int>(s.members.size())) throw std::invalid_argument("member_label: level " + std::to_string(level) + " outside the set");
    codes = &s.members[static_cast<std::size_t>(level)].member_codes;
  }
  if (!codes->empty() || !s.member_of.empty()) {
    if (member < 0 || member >= static_cast<std::int64_t>(codes->size())) throw std::invalid_argument("member_label: member " + std::to_string(member) + " outside the set");
    return (*codes)[static_cast<std::size_t>(member)];
  }
  if (member < 0 || member >= static_cast<std::int64_t>(s.members.size())) throw std::invalid_argument("member_label: member " + std::to_string(member) + " outside the set");
  return s.members[static_cast<std::size_t>(member)].code;
}

std::optional<std::int64_t> label_at(const Selection& s, std::int64_t element, int level) {
  const std::int64_t m = member_at(s, element, level);
  if (m < 0) return std::nullopt;
  return member_label(s, m, level);
}

std::vector<std::int64_t> members_with_label(const Selection& s, std::int64_t label, int level) {
  need_set(s, "members_with_label");
  const std::vector<std::int64_t>* codes = &s.member_codes;
  if (level >= 0 && has_levels(s) && level != static_cast<int>(s.members.size()) - 1) codes = &s.members.at(static_cast<std::size_t>(level)).member_codes;
  std::vector<std::int64_t> out;
  for (std::size_t m = 0; m < codes->size(); ++m) if ((*codes)[m] == label) out.push_back(static_cast<std::int64_t>(m));
  return out;
}

Selection member_indices(const Selection& s, std::int64_t member, int level) {
  need_set(s, "member_indices");
  const std::optional<std::int64_t> label = member_label(s, member, level);   // refuses a member outside the set
  std::vector<std::int32_t> out;
  for (std::int64_t v = 0; v < static_cast<std::int64_t>(s.member_of.size()); ++v)
    if (member_at(s, v, level) == member) out.push_back(static_cast<std::int32_t>(v));
  return indices(std::move(out), label);
}

bool disjoint(const Selection& s) {
  need_set(s, "disjoint");
  if (!s.factors.empty())
    return std::all_of(s.factors.begin(), s.factors.end(), [](const Selection& f) { return f.type != Type::set || disjoint(f); });
  if (!s.member_of.empty()) return true;                              // each element is in one member
  if (s.tiling) return s.tiling->hop >= s.tiling->width;
  if (std::all_of(s.members.begin(), s.members.end(), line_span)) {
    std::vector<std::pair<double, double>> e;
    for (const Selection& m : s.members) e.emplace_back(m.extents[0].start, m.extents[0].stop);
    std::sort(e.begin(), e.end());
    for (std::size_t i = 1; i < e.size(); ++i) if (e[i].first < e[i - 1].second) return false;
    return true;
  }
  std::set<std::int32_t> seen;
  for (const Selection& m : s.members) {
    if (m.type != Type::indices) return false;
    for (std::int32_t v : m.indices) if (!seen.insert(v).second) return false;
  }
  return true;
}

Selection resolve_distance(const Selection& s, chart::ChartKit& kit) {
  if (s.type != Type::distance) throw std::invalid_argument(std::string("resolve_distance: a distance, not a ") + type_name(s.type));
  if (*s.center >= kit.n()) throw std::invalid_argument("resolve_distance: centre " + std::to_string(*s.center) + " outside the surface");
  std::vector<std::int32_t> out;
  for (int v : kit.disc(static_cast<int>(*s.center), s.radius)) out.push_back(static_cast<std::int32_t>(v));
  return indices(std::move(out), s.code);
}

// ── products and levels ───────────────────────────────────────────────────────────────────
bool has_levels(const Selection& s) {
  if (s.type != Type::set || s.members.empty()) return false;
  return std::all_of(s.members.begin(), s.members.end(), [](const Selection& m) { return m.type == Type::set && m.level >= 0; });
}

int level_ordinal(const Selection& s, int k) {
  if (!has_levels(s)) throw std::invalid_argument("level: the selection has no levels");
  std::string have;
  for (std::size_t i = 0; i < s.members.size(); ++i) {
    if (s.members[i].level == k) return static_cast<int>(i);
    have += (have.empty() ? "" : ", ") + std::to_string(s.members[i].level);
  }
  throw std::invalid_argument("level: no level " + std::to_string(k) + " (the set has " + have + ")");
}

Selection level(const Selection& s, int k) {
  const int i = level_ordinal(s, k);
  const Selection& found = s.members[static_cast<std::size_t>(i)];
  if (s.member_of.empty()) return found;                               // a ladder of tilings: the level is its own selection
  const bool finest = i == static_cast<int>(s.members.size()) - 1;
  std::vector<std::int32_t> member_of(s.member_of.size());
  for (std::size_t v = 0; v < member_of.size(); ++v) member_of[v] = static_cast<std::int32_t>(member_at(s, static_cast<std::int64_t>(v), i));
  Selection out = set_of_members(std::move(member_of), found.member_codes, finest ? s.lookup : Lookup{});   // the lookup names the finest labels only
  out.unit = s.unit;
  out.level = found.level;
  out.code = found.code;
  out.band = found.band;
  out.band_unit = found.band_unit;
  return out;
}

Selection product(std::vector<Selection> factors) {
  std::vector<Selection> flat;
  for (Selection& f : factors) {
    if (f.type == Type::set && !f.factors.empty()) {                  // a product of products is one product, its factors in order
      for (Selection& g : f.factors) flat.push_back(std::move(g));
      continue;
    }
    if (has_levels(f)) throw std::invalid_argument("product: factor " + std::to_string(flat.size()) + " has levels; name one (level(set, k))");
    flat.push_back(std::move(f));
  }
  if (flat.size() < 2) throw std::invalid_argument("product: two factors or more");
  Selection s; s.type = Type::set; s.factors = std::move(flat);
  count(s);                                                            // refuses a product too large to number
  return s;
}

std::vector<std::int64_t> factor_ordinals(const Selection& p, std::int64_t k) {
  need_product(p, "factor_ordinals");
  const std::int64_t n = count(p);
  if (k < 0 || k >= n) throw std::invalid_argument("factor_ordinals: member " + std::to_string(k) + " outside 0.." + std::to_string(n - 1));
  std::vector<std::int64_t> out(p.factors.size());
  for (std::size_t f = p.factors.size(); f-- > 0;) {                  // row-major: the last factor fastest
    const std::int64_t c = count(p.factors[f]);
    out[f] = k % c;
    k /= c;
  }
  return out;
}

std::int64_t product_ordinal(const Selection& p, const std::vector<std::int64_t>& at) {
  need_product(p, "product_ordinal");
  if (at.size() != p.factors.size()) throw std::invalid_argument("product_ordinal: " + std::to_string(p.factors.size()) + " ordinals, one per factor");
  std::int64_t k = 0;
  for (std::size_t f = 0; f < at.size(); ++f) {
    const std::int64_t c = count(p.factors[f]);
    if (at[f] < 0 || at[f] >= c) throw std::invalid_argument("product_ordinal: ordinal " + std::to_string(at[f]) + " outside factor " + std::to_string(f));
    k = k * c + at[f];
  }
  return k;
}

std::vector<std::optional<std::int64_t>> factor_labels(const Selection& p, std::int64_t k) {
  const std::vector<std::int64_t> at = factor_ordinals(p, k);
  std::vector<std::optional<std::int64_t>> out;
  for (std::size_t f = 0; f < at.size(); ++f) {
    const Selection& x = p.factors[f];
    if (x.type != Type::set) out.push_back(x.code);
    else if (x.tiling) out.push_back(std::nullopt);
    else out.push_back(member_label(x, at[f]));                        // a listed member's code, a set per element's member label
  }
  return out;
}

}  // namespace nxr::selection
