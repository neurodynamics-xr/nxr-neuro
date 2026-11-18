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
const char* family_name(Family f) { return f == Family::grid ? "grid" : "irregular"; }

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
    if (e.axis < 0) throw std::invalid_argument("span: axis " + std::to_string(e.axis));
    if (!axes.insert(e.axis).second) throw std::invalid_argument("span: axis " + std::to_string(e.axis) + " bounded twice");
  }
  std::sort(extents.begin(), extents.end(), [](const Extent& a, const Extent& b) { return a.axis < b.axis; });
  Selection s; s.type = Type::span; s.extents = std::move(extents); s.code = code; s.unit = std::move(unit);
  return s;
}

Selection span_about(std::vector<About> about, std::optional<std::int64_t> code, std::string unit) {
  std::vector<Extent> e;
  e.reserve(about.size());
  for (const About& a : about) {
    if (!std::isfinite(a.center) || !std::isfinite(a.extent) || a.extent < 0)
      throw std::invalid_argument("span_about: axis " + std::to_string(a.axis) + " needs a finite centre and an extent >= 0");
    e.push_back(Extent{a.axis, a.center - a.extent / 2, a.center + a.extent / 2});
  }
  return span(std::move(e), code, std::move(unit));
}

std::vector<About> about(const Selection& s) {
  if (s.type != Type::span) throw std::invalid_argument(std::string("about: a span, not a ") + type_name(s.type));
  std::vector<About> out;
  for (const Extent& e : s.extents) out.push_back(About{e.axis, (e.start + e.stop) / 2, e.stop - e.start});
  return out;
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
  // a ladder (members that are levels of tilings) is a grid with levels; any other set of listed members is irregular
  if (!s.members.empty() && std::all_of(s.members.begin(), s.members.end(), [](const Selection& m) { return m.type == Type::set && m.level >= 0 && m.tiling; })) s.family = Family::grid;
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
  s.family = t.hop >= t.width ? Family::grid : Family::irregular;     // an overlapping tiling is a family of segments, not a partition of its line
  return s;
}

Selection elements_grid(std::int64_t size) {
  if (size < 0) throw std::invalid_argument("elements_grid: a manifold of size >= 0");
  Selection s; s.type = Type::set; s.grid = Grid{Grid::Kind::elements, size, 0, 0};
  s.family = Family::grid;
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

Selection dyadic_grid(std::vector<std::int32_t> codes, int depth, std::int64_t slots, std::vector<Selection> levels) {
  if (depth < 0 || depth > 30 || slots < 1) throw std::invalid_argument("dyadic_grid: depth 0..30 and at least one piece slot");
  const std::int64_t n = slots << depth;
  if (n > std::numeric_limits<std::int32_t>::max()) throw std::invalid_argument("dyadic_grid: more tiles than a 32-bit member number holds");
  for (std::int32_t c : codes)
    if (c < -1 || c >= n) throw std::invalid_argument("dyadic_grid: tile code " + std::to_string(c) + " outside 0.." + std::to_string(n - 1));
  if (!levels.empty()) {
    if (static_cast<int>(levels.size()) != depth + 1) throw std::invalid_argument("dyadic_grid: " + std::to_string(depth + 1) + " levels (0 .. depth), coarsest first");
    for (std::size_t l = 0; l < levels.size(); ++l) {                  // a level holds only its label and band: its members are the closed form
      Selection& x = levels[l];
      if (x.level < 0) x.level = static_cast<int>(l);
      if (x.level != static_cast<int>(l)) throw std::invalid_argument("dyadic_grid: the level at position " + std::to_string(l) + " is labelled " + std::to_string(x.level));
      x.type = Type::set;
      x.family = Family::grid;
      x.code = x.level;
      x.members.clear(); x.member_of.clear(); x.member_codes.clear(); x.factors.clear(); x.tiling.reset(); x.grid.reset();
    }
  }
  Selection s; s.type = Type::set; s.member_of = std::move(codes); s.grid = Grid{Grid::Kind::dyadic, 0, depth, slots}; s.members = std::move(levels);
  s.family = Family::grid;
  return s;
}

// ── reading ──────────────────────────────────────────────────────────────────────────────
namespace {
void need_set(const Selection& s, const char* what) {
  if (s.type != Type::set) throw std::invalid_argument(std::string(what) + ": a set, not a " + type_name(s.type));
}
bool line_span(const Selection& m) { return m.type == Type::span && m.extents.size() == 1 && m.extents[0].axis == 0; }
bool is_product(const Selection& s) { return s.type == Type::set && !s.factors.empty(); }
bool is_dyadic(const Selection& s) { return s.type == Type::set && s.grid && s.grid->kind == Grid::Kind::dyadic; }
bool is_elements(const Selection& s) { return s.type == Type::set && s.grid && s.grid->kind == Grid::Kind::elements; }
void need_product(const Selection& s, const char* what) {
  if (!is_product(s)) throw std::invalid_argument(std::string(what) + ": a product (product(factors))");
}
std::int64_t checked_product(const std::vector<std::int64_t>& radices) {
  std::int64_t n = 1;
  for (std::int64_t c : radices) {
    if (c > 0 && n > std::numeric_limits<std::int64_t>::max() / c) throw std::invalid_argument("product: more members than a 64-bit code holds");
    n *= c;
  }
  return n;
}
}  // namespace

std::int64_t count(const Selection& s) {
  if (s.type != Type::set) return 1;
  if (!s.factors.empty()) return checked_product(s.radices);
  if (s.tiling) return s.tiling->count;
  if (s.grid) return s.grid->kind == Grid::Kind::dyadic ? (s.grid->slots << s.grid->depth) : s.grid->size;   // elements, and a time grid's samples (its finest level)
  if (!s.member_codes.empty() || !s.member_of.empty()) return static_cast<std::int64_t>(s.member_codes.size());
  return static_cast<std::int64_t>(s.members.size());
}

bool has_levels(const Selection& s) {
  if (s.type != Type::set || s.members.empty()) return false;
  return std::all_of(s.members.begin(), s.members.end(), [](const Selection& m) { return m.type == Type::set && m.level >= 0; });
}

std::optional<std::string> factor_refusal(const Selection& s) {
  if (s.type != Type::set)
    return std::string("a ") + type_name(s.type) + " is one selection, not a set: a box on the product manifold is a span with one extent per factor, and a single band "
           "or range restricts the input (`at`) rather than being a factor";
  if (!s.factors.empty()) return std::nullopt;
  if (has_levels(s)) return std::string("a grid with levels: name one level (level(set, k)), or take the level a band sets (grid.h: time_level_of_band, space_level_of_band)");
  return std::nullopt;                                                 // Studio D100: any set, a grid or an irregular measurement layout (an atlas x time windows)
}

bool is_grid(const Selection& s) { return s.type == Type::set && s.family == Family::grid; }

namespace {
// does a grid's member number roll up by a shift (code >> shift)? a dyadic grid, or a disjoint tiling of a line
bool shifts(const Selection& g) { return g.type == Type::set && g.family == Family::grid && (is_dyadic(g) || (g.tiling && g.tiling->hop >= g.tiling->width)); }
}  // namespace

std::optional<std::string> rollup_refusal(const Selection& s, int factor) {
  if (s.type != Type::set) return std::string("a ") + type_name(s.type) + " is not a set: nothing to roll up";
  if (is_product(s)) {
    if (factor >= static_cast<int>(s.factors.size())) return "factor " + std::to_string(factor) + " is outside the product's " + std::to_string(s.factors.size());
    for (std::size_t f = 0; f < s.factors.size(); ++f) {
      if (factor >= 0 && static_cast<int>(f) != factor) continue;
      if (s.factors[f].family != Family::grid)
        return "factor " + std::to_string(f) + " is an irregular set: it has no levels and does not roll up (a roll-up runs along a grid factor; measure an irregular factor member by member, or summarise it)";
      if (!shifts(s.factors[f])) return "factor " + std::to_string(f) + " is a grid whose members do not roll up by a shift (a dyadic grid or a disjoint tiling does)";
    }
    return std::nullopt;
  }
  if (s.family != Family::grid)
    return std::string("an irregular set (bursts, events, scouts, an atlas) has no levels and does not roll up: its members are measured one by one and summarised across; roll up a grid");
  return std::nullopt;
}

std::vector<std::int64_t> rolled_radices(const Selection& p, int factor, int shift) {
  need_product(p, "rolled_radices");
  if (factor < 0 || factor >= static_cast<int>(p.factors.size()) || shift < 0 || shift > 62) throw std::invalid_argument("rolled_radices: a factor of the product and a shift 0..62");
  std::vector<std::int64_t> r = p.radices;
  r[static_cast<std::size_t>(factor)] = (r[static_cast<std::size_t>(factor)] + (std::int64_t(1) << shift) - 1) >> shift;
  return r;
}

std::int64_t coarser_code(const Selection& p, std::int64_t code, int factor, int shift) {
  need_product(p, "coarser_code");
  if (const auto why = rollup_refusal(p, factor)) throw std::invalid_argument("coarser_code: " + *why);
  std::vector<std::int64_t> at = product_codes(p.radices, code);     // unpack (the first factor fastest), shift the one factor, repack with the coarser radices
  at[static_cast<std::size_t>(factor)] >>= shift;
  return product_code(rolled_radices(p, factor, shift), at);
}

Selection member(const Selection& s, std::int64_t k) {
  need_set(s, "member");
  if (is_product(s)) {                                                  // the tuple of the factors' members, each one member: a product of one member
    const std::vector<std::int64_t> at = product_codes(s.radices, k);
    Selection out; out.type = Type::set;
    for (std::size_t f = 0; f < s.factors.size(); ++f) {
      const Selection& x = s.factors[f];
      if (at[f] >= count(x)) throw std::invalid_argument("member: code " + std::to_string(k) + " is one of the joint layout's empty members (factor " + std::to_string(f) + ")");
      Selection one = (!x.member_of.empty()) ? member_indices(x, at[f]) : member(x, at[f]);
      out.factors.push_back(std::move(one));
      out.radices.push_back(1);
    }
    return out;
  }
  if (s.tiling) {
    if (k < 0 || k >= s.tiling->count) throw std::invalid_argument("member: " + std::to_string(k) + " outside 0.." + std::to_string(s.tiling->count - 1));
    const double start = s.tiling->origin + static_cast<double>(k) * s.tiling->hop;
    return span({Extent{0, start, start + s.tiling->width}}, std::nullopt, s.unit);
  }
  if (is_elements(s)) {
    if (k < 0 || k >= s.grid->size) throw std::invalid_argument("member: " + std::to_string(k) + " outside 0.." + std::to_string(s.grid->size - 1));
    return point(k);
  }
  if (!s.member_of.empty()) return member_indices(s, k);
  if (k < 0 || k >= static_cast<std::int64_t>(s.members.size())) throw std::invalid_argument("member: " + std::to_string(k) + " outside the set");
  return s.members[static_cast<std::size_t>(k)];
}

std::vector<std::int64_t> members_at(const Selection& s, double value) {
  need_set(s, "members_at");
  if (is_product(s)) throw std::invalid_argument("members_at: a product lies on several lines; ask each factor (factors[f]) and combine with product_code");
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
  if (is_elements(s)) {
    if (element < 0 || element >= s.grid->size) throw std::invalid_argument("member_at: element " + std::to_string(element) + " outside the grid");
    return element;
  }
  if (s.member_of.empty()) throw std::invalid_argument("member_at: the set has no member per element");
  if (element < 0 || element >= static_cast<std::int64_t>(s.member_of.size())) throw std::invalid_argument("member_at: element " + std::to_string(element) + " outside the set");
  const std::int64_t m = s.member_of[static_cast<std::size_t>(element)];
  if (m < 0 || level < 0) return m;
  if (!is_dyadic(s)) throw std::invalid_argument("member_at: an irregular set has no levels");
  if (!has_levels(s) || level >= static_cast<int>(s.members.size())) throw std::invalid_argument("member_at: level " + std::to_string(level) + " outside the grid's levels");
  const int k = s.members[static_cast<std::size_t>(level)].level;
  if (k > s.grid->depth) throw std::invalid_argument("member_at: level " + std::to_string(k) + " is finer than the grid's codes (" + std::to_string(s.grid->depth) + ")");
  return m >> (s.grid->depth - k);                                     // the closed form: a tile's ancestor is its code shifted
}

std::optional<std::int64_t> member_label(const Selection& s, std::int64_t member) {
  need_set(s, "member_label");
  const std::int64_t n = count(s);
  if (member < 0 || member >= n) throw std::invalid_argument("member_label: member " + std::to_string(member) + " outside the set");
  if (s.tiling || s.grid || is_product(s)) return std::nullopt;       // a grid's member: its number names it
  if (!s.member_of.empty()) return s.member_codes[static_cast<std::size_t>(member)];
  return s.members[static_cast<std::size_t>(member)].code;
}

std::optional<std::int64_t> label_at(const Selection& s, std::int64_t element) {
  if (s.grid) return std::nullopt;
  const std::int64_t m = member_at(s, element);
  if (m < 0) return std::nullopt;
  return member_label(s, m);
}

std::vector<std::int64_t> members_with_label(const Selection& s, std::int64_t label) {
  need_set(s, "members_with_label");
  std::vector<std::int64_t> out;
  for (std::size_t m = 0; m < s.member_codes.size(); ++m) if (s.member_codes[m] == label) out.push_back(static_cast<std::int64_t>(m));
  return out;
}

Selection member_indices(const Selection& s, std::int64_t member) {
  need_set(s, "member_indices");
  const std::optional<std::int64_t> label = member_label(s, member);   // refuses a member outside the set
  if (is_elements(s)) return indices({static_cast<std::int32_t>(member)});
  std::vector<std::int32_t> out;
  for (std::size_t v = 0; v < s.member_of.size(); ++v)
    if (s.member_of[v] == member) out.push_back(static_cast<std::int32_t>(v));
  return indices(std::move(out), label);
}

bool disjoint(const Selection& s) {
  need_set(s, "disjoint");
  if (is_product(s)) return std::all_of(s.factors.begin(), s.factors.end(), [](const Selection& f) { return disjoint(f); });
  if (s.grid || !s.member_of.empty()) return true;                    // each element is in one member
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
  std::vector<std::int32_t> out{static_cast<std::int32_t>(*s.center)};   // the centre is always held (a radius 0 holds it alone)
  for (int v : kit.disc(static_cast<int>(*s.center), s.radius)) out.push_back(static_cast<std::int32_t>(v));
  return indices(std::move(out), s.code);
}

// ── levels ────────────────────────────────────────────────────────────────────────────────
int level_ordinal(const Selection& s, int k) {
  if (!has_levels(s)) throw std::invalid_argument("level: the selection has no levels");
  std::string have;
  for (std::size_t i = 0; i < s.members.size(); ++i) {
    if (s.members[i].level == k) return static_cast<int>(i);
    have += (have.empty() ? "" : ", ") + std::to_string(s.members[i].level);
  }
  throw std::invalid_argument("level: no level " + std::to_string(k) + " (the grid has " + have + ")");
}

Selection level(const Selection& s, int k) {
  const int i = level_ordinal(s, k);
  const Selection& found = s.members[static_cast<std::size_t>(i)];
  if (!is_dyadic(s)) return found;                                      // a ladder of tilings: the level is its own rule
  if (k > s.grid->depth) throw std::invalid_argument("level: " + std::to_string(k) + " is finer than the grid's codes (" + std::to_string(s.grid->depth) + ")");
  const int shift = s.grid->depth - k;
  std::vector<std::int32_t> codes(s.member_of.size());
  for (std::size_t v = 0; v < codes.size(); ++v) codes[v] = s.member_of[v] < 0 ? -1 : (s.member_of[v] >> shift);   // the closed form
  Selection out = dyadic_grid(std::move(codes), k, s.grid->slots);
  out.unit = s.unit;
  out.level = found.level;
  out.code = found.code;
  out.band = found.band;
  out.band_unit = found.band_unit;
  out.lookup = s.lookup;
  return out;
}

// ── products ──────────────────────────────────────────────────────────────────────────────
Selection product(std::vector<Selection> factors) {
  std::vector<Selection> flat;
  std::vector<std::int64_t> radices;
  for (Selection& f : factors) {
    if (is_product(f)) {                                                // a product of products is one product, its factors in order
      for (std::size_t g = 0; g < f.factors.size(); ++g) { radices.push_back(f.radices[g]); flat.push_back(std::move(f.factors[g])); }
      continue;
    }
    if (const auto why = factor_refusal(f)) throw std::invalid_argument("product: factor " + std::to_string(flat.size()) + " cannot be a factor: " + *why);
    radices.push_back(count(f));
    flat.push_back(std::move(f));
  }
  if (flat.size() < 2) throw std::invalid_argument("product: two factors or more");
  Selection s; s.type = Type::set; s.factors = std::move(flat); s.radices = std::move(radices);
  s.family = std::all_of(s.factors.begin(), s.factors.end(), [](const Selection& f) { return f.family == Family::grid; }) ? Family::grid : Family::irregular;
  count(s);                                                            // refuses a code space too large
  return s;
}

Selection joint(std::vector<Selection> factors) {
  Selection s = product(std::move(factors));
  std::int64_t r = 1;
  while (r < s.radices[0]) r <<= 1;                                     // the frames' radix: 2^bits, bits = ceil(log2(frames))
  s.radices[0] = r;
  count(s);
  return s;
}

std::int64_t product_code(const std::vector<std::int64_t>& radices, const std::vector<std::int64_t>& codes) {
  if (codes.size() != radices.size()) throw std::invalid_argument("product_code: " + std::to_string(radices.size()) + " codes, one per factor");
  std::int64_t code = 0, scale = 1;
  for (std::size_t i = 0; i < radices.size(); ++i) {                   // the first factor fastest
    const std::int64_t c = codes[i];
    if (c < 0) return -1;
    if (c >= radices[i]) throw std::invalid_argument("product_code: member " + std::to_string(c) + " of factor " + std::to_string(i) + ", which has " + std::to_string(radices[i]));
    code += c * scale;
    scale *= radices[i];
  }
  return code;
}

std::vector<std::int64_t> product_codes(const std::vector<std::int64_t>& radices, std::int64_t code) {
  if (code < 0) throw std::invalid_argument("product_codes: code " + std::to_string(code));
  std::vector<std::int64_t> out;
  out.reserve(radices.size());
  std::int64_t rest = code;
  for (std::int64_t n : radices) {
    if (n < 1) throw std::invalid_argument("product_codes: a factor with no members");
    out.push_back(rest % n);
    rest /= n;
  }
  if (rest != 0) throw std::invalid_argument("product_codes: " + std::to_string(code) + " is past the product of the radices");
  return out;
}

std::vector<std::optional<std::int64_t>> factor_labels(const Selection& p, std::int64_t k) {
  need_product(p, "factor_labels");
  const std::vector<std::int64_t> at = product_codes(p.radices, k);
  std::vector<std::optional<std::int64_t>> out;
  for (std::size_t f = 0; f < at.size(); ++f) {
    const Selection& x = p.factors[f];
    out.push_back(at[f] < count(x) ? member_label(x, at[f]) : std::nullopt);   // a grid's member: none (its number names it)
  }
  return out;
}

}  // namespace nxr::selection
