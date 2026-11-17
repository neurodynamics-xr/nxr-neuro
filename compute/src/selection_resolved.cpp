// nxr::selection (compute) — the resolved forms (resolved.h): a selection as the arrays a compute loops over.
#include "nxr/selection/resolved.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace nxr::selection {

namespace {
constexpr double kTolerance = 1e-9;                                    // a corner within this many points of a point is on it

[[noreturn]] void refuse(const std::string& what) { throw std::invalid_argument(what); }

std::string describe(const Selection& s) {
  if (s.type != Type::set) return std::string("a ") + type_name(s.type);
  if (!s.factors.empty()) return "a product";
  if (s.tiling) return "a set by a rule (a tiling)";
  if (!s.codes.empty()) return "a set by codes";
  return "a set of listed members";
}

// a product and a set with levels are resolved only by name: resolve_product, level(set, k)
void need_plain(const Selection& s, const char* what) {
  if (s.type == Type::set && !s.factors.empty()) refuse(std::string(what) + ": a product; resolve it with resolve_product");
  if (has_levels(s)) refuse(std::string(what) + ": a set with levels; name one (level(set, k))");
}

void need_line(const Line& line, const char* what) {
  if (line.count < 0 || !(line.spacing > 0) || !std::isfinite(line.spacing) || !std::isfinite(line.origin))
    refuse(std::string(what) + ": a line of count >= 0 points, spacing > 0 apart, from a finite origin");
}

void need_unit(const Selection& s, const Line& line, const char* what) {
  if (!s.unit.empty() && !line.unit.empty() && s.unit != line.unit)
    refuse(std::string(what) + ": the selection is in " + s.unit + ", the line in " + line.unit);
}

double position(double x, const Line& line) { return (x - line.origin) / line.spacing; }          // a coordinate, in points from the line's first
std::int64_t first_point_from(double p) { return static_cast<std::int64_t>(std::ceil(p - kTolerance)); }   // the first point at or after p

std::int64_t floor_div(std::int64_t a, std::int64_t b) {                // b > 0
  std::int64_t q = a / b;
  if (a % b != 0 && a < 0) --q;
  return q;
}

std::int32_t as_code(std::int64_t m, const char* what) {
  if (m > std::numeric_limits<std::int32_t>::max()) refuse(std::string(what) + ": more members than a 32-bit code holds");
  return static_cast<std::int32_t>(m);
}

// Intervals, member by member: each member's intervals clipped to the line, sorted, overlapping or touching ones merged
struct IntervalBuilder {
  Intervals out;
  std::vector<std::pair<std::int64_t, std::int64_t>> pending;
  explicit IntervalBuilder(std::int64_t size) { out.size = size; }
  void add(std::int64_t a, std::int64_t b) {
    a = std::max<std::int64_t>(a, 0);
    b = std::min(b, out.size);
    if (b > a) pending.emplace_back(a, b);
  }
  void close() {
    std::sort(pending.begin(), pending.end());
    for (const auto& [a, b] : pending) {
      const bool joins = out.start.size() > static_cast<std::size_t>(out.offsets.back()) && a <= out.stop.back();
      if (joins) out.stop.back() = std::max(out.stop.back(), b);
      else { out.start.push_back(a); out.stop.push_back(b); }
    }
    out.offsets.push_back(static_cast<std::int64_t>(out.start.size()));
    pending.clear();
  }
};

// the intervals of one member (a selection that is not a set) on a line
void add_member(IntervalBuilder& b, const Selection& m, const Line& line) {
  switch (m.type) {
    case Type::point:
      b.add(*m.center, *m.center + 1);
      return;
    case Type::span: {
      if (m.extents.size() != 1 || m.extents[0].axis != 0)
        refuse("resolve_intervals: a span on several axes is a box on a product manifold; write it as product(spans on each line) and use resolve_product");
      need_unit(m, line, "resolve_intervals");
      const Extent& e = m.extents[0];
      if (e.start == e.stop) {                                         // an instant: the point nearest it
        const auto i = static_cast<std::int64_t>(std::llround(position(e.start, line)));
        b.add(i, i + 1);
      } else {
        b.add(first_point_from(position(e.start, line)), first_point_from(position(e.stop, line)));
      }
      return;
    }
    case Type::distance: {                                             // on a line, the span [c - r, c + r)
      need_unit(m, line, "resolve_intervals");
      const double c = line.origin + static_cast<double>(*m.center) * line.spacing;
      b.add(first_point_from(position(c - m.radius, line)), first_point_from(position(c + m.radius, line)));
      return;
    }
    case Type::indices: {
      std::vector<std::int32_t> v = m.indices;
      std::sort(v.begin(), v.end());
      for (std::size_t i = 0; i < v.size();) {                         // a run of consecutive elements is one interval
        std::size_t j = i + 1;
        while (j < v.size() && v[j] <= v[j - 1] + 1) ++j;
        b.add(v[i], static_cast<std::int64_t>(v[j - 1]) + 1);
        i = j;
      }
      return;
    }
    case Type::set:
      refuse("resolve_intervals: a member that is itself a set (" + describe(m) + "); resolve it on its own");
  }
}
}  // namespace

const char* form_name(FormKind kind) {
  switch (kind) {
    case FormKind::rule: return "rule";
    case FormKind::intervals: return "intervals";
    case FormKind::elements: return "elements";
    case FormKind::codes: return "codes";
  }
  return "?";
}

std::int64_t member_count(const Form& f) {
  if (const auto* r = std::get_if<Rule>(&f)) return r->count;
  if (const auto* i = std::get_if<Intervals>(&f)) return i->members();
  if (const auto* e = std::get_if<Elements>(&f)) return e->members();
  return std::get<Codes>(f).members;
}

std::int64_t form_size(const Form& f) {
  if (const auto* r = std::get_if<Rule>(&f)) return r->size;
  if (const auto* i = std::get_if<Intervals>(&f)) return i->size;
  if (const auto* e = std::get_if<Elements>(&f)) return e->size;
  return std::get<Codes>(f).size();
}

// ── resolving ────────────────────────────────────────────────────────────────────────────
Rule resolve_rule(const Selection& s, const Line& line) {
  need_line(line, "resolve_rule");
  need_plain(s, "resolve_rule");
  if (s.type != Type::set || !s.tiling) refuse("resolve_rule: a set by a rule (a tiling), not " + describe(s) + "; resolve_intervals takes any selection on a line");
  need_unit(s, line, "resolve_rule");
  const Tiling& t = *s.tiling;
  auto whole = [&](double x, const char* name) {
    const double p = x / line.spacing, r = std::round(p);
    if (std::abs(p - r) > kTolerance * std::max(1.0, std::abs(p)))
      refuse(std::string("resolve_rule: the tiling's ") + name + " is not a whole number of points of the line; resolve_intervals rounds its members one by one");
    return static_cast<std::int64_t>(r);
  };
  Rule r;
  r.origin = whole(t.origin - line.origin, "origin");
  r.width = whole(t.width, "width");
  r.hop = whole(t.hop, "hop");
  r.count = t.count;
  r.size = line.count;
  if (r.width < 1 || r.hop < 1) refuse("resolve_rule: the tiling's width and hop are at least one point of the line");
  return r;
}

Intervals resolve_intervals(const Selection& s, const Line& line) {
  need_line(line, "resolve_intervals");
  need_plain(s, "resolve_intervals");
  IntervalBuilder b(line.count);
  if (s.type != Type::set) {
    add_member(b, s, line);
    b.close();
    return std::move(b.out);
  }
  if (s.tiling) {                                                      // member k by the span rule, so a tiling and its listed spans resolve alike
    need_unit(s, line, "resolve_intervals");
    const Tiling& t = *s.tiling;
    for (std::int64_t k = 0; k < t.count; ++k) {
      const double start = t.origin + static_cast<double>(k) * t.hop;
      b.add(first_point_from(position(start, line)), first_point_from(position(start + t.width, line)));
      b.close();
    }
    return std::move(b.out);
  }
  if (!s.codes.empty()) {
    if (static_cast<std::int64_t>(s.codes.size()) != line.count)
      refuse("resolve_intervals: a set by codes over " + std::to_string(s.codes.size()) + " elements, on a line of " + std::to_string(line.count) + " points");
    Codes c;
    c.codes = s.codes;
    c.members = ordinal_count(s);
    return intervals_of(c);
  }
  for (const Selection& m : s.members) { add_member(b, m, line); b.close(); }
  return std::move(b.out);
}

Elements resolve_elements(const Selection& s, std::int64_t size) {
  need_plain(s, "resolve_elements");
  if (size < 0) refuse("resolve_elements: a manifold of size >= 0");
  Elements out;
  out.size = size;
  auto check = [&](std::int64_t v) {
    if (v < 0 || v >= size) refuse("resolve_elements: element " + std::to_string(v) + " outside 0.." + std::to_string(size - 1));
  };
  auto add_one = [&](const Selection& m) {
    const auto from = static_cast<std::ptrdiff_t>(out.elements.size());
    switch (m.type) {
      case Type::point: check(*m.center); out.elements.push_back(*m.center); break;
      case Type::indices: for (std::int32_t v : m.indices) { check(v); out.elements.push_back(v); } break;
      case Type::distance: refuse("resolve_elements: a distance needs its manifold's metric; resolve_distance(selection, chart) first (on a line: resolve_intervals)");
      case Type::span: refuse("resolve_elements: a span lies on a line; resolve_intervals(selection, line), then elements_of");
      case Type::set: refuse("resolve_elements: a member that is itself a set (" + describe(m) + "); resolve it on its own");
    }
    std::sort(out.elements.begin() + from, out.elements.end());
    out.elements.erase(std::unique(out.elements.begin() + from, out.elements.end()), out.elements.end());
    out.offsets.push_back(static_cast<std::int64_t>(out.elements.size()));
  };
  if (s.type != Type::set) { add_one(s); return out; }
  if (s.tiling) refuse("resolve_elements: a tiling lies on a line; resolve_rule or resolve_intervals(selection, line), then elements_of");
  if (!s.codes.empty()) {
    if (static_cast<std::int64_t>(s.codes.size()) != size)
      refuse("resolve_elements: a set by codes over " + std::to_string(s.codes.size()) + " elements, on a manifold of " + std::to_string(size));
    Codes c;
    c.codes = s.codes;
    c.members = ordinal_count(s);
    return elements_of(c);
  }
  for (const Selection& m : s.members) add_one(m);
  return out;
}

Codes resolve_codes(const Selection& s, std::int64_t size) {
  need_plain(s, "resolve_codes");
  if (s.type == Type::set && !s.codes.empty()) {                     // already in this form
    if (static_cast<std::int64_t>(s.codes.size()) != size)
      refuse("resolve_codes: a set by codes over " + std::to_string(s.codes.size()) + " elements, on a manifold of " + std::to_string(size));
    Codes c;
    c.codes = s.codes;
    c.members = ordinal_count(s);
    return c;
  }
  return codes_of(resolve_elements(s, size));
}

Codes resolve_codes(const Selection& s, const Line& line) {
  need_line(line, "resolve_codes");
  need_plain(s, "resolve_codes");
  if (s.type == Type::set && !s.codes.empty()) return resolve_codes(s, line.count);
  return codes_of(resolve_intervals(s, line));
}

Form resolve(const Selection& s, const Space& space) {
  switch (space.form) {
    case FormKind::rule:
      if (!space.line) refuse("resolve: the rule form lies on a line; the space names none");
      return resolve_rule(s, *space.line);
    case FormKind::intervals:
      if (!space.line) refuse("resolve: the intervals form lies on a line; the space names none");
      return resolve_intervals(s, *space.line);
    case FormKind::elements:
      if (space.line) return elements_of(resolve_intervals(s, *space.line));
      return resolve_elements(s, space.size);
    case FormKind::codes:
      if (space.line) return resolve_codes(s, *space.line);
      return resolve_codes(s, space.size);
  }
  refuse("resolve: an unknown form");
}

// ── conversions ──────────────────────────────────────────────────────────────────────────
Intervals intervals_of(const Rule& r) {
  IntervalBuilder b(r.size);
  for (std::int64_t k = 0; k < r.count; ++k) {
    const auto [a, z] = rule_member(r, k);
    b.add(a, z);
    b.close();
  }
  return std::move(b.out);
}

Intervals intervals_of(const Elements& e) {
  IntervalBuilder b(e.size);
  for (std::int64_t m = 0; m < e.members(); ++m) {
    const auto lo = static_cast<std::size_t>(e.offsets[static_cast<std::size_t>(m)]), hi = static_cast<std::size_t>(e.offsets[static_cast<std::size_t>(m) + 1]);
    for (std::size_t i = lo; i < hi;) {
      std::size_t j = i + 1;
      while (j < hi && e.elements[j] <= e.elements[j - 1] + 1) ++j;
      b.add(e.elements[i], e.elements[j - 1] + 1);
      i = j;
    }
    b.close();
  }
  return std::move(b.out);
}

Intervals intervals_of(const Codes& c) { return intervals_of(elements_of(c)); }

Elements elements_of(const Intervals& in) {
  Elements out;
  out.size = in.size;
  for (std::int64_t m = 0; m < in.members(); ++m) {
    for (auto i = in.offsets[static_cast<std::size_t>(m)]; i < in.offsets[static_cast<std::size_t>(m) + 1]; ++i)
      for (auto v = in.start[static_cast<std::size_t>(i)]; v < in.stop[static_cast<std::size_t>(i)]; ++v) out.elements.push_back(v);
    out.offsets.push_back(static_cast<std::int64_t>(out.elements.size()));
  }
  return out;
}

Elements elements_of(const Codes& c) {
  Elements out;
  out.size = c.size();
  std::vector<std::int64_t> counts(static_cast<std::size_t>(c.members), 0);
  for (std::int32_t k : c.codes) {
    if (k < -1 || k >= c.members) refuse("elements_of: code " + std::to_string(k) + " outside -1.." + std::to_string(c.members - 1));
    if (k >= 0) ++counts[static_cast<std::size_t>(k)];
  }
  out.offsets.assign(static_cast<std::size_t>(c.members) + 1, 0);
  for (std::size_t m = 0; m < counts.size(); ++m) out.offsets[m + 1] = out.offsets[m] + counts[m];
  out.elements.resize(static_cast<std::size_t>(out.offsets.back()));
  std::vector<std::int64_t> next(out.offsets.begin(), out.offsets.end() - 1);
  for (std::size_t v = 0; v < c.codes.size(); ++v)                     // in element order, so each member's elements come sorted
    if (c.codes[v] >= 0) out.elements[static_cast<std::size_t>(next[static_cast<std::size_t>(c.codes[v])]++)] = static_cast<std::int64_t>(v);
  return out;
}

Codes codes_of(const Elements& e) {
  Codes out;
  out.members = e.members();
  as_code(out.members, "codes_of");
  out.codes.assign(static_cast<std::size_t>(e.size), -1);
  for (std::int64_t m = 0; m < e.members(); ++m)
    for (auto i = e.offsets[static_cast<std::size_t>(m)]; i < e.offsets[static_cast<std::size_t>(m) + 1]; ++i) {
      const std::int64_t v = e.elements[static_cast<std::size_t>(i)];
      if (v < 0 || v >= e.size) refuse("codes_of: element " + std::to_string(v) + " outside 0.." + std::to_string(e.size - 1));
      std::int32_t& slot = out.codes[static_cast<std::size_t>(v)];
      if (slot >= 0) refuse("codes_of: members " + std::to_string(slot) + " and " + std::to_string(m) + " share element " + std::to_string(v) + "; codes need disjoint members");
      slot = static_cast<std::int32_t>(m);
    }
  return out;
}

Codes codes_of(const Intervals& in) {
  Codes out;
  out.members = in.members();
  as_code(out.members, "codes_of");
  out.codes.assign(static_cast<std::size_t>(in.size), -1);
  for (std::int64_t m = 0; m < in.members(); ++m)
    for (auto i = in.offsets[static_cast<std::size_t>(m)]; i < in.offsets[static_cast<std::size_t>(m) + 1]; ++i)
      for (auto v = in.start[static_cast<std::size_t>(i)]; v < in.stop[static_cast<std::size_t>(i)]; ++v) {
        std::int32_t& slot = out.codes[static_cast<std::size_t>(v)];
        if (slot >= 0) refuse("codes_of: members " + std::to_string(slot) + " and " + std::to_string(m) + " share point " + std::to_string(v) + "; codes need disjoint members");
        slot = static_cast<std::int32_t>(m);
      }
  return out;
}

Codes codes_of(const Rule& r) {
  Codes out;
  out.codes = rule_codes(r, 0, r.size);
  out.members = r.count;
  return out;
}

// ── the closed form ──────────────────────────────────────────────────────────────────────
std::pair<std::int64_t, std::int64_t> rule_member(const Rule& r, std::int64_t k) {
  if (k < 0 || k >= r.count) refuse("rule_member: " + std::to_string(k) + " outside 0.." + std::to_string(r.count - 1));
  const std::int64_t a = r.origin + k * r.hop, b = a + r.width;
  const std::int64_t lo = std::clamp<std::int64_t>(a, 0, r.size), hi = std::clamp<std::int64_t>(b, 0, r.size);
  return {lo, std::max(lo, hi)};
}

std::pair<std::int64_t, std::int64_t> rule_members_at(const Rule& r, std::int64_t i) {
  if (i < 0 || i >= r.size || r.hop < 1) return {0, 0};
  // member k holds i when origin + k hop <= i < origin + k hop + width
  const std::int64_t first = std::max<std::int64_t>(floor_div(i - r.origin - r.width, r.hop) + 1, 0);
  const std::int64_t stop = std::min<std::int64_t>(floor_div(i - r.origin, r.hop) + 1, r.count);
  return stop > first ? std::make_pair(first, stop) : std::make_pair(std::int64_t(0), std::int64_t(0));
}

std::vector<std::int32_t> rule_codes(const Rule& r, std::int64_t first, std::int64_t n) {
  if (r.hop < r.width) refuse("rule_codes: the rule's members overlap (hop < width); a code per point needs disjoint members");
  if (n < 0) refuse("rule_codes: n >= 0");
  as_code(r.count, "rule_codes");
  std::vector<std::int32_t> out(static_cast<std::size_t>(n), -1);
  for (std::int64_t j = 0; j < n; ++j) {
    const auto [a, b] = rule_members_at(r, first + j);
    if (b > a) out[static_cast<std::size_t>(j)] = static_cast<std::int32_t>(a);
  }
  return out;
}

// ── products ─────────────────────────────────────────────────────────────────────────────
std::int64_t ProductForm::members() const {
  std::int64_t n = 1;
  for (std::int64_t c : counts) n *= c;                               // resolve_product refused an overflow (ordinal_count)
  return n;
}

std::vector<std::int64_t> ProductForm::ordinals(std::int64_t k) const {
  if (k < 0 || k >= members()) refuse("ProductForm::ordinals: member " + std::to_string(k) + " outside the product");
  std::vector<std::int64_t> out(counts.size());
  for (std::size_t f = counts.size(); f-- > 0;) { out[f] = k % counts[f]; k /= counts[f]; }
  return out;
}

std::int64_t ProductForm::ordinal(const std::vector<std::int64_t>& at) const {
  if (at.size() != counts.size()) refuse("ProductForm::ordinal: " + std::to_string(counts.size()) + " ordinals, one per factor");
  std::int64_t k = 0;
  for (std::size_t f = 0; f < at.size(); ++f) {
    if (at[f] < 0 || at[f] >= counts[f]) refuse("ProductForm::ordinal: ordinal " + std::to_string(at[f]) + " outside factor " + std::to_string(f));
    k = k * counts[f] + at[f];
  }
  return k;
}

ProductForm resolve_product(const Selection& p, const std::vector<Space>& spaces) {
  if (p.type != Type::set || p.factors.empty()) refuse("resolve_product: a product, not " + describe(p));
  if (spaces.size() != p.factors.size())
    refuse("resolve_product: " + std::to_string(p.factors.size()) + " factors, " + std::to_string(spaces.size()) + " spaces (one per factor)");
  ordinal_count(p);                                                    // refuses a product too large to number
  ProductForm out;
  for (std::size_t f = 0; f < spaces.size(); ++f) {
    Form form = resolve(p.factors[f], spaces[f]);
    const std::int64_t c = member_count(form);
    if (c != ordinal_count(p.factors[f]))                              // every form keeps every ordinal; a mismatch is a bug, never a silent renumbering
      refuse("resolve_product: factor " + std::to_string(f) + " resolved to " + std::to_string(c) + " members, not its " + std::to_string(ordinal_count(p.factors[f])));
    out.counts.push_back(c);
    out.factors.push_back(std::move(form));
  }
  return out;
}

}  // namespace nxr::selection
