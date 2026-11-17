// nxr::selection (compute) — THE RESOLVED FORMS: the few fixed array forms a compute takes a selection in (spec docs/specs/2026-11-17-selection-module.md,
// section 8). A selection (selection.h) says what a part of a manifold IS; its resolved form says which elements it holds, as arrays a compute loops over.
// Every compute that works over a selection takes one of these, resolved from a `Selection` by the resolvers below, never a parameter of its own (a tile
// width and count, a segment length and overlap, bin groups, a tile per sample, a `good` mask, a code array under another name).
//
//   Rule        CLOSED FORM: a regular family of intervals on a line, by a rule        origin, width, hop, count (in elements), the line's size
//               member k = [origin + k hop, origin + k hop + width) ∩ [0, size); evaluated, never materialized (a time tiling, Welch's segments, chunks)
//   Intervals   per member, sorted disjoint [start, stop) index pairs on a line (CSR)   offsets [members + 1], start, stop [intervals], the line's size
//               (events, bursts, bad segments, a band's bins, a run's span, the good samples: one member of many intervals)
//   Elements    per member, its sorted elements (CSR)                                    offsets [members + 1], elements, the manifold's size
//               (a drawn region, picked channels, a disc's vertices; members may overlap)
//   MemberOfElement  the MEMBER NUMBER of each element (-1: in none) and the member count   member_of [size], members
//               (a parcellation, a level of a tile ladder, a tile per sample); DISJOINT members only. A number, never a label: labels are the set's
//               `member_codes` (selection.h)
//
// A member's IDENTITY is its member number (dense, 0 .. members - 1); every form keeps every member, an empty one included, so a measurement's k-th value is
// always member k's. Indices are int64 on a line (hours of samples) and in Elements (a product manifold's cells); member numbers are int32, as every compute
// that takes them reads them.
//
// ROUNDING ONTO A LINE (a span's corners in the line's coordinates → sample indices): a span [a, b) holds the points of the line whose coordinate lies in
// it, i.e. indices [ceil((a - origin) / spacing), ceil((b - origin) / spacing)), each ceil taken with a 1e-9 tolerance so that a corner ON a point is that
// point. An INSTANT (a = b) holds the one point nearest it. A distance on a line, centre element c and radius r, is the span [c - r, c + r) in coordinates.
// A tiling resolves to a Rule only when its origin, width and hop are whole points of the line (within 1e-9), else to Intervals, member by member.
//
// PRODUCTS (section 9): a product selection resolves to its factors' forms plus the row-major numbering (`ProductForm`): a compute reads member
// (i_0, .., i_{n-1}) from the factors' forms and never materializes the product.
//
// Pure functions of their arguments: no file, no stored id; std::invalid_argument names the type or property refused and the resolver to use instead.
#pragma once
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "nxr/selection/partition.h"
#include "nxr/selection/selection.h"

namespace nxr::selection {

// ── the forms ─────────────────────────────────────────────────────────────────────────────
/** CLOSED FORM on a line of `size` points: member k (0 <= k < count) is [origin + k hop, origin + k hop + width) clipped to [0, size). origin may be
 *  negative or past the line (a tiling that starts before a block's first sample); width > 0, hop > 0. */
struct Rule {
  std::int64_t origin = 0, width = 0, hop = 0, count = 0;
  std::int64_t size = 0;                     // the line's points
};

/** per member, sorted, disjoint, non-adjacent [start, stop) index intervals on a line of `size` points (CSR: member m's are offsets[m] .. offsets[m+1]) */
struct Intervals {
  std::int64_t size = 0;                     // the line's points
  std::vector<std::int64_t> offsets{0};      // [members + 1]
  std::vector<std::int64_t> start, stop;     // [intervals]
  std::int64_t members() const { return static_cast<std::int64_t>(offsets.size()) - 1; }
};

/** per member, its sorted elements, each once, on a manifold of `size` elements (CSR: member m's are offsets[m] .. offsets[m+1]); members may overlap */
struct Elements {
  std::int64_t size = 0;                     // the manifold's elements
  std::vector<std::int64_t> offsets{0};      // [members + 1]
  std::vector<std::int64_t> elements;        // [total]
  std::int64_t members() const { return static_cast<std::int64_t>(offsets.size()) - 1; }
};

/** the member NUMBER of each element (-1: in no member), members 0 .. members - 1: DISJOINT members only */
struct MemberOfElement {
  std::vector<std::int32_t> member_of;       // [size]
  std::int64_t members = 0;
  std::int64_t size() const { return static_cast<std::int64_t>(member_of.size()); }
};

/** one of the forms */
using Form = std::variant<Rule, Intervals, Elements, MemberOfElement>;
enum class FormKind { rule, intervals, elements, member_of };
const char* form_name(FormKind kind);

/** how many members a form has (empty ones included) */
std::int64_t member_count(const Form& form);
/** the size of the manifold (or line) a form lies on */
std::int64_t form_size(const Form& form);

// ── resolving a selection ─────────────────────────────────────────────────────────────────
// A selection with levels is refused everywhere (name one: level(set, k)); a product is refused by all but resolve_product.

/** CLOSED FORM. Accepts: a set by a rule (a tiling) on `line`, whose origin, width and hop are whole points of it. Refuses every other selection
 *  (resolve_intervals) and a tiling that is not whole points (resolve_intervals rounds its members one by one). */
Rule resolve_rule(const Selection& s, const Line& line);

/** INTERVALS on `line`. Accepts (each with its unit, when it has one, equal to the line's):
 *   point     one member: its element                         span      one member: its axis-0 bounds (a span on several axes: a product)
 *   distance  one member: [c - r, c + r) in coordinates        indices   one member: its runs of consecutive elements
 *   set       a tiling (member k its interval), a set per element over the line's points (member m: the runs of m), listed members (each a point, a span on one
 *             axis, a distance or indices: its intervals; a member that is a set is refused)
 *  Intervals are clipped to the line; a member that falls outside it is kept, empty. */
Intervals resolve_intervals(const Selection& s, const Line& line);

/** ELEMENTS on a manifold of `size` elements. Accepts: a point, indices (one member), a set per element over the `size` elements (member m: its elements), a set
 *  of listed points and indices. Refuses a distance on a surface (resolve_distance first: it needs the metric), a span or a tiling (they lie on a line:
 *  resolve_intervals, then elements_of), and an element outside 0 .. size - 1. */
Elements resolve_elements(const Selection& s, std::int64_t size);

/** MEMBER-OF-ELEMENT on a manifold of `size` elements: as resolve_elements, then member_of_from (a set per element is taken as it is). Refuses members that share an element. */
MemberOfElement resolve_member_of(const Selection& s, std::int64_t size);
/** MEMBER-OF-ELEMENT on a line's points: as resolve_intervals, then member_of_from (a tile per sample). Refuses members that share a point. */
MemberOfElement resolve_member_of(const Selection& s, const Line& line);

/** where a factor of a product (or any selection) lies, and the form wanted: on a line (`line` set) or on `size` elements */
struct Space {
  FormKind form = FormKind::intervals;
  std::optional<Line> line;                  // a line: rule, intervals or member numbers over its points; none: elements or member numbers over `size` elements
  std::int64_t size = 0;
};
/** a selection in the form `space` asks for (the resolver above that form names) */
Form resolve(const Selection& s, const Space& space);

// ── conversions between forms (exact; member numbers need disjoint members) ────────────────────────
Intervals intervals_of(const Rule& rule);
Intervals intervals_of(const Elements& elements);   // each member's runs of consecutive elements
Intervals intervals_of(const MemberOfElement& member_of);
Elements elements_of(const Intervals& intervals);
Elements elements_of(const MemberOfElement& member_of);
/** std::invalid_argument when two members share an element, or a member count beyond int32 */
MemberOfElement member_of_from(const Elements& elements);
MemberOfElement member_of_from(const Intervals& intervals);
MemberOfElement member_of_from(const Rule& rule);

// ── the closed form, evaluated ────────────────────────────────────────────────────────────
/** member k of a rule, clipped to the line: [start, stop) (start == stop: outside it) */
std::pair<std::int64_t, std::int64_t> rule_member(const Rule& rule, std::int64_t k);
/** the members holding point i: [first, stop) of member numbers (empty: none); several where hop < width */
std::pair<std::int64_t, std::int64_t> rule_members_at(const Rule& rule, std::int64_t i);
/** the member of each point first .. first + n - 1 (-1: none, or outside the line): the tile per sample of a block, without the whole line's member numbers.
 *  std::invalid_argument when members overlap (hop < width). */
std::vector<std::int32_t> rule_member_of(const Rule& rule, std::int64_t first, std::int64_t n);

// ── products (section 9) ──────────────────────────────────────────────────────────────────
/** a product resolved: each factor in its own form and the row-major numbering of the tuples; the product itself is never materialized */
struct ProductForm {
  std::vector<Form> factors;                 // factor f in the form its Space asked for
  std::vector<std::int64_t> counts;          // factor f's member ordinals (= member_count(factors[f]))
  /** the product's members: the product of counts */
  std::int64_t members() const;
  /** member k's factor ordinals (the last factor fastest) */
  std::vector<std::int64_t> ordinals(std::int64_t k) const;
  /** the member of the factors' ordinals */
  std::int64_t ordinal(const std::vector<std::int64_t>& factor_ordinals) const;
};
/** a product, each factor resolved in its Space (one per factor, in order). std::invalid_argument for a selection that is not a product, a count of spaces
 *  that is not the factors', or a factor its form refuses. A factor that is not a set resolves to one member. */
ProductForm resolve_product(const Selection& product, const std::vector<Space>& spaces);

}  // namespace nxr::selection
