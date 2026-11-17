// nxr::selection (compute) — A SELECTION IN MEMORY: the Cortical Flow Studio's model (the schema's `selection` and its detail rows), as compute creates it
// before io stores it (spec docs/specs/2026-11-17-selection-module.md, section 0). Compute takes arrays; a selection here holds what its type is described
// by, never a stored id or path (the pipeline and io add those).
//
//   point      one element                              center
//   span       bounds per axis, [start, stop)           extents (start == stop on a line: an instant)
//   distance   a centre and a radius under the metric   center, radius, unit
//   indices    an explicit list of elements             indices
//   set        members, given one of four ways:         members (selections: spans, regions, …) · member_of (a MEMBER NUMBER per element) · tiling (spans by a
//                                                       rule) · factors (a PRODUCT: its members are the tuples of the factors' members, on the product manifold)
//              and possibly LEVELS (a grid hierarchy: nested partitions, each with its own member numbers and a parent map to the level above)
//
// IDENTITY AND LABEL ARE TWO THINGS. A member's IDENTITY is its member number (its ordinal: 0 .. count - 1, dense); its LABEL is its `code`, read with its set's
// lookup. Five bursts are five span members with one label; two members may share a label. PARTITION is a property (disjoint members covering the manifold), not
// a type. A kernel is not a selection: it is the weights a measurement uses.
//
// MEMBER NUMBERS. Listed members: their position (the label is the member's `code`). A tiling: k (no label). A set per element (`member_of`): each element's member
// number, dense 0 .. count - 1, -1 for an element in no member; each member carries its label in `member_codes` (one per member), exactly as a listed member
// carries `code`. ONE count: the members (`count`). To build such a set from labels per element (an atlas whose elements carry 1035, 2035), `from_labels` maps the
// labels to member numbers once; nothing after that reads a label as a number. A selection that is not a set: one member, number 0 (itself). A PRODUCT of
// factors F_0 .. F_{n-1}: the tuple (i_0, .., i_{n-1}) of the factors' member numbers, numbered ROW-MAJOR (the last factor fastest): k = ((i_0 n_1 + i_1) n_2 +
// i_2) .. ; its label is the tuple of the factors' labels, each read with its own factor's lookup (a product carries no lookup of its own).
//
// LEVELS. A set with levels has the levels as `members`, coarsest first (a dyadic partition) or finest first (a constant-Q ladder, whose levels are tilings); a
// level's IDENTITY is its position among them, its LABEL (`code`, and `level`) its power of two (tiles of 2^level samples; 2^level tiles a piece), and `band`
// what it resolves. A level of a set per element holds its own members' labels (`member_codes`) and a PARENT map (`parent`: member j of the level -> its member
// in the level above); the set's own `member_of` and `member_codes` are the finest level's. The resolved forms are in resolved.h.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/selection/lookup.h"

namespace nxr::chart { class ChartKit; }

namespace nxr::selection {

enum class Type { point, span, distance, indices, set };
const char* type_name(Type t);

/** a span's bounds on one axis, in that axis's coordinates */
struct Extent { int axis = 0; double start = 0, stop = 0; };
/** a set's members by a rule: `count` spans on axis 0 of `width`, a new one every `hop`, from `origin` (member k: [origin + k hop, origin + k hop + width)) */
struct Tiling { double origin = 0, width = 0, hop = 0; std::int64_t count = 0; };
/** a span on a line as an input: [start, stop) and its label */
struct SpanRow { double start = 0, stop = 0; std::optional<std::int64_t> code; };

struct Selection {
  Type type = Type::set;
  std::optional<std::int64_t> code;        // its label among its siblings (read with its set's lookup)
  std::string unit;                         // a span's or a tiling's coordinates · a distance's radius · a set per element on a surface: the positions'

  std::optional<std::int64_t> center;      // point, distance: the centre element
  double radius = 0;                        // distance

  std::vector<Extent> extents;              // span

  std::vector<std::int32_t> indices;        // indices (sorted, each once)

  // set: its members, one of four ways
  std::vector<Selection> members;           // as selections (a member's number is its position); with levels, the levels in order (a level of a set per element
                                            // holds its members' labels and its parent map, never a number per element: member_at(the set, element, level))
  std::vector<std::int32_t> member_of;      // as a MEMBER NUMBER per element (0 .. count - 1, -1: none); with levels, the finest level's
  std::vector<std::int64_t> member_codes;   // a set per element, or one of its levels: each member's LABEL (one per member), read with the lookup
  std::vector<std::int32_t> parent;         // a level of a set per element: member j of this level -> its member in the level above (empty for the first level)
  std::optional<Tiling> tiling;             // as spans by a rule
  std::vector<Selection> factors;           // as a PRODUCT of these selections (each on its own manifold; a factor that is not a set is one member)
  int level = -1;                           // a level: its label, a power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit` (Hz; 1/unit^2)
  std::string band_unit;
  Lookup lookup;                            // a set: what its members' labels mean
};

// ── making ─────────────────────────────────────────────────────────────────────────────
Selection point(std::int64_t center, std::optional<std::int64_t> code = std::nullopt);
/** a span: bounds per axis; std::invalid_argument for stop < start, a non-finite bound or an axis bounded twice */
Selection span(std::vector<Extent> extents, std::optional<std::int64_t> code = std::nullopt, std::string unit = {});
Selection distance(std::int64_t center, double radius, std::string unit, std::optional<std::int64_t> code = std::nullopt);
Selection indices(std::vector<std::int32_t> elements, std::optional<std::int64_t> code = std::nullopt);
/** a set whose members are the given selections, labelled through `lookup` (each member's code must be in it, unless it is empty) */
Selection set(std::vector<Selection> members, Lookup lookup = {});
/** a set of spans on a line (events, bursts, bad segments, named bands), in time order */
Selection set_of_spans(std::vector<SpanRow> rows, std::string unit, Lookup lookup = {});
/** a set of spans on a line by a rule (a tiling) */
Selection tiled(Tiling tiling, std::string unit);
/** a set per element by MEMBER NUMBER: member_of[v] in 0 .. n - 1 (-1: none) and each member's label (`member_codes`, n of them; two members may share a label;
 *  empty: member i is labelled i), named through `lookup` (each label must be in it, when it is not empty). std::invalid_argument for a number outside 0 .. n - 1
 *  (n is the labels' count, or the largest number + 1 when none are given). */
Selection set_of_members(std::vector<std::int32_t> member_of, std::vector<std::int64_t> member_codes = {}, Lookup lookup = {});
/** a set per element from LABELS per element (an imported atlas: 1035, 2035, flags; -1: none), mapped to member numbers ONCE: one member for each distinct label,
 *  numbered in ascending label order, its label kept in `member_codes`. The one place a label becomes a number. */
Selection from_labels(const std::vector<std::int32_t>& element_labels, Lookup lookup = {});
/** the PRODUCT of two or more selections, each on its own manifold (time x frequency, time x cortex, time x channels x frequency): a set whose members are the
 *  tuples of the factors' members, numbered row-major (the last factor fastest). A factor that is itself a product is flattened into this one; a factor that is
 *  not a set is one member (a burst = product(a span on time, the indices of its channels)). std::invalid_argument for fewer than two factors or a factor with
 *  levels (name one: level(set, k)). */
Selection product(std::vector<Selection> factors);

// ── reading ────────────────────────────────────────────────────────────────────────────
/** a set: how many members, the one count (listed members, a tiling's count, a set per element its members, a product the product of its factors'; a set with
 *  levels its finest level's members, or its levels when they are tilings); 1 for a selection that is not a set. std::invalid_argument for a product too large to number. */
std::int64_t count(const Selection& set);
/** does the set have LEVELS (members that are levels: a constant-Q ladder's tilings, a dyadic partition's levels)? */
bool has_levels(const Selection& s);
/** a set with levels: the level labelled `k` (its power of two) as a selection of its own, the grammar's `level(k)`: a ladder of tilings gives that level's tiling; a
 *  set per element gives the member numbers at that level (through the parent maps), the level's labels and band. std::invalid_argument for a set without levels or
 *  a level it lacks. */
Selection level(const Selection& set, int k);
/** a set with levels: the ORDINAL (identity) of the level labelled `k`; std::invalid_argument when it has none */
int level_ordinal(const Selection& set, int k);
/** a product: member k's factor ordinals (row-major); std::invalid_argument for k outside the product */
std::vector<std::int64_t> factor_ordinals(const Selection& product, std::int64_t k);
/** a product: the member ordinal of the factors' ordinals (one per factor, each inside its factor) */
std::int64_t product_ordinal(const Selection& product, const std::vector<std::int64_t>& ordinals);
/** a product: member k's LABEL, the factors' labels (one per factor, each read with its factor's lookup): a listed member's code, a set per element's member
 *  label, none for a tiling's member, a selection that is not a set its own code */
std::vector<std::optional<std::int64_t>> factor_labels(const Selection& product, std::int64_t k);
/** a set given by members or a tiling: member k (a tiling's as a span); a product: the product of its factors' members (one member each) */
Selection member(const Selection& set, std::int64_t k);
/** a set of spans on a line (listed or tiled): the members holding `value` (none, one, several where they overlap; an instant holds its own value) */
std::vector<std::int64_t> members_at(const Selection& set, double value);
/** a set per element: the MEMBER NUMBER of `element` at level ORDINAL `level` (a set with levels; -1: the set's own, the finest); -1 for none */
std::int64_t member_at(const Selection& set, std::int64_t element, int level = -1);
/** a set per element: the LABEL of the member holding `element` at `level` (as member_at), or nothing when the element is in none */
std::optional<std::int64_t> label_at(const Selection& set, std::int64_t element, int level = -1);
/** a set: member `member`'s LABEL (at level ORDINAL `level`, -1: the set's own): a listed member's code, a set per element's `member_codes`, none for a tiling's
 *  member; std::invalid_argument for a member outside the set */
std::optional<std::int64_t> member_label(const Selection& set, std::int64_t member, int level = -1);
/** a set per element: the member numbers (at `level`) whose label is `label`: none, one, or several (two members may share a label) */
std::vector<std::int64_t> members_with_label(const Selection& set, std::int64_t label, int level = -1);
/** a set per element: member number `member` (at `level`) as an indices selection carrying its label (an atlas region, a tile); empty when it holds no element;
 *  std::invalid_argument for a member outside the set */
Selection member_indices(const Selection& set, std::int64_t member, int level = -1);
/** a set: are its members disjoint (no element in two)? Per element: always. A tiling: hop >= width. Spans: no two overlap. Indices: no shared element.
 *  A product: every factor's are (a factor that is not a set is one member). Other members: false (not decided here). */
bool disjoint(const Selection& set);
/** distance: the elements it holds, by heat-method geodesic distance (geometry-central, through the chart kit) */
Selection resolve_distance(const Selection& distance, chart::ChartKit& kit);

}  // namespace nxr::selection
