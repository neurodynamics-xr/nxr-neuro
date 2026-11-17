// nxr::selection (compute) — A SELECTION IN MEMORY: the Cortical Flow Studio's model (the schema's `selection` and its detail rows), as compute creates it
// before io stores it (spec docs/specs/2026-11-17-selection-module.md, section 0). Compute takes arrays; a selection here holds what its type is described
// by, never a stored id or path (the pipeline and io add those).
//
//   point      one element                              center
//   span       bounds per axis, [start, stop)           extents (start == stop on a line: an instant); on a product manifold one extent per factor: a BOX
//   distance   a centre and a radius under the metric   center, radius, unit
//   indices    an explicit list of elements             indices
//   set        members, given one of four ways:         members (listed selections) · member_of (a MEMBER NUMBER per element) · tiling (spans by a rule) ·
//                                                       factors (a PRODUCT of grids, on the product manifold)
//
// INDEX, LABEL, LOOKUP are three things. A member's INDEX is its member number (its position, 0 .. count - 1: its identity); its LABEL is its `code`; the LOOKUP
// (the set's dictionary) says what a label means (a name, a colour; label 0 is `unassigned` where a dictionary has it, as the Studio's do). Five bursts are
// five members with one label; two members may share a label. A kernel is not a selection: it is the weights a measurement uses.
//
// TWO FAMILIES OF SETS (spec section 0):
//   GRID        a PARTITION OF THE MANIFOLD, defined before any data: a chunk grid, a frame tiling, a constant-Q ladder's levels, the dyadic tiles of a surface, the
//               elements of a points manifold (each channel its own member). Disjoint, covering, regular. Its HIERARCHY IS CLOSED FORM: a dyadic level l is the
//               finest member >> (depth - l), a constant-Q level is a tiling rule; nothing per level is held but its label and band. Measurements roll up
//               only on grids, by that arithmetic.
//   IRREGULAR   a PARTITION OF THE FIELD, made by detection on measurements over a grid (thresholding an envelope or a band power): bursts, events, scouts, the
//               DK, DKT and Destrieux atlases. Sparse, may overlap, any extent; NO levels and no roll-ups (statistics across their members are summaries).
//               Listed members (each described by its CENTRE and EXTENT: `span_about` on a line, `distance` on a surface), or a member number per element with
//               a label per member (`member_codes`). Not indexed by a grid: the grid a family was detected over is provenance; a member's grid address is a
//               lookup on request (grid.h).
//
// MEMBER NUMBERS. Listed members: their position (the label is the member's `code`). A tiling: k (no label). An irregular set per element: each element's member
// number, dense, -1 for an element in no member, one label per member in `member_codes`. A dyadic GRID: the member number is the tile code h 2^depth + t (empty
// tiles are empty members: the closed form needs every code). To build an irregular set from labels per element (an atlas whose elements carry 1035, 2035),
// `from_labels` maps the labels to member numbers once; nothing after that reads a label as a number. A selection that is not a set: one member, number 0.
//
// PRODUCTS (the Studio's D100 `productSelection`): a set whose FACTORS are GRIDS, each on its own manifold, on their product manifold. Its member code is MIXED
// RADIX, the FIRST factor fastest: code = c0 + n0 (c1 + n1 (c2 + ..)) (`product_code`, `product_codes`); the radices are fixed when it is made. Its label is the
// tuple of the factors' labels, each read with its own factor's lookup (a product carries no lookup of its own). A box on a product manifold is a `span` with one
// extent per factor, not a product. The JOINT partition (the Studio's D129) is a product whose first (time) factor's radix is rounded up to a power of two, so a
// code is (tile << bits) | frame (`joint`).
//
// LEVELS. A grid with levels (a constant-Q ladder, a dyadic grid) has its levels as `members`: a level's IDENTITY is its position among them, its LABEL (`level`,
// and `code`) its power of two (tiles of 2^level samples; 2^level tiles a piece), and `band` what it resolves. A constant-Q level is a tiling (its rule); a dyadic
// level holds only its label and band: its members are the closed form of the grid's own `member_of`. The resolved forms are in resolved.h; coupled levels (a
// band setting another factor's level) and grid addresses in grid.h.
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

/** a span's bounds on one axis, in that axis's coordinates (on a product manifold an axis is a factor, by ordinal) */
struct Extent { int axis = 0; double start = 0, stop = 0; };
/** an IRREGULAR member's description on one axis: its CENTRE and its EXTENT (a duration, a width), in the axis's coordinates; its bounds are derived,
 *  [center - extent / 2, center + extent / 2). The centre lies in some cell of a grid; the extent decides the grid's level (grid.h). */
struct About { int axis = 0; double center = 0, extent = 0; };
/** a set's members by a rule: `count` spans on axis 0 of `width`, a new one every `hop`, from `origin` (member k: [origin + k hop, origin + k hop + width)) */
struct Tiling { double origin = 0, width = 0, hop = 0; std::int64_t count = 0; };
/** a span on a line as an input: [start, stop) and its label */
struct SpanRow { double start = 0, stop = 0; std::optional<std::int64_t> code; };

/** a GRID's closed form, for the grids that are not tilings (a tiling is its own rule) */
struct Grid {
  enum class Kind {
    elements,                               // every element of a manifold of `size` elements is its own member (the channels of a sensor array); nothing per element is held
    dyadic                                  // the dyadic tiles of a surface: `member_of` holds the finest tile code h 2^depth + t; level l's member = that code >> (depth - l)
  };
  Kind kind = Kind::elements;
  std::int64_t size = 0;                    // elements: the manifold's elements
  int depth = 0;                            // dyadic: the level its `member_of` codes are at (the finest, for the whole grid)
  std::int64_t slots = 0;                   // dyadic: the piece slots the codes reserve (h in 0 .. slots - 1): level l has slots << l members
};

struct Selection {
  Type type = Type::set;
  std::optional<std::int64_t> code;        // its label among its siblings (read with its set's lookup)
  std::string unit;                         // a span's or a tiling's coordinates · a distance's radius · a set per element on a surface: the positions'

  std::optional<std::int64_t> center;      // point, distance: the centre element
  double radius = 0;                        // distance (an irregular member on a surface: its extent about the centre)

  std::vector<Extent> extents;              // span

  std::vector<std::int32_t> indices;        // indices (sorted, each once)

  // set: its members, one of four ways
  std::vector<Selection> members;           // as selections (a member's number is its position); a grid with levels: the levels in order
  std::vector<std::int32_t> member_of;      // as a MEMBER NUMBER per element (-1: none); a dyadic grid: the tile codes at its depth
  std::vector<std::int64_t> member_codes;   // an irregular set per element: each member's LABEL (one per member), read with the lookup
  std::optional<Tiling> tiling;             // as spans by a rule
  std::vector<Selection> factors;           // as a PRODUCT of these GRIDS (each on its own manifold)
  std::vector<std::int64_t> radices;        // a product: each factor's radix, fixed when it is made (its member count; a joint's time factor: the next power of two)
  std::optional<Grid> grid;                 // a grid that is not a tiling: its closed form
  int level = -1;                           // a level: its label, a power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit` (Hz; 1/unit^2)
  std::string band_unit;
  Lookup lookup;                            // a set: what its members' labels mean
};

// ── making ─────────────────────────────────────────────────────────────────────────────
Selection point(std::int64_t center, std::optional<std::int64_t> code = std::nullopt);
/** a span: bounds per axis (a BOX on a product manifold: one extent per factor it constrains, an axis left out is all of it); std::invalid_argument for
 *  stop < start, a non-finite bound or an axis bounded twice */
Selection span(std::vector<Extent> extents, std::optional<std::int64_t> code = std::nullopt, std::string unit = {});
/** an irregular member by CENTRE + EXTENT on each axis it constrains: the span [center - extent / 2, center + extent / 2) per axis (an extent 0: an instant,
 *  which resolves to the sample nearest its centre). std::invalid_argument for a negative or non-finite extent, or an axis given twice. */
Selection span_about(std::vector<About> about, std::optional<std::int64_t> code = std::nullopt, std::string unit = {});
/** a span's description by centre and extent, one per axis it constrains (the inverse of span_about); std::invalid_argument for a selection that is not a span */
std::vector<About> about(const Selection& span);
/** a distance: a centre element and a radius >= 0 under the manifold's metric (an irregular member on a surface: centre vertex + extent; a radius 0 holds the centre) */
Selection distance(std::int64_t center, double radius, std::string unit, std::optional<std::int64_t> code = std::nullopt);
Selection indices(std::vector<std::int32_t> elements, std::optional<std::int64_t> code = std::nullopt);
/** a set whose members are the given selections, labelled through `lookup` (each member's code must be in it, unless it is empty) */
Selection set(std::vector<Selection> members, Lookup lookup = {});
/** a set of spans on a line (events, bursts, bad segments, named bands), in time order: irregular */
Selection set_of_spans(std::vector<SpanRow> rows, std::string unit, Lookup lookup = {});
/** a set of spans on a line by a rule (a tiling): a GRID when hop >= width (chunks, frames); overlapping segments are a family, not a partition */
Selection tiled(Tiling tiling, std::string unit);
/** the GRID of a manifold of `size` elements (a points manifold: the channels): every element its own member, nothing per element held */
Selection elements_grid(std::int64_t size);
/** an IRREGULAR set per element by MEMBER NUMBER: member_of[v] in 0 .. n - 1 (-1: none) and each member's label (`member_codes`, n of them; two members may share
 *  a label; empty: member i is labelled i), named through `lookup` (each label must be in it, when it is not empty). std::invalid_argument for a number outside
 *  0 .. n - 1 (n is the labels' count, or the largest number + 1 when none are given). */
Selection set_of_members(std::vector<std::int32_t> member_of, std::vector<std::int64_t> member_codes = {}, Lookup lookup = {});
/** an irregular set per element from LABELS per element (an imported atlas: 1035, 2035, flags; -1: none), mapped to member numbers ONCE: one member for each
 *  distinct label, numbered in ascending label order, its label kept in `member_codes`. The one place a label becomes a number. */
Selection from_labels(const std::vector<std::int32_t>& element_labels, Lookup lookup = {});
/** a dyadic GRID from the tile code of every element at level `depth` (h 2^depth + t; -1: none) and its levels' descriptions (label and band, coarsest first:
 *  level l at position l; empty: a single level). std::invalid_argument for a code outside 0 .. (slots << depth) - 1 or levels that are not 0 .. depth. */
Selection dyadic_grid(std::vector<std::int32_t> codes, int depth, std::int64_t slots, std::vector<Selection> levels = {});
/** the PRODUCT of two or more GRIDS, each on its own manifold (time x cortex, time x channels, time x cortex x eigenvalue band): the Studio's D100
 *  `productSelection`. Its members are every combination of the factors' members, coded mixed radix with the FIRST factor fastest. A factor that is itself a
 *  product is flattened into this one. std::invalid_argument for fewer than two factors, a factor that is not a grid (an irregular set, a listed member, a single
 *  span: `grid_refusal` says why and what to use), a grid with levels (name one: level(set, k), or the level a band sets: grid.h), or a code space too large. */
Selection product(std::vector<Selection> factors);
/** the JOINT partition (the Studio's D129): product(factors) with the FIRST factor (time: the frames) given a radix of 2^bits, bits = ceil(log2(its count)), so a
 *  code is (rest << bits) | frame and the array layout is [.. x tile x frame]. The codes from count(frames) to 2^bits - 1 are empty members. */
Selection joint(std::vector<Selection> factors);

// ── reading ────────────────────────────────────────────────────────────────────────────
/** a set: how many members, the one count (listed members, a tiling's count, an irregular set per element its members, a dyadic grid slots << depth, an elements
 *  grid its size, a product the product of its radices; a ladder of tilings: its levels; a dyadic grid with levels: its tiles at its depth); 1 for a selection
 *  that is not a set. */
std::int64_t count(const Selection& set);
/** is it a GRID (a partition of the manifold by a rule: a disjoint tiling, an elements grid, a dyadic grid or one of its levels, a product of grids)? */
bool is_grid(const Selection& s);
/** why `s` cannot be a factor of a product (a sentence naming what to use instead), or nothing when it can */
std::optional<std::string> grid_refusal(const Selection& s);
/** does the set have LEVELS (members that are levels: a constant-Q ladder's tilings, a dyadic grid's levels)? */
bool has_levels(const Selection& s);
/** a grid with levels: the level labelled `k` (its power of two) as a grid of its own, the grammar's `level(k)`: a ladder of tilings gives that level's tiling; a
 *  dyadic grid gives the member numbers at that level by the closed form (the code >> (depth - k)) and the level's band. std::invalid_argument for a set
 *  without levels or a level it lacks. */
Selection level(const Selection& set, int k);
/** a set with levels: the ORDINAL (identity) of the level labelled `k`; std::invalid_argument when it has none */
int level_ordinal(const Selection& set, int k);

/** the Studio's productCode: the code of one member per factor, FIRST factor fastest: c0 + n0 (c1 + n1 (..)); -1 when any factor's member is -1 (none).
 *  std::invalid_argument for a member outside its radix or a count of codes that is not the radices'. */
std::int64_t product_code(const std::vector<std::int64_t>& radices, const std::vector<std::int64_t>& codes);
/** the Studio's productCodes: the member of each factor a product code names (first factor fastest); std::invalid_argument for a code outside the product */
std::vector<std::int64_t> product_codes(const std::vector<std::int64_t>& radices, std::int64_t code);
/** a product: member k's LABEL, the factors' labels (one per factor, each read with its factor's lookup): none for a grid's member (its number names it) */
std::vector<std::optional<std::int64_t>> factor_labels(const Selection& product, std::int64_t k);
/** a set given by members or a rule: member k (a tiling's as a span; an elements grid's as a point; a dyadic grid's as its indices); a product: the product of its
 *  factors' members, one member each */
Selection member(const Selection& set, std::int64_t k);
/** a set of spans on a line (listed or tiled): the members holding `value` (none, one, several where they overlap; an instant holds its own value) */
std::vector<std::int64_t> members_at(const Selection& set, double value);
/** a set per element (or an elements grid): the MEMBER NUMBER of `element`; a dyadic grid at level ORDINAL `level` (-1: its own depth) by the closed form;
 *  -1 for none */
std::int64_t member_at(const Selection& set, std::int64_t element, int level = -1);
/** an irregular set per element: the LABEL of the member holding `element`, or nothing when the element is in none (or the set is a grid: no labels) */
std::optional<std::int64_t> label_at(const Selection& set, std::int64_t element);
/** a set: member `member`'s LABEL: a listed member's code, an irregular set per element's `member_codes`, none for a grid's member; std::invalid_argument for a
 *  member outside the set */
std::optional<std::int64_t> member_label(const Selection& set, std::int64_t member);
/** an irregular set per element: the member numbers whose label is `label`: none, one, or several (two members may share a label) */
std::vector<std::int64_t> members_with_label(const Selection& set, std::int64_t label);
/** a set per element (irregular, or a dyadic grid at its own depth): member number `member` as an indices selection carrying its label (an atlas region, a
 *  tile); empty when it holds no element; std::invalid_argument for a member outside the set */
Selection member_indices(const Selection& set, std::int64_t member);
/** a set: are its members disjoint (no element in two)? A grid: always (a tiling: hop >= width). Per element: always. Spans: no two overlap. Indices: no shared
 *  element. Other members: false (not decided here). Roll-ups and `within` need this; a measurement over members does not. */
bool disjoint(const Selection& set);
/** distance: the elements it holds, by heat-method geodesic distance (geometry-central, through the chart kit); a radius 0 holds the centre */
Selection resolve_distance(const Selection& distance, chart::ChartKit& kit);

}  // namespace nxr::selection
