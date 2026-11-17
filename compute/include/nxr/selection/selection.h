// nxr::selection (compute) — A SELECTION IN MEMORY: the Cortical Flow Studio's model (the schema's `selection` and its detail rows), as compute creates it
// before io stores it (spec docs/specs/2026-11-17-selection-module.md, section 0). Compute takes arrays; a selection here holds what its type is described
// by, never a stored id or path (the pipeline and io add those).
//
//   point      one element                              center
//   span       bounds per axis, [start, stop)           extents (start == stop on a line: an instant)
//   distance   a centre and a radius under the metric   center, radius, unit
//   indices    an explicit list of elements             indices
//   set        members, given one of four ways:         members (selections: spans, regions, …) · codes (a code per element) · tiling (spans by a rule) ·
//                                                       factors (a PRODUCT: its members are the tuples of the factors' members, on the product manifold)
//              and possibly LEVELS (a grid hierarchy: nested partitions, each refining the one above)
//
// A selection's `code` is its LABEL among its siblings, read with its set's lookup; a member's IDENTITY is its position (ordinal): five bursts are five span
// members with one label. PARTITION is a property (disjoint members covering the manifold), not a type. A kernel is not a selection: it is the weights a
// measurement uses.
//
// MEMBER ORDINALS. Listed members: their position. A tiling: k. A set by codes: the code itself (the codes ARE the ordinals, 0 .. ordinal_count - 1; a code
// that holds no element is an empty member). A selection that is not a set: one member, ordinal 0 (itself). A PRODUCT of factors F_0 .. F_{n-1}: the tuple
// (i_0, .., i_{n-1}) of the factors' ordinals, numbered ROW-MAJOR (the last factor fastest): k = ((i_0 n_1 + i_1) n_2 + i_2) .. ; its label is the tuple of
// the factors' labels, each read with its own factor's lookup (a product carries no lookup of its own). The resolved forms are in resolved.h.
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
  std::string unit;                         // a span's or a tiling's coordinates · a distance's radius · a set by codes on a surface: the positions'

  std::optional<std::int64_t> center;      // point, distance: the centre element
  double radius = 0;                        // distance

  std::vector<Extent> extents;              // span

  std::vector<std::int32_t> indices;        // indices (sorted, each once)

  // set: its members, one of three ways
  std::vector<Selection> members;           // as selections (a member's ordinal is its position); with levels, the levels by level, ascending (a level of a set
                                            // by codes holds no codes of its own: code_at(the set, element, level))
  std::vector<std::int32_t> codes;          // as a code per element (-1: none); with levels, the finest level's
  std::optional<Tiling> tiling;             // as spans by a rule
  std::vector<Selection> factors;           // as a PRODUCT of these selections (each on its own manifold; a factor that is not a set is one member)
  int depth = 0;                            // a set with levels by codes: the finest level; level l's code is codes >> (depth - l)
  int level = -1;                           // a level: its power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit` (Hz; 1/unit^2)
  std::string band_unit;
  Lookup lookup;                            // a set: what its members' (or codes') labels mean
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
/** a set by codes: a code per element (-1: none) and the lookup naming them (a parcellation, flags) */
Selection labels(std::vector<std::int32_t> codes, Lookup lookup = {});
/** the PRODUCT of two or more selections, each on its own manifold (time x frequency, time x cortex, time x channels x frequency): a set whose members are the
 *  tuples of the factors' members, numbered row-major (the last factor fastest). A factor that is itself a product is flattened into this one; a factor that is
 *  not a set is one member (a burst = product(a span on time, the indices of its channels)). std::invalid_argument for fewer than two factors or a factor with
 *  levels (name one: level(set, k)). */
Selection product(std::vector<Selection> factors);

// ── reading ────────────────────────────────────────────────────────────────────────────
/** a set: how many members (listed, or by its tiling); a set by codes: how many distinct codes; a product: ordinal_count */
std::int64_t count(const Selection& set);
/** how many member ORDINALS a selection has (the stride a product and the resolved forms use): listed members, a tiling's count, a set by codes the largest
 *  code (in its codes or its lookup) + 1, a product the product of its factors', 1 for a selection that is not a set. std::invalid_argument for a set with levels. */
std::int64_t ordinal_count(const Selection& s);
/** does the set have LEVELS (members that are levels: a constant-Q ladder's tilings, a dyadic partition's levels)? */
bool has_levels(const Selection& s);
/** a set with levels: level `k` (its power of two) as a selection of its own, the grammar's `level(k)`: a ladder of tilings gives that level's tiling; a set
 *  by codes gives the codes at that level (codes >> (depth - k)), with the level's band. std::invalid_argument for a set without levels or a level it lacks. */
Selection level(const Selection& set, int k);
/** a product: member k's factor ordinals (row-major); std::invalid_argument for k outside the product */
std::vector<std::int64_t> factor_ordinals(const Selection& product, std::int64_t k);
/** a product: the member ordinal of the factors' ordinals (one per factor, each inside its factor) */
std::int64_t product_ordinal(const Selection& product, const std::vector<std::int64_t>& ordinals);
/** a product: member k's LABEL, the factors' labels (one per factor, each read with its factor's lookup): a listed member's code, a set by codes' code, none for a
 *  tiling's member, a selection that is not a set its own code */
std::vector<std::optional<std::int64_t>> factor_labels(const Selection& product, std::int64_t k);
/** a set given by members or a tiling: member k (a tiling's as a span); a product: the product of its factors' members (one member each) */
Selection member(const Selection& set, std::int64_t k);
/** a set of spans on a line (listed or tiled): the members holding `value` (none, one, several where they overlap; an instant holds its own value) */
std::vector<std::int64_t> members_at(const Selection& set, double value);
/** a set by codes: the code of `element` at `level` (a set with levels; -1: the set's own codes); -1 for none */
std::int64_t code_at(const Selection& set, std::int64_t element, int level = -1);
/** a set by codes: code `code` (at `level`) as an indices selection carrying that code (an atlas region, a tile) */
Selection member_indices(const Selection& set, std::int64_t code, int level = -1);
/** a set: are its members disjoint (no element in two)? By codes: always. A tiling: hop >= width. Spans: no two overlap. Indices: no shared element.
 *  A product: every factor's are (a factor that is not a set is one member). Other members: false (not decided here). */
bool disjoint(const Selection& set);
/** distance: the elements it holds, by heat-method geodesic distance (geometry-central, through the chart kit) */
Selection resolve_distance(const Selection& distance, chart::ChartKit& kit);

}  // namespace nxr::selection
