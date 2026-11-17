// nxr::selection (compute) — A SELECTION IN MEMORY: the Cortical Flow Studio's model (the schema's `selection` and its detail rows), as compute creates it
// before io stores it (spec docs/specs/2026-11-17-selection-module.md, section 0). Compute takes arrays; a selection here holds what its type is described
// by, never a stored id or path (the pipeline and io add those).
//
//   point      one element                              center
//   span       bounds per axis, [start, stop)           extents (start == stop on a line: an instant)
//   distance   a centre and a radius under the metric   center, radius, unit
//   indices    an explicit list of elements             indices
//   set        members, given one of three ways:        members (selections: spans, regions, …) · codes (a code per element) · tiling (spans by a rule)
//              and possibly LEVELS (a grid hierarchy: nested partitions, each refining the one above)
//
// A selection's `code` is its LABEL among its siblings, read with its set's lookup; a member's IDENTITY is its position (ordinal): five bursts are five span
// members with one label. PARTITION is a property (disjoint members covering the manifold), not a type. A kernel is not a selection: it is the weights a
// measurement uses.
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

// ── reading ────────────────────────────────────────────────────────────────────────────
/** a set: how many members (listed, or by its tiling); a set by codes: how many distinct codes */
std::int64_t count(const Selection& set);
/** a set given by members or a tiling: member k (a tiling's as a span) */
Selection member(const Selection& set, std::int64_t k);
/** a set of spans on a line (listed or tiled): the members holding `value` (none, one, several where they overlap; an instant holds its own value) */
std::vector<std::int64_t> members_at(const Selection& set, double value);
/** a set by codes: the code of `element` at `level` (a set with levels; -1: the set's own codes); -1 for none */
std::int64_t code_at(const Selection& set, std::int64_t element, int level = -1);
/** a set by codes: code `code` (at `level`) as an indices selection carrying that code (an atlas region, a tile) */
Selection member_indices(const Selection& set, std::int64_t code, int level = -1);
/** a set: are its members disjoint (no element in two)? By codes: always. A tiling: hop >= width. Spans: no two overlap. Indices: no shared element.
 *  Other members: false (not decided here). */
bool disjoint(const Selection& set);
/** distance: the elements it holds, by heat-method geodesic distance (geometry-central, through the chart kit) */
Selection resolve_distance(const Selection& distance, chart::ChartKit& kit);

}  // namespace nxr::selection
