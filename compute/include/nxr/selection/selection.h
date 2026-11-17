// nxr::selection (compute) — A SELECTION IN MEMORY: the Cortical Flow Studio's model (the schema's `selection` and its detail rows), as compute creates it
// before io stores it (spec docs/specs/2026-11-17-selection-module.md, section 0). Compute takes arrays; a selection here holds what its type is described
// by, never a stored id or path (the pipeline and io add those).
//
//   point      one element                              center
//   span       bounds per axis, [start, stop)           extents
//   distance   a centre and a radius under the metric   center, radius, unit
//   spans      spans on a line: rows, or a closed form  rows | tiling, unit
//   indices    an explicit list of elements             indices
//   set        members that are selections, or a code per element (a partition), or LEVELS (nested partitions; the Studio's ladder)
//
// A selection's `code` is its LABEL among its siblings, read with its set's lookup; a member's IDENTITY is its position (ordinal). A spans row has a code of
// its own (five bursts: five rows, one label). A kernel is not a selection: it is the weights a measurement uses.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/selection/lookup.h"

namespace nxr::chart { class ChartKit; }

namespace nxr::selection {

enum class Type { point, span, distance, spans, indices, set };
const char* type_name(Type t);

/** a span's bounds on one axis, in that axis's coordinates */
struct Extent { int axis = 0; double start = 0, stop = 0; };
/** a row of a spans selection: [start, stop) (start == stop: an instant) and its label */
struct SpanRow { double start = 0, stop = 0; std::optional<std::int64_t> code; };
/** spans in closed form: `count` spans of `width`, a new one every `hop`, from `origin` (row k: [origin + k hop, origin + k hop + width)) */
struct Tiling { double origin = 0, width = 0, hop = 0; std::int64_t count = 0; };

struct Selection {
  Type type = Type::set;
  std::optional<std::int64_t> code;        // its label among its siblings (read with its set's lookup)
  std::string unit;                         // spans: the coordinates' unit · distance: the radius' · a set by codes on a surface: the positions'

  std::optional<std::int64_t> center;      // point, distance: the centre element
  double radius = 0;                        // distance

  std::vector<Extent> extents;              // span

  std::vector<SpanRow> rows;                // spans, by rows (sorted by start)
  std::optional<Tiling> tiling;             // spans, in closed form

  std::vector<std::int32_t> indices;        // indices (sorted, each once)

  std::int64_t elements = 0;                // set by codes: the manifold's element count
  std::vector<std::int32_t> codes;          // set by codes: a code per element (-1: none); with levels, the finest level's
  int depth = 0;                            // set with levels by codes: the finest level; level l's code is codes >> (depth - l)
  std::vector<Selection> members;           // set: its members (a member's ordinal is its position); with levels, the levels by level, ascending (a level of a set
                                            // by codes holds no codes of its own: code_at(the set, element, level))
  int level = -1;                           // a level of a set with levels: its power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit` (Hz; 1/unit^2)
  std::string band_unit;
  Lookup lookup;                            // set, spans: what its members' (or rows', or codes') labels mean
};

// ── making the simple types ────────────────────────────────────────────────────────────
Selection point(std::int64_t center, std::optional<std::int64_t> code = std::nullopt);
Selection span(std::vector<Extent> extents, std::optional<std::int64_t> code = std::nullopt);
Selection distance(std::int64_t center, double radius, std::string unit, std::optional<std::int64_t> code = std::nullopt);
/** spans by rows: sorted; std::invalid_argument for a row with stop < start or not finite, or a label the lookup lacks */
Selection spans(std::vector<SpanRow> rows, std::string unit, Lookup lookup = {});
/** spans in closed form */
Selection spans(Tiling tiling, std::string unit);
Selection indices(std::vector<std::int32_t> elements, std::optional<std::int64_t> code = std::nullopt);
/** a set whose members are the given selections, labelled through `lookup` */
Selection set(std::vector<Selection> members, Lookup lookup = {});
/** a set by codes: a code per element (-1: none) and the lookup naming them (a parcellation, flags) */
Selection labels(std::vector<std::int32_t> codes, Lookup lookup = {});

// ── reading ────────────────────────────────────────────────────────────────────────────
/** spans: how many rows (by rows or closed form) */
std::int64_t count(const Selection& spans);
/** spans: row k */
SpanRow row(const Selection& spans, std::int64_t k);
/** spans: the rows holding `value` (none, one, several where they overlap) */
std::vector<std::int64_t> rows_at(const Selection& spans, double value);
/** set by codes: the code of `element` at `level` (a set with levels; -1: the set's own codes); -1 for none */
std::int64_t code_at(const Selection& set, std::int64_t element, int level = -1);
/** set by codes: member `code` (at `level`) as an indices selection carrying that code (an atlas region, a tile) */
Selection member_indices(const Selection& set, std::int64_t code, int level = -1);
/** spans: row k as a span selection on axis 0, carrying the row's code */
Selection member_span(const Selection& spans, std::int64_t k);
/** are its members disjoint (no element in two)? A set by codes always is; spans by rows when no two rows overlap; a closed form when hop >= width; a
 *  set of member selections when no two of its indices members share an element (other members: false) */
bool disjoint(const Selection& s);
/** distance: the elements it holds, by heat-method geodesic distance (geometry-central, through the chart kit) */
Selection resolve_distance(const Selection& distance, chart::ChartKit& kit);

}  // namespace nxr::selection
