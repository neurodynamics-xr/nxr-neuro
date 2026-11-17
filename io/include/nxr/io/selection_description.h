// nxr-io — A SELECTION'S DESCRIPTION: what a selection is, as the toolbox holds it in memory. io's own struct (io never includes compute), a field-for-field MIRROR of
// compute's `nxr::selection::Selection` (compute/include/nxr/selection/selection.h; its Type, Extent, Tiling, Entry and Lookup). A selection compute makes and one io
// loads from file are the same description and follow the same rules; the pipelines (the only place compute and io meet) convert between the two structs field by field
// (pipelines/include/nxr/pipelines/selection_bridge.h). Spec: docs/specs/2026-11-17-selection-module.md, section 0.
//
//   point      one element                                 center
//   span       bounds per axis, [start, stop)              extents
//   distance   a centre and a radius under the metric      center, radius, unit
//   indices    an explicit list of elements                indices
//   set        members, given as selections, as a code per element, or by a rule (a tiling); possibly with LEVELS
//
// How a description is STORED (selection_store.h maps it onto the schema's rows; the schema is unchanged):
//   point, span, distance, indices        the rows of that type (a span's corners are `selection_extent` rows; indices are the array at `array_path`)
//   set by a tiling                       a `spans` row with `params_json.tiling` {origin, width, hop, count} and no member rows
//   set of span members                   a `spans` row with `selection_span` rows (the members' [start, stop) and codes)
//   set by codes                          a `set` row with its array at `array_path` and its lookup as the dictionary
//   set of other members                  a `set` row with member rows (`member_of_id`, `member_ordinal`)
//   set with levels                       a `ladder` row, its members the levels (each a stored selection), the level lookup its dictionary
#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nxr::io::selection {

enum class Type { point, span, distance, indices, set };
const char* type_name(Type t);

/** a span's bounds on one axis, in that axis's coordinates */
struct Extent { int axis = 0; double start = 0, stop = 0; };
/** a set's members by a rule: `count` spans on axis 0 of `width`, a new one every `hop`, from `origin` (member k: [origin + k hop, origin + k hop + width)); closed form: no rows */
struct Tiling { double origin = 0, width = 0, hop = 0; std::int64_t count = 0; };
/** one code and what it means (a dictionary entry) */
struct Entry {
  std::int64_t code = 0;
  std::string name;
  std::optional<std::array<int, 3>> color;   // 0..255 each
};
/** the entries of a lookup, one per code; empty: the codes are their own names (stored as the selection's dictionary) */
using Lookup = std::vector<Entry>;

struct Description {
  Type type = Type::set;
  std::optional<std::int64_t> code;        // its label among its siblings (read with its set's lookup)
  std::string unit;                         // a span's or a tiling's coordinates · a distance's radius

  std::optional<std::int64_t> center;      // point, distance: the centre element
  double radius = 0;                        // distance

  std::vector<Extent> extents;              // span
  std::vector<std::int32_t> indices;        // indices (in the order given: picked channels keep their order)

  // set: its members, one of three ways
  std::vector<Description> members;         // as selections (a member's ordinal is its position); with levels, the levels, ascending
  std::vector<std::int32_t> codes;          // as a code per element (-1: none); with levels, the finest level's
  std::optional<Tiling> tiling;             // as spans by a rule
  int depth = 0;                            // a set with levels by codes: the finest level; level l's code is codes >> (depth - l)
  int level = -1;                           // a level: its power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit`
  std::string band_unit;
  Lookup lookup;                            // a set: what its members' (or codes') labels mean
};

}  // namespace nxr::io::selection
