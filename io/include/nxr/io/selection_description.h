// nxr-io — A SELECTION'S DESCRIPTION: what a selection is, as the toolbox holds it in memory. io's own struct (io never includes compute), a field-for-field MIRROR of
// compute's `nxr::selection::Selection` (compute/include/nxr/selection/selection.h; its Type, Extent, Tiling, Entry and Lookup). A selection compute makes and one io
// loads from file are the same description and follow the same rules; the pipelines (the only place compute and io meet) convert between the two structs field by field
// (pipelines/include/nxr/pipelines/selection_bridge.h). Spec: docs/specs/2026-11-17-selection-module.md, section 0.
//
//   point      one element                                 center
//   span       bounds per axis, [start, stop)              extents
//   distance   a centre and a radius under the metric      center, radius, unit
//   indices    an explicit list of elements                indices
//   set        members, given as selections, as a member NUMBER per element, or by a rule (a tiling); possibly with LEVELS
//
// A member's identity is its number (dense, 0 .. n - 1); its label is its code (`member_codes`, read with the lookup). The two are never the same field.
//
// How a description is STORED (selection_store.h maps it onto the schema's rows; the schema is unchanged):
//   point, span, distance, indices        the rows of that type (a span's corners are `selection_extent` rows; indices are the array at `array_path`)
//   set by a tiling                       a `spans` row with `params_json.tiling` {origin, width, hop, count} and no member rows
//   set of span members                   a `spans` row with `selection_span` rows (the members' [start, stop) and codes)
//   set per element                       a `set` row with its array of MEMBER NUMBERS at `array_path` (params_json `member_numbered`: 1) and a dictionary with ONE ENTRY
//                                         PER MEMBER (entry code = the member number, ordinal = the number, name and colour = the label's lookup entry, attributes_json
//                                         {"label": the member's label}); an array stored without the marker holds labels per element (an imported atlas) and is
//                                         mapped to member numbers on load (`from_labels`'s rule: ascending labels)
//   set of other members                  a `set` row with member rows (`member_of_id`, `member_ordinal`)
//   set with levels (tilings)             a `ladder` row, its members the levels (each a stored selection: member_ordinal = the level's identity, code = its power of
//                                         two, its label), the level dictionary (code = the power, ordinal = the level's identity)
//   set per element with levels           the `set` row above, its levels in params_json `levels`: [{level (the label), band, band_unit, member_codes, parent}], in
//                                         level order; the array and dictionary are the finest level's
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
  std::vector<Description> members;         // as selections (a member's number is its position); with levels, the levels in order
  std::vector<std::int32_t> member_of;      // as a member NUMBER per element (0 .. n - 1, -1: none); with levels, the finest level's
  std::vector<std::int64_t> member_codes;   // a set per element, or a level of one: each member's LABEL (one per member), read with the lookup
  std::vector<std::int32_t> parent;         // a level of a set per element: member j -> its member in the level above (empty for the first level)
  std::optional<Tiling> tiling;             // as spans by a rule
  int level = -1;                           // a level: its label, a power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit`
  std::string band_unit;
  Lookup lookup;                            // a set: what its members' labels mean
};

}  // namespace nxr::io::selection
