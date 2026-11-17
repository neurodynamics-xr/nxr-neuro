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
// How a description is STORED (selection_store.h maps it onto the schema's rows and arrays; docs/specs/2026-11-17-selection-module.md, section 9, and model.sql):
//   point, span, distance, indices        the rows of that type (a span's corners are `selection_extent` rows; indices are the array at `array_path`)
//   set by a rule (a tiling)              a `spans` row with `params_json.tiling` {origin, width, hop, count} and no member rows: NOTHING is stored per member
//   set of listed span members            a `spans` row with `selection_span` rows (the members' [start, stop) and codes): events, bursts, bad segments, named bands
//   set per element                       a `set` row, FILE-BACKED (a time partition may have millions of members; no row per member):
//                                           P = `array_path`   the member NUMBER of each element (dense 0 .. n - 1, -1: in none)
//                                           P.labels            each member's label, one dictionary code per member (int64); ABSENT when member i is labelled i
//                                         and the dictionary (`dictionary_id`), ONE ROW PER LABEL (the lookup's entries: code = the label, name, colour); no row per member,
//                                         and none at all when the lookup is empty (the labels are their own names). The sibling arrays are a convention of the store, named
//                                         by suffix from P (member_labels_path)
//   grid (a partition: chunk grid, tiling, dyadic tiles, constant-Q ladder)   its hierarchy is CLOSED FORM, never stored as rows or parent arrays: a tiling is a rule
//                                         (above); a dyadic grid is the finest member array above (member number = tile code h 2^D + t, so a member's number at level l is that
//                                         number >> (D - l)) with its rule in `params_json.grid` {kind, depth, pieces} (count at level l = pieces << l); a constant-Q ladder
//                                         is the `ladder` row below. IRREGULAR selections (bursts, events, scouts, atlases: DK, DKT, Destrieux) have NO levels
//   set of other members                  a `set` row with member rows (`member_of_id`, `member_ordinal`)
//   set with levels (tilings)             a `ladder` row, its members the levels (each a stored selection: member_ordinal = the level's identity, code = its power of
//                                         two, its label), the level dictionary (code = the power, ordinal = the level's identity)
//   product (`factors`)                   a `set` row, nothing per member: `selection_factor` rows name its factors (stored selections) in order, `n_members` is the
//                                         product of theirs; its measurements are keyed by the product's member ordinal, row-major, the last factor fastest
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
  std::vector<std::int32_t> parent;         // not stored (a grid's hierarchy is a rule); kept so compute's field copies across
  std::optional<Tiling> tiling;             // as spans by a rule
  std::vector<Description> factors;         // as a PRODUCT of these (each a stored selection of its own: the factors are given to save by id, Placement::factor_ids; a load fills them)
  int level = -1;                           // a level: its label, a power of two (tiles of 2^level samples; 2^level tiles a piece)
  std::optional<std::pair<double, double>> band;   // a level: the band of the spectrum it resolves, in `band_unit`
  std::string band_unit;
  Lookup lookup;                            // a set: what its members' labels mean
};

}  // namespace nxr::io::selection
