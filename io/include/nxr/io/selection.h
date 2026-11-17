// nxr-io — THE SELECTION (a port of the Cortical Flow Studio's backend/src/entities/selection.ts; the model is the schema's, register D30–D36/D46/D48/D60/D87/D94/D95):
// a named part of a manifold (or a position on a space), with a code, its detail rows (extents: a span's corners; spans: a spans set's members; elements: a partition's codes
// or a subset's picks, as its array) and what was measured on it. What the app adds is RESOLUTION: which index ranges of its manifold the selection is, so a field can be read
// exactly there; and a spans set's members by ordinal, in closed form or by rows.
//
//   const auto s = nxr::io::selection::Selection::read(db, store, id);
//   s.count(); s.at(k);                 // a spans set's member k: [start, stop) and its code (a constant-Q level in closed form: an O(1) multiply)
//   s.index_at(t);                      // the member holding t
//   s.resolve();                        // crop (index ranges per axis) · indices (a gather) · weights (a kernel's rule)
//   s.elements(); s.partition();        // the per-element array; a partition's code -> element count
//
// THIS FILE is the entity (resolution of a stored selection). The selection CRUD is gathered in io's selection module, in four files:
//   selection_description.h   the in-memory description (a mirror of compute's `nxr::selection::Selection`)
//   selection_store.h         save / load / update / remove a selection (all five types), its lookup (dictionary), and the rows beneath it
//   selection_measurement.h   what is measured on a selection: the measurement rows and their arrays
//   selection.h               this: the stored selection, resolved (which part of its manifold)
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nxr/io/records/rows.h"
#include "nxr/io/selection_description.h"

namespace nxr::io {
class Database;
class Store;
}  // namespace nxr::io

namespace nxr::io::selection {

/** One member of a spans set: its ordinal, [start, stop) in the line's coordinate system, its code (an event type; a closed form's member: its ordinal). */
struct Span { std::int64_t ordinal = 0; double start = 0, stop = 0; std::optional<std::int64_t> code; };


/** What part of its manifold a selection is: index ranges per axis (a crop), picked elements (a gather), or a weighting rule the toolbox evaluates. */
struct Resolution {
  enum class Kind { crop, indices, weights };
  Kind kind = Kind::crop;
  std::vector<std::pair<std::int64_t, std::int64_t>> slice;   // crop: [lo, hi) per axis of the manifold's index shape
  std::vector<std::int32_t> indices;                          // indices
  std::optional<std::string> weight_fn;                       // weights
  Json params;
};

class Selection {
 public:
  /** a stored selection by id (any status); Error(no_such_row) when there is none. `store` reads its array; both must outlive it. */
  static Selection read(const Database& db, const Store& store, std::string_view id);
  const records::SelectionRow& row() const { return row_; }
  const std::string& type() const { return row_.type; }

  // ── relations
  std::vector<records::SelectionRow> members() const;                 // a set's (or ladder's) member selections, by ordinal
  std::vector<records::SelectionExtentRow> extents() const;           // a span's corners, by axis
  std::vector<records::SelectionMeasurementRow> measurements(bool own = true) const;
  /** what this selection's `code` MEANS: its SET's dictionary first (what it is among its siblings), its own only when it is not a member */
  std::optional<records::DictionaryEntryRow> entry() const;
  std::optional<double> measurement(const std::string& measure, const std::optional<std::string>& of_field_id = std::nullopt) const;   // its own value

  // ── a spans set: its members by ordinal, in closed form or by rows (one type, D87)
  std::optional<Tiling> tiling() const;
  double end() const;                                                  // where the line ends (the closed form's `end`, else the manifold's axis)
  std::int64_t count() const;                                          // a spans set's members, or a set's
  Span at(std::int64_t k) const;                                       // Error(invalid_argument) for another type or no member k
  std::optional<std::int64_t> index_at(double value) const;            // the member with start <= value < stop

  // ── resolution: which part of the manifold
  /** on its own manifold, or (for a span) the product it lies on a factor of (`on`: a manifold id). A span crops by its extents rounded onto closed-form axes (element
   *  indices on an axis with none); a point is one element; indices and a partition are a gather; a kernel is its rule; a spans set resolves member by member (at). */
  Resolution resolve(const std::optional<std::string>& on = std::nullopt) const;
  /** a POSITION on a space (D95): the vertex of `line` (a manifold id) nearest it, or nothing when the line does not reach it */
  std::optional<std::int64_t> vertex_on(const std::string& line) const;

  // ── its array: a partition's codes, a subset's picks (nothing for a closed-form selection)
  std::vector<std::int32_t> elements() const;
  std::map<std::int32_t, std::int64_t> partition() const;              // code -> element count

 private:
  Selection(const Database& db, const Store& store, records::SelectionRow row) : db_(&db), store_(&store), row_(std::move(row)) {}
  const Database* db_;
  const Store* store_;
  records::SelectionRow row_;
};

}  // namespace nxr::io::selection
