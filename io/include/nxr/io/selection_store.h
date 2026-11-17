// nxr-io — THE SELECTION CRUD: saving, loading, updating and deleting a selection (all five types: point, span, distance, indices, set), its lookup (the dictionary its
// codes are read with), and the rows beneath it. The one place io writes or reads a selection's rows; the writer (SubjectWriter::selection / set_spans / set_extents),
// the task API (Unit::selection, set_params) and the pipelines call into it and carry no SQL of their own for selections. Spec:
// docs/specs/2026-11-17-selection-module.md, section 0 and "where it lives".
//
//   Create   save(sub, description, placement)        a made selection (selection_description.h) -> its rows, array and dictionary; returns its id (and its members')
//   Read     load(db, store, id)                      a stored selection -> the same description a made one has (+ its row); the entity `Selection::read` (selection.h) resolves it
//   Update   update(sub, id, description)             replace what the description holds (a point's centre, a span's extents, a set's spans, tiling, codes, picks, lookup)
//   Delete   remove(sub, id)                          the selection, its members (they cascade), its arrays
//   Lookup   save_lookup / load_lookup / update_lookup
//   Rows     the row-level functions the writer's composition calls (insert_row, replace_spans, ...)
//   Measurements on a selection: selection_measurement.h.
//
// The mapping of a description onto the schema's rows (`spans` and `ladder` stay stored types; io maps them to and from a set) is in selection_description.h.
// io stores what it is given: it designs nothing (a ladder's levels, tilings and bands are made by compute and handed in), and does no maths beyond decoding.
#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/records/rows.h"
#include "nxr/io/selection_description.h"
#include "nxr/io/writer.h"

namespace nxr::io {
class Store;
}

namespace nxr::io::selection {

/** WHERE a description is stored: its name, path and manifold, and what the row says about it that the description does not. */
struct Placement {
  std::string name;
  std::optional<std::string> path;                // store-relative; the array of indices or codes is written here (required for those)
  std::optional<std::string> manifold_id;         // what it lies on
  std::optional<std::string> of_field_id, id, created_utc, created_by, label, description;
  std::optional<std::string> acquisition;         // an acquisition LABEL (made on first use) ...
  std::optional<std::string> acquisition_id;      // ... or its id
  std::optional<std::string> cell;                // which cells of the manifold (default by type: point, indices, codes '0'; span, spans '1')
  std::optional<std::string> dictionary_id;       // an existing dictionary the code reads with (else the description's lookup is saved under `name`)
  std::optional<std::int64_t> n_members;          // a set given by a rule with nothing stored (the joint cells of a product): how many members it has
  Json params;                                    // params_json (object; null: none); a tiling and the levels' windows are added by save
  Cols cols;                                      // any further columns of the row
};

/** what save made */
struct Saved {
  std::string id;
  std::vector<std::string> member_ids;            // a set's (a ladder's levels') member selections, by ordinal
  std::optional<std::string> dictionary_id;
};

/** a stored selection as a description, with its row (a position on a space is in the row: the description holds elements) */
struct Loaded {
  records::SelectionRow row;
  Description description;
};

// ── create ───────────────────────────────────────────────────────────────────────────────────────────────────────
/** SAVE a selection: its row, its detail rows (extents, span members, member selections), its array (indices, codes) and its dictionary. Inside the caller's composition.
 *  A set stores as: a tiling -> `spans` (closed form); members all spans -> `spans` rows (an empty set too, unless `n_members` is given: then a `set` with nothing
 *  stored); codes -> `set` with its array and lookup (an empty lookup: each code its own name); other members -> `set` with member rows; members all tilings with
 *  levels -> `ladder` (a dictionary of the levels, an empty tile dictionary per level, each level a `spans` member holding its tiling).
 *  A ladder's line (its manifold) must be a closed-form line: its sample rate and length give each level's window in samples. */
Saved save(SubjectWriter& sub, const Description& description, const Placement& at);

// ── read ─────────────────────────────────────────────────────────────────────────────────────────────────────────
/** a stored selection (any status) as its description; Error(no_such_row) when there is none; a kernel is not a selection (Error(invalid_argument)). */
Loaded load(const Database& db, const Store& store, std::string_view id);
/** the complete selection of that name in the subject (optionally of a stored type), or nothing */
std::optional<records::SelectionRow> find_by_name(const Database& db, const std::string& subject_id, const std::string& name, const std::optional<std::string>& stored_type = std::nullopt);
/** the complete selection of that name on the manifold (optionally of a stored type), or nothing */
std::optional<records::SelectionRow> find_on_manifold(const Database& db, const std::string& manifold_id, const std::string& name, const std::optional<std::string>& stored_type = std::nullopt);
/** the complete selection of stored type `stored_type` on the manifold whose params_json holds the key `param_key` (a ladder holding `voices`), with its members */
std::optional<Saved> find_with_param(const Database& db, const std::string& subject_id, const std::string& manifold_id, const std::string& stored_type, const std::string& param_key);
/** a spans selection's member rows, by ordinal */
std::vector<records::SelectionSpanRow> spans_of(const Database& db, std::string_view selection_id);

/** a closed-form time line as stored: what a tiling is read against (its sample rate is 1 / spacing) */
struct TimeLine {
  std::string name;
  std::int64_t count = 0;                         // samples
  double spacing = 0, origin = 0;                 // seconds
  std::optional<std::string> acquisition_id;
};
/** the line `manifold_id` (a closed-form `line`); Error(no_such_row | invalid_argument) otherwise */
TimeLine time_line(const Database& db, const std::string& manifold_id);

// ── update ───────────────────────────────────────────────────────────────────────────────────────────────────────
/** UPDATE a stored selection to `description` (its type must be the stored one): a point's or distance's centre and radius, a span's extents, a spans set's rows or
 *  tiling, the codes or picks of a set or indices (the array is rewritten), and its lookup when the description has one. A set with member selections (or a ladder)
 *  is updated member by member. Stamps modified_utc. Error(invalid_argument) for a description of another type. */
void update(SubjectWriter& sub, const std::string& id, const Description& description);
/** a selection's parameters (params_json), replaced */
void set_params(Database& db, const std::string& selection_id, const Json& params);

// ── delete ───────────────────────────────────────────────────────────────────────────────────────────────────────
/** DELETE a selection of this subject: its rows (its members and detail rows cascade) and its arrays. */
void remove(SubjectWriter& sub, const std::string& id);

// ── the lookup (a selection's dictionary) ────────────────────────────────────────────────────────────────────────
/** a dictionary of that name in the subject: its entries (code, name, colour), made or found; returns its id */
std::string save_lookup(SubjectWriter& sub, const std::string& name, const Lookup& lookup, const std::optional<std::string>& description = std::nullopt);
/** the entries of a dictionary, by ordinal and code */
Lookup load_lookup(const Database& db, const std::string& dictionary_id);
/** replace a dictionary's entries (in the caller's update) */
void update_lookup(SubjectWriter& sub, const std::string& dictionary_id, const Lookup& lookup);

// ── the rows (the writer's composition calls these; they take the database, never decide) ────────────────────────
/** a selection row, its counts (`n_elements`, and `n_members` when given) and its detail rows in one go; the composition (pending/complete, arrays) is the writer's */
void insert_row(Database& db, const Row& row, std::optional<std::int64_t> n_elements, std::optional<std::int64_t> n_members, const std::optional<std::vector<SpanSpec>>& spans,
                const std::vector<ExtentSpec>& extents, const std::vector<Cols>& own_measurements);
void replace_spans(Database& db, const std::string& selection_id, const std::vector<SpanSpec>& spans);       // n_members follows
void replace_extents(Database& db, const std::string& selection_id, const std::vector<ExtentSpec>& extents);

}  // namespace nxr::io::selection
