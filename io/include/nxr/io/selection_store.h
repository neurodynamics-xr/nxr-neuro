// nxr-io — THE SELECTION CRUD: saving, loading, updating and deleting a selection (all five types: point, span, distance, indices, set), its lookup (the dictionary its
// labels are read with), and the rows beneath it. The one place io writes or reads a selection's rows; the writer (SubjectWriter::selection / set_spans / set_extents),
// the task API (Unit::selection, set_params) and the pipelines call into it and carry no SQL of their own for selections. Spec:
// docs/specs/2026-11-17-selection-module.md, section 0 and "where it lives".
//
//   Create   save(sub, description, placement)        a made selection (selection_description.h) -> its rows, array and dictionary; returns its id (and its members')
//   Read     load(db, store, id)                      a stored selection -> the same description a made one has (+ its row); the entity `Selection::read` (selection.h) resolves it
//   Update   update(sub, id, description)             replace what the description holds (a point's centre, a span's extents, a set's spans, tiling, member numbers and labels, picks, lookup)
//   Delete   remove(sub, id)                          the selection, its members (they cascade), its arrays
//   Lookup   save_lookup / load_lookup / update_lookup
//   Rows     the row-level functions the writer's composition calls (insert_row, replace_spans, ...)
//   Measurements on a selection: selection_measurement.h.
//
// The mapping of a description onto the schema's rows (`spans` and `ladder` stay stored types; io maps them to and from a set) is in selection_description.h.
// io stores what it is given: it designs nothing (a ladder's levels, tilings and bands are made by compute and handed in), and does no maths beyond decoding.
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nxr/io/database.h"
#include "nxr/io/json_text.h"
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
  std::optional<std::string> path;                // store-relative; the array of indices or member numbers is written here (required for those)
  std::optional<std::string> manifold_id;         // what it lies on
  std::optional<std::string> of_field_id, id, created_utc, created_by, label, description;
  std::optional<std::string> acquisition;         // an acquisition LABEL (made on first use) ...
  std::optional<std::string> acquisition_id;      // ... or its id
  std::optional<std::string> cell;                // which cells of the manifold (default by type: point, indices, a set per element '0'; span, spans '1')
  std::optional<std::string> dictionary_id;       // an existing dictionary the code reads with (else the description's lookup is saved under `name`)
  std::optional<std::int64_t> n_members;          // a set given by a rule with nothing stored: how many members it has (a product's is the product of its factors', read from their rows)
  std::vector<std::string> factor_ids;            // a product: its factors' stored selections, in order (the description's `factors` are what a load fills; a save reads these)
  Json params;                                    // params_json (object; null: none); a tiling and the levels' windows are added by save
  std::vector<Json> level_params;                 // a ladder: what the design settled about each level, by position (stride, rate_hz, ...): merged into the level's record
  std::vector<Cols> measurements;                 // its OWN measurement rows (array-backed or valued), written with the row (a joint set's per-field arrays)
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
/** SAVE a selection: its row, its detail rows (extents, span members, member selections), its array (indices, member numbers) and its dictionary. Inside the caller's composition.
 *  A set stores as (selection_description.h): a tiling -> `spans` (closed form, nothing per member); members all spans -> `spans` rows (an empty set too, unless `n_members`
 *  is given: then a `set` with nothing stored); a set per element -> `set` with its array of member numbers, the sibling array of the members' labels (when they have any)
 *  and a dictionary of one row per LABEL; a grid's levels are a rule in `params` ("grid"), never rows or arrays (a dyadic grid at one level: its tile codes and
 *  `params.grid`; with all its levels: save_grid); an elements grid -> `set` with nothing stored and `params.grid`; a product of GRIDS (`factors`, `radices`,
 *  Placement::factor_ids; an irregular factor is refused) -> `set` with `selection_factor` rows and `params.product` [{selection, members}], codes first factor fastest; other members -> `set` with member rows; members all tilings with levels -> `ladder` (a
 *  dictionary of the levels, an empty tile dictionary per level, each level a `spans` member holding its tiling: its member_ordinal the level's identity, its code its
 *  power of two). The arrays are written before the row, in the caller's composition.
 *  A ladder's line (its manifold) must be a closed-form line: its sample rate and length give each level's window in samples. */
Saved save(SubjectWriter& sub, const Description& description, const Placement& at);

// ── the arrays beside a set per element's element array (a convention of the store: siblings of P = `array_path`, named by suffix) ───────────────────────
std::string member_labels_path(const std::string& array_path);                          // P.labels: each member's label (int64), absent when member i is labelled i
/** a set per element from LABELS per element (an importer's atlas, flags; -1: none): one member per distinct label, numbered in ascending label order, the label kept as the
 *  member's. Decoding an array through its dictionary, the one place io turns labels into numbers (compute's `from_labels` is the same rule, on the compute side). */
struct MemberNumbering { std::vector<std::int32_t> member_of; std::vector<std::int64_t> member_codes; };
MemberNumbering number_labels(const std::vector<std::int32_t>& element_labels);
/** a description of a set per element from labels per element and the lookup they are read with (number_labels, then the description) */
Description set_from_labels(const std::vector<std::int32_t>& element_labels, Lookup lookup = {});

// ── a dyadic GRID (spatial tiles) ────────────────────────────────────────────────────────────────────────────────
/** one level of a dyadic grid: its row's placement (`name`, `manifold_id`, `dictionary_id` of its tiles' names, `params`, `cols`), the node's own path for its array (`array_path`) and the
 *  tile of every element, as the tile CODE h 2^l + t, which IS the member number (-1: none), `tiles` the level's member count (empty tiles included) */
struct GridLevel {
  Placement at;
  std::string array_path;
  std::vector<std::int32_t> tile_codes;
  std::int64_t tiles = 0;
};
/** SAVE a dyadic grid: the `ladder` row (`ladder`: name, manifold, the levels' dictionary in `dictionary_id`, params including the rule `grid` {kind, depth, pieces}) and one `set` row per level
 *  (member_ordinal = l, code = l) with its array of tile codes. The hierarchy is the closed form (a member's number at level l is its finest number >> (depth - l)): no parent arrays. */
Saved save_grid(SubjectWriter& sub, const Placement& ladder, const std::vector<GridLevel>& levels);

/** a set per element's members' labels, one per member (the sibling array, or 0 .. n_members - 1 when it has none): what turns a label an importer knows into the member number */
std::vector<std::int64_t> member_labels(const Database& db, const Store& store, const std::string& selection_id);

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
 *  tiling, the member numbers (and labels) or picks of a set or indices (the array is rewritten), and its lookup when the description has one. A set with member selections (or a ladder)
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


// ── the app-wide vocabularies (the band names, the measure registry): seeded through the store ────────────────────────────────────────
/** one entry of an app lookup, with every column a vocabulary uses */
struct AppEntry {
  std::int64_t code = 0;
  std::string name;
  std::optional<std::array<int, 3>> color;
  std::optional<double> measurement;
  std::string measurement_unit, function;          // empty: none
  Json params, attributes;                         // null: none
  std::optional<std::int64_t> ordinal;
};
struct AppLookup {
  std::string name, description;
  std::string scope = "app";
  std::optional<std::string> by;                  // created_by / modified_by on the dictionary row
  std::vector<AppEntry> entries;
  bool refresh_when_incomplete = false;           // a dictionary that lacks one of these names is rewritten, entries and all (the measure registry); else it is kept as found
};
/** the dictionary of that scope and name, made with its entries when the database lacks it (and refreshed when asked and incomplete); returns its id */
std::string ensure_app_lookup(Database& db, const AppLookup& spec);

// ── lookups of other layers (a member by name, a lookup's size, the names a channel set carries) ──────────────────
/** how many entries the dictionary has */
std::int64_t lookup_size(const Database& db, const std::string& dictionary_id);
/** the id of the dictionary of that scope (app, dataset, user) and name (the lowest id when several), or nothing */
std::optional<std::string> find_lookup(const Database& db, const std::string& scope, const std::string& name);
/** the entries of a dictionary as stored rows (all columns: measurement, attributes, ...), by ordinal then code */
std::vector<records::DictionaryEntryRow> lookup_entries(const Database& db, const std::string& dictionary_id);
/** is any selection reading with this dictionary? (a dictionary in use is kept) */
bool lookup_in_use(const Database& db, const std::string& dictionary_id);
/** the member selection of the set `parent_id` of that name, or nothing */
std::optional<records::SelectionRow> find_member(const Database& db, const std::string& parent_id, const std::string& name);
/** the top-level selection on the manifold whose name ends with `suffix` (a channel set's `_names`), or nothing */
std::optional<records::SelectionRow> find_on_manifold_ending(const Database& db, const std::string& manifold_id, const std::string& suffix);
/** the complete selection of that name over that field in the subject, or nothing */
std::optional<records::SelectionRow> find_of_field(const Database& db, const std::string& subject_id, const std::string& of_field_id, const std::string& name);
/** the ids of the subject's complete selections */
std::vector<std::string> complete_ids(const Database& db, const std::string& subject_id);
/** the member rows (the levels) of a set per element, and the factors of a product, as stored rows by ordinal */
std::vector<Row> subject_factor_rows(const Database& db, const std::string& subject_id);
/** DELETE every selection of a subject (a subject's delete does this first: a dictionary in use is restricted) */
void remove_all(Database& db, const std::string& subject_id);

// ── the arrays and paths a selection names (stored-array lists, named paths, orphan checks) ────────────────────────
/** a stored array of some subject of a dataset (a selection's, or a measurement's) */
struct SubjectArray { std::string subject_path, subject_name, path; };
/** how a dataset is asked a question when it may be open read-only (no Database): the rows an SQL text answers */
using Query = std::function<std::vector<Row>(const std::string& sql)>;
/** every array the selections of a dataset name (`array_path`) */
std::vector<SubjectArray> array_paths_in_dataset(const Query& all);
/** the (selection id, path) pairs a subject's selections name: their own `path`, their `array_path` */
std::vector<std::pair<std::string, std::string>> named_paths_of(const Database& db, const std::string& subject_id);
/** the (selection id, params_json) of a subject's selections whose parameters may name an array (`weights`, `atlas`) */
std::vector<std::pair<std::string, std::string>> params_naming_arrays(const Database& db, const std::string& subject_id);

// ── the rows as they travel (the subject bundle, the catalogue) ─────────────────────────────────────────────────
/** the subject's complete selections, by path and name, as stored rows */
std::vector<Row> subject_rows(const Database& db, const std::string& subject_id);
/** the extent rows and the span-member rows of the subject's selections */
std::vector<Row> subject_extent_rows(const Database& db, const std::string& subject_id);
std::vector<Row> subject_span_rows(const Database& db, const std::string& subject_id);
/** the vocabulary dictionaries of those names a subject sees (app, user, its dataset's, its own) */
std::vector<Row> vocabulary_rows(const Database& db, const std::vector<std::string>& names, const std::string& dataset_id, const std::string& subject_id);
/** dictionaries by id, and their entries by dictionary and code, as stored rows */
std::vector<Row> dictionary_rows(const Database& db, const std::vector<std::string>& ids);
std::vector<Row> dictionary_entry_rows(const Database& db, const std::vector<std::string>& ids);
/** the shared dictionaries (scope app or dataset) with each entry as a JSON array of all its columns (dictionary_id left out), by scope and name */
struct SharedLookup { std::string scope, name; Json entries; };
std::vector<SharedLookup> shared_lookups(const Database& db);
/** every dictionary's (id, shared key): `scope/name` for a shared one (app, dataset, user), empty for a subject's own */
std::vector<std::pair<std::string, std::string>> dictionary_keys(const Database& db);

// ── the rows (the writer's composition calls these; they take the database, never decide) ────────────────────────
/** a selection row, its counts (`n_elements`, and `n_members` when given) and its detail rows in one go; the composition (pending/complete, arrays) is the writer's */
void insert_row(Database& db, const Row& row, std::optional<std::int64_t> n_elements, std::optional<std::int64_t> n_members, const std::optional<std::vector<SpanSpec>>& spans,
                const std::vector<ExtentSpec>& extents, const std::vector<Cols>& own_measurements);
void replace_spans(Database& db, const std::string& selection_id, const std::vector<SpanSpec>& spans);       // n_members follows
void replace_extents(Database& db, const std::string& selection_id, const std::vector<ExtentSpec>& extents);

}  // namespace nxr::io::selection
