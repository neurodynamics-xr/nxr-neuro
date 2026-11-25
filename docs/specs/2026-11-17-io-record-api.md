# The io record API: every database access behind typed repositories in io (spec, 2026-11-17)

Status: **design, no code** (the inventory script, `tools/db_inventory.py`, aside). It inventories every place outside `io/` that writes SQL or holds the database handle, specifies the typed API that
replaces them (ported from the Studio's repositories), the rule and the check that keep SQL inside io, and the migration in PR-sized steps. The
migration starts after the subject-package format work now open (the pipelines' open and write paths moving to packages) has landed. Section 7 lists
the open questions, each with a recommendation.

## 0. Decision (maintainers, 2026-11-17): two levels

The repositories of sections 3 to 3.4 are the **inside** of io, not what orchestration sees.

1. **Level 1, the repositories (`nxr::io::records`, the Studio's shape: row structs, `Where` structs, `read/list/find/create/update/remove`)** stay **inside io**.
   They are used by io itself, by bindings that serve the Studio (`bindings/wasm-io`), and by record tools. Pipelines and the command line never see a row
   struct, a `Where` or a table name.
2. **Level 2, a small TASK-LEVEL API in io is the only thing pipelines and the command line use** (`nxr/io/task/*.h`, namespace `nxr::io::task`), implemented
   on level 1. It is named by the task, never as CRUD. HTTP is not built now.

The task API:

| task | calls (all take a `Subject`, io's handle to one package or dataset subject; they never take a database) |
|---|---|
| **find inputs** | io's existing `finder` completed: `find(subject, Want{cls, name, kind, manifold, relation, params, member_of, ordinal})` returns a `Found` (its id, name, class, manifold ids, value type, unit, time line, and the stored facts a step needs: `params`, `producer`, `n_vertices`, `grid`, ...) or throws one clear error (none, or several: both name the candidates). `find_all` for the few steps that list. Geometry facts a step reads (grid spacing and origin, coordinate unit, factor lines, cells) are members of `Found` for a manifold; a partition's `members(found)` and a selection's `spans(found)` are calls on it. |
| **open arrays** | `open(found)` returns an array handle for streaming (the chunk service's source, unchanged); `node_path(subject, manifold, name)` and `atlas_node(...)` stay as the naming rules. |
| **commit a result as ONE unit** | `auto u = subject.unit(by)`; every write inside it (a field, operator, selection, statistic, measurement rows, a dictionary) is one atomic composition: rows, arrays, provenance (`By`, producer JSON), the cache key and the measurements together or nothing (`commit()`; destroyed without it: rolled back). It is `SubjectWriter`'s node procedure with `compose`/`mark`/`rollback_to`, typed once. Replaces `db().tx`, `write_measurements`, `delete_measurements`, `UPDATE selection SET params_json`. |
| **cache lookup** | `cached(subject, kind, CacheKey, path)` returns the stored `Found` or none (today `find_cached`, moved into io). |
| **query measurements** | `measure(subject).kinds(selection)` (what is measured, its `computed_by`, unit), `.has(selection, measure, of_field)`, `.query(MeasurementQuery)`: the read-time reduction over partitions, members, code shifts, space and time levels, and `.values(...)` for the statistics' input; `.arrays(...)` for array-backed rows. |
| **subjects and statistics, by task** | `catalogue`/`subjects` calls of section 3.4 (list, find, status) for the cli; `statistics.of(subject, field)` and `map_slices` as `find` of class `statistic`; dictionary entries by name or code (`Found::entries()`), and `recipes.get(name)`. |

**The 228 pipeline statements, by group, and their task call** (counts: `tools/db_inventory.py`; "found" is `task::find`):

| group (pipeline statements) | task call |
|---|---|
| selections found by name, type, manifold, parameters (29 + 6 + 2) | `find(Want{cls: selection, ...})`; the parameter lookups are `Want::params` (equality and presence on the selection's parameters) |
| manifold geometry and one manifold by id, found by name and type (21 + 20 + 7 + 2) | `find` of a manifold and the `Found`'s geometry members; by id: `find_by_id(subject, cls, id)` |
| fields by name or kind, virtual or pending, on a manifold, by id (30 + 4 + 4 + 1) | `find(Want{cls: field, ...})`, `find_all` |
| operators by kind, name, target (36) | `find(Want{cls: operator, ...})` |
| members of a partition, spans of a selection (13 + 11) | `members(found)`, `spans(found)` |
| what is measured, provenance, array-backed rows (8 + 5 + 4) | `measure(subject).kinds / has / arrays` |
| measurement delete (1) | a call inside the unit |
| selection writes (4) | `unit.selection(...)`, `unit.set_params(...)` |
| cache (the cache lookup, its one statement) | `cached(...)` |
| subjects, generic row, statistics, dictionaries, topology cells (3 + 2 + 1 + 2 + 8 + 2) | `catalogue.subject`, `find`, `statistics.of`, `Found::entries()`, `recipes.get`, `Found::cells` |

Also the 137 handle arguments (`node_path`, `find_cached`, `delete_measurements`, `write_measurements`, `value_of`, `statistics_of`, `map_slices`, ...):
each becomes a task call on the `Subject`; the handle never leaves io.

**The rule** (section 4, amended): pipelines and the command line include only `nxr/io/task/` (and the writer and chunk service as before); **they include no
`nxr/io/records/` header and no `nxr/io/database.h`, and name no row, table or SQL**. Only io, `bindings/wasm-io` (it serves the Studio) and io's tests include `records/`.
The layering check enforces it.

**The migration, revised** (one PR each, each based on the latest main; each shrinks the allowlist):

| step | PR |
|---|---|
| 1 | the repositories (inside io), the probe, the layering check with its allowlist (this PR) |
| 2 | the task API in io on the repositories and the existing io helpers: `find`/`find_all`/`members`/`spans`, `measure`, `cached`, `Unit`, `By`, the prepared-statement cache; io tests compare it with the SQL it replaces |
| 3 | pipelines: the small steps |
| 4 | pipelines: the runner (`runner.h`/`registry.h` signatures take the `Subject`, not a handle) |
| 5 | pipelines: partitions and pyramid, group, measure (performance measured before and after) |
| 6 | `meeg.cpp` |
| 7 | the command line |
| 8 | the binding (it uses the repositories) |
| 9 | tests (the task API and the probe) |
| 10 | close: empty allowlist; `nxr/io.h` drops `database.h`; ARCHITECTURE.md gains the rule |

Sections 3 to 3.4 below describe level 1; read "pipelines" in section 4 and the migration table of section 5 as amended here.

## 1. Why

The architecture has three layers: **compute** (functions of matrices), **data** (`io/`: loading and storing) and **orchestration** (`pipelines/`,
`cli/`: composition only). Compute and `meeg/` already never touch the database. Orchestration does, everywhere:

- **225 SQL statements in 34 pipeline files** (220 of them `SELECT`), plus 16 in the command line and 78 in the WebAssembly binding; 293 direct
  database calls in pipelines (`one` 155, `all` 65, `read` 60, `tx` 7, `execute` 5, `insert` 1); the raw handle passed into io helpers 137 times
  (section 2 and appendix A).
- A pipeline that writes `SELECT * FROM operator WHERE subject_id = ? AND kind = 'kernel' AND name = ? AND status = 'complete'` knows the table, its
  columns, the status protocol and the naming rule. That is io's business. The same lookup is written 15 times with small differences (some forget
  `status`, some `path IS NOT NULL`), so the same question gets different answers in different pipelines.
- A schema change (the subject packages, schema 51, a renamed column) has to be chased through orchestration code, and every pipeline that reads the
  database is a place a package's layout can leak into.
- The Studio already has the design: one repository per entity, the only read and write path for its rows (`backend/src/db/repository.ts`), and a
  service that routes each call to the database that owns the row (`backend/src/service/model.ts`). nxr-neuro's WebAssembly binding re-implements
  that surface with 78 inline statements (`bindings/wasm-io/src/binding.cpp`). Porting the repositories into io gives the binding, the pipelines and
  the command line one implementation, and keeps the C++ and TypeScript surfaces in step (section 6).

**The rule this spec establishes** (section 4): only io includes `database.h` or `sqlite3.h` or writes SQL. Pipelines and the command line ask io
typed questions (`fields.find`, `selections.members`, `measurements.query`) and get typed rows back.

## 2. The inventory

Measured by `tools/db_inventory.py` (this PR), rerunnable at any commit; appendix A is its output on `main` at the time of writing. How it counts:

- **SQL statement**: a C++ string literal (adjacent literals joined, raw strings read, comments skipped) whose text starts with an SQL verb and its
  second keyword (`SELECT … FROM`, `INSERT INTO`, `UPDATE t`, `DELETE FROM`, `PRAGMA`, …). SQL is written in upper case throughout, so prose in
  literals is not counted. A statement assembled in code is counted once, at its head; its pieces (`" AND status = 'complete'"`) are counted
  separately as **fragments**.
- **Database call**: `db.` / `db().` / `db->` (and `ds.query`, io's `Dataset::query`) followed by a method of `nxr::io::Database`.
- **Handle passed**: `f(sub.db(), …)` or `f(db, …)`: an io (or Brainstorm-layer) helper called with the raw handle.
- **Groups**: each statement is classified by what it does (the rules are a table in the script, first match wins) and given the proposed API call.
- **Tests** (`*/test/*`, the binding's smoke test) are counted separately.

### 2.1 Headline numbers

| area | files | SQL statements | fragments | database calls | handle passed | `Database::open` | includes `database.h` |
|---|---:|---:|---:|---:|---:|---:|---:|
| pipelines | 34 | 225 | 16 | 293 | 137 | 1 | 2 |
| cli | 6 | 16 | 2 | 12 | 9 | 1 | 2 |
| bindings (wasm-io) | 1 | 78 | 12 | 86 | 48 | 0 | 0 |
| tests | 47 | 247 | 0 | 259 | 134 | 0 | 0 |

By entity, outside tests (319 statements): manifolds and geometry 66, selections 61, fields 46, operators 39, subjects and datasets 35,
measurements 20, dictionaries 13, spans and extents 15, generic (table named at run time) 10, storage listing 6, jobs 5, statistics 2, acquisitions 1.

### 2.2 What the statements do

The pipelines **read**: 220 of their 225 statements are `SELECT`. Their writes already go through io (`SubjectWriter`'s node methods,
`write_measurements`); the five that do not are four `UPDATE selection SET params_json` and one `DELETE FROM selection_measurement`. The binding is the
reverse: it is a write surface (the Studio's repositories) with 25 `execute` and 15 `tx` calls.

| what it does | pipelines | cli | bindings | tests | the API call that replaces it |
|---|---:|---:|---:|---:|---|
| find a selection (labelling, partition, tiling) by name and type | 29 | 1 | 3 | 73 | `selections.find` / `list(SelectionWhere)` |
| read a manifold's geometry (grid axis spacing and origin, coordinate system, unit, factor lines) | 21 | 0 | 4 | 3 | `manifolds.read(id)` → `ManifoldRecord` |
| read one manifold by id (type, name, vertex count) | 20 | 0 | 8 | 0 | `manifolds.read(id)` |
| find a field by name (and kind) on a subject | 30 | 1 | 0 | 55 | `fields.find(FieldWhere)` |
| find an operator by kind (and name, target manifold) | 36 | 0 | 1 | 5 | `operators.find` / `list(OperatorWhere)` |
| read the members of a partition (`member_of_id`) | 13 | 0 | 0 | 8 | `selections.members(partition)` |
| read a selection's time spans | 11 | 0 | 0 | 13 | `selections.spans(id)` |
| what is measured on a selection (kinds, existence, provenance) | 13 | 0 | 0 | 24 | `measurements.kinds(selection)` |
| array-backed measurement rows (`array_path`) | 4 | 0 | 0 | 12 | `measurements.list(MeasurementWhere)` |
| find selections by their parameters (`json_extract`, `LIKE` on `params_json`) | 6 | 0 | 0 | 18 | `selections.list(SelectionWhere{params})` (Q3) |
| subjects: find, list, count, write | 5 | 5 | 25 | 6 | `catalogue.subject` / `subjects(SubjectWhere)` / `status` |
| dictionaries: entries (code → name), recipe parameters, writes | 8 | 0 | 5 | 5 | `dictionaries.read` / `find`, `recipes.get(name)` |
| fields: virtual or pending (`path IS NULL`), on a manifold, by id | 9 | 1 | 3 | 5 | `fields.list(FieldWhere{stored, manifold})`, `fields.read` |
| a row of a table named at run time (`"SELECT * FROM " + table`) | 1 | 1 | 8 | 0 | `records.read(Ref)` / `records.remove(Ref)` |
| storage listing: every stored array of a dataset | 0 | 6 | 0 | 0 | `records.stored_arrays()` |
| statistics: find and read | 2 | 0 | 0 | 3 | `statistics.find` / `list(StatisticWhere)` |
| writes: selections, spans, extents, fields, operators, manifolds, measurements, jobs | 5 | 0 | 20 | 1 | each repository's `create` / `update` / `remove` |

Smaller groups (manifolds found by name and type 12, measurement rows 9, a selection of a field 4, topology cells 3, an acquisition's label 1)
are in appendix A.

Besides SQL text, **`db.read(table, id)`** (a row by id) appears 60 times in pipelines (`selection` 37, `manifold` 22, `field` 1), and the
**cache-key lookup** is a pipelines function (`find_cached`, `pipelines/src/cache.cpp`) that scans every complete node of a subject and compares
`producer_json.cache` in C++: it is called 18 times and becomes `records.find_cached` in io.

The **raw handle leaves orchestration** 137 times in pipelines, into helpers that take `const Database&`: `node_path` (27), `find_cached` (18),
`delete_measurements` (17), `write_measurements` (12), `finder::value_of` (6), `statistics_of` (6), `map_slices` (4), `atlas_node` (4),
`measured_by` (4), `query_measurements` (3), and the Brainstorm layer's `band_ref` (8) and `find_frequency` (4). The binding passes it 48 times
(mostly `drain`, `upsert` and `check_cell`). These helpers are already in io; the migration turns them into repository methods or io-internal functions, so the handle itself stops
leaving io.

Two public pipelines headers carry the handle in their signatures (`runner.h`: `plan`, `plan_runs`, `Plan::text`; `registry.h`: a `const Database&`
member), and two cli files include `database.h` and open a dataset database directly (`cli/run_command.cpp`, `cli/main.cpp`). io's own public
`Dataset::query(sql)` is a raw-SQL door used by `cli/read_command.cpp`.

The per-file table (file → statements → groups) is in appendix A; one line per statement, with its group and proposed call, is
`tools/db_inventory.py --section statements`.

### 2.3 Metadata duplicated between rows and Zarr attributes

Measured from `io/schema/model.sql` (the `<table>_attributes` views, which the drain writes into each node's `zarr.json`: `attributes_text` in
`io/src/database.cpp`) and the row tables: for each node table, the columns the view copies, and the children it embeds. The script's `--section
schema` and a short comparison of the view's `json_object` keys with the table's columns (the method is in appendix B).

| node | row columns | copied into `zarr.json` attributes | embedded children | also stated by Zarr's own array metadata |
|---|---:|---:|---|---|
| field | 29 | 29 (all) | contributions | `data_type`, `chunk_shape_json`, `codecs_json`, `fill_value_json` (zarr.json's `data_type`, `chunk_grid`, `codecs`, `fill_value`) |
| operator | 36 | 36 (all) | factors, both coordinate systems and spaces | the same four, plus `n_rows`, `n_cols`, `nnz` (a sparse group's own attributes `format`, `shape` say them again) |
| statistic | 30 | 27 (all but the three array-layout columns) | none | `data_type` |
| selection | 28 | 28 (all) | extents, spans, its own measurements (`code IS NULL`), its dictionary and every entry | `array_path`'s array |
| manifold | 27 | 27 (all) | topology with cells, geometry, dimensions, coordinate system, space | none |

**Finding.** The duplication is total and deliberate: the schema's header states that a row's columns ARE the node's attributes and that no key
lives only in `zarr.json` (so a lost database is rebuilt from the stores, and a store is readable without the database). It is consistent because one
writer produces both (the views, in SQL). Three costs are worth recording for the API:
- **Array layout is stated three or four times** (row column, attribute copy, Zarr's native metadata, a sparse group's attributes). Zarr's own
  metadata is the truth for the bytes; the row columns are a cache of it. The API returns them from the row and never asks callers to compare.
- **Each selection carries its dictionary's every entry** in its attributes, so a dictionary update rewrites every selection that reads it (the
  Studio's repository does exactly this, by design).
- **Writing member measurements resyncs the selection**: the trigger `selection_measurement_ai` enqueues a selection upsert for every insert, but the
  view embeds only the selection's own rows (`code IS NULL`); a batch of member rows rewrites a `zarr.json` whose content does not change. A
  `WHEN NEW.code IS NULL` on the two triggers would avoid it. The schema is shared with the Studio and nxr, so this is a question for all three (Q12).

No reader in `pipelines/` or `cli/` takes node metadata from `zarr.json` instead of the row: `cli/read_command.cpp` prints a node's document (its
purpose), and `io/src/sparse.cpp` reads a sparse group's structural attributes. The API keeps that: **rows are read from the database, bytes from the
store**, and attributes are written only by the drain.

### 2.4 Tables and columns code never names

Appendix B.1 counts, per table, the literals in code that name it (SQL text or a whole literal, e.g. a table list); model.sql and its generated views
are not counted as code.

| table | finding |
|---|---|
| `selection_element` | **Unused.** Named only in io's lists of tables to copy or remap (`migrate.cpp`, the package's row set). Nothing writes or reads a row. Labels live in arrays (`array_path`). |
| `datastore` | **Retiring.** The package format keeps one hint row; nothing reads it to find data. io 12 literals (open, package, catalogue code), cli 1, binding 2. |
| `sync_job` | **io-internal, used.** The attribute-sync queue (triggers enqueue, the drain writes `zarr.json`): io 15, tests 1, no other layer. No API. |
| `job` | **Studio parity only.** The binding's `job(...)` (12 literals) and one pipelines use; nxr-neuro's own runs do not write it. |
| `selection_extent`, `operator_factor` | Written only by the binding (the Studio's repositories); read only through the attribute views and generic copies. |
| `variable_dictionary`, `subject_variable` | io only (the participants table of the package format, new). |
| `field_contribution` | io 4, binding 2: group fields (pyramid reduce) and the Studio. |

Columns named by at most two literals outside tests (appendix B.2): `acquisition.ses`, `acq`, `proc` (io's BIDS entity writer only);
`topology.orientable`, `selection.center`, `selection.weight_fn`, `operator_factor.transposed`, `job.inputs_json`, `job.result_json` (the binding
only); `space.parameters_json`, `coordinate_system.is_reference`, `transition_json`, `operator.from_/to_coordinate_system_id`, and
`statistic.span_json`, `members_skipped`, `components_json`, `scaling_json` (written by io, read by nobody but the attribute view). None is unused
outright: every column is named somewhere, because io's writers and the binding set them by name. The typed API covers every column anyway (generated
rows, section 3.1); which ones are worth keeping is a schema question for the three implementations, not this API's.

## 3. The API

### 3.1 Shape: the Studio's repositories, in C++

Namespace `nxr::io::records`, headers `io/include/nxr/io/records/*.h`, one per repository. The entities, verbs and names are the Studio's
(`backend/src/db/repository.ts`) wherever they fit:

| Studio (`Repositories`) | C++ (`Records`) | verbs (Studio's) | added for nxr-neuro |
|---|---|---|---|
| `manifolds` | `manifolds` | `read(id)` with topology, geometry, dimensions · `list(where)` · `create(row, children, by)` · `update` · `delete` · `sync` | `find(where)`, `factors(id)` |
| `selections` | `selections` | `read(id)` with extents, measurements · `list(where)` · `create` · `update` · `delete` · `sync` | `find`, `members(partition[, ordinal])`, `spans(id)` |
| `fields` | `fields` | `read` · `list` · `create(row, by, children)` · `update` · `delete` · `sync` | `find` |
| `operators` | `operators` | `read` · `list` · `create(row, by, children)` · `update` · `delete` · `sync` | `find` |
| `dictionaries` | `dictionaries` | `read(id)` with entries · `list(scope)` · `create` · `update` · `delete` | `find(scope, name)`, `codes(id)` |
| `measurements` | `measurements` | `create(rows, by)` · `query(q)` → `{code, value, weight, members}` · `kinds(selection)` · `delete(where)` | `list(where)` (array-backed rows), `values(...)` |
| `jobs` | `jobs` | `read` · `list(where)` · `create` · `update` · `delete` | |
| | `statistics` | | `read` · `find` · `list` · `slices` (map slices) · `create` via the writer |
| | `recipes` | | `get(name)` |
| | `subjects`, `catalogue` | | section 3.4 |
| | `templates`, `registrations` | | section 3.3 |
| | `records` (generic) | | `read(Ref)`, `remove(Ref)`, `find_cached`, `stored_arrays` |

C++ idiom:
- **Rows are structs**, one per table, generated from `model.sql` (Q1) with `to_json`/`from_json` whose keys are the column names (the Studio's row
  objects and the binding's JSON, unchanged). Nullable columns are `std::optional<T>`; JSON columns are `Json`; `status` is an enum.
- **Filters are `Where` structs** with `std::optional` members, named as the Studio names them (`subject_id` for `subjectId`, `of_field_id` for
  `ofFieldId`): an unset member does not filter. `list` returns `std::vector<Row>`; `find` returns `std::optional<Row>` and throws
  `code::ambiguous` when more than one row matches (today several pipelines silently take the first).
- **`delete` is a C++ keyword**: the verb is `remove` (as `SubjectWriter::remove` already is). The binding and TypeScript keep `delete`.
- **Bulk measurement writes take `std::span<const NewMeasurement>`**; reads that can be large have a visitor form (`for_each(where, fn)`) so a
  caller never needs the whole result in memory.
- **`by`** (who writes) is a `By {tool, version}` value that every write stamps into `created_by`/`modified_by`/`computed_by` (the Studio's `by`).
- Errors are `nxr::io::Error` with io's codes (`no_such_row`, `ambiguous`, `invalid_argument`, `schema_version`); a missing row is
  `std::nullopt` from `read`/`find`, never an exception.

```cpp
namespace nxr::io::records {

enum class Status { pending, complete };

struct FieldRow {                       // generated from model.sql (one struct per table)
  std::string id; Status status; std::string subject_id; std::string name;
  std::optional<std::string> path; std::string kind; std::optional<std::string> manifold_id;
  /* ... every column ... */ Json producer;
};

struct FieldWhere {                     // the Studio's keys first, then nxr-neuro's
  std::optional<std::string> subject_id, kind, manifold_id, of_field_id, of_operator_id, acquisition_id;
  std::optional<std::string> name;
  std::optional<bool> stored;           // path IS NOT NULL / IS NULL (virtual or not yet written)
  std::optional<Status> status = Status::complete;   // Q2: complete rows unless asked
};

class Fields {
 public:
  std::optional<FieldRow> read(std::string_view id) const;
  std::vector<FieldRow> list(const FieldWhere&) const;
  std::optional<FieldRow> find(const FieldWhere&) const;        // one or none; several: Error(ambiguous)
  FieldRow create(const NewField&, const By&, const FieldChildren& = {});   // row-only (virtual) fields; arrays: section 3.2
  FieldRow update(const FieldRow&, const By&);
  void remove(std::string_view id);
  void sync();
};

struct ManifoldRecord {                 // manifolds.read: the Studio's row & {topology, geometry, dimensions}
  ManifoldRow row; TopologyRow topology; std::vector<TopologyCellRow> cells;
  std::optional<GeometryRow> geometry; std::vector<GeometryDimensionRow> dimensions;
  std::optional<CoordinateSystemRow> coordinate_system;
};

struct SelectionWhere {
  std::optional<std::string> subject_id, manifold_id, of_field_id;   // Studio
  std::optional<SelectionType> type; std::optional<std::int64_t> code; // Studio
  std::optional<std::string> name, member_of_id, dictionary_id;
  std::optional<bool> stored;           // array_path IS NOT NULL
  std::vector<ParamEquals> params;      // Q3: {json path, value} equalities on params_json
  std::optional<Status> status = Status::complete;
};

struct MeasurementQuery {               // the Studio's MeasurementQuery, then the pyramid's read-time roll-up
  std::string selection_id, measure;
  std::optional<std::optional<std::string>> of_field_id;   // unset: any; set to nullopt: the geometric measurements
  int shift = 0; std::optional<std::string> reduction;
  std::optional<double> min_value, max_value; std::optional<std::int64_t> limit; Order order = Order::desc;
  std::optional<std::string> within_selection_id; std::optional<std::int64_t> within_code;
  std::optional<std::string> of_statistic_id; std::optional<std::int64_t> of_index;
  std::optional<int> space_level, time_level; std::optional<std::int64_t> band, spatial_band;
};
struct MeasurementCell { std::int64_t code; std::optional<std::int64_t> tile, band; double value, weight; std::int64_t members; };

class Measurements {
 public:
  std::int64_t create(std::span<const NewMeasurement> rows, const By&);   // a derived measure refused; the measure dictionary ensured
  std::vector<MeasurementCell> query(const MeasurementQuery&) const;     // THE readback reduction (section 3.3)
  std::vector<MeasurementKind> kinds(std::string_view selection_id) const;   // + computed_by, unit per kind
  std::vector<MeasurementRow> list(const MeasurementWhere&) const;        // array-backed rows: array_path, component
  MeasurementValues values(const MeasurementWhere&) const;                // [rows x members], the statistics' input (today measurement_values)
  std::int64_t remove(const MeasurementWhere&);
};

class Records {                         // one database: a subject package, a group package, a format-52 dataset
 public:
  Manifolds& manifolds(); Selections& selections(); Fields& fields(); Operators& operators();
  Dictionaries& dictionaries(); Measurements& measurements(); Statistics& statistics(); Jobs& jobs();
  Recipes& recipes(); Templates& templates();
  std::optional<Ref> find_cached(Kind, std::string_view subject_id, const Json& cache_key, std::optional<std::string> path = {}) const;
  Unit unit(const By&);                 // a scoped unit of work (section 3.2)
};
}  // namespace nxr::io::records
```

### 3.2 Writes and the unit of work

- **One write path.** A node with arrays (a manifold, field, operator, selection or statistic with bytes) is still written by `SubjectWriter`'s node
  procedure (row `pending` → keys → `complete`, recovery by key prefix); that stays as it is, already typed, already in io. The repositories'
  `create`/`update`/`remove` are the Studio's row-level writes (a virtual field, an analytic operator, a selection with its children, a dictionary
  with its entries) and are implemented on the same procedure, so a row never reaches the database by another route. Pipelines that today run
  `UPDATE selection SET params_json` call `selections.update`.
- **`Unit`: a scoped unit of work** replacing `db().tx(...)` and `SubjectWriter::compose`. `auto u = records.unit(by);` begins; every repository
  write inside it joins; `u.commit()` completes the nodes it made and drains the sync queue once; a `Unit` destroyed without `commit` rolls back
  (rows deleted, keys removed: `SubjectWriter::abort`). A nested unit is a savepoint (`SubjectWriter::mark` / `rollback_to`). The cache rule
  (`cached`) keeps its shape: `store(result)` runs inside a unit.
- `sync()` is kept for parity (the Studio's); in C++ a committed unit has already drained.

### 3.3 What nxr-neuro needs beyond the Studio

- **Statistics.** `statistics.read(id)`, `find(StatisticWhere{subject_id, name})`, `list(StatisticWhere{subject_id, of_field_id, of_statistic_id,
  partition_selection_id, measure, method, status})`, `slices(subject_id, manifold_id)` (today `map_slices`); arrays read through `read_statistic`.
  Their writes stay `SubjectWriter::statistic`.
- **Spans.** `selections.spans(id)` → `std::vector<SpanRow>{ordinal, start, stop, code}` (11 pipeline statements read exactly this); written as a
  selection's children.
- **The cache-key lookup.** `records.find_cached(kind, subject_id, key, path)` moves from `pipelines/src/cache.cpp` into io; pipelines keep the
  `CacheKey`, `stamp` and `cached` rule. Today it scans every complete node of the table and compares JSON in C++; an indexed key column is Q4.
- **Provenance.** `By` on every write; `recipes.get(name)` → the recipe's parameters (today two copies of a nested `SELECT` on `dictionary_entry`);
  `measurements.kinds` returns `computed_by` and `unit` per kind (five pipeline statements read them separately); the package's origin records
  (`origin_of`, `set_origin`) become `records.origin(ref)`.
- **Partition reductions at read: `measurements.query` is THE readback reduction.** The cardinal rule (reductions are measurements over partition
  selections) has one read side: `query` groups by member, by `code >> shift` (the dyadic roll-up of a tile partition), by the space and time levels
  of a pyramid (`space_level`, `time_level`, `band`), restricted by `within_selection_id`/`within_code`, and computes derived measures from rolled-up
  ingredients. No caller outside io writes `SUM`, `AVG`, `GROUP BY` or a roll-up over measurement rows; the one pipeline statement that aggregates
  measurement rows today (`COUNT(*) … GROUP BY computed_by`, `measure.cpp`) becomes `kinds`.
- **Templates and registrations.** `templates.find(name, version)` → the template package's reference (subject id, content hash, location hint, from
  the catalogue's `template_reference`); `templates.anchor(ref)` → an anchor row of the template a package's rows point at; `registrations.of(subject)`
  → the registration a subject records (schema 51: the `registration` subject kind). Reads only: templates are written by import.

### 3.4 Per subject package, group packages and the catalogue

The format (spec `2026-11-16-subject-centred-datasets.md`) gives one database per subject (`<subject>.nxr.zarr/subject.sqlite`), group packages of
kind `group`, and a light catalogue (`catalogue.sqlite`). The API follows it:

- **A `Records` is opened on one database**: `Records::of(SubjectPackage&)` (read), `Records::of(SubjectWriter&)` (write, the writer's database
  and lock), and, while format-52 datasets exist, `Records::of(DatasetWriter&)`. A pipeline on one subject holds one `Records`, the subject's; it
  never sees another subject's file. The template's rows it needs are anchor rows in its own package, read by the same repositories.
- **The catalogue is its own small API**, over `catalogue.sqlite`, not a repository of node rows: `catalogue.subjects(SubjectWhere{kind, status,
  names, where})` (the `--subjects`/`--where` selection, today `select_subjects`; its provenance with each member's content hash),
  `catalogue.subject(name_or_id)`, `catalogue.status(subject)` (the counts of the model's `subject_status` view, today eight `count(*)` statements
  in the binding), `catalogue.open(entry, Mode)` → that member's `Records`.
- **Group work** reads each member through its own `Records` (opened read-only from the catalogue, its hash checked) and writes one group package
  through its `Records`; contributions name members through anchor rows. Nothing reads `index.sqlite` to compute (it is a browsing cache).
- **Routing, ported from `model.ts`.** `RecordRouter` holds several `Records` (the members of a catalogue, or the datasets a service has open) and
  routes each call to the database that owns its row, subject or selection, as the Studio's service does. The binding (and through it the Studio's
  `NXR_BACKEND=nxr-neuro`) uses the router; a pipeline uses one `Records`.
- **Generic reads for the renderer.** The Studio's renderer reads named views (`model.list(view, where)`, `model.read(table, id)`); the binding's
  `view(name, whereJson, limit)` and `row(table, id)` become `records.view(name, where, limit)` (the allowed view names only) and
  `records.read(Ref)`, implemented in io.

## 4. The rules

1. **Only io includes `nxr/io/database.h` or `sqlite3.h`, opens a database, or writes SQL.** That includes io's own public headers: at the end of the
   migration `nxr/io.h` stops including `database.h`, and the io helpers that take `const Database&` (`finder.h`, `measures.h`, `statistic.h`,
   `layout.h`, `bundle.h`, …) move behind the repositories (`io/src/` or `nxr/io/detail/`). `Dataset::query(sql)` is removed.
2. **Pipelines and the command line use only the record API** (and `SubjectWriter` for node writes, and the chunk service for recordings). No `.db()`,
   no `Database`, no `Row` from a query, no `V(...)` parameter lists.
3. **The bindings** use the record API too; `bindings/wasm-io` becomes a JSON adapter over `Records`/`RecordRouter` (the dependency table gains
   "`bindings/wasm-io`: io's record API").
4. **Tests may use one small test-only helper, also in io**: `nxr/io/testing/probe.h`, in a target `nxr_io_testing` that only test executables
   link. It offers what the API deliberately does not: `probe.count(table, where)`, `probe.rows(table, where)` (read-only, any table, for assertions
   about what was written), `probe.integrity()` (`PRAGMA foreign_key_check` and `integrity_check`), and `probe.tamper(sql)` (named so a test that
   corrupts a database on purpose says so). Tests otherwise use the same API as pipelines.
5. **The check (a CTest, like `formats_layering`)**: `cmake/check_records_layering.cmake`, test `layering_sql_only_in_io`, with
   `layering_sql_only_in_io_catches` proving it fails on a bad tree. Over `pipelines/`, `cli/`, `bindings/`, `compute/`, `meeg/` (vendored trees
   skipped), it fails on:
   - an `#include` of `nxr/io/database.h` or `sqlite3.h`, or of `nxr/io/testing/` outside a `test/` folder;
   - a string literal that starts with an SQL verb and its second keyword (the inventory script's pattern, upper case);
   - `.db()`, `Database::open`, `nxr::io::Database`, `sqlite3_`;
   - in `test/` folders: the same, except SQL inside `probe.tamper(...)`.
   During the migration it reads an **allowlist** (`cmake/records_layering_allow.txt`: the files not yet migrated). Each migration PR removes its
   files from the list; a file not on the list that gains SQL fails at once, so no new SQL lands while the old is removed. The last step deletes
   the list. `tools/db_inventory.py` stays the progress measure (its counts reach zero outside io).

## 5. The migration

Starts after the format completion has landed (open now: pipelines computing on subject packages, schema 51, group results split and join, the
content hash). Those change the pipelines' **open and write paths** (`cache.cpp`/`cache.h`, `eigen.cpp`, `group.cpp`, `measure.cpp`, `pyramid.cpp`,
the cli commands); this migration changes their **lookups**. Steps 1 and 2 add new files in io only and can be written in parallel; pipeline steps
wait for the format work to merge, and each pipeline step touches only lookup lines, never an open or write path, so a later format change rebases
cleanly.

| step | PR | contents | keeps main green by |
|---|---|---|---|
| 1 | io: the read repositories | generated row structs (`tools/gen_record_rows.py` → `io/include/nxr/io/records/rows.h`, a test that it is current); `Records`, `Unit`; read side of `manifolds`, `fields`, `operators`, `selections` (`read`, `list`, `find`, `members`, `spans`), `records.read(Ref)`; `nxr_io_testing` with the probe; the layering check with an allowlist of every current file | new files only; io tests compare each repository call with the SQL it replaces on the test datasets |
| 2 | io: measurements, statistics, dictionaries, cache, provenance | `measurements` (typed wrappers of `write_measurements`, `query_measurements`, `measurement_kinds`, `delete_measurements`, `measurement_values`), `statistics`, `dictionaries`, `recipes`, `find_cached` moved into io (`pipelines/src/cache.cpp`'s one statement goes; its callers unchanged), `By`, origin records | as step 1, after the format work's change to `cache.cpp` |
| 3 | pipelines: the small steps | `cores_steps`, `crossings_steps`, `extent_steps`, `frame_tiling`, `frames`, `joint`, `listing`, `null_compute`, `patches`, `pointspread`, `run_repeat`, `spatial_band`, `spatial_events_steps`, `spatial_null`, `speedspectrum`, `statistics`, `wishart_null`, `measure_time`, `input` (54 statements) | their tests unchanged and passing; files leave the allowlist |
| 4 | pipelines: the runner | `registry.cpp`, `runner.cpp` (19 statements, 35 calls), `finder` use, and the public signatures of `runner.h`/`registry.h` (`const Database&` → `const Records&`) | the runner's plan and execute tests |
| 5 | pipelines: partitions and pyramid | `tiles`, `scaleselection`, `cellquadrature`, `pyramid`, `group`, `measure`, `eigen` (81 statements; the selection-parameter lookups, Q3) | after the format work's changes to `group`, `measure`, `pyramid`, `eigen` have merged |
| 6 | pipelines: `meeg.cpp` alone | 70 statements, 27 operator lookups by kind and name: `operators.find` | its own PR because of its size |
| 7 | the command line | `nxr_neuro.cpp`, `measure_command.cpp`, `read_command.cpp` (storage listing → `records.stored_arrays`), `run_command.cpp`, `main.cpp`; `Dataset::query` removed | cli tests |
| 8 | the WebAssembly binding | `binding.cpp`'s 78 statements onto `Records`/`RecordRouter` (every write through the repositories, the `job` repository, `view`/`row`) | the binding's call-for-call tests against the Studio's service, unchanged |
| 9 | tests | the 47 test files onto the API and the probe, in two or three PRs by folder | the suites themselves |
| 10 | close | allowlist deleted; io's public headers stop exposing `Database`; `nxr/io.h` drops `database.h`; ARCHITECTURE.md's dependency table and PR checklist gain the rule | the layering check with no allowlist |

**Which pipelines first, and why.** The small steps (3) exercise every read repository on simple, well-tested code, with no overlap with the format
work. The runner (4) next, because its public signature is the one every later step builds on. `meeg.cpp` (6) is the largest single file and is
mostly one pattern (operator lookups), so it is fastest once `operators.find` is proven. The binding (8) last among code because its tests compare
it with the Studio, which the repositories must already match.

## 6. Parity with the Studio

**Where the C++ API and the TypeScript repositories must agree** (the binding is tested call for call against the Studio's service, and the
Studio's `NXR_BACKEND=nxr-neuro` routes its `ModelService` through the binding):
- entity names and verbs: `manifolds`, `selections`, `fields`, `operators`, `dictionaries`, `measurements`, `jobs`; `read`, `list`, `create`,
  `update`, `delete` (C++ `remove`), `sync`; `measurements.create/query/kinds/delete`;
- the `where` keys (camelCase in JSON, snake_case in C++, one mapping, generated), and their meaning (an absent key does not filter; `ofFieldId:
  null` asks for the geometric measurements, absent asks for any);
- the children shapes: `SelectionChildren{extents, spans, measurements}`, `ManifoldChildren`, `FieldChildren{contributions, selections}`,
  `OperatorChildren{factors}`, `DictionaryChildren{entries}`;
- `MeasurementQuery`'s keys and the result `{code, value, weight, members}`, `kinds`' result, `delete(where)` returning a count, derived measures
  refused on write and computed at read;
- the write rules: ids minted (uuid7) when absent, `created_utc`/`modified_utc` stamped, `by` stamped, children given are replaced, the sync drained
  after a write, a dictionary update touching the selections that read it, job statuses and their time stamps;
- row JSON: keys are column names, exactly the Studio's row objects.

The binding's two recorded differences (a manifold's delete takes its topology when nothing else stands on it; a column a row does not give is
written NULL) stay recorded in one place, the binding's tests.

**What the Studio would need later (not changed here):**
- a `statistics` repository and `selections.spans`/`members`, if its renderer wants them (nxr-neuro's additions are a superset; the Studio's calls
  keep working without them);
- `ofStatisticId`/`ofIndex` are already in its `MeasurementQuery`; the pyramid keys (`spaceLevel`, `timeLevel`, `band`, `spatialBand`) would join it
  if the Studio reads pyramids through `query`;
- the per-package routing (its `model.ts` routes over dataset databases; with packages it would route over a catalogue's members, as the format spec,
  section 7.1, describes);
- optionally, typed per-repository methods on the WebAssembly client (`selections.create(row, children, by)`) in place of the current
  `selection(op, rowJson, childrenJson, by)`; the op form stays until the Studio asks.

## 7. Open questions, each with a recommendation

1. **Row structs: written by hand or generated from `model.sql`?** Recommend **generated** (`tools/gen_record_rows.py`, as the Studio generates
   its attribute views), with a test that fails when the header is stale. One schema, three implementations; hand-written structs drift.
2. **Do `list`/`find` return only `complete` rows by default?** Recommend **yes** (`status = complete` unless the `Where` says otherwise). 129 of
   the pipelines' 225 statements filter on it; of the rest, all but 13 read a row by id or a child table (which have no status). The 13 node
   lookups by name without it (6 manifold, 2 subject, 2 selection, 2 virtual-field, 1 statistic) are either deliberate (they then
   say so in the `Where`) or the inconsistency the default removes.
3. **Selections found by their parameters** (`json_extract(params_json, …)`, `LIKE`; 6 pipeline and 18 test statements: joint pyramids, atlases,
   tile roles). Recommend **a `params` equality filter in `SelectionWhere` now** (a list of {JSON path, value}), ported from the existing uses, and
   promoting the keys used most to columns in a later schema step shared with the Studio and nxr.
4. **The cache lookup scans and parses every complete node of a table.** Recommend **an indexed `cache_key` column** (a hash of the canonical key
   JSON) in a later schema step; `find_cached`'s signature does not change.
5. **`find` with several matches**: first row (today's behaviour) or an error? Recommend **`Error(ambiguous)`**, naming the rows; a caller that
   means "the newest" says so in the `Where` (an `order`/`limit` pair, rarely needed).
6. **Where row-level writes live**: repositories or `SubjectWriter`? Recommend **both on one procedure**: `SubjectWriter` keeps node writes with
   arrays (typed specs), the repositories' `create`/`update`/`remove` are the Studio's row-level writes implemented on the same procedure; no second
   path to the tables.
7. **Ids as strong types** (`FieldId`, `SelectionId`)? Recommend **not now**: plain `std::string` in named struct members; revisit once the
   migration shows real mix-ups.
8. **One `Records` per subject package, or one over all of a dataset's packages?** Recommend **one per package** for computing (the format's rule:
   a job on one subject needs that package only) and the `RecordRouter` for services and group work.
9. **The binding's dependency.** The dependency table says bindings depend on `compute` only; `wasm-io` already depends on io. Recommend recording
   "`bindings/wasm-io`: io's record API" in the table in step 10, never `database.h`.
10. **Tests: the API or the probe?** Recommend **the API by default**; the probe only for counts, raw assertions on what was written, integrity
    checks and deliberate corruption, so tests also prove the API answers the questions pipelines ask.
11. **Performance**: a statement prepared per call today. Recommend **a prepared-statement cache inside `Records`** and visitor forms for large reads
    (`for_each`), measured on the largest pipeline test before and after step 5.
12. **Schema questions for the three implementations** (not this API's to decide): drop `selection_element` (unused); add `WHEN NEW.code IS NULL` to
    the member-measurement sync triggers (section 2.3); retire `datastore` with the package format; keep `job` for the Studio. Recommend raising them
    with the next shared `model.sql` step, and leaving them out of the API (no repository for `selection_element`, `sync_job` or `datastore`).

## Checklist (the architecture policy)

- Compute untouched; it never sees a row.
- Every file access, every row read and written, inside io; orchestration composes and asks io typed questions.
- The caller still names every input; the API finds what it is asked for and never picks data.
- Ported, not redesigned: the Studio's repositories and its routing service, with their names.

## Appendix A: the inventory (`tools/db_inventory.py`)

Regenerate with `python3 tools/db_inventory.py` from the repository root (all sections below), `--section statements` for one TSV line per
statement (file, line, group, table, proposed call, SQL), `--section schema` for appendix B.

<!-- BEGIN GENERATED: tools/db_inventory.py -->
### Totals by area

| area | files | SQL statements | SQL fragments | database calls | handle passed to a helper | `Database::open` | includes `database.h`/`sqlite3.h` |
|---|---:|---:|---:|---:|---:|---:|---:|
| pipelines | 34 | 228 | 16 | 296 | 137 | 1 | 2 |
| cli | 6 | 16 | 2 | 12 | 10 | 1 | 2 |
| bindings | 1 | 78 | 12 | 86 | 48 | 0 | 0 |
| tests | 48 | 252 | 0 | 264 | 136 | 0 | 0 |

### Database calls by method

| area | one | all | execute | tx | read | insert | query |
|---|---:|---:|---:|---:|---:|---:|---:|
| pipelines | 157 | 66 | 5 | 7 | 60 | 1 | 0 |
| cli | 6 | 3 | 0 | 0 | 0 | 0 | 3 |
| bindings | 41 | 5 | 25 | 15 | 0 | 0 | 0 |
| tests | 175 | 74 | 3 | 0 | 12 | 0 | 0 |

### `db.read(table, id)` (a row by id, no SQL text) by table

| table | pipelines | cli | bindings | tests | proposed API call |
|---|---:|---:|---:|---:|---|
| `field` | 1 | 0 | 0 | 3 | fields.read(id) |
| `manifold` | 22 | 0 | 0 | 0 | manifolds.read(id) |
| `selection` | 37 | 0 | 0 | 9 | selections.read(id) |

### SQL statements by what they do

| group | pipelines | cli | bindings | tests | proposed API call |
|---|---:|---:|---:|---:|---|
| schema and metadata probes | 0 | 0 | 0 | 2 | none: io internal (Session::open runs migrations; the schema floor retires has_table/has_column) |
| storage listing: every stored array of a dataset | 0 | 6 | 0 | 0 | records.stored_arrays(Where{subject}) -> vector<StoredArray> |
| generic: one row of a table named at run time | 1 | 1 | 8 | 0 | records.read(Ref) / records.remove(Ref) (Ref = {Kind, id}) |
| subjects: count nodes per subject (status) | 0 | 0 | 8 | 3 | catalogue.status(subject) (the subject_status view) |
| subjects: write | 0 | 0 | 3 | 2 | subjects.update(row, by) / subjects.remove(id) (io writer) |
| subjects: find one by name/path/id | 3 | 2 | 12 | 0 | catalogue.subject(name_or_id) -> optional<SubjectRow> |
| subjects: list members | 3 | 3 | 2 | 1 | catalogue.subjects(SubjectWhere{kind, status, where}) / session.dataset() |
| dictionaries: recipe parameters by name | 2 | 0 | 0 | 3 | recipes.get(name) -> optional<Json> |
| dictionaries: entries (code -> name) | 8 | 0 | 2 | 2 | dictionaries.read(id).entries / dictionaries.find(DictionaryWhere{scope, name}) |
| dictionaries: write | 0 | 0 | 3 | 0 | dictionaries.create(row, entries, by) / update / remove |
| measurements: write | 0 | 0 | 1 | 0 | measurements.create(span<const NewMeasurement>, by) |
| measurements: delete | 1 | 0 | 1 | 0 | measurements.remove(MeasurementWhere{selection, ofField, measure, ofStatistic}) |
| measurements: provenance (computed_by, unit) | 5 | 0 | 0 | 3 | measurements.kinds(selection) (computed_by, unit per kind) |
| measurements: what is measured (kinds, existence) | 8 | 0 | 0 | 21 | measurements.kinds(selection) / measurements.exists(MeasurementWhere) |
| measurements: array-backed rows (array_path) | 4 | 0 | 0 | 12 | measurements.list(MeasurementWhere{selection, ofField, measure}) -> rows with array_path |
| measurements: read rows | 0 | 0 | 0 | 9 | measurements.list(MeasurementWhere) |
| spans: read a selection's time spans | 11 | 0 | 0 | 13 | selections.read(id).spans / spans.list(selection) -> vector<SpanRow> |
| spans and extents: write | 0 | 0 | 4 | 0 | selections.create/update(row, SelectionChildren{spans, extents}, by) |
| statistics: find and read | 2 | 0 | 0 | 3 | statistics.list(StatisticWhere{subject, name, ofField, status}) / read(id) |
| manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) | 21 | 0 | 4 | 3 | manifolds.read(id) -> ManifoldRecord{row, topology, geometry, dimensions, coordinate_system} |
| manifolds: topology cells | 2 | 0 | 1 | 0 | manifolds.read(id).topology.cells |
| manifolds: one by id (type, name, size) | 20 | 0 | 8 | 0 | manifolds.read(id) |
| manifolds: find by name/type on a subject | 7 | 1 | 0 | 4 | manifolds.list(ManifoldWhere{subject, type, name, componentOf, status}) |
| manifolds and acquisitions: write | 0 | 0 | 2 | 0 | manifolds.create(row, ManifoldChildren, by) / update / remove |
| acquisitions: label | 1 | 0 | 0 | 0 | acquisitions.read(id) |
| fields: one by id | 1 | 0 | 3 | 0 | fields.read(id) |
| fields: virtual or pending (path IS NULL) | 4 | 0 | 0 | 6 | fields.list(FieldWhere{subject, name, stored=false}) |
| fields: on a manifold | 4 | 1 | 0 | 0 | fields.list(FieldWhere{subject, manifold, status}) |
| fields: find by name (and kind) on a subject | 30 | 1 | 0 | 58 | fields.find(FieldWhere{subject, name, kind}) -> optional<FieldRow> |
| fields: write | 0 | 0 | 2 | 0 | fields.create(row, by, FieldChildren) / update / remove |
| operators: find by kind (and name, target) | 36 | 0 | 1 | 5 | operators.find / list(OperatorWhere{subject, kind, name, toManifold, stored}) |
| operators: write | 0 | 0 | 2 | 0 | operators.create(row, by, OperatorChildren) / update / remove |
| selections: members of a partition (member_of_id) | 13 | 0 | 0 | 8 | selections.members(partition_id[, ordinal]) -> vector<SelectionRow> |
| selections: found by their parameters (json_extract, LIKE) | 6 | 0 | 0 | 18 | selections.list(SelectionWhere{..., params}) (open question 3) |
| selections: of a field | 2 | 0 | 0 | 2 | selections.list(SelectionWhere{subject, ofField, name}) |
| selections: labellings and partitions by name/type | 29 | 1 | 3 | 73 | selections.find / list(SelectionWhere{subject, manifold, type, name, stored, status}) |
| selections: write (params, rows) | 4 | 0 | 3 | 1 | selections.create(row, children, by) / update / remove |
| jobs | 0 | 0 | 5 | 0 | jobs.read / list(JobWhere) / create / update / remove |

### Tables named by statements outside tests

`selection` 64, `field` 46, `manifold` 46, `operator` 39, `subject` 25, `selection_measurement` 21, `selection_span` 13, `geometry_dimension` 13, `dictionary_entry` 11, `(named at run time)` 10, `topology_cell` 5, `dictionary` 5, `job` 5, `geometry` 3, `dataset` 3, `statistic` 2, `selection_extent` 2, `topology` 2, `operator_factor` 2, `field_contribution` 2, `acquisition` 1, `coordinate_system` 1, `space` 1

### File -> statements -> groups (each group's API call: the groups table above)

| file | SQL | fragments | calls | handle args | groups (count) |
|---|---:|---:|---:|---:|---|
| `bindings/wasm-io/smoke.cpp` | 2 | 0 | 2 | 0 | subjects: list members (1); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (1) |
| `bindings/wasm-io/src/binding.cpp` | 78 | 12 | 86 | 48 | subjects: find one by name/path/id (12); generic: one row of a table named at run time (8); manifolds: one by id (type, name, size) (8); subjects: count nodes per subject (status) (8); jobs (5); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (4); spans and extents: write (4); subjects: write (3); selections: write (params, rows) (3); selections: labellings and partitions by name/type (3); fields: one by id (3); dictionaries: write (3); dictionaries: entries (code -> name) (2); subjects: list members (2); operators: write (2); manifolds and acquisitions: write (2); fields: write (2); measurements: delete (1); measurements: write (1); manifolds: topology cells (1); operators: find by kind (and name, target) (1) |
| `cli/main.cpp` | 0 | 0 | 0 | 0 | (includes or handle only) |
| `cli/measure_command.cpp` | 2 | 0 | 2 | 1 | selections: labellings and partitions by name/type (1); fields: find by name (and kind) on a subject (1) |
| `cli/nxr_neuro.cpp` | 5 | 2 | 6 | 6 | subjects: list members (2); fields: on a manifold (1); manifolds: find by name/type on a subject (1); generic: one row of a table named at run time (1) |
| `cli/package_command.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `cli/read_command.cpp` | 8 | 0 | 3 | 0 | storage listing: every stored array of a dataset (6); subjects: find one by name/path/id (1); subjects: list members (1) |
| `cli/run_command.cpp` | 1 | 0 | 1 | 2 | subjects: find one by name/path/id (1) |
| `pipelines/include/nxr/pipelines/cache.h` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/include/nxr/pipelines/registry.h` | 0 | 0 | 0 | 0 | (includes or handle only) |
| `pipelines/include/nxr/pipelines/runner.h` | 0 | 0 | 0 | 0 | (includes or handle only) |
| `pipelines/src/cache.cpp` | 1 | 2 | 1 | 1 | generic: one row of a table named at run time (1) |
| `pipelines/src/cellquadrature.cpp` | 8 | 0 | 10 | 13 | operators: find by kind (and name, target) (2); measurements: provenance (computed_by, unit) (1); fields: find by name (and kind) on a subject (1); selections: found by their parameters (json_extract, LIKE) (1); selections: labellings and partitions by name/type (1); spans: read a selection's time spans (1); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (1) |
| `pipelines/src/cores_steps.cpp` | 2 | 0 | 3 | 0 | manifolds: one by id (type, name, size) (2) |
| `pipelines/src/crossings_steps.cpp` | 2 | 0 | 3 | 0 | selections: members of a partition (member_of_id) (1); spans: read a selection's time spans (1) |
| `pipelines/src/eigen.cpp` | 2 | 0 | 2 | 4 | manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (2) |
| `pipelines/src/extent_steps.cpp` | 0 | 0 | 4 | 0 | (includes or handle only) |
| `pipelines/src/frame_tiling.cpp` | 2 | 0 | 3 | 1 | fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1) |
| `pipelines/src/frames.cpp` | 2 | 0 | 2 | 2 | fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1) |
| `pipelines/src/group.cpp` | 22 | 0 | 22 | 2 | selections: labellings and partitions by name/type (4); selections: members of a partition (member_of_id) (4); selections: found by their parameters (json_extract, LIKE) (3); measurements: array-backed rows (array_path) (2); measurements: what is measured (kinds, existence) (2); subjects: find one by name/path/id (2); subjects: list members (1); spans: read a selection's time spans (1); fields: find by name (and kind) on a subject (1); fields: virtual or pending (path IS NULL) (1); selections: write (params, rows) (1) |
| `pipelines/src/input.cpp` | 6 | 1 | 6 | 0 | manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (2); manifolds: topology cells (2); selections: members of a partition (member_of_id) (1); dictionaries: entries (code -> name) (1) |
| `pipelines/src/joint.cpp` | 3 | 0 | 4 | 2 | fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1); selections: members of a partition (member_of_id) (1) |
| `pipelines/src/listing.cpp` | 4 | 0 | 4 | 0 | fields: find by name (and kind) on a subject (2); operators: find by kind (and name, target) (1); statistics: find and read (1) |
| `pipelines/src/measure.cpp` | 15 | 10 | 21 | 15 | dictionaries: entries (code -> name) (3); selections: labellings and partitions by name/type (2); measurements: provenance (computed_by, unit) (2); dictionaries: recipe parameters by name (2); fields: on a manifold (2); subjects: list members (2); measurements: delete (1); subjects: find one by name/path/id (1) |
| `pipelines/src/measure_time.cpp` | 5 | 0 | 5 | 2 | selections: labellings and partitions by name/type (1); selections: members of a partition (member_of_id) (1); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (1); fields: on a manifold (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/src/meeg.cpp` | 70 | 0 | 92 | 18 | operators: find by kind (and name, target) (27); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (11); fields: find by name (and kind) on a subject (9); manifolds: one by id (type, name, size) (7); selections: labellings and partitions by name/type (5); manifolds: find by name/type on a subject (3); fields: virtual or pending (path IS NULL) (2); dictionaries: entries (code -> name) (1); selections: of a field (1); acquisitions: label (1); fields: on a manifold (1); spans: read a selection's time spans (1); fields: one by id (1) |
| `pipelines/src/null_compute.cpp` | 3 | 0 | 4 | 1 | fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/src/patches.cpp` | 3 | 0 | 5 | 5 | fields: find by name (and kind) on a subject (1); dictionaries: entries (code -> name) (1); statistics: find and read (1) |
| `pipelines/src/pointspread.cpp` | 5 | 0 | 7 | 1 | operators: find by kind (and name, target) (4); fields: find by name (and kind) on a subject (1) |
| `pipelines/src/pyramid.cpp` | 22 | 0 | 26 | 10 | selections: labellings and partitions by name/type (7); fields: find by name (and kind) on a subject (4); spans: read a selection's time spans (2); dictionaries: entries (code -> name) (2); selections: members of a partition (member_of_id) (2); operators: find by kind (and name, target) (2); selections: write (params, rows) (1); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (1); selections: of a field (1) |
| `pipelines/src/registration.cpp` | 0 | 0 | 0 | 2 | (includes or handle only) |
| `pipelines/src/registry.cpp` | 11 | 0 | 26 | 0 | manifolds: one by id (type, name, size) (5); manifolds: find by name/type on a subject (3); spans: read a selection's time spans (2); selections: members of a partition (member_of_id) (1) |
| `pipelines/src/run_repeat.cpp` | 3 | 0 | 4 | 1 | fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/src/runner.cpp` | 8 | 0 | 9 | 39 | manifolds: one by id (type, name, size) (4); spans: read a selection's time spans (1); measurements: provenance (computed_by, unit) (1); measurements: array-backed rows (array_path) (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/src/scaleselection.cpp` | 7 | 0 | 7 | 6 | selections: found by their parameters (json_extract, LIKE) (2); fields: find by name (and kind) on a subject (1); manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (1); measurements: array-backed rows (array_path) (1); measurements: provenance (computed_by, unit) (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/src/spatial_band.cpp` | 2 | 0 | 2 | 0 | selections: labellings and partitions by name/type (1); selections: members of a partition (member_of_id) (1) |
| `pipelines/src/spatial_events_steps.cpp` | 3 | 0 | 5 | 0 | manifolds: one by id (type, name, size) (2); spans: read a selection's time spans (1) |
| `pipelines/src/spatial_null.cpp` | 1 | 0 | 1 | 1 | fields: find by name (and kind) on a subject (1) |
| `pipelines/src/speedspectrum.cpp` | 1 | 0 | 1 | 0 | manifolds: find by name/type on a subject (1) |
| `pipelines/src/statistics.cpp` | 2 | 0 | 2 | 4 | selections: labellings and partitions by name/type (1); fields: find by name (and kind) on a subject (1) |
| `pipelines/src/tiles.cpp` | 8 | 3 | 10 | 5 | fields: find by name (and kind) on a subject (2); selections: write (params, rows) (2); selections: labellings and partitions by name/type (1); selections: members of a partition (member_of_id) (1); fields: virtual or pending (path IS NULL) (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/src/wishart_null.cpp` | 5 | 0 | 5 | 1 | manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (2); fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1); spans: read a selection's time spans (1) |
| `pipelines/test/test_apply_kernel_form.cpp` | 0 | 0 | 1 | 1 | (includes or handle only) |
| `pipelines/test/test_apply_kernel_levels.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/test/test_band_bursts.cpp` | 12 | 0 | 18 | 4 | measurements: what is measured (kinds, existence) (4); fields: find by name (and kind) on a subject (2); spans: read a selection's time spans (2); selections: labellings and partitions by name/type (1); measurements: read rows (1); dictionaries: entries (code -> name) (1); measurements: array-backed rows (array_path) (1) |
| `pipelines/test/test_burst_extent.cpp` | 7 | 0 | 7 | 5 | selections: labellings and partitions by name/type (4); selections: found by their parameters (json_extract, LIKE) (1); spans: read a selection's time spans (1); statistics: find and read (1) |
| `pipelines/test/test_cell_phase_current.cpp` | 6 | 0 | 6 | 6 | selections: labellings and partitions by name/type (2); measurements: what is measured (kinds, existence) (2); selections: found by their parameters (json_extract, LIKE) (1); measurements: array-backed rows (array_path) (1) |
| `pipelines/test/test_cellquadrature.cpp` | 2 | 0 | 2 | 1 | selections: found by their parameters (json_extract, LIKE) (1); selections: labellings and partitions by name/type (1) |
| `pipelines/test/test_cortical_cell_pairs.cpp` | 2 | 0 | 2 | 6 | selections: found by their parameters (json_extract, LIKE) (1); selections: labellings and partitions by name/type (1) |
| `pipelines/test/test_cortical_cell_pairs_long.cpp` | 1 | 0 | 1 | 1 | selections: found by their parameters (json_extract, LIKE) (1) |
| `pipelines/test/test_cortical_cells.cpp` | 7 | 0 | 7 | 7 | selections: labellings and partitions by name/type (2); selections: found by their parameters (json_extract, LIKE) (1); fields: find by name (and kind) on a subject (1); measurements: what is measured (kinds, existence) (1); dictionaries: recipe parameters by name (1); measurements: provenance (computed_by, unit) (1) |
| `pipelines/test/test_crossspectrum.cpp` | 4 | 0 | 6 | 4 | selections: labellings and partitions by name/type (3); fields: find by name (and kind) on a subject (1) |
| `pipelines/test/test_cyclepattern.cpp` | 2 | 0 | 2 | 1 | selections: labellings and partitions by name/type (1); fields: find by name (and kind) on a subject (1) |
| `pipelines/test/test_filter_pipeline.cpp` | 4 | 0 | 4 | 1 | selections: labellings and partitions by name/type (2); spans: read a selection's time spans (1); fields: find by name (and kind) on a subject (1) |
| `pipelines/test/test_frame_tiling_pipeline.cpp` | 2 | 0 | 2 | 1 | selections: labellings and partitions by name/type (2) |
| `pipelines/test/test_frames_pipeline.cpp` | 3 | 0 | 3 | 0 | fields: find by name (and kind) on a subject (2); selections: labellings and partitions by name/type (1) |
| `pipelines/test/test_global_bursts.cpp` | 10 | 0 | 10 | 5 | selections: labellings and partitions by name/type (5); spans: read a selection's time spans (2); selections: found by their parameters (json_extract, LIKE) (1); statistics: find and read (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/test/test_headmodel.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/test/test_hodge_power.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/test/test_measure_pipeline.cpp` | 9 | 0 | 9 | 11 | selections: labellings and partitions by name/type (2); measurements: read rows (2); dictionaries: recipe parameters by name (2); fields: find by name (and kind) on a subject (2); measurements: provenance (computed_by, unit) (1) |
| `pipelines/test/test_measure_time.cpp` | 3 | 0 | 3 | 1 | measurements: what is measured (kinds, existence) (2); measurements: array-backed rows (array_path) (1) |
| `pipelines/test/test_meeg_pipelines.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/test/test_opticalflow_pipeline.cpp` | 12 | 0 | 12 | 4 | fields: find by name (and kind) on a subject (7); selections: labellings and partitions by name/type (4); measurements: what is measured (kinds, existence) (1) |
| `pipelines/test/test_package_pipelines.cpp` | 0 | 0 | 0 | 2 | (includes or handle only) |
| `pipelines/test/test_patches_pipeline.cpp` | 8 | 0 | 8 | 0 | fields: find by name (and kind) on a subject (4); measurements: read rows (2); selections: labellings and partitions by name/type (1); statistics: find and read (1) |
| `pipelines/test/test_phasegradient.cpp` | 9 | 0 | 9 | 3 | fields: find by name (and kind) on a subject (5); selections: of a field (2); measurements: what is measured (kinds, existence) (1); operators: find by kind (and name, target) (1) |
| `pipelines/test/test_phasegradient_band.cpp` | 17 | 0 | 17 | 2 | selections: labellings and partitions by name/type (5); fields: find by name (and kind) on a subject (5); measurements: array-backed rows (array_path) (3); fields: virtual or pending (path IS NULL) (1); measurements: read rows (1); spans: read a selection's time spans (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/test/test_pipelines.cpp` | 0 | 0 | 0 | 10 | (includes or handle only) |
| `pipelines/test/test_pointspread.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/test/test_psd_pipeline.cpp` | 2 | 0 | 3 | 1 | manifolds: geometry of one (grid axis, coordinate system, unit, factor lines) (2) |
| `pipelines/test/test_pyramid_pipeline.cpp` | 37 | 0 | 37 | 7 | fields: find by name (and kind) on a subject (11); selections: found by their parameters (json_extract, LIKE) (7); selections: labellings and partitions by name/type (6); subjects: count nodes per subject (status) (3); subjects: write (2); measurements: read rows (2); measurements: array-backed rows (array_path) (2); measurements: what is measured (kinds, existence) (2); operators: find by kind (and name, target) (1); spans: read a selection's time spans (1) |
| `pipelines/test/test_pyramid_reduce.cpp` | 28 | 0 | 28 | 2 | fields: find by name (and kind) on a subject (9); selections: labellings and partitions by name/type (8); fields: virtual or pending (path IS NULL) (4); selections: found by their parameters (json_extract, LIKE) (2); manifolds: find by name/type on a subject (1); measurements: array-backed rows (array_path) (1); measurements: read rows (1); measurements: what is measured (kinds, existence) (1); schema and metadata probes (1) |
| `pipelines/test/test_quadrature_pipeline.cpp` | 4 | 0 | 4 | 2 | measurements: what is measured (kinds, existence) (2); selections: labellings and partitions by name/type (1); fields: virtual or pending (path IS NULL) (1) |
| `pipelines/test/test_registration.cpp` | 4 | 0 | 4 | 3 | operators: find by kind (and name, target) (3); schema and metadata probes (1) |
| `pipelines/test/test_runner_execute.cpp` | 0 | 0 | 2 | 1 | (includes or handle only) |
| `pipelines/test/test_runner_plan.cpp` | 0 | 0 | 0 | 16 | (includes or handle only) |
| `pipelines/test/test_scale_selection.cpp` | 5 | 0 | 5 | 1 | selections: labellings and partitions by name/type (2); selections: found by their parameters (json_extract, LIKE) (1); measurements: array-backed rows (array_path) (1); spans: read a selection's time spans (1) |
| `pipelines/test/test_scale_step.cpp` | 1 | 0 | 1 | 1 | selections: labellings and partitions by name/type (1) |
| `pipelines/test/test_source_psd.cpp` | 0 | 0 | 0 | 2 | (includes or handle only) |
| `pipelines/test/test_source_tensor.cpp` | 1 | 0 | 1 | 2 | fields: find by name (and kind) on a subject (1) |
| `pipelines/test/test_spatial_analytic.cpp` | 4 | 0 | 4 | 1 | selections: found by their parameters (json_extract, LIKE) (1); fields: find by name (and kind) on a subject (1); selections: labellings and partitions by name/type (1); measurements: provenance (computed_by, unit) (1) |
| `pipelines/test/test_spatial_bands.cpp` | 8 | 0 | 8 | 7 | selections: members of a partition (member_of_id) (3); selections: labellings and partitions by name/type (2); spans: read a selection's time spans (1); selections: write (params, rows) (1); manifolds: find by name/type on a subject (1) |
| `pipelines/test/test_spatial_events_pipeline.cpp` | 1 | 0 | 1 | 1 | manifolds: find by name/type on a subject (1) |
| `pipelines/test/test_spatial_events_planted.cpp` | 6 | 0 | 6 | 1 | selections: labellings and partitions by name/type (4); spans: read a selection's time spans (1); measurements: array-backed rows (array_path) (1) |
| `pipelines/test/test_spatial_threshold.cpp` | 2 | 0 | 2 | 1 | selections: labellings and partitions by name/type (2) |
| `pipelines/test/test_speedspectrum_pipeline.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |
| `pipelines/test/test_tiles_pipeline.cpp` | 10 | 0 | 10 | 2 | selections: labellings and partitions by name/type (4); selections: members of a partition (member_of_id) (2); measurements: what is measured (kinds, existence) (2); dictionaries: entries (code -> name) (1); manifolds: find by name/type on a subject (1) |
| `pipelines/test/test_vortex_cores_pipeline.cpp` | 17 | 0 | 17 | 2 | selections: labellings and partitions by name/type (5); fields: find by name (and kind) on a subject (5); selections: members of a partition (member_of_id) (3); spans: read a selection's time spans (2); measurements: array-backed rows (array_path) (1); measurements: what is measured (kinds, existence) (1) |
| `pipelines/test/test_wishart_null.cpp` | 0 | 0 | 0 | 1 | (includes or handle only) |

<!-- END GENERATED: tools/db_inventory.py -->

## Appendix B: schema use (`tools/db_inventory.py --section schema`)

How it measures: every string literal in C++ (comments skipped, the same reader as appendix A) and in the binding's JavaScript; a table counts when a
literal names it after `FROM`, `JOIN`, `INTO`, `UPDATE` or `TABLE`, or is exactly its name (a table list); a column counts when a literal contains it
as a word. `io/schema/model.sql` (with its generated views) is the schema, not code, and is not scanned. Section 2.3's attribute comparison takes each
`<table>_attributes` view's `json_object` keys and compares them with the table's columns.

<!-- BEGIN GENERATED: schema -->
### Tables named in code (literals naming the table)

| table | io | pipelines | cli | bindings | tests |
|---|---:|---:|---:|---:|---:|
| `meta` | 40 | 0 | 0 | 0 | 16 |
| `datastore` | 18 | 0 | 1 | 2 | 3 |
| `dataset` | 33 | 11 | 8 | 5 | 3 |
| `subject` | 86 | 27 | 23 | 20 | 33 |
| `acquisition` | 17 | 1 | 0 | 1 | 0 |
| `topology` | 14 | 0 | 1 | 13 | 2 |
| `topology_cell` | 9 | 2 | 2 | 2 | 0 |
| `space` | 9 | 7 | 0 | 1 | 5 |
| `coordinate_system` | 11 | 4 | 0 | 2 | 1 |
| `geometry` | 18 | 11 | 2 | 16 | 2 |
| `geometry_dimension` | 23 | 11 | 1 | 3 | 2 |
| `manifold` | 93 | 83 | 5 | 20 | 101 |
| `dictionary` | 29 | 5 | 0 | 9 | 6 |
| `dictionary_entry` | 20 | 8 | 0 | 3 | 12 |
| `selection` | 71 | 217 | 10 | 21 | 140 |
| `selection_element` | 3 | 0 | 0 | 0 | 0 |
| `selection_extent` | 5 | 0 | 0 | 2 | 0 |
| `selection_measurement` | 31 | 21 | 2 | 2 | 55 |
| `selection_span` | 8 | 11 | 0 | 2 | 14 |
| `field` | 60 | 107 | 7 | 12 | 117 |
| `field_contribution` | 10 | 0 | 0 | 2 | 7 |
| `operator` | 46 | 60 | 5 | 29 | 44 |
| `operator_factor` | 4 | 0 | 0 | 2 | 0 |
| `variable_dictionary` | 12 | 0 | 0 | 0 | 0 |
| `subject_variable` | 8 | 0 | 0 | 0 | 1 |
| `sync_job` | 16 | 0 | 0 | 0 | 2 |
| `job` | 4 | 1 | 0 | 12 | 0 |
| `statistic` | 29 | 105 | 2 | 0 | 17 |
| `key_digest` | 16 | 0 | 0 | 0 | 6 |
| `subject_entry` | 7 | 0 | 0 | 0 | 2 |
| `template_reference` | 7 | 0 | 0 | 0 | 2 |

### Columns named by at most two literals outside tests


<!-- END GENERATED: schema -->

## Amendment (2026-11-24): the template holds the cohort (format 56)

This amendment replaces what this document says about group packages and about a subject naming its template. A dataset has one template
subject, its own copy. A subject package is self-contained: it names no template (no `template_id`, no record reference, no anchor rows from a
template). The template holds each subject's registration onto it, each subject's results carried onto it, and the averages and statistics over
them; it refers to its subjects. Cohort results (group fields, roll-ups, fingerprint statistics, their nulls) are written into the template, with
their members and the members' content hashes in their provenance; there is no group package and no `--into`. Datasets made before are brought to
format 56 by `migrate template-links`. Released templates and registrations between templates remain, for building and comparing templates.
