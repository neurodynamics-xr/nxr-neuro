# One io service, read and write, faced in Python and WebAssembly (spec, 2026-11-19)

Status: **design, no code.** Decided by the maintainers (2026-11-19), in this order:
1. the Python binding and the WebAssembly binding each expose nxr-neuro's io **in full**, reading AND writing, with one surface;
2. nxr's Python `nxr.io` is then re-implemented on that surface (a thin shim; the rest of nxr unchanged);
3. the Cortical Flow Studio reaches the same surface through `bindings/wasm-io` (its step 2, after its own backend reaches schema 54).

This spec gives the surface (section 2), how each face maps it (section 3), the coverage of `nxr.io` (section 4), concurrency, schema versions and
store forms (section 5), what is out of scope (section 6), the implementation plan (section 7) and the questions for the maintainers (section 8).
It builds on `2026-11-17-io-record-api.md` (io in two levels), `2026-11-17-compute-and-run.md` (the faces' `run` and `read`),
`2026-11-11-wasm-io.md` (the Studio's `DataClient` over WebAssembly) and `2026-11-16-subject-centred-datasets.md` (packages and the catalogue).

## 1. Where it stands (inventory, on `feat/phase-measures` at 6cfdd20)

### 1.1 io's task API (level 2): `io/include/nxr/io/task/`

`task.h` (474 lines) and `dataset.h` (69 lines); sources `io/src/task.cpp`, `io/src/task_dataset.cpp`.

| group | calls (file:line) |
|---|---|
| handles | `Subject::of(SubjectWriter&)`, `of(const SubjectPackage&)`, `of(const DatasetWriter&, id)`, `of(const Database&, id)` (task.h:327-330); `Dataset::of(DatasetWriter&)`, `of(const Database&)`, `open_readonly(folder)` (dataset.h:43-45) |
| find | `find`, `require`, `find_all` (by `Want`: class, name, kind, type, manifold, from/to manifold, of_field, member_of, dictionary, code, `name_like`, `stored`, parameter equalities and presence, `any_status`), `by_id`, `at_path`, `require_id`, `by_id_or_name`, `line_named`, `manifold_named` (335-377) |
| describe | `describe` (finder::Value), `manifold_of`, `factors`, `geometry`, `charts`, `members`, `member`, `spans`, `element_members`, `fields_on_axis`, `fields_of_operator`, `entry_count`, `dictionary_entries`, `grid_levels`, `map_slices`, `map_slice`, `origin`, `key_digests`, `node_counts`, `recording_time_lines`, `recording_time_line`, `projection_of` (341-419) |
| naming rules | `node_path`, `atlas_node` (371-372) |
| measurements (read) | `measured_kinds`, `measured_selections`, `has_measures`, `has_measure`, `read_measurements` (the Studio's query), `query_members` (filters and roll-ups in SQL) (355, 393-401) |
| cache | `cached(cls, key, path)`, `stored_operator(key)` (384-386) |
| write | `unit(by)` / `commit(body, by)`; outside a unit: `write_measurements`, `remove_measurements`, `set_params`, `write_member_rows` (362, 409-416). Inside `Unit`: `field`, `statistic`, `operator_`, `selection` (with grids, time grids, products, tilings, spans), `measurements` (array-backed, and pair rows), `line`, `surface`, `product`, `points`, `cell_complex`, `path_of`, `replace` (moves aside, never deletes), `write_measurements`, `remove_measurements`, `set_params`, `commit`, `rollback` (436-463) |
| dataset | `subjects`, `subject_kinds`, `subject_named`, `find_subject`, `stored_arrays`; on a writer: `recover`, `drain`, `remove_subject`, `remove(cls, id)`, `package_subject_id` (dataset.h:48-61) |

Arrays handed to a unit are `Values` (C order; exactly one of float32, float64, int32, uint8 filled). Outside the task API, io also has: the chunk service
(`chunk_loop.h`), `ArrayReader` / `layout_array` / `write_region` / `populate_array` (`array_reader.h`, `array_writer.h`), `create_dataset`, `open_writer`,
`SubjectWriter::create|open` (`writer.h`), `Catalogue`, `create_catalogue`, `register_packages`, `scan_packages`, `select_subjects`, `open_subject_package`,
`create_group_package` (`catalogue.h`), `SubjectPackage::open`, `create_package`, `open_package_writer` (`subject_package.h`), `split_dataset`,
`join_dataset`, `migrate_selection_storage`, `migrate_measurement_storage` (`migrate.h`), `compact_dataset` (`compact.h`), `subject_bundle`,
`bundle_by_path` (`bundle.h`), `folder_for` (`layout.h`), `read_mesh` (`surface.h`), the writer lock (`writer_lock.h`), and the importers
(`io/include/nxr/convert/`: bids, fif_objects, freesurfer_import, gifti_import, nifti_import, cifti_import, recording, fibers, templates, and the
Brainstorm converter).

What the task API does NOT yet have: opening a dataset or package by folder in a stated mode; creating a dataset, catalogue or subject; reading a
node's arrays (the face reads them with `ArrayReader` itself); a general manifold write (a sensor set, a volume grid, a fibre set), a standalone
dictionary write, a row-level `update`, a nested unit (savepoint); the Studio's `job` rows; `layout`'s `folder_for`; raw array keys (create, write a
region, remove, list, exists, text); the bundle and the renderer's named views.

### 1.2 The pipeline face: `pipelines/include/nxr/pipelines/face.h`

`stamp`, `pipelines`, `run(dataset, subject, RunRequest)`, `run_group`, `datasets(datastore)`, `subjects(dataset, names, where)`,
`results(dataset, subject, what)`, `measurement_query(dataset, subject, MeasurementQuery)` (`where`: `io` in SQL or `compute` in memory),
`read(dataset, subject, name, ReadRequest)` (a statistic, operator, field, selection or manifold; a time-resolved field by sample span and rows through
the chunk service, with `projectors` and `reading = shape`; a selection's measurement table; a manifold's labelling by `family`), and
`synthetic_dataset`. Every call takes folder paths and opens the dataset again (`open_for`, face.cpp:50). `read` and `results` open the dataset with
`OpenOptions{lock = false}` (face.cpp:507-516).

### 1.3 The Python binding (`bindings/python`)

`src/pipeline.cpp` binds the face: `_pipelines`, `_datasets`, `_subjects`, `_results`, `_run`, `_read`, `_synthetic_dataset`
(pipeline.cpp:25-93); `measurement_query` and `run_group` are not bound. `nxr_neuro.list` (`pipelines`, `datasets`, `subjects`, `results`) and
`nxr_neuro.read` / `nxr_neuro.run` (`_pipeline.py:72,100`) wrap them, arrays as NumPy (`np.frombuffer`, little-endian, copied).
**`nxr_neuro.io` is the file-format readers** (`io.py`: `open_edf|bdf|brainvision|eeglab|fif`, `read_nifti|gifti|tsv`; `src/formats.cpp`), not
the dataset io. The Python face writes nothing except through `run`.

### 1.4 The WebAssembly binding (`bindings/wasm-io/src/binding.cpp`, 1376 lines)

One Embind class `Datastore` on a **datastore** folder (its `catalog.json`), 28 functions (binding.cpp:1348-1375): `status`, `bundle`, `view`, `row`,
`read`, `create`, `put`, `remove`, `updateSubject`, `removeSubject`, `createField`, `selection`, `measurements`, `operator`, `fieldUpdate`,
`fieldDelete`, `manifold`, `dictionary`, `job`, `layout`, `datasetCreate`, `datasetDelete`, `catalog`, `cli`, `importSubjects`, `setEnv`,
`releaseDataset`, `close`.

- **Level used: neither the repositories nor the task API.** It includes `writer.h`, `measures.h`, `bundle.h`, `layout.h`, `array_*.h`,
  `writer_lock.h` and works on `DatasetWriter::db()` directly: 78 SQL statements or `query`/`execute` calls in the binding (step 8 of the record-API
  migration, not yet done). It knows one-database datasets only (no catalogue, no subject packages).
- **JSON conventions:** rows travel as JSON text whose keys are column names (the Studio's row objects); the Studio's write operations come as
  `(op, rowJson, childrenJson, by)` with `op` in create · update · delete; arrays as `{dtype, shape, bytes}` (`Uint8Array` copies); a failure
  returns `{error, code}` and `js/index.mjs` throws it as `NxrIoError` with `err.code` (io's `code::`). `js/index.mjs` maps the module onto the
  Studio's `DataClient` shape (`fields.post`, `selections.put`, `measurements.get`, ...; `js/index.d.ts`).
- **Holding:** a dataset written is opened once and held with its writer lock until `close()` / `releaseDataset`; reads use a cached reader opened
  with `lock = false` (binding.cpp:1312-1336).
- **Tests:** `test/parity_writes.mts`, `parity_studio.mts`, `parity_imports.mts`, `parity_analysis.mjs`, `store_keys.mjs`: call for call against the
  Studio's TypeScript service on stores made by `make_store.py` (synthetic).

### 1.5 nxr's `nxr.io` (Python, schema 50, about 7.8k lines in 30 modules)

Public package surface (`__init__.py`): `ArrayReader`, `Datastore`, `Dataset`, `OpenPreset`, `PortRequest`, `Subject`, `open`, `read`, `write`. What
the rest of nxr uses is in section 4. Its facts: schema 50 (`PRAGMA user_version` of its `model.sql`), store format 52 (subject small keys in
`small_files.sqlite`), one database per dataset with a datastore `catalog.json`; the writer lock `<dataset>/.writer.lock`, the same file, JSON and
`NXR_WRITER_LOCK_HELD` handshake as io's (`writer_lock.h:1-9`: "byte-compatible"); a reader refuses any `user_version` other than 50 and a 49 is
migrated only by the explicit `nxr migrate`.

## 2. The service surface

### 2.1 Where it lives

**A new io header `io/include/nxr/io/service/service.h`, namespace `nxr::io::service`: a `Service` handle and its operations, implemented ON the
task API** (and, for the Studio's two generic reads, `view` and `row`, on the repositories, as the record-API spec already allows for wasm-io). It is
io's own code: JSON and arrays in, JSON and arrays out, no binding type. Both bindings are converters over it and hold no logic. The command line may
call it too (its `read`, `list`, `dataset`, `subject` verbs), so there is one implementation of each operation.

- The task API stays what pipelines use. The service is the task API **opened by folder, named for a face, and complete for writing**; every NEW
  operation below is first added to the task API (or to io) and only then named in the service.
- The service does not depend on compute. The two face calls that need compute (`run` and `measurement_query` with `where = compute`) stay in
  `pipelines/face.h` and are bound beside the service (section 3).
- `face::read`, `face::datasets`, `face::subjects`, `face::results` move into the service (they are io work); the pipeline face keeps `run`,
  `run_group`, `pipelines`, `measurement_query`, `stamp`, `synthetic_dataset`, and calls the service where it read before.

### 2.2 Values at the boundary

- **JSON**: `nxr::io::Json`. Keys are snake_case as the task structs name their members (`manifold_id`, `of_field_id`); the Studio's row objects
  keep their column names (also snake_case). The Studio's camelCase `where` keys (`subjectId`, `ofFieldId`) are accepted only by `view` and
  `readMeasurements`, as today (one generated mapping, record-API spec section 6).
- **Arrays**: `service::Array {name, dtype, shape, bytes}` (C order, little-endian), the face's `StoredArray` moved into io. Writes take the same
  struct. dtypes: those io's array writer writes (float32, float64, int8/16/32/64, uint8/16/32/64); complex arrays are written as their real
  components on a trailing axis (`value_type` complex, `components` [re, im]), as io's `DType::Complex*` is read-only (types.h:14).
- **Nodes** are returned as the task API's `Node` in JSON (`cls`, `id`, `name`, `kind`, `type`, `manifold_id`, ..., `params`, `producer`, `row`).
- **Errors** are `nxr::io::Error` with io's `code::` (`no_such_row`, `ambiguous`, `invalid_argument`, `schema_version`, `writer_lock_held`,
  `not_writable`, `already_exists`, ...). Each face maps a code to one exception class (section 3).

### 2.3 The operations

"Exists" names the C++ entry today; "NEW" is added to the task API (or io) first. The face name is the C++ word, verbatim (section 3.3).

**Handle and maintenance**

| operation | C++ entry | arguments → result |
|---|---|---|
| `open` | NEW `Service::open(path, Mode)`; on `Dataset::open_readonly`, `SubjectPackage::open`, `Catalogue::open`, `open_writer`, `open_package_writer` | a datastore (`catalog.json`), a dataset (one database, or a catalogue) or one subject package; `mode` read or write → the handle. Write takes the writer lock (section 5.1) and migrates the schema (5.2); read opens SQLite read-only and never migrates, recovers or locks |
| `close` | NEW (the wasm `close`) | releases every database and lock the handle holds; `close(dataset)` releases one (the wasm `releaseDataset`) |
| `status` | exists in wasm (`status`); NEW in io | {path, form: datastore · database · catalogue · package, schema_version, store_format, mode, lock: {holder} or null, catalog} |
| `recover` | exists `task::Dataset::recover` | → {rolled_back, finished} |
| `drain` | exists `task::Dataset::drain` | the deferred deletions and the zarr.json attribute sync |
| `verify` | NEW, on `key_digests` and io's attribute sync | → the keys whose digest or attributes differ from the rows (empty: consistent) |
| `lockHandshake` | NEW, on `WriterLock` | → the `NXR_WRITER_LOCK_HELD` value a child process is given to borrow this handle's lock |

**Datastore, datasets, subjects**

| operation | C++ entry | arguments → result |
|---|---|---|
| `datasets` | exists `face::datasets` (moves) | → {id, datasets: [{id, name, path, database, subjects}]} |
| `catalog` | exists in wasm, io's `write_catalog` | the datastore's `catalog.json`, rewritten from the databases |
| `createDataset` | exists `create_dataset`, `create_catalogue` (NEW in the task API) | name, source {tool, protocol, path}, form (database · catalogue), store format → the dataset row; idempotent by name (the wasm `datasetCreate`) |
| `removeDataset` | exists in wasm (`datasetDelete`); NEW in io | name → {deleted}; refused while another writer holds it |
| `subjects` | exists `face::subjects` (moves), `task::Dataset::subjects`, `select_subjects` | names or `where` (participants filter, catalogue only) → [{id, name, kind, status, path, ...}] |
| `subject` | exists `subject_named`, `node_counts` | name or id → the entry and its node counts |
| `createSubject` | exists `SubjectWriter::create`, `create_package` (NEW in the task API) | name, kind (subject · template · group), folder → the entry |
| `updateSubject` | exists in wasm (SQL); NEW in the task API | id, patch (name, kind, template_id, source_format, source_path only) → the row |
| `removeSubject` | exists `task::Dataset::remove_subject` | id |
| `registerPackages`, `scanPackages` | exist `register_packages`, `scan_packages` | the catalogue's members |
| `createGroup` | exists `create_group_package` | group name, members → its package |

**Find and describe** (all take the subject by name or id)

| operation | C++ entry | result |
|---|---|---|
| `find`, `findAll`, `require` | exist `Subject::find`, `find_all`, `require` | a `Want` as JSON → a Node, a list, or a clear error |
| `byIdOrName`, `atPath` | exist | Node or null |
| `describe`, `geometry`, `charts`, `factors`, `members`, `spans`, `elementMembers`, `gridLevels`, `dictionaryEntries`, `mapSlices`, `origin`, `keyDigests`, `nodeCounts`, `recordingTimeLines`, `fieldsOfOperator`, `fieldsOnAxis` | exist (the `Subject` calls of the same names) | JSON |
| `nodePath`, `atlasNode` | exist | the store key a node of that name takes |
| `layout` | exists in wasm (`layout`, on `folder_for`); NEW in the task API | {name, manifold or type, role, operator, manifoldName, factors} → {path, folder} |
| `cached`, `storedOperator` | exist | a cache key → Node or null |
| `results` | exists `face::results` (moves) | what → [{name, kind, class}] |
| `bundle` | exists `subject_bundle`, `bundle_by_path` (wasm `bundle`) | a subject's rows as the Studio's `bundleOf` (with the template's anchor rows) |
| `view`, `row` | exist in wasm (SQL in the binding); NEW `records.view`, `records.read(Ref)` in io | the Studio's renderer reads: a named view (the allowed names only) with a `where`, a row by table and id |

**Read values**

| operation | C++ entry | arguments → result |
|---|---|---|
| `read` | exists `face::read` (moves into the service) | name, and `ReadRequest` (measure, of_field, start, stop in samples, rows, projectors, reading, family) → {meta, arrays}. A time-resolved field is read through the chunk service, one chunk at a time; its span is the caller's |
| `readArray` | exists in wasm (`read`); NEW in io on `ArrayReader` | a subject and a store key, a region [[lo, hi), ...] or none → one Array |
| `readMeasurements` | exists `Subject::read_measurements` (wasm `measurements` op query) | the Studio's `MeasurementQuery` → [{code, value, weight, members}] |
| `queryMembers` | exists `Subject::query_members` | io's query (filters, roll-ups in SQL, across subjects) → rows |
| `measuredKinds`, `measuredSelections`, `hasMeasure` | exist | JSON |
| `keys`, `exists`, `readText` | NEW, on the subject's `Store` | a prefix → its keys; a key → bool; a key → its text (a `zarr.json`) |

**Write** (a unit is the scope; a write call made outside one is its own unit)

| operation | C++ entry | arguments → result |
|---|---|---|
| `unit` | exists `Subject::unit(By)` | a subject, by {tool, producer} → a Unit handle with the calls below, `commit`, `rollback` |
| `field`, `statistic`, `operator_`, `selection`, `measurements`, `line`, `surface`, `product`, `points`, `cell_complex` | exist `Unit::` calls | the `*Out` struct as JSON, its `Values` as Arrays → the new id |
| `field` of an array already written | NEW (`FieldOut` naming a stored array; the wasm `createField` "stored" case, D59) | the array's key; its layout read from the store, its value count checked against the manifold's cells |
| `manifold` | exists in wasm (`manifold` create, SQL); NEW `Unit::manifold(ManifoldOut)` | a manifold of any type with its topology, cells, geometry and dimensions (sensor sets, volume grids, fibre sets) |
| `dictionary` | exists in wasm (SQL); NEW `Unit::dictionary` (on `SubjectWriter::dictionary`) | name, scope, entries [{code, name, color}] → id |
| `update` | exists in wasm (`fieldUpdate`, `selection`/`operator`/`manifold`/`dictionary` op update, SQL); NEW `Unit::update(cls, id, patch)` | the columns the Studio's repositories allow to change → the row |
| `remove` | exists `task::Dataset::remove(cls, id)` (wasm `fieldDelete`, op delete) | a node and what depends on it; its arrays go only when no remaining row names them |
| `replace`, `pathOf` | exist | moves a node at a path aside (never deletes) |
| `writeMeasurements`, `removeMeasurements`, `setParams` | exist | rows; a filter → the count |
| `mark`, `rollbackTo` | NEW (a nested unit as a savepoint: `SubjectWriter::mark`, `rollback_to`) | inside a unit |
| `createArray`, `writeArray`, `removeArray` | exist in wasm (`create`, `put`, `remove`); NEW in io on `layout_array`, `write_region`, `erase_chunks` | a key and {shape, dtype, chunks?, fill_value?}; a region and its bytes (read-modify-write of the chunks it touches); remove refused while a row names the key (io's named paths) |
| `job` | exists in wasm (SQL); NEW `records.jobs` behind the service | the Studio's job rows: get, create, update |

**Convert and import**

| operation | C++ entry | arguments → result |
|---|---|---|
| `importBids`, `importFif`, `importTransform`, `importFreesurfer`, `importGifti`, `importNifti`, `importRecording` | exist (`io/include/nxr/convert/*.h`, the `nxr-neuro import ...` verbs) | the request as the command line takes it → {subject, counts} |
| `convertSubject` | exists (the Brainstorm converter, `nxr-neuro convert subject`) | the same |
| `importSubjects` | exists in wasm (the Studio's `importSubjects` over `convert`) | kept for the Studio; implemented on `convertSubject` |
| `splitDataset`, `joinDataset`, `compactDataset`, `migrateSelectionStorage`, `migrateMeasurementStorage` | exist (`migrate.h`, `compact.h`) | the folder and options → the report |

## 3. The faces

### 3.1 Python (`nxr_neuro`)

- **The module.** `nxr_neuro.io` becomes the dataset service: `svc = nxr_neuro.io.open(path, mode="read")`, a context manager (`with ... as svc`),
  its methods the operations of section 2.3; `svc.unit(subject, by)` a context manager that commits on exit and rolls back on an exception. The
  file-format readers move to `nxr_neuro.formats` (the C++ `nxr::formats`, `io/formats/`); the old `nxr_neuro.io.open_edf` etc. are refused with
  the new name (no alias). This rename is a question for the maintainers (section 8, Q1).
- **Arrays:** NumPy in and out. `Array` becomes an `np.ndarray` of its dtype and shape (the binding copies once, as `_read` does); `Values`
  inputs are taken from any array-like with `np.ascontiguousarray`, keeping its precision (float32 stays float32).
- **JSON:** dicts and lists (the binding passes JSON text and the package parses it, as `_pipeline.py` does; or nanobind's dict conversion).
- **Errors:** one exception class per io code in `nxr_neuro._errors` (`WriterLockHeld`, `NotWritable`, `SchemaVersionError`, `NoSuchRow`,
  `Ambiguous`, `AlreadyExists`), each a subclass of `nxr_neuro.Error`, the code kept on `.code`.
- **The GIL** is released for every call that touches the disk; a progress callback takes it back (as `_run`).
- **Unchanged:** `nxr_neuro.run`, `run_group` (to be bound), `nxr_neuro.list.pipelines`, `measurement_query` (to be bound, from `face.h`).
  `nxr_neuro.read(dataset, subject, name, ...)` and `nxr_neuro.list.datasets|subjects|results` stay as one-call shortcuts that open, call the service
  and close.

### 3.2 WebAssembly (`bindings/wasm-io`)

- **The module** binds `Service` (Embind class) and `Unit` (returned by `unit`, with its methods). JSON as text; arrays as `{dtype, shape, bytes}`
  with `bytes` a `Uint8Array` (typed-array views in `js/index.mjs`).
- **wasm-io moves to the service** (and through it to the task API and, for `view`/`row`/`job`, the repositories). Its 78 SQL statements leave
  the binding: this is step 8 of the record-API migration, done as part of this work. It gains catalogues and subject packages for free.
- **The Studio's `DataClient`** stays the JS layer's job: `js/index.mjs` maps `fields.post`, `selections.put`, `measurements.get`, ... onto the
  service's calls, so the Studio sees no change; `index.d.ts` gains the service's own typed surface beside `wasmClient`.

The 28 functions:

| today | becomes | why |
|---|---|---|
| `status`, `bundle`, `view`, `row`, `layout`, `catalog`, `close`, `job`, `removeSubject`, `updateSubject`, `setEnv` | kept (same name) | already the service's word |
| `read`, `create`, `put`, `remove` | renamed `readArray`, `createArray`, `writeArray`, `removeArray` | `read` is the service's read of a node by name; the key-level calls say they act on an array |
| `createField`, `fieldUpdate`, `fieldDelete` | `field`, `update`, `remove` (inside or outside a unit) | one verb per thing across classes |
| `selection(op, ...)`, `operator(op, ...)`, `manifold(op, ...)`, `dictionary(op, ...)` | `selection`, `operator_`, `manifold`, `dictionary` (create), `update`, `remove` | the op form splits into the verbs |
| `measurements(op, ...)` | `writeMeasurements`, `readMeasurements`, `measuredKinds`, `removeMeasurements` | the same |
| `datasetCreate`, `datasetDelete` | `createDataset`, `removeDataset` | verb first, as every other write |
| `releaseDataset` | `close(dataset)` | one word for letting go |
| `importSubjects` | kept, on `convertSubject` | the Studio's word |
| `cli` | kept, outside the service | the command line in process: a face of a face, not io |

A renamed function's old name is refused with the new one named (the consolidation rule), never kept as an alias; `js/index.mjs` is changed in the
same PR, so the Studio's calls do not break.

### 3.3 Names

One word per operation in C++, Python and JavaScript, as the decided structure requires ("the C++ word verbatim, camelCase, in every face"). io's
task API is snake_case (`find_all`, `read_measurements`); the service, which is what the faces see, names its operations in camelCase
(`findAll`, `readMeasurements`), as the compute bindings do (`signal_bandAnalytic`). Struct members stay snake_case (they are JSON keys and
column names). Whether the service is camelCase over a snake_case task API is a question (section 8, Q4); the recommendation is yes.

## 4. `nxr.io` coverage: what the rest of nxr uses → the service

Counts are uses outside `nxr/io` in `src/nxr` (static references; the per-method counts on `SubjectWriter` and `Dataset` are grep estimates).
"Shim" means pure Python that stays in nxr's new `nxr.io` (no file access). "Gap" has no service operation.

| nxr.io name (uses / modules) | service operation | note |
|---|---|---|
| `reader.open` (22/19), `Dataset` (19/13), `Datastore` | `open(path, "read")` | read-only, any schema 50-54 |
| `Dataset.bundle` (~33) | `bundle` | |
| `Dataset.reader` (~31), `Dataset.array` (~17), `ArrayReader` | `readArray` | |
| `Dataset.subject_row`, `subject`, `subjects`, `store` (~11) | `subject`, `subjects`, `keys` | |
| `Dataset.q` (~46, in pipelines, scopes, analyses) | **gap** | raw SQL text (section 4.1) |
| `writer.open_writer` (16/14), `DatasetWriter` (4/2) | `open(path, "write")` | |
| `SubjectWriter` (69/23), `.open` (19/13), `.create` (9/6) | `unit(subject, by)`; `createSubject` | the writer's composition is the unit |
| `SubjectWriter.composition` (~29), `mark`/`rollback_to` (~6) | `unit`, `mark`, `rollbackTo` | NEW nested unit |
| `SubjectWriter.field` (~32), `operator` (~19), `selection` (~18), `statistic` (~1), `product` (~6) | `field`, `operator_`, `selection`, `statistic`, `product` | the Out structs carry what the keyword arguments carried |
| `SubjectWriter.manifold` (~17), `topology`, `geometry`, `cs`, `acquisition` | `manifold` (NEW), `line`, `surface`, `points`, `cell_complex`; `cs_key`, `acquisition` members | |
| `SubjectWriter.dictionary` (~10) | `dictionary` (NEW) | |
| `SubjectWriter.find` (~31), `at` (~25), `by_name` (~2) | `find`, `atPath`, `byIdOrName` | |
| `SubjectWriter.write_array` (~14), `writer.populate_array` (2), `array_meta` (7), `default_chunks` (1) | `createArray`, `writeArray` (+ `field` of an array already written) | blockwise writes of a recording |
| `writer.remove_node` (19/11), `update_node` (1) | `remove`, `update` (NEW) | |
| `writer.create_dataset` (3), `delete_dataset` (1), `write_catalog` (4), `read_catalog` (1), `ensure_band_dictionary` (2), `BAND_DICTIONARY` (1) | `createDataset` (makes the band dictionary), `removeDataset`, `catalog`, `datasets` | |
| `writer.recover` (1), `database.drain` (2), `database.verify` (1) | `recover`, `drain`, `verify` (NEW) | |
| `layout.folder_for` (38/20), `folder_of` (1) | `layout` | |
| `layout.node_path` (17/11), `atlas_node` (3) | `nodePath`, `atlasNode` | |
| `small_files.open_array` (33/15), `open_group` (2), `exists` (5), `read_text` (2) | `readArray`, `createArray`, `writeArray`, `keys`, `exists`, `readText` | the shim returns a small array proxy (slicing, shape, dtype, attributes), not a `zarr.Array` |
| `writer_lock.WriterLockHeld`, `DatasetNotWritable` (6/3) | error codes `writer_lock_held`, `not_writable` | the shim re-exports the face's exception classes |
| `database.schema_version` (2) | `status` | |
| `measurements.query_measurements` (1), `MeasurementError` | `readMeasurements` | |
| `eigen.store_package`, `refuse_existing` (4/1) | `unit` with `points`, `operator_`, `field`, `replace` | or the `eigenmodes` definition through `run` |
| `writer.jdump` (66/22), `lattice_counts` (5), `database.now_utc` (7), `uuid7` (4), `chunks.*` (11), `grid.*` (25), `write.jsonencode` (3), `ports.ChunkSource`, `array_chunks` (3), `subject.Subject` (1), `adapter.*_from_rows` (66/14) | shim | pure Python over `bundle` and `readArray`; unchanged |
| `read.surface` (2), `write.bundle` (3) | out of scope | the Rheome viewer's files, not the dataset |
| `registry.measure`, `is_derived` (3) | the measures registry of compute (to be bound in the compute face) | not io |
| `mesh.read_mesh`, `mesh_name` (2) | io's `read_mesh` (surface.h), bound as a format reader | not the dataset service |
| `copy.copy_eigen`, `copy_kernel` (2) | **gap** | a node copied from another dataset |
| `migrate.main` (1, schema 49 → 50) | **gap** (kept in nxr) | io migrates 50 → 54 only |
| `shared_dataset.migrate`, `fragment`, `merge` (3) | **gap** | parallel-job datasets; subject packages and `splitDataset`/`joinDataset` replace them |
| `database.ensure_groups`, `remove_folder` (4) | **gap** (internal helpers) | callers move to the service's writes |

**Summary.** About 400 uses map directly onto service operations (about 15 of them on NEW operations: `manifold`, `dictionary`, `update`,
nested units, the key-level array calls, `verify`); about 190 are pure-Python helpers that stay in the shim unchanged; 5 are out of scope.

### 4.1 The biggest gap: SQL written outside `nxr.io`

The rest of nxr is not only a client of `nxr.io`'s functions: **about 43 modules outside `nxr/io` write SQL themselves** (`Dataset.q(sql)` about 46
times; `db.one`, `db.all`, `db.execute`, `db.read` on the writer's database in about 165 lines: mostly `SELECT`, with about 40 `INSERT`,
`UPDATE` and `DELETE` statements), chiefly in `nxr/convert` (16 modules), `nxr/pipelines` (14), `scopes` (6) and `analyses` (4). A thin shim
cannot serve these without handing nxr a database connection, which the decided structure reserves to io. Three ways (section 8, Q2):
1. **Pass-through (an exception):** the service offers `sql(text, params)` read-only on a read handle, for nxr only, while its call sites are
   moved; writes are not passed through.
2. **Move the call sites** in nxr onto `find`, `readMeasurements`, `queryMembers`, `view` (the change to nxr is then not "only `nxr.io`").
3. **Both, in order:** the read pass-through first (so the shim lands and nxr keeps working), then each module moved, the pass-through removed
   when the count reaches zero (an allowlist that only shrinks, as `cmake/records_layering_allow.txt`). The ~40 SQL writes move before the shim
   lands, since a write outside a unit would bypass the writer lock's recovery and the attribute sync.

Recommended: 3.

## 5. Concurrency, schema versions, store forms

### 5.1 The writer lock

- **One writer a dataset across processes**, one lock file: `<dataset>/.writer.lock` for a one-database dataset, `<package>/.writer.lock` for a
  subject package (a catalogue's packages are locked one by one). The format is shared by io, nxr's `writer_lock.py` and the Studio's
  `writer-lock.ts`: JSON {pid, host, program, started_utc}, created with `O_CREAT|O_EXCL`; stale on the same host only (a dead pid, a start before
  boot, an unreadable file older than 30 s, taken over through `<lock>.break`); `NXR_WRITER_LOCK_HELD=<lock file>:<pid>` borrows a lock (no create,
  no remove, no recovery).
- **After the shim, nxr takes the lock through the service** (`open(path, "write")`); nxr's `writer_lock.py` is retired. A child process nxr
  starts is given `lockHandshake()` in its environment, as now.
- **A handle opened to write holds the lock until `close`** (Python: the context manager, and `atexit`; wasm: as the binding holds datasets today).
  Recovery (pending rows rolled back) runs only in the process that took the lock.
- **Readers take no lock and open SQLite read-only.** Today io's "readers" do not: `open_writer` and `open_package_writer` with `lock = false` open
  the database read-write (`Database::open` defaults `read_only = false`, writer.cpp:250, subject_package.cpp:656), so a read of a schema 50-53
  database **migrates it in place without the lock**, and a package read also runs `exec_script(kPackageTables)`. The face's `read`, `results`,
  `subjects` and the wasm binding's readers all open this way. The service's read mode opens with `read_only = true` (as `Dataset::open` and
  `SubjectPackage::open` already do); this is fixed in step 1 of the plan, with a test.
- Within one process, SQLite's busy timeout (5 s) covers a reader meeting the writer's transaction; a unit is one transaction.

### 5.2 Schema versions

- **Writable open migrates in place** from 50, 51, 52 or 53 to 54 (additive steps, under the lock; `Database::open`, database.cpp:330-340).
- **Read-only open reads 50 to 54 as they are** (the queries ask the database which columns it has); a version above 54 or below 50 is refused with
  its remedy (49: nxr's `nxr migrate`; older: re-convert).
- **The transition for nxr:** nxr's own reader refuses anything but 50. Once a dataset is opened for writing by the service it is at 54, and an nxr
  without the shim can no longer read it. So nxr's reader is first taught to read 50-54 as they are (a small nxr PR, before the shim), and the shim
  lands in one switch.
- The store format follows the form: a one-database dataset is written at format 52 (small keys in each subject's `small_files.sqlite`), a subject
  package at 53. A dataset at format 50 or 51 is read; `compactDataset` brings it to 52.

### 5.3 Store forms

- **Datastore:** a folder with `catalog.json` listing its datasets.
- **One-database dataset:** `<dataset>/dataset.sqlite` (every subject's rows) and one zarr v3 store a subject, `<subject>.nxr.zarr/`, with its
  `small_files.sqlite` at format 52.
- **Catalogue dataset:** `<dataset>/catalogue.sqlite`, `participants.sqlite`, and one **subject package** a subject, `<subject>.nxr.zarr/` with its
  own `subject.sqlite` (its rows, anchor rows of its template, its small keys, a digest for every key, its content hash); group packages of kind
  `group`. `splitDataset` and `joinDataset` convert between the two forms in place, verified and reversible.
- The service opens all three (and a single package alone) behind one handle; operations that need a form say so (`registerPackages`: a
  catalogue; `where`: a catalogue). Routing a call to the database that owns its subject is the `RecordRouter` of the record-API spec (3.4),
  inside io.

## 6. Out of scope

- **Analysis code:** compute's nodes, the pipelines and their definitions. The faces reach them through `compute` and `run`, not through io.
- **The Studio's GPU:** its WebGPU chains, residency and rendering. The service supplies stored data and stores results; it does no device work.
- The Rheome viewer's bundle files (nxr's `write.bundle`, `read.surface`); exports to the field's formats (`nxr-neuro export ...`) are a later
  addition to the service, not in this round.
- Reading over HTTP (published roots): the Studio's served client covers it.
- nxr's analyses, plots and port adapters: they stay in nxr, on the shim.

## 7. Implementation plan (one PR each; scoped tests only; synthetic data only)

Every PR builds only the targets it changes (`tools/tests.py run REGEX`), runs its own tests once, and goes through one CI round. Stores in tests
are made by io's own writers or `face::synthetic_dataset`, never from a lab dataset.

| step | PR | tests |
|---|---|---|
| 1 | **Readers read-only.** A `Mode::read` in io that opens SQLite read-only and never migrates or recovers; `face::read`, `results`, `subjects` and wasm-io's reader use it | io: a schema 53 database opened to read keeps `user_version` 53 and its bytes; a package read leaves its file unchanged |
| 2 | **The task API's missing writes**: `Unit::manifold`, `Unit::dictionary`, `Unit::update`, nested units (`mark`, `rollback_to`), `field` of an array already written, `createSubject`/`createDataset`/`removeDataset` | io: each against the rows `SubjectWriter` makes for the same input; a nested rollback leaves the outer unit's rows |
| 3 | **io's key-level array calls and reads**: `readArray`, `createArray`, `writeArray` (regions), `removeArray` (refused while named), `keys`, `exists`, `readText`, `verify` | io: a region round trip on format 51, 52 and package stores; remove refused while a row names the key |
| 4 | **The service** (`nxr/io/service/service.h`): `open`/`close`/`status`, every operation of 2.3 on the task API; `face::read`, `datasets`, `subjects`, `results` move into it and the face calls it | io: one test file exercising each group on a synthetic dataset in both forms (one database, catalogue), read and write modes |
| 5 | **Python face**: `nxr_neuro.io` = the service, `nxr_neuro.formats` = the readers (Q1), exceptions, `Unit` as a context manager; bind `measurement_query` and `run_group` | pytest: open, write a field, statistic, selection and measurements in a unit, read them back as NumPy; a rolled-back unit leaves nothing; the lock refused across two processes |
| 6 | **wasm-io on the service**: the renames of 3.2, SQL removed from the binding (record-API step 8), `js/index.mjs` mapped, catalogues and packages opened | the existing parity tests (`parity_writes.mts`, `parity_studio.mts`, `parity_imports.mts`) unchanged and green; one new parity run on a catalogue dataset |
| 7 | **The importers in the service** (`import*`, `convertSubject`, `splitDataset`, `joinDataset`, `compactDataset`) | one import a format, on the synthetic files the importers' tests already use |
| 8 | **nxr (in its own repository):** its reader reads 50-54; then the shim (`nxr.io` on `nxr_neuro.io`), the SQL writes moved first, the read pass-through (Q2) with its shrinking allowlist | nxr's own `tests/` that touch `nxr.io`, scoped, on synthetic datasets |

Steps 1 to 3 touch io only and can be written one after another in one checkout; 5 and 6 can follow 4 in either order.

## 8. Questions for the maintainers

1. **`nxr_neuro.io` is today the file-format readers.** Recommend: `nxr_neuro.io` becomes the dataset service and the readers move to
   `nxr_neuro.formats` (the C++ namespace `nxr::formats`), the old names refused. A public rename of a released module.
2. **SQL written by nxr outside `nxr.io`** (section 4.1: about 43 modules). Recommend: a read-only `sql` pass-through on read handles, for the shim
   only, behind a shrinking allowlist, the SQL writes moved before the shim lands. This is an exception to "only io writes SQL" for its duration.
3. **`view` and `row` in the Python face.** They are the repositories' generic reads (level 1), which the decided structure allows only to io,
   wasm-io and io's tests. Recommend: allow them in the service for both faces (named views and rows by id, no SQL text), amending the rule.
4. **Names: camelCase service over a snake_case task API.** Recommend: camelCase in the service and every face, the task API unchanged; the
   C++-keyword case named `operator_` in every face, verbatim.
5. **wasm-io's renames** (3.2: `read` → `readArray`, `createField` → `field`, `datasetCreate` → `createDataset`, ...). Recommend: renamed with the
   old names refused, `js/index.mjs` changed in the same PR so the Studio's `DataClient` is unchanged.
6. **nxr's 49 → 50 migration and its parallel-job datasets** (`shared_dataset`). Recommend: neither moves into nxr-neuro; nxr keeps `nxr migrate`
   until no 49 dataset is left, and `shared_dataset` is retired in favour of subject packages.

## Checklist (the architecture policy)

- Compute is untouched; the service is io, and the faces are converters with no maths, no SQL and no logic.
- Every file access and every row read or written is inside io; the task API stays what pipelines use.
- The caller names every input and every span; the service finds what it is asked for and picks no data.
- Ported, not redesigned: the Studio's `DataClient` and repositories, io's task API, the face's `read`.
