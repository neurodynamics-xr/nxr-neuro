# The dataset API in Python and WebAssembly: the command line's verbs, read and write (spec, 2026-11-19)

Status: **design, no code.** Decided by the maintainers (2026-11-19):
1. the Python binding and the WebAssembly binding each expose nxr-neuro's dataset io **in full**, reading AND writing;
2. **the audience is anyone who works with nxr-neuro datasets**, as the command line's is. The bindings are built for that audience, not shaped
   by any program that uses them today. Other programs (nxr's Python io, the Cortical Flow Studio's TypeScript io) are retired and rewritten
   onto these bindings; the bindings do not adapt to them.

So the Python and WebAssembly dataset API **mirrors the command line**: the same verbs, nouns and option names, returning arrays and dictionaries
instead of printed JSON. Section 1 states where things stand; section 2 the rules of the mirror; section 3 the table, command by command; section 4
the new verbs (for the three faces); section 5 the implementation underneath; section 6 concurrency, versions and forms; section 7 what is out of
scope; section 8 the plan; section 9 the questions left. Appendix A lists what nxr will have to rewrite; appendix B what the Studio uses today
that has no public equivalent.

## 1. Where it stands (on `feat/phase-measures` at 6cfdd20)

- **The command line** (`cli/`): verb first, `nxr-neuro <verb> <noun> [options]`, one table (`cli/command_table.cpp`) from which the parsers'
  check, `--help`, the man pages and the shell completion are made. 20 verbs; besides `run` (every pipeline) and `compute` (reserved), **62
  commands**. Old spellings (`dataset create`, `convert subject`, `psd compute`, ...) are refused with exit status 2, naming the new form
  (`docs/specs/2026-11-16-command-line.md`). A computation over a recording requires `--from/--to` or `--all`. Output: JSON lines, the last a
  `"stage": "result"` object.
- **io's task API** (`io/include/nxr/io/task/task.h`, `dataset.h`): find, describe, read measurements, the cache, and writing as one atomic `Unit`
  (`field`, `statistic`, `operator_`, `selection`, `measurements`, `line`, `surface`, `product`, `points`, `cell_complex`, `replace`), and on a
  dataset `subjects`, `recover`, `drain`, `remove_subject`, `remove`. It is what pipelines and the command line use; only io writes SQL.
- **The pipeline face** (`pipelines/include/nxr/pipelines/face.h`): `run`, `run_group`, `pipelines`, `datasets`, `subjects`, `results`,
  `measurement_query`, `read` (a stored node by name: a recording by sample span through the chunk service, a selection's measurement table, a
  manifold's labelling), `synthetic_dataset`.
- **The Python binding** binds that face (`nxr_neuro.run`, `nxr_neuro.read`, `nxr_neuro.list.pipelines|datasets|subjects|results`); it writes
  nothing except through `run`. `nxr_neuro.io` holds the file-format readers (`open_edf`, `read_nifti`, ...).
- **The WebAssembly binding** (`bindings/wasm-io`) is a `Datastore` class with 28 functions shaped as the Studio's `DataClient` (`createField`,
  `selection(op, ...)`, `view`, `row`, `job`, ...), with 78 SQL statements in the binding and one-database datasets only.
- **A defect found while writing this** (issue #144): io's "readers" (`OpenOptions{lock = false}`) open SQLite read-write
  (`writer.cpp:250`, `subject_package.cpp:656`), so a read of a schema 50-53 dataset migrates it in place without the writer lock. The face's
  `read`, `results`, `subjects` and the wasm binding's readers all open this way. It is fixed first (section 8, step 1).

## 2. The rules of the mirror

1. **One command, one function, in each face.** `nxr-neuro <verb> <noun> [options]` is `nxr_neuro.<verb>.<noun>(**options)` in Python and
   `nxr.<verb>.<noun>(options)` in JavaScript (a Promise). Verb first, as on the command line. `run` and `compute` are verbs with no noun:
   `nxr_neuro.run(pipeline, ...)`. The pattern already exists: `nxr_neuro.list.subjects`.
2. **The same option names.** A long option `--from-sphere` is the argument `from_sphere`: dashes become underscores, nothing else changes. A
   positional argument (`import subject PROTOCOL`, `read dataset FOLDER`) is the first positional parameter. A flag is a boolean. A repeatable option
   (`--frequency+`) is a list. Python keywords take a trailing underscore (`--from` is `from_`, the verb `import` is the module `import_`);
   JavaScript needs none (`nxr.import.bids({from: ...})`). Required options are required, defaults are the command line's.
3. **The same refusals.** An old spelling refused on the command line has no function in the faces. A missing time range is refused with the same
   message. Errors carry io's code: a Python exception per code (`nxr_neuro.WriterLockHeld`, `NotWritable`, `SchemaVersionError`, `NoSuchRow`,
   `Ambiguous`, `AlreadyExists`, subclasses of `nxr_neuro.Error`); a JavaScript `NxrError` with `.code`.
4. **What comes back.** The command's result object (its `"stage": "result"` line) as a dictionary or a JS object; progress lines go to an optional
   `progress` callback instead of stdout. **Where the command line writes an array to a file or prints its checksum, the face returns the array**:
   NumPy in Python (in its stored precision), a typed array with `shape` and `dtype` in JavaScript. Where the command's purpose is to write files
   in another format (`export ...`), the face writes them and returns their paths.
5. **One unit per command.** A command is atomic (a unit of io's task API: everything it stores, or nothing); so is each function. The faces add
   no transaction across commands, as the command line has none.
6. **A dataset is a path.** Every function takes `dataset` (and `subject`, `package`, `datastore`) as the command line does and opens what it
   needs for that call. A convenience handle, `nxr_neuro.open(dataset)`, returns an object whose methods are the same functions with `dataset` bound
   and the database kept open between calls (no other difference); in JavaScript, `nxr.open(dataset)`.
7. **The command table is the vocabulary.** A test reads `cli/command_table.cpp` and fails when a command has no Python or JavaScript function, or a
   function's arguments differ from the command's options (the names, the required ones, the defaults), as `__check-options` checks the parsers.

This answers the four questions the first draft of this spec left open:
- the module is not renamed: the dataset API lives under the verbs (`nxr_neuro.create`, `nxr_neuro.read`, ...); `nxr_neuro.io` keeps the format
  readers;
- there is no SQL pass-through: no command passes SQL;
- there are no repository-level `view` / `row` reads in the faces: no command offers them;
- camelCase does not arise: the names are the command line's words and its options with underscores.

## 3. The table: every command, its Python and JavaScript function

The library entry is the function the command calls (section 5); logic still in `cli/` moves into it first. Arguments are the command's options
(rule 2); only notable ones are named. Python is `nn` = `nxr_neuro`; JavaScript is `nxr`.

### 3.1 Datasets, subjects, packages, catalogues

| command | Python · JavaScript | returns |
|---|---|---|
| `create dataset` | `nn.create.dataset` · `nxr.create.dataset` | the dataset (folder, form, id); `--templates` makes the default subject through `run` |
| `create subject` | `nn.create.subject` · `nxr.create.subject` | the subject |
| `create package` | `nn.create.package` · `nxr.create.package` | the package record |
| `create index` | `nn.create.index` · `nxr.create.index` | the index built |
| `create template` | `nn.create.template` · `nxr.create.template` | the template package |
| `create registration` | `nn.create.registration` · `nxr.create.registration` | the registration package |
| `create recording` (`--synthetic`) | `nn.create.recording` · `nxr.create.recording` | the recording stored |
| `delete dataset` | `nn.delete.dataset` · `nxr.delete.dataset` | what was deleted |
| `delete subject` (`--node`) | `nn.delete.subject` · `nxr.delete.subject` | what was deleted |
| `show dataset` | `nn.show.dataset` · `nxr.show.dataset` | subjects by kind, source, dates |
| `show subject` | `nn.show.subject` · `nxr.show.subject` | its node counts |
| `show package` | `nn.show.package` · `nxr.show.package` | the package record |
| `list datasets` | `nn.list.datasets` · `nxr.list.datasets` | [dataset] |
| `list subjects` (`--subjects`, `--where`) | `nn.list.subjects` · `nxr.list.subjects` | [subject], with the selection's provenance |
| `list members` | `nn.list.members` · `nxr.list.members` | [package, state] |
| `list recordings` | `nn.list.recordings` · `nxr.list.recordings` | [recording] |
| `list kernels` | `nn.list.kernels` · `nxr.list.kernels` | [kernel] |
| `list results` | `nn.list.results` · `nxr.list.results` | [field or statistic] |
| `list templates` | `nn.list.templates` · `nxr.list.templates` | [template] |
| `list pipelines` | `nn.list.pipelines` · `nxr.list.pipelines` | [pipeline, its parameters with meaning, default, unit] |
| `register package` | `nn.register.package` · `nxr.register.package` | each outcome and refusal |
| `verify package` | `nn.verify.package` · `nxr.verify.package` | the keys that differ (none: verified) |
| `release template` | `nn.release.template` · `nxr.release.template` | the released package |
| `release redistributable` | `nn.release.redistributable` · `nxr.release.redistributable` | the edition and what it leaves out |
| `split dataset` | `nn.split.dataset` · `nxr.split.dataset` | the migration report |
| `join dataset` | `nn.join.dataset` · `nxr.join.dataset` | the migration report |
| `compact dataset` | `nn.compact.dataset` · `nxr.compact.dataset` | what moved (a dry run: what would) |
| `recover dataset` | `nn.recover.dataset` · `nxr.recover.dataset` | rows rolled back, deletions finished |
| `migrate measurements` | `nn.migrate.measurements` · `nxr.migrate.measurements` | the migration report |
| `add surface` (FILE) | `nn.add.surface` · `nxr.add.surface` | the surface; the faces also take `vertices`, `faces` arrays in place of FILE (section 4) |

### 3.2 Reading and querying

| command | Python · JavaScript | returns |
|---|---|---|
| `read dataset FOLDER` (`--array SUBJECT PATH`, `--region`, `--out`) | `nn.read.dataset` · `nxr.read.dataset` | without `array`: every array the rows name (shape, dtype, checksum); with it: **the array** (or its region). `out` writes it as the command does |
| `read crossspectrum` | `nn.read.crossspectrum` · `nxr.read.crossspectrum` | **the arrays** (coherency, phase, delay or phase-slope index) and their description |
| `query measurements` | `nn.query.measurements` · `nxr.query.measurements` | **the values** [rows x members], weight, the level read and its snapped size |
| NEW `read field`, `read statistic`, `read operator`, `read selection`, `read manifold` | `nn.read.field` ... · `nxr.read.field` ... | the stored node by name and **its arrays** (section 4.1) |

### 3.3 Import and export

| command | Python · JavaScript | returns |
|---|---|---|
| `import list PROTOCOL`, `import scan PROTOCOL` | `nn.import_.list`, `nn.import_.scan` · `nxr.import.list`, `nxr.import.scan` | the protocol's subjects; its description |
| `import subject`, `template`, `group`, `templates`, `condition` | `nn.import_.subject` ... · `nxr.import.subject` ... | what was imported (the counts) |
| `import surface`, `mri`, `inverse`, `map`, `resample`, `badsegments`, `events` | `nn.import_.<noun>` · `nxr.import.<noun>` | the nodes stored |
| `import recording`, `fif`, `nifti`, `cifti`, `gifti`, `freesurfer`, `transform` | `nn.import_.<noun>` · `nxr.import.<noun>` | the nodes stored |
| `import bids` | `nn.import_.bids` · `nxr.import.bids` | the packages made, per subject |
| `import participants` | `nn.import_.participants` · `nxr.import.participants` | the variables stored |
| `export measurements`, `tsv`, `gifti`, `fif`, `bids-derivatives`, `participants` | `nn.export.<noun>` · `nxr.export.<noun>` (`bids_derivatives`) | the files written (paths); the files are the command's purpose |

That is 23 `import` and 6 `export` commands, each one function a face.

### 3.4 Pipelines

| command | Python · JavaScript | returns |
|---|---|---|
| `run <pipeline>` (a built-in, a definition, a `.yaml` path; `--from/--to` or `--all`, `--replace`, `--dry-run`, `--<parameter>`) | `nn.run(pipeline, dataset=, subject=, from_=, to=, all=, replace=, dry_run=, **parameters)` · `nxr.run(pipeline, {...})` | the result (cache hit or computed, what was stored, the range used); the group scope with `subjects`, `into` |
| `compute` | reserved, as on the command line | |

**In all: 62 commands plus `run` (and the reserved `compute`), each with one Python and one JavaScript function: 64 rows a face.** The existing
Python functions change to the mirror: `nn.read(dataset, subject, name, ...)` becomes `nn.read.<class>` (section 4.1); `nn.run`'s
`start` / `stop` in seconds become `from_` / `to`; both old forms are refused with the new one named.

## 4. New verbs (all three faces: command line, Python, JavaScript)

The command line cannot today store values a user computed, nor read one stored node by name. Both are added as commands, named as the command
line names things, so the three faces stay one vocabulary. On the command line the arrays come in and go out as `.npy` files (io already reads
and writes NumPy files, `io/include/nxr/io/npy.h`); in the faces they are arrays.

### 4.1 `read <class>`: a stored node by name

`read field | statistic | operator | selection | manifold --dataset D --subject S --name N [options] [--out FILE.npy]`: what `face::read` does
today, one noun a class (the command line's nouns are classes: `list recordings`, `list kernels`). Options, as the face's `ReadRequest`:
- `read field`: a time-resolved field (a recording, a source series) **requires `--from/--to` or `--all`** (seconds, as every command over a
  recording; the chunk service reads one chunk at a time), `--rows`, `--projectors active|none`, `--shape-only` (the description, no sample);
- `read selection`: its array, or its spans (`start`, `stop`, `code`, the dictionary); `--measure M [--field F]` reads its measurement table;
- `read manifold`: positions, cells, axes, element names; `--family NAME` reads one labelling (codes and dictionary);
- `read statistic`, `read operator` (a sparse operator: `data`, `indices`, `indptr`).

### 4.2 `add <class>`: store what a user computed

The verb is the command line's `add` ("add a surface to a subject"). Each stores one node as one unit, with a producer record naming the program
and the user's `--description`, and refuses a name already used unless `--replace` (moved aside, never deleted, as `run --replace`).

| command | stores | main options |
|---|---|---|
| `add field` | a Field on a manifold | `--name`, `--manifold`, `--values FILE.npy` ([n] or [n x components]), `--value-type`, `--unit`, `--cell`, `--derived-from` |
| `add selection` | a partition or set (a member per element), indices, or spans on a time line | `--name`, `--manifold`, `--type`, `--members FILE.npy` or `--indices` or `--spans FILE.npy` (start, stop, code), `--entries` (the members' names) |
| `add statistic` | a Statistic | `--name`, `--measure`, `--method`, `--values`, `--axes`, `--of-field`, `--partition`, `--unit` |
| `add measurements` | measurements of a field over a selection's members | `--selection`, `--field`, `--measure M=FILE.npy` (repeatable), `--unit` |
| `add operator` | an Operator between two manifolds, dense or sparse | `--name`, `--kind`, `--from-manifold`, `--to-manifold`, `--values` (dense) or `--data --indices --indptr` (csr) |
| `add points`, `add line` | a manifold of points (one coordinate each), a closed-form line | `--name`, `--positions`; `--n --spacing --origin --unit` |
| `add surface` (extended) | a surface | FILE as today, or `--vertices FILE.npy --faces FILE.npy` |

These are io's `Unit` calls (`field`, `selection`, `statistic`, `measurements`, `operator_`, `points`, `line`, `surface`), which exist; the commands
are new. Precision follows the input (a float32 array is stored as float32).

## 5. The implementation underneath

- **One function a command, in the library; the command line, Python and JavaScript all call it.** Each takes an options struct whose members are
  the command's options (snake_case, the command table's names) and returns `{result: Json, arrays: [{name, dtype, shape, bytes}]}`.
  - io-only commands (datasets, subjects, packages, catalogues, imports, exports, `read dataset`, `read <class>`, `add <class>`): in io, a new
    header `io/include/nxr/io/task/commands.h`, namespace `nxr::io::task`, implemented on the task API (and on io's existing `create_dataset`,
    `Catalogue`, `register_packages`, `split_dataset`, `compact_dataset`, the importers in `io/include/nxr/convert/` and the exporters).
  - commands that need compute (`run`, `query measurements` with its in-memory roll-up, `read crossspectrum`, `create registration`,
    `create dataset --templates`): in `pipelines/face.h`, as `run` already is.
  - Where a command's code in `cli/` still does more than parse and print, that logic moves into its library function (the command line's own rule:
    it "composes nothing the library doesn't").
- **No SQL, no row and no repository in any face.** The bindings convert arguments and results only. The WebAssembly binding moves off its own SQL
  and its repository-level calls onto these functions (its 78 statements leave the binding: step 8 of the record-API migration, done here).
- **Arrays at the boundary.** C order, little-endian, the stored dtype. Python: NumPy, copied once; inputs taken from any array-like with
  `np.ascontiguousarray`, keeping their precision. JavaScript: `Float32Array`, `Float64Array`, `Int32Array`, ... with `shape` and `dtype`.
  Complex values are stored as their components on a trailing axis (`value_type` complex), as io writes no complex dtype (`types.h:14`).
- **The GIL** is released for every call that touches the disk; the `progress` callback takes it back.

## 6. Concurrency, schema versions, store forms

### 6.1 The writer lock

- One writer a dataset across processes: `<dataset>/.writer.lock` for a one-database dataset, `<package>/.writer.lock` for a subject package (a
  catalogue's packages are locked one by one). JSON {pid, host, program, started_utc}, created with `O_CREAT|O_EXCL`; stale on the same host only
  (a dead pid, a start before boot, an unreadable file older than 30 s, taken over through `<lock>.break`); `NXR_WRITER_LOCK_HELD=<lock>:<pid>`
  lets a child process borrow its parent's lock (no create, no remove, no recovery). (`io/include/nxr/io/writer_lock.h`.)
- **A write command takes the lock for its own duration and releases it**, as on the command line. A handle from `nxr_neuro.open(dataset,
  mode="write")` holds it until `close()` (a context manager in Python; `atexit` releases it); recovery runs only in the process that took it.
- **A read takes no lock and opens SQLite read-only** (#144): it never migrates, recovers or runs `exec_script`. A read meets a writer's
  transaction through SQLite's busy timeout (5 s); a command's unit is one transaction.

### 6.2 Schema versions

- **A writable open migrates in place** from 50, 51, 52 or 53 to 54 (additive steps, under the lock; `Database::open`, database.cpp:330-340).
- **A read-only open reads 50 to 54 as they are** (the queries ask which columns the database has). Above 54 or below 50 is refused with its remedy
  (`schema_version`).

### 6.3 Store forms

- **Datastore:** a folder holding dataset folders (`catalog.json`), named by `--datastore` (`list datasets`, `create dataset --datastore`,
  `delete dataset`).
- **Catalogue dataset (the default form):** `catalogue.sqlite`, `participants.sqlite` and one **subject package** a subject, `<subject>.nxr.zarr/`
  with its own `subject.sqlite` (its rows, the anchor rows of its template, its small keys, a digest for every key, its content hash); group
  packages of kind `group`. Format 53.
- **One-database dataset:** `dataset.sqlite` and a zarr v3 store a subject with its `small_files.sqlite` (format 52); 50 and 51 are read,
  `compact dataset` brings them to 52. `split dataset` and `join dataset` convert between the two forms in place, verified and reversible.
- Every function opens either form from the same `dataset` path, as every command does (`cli/cli.h` `open_dataset`).

## 7. Out of scope

- Analysis code (compute's nodes, the pipelines, their definitions): reached through `run` (and later `compute`), not through io.
- The Studio's GPU: its WebGPU chains, residency and rendering.
- Reading over HTTP (published roots).
- Anything a command does not do. A program that needs more (appendices A and B) asks for a command; the command line, Python and JavaScript get
  it together.

## 8. Implementation plan (one PR each; scoped tests; synthetic data only)

Each PR builds only the targets it changes (`tools/tests.py run REGEX`), runs its own tests once, and goes through one CI round. Test datasets are
made by io's writers, `face::synthetic_dataset` or `create recording --synthetic`, never from a lab dataset.

| step | PR | tests |
|---|---|---|
| 1 | **Readers read-only (#144)**: a read mode that opens SQLite read-only and never migrates, recovers or locks; the face's `read`, `results`, `subjects` and wasm-io's readers use it | a schema 53 database read keeps `user_version` 53 and its bytes; a package read leaves its file unchanged |
| 2 | **The command functions** (`io/include/nxr/io/task/commands.h`, `face.h`): one a command, the command line calling them; logic left in `cli/` moved in. Two PRs: datasets, subjects, packages, read, query; then imports and exports | the existing CLI tests, unchanged and green (they now test the functions too) |
| 3 | **The new verbs** `read <class>` and `add <class>`, in the library and on the command line, in the command table | one round trip a class: `add` from synthetic arrays, `read` gives them back byte for byte; `--replace` moves aside; a name used is refused |
| 4 | **Python**: the verb modules (`nn.create`, `nn.list`, `nn.show`, `nn.read`, `nn.query`, `nn.add`, `nn.delete`, `nn.import_`, `nn.export`, ...), `nn.run` and `nn.read` changed to the mirror (old forms refused), `nn.open`, the exception classes; the coverage test against the command table | pytest: the coverage test; create a dataset and subject, `add` and `read` each class, `query measurements`; the writer lock refused across two processes |
| 5 | **WebAssembly**: the same verbs on the same functions; the `Datastore` class and its 28 functions removed (no aliases); `js/index.mjs` and `index.d.ts` give `nxr.<verb>.<noun>` | the coverage test; the same round trips in Node on a synthetic dataset |

Steps 1 and 2 touch io, pipelines and the command line only; 4 and 5 can follow 3 in either order. The wasm-io parity tests against the Studio's
service are retired with the `Datastore` class (the Studio is rewritten onto the public API).

## 9. Questions for the maintainers

1. **The new verbs' names and grammar**: `read field | statistic | operator | selection | manifold` (a noun a class) and `add field | selection |
   statistic | measurements | operator | points | line` (and `add surface` from arrays), with `.npy` files as the command line's array format.
   Recommend as written.
2. **Python keywords**: the verb `import` as the module `nxr_neuro.import_`, and the options `--from` as `from_` (an import's source and a time
   range's start). Recommend the trailing underscore (Python's convention); JavaScript keeps the bare words.
3. **The two Python functions already released change** (`nn.read(...)` → `nn.read.<class>`, `nn.run(start=, stop=)` → `from_=`, `to=`), with
   the old forms refused. Recommend yes: the mirror is the rule.

## Appendix A: what nxr will have to rewrite (input for its migration, not a design input)

nxr's `nxr.io` (schema 50, one-database datasets, store format 52; its writer lock is the same file format) and the rest of nxr's use of it:

| nxr uses | public equivalent |
|---|---|
| `io.open`, `Dataset`, `Dataset.bundle`, `.reader`, `.array` (about 100 uses) | `read <class>`, `read dataset --array`, `list results`, `show subject` |
| `writer.open_writer`, `SubjectWriter` with `field`, `selection`, `statistic`, `operator`, `manifold`, `product`, `dictionary`, `write_array`, `composition` (about 250 uses) | `add <class>` (one unit a command), `run` for pipelines; no public `dictionary`, `product` or multi-node composition |
| `writer.create_dataset`, `delete_dataset`, `write_catalog`, `recover`, `remove_node` | `create dataset`, `delete dataset`, `list datasets`, `recover dataset`, `delete subject --node` |
| `layout.folder_for`, `node_path`, `atlas_node`; `small_files.open_array`, `exists`, `read_text` (about 90 uses) | none: store keys are io's business; nxr names nodes, not keys |
| `measurements.query_measurements` | `query measurements` |
| `writer_lock` exceptions | `nxr_neuro.WriterLockHeld`, `NotWritable` |
| `migrate` (49 → 50), `shared_dataset` (fragment, merge), `copy` (eigen, kernel between datasets) | none (49 stays nxr's; subject packages and `split`/`join` replace fragments) |
| **SQL written outside `nxr.io`**: about 43 modules (`Dataset.q` about 46 times, `db.one/all/execute` in about 165 lines, about 40 of them writes), in `convert`, `pipelines`, `scopes`, `analyses` | none: each becomes a command call or a request for a command |
| pure-Python helpers (`jdump`, `uuid7`, chunk and grid constants, the Studio-port adapters over the bundle) | stay in nxr |

nxr's reader accepts only schema 50: a dataset written by nxr-neuro (54) is unreadable to an nxr not yet moved.

## Appendix B: what the Studio uses today with no public equivalent (input for its rewrite, not API)

From the WebAssembly binding's 28 functions and `js/index.mjs`:
- the renderer's generic reads: `view(name, where, limit)` over named views and tables, `row(table, id)`, and `bundle(storePath)` (every row a
  subject's model needs);
- key-level array access: `read` / `create` / `put` / `remove` of an array by store key and region (the public face reads and adds nodes, not keys);
- row-level writes in the repositories' op form: virtual fields (`createField` naming an input and an operator, no bytes), `fieldUpdate`,
  `selection` / `operator` / `manifold` / `dictionary` create · update · delete with children, `measurements` post · delete, `updateSubject`;
- `layout` (where a node would be stored), `job` rows, `releaseDataset`, `setEnv`, and `cli(argv)` (the command line in process);
- `datasetCreate`, `datasetDelete`, `catalog`, `importSubjects`, `removeSubject`, `status` have public equivalents (`create dataset`,
  `delete dataset`, `list datasets`, `import subject`, `delete subject`, `show dataset`).

## Checklist (the architecture policy)

- Compute is untouched; every command's function is io (or the pipeline face for the ones that compute).
- Every file access and every row read or written is inside io; the faces convert arguments and results only, with no SQL and no logic.
- The caller names every input and every time range; nothing is defaulted that the command line does not default.
- One vocabulary: the command table, checked against every face by a test.
