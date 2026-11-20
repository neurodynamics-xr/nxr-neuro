# nxr-neuro

The project's combined C++ toolkit for measuring the spatiotemporal patterns of brain activity from MEG and EEG: **the format** (the nxr dataset), **the maths** (geometry and
spectral operators on the cortex), **the MEG/EEG source imaging** (`meeg/`: the noise covariance and the minimum-norm kernel), **the pipelines** that join them, and **the bindings** that carry the maths into MATLAB, Node and the browser.
One repository and one build; the libraries inside stay separate. **Each modality is its own folder**: `meeg/` (MEG and EEG) now; `pet/` and others later.

**Why:** the field cannot yet measure travelling, rotating and standing patterns of cortical activity reliably, in physical units, at declared scales. nxr-neuro computes those measurements
(modes, divergence, curl, speed, quadrature, the speed spectrum) on each person's own cortex, each with a named reference method it reproduces, a parity test that proves it, and its full
provenance stored with the result.

**What it is not:**
- **not a job runner:** a cluster or workflow system schedules it; nxr-neuro is a command such a system can call.
- **not a viewer:** the Cortical Flow Studio and Viewer show results.
- **not the Python analysis code:** `nxr` stays the reference that the C++ is tested against.
- **no Brainstorm code:** Brainstorm is GPLv3; the maths here is written from the papers, and Brainstorm's outputs are only goldens.

**Not done yet:** the default subject's option resolutions, HCP-1065 and Frequency manifold; BEM head models (sphere models only); MNE-Python's filter option set; modalities other than MEG/EEG; the dataset API in MATLAB (the Python and JavaScript faces carry every command; MATLAB carries `run`, `list`, `read`; `run` in the faces takes the definitions: the other hand-wired pipelines follow).

## Layout

```
nxr-neuro/
  compute/     nxr_compute   pure maths: geometry-central + Eigen + Spectra + pocketfft; no file I/O  (was nxr-compute)
  io/          nxr_io        the nxr dataset format (SQLite rows + Zarr v3), writer, Brainstorm importer (nxr::bst)   (was nxr-io's cpp/ core)
  pipelines/   nxr_pipelines the glue: read with io, compute with compute, write with io, cached; every verb of the command below (the surface: eigen, frames, tiles; the sensors:
               filters, psd, noisecov; the sources: headmodel, inverse, sourcepsd, quadrature, hodgepower; the patterns: flow_modes, speedspectrum)
  meeg/        nxr_meeg      MEG/EEG source imaging: whitener, depth weights, unconstrained minimum-norm kernel, chunk-wise noise covariance; Eigen only; no file I/O
  bindings/    MEX, Node (N-API), wasm and a CLI smoke around nxr_compute (Python and WebAssembly carry every command of the command line, MATLAB the pipeline level `run`, `list`, `read`, through `pipelines/face.h` and io's task commands); python/: the wheel (nxr_compute, io/formats and the dataset API)
  cli/         the one command `nxr-neuro`
  cmake/       shared CMake (Eigen 3.4)
  CMakeLists.txt   the one top-level build
```

Each library keeps its include paths and namespaces: compute's headers are `nxr/<toolbox>/<header>.h` and its namespaces `nxr::<toolbox>` (ARCHITECTURE.md, "The toolboxes"; there is no `nxr/compute.h`); `nxr/io/…`, `nxr/bst/…`, `nxr::io`, `nxr::bst`; `nxr/pipelines/…`.
`compute/` and `io/` were imported with their full history (`git subtree`), so `git log -- compute/src` and `git blame` reach back through the original repositories.

## Dependency rule (per library)

| library | may depend on |
|---|---|
| `compute` | Eigen, Spectra (v1.2.0 since 2026-11-12: its complex Hermitian solver, `HermEigsSolver`, for the magnetic Laplacian), geometry-central, pocketfft (header-only, vendored; amended 2026-11-10, see `compute/CONTRIBUTING.md`). No file I/O. (Welch's PSD, `nxr/signal/spectral.h`, is here.) Never includes `io`. |
| `io` | SQLite, the Zarr codecs (zstd, blosc, zlib), matio, HDF5, nlohmann/json, rapidyaml (single header, vendored in `io/extern/ryml`, v0.15.2, MIT; amended 2026-11-12, the maintainers' decision 1 of the chain-graph spec: pipeline definitions are nxr's YAML). Never includes `compute`. |
| `pipelines/graph` (`nxr_graph`) | `compute`, `meeg`, Eigen and nlohmann/json (a header; amended 2026-11-17, question 1 of the compute-and-run spec: the graph executor, nodes with typed ports, a graph checked on types and run block by block). **Never `io`**: no file, subject or database; the layering guard fails on an `io`, `bst` or `pipelines` include there, and the dispatch guard on a node called outside `pipelines/graph/src/dispatch.cpp`. Not yet used by the runner (spec steps C4 and C5). |
| `pipelines` | `compute`, `io` and `meeg`; the only place they meet. |
| `meeg` | Eigen only. Never includes `compute` or `io`; reads no file (the pipelines hand it arrays). |
| `bindings` | every library (`compute`, `meeg`, `io`, `pipelines`), as the command line does (amended 2026-11-17; the earlier "`compute` only" rule came from the old nxr-compute repository). Bindings are faces: they convert arguments and results and call the library, with no maths, no SQL and no logic of their own. |

## The store: where a subject's keys live (`io/include/nxr/io/store.h`; spec `docs/specs/2026-11-14-dataset-wide-runs.md` section 1, phase A)

Every array and every node's `zarr.json` is a zarr KEY (`source/m/vertices/zarr.json`, `source/m/vertices/c.0.1` (format 51; an older array has `c/0/1`): '/'-separated, relative to a store's root) read and written through `nxr::io::Store`, whose model is the Studio's zarrita `Readable`:
`get`, ranged `get_range`, `size`, `exists`, `exists_prefix`, `list` (the immediate children), `set`, a streamed `open_writer` (a shard is written chunk by chunk, never held whole), `erase`, `erase_prefix`, `rename_prefix`.

- **`FolderStore`** is the folder tree: key `a/b` is the file `root / a / b`, what nxr has always written (formats 50 and 51). No stored byte or path changes (`store_identity`, a test, compares every folder and file of the conformance scenario with a golden made before the store existed). **`SmallFileStore`** and **`DatastoreStore`** are format 52's (next section).
- **`PrefixStore`** is a store's keys under a prefix: a subject's keys (`SubjectWriter::store()`: what its rows' `path` columns name) inside the dataset's store (`DatasetWriter::store()`: `<dataset>/<subject>/…`, what the drain and the node procedure use).
- **What goes through it:** `ArrayReader::open(store, key)`, `ChunkSource(store, key, …)`, `layout_array` · `populate_array` · `write_region` · `SlabWriter` (store, key), the drain and `text_parity`, `create_node` · `remove_node` · `recover`, `Populate` (given the store and the node's key), `remove_store_node` · `deleting_rows`, the `_replaced-<date>` move (`rename_prefix`), the CLI's `read`, the WASM `read` · `create` · `put` · `remove`, `ZarrStore` (the CRUD helper, now over a `FolderStore`).
- **What stays on the filesystem:** the dataset folder (create, open, delete, the catalog, the writer lock), the database file, and what is not a store key (Brainstorm and mesh sources, `.npy`, a template's source files). The layering guard (`cmake/check_layering.cmake`) fails when the array reader, the array writer or the store-node removal use `std::filesystem` / fstream, or when pipelines, cli or bindings reach a subject's folder.

## Format 52: a subject's small keys in one SQLite file (`io/include/nxr/io/small_file_store.h`; spec `docs/specs/2026-11-14-dataset-wide-runs.md` section 1, phase B;, 2026-11-15: "fix it already")

A cluster file system pays a metadata round trip per file and per folder, and a subject of the fixture was 7 440 files in 8 242 folders for 3.8 GB (2 984 `zarr.json` and 6 226 files under 1 MB holding 442 MB). **Format 52** keeps the same zarr keys and the same bytes and changes only where the small ones live:

- **One database a subject:** `<subject>.nxr.zarr/small_files.sqlite`, a table `small_files(key TEXT PRIMARY KEY, value BLOB)`. It holds **every `zarr.json`** and **every chunk or shard whose value is under 1 MiB** (`kSmallFileThreshold`; "under": a value of 1 MiB or more is a file). A value at or over the threshold (the large shards, formats 50/51's 64 MiB shards) stays **a file** in the subject's folder tree. A key lives in one place; a write routes by size and removes the key from the other.
- **`SmallFileStore`** (a `Store` over a `FolderStore`): `get`, `get_range`, `size`, `exists`, `exists_prefix` look in the table first, then at the files; `set` and the streamed `open_writer` route by size (a streamed value is held until it reaches the threshold, then spills to a file; it comes into being at `close`, not at the call as a FolderStore's does); `list` merges rows and files (folder names are derived from the keys, the database file is not a key); `erase`, `erase_prefix` and `rename_prefix` act on both. Journal: WAL while open (`synchronous = NORMAL`, 5 s busy timeout; concurrent readers, one writer: the dataset's writer lock already serialises writers); **at rest the database is one file** (closing returns the journal to the default and removes the WAL files; WebAssembly has no shared memory and keeps the default journal). A read-only opener (`SmallFileOptions::read_only`) never creates or writes it, and opens a database on read-only media as immutable.
- **`DatastoreStore`** is the dataset-level store (`DatasetWriter::store()`, keys `<dataset>/<subject>/…`): the first two segments name a subject, whose keys go to its SmallFileStore (a dataset at format 52, or a subject that already has its `small_files.sqlite`: a dataset half way through `compact`) or its plain folder; keys of fewer than three segments are the folder tree's. A subject erased or renamed whole takes its database with its folder; a node moved aside into `_replaced-<date>/<subject>/…` becomes plain files there (an aside is a plain zarr folder). `DatasetWriter::close()` closes every subject's database.
- **`meta.store_format`** is the version: 52 for a dataset made now (`CreateOptions::store_format`, default 52; `dataset create --store-format 51` makes the folder tree). A format 51 or 50 dataset opens as before, readable and writable through plain files; **a build refuses a store format above the newest it knows** (`schema_version` error). `store_format_number(db)` reads it.
- **Atomicity:** none is needed beyond the node procedure (`pending` → keys → `complete`); recovery and node removal erase a node's keys by prefix, in the table and in the folder.
- **Reading outside a writer:** `nxr-neuro read` opens the dataset's keys through a read-only `DatastoreStore`. **nxr (Python: `nxr.io.small_files`) and the Studio (`backend/src/small-files.ts`) read and write format 52** (phase C, `docs/open-items.md`) and create it by default; `--store-format 51` (nxr: `NXR_STORE_FORMAT=51`) makes the folder tree, and `--no-small-files` keeps `dataset compact` at 51.
- **Counts** (`test_format52`, one synthetic subject: a surface, 24 small fields, a 2.4 MB chunk and a 13 MB shard): **59 files in 30 folders at format 51, 3 files in 3 folders at format 52** (the database, the medium chunk, the large shard); every key and byte identical to the same data at 51 (the `zarr.json` of two datasets differ only in their ids). `store_identity` (50 and 51) still passes: its scenario writes format 51 (`NXR_SCENARIO_STORE_FORMAT=52` writes 52).
- **What the files at 52 are not:** the dataset's own files (`dataset.sqlite`, the writer lock, the catalog, `zarr.json`) stay beside the subjects; a subject's large shards are still one file each (a sharded array keeps the 64 MiB shard target of format 51).

## Compacting a dataset: format 50 to format 51, and 51 to 52 (`dataset compact`, `io/include/nxr/io/compact.h`; spec `docs/specs/2026-11-14-dataset-wide-runs.md` section 1)

Format 51 writes the chunk key separator `.` (`c.0.1` in the array's folder, no `c/` folders) and shards an array over 4 MiB by the default policy (`kShardBytes`, ~64 MiB shards of 2 MiB inner chunks). A dataset made before has `c/0/1` and thousands of chunk files; `dataset compact` rewrites it in place, **one array at a time, through the store interface**, and **never on a dataset another writer holds** (the writer lock; nothing is recovered either).

- **What it does to an array** (every array under every subject, found by its `zarr.json`; links are not followed and are reported):
  1. the new layout is written *beside* the old, in `<array>/.compact/`, slab by slab (128 MiB at most in memory);
  2. every slab is read back through `ArrayReader` and compared with the old array's **byte for byte** (the decoded values; a mismatch refuses the array and leaves it as it was). An old array that stored its all-fill chunks (written dense) keeps them;
  3. `.compact/.verified` says the new layout is whole and checked;
  4. the swap: the new chunk files move up next to the old `c/` folder (other keys, nothing overwritten), then the array's `zarr.json` is written (**the commit**: from then on it is format 51), then the old `c/` and `.compact/` are removed. The attributes text is carried verbatim.
- **Rows and the dataset:** a node's row (field, operator, statistic) has its `codecs_json` and `chunk_shape_json` brought to the file's, and its `zarr.json` is rewritten as the view's canonical text (`text_parity` stays 0). `meta.store_format` is set to the format-51 text when no array of the dataset is left with the `/` separator. Arrays no row names keep their attributes as they were.
- **Resumable and idempotent.** A crash before the commit leaves the old layout intact (the next run restarts the array, or, when the marker is there, finishes the swap); a crash after it leaves a format-51 array with debris the next run removes. An array already in format 51 is skipped. A second run changes no byte (`test_compact`: a crash at each of the four stages, then the same tree as a clean run).
- **`--dry-run`** reports what would change and the files and folders under the arrays before and after (after: at most the grid's cells; a dry run writes nothing). `--subject S` compacts one subject. `--no-shard` changes only the separator (for an array that is written region by region, which a sharded one refuses).
- **An array written in pieces is kept unsharded.** The Studio's WASM `put` lays an array out with `ArrayOptions::in_pieces` (unsharded, as region writes need) and its `zarr.json` records it as the extension field `nxr_written_in_pieces: {"must_understand": false}` (zarr's own provision for extra fields; zarr-python and zarrita ignore it, the drain's canonical text keeps it, no schema change). `compact` changes such an array's separator only, whatever its size (`test_compact`: it survives compact unsharded and still accepts a region write).
- **Format 52:** after a subject's arrays, `compact` moves **every small key of its folder tree** (every `zarr.json`, every chunk or shard under 1 MiB) into the subject's `small_files.sqlite`, a batch at a time (1 024 keys or 64 MiB): the batch is written in one transaction, **read back and compared with its files byte for byte**, and only then are its files removed; the empty folders go last. A crash leaves a key in both places (the same bytes; the table is read first) and the next run finishes (`test_format52`: a crash after the first batch is in the table and one after the files are removed, then the same keys and bytes and the same tree as a clean run). `meta.store_format` becomes 52 when every array is format 51 and no subject has a small key left as a file. `--no-small-files` stops at format 51. A dry run reports `small_keys_moved` (to move) and the files on disk after as an estimate.
- **Output** (JSON): `arrays`, `compacted`, `already_format_51`, `links_skipped`, `files_before`, `folders_before`, `files_after`, `folders_after` (keys of the store: a key in a database still counts as a file), `files_on_disk_before`, `folders_on_disk_before`, `files_on_disk_after`, `folders_on_disk_after` (what the file system holds), `small_keys_moved`, `store_format_set`, `store_format`, and the same per subject.
- **Tests:** `test_compact` (values identical, counts drop, rows and `zarr.json`, idempotence, crash-resume at every stage, one subject, the lock, `--no-shard`), and `store_identity_v50_compacts`: the format-50 golden scenario compacts to the tree of the format-51 golden, file by file.
- **What it cannot do:** it does not delete anything the old layout held beyond its own chunk files, and it asks no one: run it on a real dataset only with the maintainers' OK (never delete data without it; the old chunks are replaced by the verified new ones).

## Format 53: the subject package (`io/include/nxr/io/subject_package.h`; spec `docs/specs/2026-11-16-subject-centred-datasets.md` sections 2 and 6, phase B)

A subject is one folder, `<subject>.nxr.zarr/`, that can be copied, archived, shared and opened on its own: no catalogue, participants table, index or other subject's file is needed to read, compute on or write it. Formats 50 to 52 are read and written exactly as before; nothing here changes a dataset.

- **Layout:** `subject.sqlite` (the model's tables, `model.sql` unchanged, holding the subject's own rows; **and** the `small_files` table of format 52), `.writer.lock` while a writer holds it, and the large shards, one file each, as in format 51 and 52. `SmallFileStore` has a second constructor over a `Database` that is already open: the same connection, so **a node's rows and its small keys are one transaction** (no `ATTACH`), and the package is one file at rest (the default rollback journal, no WAL).
- **Tables the package adds** (created by the package code, not in `model.sql`): `key_digest(key, size_bytes, sha256)`, the size and SHA-256 of every key's value, written as the value is written (a streamed shard is hashed as it goes, with `Sha256`); `anchor_row(table_name, row_id, from_subject_id)`, the rows of another package this one's rows point at (a template's; empty until templates are packages); `meta.store_format` = 53 and `meta.package_json`, the package's record (`PackageRecord`: subject id, name, kind, schema version, store format, the template reference, the content hash and when it was computed). The package also holds the home `dataset` and `datastore` rows (where it was made: provenance, never membership) and a copy of the shared `frequency bands` vocabulary.
- **The content hash** is the SHA-256 of the lines `<key> <size_bytes> <sha256>` of `key_digest` in key order, a line `rows`, then one line per row of every table except `meta`, `sync_job`, `small_files` and `key_digest` (tables by name, rows by primary key): `<table>` TAB the JSON array of the row's values in the table's column order. It is recomputed when a writer closes the package, from `key_digest` (no shard is re-read), and written only when it changed, so opening and closing a package without writing leaves it as it was. `SubjectPackage::verify` recomputes every key's digest from its bytes and the hash from them, and names each mismatching key; it never repairs.
- **Reader:** `SubjectPackage::open(folder)` (read only). It refuses, by name and without opening anything partly, a folder with no `subject.sqlite` (`not_found`), a database at store format below 53 (`not_a_database`: that is a dataset database) and one above 53 or at another schema version (`schema_version`). `bundle()` runs `subject_bundle` on the package's database: the same SQL, and the same bundle as the subject written into a dataset (`test_subject_package` compares them).
- **Writer:** `create_package(folder, subject_name, CreatePackageOptions)` and `open_package_writer(folder)` return a `DatasetWriter` in package mode (`is_package()`, `node_store()`): its keys are `<home dataset path>/<subject path>/...` as a dataset's, mapped onto the package's own keys, so `SubjectWriter` and every pipeline written for a dataset write a package unchanged. The home dataset is named by the caller (`CreatePackageOptions::home_dataset`; there is no default). **One package, one writer lock** (the lock's file and rules unchanged: same-host stale detection, the handshake); two jobs on two subjects share no file they write. Opening takes the lock before it recovers (`pending` rolled back, `deleting` finished); a refused creation leaves nothing behind.
- **Ids:** every id is a UUIDv7; packages written independently do not collide in practice, and nothing checks at write time because no shared database exists. `find_id_clashes(packages)` checks across packages (registration and the index build use it): an id two packages both own is reported with its table and both packages, except the home `dataset` and `datastore` rows (shared by the packages of one dataset) and the rows a package lists in `anchor_row` (another package's own ids).
- **Not here yet:** the catalogue, `register`, templates and registrations as packages, and `split`/`join` of an existing dataset (later steps of the same spec). Schema stays at 50 (`model.sql` is untouched in this step).
- **Tests:** `test_subject_package` (synthetic: one folder, no `small_files.sqlite`, digest of every key, the hash stable under a no-op writer and changed by a write, a flipped byte in a shard named by key, a copy hashing equal, recovery of a pending node, a second writer refused, two packages written at once, versions refused, id clashes and anchors, `Sha256` in pieces equal to whole).

## The dataset module: a catalogue of subject packages (`io/include/nxr/io/catalogue.h`, `participants.h`, `subject_filter.h`, `dataset_index.h`; spec `docs/specs/2026-11-16-subject-centred-datasets.md` sections 3, 5 and 6, phase C)

A dataset is a folder at any path, addressed by that path (there is no datastore and no `catalog.json`). It holds `catalogue.sqlite`, optionally `participants.sqlite`, a deletable `index.sqlite`, group packages and, by default, its subjects' packages. **The dependency goes one way:** this module reads subject packages; a package never reads it. Library functions only; the command line wires them separately.

- **The catalogue** (`create_catalogue`, `Catalogue::open`): a database in the model's schema holding the dataset's own `dataset` row, one `subject` row per member (a copy of the package's own row, its `path` relative to the dataset folder or absolute for a package elsewhere), and three tables of its own: `subject_entry` (schema version, store format, **content hash**, registered time and by whom, `status` registered, missing, stale or replaced, and the counts of the model's `subject_status` view, so a subject list opens no package), `template_reference` (name, edition, version, the template's subject id and hash, a location hint) and `vocabulary` (the shared vocabulary's digest by key). It holds no manifold, field, selection, operator or participant row. `Catalogue::package_options()` gives `create_package` the dataset as the package's home.
- **`register_packages`** (idempotent; no array is read or moved): per package, the versions are known; no writer lock; no pending node; the record's content hash is the hash of the rows and digests as they are; its ids clash with no other member's and no other candidate's in the call (`find_id_clashes`, one pass); its shared vocabulary (app- and dataset-scope dictionaries) has the entries the dataset's first copy has (a differing copy is refused by name, never merged); the template it names is the one the dataset's other subjects use (name, edition, version, hash). Outcomes per package: `registered`; `unchanged` (same id, same hash: a rerun changes nothing); `updated` (same id, new hash: the package is the authority on its own content); `replaced` (another id under a held name or path, only with `replace`: the old entry becomes `replaced` and its package is moved to `<folder>.replaced-<date>`, never deleted); `refused` with the reason, naming the package and the other party. A failed package registers nothing and does not stop the others. A package registered under another path with the same id as an entry whose package is still in place is refused (a copy). `scan_packages` lists the `*.nxr.zarr` folders directly inside the dataset; `refresh_member_status` marks entries `missing` (package gone) or `stale` (hash differs).
- **Participants** (`participants_import`, `participants_export`, `participant_variables`, `participant_values`): `participants.sqlite`, its own file, holds `variable_dictionary` and `subject_variable` with the model's columns (the foreign keys to rows of another file dropped). Import reads BIDS `participants.tsv` (a `participant_id` column matched to subject names, `n/a` missing) and the `participants.json` sidecar (`Description`, `Units`, `Levels`, and as an extension `Type`: number, string or boolean; without it a variable is a number when every value is one). All or nothing: an unknown participant, a value of the wrong type or outside its levels is refused with its row. A missing value is a missing row. Export writes only to the paths it is given. The catalogue, the packages and the index never carry participants.
- **Selecting subjects** (`select_subjects`): exactly one of an explicit list (names or ids) or a `where` filter; neither or both is refused; there is no "all subjects" default; a selection matching nothing, an unknown name, a member that is not `registered`, whose package is missing, or whose package differs from its entry (`stale`), and a template or group among the members are refused. The result carries each member's name, id and content hash and, for a filter, its text and the language version, as `provenance()`.
- **The filter language, version 1** (`SubjectFilter`, `kFilterLanguageVersion`): comparisons `== != < <= > >=`, `in (...)`, `is missing`, `and` / `or` / `not`, parentheses; a variable with spaces is written between backticks; strings in single quotes (`''` is a quote). `not` binds tightest, then `and`, then `or`. It is parsed, type-checked and evaluated in memory over the participant variables only (never SQL, never another table). Refused before anything is read: an unknown variable (naming those that exist, or that there is no participants table), a type mismatch (a number variable takes numbers; a string variable takes strings with `==`, `!=`, `in`; a boolean takes `true`/`false` with `==`, `!=`), a value outside a categorical variable's levels (naming them), and syntax errors with their position (`=` alone gets a hint to write `==`). **Missing values follow SQL's three-valued logic**: a comparison or `in` on a missing value is unknown, `not unknown` is unknown, `unknown and false` is false, `unknown or true` is true; a subject is selected only when the whole filter is true; `is missing` is the one test true on a missing value. A new construct is a new language version.
- **The index** (`build_index`, `index_is_current`): `index.sqlite`, a model-schema database of every registered member's rows (the shape of a dataset database), built from the packages and never written otherwise. Its `meta.index_json` records the content hash of each package it was built from; it is stale when a member's hash differs or a member is new or gone, and rebuilt (beside the old file, then renamed over it). The home dataset and datastore rows become the catalogue's one; a shared vocabulary dictionary is matched across packages by `(scope, name)`, the first copy kept and references remapped; two copies with different entries are a conflict and a duplicate id a clash: the build reports each naming the packages and writes nothing. A package at an unknown version refuses the build. Deleting the index loses nothing.
- **Group packages** (`create_group_package`, `group_member_status`): a package of kind `group`, registered like any member, whose record lists the subjects it was computed from with each one's content hash. `group_member_status` says which members still are as the group saw them. (The group fields themselves are written by the group analysis, a later step.)
- **Tests:** `test_subject_filter` (the spec's examples, precedence, three-valued logic, every refusal); `test_catalogue` (create, register: registered, unchanged, updated, replaced, refusals for a held lock, a package changed outside a writer, a name, an id clash, a vocabulary difference; participants import and export; selection; the index as a cache; groups; status).

## Migration: `split` and `join` (`io/include/nxr/io/migrate.h`; spec `docs/specs/2026-11-16-subject-centred-datasets.md` section 8, phase E)

An existing dataset (one `dataset.sqlite`, store formats 50 to 52) becomes subject packages with a catalogue, and back, **in place**, under the dataset's writer lock, **subject by subject**, verified before anything is marked done, **resumable and reversible**; the old database is kept, renamed, never deleted. It is one explicit call per dataset (never on first use): a build that knows format 53 still reads 50 to 52. Nothing here has been run on a real dataset.

- **`split_dataset`:** (1) a dataset below 52 is brought to 52 by `compact_dataset` first. (2) The catalogue is made carrying the dataset's own `dataset` and `datastore` rows (the ids stay). (3) **Templates first**: each subject of kind `template` becomes a package; the default subject is referenced as `<dataset>-default`, edition `full`, version `local`, with its content hash, and every subject whose `template_id` names it records that reference. It is not released into a shared templates folder; replacing it with a released template of the same anatomy is a separate step. (4) **Each subject** is built in its own folder as `subject.sqlite.building`: the rows it **owns** (every row with its `subject_id`; the children of its rows; field contributions follow their field; the home dataset and datastore rows; **every app-, dataset- and user-scope dictionary with its entries, ids unchanged**, so a shared vocabulary is the same row in every package), the **anchor rows** its rows point at that another subject owns (a template's manifolds, a group's members and source fields) with their parents and what a bundle follows (a topology's cells, a geometry's dimensions, a dictionary's entries, a manifold's components), listed in `anchor_row`; its small keys from `small_files.sqlite` into the package's `small_files` table; `key_digest` of every key (shards included); the record and the content hash. **Verified before it is committed:** no foreign-key violation; every row of the package is the original's, value for value; the bundle read from the package equals, exactly, the bundle read from the original database; the package's keys are the subject's keys and each small key's bytes are unchanged (no shard moves). Then `subject.sqlite.building` is renamed `subject.sqlite`, the old `small_files.sqlite` is kept as `small_files.sqlite.format52`, and the package is registered in the catalogue (that is the subject's commit). (5) **The whole**: the index is built from the packages and compared with the original database table by table, row for row **in both directions**: a row of the original that no package holds, or a value that differs, refuses the end (nothing is renamed). Participant rows, if the original ever held any, move to `participants.sqlite`. (6) The original is renamed `dataset.sqlite.format52`.
- **Resumable:** a crash leaves the old dataset whole or a subject finished; a half-built `*.building` is removed and started again; a package committed before its registration is checked (its content hash is its record's) and registered; a second run on a finished dataset changes no byte. Hooks at the stages `built`, `verified`, `committed`, `registered` let a test stop there.
- **`join_dataset`:** the index build is written as `dataset.sqlite` (the original's `meta` when `dataset.sqlite.format52` is kept; participants back into the model's two tables), committed first; then each subject's small keys go back into `small_files.sqlite` (compared byte for byte with the package's table) and its `subject.sqlite` is renamed `subject.sqlite.format53`; last the catalogue and participants file are renamed `*.format53` and the index (a cache) is removed. Names kept aside are never overwritten (`-2`, `-3`, ...). `split` then `join` gives back the original database table by table and every key's bytes (`test_migrate`), and the joined dataset opens as a writer and passes `text_parity`.
- **Left for a later, explicit step:** group fields that `run roll_up` wrote into the default subject stay in the template's package (moving them changes the rows' owner, which a verified row-for-row migration cannot do); `dataset retemplate`, which replaces a dataset-local template with a released one.
- **Cost:** `key_digest` reads every key once (shards included), and registering a package checks its ids against every member's.
- **Tests:** `test_migrate` (a synthetic dataset with a template, two subjects registered to it with anchors, a group subject and large shards; a dry run; the report; every package verified and its bundle the original's; keys unchanged; the index equal to the original in both directions; a second run changing no byte; a crash at each of four stages then the same command again giving the same package hashes as an uninterrupted run; `join` and a second `split`; a format 51 dataset brought to 52 first).

## Templates as packages (`io/include/nxr/io/templates.h`; spec `docs/specs/2026-11-16-subject-centred-datasets.md` section 4, phase D)

The anatomy every subject is measured against is a subject package of kind `template`, opened exactly as any subject is, shared and versioned instead of rebuilt in every dataset. Library functions only.

- **Origin.** Every item of a template (a manifold, field, operator, selection or statistic with a path) carries an **origin record** in the package's `origin` table (`set_origin`, `origin_of`): `source`, `licence`, `attribution` and `redistributable` (whether its licence lets it be redistributed). `package_items` lists the items; `items_without_origin` those that have none. (The origin is stored beside the model's rows, in a table of the package format; `model.sql` is unchanged.)
- **Release and freezing** (`release_package`): copies a `template` (or `registration`) package into a templates folder as **`<name>@<version>.nxr.zarr`** and writes the release (name, edition, version, who, when, program) into its record. It refuses a package with an item that has no origin record (naming the items), a package whose rows no longer hash to its record, and a name and version already released with another content hash (a released version is frozen: a change is a new version, a new package). The same release again is a no-op. **The content hash does not change by being released**: it is what every reference names. A released package is frozen: `open_package_writer` refuses it (`not_writable`). The working package is left as it was. (A new version is built again from its sources; a deliberate fork of a package under a new id, `subject fork`, is not built yet.)
- **Two editions** (`release_redistributable`): `NAME@VERSION` is the **full** edition; `NAME@VERSION:redistributable` keeps only the items whose origin says `redistributable`, **byte for byte the full edition's** (the function compares the key digests of every kept item and refuses a difference), lists what it left out (the dropped items and whatever depended on them, which the model's cascade removes), has its own content hash and records the full edition's in `release.derived_from`. As a folder name it is `<name>@<version>+redistributable.nxr.zarr` (':' is not portable in a file name; this spelling is ours, not the spec's).
- **The template search path** (`template_search_path`, `resolve_template`, `list_templates`): in order, the locations the command names (`TemplateSearch::command_paths`, the `--templates` of a command), the environment variable **`NXR_TEMPLATE_PATH`** (a list separated as `PATH` is: `:` on Linux and macOS, `;` on Windows), the user setting **`template_path`** (a JSON list) in `$XDG_CONFIG_HOME/nxr/settings.json` (`~/.config/nxr/settings.json`) on Linux, `~/Library/Application Support/nxr/settings.json` on macOS, `%APPDATA%\nxr\settings.json` on Windows (`user_settings_file`), then the location hint recorded in the reference. **A reference names name, edition, version and content hash**; the first released package found whose hash is the reference's is used; one with the right name and version and another hash is skipped and reported with both hashes; none found is refused, listing every location searched. The path says only where to look: which template is used is fixed by the reference, so no setting chooses data.
- **A subject and its template** (`register_to_template`, `record_registration`): the subject package gets the **anchor rows** of the template its rows reach (`add_anchor_rows`: the template's `subject` row and the manifolds named, with their parents and what a bundle follows; listed in `anchor_row`; no array is copied), `subject.template_id` the template's subject, and the record's `template_reference` (name, edition, version, hash, subject id). So **a subject's bundle is complete without opening its template** (`test_templates` reads the template's surface from the package's bundle); the template package is opened only to read its arrays. `anchor_differences` says which anchor rows differ from the template's own. `record_registration` records the registration between templates the subject's carried nodes use. A subject is moved to another version only by registering it again to that version's template, at the caller's request.
- **Registering in a catalogue** (`RegisterOptions::templates`): when given, a package that names a template must find it by name, edition, version and hash, and every anchor row must equal the template's, value for value. `select_subjects` refuses members registered to different template versions (a cohort is never pooled across them), naming each version.
- **A registration between two templates** (`create_registration_package`): a package of kind `registration` (the kind is added to the package's schema in place, additively with `user_version` unchanged, as a value type is; the spec's schema 51 bump is left for the three implementations to make together). It anchors the two templates' manifolds, lets the caller write the resampling operators (each `from_manifold_id` on one template and `to_manifold_id` on the other, one per direction), records both templates (`template_reference.from` and `.to`), the method and what it cannot resolve (`description.method`, `description.warnings`) and gives every item the origin it was told; then `release_package` versions it like a template, and it is found on the search path as `<name>@<version>.nxr.zarr`. **Not built here:** the computation of the operators (`registration create --method`: the resampling matrices on the registered spheres) is a compute function composed by a pipeline and is left to a later step; `test_templates` writes permutation operators by hand and applies them back and forth.
- **The datastore and `catalog.json` are retired:** nothing of format 53 reads or writes them (`test_templates` checks that no `catalog.json` is made). `create_dataset`, `read_catalog`, `write_catalog` and `list_datasets` stay only to read and migrate a dataset of formats 50 to 52.
- **Tests:** `test_templates` (names; refusal without an origin; release, the same hash, frozen, the same release again; the search path in its order, with a decoy of the same name and another hash, the environment, the settings file, the hint; the redistributable edition and its kept items equal; a subject's anchors, record and bundle; the catalogue's anchor check and the version-pooling refusal; a registration between two templates released and found, its operators applied by hand forward and back).

## Template importers (`io/convert/`, `io/include/nxr/convert/templates.h`; spec `docs/specs/2026-11-16-subject-centred-datasets.md` section 4)

`create template --recipe fsaverage|icbm152 --sources DIR --package P [--name N] [--surface S ...] [--parcellation NAME ...] [--cortex-mask] [--hcp-trk FILE --hcp-end-surface S [--max-end-distance MM] [--hcp-points N]] [--accept-unlisted-sources]` builds a template package natively (no MATLAB or Brainstorm at run time) and `release template` / `release redistributable` freeze it and make its two editions.

- **fsaverage** (FreeSurfer subject folder): `cortex_pial` (primary), `cortex_white`, `cortex_inflated`, the registered sphere `cortex_pial_sphere` (`sphere.reg`), hemispheres as components; `--parcellation aparc` (Desikan-Killiany), `aparc.a2009s` (Destrieux) or any `<hemi>.<name>.annot`; `--cortex-mask`.
- **icbm152** (Brainstorm's default-anatomy folder): the `tess_cortex_*.mat` surfaces named by `--surface` (default `tess_cortex_pial_high.mat`) with their registered spheres (`<surface>_sphere`); `--parcellation` names atlases of the surface file (none is written unless named). `Structures` (the hemispheres) is always the surface's own.
- **HCP-1065** (`--hcp-trk`, icbm152 only): the streamlines as the curve manifold `hcp1065` (SCS metres, `--hcp-points` per streamline at equal arc length, both ends exact), each end's distance to `--hcp-end-surface` (`end_distance_mm`, stored for every end), and the operator `hcp1065_ends` (csr) holding the ends within `--max-end-distance` (default 5 mm). The MNI-to-anatomy transform is read from the anatomy's `subjectimage_T1.mat` (its NCS and SCS frames). Origin: CC BY-SA 4.0 together with the WU-Minn HCP Open Access Data Use Terms; cite Yeh 2022, Nat Commun 13:4933.
- **The scalp and the landmarks** (`--scalp`, `BuildRequest::scalp`; for the standard electrode montages, audit `docs/audits/2026-11-17-sleep-events-and-montages.md` section 4.3): the head surface as the surface `head` (a scalp for PLACEMENT, not a boundary-element model: no skull layers are imported) and the landmarks nasion, LPA and RPA as the points manifold `head_landmarks` (names as a labelling), in the template's own coordinate system, metres. icbm152: `tess_head.mat` (SCS metres) and `subjectimage_T1.mat` (`SCS.NAS/LPA/RPA` through its SCS frame, `scs_mm = R (voxel * Voxsize) + T`). fsaverage: MNE-Python's `bem/outer_skin.surf` (the scalp MNE derives from fsaverage's MRI; FreeSurfer's terms) and `bem/fsaverage-fiducials.fif` (`nxr::formats::fif::read_fiducials`: one isotrak block, refused unless in FIF frame 5, the MRI's surface RAS, metres). **No template carries an inion**: it is added with the montage (`nxr::meeg::inionFromLandmarks` is the default). The manifest lists `tess_head.mat` of the icbm152 recipe with its checksum; the fsaverage files and the ICBM152_2023b `tess_head.mat` are not yet listed (their checksums are to be added from the downloaded files), so they need `--accept-unlisted-sources` and the checksum read goes into the origin.
- **Coordinate system of fsaverage (G13, first part)**: the fsaverage surfaces are written in `freesurfer-tkr` (metres, FreeSurfer's surface RAS) and no longer in the unit-less `fsaverage` key. The default subject (`io/src/default_subject.cpp`) and the `mni` to `scs` transition of the streamlines are G13's later parts and are not changed here.
- **Origin of every item**, the source manifest and the refusals: see `io/convert/README.md`. An atlas's terms come from a table the importer carries (`atlas_terms`); an unknown atlas is `not confirmed` and not redistributable.
- **What it cannot resolve**: an end's vertex is a location on the chosen surface, not the depth at which a fibre ends; the streamlines are one random realisation of the tracking.
- **Tests**: `test_template_import` (synthetic octahedra as a FreeSurfer subject and as a Brainstorm default anatomy; a tiny `.trk`): each importer, the origin on every item, parcellations only when named, the manifest refusals, the redistributable edition leaving out Hammers, LPBA40 and unconfirmed atlases, the `.trk` reader against numbers made by nibabel from the same bytes (four header orders), the MNI frame, and the distance gate.

## Registration between templates (`compute/include/nxr/sphere_registration.h`, `pipelines/include/nxr/pipelines/registration.h`; spec `docs/specs/2026-11-16-subject-centred-datasets.md` section 4.4)

`create registration` carries a field from one template's surface to another's through their **registered spheres** (FreeSurfer's `sphere.reg`, or any sphere on which two surfaces are put in correspondence), and stores the result as a package of kind `registration`.
It is a graph of one pure node: `nxr::operators::registration::resample_operator` (matrices in, a sparse matrix out, no file and no subject), prepared by `registration_create` (finds both released templates by name, edition, version and hash on the template search path, reads their spheres, checks them) and stored by io through `create_registration_package`.

- **Methods** (named options; the default is the reference's): `shepard` (default; the `--neighbours` K = 8 nearest source vertices on the unit-normalised spheres, the modified Shepard weights of Franke and Nielson 1980: weight k = ((d_K - d_k) / (d_K d_k))^2 for k < K with d the squared chord distance, normalised to sum 1; the K-th neighbour sets the radius and takes no weight; a zero distance is replaced by the smallest positive number of the type; a row whose weights are all 0 gives its nearest neighbour weight 1), `nearest` (the nearest source vertex), `barycentric` (the three vertices of the source triangle that holds the point; a point on no triangle of its nearest vertices takes its nearest vertex, and the number that did is reported).
- **Pieces** (a cortex's two hemispheres, as runs of vertices, left then right) are found from the faces and matched one to one in order: a destination vertex looks only among the source vertices of its own piece. Spheres with a different number of pieces are refused.
- **Stored**: one operator per direction (`<from>_to_<to>`, kind `resampling`, CSR [destination x source], `from_manifold_id` on the source template and `to_manifold_id` on the destination, both anchored in the package), unless `--one-direction`. The package record holds both templates (name, edition, version, hash), the method and `warnings`; each operator's `producer_json` the method, K, the empty columns (source vertices no destination takes from) and the largest chord from a destination point to its nearest source vertex.
- **Origin**: derived from the two spheres' own origin records (the licences and attributions joined; redistributable only when both are), unless the caller gives `--licence`, `--attribution`, `--source` or `--redistributable`.
- **What it cannot resolve**: the correspondence is the spheres', not anatomy's; every method smooths (a value carried is a mixture of source values), so a field carried there and back is not the field; where the destination is the finer surface, a source vertex may take part in no row.
- **Tests**: `test_sphere_registration` (compute; synthetic icospheres: the rows sum to 1, a linear field is carried within a stated error by each method, the identity on one sphere, two pieces kept apart, float32, the refusals) and `test_registration` (pipelines; two released synthetic templates, the stored operators equal the compute node's, the derived origin, the refusals, a frozen release).

## The structural connectome's pure functions and nodes (`compute/include/nxr/connectome.h`, `nearest.h`, `congruence.h`, `pipelines/src/connectome_steps.cpp`; spec `docs/specs/2026-11-16-structural-connectome.md` section 4)

The pure part of `compute connectome`. The pipeline definition and the command come with the runner's declared ports; until then the nodes are called alone (`dispatch`) or from a test.

- **Pure functions** (`nxr::meeg`, `nxr::geometry::nearest`, `nxr::core`; float32 and float64, never a silent cast): `fiberEnds`, `nearestVertices` (a k-d tree; equal to the brute force, the lowest index on a tie), `withinGate` (both ends within the distance), `registrationMatch` + `carryEnds` (Shepard weights; **a template vertex with no subject neighbour goes to its nearest subject vertex on the registration sphere, within its piece, never to vertex 0**), `smoothingStep`/`smoothingOperator`, `memberships` + `smoothMemberships`, `partitionCounts`, `endConnectome`, `connectionWeights` (`sparseCongruence`: smoothing · counts · smoothingᵀ, diagonal removed), `pruneBelow`, `componentsOf` + `largestComponent`.
- **Counting over a partition is a measurement**: the fibers are the elements, the region pairs the members. The smoothing is pushed onto the memberships as its ADJOINT (the transpose of the mesh's smoothing step, applied `hops` times to the one-hot memberships), which makes the counts equal the smoothed vertex matrix (smoothing · counts · smoothingᵀ) summed over the region pairs exactly (the rows of the smoothing sum to 1, its columns do not on an irregular mesh). A fiber with both ends in one region adds 2 at that region's diagonal (both orders, as the vertex matrix has them): the within-region count is stored. The vertex matrix drops its diagonal; a partition does not need to.
- **Nodes** (registered, kind `setup`, no time axis; sparse values travel as COO arrays `[nnz x 3]` with meta `{form: coo, shape}`, precision as meta `precision`): `meeg.fiber_ends`, `meeg.nearest_vertex`, `meeg.end_gate` (`max_end_distance_mm`, default 5, `none` = no gate; the number left out in the output's meta), `meeg.registration_match` (`neighbours`, default 8), `meeg.carry_ends`, `meeg.smooth_memberships` (`hops`, default 3; `regions`), `meeg.partition_counts`, `meeg.end_connectome` (`n_vertices`), `meeg.connection_weights` (`smooth_hops`, default 3), `meeg.prune_below` (`meeg.prune_below`, default 1e-4), `largest_component`.
- **What it cannot resolve**: tractography ends are not the fibers' cortical terminations (streamlines stop at the grey/white boundary); an end's vertex is a location on the chosen surface, not a depth; smoothing spreads ends over the mesh and is not conserving; a template's fibers on a subject's cortex carry the template's anatomy through a spherical correspondence, not the subject's own white matter. The gate measures an end's distance to the end surface before any registration and does not gate the registration's own displacement.
- **Definitions** (`connectome` and `connectome_vertices` run; `connectome_template` is drafted in `pipelines/definitions/pending/`, waiting for the runner's `registration` input): `connectome` (own fibers, counts over a partition: `--fibers`, `--end-surface`, `--over` required, `--smooth-hops` default 0, `--max-end-distance` default 5 mm or `none`; the counts a `measurement` output with two codes a row, within-region pairs stored), `connectome_template` (a template's fibers through the registration; the bundle's `origin` and licence, HCP-1065 CC BY-SA 4.0 with the WU-Minn HCP Data Use Terms, are carried into the result: ShareAlike, with attribution) and `connectome_vertices` (the materialised vertex matrix, smoothing default 3, pruning and the largest component, only here). They are separate because a definition has no conditional steps. The runner now has the input classes `curve` and `surface`, a required-parameter default, an output of kind `measurement` with `codes: 2` (a sparse matrix over a selection's members, a row a pair), a sparse `field` on a product manifold (`on: surface x surface`) and `carry_origin`; the fibers and surface keep their stored precision (a mix is refused). `connectome_template` still waits for the `registration` input. The 11 nodes have signatures (`make` over constant arrays). `test_connectome_definition` checks each draft against the registered nodes' ports and parameters and the signatures.
- **Tests**: `test_connectome` (one test a function; the vertex-level fiber counts, connection weights, pruning and components against an independent dense implementation of the reference's behaviour on synthetic fibers and a synthetic sphere, gate none), `test_connectome_partition` (region results against a direct sum over the partition at 0, 1 and 3 hops, a roll-up, the order of the fibers), `test_connectome_gate`, `test_connectome_registration` (the quirk fix against a brute-force search), `test_connectome_nodes` (pipelines; each node through the dispatch point). `test_connectome_golden` (against the maintainers' MATLAB `tess_connectome` on the same synthetic sphere and fibers: `compute/test/golden/connectome_golden.txt`, made once by `gen_connectome_golden.m`, which only calls the reference; fiber counts identical, smoothed weights, pruning and the largest component at 0, 1 and 3 hops and EdgeThr 0, 1e-4, 0.05).

## Standard electrode montages: the 10-05 rule (`compute/include/nxr/montage.h`, node `meeg.eeg_montage_1005` in `pipelines/src/montage_steps.cpp`; audit `docs/audits/2026-11-17-sleep-events-and-montages.md` section 4)

- **Why**: a recording without digitised positions can only be placed on a head by the standard system's names. The systems are RULES, not data: positions at fixed fractions of arcs between the nasion, the inion and the preauricular points. nxr-neuro computes them on a template's own scalp, so no third-party position table is shipped.
- **Method and reference (default, 2026-11-17)**: Oostenveld R, Praamstra P (2001) The five percent electrode system for high-resolution EEG and ERP measurements, *Clin Neurophysiol* 112:713-719 (after Jasper 1958 and Chatrian et al. 1985; the names and the 10-10 subset follow Jurcak V, Tsuzuki D, Dan I 2007, *NeuroImage* 34:1600-1611). The code is written from the papers. The reference the default is held equal to is MNE-Python's `spherical_1005` (BSD-3; the `eeg_positions` package computes it): on a sphere both give the same positions. The golden is MNE's output (`compute/test/golden/montage_golden.txt`, made by `gen_montage_golden.py`, which only calls MNE).
- **The rule** (`nxr::meeg::tenFive(vertices, faces, landmarks)`): every contour is the scalp mesh cut by the plane through three points, and its points are at fixed fractions of its length. (1) The midline Nz - Cz - Iz at 5% steps; Cz is the point at 50% of both the midline and the coronal contour LPA - Cz - RPA (found by alternating the two, to a stated tolerance; refused if it does not settle). (2) Nz - T10 - Iz and Nz - T9 - Iz (T9 = LPA, T10 = RPA) and the coronal T9 - Cz - T10, at 5% steps. (3) NFpz - T9h - OIz and NFpz - T10h - OIz, and Fpz - T7 - Oz and Fpz - T8 - Oz. (4) The 14 coronal rows from AFp to POO, at 1/16 steps. A point an earlier contour made is not moved by a later one. The four landmarks are the caller's (the inion above all: it is a landmark, not a measurement); each moves to its nearest point of the scalp and the shift is reported. The handedness is read from the landmarks.
- **The inion**: the rule needs it as a landmark. `inionFromLandmarks` is a DEFAULT for a template whose scalp has none stated: the farthest point where the line from the nasion through the midpoint of the preauricular points crosses the scalp. It is a stated convention, not the bump the inion is (the audit recommends the inion at its standard MNI position; the published coordinates were not available to check, so the maintainers confirm or give them); a caller who knows the inion passes it.
- **Names**: 345 positions (Nz and Iz included). The 10-10 (77 names) and the 10-20 (21) are subsets of them, not separate montages. The old 10-20 names are aliases (T3 = T7, T4 = T8, T5 = P7, T6 = P8). The 10-10 list is the standard's (it includes AF3, AF4, PO3, PO4, TP9, TP10, which MNE's `spherical_1010` leaves out: the maintainers confirm this choice).
- **Node** `meeg.eeg_montage_1005` (kind `setup`; inputs `scalp_vertices` [V x 3], `scalp_faces` [F x 3] (0-based indices), `landmarks` [4 x 3] (nasion, inion, LPA, RPA); output `positions` [345 x 3], row k named `meta.labels[k]`, with `meta.systems`, `meta.old_names`, `cz_iterations`, `cz_residual`, `landmark_shift` and `warnings`). Parameters: `tolerance` (when Cz stops moving, a fraction of the nasion - inion distance, default 1e-7), `max_iterations` (a count, default 100). Precision follows the input (the arcs are computed in double).
- **`import recording --montage NAME`** (`RecordingImport::montage`): matches each channel's name to the labels of a montage already stored in the subject (a points manifold with its names labelling, as `run montage` makes it), EXACTLY (the case as written; no prefix or suffix is stripped, nothing is guessed); the montage's recorded aliases (T3 = T7) count and are marked `via: alias`. The montage, the matched channels and the channels no label matches are recorded in the recording's producer (`montage`) and listed in the result (`montage.matched`, `montage.unmatched`). A montage the subject does not hold is refused before anything is written. Test: `test_import_recording`.
- **`run montage`** (the definition `pipelines/definitions/montage.yaml`, since #115; it replaced the hand-wired `montage_create`): `nxr-neuro run montage --dataset PACKAGE --subject S [--scalp head] [--landmarks head_landmarks] [--name eeg_1005]` (the parameter `inion` gives the inion). The graph: the scalp -> `geometry.surface_parts` (vertices, faces); with the landmarks (the points NAS, LPA, RPA, in the scalp's coordinate system, refused otherwise) -> `meeg.inion_from_landmarks` (the inion by `nxr::meeg::inionFromLandmarks`, or the caller's `inion`; the convention and the point are in the producer) -> `meeg.eeg_montage_1005` -> `meeg.montage_labels` (names, system). Stored: the points manifold `<name>` (positions [345 x 3] in the scalp's coordinate system, aliases and counts in its producer), its names `<name>_names` (one level per point) and the partition `<name>_system` (levels 10-20, 10-10, 10-05: the narrowest system a point belongs to). The licence of the scalp and the landmarks is not carried to the montage by the task API (issue filed).
- **What it cannot resolve**: the system's positions on this head model, not where a cap sits on a person (the cap's placement, the hair, the head's shape under it); arcs are lengths along a triangle mesh, which shortens them by a term that goes as the square of the edge length; a scalp that is open (a neck cut) is fine while the contours stay on it. Not built: the second named option of the audit, `placement: sphere_projection`.
- **Tests**: `test_montage` (compute: the names and subsets against MNE's lists; a moved, scaled sphere against MNE's positions after a similarity alignment (Umeyama), the error reported in mm (icosphere level 5, 5.9 mm edge: mean 0.007 mm, max 0.02 mm); an ellipsoid whose midline points are at 5% steps of the ellipse's own arclength, with Cz at its top, mirror images and points on the surface; the refusals) and `test_montage_nodes` (pipelines: equal to the function, precision, parameters and inputs refused, traits declared).

## Fiber bundles in packages (`io/include/nxr/io/fibers.h`, `io/include/nxr/convert/fibers.h`; spec `docs/specs/2026-11-16-structural-connectome.md` sections 3, 4.1, 6 and 7)

The fiber support of `compute connectome`: storing and reading a bundle and its ends, and importing a `.trk`.

- **Stored** (io, through the task API): a bundle is a `curve` manifold `<name>` at `fibre/<name>` (fiber-major points, SCS metres, float32 or float64 as given; `producer_json` holds `points_per_component`, `precision` and, for a bundle that came from somewhere, its `origin`: source, citation, licence, terms). The ends on an end surface, per bundle and surface: the operator `<bundle>_ends_<surface>` (kind `fibre ends`, CSR [surface vertices x bundle points]: each fiber's first and last point on the nearest surface vertex; it does not depend on any gate) and the array `<curve>/end_distance_<surface>` ([2nF], metres, the fibers' precision; fiber f's first end at 2f, its last at 2f + 1). `derived_origin` copies a bundle's `origin` into what is derived from it (licence, attribution, and the ShareAlike notice), so a result made from the template's HCP-1065 bundle carries CC BY-SA 4.0 together with the WU-Minn HCP Data Use Terms.
- **Imported**: `nxr::convert::import_trk` reads a DSI Studio / TrackVis `.trk` (the reader of the template importers), applies the caller's RAS-mm-to-SCS-metres transform (a choice of data: no default), resamples each streamline to a fixed number of points at equal arc length (ends exact) and stores the bundle with the origin the caller gives. The template importer writes the HCP-1065 bundle with its `origin`. `nxr::convert::import_trx` does the same for a TRX archive (`io/formats` `trx.h` over `zip.h`: stored and deflate entries, ZIP64, CRC-32 checked; positions float16, float32 or float64 in RAS mm, offsets N or N + 1; the archive's data per vertex, per streamline and groups are named in `not_imported`, not read). The ZIP container is read with zlib's raw deflate, already linked: no zip library.
- **Tests**: `test_fibers` (io; synthetic octahedron and bundles: float32 and float64 stored and read back equal, the ends and distances in the stored order, the origin carried to a derived result, the refusals, a `.trk` and a `.trx` (float32 and float16) imported with a caller's transform and resampled), `test_formats_zip` and `test_formats_trx` (archives built byte by byte and by the writer: every half-precision value, both offset conventions, ZIP64, CRC and size checks, the refusals).

## The commands over subject packages (`cli/command_table.cpp`, `cli/package_command.cpp`)

In the verb-first grammar, each with its own `--help`; they parse, call the functions above and print (JSON lines, or the readable form in a terminal), and compose nothing the library does not. A dataset is addressed by its own path (`--dataset PATH`).

| command | what it does |
|---|---|
| `create dataset --dataset D [--name N]` | **the default form**: a dataset as a catalogue of subject packages, in a folder of its own. `create dataset --datastore DIR --name N` (the old spelling) still makes the one-database dataset DIR/N, with a note on stderr; `split dataset` turns it into a catalogue. (`create catalogue`, made on 2026-11-16 and never released, is removed.) |
| `create subject --dataset D --name S` | on a catalogue: a new package `D/S.nxr.zarr`, registered |
| `create package --package P --name S --home-dataset N [--kind subject\|template\|group]` | a subject package |
| `verify package --package P`, `show package --package P` | recompute every digest and the content hash; show the record |
| `register package --dataset D (--package P ... \| --scan) --registered-by WHO [--replace] [--templates DIR ...]` | register (idempotent; each outcome and refusal reported) |
| `list members --dataset D` | the packages with their state (registered, missing, stale) |
| `list subjects --dataset D [--subjects A,B \| --where FILTER]` | on a catalogue: the members, or those selected (the matched subjects, their hashes and the filter text with its language version are printed); on a one-database dataset as before |
| `import participants --dataset D --tsv FILE [--sidecar FILE]`, `export participants ...` | BIDS participants in and out of the dataset's own participants file; written only where named |
| `create index --dataset D [--force]` | the cross-subject index (a cache) |
| `split dataset --dataset D --registered-by WHO [--dry-run]`, `join dataset --dataset D` | the migration |
| `release template --package P --name N --template-version V --released-by WHO --into DIR`, `release redistributable --package P --into DIR --released-by WHO` | freeze a template or registration; its redistributable edition |
| `create registration --package P --name N --from NAME@VERSION --to NAME@VERSION --from-sphere S --to-sphere S [--method shepard\|nearest\|barycentric] [--neighbours K] [--one-direction] [--templates DIR ...]` | the registration between two released templates through their registered spheres, as a `registration` package (see "Registration between templates"); `release template` freezes it |
| `list templates [--templates DIR ...]` | the released templates on the search path (the `--templates` locations, then `NXR_TEMPLATE_PATH`, then the user setting `template_path`) |

**Computing on packages (`cli/cli.h` `open_dataset`; `io::open_subject_package`).** Every command that names `--dataset` and `--subject` opens the subject through the dataset: on a catalogue it finds the subject's package in the catalogue and opens it as the writer (the package's own writer lock; a read opens it without one), so everything a `compute` verb stores goes into that package and no other file is written; on a one-database dataset (`dataset.sqlite`, store formats 50 to 52) it opens as it always did, with a note on stderr that `split dataset` makes it a catalogue. The pipelines are unchanged: they take the writer and the subject, and a package writer is a `DatasetWriter` in package mode (`node_store()`). The result of a chain on a package equals the same chain on the one-database form byte for byte (`test_package_pipelines`). A subject registered to a template reads the template's arrays from the template's package, read-only, found when the first one is read: the catalogue's own members first (a dataset-local template, as `split` makes), then the template search path (`--templates DIR ...`, repeatable, on every such command; then `NXR_TEMPLATE_PATH`, then the user setting `template_path`). The commands that make or change a one-database dataset itself (`import subject`, `import template`, `import group`, `compact dataset`, `recover dataset`, `delete subject`) refuse a catalogue and say so. A replaced result is moved aside inside the package (`_replaced-<date>/`), never deleted. **Group results** (`reduce pyramid --dataset D --subjects A,B,C --into NAME`, `pyramid_reduce_group`; spec section 5): on a catalogue the members are read each from its own package (read only, no lock) and the sums are written into the **group package** `NAME` of the dataset, made if absent, else its group fields are replaced (one package per analysis the caller names, never a default one; `--into` is required). The group package holds the default subject's rows the results are on (its manifolds, selections, canonical frames) and the members' subject and source-field rows the contributions name as anchor rows; no array is copied, the default subject's arrays are read from its own package. Members registered to different template versions are refused (never pooled), as are the sums' own refusals (`run roll_up` above). The result equals the one-database reduce's byte for byte (`test_pyramid_reduce`). Limit of this step: the default subject must be one of the dataset's own members, which is what `split dataset` makes; a released template that has its own home dataset needs `SubjectWriter::open` by id, a later step.

The shared `--dataset` help now says what it is: the path of the dataset's folder. The word "datastore" is gone from the help; `--datastore` stays as the name of the option of `create dataset`, `list datasets` and `delete dataset`, which name the folder that holds dataset folders.

## The Python face (`nxr_neuro`, `bindings/python`; spec `docs/specs/2026-11-17-interoperability.md` section 6.1)

The first slice of the Python face: **one wheel per platform** (Linux x86_64 manylinux_2_28, macOS arm64 13.3+; nanobind, stable ABI, so one wheel serves
Python 3.12 and newer), built by `.github/workflows/wheels.yml` with cibuildwheel, which runs `bindings/python/tests` on each built wheel. Pinned build
tools (`bindings/python/pyproject.toml`): nanobind 3.1.0, scikit-build-core 1.1.2, cibuildwheel 4.3.5. The version is the project's (`project(nxr_neuro VERSION ...)` in
`CMakeLists.txt`); the wheel carries `LICENSE`, `AUTHORS` and `THIRD_PARTY_NOTICES.md` in its `.dist-info/licenses`. Build it by hand:

    cd bindings/python && python -m build --wheel      # or: pip wheel .   (CMake, a C++17 compiler and the fetched dependencies, as the main build)

A binding only binds: each Python function is one call into the library (`nxr_compute` for the maths, `nxr_io_formats` for the formats); the options
carry the library's names and defaults, and their docstrings are the options structs' comments. The dataset format is read and written through the dataset API below
(every command of the command line as a function).

**Arrays.** A signal is `[channels, samples]`, C-ordered. float32 and float64 are accepted (anything else is converted to float64); the library computes
in float64 and the result comes back in the input's precision. float64 results are handed over without a copy and may be Fortran-ordered. The C++ releases
the GIL while it runs. Errors are `nxr_neuro.Error` (with `.code`) and its subclasses `FormatError`, `UnsupportedError` (also `NotImplementedError`),
`InvalidArgumentError` (also `ValueError`), `IOError_` (also `OSError`), `NotConvergedError`, `CancelledError`, and io's codes: `WriterLockHeld`, `NotWritable`
(also `PermissionError`), `SchemaVersionError`, `NotADatabaseError`, `NoSuchRow` and `Ambiguous` (also `LookupError`), `AlreadyExists` (also `FileExistsError`), `NotFound`
(also `FileNotFoundError`); an option name that is not one of the library's is a `ValueError`.

**The dataset API: the command line as functions** (spec `docs/specs/2026-11-19-io-service.md`). `nxr-neuro <verb> <noun> --option VALUE` is
`nn.<verb>.<noun>(option=VALUE)`, for every command of the command table (`cli/command_table.cpp`): `nn.create`, `nn.list`, `nn.show`, `nn.delete`, `nn.read`, `nn.add`, `nn.edit`,
`nn.query`, `nn.import_`, `nn.export`, `nn.register`, `nn.verify`, `nn.release`, `nn.split`, `nn.join`, `nn.compact`, `nn.recover`, `nn.migrate`, and `nn.compile` and
`nn.run` (verbs with no noun).
- **Names.** The command line's options with dashes as underscores (`--from-sphere` is `from_sphere`; `export bids-derivatives` is `nn.export.bids_derivatives`); a Python
  keyword takes a trailing underscore: the verb `import` is the module `nn.import_`, the option `--from` is `from_`. The positional argument (`import subject PROTOCOL`,
  `read dataset FOLDER`, `compile DEFINITION`, `import templates TEMPLATES_DIR` as `templates_dir`) is the one positional parameter; every option is given by name.
- **Values.** A flag is a boolean; a repeatable option (`--templates DIR ...`) a list; an option the command line takes comma-separated (`--subjects A,B`, `--rows 0,2`) a
  list or that text. Required options are required; an option left out takes the command line's default (the library fills the same request). Arrays go in as NumPy
  arrays (any array-like, in its own precision) or a path to a .npy file: the face hands them to the library in memory (dtype, shape, bytes; no temporary file), under the name the
  producer records (named after the option, `values.npy`). `add measurements --measure NAME=FILE.npy` is `measure={"NAME": array}`; `compile --set NAME=VALUE` is
  `set={"NAME": value}` (or the command line's strings), `--signatures` and `--inputs` a dict (or the JSON file's path).
- **Results.** The command's result object (its `"stage": "result"` line) as an `nn.Result`, a dict; where the command line writes an array to a file or prints its
  checksum (`read <class>`, `read dataset --array`), the array is in `Result.arrays` (NumPy, C order, its stored dtype). `export ...` writes its files and returns their
  paths, as `read <class>` does with `out`. `progress=callable` receives each progress line as a dict. `nn.list.pipelines()` gives the pipelines `run` takes as objects with
  `.help()`.
- **Binding.** One native call (`_core._command(verb, noun, options)`) converts the options to the command's library request (`bindings/common/commands.h`: conversion only,
  shared with the WebAssembly face) and calls the command's function once (`nxr/io/task/commands.h`, `nxr/pipelines/face.h`); an option the command does not take is refused.
  The GIL is released while the library runs.
- **`nn.run(pipeline, dataset, subject, *, from_=, to= | all=True, replace=, dry_run=, preset=, **parameters)`**: the time range is required (seconds); a group definition
  takes `subjects=[...]` and `into=`. Returns {pipeline, subject, cache, stored, runs, seconds}; read what was stored with `nn.read.<class>`.
- **`nn.open(dataset, mode="read")`**: a handle whose methods are the same functions with the dataset bound (`d.read.field(subject=..., name=..., all=True)`,
  `d.run("psd", "S1", all=True)`, `d.compile(...)`). `mode="write"` takes the dataset folder's writer lock now and holds it until `close()` (a context manager; released at
  exit): the handle's commands borrow it (io's handshake `NXR_WRITER_LOCK_HELD`, lent by the library's `task::Handle`), another process's write is refused (`WriterLockHeld`),
  and so is another write in the same process that does not go through the handle (in JavaScript also from another instance of the WebAssembly module: the locks a process holds are recorded on `globalThis`, shared by every instance).
  The library keeps the subject's database open between the handle's reading calls (`task::Handle`, `open_reader`: read-only, no lock, no migration); a write command opens
  its own database for the length of its unit, since a package's record and content hash are finished when its writer closes.
- **Refused, naming the new form** (the first wheel's calls; no aliases): `nn.read(dataset, subject, name, ...)` (now `nn.read.<class>`, a recording's range as `from_`/`to`
  in seconds), `nn.run(..., start=, stop=)` (now `from_=`, `to=`), `nn.list.results(dataset, subject, kind)` (now `nn.list.results | recordings | kernels`),
  `nn.list.subjects(names=)` (now `subjects=`). `nn.Stored` and `nn.RunResult` are removed.
- **The coverage test** (`bindings/python/tests/test_commands.py`) reads `cli/command_table.cpp` and fails when a command has no function, a function's arguments differ from
  the command's options (names, the positional, required, flags), or a function has no command.

**`nn.compute`** (the pure functions: no file, no dataset, no recording):
*(Named `nn.ops` in the first wheel; renamed `compute` by the maintainers' decision of 2026-11-17, the same in MATLAB as `nxr.compute`. `nn.ops` stays one release as a deprecated alias that warns. Pipelines will be `run`, later.)*

| function | the library's | what |
|---|---|---|
| `compute.signal.psd(x, sfreq, segment_samples=256, overlap=0.5, window="hann", periodic=True, detrend="constant", scaling="density", good=None)` | `nxr::signal::spectral::Welch` | Welch's PSD; with those defaults equal to `scipy.signal.welch(nperseg=256)`; `good` leaves out segments touching a bad sample |
| `compute.signal.xpsd(x, sfreq, bins, ..., mean=True, quadrature=False)` | `nxr::signal::spectral::CrossWelch` | the cross-spectral density between channels, per group of bins; optionally its imaginary part |
| `compute.signal.lowpass`, `highpass`, `bandpass`, `bandstop`, `notch` | `nxr::signal::filters` | Brainstorm's zero-phase FIR and IIR filters, options as `FilterOptions` |
| `compute.signal.decimate(x, q, order=0, left=0, width=None, offset=0)`, `decimationHalo(q, order=0)` | `nxr::signal::decimate`, `decimationHalo` | the signal at rate / q: a window-method low-pass (order 20 q, Hamming, edge at the new Nyquist frequency) applied centred, then every q-th sample (`scipy.signal.decimate(x, q, ftype='fir', zero_phase=True)`); on a block with `decimationHalo(q)` samples of context on each side, the decimated chunks laid end to end are the whole signal's. Nothing above the new Nyquist frequency survives; the band just below it is attenuated by the transition |
| `compute.stats.pca(x, scale="none")`, `compute.stats.reconstruct(...)` | `nxr::stats` | principal components of observations x features (scikit-learn's `svd_solver="full"`, sign convention fixed) and the rank-k reconstruction |
| `compute.operators.eigen(vertices, faces, k, ...)` | `nxr::geometry::solve::eigen` | the Laplace-Beltrami eigenpairs of a triangle mesh (cotangent Laplacian, mixed-Voronoi mass; the shift ladder by default) |
| `compute.geometry.surfaceArrays(vertices, faces, intrinsic_delaunay=True, mollify=1e-6, flip_eps=1e-6, surface="")` | `nxr::geometry::surfaceArrays` | the surface as data, made once: the input mesh and its intrinsic mollified Delaunay triangulation (with its record: ε, flips, boundary vertices), a dict of arrays passed to every surface function below (nothing is cached; no function flips again) |
| `compute.geometry.complex(surface)` | `nxr::geometry::IntrinsicSurface::complex` | the operator mesh in the face-corner layout: intrinsic faces, edges (tail → tip), signed face edges, lengths, signpost angles, cone angles |
| `compute.geometry.meshTopology(surface)`, `eulerCharacteristic(surface)` | `nxr::geometry::meshTopology`, `eulerCharacteristic` | the input's halfedge topology as arrays (boundary edges and vertices from `heIsInterior`), and χ |
| `compute.geometry.vertexFrames(surface)`, `frames(surface)` | `nxr::geometry::vertexFrames`, `frames` | the tangent frame (e1, e2 = n × e1, n) per vertex (the gauge) and per input face |
| `compute.geometry.heat(surface, sources, t_coef=1.0, mesh="intrinsic")` | `nxr::geometry::heat` | heat-method geodesic distances from a source set, or one column per set |
| `compute.geometry.logMap(surface, source, strategy="affine_local", t_coef=1.0)` | `nxr::geometry::transport::logMap` | the log map (geodesic polar coordinates of every vertex, norm = geodesic distance) at a source vertex, or a list of them on one solver, in the vertex's tangent frame (`sourceE1`, `sourceE2`, `sourceNormal` in R^3) |
| `compute.geometry.logMap(surface, (face, (w0, w1, w2)), ...)` | `nxr::geometry::transport::logMap (`FacePoint`)` | the same at a point of an INPUT face (a list of pairs: one solver): the blend of the three corners' log maps carried into the face's plane (geometry-central's face-point rule; first order in the face size, exact at a corner, where it equals the vertex form). One frame in R^3: the heaviest corner's rotated into the face's plane (`sourceNormal` = the face normal), and `sourcePoint` = sum w_k p_k. Weights non-negative summing to 1 (1e-9), a face outside the mesh: refused. On a fine unit sphere at a face centroid the distance error is about a tenth of taking the nearest vertex's log map |
| `compute.operators.assembleManifoldOperators(surface, mass="mixedVoronoi")` | `nxr::operators::assembleManifoldOperators` | the cotan Laplacian, the mass, the dual areas and the vertex normals of the operator mesh (sparse matrices as `scipy.sparse`) |
| `compute.operators.assembleDECOperators(surface)` | `nxr::operators::assembleDECOperators` | d0, d1, ⋆0, ⋆1 (raw, not floored), ⋆2 and ⋆1⁻¹ of the operator mesh; d0's rows are `complex`'s edges |
| `compute.operators.hodgedirac.assemble(surface, star1_floor=0.0, mollify=1e-6, flip_eps=1e-6)` | `nxr::operators::hodgedirac::assemble` | the Hodge–Dirac pencil (K, M) with d0, d1, the stars, the complex, the construction's record, the topology (χ, boundary loops, genus) and the ⋆1 counts (`star1Clamped`, `star1Negative`: boundary edges opposite an obtuse angle) |
| `compute.operators.hodgeDiracModes(pencil, count, both_signs=False, **solve options)` | `nxr::operators::hodgeDiracModes` | the modes by the exact/coexact split: μ, the vectors (0-, 1-, 2-form rows), kind (0 exact, 2 coexact). Refused on a mesh other than a closed genus-0 surface or a disc (no harmonic 1-forms are built), and on genuinely negative ⋆1 unless a floor was named |
| `compute.operators.hodgeLadders(L0, M0, S2, M2, count, **solve options)` | `nxr::operators::hodgeLadders` | the exact and coexact ladders separately, the λ cut and the merge by μ |
| `compute.operators.poisson(K, M, density_vertices, density_values)` | `nxr::operators::poisson` | K φ = −M (ρ − ρ̄) |
| `compute.signal.intrinsicComplex(surface)` | `nxr::signal::intrinsicComplex` | the intrinsic triangulation as a cell complex (edges tail → tip, faces, lengths, d0, d1, the three stars, the signposts placing its edges in R³), built once and passed to every function below |
| `compute.signal.sharp(complex, w)`, `flat(complex, J)` | `nxr::signal::sharp`, `flat` | a 1-form on the intrinsic edges ↔ a tangent vector field at the vertices (in R³), through the signposts' embedding; exact for a constant field on a flat mesh, the boundary included |
| `compute.signal.phaseCurrent(operator, z, normals=None, components=1)`, `phaseGradient(complex, x)` or `phaseGradient(j, p)`, `vertexGradient(positions, faces)` | `nxr::signal::phaseCurrent`, `phaseGradient`, `vertexGradient` | the phase current Im(z̄ dz) and the phase gradient dφ, ONE quantity each, discretised by the operator the caller passes: the intrinsic complex (a signed number an edge: Im(conj(z_tail) z_tip), arg(x_tip conj(x_tail))) or a vertex gradient (a vector a vertex, the reference: Im(z̄ ∇z), j / \|z\|²) |
| `compute.signal.modesOf(modes, pencil)` | `nxr::signal::modesOf` | a Hodge–Dirac solve's result as the forms read it: P (0-form parts), X2 (2-form parts), μ, kind |
| `compute.signal.projectOneForm(complex, modes, w)`, `synthesizeOneForm(complex, modes, coef)` | `nxr::signal::projectOneForm`, `synthesizeOneForm` | a 1-form ↔ its coefficients in the modes' 1-form parts (inner products in ⋆1, no solve); zeroing rows by kind gives the curl-free or divergence-free part, by μ a band-limited form |
| `compute.signal.projectTwoForm(...)`, `synthesizeTwoForm(...)` | `nxr::signal::projectTwoForm`, `synthesizeTwoForm` | the same for a 2-form on the intrinsic faces, against the coexact modes' 2-form parts |
| `compute.signal.exactPotential(modes, coef)`, `coexactStream(complex, modes, coef)` | `nxr::signal::exactPotential`, `coexactStream` | the scalar potential Φ (d0 Φ = the exact part) and the stream function Ψ of the coexact part, at the vertices |
| `compute.signal.splitPhase(complex, modes, w)` | `nxr::signal::splitPhase` | a 1-form on the intrinsic edges (a phase step, a phase current, any edge values) split by the modes into its exact (curl-free) and coexact (divergence-free) parts, POINTWISE only: the coefficients, the parts, the potential, the stream function, the propagation direction. The shares and the misfit share are derived measures of reductions (below). **Boundary condition (a fact of the method, not an option):** on an open surface the potential is Neumann (zero normal flux) and the stream function Dirichlet (zero on the rim); a closed surface has no boundary. **Reading:** for a phase-step input (`phaseGradient`) the exact part is the timing (its potential a time-of-arrival map, potential / ω) and the coexact part the rotation (winding at phase singularities); for an amplitude-weighted phase current (`phaseCurrent`) the exact part is the curl-free part of the current, not a timing. **What it cannot resolve:** a phase step per edge is taken in (−π, π], so a wave with more than half a cycle per edge aliases |
| `compute.signal.vertexEnergy(complex, w)` | `nxr::signal::vertexEnergy` | the ⋆1 energy of a 1-form at the vertices (each edge's ⋆1 w² half at each end): a tile's energy is the SUM over its vertices. Summed over a partition's members, the energies of w, of the exact part, of the coexact part and of w − exact are the measures `one_form_energy`, `exact_energy`, `coexact_energy`, `exact_misfit_energy`; the derived measures `exact_share`, `coexact_share`, `residual_share` and `exact_misfit_share` are their ratios, computed at read |
| `compute.select.median(x)`, `mad(x)` | `nxr::select::median`, `mad` | the median, and the robust SD 1.4826 * median absolute deviation, of a 1-D series |
| `compute.select.crossings(x, entry, exit)`, `mergeSpans(spans, gap)`, `keepSpans(spans, min_length, max_length)` | `nxr::select::crossings`, `mergeSpans`, `keepSpans` | spans where a series is above a hysteresis threshold, joined across short gaps, kept by length (samples); a span is `start, stop, peak, value` arrays (0-based, stop exclusive) |
| `compute.select.detectBursts(amplitude, step, enter_sd=1.5, exit_sd=1.0, merge_s=0.1, min_s=0.4, max_s=inf, baseline=None)` | `nxr::select::detectBursts` | the bursts of an amplitude series: robust baseline, hysteresis, merge, length; the spans with the baseline, levels and durations used |
| `compute.select.risefall(amplitude, sfreq, t0, t1, model="linear", margin=1.0, smooth=0.1, tau_max=1.0, min_r2=None)`, `stagesOf(t_on, t_off, tau_rise, tau_fall)` | `nxr::select::risefall`, `stagesOf` | the double-sigmoid fit of an episode's switch-on and switch-off (`kept` is the fit gate when `min_r2` is given), and its onset, plateau and offset stages |
| `compute.signal.bandAnalytic(x, sfreq, bands, shape="morlet", half_width=1/3)`, `bandEnvelope(...)` | `nxr::signal::bandAnalytic`, `bandEnvelope` | the analytic signal (complex) and the envelope (real) of bands, `[bands, channels, samples]`, designed on the window given |
| `compute.signal.timeBank(sfreq, n, family, sides="analytic", gain=1.0, ...)` | `nxr::signal::timefilterbank::timeBank` | a time filterbank for inspection: bins, gains `[members, bins]`, centre frequencies, frame bounds |
| `compute.signal.wt(x, sfreq, bank)`, `iwt(W, sfreq, n, bank, synthesis="dual")` | `nxr::signal::wt`, `iwt` | the wavelet transform of a signal by a time bank (a dict of the design fields, designed on x's own n samples) and its inverse; complex `[members, channels, samples]` |
| `compute.signal.graphBank(family, eigenvalues=None, lmax=None, ...)` | `nxr::signal::graphfilterbank::graphBank` | a graph filterbank (mexhat, heat, itersine, logitersine, meyer, weyl, rank) for inspection: gains at the eigenvalues, roles, scales, centre wavenumbers, frame bounds |
| `compute.signal.gwt(F, bank, eigenvalues=, eigenvectors=, mass=)` or `(F, bank, laplacian=, mass=, order=40, lmax=0)`, `igwt(...)` | `nxr::signal::gwt`, `igwt` | the wavelet transform of a field on a graph, exact through the operator's modes or by a Chebyshev series; `[members, vertices, samples]` |
| `compute.signal.jwt(F, sfreq, graph_bank, time_bank, eigenvalues=, ..., kernels=None)`, `ijwt(...)` | `nxr::signal::jwt`, `ijwt` | the joint time-vertex wavelet transform on mode series or on vertex fields, with optional speed or ridge kernels; complex `[members, rows, samples]` and the member index table (graph member fastest) |

**`nn.io`** (the pure formats, `io/formats`; native, no MATLAB and no Python library at run time): `open_edf` / `read_edf` (EDF, EDF+, BDF, BDF+; `open_bdf` and
`read_bdf` are the same), `open_brainvision` / `read_brainvision`, `open_eeglab` / `read_eeglab`, `open_fif` / `read_fif` (Neuromag / MNE raw; `allow_maxshield=False`) return a recording: `rec.header` (a dictionary holding every field of
the file's header, channels with their units and locations, annotations, gaps) and `rec.read(start, stop, first_channel, last_channel)`, a block `[channels, samples]` in SI units, float32
or float64 as the file holds them. **The span is always stated** (`start` and `stop` in samples, or `all=True` for `read_*`); a block's values never depend on the block. `read_nifti(path)` gives
`(data, header)` (the file's axes, x first, with the 4 x 4 affine), `read_gifti(path)` a dictionary of arrays with their intents, metadata, labels and transforms, `read_tsv(path)` a dictionary of columns.

**Tests** (`bindings/python/tests`, pytest; synthetic data only): Welch against `scipy.signal.welch` and a NumPy cross-spectrum; the filters against their passbands, stopbands and zero phase; PCA against the NumPy SVD with
the sign convention; the sphere's eigenvalues l(l+1); the readers against the MNE-Python, nibabel and pandas goldens of `io/formats/test` and their refusals; the dataset API's coverage test against the command table, the old forms' refusals, and on synthetic datasets: datasets, subjects
and packages created, listed and shown, every class added from arrays and read back byte for byte, `compile`, `run` with `from_`/`to`, and the writer lock of a write handle
refusing another process.

## The MATLAB face (`+nxr`, `bindings/mex`; spec `docs/specs/2026-11-17-interoperability.md`)

The same first slice as the Python wheel, **with the same names and options**, under the `+nxr` package: `nxr.compute.signal.psd`, `cross_spectrum`;
`nxr.compute.signal.lowpass`, `highpass`, `bandpass`, `bandstop`, `notch`; `nxr.compute.stats.pca`, `reconstruct`; `nxr.compute.operators.eigen`; `nxr.compute.select.{median, mad, crossings, mergeSpans, keepSpans, detectBursts, risefall, stagesOf}` and `nxr.compute.signal.{bandAnalytic, bandEnvelope, timeBank, wt, iwt, graphBank, gwt, igwt, jwt, ijwt}` (the C++ word verbatim, camelCase, in every face; the options keep their snake_case names; a bank is a struct of the design fields; spans are structs of columns with 1-based `start` and `peak` and `stop` the last sample, inclusive: the wheel's 0-based, stop-exclusive arrays converted at the boundary; the member index of `jwt` is 1-based); `nxr.io.open_edf` / `open_bdf` /
`open_brainvision` / `open_eeglab` / `open_fif` (a `nxr.io.Recording`: `header` as a struct, `read(first, last)`, `read_all`), `read_<format>` (header and block in one call, the span
stated), `read_nifti`, `read_gifti`, `read_tsv`. FIF raw (`io/formats/fif.h`: MNE's raw reader, split files, skips, annotations, the whole `info`) is in both faces (`nn.io.open_fif` / `read_fif`).
A binding only binds: each function is one command of the one `nxr_compute` mexFunction (`compute.*` and `io.*` commands, `bindings/mex/src/first_slice*.cpp`) into the library; the
options are name=value with the library's names and defaults, and each function's help text is the options struct's comments (meaning, default, unit). The earlier
commands (`nxr_compute('create', ...)`, `nxr.manifold.*`) are unchanged.

**Conventions.** Signals are `[channels x samples]`. Precision follows the input (single in, single out; integers and logicals are computed as double); a recording reads as the file holds it
(single for integer and 32-bit float files, double for 64-bit). **Indices are 1-based** (faces, bins, samples, channels, the gaps' first sample, a GIFTI triangle's vertex indices),
the wheel's 0-based; the conversion is made at the boundary, once. Per-component lists (`singular_values`, `mean`, ...) are rows, `eigenvalues` a column; a cross-spectrum is `[groups x channels x channels]`.
Errors have identifiers: `nxr:invalidArgument`, `nxr:formatError`, `nxr:unsupported`, `nxr:io`, and the library's (`nxr:...`).

**Build and symbols.** `cmake --build build --target nxr_compute_mex` (MATLAB found by CMake; `bindings/mex/matlab/README.md`). The MEX shares MATLAB's process, which ships its own HDF5 and zlib: HDF5 and matio are
linked statically and hidden (`nxr_io_hide_static_symbols`), zlib too where a static archive is found (`-DZLIB_USE_STATIC_LIBS=ON`; Linux, and Homebrew's on macOS), the C++ runtime on Linux; only
`mexFunction` and `mexfilerequiredapiversion` are exported. SQLite (the pinned amalgamation) and zstd are io's and are not in it. `bindings/mex/test/check_symbols.sh` verifies it (`nm`, `otool`/`ldd`).

**Tests and CI.** `bindings/mex/test/first_slice` (class-based `matlab.unittest`): the first slice against the C++ goldens (scipy, Brainstorm, numpy, MNE-Python, nibabel, pandas) and the same synthetic inputs and
expected values as `bindings/python/tests`. `.github/workflows/matlab.yml` builds the MEX on Linux x86_64 and macOS arm64 and runs them with `matlab-actions/setup-matlab` and `run-tests` (tags, dispatch, and pull
requests touching `bindings/mex/**`). Not in this slice: the pipeline commands (`nxr.compute.<verb>`), which wait for io's task-level API.

## Exporters (`io/export/`, `io/include/nxr/exporters/`; spec `docs/specs/2026-11-17-interoperability.md` sections 3.3 and 5; D6: exports first)

An exporter finds stored nodes through io's task API, opens their arrays, and hands them to a pure writer of `io/formats/`: it writes what is stored, in the units stored, or refuses, and does no maths. Every export:
- writes a **provenance sidecar** `<first file>.json` (`nxr-export/1`): the producer and its version, the command and every option, the time range read (`span`), the package's name and content hash, each input (class, name, kind, unit, producer record, and the digest of its arrays: key count, bytes and one SHA-256 over the keys' recorded digests), the **terms** and each output file with its SHA-256. The same command writes the same bytes (no timestamps);
- carries **licence and attribution** from the inputs' `origin` records: a ShareAlike licence among them (CC BY-SA, ODbL) becomes the output's licence and the output is flagged `share_alike`; `redistributable` only when every input's record says so; an input with no record is `unknown` and not redistributable; two different ShareAlike licences cannot be combined and are refused;
- states the time range explicitly where it reads a series: `--from`/`--to` in seconds (or `--all`), required, refused for a map. The series is read chunk by chunk through the chunk service.

**`export gifti`** (`nxr/exporters/gifti.h`): a Field on a surface (a map: `.func.gii` or `.shape.gii` with `--kind shape`; a series on surface x time line: one `NIFTI_INTENT_TIME_SERIES` array per frame of the range, each with `nxr-neuro:time`, the file with `TimeStep`), or a labelling (a partition or set Selection: `.label.gii`, an int32 array and the label table from the dictionary's names and colours; code -1 is key 0 `???` when the dictionary leaves 0 free, else refused). `--with-surface` adds `.surf.gii`; `--hemispheres` writes one file of each kind per hemisphere (`_hemi-L`, `_hemi-R`, `AnatomicalStructurePrimary` CortexLeft / CortexRight) from the components the glued cortex stores. The file metadata carries `nxr-neuro:version`, `:command`, `:producer`, `:origin`, `:licence`, `:attribution`, and per data array `Name`, `nxr-neuro:field`, `nxr-neuro:unit`. What GIFTI cannot hold is recorded in the sidecar, not hidden: values are written as float32 (`details.converted` says when the field was float64), and coordinates are written in the stored unit unless `--surface-unit m|mm` scales them. Only scalar fields are written. A GIFTI file is one XML document, so the frames of the stated range are held (as float32) to write it: the range bounds the memory. Tests: `test_export_gifti` (round trips with the pure reader) and `io/test/parity/test_export_gifti.py` (nibabel reads the files; with `NXR_EXPORT_GOLDEN_DIR` set the test keeps its files for it).

**`export tsv`** (`nxr/exporters/tsv.h`): `--partition P --field F --measures a,b` writes one row per member of the labelling (`member`, `code`, a column per measure; scalar measurement rows, the values as stored and the roll-up io's measurement query makes; a measure derived at read, such as sd or variance, is a derived column, named so in the sidecar), or `--statistic S` writes the statistic in long form (a column per axis: a frequency or other line axis as its coordinate in the axis's unit, a bin axis as `<axis>_low` and `<axis>_high`, a selection axis as the member's name, any other axis as the index; `component` when it has components; `value`). `<STEM>.tsv` is UTF-8 with LF, tab-separated, `n/a` for NaN (an infinite value is refused: BIDS has none). `<STEM>.json` is the BIDS sidecar: each column's `Description`, `Units` (when the store records a unit; else `FieldUnits`) and `Levels`, `Partition` with its dictionary, `Statistic` with its measure, method, scaling and span, and the provenance record under `GeneratedBy`. `export measurements` (the dataset-wide table) keeps its own folder of CSV files and dictionary; its BIDS sidecar is to follow. Test: `test_export_tsv`.

**`export fif`** (`nxr/exporters/fif.h`, the pure writer `nxr/formats/stc.h`): a source map (one time point, start 0 s, step 1 s) or a source series on a glued cortex whose left and right hemispheres are stored as components, written as `<STEM>-lh.stc` and `<STEM>-rh.stc` (`mne.read_source_estimate`). Each hemisphere's vertices are numbered from 0 in its own surface (the source space is every vertex). A series is read through the chunk service over the stated range (`--from`/`--to` or `--all`, required) and written one time point at a time, so memory is one frame whatever the range; `.stc` stores time-major float32 with times in milliseconds (MNE's layout, byte for byte), so a float64 field is converted (recorded in the sidecar). The file has no room for metadata: `<STEM>-lh.stc.json` holds the provenance and the field's unit. **`export fif --covariance NAME --out STEM`** writes a covariance statistic as `<STEM>-cov.fif` (`mne.read_cov`): the noise covariance (kind 1) with the channel names of the statistic's channel Selection (the names labelling of the channel manifold), `nfree` from the statistic's degrees of freedom, values as stored (the unit goes to the sidecar; projectors and bads are not carried); read back with MNE in `io/test/parity/test_export_fif.py`. Forward solutions and inverse operators are **not written**: an MNE forward needs each channel's FIF info (kind, coil type, position), which a package does not keep, and an MNE inverse operator is the eigen-decomposition of the whitened forward, which a stored kernel does not hold; the command refuses them with that reason. Tests: `test_formats_stc` (MNE's bytes, from `gen_stc.py`), `test_export_fif` (including a series stored in several chunks) and `io/test/parity/test_export_fif.py` (MNE reads the files).

**`export bids-derivatives`** (`nxr/exporters/bids.h`; spec 5.4, question 9: `desc-<what>` and documented suffixes now): `--package P --out DERIVATIVES --results REGEX [--space NAME] [--session S --task T --run N] [--from/--to or --all] [--with-surface] [--stc] [--measurements PARTITION:FIELD:m1,m2]`. Each field on a cortical surface, labelling and statistic whose whole name matches is written by the exporter of its kind (the folder is `--out`, not `--to` as the spec has it, because `--to` is the end of a time range): `sub-<S>[/ses-<E>]/sub-<S>[_ses-<E>][_task-<T>][_run-<R>]_space-<SP>[_hemi-<L|R>]_desc-<name>_<suffix>.<ext>` with the suffixes `map`, `series` (`.func.gii`), `dseg` (`.label.gii`), `stc` (`_stc-lh.stc`, `-rh.stc`), `surf`, `stat` and `measures` (`.tsv`). `<name>` is the stored name reduced to letters and digits (two names that reduce alike are refused); the space is the one named, or the template the subject is registered to; session, task and run only when named. Each file has a BIDS sidecar `<stem>.json` holding the exporter's provenance under `GeneratedBy` and `Sources` as BIDS URIs (`bids::<path>`) for inputs imported from a BIDS dataset. `dataset_description.json` (`DatasetType: derivative`, `GeneratedBy`, `SourceDatasets`, `License`, and `NxrNeuro.Suffixes`, which lists nxr-neuro's suffixes with what each holds) is merged by a re-run into the same folder; the licence is the ShareAlike one when an input carries it (two different ShareAlike licences are refused), else the common one, else "see the sidecar of each file". `participants.tsv` is never written. A series needs its stated range; a pattern that matches nothing exportable is refused. Test: `test_export_bids`.

## Build

CMake 3.20+, Ninja, a C++17 compiler; SQLite, zlib and zstd from the system (`brew install zstd`). c-blosc, HDF5, matio, Spectra and Eigen 3.4 are fetched into the build
directory; geometry-central is a git submodule (with its own), so clone with submodules:

    git clone --recurse-submodules git@github.com:neurodynamics-xr/nxr-neuro.git     # or: git submodule update --init --recursive
    cmake -S . -B build -G Ninja && cmake --build build && (cd build && ctest)

(Large build trees can be placed on another disk by making `build/` a link to it before the first configure.)

Options (all ON by default): `NXR_NEURO_BUILD_COMPUTE`, `_IO`, `_PIPELINES` (needs both), `_CLI` (needs io), `_BINDINGS` (needs compute), `_TESTS`. `-DNXR_EIGEN_DIR=<an Eigen 3 source folder>` uses one
you already have instead of fetching. The installed Eigen 5 does not satisfy geometry-central's check, which is why Eigen 3.4 is fetched (and why geometry-central never downloads one
into its source tree).

Bindings: MATLAB MEX builds when MATLAB is found (`build/bindings/mex/nxr_compute.mex*`); the Node addon is built only through cmake-js (`bindings/scripts/build.sh`, after `npm install` in
`bindings/`); wasm under Emscripten (`bindings/scripts/build-wasm.sh`, which builds `compute` alone).
**nxr-io in WebAssembly** (2026-11-11): under Emscripten `io/` builds whole, the Brainstorm importers with matio (no HDF5), with SQLite 3.51 from its amalgamation (the schema's views need 3.44 or newer; Emscripten's port is 3.39), zlib from Emscripten,
the zstd c-blosc carries, and WebAssembly exceptions (io reports errors by throwing). `bindings/wasm-io/smoke.sh` builds it and checks it against the native CLI
both ways (WASM writes, the CLI reads; the CLI writes, WASM reads).
## The JavaScript face (`@nxr/neuro`, `bindings/wasm-io`; spec `docs/specs/2026-11-19-io-service.md`, step 5)

The command line as functions, in Node: `nxr-neuro <verb> <noun> --option VALUE` is `nxr.<verb>.<noun>({ option: VALUE })`, a Promise, for every command of the command table
(`cli/command_table.cpp`), `nxr.run` and `nxr.open`; and the geometry compute on meshes in memory (`nxr.compute`), in the same module instance. The module is the C++ library
under Emscripten (SQLite 3.51 from its amalgamation, zlib, the zstd c-blosc carries, WebAssembly exceptions; no HDF5, so a v7.3 `.mat` is refused by name, and a Brainstorm import skips it and names it: see *MATLAB v7.3 files in the WebAssembly build*); in Node it reads and
writes the real disk (NODERAWFS). The browser's file connection (a folder the user grants) comes later. The old `Datastore` class (28 functions, its own SQL) and `wasmClient` are
removed, with no aliases; the parity tests against the Studio's service went with them.

    import { nxrNeuro } from '@nxr/neuro'
    const nxr = await nxrNeuro()
    await nxr.create.dataset({ dataset: 'study/demo' })
    await nxr.create.subject({ dataset: 'study/demo', name: 'sub-01' })
    await nxr.add.field({ dataset: 'study/demo', subject: 'sub-01', name: 'w', manifold: 'sensors', values: new Float32Array([0.5, -1.25, 3]) })
    const r = await nxr.read.field({ dataset: 'study/demo', subject: 'sub-01', name: 'w' })   // r.arrays.values: a Float32Array with .shape and .dtype

- **Names.** The command line's options with dashes as underscores (`--from-sphere` is `from_sphere`; `export bids-derivatives` is `nxr.export.bids_derivatives`); JavaScript needs
  no trailing underscore (`nxr.import.bids({ from })`, `nxr.import.subject(...)`). The positional argument (`protocol`, `folder`, `file`, `definition`, `templates_dir`) is a key of
  the options or the first argument (`nxr.read.dataset(folder, { array })`). A flag is a boolean; a repeatable option an array.
- **Arrays in** are typed arrays (their precision is kept), `{ data, shape, dtype? }`, or the path of a .npy file; the face hands them to the library in memory, under the name the
  producer records (`values.npy`). `add measurements --measure` is `measure: { NAME: array }`; `compile --set` is `set: { NAME: value }`.
- **Arrays out.** The command's result object (its `"stage": "result"` line); where the command line writes an array to a file or prints its checksum, the object has `arrays`:
  typed arrays by name in the stored precision (a 64-bit integer array is a `BigInt64Array`), each with `shape` and `dtype`. `progress` (a function) receives each progress line.
- **Errors** are an `NxrError` with the library's `.code` (`invalid_argument`, `no_such_row`, `ambiguous`, `already_exists`, `not_found`, `writer_lock_held`, `not_writable`,
  `schema_version`, ...).
- **`nxr.run(pipeline, { dataset, subject, from, to | all, replace, dry_run, preset, subjects, into, ...parameters })`**: the time range is required (seconds); `nxr.list.pipelines()`.
- **`nxr.open(dataset, mode = 'read')`**: a handle with the verbs' functions and the dataset bound (`d.read.field({ subject, name })`, `d.run('psd', { subject, all: true })`,
  `d.close()`). The library keeps the subject's database open between reading calls (read-only); `'write'` takes the dataset's writer lock now and holds it until `close()`.
- **One native call** (`command(handle, verb, noun, optionsJson, arrays, progress)` in `bindings/wasm-io/src/binding.cpp`) converts through `bindings/common/commands.h`, the same
  code as the Python face; the face has no SQL and no logic. `js/commands.mjs` and `js/commands.d.ts` are generated from the command table (`node tools/command_table.mjs`).
- **Tests.** `bindings/wasm-io/test/run.sh [<wasm build folder>]` builds the module and runs `test/commands.test.mjs` (every command has its function with the table's options, no
  function without a command, the generated files are current) and `test/dataset.test.mjs` (create, add and read each class, query measurements, compile and run, the writer lock
  across processes) with `node --test`, on synthetic datasets. `bindings/wasm-io/smoke.sh` still builds the io-only module and checks it against the native command both ways.
- **What the Cortical Flow Studio used** (`Datastore`: views, rows, key-level array access, the repositories' row-level writes, `layout`, jobs) has no public equivalent by design
  (spec, Appendix B); the Studio's opt-in `NXR_BACKEND=nxr-neuro` backend is rewritten onto this API in the Studio's own step.

## The command `nxr-neuro`

`build/cli/nxr-neuro`: one grammar, **verb first**: `nxr-neuro <verb> <what> [options]`. `nxr-neuro --help` lists the verbs, `nxr-neuro <verb> --help` the commands of a verb
(`run` groups the built-in pipelines by topic: source imaging, geometry, time, spectra, cortical flow and patterns, measurements, nulls), and `nxr-neuro <verb> <what> --help` one command: each
option with its meaning, default and unit, and an example. The help, the man pages (`nxr-neuro __man DIR`) and the shell completion (`nxr-neuro completion bash|zsh|fish`) are made from
one table (`cli/command_table.cpp`), checked against the parsers by a test. The grammar, the old spellings and the rules are in `docs/specs/2026-11-16-command-line.md`.

**A time range is required** for every computation over a recording: `--from <s> --to <s>` (seconds or `m:ss`) or `--all`. Without one the command refuses and names the options.
The engine still reads one chunk at a time with bounded memory, whatever the range, and records the range it used (after snapping to the method's boundaries) in the result and the provenance.

**Output.** In a terminal: a progress bar on stderr, then a result line (done, `cache hit` or `computed`, seconds, the range) and what was stored. With `--json`, or when stdout is not a terminal,
the JSON lines (`"stage"` keyed; the result line has `"cache": "hit" | "miss"`) are unchanged. `--quiet` prints only the result line; colour only in a terminal and never with `NO_COLOR`.
A failure is a message on stderr (a `"stage": "error"` object, with its `code`, under `--json`) and a non-zero exit status (1 a failure, 2 a usage error).

**The old spellings of the pipelines are not kept** (amended 2026-11-17, `docs/specs/2026-11-17-compute-and-run.md`, section 6.3): `compute <what>`, the old words (`psd compute`, `eigen solve`, `pyramid build`, `lowpass`, ...; the last column of the table below) and `run --list` are refused with exit status 2 and a message naming the new spelling (`'psd compute' is now 'run psd'`); no alias runs. The other old two-word spellings (`dataset list`, `convert subject`, `measure query`, `surface add`, ...) are refused the same way (nxr-neuro is unreleased: no deprecation period). The one-word `read FOLDER` is kept as a hidden alias.

### `run`: Run a pipeline on a subject: a built-in (psd, crossspectrum, phasegradient, measure, ...), a built-in definition, or a definition file

| command | what it does | old spelling (refused) |
|---|---|---|
| `nxr-neuro run noisecov` | The noise covariance of a recording's good channels. | `noisecov compute` |
| `nxr-neuro run inverse` | The depth-weighted minimum-norm inverse kernel, from a noise covariance and a head model. | `inverse compute` |
| `nxr-neuro run headmodel` | The sphere head model (gain matrix) for a subject's cortex, MEG and EEG. | `headmodel compute` |
| `nxr-neuro run pointspread` | The instrument's blur: unit currents at sample vertices through the head model and inverse kernel. | `pointspread compute` |
| `nxr-neuro run eigenmodes` | The Hodge-Dirac eigenmodes of a stored surface, with the intrinsic complex and the modes' full forms (cached); `--glued` for a cortex of two hemispheres. | `eigen solve` |
| `nxr-neuro run surface_frames` | The canonical (north, west) and anatomical frames of a surface. | `frames compute`, `run surface_frames` |
| `nxr-neuro run select_dyadic_surface` | The dyadic equal-area tiles of a surface as one grid with closed-form levels, from a stated finest tile (subject, group tree or transfer). | `tiles compute`, `run select_dyadic_surface`, `run select_dyadic_surface --rule group`, `run group_tiles` |
| `nxr-neuro run select_time_grid` | The dyadic time grid of a recording's line: tiles, band, frames (density and origin are options), constant-Q, segments, windows, chunks. | `constantq compute`, `run select_time_grid --rule frames` |
| `nxr-neuro run select_bands` | A band set on the Frequency or Eigenvalue line, each band a quantity. | `run pyramid_bands`, `run spatial_bands` |
| `nxr-neuro run select_product` | A product of grids and band sets (a box, cells, the joint partition), levels coupled to a band. | `joint compute`, `run select_product` |
| `nxr-neuro run glue` | The two hemispheres of a cortex glued by a bridge: a derived manifold with its map and bridge selection. | `--glued` of `run eigenmodes` |
| `nxr-neuro run roll_up` | A measurement rolled up through its grid's hierarchy, or across a group's subjects (scope group). | `run roll_up` |
| `nxr-neuro run band_analytic`, `cortical_band`, `member_activation`, `band_outer` | Field chains: pointwise fields (a band's analytic signal, the band on the cortex, one filter-bank member's activation, z z^H), run alone or composed into `measure`. | the steps of `pyramid`, `cortical_band_cells`, `opticalflow` |
| `nxr-neuro run lowpass` | A zero-phase low-pass filter, written as a new recording. | `lowpass` |
| `nxr-neuro run highpass` | A zero-phase high-pass filter, written as a new recording. | `highpass` |
| `nxr-neuro run bandpass` | A zero-phase band-pass filter, written as a new recording. | `bandpass` |
| `nxr-neuro run bandstop` | A zero-phase band-stop filter, written as a new recording. | `bandstop` |
| `nxr-neuro run notch` | A zero-phase notch filter (line noise and harmonics), written as a new recording. | `notch` |
| `nxr-neuro run sourcepsd` | The band power maps of the sources (the Welch PSD of the sources, grouped in bands). | `sourcepsd compute` |
| `nxr-neuro run sourcepsd_bins` | The power of the sources at every Welch bin of one band (nothing reduced over frequency), through a named kernel. | `sourcepsd compute --resolve bins` |
| `nxr-neuro run pca` | The principal components across the members of a stored statistic (members by features), with scores, explained variance and centring. | `pca compute` |
| `nxr-neuro run sourcetensor` | The band cross-spectral tensor of the current, its Stokes quaternion, normal power and share. | `sourcetensor compute` |
| `nxr-neuro run crossspectrum` | The sensor cross-spectral density per band (Welch, or per time window with the analytic estimator). | `crossspectrum compute` |
| `nxr-neuro run quadrature` | The quadrature spectrum of the sources per band: fraction, energy map and leading pairs (the definition `quadrature`). | (the definition) |
| `nxr-neuro run hodgepower` | The band power of the Hodge-Dirac eigenmode coefficients of a free-orientation kernel. | `hodgepower compute` |
| `nxr-neuro run speedspectrum` | The speed spectrum: mode energy per phase speed from the joint eigenvalue and frequency bank. | `speedspectrum compute` |
| `nxr-neuro run phasegradient` | The phase gradient of a band from a stored cross-spectrum: phase current, power, speed and travelling index. | `phasegradient compute` |
| `nxr-neuro run flow_modes` | The Hodge-Dirac flow weights of the apparent motion of one filter-bank member's source map (reads the stored amplitude by name). Its divergence, curl and speed are measurements: `run measure --set flow_max_amplitude` first (the unit-maximum amplitude), then the `measure` definition over the dyadic grid. | `flow_modes` |
| `nxr-neuro run cellquadrature` | Quadrature (travelling vs standing) per joint cell and per burst, from the stored cross-spectrum. | `cellquadrature compute` |
| `nxr-neuro run patches` | Patches on a static cortical map: where it stands out from its surroundings, at the size of its tiles. | `patches compute` |
| `nxr-neuro run scaleselection` | Each event's own size and rhythm: the best spatial and temporal level by standardised response. | `scaleselection compute` |
| `nxr-neuro run measure` | A field (stored, or a chain composed in) or a stored statistic (a PSD over its channels, schema 54) reduced over a selection's members (stored, or made by a maker), with presets. | `measure compute`, `run measure --set cortical_band_pyramid`, `run measure --set cortical_band_cells`, `run measure --set cortical_band_cells` (lists), `run cell_phase_current`, `run band_amplitude`, `run channel_mean`, `run crossspectrum_analytic`, `run measure --set surrogate_lagfree_envelope`, `run flow_amplitude`, `run measure_over_time` |
| `nxr-neuro run summarize` | A stored measurement summarised over its selection's members. | `statistic compute`, `run summarize` |
| `nxr-neuro run surrogate_lagfree_pairs` | Each burst's lagged share against lag-free Wishart draws of its own second moment (a definition). | — |
| `nxr-neuro run surrogate_lagfree_spatial_events_compare` | The spatial events against lag-free surrogate draws (early stopping): per-vertex p and q, the patch sizes. | — |
| `nxr-neuro run surrogate_lagfree_vortex_compare` | The vortex cores' tile crossings against lag-free surrogate draws (early stopping): per-tile p and q, the run repeat. | — |
| `nxr-neuro run repeat` | The vortex cores' run-to-run repeat: two recordings' stored tile crossings compared. | `repeat compute` |

**`run` takes every pipeline** (`nxr-neuro list pipelines` lists them): the hand-wired ones above, each dispatched to its own function with the options it always had, and the YAML definitions, shipped
(`pipelines/definitions/`, compiled in) or your own (`nxr-neuro run my_pipeline.yaml ...`, a path ending in `.yaml` or holding a `/`). A pipeline's name has no dot (a dot is a node's:
`<group>.<name>`), and a name is in one table only (a test checks it).

| command | what it does |
|---|---|
| `nxr-neuro run <built-in>` | one of the commands in the table above |
| `nxr-neuro run <definition> --dataset D --subject S [--<parameter> VALUE ...] (--from T --to T \| --all) [--replace] [--dry-run]` | a shipped definition (`band_bursts`, `gabor_bursts`, `global_bursts`, `cortical_band_cells`, `spatial_events`, ...) on a subject |
| `nxr-neuro run FILE.yaml ...` | your own definition, the same options |

A definition's parameters are options, `--<parameter> VALUE` with dashes for underscores (`--space-level 5`, `--band '[8,16]'`; VALUE is read as YAML); `--set NAME=VALUE` is gone (refused, naming
the new form). **Help is generated from the definition.** A parameter is written in the definition's long form,
`band: {default: [8.0, 16.0], meaning: "the time level's band", unit: Hz}` (and `choices:` for a closed list), and `nxr-neuro run <definition> --help` prints each parameter's meaning, default and unit.
The short form (`band: [8.0, 16.0]`, the default alone) is accepted for your own definitions; their help says the parameter is undocumented. A shipped definition must declare all three
(the test `cli_pipelines`). A parameter named as one of the verb's own options (`dataset`, `subject`, `from`, `to`, `all`, `replace`, `dry_run`, `json`, `help`, `quiet`, `no_color`, `human`, `version`,
`chunks`, `start`, `stop`) is refused when the definition is read. The time range is `--from/--to` or `--all`, as for every pipeline over a recording (`--chunks`, `--start` and `--stop` are hidden
developer options).

`nxr-neuro compute` is reserved for applying a node or a graph to arrays and files (a later step of the compute-and-run spec); today it prints that it is not available, and `compute <pipeline>` is refused
with the `run` spelling.

### `compile`: Compile a pipeline definition without running it

`nxr-neuro compile <definition> [--dataset D --subject S] [--set NAME=VALUE ...] [--signatures FILE.json] [--inputs FILE.json]` expands a definition (its `for:`, `when:`,
`chain:` and `make:` written out for the parameters), checks every connection before any data, and prints the **compiled graph**: the nodes (name, section, kind, every
parameter resolved), the edges (from an input or a node's output port to a node's input port), every value's type in the structured JSON form, the halo, the span grid, the
outputs, and the inputs to bind. Nothing is read beyond metadata and nothing is written. The library function is `nxr::pipelines::face::compile` (one function, one command;
the Python and WebAssembly faces will call it as `compile(definition, ...)`).
- With `--dataset` and `--subject` the inputs are found and typed from the subject's stored records exactly as `run` finds them (the dataset is opened read-only), and the graph is
  checked as `run` checks it: `bind` names the stored node each input that is not streamed is bound to, `streams` the recording the chunk loop reads. A stored field takes the
  type its record gives (its manifolds, cell, value type, components, unit, time line); a `kind` outside the quantity vocabulary (an imported map's) leaves only its quantity
  unset, so a node reading a named quantity refuses it and one reading any quantity takes it. A stored manifold given as an input is typed `manifold<name(id)>`, with its kind,
  size and coordinate system.
- Without them, `--inputs` gives each input's type (the structured form or the printed form); the graph is checked by the node signatures alone, so a node typed only by its
  plan (today `signal.psd`, among others) is refused, and `halo` is null.
- `--signatures` declares nodes another program implements (a renderer's sinks) by their signatures: checked with nxr-neuro's nodes, never run. A definition naming a node that is
  neither registered nor declared is refused; a declared signature never replaces a registered node.

A refusal is the checker's message (`<use> -> <output>: port <port>: <why> (<value> = <printed type>, from <use>)`), exit status 1.

### `list`: List datasets, subjects, the nodes of a subject (by class), recordings, kernels, results or pipelines

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro list datasets` | List the dataset folders in a folder (`--datastore`): every dataset of either form (`form`: `one-database`, or `catalogue` as `create dataset --dataset` makes it), with its id, name, path, database and subject count; an empty folder is an empty list, a folder that does not exist is `not_found`. | `dataset list` |
| `nxr-neuro list subjects` | List the subjects of a dataset. | `subject list` |
| `nxr-neuro list recordings` | List the recordings of a subject. |  |
| `nxr-neuro list kernels` | List the inverse kernels of a subject. |  |
| `nxr-neuro list results` | List the fields and statistics the computations stored for a subject. |  |
| `nxr-neuro list manifolds` · `fields` · `selections` · `operators` · `statistics` · `dictionaries` · `spaces` · `acquisitions` · `jobs` | The nodes of one class of a subject, one description each (or only the columns asked for), filtered and paged by a cursor (below). | |
| `nxr-neuro list pipelines` | List the pipelines `run` takes (the hand-wired ones and the definitions) and the steps the definitions are made of. | (`run --list`, refused) |

### `show`: Show a dataset or a subject

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro show dataset` | Show a dataset: its subjects by kind, its source and its creation date, and the count of each class of node (`nodes`). | `dataset show` |
| `nxr-neuro show subject` | Show how many manifolds, fields, operators and selections a subject holds. | `subject show` |

### `read`: Read stored arrays and derived readings

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro read subject` | Everything a subject holds as one index: every node's description, no arrays (below). | |
| `nxr-neuro read dataset FOLDER` | Read an nxr dataset folder: list its arrays, or read one array or region. | `read` |
| `nxr-neuro read crossspectrum` | Read coherency, phase, delay or phase-slope index from a stored band-resolved cross-spectrum. | `crossspectrum read` |
| `nxr-neuro read field` | A stored field by name: its values (a time-resolved field over a stated time range), its manifold and cells. | |
| `nxr-neuro read statistic` | A stored statistic by name: its values on its axes. | |
| `nxr-neuro read operator` | A stored operator by name: dense values, or a sparse operator's `data`, `indices`, `indptr`. | |
| `nxr-neuro read selection` | A stored selection by name: its per-element array, its spans and dictionary, or (`--measure M [--field F]`) its measurement table. | |
| `nxr-neuro read manifold` | A stored manifold by name: positions, cells, axes, element names, coordinate system; or (`--family NAME`) one labelling. | |

**`read <class>`** (spec `docs/specs/2026-11-19-io-service.md`, 4.1) reads ONE stored node by name or id: `--dataset D --subject S --name N`, the class named by the command (a name of
another class is refused, naming the command that reads it). Nothing is written: the dataset is opened read-only and every file is left as it was. The result line is the node's
record (`class`, `name`, `id`, unit, kind, producer, ...) with `arrays` [{name, dtype, shape}]. With `--out FILE.npy` the arrays are written as NumPy files (C order, little-endian,
the stored precision: a float32 node is read as float32) and the result says where (`files`): one array to FILE itself; several to `<stem>.<array>.npy` (a sparse operator:
`<stem>.data.npy`, `<stem>.indices.npy`, `<stem>.indptr.npy`, scipy.sparse.csr_matrix's arrays; a manifold: `positions` and `cells_<rank>`; spans: `start`, `stop`, `code`).
A sparse operator is written as three .npy files rather than one .npz: the command line has one array format (.npy, read and written by io itself; a .npz is a zip archive
io has no reader for), and the three arrays are exactly what `add operator --data --indices --indptr` takes back.
- **A part, and the read's budget.** `--region START:STOP,...` reads [START, STOP) along each leading axis of each array the read gives (a manifold's `positions` and `cells_<rank>`, a
  selection's spans, a statistic's values; a STOP past an array's extent is cut to it, and the result's `regions` {array: [[start, stop], ...]} say what each array read); the form is
  `add field --append`'s. A time-resolved field is read by `--rows` and `--from/--to` instead, and a sparse operator or field whole. `--budget BYTES` (default 1e9, 1 GB) is the most a
  read may hold: before any array is read (or made), the bytes of every array the read would give are added up and a read above the budget is refused, naming its size, each array's
  shape and dtype, and the `--region` (or, for a time-resolved field, the time range) that fits. A large curve of fibres, read whole, is refused this way at once instead of failing
  to allocate.
- `read field`: a **time-resolved field** (a recording, a source series) needs its time range, `--from S --to S` (seconds or m:ss) or `--all`, and is read through the chunk
  service one chunk at a time; `--rows I,J` picks rows in order, `--projectors active|none` (a recording's active projectors applied, or the samples as stored), `--shape-only` the
  description alone. A field with no time line is read whole. The result names the field's `manifold` (and `manifold_id`) and `cell` ("0" vertices, "1" edges, "2" faces).
- `read manifold`: `positions` [n x 3] and `cells_<rank>`, its `axes`, element `names`, and `coordinate_system` {name, unit, space} (the chart its positions are in).
- `read selection` of spans gives `start`, `stop` (in the line's unit) and `code`, one a member; a **tiling** (a closed-form spans selection: an origin, a width, a hop and a count, no span stored) gives
  the same arrays, its members made from its rule (the last cut at the line's end; `code` the member number), and its rule stays in `params`.

### Queries over a dataset: `read subject`, `list <class>`, the counts of `show dataset` (`io/src/commands_query.cpp`; spec `docs/specs/2026-11-20-data-layer-queries.md`, 3.1 and 3.2)

nxr-neuro is the data layer: a client (an application's interface, an analysis script) asks it what a dataset holds and gets nxr-neuro's typed description of each node, never a row of the
catalogue. The schema and the store layout can change without changing these results. Every call opens the dataset read-only and leaves every file as it was; all of it runs in io (no SQL
in a face), so the command line, Python (`nn.read.subject`, `nn.list.fields`, ...) and JavaScript (`nxr.read.subject`, ...) give the same result.

**`read subject --dataset D --subject S [--classes manifolds,fields,...] [--related]`** returns
`{"schema": "nxr-neuro-subject@1", "subject": {id, name, kind, template}, "manifolds": [...], "fields": [...], "selections": [...], "operators": [...], "statistics": [...],
"dictionaries": [...], "spaces": [...], "acquisitions": [...]}`, each array the descriptions of that class in name order (`--classes` keeps only the classes named). A description is named, nested
and typed: a node is named, never numbered; what it refers to is named by its name (`"manifold": "sensors"`; a node of another subject is `SUBJECT:NAME`); no key is a column of the schema
(`*_id`, `*_json`). Only complete nodes are described. `id` is the node's identity (stable across calls). **Every typed node carries `type`: the JSON `compile` gives a value** (the type
system's: `class`, `domain` with its factor manifolds (`kind`, `topology`, `coordinate_system` {space, name, unit}, `n`, `origin`, `spacing`, `unit`), `value` (real, complex, ...), `components`, `unit`,
`quantity`, `sfreq` for a stream); a manifold's `type` is its manifold term. It is built the way the runner types a stored input, so a client reads a stored node and a compiled graph's value in one
vocabulary, and the same stored field has the same `type` in `read subject` and in `compile`. The storage words (`value_type`, `cells`, `data_type`, `layout`, and a domain, unit or components of
their own) are not keys: `type` says them. `kind` is the stored kind (a field's, an operator's, a selection's type, a manifold's type); `type.quantity` is the typed quantity. The other keys:

| class | description (besides `id`, `name`, `kind`, `type`) |
|---|---|
| manifold | `label`, `description`, `primary`, `acquisition`, `coordinates` {system, unit, kinds, axes} (the coordinate system its points are in; `kinds` when each point of a line carries a code; `axes` [{name, size, origin, spacing, unit}], the coordinate of a closed-form geometry along each axis, a line's one: its first point, the step between points and the unit), `size` {vertices, edges, faces, volumes}, `factors` (a product's, by name), `component_of`, `component_ordinal`, `from_index` (a component's first vertex in the whole's vertex order), `derived_from`, `derived_by`, `stored`, `provenance` |
| field | `label`, `acquisition`, `description`, `stored`, `lineage` {derived_from, of_field, of_operator, by_operator, function, contributions [{field, weight}]}, `provenance` |
| selection | `label`, `acquisition`, `description`, `elements`, `members` (their count), `member_of`, `member_ordinal`, `code` (a member of a set, such as each atlas of the set of a surface's parcellations or each level of a grid, is a selection of its own, described like any other: the set it belongs to by name, its place in it and its code; null for a selection that is no member), `of_field`, `weights` (one weight a member is stored), `stored`, `dictionary` (its name), `entries` [{code, name, colour, label, description}], `provenance` |
| operator | `label`, `acquisition`, `description`, `from` and `to` {manifold, selection}, `method`, `parameters`, `shape` {rows, columns, nonzeros}, `stored`, `lineage` {derived_from, of_statistic, inverts, inverse_kind}, `provenance` |
| statistic | `measure`, `method`, `label`, `acquisition`, `description`, `of_field`, `of_statistic`, `partition`, `members`, `members_skipped`, `dof`, `axes` [{name, manifold, selection, size, unit}], `stored`, `provenance` (its `type` is untyped past its class, value and unit: its domain is typed by the node that made it) |
| dictionary | `name`, `scope`, `description`, `entries` [{code, name, colour (#rrggbb), label, description, measurement, unit, function, params}], `provenance` (every dictionary the subject sees, each with its `scope`: the subject's own, the dataset's, and every app-scope one, whether or not a selection of the subject reads it); no `type` |
| space | `name`, `geometry`, `rank`, `parameters`, `description`, `coordinate_systems` [{name, unit, reference, transition, description}]; no `type` |
| acquisition | `name` (its label), `session`, `task`, `run`, `acq`, `processing`, `split`; no `type` |

`provenance` is {source, comment, created_utc, created_by, modified_utc, modified_by, producer}: `producer` is the node's producer record as `read <class>` gives it (the command or pipeline that made it, the program, its inputs, and what the maker gave with `--producer`; null when none; a selection's is kept in its parameters, a statistic's provenance has no source or comment). `type` is built by the pipelines' face (`face::read_subject`, `face::list_class`); io reads and pages
the rows (a build without the pipelines gives `"type": null`; a node the type system refuses to type, such as a tangent field with no gauge record, gives `"type": {"refused": "<why>"}`).

`--related` adds `"related": [{subject, kind, relation, description}]`: the subjects this one refers to, with `relation` `template` (the template its anatomy is taken from), `source` and
`target` (the owners of the manifolds its operators map from and to: a registration's two templates) and `domain` (of the manifolds its fields lie on), each with its description as far as
the database holds it (a one-database dataset holds the whole subject; a package holds the anchor rows of its template).

**`list <class> --dataset D [--subject S] [--name TEXT] [--kind TEXT] [--on MANIFOLD] [--limit N] [--after CURSOR] [--columns a,b,...]`**, `<class>` one of `manifolds`, `fields`,
`selections`, `operators`, `statistics`, `dictionaries`, `spaces`, `acquisitions`, `jobs`, returns `{"schema": "nxr-neuro-list@1", "class", "subject", "rows", "next"}`. A row is the class's
description, exactly as `read subject` gives it, with the `subject` it belongs to first; with `--columns` it holds only the named keys of the description (and `subject`), in that order,
and a name that is not one is refused with the valid ones. A catalogue of packages is listed one subject at a time (`--subject` is required); a one-database dataset without `--subject` lists
every subject's nodes. The filters are named options, never SQL: `--name` the nodes whose name contains TEXT (case-insensitive; `%` and `_` are ordinary characters), `--kind` a field's kind, a
manifold's, selection's or operator's type or kind, a statistic's measure, a dictionary's scope, a space's geometry or a job's kind, `--on` the nodes that lie on the manifold of that name (a
field on it or on a product of it, a selection on it, an operator from or to it, a statistic with an axis along it, the manifolds built on it: its products, components and derivations).
Rows come in name order; `--limit N` gives N rows a page and `next`, an opaque cursor (null after the last row), is passed back as `--after` for the following page: every row comes exactly
once. `list jobs` reads the `job` table (what ran, when, the program and the outcome): rows come from `run` and `import` (see *Writes in pieces, deletes by class and jobs*).
`list recordings`, `list kernels` and `list results` are unchanged.

**`show dataset`** gains `nodes`: {manifolds, fields, selections, operators, statistics, dictionaries, spaces, acquisitions, jobs}, the count of each class summed over the dataset's
subjects (as `list` returns them); for a catalogue it opens each package read-only, and one that cannot be opened is named in `nodes_unavailable`.

### `query`: Query stored measurements

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro query measurements` | Query a stored measurement at any spatial and temporal scale, stated as quantities (a tile in mm, a duration in s, a band in Hz). | `measure query` |

### `export`: Export stored results: measurements as a table, cortical maps as GIFTI

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro export measurements` | Export stored measurements as a table for statistics (a row per subject), with its dictionary and provenance. | `measure export` |
| `nxr-neuro export gifti` | Write a cortical Field (a map, or a series over a stated time range) or a labelling as GIFTI (.func.gii, .shape.gii, .label.gii), with the surface on request and a provenance sidecar. | - |
| `nxr-neuro export tsv` | Write stored measurements (a row per member of a partition, a column per measure) or a statistic (long form) as a BIDS-style TSV with a JSON sidecar of units, levels and provenance. | - |
| `nxr-neuro export fif` | Write a source map or a source series on the two hemispheres of a cortex as MNE source estimates (-lh.stc and -rh.stc), with a provenance sidecar. | - |
| `nxr-neuro export bids-derivatives` | Write results (maps, series, labellings, statistics, measurements, source estimates) as a BIDS derivatives folder: files named with BIDS entities, sidecars with Sources and provenance, and a dataset_description.json listing nxr-neuro's suffixes. | - |

### `import`: Import data from Brainstorm into a dataset

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro import list PROTOCOL` | List the subjects of a Brainstorm protocol. | `convert list` |
| `nxr-neuro import scan PROTOCOL` | Scan a Brainstorm protocol and describe what it holds. | `convert scan` |
| `nxr-neuro import subject PROTOCOL` | Import a Brainstorm subject's anatomy into a dataset. | `convert subject` |
| `nxr-neuro import template PROTOCOL` | Import a Brainstorm protocol's template subject into a dataset. | `convert template` |
| `nxr-neuro import group PROTOCOL` | Import a Brainstorm protocol's group subject into a dataset. | `convert group` |
| `nxr-neuro import templates TEMPLATES_DIR` | Import the fsaverage reference templates (and optional fibre tracts) into a dataset. | `convert templates` |
| `nxr-neuro import surface` | Import Brainstorm tessellation files as surfaces of a subject. | `convert surface` |
| `nxr-neuro import mri` | Import an anatomical MRI of a subject. | `convert mri` |
| `nxr-neuro import recording` | Copy a recording file (FIF, EDF, BDF, BrainVision, EEGLAB, CTF, KIT) into a subject package, with its channels, events, bad channels and an origin record. `--in-place` links the file instead of copying its samples: the chunk service streams from the original, the package records its path and SHA-256, and a changed file is refused. `--anonymize` drops the identifying header fields. | - |
| `nxr-neuro import fif` | Import an MNE FIF object into a subject package: a source space (a surface source space is the full surface in its frame plus an `indices` Selection of the used vertices; a volume one a lattice in its voxel chart; a discrete one a points manifold), a covariance (a statistic on the channels' Selection), a forward solution (an operator, source to channels, with both Selections), an inverse operator (its parts as operators, fields and the noise-covariance statistic; no kernel), events and annotations (added to `--recording`, never overwriting its events; annotations keep their channels) or a head-to-MRI transform (a transition operator). The kind is found from the file; channels are matched by name in `--channels`. `--anonymize` drops the file names and command line the files record. | - |
| `nxr-neuro import nifti` | Import a NIfTI-1/2 volume into a subject package: a lattice manifold in its voxel chart `voxel:<name>_volume` (the voxel size as spacing in mm), its intensities as a Field in C order (or, with `--parcellation NAME`, a codes Selection: a label volume is imported only when named; `--label-names` names its entries), and the file's voxel-to-world transform as a transition operator in millimetres to the system its transform code names (1 `scanner-ras`, 2 `nifti-aligned`, 3 `talairach`, 4 `mni`; code 0: no transition, nothing guessed). A series of volumes is refused. | - |
| `nxr-neuro import transform` | Import a head-to-MRI transform (an MNE `-trans.fif`, either direction) as a transition operator from the head frame to FreeSurfer's surface RAS (`freesurfer-tkr`, metres). | - |
| `nxr-neuro import bids` | Import a BIDS dataset (MEG, EEG, iEEG) as a new dataset of subject packages: one package per `sub-<label>`, each recording imported through `import recording` (copied, or `--in-place`), `participants.tsv` as the participant variables (so `--where` works), `channels.tsv` (types, bad channels), `events.tsv` (the recording's events) and `electrodes.tsv` with `coordsystem.json` (channel positions; the coordinate system is recorded verbatim, never interpreted). The dataset's licence and authors are the origin of every item. | - |
| `nxr-neuro create recording --synthetic` | Make a small synthetic recording in a subject package: sine waves (`--frequency`, repeatable) plus white noise from a `--seed`, with `--channels`, `--sfreq` and `--duration` as options. Stored as an imported recording is stored, with an origin record saying it is synthetic (CC0-1.0, redistributable); the same seed and options give the same samples. Used by the release smoke test. | - |
| `nxr-neuro import condition PROTOCOL` | Import a Brainstorm condition (its recordings, channels and head models) into a dataset. | `convert convert` |
| `nxr-neuro import inverse` | Import a Brainstorm inverse kernel as an operator of a subject. | `convert inverse` |
| `nxr-neuro import map` | Import Brainstorm source or matrix maps as fields of a subject. | `convert map` |
| `nxr-neuro import resample` | Resample a recording to a new sampling rate, as a new recording. | `convert resample` |
| `nxr-neuro import badsegments` | Import a recording's bad segments from a file. | `convert badsegments` |
| `nxr-neuro import events` | Import a recording's events from a file. | `convert events` |

**MATLAB v7.3 files in the WebAssembly build.** A `.mat` saved with `-v7.3` is an HDF5 file. The native builds read it (matio over HDF5), as before. The WebAssembly build has no HDF5
(and is 32-bit, with 4 GB at most), so there `import subject`, `import condition`, `import template` and `import group` **skip each v7.3 file and import the rest**, instead of
rolling the whole subject back: a fibres file, a non-primary surface, a subject image, a recording, a time-frequency file, a kernel (or its head model) or a map. Each skipped file is
named with the reason "MATLAB v7.3 (HDF5): not readable in this build; import it with a native build" in the command's result (`skipped` [{what, reason}] and a `message` naming
them), as a `skipped` progress line, and in the import's job row (its `result`, which `list jobs` reads). The remedy is the native command line or Python package (or re-saving the
file as v7). Nothing else is skipped silently: a file that cannot be read for another reason is refused as before, and so is a v7.3 file the import cannot go on without (the
condition's channel file, the surface its kernels index, a template's only cortex), named with the same reason.

### `create`: Create a dataset or a subject

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro create dataset` | Create a dataset: a folder holding a catalogue of subject packages (the default form). --datastore makes the old one-database form. | `dataset create` |
| `nxr-neuro create subject` | Create a subject (or a template or group subject) in a dataset. | `subject create` |

### `delete`: Delete a dataset or a subject

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro delete dataset` | Delete a dataset: its folder, inside the folder given by `--datastore`. | `dataset delete` |
| `nxr-neuro delete subject` | Delete a subject, or one of its nodes (`--node CLASS:NAME`), from a dataset. | `subject delete` |

### `compact`: Compact a dataset

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro compact dataset` | Pack a dataset's small files and shards to save space (a dry run shows what would change). | `dataset compact` |

### `recover`: Recover a dataset after an interrupted write

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro recover dataset` | Finish or roll back an interrupted write in a dataset, and roll back the unfinished writes in pieces. | `dataset recover` |

### `reduce`: Reduce stored results across subjects

`reduce pyramid` is gone: the group reduce is the definition `run roll_up` (group scope; `--subjects A,B,C`, and `--into` for a catalogue dataset).

### `add`: Add to a subject what was computed elsewhere

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro add surface FILE` | Add a mesh file (OBJ, PLY or STL), or `--vertices V.npy --faces F.npy` with `--name`, to a subject as a surface. | `surface add` |
| `nxr-neuro add field` | Values on a manifold (`--manifold M --values V.npy`), optionally time-resolved on a line (`--time LINE`); its lineage with `--by-operator`, `--contribution SUBJECT:FIELD[:WEIGHT]`, `--producer JSON`. | |
| `nxr-neuro add selection` | A partition or set (`--members`), a pick (`--indices`) or spans on a line (`--spans`), `--type` naming which; `--of-field`, `--weights V.npy`, `--producer JSON`. | |
| `nxr-neuro add statistic` | Values on named axes (`--axes NAME:MANIFOLD ...`), with `--measure` and `--method` from the registries, of a field (`--of-field`) or detached. | |
| `nxr-neuro add measurements` | One value a member of a selection for each measure (`--measure NAME=V.npy ...`), of a field (`--field`) or the members' own. | |
| `nxr-neuro add operator` | A map between two manifolds, dense (`--values` [to x from]), sparse CSR (`--data --indices --indptr`) or analytic (`--layout analytic --method NAME --params JSON`, no matrix); `--from-selection`, `--producer JSON`. | |
| `nxr-neuro add points` | A manifold of points: positions [n x 3] or one coordinate each [n]. | |
| `nxr-neuro add line` | A closed-form line: `--n`, `--spacing`, `--origin`, `--unit`; `--dimension time` (seconds) makes a time line fields can be time-resolved on. With `--coordinates C.npy --dimension eigenvalue [--kinds K.npy]`, a line whose points are not evenly spaced. | |
| `nxr-neuro add product` | The product manifold of stored factors (`--name N --factors A,B`: a line over time, a surface over time). | |
| `nxr-neuro add dictionary` | A dictionary of codes of the dataset or a subject (`--name N --scope dataset\|subject --entries E.json`). | |

### `edit`: Edit what a subject holds without touching its arrays

| command | what it does | old spelling |
|---|---|---|
| `nxr-neuro edit manifold` | A manifold's label, description and primary flag (`--name N [--primary yes\|no] [--label TEXT] [--description TEXT]`). | |
| `nxr-neuro edit dictionary` | One entry of a dictionary (`--name N --entry CODE [--name-of] [--label] [--colour] [--description] [--params]`), or every entry (`--entries E.json --replace`). | |

**`add <class>`** (spec `docs/specs/2026-11-19-io-service.md`, 4.2) stores what a user computed, from .npy files, as ONE unit (all of it or nothing) through io's typed writes:
- the arrays keep their precision (float32 stays float32; integers are stored as int32, refused when a value does not fit; a manifold's coordinates and measurement rows are float64);
- they are checked against what they lie on, and a mismatch is refused saying what was expected: a field's first axis is its manifold's vertices (or edges, faces by `--cell`), a
  time-resolved field's second its line's samples, a value type's components trail (complex [.. x 2], vector3 [.. x 3]); a partition's labels one an element; an operator
  [to x from]; a measure one value a member;
- each node records a producer naming the command, the program and the input files by NAME (never their paths), with `--description`;
- a name already used in the subject is refused unless `--replace`, which moves the old node aside (never deletes it: a selection's weights array moves with its values) and says where (`replaced`).

### Writes in pieces, deletes by class and jobs (spec `docs/specs/2026-11-20-data-layer-queries.md`, 3.7 and 3.8; `io/src/commands_pieces.cpp`, `commands_delete.cpp`, `jobs.cpp`)

**`add field | statistic` in three steps** for values too large to hand over at once. The three steps are ONE unit: until the commit the node is not in the subject (`read subject`, `list <class>` and
`read <class>` do not see it).
- `--begin` takes the full `--shape N,N,...` of the values (as the one-shot `--values` would be) and the stored `--dtype` (float32, float64, int32, int64, uint8 or bool; no default: precision follows the
  caller's data), with the command's usual options (`--subject`, `--name`, `--manifold`, `--time`, `--value-type`, `--axes`, ...). Everything the one-shot command checks without the values is checked now
  (the manifold, the name, the shape, the axes). It prints a write id.
- `--append --write ID --region START:STOP,START:STOP --values R.npy` writes one region, `[START, STOP)` along each axis, comma separated (`0:5000,0:3`); the values are the region's own shape and the begin's
  dtype. A region **may not overlap one already written**, and a region given twice is refused as such (a region is written once). Regions may come in any order.
- `--commit --write ID` stores the node by the one-shot command's own unit, once every value of the shape is covered; otherwise it is refused and names the parts missing (`[2:3, 0:20]`, at most six, and
  the count). The result is the one-shot command's, with `write` and `regions`. The stored values equal the one-shot's byte for byte (`test_pieces_deletes_jobs`).
- **Where an unfinished write lives:** in io's staging folder `<dataset>/.writes/<write id>/` (`write.json`: what the command was asked, the shape, the dtype and the regions written; `values.npy`: the values, a
  sparse file, zeros where nothing is written), outside the database. So the write id works across separate command-line processes. The commit loads the staged values as the one-shot command loads
  its file (memory: the whole array at the commit; the appends hold one region each).
- **Rolled back, never stored:** `recover dataset` removes every unfinished write (`writes_rolled_back` in its result; on a catalogue it removes only these, the packages recover when opened); a handle
  (`nn.open(..., mode="write")`, `nxr.open`) rolls back the writes it began when it closes, and a writable open removes the writes of a handle whose process is gone. A write begun through the command line
  (no handle) waits for its commit or `recover dataset`.
- **Handles:** a write begun through a handle belongs to it: another handle or process is refused (`belongs to another handle`). Through the handle `--write` may be left out while it holds one
  unfinished write (several: refused, naming them).
- **Refused:** a region outside the shape or with start >= stop; one given twice or overlapping one written; values of another shape or dtype than the region and begin; a commit with regions missing;
  a write id that is unknown, committed, rolled back, another handle's, or of the other class (a field's write committed as a statistic).

**`delete subject --node CLASS:NAME`** removes one node by its class (manifold, field, selection, operator, statistic or dictionary) and name, since names repeat across classes. io removes what the node owns:
a manifold's geometry and topology, a field's arrays and the measurements made of it, a selection's members and its weights array, and an operator's by-products (its eigenvalues). The operator a field was made by
(an inverse kernel, a forward model) is a stored node of its own and is never removed with the field: it is deleted only when named. It **refuses a node other nodes depend on, naming them** (`error: manifold sensors is used by field f1, operator mix: delete those first`):
for a manifold, the fields, selections, operators, statistics, products and derived manifolds on it; for a field, the fields derived from it, the selections and statistics that name it as theirs and the fields
mapped from it; for a selection, the operators, statistics and joint selections that read it; for an operator, the fields made by it and the operators derived from or composed of it; for a dictionary, the
selections that read it. The old `--node NAME` form (a store path, or a name) is kept: a path is unique; a name only when one class holds it, else the delete is refused (`ambiguous`) naming the
`CLASS:NAME` forms. A catalogue's node is deleted in its package; deleting a whole subject still needs a one-database dataset.

**Jobs.** `run` and `import <what>` leave a row in the dataset's `job` table (a package's, for a file importer), written after the work in its own short unit: `kind` (`run:psd`, `import:recording`),
`params_json` (the pipeline and its parameters as given; the import's options), `inputs_json` (the nodes a run read), `result_json` (what it stored), `requested_by` (the program), `started_utc`,
`finished_utc`, `status` (`done`, or `failed` with the `error` message), and the subject. A run whose result was a cache hit did no work and leaves no row; a dry run and a plan refused before any
data was read leave none; a run made inside another command (a surrogate's draws) leaves none. A request refused as given (a usage error) did not run: no row. A failure while it ran is recorded and the
error goes on. When the row cannot be written, the outcome of the work stands (a note, or `job_error` in an import's result). The rows are not content: the package hash does not cover them. `list jobs`
reads them back (filter `--kind run:psd`).

**The options of `add`, dictionaries and `edit`** (spec `docs/specs/2026-11-20-data-layer-queries.md`, 3.3 to 3.6; each is one unit, in `io/src/commands_add.cpp` and `commands_edit.cpp`; the
refusals store nothing). Each option is stored in a column the schema already has:
- **`add line --coordinates C.npy --dimension eigenvalue --unit U [--kinds K.npy]`** stores a line whose points are not evenly spaced: the coordinates [n] (float32 or float64, stored as float64)
  must be strictly increasing, else refused; its topology is a path and its geometry stored, in a coordinate system of the dimension's name (a line space). `--kinds` is one integer code a point,
  stored beside the coordinates and named in the manifold's producer record. Only `eigenvalue` (an operator's spectrum) is accepted: the layout places a line by its role (time, frequency or
  eigen), and a field on the product of a line with another line needs a folder; `read manifold` gives `positions` (the coordinates) and `kinds`; `read subject` describes the line with
  `coordinates` {system, unit, kinds}.
- **`add product --name N --factors A,B`** makes the product manifold of stored factors in axis order; a factor that does not exist is refused (`no_such_row`), as is a name in use or factors that
  already make a product under another name. A field added on a factor with `--time` makes the same product.
- **`--by TEXT`** (every `add`: surface, field, selection, statistic, measurements, operator, points, line, product, dictionary; `by` in Python and JavaScript) is the author recorded as the node's
  `created_by` (and `modified_by`): the person or the application that computed the values. Without it the author is the program (nxr-neuro and its version); the producer record names the program
  either way (`program`). A write in pieces keeps it from `--begin` (or takes it at `--commit`).
- **`add field --by-operator NAME`** records the operator that made the field (its lineage `by_operator`); **`--contribution SUBJECT:FIELD[:WEIGHT]`** (repeatable) names the fields a mapped field is
  made from (the subject by name, in the dataset; the weight a trailing number, default 1) as its `contributions`; **`--producer JSON`** adds the maker's facts to the producer record (not under the
  keys the command records itself: `command`, `program`, `inputs`, `output`, `cache`).
- **`add selection --of-field NAME`** names the field (a recording) the selection belongs to; **`--weights W.npy`** is one weight a member (a set's labelled members, the picks, the spans), stored beside
  the selection and named in its parameters (`weights`; `read selection` gives the array `weights`, `read subject` the flag `weights`); **`--producer JSON`** as above.
- **`add operator --layout analytic --method NAME [--params JSON] [--from-selection NAME]`** stores an operator by its method, with no matrix (`stored` false, no array to read); `--from-selection` names the
  selection of the from manifold it reads, for any layout; **`--producer JSON`** as for `add field`.
- **`add dictionary --name N --scope dataset|subject [--subject S] --entries E.json`**: the entries are a JSON list (the file, or the text itself; in Python and JavaScript a list of objects) of
  `{code, name, label, colour, description, params}`; codes are unique and a colour is `#rrggbb`, else refused; a dictionary of that name and scope that exists is refused. `--subject` opens the unit
  (required for a subject scope and for a catalogue of packages; a one-database dataset's dataset-scope dictionary needs none). The label and the description are kept with the entry's attributes; `read
  subject` and `list dictionaries` give each entry as `{code, name, colour, label, description, measurement, unit, function, params}`.
- **`edit dictionary --name N [--subject S] --entry CODE [--name-of TEXT] [--label TEXT] [--colour #rrggbb] [--description TEXT] [--params JSON]`** changes one entry (an empty text clears a label or description);
  **`--entries E.json --replace`** replaces every entry, and refuses to drop a code a selection reads. The subject's dictionary of that name is found first, then the dataset's; the app's are the program's own and are not edited.
- **`edit manifold --name N [--primary yes|no] [--label TEXT] [--description TEXT]`**: making a surface primary clears the previous primary surface of the subject in the same unit; only a surface is
  primary, else refused. A manifold's `label` and `description` are in `read subject` (`description` is the manifold's comment).

Every pipeline command writes JSON lines (progress events, then one `"stage": "result"` line with `"cache": "hit" | "miss"`); a rerun with the same options is a cache hit, and `--replace` recomputes. Details of the dataset commands and of the format: `io/README.md`.

## The default subject (`io/include/nxr/io/default_subject.h`; spec `docs/specs/2026-11-11-default-subject.md`)

Every subject of a dataset is registered to one average anatomy, Brainstorm's `@default_subject`. `dataset create --templates <FreeSurfer subjects folder>` writes it with the dataset (both, or
neither): a subject `@default_subject` of kind `template`, written as any subject is, holding fsaverage5's `cortex_pial` (primary, with its hemispheres and `sphere.reg`), `cortex_white` and
`cortex_inflated` in metres (`--default` picks another resolution), and the pial's parcellations: the selection set `source/cortex_pial_atlases` with Structures, Desikan-Killiany
(`label/{lh,rh}.aparc.annot`) and Destrieux (`aparc.a2009s`), each a partition (a code per vertex) whose app-scope dictionary names "<region> L|R" with FreeSurfer's colours, as nxr's default
subject writes them. `--modes K` (with the pipelines built) also SOLVES the pial's Hodge–Dirac basis with `run eigenmodes`'s pipeline (about K modes; the glued surface, its bridge, the eigen axis and the eigenvectors, stored as below); without it,
run `nxr-neuro run eigenmodes --dataset D/N --subject @default_subject --surface cortex_pial` on the default subject. io never solves: the basis is compute's, so it is made by the pipelines.
With the pipelines built it also writes the pial's **frames** (`run surface_frames` on the default subject: the canonical frames, north / west from its registered sphere, and its
anatomical frame from FreeSurfer's RAS axes, the template every subject's anatomical frame is carried from) and the **group tree**: the dyadic ladder of the default pial (`cortex_pial tiles`, Selection type `ladder`, decision D85; its levels
`cortex_pial tiles L0 … L<finest>` are partitions, one tile code `h·2^l + t` per vertex), as `run select_dyadic_surface` writes any grid but with `tree: "group"` (`run select_dyadic_surface --rule group`, a definition: `dataset create` runs it) and the dataset's default atlas
definitions in its params (`atlas`: leaf depth `--atlas-depth`, 8 by default; resolution; frames (the canonical frames' id); the time convention, the line's time grid (levels of 2^k samples from the first sample, closed form); the default bands, octaves 1–64 Hz; the Levi-Civita levels; the trajectory scales), the settings every subject's atlas and the group sums share. On fsaverage5 its
codes equal the Studio's `surfaceLadder` and nxr's at every level. The group tree carries the **Levi-Civita connection** between its tile centres at `--lc-levels` (4,6 by default), as nxr
stores it: the centres (`cortex_pial_lc_centres_L<l>`) and Ω[a, b], the angle at centre b of centre a's north carried there, as `levi-civita connection` measurements on level l's Selection.
Its antisymmetrisation is not nxr's: nxr's halves an angle on the circle, which turns every |Ω| > 90° by 180° (a quarter of the pairs on fsaverage5; shown on exact spheres, where Ω is known).
The transport between distant tiles depends on the path and on the mesh (its warnings say so). The canonical frames follow the format's definition (north = the closed-form meridian of the subject's registered sphere, pushed to the cortex by 1-ring Jacobians). They equalled nxr's to 0.003° (median; max 1.3°) only on fsaverage5's regular sphere: nxr's sphere solve departed from the meridian on warped registered spheres (median ~1°, max 25-52°); fixed in nxr #87 (default `field=meridian`, the solve kept as `field=solved`).
Subjects created later point at the default subject (`template_id`). Not yet: other resolutions, HCP-1065, the Frequency manifold.

## Reading recordings: the chunk service (`io/include/nxr/io/chunk_loop.h`)

A recording can be hours long, so it is never loaded whole. The chunk service, a port of the Studio's `ChunkSource` and `chunkLoop` (nxr: `nxr/io/chunks.py`), is the
one place a recording is read and looped over.
- **The chunk.** 2^15 samples of the time axis, every channel. Chunk k is samples [k·2^15, min((k+1)·2^15, n)), and the last is partial.
  A recording's chunk width is its row's `chunk_shape_json[1]`, never more than 2^15.
- **The span of a run.** `--chunks K` (one chunk), `--chunks A-B` or `--chunks all` (the whole recording, as a loop over its chunks), or an explicit
  `--start SEC --stop SEC`. **The default is chunk 0, never the whole recording.**
- **Time is one dyadic tile partition.** A tile of level k is samples [m·2^k, (m+1)·2^k) of the recording's line, counted from its first
  sample (m, the tile's ordinal, is its code); a chunk is level 15, a block level 17. A run's span is a run of whole tiles of the level it measures on (its
  window or Welch segment): `--start/--stop` snap to the nearest tile boundary (`nxr::io::SpanRequest::level`, `resolve_span`), an end at the recording's end
  stays there, a span that snaps to nothing is refused, and the producer record keeps the seconds asked, the snapped extent and the level
  (`analysis_window`: `asked_s`, `start_s`/`stop_s`, `tile_level`). Pointwise verbs (filters, frames, the wavelet transforms) snap at level 0,
  a sample. Spec: the time-tile amendment of `docs/specs/2026-11-13-cycle-pattern.md`.
- **What a pipeline does.** It declares the recording, its rows and its span, and folds each chunk handed to it into its accumulators: Welch, the covariance, the
  filters, the wavelet bank. The maths never sees a chunk, only a matrix. A computation's own windows are its own choice: a Welch segment's length, taper and
  overlap set the frequency resolution and are carried across chunk edges, so a result does not depend on the chunk width (tested).
- **The projectors.** An import stores the samples as recorded and the recording's projectors (SSP, ICA) as Operators with their status. The chunk service applies
  the ACTIVE ones (status 1) to every block it loads, by default, as Brainstorm's `in_fread` and MNE-Python's reading do: the selected SSP vectors stacked and
  re-orthogonalised together (a vector whose singular value is below 1% of the largest is dropped), the others (ICA) in the file's order. `--projectors none` reads
  the samples as recorded. What was applied is in each run's provenance (`run.projectors`) and, when the recording has active projectors, in its cache key. A
  projector is idempotent (P P = P), so samples that already carry one are read back unchanged up to rounding.
- **What is recorded.** Each producer record states the span (`analysis_window.chunks`) and the chunks read (`chunks_read`, first and last).
- **The halo** (the Studio's `haloOf`): samples of context a computation needs on each side of its span, so its edges are computed with real samples, never a
  reflection. The service reads them from the neighbouring chunks and clips them where the recording ends (`with_halo`). Its size is the computation's: optical
  flow's member asks for its taps (`haloOf`); a Welch spectrum asks for n - 1 samples to the RIGHT (an accumulating node's halo: a segment belongs to the chunk that holds its start and reads into the halo; it never reaches past the span's end, and nothing is carried between chunks).
- **The job** (the Studio's `chunkLoop`):
  - Every chunk is read **whole** (the Studio's residency) and handed over cut to the span and its halo.
  - Chunk k + 1 is **read while chunk k is folded**. The memory held is two chunks.
  - **Progress:** a `"stage": "chunk"` line after each chunk.
  - **Cancel:** Ctrl-C stops the job cleanly. Nothing of it is stored (`code::cancelled`), and a second Ctrl-C kills the process.
  - **One job at a time per folder of datasets:** a queue file, `.nxr-chunk-queue.lock`, in the folder that holds the dataset's folder, held for the job.
  - The command line sets these once (`set_chunk_loop_defaults`); no pipeline passes them. In the WebAssembly build, without threads, the next chunk is read when
    it is needed and the queue is the Studio's.
- **The guard.** The test `layering_recordings_read_by_chunk_service` fails if code outside io reads a recording or works out chunks itself.
- **The filters.** They walk every chunk, because they write a whole new recording. Their span is to be decided by the maintainers.

## The constant-Q time ladder (`io/include/nxr/io/constant_q.h`; `run select_time_grid --rule constant_q`)

**Why.** Time is partitioned as space is (the cardinal rule): measurements over windows of a recording need the windows to be a Selection. The
Studio's time-tile policy defines them once, and nxr-neuro ports it.
- **The reference:** the Studio's `backend/src/db/constant-q.ts`, line for line. Its rows are those the Studio's script `backend/scripts/constant-q.mjs`
  writes.
- **The policy:**
  - bands at 6 voices an octave from the top down;
  - a band's window holds `cycles` cycles of its lowest frequency, rounded up to a power of two of samples, so a coarse tile is the union of finer ones
    and no tile straddles a stored chunk;
  - one level per distinct window (D84: a tiling of time; what a window resolves in Hz is derived and references nothing);
  - the `hertz` anchor (default) puts the octave edges on 1, 2, 4… Hz, capped at 64 Hz and floored at 0.5 Hz; `cycles` pins the cycle count instead.
- **The rows:**
  - a `ladder` Selection on the recording's time line (`<line>_constantQ`), its members the levels;
  - each level a `spans` member whose membership is the closed form `params_json.tiling` {origin, width, hop, count} (D87: the ordinal is the code;
    nothing per sample);
  - a level dictionary (window in samples and seconds, tile count, finest tiles, what it resolves);
  - an empty tile dictionary per level (where a tile's label or note goes, D83).
- **The command:** `run select_time_grid --rule constant_q` writes it on a recording's time line (`--recording`, or every recording's), the range a quantity (`--range "0.5-64 Hz"`). It is
  idempotent. The importer does not write it, so its parity with nxr's converter holds.
- **On the fixture** (1200 Hz): 7 levels, windows 256…16,384 samples, resolving 32–64, 16–32, 8–16, 4–8, 2–4, 1–2 and 0.5–1 Hz.
- **Tested** (`test_constant_q`):
  - against the Studio's own functions on five lines (1200, 2400 and 250 Hz, the `cycles` method, 3 voices): bands, levels, entries, the ladder's and
    members' labels, descriptions and closed forms;
  - the writer's rows, and its idempotence;
  - the tile arithmetic.

## MEG/EEG source imaging (`meeg/`; spec: `docs/specs/2026-11-10-neuro-forward-inverse.md`)

When a dataset has no head model from import, `run headmodel` computes a sphere one (MEG: Sarvas in one sphere or overlapping spheres fitted to the inner skull; EEG: three shells fitted to the scalp), stored as an imported one (spec section 8).
When a dataset has no inverse kernel from import, `run inverse` makes one: the **unconstrained minimum-norm kernel with depth weighting** from the stored head model (the gain) and a
noise covariance. When it has no noise covariance either, `run noisecov` estimates one from a recording (the good channels over a time window, the mean removed per block, read chunk by
chunk). The defaults reproduce Brainstorm's default output (depth exponent 0.5, limit 10, SNR 3, regularisation 'reg' 0.1, per-type whitening); `--mne` and the named options give MNE-Python's.
**The projectors.** The kernel applies the channels' projectors as each reference does. Brainstorm (the default, `process_inverse_2018`): the projectors of status 1 and 2
on all the channels, applied to the gain and to both sides of the noise covariance, then the good channels and the average reference. MNE-Python (`--mne`, `make_inverse_operator`):
every projector of the channels (the average reference one of them) on the picked channels, each vector renormalised; the covariance projected with those and with the ones its
recording was read with; the gain used as it is; the kernel ending with the projector. A covariance of the data as recorded and one of the projected data give the same kernel.
Goldens: `meeg/test/golden/gen_kernel_projectors_golden.py`/`.m` (one active SSP projector, both covariances), checked to 1e-12 (measured 4e-15 against Brainstorm, 6e-14 against MNE-Python).
The kernel and its `_channels`, `_noise_cov`, `_whitener` and `_source_rr/_nn` are stored as the importer stores them, so a reader tells a computed kernel from an imported one by its producer string
only. The maths is written from the papers, not translated from Brainstorm (GPLv3); Brainstorm and MNE-Python are reference outputs (goldens in `meeg/test/golden/`, with their generators).

## Temporal filters (`compute/include/nxr/filter.h`, `lowpass|highpass|bandpass|bandstop|notch`)

MATLAB's verb names, zero phase and no delay, applied chunk by chunk (a state fed chunk after chunk; the output equals the whole-record filter to ~1e-14 for any chunk size). The default option
set reproduces Brainstorm's output: lowpass/highpass/bandpass a Kaiser FIR (kaiserord + fir1 'noscale', ripple 1e-3, 0.5 or 1 Hz transition bands, record mean removed and restored only without a
high-pass part, both ends padded with the mean of the first/last K samples); notch the 2nd-order IIR notch forward-backward, one frequency after the other; bandstop butter(4) forward-backward
(MATLAB filtfilt's ends). `nxr-neuro <verb> --dataset D/N --subject S --recording R --fpass HZ[,HZ] | --fstop LO,HI | --frequencies F,.. | --line F [--harmonics N]` writes a new recording
`<R>_<verb>` on the same channels and time line, with flags, events, bad segments, projectors and CTF compensation carried over. MNE-Python's option set: not done.

## Declared outputs and presets of a definition (`pipelines/include/nxr/pipelines/declared_outputs.h`; spec `docs/specs/2026-11-17-definition-outputs.md`, sections 2, 3 and 6)

A definition's `outputs:` entry may DECLARE what it stores: the kind, the found inputs it lies on or is of, the stored name. The node that made the values DESCRIBES them (`Array::meta["describe"]`: measure, method,
axes with their sizes and a closed-form line, the segment tiling, the scaling record, the unit); the runner binds the two, checks them, and io writes every output of the run through ONE `task::Unit`
(provenance and cache key with each, all or nothing). A manifold an output lies on, made by the same run, is written first.

| kind | declaration | stores |
|---|---|---|
| `field` | `kind_name`, `on: LINK` | a Field on that manifold |
| `statistic` | `of: INPUT`, `axes: [{name, on: LINK} \| {name, line: NAME} \| {name}]`, `partition: {tiling: NAME, on: LINK}` | the statistic, its frequency (or other) line, the segment partition |
| `operator` | `operator_kind`, `from: LINK`, `to: LINK` | an Operator between two manifolds: dense, or SPARSE (csr or csc) when the node describes `sparse` (indptr, indices; its array the nnz values), with the cell keys `from_cell`, `to_cell` (d0: '0' to '1') and a stored `chunks`/`shards` layout |
| `selection` | `type: partition \| set \| spans \| tiling`, `on: LINK` | a Selection on that manifold, with its dictionary |
| `manifold` | `type: line \| surface \| product \| points \| complex` | the manifold, found by name and reused when identical. `points`: one coordinate each (an operator's mode axis, the eigenvalues); `complex`: a CELL COMPLEX, a surface whose vertices, edges ([nE x 2]) and faces are stored |

A field's description may name the cells it lies on (`cell`: '0' vertices, '1' edges, '2' faces) and store integers (`dtype`: `i32` or `u8`). A link may name ALTERNATIVES (`glued|surface.manifold.0`: the first one made by the run or found); an output may be
`optional: true` (a node that makes none hands an array marked `absent`: a glued surface of a cortex of two pieces), and a field may lie on a manifold the run itself makes. Replacing a manifold (`--replace`) moves what lies on it aside with it.

A name is a template: `{input}` the found input's name, `$parameter` a parameter's value, `{name}` what the step's plan fixes (`facts.names`). The short forms (`kind: spans`, `kind: labelling`, `kind: measurement`, a statistic without `axes:`, a Field's own kind
name such as `band power`) are stored as before. A `node:` use (grouped names such as `signal.psd`) and a `step:` use are the same.

`presets:` names parameter sets (each a named alternative); `run NAME --set PRESET` starts from one. Precedence: the parameters' defaults, then the preset's values, then the caller's own `--NAME VALUE` or
`--set NAME=VALUE`. A preset names only declared parameters (refused when the definition is read); its name is in the provenance and the cache key (`preset`), beside the resolved parameters.
`run NAME --help` lists every preset with the values it sets.
A statistic's `describe` may carry `producer`, what its readers ask of the provenance beside the run's own record (a spectrum's `frequency_resolution_hz` and `sfreq`, its segments, the share of samples left out, the scaling
record). A native node that applies a stored dense Operator (`signal.linear_map`) is handed its matrix and the channels it lies over by the runner (`meta.channels`); it reads no store.

## The Hodge-Dirac eigenmodes (`run eigenmodes`, the step `hodge_dirac`; spec `docs/specs/2026-11-17-eigenmodes-as-definition.md`)

`run eigenmodes --surface NAME [--modes K] [--glued]` solves the Hodge-Dirac modes of a stored surface on its intrinsic Delaunay complex (the split solve: the Laplace-Beltrami ladder gives the exact modes, the 2-form
ladder the coexact modes; one lambda cut, the constants out) and stores them with their full forms. Default: the reference method of nxr and Rheome; the other methods are named options (`--shift`, `--sigma`, `--slice`,
`--normalize`, `--star1-floor`, `--mollify`, `--flip-eps`), all in the provenance and the cache key. `--glued` is for a cortex of TWO pieces (its hemispheres): each medial wall cut, a bridge grown, solved on the glued
surface; a surface of two pieces without it, or of one piece with it, is refused with the reason; three or more pieces are refused. Eigenmodes come first: a spatial band that needs modes that are not stored has the runner
make them with this run, said in a message. What it cannot resolve: a mode above the mesh's resolution is not a property of the cortex, and the bridge is not cortex.
Stored (stem `hodgeDirac`, or `hodgeDirac__<surface>` for a surface that is not the primary one; the layout is unchanged from `eigen solve`): the mode axis `<stem>` (points, coordinate mu = sqrt(lambda), ascending), the field
`<stem>_kind` (0 exact, 2 coexact; of the eigenvectors operator), the operators `<stem>_eigenvectors` (axis to surface, [V x K] float32) and `<stem>_forms2` (axis to the complex's faces, cell 2), the selection `<stem>_support`; the
glued surface `<surface>_glued` with its bridge `<surface>_glued_bridge`; and the cell complex `<solved>_intrinsic` (vertices, edges, faces) with the sparse operators `_d0` (cell 0 to 1) and `_d1` (cell 1 to 2) and the fields
`_length`, `_star0`, `_star1`, `_star2`, `_angle_tail`, `_angle_tip`, `_angle_sum`, `_reference` on its cells. The library's `eigen_solve` (`pipelines/eigen.h`) is a thin caller of this definition.

## The Welch PSD (`compute/include/nxr/welch.h`, the definition `run psd`)

`nxr::signal::spectral::Welch` is a streaming estimator with no carried segment state: a block owns its first `own` columns and carries the right halo (n - 1 samples, fewer where the span ends); it forms every segment that STARTS in its own columns
(`stft`), so segments cross chunk edges and **the result is identical to the last bit whatever the chunk size** (`test_welch`, `test_welch_halo`: 1 sample to the whole record). Memory: one block with its halo + one segment; the estimator holds only the sum over the segments. A sample mask leaves out every segment that touches a bad sample (the segment grid does not move).
Options (each a named field of `WelchOptions`): segment length (samples, or seconds with `segmentSamplesFor`), overlap (a fraction; hop = n - floor(n*overlap)), window (Hann · Hamming · Blackman · Rectangular, periodic or symmetric),
detrend (none · constant · linear), scaling (`Density` power per Hz = scipy "density" and Brainstorm's physical units · `Spectrum` · `Chain` = nxr.spectral.chain's per-bin power). One-sided: interior bins doubled.
The references are named option sets: `brainstormPsdOptions()` (4 s, 50 %, Hamming **symmetric**, the mean of each segment removed, power per Hz), `scipyWelchOptions()` (Hann periodic, constant detrend; nxr's `qc.sensors.psd` is this at 4 s),
`nxrChainWelchOptions()` (`nxr.spectral.chain._welch`).
**`run psd` is a definition** (`pipelines/definitions/psd.yaml`, since D1c of `docs/specs/2026-11-17-definition-outputs.md`): the node `signal.psd` (an accumulate node of `nxr::signal::spectral::Welch`; the segment rule is the pure
`nxr::signal::spectral::segmentFor`) streamed by the runner, and a declared `statistic` output with its channel axis, its frequency line and its segment partition. The sets below are its **presets** (`--set tiles|brainstorm|scipy|nxr-chain`);
the Welch options are its parameters (`--segment-seconds`, `--segment-samples`, `--segment-level`, `--segment-rule tile|round|even`, `--overlap`, `--window`, `--periodic true|false`, `--detrend`, `--scaling`,
`--bad-segments exclude|ignore`; `--name`; `--recording`, none = every recording), each with its meaning, default and unit in `run psd --help`; the caller's own values win over a preset's. The old `--no-bad-segments` is
`--bad-segments ignore`, `--periodic`/`--symmetric` are `--periodic true|false`, `--through` is its own definition `rotary_psd` (below). The result is tested against the references (`test_psd_golden`: scipy, nxr's chain and Brainstorm's
output, and `rotary_welch_golden.txt`, to float32 storage), not against an older output.
**The pipelines' sets (`--set`, every Welch producer: `psd`, `sourcepsd`, `sourcetensor`, `crossspectrum`, `quadrature`, `hodgepower`, `speedspectrum`):**
`tiles` (THE DEFAULT since 2026-11-13, the time-tile amendment: a segment is a TILE of 2^k samples on the recording's line, k = round(log2(`--segment-seconds`
· fs)), the nearest level, as the Studio's `anchorsFor` rounds (the Studio's `welchSeg` rounds up), or `--segment-level k`, or a power-of-two `--segment-samples`;
half overlapping, so the hop is level k − 1 and the overlap must keep it a tile (0, ½, ¾, …); Brainstorm's taper, detrend and density; the span snaps to level k;
4 s gives level 12 at 1200 Hz, 4096 samples, 3.41 s) · `brainstorm` (Brainstorm's exact behaviour, the default until 2026-11-13: round(4 · fs) samples from the
span's start) · `scipy` · `nxr-chain`. Segments still cross chunk edges at ½ overlap: the right halo carries the samples, no state does. Parity (`compute/test/test_welch.cpp`, goldens in `compute/test/golden/`, synthetic signals; generators `gen_welch_golden.py`/`.m`): scipy, nxr and Brainstorm's `bst_psd` outputs agree to 1e-15.

`run psd` reads a recording chunk by chunk, leaves out the recording's bad segments (the Selection `<recording>_bad_segments`) and stores the PSD as
a STATISTIC (2026-11-13, `docs/specs/2026-11-13-statistics.md`; until then a Field on channels × frequency, which stays readable): `<recording>_psd_welch`,
measure `power_density` (`power` for the spectrum and chain scalings), method `welch`, OF the recording, computed from its segment partition
`<name>_segments` (Welch's segments as a closed-form spans Selection on the time line: width, hop, count) and resting on `n_members` segments
(`members_skipped` left out for a bad sample), axes channel × the frequency line `<recording>_freq`, float32, unit `<unit>^2/Hz`, the span, every Welch
option in `options_json`, the spectral scaling record, and in `producer_json` the cache key, the share of samples excluded and the
projectors applied on load (`run.projectors`). All channels are kept, the bad ones too.

### `run rotary_psd --operator OPERATOR`: the rotary spectrum (2026-11-16; nxr's `docs/specs/2026-11-15-nxr-scopes.md`, "Rotary spectra ..."; its own definition since 2026-11-17, the old `psd --through`)

**What it is for.** A rotating pattern is one pair of sensor filters turning from x towards y: its power sits at ONE sign of frequency. The rotary spectrum is the two-sided spectrum of
that pair, and its sense index (P₊ − P₋)/(P₊ + P₋) over a band (derived at read, `nxr.spectral.rotary.sense_index`) says which way the pair turns, from +1 (all forward) to 0 (no preferred sense).
`OPERATOR` is a stored dense `[2 × C]` Operator over the recording's channels, normally the subject's `<kernel>_rotationPlane` (rows fx, fy; made by nxr's `nxr kernel rotationplane`, which
already negates fy when −f wins, so +f is forward and nothing is flipped here).

**The graph** (the Editor's chain: members → filter → transform, a linear map before any non-linear step): the chunk service gives each block of the recording (rows = the operator's channels, in its
order, the bad-sample mask) → *members*: z = fx·X + i fy·X (`nxr::signal::linearMap`, a pointwise node) → *spectrum*: the two-sided Welch periodogram of each segment (`nxr::signal::spectral::ComplexWelch`, pure)
→ *reduce*: the mean over the segment partition, the estimator's own reduction, the same segments, window, detrend, scaling, carrying across chunk edges and bad-segment rule as the sensor PSD
(`--set`, `--segment-*`, `--window` … apply as they do to `run psd`: the same parameters and presets). The host type-checks before any data flows: a dense Operator of two rows whose channels (from_selection) are on the
recording's channel manifold and inside it; otherwise it is refused with its reason.

**Stored.** The STATISTIC `<recording>_<operator>_psd_welch` (`--name`), measure `power_density` (`power` for the spectrum and chain scalings), method `welch`, OF the recording, from its segment
partition, axes **[pair (1), frequency]**, the frequency on a SIGNED closed-form line `<recording>_freq_signed` (−fs/2 … fs/2 − fs/n, n bins; for odd n the line is symmetric). `options_json`
has every Welch option, `sides: two`, and `input` = the operator, its id, its rows and the band it was made for (the operator's own `band`); its unit is the operator's output squared per Hz.
It is told from the sensor PSD by its axes (`pair` × a signed frequency). The reference is nxr's `nxr.timefreq.psd` of a complex series (scipy.signal.welch, `return_onesided=False`, f ascending): parity
to 1e-16 on synthetic signals (`compute/test/golden/gen_rotary_welch_golden.py`), a forward pair peaking at +f and a backward pair at −f (`test_welch`); the stored statistic, the chunk-width
independence to the last bit and the mirror (fy negated gives P(−f)) are in `test_psd_pipeline`; the graph is `signal.linear_map` (a pointwise linear map) then `signal.psd` (a complex series) (an accumulate node), the operator handed to the node as its matrix and channels by the runner.

## Source maps: band power, quadrature, Hodge power (`pipelines/include/nxr/pipelines/meeg.h`)

All three read a recording chunk by chunk through a stored inverse kernel (default: the subject's only one), take the sensors' cross-spectrum per band with the Welch options of `run psd`
(the same `--set` and overrides), and never form the N x N source matrix. Bands default to Brainstorm's (delta 2-4, theta 5-7, alpha 8-12, beta 15-29, gamma1 30-59, gamma2 60-90; the mean of the bins
from lo to hi inclusive).

- `run sourcepsd`: Brainstorm's PSD on sources grouped in bands: diag(K S K^T) per band, a free kernel's three orientations summed in power, `--normalisation relative` divides by the
  source's whole-spectrum power. Stored as a STATISTIC read as maps (2026-11-13): `<R>_<kernel>_psd`, `power_density` (`power`; `relative_power`) by `welch`, OF
  the recording, from its segment partition, axes vertex (the kernel's target) × band (each entry named, with its edges and bins), so band k's map is
  `<R>_<kernel>_psd_<band>` (as its Field was named; one stored before as Fields stays readable). `io::map_slices` lists a surface's maps of statistics and
  `read_map_slice` reads one; `run measure` and `measure export` take them as they take Fields (their rows name the statistic and the band:
  `of_statistic_id`, `of_index`, added to the measurement table in place).
- `run sourcepsd_bins` (spec `docs/specs/2026-11-16-meg-fingerprinting.md`, 3.1; the node `meeg.source_bins`): the same Welch pass as `run sourcepsd` keeps every bin of ONE band instead of its mean: the frequency
  partition's members are the bins, nothing reduced over frequency. `--kernel`, `--band-name`, `--lo` and `--hi` are required (each a choice of data); one run a band (a band is a selection that selects; resolving
  frequency within it is this transform). Stored: the statistic `<recording>_<kernel>_psd_<band>_bins` (measure `power_density`, `power` for the other scalings; method `welch`; OF the recording, from its segment
  partition), axes vertex (the kernel's target) x frequency (the band's bins, a line on the Frequency manifold), in the precision of the recording; the band statistic of `run sourcepsd` is the mean of these bins. No
  normalisation (a total over all bins is not kept bin by bin: divide at read). **Accumulation is on the source side** (`nxr::signal::spectral::LinearWelch`, `compute/include/nxr/welch.h`: the kernel applied to each
  segment's Fourier coefficients, |K X|^2 summed per source per bin, a free kernel's three orientations summed in power): sources x bins x 8 bytes (15 000 sources, 299 bins: ~36 MB) and a block of at most 4096 mapped
  rows, where the sensor side would hold channels^2 x bins (270 channels, 299 bins: ~350 MB). `LinearWelch` is tested in `compute/test/test_welch.cpp`: equal to Welch on K x over the bins kept, with a bad span and odd
  segment lengths, independent of the chunk size. **What it cannot resolve:** neighbouring bins are correlated (the window's width), so the bins are not independent estimates; each source's spectrum is the kernel's
  estimate, with the kernel's spread and orientation bias, and a spectral feature two sources share through that spread is not told apart.
- `run quadrature` (a definition: `signal.xpsd` on the kernel's channels, then the node `signal.source_planes`; `docs/specs/2026-11-17-flow-verbs.md`): the quadrature (imaginary) part of the source cross-spectrum, which no zero-delay mixture produces. Per band: `quadratureFraction` ||Im C|| / ||C|| (in the
  fields' params), the `quadratureEnergy` map per source `<stem>_energy_<band>`, and the top K quadrature pairs (p leads q by a quarter cycle) `<stem>_pair<k>_p_<band>` / `_q_`
  (vector3 fields for a free kernel). nxr's `lagged.py` is the reference. Since 2026-11-13 they are MAP STATISTICS by `welch`: `<stem>_energy`
  (`quadrature_energy`, vertex × band, each band entry with its quadrature fraction and the sensors' rank) and `<stem>` (`quadrature_pattern`, vertex ×
  patterns, vector3 for a free kernel; an entry a band's p or q of pair k with its sigma, share and rotary sense); each map keeps its Field's name.
- `run hodgepower`: the band power of the Hodge–Dirac mode coefficients of the sources: the free-orientation kernel fused into the subject's Hodge–Dirac package (`nxr.inverse.fused`), two
  parts per band on the eigen axis, `<stem>_<band>_current` (coexact) and `<stem>_<band>_flow`: since 2026-11-13 one MAP STATISTIC `<stem>` (`power_density`
  by `welch`, axes mode × band part), each part's map keeping its Field's name. A fixed-orientation kernel is refused.
- `run sourcetensor`: the VECTOR analogue of the band power map. Per source and band, the complex 3×3 Hermitian cross-spectral tensor of the current,
  T = K_v (S + i Sq) K_vᵀ (K_v the source's three kernel rows; S + i Sq the band's sensor cross-spectrum with its quadrature part), stored as value type
  `hermitian3` (float32 [N × 3 × 3 × 2], real and imaginary parts; ambient axes, gauge-free). Since 2026-11-13 its four outputs are MAP STATISTICS by
  `welch` on the axes vertex × band, each with its value type: `<stem>` (`cross_spectral_density`, hermitian3), `<stem>_stokes` (`stokes`, quaternion),
  `<stem>_normalpower` (`power_density`, scalar), `<stem>_normalshare` (`share`, scalar); a band's map keeps its Field's name, and `run measure` reads
  a hermitian3 map as it reads a hermitian3 Field (the `tensor_*` sums). Its trace IS the band power of `sourcepsd` with the same
  options (tested). Read from it, per vertex: the tangential Stokes parameters in the canonical frames as a quaternion `[S0, S1, S2, S3]` (S0 the
  tangential power, S1 + iS2 the linear polarisation — axis ½·arg from north, strength |S1 + iS2| —, S3 the circular, > 0 turning from north towards west,
  counter-clockwise about the outward normal; S0, S3 and |S1 + iS2| do not depend on the frames), the normal power nᵀ Re(T) n and the normal share.
  Needs the canonical frames (`run surface_frames`); a free-orientation kernel only. What it cannot show: the tensor is the KERNEL's estimate — a minimum-norm
  kernel spreads each source over its neighbours, and resolves its orientation only as far as the kernel does (on OMEGA sub-0181 the MNE kernel gives a
  median normal share of 0.35, an anisotropy of 0.57 and a circularity of 3 % per vertex in alpha).
- `run crossspectrum` (named `csd compute` until 2026-11-13; renamed so it is not read as current source density): the sensor cross-spectral density per band, **stored**: the root every quantity above is read from. **Why:** the source cross-spectrum
  between every pair of sources (3N × 3N complex, ~16 GB a band for 15k sources) is K C Kᵀ, so it holds nothing beyond the band's sensor cross-spectrum C
  (M × M, ~0.7 MB) and the kernel. Stored, any block between two places is read on demand, T_ij = K_i C K_jᵀ (`nxr::core::crossTensor`; T_ii is
  `sourcetensor`'s), and so is a region-to-region block, (Σ_A K_i) C (Σ_B K_j)ᵀ, without forming the full matrix. C does not depend on a kernel: one C
  serves every kernel whose channels it covers. Welch as `sourcepsd` (the same options, the mean over the band's bins, the quadrature part kept),
  C_ab = ⟨X_a conj(X_b)⟩. Stored as a STATISTIC per band (2026-11-13, `docs/specs/2026-11-13-statistics.md`): `<stem>_<band>` (stem default
  `<R>_crossspectrum`), measure `cross_spectral_density` (`cross_spectrum` for the spectrum and chain scalings), method `welch`, OF the recording, from
  the segment partition `<stem>_segments`, axes channel × channel (the Selection `<stem>_channels`: the channels of `--channels-of KERNEL`, else the good
  channels of `--types`), value_type `complex` (float64 [M × M × 2]), the band in its options. One stored before as an Operator of kind `cross-spectral density` is still
  read (no migration). `crossspectrum_read` reads one band, optionally on a kernel's channels in the kernel's order (a channel it does
  not cover is refused). Tested: Hermitian; K C Kᵀ's diagonal blocks equal `sourcetensor`'s tensors; independent of the chunk size. What it cannot show:
  it is a sensor quantity, and every source reading of it carries the kernel's spread and orientation bias; the band's bins are averaged, so the shape of
  the cross-spectrum within the band is not kept.
  **`--estimator analytic`** (spec `docs/specs/2026-11-13-cycle-pattern.md`, stage 1, as amended 2026-11-13: the time tiles): a cross-spectrum PER WINDOW,
  for what repeats across a rhythm's cycles. Per band, the band's analytic signal (the Studio's Meyer band, `signal::wtBlock`, gain 2: Re z is the
  band-passed signal) read chunk by chunk with the band's halo, decimated by the constant-Q ladder's stride (`selection::constantq::strideFor` at the band's low edge, a
  power of two leaving ≥ 24 samples a cycle; alpha at 1200 Hz: 4, so 300 Hz; the samples kept are its multiples on the recording's line; not a filter: a band
  reaching the decimated Nyquist rate is refused), and Σ z zᴴ per TILE of the window level (`partition::crossSpectrum`, m = line sample >> level). The level
  is `--level K`, or `--window-seconds` snapped to the nearest level, by default the finest of the bands' constant-Q windows (`selection::constantq::windowFor`: the
  ladder's, 6.8 cycles of the low edge rounded up to 2^k; alpha at 1200 Hz: level 10, 0.85 s), at most 15 (a chunk). The span snaps to the tiles; a tile
  holding a bad sample or cut by the span's end is dropped. Stored per band (since 2026-11-13) as a STATISTIC `<stem>_<band>_windows`, `cross_spectrum_sum` by `analytic`, value_type `complex`, axes window (the
  kept tiles of `<stem>_windows`, by ordinal; that tiling its partition) × channel × channel (one stored before as an Operator of kind `cross-spectrum series` is still read),
  [W × M × M × 2], the SUMS (the mean is sum / `window_samples`, the decimated points a tile, at read), `window_index` the tiles' ordinals on the line; and the
  level's tiling of the line, the closed-form `spans` Selection `<stem>_windows` (the envelope's form: `params_json.tiling`). A COARSER WINDOW IS A SUM AT
  READ: `crossspectrum_windows_read(…, level)` sums the stored tiles m with m >> (level − stored), keeping a window only when all its tiles are stored (the
  Studio's envelope roll-up). Memory and storage: (span / 2^level) · M² · 16 bytes a band (chunk 0 of a 268-channel recording at level 10: 37 MB). The Welch
  options do not apply. Tested (`test_crossspectrum`): Hermitian; the tiles equal a full-recording reference to the halo's truncation (1.1e-5), and a
  stride-2 band's sums × 2 equal the full-rate ones (1.5e-3); the same tiles with chunks of 128, 512, 700 and 1777 samples (finer and coarser than the tile) to
  the halo's truncation (< 1.5e-5); level 10 stored directly equals the level-8 tiles rolled up; a bad span drops its tiles and the coarser windows that hold
  them; a span in seconds snaps to the tiles. What it cannot show: a window of a band of width B and length T holds about B·T independent complex samples
  (stored as `independent_estimates_per_window`), so a statistic of one window has at most that many degrees of freedom.
  **`--resolve bins`** (spec `docs/specs/2026-11-15-band-resolved-cross-spectrum.md`, 2026-11-15; Welch only, the analytic estimator is refused): the same pass keeps the band's cross-spectrum
  bin by bin. The only change is the frequency partition of the last reduction (`nxr::signal::spectral::CrossWelch`'s `BinGroup`s: one group a bin instead of one for the band; no new compute function).
  `--resolve band` is the default, its output and cache key are those of before. With `bins` the run ALSO stores, per band, the STATISTIC `<stem>_<band>_bins` (measure `cross_spectral_density`, method `welch`, OF the
  recording, from `<stem>_segments`), axes channel × channel × frequency (the bins' centres in Hz and `bins: [first, last]`, the PSD's frequency line), complex float64 [M × M × F × 2], Hermitian in the
  first two axes at every bin; `resolve: bins` and every Welch option are in its options, its producer and the cache key. The band statistic is the mean of its bins (tested to 1e-13, and equal to a
  `--resolve band` run's). Size: M² · F · 16 bytes a band (alpha at 4 s, 268 channels: ~20 MB). Read: `crossspectrum_bins_read` (io only), then the pure functions of `meeg/include/nxr/signal/crossspectral.h`:
  `coherency` (C_ab / √(C_aa C_bb) per bin), `phaseSlope` (weighted least squares of the phase, unwrapped by adjacent bins, on f: a delay τ = (1/2π) dφ/df in ms and the fit's rms residual; `weights`
  `coherence` (|coherency|²) or `none`) and `phase_slope_index` (Nolte et al. 2008, PRL 100, 234101: Ψ = Im Σ conj(𝒞(f)) 𝒞(f + δf), written from the paper; not normalised by its jackknife standard
  deviation, which needs the segments' C). Sign: arg C_ab > 0 when a LEADS b, so τ > 0 and Ψ > 0 when the row channel leads the column channel (if a leads b by d, C_ab = |X_a|² e^{+i2πfd}, so τ = +d). `nxr-neuro crossspectrum read --name <stem>_<band>_bins
  (--pair A,B | --seed A) --reading coherency|imag|phase|delay|psi` prints them as JSON (A, B recording channel indices). Tested (`test_crossspectrum`): the bins' mean equals the band statistic; independent of
  the chunk size; a 12 ms delay between a source and its noisy copy is recovered from 5-30 Hz (11.6 ms; the 8-12 Hz band alone gives only the sign: a few independent bins of 6 segments); `meeg/test_crossspectrum_bins`:
  a known delay through many phase wraps, and the PSI against MNE-Connectivity 0.9.0's `_compute_psi` (`meeg/test/golden/gen_psi_golden.py`, 1.7e-16). **What it cannot resolve:** the delay is the slope of
  the phase over the band's bins, ONE delay per pair for the whole band, from a band only a few Hz wide; within that range a constant delay and a phase that is not linear in frequency are not told
  apart beyond the fit's residual. Neighbouring bins are correlated (the window's width), so the bins are not independent estimates; on sensors a pair's phase is mixed by the field spread, so a
  sensor delay is not a transmission delay between the sources.
- `run phasegradient`: whether the band's activity MOVES across the cortex, which way and how fast, in physical units. nxr's phase reader
  (`nxr.detect.core_phase`, `PhaseReader.read_out`: ∇φ = Im(z̄ ∇z)/|z|², no unwrapping; velocity u = −ω ∇φ/|∇φ|²) in its spectral form, read from a
  stored band cross-spectrum (`--cross-spectrum`, the first reader of `run crossspectrum`) through a free kernel and the mesh's P1 gradient (the area-weighted mean of the
  face gradients, projected on the tangent plane): with z_v = a_v X the signal at v and g_v its tangent gradient, the sums P = ⟨|z|²⟩ = a S aᵀ,
  H = Im⟨z̄ g⟩ = B Sq aᵀ and Q = ⟨|g|²⟩ = tr(B S Bᵀ) (`nxr/signal/phase_gradient.h`) are power-weighted over the band's segments and bins and add over regions. The
  signal is the full current as the inverse gives it, the x y z components summed (`--signal ambient`, default) or `--signal normal`, the normal current n·J.
  **Stored as measurements over the dyadic tile grid** (spec `docs/specs/2026-11-14-measurement-pyramid.md`: a reduction over space is a measurement over a partition, never a
  per-vertex ratio; the earlier per-vertex Fields broke the cardinal rule and are no longer written; those already in a dataset are untouched):
  - **The leaves** are the finest level of the surface's dyadic grid (`finest`, a quantity: a tile diameter, an area or parts per hemisphere; the grid is made by `run select_dyadic_surface` when the surface has none), the same leaves as `run measure --set cortical_band_pyramid`. `run surface_frames` first: the current is stored in the canonical frames.
  - **What is stored** (the additive INGREDIENTS, array-backed `[leaf]` measurements of a byte-less Field `<stem>_<band>` of kind `phase gradient`, by_operator the kernel, in the partition's node `<node>/phase/<name>/`,
    within the band when it is one of the Frequency manifold's `bands`), each a SUM over the leaf's vertices weighted by the vertex's lumped area a, with nxr's registry names:
    `phase_power` = Σ a P, `phase_gradient_power` = Σ a Q, `phase_current_1`, `phase_current_2` = Σ a (H·e1), Σ a (H·e2) (the phase current H = Im⟨z̄ g⟩ in the canonical frames, e1 north, e2 west: nxr's
    `vector_stats` convention), `phase_current_magnitude` = Σ a |j|. Length units are converted to metres at the store (rad/m × power; Q per m²). A vertex with no canonical frame (the registered
    sphere's poles) is out of every sum (counted in `producer_json.leaves.vertices_without_frame`).
  - **Derived at read** (`measure query` / `query_measurements`: ratios of the rolled-up sums, never stored, never rolled up): `phase_gradient_1`, `phase_gradient_2` (Σj / ΣP, rad/m, against the travel),
    `phase_gradient_magnitude` = |Σj| / ΣP, `phase_gradient_direction_deg`, `travelling_index` = |Σj| / √(ΣP ΣQ) (1 one plane wave, 0 a standing pattern or opposite waves cancelling), `phase_coherence` = |Σj| / Σ|j|;
    the speed is ω / magnitude (ω = 2π × the band's centre, `producer_json.omega_rad_s`). A coarser level is the sums rolled up (`--spatial-scale`), so a region's gradient is the power-weighted mean of its vertices'.
    (New in the lookup: a derived measure of ARRAY-BACKED sums is computed cell by cell from its rolled-up ingredients; a derived measure whose ingredient is a mean, not a sum, is refused.)
  - **Time.** The input is a time-averaged cross-spectrum, so a leaf has one value and no frame axis; its time is the cross-spectrum's own span, recorded in the Field's `producer_json.time` (`axis none`, the window, the
    segments and whether bad segments were excluded).
  - **What changed in meaning.** A leaf's gradient is the power-weighted mean of its vertices' phase currents in the leaf's canonical frames, not the gradient at a vertex: a leaf larger than the wavelength averages a
    turning wave out (said in the warnings). The per-vertex velocity and speed are gone (they were ratios of per-vertex sums); the speed at a leaf is ω / `phase_gradient_magnitude`.
  - **Tested** (`test_phasegradient`, synthetic spheres, never a real dataset): the leaf sums equal the vertex-area-weighted per-vertex sums computed directly (**2.6e-15** relative, 512 leaves); the derived gradient equals the
    power-weighted mean of the vertices' gradients (**6.5e-16**; it equals the area-weighted mean of H/P only where P is uniform, which the planted wave's is to 3 decimals); a roll-up D → D−1 equals a run on the merged leaves
    (**2.9e-14**, every stored sum and the derived magnitude, travelling index and coherence); against ground truth at leaf level (a wave travelling round two spheres, through `run crossspectrum`): speed within
    1.2 % (median; 3.9 % worst), travelling index 0.999 against 0.002 for the standing pattern, the ambient signal within 1.1 %, the vertices' direction within 11° of the ground truth (cos > 0.98); `test_wave`:
    `phaseCurrentInFrames`.
  - What it cannot show: the kernel spreads each source with
    no lag, which shrinks the phase gradient (the speed is overestimated where the spread is wide; not yet measured on a real head model); the readout is a
    power-weighted mean over the whole window and the band (waves that change direction cancel, nothing is resolved in time); ω is the band's centre;
    the P1 gradient resolves a wavelength of a few edges at best.
- **Retired: `phasegradient --spatial-band` (the per-spatial-band route, the timing map and its consistency).** The spatial band is now a band SET selection (`run select_bands --rule levels|eigenvalue|wavelength`) read by `run measure`;
  the intrinsic-form maths (`compute/include/nxr/intrinsic_forms.h`, the phase 1-form split, `nxr::intrinsic::vertexOutflow` / `vertexCirculation`: a tile's flux and circulation are the SUM of the pointwise values by `reduce`) stays in compute.
  Whether the timing map and its fits return as a node of the measurement system is an open design question.

## The scaling of a spectral quantity, chosen at read (`compute/include/nxr/spectral_scaling.h`, `pipelines/include/nxr/pipelines/spectral_record.h`; spec `docs/specs/2026-11-13-spectral-scaling.md`)

Each spectral estimator stores ONE quantity and, in `producer_json.spectral_scaling`, what a conversion needs: what is stored (density · spectrum ·
band-power), the bandwidth one value spans, the Welch window's noise bandwidth per bin, the degree (2 for quadrature's (Im C)²), the power unit. The
reader picks: `stored` · `density` (units²/Hz) · `power` (units² in the band) · `amplitude` (√power) · `amplitude-density` · `db` · `db-density`.
Density and power are factors (they apply to cross-spectra); the others apply to a real power. Read through `crossspectrum_read` /
`crossspectrum_windows_read` (`scaling`) and `measure export --scaling`. A Welch band's power is its density × its width (the broadband reading; a tone's
power is the spectrum-scaled value itself); the analytic estimator's band power is ⟨|z|²⟩ / 2, its density that over the band-pass's noise bandwidth
(`timebank::noiseBandwidth`). Tested: white noise reads as the same density through Welch density, Welch spectrum and the analytic signal (within 0.6 %);
a 10 Hz cosine's band power (0.5) reads 0.4994 (Welch) and 0.4978 (analytic). Not converted: a relative power, nxr's chain scaling, anything stored
before 2026-11-13 (refused any scaling but stored), `speedspectrum`'s energies (no record yet).

## Pipelines: definitions and the runner (`pipelines/include/nxr/pipelines/{definition,registry,runner}.h`, `io/include/nxr/io/{finder,yaml}.h`; `run`)

**Why.** Every analysis is a chain of pure steps run over a recording's chunks, then reduced or stored. Written as a DEFINITION (nxr's YAML format, the same
file in both tools), a chain can be read, checked and rerun on any subject without new code; the four parts of the architecture are explicit in it
(loading: the inputs; preparation: the plan; the graph: the steps; readback: the outputs). Spec: `docs/specs/2026-11-12-chain-graph.md` (approved); the
reference is nxr's `nxr/pipelines/` (`definition.py`, `registry.py`, `inputs.py`, `runner.py`), ported with its names.

- **A definition** has five sections: the header, `inputs` (found by the model: a field, operator or selection of a kind, or a stored **statistic of a method** (`statistic: histogram`, found by `where: {name: …}`: another run's
  global statistic read BY NAME; it lies on no manifold), narrowed by links such as
  `from: recording.manifold.0` and by `where` on its columns or `params_json`; a `where` value that is a `$parameter` holding a name may carry `{input}`, e.g. `bursts: "global_bursts.bursts_kept.{recording}"`; `each: true` the anchor, one run per match), `parameters` (defaults,
  overridden by `--set`), `steps` (`before` · `chunks: {over, do}` · `after`), `outputs`. `parse` refuses every structural fault with where it is.
  nxr-neuro's own definitions are in `pipelines/definitions/` (compiled in; `nxr-neuro list pipelines`); any `.yaml` path runs too.
  **Every definition names its recording and kernel ().** `--set recording=NAME` runs on that one recording (default `none`: every recording of the subject, one run each, as before; an empty-room recording without its own kernel is left out by naming the others, one run each); `--set kernel=NAME` picks that inverse kernel (default `none`: found by `method`, as before). The finder's rule: an input's `where: {name: $recording}` entry whose `$parameter` is `none` (or null) filters nothing. A named kernel must also be of `method` (its `params.inverse.measure`, default MNE): name its method too (`--set kernel=K --set method=dSPM`). Adding the parameters changes each definition's text, so its cached results are recomputed once.
- **The steps** (`registry.cpp`): `signal.band_filter`, `analytic`, `envelope`, `meeg.apply_kernel`, `form`, `power`, `phase_current`, `signal.gradient_power` (chunk:
  pointwise), `measure` (accumulate: a reduction over the run's span, per element; the cardinal rule; or, with `over` a constant-Q ladder, `amplitude` /
  `power` per TILE of the level of the band, D84: the level IS the band) and `detect_spans` (after: a measurement's tiles labelled with events, Rheome's
  `detect.spindle` rule). Each declares its ports, parameters and the
  function it is held equal to, and CHECKS ITS ARROWS on the model before any data is read: Nyquist, a kernel from the field's own channels, D106 (no
  linear map after a non-linear step), a phase only of a complex field whose phase is kept, a form only of its type, no measure of a complex field.
  `analytic` and `envelope` take their band as `signal.band_filter` does: the Studio's analytic signal of a band is the band's even taps (the real part) and its
  odd, Hilbert taps (the imaginary) on the same raw rows; `band: null` is the plain Hilbert, cut at 1024 taps (its lowest frequencies lose their quadrature).
  **`shape`: `morlet` (the DEFAULT) or `meyer`.** Morlet is the project's fixed-band rule (nxr `vortex.morlet_gain`, `bands_fixed`): a Gaussian in frequency,
  unit peak, centred on the band's ARITHMETIC centre (lo + hi)/2, the band's width hi − lo its full width at half maximum (σ = (hi − lo)/(2√(2 ln 2)); `filterbank::morletBand`),
  taken to taps as any gain; Re z is the band-passed signal. `meyer` (compact edges, `half_width` octaves, default 1/3) is the named option for analyses that split power across
  bands: Morlet bands overlap and do not add up. Both options are in the step's parameters, so in the run's provenance and cache key.
  **The phase current's `discretisation`** (`signal.field_form` with `form: phase_current`): `vertex_gradient` (the DEFAULT, the reference: nxr core_phase, Im(z̄ ∇z) with the
  mesh's vertex gradient on the tangent plane) or `intrinsic_edges` (the same quantity on the edges of the surface's intrinsic Delaunay complex, Im(z̄_tail z_tip), back at the
  vertices by `sharp`; the complex is built once in preparation with `eigen solve`'s default options). The same ports and measures either way; the option is in the provenance
  and the cache key; any other form refuses it.
- **`signal.phase_split`** (`form_steps.cpp`; spec `docs/specs/2026-11-19-intrinsic-form-nodes.md`): a complex scalar field on the surface a Hodge-Dirac package is solved on (the
  glued surface of a cortex: `meeg.apply_kernel` with `glued`) and the package → per sample, at the vertices: `potential` (the exact part's potential φ, d0 φ = the exact part of the
  1-form; for a phase step, a timing map, φ / ω), `stream` (the coexact part's stream function), `propagation` (−∇φ, vector3), and the ⋆1 energies `one_form_energy`, `exact_energy`, `coexact_energy`,
  `exact_misfit_energy` (`splitPhase`, `vertexEnergy`) of the 1-form the parameter **`one_form`** names: `phase_gradient` (the DEFAULT, the reference: the earlier lab design used phase lags;
  `nxr::signal::phaseGradient(complex, x)`, the phase step on the intrinsic edges: exact part = timing, coexact part = rotation) or `phase_current` (`nxr::signal::phaseCurrent(complex, x)`,
  Im(conj(x_tail) x_tip): the exact part is the curl-free part of the current, not a timing); any other value is refused at the check. `one_form` is a step parameter, so it is in the provenance and
  the cache key. The provenance record also holds `boundary`: `none` (closed surface) or `neumann_potential_dirichlet_stream` (open surface: Neumann potential, Dirichlet stream function), a fact of the method.
  The 1-forms stay inside the node. **Phase-wrap limit:** a phase step per edge is taken in (−π, π], so a wave with more than half a cycle per edge aliases; nothing in the output flags it. The runner binds the package's 2-form parts
  and its complex's options (mollify, flip_eps, star1_floor, from the solve's provenance) with its eigenvectors, in preparation; the node rebuilds the same complex and refuses
  2-form parts of another triangulation. The shares are measurements: each energy reduced (`measurement.reduce`, measure `sum` under the energy's name) over any partition,
  `exact_share`, `coexact_share`, `residual_share`, `exact_misfit_share` derived at read (over a tile the parts are not orthogonal: a share may pass 1 there). Not changed: `crossspectrum --estimator analytic`
  keeps its Meyer band. Morlet is not compact: its tail leaks across DC and Nyquist for a wide band near either (the band is still checked below Nyquist). The definitions
  `cortical_band_cells` (lists) and `global_bursts` (lists) set `shape: meyer` (scale selection splits power across bands); `global_bursts`, `band_bursts`, `cortical_band_cells`,
  and `cortical_band_cells_current` take the new default, so their stored numbers differ from before 2026-11-14 (rerun with `--replace`).
- **The plan** (`run --dry-run`): the inputs found (absent or ambiguous inputs refused, named), every step planned in order, what each makes (its
  manifolds, value type, ℝ or ℂ), the outputs' places. Nothing is read beyond the rows; nothing is written.
- **The run:** each step's constants bound once (the taps; the kernel padded to every channel of the recording; the surface's gradient and area-weighted
  normals); the recording read by io's chunk service, each chunk of the span (chunk 0 by default) handed over WITH ITS HALO from the neighbouring chunks
  (`haloed_loop`: the Studio's `#haloed`; the halos of the chunk steps composed, each leaving the halo the steps after it need), so every chunk is exact on
  its own columns and the result does not depend on the chunk width (`test_runner_execute`: 3 or 9 chunks, both within 1e-7 of the whole recording
  computed at once); `measure` folds the Studio's `reduceTiles` over the good samples (the bad spans the chunk service masks); after steps; then STORE.
  **Time sub-blocks** (2026-11-13): after the last chunk step that needs a halo (a band's taps), every step is pointwise in time, so that tail runs on the
  chunk's own columns a SUB-BLOCK at a time, in time order (`--block-samples`, default 1024; 0 the whole chunk). Only memory changes: a free kernel on a
  20 484-vertex cortex holds 61 452 rows × the block, not × the chunk.
  The width is in the cache key (the folds' sums may round differently); `test_global_bursts` checks 0, 333 and 1024 give the same values.
- **What is stored:** each output a Field `<pipeline>.<output>.<anchor>` on the manifold its steps made it on, in that manifold's folder, with its
  provenance (the pipeline, the definition's sha256, the parameters, the inputs' ids, the span, the chunks read, the halo) and its CACHE KEY (the same four
  things): a rerun is a hit; a node made with another key is refused before any data is read, unless `--replace` (then moved aside, never deleted).
- **Three kinds of output** (`outputs: {name: {kind: …}}`): a Field (above); `kind: measurement`, a measurement over a time partition stored as the
  envelope is (an array [rows × tiles × 2] beside the recording, `selection_measurement` rows on the level, `computed_by` naming the pipeline and its cache
  key's hash). **Packed:** the measurement outputs of one run over the SAME Selection with the same rows, members and
  counts are stored in ONE array [rows × members × (components + 1)] (each measure's components in the definition's order, the count last), one row a measure
  component pointing at it with its `component` (the schema's array-backed rule; the measurements of a band set share arrays the same way). `global_bursts` stores its 24
  measurement outputs in 3 arrays, not 24. The report's `stored` entry says its `array`, first `component` and `components`; the lookup (`measurement_values`,
  `measure query`) and `read` find a measure by its row's component. A rerun with `--replace` of a run stored before this (one array per measure) rewrites the
  rows to the packed array and leaves the old arrays unreferenced (`dataset compact` is not the tool; delete them only with the maintainers' OK); `kind: spans` with `types: [name]`, an event set stored as the importer stores events (a spans Selection on the time line, code = the type
  through its dictionary, the channel in `params_json.channels`, how it was made in `params_json`, as the Studio's `searchAtlas` records its query).
- **Where nxr-neuro differs from nxr:** nxr's `mean_square` accumulator is the pointwise `power` followed by `measure` (reductions belong to the Selection
  system); nxr is to take the same steps (both tools share the step names).
- **Not built yet:** `measure` over a stored time partition other than a constant-Q level of a band, and the steps' reductions over a surface partition (the phase sums written as
  measurements); workers (one chunk at a time, as decided); parity against `sourcepsd` and nxr; today's verbs re-expressed as definitions.

## The pipeline type system: types (`pipelines/graph/include/nxr/graph/types.h`; spec `docs/specs/2026-11-17-type-system.md`, step T1)

A `PortType` says what a port carries before any data flows. Beside C1's members (class, value type, rows, axis, rate, unit, `on`/`to` labels, `facts`) it holds, once `typed` is set:

| member | meaning |
|---|---|
| `domain`, `target` | where the values lie: a product of **manifold terms** (`Manifold`: a stored id, an anonymous array manifold, a signature variable, one a node makes, a restriction `M|selection`, a member set `members(P)`, a derived one `glued(M)`), and the cells; an operator's `domain` is its source and `target` its target |
| `quantity` | what the values are (field), the measure (measurement, statistic), the kind (operator): a **closed vocabulary**, `nxr::graph::quantities()`; a name outside it is refused when a node's signature is registered |
| `res` | the time band (Hz, shape) and the spatial level (ladder, level or stack of levels, shape): the level IS the band |
| `sel` | a selection's shape: its type, whether its members are disjoint (`partition`: a measurement may be taken over it), its identity, what it was detected over |
| `of` | what a statistic is of: the quantity, resolution and domain before the reduction |
| `spin`, `gauge` | G5: 0 scalar, 1 tangent vector, n an n-RoSy, -1 a conjugate; the gauge of a tangent value (empty for a scalar and for an embedded `vector3`) |
| `transition` | an operator of kind `transition`: a change of coordinate system of one space |

The printed form (`print`, `parse_type`) is the one used by messages and `--dry-run`: `field<cortex(kernel) × frames(tiling); complex vector3; current; band 8-12 Hz morlet; level tiles@5>`. A manifold prints as `label(id)`, a made one `label(~use.output)`, an anonymous one `#n`, a restriction `M|selection`, a member set `members(P)`, a derived one `relation(M)`. `unify` binds the variables of a signature to the terms they meet; its only rules are equality of ids, anonymous numbers, member sets, restrictions and derived terms, a variable's constraints (a unit: a time variable binds a line in seconds; raw samples: not tiled from another line; a topology: path or cycle), and one subsumption at an operator's source (a kernel from a restriction reads the whole manifold, or the same restriction).

**The structured JSON form** (`nxr/graph/type_json.h`: `to_json`, `port_type_from_json`) carries what the printed form leaves out, for programs that build on the types:
`{"class", "value", "complex_parts", "components", "domain": {"factors": [manifold, ...], "cells"}, "target", "axis", "streamed", "sfreq", "rows", "quantity", "unit",
"resolution": {"band", "band_shape", "ladder", "levels", "spatial_shape"}, "nonlinear", "spin", "gauge", "selection", "of", "transition", "typed", "text"}`, a manifold as
`{"form", "key", "label", "kind", "topology", "coordinate_system": {"space", "name", "unit"}, "n", "spacing", "origin", "rate", "unit", "samples", "made_from": {"relation", "of"}, "of", "by", "irregular", "text"}`. A line has no role: it is a 1-D mesh with a `topology` (`path` or `cycle`), a coordinate (system, origin, spacing, unit) and, when made from another line, `made_from` (`tiled` for a frame line, `transform` for a frequency line). A signature that sends `role` is refused. A string is
read as the printed form. The two forms are held equivalent by `test_graph_type_json`: a type read back from its JSON prints as the type did. `facts` are never part of it.

**External signatures** (`external_nodes`): a node another program implements is declared as JSON in the terms of the registered signatures:
`{"nodes": [{"name": "render.surface", "kind": "array", "move": "pointwise", "inputs": [{"port": "field", "class": "field", "on": [{"form": "variable", "key": "S", "kind": "surface"}],
"value": ["real"], "components": 1}, {"port": "surface", "class": "manifold", "on": ["S"]}], "outputs": [], "parameters": [{"name": "colormap", "default": "viridis", ...}]}]}`.
An input pattern takes a class, its domain's factors (`on`; an operator's `to`), value types, `streamed`, `linear`, `optional`, `over`, quantities and `components`; a variable's
`kind` binds only a manifold of that kind. Such a node is checked by the shared checker like any other and refuses to run.

### Signatures and the shared checker (`signature.h`, `checker.cpp`, `signatures.cpp`; step T2)

A node declares a **signature**: a move, and its ports' types written with manifold variables (`var`, `time_var`, `frames_var`, `var_rest`, an operator's source `from_or_restriction_of`). `type_use` solves a use by unification, builds the outputs' types from the signature (so nothing a node's own code does can get around the rules), applies the move's rule, then runs the node's small `refine` (what the signature cannot say: a parameter that picks a band, a level, a form), which may only narrow the outputs or refuse. It is **one function** for every graph: `check` runs it beside a node's old `plan` when every input is typed, `type_check` runs a graph of signed nodes by itself.

| move | rule |
|---|---|
| `pointwise` | the domain is unchanged; non-linear nodes set `nonlinear` |
| `apply` | an operator maps one factor (its source may be a restriction: the one implicit move); the field is linear (D106); a tangent field needs an operator built in its own gauge (`geometry.frame_rotate` otherwise) |
| `compose` | two operators meet: A to B, B to C gives A to C |
| `restrict` | at most one factor narrowed (`select.at_level`: the stack of levels) |
| `measure` | the **only** move that removes a factor: a partition's members replace the factors it partitions; the partition must be disjoint; a ladder's level is chosen by the field's band; a joint partition's band and level must equal the field's (the level IS the band); a partition of the streamed factor needs an accumulate node |
| `summarize` | across a partition's members: a statistic that remembers what it is of; a field is measured first (a fused estimation node declares the partition it reduces over) |
| `detect` | events found on a scalar measurement over a time partition's members: a selection with an identity and what it was detected over, and measurements over its members |
| `make` | no field input |

The checker never inserts a transition or a restriction: it refuses, names the rule and what to add. A refusal reads `<use> -> <output>: port <port>: <why> (<value> = <printed type>, from <the use that made it>)`. `builtin_signature(name)` holds the signatures of the registered nodes the shipped definitions use; `Registry::add` attaches one to a node of that name when the ports are the node's. Tests: `test_graph_moves` (a graph per move and rule), `test_definitions_typed` (the 15 shipped definitions check; the spec's ten seeded mistakes are refused). The closed vocabulary of quantities is `nxr::graph::quantities()`; a signature naming a quantity outside it is refused when registered.

## Events on the time axis (`run band_bursts`; spec `docs/specs/2026-11-13-time-events.md`, audit `docs/audits/2026-11-13-time-event-labelling.md`)

**Why.** Standard time-series event labelling: the time axis of a recording labelled with events (bursts, spindles, slow waves, bad
segments, beside the imported Brainstorm events), through the selection/measurement system. An event set is stored as the importer stores Brainstorm's
events, so every reader of `<rec>_events` reads it.

- **`band_bursts`** (a definition, `pipelines/definitions/band_bursts.yaml`): the band's envelope (`envelope`), its `amplitude` (mean |z|) per tile of the
  constant-Q level of the band (`measure` over the ladder; `run select_time_grid --rule constant_q` first; a band that is no level's is refused, D84), then `detect_spans`.
  - Per channel (`combine each`, `detect.spindle`'s rule) or on the mean over channels (`combine mean`): the median and robust SD (1.4826 × MAD) over the
    good tiles of the run's span; tiles ≥ median + 1.5 SD entered, extended while ≥ + 1.0 SD, gaps ≤ 0.1 s joined, spans ≥ 0.4 s kept.
  - The reference parameters are Rheome's alpha episodes (`flow.labelalpha` → `detect.spindle` in envelope mode); the default band is alpha's level, 8–16 Hz
    at the hertz anchor.
  - The thresholds' ingredients are a STATISTIC (2026-11-13): `summarize` stores the median and robust SD per channel of the tile amplitudes (output
    `baseline`), and `detect_spans` reads them. A burst's peak and mean amplitude are the measurement rows of its tiles, read through the lookup, not stored again.
  - Stored: the amplitude rows on the level, and `band_bursts.bursts.<recording>`, a spans Selection (type `burst`). A rerun is a hit; another job's outputs
    are refused unless `--replace`.
  - **Tiles and events:** the tiles of the level are a dense PARTITION of the time axis; their measurements are the recording's windowed
    features: `amplitude` (mean |z|) and `peak` (`envelope_max`, the largest |z|) per tile, in one chunk loop, sharing one `count` row. The bursts are a
    SPARSE Selection derived from those tile measurements: each is a run of tiles of its band's level, so it is measured FROM its tiles, exactly, with no
    second pass over the recording (`measure_events`, `nxr::select::rollupEvents`): `burst_amplitude` (the count-weighted mean of the tile means),
    `burst_peak` (the max of the tile maxima), `burst_duration` (`duration`: tiles x the tile width, s) and `burst_interval` (`interval_next`: the next burst's
    start on the same row minus this stop, s; NaN for the last; reduction none). Stored array-backed over the run's own bursts Selection, [rows x bursts x 2]
    (the value, the samples counted), the member axis the bursts' order; a burst is on the row of the channel that found it and NaN on the others (the mean
    trace's bursts: every row, each on its own tiles). A burst holding a tile not whole and good is NaN for the sampled measures. The lookup
    (`measurement_values`) reads them back as [rows x bursts].
  - **The statistics across the bursts** (`burst_*_stats`, `summarize`): `count`, `sum`, `median`, `mean`, `sd`, `percentile:25/75/90` by default (the
    parameter `burst_summaries`), per channel. Each statistic records its span and the span's GOOD SECONDS in its `options` (`good_s` = the span's samples
    less the bad ones, over the rate), so the burst rate (count / good_s x 60 per minute), the occupancy (sum of `duration` / good_s) and the interval's CV
    (sd / mean) are DERIVED AT READ, never stored.
  - **Not built:** a median per burst (medians do not roll up; it needs the samples), `n_cycles` (needs the burst's peak frequency), `relative_peak`, the rise
    and fall fit, and events not derived from tiles (imported events: arbitrary edges; a generic `event_measures` pipeline, where the band needs a place in the row's key).
- **Tested** (`test_band_bursts`, synthetic): the amplitude per tile equals the direct mean of |z| (5e-8); planted bursts found on their channels and no
  others; the same with another chunk width (8e-8); a bad span's tiles NaN; `combine mean`; a rerun a hit; a band that is no level's refused.
- **What it cannot resolve:** the time resolution is the band's tile (alpha at 1200 Hz: 0.85 s), and the band's response can carry an edge into the next
  tile; the band is an octave level, so 8–12 Hz is no band here; a burst on a sensor says nothing about where in the cortex it is.

## Gabor bursts: bursts in time matched by atoms of free width (`run gabor_bursts`; spec `docs/specs/2026-11-20-gabor-bursts.md`)

**Why.** A burst is a transient oscillation with a centre, a frequency, a duration and an amplitude. A band's envelope (`band_bursts`) ties the time resolution to the band: a short and a long
burst at one frequency are read through the same filter. A unit-energy Gabor atom ψ = (πσ²)^(−1/4) e^(−(t−t0)²/2σ²) e^(i2πf(t−t0)) with a FREE width σ is the matched filter of a Gaussian burst
A e^(−(t−t0)²/2s²) cos(2πf(t−t0) + φ): |W| = |∫ x ψ*| is largest at σ = s, where |W| = (A/2) π^(1/4) √σ. So one peak of |W| over (t0, f, σ) gives the burst's centre, frequency, width (FWHM 2.355 σ),
cycles, amplitude A = 2|W| / (π^(1/4) √σ) and phase, in physical units, and two bursts at one frequency are told apart by σ.

- **The method** (the default reproduces the reference detector the spec ports, written in Python; chosen 2026-11-20):
  - the atoms: the time filter bank's `Family::Gabor` (`timefilterbank.h`: one member an atom, gain G times the unit-energy atom, `Normalisation::UnitEnergy` recorded on the bank; analytic
    only; not a frame), on the grid `gaborGrid`: widths σ_min · 2^(k/per_octave) up to σ_max, at each width the band's frequencies widened by one σ_f = 1/(2πσ) in steps of σ_f/2, f·σ ≥ min_f_sigma.
    The node `signal.gabor` (accumulating, with the halo) gives |W| on every atom, EACH WIDTH AT ITS OWN RATE (the reference's): one spectrum of the block, then one width at a time, each atom read on
    its own bins at every D-th sample (`nxr::signal::wtDecimated`: exactly the full-rate values there), D the largest power of two keeping at least 3 samples a σ; each width's |W| is a series on its own
    frames, a line tiled from the recording's samples (frame j at sample j·D). Rows atom-major (row = atom × channels + channel), one row group a width. `decimation: none` reads every width at every sample (for checks).
  - each atom's own LEVEL: the median and robust SD (1.4826 × MAD) of its |W| over its width's frames (`measurement.summarize`, extended to row groups each over its own members): a statistic, never
    computed inside a node.
  - the candidates (`select.events.peaks`): per channel and width, the local maxima of |W| in (frequency × frame) (3 × 3, the box clipped at the edges) at least median + k × SD; frames within
    max(edge, 3σ) of the run's ends left out. A candidate's time is its frame's instant.
  - the fit (`select.gabor_fit`, on the channel's samples, `collect`, as `select.risefall`): a parabola through log|W| in frequency (± σ_f/2) and in log width (× 2^(±1/8)), twice, by the inner
    product itself (`nxr::signal::gaborFit`); then t0, f, σ, FWHM, cycles at half maximum (2.355 σ f), amplitude, phase, |W|, the width test (`atom_width_peak`: |W| larger than at σ × 2^(±1/per_octave),
    inside the grid's widths) and R² (`atom_r2`) of the atom against the band's signal in its half-maximum window.
  - the gates (`select.events.spans` rule `keep_where`, extended with `x` and `where`): a peak along width, at least `min_cycles` (2) cycles, f inside the band; then the suppression
    (`select.events.suppress`: strongest |W| first; an atom within the larger time resolution σ and the larger frequency resolution 1/(2πσ) of a kept one on its channel is dropped); then R² ≥ `min_r2` (0.7).
  - the edges (`select.events.spans` rule `edges`): `tangent` (the default): the envelope of the channel read by an analytic Morlet member at the burst's f, width σ_r = max(σ/2, 3/(2πf)), over
    t0 ± max(3 s, 6σ); its peak inside the half-maximum window; the troughs either side (the first local minimum below half the peak, else the lowest point within 1 s); onset and offset where the
    tangents at the steepest rise and fall meet the troughs' levels (`nxr::select::tangentEdges`). `half_max`: t0 ∓ FWHM/2. A burst whose envelope has no such rise and fall has no tangent edges and is
    counted, not kept.
  - the global windows (`select.events.spans` rule `union`): the bursts of every channel joined where they overlap in time and in frequency (f ± 1/(2πσ)), chained (`nxr::select::overlapGroups`);
    each window records its bursts. `overlap: time` joins by time alone.
- **Defaults** (the reference detector's): band 8–35 Hz, σ 25–640 ms, 4 widths an octave, min_f_sigma 0.5, k = 4, edge 0.5 s, 2 rounds of the fit, 2 cycles, R² ≥ 0.7 against the band's signal
  (`r2_shape: morlet_ladder`, the reference's: the recording through Σ g² of the members of a Morlet ladder, 1–64 Hz, 8 an octave, whose centres lie in the band, by `signal.band_filter`; `meyer` and `morlet` are
  named options), tangent edges, windows by time-frequency overlap, each width at its own rate. The halo is the widest atom's reach down to `tolerance` (1e-6) of its response, so the result does not depend on the
  chunk size. Every option is a parameter, recorded in the provenance and the cache key.
- **Stored:** `gabor_bursts.candidates.<recording>` (spans, type `atom`: every candidate, its frame) and `atoms` over them (`atom_time`, `atom_frequency`, `atom_width`, `atom_fwhm`, `atom_cycles`,
  `atom_amplitude`, `atom_phase`, `atom_coefficient`, `atom_width_peak`, `atom_r2`, [channels × candidates], NaN off a candidate's channel); `bursts` (spans, type `burst`, [onset, offset) on the channel)
  and `burst_values` over them (the atom's measures, `onset`, `offset`); `windows` (spans, type `window`, no channel; each window's bursts in its params). The measures are in the registry (codes 1067–1078).
- **Memory, stated and checked.** One block of `signal.gabor` holds channels × (the padded block × 8 + its spectrum × 16) bytes, then one width at a time: one atom's decimated coefficients and the
  width's |W| on the block's frames (atoms of the width × channels × the block's frames × 4). Before any data flows the node computes this for the largest chunk (2^15 samples and the halo) and refuses a
  block over `budget` (1 GB by default), saying the size and the remedy (fewer channels, a narrower band, a smaller σ_max, a shorter chunk). Worked: 270 channels at 1200 Hz, the default grid
  (8–35 Hz, 1218 atoms a channel): 0.50 GB a block (13–30 Hz, 804 atoms: 0.49 GB); every width at the recording's rate it would be about 108 GB. The run also holds each width's |W| over the span:
  Σ over widths of atoms × channels × (rate / decimation) × 4 bytes a second: 30.7 MB per second of recording at 270 channels and 1200 Hz on the default grid (21.7 MB on 13–30 Hz), so a 5-minute span
  holds about 9.2 GB; `collect` holds the span's samples twice besides. That rate times the span asked, plus the block, is checked against `budget` on the first block, before anything is held, and
  refused with both figures, the longest span that fits and the remedies (a shorter span, fewer channels, a narrower band, a smaller σ_max, a larger `budget`). A level computed without holding the
  series is not built (#189).
- **Tested** (`test_gabor`, `test_gabor_golden`, `test_gabor_bursts`, synthetic): the decimated transform equals the full rate at every D-th sample (1e-12); against the reference detector (`golden/gabor_golden.txt`, its generator `gen_gabor_golden.py`; the reference's folder in
  `NXR_GABOR_REFERENCE`): the grid (6e-14), the fit of each of its candidates (f, σ, amplitude, phase, |W| to 5e-13), its burst test, its merge (the same atoms, in order) and its tangent edges (1e-9 s); the grid; unit energy; |W| = (A/2) π^(1/4) √σ for a matched burst (A recovered to 1e-6) and the inner product equal to the bank's
  transform; a planted burst's f, σ, A and φ recovered from the nearest grid atom; two bursts at one frequency told apart by σ; the width test; R²; the local maxima, the suppression, the tangent edges
  of a Gaussian (t0 ∓ 2σ), the groups by overlap. In the pipeline: five planted bursts on three channels recovered once each within their resolution; every burst passes the gates; a burst narrower than
  the ladder is a candidate and no burst; a stricter R² keeps fewer; half-maximum edges t0 ∓ 1.1775 σ; the same bursts to the last bit with chunks of 3000 samples; each width at its own rate gives the bursts every width at the
  recording's rate gives, within their resolution; a block over the budget refused at the plan with its size, a run whose series over the span is over it refused with the longest span that fits; the windows by time-frequency overlap and by time.
- **What it cannot resolve:** an atom's time and frequency are known to its own resolution (σ and 1/(2πσ)): two bursts closer than that are one atom, and a burst's time is its frame's instant, within σ/6 (the
  fit refines frequency and width, not time). A burst that is not Gaussian in its envelope is matched by the nearest width; R² says how much of the band's signal one atom explains there. A burst
  outside the widths (narrower than σ_min, wider than σ_max) fails the width test. The levels are each atom's own over the run's span, so a recording full of bursts raises them. Per channel only:
  a burst on a sensor says nothing about where in the cortex it is; the spatial adjustment of the reference detector (its heat-kernel blobs on the sensor graph) is not built.

## Patches on a static cortical map (`run patches`; `compute/include/nxr/regions.h`; spec `docs/specs/2026-11-13-cortical-events.md`, phase 1)

**Why.** The spatial equivalent of the bursts of a band, on a map with no time axis (a PET map; any scalar Field on a surface): where the map stands out from its
surroundings at the size of a level of the surface's tile ladder, as where, how big, how strong. The same idea as `band_bursts` with the selection system: a level IS
its band, the baseline is a statistic, an event is a connected run of tiles above it, each event measured from its tiles. Only the neighbourhood differs.

- **The graph**: preparation reads the map, the surface and the level's tiling and type-checks them (a scalar Field on a
  surface; a ladder made by `run select_dyadic_surface`; the level exists; no hole in the map), then the nodes, each a pure function of matrices, and the results go back to io.

      map → spatial_band → measure (per tile) → summarize (baseline) → sign → detect_regions → measure (per patch) → summarize (across patches)

  - **spatial_band** (`nxr::signal::gwt(bank, member, ...)`): the Weyl band of the level, member `l` of `weyl(mean piece area, l + 1)`. The pairing is by construction and is
    checked on every run: the member's λ range equals the level's `lambda_min … lambda_max` in the ladder's record (a ladder made with another area is refused). Meyer
    edges ± 1/3 octave. The transform is EIGENMODE FIRST, CHEBYSHEV SECOND (below: *The spatial band's transform*); `--transform auto` is the default. The series (`chebyshev`, or the announced fallback) is a
    Chebyshev series in the cotangent pencil of the intrinsic Delaunay mesh (the degree is the smallest of 40, 80, … 5120 whose series is within `--tolerance` 0.01 of the band's gain, else the run is refused,
    never silently coarse: coarse levels and fine meshes need a high degree); `--transform exact` is the dense modes of the mesh (at most 4096 vertices; the reference the tests compare to).
    Stored as the Field `<map>_band_L<l>` (float64, the map's unit): the band-passed map; its `producer_json` records the transform, the source of the modes (or the degree) and any `warnings`.
  - **measure**: the area-weighted mean, max, min and abs_max of the band over the level's tiling (`nxr::select`, `run measure`, recipe `patches-barycentric`).
    These rows are OF THE BAND FIELD. A plain `run measure` of the map over the same tiling is a different row (a different `of_field_id`) and is never overwritten.
  - **summarize**: the baseline of the level, the median and robust SD (1.4826 MAD) of the tile means (`<map>_band_L<l>_baseline`, a statistic).
  - **sign** (`nxr::select::applySign`): positive (x), negative (−x: dips), magnitude (|x|). The caller's; the detection never knows. For magnitude the baseline is of the
    absolute tile means, computed with the same functions (the stored statistic is of the signed means); the record says so.
  - **detect_regions** (`nxr::select::detectRegions`, one pure function on a graph: values per node, neighbour pairs, node weights, thresholds, size limits → a label per node):
    the tiles ≥ median + `--exit-sd` SD form connected regions over the pairs of tiles joined by a mesh edge (the level's stored adjacency; hemispheres never touch); a region is
    kept when a tile reaches median + `--enter-sd` SD and its area is within `--min-area` … `--max-area` (m², the weights' unit; none by default). The thresholds default to
    Rheome's alpha-episode rule (1.5 and 1.0), as `band_bursts`. In time the same function on the line graph reproduces `crossings` + `keepSpans` (tested). NO merging
    across gaps; a named merge option is not built. NaN (no value) is in no region.
  - **measure per patch**: each patch is a union of tiles, so its mean, max, min and abs_max OVER THE PATCHES Selection are the exact roll-up of its tiles (the area-weighted
    mean of the tile means; the max of the tile maxima); `area` (m²) and `n_elements` are its geometry. `n_tiles` and `centre_vertex` (the vertex nearest the area-weighted
    centroid) are nxr-neuro's own measures (`measures_nxr.json` 1012, 1013; a count and a place).
  - **summarize across the patches**: `count, sum, median, mean, sd, percentile:25/75/90` (`--summaries`) of the patches' area, n_tiles, mean and peak (max for positive, min for
    negative, abs_max for magnitude), as statistics `<map>_patches_L<l>_<sign>_<what>` (`run statistic --members-own` is the area and n_tiles: the members' own rows).
- **Stored**: the band Field; the tiles' rows; the baseline statistic; the patches as a `set` Selection `<map> patches L<l> <sign>` on the surface (a code per vertex = its patch's
  ordinal numbered by its lowest tile, −1 in none; `n_members` = the patches (the writer does not count the −1); dictionary `patch k` with its tiles and centre); its record
  (`params_json`: map, level, band with its λ range and transform, sign, thresholds, baseline, limits, cache key); the patches' rows; the statistics. A rerun with the same
  inputs and options is a cache hit at every stage; another choice at a stage whose result exists is refused unless `--replace`, and a refusal leaves what is stored untouched.
- **Reference.** The method is nxr-neuro's own, assembled from named parts: the Weyl pairing of the ladder (the Studio's `engine/ladder.ts`, nxr `geom.ladder`), the graph filterbank
  (Rheome `@graphfilterbank`, nxr, the Studio's dyadic frame), the measures of the Studio's `measureOverPartition` (`nxr::measurement::reduce`), and the hysteresis rule of nxr `spindle_envelope` / Rheome `detect.spindle`.
  Rheome's `blobscale` is the same idea on a Mexican-hat ladder.
- **What it does NOT do.** No instrument limit is built in (a resolution, a leakage, a smallest size are the caller's `--min-area`, named; the maintainers forbid building them into tools).
  One level per run: a coarser level is not a roll-up of a finer one (it is its own band). It finds events of one sign at a time. It does not merge, track or link patches across
  levels or time (space × time is phase 2).
- **What the band-pass does, and the reader should know.** A band-passed bump has side lobes of the opposite sign around it (a Mexican-hat response), so a strong dip has a
  positive ring of the order of half its own tile mean: at 1.5 SD (the default) that ring is a positive patch; at 3 SD it is not (the test uses 3). The tile mean of the band is the local height
  relative to the surroundings AT THE LEVEL'S SCALE, not the map's value. The baseline is across the level's tiles: with few tiles (a coarse level) the median and robust SD are
  coarse estimates; a map with no variation at the level is refused (the SD is round-off).
- **Tested** (`test_regions`, `test_patches_pipeline`, synthetic only): a line graph gives the spans of `crossings`; a planar graph, NaN nodes, size limits, the sign; on a
  two-hemisphere brain, planted smooth bumps found at the level of their size as one patch of the centre's tile each and no others, the label of every vertex, area, mean and peak
  against an independent computation from the stored band and the mesh (to 1e-9), a dip found with sign negative, a larger bump not found at that level and found at a coarser one,
  plain tile means and the band's separate, a rerun a hit and another choice refused unless `--replace`, the exact modes and the Chebyshev series agreeing, and the refusals.

## The spatial band's transform: eigenmode first, Chebyshev second (`nxr/pipelines/spatial_band.h`)

> The default spatial transform is the stored eigenmodes when they exist, with a Chebyshev fallback that is never silent: when eigenmodes are absent they are computed, with a message. Eigenmodes first, Chebyshev second.

**Why.** The level-l band F_l of a surface's fields (`run patches`, `meeg.apply_kernel` with `tiles`/`level` in `run measure --set cortical_band_cells` and `run global_bursts`) is a function of the surface's Laplace–Beltrami spectrum. The subject's stored
Hodge–Dirac package holds that spectrum (its exact modes); the series only approximates it. So the stored modes are the first choice, and the series never a silent one.

- **`spatial_shape`: a spatial level is the hat peaked at its band's centre.** `meeg.apply_kernel` with `tiles`/`level` takes `spatial_shape`:
  - **`mexhat` (the DEFAULT of the step and of the definitions that use it)**: g(λ) = λs²·e^(−λs²/2) / max with s² = 2/c, i.e. g = (λ/c)·e^(1−λ/c): unit peak at λ = c, 0 at 0
    (`filterbank::mexicanHat`; nxr `vortex_cores.hat_operator`). **c is the GEOMETRIC centre sqrt(λ_min · λ_max) of the level's Weyl band** as the ladder records it (nxr's 4π·2^(l−0.5)/A;
    level 0, whose band starts at 0: λ_max/√2, as if it began one octave below). The record keeps the centre, the floor and the reach (`spatial_shape`, `hat_centre_lambda`,
    `hat_gain_floor`, `hat_reach_lambda`), and the plan's cache key carries the shape and the centre.
  - **`meyer` (named)**: the level IS its Weyl band (compact edges, the frame: levels add up). Hats overlap and do not add up, so every analysis that splits power across levels or
    bands (scale spectra, the joint eigenmode × frequency plane, power shares, scale selection) sets `spatial_shape: meyer` (and `shape: meyer`): the definitions `cortical_band_cells` (lists) and `global_bursts` (lists)
    do. `global_bursts`, `cortical_band_cells` and `cortical_band_cells_current` take the hat (their stored numbers differ from before 2026-11-14; rerun with `--replace`). The library's `SpatialBandOptions`
    stays `meyer` by default, so `run patches` and the cell quadrature are unchanged.
  - **The tail.** The hat has no compact support: the eigen route needs modes to where its gain is under 1e-3 (about 10 centres up, `hat_reach_lambda`), against 2^(1/3) · λ_max for Meyer. A stored package
    that reaches the Meyer band often does not reach the hat: `auto` then falls back to the Chebyshev series, ANNOUNCED (a warning naming the hat, the level, the package and the reach); forced `eigen` refuses
    with that reason; a computed package is sized for the reach (Weyl's law). The series follows the hat's smooth gain within `tolerance`. The hat's gain below its peak is linear in λ: it passes the
    large scales of the field (the constant and the lowest modes with a small weight), where Meyer's band is cut.
  - **Tested** (`test_hat`, `test_spatial_hat`): the gain equals nxr's (golden `gen_hat_golden.py`: the gain lines of `hat_operator` on a synthetic sphere spectrum, levels 1 to 5; nxr scales by the sample maximum, here the analytic one);
    the dense exact band, the stored/computed eigenmodes and the Chebyshev series agree on a sphere (eigen vs exact 4×10⁻⁷, series vs exact 4×10⁻¹¹), the Riesz part follows the hat, the long tail falls back announced and forced
    `eigen` refuses.
- **`form`: the curl of the tangent current, folded into the operator.** `meeg.apply_kernel` takes `form`: `none` (the default), `curl_tan`, `div_tan` or `normal` (n·x). For a FREE kernel (x y z a vertex)
  the form is folded in preparation into ONE linear operator: kernel → the current's tangent part → the form → the spatial band (hat by default, `spatial_shape: meyer` named), so the chain stays one linear map before any
  non-linear step (D106). The output is ONE number a vertex, complex when x is (the analytic signal's curl): the value the project's vortex cores are cut from (nxr `vortex.Cortex.readings`: `div_curl(tangent(q))`, then
  `vortex_cores.hat_operator` on it). The form is the Studio's (`forms::formOfFreeColumns`, a matrix of free currents in, a matrix of forms out, float32 inside; the same `curl_tan` as the `form` step), taken on the native
  mesh the kernel lives on; the band is the glued package's, read back on the kept vertices. **Order matters:** the curl BEFORE the hat is not the hat before the curl (the `form` step after `meeg.apply_kernel` is the latter).
  `form` is in the run's key and provenance; it needs a free kernel on a surface and refuses `envelope: spatiotemporal`.
  - **On the glued surface, the bridge filled by the hat:** with a spatial band (`tiles` and `level`) the output is on EVERY vertex of the GLUED surface (`source/<surface>_glued`: the kept
    vertices, then the bridge, the corpus callosum included), not only on the kernel's kept vertices. The form is zero on the cut medial wall and on the bridge BEFORE the band; the band, taken through the glued
    surface's modes, fills them (exactly nxr `Cortex.coords` → the gain → `P.T`, read at every glued vertex, where nxr's `hat_operator` reads only the kept ones). Still ONE linear operator bound in preparation, now
    [glued vertices × channels]. The output's surface is the glued one (a cortex of two hemispheres; the surface itself for one closed piece). **Eigenmode first, Chebyshev second, announced:** the stored glued
    eigenmodes (`hodgeDirac_eigenvectors`, `hodgeDirac_support` maps them to the kept vertices) when they reach where the hat's gain falls under 1e-3; else the Chebyshev series of the GLUED mesh (never the native mesh: it
    has no bridge), with a warning in the progress and the provenance; `exact` is the dense modes of the glued mesh. The glued surface must be stored (`nxr-neuro run eigenmodes`; the run is refused, saying so, when it is not).
    Without `tiles` there is no band and the form stays on the kernel's own vertices. The Riesz part is not built on the glued surface.
- **`levels` and `glued` (; spatial-events spec, piece 3): the scale space as ONE operator.** `levels: [4, 5, 6, 7, 8]` (instead of `level`) binds one operator a hat level in preparation and stacks them: the output is
  [vertex × level] per component, row = (vertex · levels + level) · components + component (`facts.levels` names them; `select.at_level` reads one back). Each level's band is the single-level step's (same `transform`, `modes`, `spatial_shape`,
  eigenmode first and Chebyshev announced, each level's choice in the run's key and in `prepared`); one level of the list equals the same level asked alone. `glued: true` evaluates the band on EVERY glued vertex for a field with NO form
  too (a free current: each component through the band, the bridge zero before it and filled by it; a form is always glued): the field `power` then sums over the components, with no orientation assumed. **Memory:** the bound
  operator is [glued vertices · levels · components rows × the kernel's channels] in double, held for the run (25 000 vertices, 5 levels, 3 components, 300 channels: ≈ 900 MB; with a form, one component: ≈ 300 MB), and a block of
  frames holds rows × block_frames × 8 bytes a copy (`--block-frames`). **Several levels are refused with `envelope: spatiotemporal`** (the monogenic field is of one level).
- **`select.at_level`** (step, chunk, pointwise): the rows of one level of a multi-level field, [vertex × components], as `meeg.apply_kernel` with `level` makes them (refused for a level the field lacks, or for a single-level field).
  **Tested** (`test_apply_kernel_levels`, synthetic two-hemisphere cortex): curl_tan with `levels: [2, 3]`, then each level's band power per glued vertex equals the direct dense exact hat (the operator nxr's `hat_operator` golden of
  `test_glued_form` fixes at levels 2 and 3), 5×10⁻⁸; the free current with `glued: true`, power summed over the components, 6×10⁻⁸; level 3 of the pair equals level 3 alone exactly; the refusals.
  - **What it cannot resolve:** the sense of rotation read from the CURL MAP is right in 88-100 % of 25-40 mm patches (lab, 2026-10-30); from the current itself it is chance. The bridge's values are the hat's
    smooth continuation of the two hemispheres' curl across the callosal gap, not a measurement of cortex there.
  - **Tested** (`test_form_columns`, golden `gen_curl_golden.py`: nxr's `FaceOperators.div_curl(tangent(q))` and the normal part on a jittered icosphere with a planted rigid rotation: 1e-7 of the largest value; the rotation's
    curl equals 2 cos(colatitude)/R; linearity; refusals), (`test_glued_form`, golden `gen_glued_hat_golden.py`: a SYNTHETIC glued pair of spheres, nxr's `glue_mesh` (the same glued vertices and faces) and the dense
    hat through the glued mesh's modes, read at every glued vertex: stored eigenmodes 1e-4 of the largest value, the dense modes 2e-8, the Chebyshev series of the glued mesh 2e-8; the bridge carries the band; the native read-back is its kept
    vertices; a level beyond the modes falls back announced; one closed piece is its own glued domain) and (`test_apply_kernel_form`: the run's band power on every glued vertex equals the direct operator
    `F_l · form(K)` for the three forms (5e-8); the order against the form after the hat).
- **`transform` = `auto` (the default).**
  1. **Stored modes reach the band**: the package of the surface (`hodgeDirac[__<surface>]`, as `run eigenmodes` stores it) has an exact-mode eigenvalue λ_max at or above the band's top *with its Meyer edge* (`hi · 2^(1/3)`).
     F_l = Φ g(λ) Φᵀ ⋆0 on the package's exact modes (those with a non-zero gain; the constant is added exactly for the level-0 scaling member, which no package stores), g the level's Weyl member, Φ unit-norm in the glued mesh's vertex mass
     (read off the stored potentials ψ/√2).
  2. **No package**: the modes are COMPUTED (`eigen_solve`, enough by Weyl's law: exact modes ≥ 1.25 · A · reach / 4π + 16, A the surface's area; package size = `--modes`, default max(3000, 2 × that), so a later `run eigenmodes` of the standard size is a hit),
     STORED as `run eigenmodes` stores them, and SAID: `eigenmodes computed and stored: N modes (n exact), largest exact eigenvalue λ 1/m^2, name hodgeDirac` in the progress (`{"stage": "spatial_band", "message": …}`), in the
     result's `messages`, and in the provenance (`spatial_band.messages`, `spatial_band.package.computed`).
  3. **Stored modes that do not reach the band** (a fine level), or modes that cannot be computed (a cortex of two pieces with no Desikan–Killiany medial wall, more than two pieces, a package above `max_modes` = 8000): **Chebyshev, ANNOUNCED**:
     a warning in the progress, the report's `warnings`, and the stored `producer_json.warnings`, e.g. `spatial band of level 6 of cortex: CHEBYSHEV fallback, not eigenmodes: the stored eigenmodes (hodgeDirac: 51 Laplace-Beltrami modes, largest
     eigenvalue 8068.98 1/m^2) do not reach level 6 of cortex: its band tops at 10047.9 1/m^2 (12659.5 with the Meyer edge); the series used has degree N (its largest |gain - series| e)`.
- **`eigen`**: 1 or 2, and REFUSED (with the reason, naming the level, the band's top and the modes' top) where `auto` would fall back. **`chebyshev`**: the series, as before, silent. **`exact`**: the dense eigen-decomposition of the input mesh itself
  (at most 4096 vertices): kept as the test reference (the stored-modes band equals it to 7e-8 on a sphere), not a way to run a cortex.
- **The domain of the stored modes is the GLUED surface.** A cortex's package is solved on its hemispheres glued by anatomy (medial walls cut, a bridge grown). F_l is applied on that surface: the field is put on the kept vertices (the package's `support`), zero on the bridge, band-passed,
  and read back on the kept vertices; the cut medial wall's vertices get 0. So the band near the wall is the glued cortex's, not that of two separate hemispheres (the series, on the input mesh, is the latter); the difference is in the provenance (`spatial_band.package.glued_surface`,
  `bridge_vertices`, `domain`). Tested against the dense band of the same glued surface (4e-8).
- **Provenance and cache key (method rule 3).** The band's `info` records the requested transform, the transform used, the source (`stored` | `computed` | `input mesh`), for eigenmodes the package, its mode counts, λ_max, the band's top, the glued surface and the bridge,
  for the series the degree and its error, plus `warnings` and `messages`. The cache key takes what the request RESOLVES to on the stored state (read before anything is computed): `{"transform": "eigen", "package": …}` or `{"transform": "chebyshev", order, tolerance}`;
  in `run`, a step's `resolve` hook puts it in `key.params.resolved` (`apply_kernel:sources`). The key is the same before and after the modes are computed (a rerun is a hit), and differs between transforms.
- **Cost, for later.** The solve is the package's (the standard 3000 modes on a 20k-vertex cortex: minutes to hours, memory of the eigenvectors V × K × 4 bytes ≈ 240 MB); a level needs about 1.6 · 2^l · pieces exact modes (two hemispheres: level 5: ~100, level 8: ~800, level 9: ~1600), so a 3000-mode
  package (about 1500 exact) reaches level 9, not 10. Reading a package reads the whole eigenvector file once per band (the columns of the band are kept); a column-wise read would avoid it. The package is the subject's: computing it here stores it for every later user.
- **Cannot resolve / limits.** Modes are the glued surface's (above). A band whose λ range exceeds the package is the series (announced), never extrapolated. Eigenvalues come from the package's own solve (its mollified intrinsic Delaunay mesh), the series from the unmollified one: they agree to the solver's tolerance, not bit for bit.
- **Tested** (`test_spatial_band`, synthetic): no stored modes → computed, stored (`run eigenmodes`'s rows), the message in the progress, the result and the provenance, the band equal to the dense exact one (7e-8); stored modes → used with no computation and no message, the same key and result; a level beyond the
  stored modes → the series with the warning (level, both λ, the degree), forced `eigen` refused; too few modes asked, over the cap, or a two-piece surface with no atlas → the announced fallback, forced `eigen` refused, forced `chebyshev` silent and writing nothing; a two-hemisphere cortex glued by anatomy → the stored
  modes on the glued surface equal the dense glued band, the wall carries 0; the plan's key differs by transform; `test_cortical_cells` and `test_patches_pipeline` check the announced fallback of the default run.

## Global bursts from the cortex, and Rheome's rise and fall (`run global_bursts`; `compute/include/nxr/risefall.h`; spec `docs/specs/2026-11-13-cortical-events.md`, "Global bursts: WHEN, from the cortex")

**Why.** WHEN the cortex, as a whole, is in a band's burst (the whole-brain or each hemisphere's), found on the source estimate's cells and refined on samples; WHERE it is, and how big, is the next section (the same run).

- **The spatial band** of the cells (`meeg.apply_kernel` with `tiles`/`level`) takes `transform: auto` and `modes` as `cortical_band_cells` does: stored eigenmodes first, computed and stored when absent (said), Chebyshev an announced fallback (*The spatial band's transform*); the run's `warnings` and `messages` are in the report and the stored provenance.
- **Detection on the time tiles of a trace** (`reduce_cells`): per time tile, the MAX over the level's spatial tiles of the joint cell amplitude (phase 2a's cells), whole brain (`groups: whole_brain`, default) or per connected piece (`per_hemisphere`);
  then `detect_spans` and `summarize` as `band_bursts` (Rheome's 1.5 / 1.0 SD, 0.1 s, 0.4 s; the baseline a statistic). Whole brain with `combine: mean` gives GLOBAL events (a spans Selection of type `burst`, no row); per hemisphere
  needs `combine: each` (the plan refuses the mean over hemispheres) and each event carries its hemisphere in `params_json.channels` (the row: 0 the first piece). One set, the hemisphere a property of the event.
- **Edges refined on SAMPLES, coarse to fine** (Rheome's design): in the same chunk loop `tile_max` makes, per sample, the area-weighted mean |z| per spatial tile (`nxr::partition::meanOverMembers`, a spatial reduction with time unreduced) and the max over the
  tiles of each group; `collect` keeps that [groups × samples] series (held in memory: groups × samples doubles; not stored; at the recording's rate: no multirate). The [tiles × samples] intermediate is never stored.
- **`select.risefall`** (`nxr::select`, a port of `rheome.detect.risefall`): each burst fitted on the sample series over [start − margin, stop + margin] by the double sigmoid (`linear`, or `quadrature`: an episode adds to a background in power), the series
  smoothed (0.1 s), scaled to its maximum, bounds and start as Rheome's: per burst `t_on`, `t_off` (half-rise and half-fall, s on the recording's clock), `tau_rise`, `tau_fall`, `fit_baseline`, `fit_height`, `fit_r2`, `plateau_duration` (new measures 1014 … 1021,
  reduction none), stored over the bursts as the roll-up measures are, with their statistics across the bursts. `min_r2` (labelalpha's is 0.7) sets the fit of a burst below it NaN; the default keeps every fit and stores its r². The `posterior` model is not
  built and is refused by name. The optimizer is a bounded Levenberg–Marquardt where Rheome's is `lsqcurvefit`: the same minimum where the fit is well posed, tested against Rheome's own outputs (golden, MATLAB): timings to 1e-5 s.
- **Per burst also**: the trace's amplitude (count-weighted mean of the tile means) and peak (max of the tile maxima), duration and interval (`measure_events`, reused), each with `count, sum, median, mean, sd, percentile:25/75/90` across the bursts.
- **Tested** (`test_risefall`, `test_global_bursts`, synthetic): the fit equals Rheome's on four episodes (two models) and the planted parameters are recovered; two planted episodes (one tile's own pattern in each hemisphere, 12 Hz under a double sigmoid) found as two global
  bursts, the refined onset and offset within 0.01 s of the planted ones (the tile edges are 0.8 s away), one burst on each hemisphere's row with `per_hemisphere`, `min_r2`, a rerun a hit, the refusals.
- **The bursts' options.** The defaults are unchanged (seconds; every fit kept; the unfiltered bursts); the options are named, and the three
  new outputs are added to every run:
  - **`durations: cycles`** (steps `detect_spans`, `select.risefall`; the definition's `durations`): the minimum (`min_cycles` 3), the merge gap (`merge_cycles` 1) and the maximum (`max_cycles`, none) of the detection, and risefall's margin (`margin_cycles` 10), smoothing
    (`smooth_cycles` 1) and bound on the rise and fall times (`tau_max_cycles` 10, nxr's `burst_stages` calls), are in CYCLES of the CENTRE of the band the tiles resolve (a cycle is 1/centre s: 12 Hz for the band [8, 16]). They become seconds in the step, then are compared in whole tiles
    as ever; the resolved seconds are in the detection's record (`detection.durations`) and the fits' (`risefall.durations`); the seconds parameters (`merge_s`, `min_s`, `max_s`, `margin`, `smooth`, `tau_max`) are not used with `cycles`. Unlike nxr, the detection is on the joint cells' tiles
    (the time level's width), not smoothed per sample: a duration is resolved to a whole tile.
  - **`fit_gate`** (the new step; output `bursts_kept`): a burst is kept when its rise-fall fit is ok, its r² is at least `gate_min_r2` (0.7, labelalpha's, nxr's) and neither time is at its bound (within 1 % of `tau_max`): `nxr::select::passesGate`. A spans Selection of the KEPT bursts, whose provenance
    (`detection.gate`) holds the gate, how many bursts it kept and refused for which reason, and each kept burst's ordinal among the detected ones; **`bursts` (unfiltered) and every measurement over it stay as they were.**
  - **`stages`** (the new step `burst_stages`; outputs `burst_onset`, `burst_within`, `burst_offset`): from each KEPT burst's stored fit, the onset [t_on − 2 τ_rise, t_on + 2 τ_rise], the within (the plateau [t_on + 2 τ_rise, t_off − 2 τ_fall], none when they meet) and the offset [t_off − 2 τ_fall, t_off + 2 τ_fall] as three spans
    Selections of types `onset`, `within`, `offset` (`nxr::select::stagesOf`), clipped to the run; the span's edges are seconds, not tile edges (a roll-up over tiles takes the bursts, not the stages). Each stage's `detection.stages` says what it is and how many bursts have it.
    The `signal.analytic_frames` step's `within` input takes any of them.
  - **Tested** (`test_burst_stages`, golden `gen_burst_stages_golden.py`: nxr's `select.risefall` (quadrature, margin 10 cycles, smoothing 1 cycle, tau ≤ 10 cycles) and its gate on three synthetic series, a clean episode, one whose fall is at its bound and one in noise: the gate decides as nxr's on all three, the stages arithmetic is
    nxr's to 1e-12 and the fit's numbers agree to 5 ms where the fit is well posed; `test_global_bursts`: the planted bursts all kept, their stages equal t_on ± 2 τ_rise etc. of the stored fits, a gate between the two fits' r² keeps one and drops its stages, cycles give the same two bursts with the resolved seconds
    recorded, a minimum longer than any burst gives empty sets). The dev check of the spec (both detectors on the fixture and on a stationary lag-free surrogate) is not done here: it needs the surrogate of piece 10.
- **Not built**: the trace stored as a Field, the decimated trace, the `posterior` model.

## Vortex cores: the threshold's distribution and the patches (`run vortex_core_threshold`, `run vortex_cores`; `compute/include/nxr/histogram.h`, `cores.h`; spec `docs/specs/2026-11-14-vortex-cores-port.md`, piece 5)

**Why.** The project's vortex cores are the connected patches where the hatted curl of the current is strong, one sense each, in each frame of the bursts (nxr `vortex._core_threshold`, `_core_patches`). Their directions of travel are the
robust population result; their speed and lifetime are not (they equal the lag-free null). **State: the pure functions, the two runs and the storage are built and tested** (the maintainers' decisions of 2026-11-14: two definitions, the
histogram's range from the first 3000 burst frames, the cores stored as point events).
- **`nxr::stats::histogram`** (the distribution of a non-negative series, folded per tile; nxr `_core_threshold`): 4096 log bins, hi = 4 × the first frames' maximum, lo = hi · 1e-6; counts are additive over tiles, so any cut of the frames into blocks
  gives the same distribution; `percentile(p)` is the upper edge of the first bin that reaches p (exact to one bin, 0.34 %).
- **`nxr::select::corePatches`** (a pure function of a [vertices × frames] map, the mesh's edges, the vertex areas and positions, and a threshold; nxr `_core_patches`): per frame and per sense (+1 first), the connected vertices with
  sense · value ≥ threshold; no merging of patches that do not touch. Each core has its frame, sense, centre (the |value| · area-weighted centroid, in the positions' unit), area, and nxr-neuro's own **centre vertex** (the patch's vertex
  nearest the centre) and **peak** (the largest sense · value). Frames are independent: blocks of frames, each given its first frame, give the same list (the memory of a call is the block's frames × vertices).
- **The chain (both runs; one linear map before any non-linear step, D106).** `analytic` (band, Morlet) → `frames` (the frame tiling of `run select_time_grid --rule frames`, with `within` the named bursts: the good frames are the burst frames, and not the bad segments')
  → `meeg.apply_kernel` (form `curl_tan`, the level's Mexican hat, on the GLUED surface: one cortex, the bridge filled by the hat; linear, so it commutes with the frame sampling, and runs on the frames in blocks) → `core_histogram` or `cores`. The bursts (`bursts`, e.g. `global_bursts.bursts_kept.{recording}`) and the
  tiling (`tiling`, `{recording} frames <frame_s> s`) are stored nodes read by name; the caller names them. A complex field is read by its real part (Re(M z) = M Re(z) for a real linear map: the real band's curl).
- **Pass one, `run vortex_core_threshold` (step `core_histogram`, accumulate).** The distribution of |curl| over the good frames, pooled over the vertices, in 4096 log bins, a **statistic** `vortex_core_threshold.histogram.<recording>`
  (measure `count`, method `histogram`, `n_members` = the frames counted, `options` = lo, hi, bins, the frames, the chain it was made from, the bin axis's upper edges). **The range is fixed from the run's first 3000 good frames** (`range_frames`, nxr's chunk
  of 3000, counted from the run's first good frame, so it does not depend on the chunk size): hi = 4 × their largest |curl|. Those frames are held until the range is known (memory: 3000 × the vertices, float32). The percentile is **not stored**: it
  is derived at read.
- **Pass two, `run vortex_cores` (step `cores`, accumulate).** Reads the stored histogram BY NAME as an input (`threshold_statistic`, default `vortex_core_threshold.histogram.{recording}`), derives the `percentile`th percentile (98, nxr's) from its
  counts and edges in preparation, and cuts `corePatches` from each block of frames (`block_frames`: memory only). It refuses a threshold made from another band, spatial level and shape, form or frame tiling (a percentile of another chain's distribution is no
  threshold of this one). The threshold, the percentile, the statistic's name and id, and the mesh are in the run's provenance (`params_json.detection.record` of the cores, `prepared`).
- **What is stored: POINT EVENTS** (`pipelines/include/nxr/pipelines/point_events.h`, shared with the spatial-events work;, spatial-events decision 1; no new point type): a **spans Selection** `vortex_cores.cores.<recording>` (event type `core`),
  **one member per core whose span is the core's frame** [t_j, t_j + 1/fps) (several members share a frame; the model keys a span by its ordinal; order: frame, sense +1 first, lowest vertex), and the cores' attributes as **measurements over that
  Selection** (array-backed, the member axis = the cores' order, count 1: one frame): `frame` (the frame tiling's index), `sense` (+1 counter-clockwise about the normal, −1), `centre_vertex` (a vertex of the glued surface), `centre`
  (`centre_x`, `centre_y`, `centre_z`, m: the |curl|·area-weighted centroid), `area` (m²: Σ the patch's barycentric vertex areas) and `peak` (the largest sense · curl). `frame`, `sense`, `centre_x/y/z` and `peak` are nxr-neuro's own measures
  (1044-1049, reduction none: places and per-event values do not roll up).
- **The helper for the next step.** `point_event_arrays(PointEvents)` (frame tiling's rate and origin, one frame per event, attributes `{measure, comps, values [events × comps]}`) makes the runner's arrays: the spans (the frame tiling's tile width and origin in
  `tiles_of`) and one measurement array per attribute; `point_events_value(x, facts)` and `point_attribute_value(x, measure, unit)` make the plan's Values. The runner stores them as any spans output and any measurement over the run's events (a vector measure
  `centre` is stored as three rows). A step that finds events on frames (the spatial events: patches of relative band power) fills a `PointEvents` and declares those outputs; nothing in the runner is specific to cores.
- **Not resolved / limits.** A core is a patch of the curl map above a percentile, a description of where the field rotates; the sense read from the curl map is right in 88-100 % of 25-40 mm patches, from the current itself it is chance. The
  threshold is relative (a percentile of the run's own frames), so a recording of weak rotation has cores too: that is why the null (piece 10) is the evidence, not the cores. The curl is the band's, per frame (the frames are samples of the band's signal, not
  windows); the distribution's percentile is exact to one bin (0.34 %); the centre is in the surface's own unit (m). The `signal.analytic_frames` step's tail runs on the whole chunk's frames (memory: the glued vertices × the chunk's frames, complex float32; use `--chunks` to cut
  the run), the cores' fold itself works `block_frames` at a time.
- **Tested** (`test_histogram`; `test_cores`, golden `gen_cores_golden.py`: nxr's `_core_patches` on a synthetic icosphere cortex with planted blobs of both senses over 8 frames; `test_vortex_cores_pipeline`, synthetic two-hemisphere source space,
  4140 burst frames with a bad span inside: the stored histogram equals `nxr::stats::histogram` of the direct field (the same counts, hi to 1e-5), the threshold read by name is the direct 98th percentile, the stored cores (7642) equal `corePatches` called directly
  on the direct field (frame, sense, vertex identical; centre and area to 1e-7), 3120 distinct frames hold the 7642 members, the spans are their frames, chunks of 4096 and of 1000 give the same distribution and the same cores, the 90th percentile gives a
  lower threshold and more cores from the same stored counts, a rerun is a hit, a threshold of another spatial shape, a field not on the frames and a percentile out of range are refused).

### Tracks of the cores (`run vortex_cores`, step `tracks`, `compute/include/nxr/tracks.h`; spec piece 6)

**Why.** A track is a core followed from frame to frame: where it went. The project's population work found track speed and lifetime EQUAL THE NULL's, so tracks are **descriptions**, never evidence that anything travels; they are stored because the
crossings per tile (piece 7) are made of their steps, and so a reader can look.
- **`nxr::select::Linker`** (pure maths on arrays; nxr `_track_stats`' linking): cores come by frame; in the order given (sense +1 first, then the place in the frame) each core joins the nearest still-free track of the PREVIOUS frame of the SAME sense within `link`
  of that track's last centre (greedy: the first core in the order takes the nearest; a track takes one core a frame), else starts a track; ONLY CONSECUTIVE FRAMES link (a frame with no core, or a bad segment, closes every track). Tracks are numbered by the order
  they start. It is a streaming accumulator (`feed` in blocks of any size, a frame may be split between two; `finish`): the open tracks are carried, and any cut of the cores gives the same tracks (tested). `trackStats` gives each track's first and last frame,
  frames held, sense, path (Σ the steps' lengths) and displacement (|last centre − first centre|).
- **The step `tracks` (kind `after`).** It reads the run's cores as `cores` accumulated them (arrays after the chunk loop, never recording data), links them once (the accumulator fed the whole list: the cores do not depend on the chunk size, so neither do the tracks)
  and stores POINT EVENTS of kind `tracks`: a spans Selection **`vortex_cores.tracks.<recording>`** (type `track`), one member per track of at least `min_frames` frames (default 2: a lone core is no track; nxr's lifetime statistics use 3 or more: filter `n_frames` at read), its
  span the first core's frame start to the last core's frame end, in the order the tracks start; as measurements over it: `first_frame`, `last_frame`, `n_frames`, `path_length` (m), `displacement` (m) and `sense`. **The labelling core → track** is the measurement
  **`track`** over the CORES' Selection: the member ordinal of the core's track in the tracks' Selection (−1: a core in no stored track). `link_mm` (15, nxr's CORE_LINK) is in mm and converted to the surface's unit (m). A run that makes two sets of point events
  names each by its kind (`facts.events_kind`; a measurement is over the set of its kind: `point_events.h`).
- **Measures** (nxr-neuro's own, 1116-1120, reduction none: a place or a label does not roll up): `first_frame`, `last_frame`, `n_frames`, `displacement`, `track`.
- **What it cannot resolve.** Linking is by nearest centre between consecutive frames of the band's signal: a core that vanishes for one frame and returns is two tracks; two cores that cross may swap; a track is a chain of detections, not a physical object. The provenance
  (`producer_json.warnings` and the Selection's `params_json.detection.record.warnings`) says: tracks describe where a core went, not that anything travels; speed and lifetime equal the lag-free null's (the null is the evidence).
- **Tested** (`test_tracks`, golden `gen_tracks_golden.py`: nxr's `_track_stats` links on planted moving vortices of both senses on a sphere, two same-sense cores passing within the link distance, lone cores, empty frames and a gap: the same links (sense, from,
  displacement, track, frame) to 0 mm, the same (frames, path, displacement) of the tracks of 3 frames or more; any cut into blocks gives the same labels; hand-made cases: a gap breaks the track, the sense, the inclusive distance, greedy order, one core a frame per
  track; refusals. `test_vortex_cores_pipeline`: on the synthetic source space the stored tracks (2 060 of 2+ frames from 8 228 cores) equal `nxr::select::link` on the stored cores (frames, senses, spans identical; path and displacement to 1e-7), the labelling
  `track` is as linked directly, chunks of 1000 and 4096 give the same tracks, a longer `link_mm` joins more, `min_frames 1` stores the lone cores too).

### Crossings of the tiles by the tracks (`run vortex_cores`, step `select.tile_crossings`, `compute/include/nxr/crossings.h`; spec piece 7)

**Why.** Does the direction cores move in agree, tile by tile, more than chance? The project's population work (2026-10-29/30) found this, the path consistency of the cores' crossings and its repeat across two runs, one of the few robust results; speeds and lifetimes were null. A
tile's crossings are counted so that places with many tracks do not outvote places with few.
- **A crossing** is ONE TRACK'S STEPS inside ONE TILE of the surface ladder at `space_level` (one sense). A step is a core after the first of a stored track, from the previous core's centre; it sits at the surface vertex NEAREST its start (that vertex's tile and normal), its
  displacement is projected on the tangent plane there, and the unit vector is **carried to the tile's centre** along the surface by the vector heat method (the Levi-Civita form of `nxr/select/dyadic.h`: the number c1 + i c2 in the vertex's frame turned by e^(−i ρ), the arithmetic of
  `ladder::lcResultant`; nxr `Tiles.crossings`, `Pooling.carry`): vectors at different places of a fold are never added in 3-D. One track's carried steps in a tile are ADDED and made a unit vector (a track that crosses a tile in many steps is one crossing); the crossings of
  a tile and sense are then added and counted. **`nxr::select::track_crossings::crossings`** is that, a pure function of arrays (the steps, the surface's per-vertex frames, ρ and tile; `embedAtCentres` reads the sums back as 3-D tangent vectors at the centres; a k-d tree finds the nearest vertex).
- **Stored: ONE measurement `select.tile_crossings` over the level's tiles** (the level Selection `<surface> tiles L<l>`, of the recording), array-backed **[rows × members × 4]**, the vector-measurement convention of #139: components 0-2 the summed vector (`crossing_sum_x/y/z`), component 3 the
  crossings' count (`crossings`). **Rows** are stage × sense: row 2i the stage i's sense +1, row 2i+1 its sense −1 (`stages`, default `[burst]`: every step; `onset`, `within`, `offset` count the steps whose frame starts inside that stage's spans, the inputs of those names, by default
  `global_bursts.burst_onset|within|offset.{recording}`). **Members** are the level's tiles, then a **`bridge` member** (the last, index = the tiles): crossings whose nearest vertex is on the callosal bridge (the glued surface's, not cortex: no tile and no centre to carry to); its unit
  vectors are added in 3-D. No `count` (samples) row is written: the count is the crossings.
- **Derived at read** (never stored; `measure` registry, nxr-neuro's own 1121-1126): **`crossing_consistency`** R = |S| / n and the unbiased **`crossing_rho2`** = (|S|² − n) / (n (n − 1)) (undefined for n < 2), read member by member with `derived_measurement_values`. R is biased upward for
  small n (one crossing has R = 1): **compare places of different counts by ρ², never R, and report the counts** (the project's trap, 2026-10-30; the data-to-null crossing ratio comes with the null, piece 10). The stored sums and counts do not roll up the ladder (vectors of different tiles are never
  added; reduction none): they are at the level they were made.
- **Frames and transport.** The canonical frames when `run surface_frames` stored them, else the signpost basis (said); each tile's centre is the non-singular member nearest its area-weighted centroid, and ρ is `ladder::leviCivita(...).rho` on the NATIVE surface (the cores' glued vertices map to it through the stored
  package's support, the bridge has none).
- **What it cannot resolve.** A crossing is the direction a core moved through a tile between frames of the band's signal. The transport is path-dependent on a folded cortex (the Studio's intrinsic-Delaunay heat method against nxr's geometry-central one: 0.08° on a sphere, more on a fold); a tile's R by itself
  is no evidence of travelling: the lag-free null is (piece 10). Tracks shorter than `min_frames` make no crossing. Warnings are in the provenance.
- **Tested** (`test_crossings`, golden `gen_crossings_golden.py`: nxr's `Tiles.crossings` on two icosphere hemispheres (nxr's ladder level 3) with planted moving vortices of both senses, a stage's subset of the frames: the counts per tile and sense are nxr's EXACTLY, the carried sums to 0.003 of 3.8
  (directions within 0.08°); hand-made cases: one track's steps in a tile are one crossing, senses separate, a step with no tangent part has no direction, the bridge in 3-D, the mask, the carry by e^(−iρ), the nearest vertex; refusals. `test_vortex_cores_pipeline`: the stored `select.tile_crossings` (3 901 steps of
  the stored tracks, 1 358 crossings in 16 tiles) equal `nxr::select::track_crossings::crossings` on the stored cores with the frames and transport made in the test (counts identical, sums to 2e-4 of 172), R and ρ² are the formulas, a stage's rows equal the direct run with the mask, a rerun is a hit, a stage named without
  its Selection is refused). The synthetic source space has no crossing on its bridge: the bridge member is tested by hand only.

### Run-to-run repeat of the crossings (`run repeat`, `pipelines/include/nxr/pipelines/run_repeat.h`, `nxr::select::track_crossings::repeat`; spec piece 8)

**Why.** If cores really move in a preferred direction in a tile, it should be the same direction when the subject is recorded again. The project's population work used the repeat across two rest runs as its test (nxr `repeat`, `tile_dot`).
- **`nxr-neuro run repeat --dataset D/N --subject S --recording-a A --recording-b B [--over TILES] [--stage NAME|INDEX] [--min-crossings 10] [--name NAME] [--replace]`**: a statistic over the two recordings' stored `select.tile_crossings` (`run vortex_cores` first on both, on
  the same tile level; read through io, never a recording). A tile's two senses are POOLED (a core turns sense each half cycle: the direction is the tile's, not the sense's; the two 3-D sums are vectors at the same centre). Only tiles with at least `min_crossings` (10, nxr's
  NMIN_TILE) crossings in BOTH runs count:
  - **per tile:** the dot product of the two runs' mean resultant vectors (S/n), large only when both runs are consistent and agree, near 0 when either is inconsistent, negative when they go opposite ways (NaN with fewer crossings);
  - **whole recording:** the count-weighted mean cosine of the tiles' directions, weight min(n_a, n_b) (a tile whose sum vanishes in a run has no direction and is left out).
- **Stored** as two statistics of the first run's Field (the other named in `options`): `vortex_cores.run_repeat.<a>.<b>.tiles` (measure `repeat_dot`, method `select.run_repeat`, [tiles], the axis the level's Selection) and `….whole` (measure `repeat_cosine`, [1]); `--stage NAME` (burst, onset, within, offset) takes that stage's pair of rows (2i, 2i+1) in EACH run: `run vortex_cores` writes the stage names on the crossing rows (` · stages burst,within` at the end of their `computed_by`; no schema change), so two runs may list their stages in another order, and a name a run lacks is refused with the names it has. `--stage INDEX` takes rows 2S and 2S+1 of both (crossings stored before the names exist have only this); the default is stage 0 (`burst`). A stage other than the first names the statistic `….<stage>` (`….stageS` by index). A rerun is a hit; `--replace` moves the old aside. Maths: `nxr::select::track_crossings::poolSenses`, `repeat` (pure functions of arrays).
- **The p-value is `run surrogate_lagfree_vortex_compare`'s** (`nxr-neuro run surrogate_lagfree_vortex_compare --recording-a A --recording-b B`: a tile's dot against the lag-free surrogate's draws, run A's draw i against run B's, Benjamini-Hochberg over the tiles, stored as `….p` and `….whole_p`; see *The lag-free surrogates in the graph and their comparisons*); `run repeat` stores the statistic only (said in its provenance warnings).
- **What it cannot resolve.** Two runs of one subject share the cortex, the kernel and the instrument's leakage: agreement is evidence only against the null. A tile needs 10 crossings in each run: with few cores most tiles are left out (`tiles_used` of `tiles`
  are reported).
- **Tested** (`test_crossings`, golden `gen_repeat_golden.py`: nxr's `repeat` and `tile_dot` on planted tile sums of two runs, most tiles agreeing, three opposed, a few with too few crossings: per-tile dots to 1e-16, the cosine to 1e-12; the same motion twice gives 1, reversed -1,
  too few crossings NaN; refusals. `test_vortex_cores_pipeline`: `rest` and `rest_b` (the same samples in other chunks: the same motion twice) give cosine 1.000 over 3 of 16 tiles, equal to the direct computation on the stored arrays; with one run's vectors negated -1.000 and every dot
  <= 0; a rerun is a hit; the same recording twice, a missing run and a stage out of range are refused).

### The lag-free surrogates: the draw, the Wishart draw, the sequential p (`compute/include/nxr/synthesis/lagfree.h`, `nxr::synthesis`; `compute/include/nxr/nulltest.h`, `nxr::stats`; spec `docs/specs/2026-11-17-surrogates.md`)

**Why.** A core rate, a direction consistency in a tile or a repeat between two runs is evidence of travelling only against what mixing and amplitude modulation alone produce. The lag-free surrogate (Rheome `lagfreenull`, `envelopenull`, `pairnull`, `pairdof`) keeps the data's
spectrum (never flat), its zero-lag covariance and its amplitude envelope, and removes every lag.
- **The surrogate is a function of the sample index** (`shapedNoiseBlock`, mixed: `lagfreeBlock`): one standard normal per stream and sample from a counter-based generator (`standardNormalAt`: splitmix64 of seed, stream, sample; Box-Muller), the zero-phase taps made from the data's spectrum (`spectrumShapingTaps` of `channelMeanSpectrum`), the mixing
  √(Re C) at unit mean channel variance (`zeroLagMixing`, `symmetricSquareRoot`) and a per-sample scale (`envelopeScale`, `analyticGainPower`, `interpolateTileSeries`) (the data's one-cycle RMS envelope over the surrogate's expected one). A chunk of it does not depend on the chunk size (tested to the last bit with four unequal cuts).
- **Parity with nxr** (`test_surrogate`, golden `compute/test/golden/surrogate_golden.txt`, generator `gen_surrogate_golden.py`; a synthetic 3-channel recording with a 10 Hz source reaching the channels with lags of 0, 2 and 4 samples; the mean of 40 draws of each): the spectrum agreement with the data's
  (Welch, 2-30 Hz) 0.9974 against nxr's 0.9966; the cross-spectrum of two channels in 6-14 Hz, Im / √(P0 P1) -0.002 against nxr's 0.010 (the data's: -0.93) and Re 0.235 against 0.251; the covariance within 0.8 % of the variance; the level of the envelope 0.977 against 0.998. A
  20 ms delay of one channel gives Im -0.56, so the test sees a lag.
- **One stated difference: the envelope.** nxr rescales each draw to the data's RECALISED envelope (the draw's RMS over channels and one cycle: a reduction inside the function), so its draw's envelope IS the data's (correlation 0.98 here). The cardinal rule forbids that reduction in a node: the scale here is the
  data's envelope over the EXPECTED envelope of the unscaled draw (pointwise), so a draw follows the data's envelope up to the noise's own sampling variation (correlation 0.85 on 3 channels and 10 samples a cycle, the most it can differ; the level matches). The maintainers may ask for nxr's form, which needs the one-cycle channel mean as a measurement.
- **`lagfreeSecondMomentDraw`** (Rheome `pairnull`): Q ~ complex Wishart(Re Q0 / k, k), the degrees of freedom k from a burst's duration × the band's effective bandwidth (`effectiveBandwidth`, nxr `dof_fraction` × the sampling rate: a flat spectrum of width B gives B), seeded by (seed, draw): the mean is Re Q0 with no imaginary part (tested), the spread falls as 1/k.
- **The comparison (`nxr::stats`)**: `sequentialP` / `sequentialPs` (Besag-Clifford: h / n when stopped, else (e + 1) / (n + 1)), `fixedDrawPs` ((e + 1) / (n + 1), Rheome `tilepq`), `benjaminiHochbergQ` (NaN members left out of the tests; Rheome `bh`), the draw tally (`tallyDraw`, `rollUp`, `undecidedMembers`, `nullMean`), `countRatio`, `unbiasedConsistency`, `meanResultants`, `occupancyFraction`, `patchShares`, `compareMeanOverMembers`: pure functions of the data's and the draws' values over the same members.
- **What it cannot resolve.** The null has the data's covariance at zero lag, its spectrum and its envelope, and nothing else: a result beyond it is beyond what mixing and amplitude modulation produce, never a proof of a wave. Its spatial structure is the band's (the cross-spectrum's real part), the same at every frequency.

### The lag-free surrogates in the graph and their comparisons (`run measure --set surrogate_lagfree_envelope`, `run surrogate_lagfree_vortex`, `run surrogate_lagfree_vortex_compare`; steps `power_series`, `synthesis.lagfree_mixing`, `synthesis.lagfree`; spec `docs/specs/2026-11-17-surrogates.md`)

**Why.** Cores, tracks and crossings are descriptions; a tile's direction consistency and the run-to-run repeat are evidence only beyond the lag-free null. The null is the SAME vortex graph run on a surrogate of the recording.
- **Ingredients (all stored, read by name):** the recording's Welch PSD (`run psd`: the spectrum's shape, never flat), the band's cross-spectrum (`run crossspectrum --channels-of KERNEL --bands alpha:8-12`: √(Re C) mixes the draw and its channels are the null's), and the data's band power
  in dyadic time tiles (`run measure --set surrogate_lagfree_envelope`: step `power_series`, a reduction over a time partition folded chunk by chunk; the statistic `lagfree_envelope.series.<recording>`, [tiles x (sum, count)], measure `power_sum`, method `power_tiles`; the mean is derived at read; bad samples leave the tiles).
  The stored band cross-spectrum is one matrix a band (the bins averaged): it carries the covariance but not the spectrum's shape within the band, which is why the shape comes from the PSD.
- **`synthesis.lagfree`** (a chunk step; the recording is declared `read: shape`: the chunk service hands its rows, rate, span and good samples and reads NO sample): `nxr::synthesis::shapedNoiseBlock`, one UNMIXED stream for each of the cross-spectrum's channels (the others are 0, the kernel uses none), scaled by √(the data's tile power / the draw's expected power through the band's analytic gain), so the analytic signal of the draw
  has the data's envelope. A function of the sample index: chunk-size independent (tested in compute to the last bit). `seed` is a parameter; a draw is the graph with another seed, in the cache key and, by the definition header `draw: seed`, in the stored node's name.
- **The mixing is composed, not applied.** `synthesis.lagfree_mixing` (before the loop) makes √(Re C) from the cross-spectrum; the unmixed draw AWAITS it through the linear time steps (analytic, frames: the same filter on every channel, so they commute with it, as the per-sample scale does), and `meeg.apply_kernel` composes it ONCE into its operator (`pre_operator`: kernel · tangent part · curl · hat · mixing, one [vertices × channels] matrix, cached across the draws by its inputs' provenance). A non-linear step or a reduction of a value still awaiting its operator is refused at plan.
- **`run surrogate_lagfree_vortex`** is `vortex_cores` after `synthesis.lagfree`: analytic, frames on the DATA's burst frames, the curl of the hat (the fused operator), cores at the DATA's stored threshold, tracks, `select.tile_crossings` with `as: statistic`. Only the draw's tile crossings are stored, as a statistic beside the recording
  (`surrogate_lagfree_vortex.draw<seed>.null_crossings.<recording>`, [stage x sense, tiles + bridge, (x y z, count)], measure `crossing_sum`, method `lagfree_null`, over the level's Selection: small); the surrogate, cores and tracks of a draw are never stored.
- **`nxr-neuro run surrogate_lagfree_vortex_compare --recording-a A [--recording-b B] [--set NAME=VALUE ...]`** runs the draws in batches (default 20; at least 20, at most 100; `--batch`, `--min-draws`, `--max-draws`), reading each from the store, refuses a draw over another Selection than the data's crossings or made by another definition, and compares the data's crossings with them (stage by `--stage NAME|INDEX`, the senses pooled, tiles with at least `--min-crossings` 10):
  - the **crossing consistency** per tile: the unbiased ρ² = (|S|² - n) / (n (n - 1)) (never R), one-sided; with B, the **run repeat** per tile (the dot of the two runs' mean resultants, draw i of A against draw i of B: the two runs' seeds are 100000 apart so their noise is independent) and the count-weighted mean cosine;
  - the **sequential p** of Besag and Clifford (h = 10 exceedances: h / n, else (e + 1) / (n + 1)); the draws stop when every tile with a statistic has h exceedances (after the minimum), else at the maximum; **Benjamini-Hochberg** q over the tiles. A rerun is a hit; draws that exist are kept, so a larger `--max-draws` continues.
  - **Stored** (statistics of the recording over the level's Selection): `surrogate_lagfree_vortex_compare.<rec>.consistency_p` [tiles x (p, q, exceedances)] (measure `null_p`, method `besag_clifford`), `.crossing_ratio` [tiles] (the data's crossings over the null's mean: the **data-to-null crossing ratio**, measure `crossing_ratio`), `.resultant` [tiles x (x, y, z, mean crossings)]
    (the null's mean resultant per tile, what a group map subtracts: measure `null_resultant`); with B, `vortex_cores.run_repeat.<a>.<b>.p` [tiles x (p, q, exceedances)] and `.whole_p` [(p, exceedances, draws)].
  - Tested (`test_vortex_cores_pipeline`, synthetic): the envelope statistic's mean power equals the direct (0.9909 against 0.9906); a draw is a function of its seed (the same seed replaced is identical, another differs, a rerun is a hit, the data's own crossings are untouched); the stored p and exceedances equal the count made directly from the stored draws; the stage-name selection; stopping at the minimum when no tile can be tested.
- **What it cannot resolve.** The null has the data's zero-lag covariance, spectrum and envelope and nothing else; the threshold and the burst frames are the data's. A null tile with fewer crossings than the minimum has no statistic and is not an exceedance: read the crossing ratio beside the p. The envelope is scaled pointwise (see above), so the draws' envelope follows
  the data's up to their sampling variation.
- **Not yet built:** the subtraction of each subject's surrogate in a group map (the resultants are stored for it).

## Spatial events: where a pattern happens, as an event with an extent (`run spatial_baseline`, `run spatial_threshold`, `run spatial_events`; spec `docs/specs/2026-11-14-spatial-events.md`)

Bursts say WHEN (global, on the envelope: never per vertex). A spatial event says WHERE: inside a burst, the cortex's band power above each vertex's own baseline, cut into connected patches on the glued cortex, each with a place,
an area and a characteristic size. This section grows with the spec's pieces (the checklist at its end of the spec says which are done).

- **The baseline (piece 1; `run spatial_baseline`, step `baseline`, accumulate).** The band power each glued vertex has at each hat level when no burst is on. The chain is `analytic` → `frames` (the frame tiling, no `within`) →
  `meeg.apply_kernel` (NO form, `levels`, `glued: true`: the free current as the inverse gives it, each component through each level's hat, the bridge filled; no orientation assumed) → `power` (|z|² summed over the components) →
  `baseline`, which means x over the GOOD frames: the run LEAVING OUT `exclude` (the bad segments) and `outside` (the named bursts, or a stage of them: the run outside them). A time Selection "the run minus the bursts minus
  the bad segments" is not a new Selection: a step leaves out the UNION of its `exclude` and `outside` Selections (the runner builds one exclusion; two accumulate steps with different ones are refused), the same
  exclusion the chunk service applies to the samples. It is a mean over a partition's members folded by additive sums (a measurement), never a reduction inside a pointwise step. **Stored:** a STATISTIC
  `spatial_baseline.baseline.<recording>` (measure power, method mean, `[vertex × level]`, the vertex axis on the glued surface, the levels named `level<l>` and listed in `options`; the chain it is of in `options.chain`; `n_members` the
  good frames), beside the recording. A cut of the run (`--chunks`) cuts it. **Memory** per block of frames: see `meeg.apply_kernel` `levels` (the operator) and `--block-frames`.
  `nxr-neuro run spatial_baseline --dataset D --subject S --band '[8.0,12.0]' --levels '[2,3,4,5,6,7,8]' --bursts '<the' kept bursts> --all`. **What it cannot resolve:** the baseline is the power with no named burst
  on, not the brain's noise-free state: bursts that were missed, or activity below the detection threshold, are in it.
- **`relative` (piece 2; step, chunk, pointwise).** x (a band's power per vertex and level) divided row by row by a stored baseline read BY NAME (`baseline`: a statistic of method mean, measure power): a diagonal scaling,
  `nxr::scaling::divideRows` (compute/, general algebra: a matrix and a vector of divisors in, a matrix out; refuses a divisor that is not finite and above 0). It is bound once in preparation, refuses a baseline of another chain (band,
  spatial band and levels, form, glue, frame tiling) and one that is not above 0 at a row. The result is unitless: 1 is the vertex's own baseline.
- **`scale` (piece 4; step, chunk, pointwise; `compute/include/nxr/scalespace.h`).** The CHARACTERISTIC LEVEL of each vertex and frame: x is the band power of a multi-level field (`meeg.apply_kernel` with `levels`, `glued`, then `power`); at each
  vertex the level whose power over the squared L2 norm of its atom is the largest (Lindeberg 1998; decision 4). The atom of a vertex at a level is the operator's column there (the level's hat on the native sources, read at that glued vertex);
  its norm is the Euclidean norm of that column over the native vertices, computed exactly in preparation (the band applied to the identity, 256 columns a block: memory glued vertices × 256 doubles; time one band application to as many columns as
  there are native vertices, per level, repeated by every run: on a 20 000-vertex cortex a few minutes with the stored eigenmodes). So a blob of a level's own size gives its largest normalised power at that level, and white noise the same at every
  level. The maximum is over the LEVEL axis (the scale space); nothing is reduced over time or space. The output is the ladder's level as a float, one number a vertex and frame; the size in cm is the level's centre wavelength 2π/√λ_c
  (read at the events). `nxr::select::scale_selection::selectLevel` (power, squared norms, levels → index and response) is a pure function of matrices. **What it cannot resolve:** one level per vertex and frame (a patch of two sizes shows its dominant one); the
  levels are octave-spaced, so the size is known to a level, not continuously; the atom is the operator's, so the instrument's blur is inside the size.
- **`patches` and `run spatial_events` (pieces 5 and 6; step `patches`, accumulate).** The chain (one linear map before any non-linear step, D106): `analytic` (Morlet) → `frames` (with `within` the named bursts: the good frames are the burst
  frames, and not the bad segments') → `meeg.apply_kernel` (no form, `levels`, `glued: true`) → `power` → two branches: `relative` (÷ the stored baseline, read by name) → `select.at_level` (`reference_level`) → `patches`; and `scale` → `patches`.
  `patches` cuts the relative power at the THRESHOLD with `nxr::select::corePatches`, the vortex cores' finder (it never knows its input). **The threshold is set by its false-alarm rate, measured, as the time burst
  detector's threshold comes from the envelope's own distribution:** a vertex's single-frame relative power is about exponential, so a fixed ratio of 2 is exceeded by about 13.5 % of the vertices of every frame on noise alone. The default is
  `threshold: percentile`, `percentile` 99: the 99th percentile of the relative power OUTSIDE the bursts (and bad segments), pooled over the glued vertices, derived in preparation from the stored distribution `threshold_statistic`
  (default `spatial_threshold.histogram.{recording}`, made by `run spatial_threshold`, read by name). It is refused when that distribution is missing, or made from another band, band shape, levels, reference level, spatial shape, tiling or baseline.
  The derived threshold value, the percentile and the statistic used are in the provenance (`params.detection.record`). `threshold: ratio` (with `ratio`, 2.0) is the named option: relative power at least `ratio` × the vertex's baseline.
  **`run spatial_threshold` (step `relative_histogram`, accumulate).** The chain of `spatial_events` up to `relative` and `select.at_level` (the reference level), on the frames OUTSIDE the named bursts and bad segments (`frames` without `within`; `outside`
  is the existing exclusion mechanism of `baseline`), accumulating an `nxr::stats::histogram` (4096 log bins, the range fixed from the run's first 3000 good frames counted from its first, so independent of the chunk size, as `vortex_core_threshold`)
  of the relative power, stored as the statistic `spatial_threshold.histogram.<recording>` (measure count, method histogram; the chain, with the bursts left out, in its options). **Stored as POINT EVENTS** (`point_events.h`; decision 1):
  a spans Selection `spatial_events.events.<recording>` (type `patch`), ONE MEMBER PER PATCH whose span is its frame (several may share one), and per patch as measurements over it: `frame`, `burst_ordinal` (the ordinal of the bursts
  Selection's span holding the frame), `centre_vertex` (a glued vertex), `centre` (`centre_x/y/z`, the power-weighted centroid, m), `area` (m²), `diameter_cm` (the equal-area diameter 2√(area/π)), `characteristic_level` (the `scale` map's level at
  the centre vertex and frame), `size_cm` (that level's centre wavelength 2π/√λ_c, cm), `relative_peak` (the largest relative power in the patch; unitless, 1 is the baseline). `min_area` (m², 0: none) drops smaller patches. **Home map (piece 6):**
  a second output, the statistic `spatial_events.home.<recording>` (measure count, method occupancy, `[vertex × (frames in a patch, burst frames)]`): counts, additive over frames; the fraction frames_in_patch / frames is DERIVED at read.
  A vertex is in a patch when its relative power is at least the threshold (`min_area` drops events, not vertices). Frames are independent: the result does not depend on the storage chunk or on `--block-frames` (tested).
  Run order on a recording: `run select_time_grid --rule frames` (its band), `run global_bursts` (the bursts), then
  `nxr-neuro run spatial_baseline --dataset D --subject S --band '[8.0,12.0]' --levels '[2,3,4,5,6,7,8]' --all`,
  `nxr-neuro run spatial_threshold --dataset D --subject S --band '[8.0,12.0]' --levels '[2,3,4,5,6,7,8]' --reference-level 6 --all` and
  `nxr-neuro run spatial_events --dataset D --subject S --band '[8.0,12.0]' --levels '[2,3,4,5,6,7,8]' --reference-level 6 --all` (`--bursts …` names other bursts; `--percentile '…`,' `--threshold ratio --ratio '…`,' `--transform …`).
  **What it cannot resolve:** a patch is where the band is strong inside a burst, not that a local generator is there (volume conduction and the inverse spread power over the cortex; the point-source reference is a separate measurement);
  the size is octave-spaced and one level per patch; the finest level bounds the resolution; the atom norms of `scale` are recomputed every run (open: storing them as a node read by name); the false-alarm threshold is a statement about the noise outside the bursts, so it assumes the outside frames are a fair sample of the noise inside them (a burst's own broadband tail is not).
- **Tested** (`test_scaling`; `test_spatial_events_pipeline`, a synthetic two-hemisphere cortex, never a real dataset): the stored baseline equals the direct computation (the band's wavelet read at the frames, each level's dense exact hat) to 10⁻⁹ of the largest
  value, with two bad spans and a window of bursts; the same from storage chunks of 4096 and 1000, and from blocks of 100 and 256 frames; a rerun is a hit; bursts covering the run leave no baseline (refused); the burst frames' mean of
  power / baseline equals the direct mean over the baseline at each level (10⁻⁷); a baseline of another chain, and a field that is no power, are refused.
- **Tested, events** (`test_spatial_events_planted`, a synthetic 642-vertex sphere, an identity kernel, white noise, two blobs on in turn inside the window, 2 cm and 5 cm wide, levels 1-4): 2 513 of 2 520 frames have their strongest event at the planted blob; the wide blob's events are the larger in `size_cm` (37.3 against 18.6 cm) and diameter (14.8 against 10.6 cm); each blob's vertex lies in a patch in 0.50 of the burst frames (the home map); the same events from storage chunks of 4096 and 1000 and from blocks of 100 and 256 frames; a rerun is a hit; the default threshold (percentile 99 of the relative power outside the window, `run spatial_threshold`) still finds the blobs (2 499 of 2 520 frames); refusals (a baseline of another chain, ratio 0, `percentile` with no stored distribution, a level the field lacks). The sphere is small (its finest hat is 18.6 cm), so a patch is wider than the blob: the test fixes the chain's agreement with the plant, not the instrument's resolution.
- **Tested, threshold** (`test_spatial_threshold`, a synthetic 642-vertex sphere, an identity kernel, white noise only, a window of bursts): the stored histogram equals `nxr::stats::histogram` of the direct outside-frame relative power (the band's wavelet at the frames, dense exact hat, ÷ baseline; L1 difference 50 of 2 168 676 counts, float rounding at bin edges), and its 99th percentile is the direct one (4.36 × the baseline); on pure noise the share of vertex-frames above the percentile-99 threshold inside the window is 1.61 % (the same share counted directly: 1.61 %; tolerance 0.5-2.2 %, the noise's spatial correlation leaves about ten independent regions on this small sphere), against 14.4 % for `ratio` 2 (tolerance 9-18 %; exp(-2) = 13.5 %); the histogram is identical from storage chunks of 4096 and 1000 and from blocks of 100 and 256 frames; a rerun is a hit; refusals (a distribution of another reference level, percentile 100, no stored distribution).
- **Tested, scale** (`test_scalespace`; `test_scale_step`, a synthetic 642-vertex sphere of 10 cm radius, an identity kernel, Gaussian blobs 1.2, 2, 3.5 and 6 cm wide, hats at levels 1-4): the stored level of EVERY vertex equals the direct one (the dense exact hats, squared response over the squared row norm: 0 of 642 differ); the level at the blob's centre tracks the planted width (4 4 3 2: wider, coarser; three distinct levels); refusals (not power, a single level, not glued).

### Spatial events, wrap-up: coarser levels by default, and the noise reference (`run spatial_events_outside`, the `summary` statistic)

- **The scale space reaches the coarse sizes.** The default `levels` of `spatial_baseline`, `spatial_threshold` and `spatial_events` is `[2, 3, 4, 5, 6, 7, 8]` (it was `[4, 5, 6, 7, 8]`): a patch is no longer cut off at level 4's size. A run asks for the levels the baseline and the threshold were made with (a mismatch is refused, as before); the operator holds one hat a
  level, so MEMORY grows with the number of levels (7 against 5: a 40 % larger bound operator; `--block-frames` still bounds the frames held).
- **The `summary` statistic** (every `patches` run; `spatial_events.summary.<recording>`, measure `patch_summary`, method `patch_statistics`): what the patches are, in a form that is the same inside the bursts and outside them: counts by AREA (32 log bins, 1e-7 to 1 m², each entry names its upper edge), by CHARACTERISTIC LEVEL (each entry its size in cm), by
  PATCHES PER FRAME (0 to 15 and 16 or more: frames holding that many), and the totals `frames` and `patches`. Counts are additive over frames; a rate is derived at read.
- **`run spatial_events_outside`: the noise reference.** The same chain, the same stored baseline and threshold, on the frames OUTSIDE the named bursts (and the bad segments): `frames` without `within`, `patches` with `outside: bursts` (the run leaves the bursts out) and `events: false` (the threshold is a percentile of the relative power outside the bursts, so about
  1 % of the vertices of every frame are above it: those patches are COUNTED, never stored as events). Stored as `spatial_events_outside.summary.<recording>`. A reader compares the two summaries per frame: the same threshold finds `0.39` patches a frame outside the planted window and `1.09` inside it in the test (synthetic: two blobs on a sphere), which is
  what the events inside are to be read against. `patches` takes exactly one of `bursts` (the frames are inside) and `outside`.
- **What it cannot resolve.** The reference is the band outside the bursts under the same instrument (the inverse's spread, volume conduction): it says what the threshold finds where no burst is on, not that patches inside the bursts are anything but more numerous or larger than that. It is not the lag-free null (below).

### The spatial events against the lag-free surrogate, and the instrument's blur (`run surrogate_lagfree_spatial_events`, `run surrogate_lagfree_spatial_events_compare`, `run pointspread`)

- **`run surrogate_lagfree_spatial_events`** is `run spatial_events` on a lag-free surrogate of the recording (`synthesis.lagfree`, its mixing composed into each hat level's operator: the recording's own spectrum, the band's real cross-spectrum as its zero-lag covariance, the data's band envelope; see the lag-free null above), the baseline, the threshold and the burst frames the DATA's. A draw is the run with `seed: i`
  (`draw: seed`: its own node, its own cache key); only its home map and its patch summary are stored (`surrogate_lagfree_spatial_events.draw<seed>.home.<recording>`, `.summary.<recording>`), never the surrogate or the patches (`events: false`).
- **`nxr-neuro run surrogate_lagfree_spatial_events_compare --recording R [--set NAME=VALUE ...]`** runs the draws in batches (20, at least 20, at most 100, h = 10: Besag-Clifford), refuses a draw on another surface or with other summary entries than the data's, and compares the data's `spatial_events.home.<R>` and `.summary.<R>`: per glued vertex the HOME OCCUPANCY (the fraction of burst frames in a patch) one-sided against the draws, BH q over the vertices; and the SIZE
  DISTRIBUTION: the share of the patches at each characteristic level and in each area bin, and the patches per frame, two-sided (the draws at least as large, and at most as large). Stored (statistics of the recording): `surrogate_lagfree_spatial_events_compare.<R>.home_p` [vertex x (p, q, exceedances)], `.home_null` [vertex] (the null's mean occupancy, what the home map is read against),
  `.size_null` [entry x (data, null mean, p_upper, p_lower, q_upper, q_lower)].
- **What the null says.** It keeps the data's zero-lag covariance, so a STATIONARY spatial structure (a place that is always strong in the band) is reproduced by it: in the test (two planted blobs on a sphere) the null occupies the narrow blob's vertex in 0.97 of the burst frames against the data's 0.99. What the data has beyond it is what is left: an excess over the null, or a pattern that moves.
  It is not the instrument's blur, which is the next measurement.
- **`nxr-neuro run pointspread --recording R [--forward NAME] [--kernel NAME] [--vertices 20 | --list V,V,..]`** is the instrument's blur, a SEPARATE measurement (never built into a tool): a unit current at each sampled native vertex, along x, y and z separately (no orientation assumed), through the stored head model's gain (the operator of kind `forward`:
  `run headmodel`, `headmodel_surf_os_meg` and the like) and the stored inverse kernel (the resolution matrix's column of that source: K G e), read by the SAME chain the events are: each level's hat on the glued cortex, the power summed over the components, the patch where the reference level's power is at least `--fraction` (0.5) of its peak
  (a point source has no noise floor, so the half maximum, not the events' false-alarm threshold; `nxr::select::corePatches`), and the characteristic level at the patch's centre (`nxr::select::scale_selection`, the atoms' norms as `scale`). Stored as ONE statistic of the recording, `pointspread.<kernel>` (measure `point_spread`, method `point_source`) [source vertex x orientation x
  (peak_error_cm, diameter_cm, area_cm2, characteristic_level, size_cm, peak_power)]. A reader compares the events' sizes with it: a patch no larger than a point source's is the instrument's blur, not the source's extent.
  Tested (`test_pointspread`: a 10 cm sphere, a Gaussian smoothing gain of KNOWN width, a kernel that hands each sensor back to its vertex): the peak is at the source (0 cm over 54 sources); the patch diameter grows with the blur (4.4 cm for the identity, 4.9 for a 2 cm Gaussian, 8.1 for a 6 cm one); the three orientations agree where the model has no preference; a rerun is a hit. The sphere is small
  (its finest hat is 18.6 cm), so the characteristic LEVEL is the coarsest at every source: the test fixes the order and the localization, not the absolute width. **What it cannot resolve:** the blur of THIS head model and kernel (their noise model and regularisation) for a source ALONE, not of many together; sizes are octave-spaced levels.

### The lag-free pairs surrogate of the second-moment measures (`run surrogate_lagfree_pairs`, node `synthesis.lagfree_pairs`, `nxr::synthesis::lagfreeSecondMomentDraw`)

- **Why a second null.** The surrogate (above) is a null for cores, tracks and patches (thresholds, links: not quadratic in the data). A quantity LINEAR in a burst's second moment Q_b = Σ z zᴴ (the quadrature, the lagged share, a phase current) needs no surrogate recording: its null is a distribution of Q. Under no lag Q_null ~ complex Wishart(Re Q_b / ν, ν), the burst's own
  zero-lag covariance with ν degrees of freedom (nxr `wishart`): 1000 draws a burst cost almost nothing. It complements the sign-flip test (is a direction SHARED across bursts?): the Wishart null asks whether LAGS EXIST AT ALL in each burst.
- **`nxr-neuro run surrogate_lagfree_pairs --recording R --windows <stem>_<band>_windows --bursts NAME --psd NAME --band LO,HI [--draws 1000] [--seed 0]`** is a statistic node on STORED inputs only: the analytic cross-spectrum windows (`run crossspectrum --estimator analytic`), summed over the tiles wholly inside each burst (products add: exact; a burst with a missing tile has no test), the Welch PSD
  (the band's EFFECTIVE BANDWIDTH: `nxr::synthesis::effectiveBandwidth` of the channels' mean PSD through the band's analytic gain, so ν = the burst's duration × that bandwidth), seeded draws (a draw is a function of seed, burst and draw index). What it measures is the **sensor-level lagged share** of each burst,
  ‖Im Q_b‖²_F / ‖Q_b‖²_F: per burst the draws at least as large (p = (e + 1) / (draws + 1)), Benjamini-Hochberg q over the bursts, and for the run the mean share against the null's mean over the same bursts. Stored: `surrogate_lagfree_pairs.<bursts>` [bursts × (lagged_share, null_mean, p, q, dof)] (measure `null_p`, method `lagfree_pairs`) and `….run` [(mean share, null mean, null sd, p)].
- **Tested** (`test_wishart_null`, synthetic: a 10 Hz band process on six channels, no lag in bursts 0 and 2, 24 ms a channel in 1 and 3; 400 draws): the lagged bursts' shares 0.24 and 0.31 against null means 0.04 (p 0.005 and 0.0025, q 0.01), the lag-free bursts' 0.005 and 0.09 against 0.02 (p 0.59 and 0.027); the run's mean share is beyond its null; the same seed is the same result, a rerun is a hit.
- **What it cannot resolve; what is NOT built.** The null is a zero-lag Gaussian process with the burst's covariance and ν independent samples; a burst beyond it has lags beyond sampling noise, not a wave. ν is a flat-ish band's duration × bandwidth: structure of the spectrum inside the band changes it. The functional measured is the SENSOR-level share; the CORTICAL quantities (the burst phase current of #139 and its
  direction consistency, the tile quadrature of `measurement.cellquadrature`) are the same linear forms of Q applied through the kernel, the level's spatial band and the gradient to every draw: that operator is not bound here (a bank of one Hermitian form per tile, 1000 draws a burst), so the Wishart null of the phase-current consistency is OPEN, and so is the direction-consistency statistic across bursts itself
  (the burst-phase-current spec leaves it, with its null, to the maintainers).

## Where per global burst: the space × time extent on the joint cells (`run global_bursts`, steps `cell_extent` and `extent_measure`; `compute/include/nxr/extent.h`; spec `docs/specs/2026-11-13-cortical-events.md`, "Phase 2b: where per global burst")

**Why.** A global burst says WHEN (a span of time tiles). Its WHERE is the cells it covers at the run's pair (spatial level, time level): which spatial tiles are above their own baseline, in which time tiles, how big that is, where its centre is and how it moves.
The same rule as in time and on a static map (hysteresis on a graph, `detectRegions`, unchanged); only the graph is the joint one. Descriptive: statistics across events are distributions, with no nulls.

- **Detection** (`cell_extent`, an after-step in the same run: the cells and the bursts are in it). Per spatial tile, the baseline is the median and robust SD (1.4826 MAD) of its cell amplitude across the run's time tiles (`summarize`, stored as `cell_baseline`); each cell is
  standardized, z = (amplitude − median) / SD (a tile with no spread is left out and counted). The cell graph is the level's touching tiles within a time tile (`tileAdjacency`) plus the same tile in the next time tile (`nxr::select::spaceTimePairs`). Per burst, on the
  cells of ITS time tiles (per hemisphere: its row's tiles), `detectRegions` with `extent_enter_sd` 1.5 / `extent_exit_sd` 1.0 (the bursts' own), kept when ≥ `min_cells` (1) cells and ≥ `min_area` (0 m²). **Every region is an event** (not only the largest), linked to its burst by
  the burst's ordinal; events are numbered by burst, then by their lowest cell. No merging across gaps.
- **The labelling** (`extents`, a new output kind `labelling`): a `set` Selection on the joint partition, an int32 array [spatial tile × time tile] (the cell layout) holding the event's ordinal (−1: none; a negative code is in no member), a dictionary `event k`, `n_members` = the
  events, `params_json` = the pipeline, parameters, inputs, run, the step's record (thresholds, the baseline summaries, each burst and its events, the tiles without spread, the layout), the bursts Selection it was found against, and the cache key. A rerun is a hit; another choice is
  refused unless `--replace`. The cell code of the atlas ((tile << B) | time tile, B = `bitsFor(F)`) and its inverse are `nxr::select::cellCode` / `cellOf` (checked: a frame beyond the layout, a gap code and a negative code are refused).
- **Per event** (`extent_measure`: the exact roll-up of its cells; the measurement is over the labelling, members = the events; new measures `n_cells`, `first_tile`, `last_tile`, `burst_ordinal`, `path_length`, codes 1024 … 1028): `n_cells`; `n_tiles` (distinct spatial tiles); `area` (the
  union of those tiles, m²); `amplitude` (area- and sample-weighted mean of the cells' amplitude); `envelope_max` (the largest cell peak); `first_tile`, `last_tile`, `duration` (s); `burst_ordinal`; `path_length` (the sum of the straight distances between the centre vertices of consecutive
  time tiles: a chord, never longer than the path over the surface; path dependent, so not a roll-up). **Per time tile** (over the time level, rows = the events): `area` (Σ the tiles' areas) and `centre_vertex` (the vertex nearest the area-weighted centroid of its tiles' vertices,
  `nxr::select::tileCentres`; NaN where the event has no cell). Statistics across the events (`count, sum, median, mean, sd, percentile:25/75/90`) of `n_cells`, `area`, `amplitude`, `envelope_max`, `duration`, `path_length`.
- **Options** (`--set`): `extent_enter_sd`, `extent_exit_sd`, `min_cells`, `min_area`, with `groups` and `combine` as the bursts'.
- **What it cannot resolve.** The extent is a set of cells of ONE pair of levels, quantized to the level's tiles and time tiles; at a coarse level the pattern of one tile has a footprint over the whole hemisphere (level 4 on the test cortex), so thresholds relative to each tile's
  baseline can give a hemisphere-wide extent: place events to within the band's scale, compare cells of one level. A region is "above its tile's own baseline", not a propagating oscillation: speed and direction are measurements of phase (phase 3). The centre's path is a chord of
  nearest-vertex centres, not a geodesic. Standardizing by the run's own history means a run of one chunk (the default) has a baseline from that chunk's time tiles.
- **Tested** (`test_extent`, `test_burst_extent`, synthetic): the cell code round-trips over layouts of 1 … 1024 time tiles; the joint graph's size; the standardization; an event of a path of tiles moving one tile per time tile is one event through the time links, a static one another;
  min cells, min area and a NaN cell; the roll-ups against direct sums. In the pipeline, a pattern of one tile handing over to a neighbour's inside one burst (left hemisphere) and a static one in the other (a second burst): the labelling equals an INDEPENDENT flood fill of the stored cells
  (0 mismatching cells), the event's strongest tile moves from the first tile to the second, every measure equals its direct computation from the labelling (cells, tiles, union area, amplitude, peak, first and last time tile, duration, burst, the area and centre per time tile, the path
  length), the moving event's path is longer than the static one's, the statistics are stored, a rerun a hit, another threshold refused, no event with a high cell minimum (stored all −1, a hit), per hemisphere the events lie in their hemisphere, the refusals.

## The spatial analytic signal: a phase-free spatial envelope (`compute/include/nxr/monogenic.h`, `envelope: spatiotemporal`; spec `docs/specs/2026-11-13-cortical-events.md`, "Spatial analytic signal (monogenic)")

**Why.** A spatial band leaves a SIGNED field (lobes of both signs). `|z|` of it is the spatial analogue of measuring a band-passed x(t) instead of its envelope: where lobes of opposite sign meet, the field passes through zero and `|z|` dips, so two overlapping
patterns cancel in the cells and one event can split in two (the phase 2b hand-over). The spatial analogue of the analytic signal is the MONOGENIC signal (Felsberg and Sommer 2001): with the Riesz transform `R f = ∇ Δ^{-1/2} f` (a tangent vector field),
the spatial amplitude `A = √(f² + |R f|²)` is unsigned and has no zero-crossing dips; the spatial phase `atan2(|R f|, f)` (0 … π) advances across lobes; the orientation `R f / |R f|` is the local wave-vector direction. **No reference has it**: Rheome, nxr and the
Studio have no Riesz transform, so this is new, and the maths is written from the paper.

- **Eigenmode first.** With the band's mode coefficients `c_k = ⟨ψ_k, f⟩`, `R f = Σ g_k c_k ∇ψ_k / √λ_k`: the exact 1-forms of the Hodge–Dirac pairing as tangent vectors per vertex (the mesh's P1 gradient, `phase::vertexGradient`, as `phasegradient`). The constant mode has
  no Riesz part. Computed as the band's own transform with the gain `g/√λ` and then one gradient, so the transform choice is the band's (`transform`: stored eigenmodes, else computed and stored, said; else Chebyshev, announced; `exact` for a test). A Chebyshev Riesz part
  follows its own gain `g/√λ` to within `tolerance` of its peak (the degree is raised from the band's, up to 5120; a scaling level, whose gain is not 0 at λ = 0, is refused).
- **Space × time.** `z(v, t)` is complex in time (the band's analytic signal), `R` real and linear, so `A² = Σ_a |z_a|² + Σ_a |R z_a|²` over the source components `a` (a free kernel: per component, then summed; no orientation assumed, as `magnitude`).
- **The option.** `envelope: temporal | spatiotemporal` on `run measure --set cortical_band_cells`, `cortical_band_cells` (lists) and `global_bursts` (the cell amplitudes that `cell_extent` reads). **Default `temporal`: the chain and its values are the old ones** (tested against the direct
  computation, 4e-8). `spatiotemporal` is a parameter of the step `meeg.apply_kernel` (with `tiles`/`level`): in preparation it binds `M_l = (F_l ⊗ I) K` and `R_l = (∇ Δ^{-1/2} F_l ⊗ I) K` STACKED per vertex (4c rows a vertex, c the kernel's components): a field of type
  `monogenic`, which only `magnitude` reads (it checks the type); `magnitude` then gives `A` through `nxr::signal::amplitude`. The rest of the chain (`measure` over the joint cells, `tile_max`, `cell_extent`) is unchanged.
- **Nodes (compute, pure).** `rieszGain` (g → g/√λ), `rieszTransform` (the gradient per component), `amplitude`, `phase`, `orientation`. Phase and orientation are tested and not stored: no registry output asks for them yet.
- **Rows of the two envelopes.** The stored measures stay `amplitude` / `envelope_max` on the pair's joint Selection, and a stored measurement is unique by (Selection, recording, measure): the two envelopes of one pair cannot sit side by side. The run's cache key holds
  `parameters.envelope`, so the other envelope is a miss and the existing rows are REFUSED unless `--replace` (never silently replaced), and `producer_json.parameters.envelope` records which one the rows are. Both side by side need a measure name of their own (open item).
- **Memory.** The stacked map is 4c rows a vertex instead of c, float64, held for the run and shared by every band reading the level (20k-vertex free cortex, 3 components, 300 channels: about 576 MB a level instead of 144 MB); a chunk's stacked sources are
  `[4c·V × (chunk + halo)]` complex float32 until `magnitude` releases them.
- **Cannot resolve.** The Riesz part of a field on a mesh is the P1 gradient of `Δ^{-1/2} f`: near the cut medial wall (the glued band is 0 there) the gradient uses only the faces whose vertices are all kept, so the wall's edge is one-sided; a band of a few lobes on a
  coarse mesh has `|R f|` smoothed by the P1 gradient. `A` is a local amplitude of the BAND's pattern: it removes the dip of a sign change, not the real cancellation of two superposed patterns in anti-phase in time (their sum is the field). An event's extent still comes
  from the cell means.
- **Tested.** `test_monogenic` (unit sphere, a planted degree-5 zonal pattern band-passed, 642 vertices): `A` follows the continuum `√(P₅² + sin²θ P₅'²/30)` (within 8% of its peak), no dip at the zero crossings (`|f|` reads 0, `A` above 0.3 of its median), the phase < π/2 on
  positive lobes and > π/2 on negative ones, the orientation along the pattern's gradient with the sign of `P₅'`; `R` by eigenmodes equals the dense matrix function (1e-9) and the Chebyshev series follows it; a mode's sign flip changes nothing. `test_spatial_band`: the stored-modes
  Riesz part equals the dense exact one (1.4e-7), the series (4.2e-4), level 0 Chebyshev refused. `test_spatial_analytic` (pipeline, the phase 2b hand-over, never a real dataset): both envelopes' cells equal the direct computation (4e-8); `temporal` named = unnamed (a hit);
  the other envelope refused without `--replace`; chunk widths 4096 and 1000 identical; a rerun a hit; a free kernel (1, 2, 2) gives 3 times the fixed cells; `cortical_band_cells` (lists) the same cells; the hand-over is TWO events under `temporal` and ONE under `spatiotemporal` at one
  exit level (extent z 5909 vs 6988 at the trough).

## Quadrature per cell and per burst: travelling or standing (`run cellquadrature`; `compute/include/nxr/quadrature.h`; spec `docs/specs/2026-11-13-cortical-events.md`, phase 2d)

**Why.** Is the activity at a joint cell, or in a global burst, travelling or standing? A travelling or rotating pattern has a source cross-spectrum with a quadrature (imaginary) part, which no zero-delay mixing makes; a standing pattern (one real
pattern times one phase) has none. Read per spatial tile of a level and per time band at its constant-Q level, so each cell and each burst carries its own answer, from stored products only: no recording pass.

- **The inputs (all stored):** the analytic cross-spectrum's tiles (`run crossspectrum --estimator analytic`: Σ z zᴴ per time tile, `<stem>_<band>_windows`), the inverse kernel (free or fixed; every component is used, no orientation assumed), the joint Selection of
  the pair (`run select_product`; found by the cross-spectrum's recording, `--spatial-scale` and its tile level, or `--joint NAME`) and, for bursts, a spans Selection on the recording's line (`--bursts NAME`, e.g. `run global_bursts`'s `bursts`).
  **The cross-spectrum's tile level must equal the joint pair's time level** (window_samples = 2^level): otherwise it is refused, with the levels the recording does have joints for. The cross-spectrum's band is the caller's: a band that is not
  the one the joint's level resolves is not refused but reported in `warnings`.
- **The quantity.** For a spatial tile s of level l, B_s are the rows of M_l = F_l K of the tile's vertices (all components; M_l bound once, the level's spatial band folded in: eigenmodes first, Chebyshev an announced fallback, *The spatial band's transform*), T = B_s C B_sᵀ the
  tile's source cross-spectrum, and the reading ‖Im T‖_F / ‖T‖_F (nxr's `lagged_fraction`; `run quadrature`'s per-band figure, here per tile and per time tile or burst). For ONE travelling pattern it is at most 1/√2 ≈ 0.707; 0 for any standing pattern; up to 1 for several patterns at different phases.
- **Without forming T:** with the channel Gram G = B_sᵀ B_s, ‖T‖² = tr(Cr G Cr G) − tr(Cq G Cq G) and ‖Im T‖² = −tr(Cq G Cq G) (C = Cr + i Cq): two channels × channels products per (tile, time tile) and nothing larger (`quadrature::groupedCongruenceNorms`, a function of matrices).
  Cost: 2 m³ flops per spatial tile and time tile (m channels); 128 tiles × 600 time tiles × 270 channels is about 10 minutes. Memory: M_l (rows × channels doubles) and the tiles of C (tiles × m² complex).
- **What is stored: the ingredients, never the ratio.** `cross_spectrum_norm2` = Σ|T_ij|² and `cross_spectrum_quadrature_norm2` = Σ|Im T_ij|² (nxr-neuro's own measures 1022 and 1023, reduction sum), as measurements OF the recording over (a) the JOINT cells: an array beside the recording
  [spatial tiles × time tiles × 2] float64 (NaN where a time tile is not stored: a bad sample or the span's end dropped it), and (b) the BURSTS: [spatial tiles × bursts × 2], rows the level's tiles, members the Selection's events in order. The quadrature fraction `quadrature_fraction` (1111) is
  a ratio of sums, **derived at read** (`derived_measurement_values`), never stored, never averaged. Provenance: `computed_by` names the pipeline and its key's hash, the recipe (every option, the inputs, what the spatial band resolved, the warnings) is an entry of the `recipes` dictionary. A rerun is a hit;
  other rows of the same Selection and measure are refused unless `--replace`.
- **Which roll-ups are exact.** Over TIME the exact thing is the cross-spectrum: C summed over a burst's time tiles IS the burst's (products add), so the per-burst quantity is made from the SUMMED C (`partition::sumOverMembers`), then the ingredients per spatial tile. It is NOT the cells' ingredients summed
  over the burst's tiles (Σ_t ‖T_t‖² ≠ ‖T of Σ_t C_t‖²: on a stable travelling wave of n tiles the second is n times the first, tested). Over SPACE the ingredients of a union of tiles are not the sum of the tiles' (the cross-tile blocks are missing). The stored measures say so in their descriptions.
  A burst's time tiles are those stored tiles wholly inside its [start, stop]; a burst with a missing tile (dropped for a bad sample, or outside the stored span) reads NaN, never a partial sum (`events_read` in the result).
- **What it cannot show.** The cells' ingredients are a tile's own internal quadrature only; the quadrature BETWEEN two tiles (a wave running from one tile into the next) is in the cross-tile blocks, which are the tile pairs (next item). A source-space quadrature is the sensors' quadrature through the kernel
  (no zero-delay mixing adds any), and the kernel's spread blurs a pattern into neighbouring tiles. The fraction is a summary: it discards which vertices lead which, the direction and the speed (`phasegradient`, `speedspectrum`, `run quadrature`'s pairs).
  Noise alone reads a high fraction on a short window (a band of width B over T seconds holds about B·T independent samples), so a fraction is read where the tile carries power (its `cross_spectrum_norm2`). No nulls (out of scope).
- **Area weights (`--weights area|none`, default `area`.** The ingredients are Σ_ij a_i a_j |T_ij|² and Σ_ij a_i a_j |Im T_ij|², a the barycentric vertex areas (the tiles' areas are sums of them), so the value does not change when a region of the mesh is sampled more
  finely; `none` counts every vertex equally (the behaviour before this option; same cache key and rows). The trace form with W = diag(a): G = Bᵀ W B, ‖W^½ B X Bᵀ W^½‖² = tr(X G Xᵀ G), the same two products. It is a new measure definition: rows made under the other setting are another job's and refused unless `--replace`.
- **Tile pairs (`--tile-pairs on|off`, default on).** For each pair (s, s') of tiles of the level joined by a mesh edge (`ladder::tileAdjacency`, each pair once, s < s'): T_ss' = B_s C B_s'ᵀ, the block coupling the two tiles, where a wave running from one tile into the next lives. Stored `tile_pair_cross_spectrum_norm2` (1029) and
  `tile_pair_cross_spectrum_quadrature_norm2` (1030): array-backed rows = the tile pairs, over the joint cells ([pairs × time tiles × 2]) and over the bursts ([pairs × bursts × 2]), the pair list as int32 [pairs × 2] beside (`<array>_pairs`); the fraction is derived at read (`tile_pair_quadrature_fraction`, 1112). Without forming T_ss':
  ‖Im T_ss'‖² = −tr(Cq G_s' Cq G_s) (and the same for Re), each trace Σ_ij (X G_s')_ij (X G_s)_ji, so with X G_s held per tile a pair costs m² on top of the tiles' own products (`quadrature::pairCongruenceNorms`). Same per-burst rule (C summed first) and NaN rule. Limits: only touching tiles; the kernel's spread
  puts a pattern into the next tile; the fraction is a summary; noise reads high on a short window. Spec: *Phase 2d amendment*.
- **Tested** (`test_cellquadrature`, `test_quadrature`, `test_partition`, `test_measures`; the amendment above: weighted and pair forms against the dense forms, a refined region leaves the weighted energy unchanged, a travelling wave reads higher between touching tiles than within them, `weights: none` reproduces the old values; synthetic: a two-hemisphere surface, one channel a vertex, 12 Hz travelling along z in four time tiles and standing in four others, a bad span): the ingredients equal the dense T's squared norms to 2e-14 for a fixed and a free
  (3-component) kernel; the planted travelling tiles read 0.50 and the standing 0.0005 (cells and bursts); a burst's ingredients equal the dense form on the summed C (1e-14) and are 4 times the cells' sum on a 4-tile wave; a burst with a dropped tile is NaN; the derived fraction equals √(‖Im T‖²/‖T‖²); a rerun is a hit; other rows refused unless `--replace`;
  the refusals (tile level of the cross-spectrum not the pair's, no joint at the level, not a cross-spectrum, not events).

## Scale selection: each event's own size and rhythm across level pairs (`run global_bursts` (lists of bands and spatial scales), `run scaleselection`; `compute/include/nxr/scale.h`; spec `docs/specs/2026-11-13-cortical-events.md`, "Phase 2e")

**Why.** An event has its OWN size and rhythm; we do not impose them. Detected over a declared set of (spatial level, time level) pairs, the same episode is seen at several pairs. The pair of its strongest response is its characteristic size (the level's tile,
its Weyl band) and rhythm (the time level's band): scale selection as in blob detection (Rheome's `detect.blobscale`: the scale of the largest normalised response, read AT the blob; a selection on the ladder's edge is unresolved). Its response across the pairs
(the footprint) says whether it is a single-scale rhythm or spread over scales. Descriptive: no nulls, no tests (out of scope).

    nxr-neuro run cortical_event_pairs --dataset D --subject S --spatial-scales '["parts 8","parts 16","parts 32"]' --bands '[[4.0,8.0],[8.0,16.0]]' --all     # one pass: every pair's cells, global bursts and extents
    nxr-neuro run scaleselection --dataset D --subject S --recording rest --spatial-scales '["parts 8","parts 16","parts 32"]' --bands '["8-16 Hz","4-8 Hz"]'                        # stored rows only: the selection

- **The per-pair half (`run global_bursts` (lists of bands and spatial scales)).** `cortical_band_cells` (lists)' chain for every pair of `space_levels` × `bands`, and, after the loop and for each pair, what `global_bursts` does with one: the trace (the max over the level's tiles of the cells' amplitude),
  its baseline, the pair's OWN global bursts (`detect_spans`) and the extents of the bursts on the pair's cells (`cell_extent`, a labelling of the joint cells). The same steps, unchanged, written once a pair by `for:`; the executor has nothing analysis-specific.
  The joint Selections of every pair must exist (`run select_product`; a missing one is refused naming every missing pair and its command). Not here (`global_bursts` has them for one pair): Rheome's fit of the burst's edges and the per-event measures. Options as `global_bursts`'
  (`enter_sd`, `exit_sd`, `merge_s`, `min_s`, `extent_enter_sd`, `extent_exit_sd`, `min_cells`, `min_area`, `groups`, `combine`). Stored per pair: the cells (as `cortical_band_cells` (lists)), `bursts_l<l>_b<band>` (spans), `cell_baseline_l<l>_b<band>` (median and robust SD
  a tile), `extents_l<l>_b<band>` (the labelling).
- **The comparable unit.** Each event's response at a pair is read in standardized units: z = (cell amplitude − the tile's median) / its robust SD over the run's time tiles, as the extents were found. A raw amplitude (or the raw max over tiles) is not comparable across
  levels (it rises with the number of tiles and the band's noise). `--response peak` (default; the pair of the largest peak z, blob-detection selection) or `mean` (the mean z over the event's cells, less sensitive to one cell and to the cell count).
- **Matching across pairs: the ladders' exact nesting, no distances.** Two events of different pairs are the same cross-pair event when at least `--min-link` (1) pairs of their cells nest in space (tile s of level l lies in tile s >> (l − l′) of the coarser level l′; a
  hemisphere is in the code) and overlap in time (the tiles' sample intervals intersect; dyadic windows nest). The cross-pair events are the connected components of those links, numbered by their earliest sample. Events of one pair never link directly.
- **Best pair and footprint.** The best pair is the pair of the largest response (a tie: the first in the declared order, space level then time level); the footprint is the event's response at every pair, NaN where none of its events was detected there (not 0, not
  the sub-threshold response). `scale_edge` (a bit mask) flags a best pair on the set's edge, where the true scale may lie beyond (`blobscale`'s `atFloor`): 1 the finest spatial level, 2 the coarsest, 4 the shortest window (the fastest band), 8 the longest; set only
  for a dimension with at least two values.
- **What is stored** (registry 1031 … 1040, none rolls up). The cross-pair events are a `spans` Selection on the recording's time line (`<recording>_scale_selection`; each event's span in time; `params_json.scale` = the pair table: levels, bands, tile areas, the joint
  and labelling of each pair, the options, the warnings). Measurements OF the recording over it: `scale_space_level`, `scale_time_level`, `scale_peak` (the response at the best pair), `scale_tile_area` (m², the mean tile of the best level: the event's size),
  `scale_band_lo`, `scale_band_hi` (Hz: its rhythm), `scale_pairs` (how many pairs hold a member), `scale_edge` [1 × events × 8]; the footprint `scale_response` [pairs × events × 1], rows the pairs in the declared order; and over each pair's own labelling `scale_event`
  [1 × its events]: the cross-pair event of each event. `computed_by` names the pipeline and its key's hash; a rerun with the same inputs and options is a hit; another option is refused unless `--replace`.
- **What it cannot resolve.** The size and rhythm are the best pair of the DECLARED set, quantized to its levels and bands, not a fitted scale. Standardization is not free of selection bias: a pair with more cells has more chances of a large noise z, so the peak favours fine
  levels and short windows (use `--response mean` to see it). A coarse level's footprint is wide (its tiles hold the pattern of a neighbourhood). A response is a standardized amplitude, not the pattern's shape; speed and direction are phase measurements.
- **Tested** (`test_scale`, `test_scale_selection`, `test_measures`; synthetic): the nesting (the hemisphere bit, siblings, the level difference) and time overlap; the links (nesting AND overlap needed; `min_link`), the numbering, the footprint, the best pair and ties,
  the mean response and the refusals, on hand-made pairs. In the pipeline, two episodes (A the level-3 pattern of one left tile at 6 Hz, B the level-4 pattern of one right tile at 12 Hz) over two levels × two bands: every pair's events of an episode are ONE cross-pair
  event; A's best pair is (level 3, 4-8 Hz) and B's (level 4, 8-16 Hz) with a response 9 to 11 times the other pair's of the same band; the footprint, tile area, band, pair count, edge flags and span equal an independent computation from the stored labellings and cells; a
  rerun a hit; the mean response; `min_link` large splits the events; the refusals (every missing pair named with its command, no run, another source, a repeated level, an unknown recording and response).

## Statistics (`io/include/nxr/io/statistic.h`; spec `docs/specs/2026-11-13-statistics.md`)

**Why.** A result over a whole recording that does not fit the selection/measurement system — a Welch PSD, a cross-spectrum, a distribution, a percentile —
is a STATISTIC: "field -> selection/measurements -> statistic. the statistic is computed from the partitions". Brainstorm's `timefreq` file
(the values with their axes, Measure, Method, Options, nAvg, the parent DataFile) is the model.

- **The node** (table `statistic`, a zarr array at its path, its `zarr.json` the row): `measure` and `method` (names of `io/schema/statistics.json`), the
  `value_type` of each value as a field's (scalar · integer · complex · vector2 · vector3 · matrix3 · quaternion · hermitian3, its shape trailing the axes:
  complex [2], quaternion [4], hermitian3 [3, 3, 2];: so readers know from the metadata how to treat it), OF a
  field or a statistic, the partition it is computed from (`partition_selection_id`) and how many of its members (`n_members`, `members_skipped`, `dof`),
  the span (`span_json`), the axes of its values in order (`axes_json`: each a manifold, a Selection's members, closed-form edges or its own values), the
  components of the last axis, the unit and spectral scaling, every option of the method, the provenance and cache key. Written by
  `SubjectWriter::statistic`, read by `read_statistic`, listed by `statistics_of`; `find_cached` takes the table.
- **Proposed for the shared model, started in nxr-neuro.** New datasets have the table; a dataset made before it gets it IN PLACE with its
  first statistic (`ensure_statistics`: the statistic block of `model.sql`, sync_job's table list widened; `user_version` unchanged, recorded in `meta`), and
  until then reads as one with no statistics: no migration. The parity test against nxr's `model.sql` compares everything but that block.
- **Tested** (`test_statistic`): written and read back with its axes and record, its zarr.json, listed by its field, refusals (an unknown measure or method,
  axes that are not the values', no parent), removed; a dataset made before the table opens, recovers, has none, and gets the table with its first one.
- **Statistic summaries** (`run summarize`, the runner step `measurement.summarize`; spec `docs/specs/2026-11-13-statistic-summaries.md`): a stored measurement summarised
  over its partition's members into a statistic (method `summary`; a statistic ACROSS the members: MATLAB's ScalarizationMethod, by contrast, reduces within a frame, our measurement layer), per row, pooled or of the mean over rows; summaries
  mean · sd (n − 1) · median · mad (1.4826 MAD) · percentile:p (numpy linear) · peak · kurtosis · skewness · min · max (`nxr::stats`, golden from
  numpy/scipy). It reads either stored form (array-backed or scalar rows), never the recording; a rerun is a hit. `band_bursts` stores its thresholds'
  ingredients this way (its baseline) and detects from them.
- **The `pca` statistic** (`run pca`, the node `stats.pca`, `compute/include/nxr/pca.h`; spec `docs/specs/2026-11-16-meg-fingerprinting.md`, 4.4): the principal components ACROSS the members of a table (members x
  features), method `pca`, generalising `eigendecomposition` (which decomposes one second-moment statistic; unchanged). The table is a stored real statistic of two axes (`--of NAME --members AXIS`, by name or position; the
  other axis is the features; a statistic of more axes is reduced to a table first). The maths is `nxr::stats::pca`, a pure function of a matrix: the thin SVD of the table with each column's mean removed (scikit-learn's PCA,
  `svd_solver="full"`), never the features x features covariance; `--scale none` (default) or `unit-variance` (ddof 1; a constant feature is refused). Every component up to the table's rank is kept (numpy's
  `matrix_rank` rule; at most members - 1), so any k is a read. **Sign convention, fixed:** each component's entry of largest absolute loading is positive, its scores' column the same sign (scikit-learn's `svd_flip`,
  `u_based_decision=False`). Explained variance is S^2 / (n - 1), its ratio over the table's total variance. Stored as four statistics, of the input statistic: `<name>` (`principal_components`, [component x feature]),
  `<name>_scores` (`principal_scores`, [member x component] = U S), `<name>_variance` (`explained_variance`, [component x 3], float64) and `<name>_centring` (`pca_centring`, [2 x feature]); the first, second and fourth in
  the input's stored precision. **Reconstruction at k** is derived at read: `nxr::stats::reconstruct(scores, components, mean, scale, k)`. `nxr::stats` is tested against numpy's SVD
  (`compute/test/golden/gen_pca_golden.py`). **What it cannot show:** a component is a direction of variance in THIS sample; with n members at most n - 1 exist and the leading ones of a small sample are not stable
  estimates of a population's.
- **Next** (the spec's order): the other spectral producers, the histogram, Brainstorm's sensor-level timefreq on import. (`run psd`'s PSD is a statistic since 2026-11-13; checked
  against the code 2026-11-14: the `<recording>_psd_welch` Field of the spec's table is already gone.)

## One selection and measurement system (spec `docs/specs/2026-11-17-consolidation.md`; definitions in `pipelines/definitions/`)

**Why.** Selections and reductions had grown several competing definitions (tiles, joint cells, frame tilings, pyramids, per-verb sums). There is now ONE of each: five
selection makers, four measurement operations, one table for values over members, and ONE measuring definition fed by field chains. Old names are not kept as aliases: a
retired name is refused with the name of what replaces it (`nxr-neuro run tiles` says `run select_dyadic_surface`).

- **Storage and reading (schema 53).** One table holds a value over a selection's members (`selection_measurement`, array-backed `[rows x members x components]`); a product's factor that is a level of a grid
  names the grid and the level's IDENTITY (its ordinal in the grid's level dictionary; the level's code is its label), and a dyadic grid's level dictionary records the mean tile diameter in mm (a time grid's, the
  tile's duration in s), which is what a query by physical size reads. The measures registry (`io/schema/measures.json`, `measures_nxr.json`) has one rule key, `merge` (sum, max, min, mean, none, derived), and a `form`
  column (the form of the field a measure is taken of); io applies the merge rule in SQL (`nxr::io::query`: filters by selection, member, level in physical units, subject or group; roll-ups through a grid's hierarchy
  and across subjects) and compute in memory (`roll_values`), and a parity test (`test_rollup_parity`) compares the two. `nxr-neuro migrate measurements --dataset D [--dry-run] [--remove-statistics]` makes a
  dataset made before schema 53 follow it, subject by subject and resumable; the old statistics are kept unless `--remove-statistics` (the explicit second step).
- **The glued cortex.** The imported two-hemisphere mesh is the manifold of record; the glued surface is derived (`run glue`) and is the default computational manifold. A Field, a selection or a measurement made on the
  glued surface is stored on the original: the elements are restricted back through the glue map (the bridge has no counterpart and is dropped, like the medial wall) and the record says so (`restricted_from`).
- **Operators a node builds** (the fused kernel of a chain, with its band and pre-operator) are stored through io as operators with their provenance and found by their key on the next run; no process-wide cache.
- **Quantities.** A range may be open: `>=0.4 s`, `<=5 s`; a duration may be in `cycles` of the band's centre (`>=3 cycles`).
- **The makers (selections are made, never recomputed from parameters).** Each is a definition that also runs alone and stores its selection once (get-or-create):
  - `run select_dyadic_surface --rule subject|group|transfer`: the dyadic equal-area tiles of a surface as one grid with closed-form levels (was `tiles`, `group_tree`,
    `group_tiles`); the finest tile is a quantity (`--finest "20 mm"`, `"parts 256"`, `"area 400 mm2"`), the level and its snapped quantity are recorded;
  - `run select_time_grid --rule tiles|band|frames|constant_q|segments|windows|chunks`: the dyadic time grid of a line (was `frametiling`, `constantq`). **Frame density and origin
    are options with their defaults in the YAML** (2026-11-17): `frames_per_cycle` (default 36, the 10 degree step of phase work) or `frames_per_tile`, and `origin` as a
    point: `start` (the recording's first sample), `clock`, or `event` (a stored selection's first member). No method constant is hard-wired in compute;
  - `run select_bands --rule frequency|eigenvalue|wavelength|parts|levels`: a band set on the dataset's Frequency or Eigenvalue line (was `pyramid_bands`, `spatial_bands`);
  - `run select_product`: a product of grids and band sets (a box, a grid of cells, the joint partition), levels coupled to a band in closed form (was `joint`);
  - `select.events.spans|regions|cores|tracks`: irregular selections made by detection (nodes inside `band_bursts`, `global_bursts`, `patches`, `spatial_events`, `vortex_cores`).
- **The glued cortex is made by its own definition** (2026-11-17): `run glue --surface cortex_pial` stores the glued surface as a manifold, its map (glued vertex to
  original, -1 on the bridge) and the bridge as a labelled irregular selection. The imported two-hemisphere mesh stays the manifold of record (atlases, kernels, exports,
  figures); the glued surface is the default computational manifold for spatiotemporal patterns, results are restricted back to the original hemispheres, and orientations are
  also reported in the anatomical frame as a check of the bridge's transport. `eigenmodes` and `meeg.apply_kernel` take the glued surface by name; `--glued` is gone.
- **The four measurement operations**, one implementation each (`compute/include/nxr/measurement/`):
  - `reduce`: a field (or a measurement) over any selection's members, by the measures of the registry (the only move that removes a factor);
  - `measurement.roll_up`: through a grid's closed-form hierarchy, or across a group's subjects (`run roll_up`, `scope: group`; was `pyramid reduce`);
  - `summarize`: members to a statistic (`run summarize`, was `statistic`);
  - `estimate`: a statistic from a partition of the stream (the Welch family, histograms).
  **Roll-ups exist in io's query as well as in compute** (2026-11-17): each measure's merge rule (sum, max, min, mean via sum and count, ratios derived at read) is DATA in the
  measures registry, applied by io in SQL and by compute in memory, with a parity test between the two.
- **Quantities, never level indices.** Every selection parameter is a number with a unit in a named chart of the space it lies in: `8-12 Hz`, `2-5 s`, `20 mm`,
  `wavelength 40 mm`, `parts 16`, `area 100-400 mm2`. A bare level index is refused outside the explicit `index` chart; a definition declares `unit` and `chart` for each
  such parameter. The run records the stated quantity, the resolved level and the snapped quantity.

  | was | is |
  |---|---|
  | `--depth`, `--atlas-depth`, `--lc-levels` | `--finest`, `--atlas-finest`, `--lc-scales` |
  | `--space-level`, `--levels`, `--reference-level`, `--space-levels` | `--spatial-scale`, `--spatial-scales`, `--reference-scale` |
  | `--time-level`, `--time-levels`, `--level` (tile of 2^k samples) | `--tile "<s>"` or `--band "<Hz>"` |
  | `--frame-level`, `frames_per_cycle` | frames from a band and `--frames-per-cycle` (default 36) or `--frames-per-tile` |
  | `--segment-level` | `--segment-seconds` or `--segment-samples` |
  | `min_s`, `max_s`, `merge_s`, `min_cells` | `duration` (a range), `gap`, `area` (mm2): in seconds or cycles |

- **`run measure` (version 2)** reduces a stored field, or a field chain composed in (`band_analytic`, `cortical_band`, `member_activation`, `band_outer`; `kind: chain`
  definitions that also run alone), over a stored selection or one made by a maker composed in. The `--set NAME` presets are only named parameter sets; the cache key holds the
  resolved parameters, so two presets with equal parameters are one cached result.

  | preset | replaces |
  |---|---|
  | `cortical_band_cells`, `cortical_band_cells_current` | `cortical_cells`, `cortical_cell_pairs`, `cell_phase_current` |
  | `cortical_band_pyramid` | `pyramid` (array `[band x leaf x frame]` at nxr's atlas node path) |
  | `sensor_band_amplitude` | `band_amplitude`, `measure.band_amplitude` |
  | `analytic_cross_spectrum_tiles` | `crossspectrum_analytic` |
  | `channel_span_mean` | `channel_mean` |
  | `surrogate_lagfree_envelope`, `psd_channel_mean` | `lagfree_envelope`, the PSD's channel mean inside the surrogate |
  | `flow_max_amplitude` | `flow_amplitude` (stored, then read by name by `flow_modes`) |

  Several (band x spatial scale) pairs are lists of `--band` and `--spatial-scale`: one plan each, each cached on its own (this replaces `cortical_cell_pairs` and
  `cortical_event_pairs`). `cyclepattern` is the `estimate` of the window cross-spectrum series then `summarize` (method `cycle_pattern`); `opticalflow` is `run flow_modes`
  after `run measure --set flow_max_amplitude`; `phasegradient` is the definition `run phasegradient`.
- **`measure query`** reads stored measurements at a scale stated as quantities (`--band`, `--spatial-scale`, `--tile`); the roll-up to a coarser level is applied at read,
  the level is printed with its snapped quantity. `measure export` is unchanged.
- **Storage.** One table for values over members (`selection_measurement`, array-backed); `statistic` holds statistics only (a Welch PSD, a covariance, a histogram's bins,
  the summaries across members), and the checker refuses a statistic output with a member axis. Schema 53 adds `selection_factor.level`; the migration of schema-52 datasets is
  `package migrate-measurements` (written; run only on the maintainers' word).
- **Tests.** `test_select_makers` (the maker definitions, the quantity refusal, the glued cortex); the CLI test refuses every retired spelling naming its replacement.


## Optical flow (`compute/include/nxr/opticalflow*.h`, `run flow_modes`)

Rheome's `flow.apparent` for a recording, streamed (the memory held does not depend on the window's length). One member of a time filterbank (`--family`, `--gain`, `--freq`: no defaults) is
applied at the sensors and read at the frames of a frame grid made first (`run select_time_grid --rule frames`, input `grid`): the transform returns the member's analytic signal at the recording's full rate and never decimates, resamples or windows it; the frames are the grid's. They are carried through the kernel, and reduced to the amplitude |J| per source (`--input signed`: the band signal itself, fixed kernel only). Horn–Schunck optical flow is solved per
frame on the cortex, and each velocity is read at once in the Hodge–Dirac modes: **exact** coefficients (sources and sinks of the motion) and **coexact** coefficients (its rotation).

- Defaults: the **connection** regulariser, the Cortical Flow Studio's connection Laplacian on the signpost gauge with its gauge data term (ported line for line), solved by
  **pcg**. Named options: `--regulariser ambient` (Lefevre and Baillet's form; `--datafit triangle`), `--method direct` (Rheome's exact solve, kept for parity), `--method perscale` (one series per
  band-pass member of a graph filterbank, `--scale-family`). `--alpha` unset takes the regulariser's own default (connection 0.01, ambient 1); vary it and report the range.
- Stored: `<stem>_flow_modes` (float32 [modes x frames-1]) on the eigen axis x `<stem>_time` (the coefficients are per frame, not a reduction), and, the
  divergence, curl, speed and velocity as additive **sums over a product of the dyadic surface grid and the time grid**: a byte-less Field `<stem>_flow` and, per leaf of the grid (`finest`, a quantity) and per frame of the recording
  (`frames_per_cycle` or `frames_per_tile`, `origin`), sums of the vertex values weighted by vertex area, per second, over the flow steps in the frame: `flow_divergence`, `flow_curl` (1/s), `flow_speed` (m/s, the tangent speed),
  `flow_velocity_1` / `flow_velocity_2` (v . e1 / e2 in the canonical frames, with `run surface_frames`) and `flow_velocity_anterior` / `flow_velocity_second` (the anatomical frame), on a joint Selection
  `<recording> x <stem> <leaves> x <frame_s> s`; the frame tiling's `count` of the Field is the steps a frame holds. The time means are derived at read (`measure query ... --measure flow_speed_density`: sum / (area x
  steps), at any space level and time level; the mean speed is the area-weighted mean of the vertices' mean speeds, `test_opticalflow_pipeline`: the leaf sums equal the steps x the vertex time means to 1e-6, the whole-cortex mean
  speed to 7 digits). **No per-vertex time-mean Field is written any more** (a correction, no option; Fields of earlier runs are untouched; a rerun over one needs `--replace`). A frame the member's rate does not
  reach holds no step and no mean. The per-frame velocity is not stored. `--tiles` summarises the run's own time-mean maps per ladder level (none on a cache hit).
- What it cannot resolve (also in `producer_json.warnings`): the motion is kinematic, not physical; it is aperture-limited (alpha); it is per frame, not per second; the window's edges; the
  record's bad segments are **not** left out (choose a window free of them).
- Findings behind the defaults: `docs/audits/2026-11-11-opticalflow-smoothness.md`. Side-by-side runs of the methods: `cli/scripts/opticalflow-compare.sh`.

## Field forms (`compute/include/nxr/forms.h`; the `form` step of a pipeline)

**Why.** Between a source estimate and any measure (a phase gradient, a power map, a tensor) a scalar or vector is drawn from the field: the normal
current, the tangential flow, the divergence, the rotation. The Studio names these **forms** (your spec "Types and forms", 2026-10-01), and nxr-neuro
ports them.
- **The reference:** the Studio's `fieldForms.ts` (`FORMS`, `formValues`) and the face-gradient operator it stands on (`faceGradient.ts`:
  `assembleFaceGradient`, the face scalars, the area-weighted face → vertex mean, the normal part, the mean curvature). Ported line for line, with
  JavaScript's arithmetic (float32 stores, V8's `Math.hypot`, no fused multiply-add).
- **The types:** ℝ (a number), ℂ (a tangent vector, embedded as 3 numbers) and ℍ ([w x y z]; a free-orientation current is ℍ with w = 0).
- **The forms:**
  - ℝ: `f`;
  - ℂ: `abs`, `div`, `curl` and their variants (`div_tan`, `div_3d`, `div_curv`, `div_2H`, `curl_tan`, `curl_3d`, `curl_curv`), `v`, `Jv`;
  - ℍ: `absq`, `nx`, `w`, the same divergence and curl variants of x, `xt`, `x`.

  Frames are stacked; each is computed alone.
- **Complex fields** (the analytic signal), the extension for phase: a linear form of re + i·im is form(re) + i·form(im), so the phase is kept and the
  result may feed a phase gradient. `abs` and `absq` give the envelope (over every real and imaginary component), a real number with no phase.
  `isLinear` says which is which.
- **Tested** (`test_forms`): the operator and all 27 forms equal the Studio's own functions value by value (float32, to the bit) on a bumpy icosphere
  with random fields. The golden comes from the Studio's code (`gen_forms_golden.mts`).
- **What it cannot show:** a form is computed on the mesh as given. The divergence and curl are the face-gradient (P1) operators averaged to vertices,
  so they resolve nothing finer than a few edges, and `curl_curv` is mesh error only.

## Zero-phase taps and the analytic signal (`compute/include/nxr/taps.h`; the `signal.band_filter` step and its analytic signal)

**Why.** A pipeline filters a recording to a band one chunk at a time, and reads the band's phase. A filter that can be computed chunk by chunk with a
known halo is what makes that exact.
- **The reference:** nxr's `filters/taps.py` (`gain_taps`, `convolve_rows`), which is the Studio's `gainTaps` and `convolveRows` (`deviceChain.ts`).
  Ported line for line, numpy's pairwise sums included.
  - A real gain G(f) (a Morlet band, `filterbank::morletBand`, nxr's `morlet_gain`/2, the pipelines' default; or a Meyer band, `filterbank::meyerBand`, nxr's `meyer_band`) is taken to the time domain once: the inverse FFT on a 2^14 grid,
    truncated where the tail holds under 10⁻⁶ of the energy (at most 1,024 taps a side), the DC restored.
  - Every row is then convolved with the taps, an index past the block's end reflected evenly.
- **The analytic signal** is the same gain's odd taps (its Hilbert transform) on the same rows: re from the even taps, im from the odd ones. A tone the
  band passes comes out as e^{iωt}.
- **The halo:** the taps are finite (K a side), so a chunk read with K samples of context on each side gives exactly the whole recording's result on its
  columns. The chunk service serves that halo.
- **Tested** (`test_taps`):
  - equal to nxr's functions to the bit: the taps of two bands (even and odd, the truncation included) and the convolutions (a block's middle and its
    reflected ends), golden from nxr's code (`gen_taps_golden.py`);
  - the halo (a chunk with its context equals the whole recording, to the bit);
  - a 10 Hz cosine's analytic signal (amplitude within 2×10⁻⁴ of 1, phase step within 10⁻⁵ rad of ω/fs).
- **Tested, Morlet** (`test_morlet`, golden `gen_morlet_golden.py` from nxr's `morlet_gain`, `gain_taps`): the gain, the even and odd taps of two bands, and a band's analytic
  signal of a synthetic series by taps against nxr's FFT route (interior, within 2×10⁻³ of the peak amplitude); the unit peak at (lo + hi)/2 and the half maximum at ± (hi − lo)/2.
- **What it cannot show:** a band narrower than the record allows is truncated at 1,024 taps a side (its response then deviates from G). At a recording's
  end the block is reflected, so the first and last K samples are the reflection's, not the signal's.

## Spatial band power: of a field, and of a second-moment statistic (`compute/include/nxr/filterbank.h`)

**Why.** Before choosing a spatial scale, look at the spatial spectrum, as an FFT comes before choosing a frequency band. A graph filterbank (MexHat,
Heat, Itersine, the Meyer octave frame `weyl`/`dyadic`) gives each band; these measure how much of a quantity lives in each.
- **A field** (vertex values: a map, a source series): `analysis(bank, transform, F)` then `memberEnergy(W)` (Rheome's `vertexSpectrum`, an unweighted sum
  over vertices) or `memberEnergy(W, areas)`, AREA-WEIGHTED: with the mass the eigenbasis is orthonormal for, it equals the band's power in mode
  coefficients (Parseval), so a tight bank's bands add up to the field's power. Complex fields too (|W|²).
- **A second-moment statistic** (a cross-spectrum, a covariance, in mode coefficients: e.g. K_Φ S K_Φᵀ with K_Φ = `forward(t, kernel)`):
  `nxr::measurement::bandMoments(gains, C, withMatrices)` gives each band's power P_m = Σ_k g_m(λ_k)² C_kk and, on request, its matrix C_m = G_m C G_m; `bandPower(bank, λ,
  diag C)` the power alone. Anything else of a band (a coherency, the lag content ‖Im C_m‖ / ‖C_m‖) is read from C_m.
- **Tested** (`test_band_moments`, an ellipsoid with its full eigenbasis): the band power of c cᴴ equals the area-weighted energy of the filtered field to
  1e-14 (Meyer and MexHat); the Meyer frame's bands add up to tr C; a pattern with no lags gives real band matrices, a travelling one does not.

## Comparing vector currents: the data-block magnetic Laplacian, and the geometry transport as an option (`magnetic::blockMagneticLaplacian`, `compute/include/nxr/transport.h`, `algebra::blockCongruence`)

**Why.** A free estimate's current is a 3-vector a vertex; comparing two vertices' currents means relating two vectors. The GEOMETRY TRANSPORT carries
a vector as a current normal to the cortex would be carried along the surface: it ENCODES the prior that currents are normal. It is a named option for anyone who wants that model, not the default and not the direction route; direction analysis makes no
orientation assumption and uses the joint data-block magnetic Laplacian (`magnetic::blockMagneticLaplacian`, below).
- `algebra::blockCongruence(A, M, pairs, 3)`: the 3 x 3 blocks T_uv = K_u C K_vᵀ of a free-orientation kernel at the pairs.
- `transport::leviCivita(normals, edges)`: the minimal rotation taking v's normal to u's (no gauge; holonomy where the surface curves);
  `transport::frames(e1, e2, n, edges)`: compute's flat full-frame transport F_u F_vᵀ (path-independent; its tangential part follows the frames' e1).
  A named option (`transportOf`); the caller chooses.
- `transport::compare(T, R, …)`: per edge the transported cross-spectrum tr(T_uv R_uvᵀ), its lag and strength.
- `magnetic::transportLaplacian(n, edges, w, θ, R, normalise)`: L = D − W with blocks W_uv = w e^{iθ} R_uv (3n x 3n, Hermitian, PSD): the connection
  Laplacian whose transport is the surface's rotation times the measured lag (the normal-current model).
- `magnetic::blockMagneticLaplacian(n, edges, blocks, normalise)`: the same operator with ARBITRARY complex b x b blocks W_uv (W_vu = W_uvᴴ), degree
  Σ_v ‖W_uv‖₂ (PSD for any blocks); b = 1 is the scalar magnetic Laplacian, blocks w e^{iθ} R the transported one. With each edge's normalised 3 x 3
  cross-spectrum block it is the JOINT DATA-BLOCK route: one complex 3-vector a vertex, orientation and lag fitted together from the data, one global
  phase free.
- **Tested** (`test_transport`): on a folded sheet (walls up to 72°) carrying a normal current with a travelling lag, the world-frame comparison is off by
  π for 3,215 of 8,440 pairs, the transported one exact (2e-16). On that developable sheet the operator has three null vectors (the three parallel fields)
  whose normal components all carry the lags; on a SPHERE (curved: the tangential fields have holonomy, the normal line is carried onto itself) its lowest
  mode is unique and is the current, along the normal (3e-12) with its lags (8e-13).
- **What the transport does not decide:** where the surface is flat, the vector's direction among parallel fields. The data-block route takes the
  direction from the data instead.

## Collapsing a region to one signal (removed)

`region.h` (`region::collapse`, the `pca_flip` weights of a region's elements) had no caller beyond its test and was removed (2026-11-18, refactor row 29); it is in the git history before that
commit. A region-level signal is a LINEAR spatial map applied to the field (a member of the chain's `members` stage); the PCA weights are a statistic of the region's covariance block.
`integrals.h` (`flux`, `circulation`, `vertexAreas`) is replaced by `operators/weak_forms.h` (`weakDivergence`, `weakCurl`: the pointwise, per-vertex weak forms of nxr's `measure.integrals`),
summed over a tile by `nxr::measurement::reduce` (measure sum), the tile's area being the sum of `massMatrix(Barycentric)`; the golden of `test_integrals` is nxr's.

## Diagnosing lags: their noise, where they fail to close, the distance they are read at, a surrogate (`compute/include/nxr/lag.h`; `timebank::equivalentBandwidth`)

**Why.** A lag is only worth mapping where it stands above its own noise, and at the scale where it lives. These tools say both, for any pairs (mesh edges,
pairs at a distance, a filterbank band's matrix).
- `lagStatistics(S, P_u, P_v, n)`: per pair the lag θ, the coherence |γ|, the lag's standard error se(θ) ≈ √((1 − |γ|²) / (2n|γ|²)) (Bendat & Piersol) and
  |θ| / se; option `debias` for |γ|² at small n.
- `timefilterbank::equivalentBandwidth(bank, member) (= nxr::stats::statisticalBandwidth of g²)` (the statistical bandwidth (∫g²)² / ∫g⁴) and `independentEstimates(bank, member, samples)`, n = B·T: the
  number of independent complex estimates a band gives over a window. **Checked against a simulation** (`test_lag`: two channels one sample apart through
  the Meyer alpha band, 500 trials of 4 s): the spread of the estimated lag equals se with n = B·T (ratio 1.10); n = 2BT would understate the noise by √2.
- `holonomy(edges, θ, faces)`: per face the wrapped sum of its edge lags: 0 where the lags close (they come from one phase map, vortices included),
  non-zero where they are frustrated.
- `pairsAtDistance(positions, edges, d, tolerance, perVertex, metric)`: pairs at a distance (straight, or along the edges); a lag against distance is their
  lags summarised over distance bins by the measurement system.
- `circularShift(Z, shifts)`: each channel's coefficients rotated in time: its spectrum kept, the lags between channels destroyed: the surrogate of a null.

## A band's coefficients on a chunk: the filterbank transform of a haloed block (`nxr::signal::wtBlock`, `haloOf`)

**Why.** The FFT route to a band's complex signal: the chunk (with its halo, from io's chunk service) Fourier transformed, the band's gain put on the bins,
transformed back; any number of members at once, so a time–frequency cross-spectrum or an average over frequencies is the same call. Every transform returns its coefficients at the input's rate: it never decimates, resamples or windows its input (`wt(bank, member, x, at)` reads one member's analytic signal at instants the caller gives, exactly). The bank is the
caller's `timebank::Design`: the Signal chain's single Meyer band (`bands = {{8, 12}}`, the Studio's `meyerBand`, not the tight frame: alpha's edges are
closer than the frame allows), Morlet centres, or any of the frame families. The bank is designed on the block's own grid (zero-padded, no wrap) and only
the own columns are kept. **The halo** (`haloOf(bank, member)`) is the member's response length, as the Studio's device chain sizes its HALO by the band's
taps (1e-6 of the energy left out): alpha [8, 12] Hz at 1000 Hz needs 783 samples, about 6 periods (the Studio's CPU rule of 3 periods leaves 1 % at the
chunk edges). Tested (`test_cross_spectrum`): chunks of 4096 with that halo equal the whole record transformed at once to 2.5e-4; the FFT route equals the
taps route (the same gain's even and odd taps) to 5e-4; a Morlet member returns a tone's amplitude. Where a recording ENDS the block has no context and
the transform zero-pads (the taps route reflects there).

## The phase of a complex field (`compute/include/nxr/phase.h`; pointwise)

**Why.** How the timing of an oscillation moves across the cortex: the phase gradient, the phase velocity, and whether a pattern travels or stands.
- **The reference:** nxr's phase reader (`nxr/detect/core_phase.py`): ∇φ = Im(z̄∇z)/|z|², with no unwrapping; u = −ω∇φ/|∇φ|². The mesh gradient is
  its GV, the P1 face gradients averaged by area onto the vertices.
- **Pointwise (the cardinal rule):** per vertex, per sample; nothing is summed across samples or vertices.
  - `power` gives |z|².
  - `phaseCurrent` gives j = Im(z̄∇z), the phase current |z|²∇φ, on the tangent plane.
  - `gradientPower` gives |∇z|².
  - `phaseGradient` gives j/|z|².
  - `phaseVelocity` gives the velocity and its length, the speed.
  - `travellingIndex` gives |j|/√(|z|²|∇z|²): 1 for one plane wave, 0 for a standing pattern.
  - For one frame these are that moment's maps.
  - **Over a span or a region** the phase gradient is a reduction: measure j and |z|² over a partition (time and/or space), and the ratio is a
    derived measure at read.
- **Vectors:** a field may hold 1 or 3 complex components a vertex. With 3 (a free-orientation current) the formulas sum over them
  (Im(Jᴴ∇J), |J|², |∇J|²): the phase of the vector as a whole, independent of the axes.
- **Tested** (`test_phase`):
  - against nxr's `core_phase`, to rounding (golden from nxr's code, `gen_phase_golden.py`);
  - on a sphere, a wave round the axis: speed within 1.3 % (median), direction within 1.4°, travelling index 1.000; a standing wave: 0.000;
  - a vector whose orientation turns over the sphere: the scalar's phase gradient to 1.1 % (median);
  - pointwise (a frame alone equals it among others); the current tangent.
- **What it cannot show:**
  - the P1 gradient resolves a wavelength of a few edges at best;
  - the phase is singular where |z| = 0 (spiral cores, the poles of a wave round an axis): NaN there;
  - for a vector field the discrete product rule lets a turning orientation leak about 1 % into the phase gradient on a 642-vertex sphere.

## The magnetic Laplacian and its eigenmodes (`compute/include/nxr/magnetic.h`, `congruence.h`; spec `docs/specs/2026-11-12-magnetic-laplacian.md`)

**Why.** Instantaneous phase maps cannot be averaged (every vertex's phase goes round the circle); the lags between neighbours can. Averaged lags need
not fit together, so one map is SOLVED from them: x_u ≈ e^{iθ_uv} x_v on every edge, relaxed to Σ|x_u|² = 1, is the lowest eigenvector of the magnetic
Laplacian (Singer, 2011: angular synchronisation). Its phase is a latency map, its magnitude where the consensus holds, its zeros the vortex cores,
and λ₁ the frustration: the misfit no single map removes. The pointwise phase gradient stays the tool for a single instant.

- `magneticLaplacian(n, edges, W, normalise)`: L = D − W from the edges' complex weights W_uv (Hermitian; D_uu = Σ|W_uv|); `normalise` none or
  symmetric (Singer's). The Laplacian does not know what W is made of: that is the caller's.
- `hermitianEigs(L, k)`: the lowest k eigenpairs, Spectra 1.2.0's `HermEigsSolver` with shift-invert through Eigen's complex LDLᵀ, then the clusters
  CLOSED: what was found is deflated (Woodbury on the same factorisation) and the solve repeated until no eigenvalue below the k-th turns up (Lanczos
  misses partners of exactly degenerate clusters: on the icosphere's graph, without it, the lowest 25 are wrong by 0.53; with it they equal the dense
  oracle to 2e-14). Each eigenvector of unit norm, its largest component real and positive. `hermitianEigsDense` is the test oracle.
- `winding(x, faces)`: the winding number of arg x round each face (±1 at a vortex core).
- `nxr::core::congruence(A, M, pairs)`: A M Aᴴ at the pairs asked for only (general algebra; a source cross-spectrum on the mesh's edges is K C Kᴴ
  on the edge pattern).
- **Tested** (`test_magnetic`, synthetic): a ring with flux (exact spectrum, degenerate pairs included); the icosphere's graph clusters against the dense
  oracle; a random graph with random lags against scipy's `eigsh` (golden `compute/test/golden/gen_magnetic_golden.py`: within 7e-15, eigenvectors
  equal up to their phase); on an icosphere, a plane wave's own lags fitted exactly (λ₁ ≈ 0, the mode's lags the wave's), a rotating wave fitted with its
  two cores found by the winding (+1 north, −1 south), and λ₁ growing with noise on the lags.
- **What it cannot resolve:** the map is defined up to a global rotation (only phase differences mean anything); lags over half a cycle between
  neighbours alias; where |x| is small the phase is undetermined; λ₁ compares only on the same graph and options.
- **Not built yet:** how W is made from data, the steps, storage, the time route.

## The cortical coordinate system: frames and tiles (`compute/include/nxr/frames.h`, `ladder.h`; audit `docs/audits/2026-11-11-cortical-coordinates.md`)

Directions and areas on the cortex need a coordinate system shared across people. Two of its steps are here:

- `run surface_frames`: the **canonical frames** (north / west from the registered sphere `<surface>_sphere`; Rheome's sphereframe), and the **anatomical frame** (one angle per vertex from north
  to anterior, the second axis superior or medial, and a reliability mask stored as a Selection of the reliable vertices). The anatomical axes come from `--template` through the registered
  spheres, or else from the surface's own SCS axes projected on its tangent planes.
- `run select_dyadic_surface`: the **dyadic tile grid** (the maker node `select.dyadic_surface`: compute's dyadic builder, stored through io's `save_grid`; the graph and the Studio's side by side are in `docs/specs/2026-11-17-tile-grids.md`; the Studio's `surfaceLadder`, nxr's `geom.ladder`; it reads the surface's positions as stored, float64, as the Studio's backend and nxr
  do: rounded to float32 first, a few vertices of the finer levels moved, 169 of fsaverage5's at level 11, fixed 2026-11-11): 2^l equal-area tiles per connected piece at level l, paired with spatial scale by Weyl's law,
  stored as Selections with their areas, centres and adjacency: the grid's hierarchy is the closed form (the member number is the finest tile code, level l's member that code `>> (D - l)`), and its record says `grid {kind dyadic, depth, slots, pieces}`: `slots` the piece slots the codes reserve (at least 2, so a one-piece surface still has codes `h 2^D + t` with h < 2), `pieces` the connected pieces. `run flow_modes` and `run quadrature` take `--tiles` to write their maps' tile summaries per level: scalars
  area-weighted, vectors in the canonical frame's components and in the Levi-Civita (transported) form, labelled, never raw 3-D sums.
  **A scalar map's summary is measurement rows** (the measurement system, 2026-11-14 and the consolidation of 2026-11-17): per level and map, nxr's `<node>/maps/<field>/{w,s1,s2,n,min,max}` arrays
  and the rows `weight`, `sum`, `sum_square`, `count`, `min`, `max` of the map's Field on the level's Selection (its `params_json.atlas` block makes it a partition the lookup reads); the mean (sum / weight),
  mean_square and variance are **derived at read** (`measure query ... --measure mean`, which also rolls the rows up the ladder); `mean` is no longer stored (`test_tiles_pipeline`: the mean at read
  equals the direct area-weighted mean to 0 at level 2 and rolled up to level 1). A map of a **statistic's slice** (the `quadrature` energy maps) gets a **byte-less Field** `<slice> tile map` (kind `tile map`, no array, `producer_json` naming the statistic and the slice; the pattern of the source estimate and `phasegradient`), so its rows satisfy the schema's array-backed rule and are the same rows as any map's; no `mean` is stored (decided 2026-11-14, `test_quadrature_pipeline`). The flow's divergence, curl and speed are not summarised per tile here (they are leaf sums, rolled up at read).

The charts the frames are built on (heat distance, vector heat transport, log map, gauge heading, geodesic disc) are in `compute/include/nxr/chart.h`: `ChartKit` is a thin wrapper of geometry-central's `HeatMethodDistanceSolver` and `VectorHeatMethodSolver`, one `IntrinsicSurface` per connected piece, in geometry-central's vertex gauge (the one gauge of nxr-neuro). The canonical frame (north / west) is stored as the angle of north in that gauge (`frames::gaugeAngle`); the anatomical axes are a reporting projection of a 3D vector (`frames::anatomicalProjection`, frame-free for the SCS definition).

## The speed spectrum (`compute/include/nxr/speedspectrum.h`, `jointbank.h`; `run speedspectrum`)

**Stored (2026-11-13) as STATISTICS** by `welch`, one per mode kind and part: `<stem>_<kind>_<part>` (energy and histogram `power`, tile_speed `speed`,
quadrature `quadrature_fraction`) on the axes graph member × time member (the histogram: × speed bin), from the Welch segment partition.

Energy of cortical activity per phase speed. The Hodge–Dirac modes' power per frequency bin (Welch, through a free-orientation kernel) is tiled by the joint filterbank: a graph bank over the
eigenvalues (`--graph-family`, no default) x a time bank over frequency (`--family`, `--gain`, `--sides`, all required). Each tile is read as energy at phase speed c with 2πf = cμ, per spatial scale,
exact and coexact modes apart, with each tile's quadrature fraction (travelling versus standing). `--speed-bins`, `--speed-min`, `--speed-max` set the speed axis.

## The Hodge–Dirac mode solve: the reference method

nxr-neuro solves the Hodge–Dirac modes **exactly as nxr and Rheome do** (`nxr.eigen.hodge_dirac.solve_hodge`, `nxr.pipelines.hodge.solved_package`; `rheome.eigen.hodge_dirac`,
`hodge_package`), by default:

- two ladders: exact Δ0 = d0ᵀ⋆1d0 against ⋆0, coexact Δ2 = (⋆2d1)⋆1⁻¹(⋆2d1)ᵀ against ⋆2;
- `want = modes/2 + 1`; each ladder asks for `min(want + 1, n − 2)` pairs (n = nV for Δ0, nF for Δ2);
- ONE shift-invert solve per ladder at the **absolute σ = −1e-3**, generalised symmetric (Spectra, tolerance 1e-14 ~ ARPACK's machine precision, `ncv = 2k+1`, all pairs must converge),
  no re-orthonormalisation; ascending, λ = max(λ, 0);
- per ladder the constants out (λ > 1e-6 · max λ), then ONE λ cut (the smaller of the two tops), μ = √λ, merged by μ (stable; exact before coexact on a tie), kinds 0 exact · 2 coexact,
  modes with μ ≤ 1e-6 · max μ left out; the result holds *about* `modes` modes (everything under the cut);
- the package: ψ/√2 (exact), W·χ/√2 (coexact; W = the ⋆2-weighted mean of χ over the faces round the vertex), stored as float32.

**A cortex is glued by anatomy first**, as nxr's `solve` and the Studio glue one. A surface of two closed pieces
(the hemispheres) has each MEDIAL WALL (the vertices its stored Desikan-Killiany atlas codes `unassigned`) cut out as a disk, and a BRIDGE of new vertices grown between the two rims
(nxr-compute's `glueMesh`, a line port of the Studio's `glue.ts`, bit-exact with its golden). The solve runs on the glued surface, and the package is stored as nxr stores one:

- `source/<surface>_glued`: the glued surface; vertex i < n_kept is `<stem>_support[i]` (the kept vertices, ascending), the rest the bridge;
- `source/<surface>_glued_bridge`: the bridge's vertices (not cortex: no sources, a forward zero there);
- `<stem>_eigenvectors`: [glued vertices × K], onto the glued surface; `<stem>_support`: the input's kept vertices.

**The full forms are stored with the package**. The modes are solved on the intrinsic Delaunay triangulation: the same vertices, its own
edges (lengths along the surface) and faces, and on the cortex some pairs of vertices are joined by two different edges, so it is not a mesh of vertex
triples. Stored as its own manifold `source/<surface>[_glued]_intrinsic` (edges and faces as its cells), with d0 and d1 as sparse operators between its
cells, the edge lengths, the stars ⋆0, ⋆1, ⋆2 and the signposts as fields on its cells, and each mode's 2-form part χ/√2 on its faces
(`hodgeDirac_forms2`, axis → faces). With the 0-form parts (the eigenvectors, ψ/√2) both 1-form parts follow exactly, and they are orthonormal in ⋆1:
reading a 1-form on the intrinsic edges into the modes is a set of inner products (tested: the 1-form Gram matrix is I/2 to 1e-7). **Why:** the modes
live on that complex; reading or synthesising fields with the input mesh's operators mixes two triangulations (the readings it gave were not
orthonormal and, on rough fields, not projections).

The medial wall is part of the cache key (another atlas, another glued surface). A surface with no Desikan-Killiany atlas is refused by name; a surface of one piece (a mesh, a sphere) is solved as
it is. **What the glue cannot resolve:** the bridge is a construction, not tissue. Modes and flows defined on the glued surface continue across it, so a pattern read on the bridge, or a
flow passing through it, is the basis's, not the cortex's: mask the bridge when reading cortex (the selection above).

The methods used before are **named options, never defaults**, the same names in the library (`HodgeSolveOptions`, `nxr/operators/eigen.h`) and the command:

| option | values | |
|---|---|---|
| `--shift` (`shift`) | `absolute` (default; `--sigma`, default −1e-3) · `relative` | `relative` = `eigenSmallest`: σ = −1e-7·λmax, then −1e-4·λmax, then a Cholesky last resort. One such solve can lose members of a multiplet (measured on the 642-vertex sphere); the absolute shift does not |
| `--slice N` (`slice`) | none (default) · window size | spectrum slicing (`eigenSliced`); the windows use their own shifts |
| `--normalize` (`normalize`) | off (default) · on | Cholesky whitening afterwards |
| `--star1-floor F` (`star1Floor`) | off (default) · F > 0 | **nxr has no ⋆1 floor and is the reference: ⋆1 is the raw intrinsic cotan weight.** Rheome's old default of 1e-6 is being changed to match. With a floor, its value and the number of clamped edges are recorded |

| `--mollify D` (`mollify`) | 1e-6 (default) | the mollification margin δ = D × the mean edge length of the intrinsic Delaunay construction (Sharp & Crane 2020; ε = max over corners of l_C − l_A − l_B + δ, added to every edge if positive). The shared name of the Studio, nxr and Rheome |
| `--flip-eps E` (`flipEps`) | 1e-6 (default) | an edge is flipped while its cotan weight is below −E. Shared name (nxr: `--flip-eps`) |

**One Delaunay construction.** The triangulation the solve works on is made by THE construction every operator in `compute/` uses (`compute/include/nxr/delaunay.h`), a line port of
the Studio's `intrinsicDelaunay.ts` (the reference; nxr's `intrinsic_delaunay.py` and Rheome's are ports of the same file): mollify; the mixed-Voronoi mass of the mollified input recorded before the first
flip; FIFO flips; signposts carried through the flips. The pipeline therefore mollifies exactly as nxr does (it did not before), and `mollify`, `flipEps`, the ε added and the flip count are recorded in the
provenance (`producer_json.solve.delaunay`) and in the cache key. In `compute/` the intrinsic mesh with the mixed-Voronoi mass is the DEFAULT of every operator (cotan Laplacian, DEC, connection Laplacian,
Hodge–Dirac); the embedded mesh (`intrinsicDelaunay: false`) and the Galerkin and barycentric masses are named options (the old `lumped`, A/3 per vertex, is now `barycentric`; no alias).

Every option is recorded in the stored result's provenance (`producer_json.solve.options`, `star1_clamped`, `lambda_top`, `tolerance`) and is part of the cache key: another option, another
cached result. Parity with nxr on the same mesh: `io/test/parity/test_hodge_reference_parity.py` (numbers in the pull request). The Delaunay construction itself is tested against the Studio's `intrinsicDelaunay.ts` and nxr's `prepare` by
`compute/test/parity/test_delaunay_parity.py` (stretched sphere, exactly cocircular lattice with the flip order deciding, slivers needing mollification; faces after the flips, twins, lengths, mass, flip count, signposts). The general Laplace–Beltrami solve (`eigenOperator(LaplacianCotan)`)
is not changed: Rheome's `eigen.modes` is a different method and the pipeline has no LB step.

## Tests

`ctest` runs the C++ tests of `compute`, `io`, `meeg` and `pipelines` (`ctest -R meeg` for the source-imaging tests, parity goldens included). The parity tests (`io/test/parity/`, pytest, run with nxr's environment; synthetic data only) compare the C++ with nxr's Python:

    export NXR=<path to the nxr checkout> NXR_NEURO=$PWD/build/cli/nxr-neuro NXR_NEURO_SCENARIO=$PWD/build/io/nxr-neuro-scenario
    uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> \
        io/test/parity/test_parity.py io/test/parity/test_write_parity.py io/test/parity/test_pipelines_parity.py

The importer suites also need `NXR_NEURO_BST_DUMP=$PWD/build/io/nxr-neuro-bst-dump` (see the headers of the files).

The Delaunay parity test (needs the Studio's checkout for its `tsx`, and nxr with `prepare(mollify=, flip_eps=)`):

    export NXR_CORTICAL_FLOW=<nxr-cortical-flow checkout> NXR=<nxr checkout> NXR_NEURO_BUILD=$PWD/build
    uv run --project $NXR python -m pytest -q --basetemp=<a scratch folder> compute/test/parity/test_delaunay_parity.py

## Design documents

| | |
|---|---|
| `docs/specs/2026-11-10-neuro-forward-inverse.md` | head model, noise covariance, minimum-norm kernel (`meeg/`) |
| `docs/specs/2026-11-11-default-subject.md` | the default subject, step 1: FreeSurfer surfaces and the imported Hodge–Dirac modes |
| `docs/specs/2026-11-10-filterbanks.md` | FFT, graph and time filterbanks; the Rheome / nxr / Studio comparison |
| `docs/audits/2026-11-10-operators-eigenmodes.md` | the operators and the eigenmode solve against nxr and Rheome |
| `docs/audits/2026-11-10-rheome-survey.md` | what Rheome has and what was brought here |
| `docs/audits/2026-11-11-cortical-coordinates.md` | frames, charts and the tile ladder |
| `docs/audits/2026-11-11-opticalflow-smoothness.md` | the optical-flow smoothness term (frame, ambient, connection) and a bug in Brainstorm's right-hand side |

Per library: `compute/README.md` and `compute/docs/`, `io/README.md` and `io/docs/`, `meeg/README.md`.

## The old repositories

`nxr-io` and `nxr-compute` are unchanged for now; the maintainers decide when to archive them. nxr-compute's open branch (`feat/wasm-geodesic-paths`) is to be brought in later with
`git subtree pull --prefix=compute git@github.com:neurodynamics-xr/nxr-compute.git <branch>`; until then a change made there is not here (its bindings now live in `bindings/`, so
a pulled change under `bindings/` needs moving by hand).

## Node traits: the settled classifications (2026-11-17)

Every registered node declares its time behaviour and precision (`ARCHITECTURE.md`, *Backends and node traits*); no node is `uncertain` any more (`test_node_traits` fails on one). What each was decided from:
- `synthesis.lagfree`: **windowed**. The noise is a function of the sample index, but each output sample is that noise filtered by the shaping taps (`taps_half` samples each side), made from the sample index: it asks no halo of its input, whose shape only it reads.
- `select.tile_crossings` (the crossings step): **windowed**, a window of two cores along a track; it runs once on the whole event table and carries no state between blocks.
- `extent` grow (hysteresis regions): **stateful**; the labelling is carried forward in time.
- `extent` roll-up and `rollup_events`: **windowed**; the window is the event's own cells or tiles (variable length).
- `summarize`: **stateful**; it needs every member of the partition.
- `fit_gate`, `burst_stages`: **pointwise over events**; events do not interact.
- `measure` (the median): **stateful**. A median is exactly independent of block order, but the node holds the samples, not a fixed-size sum, so a block-parallel executor must merge them.
Precision: nodes that compute in double on float32 input and give a field declare `follows_input` and store float32 (their notes say so); only statistics and event tables declare float64 output. Where the stored dtype of a statistic is written (the runner) is not changed by this batch.
