# Dataset-wide runs: few files, staging and merge, the group scope (spec, 2026-11-14)

## Why

nxr-neuro will run over whole datasets on HPC nodes (later GPUs), and group statistics come after. Three things stand in the way:
- **Too many files.** One real subject is thousands of files and folders for a few GB: a `zarr.json` for each array, and most files under 1 MB holding a small share of the bytes. A cluster file system pays a metadata round-trip per file and per folder. Apart from chunking or sharding a large
  array, there is no reason for a file.
- **One SQLite database per dataset.** Many jobs writing one SQLite file on a shared file system is unsafe (its locking is not reliable there) and
  serial (the writer lock). The database write was among the slowest steps of cluster runs.
- **No group scope.** Every definition is `scope: subject`; between-subject statistics have no place to run.

What exists and is kept: the dataset database as the system of record; each node's `zarr.json` (the array's description, which every zarr reader
needs, and the drained copy of its row, which lets the store rebuild the database: D141); one writer per dataset (`.writer.lock`); nodes written
`pending` then `complete`; UUIDv7 ids; the default subject (fsaverage5) with the group tile tree. a job runner can already write one-subject datastores per job
and join them, side by side only: it cannot join subjects into one dataset, because each carries its own
dictionaries under fresh ids.

## 0. Measure first

One canary job on the cluster, timed per phase: compute, row writes, drain, `zarr.json` writes, array writes, archive. The fixes below are
ordered by what it shows; nothing is rebuilt before it.

## 1. Few files: a file only for a large array's shard

**The rule:** a file exists only for a shard of an array large enough to need one. Everything else of a subject lives in ONE small-file store (one file).

- **The subject's small-file store:** a key-value table in one SQLite file per subject (`<subject>.nxr.zarr/small_files.sqlite`), holding zarr keys: every `zarr.json` (groups
  and arrays) and every chunk of a small array (under a stated size, default 1 MB a chunk). It is still zarr: the keys and bytes are exactly what the
  folder tree holds; only where they live changes.
  - Per subject, so a subject stays self-contained (copied, archived and merged as one unit) and the dataset database stays small.
  - Not transactional with the dataset database, and it need not be: files are not today either. The node protocol already covers it (rows
    `pending` → keys written → `complete`; recovery removes a pending node's keys, by key prefix instead of `remove_all`). An ATTACH across the two is
    not used: SQLite commits attached databases atomically only in rollback-journal mode, and the dataset database runs in WAL.
  - Folders disappear with the files: a key's path is a string in the table.
- **Large arrays:** sharded, shards of a stated target size (default 64 MB), one file each. On the fixture: 3.5 GB in about 60-100 files instead of
  1 223 (1 207 of them are 1-8 MB chunk files of unsharded arrays; only 8 arrays are sharded today).

**What the survey found (2026-11-14, read-only; fixture subject: 7 461 files, 8 282 folders; 2 995 `zarr.json`; 2 901 arrays, 2 857 of them a single
chunk file):** no implementation has a store abstraction. nxr-neuro's io works on `std::filesystem` paths throughout (array reader and writer, the drain,
node creation, recovery, deletes, `Populate` callbacks given a folder, the CLI's `read`, the WASM binding); nxr opens arrays by path (zarr-python 3.4.2,
`LocalStore`); the Studio reads through zarrita's pluggable `Readable` but writes `zarr.json`, chunks, renames and deletes with `node:fs` directly. No
existing SQLite or zip store to port; the Studio's `Readable` (get, ranged get) is the interface to follow. Sharded reads need ranged gets.

**So the work is phased:**
- **A. A store interface in each implementation, no format change** (nxr-neuro io, its CLI and WASM binding built 2026-11-14: `io/include/nxr/io/store.h`, README *The store*; nxr and the Studio are phase C): get, ranged get, set, delete by prefix, list, exists; the folder tree as its first
  backend; every call site moved behind it (io first, then the WASM binding and `cli/read_command.cpp`). Every existing test stays green and every
  stored byte is unchanged.
- **B. The new format version (BUILT 2026-11-15: format 52; README *Format 52*):** the small-file store backend (`SmallFileStore`, `DatastoreStore`: one SQLite
  file per subject, WAL while open and one file at rest, small values under 1 MiB as rows, larger as files), large arrays sharded at the target size (format 51's
  default, unchanged), `dataset compact` (51 to 52: moved in verified batches, resumable), recovery and deletes by key prefix, the `_replaced-<date>` move as a
  key-prefix rename. `dataset export` (a plain zarr folder or zip for tools outside nxr) is still to build. A new dataset is format 52; the SQLite form was approved.
  Synthetic subject: 59 files in 30 folders at 51, 3 files in 3 folders at 52. nxr and the Studio cannot read 52 until phase C.
- **C. nxr and the Studio** take the same interface and backend (a zarr-python `Store` subclass; a zarrita `Readable` plus the Studio's writers in
  `db/sync.ts`, `crud.ts`, `float16-write.ts`, `db/import.ts`, `service/model.ts`, `list.ts`), with parity tests on the same synthetic datasets and the
  canonical `zarr.json` text (D145) kept byte-identical in all three.
- **A new version of the nxr format.** This is the nxr format itself, not a convention of one program: every implementation of it reads and writes the
  new layout (nxr-neuro's io and its WASM module, nxr in Python, the Studio), each through the same store (`get(key)`: the small-file store, then the
  file). The schema version rises; a reader refuses a version it does not know, as today.
- **For tools outside nxr:** `dataset export --folder|--zip` writes a plain zarr folder or an uncompressed zip.
- **Existing datasets:** `dataset compact` moves a subject's small keys into its small-file store and re-shards its large arrays, verifies every key byte for byte,
  and removes the old files only with the maintainers' OK (never delete data without it).
- **Expected:** a subject from ~15 700 files and folders to ~100 files.
- **No compression twice:** arrays are already blosc/zstd; archives are uncompressed (tar or zip). Gzip, single-threaded over compressed data, is
  dropped.

## 2. Staging and merge (dataset-wide runs)

1. **A job reads** a read-only copy of the dataset database (megabytes) and the arrays in place, read-only.
2. **A job writes** a staging dataset of its own (the existing writer pointed elsewhere): its database and small-file stores on the node's local disk
   (`$SLURM_TMPDIR`; `synchronous=OFF`, journal in memory: a crashed staging dataset is rerun), its large shards straight to their final place (each
   subject's arrays are its own folder, so jobs on different subjects never touch the same files).
3. **`dataset merge --into D --from STAGING…`** (io): under the target's lock, one transaction per staging dataset: its complete nodes' rows and small-file store
   keys, statistics before what they name (recovery's order), dictionary ids mapped onto the target's (what `merge_nxr.py` could not do), one drain at
   the end. Arrays are not copied.
   - Same cache key: identical, skipped (a merge can be rerun). Same name, another key: refused unless `--replace`. A failed job leaves its staging
     dataset and nothing in the target.
   - Port `merge_nxr.py`'s checks (schema version, a live writer lock, duplicate ids) before writing any of its own.
4. **The job runner** runs one job per recording or subject, then the merge as its finalize step.

Staging changes nothing on a workstation or in the Studio: a single process still writes directly.

## 3. The group scope

Map, then reduce, in one dataset:

| phase | runs as | writes |
|---|---|---|
| per subject | parallel jobs, staged and merged | the subject's results, and the same results carried to the template |
| group | one job after the merge | group statistics on the default subject |

- **`scope: dataset`** in the runner: inputs like "each subject's home map on the template"; the subjects are a cohort Selection named by the caller,
  never all by default.
- **Carried to the template in the per-subject phase:** scalars resampled through the registration (`sphere.reg`); vectors also rotated into the
  template's frames (the group layer of the vortex-cores spec; nxr-neuro needs this resampling step).
- **Each subject's own null stored** in the per-subject phase, subtracted before any group map.
- **Group statistics** (statistic nodes on `@default_subject`): the group map G = |Σ m_s| / Σ |m_s| per vertex, sign-flip or random-rotation nulls,
  BH over vertices or tiles, covariates (PET, age) from the dataset.
- The group phase is light (subjects × ~20 k template vertices): one CPU job.

## Order

0 (the canary measurement) → 1 (the small-file store and sharding: io, then nxr and the Studio, then `dataset compact`) → 2 (staging and merge) → 3 (the
group scope). Within a job, the CPU threading of the spatial-events chain (threaded kernel product, frame blocks in parallel with an ordered fold, a
float32 operator) is separate work, before any GPU backend.

## Open decisions

**Decided:** the nxr format moves to about 100 files a subject .

1. **The small-file store's form:** an SQLite key-value file per subject (recommended, taken unless the maintainers say otherwise: transactional with the rows,
   updates and deletes in place, SQLite is already in every nxr implementation), or an uncompressed zip (append-only: an update or delete rewrites it).
2. **Thresholds:** small chunk 1 MB, shard 64 MB (to be tuned by the canary).
3. **Every implementation follows the new format version:** nxr-neuro first (io, then its WASM module), then nxr (Python) and the Studio, each with
   a parity test on the same synthetic datasets. Who ports nxr and the Studio, and when.

## Checklist (the architecture policy)

- Compute is untouched: the small-file store, the merge and the group scope are io and orchestration.
- Loops and file access stay in io (the store, the writer, the merge); the runner's dataset scope composes stored results.
- The caller names the inputs: the staging datasets to merge, the cohort, the template carry.
