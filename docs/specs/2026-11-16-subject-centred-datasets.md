# Subject-centred datasets: the subject as the unit, the dataset as a catalogue, templates as shared subjects (spec, 2026-11-16)

Status: **design, no code.** The maintainers approved the direction on 2026-11-16 (a subject-centred design). This spec says what the format becomes,
what replaces the staging and merge of `docs/specs/2026-11-14-dataset-wide-runs.md` section 2 (designed, never built), how existing datasets move, and
what is left for the maintainers to decide (section 10).

## 1. Why

Today an nxr dataset is ONE database (`<dataset>/dataset.sqlite`, the model's tables for every subject) beside the subjects' stores. Format 52
already moved each subject's small zarr keys into the subject's own `small_files.sqlite`; the rows did not move. Four things follow from that one
database:

- **HPC.** Every job that writes anything writes the one dataset database, under one writer lock. Many jobs on one SQLite file on a shared file
  system is unsafe (its locking is not reliable there) and serial. Staging and merge (a database per job, folded in afterwards) was designed to work
  around it; it keeps the shared database as the system of record and adds a merge step, an id remap and a failure mode to every cluster run.
- **Sharing.** A subject cannot be handed to someone, archived or moved without the database rows that describe it, which live in a file shared
  with every other subject. A cohort cannot be shared without the whole dataset.
- **Templates.** The default subject (fsaverage5: surfaces, `sphere.reg`, parcellations, eigenmodes, frames, the group tree) is written into every
  dataset that uses it: the same anatomy, rebuilt and stored again per dataset, with no way to say two datasets used the same one.
- **Provenance.** A group result names its member subjects by row id in the same database. Nothing records WHICH state of a member (or of the
  template) it was computed from, so a member changed later silently invalidates it.

Between-subject work needs only two things from a dataset: **the subjects registered to a common template** (their measurements carried to it, the
template's fibres and reference maps), and **grouping and filtering of subjects** for statistics between groups. Neither needs one database holding
every subject's rows. The default subject is already "just another subject" (`subject.kind = 'template'`); this spec makes that literal.

## 2. The subject unit: a subject package

A **subject package** is one folder, `<subject>.nxr.zarr/`, holding everything about one subject: its rows and its arrays. It can be copied,
archived, moved, shared and opened on its own.

```
<subject>.nxr.zarr/
  subject.sqlite        the subject's rows (the model's tables) AND its small zarr keys (the `small_files` table of format 52)
  .writer.lock          while a writer holds it (the writer lock's file, unchanged)
  <node>/.../c.0.0      a large array's shards, one file each (format 51/52's sharding, unchanged)
```

**What `subject.sqlite` holds** (store format 53):
- **The model's tables, from `model.sql` verbatim** (the schema the three implementations share byte for byte), holding only:
  - the subject's own `subject` row and every row that carries its `subject_id` (acquisition, topology, space, geometry, manifold, selection,
    field, field_contribution, operator, statistic, job, subject-scope dictionaries) and their child rows (`topology_cell`, `coordinate_system`,
    `geometry_dimension`, `selection_element`, `selection_extent`, `selection_measurement`, `selection_span`, `operator_factor`,
    `dictionary_entry`). This is exactly the row set nxr's `shared_dataset.fragment` selects for one subject (its `CHILD` table and its owner
    tables), ported;
  - its **home** `datastore` and `dataset` rows (where it was created: provenance, not membership; section 3);
  - copies of the **shared vocabulary** rows its rows name: app- and dataset-scope dictionaries with their entries (atlas levels, `frequency bands`,
    `recipes`), as nxr's fragment carries "the shared rows";
  - **anchor rows** of the template it is registered to (section 4.3): the few rows of another package its own rows point at.
- **`small_files(key, value)`**, format 52's table, unchanged: every `zarr.json` and every value under 1 MiB (`kSmallFileThreshold`).
- **`meta`**: `schema_version` (the model's `PRAGMA user_version`), `store_format` = 53, `built_by`, and the package's own record (section 2.3).
- **`key_digest(key, size_bytes, sha256)`** (new): the size and SHA-256 of every key's value, written by the store when the value is written
  (`set`, and a streamed writer's `close`, which hashes as it streams). It is what the content hash (section 2.3) and the migration's verification
  are computed from without re-reading the shards.

**Why one file for rows and small keys** (open question 1): the rows and the keys a node writes are then in ONE SQLite database, so a node's rows
and its small keys commit together (the dataset-wide-runs spec noted that an ATTACH across two databases is not atomic in WAL mode; one database
needs no ATTACH). A subject goes from 3 files (format 52: its database, its medium chunks, its shards) plus its share of the dataset database to 2 +
its shards. The node procedure (`pending` → keys → `complete`, recovery by key prefix) is unchanged.

### 2.1 How ids stay unique across subjects

- **Every id is a UUIDv7**, as today (`uuid7()`): a millisecond timestamp and 74 random bits. Packages written by independent jobs, on different
  machines, with no coordination, do not collide in practice. Nothing new is needed to make them unique; what is new is that nothing CHECKS it at
  write time, because no shared database exists. So:
- **Registration checks** (section 6): an id a package brings that another registered package already owns (other than the anchor rows of a shared
  template, which are the template's own ids by design) is refused, naming both packages. This is nxr's merge check ("an id this subject's rows bring
  must not be another subject's in the dataset"), ported.
- **Shared vocabulary rows** (an app-scope dictionary such as `Desikan-Killiany`, the `frequency bands` vocabulary) are copied into each package
  that names them, each copy with its own id. Across packages they are matched by their unique key (`scope, name`, and `dataset_id` for a dataset
  scope), exactly as nxr's merge matches `SHARED_KEYED` rows, and the cross-subject index (section 3.3) maps each copy's references onto one row. Two
  copies under one key with different entries are a conflict, reported by name and never merged silently. (Deterministic ids for these rows are the
  alternative: open question 4.)

### 2.2 The schema version, per subject

- A package carries its own versions: `PRAGMA user_version` (the model's schema version) and `meta.store_format` (53).
- **A reader refuses a package at a version it does not know**, by name, as today a reader refuses a dataset database (`schema_version` error). It
  never opens it partly.
- The model's rule that "a datastore is at one version, entirely" becomes: **a reader works on packages at versions it knows; an operation across
  subjects (an index build, a group analysis) refuses a cohort containing any package at a version it does not know**, naming each. A dataset may
  hold packages at two versions only while a migration is under way (section 8).

### 2.3 The package record and its content hash

`meta` holds the package's record, one JSON value (`package_json`): the subject's id, name and kind, the schema version and store format, the
template reference (section 4.3), and the **content hash**:

- **content hash** = SHA-256 over the sorted lines `<key> <size_bytes> <sha256>` of `key_digest`, followed by a canonical dump of the package's rows
  (every table, rows ordered by primary key, columns in the schema's order, values in the canonical text the drain uses for `zarr.json`). Rows that
  record time of writing (`modified_utc`, `job` rows) are included: a package rewritten is a different state.
- It is recomputed when a writer closes the package (from `key_digest`, so no shard is re-read) and stored with the time it was computed.
- `subject verify --package PATH` recomputes every key's digest from the bytes and the hash from them, and reports a mismatch by key.
- A group result, a catalogue entry and a template reference name a package by **id + content hash**, so a later change to the package is seen.

## 3. The dataset: a light catalogue

A dataset is a folder with a **catalogue** and (usually, not necessarily) its subjects' packages:

```
<datastore>/
  catalog.json                        the datastore's list of datasets (unchanged in role; each entry now names a catalogue)
  <dataset>/
    catalogue.sqlite                  the dataset: its subjects, its participants table, its template references
    .writer.lock                      held only while the catalogue itself is written (registration, participants), seconds
    index.sqlite                      a CACHE (section 3.3); deletable
    <subject>.nxr.zarr/               subject packages
```

### 3.1 What the catalogue holds

The catalogue is a SQLite file whose tables are, where the model already has them, **the model's own** (`model.sql`), so the three
implementations read it with code they already have:

| table | from | holds |
|---|---|---|
| `meta` | model | `catalogue_version`, `built_by` |
| `datastore`, `dataset` | model | the dataset's own row (id, name, path, source) |
| `subject` | model | one row per member: id, name, `path` (relative to the dataset folder, or a location outside it: section 3.2), kind (`subject`, `template`, `group`), `template_id`, provenance columns. A COPY of the package's own row, refreshed at registration |
| `subject_entry` | new | per member: `subject_id`, `schema_version`, `store_format`, `content_hash`, `registered_utc`, `registered_by`, `status` (`registered`, `missing`, `stale`, `replaced`) |
| `template_reference` | new | the templates the dataset's subjects are registered to: template name, version, the template package's subject id, content hash, a location hint (section 4) |
| `variable_dictionary`, `subject_variable` | model | **the participants table**: what the dataset records about its subjects (name, type, unit, levels: BIDS `participants.json`) and each subject's values (BIDS `participants.tsv`). The model has had these tables since its first version; nothing wrote them |

The catalogue holds **no** manifold, field, selection, operator or measurement row. It is small (kilobytes for hundreds of subjects) and is the
only file a job reads to find its inputs.

**Participants**: `participants import --dataset DATASTORE/NAME --from participants.tsv [--sidecar participants.json]` writes the variables (BIDS's
`participant_id` column matched to subject names; an unknown participant is refused by name, a subject without a row has no values: a missing value
is a missing row, as the model says); `participants export` writes the TSV and JSON back. Values are typed (`number`, `string`, `boolean`) as the
model's `variable_dictionary.type` says.

**Cohorts by filtering**: `--where` on any command that takes subjects (`--where "group = 'control' AND age >= 60"`, over the variables by name)
resolves, in io, to a list of subjects; the command then runs exactly as if those subjects had been named. The resolved list (ids and content hashes)
and the filter text both go into the result's provenance. There is no default cohort: a command across subjects is given `--subjects` or `--where`,
never "all" by default (`--subjects all` is an explicit choice, recorded as such).

### 3.2 Membership and location

- A subject is a member of a dataset when the catalogue has its entry. Its package is usually inside the dataset folder; a package elsewhere
  (another dataset's, a shared read-only location) can be listed by path. **A catalogue writes only packages inside its own folder**; one it lists from
  elsewhere is read-only to it (open question 5).
- The package's `dataset` row is its **home** (where it was made). Listing a subject in another catalogue does not change the package.

### 3.3 The cross-subject index is a cache

Some readers want SQL across subjects (the Studio's views over a dataset, `measure query` across a cohort). For them `dataset index --dataset
DATASTORE/NAME` builds `index.sqlite`: a database in the model's schema holding every registered package's rows and its templates' rows, in the
shape of today's `dataset.sqlite` (so every existing reader of a format-52 dataset database reads it unchanged).
- **Built from the packages, never written otherwise.** Its `meta` records the content hash of every package it was built from; a reader that finds
  a package whose hash differs (or a new member) rebuilds it (or the part for that package) before trusting it. Deleting it loses nothing.
- **Its build is nxr's merge without moving anything**: each package's rows folded in by subject, children with their parents, shared vocabulary rows
  matched by key and references remapped (nxr's `_merge_one`, ported), anchor rows taken once from the template itself.
- It is the place an id clash or a vocabulary conflict across packages shows up; the build reports them by package and does not guess.

## 4. Template subjects

### 4.1 What a template is

A template is a subject package of kind `template`: "just another subject with special items". It holds, each item with its provenance record (the
`origin` record: source, licence, attribution, being added separately to the format), whatever every registered subject is measured against:
- the surfaces (pial, white, inflated; the primary one) and the registration sphere (`<surface>_sphere`, FreeSurfer's `sphere.reg`);
- the parcellations (the `<surface>_atlases` selection set);
- the eigenmodes (the Hodge–Dirac basis, the glued surface, its bridge), the frames (canonical and anatomical);
- the group tree (the `ladder` Selection with `tree: "group"` and the atlas definitions) and its Levi-Civita connection;
- template fibres (`curve/<fibers>`, the bundle a subject without its own borrows: the structural-connectome spec);
- reference maps (fields on the template's cortex: receptor or tracer maps, cortical gradients), each a Field with its `origin`.

Nothing here is new: the default subject already stores all of the above but the fibres and maps, as nodes of a subject (default-subject spec).

### 4.2 Shared and versioned

- **A template is released**: `template release --package PATH --name NAME --version VERSION` freezes a template package: it computes the content
  hash, writes `name`, `version` and the hash into the package record, and from then on io opens it **read-only** (a released template is never
  written again; a change is a new version, a new package with a new subject id).
- **Where templates live**: in a templates folder (a datastore-level `templates/<name>/<version>/<name>.nxr.zarr`, or any folder the caller
  names). A dataset does not copy a template: its catalogue's `template_reference` names it by **name + version + content hash** with a location
  hint. io resolves it from the hint, then from the template folders the caller names (`--templates PATH`, repeatable); a package found with another
  hash is refused, by name and both hashes. Which template a dataset uses is the caller's choice at `dataset create --template NAME@VERSION`
  (recorded in the catalogue), never a default of the code.
- **Several datasets, one template**: two datasets registered to `fsaverage5@1.0` with the same hash used the same anatomy, the same group tree and
  the same eigenmodes, and that is checkable.
- **Building one** is what `dataset create --templates` does today (surfaces, parcellations, frames, group tree, `--modes`), retargeted:
  `template create --from FREESURFER_SUBJECTS --name NAME --version VERSION [...]` writes a template package with the same nodes, then `template
  release`.

### 4.3 How a subject records its registration

- **The reference**: the subject's `template_id` column (as today) holds the template package's subject id. The package record's
  `template_reference` adds the template's name, version and content hash, so the id is never ambiguous across versions.
- **Anchor rows**: a subject's rows point into its template (a registration operator's `from_manifold_id`, a group-tile partition's
  `dictionary_id`, a field carried to the template whose `manifold_id` is the template's pial). With the model's foreign keys enforced, those rows must
  exist in the subject's database. The package therefore carries **the template's rows its own rows reach, and their parents** (the template's
  `subject` row, the referenced manifolds with their topologies, geometries, coordinate systems and spaces, the dictionaries and entries), copied
  byte for byte from the released template. This is exactly the set the Studio's `bundleOf` already returns as "following the template"
  (`related_subjects`, the other subject's manifolds and their parts, `lineage_fields`, `lineage_operators`); no arrays are copied. Registration
  checks every anchor row against the template's own. (Alternatives: open question 2.)
- **The registration itself stays nodes of the subject**, as today: its registered sphere (`<surface>_sphere`), its group tree per subject
  (`<surface>_group_tiles`, with `sphere_match_median` and `sphere_match_max`), and anything carried to the template (resampled scalars, rotated
  vectors). Their `producer_json` names the template by name, version and hash.
- **Changing template version** is a re-registration: the subject's carried nodes are recomputed against the new version (new nodes, new cache keys;
  the template version is part of every such node's cache key). Old nodes keep naming the old version until removed.

## 5. Between-subject analysis

A group analysis is: **select the cohort through the catalogue → read each member's registered measurements from its package → read the template
from its package → write the group result.**

1. **Selection** (io): `--subjects NAME,NAME,...` or `--where FILTER` on the catalogue gives the members, each with its content hash. Refused: a
   member at an unknown version, a member whose package is missing or whose hash differs from its entry (`stale`: re-register first), members
   registered to different template versions (never pooled), a template as a member.
2. **Reading** (io): each member's rows are read from its own `subject.sqlite`; the template's rows and arrays from the template package. No
   cross-subject database is needed (the index is not read by an analysis; it is for browsing).
3. **`pyramid reduce`** keeps its sums, merge rules, refusals, memory bound and parity with nxr exactly as `docs/reference.md` describes them. What
   changes is where it reads (each member's package) and where it writes (below). Its provenance gains the members' content hashes and the template's
   name, version and hash.
4. **Where group results live: a group package** (open question 3; recommended). Today `pyramid reduce` writes into the default subject's store. A
   released template is read-only, and a group result is not part of an anatomy, so the result goes to a package of kind `group` (the model's own
   kind for "group averages and statistics": Brainstorm's `Group_analysis`) inside the dataset, registered in the catalogue like any member:
   - its fields are the same rows `pyramid reduce` writes today (`group <kind>`, `function = sum`, `field_contribution` rows, measurements on the
     template's tile partition, which the group package reaches through anchor rows as a subject does);
   - `field_contribution.subject_id` names a member that lives in another package: the contribution rows are the members' anchor rows' counterpart,
     recorded with the member's content hash in `producer_json`; the member's field rows are not copied;
   - one group package per analysis the caller names (`--into GROUP_NAME`), never a default one; a re-run replaces its group fields as today.
   - The group statistics (the group map, sign-flip and random-rotation nulls, covariates from the participants table) are statistic nodes of the
     same group package.

## 6. Writing on clusters

**One job writes one subject package, with no shared lock.**

- **The job** opens the subject's package (`--package PATH`, or `--dataset DATASTORE/NAME --subject NAME`, resolved through the catalogue by a read
  only), takes **the package's** writer lock (`<subject>.nxr.zarr/.writer.lock`, the existing lock file and its rules: same-host stale detection, the
  handshake), and writes rows and keys into `subject.sqlite` and its shards. It reads the catalogue and the template read-only and never writes them.
  Two jobs on two subjects share no file they write.
- **Where `subject.sqlite` is written on a cluster** is decided by the canary (dataset-wide-runs section 0): in place (one process, one host: SQLite's
  WAL is safe when every connection is on one host), or on the node's local disk and moved into the package when the job closes (one rename on the
  same file system, else copy, verify the digest, then replace). Either way the shards go straight to the package.
- **Creating a subject** on a cluster: `subject create --package PATH --template NAME@VERSION` makes a package without touching any catalogue.
- **Registering** replaces staging and merge: `dataset register --dataset DATASTORE/NAME --package PATH [PATH ...]` (or `--scan`, every package in
  the dataset folder), run once after the jobs (the workflow's last step), or by a job itself at its end. It takes the catalogue's writer lock for one
  short transaction per package (no array is read or moved; it reads the package record and checks rows) and:
  - checks: the schema version and store format are known; the package is complete (no `pending` node, no `.writer.lock` held); its anchor rows
    equal the template's; its ids clash with no other member's; its vocabulary copies agree with the dataset's by key;
  - writes or refreshes the `subject` and `subject_entry` rows (content hash, versions, time).
- **Conflict rules and idempotence**:
  - the same subject id with the same content hash: nothing to do (a rerun of `register` is a no-op);
  - the same subject id with a new hash: the entry is refreshed (the package is the authority on its own content), reported as `updated`;
  - another subject id under a name the catalogue holds: refused, unless `--replace`, which marks the old entry `replaced` and moves its package aside
    (`<subject>.nxr.zarr.replaced-<date>`), never deleting it;
  - a failed check registers nothing for that package and does not stop the others; the output names each package and its result.
- **The chunk-queue lock** (`.nxr-chunk-queue.lock`, today one per datastore, so parallel jobs on one datastore wait for each other) exists to keep
  one machine from running two chunk loops at once. Its scope becomes **the machine, not the storage**: a file the caller names (`--queue FILE`) on
  local disk, or none (on a cluster the scheduler already places one job per allocation). It is never a datastore-wide file on a shared file system
  (open question 8).

What becomes of the dataset-wide-runs spec: section 1 (few files, format 52) stands and is extended here; **section 2 (staging and merge) is
superseded** (no merge of rows is needed when no shared database exists); section 3 (the group scope) is kept with section 5 above.

## 7. Reading

The Studio's `bundleOf` (`backend/src/db/bundle.ts`), nxr's reader (`Dataset.bundle`) and nxr-neuro's `subject_bundle` (the reference's port, used by
the WebAssembly data client) issue the same SQL for a subject. **The SQL does not change.** What changes is the database it runs on.

- **The subject source.** `bundleOf` already reads through an interface, `BundleSource` (SQL plus the three top rows). The new source opens the
  package's `subject.sqlite` read-only, ATTACHes its template package's database read-only, and creates TEMP views with the model's table names, each
  the union of the subject's rows and the template's (the anchor rows, identical in both, taken once). SQLite resolves an unqualified table name in
  the temp schema first, so every existing query (`SELECT * FROM manifold WHERE subject_id = ?`, the template-following joins) runs unchanged and
  returns what it returns on a format-52 dataset database today. This is a port: one new source per implementation, no query rewritten.
  - nxr-neuro: `subject_bundle(const SubjectSource&, subject_id)` beside today's `(const Database&, ...)`; `SubjectSource::open(package_path,
    template_locations)` in io.
  - nxr: `nxr.io.reader` opens a catalogue and gives a member's package (`Dataset.subject(name)`), with the same source; `nxr.io.small_files`
    already reads the `small_files` table and now opens it in `subject.sqlite`.
  - the Studio: a `BundleSource` over the package (node:sqlite in the desktop app, SQLite-wasm in the served build).
- **Arrays**: a member's keys are read from its package (`SmallFileStore` over the package folder, as format 52); a template's arrays from the
  template package (the Studio's `Model.readFor` for a related subject resolves the template's package from the reference instead of
  `<dataset>/<template path>`).
- **The WebAssembly data client** (`wasmClient`): `status()` reads `catalog.json` and the catalogue; `bundle(storePath)` opens the package's source;
  `read`, `create`, `put`, `remove` act on the package's store. For a published dataset over HTTP this helps: one subject is one small SQLite file
  to fetch, not a dataset database.
- **Dataset-wide views** (the Studio's lists over all subjects, `measure query` over a cohort) read `index.sqlite` (section 3.3), rebuilt when stale.
- **Parity**: the bundle from a package equals, row for row, the bundle of the same subject from a format-52 dataset database (synthetic datasets,
  the WebAssembly parity script's checks extended to packages; the same test in nxr against the Studio).

## 8. Migration

Existing datasets (one `dataset.sqlite`, store format 50, 51 or 52) move to the new layout with **`dataset split --dataset DATASTORE/NAME`**, in
place, under the dataset's writer lock, **subject by subject, verified byte for byte, resumable, and reversible**, as `dataset compact` is:

1. **Format first**: a dataset below 52 is brought to 52 by `dataset compact` (unchanged), so every subject has its `small_files` table.
2. **The template**: the dataset's default subject becomes a template package in place (its rows copied into its own `subject.sqlite`), named
   `<dataset>-default@local` with its content hash, recorded in the new catalogue's `template_reference`. It is not released into a shared templates
   folder by the migration; replacing a dataset-local template with a shared released one (same anatomy, same hash) is a separate, explicit step
   (`dataset retemplate`, which compares hashes and refuses a different anatomy).
3. **Each subject**: its rows (the fragment's row set: section 2), its vocabulary copies and its anchor rows are written into `subject.sqlite` beside
   the existing `small_files` table, in one transaction; `key_digest` is filled from its keys; the package record and content hash are written.
4. **Verification, per subject** (before anything is marked done): the index build (section 3.3) is run over the packages so far, and every table of
   the result is compared with the original `dataset.sqlite`, **row for row and value for value** (as `compact` compares arrays: a mismatch refuses that
   subject and leaves it as it was); the subject's bundle from its package is compared with its bundle from the original database; every key's bytes
   are unchanged (no key moves).
5. **The commit**: the subject's catalogue entry. A crash before it leaves the old dataset whole (the package's new tables are rebuilt on the next
   run); after it, the subject is done and is skipped (a second run changes no byte).
6. **The end**: `meta.store_format` 53 in every package and the catalogue; the original database is kept, renamed `dataset.sqlite.format52`, and is
   removed only with the maintainers' OK (never delete data without it). Groups: group fields that `pyramid reduce` wrote into the default subject are
   moved into a group package (open question 3), verified as above.
7. **Reverse**: `dataset join --dataset DATASTORE/NAME` writes a format-52 `dataset.sqlite` from the packages (the index build, written as the
   dataset database) and removes the packages' row tables, leaving `small_files` as format 52 had it. `split` then `join` gives back the original
   database, compared table by table (`test_split`: synthetic datasets at 50, 51 and 52, a crash at each stage, idempotence, the round trip).

**The version step**: store format **53** (packages and catalogue) and the model's schema version **51** (the new tables: `subject_entry`,
`template_reference`, `key_digest`, the package record in `meta`), added to `model.sql` in all three implementations at once. Every reader refuses a
version above the newest it knows (`schema_version` error), so a build from before this change refuses a format-53 package instead of misreading it,
and a format-53 build still opens 50–52 datasets as today.

## 9. What it does not change

- **The array layout**: zarr v3 keys and bytes, the chunk key separator, sharding (~64 MiB shards of 2 MiB inner chunks), codecs, the 1 MiB small-key
  threshold, each node's `zarr.json` and its canonical text (the drain writes it from the row as today, into the package's store).
- **The model's tables and columns** (`model.sql`): rows keep their columns and meanings; only where they are stored changes. UUIDv7 ids, the node
  procedure (`pending` → keys → `complete`), recovery, the writer lock's file and rules.
- **The measurement pyramid**: its trees, leaves, keys and merge rules (`pyramid build`, `pyramid reduce`); the group tree and its codes.
- **The chunk service**: chunks of 2^15 samples, read only by io, one chunk by default.
- **The layering**: io owns the package, the catalogue, the index, registration, template resolution and the migration; pipelines name subjects,
  cohorts and templates as the caller gave them and never open a file; compute is untouched (it never knew about subjects). The layering guard keeps
  failing when pipelines, cli or bindings reach a subject's folder.

## 10. Phases and open questions

**Phases** (each its own PR, with its tests; nothing on a real dataset without the maintainers' OK):

| phase | what | tests |
|---|---|---|
| 0 | the canary measurement (dataset-wide-runs section 0), extended: in-place vs node-local `subject.sqlite` | timings per phase |
| A | the subject source in io (`SubjectSource`, temp-view union, `subject_bundle` over it), reading packages made by a test from a format-52 dataset | bundle from package = bundle from the database |
| B | format 53 writing: `SubjectWriter` on a package (per-package lock), `key_digest`, the content hash, `subject create --package`, `subject verify` | node procedure, crash recovery, hash stable and sensitive |
| C | the catalogue: `dataset create` (catalogue), `dataset register`, `participants import/export`, `--where`, `dataset index` | conflict rules, idempotence, id and vocabulary clashes |
| D | templates: `template create`, `template release`, resolution by name + version + hash, anchor rows | refusal of a wrong hash, anchor checks |
| E | migration: `dataset split`, `dataset join`, `dataset retemplate` | round trip at 50/51/52, crash at each stage |
| F | group: `pyramid reduce` over a catalogue cohort into a group package | the existing reduce tests and nxr parity, unchanged sums |
| G | nxr and the Studio: the same source, catalogue and package writer, parity tests on the same synthetic datasets; the WebAssembly client | three-way parity |

**Open questions for the maintainers:**

1. **One file or two per subject**: rows and small keys in one `subject.sqlite` (recommended: one transaction for a node's rows and keys, one file
   fewer), or rows in their own file beside format 52's `small_files.sqlite` (no change to the small-key file's name).
2. **References into a template**: anchor rows copied into each package (recommended: the model's foreign keys stay enforced, the rows are what
   `bundleOf` already follows), or foreign keys relaxed for columns that may point into another package (a change to `model.sql`), or the template's
   whole row set copied into each package (nxr's fragment does this; it duplicates rows, though not arrays).
3. **Where group results live**: a group package in the dataset (recommended), the catalogue, or the template (today, impossible once templates are
   released read-only). And whether existing group fields in default subjects are moved at migration.
4. **Ids of shared vocabulary rows**: per-copy ids matched by key at index time (recommended: nxr's merge, ported), or deterministic ids (a hash of
   scope, name and entries) so every copy has the same id.
5. **A subject in several datasets**: may a catalogue list a package that lives in another dataset, read-only (recommended: yes, it is how a cohort
   is shared without copying)?
6. **Templates**: where released templates live (a datastore-level `templates/` folder, a separate distribution), how versions are named, who
   releases one, and how the `origin` records' licences are checked before a template is shared outside the project.
7. **The catalogue's form**: SQLite with the model's tables (recommended: every implementation already reads them), or text files (a BIDS-like TSV and
   JSON pair, easy to diff and edit by hand). Participants import and export TSV either way.
8. **The chunk-queue lock**: per machine, named by the caller (recommended), per package, or kept per datastore on a workstation only.
9. **The content hash**: digests written at every key write (recommended: no re-read of shards to verify), or computed on demand by `subject verify`
   only (no write cost, slow to check).
10. **The datastore level**: does `catalog.json` stay as the datastore's list of datasets (recommended), and does a datastore also list its
    templates?

## Checklist (the architecture policy)

- Compute is untouched: packages, the catalogue, the index, registration, templates and the migration are io; the cohort and the group package are
  named by the caller and composed by the pipelines.
- Every file access stays in io: a pipeline names a subject, a cohort or a template; io resolves and opens them.
- The caller names the data: the cohort (`--subjects` or `--where`, never all by default), the template (name and version), the group package, the
  packages to register.
- Ported, not redesigned: the row set of a subject and the vocabulary remap (nxr's `shared_dataset` fragment and merge), the bundle's queries and its
  "following the template" (the Studio's `bundleOf`), the participants tables (the model's `variable_dictionary` and `subject_variable`), the writer
  lock, `dataset compact`'s verify-then-commit.
